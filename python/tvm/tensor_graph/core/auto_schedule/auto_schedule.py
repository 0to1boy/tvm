import tvm
import sys
import os
import tvm._ffi
import numpy as np
import multiprocessing
import multiprocessing.pool
import psutil
import signal
import queue
import copy
import json
import time
import math
import random
import heapq
from functools import reduce
import pickle

import tvm.auto_scheduler
from .schedule_state import RealScheduleState
from .hardware_config import get_hardware_config
from .schedule_merge import schedule_cuda_merge
from .schedule_allredcue import schedule_cuda_allreduce
from .schedule_buffer_output import schedule_cuda_buffer_output
from .schedule_tiling_and_binding import schedule_cuda_tiling_and_binding
from .schedule_buffer_input import schedule_cuda_buffer_input, create_buffer
from .schedule_unroll import schedule_cuda_unroll
from .utils import tile_axis, tile_axes, reorder_spatial_and_reduce_axes
from ..utils import to_tuple, to_int, can_to_int, to_int_or_None, ASSERT, ERROR

from tvm import auto_tensorize as at, tg
from tvm import auto_scheduler
from tvm.auto_scheduler.cost_model import RandomModel, XGBModel
from tvm.auto_scheduler.search_policy import SketchPolicy

from collections import OrderedDict
from tvm.auto_scheduler.workload_registry import make_workload_key
from tvm.auto_scheduler.task_scheduler import make_search_policies
from tvm.auto_scheduler.measure import ProgramMeasurer
from tvm.auto_scheduler.utils import array_mean
from tvm.auto_scheduler.task_scheduler import derive_similarity_tag
from tvm.auto_scheduler.task_scheduler import PrintTableInfo, LogEstimatedLatency
from tvm.auto_scheduler.compute_dag import LayoutRewriteOption

WORKLOAD_FUNC_REGISTRY_STR = {}
GLOBAL_ARGS = {}
# number of measures per round
MEASURES_PER_ROUND = 64
def task_func(func_name):
    return GLOBAL_ARGS.get(func_name, "GTA_schedule")

def interpret_cuda_schedule(sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
    op_to_id = {}
    op_to_state = {}
    for i, op in enumerate(subgraph.operation_list):
        op_to_id[op] = i
        op_to_state[op] = RealScheduleState("cuda")
    # reverse order, from output to input
    for op in reversed(subgraph.operation_list):
        schedule_cuda_merge(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_buffer_output(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        create_buffer(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_allreduce(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_tiling_and_binding(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_buffer_input(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_unroll(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )


def interpret_llvm_schedule(sch, tensors, subgraph, multi_entity, hd_config):
    return


@tvm._ffi.register_func("tg.autoschedule.interpret")
def interpret(sch, tensors, subgraph, target, multi_entity):
    with open("trace_debug_autoschedule.log", "a") as debug:
        if str(target.kind) == "cuda":
            hd_config = get_hardware_config("default_cuda", "cuda")
            interpret_cuda_schedule(
                sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
        elif str(target.kind) == "llvm":
            hd_config = get_hardware_config("default_llvm", "llvm")
            interpret_llvm_schedule(
                sch, tensors, subgraph, multi_entity, hd_config)
        else:
            ERROR("Currently no support for target",
                  target.kind, type(target.kind))
        return


def set_interpret(func):
    tvm._ffi.register_func("tg.autoschedule.interpret", func, True)


@tvm._ffi.register_func("tg.autoschedule.auto_tensorize_cuda")
def auto_tensorize_cuda(sch, tensors, log_file, trials):
    target = "cuda"
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)
    target_dag = at.compute_dag_from_tensors(tensors)
    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, transform_dump=True
    )
    if result.defined():
        sch, args = at.get_schedule(result.sch_app, result.params)
        return {sch: args}
    else:
        return {sch: []}


@tvm._ffi.register_func("tg.autoschedule.build_func")
def build_func(sch, args, target, target_host, name, binds):
    binds = {x: y for x, y in binds.items()}
    return tvm.build(sch, args, target, target_host, name=name, binds=binds)


class AmosAutoScheduleContextV1(object):
    def __init__(self, name, top_log_dir, subgraph, measure_option, verbose=False):
        self.measure_option = measure_option
        self.target = tvm.target.Target(measure_option.target)
        self.name = name
        self.subgraph = subgraph
        self.log_name = os.path.join(top_log_dir, "tg:" + name + ".log")
        self.logger = open(self.log_name, "a")
        # self.log_file = str(name)
        self.best_perf = at.MAX_FLOAT
        self.best_sch = None
        self.best_args = None
        self.best_result = None
        self.verbose = verbose
        self.result = None
        self.counter = 0
        self.total_trials = 0
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
    
    def __del__(self):
        self.logger.close()
    
    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s by %d trials..." %
              (self.log_name, trials), flush=True)
        print(">>> TGAutoSchedule <<< ")
        # search_group_size = 10
        # iterations = (trials + search_group_size - 1) // search_group_size
        iterations = 1
        search_group_size = 1
        self.total_trials += trials

        for i in range(iterations):
            schs = []
            args_lst = []
            results = []
            for j in range(search_group_size):
                # register function
                register_name = self.log_name + "_" + str(i) + str(j)
                inputs = self.target_dag.get_inputs()
                args = inputs + list(self.target_dag.tensors)
                registered_func = auto_scheduler.register_workload(register_name, f=task_func)
                GLOBAL_ARGS[register_name] = args
                workload_key = make_workload_key(register_name, args=[register_name])
                dag = tvm.auto_scheduler.ComputeDAG(workload_key)
                target = tvm.target.Target(self.measure_option.target)
                task = auto_scheduler.SearchTask(
                        compute_dag=dag, workload_key=workload_key, target=target, layout_rewrite_option=1
                    )

                measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
                # measure_ctx = auto_scheduler.RPCRunner(min_repeat_ms=300)
                tune_option = auto_scheduler.TuningOptions(
                    num_measure_trials=trials,
                    measure_callbacks=[auto_scheduler.RecordToFile(self.log_name)],
                    runner=measure_ctx.runner,
                    verbose=2,
                )
                # cost_model = auto_scheduler.RandomModel()
                cost_model = auto_scheduler.XGBModel()
                search_policy = auto_scheduler.SketchPolicy(task,cost_model)
                task.tune(tune_option, search_policy=search_policy)
        sch, args = task.apply_best(self.log_name) 
            
        perf = at.evaluate_schedule(sch, args, self.measure_option)
        schs.append(sch)
        args_lst.append(args)
        results.append(perf)
            
        for result, sch, args in zip(results, schs, args_lst):
            perf = result
            if perf < self.best_perf:
                self.best_perf = perf
                self.best_sch = sch
                self.best_args = args
        
        sch, args, perf = self.get_best_schedule()
        return sch, args
    
    def get_best_schedule(self):
        if self.best_perf != at.MAX_FLOAT:
            return self.best_sch, self.best_args, self.best_perf
        else:
            return None, None, at.MAX_FLOAT
    
    def get_measure_opt(self):
        return self.measure_option

class AmosAutoScheduleContextV2(object):
    def __init__(self, name, top_log_dir, subgraph, measure_option, verbose=False):
        self.measure_option = measure_option
        self.target = tvm.target.Target(measure_option.target)
        self.name = name
        self.subgraph = subgraph
        self.log_name = None
        self.log_file_name = os.path.join(top_log_dir, "tg:" + name + ".log")
        # self.logger = open(self.log_name, "a")
        # self.log_file = str(name)
        self.log_dir = os.path.join(top_log_dir, "at-" + name)
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        self.best_perf = at.MAX_FLOAT
        self.best_sch = None
        self.best_args = None
        self.best_result = None
        self.best_task = None
        self.search_policies = []
        self.tasks = []
        self.flag = True
        self.verbose = verbose
        self.result = None
        self.counter = 0
        self.tune_records = []
        self.entry_flag = False
        self.total_trials = 0
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        
    def __del__(self):
        self.logger.close()
    
    def auto_scheduleV1(self, trials, measurer, log_file):
        self.log_name = log_file
        print("Autoscheduling %s by %d trials..." %
              (self.log_file_name, trials), flush=True)
        print("---- TGAutoSchedule ----")
        iterations = 1
        search_group_size = 1
        self.total_trials += trials
        best_value = at.MAX_FLOAT
        if self.flag:
            for i in range(iterations):
                for j in range(search_group_size):
                    search_policy, task = self._tune_task(trials, measurer)
                    self.search_policies.append(search_policy)
                    self.tasks.append(task)
            self.flag = False
        
        for index in range(search_group_size):
            measure_inputs, measure_results = self.search_policies[index].continue_search_one_round(trials, measurer)
            for res in measure_results:
                cost = array_mean(res.costs)
                if cost < best_value:
                    best_value = cost
            info_record = self.tasks[index], measure_inputs, best_value
            self.tune_records.append(info_record)
        
        for info in self.tune_records:
            task, measure_input, perf = info
            if perf < self.best_perf:
                cur_sch, cur_args = task.apply_best(self.log_name)
                self.best_perf = perf
                self.best_sch = cur_sch
                self.best_args = cur_args
                self.best_task = task  
        
        return measure_input, self.best_perf, self.best_task
    
    def get_best_schedule(self):
        if self.best_perf != at.MAX_FLOAT:
            best_file = os.path.join(
                self.log_dir, f"best_cost_{self.name}.txt"
            )
            with open(best_file, "w") as fout:
                obj = {"cost": self.best_perf}
                string = json.dumps(obj)
                fout.write(string)
            return self.best_sch, self.best_args, self.best_perf
        else:
            return None, None, at.MAX_FLOAT

    def _tune_task(self, trials, measurer):
        global WORKLOAD_FUNC_REGISTRY_STR
        best_value = at.MAX_FLOAT
        register_name = "tg:" + self.name
        cost_model = auto_scheduler.XGBModel()
        target = tvm.target.Target(self.measure_option.target)
        # register function
        if register_name not in WORKLOAD_FUNC_REGISTRY_STR:
            print(f"new register_name :{register_name}")
            inputs = self.target_dag.get_inputs()
            args = inputs + list(self.target_dag.tensors)
            WORKLOAD_FUNC_REGISTRY_STR[register_name] = register_name
            registered_func = auto_scheduler.register_workload(register_name, f=task_func)
            GLOBAL_ARGS[register_name] = args
            workload_key = make_workload_key(register_name, args=[register_name])
            dag = tvm.auto_scheduler.ComputeDAG(workload_key)
            task = auto_scheduler.SearchTask(
                compute_dag=dag, workload_key=workload_key, target=target
            )
            search_policy = auto_scheduler.SketchPolicy(task, cost_model)
            
        else:
            print(f"Error! Current function {register_name} is already register.")

        return search_policy, task
    
    def get_measure_opt(self):
        return self.measure_option    

class TGAutoScheduleContext(object):
    def __init__(self, name, top_log_dir, subgraph, measure_option, verbose=False):
        self.measure_option = measure_option
        self.target = tvm.target.Target(measure_option.target)
        self.name = name
        self.subgraph = subgraph
        self.log_name = os.path.join(top_log_dir, "tg:" + name + ".log")
        self.logger = open(self.log_name, "a")
        self.best_perf = 1e-10
        self.best_result = None
        self.verbose = verbose
        self.result = None
        self.counter = 0
        self.total_trials = 0

        if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
            with open(self.log_name, "r") as fin:
                print("Loading from %s..." % self.log_name, flush=True)
                for line in fin:
                    obj = json.loads(line)
                    entity = obj["entity"]
                    perf = obj["perf"]
                    entity = tg.string_to_multi_schedule_entity(entity)
                    result = tg.get_schedule_result_from_entity(
                        name, subgraph, self.target, entity)
                    if perf > self.best_perf:
                        self.best_perf = perf
                        self.best_result = result
                    # feedback
                    tg.get_schedule_result(
                        self.name,
                        self.subgraph,
                        self.target,
                        self.measure_option.dev_id,
                        self.measure_option.timeout,
                        perf,
                        True,
                        result,
                    )

    def __del__(self):
        self.logger.close()

    def get_new_schedule(self):
        ret = None
        while ret is None:
            try:
                ret = tg.get_schedule_result(
                    self.name,
                    self.subgraph,
                    self.target,
                    self.measure_option.dev_id,
                    self.measure_option.timeout,
                )
            except Exception as e:
                if self.verbose:
                    print(e)
                pass
        return ret

    def count(self):
        self.counter = (self.counter + 1) % 16
        if self.counter == 0:
            print("\n", flush=True)
            print("Currently best performance:",
                  1 / self.best_perf, flush=True)

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s by %d trials..." %
              (self.log_name, trials), flush=True)
        self.total_trials += trials
        search_group_size = 10
        iterations = (trials + search_group_size - 1) // search_group_size
        beg = time.time()
        for i in range(iterations):
            schs = []
            args_lst = []
            results = []
            for j in range(search_group_size):
                # this one is get new schedule
                result = self.get_new_schedule()
                # measure current result
                results.append(result)
                sch = result.schedule
                args = result.tensors
                schs.append(sch)
                args_lst.append(args)

            timecosts = at.evaluate_schedules(
                schs, args_lst, self.measure_option)

            for result, timecost in zip(results, timecosts):
                # timecost = 1.0
                perf = 1.0 / (timecost + 1e-10)
                if perf > self.best_perf:
                    self.best_perf = perf
                    self.best_result = result
                    entity = tg.multi_schedule_entity_to_string(
                        result.schedule_entities)
                    log = {"entity": entity, "perf": perf}
                    print(json.dumps(log), file=self.logger, flush=True)
                    print(".B", end="", flush=True)
                elif timecost != at.MAX_FLOAT:
                    print(".N", end="", flush=True)
                self.count()
                # this one is feedback
                tg.get_schedule_result(
                    self.name,
                    self.subgraph,
                    self.target,
                    self.measure_option.dev_id,
                    self.measure_option.timeout,
                    perf,
                    True,
                    result,
                )
        end = time.time()
        print("Schedule cost %f seconds" % (end - beg))

        sch, args, perf = self.get_best_schedule()
        return sch, args

    def get_best_schedule(self):
        if self.best_result is not None:
            return self.best_result.schedule, self.best_result.tensors, 1 / self.best_perf
        else:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AnsorAutoScheduleContext(object):
    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.measure_option = measure_option
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        task_name = name

        inputs = self.target_dag.get_inputs()
        args = inputs + list(self.target_dag.tensors)

        self.total_trials = 0

        def task_func():
            return args

        registered_func = auto_scheduler.register_workload(
            task_name, f=task_func)

        target = tvm.target.Target(measure_option.target)

        self.task = auto_scheduler.create_task(
            task_name,
            (),
            target,
            hardware_params=auto_scheduler.HardwareParams(
                1024,  # cores
                16,  # vector bytes
                1024,  # cache line bytes
            ),
        )

        task_name = self.task.workload_key[2:-2]
        self.log_name = os.path.join(
            top_log_dir, "ansor:" + task_name + ".log")

        self.runner = auto_scheduler.LocalRunner(
            timeout=measure_option.timeout,
            number=measure_option.number,
            repeat=measure_option.repeat,
            min_repeat_ms=measure_option.min_repeat_ms,
            cooldown_interval=measure_option.cooldown_interval,
            enable_cpu_cache_flush=measure_option.enable_cpu_cache_flush,
        )

    def auto_schedule(self, trials, model="xgb"):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s by %d trials..." %
              (self.log_name, trials), flush=True)
        self.total_trials += trials
        sch, args = None, None
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=trials,
            # runner=self.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_name)],
        )

        if model == "random":
            cost_model = RandomModel()
        elif model == "xgb":
            cost_model = XGBModel()
        else:
            raise RuntimeError("Unsupported model: %s" % model)
        if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
            cost_model.update_from_file(self.log_name)
            search_policy = auto_scheduler.SketchPolicy(
                self.task,
                cost_model,
                init_search_callbacks=[
                    auto_scheduler.PreloadMeasuredStates(self.log_name)],
            )
        else:
            search_policy = SketchPolicy(self.task, cost_model)
        sch, args = auto_scheduler.auto_schedule(
            self.task, search_policy=search_policy, tuning_options=tune_option
        )

        return sch, args

    def get_best_schedule(self):
        try:
            inp, res = auto_scheduler.load_best(
                self.log_name, self.task.workload_key)
            sch, args = self.task.compute_dag.apply_steps_from_state(inp.state)
            return sch, args, np.mean([float(x) * 1e3 for x in res.costs])
        except:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AutoTensorizeContext(object):
    @classmethod
    def can_use(cls, name, top_log_dir, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        task_name = name
        log_name = os.path.join(top_log_dir, "at:" + task_name + ".log")
        match_result, new_state = at.auto_tensorize_compute(
            target_dag, measure_option.target, log_name, measure_option
        )
        return match_result is not None and new_state is not None

    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option
        task_name = name
        self.log_name = os.path.join(top_log_dir, "at:" + task_name + ".log")
        self.match_result, self.new_state = at.auto_tensorize_compute(
            self.target_dag, measure_option.target, self.log_name, measure_option
        )

        self.total_trials = 0

        assert self.match_result is not None
        assert self.new_state is not None

        if str(self.measure_option.target) == "cuda":
            self.schedule_gen = at.CUDAScheduleGeneratorV2(
                self.match_result, self.new_state, log_file=self.log_name
            )
            if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
                self.schedule_gen.load_from_file(self.log_name)
            sc_info = self.schedule_gen.get_schedule_compute_info()
            self.schedule_app = at.CUDAScheduleApplierV2(
                self.match_result, sc_info)
            self.checker = at.CUDAProgramChecker(
                arch=at.get_cuda_compute_version(self.measure_option.dev_id)
            )
        else:
            raise RuntimeError("Do not support target: %s" %
                               self.measure_option.target)

        self.builder = at.pebble_local_builder_build
        self.runner = at.pebble_local_runner_run

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s..." % self.log_name, flush=True)
        self.total_trials += trials
        if trials:
            value, params = at.find_optimized_parameters(
                self.match_result,
                self.schedule_gen,
                self.schedule_app,
                self.measure_option,
                self.checker,
                trials,  # policy="random",
                builder=self.builder,
                runner=self.runner,
                verbose=False,
            )

        sch, args, perf = self.get_best_schedule()
        return sch, args

    def get_best_schedule(self):
        if self.schedule_gen.has_entry():
            entry = self.schedule_gen.get_best_entry()
            # we store 1/time_cost in file
            params = entry.record
            sch, args = at.get_schedule(self.schedule_app, params)
            return sch, args, 1 / entry.value * 1e3
        else:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AutoTensorizeContextV2(object):
    @classmethod
    def can_use(cls, name, top_log_dir, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        match_results = at.get_match_results(target_dag, measure_option.target)
        return len(match_results) > 0

    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option
        task_name = name
        self.log_dir = os.path.join(top_log_dir, "at-" + task_name)
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        self.log_name = os.path.join(
            self.log_dir, "at:" + task_name + ":transform" + ".log")
        match_results = at.get_match_results(
            self.target_dag, measure_option.target)

        self.total_trials = 0

        assert len(match_results) > 0
        self.match_result = match_results[0]

        self.gen = at.MappingGenerator(
            self.match_result, log_file=self.log_name, allow_repeat=True)
        if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
            self.gen.load_from_file(self.log_name)
        self.app = at.MappingApplier(
            self.match_result, verbose=True, strict=False)

        class ScheduleContext:
            def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
                self.schedule_gen = schedule_gen
                self.schedule_app = schedule_app
                self.sc_info = sc_info
                self.checker = checker
                self.generate_schedule = generate_schedule

        self.schedule_ctx_cls = ScheduleContext

        self.schedule_context_cache = {}
        self.best_value = 1 / at.MAX_FLOAT
        self.best_ctx = None
        self.best_params = None
        self.pure_test = False
        self.drop_output = False
        self.enable_split_K = False
        self.use_shared_store = False

        self.builder = at.pebble_local_builder_build
        self.runner = at.pebble_local_runner_run

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s..." % self.log_name, flush=True)
        self.total_trials += trials
        schedule_trials = 20
        iterations = trials // schedule_trials

        if iterations == 0:
            iterations = 1
            schedule_trials = 0
            self.pure_test = True
            print("Pure testing mode...", flush=True)
        beg = time.time()
        for it in range(iterations):
            if not self.pure_test:
                feasible = False
                while not feasible:
                    record = self.gen.get_next(policy="random")
                    try:
                        tmp_app = at.MappingApplier(
                            self.match_result, strict=False)
                        tmp_app.apply(record, drop_output=self.drop_output)
                        feasible = True
                    except RuntimeError as e:
                        print("Catch an infeasible mapping:", flush=True)
                        print(record, flush=True)
            else:
                try:
                    entry = self.gen.get_best_entry()
                    record = entry.record
                except Exception as e:
                    raise RuntimeError(
                        "Can't get previous results for test mode.")
            print(f"Choose transform: {record}", flush=True)
            new_state = self.app.apply(record, drop_output=self.drop_output)

            record_key = record.as_key()
            if record_key in self.schedule_context_cache:
                sch_ctx = self.schedule_context_cache[record_key]
            else:
                current_log_file = os.path.join(
                    self.log_dir, "at:" + self.name +
                    ":mapping:" + str(record_key) + ".log"
                )
                if str(self.measure_option.target) == "cuda":
                    if not self.enable_split_K:
                        if self.use_shared_store:
                            raise NotImplementedError()
                        else:
                            schedule_gen = at.CUDAScheduleGeneratorV2(
                                self.match_result, new_state, log_file=current_log_file
                            )
                            if os.path.exists(current_log_file) and os.path.isfile(
                                current_log_file
                            ):
                                schedule_gen.load_from_file(current_log_file)
                            sc_info = schedule_gen.get_schedule_compute_info()
                            schedule_app = at.CUDAScheduleApplierV2(
                                self.match_result, sc_info)
                    else:
                        schedule_gen = at.CUDAScheduleGeneratorSplitK(
                            self.match_result, new_state, log_file=current_log_file
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierSplitK(
                            self.match_result, sc_info)
                    checker = at.CUDAProgramChecker(
                        arch=at.get_cuda_compute_version(
                            self.measure_option.dev_id)
                    )
                elif str(self.measure_option.target) == "opencl":
                    schedule_gen = at.MaliScheduleGenerator(
                        self.match_result, new_state, log_file=current_log_file
                    )
                    if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        schedule_gen.load_from_file(current_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = at.MaliScheduleApplier(
                        self.match_result, sc_info)
                    # TODO: write a checker for MALI GPU
                    checker = at.MaliProgramChecker(arch="g76")
                else:
                    raise RuntimeError("Do not support target: %s" %
                                       self.measure_option.target)

                # use tuning to find params
                if schedule_trials:
                    generate_schedule = at.find_optimized_parameters_v2(
                        self.match_result,
                        schedule_gen,
                        schedule_app,
                        self.measure_option,
                        checker,
                        schedule_trials,  # policy="random",
                        builder=self.builder,
                        runner=self.runner,
                        verbose=False,
                        search_group_size=10,
                    )
                else:
                    generate_schedule = None

                sch_ctx = self.schedule_ctx_cls(
                    schedule_gen, schedule_app, sc_info, checker, generate_schedule
                )
                self.schedule_context_cache[record_key] = sch_ctx

            if sch_ctx.generate_schedule is not None:
                value, params = next(sch_ctx.generate_schedule)
            try:
                entry = sch_ctx.schedule_gen.get_best_entry()
                # we store 1/time_cost in file
                params, value = entry.record, entry.value
                # print("Evaluation only:", params, value, flush=True)
                if not self.pure_test:
                    self.gen.feedback(record, value)
            except Exception as e:
                params = None
                value = 1 / at.MAX_FLOAT

            # record the best
            if value > self.best_value:
                self.best_value = value
                self.best_ctx = sch_ctx
                self.best_params = params

            print(
                f"Iteration: {it+1}: {value}/{self.best_value}, {str(record)}, {str(params)}",
                flush=True,
            )

            if (it + 1) % 10 == 0:
                print("Show transformation explore summary:", flush=True)
                for k, v in self.schedule_context_cache.items():
                    print(f"{str(k)}: {v.schedule_gen.num_entries()}", flush=True)

        end = time.time()
        print(f"Tensorize use time {(end - beg)} s.", flush=True)
        sch, args, perf = self.get_best_schedule()
        return sch, args

    def get_best_schedule(self):
        tmp_schedule_context_cache = {}
        if self.gen.has_entry():
            best_transform = self.gen.get_best_entry()
            transform = best_transform.record
            transform_key = transform.as_key()
            best_log_file = os.path.join(
                self.log_dir, "at:" + self.name +
                ":mapping:" + str(transform_key) + ".log"
            )
            if transform_key in self.schedule_context_cache:
                schedule_gen = self.schedule_context_cache[transform_key].schedule_gen
                schedule_app = self.schedule_context_cache[transform_key].schedule_app
            elif transform_key in tmp_schedule_context_cache:
                schedule_gen, schedule_app = tmp_schedule_context_cache[transform_key]
            else:
                new_state = self.app.apply(
                    transform, drop_output=self.drop_output)
                if str(self.measure_option.target) == "cuda":
                    if not self.enable_split_K:
                        if self.use_shared_store:
                            raise NotImplementedError()
                        else:
                            schedule_gen = at.CUDAScheduleGeneratorV2(
                                self.match_result, new_state, log_file=best_log_file
                            )
                            if os.path.exists(best_log_file) and os.path.isfile(best_log_file):
                                schedule_gen.load_from_file(best_log_file)
                            sc_info = schedule_gen.get_schedule_compute_info()
                            schedule_app = at.CUDAScheduleApplierV2(
                                self.match_result, sc_info)
                    else:
                        schedule_gen = at.CUDAScheduleGeneratorSplitK(
                            self.match_result, new_state, log_file=best_log_file
                        )
                        if os.path.exists(best_log_file) and os.path.isfile(best_log_file):
                            schedule_gen.load_from_file(best_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierSplitK(
                            self.match_result, sc_info)
                    checker = at.CUDAProgramChecker(
                        arch=at.get_cuda_compute_version(
                            self.measure_option.dev_id)
                    )
                elif str(self.measure_option.target) == "opencl":
                    schedule_gen = at.MaliScheduleGenerator(
                        self.match_result, new_state, log_file=best_log_file
                    )
                    if os.path.exists(best_log_file) and os.path.isfile(best_log_file):
                        schedule_gen.load_from_file(best_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = at.MaliScheduleApplier(
                        self.match_result, sc_info)
                    # TODO: write a checker for MALI GPU
                    checker = at.MaliProgramChecker(arch="g76")
                else:
                    raise RuntimeError("Do not support target: %s" %
                                       self.measure_option.target)
                tmp_schedule_context_cache[transform_key] = (
                    schedule_gen, schedule_app)

            if schedule_gen.has_entry():
                entry = schedule_gen.get_best_entry()
                # we store 1/time_cost in file
                params = entry.record
                sch, args = at.get_schedule(schedule_app, params)
                return sch, args, 1 / entry.value * 1e3
            else:
                return None, None, at.MAX_FLOAT
        else:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AutoTensorizeContextV3(object):
    @classmethod
    def can_use(cls, name, top_log_dir, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        match_results = at.get_match_results(target_dag, measure_option.target)
        return len(match_results) > 0

    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option
        task_name = name
        self.log_dir = os.path.join(top_log_dir, "at-" + task_name)
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        self.log_name = os.path.join(
            self.log_dir, "at:" + task_name + ":transform" + ".log")
        match_results = at.get_match_results(
            self.target_dag, measure_option.target)

        self.total_trials = 0

        assert len(match_results) > 0

        self.drop_output = False
        self.search_group_size = 5
        self.repeat_rounds = 2
        self.enable_split_K = False
        self.use_shared_store = False
        self.enable_perf_model = True

        self.builder = at.pebble_local_builder_build
        self.runner = at.pebble_local_runner_run

        self.all_matches = []
        self.all_mappings = []
        self.appliers = []
        self.mapping_weights = []
        self.weights_updates = []
        self.momentum = 0.8
        # use all_fit logic to choose the one with minimum padding
        match_result, _ = at.policy.all_fit(match_results)
        match_results = [match_result]
        self.total_matchings = 0
        self.total_mappings = 0
        for match_result in match_results:
            transform_strict = True
            self.all_matches.append(match_result)
            gen = at.MappingGenerator(match_result)
            mappings = gen.get_all()
            # filter out infeasible mappings
            feasible_mappings = []
            tmp_app = at.MappingApplier(match_result, strict=transform_strict)
            for mapping in mappings:
                try:
                    tmp_app.apply(mapping, drop_output=self.drop_output)
                    feasible_mappings.append(mapping)
                except RuntimeError as e:
                    pass
            if len(feasible_mappings) == 0:
                # relax
                transform_strict = False
            else:
                mappings = feasible_mappings
            # record the feasible mappings
            self.all_mappings.append(mappings)
            self.total_matchings += 1
            assert len(mappings) > 0
            self.total_mappings += len(mappings)
            self.mapping_weights.append(
                [1.0 / len(mappings) for m in mappings])
            self.weights_updates.append([0.0 for m in mappings])
            app = at.MappingApplier(
                match_result, verbose=False, strict=transform_strict)
            self.appliers.append(app)
        assert self.total_mappings > 0

        class ScheduleContext:
            def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
                self.schedule_gen = schedule_gen
                self.schedule_app = schedule_app
                self.sc_info = sc_info
                self.checker = checker
                self.generate_schedule = generate_schedule

        self.schedule_ctx_cls = ScheduleContext

        # global context for overall exploration
        self.schedule_context_cache = {}
        self.best_value = 1 / at.MAX_FLOAT
        self.best_ctx = None
        self.best_params = None

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s..." % self.log_name, flush=True)
        self.total_trials += trials

        if trials < self.total_mappings * self.repeat_rounds * self.search_group_size:
            print(
                f"[Warning] Too few trials, expect at least {self.total_mappings * self.repeat_rounds * self.search_group_size} trials.",
                flush=True,
            )
            trials = self.total_mappings * self.repeat_rounds * self.search_group_size
            print(
                f"Increase trials to {self.total_mappings * self.repeat_rounds * self.search_group_size}.",
                flush=True,
            )
        else:
            print("Total trials:", trials, flush=True)
        trials_per_matching = trials // self.repeat_rounds // self.total_matchings

        print("Num rounds:", self.repeat_rounds, flush=True)
        print("Num matching:", self.total_matchings, flush=True)
        print("Num mapping:", self.total_mappings, flush=True)
        print("Initial trials per matching:", trials_per_matching, flush=True)

        beg = time.time()
        for round in range(self.repeat_rounds):
            for match_id in range(self.total_matchings):
                match_result = self.all_matches[match_id]
                app = self.appliers[match_id]
                weights = self.mapping_weights[match_id]
                updates = self.weights_updates[match_id]
                tune_trials = [math.ceil(trials_per_matching * x)
                               for x in weights]
                best_values_of_mappings = []
                print("Original weights", weights, flush=True)
                print("Original trials for each mapping",
                      tune_trials, flush=True)
                print("Current explored matching:",
                      str(match_result), flush=True)
                print("Its axis mapping:", flush=True)
                for i, v in match_result.axis_map.items():
                    print(i.var, ":", [x.var for x in v], flush=True)
                for mapping_id in range(len(self.all_mappings[match_id])):
                    record = self.all_mappings[match_id][mapping_id]
                    print("Current explored mapping:", str(record), flush=True)

                    # transform compute
                    new_state = app.apply(record, drop_output=self.drop_output)
                    # prepare tune log file
                    record_key = record.as_key()
                    current_log_file = os.path.join(
                        self.log_dir, "at:" + self.name +
                        ":mapping:" + str(record_key) + ".log"
                    )
                    if record_key in self.schedule_context_cache:
                        sch_ctx = self.schedule_context_cache[record_key]
                    else:
                        schedule_gen, schedule_app, checker, sc_info = self._get_schedule_ctx(
                            match_result, new_state, current_log_file
                        )

                        # tune loop
                        schedule_trials = tune_trials[mapping_id]
                        if schedule_trials:
                            # this returns a generator
                            if self.enable_perf_model:
                                generate_schedule = at.find_optimized_parameters_v3(
                                    match_result,
                                    schedule_gen,
                                    schedule_app,
                                    self.measure_option,
                                    checker,
                                    schedule_trials,  # policy="random",
                                    builder=self.builder,
                                    runner=self.runner,
                                    verbose=False,
                                    search_group_size=self.search_group_size,
                                    build_parallel=1,
                                    run_parallel=1,
                                    perf_percentage=0.5,
                                )
                            else:
                                generate_schedule = at.find_optimized_parameters_v2(
                                    match_result,
                                    schedule_gen,
                                    schedule_app,
                                    self.measure_option,
                                    checker,
                                    schedule_trials,  # policy="random",
                                    builder=self.builder,
                                    runner=self.runner,
                                    verbose=False,
                                    search_group_size=self.search_group_size,
                                    build_parallel=1,
                                    run_parallel=1,
                                )
                        else:
                            generate_schedule = None

                        # create new schedule context
                        sch_ctx = self.schedule_ctx_cls(
                            schedule_gen, schedule_app, sc_info, checker, generate_schedule
                        )
                        self.schedule_context_cache[record_key] = sch_ctx

                    if sch_ctx.generate_schedule is not None:
                        value, params = next(sch_ctx.generate_schedule)
                    try:
                        entry = sch_ctx.schedule_gen.get_best_entry()
                        # we store 1/time_cost in file
                        params, value = entry.record, entry.value
                    except Exception as e:
                        params = None
                        value = 1 / at.MAX_FLOAT

                    # record the best value of current mapping
                    best_values_of_mappings.append(value)

                    # record the best
                    if value > self.best_value:
                        self.best_value = value
                        self.best_ctx = sch_ctx
                        self.best_params = params

                    print(
                        f"Best record value:{self.best_value} (larger is better)", flush=True)
                    print(
                        f"Round {round+1}, Match {match_id+1}, Mapping {mapping_id+1}: {value}/{self.best_value}({1/self.best_value*1e3} ms), {str(record)}, {str(params)}",
                        flush=True,
                    )

                # redistribute weights according to current best value
                max_value = max(best_values_of_mappings)
                exp_scores = [math.exp(x - max_value)
                              for x in best_values_of_mappings]
                sum_exp_scores = sum(exp_scores)
                new_weights = [x / sum_exp_scores for x in exp_scores]
                delta_weights = [new_weights[i] - weights[i]
                                 for i in range(len(weights))]
                new_updates = [
                    delta_weights[i] + self.momentum * updates[i] for i in range(len(updates))
                ]
                new_weights = [weights[i] + new_updates[i]
                               for i in range(len(new_updates))]
                exp_scores = [math.exp(x) for x in new_weights]
                sum_exp_scores = sum(exp_scores)
                new_weights = [x / sum_exp_scores for x in exp_scores]
                # update into global context
                self.mapping_weights[match_id] = new_weights
                self.weights_updates[match_id] = new_updates
                print("New weights", new_weights, flush=True)

            print("Show mapping exploration summary:", flush=True)
            for k, v in self.schedule_context_cache.items():
                print(
                    f"mapping {str(k)}: explored {v.schedule_gen.num_entries()} schedules",
                    flush=True,
                )
        end = time.time()
        print(f"Tensorize use time {(end - beg)} s.", flush=True)
        sch, args, perf = self.get_best_schedule()
        return sch, args

    def _get_schedule_ctx(self, match_result, new_state, current_log_file):
        target = self.measure_option.target
        if str(target) == "cuda":
            if not self.enable_split_K:
                if self.use_shared_store:
                    raise NotImplementedError()
                else:
                    if self.enable_perf_model:
                        schedule_gen = at.CUDAScheduleGeneratorV3(
                            match_result,
                            new_state,
                            log_file=current_log_file,
                            arch=at.get_cuda_compute_version(
                                self.measure_option.dev_id),
                            verbose_init=False,
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierV3(
                            match_result, sc_info)
                    else:
                        schedule_gen = at.CUDAScheduleGeneratorV2(
                            match_result,
                            new_state,
                            log_file=current_log_file,
                            arch=at.get_cuda_compute_version(
                                self.measure_option.dev_id),
                            verbose_init=False,
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierV2(
                            match_result, sc_info)
            else:
                if self.enable_perf_model:
                    raise NotImplementedError()
                else:
                    schedule_gen = at.CUDAScheduleGeneratorSplitK(
                        match_result,
                        new_state,
                        log_file=current_log_file,
                        arch=at.get_cuda_compute_version(
                            self.measure_option.dev_id),
                        verbose_init=False,
                    )
                    if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        schedule_gen.load_from_file(current_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = at.CUDAScheduleApplierSplitK(
                        match_result, sc_info)
            checker = at.CUDAProgramChecker(
                arch=at.get_cuda_compute_version(self.measure_option.dev_id), verbose_init=False
            )
        elif str(target) == "opencl":
            schedule_gen = at.MaliScheduleGenerator(
                match_result, new_state, log_file=current_log_file, verbose_init=False
            )
            if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                schedule_gen.load_from_file(current_log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = at.MaliScheduleApplier(match_result, sc_info)
            # TODO: write a checker for MALI GPU
            checker = at.MaliProgramChecker(arch="g76", verbose_init=False)
        elif str(target) == "llvm -mcpu=skylake-avx512":
            schedule_gen = at.LLVMScheduleGenerator(
                match_result, new_state, log_file=current_log_file, verbose_init=False
            )
            if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                schedule_gen.load_from_file(current_log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = at.LLVMScheduleApplier(match_result, sc_info)
            # TODO: write a checker for CPU
            checker = at.EmptyChecker()
        elif str(target).startswith("tenet"):
            target = str(target)
            parts = target.split(" ")
            assert len(parts) > 1
            if parts[1] == "cuda":
                schedule_gen = at.CUDAScheduleGeneratorTenet(
                    match_result,
                    new_state,
                    log_file=current_log_file,
                    arch=at.get_cuda_compute_version(
                        self.measure_option.dev_id),
                    verbose_init=False,
                )
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = at.CUDAScheduleApplierTenet(
                    match_result, sc_info)
                checker = at.CUDAProgramChecker(
                    arch=at.get_cuda_compute_version(self.measure_option.dev_id), verbose_init=False
                )
            else:
                schedule_gen = at.TenetScheduleGenerator(
                    match_result, new_state, log_file=current_log_file, verbose_init=False
                )
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = at.TenetScheduleApplier(match_result, sc_info)
                # TODO: write a checker for TENET
                checker = at.EmptyChecker()
        else:
            raise RuntimeError("Do not support target: %s" % target)
        return schedule_gen, schedule_app, checker, sc_info

    def get_best_schedule(self):
        best_sch = None
        best_args = None
        best_mapping = None
        best_params = None
        best_cost = at.MAX_FLOAT
        tmp_schedule_context_cache = {}
        best_file = os.path.join(
            self.log_dir, f"best_mappings_{self.name}.txt")
        best_mapping_params = None
        if (os.path.exists(best_file) and os.path.isfile(best_file)):
            with open(best_file, "r") as fin:
                for string in fin:
                    obj = json.loads(string)
                    best_mapping_params = obj
                    break
        for match_id in range(self.total_matchings):
            match_result = self.all_matches[match_id]
            app = self.appliers[match_id]
            for mapping_id in range(len(self.all_mappings[match_id])):
                record = self.all_mappings[match_id][mapping_id]

                # prepare tune log file
                record_key = record.as_key()
                if best_mapping_params is not None and best_mapping_params["mapping"] == record_key:
                    new_state = app.apply(record, drop_output=self.drop_output)
                    schedule_gen, schedule_app, checker, sc_info = self._get_schedule_ctx(
                        match_result, new_state, os.path.devnull
                    )
                    params_obj = best_mapping_params["params"]
                    params = schedule_gen.record_from_json(params_obj)
                    sch, args = at.get_schedule(schedule_app, params)
                    return sch, args, best_mapping_params["cost"]
                current_log_file = os.path.join(
                    self.log_dir, "at:" + self.name +
                    ":mapping:" + str(record_key) + ".log"
                )
                if record_key in self.schedule_context_cache:
                    schedule_gen = self.schedule_context_cache[record_key].schedule_gen
                    schedule_app = self.schedule_context_cache[record_key].schedule_app
                elif record_key in tmp_schedule_context_cache:
                    schedule_gen, schedule_app = tmp_schedule_context_cache[record_key]
                elif (os.path.exists(current_log_file) and os.path.isfile(current_log_file)):
                    new_state = app.apply(record, drop_output=self.drop_output)
                    schedule_gen, schedule_app, checker, sc_info = self._get_schedule_ctx(
                        match_result, new_state, current_log_file
                    )
                    # create new tmp schedule context
                    tmp_schedule_context_cache[record_key] = (
                        schedule_gen, schedule_app)
                else:
                    continue
                if schedule_gen.has_entry():
                    entry = schedule_gen.get_best_entry()
                    # we store 1/time_cost in file
                    params = entry.record
                    sch, args = at.get_schedule(schedule_app, params)
                    if 1 / entry.value < best_cost:
                        best_sch = sch
                        best_args = args
                        best_mapping = record_key
                        best_params = params
                        best_cost = 1 / entry.value
        if best_sch is not None:
            with open(best_file, "w") as fout:
                obj = {"mapping": best_mapping,
                    "params": best_params.to_json(), "cost": best_cost}
                string = json.dumps(obj)
                fout.write(string)
        return best_sch, best_args, best_cost

    def get_measure_opt(self):
        return self.measure_option


class AmosAnsorContextV1(object):
    @classmethod
    def can_use(cls, name, top_log_dir, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        match_results = at.get_match_results(target_dag, measure_option.target)
        return len(match_results) > 0
    
    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option
        task_name = name
        self.log_dir = os.path.join(top_log_dir, "at-" + task_name)
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        self.log_name = os.path.join(
            self.log_dir, "at:" + task_name + ":transform" + ".log")
        match_results = at.get_match_results(
            self.target_dag, measure_option.target)
        
        self.best_task = None
        self.workload_key = None
        self.best_mapping = None

        self.total_trials = 0
        assert len(match_results) > 0

        self.drop_output = False
        self.repeat_rounds = 1
        self.least_trails = 10
        self.enable_split_K = False
        self.use_shared_store = False
        self.enable_perf_model = False

        self.all_matches = []
        self.all_mappings = []
        self.appliers = []
        self.momentum = 0.8
        # use all_fit logic to choose the one with minimum padding
        match_result, _ = at.policy.all_fit(match_results)
        match_results = [match_result]
        self.total_matchings = 0
        self.total_mappings = 0
        for match_result in match_results:
            transform_strict = True
            self.all_matches.append(match_result)
            gen = at.MappingGenerator(match_result)
            mappings = gen.get_all()
            # filter out infeasible mappings
            feasible_mappings = []
            tmp_app = at.MappingApplier(match_result, strict=transform_strict)
            for mapping in mappings:
                try:
                    tmp_app.apply(mapping, drop_output=self.drop_output)
                    feasible_mappings.append(mapping)
                except RuntimeError as e:
                    pass
            if len(feasible_mappings) == 0:
                # relax
                transform_strict = False
            else:
                mappings = feasible_mappings
            # record the feasible mappings
            self.all_mappings.append(mappings)
            self.total_matchings += 1
            assert len(mappings) > 0
            self.total_mappings += len(mappings)
            app = at.MappingApplier(
                match_result, verbose=False, strict=transform_strict)
            self.appliers.append(app)
            
        self.selected_indices = self.sample_mappings()
        
        assert self.total_mappings > 0

        class ScheduleContext:
            def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
                self.schedule_gen = schedule_gen
                self.schedule_app = schedule_app
                self.sc_info = sc_info
                self.checker = checker
                self.generate_schedule = generate_schedule

        self.schedule_ctx_cls = ScheduleContext

        # global context for overall exploration
        # self.schedule_context_cache = {}
        self.best_value = at.MAX_FLOAT
        self.best_match_id = 0
        self.best_mapping_id = 0
        self.best_ctx = None
        self.best_params = None

    def sample_mappings(self):
        sample_matches_id = []
        sample_mappings_id = []
        num_mappings_to_select = self.total_mappings
        if self.total_mappings > 10:
            num_mappings_to_select = int(0.1 * self.total_mappings)
        else:
            num_mappings_to_select = 1

        all_indices = [(i, j) for i in range(len(self.all_mappings)) for j in range(len(self.all_mappings[i]))]

        selected_indices = random.sample(all_indices, num_mappings_to_select)
        
        return selected_indices

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s..." % self.log_name, flush=True)
        print(">>> AutoTensorize Begin <<< ")
        self.total_trials += trials

        info_record = list()
        
        if trials < self.least_trails:        
            print(
                f"[Warning] Too few trials, expect at least {self.least_trails} trials.",
                flush=True,
            )
            trials = self.least_trails
            print(
                f"Increase trials to {self.least_trails}.",
                flush=True,
            )
        else:
            print("Total trials:", trials, flush=True)
        trials_per_matching = trials
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
        cost_model = auto_scheduler.XGBModel()
        # cost_model = auto_scheduler.RandomModel()
        
        print("repeat rounds:", self.repeat_rounds, flush=True)
        print("total_matchings:", self.total_matchings, flush=True)
        print("total_mappings:", self.total_mappings, flush=True)
        print("Initial trials per matching:", trials_per_matching, flush=True)
        
        for round in range(self.repeat_rounds):
            for index in (self.selected_indices):
                match_id, mapping_id = index
                match_result = self.all_matches[match_id]
                app = self.appliers[match_id]
                best_values_of_mappings = []
                
                print("Current explored matching:",
                        str(match_result), flush=True)
                print("Its axis mapping:", flush=True)
                
                for i, v in match_result.axis_map.items():
                    print(i.var, ":", [x.var for x in v], flush=True)
                
                record = self.all_mappings[match_id][mapping_id]
                print("Current explored mapping:", str(record), flush=True)

                # transform compute
                new_state = app.apply(record, drop_output=self.drop_output)
                # prepare schedulers
                schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
                checker = at.CUDAProgramChecker()
                # use ansor to find optimized params
                target_dag = schedule_app.target_dag
                inputs = target_dag.get_inputs()
                args = inputs + list(target_dag.tensors)
                
                register_name = self.name + str(round) + str(match_id) + str(mapping_id)
                registered_func = auto_scheduler.register_workload(register_name, f=task_func)
                GLOBAL_ARGS[register_name] = args
                workload_key = make_workload_key(register_name, args=[register_name])
                
                dag = tvm.auto_scheduler.ComputeDAG(workload_key, hw_abs_dag=schedule_gen.hw_abs_dag_stage)
                target = tvm.target.Target(self.measure_option.target)
                task = auto_scheduler.SearchTask(
                    compute_dag=dag, workload_key=workload_key, target=target, layout_rewrite_option=1
                )
            
                # measure_ctx = auto_scheduler.RPCRunner(min_repeat_ms=300)
                tune_option = auto_scheduler.TuningOptions(
                    num_measure_trials=trials_per_matching,
                    measure_callbacks=[auto_scheduler.RecordToFile(self.log_name)],
                    runner=measure_ctx.runner,
                    verbose=2,
                )
                # cost_model = auto_scheduler.RandomModel()
                search_policy = auto_scheduler.SketchPolicy(task,cost_model)
                task.tune(tune_option, search_policy=search_policy)
                print(
                    f"Round {round+1}, Match {match_id+1}, Mapping {mapping_id+1}.",
                    flush=True,
                )

                sch, args = task.apply_best(self.log_name)
                info_record.append([task, sch, args, record])
        
        for info in info_record:
            task, cur_sch, cur_args, cur_record = info
            value = at.evaluate_schedule(cur_sch, cur_args, self.measure_option)
            # record the best
            if value < self.best_value:
                self.best_value = value
                self.best_task = task
                self.workload_key = task.workload_key
                self.best_mapping = str(cur_record)

        print(
            f"Best record value:{self.best_value} (larger is better), best mapping: {self.best_mapping}", flush=True)
        
        print("Curent operation is finish tune.", flush=True)        
        sch, args, perf = self.get_best_schedule()
        return sch, args
    
    def get_best_schedule(self):
        best_sch = None
        best_args = None
        best_cost = at.MAX_FLOAT
        log_file = self.log_name
        best_file = os.path.join(
            self.log_dir, f"best_mappings_{self.name}.txt")

        if self.best_task is not None:
            
            inp, _ = tvm.auto_scheduler.load_best_record(log_file, self.workload_key)
            
            if inp is None:
                print("No autotensorize records in the log file.")            
            else:
                sch, args = self.best_task.apply_best(log_file)
                value = at.evaluate_schedule(sch, args, self.measure_option)   
                if value < best_cost:
                    best_sch = sch
                    best_args = args
                    best_cost = value
                if best_sch is not None:
                    with open(best_file, "w") as fout:
                        obj = {"mapping": self.best_mapping,"cost": best_cost}
                        string = json.dumps(obj)
                        fout.write(string)
                        
        return best_sch, best_args, best_cost

    def get_measure_opt(self):
        return self.measure_option

# Add Task Scheduler to AMOS
class AmosAnsorContextV2(object):
    @classmethod
    def can_use(cls, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        match_results = at.get_match_results(target_dag, measure_option.target)
        return len(match_results) > 0
    
    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors(
            [x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option

        self.log_name = None
        task_name = name
        self.log_dir = os.path.join(top_log_dir, "at-" + task_name)
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        match_results = at.get_match_results(
            self.target_dag, measure_option.target)
        
        self.best_task = None
        self.workload_key = None
        self.best_mapping = None
        self.tune_records = {}
        self.best_sch = None
        self.best_args = None

        self.total_trials = 0
        assert len(match_results) > 0

        self.drop_output = False
        self.repeat_rounds = 1
        self.least_trails = 10
        self.enable_split_K = False
        self.use_shared_store = False
        self.enable_perf_model = False

        self.enable_pruning = True
        self.all_results = {}
        self.probability = []
        self.selected_key = {}
        self.tasks = []
        
        self.search_policies = []
        self.records = []
        self.existing_best_cost = at.MAX_FLOAT

        self.all_matches = []
        self.all_mappings = []
        self.appliers = []
        self.momentum = 0.8
        # use all_fit logic to choose the one with minimum padding
        match_result, _ = at.policy.all_fit(match_results)
        match_results = [match_result]
        self.total_matchings = 0
        self.total_mappings = 0
        for match_result in match_results:
            self.transform_strict = True
            self.all_matches.append(match_result)
            gen = at.MappingGenerator(match_result)
            mappings = gen.get_all()
            print(f"length of all mappings before filter = {len(mappings)}")
            # filter out infeasible mappings
            feasible_mappings = []
            tmp_app = at.MappingApplier(match_result, strict=self.transform_strict)
            for mapping in mappings:
                try:
                    tmp_app.apply(mapping, drop_output=self.drop_output)
                    feasible_mappings.append(mapping)
                except RuntimeError as e:
                    pass
            if len(feasible_mappings) == 0:
                print(f"entering, transform_strict = False")
                # relax
                self.transform_strict = False
            else:
                print("entering else...")
                mappings = feasible_mappings
            print(f"Total feasible mappings: {len(mappings)}")
            # record the feasible mappings
            self.all_mappings.append(mappings)
            self.total_matchings += 1
            assert len(mappings) > 0
            self.total_mappings += len(mappings)
            app = at.MappingApplier(
                match_result, verbose=False, strict=self.transform_strict)
            self.appliers.append(app)
            
        self.sample = False
        self.selected_indices = self.sample_mappings()
        
        assert self.total_mappings > 0

        class ScheduleContext:
            def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
                self.schedule_gen = schedule_gen
                self.schedule_app = schedule_app
                self.sc_info = sc_info
                self.checker = checker
                self.generate_schedule = generate_schedule

        self.schedule_ctx_cls = ScheduleContext

        # global context for overall exploration
        # self.schedule_context_cache = {}
        self.best_value = at.MAX_FLOAT
        self.best_match_id = 0
        self.best_mapping_id = 0
        self.best_ctx = None
        self.best_params = None
        self.tic = None
        self.flag = True

    def sample_mappings(self):
        num_mappings_to_select = self.total_mappings
        if self.sample:
            if self.total_mappings > 19:
                num_mappings_to_select = int(0.3 * self.total_mappings)
            elif self.total_mappings > 7:
                num_mappings_to_select = 4
            else:
                num_mappings_to_select = 1

        all_indices = [(i, j) for i in range(len(self.all_mappings)) for j in range(len(self.all_mappings[i]))]

        if self.sample:
            selected_indices = random.sample(all_indices, num_mappings_to_select)
        else:
            selected_indices = all_indices
        return selected_indices
    
    def increase_probability(self, probabilities, increase_probability_rate):
        for key in probabilities:
            if random.random() < increase_probability_rate:
                probabilities[key] += increase_probability_rate / len(probabilities)
        
    def probability_select(self):
        assert len(self.all_results) > 0
        sorted_dict =  dict(sorted(self.all_results.items(), key=lambda item:item[1], reverse=False))

        initial_probability = 1.0
        total_value = sum(1.0 / value for value in self.all_results.values())

        probabilities = {}
        for key, value in sorted_dict.items():
            probabilities[key] = initial_probability * (1.0 / value / total_value)
        
        increase_probability_rate = 0.2
        self.increase_probability(probabilities, increase_probability_rate)
        N = int(0.2 * len(self.all_results))
        top_n = heapq.nlargest(N, probabilities.items(), key=lambda item: item[1])
        top_n_key = [item[0] for item in top_n]
        
        return top_n_key

    def Ge_search_policies(self):
        search_policy = "default"
        search_policy_params=None
        self.search_policies = make_search_policies(
            search_policy,
            search_policy_params,
            self.tasks,
            MEASURES_PER_ROUND,
            2,
            None,
            None,
            False,
        )
        return 

    def auto_scheduleV1(self, trials, measurer, log_file):
        self.log_name = log_file
        print("Autoscheduling %s..." % self.log_name, flush=True)
        self.total_trials += trials
        cost = at.MAX_FLOAT
        best_value = at.MAX_FLOAT
        if trials < self.least_trails:        
            print(
                f"[Warning] Too few trials, expect at least {self.least_trails} trials.",
                flush=True,
            )
            trials = self.least_trails
            print(
                f"Increase trials to {self.least_trails}.",
                flush=True,
            )
        else:
            print("Total trials:", trials, flush=True)
        self.trials_per_matching = trials
        
        print("repeat rounds:", self.repeat_rounds, flush=True)
        print("total_matchings:", self.total_matchings, flush=True)
        print("total_mappings:", self.total_mappings, flush=True)
        print("Initial trials per matching:", self.trials_per_matching, flush=True)
        self.tic = time.time()
        if self.flag:
            for round in range(self.repeat_rounds):
                for index in (self.selected_indices):
                    print(f"iteration {index}")
                    match_id, mapping_id = index
                    task, record = self._tune_task(index, measurer)
                    self.tasks.append(task)
                    self.records.append(record)
            self.Ge_search_policies()
            self.flag = False
        
        id = 0
        for index in (self.selected_indices):
            match_id, mapping_id = index
            print(f"index : {index}")
            if self.selected_key:
                if mapping_id not in self.selected_key:
                    print(f"have been discard mapping_id : {mapping_id}")
                    id += 1
                    continue
            print(f"choice mapping_id : {mapping_id}")
            measure_inputs, measure_results = self.search_policies[id].continue_search_one_round(self.trials_per_matching, measurer)

            for res in measure_results:
                cost = array_mean(res.costs)
                if cost < best_value:
                    best_value = cost
            record = self.tasks[id], self.records[id], measure_results, best_value
            self.tune_records[id] = record
            id += 1
        
        for key, value in self.tune_records.items():
            task, cur_record, measure_inputs, cost = value
            self.all_results[key] = cost
            # record the best
            if cost < self.best_value:
                self.best_value = cost
                self.best_task = task
                self.workload_key = task.workload_key
                self.best_mapping = str(cur_record)

        if(self.enable_pruning):
            self.selected_key = self.probability_select()
            print(f"self.selected_key : {self.selected_key}", flush=True)

        print(
            f"Best record value:{self.best_value} (smaller is better), best mapping: {self.best_mapping}", flush=True)
        print(f"Best task: {self.best_task}")
        print(f"Curent operation: {self.name} is finish tune.", flush=True)

        return measure_inputs, self.best_value, self.best_task
    
    def _tune_task(self, index, measurer):
        global WORKLOAD_FUNC_REGISTRY_STR
        match_id, mapping_id = index
        register_name = self.name + "_" + str(match_id) + "_" + str(mapping_id)
        match_result = self.all_matches[match_id]
        record = self.all_mappings[match_id][mapping_id]
        target = tvm.target.Target(self.measure_option.target)
        print("Current explored matching:",
                str(match_result), flush=True)
        print("Its axis mapping:", flush=True)
        print("Current explored mapping:", str(record), flush=True) 
        for i, v in match_result.axis_map.items():
                print(i.var, ":", [x.var for x in v], flush=True)
        if register_name not in WORKLOAD_FUNC_REGISTRY_STR:
            print(f"new register_name :{register_name}")
            # transform compute
            app = self.appliers[match_id]
            new_state = app.apply(record, drop_output=self.drop_output)
            # prepare schedulers
            schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
            checker = at.CUDAProgramChecker()
            # use ansor to find optimized params
            target_dag = schedule_app.target_dag
            inputs = target_dag.get_inputs()
            args = inputs + list(target_dag.tensors)
            GLOBAL_ARGS[register_name] = args
            WORKLOAD_FUNC_REGISTRY_STR[register_name] = register_name
            registered_func = auto_scheduler.register_workload(register_name, f=task_func)
            workload_key = make_workload_key(register_name, args=[register_name])
            dag = tvm.auto_scheduler.ComputeDAG(workload_key, hw_abs_dag=schedule_gen.hw_abs_dag_stage)
            task = auto_scheduler.SearchTask(
                compute_dag=dag, workload_key=workload_key, target=target
            )
        else:
            print(f"Error! Current function {register_name} is already register.")
        
        return task, record
    
    def get_best_schedule(self):
        best_file = os.path.join(
            self.log_dir, f"best_mappings_{self.name}.txt")

        if self.best_task is not None:
            inp, _ = tvm.auto_scheduler.load_best_record(self.log_name, self.workload_key)
            if inp is None:
                print("No autotensorize records in the log file.")            
            else:
                sch, args = self.best_task.apply_best(self.log_name)
                value = at.evaluate_schedule(sch, args, self.measure_option)
                print(f"Best task in get_best_schedule: {self.best_task}, best value: {value}")
                if value < self.existing_best_cost:
                    self.best_sch = sch
                    self.best_args = args
                    self.existing_best_cost = value
                if self.best_sch is not None:
                    with open(best_file, "w") as fout:
                        obj = {"mapping": self.best_mapping,"cost": self.existing_best_cost}
                        string = json.dumps(obj)
                        fout.write(string)
                        
        return self.best_sch, self.best_args, self.existing_best_cost

    def get_measure_opt(self):
        return self.measure_option

    
class AutoScheduleGraphDispatch(object):
    working_set = {}
    results = {}

    @classmethod
    def add_task(
        cls, name, top_log_dir, subgraph, measure_option, scheduler_option="auto_tensorize_v3"
    ):
        use_at = 0
        next_id = len(AutoScheduleGraphDispatch.working_set)
        if scheduler_option == "auto_tensorize_v3" or scheduler_option == "auto_tensorize":
            if AutoTensorizeContextV3.can_use(name, top_log_dir, subgraph, measure_option):
                ctx = AutoTensorizeContextV3(
                    name, top_log_dir, subgraph, measure_option)
                use_at = 1
            else:
                # fallback to TG
                print("Fallback to TG")
                ctx = TGAutoScheduleContext(
                    name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "amos_ansor_v1":
            print("amos_ansor_v1")
            if AmosAnsorContextV1.can_use(name, top_log_dir, subgraph, measure_option):
                ctx = AmosAnsorContextV1(
                    name, top_log_dir, subgraph, measure_option)
                use_at = 1
            else:
                print("Fallback to ansor")
                ctx = AmosAutoScheduleContextV1(
                    name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "amos_ansor_v2":
            print("amos_ansor_v2")
            if AmosAnsorContextV2.can_use(subgraph, measure_option):
                ctx = AmosAnsorContextV2(
                    name, top_log_dir, subgraph, measure_option)
                use_at = 1
            else:
                print("Fallback to ansor")
                ctx = AmosAutoScheduleContextV2(
                    name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "auto_tensorize_v2":
            if AutoTensorizeContextV2.can_use(name, top_log_dir, subgraph, measure_option):
                ctx = AutoTensorizeContextV2(
                    name, top_log_dir, subgraph, measure_option)
                use_at = 1
            else:
                # fallback to TG
                print("Fallback to TG")
                ctx = TGAutoScheduleContext(
                    name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "tg":
            ctx = TGAutoScheduleContext(
                name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "ansor":
            ctx = AnsorAutoScheduleContext(
                name, top_log_dir, subgraph, measure_option)
        else:
            raise RuntimeError("Unknown scheduler: %s" % scheduler_option)
        AutoScheduleGraphDispatch.working_set[next_id] = ctx
        sch, args, perf = ctx.get_best_schedule()
        # if sch is not None:
        #   perf = at.evaluate_schedule(
        #     sch, args, ctx.get_measure_opt(), new_process=True)
        # else:
        #   perf = at.MAX_FLOAT
        AutoScheduleGraphDispatch.results[next_id] = (sch, args, perf)
        return next_id, ctx, use_at

    @classmethod
    def remove_task(cls, task_id):
        if task_id in AutoScheduleGraphDispatch.working_set:
            del AutoScheduleGraphDispatch.working_set[task_id]

    @classmethod
    def auto_schedule(cls, selected_ids, trials_lst, task_scheduler, measurer, log_file):
        if not task_scheduler:
            for tid, trials in zip(selected_ids, trials_lst):
                if not trials:
                    continue
                if tid in AutoScheduleGraphDispatch.working_set:
                    ctx = AutoScheduleGraphDispatch.working_set[tid]
                    # if isinstance(ctx, TGAutoScheduleContext):
                    ctx.auto_schedule(trials)
                    sch, args, perf = ctx.get_best_schedule()
                    # if sch is not None:
                    #   perf = at.evaluate_schedule(
                    #     sch, args, ctx.get_measure_opt(), new_process=True)
                    # else:
                    #   perf = at.MAX_FLOAT
                    AutoScheduleGraphDispatch.results[tid] = (sch, args, perf)
        else:
            trials = trials_lst
            num_measures_per_round = min(MEASURES_PER_ROUND, trials // len(selected_ids))
            # task_cts[i] saves how many times task i is tuned
            task_cts = [0 for _ in range(len(selected_ids))]
            # task_best_cts[i] saves the round task i found the best latency
            task_best_cts = [0 for _ in range(len(selected_ids))]
            # task_costs_history[i] saves the latency history of task i
            task_costs_history = [[] for _ in range(len(selected_ids))]
            early_stopping_task = 1e20
            early_stopping_all = 1e20
            dead_tasks = set()
            tune_time = best_ct = best_score = 0
            best_costs = 1e10 * np.ones(len(selected_ids))
            cur_score = sum(best_costs)
            backward_window_size = 3
            beta = 2
            alpha = 0.2
            
            best_tasks = []
            
            task_tags = []  # task_id -> tag
            tag_to_group_id = {} # tag -> group_id
            flop_cts = [] # task_id -> the number of floating ops
            group_task_ids = []  # group_id -> all task ids in this group

            if num_measures_per_round <= 0:
                raise ValueError(
                    "num_measure_trials is too small. Please set it to a higher value."
                    f"It should be at least {len(selected_ids)} for this model."
                )
            # do a round robin first to warm up
            for tid in selected_ids:
                if not task_cts[tid]:
                    if tid in AutoScheduleGraphDispatch.working_set:
                        ctx = AutoScheduleGraphDispatch.working_set[tid]
                        measure_inputs, best_value, best_task = ctx.auto_scheduleV1(num_measures_per_round, measurer, log_file)
                        sch, args, perf = ctx.get_best_schedule()
                        AutoScheduleGraphDispatch.results[tid] = (sch, args, perf)
                        
                        task_cts[tid] += 1
                        best_tasks.append(best_task)
                        tune_time += len(measure_inputs)
                        if best_value < best_costs[tid]:
                            best_costs[tid] = best_value
                            task_best_cts[tid] = task_cts[tid]
                        # Stop tuning this task in the rest of the process if its search space has been
                        # fully explored or it has no improvement for a long while.
                        no_change_trials = (
                            task_cts[tid] - task_best_cts[tid]
                        )
                        if len(measure_inputs) == 0 or no_change_trials > early_stopping_task:
                            dead_tasks.add(tid)
                        
                        task_costs_history[tid].append(best_costs[tid])

            print(f"first round end...")
            cur_score = sum(best_costs)
            best_score = cur_score
            best_ct = tune_time
            # Build similarity groups
            for i, task in enumerate(best_tasks):
                tag = derive_similarity_tag(task.compute_dag)
                task_tags.append(tag)
                flop_cts.append(task.compute_dag.flop_ct)
                if not tag:
                    continue
                
                if tag not in tag_to_group_id:
                    tag_to_group_id[tag] = len(tag_to_group_id)
                    group_task_ids.append([])
                group_task_ids[tag_to_group_id[tag]].append(i)
            
            # use the specific strategy to choose workload to tune
            task_idx = -1
            strategy = "gradient"
            while tune_time < trials and len(dead_tasks) < len(selected_ids):
                if strategy == "round-robin":
                    task_idx = (task_idx + 1) % len(selected_ids)
                    while task_idx in dead_tasks:
                        task_idx = (task_idx + 1) % len(selected_ids)
                elif strategy == "gradient":
                    gradients = []
                    # {
                    for i in range(len(selected_ids)):
                        if i in dead_tasks:
                            gradients.append(0)
                            continue
                        
                        # compute gradient from chain rule : (delta f / delta g_i)
                        delta = 1e-4
                        new_costs = list(best_costs)
                        new_costs[i] -= delta        
                        chain_grad = (
                            sum(best_costs) - sum(new_costs)
                        ) / delta
                        # compute (g_i(t_i) - g(t_i - \Delta t)) / (\Delta t)
                        if (
                            task_cts[i] - 1 < len(task_costs_history[i])
                            and task_cts[i] - 1 - backward_window_size >= 0
                        ):
                            backward_grad = (
                                task_costs_history[i][task_cts[i] - 1]
                                - task_costs_history[i][
                                    task_cts[i] - 1 - backward_window_size
                                ]
                            ) / backward_window_size
                        else:
                            backward_grad = 0
                        # compute (g_i(t_i + \Delta t) - g(t_i)) / (\Delta t)
                        g_next_1 = best_costs[i] - (best_costs[i] / task_cts[i])
                        g_next_2 = beta * 1e30
                        group_id = tag_to_group_id.get(task_tags[i], None)
                        if group_id is not None and len(group_task_ids[group_id]) > 1:
                            best_flops = max(
                                [
                                    flop_cts[j] / best_costs[j]
                                    for j in group_task_ids[group_id]
                                ]
                            )
                            g_next_2 = beta * flop_cts[i] / best_flops
                        g_next = min(g_next_1, g_next_2)
                        forward_grad = g_next - best_costs[i]
                        # combine all grads
                        grad = chain_grad * (
                            alpha * backward_grad + (1 - alpha) * forward_grad
                        )
                        assert grad <= 0
                        gradients.append(grad)
                    
                    if max(gradients) == min(gradients):
                        task_idx = np.random.choice(len(gradients))
                    else:
                        task_idx = np.argmin(gradients)
                    # }
                if task_idx in AutoScheduleGraphDispatch.working_set:
                    ctx = AutoScheduleGraphDispatch.working_set[task_idx]
                    measure_inputs, best_value, best_task = ctx.auto_scheduleV1(num_measures_per_round, measurer, log_file)
                    sch, args, perf = ctx.get_best_schedule()
                    AutoScheduleGraphDispatch.results[task_idx] = (sch, args, perf)
                    
                    task_cts[task_idx] += 1
                    
                    tune_time += len(measure_inputs)
                    if best_value < best_costs[task_idx]:
                        best_costs[task_idx] = best_value
                        task_best_cts[task_idx] = task_cts[task_idx]
                    
                    task_costs_history[task_idx].append(best_costs[task_idx])
                    cur_score = sum(best_costs)
                    # Stop tuning this task in the rest of the process if its search space has been
                    # fully explored or it has no improvement for a long while.
                    no_change_trials = (
                        task_cts[task_idx] - task_best_cts[task_idx]
                    )
                    if len(measure_inputs) == 0 or no_change_trials > early_stopping_task:
                        dead_tasks.add(task_idx)

                    # _adjust_similarity_group
                    # adjust the similarity group for the selected task
                    group_id = tag_to_group_id.get(task_tags[task_idx], None)
                    if group_id is None or len(group_task_ids[group_id]) <= 1:
                        print(f"Don't need adjust.")
                    else:
                        group_ids = group_task_ids[group_id]
                        best_group_flops = max([flop_cts[j] / best_costs[j] for j in group_ids])
                        cur_flops = flop_cts[task_idx] / best_costs[task_idx]

                        # if we tune a task for many times but it still cannot achieve
                        # a similar speed to the fastest one in its group, this means this task
                        # is actually not similar to other tasks in its group.
                        # So we will remove it from its original group.
                        if cur_flops < best_group_flops / beta and task_cts[task_idx] > 5 + max(
                            task_cts[j] for j in group_ids if j != task_idx
                        ):
                            task_tags[task_idx] = None
                            group_ids.remove(task_idx)
                    
                    if cur_score < best_score:
                        best_score = cur_score
                        best_ct = tune_time
                    elif tune_time - best_ct >= early_stopping_all and all(
                        cost < 1e9 for cost in best_costs
                    ):
                        
                        print(
                            "Stop early since no performance improvement in the last "
                            + str(early_stopping_all)
                            + " measurement trials."
                        )
                        break

    @classmethod
    def query_schedule(cls, tid):
        if tid in AutoScheduleGraphDispatch.results:
            ctx = AutoScheduleGraphDispatch.working_set[tid]
            sch, args, perf = ctx.get_best_schedule()
            # if sch is not None:
            #   perf = at.evaluate_schedule(
            #     sch, args, ctx.get_measure_opt(), new_process=True)
            # else:
            #   perf = at.MAX_FLOAT
            print(
                "Query subgraph task id: %s, perf=%f ms after %d tuning"
                % (str(tid), perf, ctx.total_trials),
                flush=True,
            )
            return (sch, args, perf)
        else:
            return (None, None, at.MAX_FLOAT)


class AutoScheduleMultiGraphContext(object):
    def __init__(
        self,
        name,
        tir_multi_graph,
        measure_option,
        scheduler_option="auto_tensorize_v3",
        gamma=0.02,
        trials=100,
        policy="equal",
    ):
        self.tir_multi_graph = tir_multi_graph
        self.performance_trace = {}
        self.schedules = {}
        self.contexts = {}
        self.graph_tag_to_tid = {}
        self.C = {}
        self.alpha = {}
        self.beta = {}
        self.X = {}
        self.gamma = gamma
        self.use_at_set = set()
        self.subgraph_count = {}
        self.log_dir = name
        self.scheduler_option = scheduler_option
        self.log_file = "%s.json" % (self.log_dir)
        
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        graphs = tg.get_graphs_from_tir_multi_graph(tir_multi_graph)
        graphs = OrderedDict(
            sorted([(x.value, y) for x, y in graphs.items()], key=lambda x: x[0]))
        for key, subgraph in graphs.items():
            new_name = name + "_subgraph" + str(key)
            if subgraph.tag in self.graph_tag_to_tid:
                self.subgraph_count[subgraph.tag] += 1
                continue
            else:
                self.subgraph_count[subgraph.tag] = 1
            tid, ctx, use_at = AutoScheduleGraphDispatch.add_task(
                new_name, self.log_dir, subgraph, measure_option, scheduler_option=scheduler_option
            )
            if use_at:
                self.use_at_set.add(subgraph.tag)
            sch, args, perf = AutoScheduleGraphDispatch.query_schedule(tid)
            self.performance_trace[tid] = [perf]
            self.C[tid] = perf
            self.alpha[tid] = perf / (32)
            self.beta[tid] = 1.0
            self.X[tid] = self.calculate_X(tid)
            self.schedules[tid] = (sch, args)
            self.contexts[tid] = ctx
            self.graph_tag_to_tid[subgraph.tag] = tid
        self.L = len(self.graph_tag_to_tid) * trials
        self.trials = trials
        self.policy = policy

    def calculate_X(self, tid):
        raw = math.sqrt(self.C[tid] / (self.alpha[tid] + 1e-10))
        return raw

    def select_next_tasks(self):
        # this is the decision part, currently use the simple decision
        ret = []
        trials = []
        sum_X = reduce(lambda x, y: x + y, self.X.values(), 0.0)

        for tid, lst in self.performance_trace.items():
            if self.policy == "equal":
                ret.append(tid)
                trials.append(self.trials)
            elif self.policy == "rebalance":
                ret.append(tid)
                raw = int(max(1, min(self.X[tid] * self.L / sum_X, self.L)))
                trials.append(raw)
                diff = 2 * (self.C[tid] / (self.alpha[tid]
                            * raw) + self.beta[tid] - lst[-1])
                self.alpha[tid] = max(
                    1e-5,
                    self.alpha[tid]
                    + self.gamma * diff *
                    self.C[tid] / (raw * self.alpha[tid] * self.alpha[tid]),
                )
                self.beta[tid] = max(1e-5, self.beta[tid] - self.gamma * diff)
                self.C[tid] = lst[-1]
                self.X[tid] = self.calculate_X(tid)

        return ret, trials

    def auto_schedule(self):
        tids, trials = self.select_next_tasks()
        task_scheduler = False
        if self.scheduler_option == "amos_ansor_v2":
            trials = self.L
            task_scheduler = True
        
        num_measures_per_round = MEASURES_PER_ROUND
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials= num_measures_per_round,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
            runner=measure_ctx.runner,
            verbose=2,
        )
        measurer = ProgramMeasurer(
            tune_option.builder,
            tune_option.runner,
            tune_option.measure_callbacks,
            tune_option.verbose,
        )

        AutoScheduleGraphDispatch.auto_schedule(tids, trials, task_scheduler, measurer, self.log_file)
        for k, lst in self.performance_trace.items():
            sch, args, perf = AutoScheduleGraphDispatch.query_schedule(k)
            self.schedules[k] = (sch, args)
            lst[-1] = perf  # only reserve one
        print("auto_schedule end...")

    def get_schedules(self):
        total = 0
        mapped = 0
        for k, v in self.subgraph_count.items():
            total += v
            if k in self.use_at_set:
                mapped += v
        print(
            "[NOTICE] totally",
            total,
            "subgraphs, mapped",
            mapped,
            "subgraphs, ratio=",
            mapped / total * 100.0,
            "%",
        )
        ret = {}
        graphs = tg.get_graphs_from_tir_multi_graph(self.tir_multi_graph)
        graphs = OrderedDict({x.value: y for x, y in graphs.items()})
        for key, subgraph in graphs.items():
            tid = self.graph_tag_to_tid[subgraph.tag]
            sch, args = self.schedules[tid]
            ret[key] = tg.ScheduleTensors(sch, args)
        return ret

    def ready(self):
        graphs = tg.get_graphs_from_tir_multi_graph(self.tir_multi_graph)
        graphs = OrderedDict({x.value: y for x, y in graphs.items()})
        for key, subgraph in graphs.items():
            tid = self.graph_tag_to_tid[subgraph.tag]
            sch, args = self.schedules[tid]
            if sch is None or args is None:
                print(f"tid = {tid}")
                print(f"sch : {sch}")
                print(f"args: {args}")
                return False
        return True


class AutoScheduleMultiGraphDispatch(object):
    working_set = {}

    @classmethod
    def add_graph_task(
        cls,
        name,
        tir_multi_graph,
        measure_option,
        scheduler_option="auto_tensorize_v3",
        trials=100,
        policy="equal",
    ):
        next_id = len(AutoScheduleMultiGraphDispatch.working_set)
        AutoScheduleMultiGraphDispatch.working_set[next_id] = AutoScheduleMultiGraphContext(
            name,
            tir_multi_graph,
            measure_option,
            scheduler_option=scheduler_option,
            trials=trials,
            policy=policy,
        )
        return next_id

    @classmethod
    def auto_schedule(cls, tid):
        assert tid in cls.working_set
        ctx = cls.working_set[tid]
        ctx.auto_schedule()

    @classmethod
    def get_schedules(cls, tid):
        assert tid in cls.working_set
        ctx = cls.working_set[tid]
        return ctx.get_schedules()

    @classmethod
    def ready(cls, tid):
        assert tid in cls.working_set
        ctx = cls.working_set[tid]
        return ctx.ready()
