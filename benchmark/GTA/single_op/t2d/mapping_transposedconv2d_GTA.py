import tvm
import os
import time
import tempfile
import shutil
import numpy as np
import random
import argparse
# from tvm import testing
from tvm import auto_tensorize as at
from tvm.contrib import tar, ndk
from tvm import auto_scheduler
from tvm.ir import transform
from tvm.driver import build_module
from tvm.runtime import Object, module, ndarray
import multiprocessing as multi
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from tvm import tg
from collections import OrderedDict

from tvm.auto_scheduler.compute_dag import ComputeDAG
from tvm.auto_scheduler.workload_registry import make_workload_key
from tvm.auto_scheduler.task_scheduler import make_search_policies
from tvm.topi.testing import conv2d_nchw_python
from tvm.auto_scheduler.measure import ProgramMeasurer
from tvm.auto_scheduler.utils import array_mean
import copy
import random
import heapq
TRIALS_PER_WORKLOAD = 48
WORKLOAD_FUNC_REGISTRY_STR = {}
GLOBAL_ARGS = {}
def task_func(func_name):
    return GLOBAL_ARGS.get(func_name, "GTA_schedule")

def zero_expand2d(inputs, stride=1):
    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    assert isinstance(stride, tuple), "type(stride)={}".format(type(stride))
    assert len(stride) == 2

    expand_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    out_height = (height - 1) * stride[0] + 1
    out_width = (width - 1) * stride[1] + 1
    return tvm.te.compute(
        (batch_size, in_channel, out_height, out_width),
        lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(h % stride[0] == 0, w % stride[1] == 0),
            inputs[b, c, h // stride[0], w // stride[1]],
            expand_zero,
        ),
    )


def zero_pad2d(inputs, padding=0):
    padding = (
        (padding, padding, padding, padding)
        if isinstance(padding, (int, tvm.tir.IntImm))
        else padding
    )
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert len(padding) == 4

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(
                h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]
            ),
            inputs[b, c, h - padding[0], w - padding[2]],
            padding_zero,
        ),
        name="Padding",
    )


def transposed_conv2d(
    N, C, H, W, K, kernel_size, stride, padding, output_padding, dilation, in_dtype, out_dtype
):
    A = tvm.te.placeholder([N, K, H, W], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, C, kernel_size, kernel_size], dtype=in_dtype, name="B")
    batch_size, input_channel, in_h, in_w = A.shape
    input_channel_w, output_channel, k_h, k_w = B.shape

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    output_padding = (
        (output_padding, output_padding)
        if isinstance(output_padding, (int, tvm.tir.IntImm))
        else output_padding
    )
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(output_padding, tuple) and len(output_padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    kernel_h = (k_h - 1) * dilation[0] + 1
    kernel_w = (k_w - 1) * dilation[1] + 1
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    rc = tvm.te.reduce_axis((0, input_channel), name="rc")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")

    expanded = zero_expand2d(A, stride=stride)
    padded = zero_pad2d(
        expanded,
        padding=(
            kernel_h - 1 - padding[0],
            kernel_h - 1 - padding[0] + output_padding[0],
            kernel_w - 1 - padding[1],
            kernel_w - 1 - padding[1] + output_padding[1],
        ),
    )
    output = tvm.te.compute(
        (batch_size, output_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (
                padded[b, rc, h + rh * dilation[0], w + rw * dilation[1]]
                * B[rc, c, k_h - rh - 1, k_w - rw - 1]
            ).astype(out_dtype),
            axis=[rc, rw, rh],
        ),
    )
    return [A, B, output]

def mapping_tensorcore(
    N,
    C,
    H,
    W,
    K,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
    sample=False,
):

    A, B, Conv = transposed_conv2d(
        N, C, H, W, K, kernel_size, stride, padding, output_padding, dilation, in_dtype, out_dtype
    )
    log_dir = "transposedconv2d-%s-%s-layer-%s" % (in_dtype, out_dtype, layer)
    log_file = "transposedconv2d-%s-%s-layer-%s-GTA.log" % (in_dtype, out_dtype, layer)
    target_device = "cuda"
    target_dag = at.compute_dag_from_tensors([Conv])
    match_results = at.get_match_results(target_dag, target_device)
    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" % target_device, flush=True)
        return
    else:
        print("Possible matchings:", flush=True)
        for i, m in enumerate(match_results):
            print(i, ":", str(m), flush=True)
    
    
    measure_opt = at.MeasureOptions(target=target_device, timeout=10, number=200, min_repeat_ms=500)
    target = tvm.target.Target(measure_opt.target)
    
    all_matches = []
    all_mappings = []
    appliers = []
    best_cost = 1e10
    total_matchings = 0
    total_mappings = 0
    repead_rounds = 1

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    measurer = ProgramMeasurer(
        tune_option.builder,
        tune_option.runner,
        tune_option.measure_callbacks,
        tune_option.verbose,
    )
    global WORKLOAD_FUNC_REGISTRY_STR
    global GLOBAL_ARGS
    tasks = []
    search_policies = []
    match_result, _ = at.policy.all_fit(match_results)
    match_results = [match_result]
    for match_result in match_results:
        transform_strict = True
        all_matches.append(match_result)
        gen = at.MappingGenerator(match_result)
        mappings = gen.get_all()
        # filter out infeasible mappings
        feasible_mappings = []
        tmp_app = at.MappingApplier(match_result, strict=transform_strict)
        for mapping in mappings:
            try:
                tmp_app.apply(mapping, drop_output=False)
                feasible_mappings.append(mapping)
            except RuntimeError as e:
                pass
        if len(feasible_mappings) == 0:
            transform_strict = False
        else:
            mappings = feasible_mappings
        all_mappings.append(mappings)
        total_matchings += 1
        assert len(mappings) > 0
        total_mappings += len(mappings)
        app = at.MappingApplier(match_result, verbose=False, strict=transform_strict)
        appliers.append(app)

    selected_indices = sample_mappings(total_mappings, all_mappings, sample)
    print(f"length of selected_indices = {len(selected_indices)}")
    print("total_matchings:", total_matchings, flush=True)
    for round in range(repead_rounds):
        for index in selected_indices:
            match_id, mapping_id = index
            match_result = all_matches[match_id]
            app = appliers[match_id]

            print("Current explored matching:",
                str(match_result), flush=True)
            print("Its axis mapping:", flush=True)
            for i, v in match_result.axis_map.items():
                print(i.var, ":", [x.var for x in v], flush=True)

            record = all_mappings[match_id][mapping_id]
            print("Current explored mapping:", str(record), flush=True)

            new_state = app.apply(record, drop_output=False)

            schedule_gen = at.LLVMScheduleGenerator(match_result, new_state)

            sc_info = schedule_gen.get_schedule_compute_info()

            schedule_app = at.LLVMScheduleApplier(match_result, sc_info)

            target_dag = schedule_app.target_dag

            inputs = target_dag.get_inputs()
            args = inputs + list(target_dag.tensors)
            register_name = log_dir + "_" + str(match_id) + "_" + str(mapping_id)

            if register_name not in WORKLOAD_FUNC_REGISTRY_STR:
                GLOBAL_ARGS[register_name] = args
                print(f"new register_name :{register_name}")
                WORKLOAD_FUNC_REGISTRY_STR[register_name] = register_name
                registered_func = auto_scheduler.register_workload(
                    register_name, f=task_func
                )

                workload_key = make_workload_key(register_name, args=[register_name])

                print(f"workload_key: {workload_key}",flush=True)
                dag = ComputeDAG(workload_key, hw_abs_dag=schedule_gen.hw_abs_dag_stage)

                task = auto_scheduler.SearchTask(
                    compute_dag=dag, workload_key=workload_key, target=target
                )

                tasks.append(task)



    total_trails = len(selected_indices)*trials
    turn_trails = 0
    all_results = {}
    selected_key = {}
    best_task = None
    trails_per_workload = min(trials, TRIALS_PER_WORKLOAD)
    search_policy = "default"
    search_policy_params=None
    search_policies = make_search_policies(
        search_policy,
        search_policy_params,
        tasks,
        trails_per_workload,
        2,
        None,
        None,
        False,
    )
    for id in range(len(selected_indices)):
        all_results[id] = 1e20

    while turn_trails < total_trails:
        id = 0
        for index in (selected_indices):
            match_id, mapping_id = index
            if selected_key:
                if mapping_id not in selected_key:
                    print(f"have been discard mapping_id : {mapping_id}")
                    id += 1
                    turn_trails += trails_per_workload
                    continue
            print(f"current search_policy id : {id}")
            measure_inputs, measure_results = search_policies[id].continue_search_one_round(trails_per_workload, measurer=measurer)
            sch, args = tasks[id].apply_best(log_file)
            value = at.evaluate_schedule(sch, args, measure_opt)
            print(f"Temp evaluate_schedule value: {value}")
            turn_trails += len(measure_inputs)


            if value < all_results[id]:
                print(f"current best_value : {value}")
                print(f"all_results[{id}]: {all_results[id]}")
                all_results[id] = value
            id += 1


        selected_key = probability_select(all_results)
        print(f"selected_key : {selected_key}", flush=True)

    best_cost = 1e10
    for index, value in all_results.items():
        if value < best_cost:
            best_cost = value
            print(f"index of best_cost = {index}, best_cost = {best_cost}")
            # best_id = index
    print("Cost of %s is %f ms" % (log_dir, best_cost))
    return best_cost


def increase_probability(probabilities, increase_probability_rate):
        for key in probabilities:
            if random.random() < increase_probability_rate:
                probabilities[key] += increase_probability_rate / len(probabilities)

def probability_select(all_results):
    assert len(all_results) > 0
    print(f"all_results: {all_results}")
    sorted_dict =  dict(sorted(all_results.items(), key=lambda item:item[1], reverse=False))
    initial_probability = 1.0
    total_value = sum(1.0 / value for value in all_results.values())
    probabilities = {}
    for key, value in sorted_dict.items():
        probabilities[key] = initial_probability * (1.0 / value / total_value)

    increase_probability_rate = 0.1  
    increase_probability(probabilities, increase_probability_rate)
    if len(all_results) > 1:
        N = int(0.2 * len(all_results))
    else:
        N = len(all_results)
    print(f"select size = {N}")

    top_n = heapq.nlargest(N, probabilities.items(), key=lambda item: item[1])
    for key, value in top_n:
        print(f"{key}: {value:.6f}")
    top_n_key = [item[0] for item in top_n]

    return top_n_key


def sample_mappings(total_mappings, all_mappings, sample):
    num_mappings_to_select = total_mappings

    all_indices = [(i, j) for i in range(len(all_mappings)) for j in range(len(all_mappings[i]))]

    if(sample):
        selected_indices = random.sample(all_indices, 2)
    else:
        selected_indices = all_indices
    return selected_indices
    

_ = None
#  (  N,   C,   H,   W,   K, kernel_size, stride, padding, dilation)
shapes_b1 = [
    (_, 3, 112, 112, 64, 3, 7, 3, 1),  # stem
    (_, 64, 56, 56, 64, 3, 1, 1, 1),  # layer1 x 4
    (_, 64, 56, 56, 128, 1, 2, 0, 1),  # layer2 downsample
    (_, 64, 56, 56, 128, 3, 2, 1, 1),  # layer2
    (_, 128, 28, 28, 128, 3, 1, 1, 1),  # layer2 x 3
    (_, 128, 28, 28, 256, 1, 2, 0, 1),  # layer3 downsample
    (_, 128, 28, 28, 256, 3, 2, 1, 1),  # layer3
    (_, 256, 14, 14, 256, 3, 1, 1, 1),  # layer3 x 3
    (_, 256, 14, 14, 512, 1, 2, 0, 1),  # layer4 downsample
    (_, 256, 14, 14, 512, 3, 2, 1, 1),  # layer4
    (_, 256, 7, 7, 512, 3, 1, 1, 1),  # layer4 x 3
]

supported_dtypes = set(
    [
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("int4", "int32"),
        ("int8", "int32"),
    ]
)

example_text = """
 example:
    python mapping_transposedconv2d_GTA.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_transposedconv2d_GTA.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_transposedconv2d_GTA.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--in_dtype",
        type=str,
        choices=["float16", "float32", "float64", "bfloat16", "int4", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        choices=["float16", "float32", "float64", "int32"],
        default="float16",
    )
    parser.add_argument("--begin", type=int, choices=list(range(len(shapes_b1))), default=0)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes_b1) + 1)), default=len(shapes_b1)
    )
    parser.add_argument("--trials", type=int, default=-1)

    args = parser.parse_args()
    batches = [args.batch]
    beg = args.begin
    num = args.num
    assert (
        args.in_dtype,
        args.out_dtype,
    ) in supported_dtypes, (
        f"The desired dtype pair {(args.in_dtype, args.out_dtype)} is not supported by Tensor Core."
    )
    for batch in batches:
        costs = []
        for i, shape in enumerate(shapes_b1[beg : beg + num]):
            (N, C, H, W, K, kernel_size, stride, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, kernel_size, stride, padding, dilation)
            layer_name = f"({N},{C},{H},{W},{K},{kernel_size},{stride},{padding},{dilation})"
            try:
                cost = mapping_tensorcore(
                    N,
                    C,
                    H,
                    W,
                    K,
                    kernel_size,
                    stride,
                    padding,
                    0,
                    dilation,
                    layer_name,
                    args.in_dtype,
                    args.out_dtype,
                    trials=args.trials,
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
