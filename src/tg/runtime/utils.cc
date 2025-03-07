#include "utils.h"
#include "tvm/runtime/profiling.h"

namespace tvm {

namespace tg {

Array<FloatImm> evaluate_graph(
  TIRMultiGraph multi_graph,
  Map<IntImm, ScheduleTensors> graph_sch_tensors_,
  Target target,
  int dev_id,
  int number) {
  std::unordered_map<IntKey, ScheduleTensors> graph_sch_tensors;
  for (auto kv : graph_sch_tensors_) {
    graph_sch_tensors[IntKey(kv.first->value)] = kv.second;
  }
  DLDevice ctx;
  if (target->kind->name == "cuda") {
    ctx = DLDevice({kDLCUDA, dev_id});
  } else if (target->kind->name == "llvm") {
    ctx = DLDevice({kDLCPU, dev_id});
  } else {
    ERROR << "Currently only support CUDA/LLVM but get " << target->kind->name << ".";
  }

  std::vector<IntKey> order;

  std::unordered_map<IntKey, int> call_order;
  std::set<IntKey> free_set;
  for (auto kv : multi_graph->graph_attrs) {
    call_order[kv.first] = kv.second->num_predecessor;
    if (kv.second->num_predecessor == 0) {
      free_set.insert(kv.first);
    }
  }

  while (!free_set.empty()) {
    std::unordered_set<IntKey> update_set;
    for (auto k : free_set) {
      order.push_back(k);
      for (auto v : multi_graph.Self()->graph_attrs[k]->successors) {
        call_order[v] -= 1;
        if (call_order[v] == 0) {
          update_set.insert(v);
        }
      }
    }
    free_set.clear();
    for (auto k : update_set) {
      free_set.insert(k);
    }
  }

  std::unordered_map<IntKey, std::vector<runtime::NDArray>> arrays;
  std::unordered_map<IntKey, IntKey> graph_keys;
  std::unordered_map<std::string, IntKey> unique_graphs;
  std::unordered_map<IntKey, runtime::PackedFunc> functions;

  for (auto k : order) {
    ASSERT(graph_sch_tensors.count(k));
    auto subgraph = multi_graph->graphs.at(k);
    if (unique_graphs.count(subgraph->tag)) {
      graph_keys[k] = unique_graphs.at(subgraph->tag);
      continue;
    }
    ScheduleTensors sch_tensor = graph_sch_tensors.at(k);
    // prepare arrays
    std::vector<runtime::NDArray> array;
    for (auto t : sch_tensor->tensors) {
      std::vector<int64_t> shape;
      for (auto p : t->shape) {
        shape.push_back(get_const_int(p));
      }
      // currently use Empty array, this should give similar results to random fill
      array.push_back(runtime::NDArray::Empty(shape, t->dtype, ctx));
    }
    arrays[k] = array;

    // prepare functions
    const auto* f = runtime::Registry::Get("tg.autoschedule.build_func");
    ASSERT(f != nullptr) << "Can't find tg.autoschedule.build_func";
    runtime::Module ret = (*f)(
      sch_tensor->schedule, sch_tensor->tensors, target,
      Target("llvm"), "main", Map<te::Tensor, tir::Buffer>()
    );
    runtime::PackedFunc func = ret->GetFunction("main");
    functions[k] = func;
    graph_keys[k] = k;
    unique_graphs[subgraph->tag] = k;
  }
  auto* call_unpack = new CallFunc<tvm::runtime::PackedFunc, tvm::runtime::NDArray>();
  Array<FloatImm> ret;
  // sync device
  runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
  for (int ad = 0; ad <= number; ++ad) {    
    auto beg = std::chrono::steady_clock::now();
    for (auto k : order) {
      IntKey real_key = graph_keys.at(k);
      (*call_unpack)(functions[real_key], arrays[real_key]);
    }
    runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
    if (ad == 0) {
      // skip the first one
      continue;
    }
    auto end = std::chrono::steady_clock::now();
    double execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() / 1e3;
    ret.push_back(FloatImm(DataType::Float(32), execution_time));
  }  // for ad
  
  return ret;
}


TVM_REGISTER_GLOBAL("tg.evaluate_graph")
.set_body_typed([](
  TIRMultiGraph multi_graph,
  Map<IntImm, ScheduleTensors> graph_sch_tensors,
  Target target,
  int dev_id,
  int number){
  return evaluate_graph(multi_graph, graph_sch_tensors, target, dev_id, number);
});


}  // namespace tg


}  // namespace tvm