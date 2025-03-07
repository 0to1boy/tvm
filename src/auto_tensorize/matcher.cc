#include <tvm/auto_tensorize/matcher.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include "../tg/autodiff/arg_util.h"

namespace tvm {
namespace auto_tensorize {

Array<IterVar> HwAbsDAGMatcher::_extract_axes_from_op(const ComputeOpNode* op,
                                                       bool include_reduce) {
  Array<IterVar> axes;
  for (IterVar axis : op->axis) axes.push_back(axis);
  for (IterVar axis : op->reduce_axis) axes.push_back(axis);
  return std::move(axes);
}

bool HwAbsDAGMatcher::_check_elemwise(const ComputeOpNode* op, Array<Array<PrimExpr>>& indices) {
  if (op->reduce_axis.size() != 0) return false;
  Array<IterVar> spatial_axes = _extract_axes_from_op(op, false);
  size_t n_axes = spatial_axes.size();
  for (Array<PrimExpr> buf_idx : indices) {
    if (buf_idx.size() != n_axes) return false;
    for (size_t i = 0; i < n_axes; ++i) {
      // const IterVarNode* ptr = buf_idx[i].as<IterVarNode>();
      // if (ptr == nullptr) return false;
      if (!spatial_axes[i]->var.same_as(buf_idx[i])) return false;
    }
  }
  return true;
}

Map<IterVar, Range> HwAbsDAGMatcher::_infer_bounds(Operation out) {
  Array<Operation> out_ops{out};
  Schedule sch = create_schedule(out_ops);
  sch = sch.normalize();
  Map<IterVar, Range> bounds = InferBound(sch);
  return bounds;
}

MatchResult HwAbsDAGMatcher::match(Tensor target, Tensor intrin, Operation main_hw_abs) {
  auto target_bounds = _infer_bounds(target->op);
  auto intrin_bounds = _infer_bounds(intrin->op);
  bool success = _match(target, intrin, main_hw_abs, target_bounds, intrin_bounds);
  return success ? this->results : MatchResult();
}

bool HwAbsDAGMatcher::_match(Tensor target, Tensor intrin, Operation main_hw_abs,
                              Map<IterVar, Range> target_bounds,
                              Map<IterVar, Range> intrin_bounds) {
  if (target->dtype != intrin->dtype) {
    return false;
  }
  const ComputeOpNode* target_op = target->op.as<ComputeOpNode>();
  const ComputeOpNode* intrin_op = intrin->op.as<ComputeOpNode>();
  if (intrin_op == nullptr) {
    // const PlaceholderOpNode* target_op = target->op.as<PlaceholderOpNode>();
    const PlaceholderOpNode* intrin_op = intrin->op.as<PlaceholderOpNode>();
    CHECK(intrin_op != nullptr) << "Intrin tensor is neither from a ComputeOp "
                                << "nor a PlaceholderOp" << intrin << ".";
    // return target_op != nullptr;
    return true;
  }

  const PrimExpr target_expr = target_op->body[target->value_index];
  const PrimExpr intrin_expr = intrin_op->body[intrin->value_index];

  if (intrin->op.same_as(main_hw_abs)) {

    Array<IterVar> intrin_axes = _extract_axes_from_op(intrin_op);
    Array<IterVar> target_axes = _extract_axes_from_op(target_op);
    HwAbsExprMatcher expr_matcher(buffer_map);
    Array<IterVarMap> possible_index_mappings;
    possible_index_mappings = expr_matcher.match(target_expr, intrin_expr, target_axes, intrin_axes,
                                                 target_bounds, intrin_bounds);
    if (possible_index_mappings.size() == 0) {  // expr matching failed
      return false;
    }
    results.Set(target->op, possible_index_mappings);
  } else {

    HwAbsExprMatcher expr_matcher(buffer_map);
    Array<Array<PrimExpr>> target_indices, intrin_indices;
    expr_matcher.extract_indices(target_expr, intrin_expr, target_indices, intrin_indices);

    CHECK(_check_elemwise(intrin_op, intrin_indices));
    if (!_check_elemwise(target_op, target_indices)) {
      return false;
    }
  }

  Array<Tensor> target_input_tensors = target_op->InputTensors();
  Array<Tensor> intrin_input_tensors = intrin_op->InputTensors();
  if (target_input_tensors.size() != intrin_input_tensors.size()) {
    return false;
  }
  size_t num_inputs = intrin_input_tensors.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    Tensor target_input_tensor = target_input_tensors[i];
    Tensor intrin_input_tensor = intrin_input_tensors[i];
    bool success = _match(target_input_tensor, intrin_input_tensor, main_hw_abs, target_bounds,
                          intrin_bounds);
    if (!success) return false;
  }

  return true;
}

void HwAbsExprMatcher::extract_indices(PrimExpr target, PrimExpr intrin,
                                         Array<Array<PrimExpr>>& target_indices,
                                         Array<Array<PrimExpr>>& intrin_indices) {
  VisitExpr(target, intrin);
  _check_intrin_const_dim();
  for (Array<PrimExpr> i : this->target_indices) {
    target_indices.push_back(i);
  }
  for (Array<PrimExpr> i : this->intrin_indices) {
    intrin_indices.push_back(i);
  }
}

void HwAbsExprMatcher::_check_intrin_const_dim() {
  bool has_const_dim = false;
  for (auto index : intrin_indices) {
    for (auto i : index) {
      if (is_const_int(i)) {
        has_const_dim = true;
      }
    }
  }
  CHECK(!has_const_dim);
}

Array<IterVarMap> HwAbsExprMatcher::match(PrimExpr target, PrimExpr intrin,
                                            Array<IterVar>& target_axes,
                                            Array<IterVar>& intrin_axes,
                                            Map<IterVar, Range> target_bounds,
                                            Map<IterVar, Range> intrin_bounds) {
  bool structure_match = VisitExpr(target, intrin);  // buffer and op
  _check_intrin_const_dim();
  if (!structure_match) {
    return Array<IterVarMap>();
  }
  Array<IterVarMap> possible_index_mappings;
  IndexExprMatcher index_matcher;
  possible_index_mappings = index_matcher.match(target_indices, intrin_indices, target_axes,
                                                intrin_axes, target_bounds, intrin_bounds);
  if (possible_index_mappings.size() == 0) {
    return Array<IterVarMap>();
  } else {
    return possible_index_mappings;
  }
}

Array<IterVarMap> generate_mappings(Array<IterVar> target_axes, Array<IterVar> intrin_axes) {
  size_t n = target_axes.size(), r = intrin_axes.size();
  if (n < r) return Array<IterVarMap>();

  std::vector<IterVar> target_axes_vec;
  for (auto axis : target_axes) target_axes_vec.push_back(axis);

  auto comp = [](const IterVar& x, const IterVar& y) { return x.get() < y.get(); };
  std::sort(target_axes_vec.begin(), target_axes_vec.end(), comp);

  Array<IterVarMap> all_itervar_mappings;

  std::vector<bool> selector(n);
  std::fill(selector.begin(), selector.begin() + r, true);

  do {
    std::vector<IterVar> comb;
    for (size_t i = 0; i < n; ++i) {
      if (!selector[i]) continue;
      comb.push_back(target_axes_vec[i]);
    }

    do {
      IterVarMap itervar_map;
      for (size_t i = 0; i < r; ++i) {
        // Need to match the axis type
        if (comb[i]->iter_type == intrin_axes[i]->iter_type)
        {
          itervar_map.Set(comb[i], intrin_axes[i]);
        }
         
      }
      all_itervar_mappings.push_back(itervar_map);
    } while (std::next_permutation(comb.begin(), comb.end(), comp));
  } while (std::prev_permutation(selector.begin(), selector.end()));

  return std::move(all_itervar_mappings);
}

bool IndexExprMatcher::_verify_index(Array<PrimExpr> target_idx, Array<PrimExpr> intrin_idx) {
  tg::CheckExprEqual check_equal(true, true);
  size_t n_dim_target = target_idx.size();
  size_t n_dim_intrin = intrin_idx.size();

  for (size_t j = 0; j < n_dim_intrin; ++j) {
    PrimExpr intrin_i = intrin_idx[j];
    bool i_matched = false;
    for (size_t k = 0; k < n_dim_target; ++k) {
      PrimExpr target_i = target_idx[k];
      // for relaxed matching, the order is important
      // target index is more general than intrin index
      if (check_equal(target_i, intrin_i)) {
        target_idx.Set(k, make_zero(target_idx[0].dtype()));
        i_matched = true;
        break;
      }
    }
    if (!i_matched) {
      return false;
    }
  }

  for (PrimExpr i : target_idx) {
    if (!is_const_int(i)) {
      return false;
    } else if (!i.as<IntImmNode>()->value == 0) {
      std::cout << "Warning: found a non-zero constant in target_idx" << std::endl;
      std::cout << "target_idx: " << target_idx << std::endl;
      std::cout << "intrin_idx: " << intrin_idx << std::endl;
    }
  }

  return true;
}

bool IndexExprMatcher::_verify_indices(Array<Array<PrimExpr>> target_indices,
                                      Array<Array<PrimExpr>> intrin_indices) {
  size_t n_indices_intrin = intrin_indices.size();

  for (size_t i = 0; i < n_indices_intrin; ++i) {
    Array<PrimExpr> target_idx = target_indices[i];
    Array<PrimExpr> intrin_idx = intrin_indices[i];
    if (!_verify_index(target_idx, intrin_idx)) {
      return false;
    }
  }
  return true;
}

Array<Array<PrimExpr>> IndexExprMatcher::_rewrite_indices(Array<Array<PrimExpr>> indices,
                                                          IterVarMap itervar_map,
                                                          Map<IterVar, Range> target_bounds,
                                                          Map<IterVar, Range> intrin_bounds) {
  IterVarRewriter itervar_rewriter(itervar_map, target_bounds);
  size_t n_indices = indices.size();
  auto simplify = [](const PrimExpr& x) { return arith::Analyzer().Simplify(x); };

  for (size_t i = 0; i < n_indices; ++i) {
    Array<PrimExpr> idx = indices[i];
    size_t n_dim = idx.size();
    for (size_t j = 0; j < n_dim; ++j) {
      PrimExpr mod_i = simplify(itervar_rewriter.VisitExpr(idx[j]));
      idx.Set(j, mod_i);
    }
    indices.Set(i, idx);
  }
  return std::move(indices);
}

Array<IterVarMap> IndexExprMatcher::match(Array<Array<PrimExpr>> target_indices,
                                          Array<Array<PrimExpr>> intrin_indices,
                                          Array<IterVar>& target_axes, Array<IterVar>& intrin_axes,
                                          Map<IterVar, Range> target_bounds,
                                          Map<IterVar, Range> intrin_bounds) {
  CHECK(target_indices.size() == intrin_indices.size());
  Array<IterVarMap> possible_itervar_mappings;
  Array<IterVarMap> all_itervar_mappings = generate_mappings(target_axes, intrin_axes); 

  for (IterVarMap itervar_map : all_itervar_mappings) {
    auto modified_target_indices =
        _rewrite_indices(target_indices, itervar_map, target_bounds, intrin_bounds);

    if (_verify_indices(modified_target_indices, intrin_indices)) {
      possible_itervar_mappings.push_back(itervar_map);
    }
  }
  return std::move(possible_itervar_mappings);
}

TVM_REGISTER_GLOBAL("auto_tensorize.MatchIntrinsic").set_body([](TVMArgs args, TVMRetValue* ret) {
    MatchResult result = HwAbsDAGMatcher().match(args[0], args[1], args[2]);
  
  Array<Operation> keys;
  Array<Array<IterVarMap>> values;
  for (auto it : result) {
    keys.push_back(it.first);
    values.push_back(it.second);
  }

  Array<Array<Array<ObjectRef>>> flattened_values;
  for (auto value : values) {  // Array<IterVarMap>
    Array<Array<ObjectRef>> flattened_value;
    for (auto val : value) {  // IterVarMap
      Array<IterVar> ks;
      Array<IterVar> vs;
      for (auto it : val) {
        ks.push_back(it.first);
        vs.push_back(it.second);
      }
      Array<ObjectRef> flattened_val{ks, vs};
      flattened_value.push_back(flattened_val);
    }
    flattened_values.push_back(flattened_value);
  }

  Array<ObjectRef> flattened_result{keys, flattened_values};
  *ret = flattened_result;
});
}  // namespace auto_tensorize
}  // namespace tvm
