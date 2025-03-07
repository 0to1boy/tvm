/*!
 * \file feature_visitor.cc
 * \brief Base class for feature extractor.
 *        These features are used for machine learning cost model
 */

#include "feature_visitor.h"

namespace tvm {
namespace tg {

// for loop
void FeatureVisitor::VisitStmt_(const ForNode* op) {
  const auto* extent = op->extent.as<IntImmNode>();
  const auto *min = op->min.as<IntImmNode>();
  int64_t loop_extent = -1;
  if (extent != nullptr)
    loop_extent = extent->value;

  AnnotationType ann = kSerial;
  switch (op->kind) {
    case ForKind ::kParallel:
      ann = kParallel;
      break;
    case ForKind::kUnrolled:
      ann = kUnrolled;
      break;
    case ForKind::kVectorized:
      ann = kVectorized;
      break;
    case ForKind::kSerial:
      ann = kSerial;
      break;
    case ForKind::kThreadBinding:
      LOG(FATAL) << "Loop ThreadBinding is reserved for future used and "
                  << "not yet supported in TIR";
      break;
  }

  if (EnterItervar_(op->loop_var, min->value, loop_extent, false, ann, nullptr, nullptr)) {
    StmtExprVisitor::VisitStmt_(op);
    ExitItervar_();
  }
}

// parallel axis, virtual thread
void FeatureVisitor::VisitStmt_(const AttrStmtNode* op) {
  // std::cout << "Found AttrStmtNode: " << op->attr_key << std::endl;
  if (op->attr_key == tir::attr::thread_extent || op->attr_key == tir::attr::virtual_thread ||
      std::string(op->attr_key).find(tir::attr::pragma_scope_prefix) == 0) {
    Var var = op->node.as<tir::IterVarNode>()->var;
    const auto *extent = op->value.as<IntImmNode>();

    size_t min = 0;
    if (auto ptr = op->body.as<tir::ForNode>())
      min = ptr->min.as<IntImmNode>()->value;
    
    CHECK(extent);

    std::string name = var.get()->name_hint;
    AnnotationType ann = kParallel;
    if (op->attr_key == tir::attr::thread_extent) {
      if (name == "blockIdx.x")
        ann = kBlockX;
      else if (name == "blockIdx.y")
        ann = kBlockY;
      else if (name == "blockIdx.z")
        ann = kBlockZ;
      else if (name == "threadIdx.x")
        ann = kThreadX;
      else if (name == "threadIdx.y")
        ann = kThreadY;
      else if (name == "threadIdx.z")
        ann = kThreadZ;
      else
        LOG(FATAL) << "invalid thread itervar " + name;
    } else if (std::string(op->attr_key).find(tir::attr::pragma_scope_prefix) == 0) {
      ann = kPragma;
    } else {
      ann = kVirtualThread;
    }
    if (EnterItervar_(var, min, extent->value, true, ann, op->attr_key.c_str(), &op->value)) {
      StmtExprVisitor::VisitStmt_(op);
      ExitItervar_();
    }
  } else if (op->attr_key == tir::attr::storage_alignment) {
    EnterAllocateNode_(op->value.as<StringImmNode>()->value);
    StmtExprVisitor::VisitStmt_(op);
    ExitAllocateNode_();
  }
}

// memory access
void FeatureVisitor::VisitExpr_(const BufferLoadNode* op) {
  EnterMem_(op->buffer->data, op->indices[0], AccessType::kRead);
  StmtExprVisitor::VisitExpr_(op);
  ExitMem_();
}


void FeatureVisitor::VisitStmt_(const BufferStoreNode* op) {
  EnterMem_(op->buffer->data, op->indices[0], AccessType::kWrite);
  StmtExprVisitor::VisitStmt_(op);
  ExitMem_();
}

}  // namespace tg
}  // namespace tvm
