// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#ifndef TVM_TL_TRANSFORM_COMMON_ASCEND_RESOURCE_SCOPE_H_
#define TVM_TL_TRANSFORM_COMMON_ASCEND_RESOURCE_SCOPE_H_

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

// Validate the final hardware operations before emitting a Vector-only kernel.
void VerifyAscendAIVKernel(const tir::PrimFunc &func);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ASCEND_RESOURCE_SCOPE_H_
