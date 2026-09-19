#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Computes the Qwen3 full-attention input projection at its represented boundary.
 *
 * `hidden` is contiguous BF16 [5120,1]. `query_key` and `gate_value` are canonical
 * Q4G64_F16S Q4N16K16 weights [7168,5120]. The four contiguous BF16 outputs are
 * query [6144,1], key [1024,1], gate [6144,1], and value [1024,1]. The implementation
 * quantizes the represented input once into caller-owned serialized storage, evaluates
 * both represented matrices, and writes the four semantic outputs directly. Each dot
 * accumulates in FP32 and rounds once at its observable BF16 output.
 *
 * This fixed-shape R9700 Op accepts only the stated T=1 all-Q4 domain. It allocates and
 * repacks nothing. Input, weight planes, workspace, and all outputs must be disjoint.
 */
void full_attention_projection_t1(const Tensor& hidden, const Weight& query_key,
                                  const Weight& gate_value, Tensor& query, Tensor& key,
                                  Tensor& gate, Tensor& value,
                                  const DeviceSpan& activation_workspace,
                                  hipStream_t stream);

} // namespace ninfer::ops
