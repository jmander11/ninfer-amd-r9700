#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Computes the Qwen3 GDN input projections for a compact T=2..4 decode batch.
 *
 * `hidden` is contiguous BF16 [5120,T]. The two canonical Q4G64_F16S Q4N16K16
 * weights are [4096,5120] and [12288,5120], and the disjoint BF16 outputs have the
 * corresponding row counts. The represented activation is quantized once in stable
 * caller-owned storage and both semantic outputs are published by one combined grid.
 */
void gdn_input_projection_decode(const Tensor& hidden, const Weight& query_key,
                                 const Weight& value_z, Tensor& query_key_output,
                                 Tensor& value_z_output,
                                 const DeviceSpan& activation_workspace,
                                 hipStream_t stream);

} // namespace ninfer::ops
