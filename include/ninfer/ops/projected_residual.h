#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Exact caller-owned A8G64 scratch for the fixed T=1 boundary. Unsupported K is invalid.
[[nodiscard]] std::size_t projected_residual_t1_workspace_capacity_bytes(
    std::int32_t columns);

/**
 * Applies the exact R9700 decode projection-plus-residual boundary.
 *
 * `input` is contiguous BF16 [K,1], `weight` is canonical Q4G64_F16S Q4N16K16
 * [5120,K], and `residual` is contiguous BF16 [5120,1]. K is exactly 6144 or 17408.
 * The R9700 implementation quantizes the represented BF16 input to signed A8G64 with FP16
 * scales, then computes the observable mutation
 *
 *   delta = BF16(sum(dequantized_A8(input) * dequantized_Q4(weight)))
 *   residual = BF16(FP32(residual) + FP32(delta)).
 *
 * The explicit intermediate BF16 projection boundary matches the unfused Linear followed by
 * residual-add composition. Activation quantization uses caller-owned graph-stable storage; the
 * Op allocates and repacks nothing. Any activation-codec failure poisons every residual element
 * with the codec's conspicuous BF16 NaN. Input, persistent weight planes, activation workspace,
 * and residual must be pairwise disjoint. Weight plane extents and workspace extent must match
 * their exact represented sizes (workspace uses the capacity query above). BF16/FP16 planes
 * require natural alignment, packed weight codes require 8-byte alignment, and workspace requires
 * 4-byte alignment. Malformed bindings throw std::invalid_argument before device work.
 */
void projected_residual_t1(const Tensor& input, const Weight& weight, Tensor& residual,
                           const DeviceSpan& activation_workspace, hipStream_t stream);

} // namespace ninfer::ops
