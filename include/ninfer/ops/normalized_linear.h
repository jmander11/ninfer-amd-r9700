#pragma once

#include "core/arena.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * Op: RMS-normalized bias-free linear projection.
 * For every token t, h[k,t] = BF16(x[k,t] / sqrt(mean_k(x[k,t]^2)+eps)
 * * (norm[k] + (unit_offset ? 1 : 0))). Let a be the decoded compile-selected
 * canonical A8G64/A4G64 activation codec over h, including represented FP16 scales,
 * exactly as Linear; out[n,t] = BF16(sum_k W[n,k]*a[k,t]). The BF16 h boundary is
 * explicit. Nonfinite/scale-overflow activation status poisons all output values with NaN.
 * The oracle evaluates normalization and projection from the represented BF16 inputs
 * and decoded signed weight codes/scales, independently of the implementation reduction.
 *
 * x is contiguous BF16 [5120,T], norm BF16 [5120], W canonical Q4G64_F16S
 * Q4N16K16 [34816,5120], out contiguous BF16 [34816,T]; T is any positive value.
 * eps is positive finite. All input/output/weight/workspace planes are disjoint.
 * BF16 and FP16 planes are two-byte aligned; packed weight codes are eight-byte
 * aligned and workspace is four-byte aligned. Weight and workspace byte extents
 * are exact, including the internal workspace padding reported by the query.
 * Caller owns the exact workspace returned below; no persistent state or allocation.
 * Stream-ordered and capture-safe. The private fused route is T1/A8; other extents
 * compose public RMSNorm and Linear using a BF16 intermediate in caller workspace.
 */
[[nodiscard]] std::size_t normalized_linear_workspace_capacity_bytes(
    std::int32_t tokens, std::int32_t columns, std::int32_t rows);
void normalized_linear(const Tensor& input, const Tensor& norm, float eps,
                       bool unit_offset, const Weight& weight, Tensor& output,
                       const DeviceSpan& workspace, hipStream_t stream);

} // namespace ninfer::ops
