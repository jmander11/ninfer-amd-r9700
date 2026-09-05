#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Prepare lane-owned BF16 prefixes as one compact ragged batch.
 *
 * source is [D,W,C], lanes/starts/ends are I32 [B], destination is contiguous BF16 [D,W,B],
 * positions is contiguous I32 [W,B], and counts is I32 [B]. For N=ends[b]-starts[b], columns j<N
 * copy source[:,j,lanes[b]], use absolute position starts[b]+j, and publish counts[b]=N. The
 * physical tail j>=N is zero-filled and repeats the final live position (or starts[b] when N=0).
 * The caller guarantees 0<=N<=W and 0<=lanes[b]<C. D must be divisible by eight and all BF16
 * column starts must be 16-byte aligned. The sole registered scheduling domain is W=1..16 and
 * B=1..4. Inputs and the three outputs are pairwise disjoint; the Op is an exact bit/I32
 * transform and owns no workspace or persistent state.
 */
void prepare_ragged_prefix(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                           const Tensor& ends, Tensor& destination, Tensor& positions,
                           Tensor& counts, hipStream_t stream);

} // namespace ninfer::ops
