#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Host execution-resource promise for swa.
 *
 * positions[0,b] is row b's exact device-resident committed-context frontier. This envelope bounds
 * every row so a fixed launch can be captured and replayed without a host read.
 */
struct SwaContextExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Op: symmetric non-causal sliding-window grouped-query attention
 *
 * The fixed optimized geometry is D=128, Hq=32, Hkv=8, group=4, and two closed windows W=2048
 * and W=4096. swa() selects W from context.capacity; both are power-of-two cyclic capacities of
 * the same Op. q/out are contiguous BF16 [128,32,T,B], query_k/query_v are contiguous BF16
 * [128,8,T,B], positions is contiguous device I32 [T,B], valid_columns and lanes are contiguous
 * device I32 [B]. Row b has V=valid_columns[b] live query columns with positions[i,b]=L[b]+i for
 * i<V; lanes[b] selects its cyclic-cache lane. Columns i>=V are an inert physical tail and produce
 * zero output.
 *
 * The read-only cyclic context contains committed absolute positions [max(0,L-W),L), with
 * absolute position p stored at physical slot p mod W. Query K/V is a separate temporary
 * segment at positions [L,L+V). For every live query position p_i, admitted populated keys satisfy
 * abs(p_j-p_i)<W. Thus distance W-1 is included, distance W is excluded, and every query
 * row sees every live temporary query row from the same batch row. scale is 1/sqrt(128).
 *
 * Context and query K/V are unchanged. out is the only observable mutation and is completely
 * overwritten. The R9700 implementation domain is T=1..16 on gfx1201 wave32.
 *
 * The registered request-batch domain is B=1..4. The caller guarantees min_context <= L <=
 * max_context, sequential nonnegative positions, and that the cyclic context contains the
 * declared live interval. The envelope is checked in the device route without a host read and
 * never changes the admitted key set.
 */
void swa(const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
         const Tensor& valid_columns, const Tensor& lanes, float scale,
         const CyclicKVCacheLayerView& context, SwaContextExecutionEnvelope envelope,
         WorkspaceArena& workspace, Tensor& out, hipStream_t stream);

/**
 * Short fixed envelopes use a workspace-free wave32 score-streaming route. Longer envelopes use
 * caller-owned split-KV partial accumulators and statistics sized by this function. Invalid
 * profiles or intervals throw.
 */
[[nodiscard]] std::size_t swa_workspace_capacity_bytes(SwaContextExecutionEnvelope envelope,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens,
                                                       std::int32_t batch_size);

} // namespace ninfer::ops
