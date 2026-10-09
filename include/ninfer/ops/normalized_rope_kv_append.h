#pragma once

#include "ninfer/ops/kv_cache_append_prefix.h"

namespace ninfer::ops {

/**
 * Normalize and rotate represented DFlash2 K, and append selected K/V prefixes to its BF16 ring.
 * fused_kv is contiguous BF16 [2048,W*B], containing K[128,8] then V[128,8] per column.
 * positions is contiguous device I32 [W,B]; counts and lanes are contiguous device I32 [B].
 * key_norm is contiguous BF16 [128], eps is positive and finite. For every i < counts[b],
 * K = BF16(RoPE(RMSNorm(raw K, key_norm, eps), positions[i,b], theta=1e7));
 * V is copied exactly. RMS uses plain gains. The oracle evaluates normalization and rotation
 * naively in FP64 from represented public inputs. The native implementation retains BF16
 * normalization staging; that private arithmetic profile is checked against the same oracle.
 * Cyclic storage and caller promises are those of kv_cache_append_prefix: 2048 slots, D128/H8,
 * B1..8, sequential nonnegative positions, distinct selected lanes, at most one ring of writes.
 * Rejected rows and every other cache byte remain unchanged. Inputs and cache planes are disjoint.
 * No transient workspace, allocation, frontier publication or growing-cache state.
 */
void normalized_rope_kv_append(const Tensor& fused_kv, const Tensor& key_norm,
                               const Tensor& positions, const Tensor& counts, const Tensor& lanes,
                               float eps, KVCacheAppendPrefixExecutionEnvelope envelope,
                               const CyclicKVCacheLayerView& cache, hipStream_t stream);

} // namespace ninfer::ops
