#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Host execution-resource promise for kv_cache_append_prefix.
 *
 * Every device count must remain in this inclusive interval on every replay. The envelope fixes
 * launch resources; it does not select or publish committed frontiers.
 */
struct KVCacheAppendPrefixExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

// Non-owning DFlash Full BF16 page view. This state is deliberately distinct from the
// asymmetric FP8-K/INT4-V Text/MTP cache and carries no allocation, transaction, or publication
// authority.
struct KVCacheAppendPrefixPagedView {
    Tensor key_pages;
    Tensor value_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
};

/**
 * Op: append device-selected exact K/V prefixes to batched paged growing-cache storage.
 *
 * k/v are contiguous BF16 [128,8,W,B], positions is contiguous device I32 [W,B], counts and
 * table_rows are contiguous device I32 [B]. For every row b and i in [0,counts[b]), the Op copies
 * k/v[:, :, i, b] bit-for-bit into logical cache position positions[i,b] through table row
 * table_rows[b]. No cache byte for any rejected physical tail is written. Inputs are unchanged,
 * and the Op neither decides nor publishes a frontier. The paged K/V planes use the DFlash Full
 * head-major order [128,64,Nphysical,8].
 *
 * The registered request-batch domain is B=1..4. The caller guarantees positive W,
 * 0 <= counts[b] <= W within the declared envelope, valid
 * sequential nonnegative positions, pairwise non-aliasing, and materialized block-table entries
 * for every represented position. The only registered geometry is D=128, Hkv=8, page size 64,
 * with head-major BF16 planes.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            const KVCacheAppendPrefixPagedView& cache, hipStream_t stream);

/**
 * Op: append device-selected exact K/V prefixes to lane-owned cyclic cache storage.
 *
 * k/v, positions, and counts have the same batch geometry as the paged overload; lanes[b] selects
 * the destination cache lane. Absolute position p maps to physical slot p mod W, where W is the
 * cyclic cache capacity, which is exactly 2048 for Qwen3.8 DFlash2. The caller
 * guarantees that each row's live interval ends immediately before positions[0,b] and that
 * advancing it by counts[b] makes every overwritten old slot dead. One row may commit at most the
 * ring capacity, so no two live writes race for one physical slot.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            const CyclicKVCacheLayerView& cache, hipStream_t stream);

} // namespace ninfer::ops
