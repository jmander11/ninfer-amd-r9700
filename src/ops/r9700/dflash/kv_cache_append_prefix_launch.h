#pragma once

#include "ninfer/ops/kv_cache_append_prefix.h"

namespace ninfer::ops::r9700::dflash {

void kv_cache_append_prefix_paged_launch(
    const Tensor& key, const Tensor& value, const Tensor& positions, const Tensor& counts,
    const Tensor& table_rows, const KVCacheAppendPrefixPagedView& cache,
    KVCacheAppendPrefixExecutionEnvelope envelope, hipStream_t stream);

void kv_cache_append_prefix_cyclic_launch(
    const Tensor& key, const Tensor& value, const Tensor& positions, const Tensor& counts,
    const Tensor& lanes, const CyclicKVCacheLayerView& cache,
    KVCacheAppendPrefixExecutionEnvelope envelope, hipStream_t stream);

} // namespace ninfer::ops::r9700::dflash
