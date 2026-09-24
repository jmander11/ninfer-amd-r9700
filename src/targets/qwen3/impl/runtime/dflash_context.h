#pragma once
#include "targets/qwen3/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "targets/qwen3/impl/runtime/layouts.h"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {

using DFlashFullBF16LayerView = ops::BidirectionalGqaBF16ContextView;

class DFlashFullBF16Cache {
public:
    DFlashFullBF16Cache(DeviceSpan backing,
                        const DFlashPersistentLayout::FullBF16Layout& layout);

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }
    [[nodiscard]] PagedKVPool& pool() noexcept { return pool_; }
    [[nodiscard]] const PagedKVPool& pool() const noexcept { return pool_; }
    [[nodiscard]] DFlashFullBF16LayerView layer_view(std::uint32_t layer) const;

private:
    PagedKVPool pool_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    std::int32_t head_dim_     = 0;
};

struct DFlashPersistentState {
    CyclicKVCache local;
    CyclicKVCache rewrite_checkpoint_local;
    CyclicKVCache staging_local;
    std::optional<DFlashFullBF16Cache> full;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor pending_features;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] DFlashFullBF16LayerView full_batch_layer(std::uint32_t layer) const;
    void save_rewrite_checkpoint(std::int32_t lane, hipStream_t stream);
    void restore_rewrite_checkpoint(std::int32_t lane, hipStream_t stream);
};

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS
