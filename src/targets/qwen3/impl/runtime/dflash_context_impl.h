#include "targets/qwen3/impl/runtime/dflash_context.h"

#include <stdexcept>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {

DFlashFullBF16Cache::DFlashFullBF16Cache(
    DeviceSpan backing, const DFlashPersistentLayout::FullBF16Layout& layout)
    : pool_(backing, layout.storage), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim) {
    if (layers_ == 0 || max_context_ == 0 || kv_heads_ <= 0 || head_dim_ <= 0 ||
        pool_.plane_count() != static_cast<std::size_t>(layers_) * 2ULL) {
        throw std::invalid_argument("DFlash Full BF16 cache layout is invalid");
    }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        const Tensor& key   = pool_.plane(static_cast<std::size_t>(layer) * 2ULL);
        const Tensor& value = pool_.plane(static_cast<std::size_t>(layer) * 2ULL + 1ULL);
        if (key.dtype != DType::BF16 || value.dtype != DType::BF16 || key.ne[0] != head_dim_ ||
            value.ne[0] != head_dim_ || key.ne[3] != kv_heads_ || value.ne[3] != kv_heads_ ||
            pool_.plane_order(static_cast<std::size_t>(layer) * 2ULL) !=
                PagedKVPlaneOrder::HeadMajor ||
            pool_.plane_order(static_cast<std::size_t>(layer) * 2ULL + 1ULL) !=
                PagedKVPlaneOrder::HeadMajor) {
            throw std::invalid_argument("DFlash Full BF16 cache layout is invalid");
        }
    }
}

DFlashFullBF16LayerView DFlashFullBF16Cache::layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("DFlash Full BF16 layer is out of range"); }
    const std::size_t base = static_cast<std::size_t>(layer) * 2ULL;
    return DFlashFullBF16LayerView{
        .key_pages   = pool_.plane(base),
        .value_pages = pool_.plane(base + 1ULL),
        .block_tables = pool_.block_tables(),
        .head_dim     = head_dim_,
        .num_kv_heads = kv_heads_,
    };
}

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout)
    : local(backing, layout.local),
      rewrite_checkpoint_local(backing, layout.rewrite_checkpoint_local),
      staging_local(backing, layout.staging_local),
      prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (layout.full) { full.emplace(backing, *layout.full); }
    const bool local_ok =
        local.layer_count() == DFlashConfig::local_layers &&
        rewrite_checkpoint_local.layer_count() == DFlashConfig::local_layers &&
        staging_local.layer_count() == DFlashConfig::local_layers &&
        local.capacity() == DFlashConfig::local_capacity &&
        rewrite_checkpoint_local.capacity() == DFlashConfig::local_capacity &&
        staging_local.capacity() == DFlashConfig::local_capacity &&
        local.num_kv_heads() == DFlashConfig::kv_heads &&
        rewrite_checkpoint_local.num_kv_heads() == DFlashConfig::kv_heads &&
        staging_local.num_kv_heads() == DFlashConfig::kv_heads &&
        local.head_dim() == DFlashConfig::head_dim &&
        rewrite_checkpoint_local.head_dim() == DFlashConfig::head_dim &&
        staging_local.head_dim() == DFlashConfig::head_dim &&
        local.lane_capacity() == rewrite_checkpoint_local.lane_capacity() &&
        staging_local.lane_capacity() == 1;
    if (!local_ok) {
        throw std::invalid_argument("DFlash persistent cache layout is invalid");
    }
    if constexpr (DFlashConfig::full_layers > 0) {
        if (!full || full->layers() != 1 || full->max_context() != layout.full->max_context ||
            full->pool().plane_count() != 2 ||
            local.lane_capacity() != full->pool().table_row_count() ||
            full->pool().plane(0).dtype != DType::BF16 ||
            full->pool().plane(0).ne[0] != DFlashConfig::head_dim ||
            full->pool().plane(0).ne[1] != kPagedKVPageSize ||
            full->pool().plane(0).ne[3] != DFlashConfig::kv_heads) {
            throw std::invalid_argument("DFlash persistent cache layout is invalid");
        }
    } else if (full) {
        throw std::invalid_argument("DFlash2 persistent cache must not allocate a Full pool");
    }
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

DFlashFullBF16LayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    if (!full) { throw std::logic_error("DFlash Full pool is unavailable"); }
    return full->layer_view(layer);
}

void DFlashPersistentState::save_rewrite_checkpoint(std::int32_t lane, hipStream_t stream) {
    rewrite_checkpoint_local.copy_lane_from(local, lane, stream);
}

void DFlashPersistentState::restore_rewrite_checkpoint(std::int32_t lane, hipStream_t stream) {
    local.copy_lane_from(rewrite_checkpoint_local, lane, stream);
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS
