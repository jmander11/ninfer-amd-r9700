#include <ninfer/targets/qwen3/decoder_state.h>

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers,
                              std::uint32_t capacity, std::int32_t kv_heads,
                              std::int32_t head_dim, std::int32_t value_group,
                              std::int32_t table_rows, std::uint32_t physical_page_groups,
                              Fp8KInt4VPlaneLayouts plane_layouts) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("FP8-K/INT4-V decoder cache geometry is invalid");
    }
    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("FP8-K/INT4-V decoder cache physical pages are below capacity");
    }
    Fp8KInt4VPagedKVSpec spec{
        .page_group_count = physical_page_groups,
        .logical_page_capacity = logical_pages,
        .table_rows = table_rows,
        .layer_count = static_cast<std::int32_t>(layers),
        .head_dim = head_dim,
        .num_kv_heads = kv_heads,
        .value_group = value_group,
        .plane_layouts = plane_layouts,
    };
    return {.storage = plan_fp8_k_int4_v_paged_kv_pool(builder, spec),
            .layers = layers,
            .max_context = capacity};
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.value_group, spec.kv_table_rows,
                                spec.text_physical_page_groups, spec.text_plane_layouts);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.value_group, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups, spec.mtp_plane_layouts);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.storage.storage), spec_(layout.storage.spec), layers_(layout.layers),
      max_context_(layout.max_context) {
    if (layers_ == 0 || max_context_ == 0 ||
        layers_ != static_cast<std::uint32_t>(spec_.layer_count) ||
        page_count(max_context_) != spec_.logical_page_capacity) {
        throw std::invalid_argument("FP8-K/INT4-V decoder cache layout is inconsistent");
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache,
                                   const PagedKVAllocation& allocation) noexcept
    : cache_(&cache), allocation_(&allocation) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0U : cache_->max_context();
}

Fp8KInt4VPagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr || allocation_ == nullptr) {
        throw std::logic_error("FP8-K/INT4-V execution view is empty");
    }
    return cache_->layer_view(static_cast<std::int32_t>(layer), *allocation_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.valid() || !allocation.belongs_to(pool_)) {
        throw std::invalid_argument("FP8-K/INT4-V execution view requires its cache allocation");
    }
    if (allocation.bound_row() < 0) {
        throw std::logic_error("FP8-K/INT4-V execution view requires a bound block-table row");
    }
    return PagedKVCacheView(*this, allocation);
}

PagedKVLayerRead::PagedKVLayerRead(Fp8KInt4VPagedKVLayerView layer,
                                   const PagedKVAllocation& allocation,
                                   const PagedKVPublication& publication,
                                   std::uint32_t publication_frontier,
                                   std::uint32_t visible_frontier,
                                   std::uint32_t mapped_pages,
                                   std::int32_t bound_row,
                                   std::uint64_t mapping_generation,
                                   std::uint64_t generation,
                                   hipStream_t ordered_stream,
                                   bool pending,
                                   const std::int32_t* device_table_row,
                                   const std::uint32_t* pool_block_tables,
                                   std::uint32_t pool_table_row_stride,
                                   std::uint32_t pool_table_row_count) noexcept
    : layer_(layer), allocation_(&allocation), publication_(&publication),
      publication_frontier_(publication_frontier), visible_frontier_(visible_frontier),
      mapped_pages_(mapped_pages), bound_row_(bound_row),
      mapping_generation_(mapping_generation), generation_(generation),
      ordered_stream_(ordered_stream), pending_(pending),
      device_table_row_(device_table_row), pool_block_tables_(pool_block_tables),
      pool_table_row_stride_(pool_table_row_stride),
      pool_table_row_count_(pool_table_row_count) {}

bool PagedKVLayerRead::valid_for(hipStream_t stream) const noexcept {
    if (stream == nullptr || allocation_ == nullptr || publication_ == nullptr ||
        !allocation_->valid() || allocation_->bound_row() != bound_row_ || bound_row_ < 0 ||
        allocation_->mapping_generation() != mapping_generation_ ||
        (device_table_row_ == nullptr && allocation_->mapped_page_count() != mapped_pages_) ||
        mapped_pages_ == 0U ||
        !publication_->healthy || publication_->transaction_generation != generation_ ||
        publication_->valid_frontier != publication_frontier_ || visible_frontier_ == 0U ||
        static_cast<std::uint64_t>(visible_frontier_) >
            static_cast<std::uint64_t>(mapped_pages_) *
            static_cast<std::uint64_t>(kPagedKVPageSize) ||
        (device_table_row_ != nullptr &&
         (pool_block_tables_ == nullptr || pool_table_row_stride_ < mapped_pages_ ||
          pool_table_row_count_ == 0U))) {
        return false;
    }
    if (pending_) {
        return publication_->transaction_open && ordered_stream_ == stream &&
               visible_frontier_ > publication_frontier_;
    }
    return !publication_->transaction_open && ordered_stream_ == nullptr &&
           visible_frontier_ == publication_frontier_;
}

PagedKVLayerRead PagedKVCache::published_layer_read(
    const PagedKVAllocation& allocation, const PagedKVPublication& publication,
    std::uint32_t layer) const {
    if (!publication.healthy || publication.transaction_open ||
        publication.valid_frontier == 0U || publication.valid_frontier > max_context_ ||
        publication.valid_frontier > allocation.mapped_token_capacity()) {
        throw std::logic_error("FP8-K/INT4-V committed layer publication is unavailable");
    }
    const auto typed_layer = layer_view(static_cast<std::int32_t>(layer), allocation);
    return PagedKVLayerRead(typed_layer, allocation, publication,
                            publication.valid_frontier, publication.valid_frontier,
                            allocation.mapped_page_count(), allocation.bound_row(),
                            allocation.mapping_generation(),
                            publication.transaction_generation, nullptr, false);
}

void PagedKVCache::truncate_publication(const PagedKVAllocation& allocation,
                                        PagedKVPublication& publication,
                                        std::uint32_t retained_frontier) const {
    if (!allocation.valid() || !allocation.belongs_to(pool_) || allocation.bound_row() < 0) {
        throw std::invalid_argument(
            "FP8-K/INT4-V publication truncation requires its bound cache allocation");
    }
    if (!publication.healthy || publication.transaction_open) {
        throw std::logic_error(
            "cannot truncate a poisoned or busy FP8-K/INT4-V publication");
    }
    if (retained_frontier > publication.valid_frontier || retained_frontier > max_context_ ||
        retained_frontier > allocation.mapped_token_capacity()) {
        throw std::invalid_argument(
            "FP8-K/INT4-V publication truncation is outside the published allocation");
    }
    if (retained_frontier == publication.valid_frontier) { return; }
    if (publication.transaction_generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FP8-K/INT4-V transaction generation exhausted");
    }
    ++publication.transaction_generation;
    publication.valid_frontier = retained_frontier;
}

Fp8KInt4VPagedKVLayerView
PagedKVCache::layer_view(std::int32_t layer, const PagedKVAllocation& allocation) const {
    if (!allocation.valid() || !allocation.belongs_to(pool_)) {
        throw std::invalid_argument("FP8-K/INT4-V decoder view requires its cache allocation");
    }
    if (allocation.bound_row() < 0) {
        throw std::logic_error("FP8-K/INT4-V decoder view requires a bound block-table row");
    }
    return fp8_k_int4_v_paged_kv_layer_view(pool_, spec_, layer, allocation.bound_row());
}

Fp8KInt4VAppendBinding PagedKVCache::bind_append(
    const PagedKVAllocation& allocation, std::int32_t layer,
    std::span<const std::uint32_t> logical_positions) const {
    return bind_fp8_k_int4_v_append(allocation, pool_, spec_, layer, logical_positions);
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0U);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept {
    return mtp_kv ? &*mtp_kv : nullptr;
}

} // namespace ninfer::targets::qwen3
