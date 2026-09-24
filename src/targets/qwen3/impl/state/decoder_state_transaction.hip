#include <ninfer/targets/qwen3/decoder_state.h>

#include "core/device.h"
#include "ops/r9700/kv/fp8_int4_kv_append.h"
#include "ops/r9700/kv/fp8_int4_kv_compact.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3 {
namespace {

ops::r9700::kv::Fp8Int4KvPlaneLayout raw_layout(Fp8KInt4VPlaneLayout layout) {
    switch (layout) {
        case Fp8KInt4VPlaneLayout::FeatureFastestPageMajor:
            return ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        case Fp8KInt4VPlaneLayout::TokenFastestHeadMajor:
            return ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest;
    }
    throw std::invalid_argument("FP8-K/INT4-V transaction has an invalid plane layout");
}

__global__ void initialize_segmented_cursor(const std::int32_t* base,
                                            std::uint32_t minimum_base,
                                            std::uint32_t maximum_base,
                                            std::uint32_t max_context,
                                            std::uint32_t* cursor,
                                            std::uint32_t* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
    *status = 0U;
    const std::int32_t represented = *base;
    // Preserve the represented device value even on rejection so the ordered host resolution
    // reports the actual bad frontier instead of an uninformative zero.
    *cursor = static_cast<std::uint32_t>(represented);
    if (represented < 0 || static_cast<std::uint32_t>(represented) < minimum_base ||
        static_cast<std::uint32_t>(represented) > maximum_base ||
        static_cast<std::uint32_t>(represented) > max_context) {
        atomicOr(status, ops::r9700::kv::Fp8Int4KvAppendInvalidPosition);
        return;
    }
}

__global__ void advance_segmented_cursor(const std::int32_t* count,
                                         std::uint32_t maximum_count,
                                         std::uint32_t max_context,
                                         std::uint32_t* cursor,
                                         std::uint32_t* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0U) { return; }
    const std::int32_t represented = count == nullptr ? static_cast<std::int32_t>(maximum_count)
                                                       : *count;
    if (represented < 0 || static_cast<std::uint32_t>(represented) > maximum_count ||
        static_cast<std::uint32_t>(represented) > max_context - *cursor) {
        atomicOr(status, ops::r9700::kv::Fp8Int4KvAppendInvalidCount);
        return;
    }
    *cursor += static_cast<std::uint32_t>(represented);
}

} // namespace

PagedKVTransaction PagedKVCache::begin_append(
    PagedKVAllocation& allocation, PagedKVPublication& publication,
    std::span<const std::uint32_t> logical_positions,
    PagedKVTransactionWorkspace workspace) {
    if (!publication.healthy || publication.transaction_open) {
        throw std::logic_error("cannot append to a poisoned or busy FP8-K/INT4-V sequence");
    }
    if (logical_positions.empty()) {
        throw std::invalid_argument("FP8-K/INT4-V append transaction cannot be empty");
    }
    for (std::size_t i = 0; i < logical_positions.size(); ++i) {
        const std::uint64_t expected =
            static_cast<std::uint64_t>(publication.valid_frontier) + i;
        if (expected >= max_context_ || logical_positions[i] != expected) {
            throw std::invalid_argument(
                "FP8-K/INT4-V append positions must be contiguous at the valid frontier");
        }
    }
    (void)bind_append(allocation, 0, logical_positions);
    if (publication.transaction_generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FP8-K/INT4-V transaction generation exhausted");
    }
    ++publication.transaction_generation;
    publication.transaction_open = true;
    return PagedKVTransaction(*this, allocation, publication, logical_positions, nullptr,
                              logical_positions.size(), nullptr, workspace, true);
}

PagedKVTransaction PagedKVCache::begin_device_append(
    PagedKVAllocation& allocation, PagedKVPublication& publication,
    const std::int32_t* device_positions, std::size_t position_count,
    PagedKVTransactionWorkspace workspace) {
    if (!publication.healthy || publication.transaction_open) {
        throw std::logic_error("cannot append to a poisoned or busy FP8-K/INT4-V sequence");
    }
    if (device_positions == nullptr || position_count == 0U) {
        throw std::invalid_argument(
            "FP8-K/INT4-V device append requires a nonempty device I32 position panel");
    }
    (void)layer_view(0, allocation);
    const std::uint64_t suffix_end =
        static_cast<std::uint64_t>(publication.valid_frontier) + position_count;
    if (publication.valid_frontier > max_context_ || suffix_end > max_context_ ||
        suffix_end > allocation.mapped_token_capacity()) {
        throw std::invalid_argument(
            "FP8-K/INT4-V device append extent is outside the contiguous cache suffix");
    }
    if (publication.transaction_generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FP8-K/INT4-V transaction generation exhausted");
    }
    ++publication.transaction_generation;
    publication.transaction_open = true;
    return PagedKVTransaction(*this, allocation, publication, {}, device_positions,
                              position_count, nullptr, workspace, true);
}

PagedKVTransaction PagedKVCache::begin_device_prefix_append(
    PagedKVAllocation& allocation, PagedKVPublication& publication,
    const std::int32_t* device_positions, std::size_t maximum_position_count,
    const std::int32_t* device_position_count, PagedKVTransactionWorkspace workspace) {
    if (!publication.healthy || publication.transaction_open) {
        throw std::logic_error("cannot append to a poisoned or busy FP8-K/INT4-V sequence");
    }
    if (device_positions == nullptr || device_position_count == nullptr ||
        maximum_position_count == 0U ||
        maximum_position_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "FP8-K/INT4-V device-prefix append requires positions, count, and a nonzero I32 extent");
    }
    (void)layer_view(0, allocation);
    const std::uint64_t suffix_end =
        static_cast<std::uint64_t>(publication.valid_frontier) + maximum_position_count;
    if (publication.valid_frontier > max_context_ || suffix_end > max_context_ ||
        suffix_end > allocation.mapped_token_capacity()) {
        throw std::invalid_argument(
            "FP8-K/INT4-V device-prefix append extent is outside the cache suffix");
    }
    if (publication.transaction_generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FP8-K/INT4-V transaction generation exhausted");
    }
    ++publication.transaction_generation;
    publication.transaction_open = true;
    return PagedKVTransaction(*this, allocation, publication, {}, device_positions,
                              maximum_position_count, device_position_count, workspace, true);
}

PagedKVTransaction PagedKVCache::begin_device_segmented_append(
    PagedKVAllocation& allocation, PagedKVPublication& publication,
    const std::int32_t* device_base_frontier, std::uint32_t minimum_base_frontier,
    std::uint32_t maximum_base_frontier, std::uint32_t visible_frontier_limit,
    PagedKVTransactionWorkspace workspace, const std::int32_t* device_table_row) {
    if (!publication.healthy || publication.transaction_open) {
        throw std::logic_error("cannot append to a poisoned or busy FP8-K/INT4-V sequence");
    }
    if (device_base_frontier == nullptr || workspace.status == nullptr ||
        workspace.cursor == nullptr || minimum_base_frontier > maximum_base_frontier ||
        publication.valid_frontier < minimum_base_frontier ||
        publication.valid_frontier > maximum_base_frontier ||
        visible_frontier_limit <= maximum_base_frontier ||
        visible_frontier_limit > max_context_) {
        throw std::invalid_argument("FP8-K/INT4-V segmented append envelope is invalid");
    }
    (void)layer_view(0, allocation);
    if (publication.transaction_generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FP8-K/INT4-V transaction generation exhausted");
    }
    ++publication.transaction_generation;
    publication.transaction_open = true;
    return PagedKVTransaction(*this, allocation, publication, {}, nullptr, 0U, nullptr,
                              workspace, true, device_base_frontier, minimum_base_frontier,
                              maximum_base_frontier, visible_frontier_limit, device_table_row);
}

PagedKVTransaction PagedKVCache::begin_compact(PagedKVAllocation& allocation,
                                                PagedKVPublication& publication,
                                                PagedKVTransactionWorkspace workspace) {
    if (!publication.healthy || publication.transaction_open) {
        throw std::logic_error("cannot compact a poisoned or busy FP8-K/INT4-V sequence");
    }
    if (!allocation.valid() || !allocation.belongs_to(pool_) || allocation.bound_row() < 0) {
        throw std::invalid_argument(
            "FP8-K/INT4-V compaction requires a bound allocation from its cache");
    }
    if (publication.transaction_generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FP8-K/INT4-V transaction generation exhausted");
    }
    ++publication.transaction_generation;
    publication.transaction_open = true;
    return PagedKVTransaction(*this, allocation, publication, {}, nullptr, 0U, nullptr, workspace,
                              false);
}

PagedKVTransaction::PagedKVTransaction(PagedKVCache& cache, PagedKVAllocation& allocation,
                                       PagedKVPublication& publication,
                                       std::span<const std::uint32_t> positions,
                                       const std::int32_t* device_positions,
                                       std::size_t position_count,
                                       const std::int32_t* device_position_count,
                                       PagedKVTransactionWorkspace workspace, bool append_mode,
                                       const std::int32_t* device_base_frontier,
                                       std::uint32_t minimum_base_frontier,
                                       std::uint32_t maximum_base_frontier,
                                       std::uint32_t visible_frontier_limit,
                                       const std::int32_t* device_table_row)
    : cache_(&cache), allocation_(&allocation), publication_(&publication),
      positions_(positions), device_positions_(device_positions), position_count_(position_count),
      device_position_count_(device_position_count),
      device_base_frontier_(device_base_frontier), device_table_row_(device_table_row),
      workspace_(workspace),
      published_frontier_(publication.valid_frontier),
      initial_frontier_(publication.valid_frontier),
      minimum_base_frontier_(minimum_base_frontier),
      maximum_base_frontier_(maximum_base_frontier),
      visible_frontier_limit_(visible_frontier_limit),
      allocation_mapping_generation_(allocation.mapping_generation()),
      allocation_bound_row_(allocation.bound_row()),
      append_mode_(append_mode), segmented_mode_(device_base_frontier != nullptr),
      positions_uploaded_(device_positions != nullptr) {
    const bool host_positions_invalid = append_mode_ && !segmented_mode_ &&
                                        device_positions_ == nullptr &&
                                        (workspace_.positions == nullptr ||
                                         workspace_.position_capacity == 0U);
    if (cache.layers() > 64U || workspace_.status == nullptr || host_positions_invalid ||
        (segmented_mode_ && workspace_.cursor == nullptr)) {
        publication.transaction_open = false;
        throw std::invalid_argument("FP8-K/INT4-V transaction workspace is invalid");
    }
}

PagedKVTransaction::~PagedKVTransaction() noexcept { abort(); }

PagedKVTransaction::PagedKVTransaction(PagedKVTransaction&& other) noexcept
    : cache_(other.cache_), allocation_(other.allocation_), publication_(other.publication_),
      positions_(other.positions_), device_positions_(other.device_positions_),
      position_count_(other.position_count_),
      device_position_count_(other.device_position_count_),
      device_base_frontier_(other.device_base_frontier_),
      device_table_row_(other.device_table_row_), workspace_(other.workspace_),
      stream_(other.stream_),
      published_frontier_(other.published_frontier_), initial_frontier_(other.initial_frontier_),
      minimum_base_frontier_(other.minimum_base_frontier_),
      maximum_base_frontier_(other.maximum_base_frontier_),
      visible_frontier_limit_(other.visible_frontier_limit_),
      compact_prefix_(other.compact_prefix_),
      layers_launched_(other.layers_launched_),
      allocation_mapping_generation_(other.allocation_mapping_generation_),
      allocation_bound_row_(other.allocation_bound_row_), append_mode_(other.append_mode_),
      segmented_mode_(other.segmented_mode_), segment_open_(other.segment_open_),
      launched_(other.launched_), positions_uploaded_(other.positions_uploaded_),
      closed_(other.closed_) {
    other.cache_ = nullptr;
    other.allocation_ = nullptr;
    other.publication_ = nullptr;
    other.closed_ = true;
}

PagedKVTransaction& PagedKVTransaction::operator=(PagedKVTransaction&& other) noexcept {
    if (this != &other) {
        abort();
        cache_ = other.cache_;
        allocation_ = other.allocation_;
        publication_ = other.publication_;
        positions_ = other.positions_;
        device_positions_ = other.device_positions_;
        position_count_ = other.position_count_;
        device_position_count_ = other.device_position_count_;
        device_base_frontier_ = other.device_base_frontier_;
        device_table_row_ = other.device_table_row_;
        workspace_ = other.workspace_;
        stream_ = other.stream_;
        published_frontier_ = other.published_frontier_;
        initial_frontier_ = other.initial_frontier_;
        minimum_base_frontier_ = other.minimum_base_frontier_;
        maximum_base_frontier_ = other.maximum_base_frontier_;
        visible_frontier_limit_ = other.visible_frontier_limit_;
        compact_prefix_ = other.compact_prefix_;
        layers_launched_ = other.layers_launched_;
        allocation_mapping_generation_ = other.allocation_mapping_generation_;
        allocation_bound_row_ = other.allocation_bound_row_;
        append_mode_ = other.append_mode_;
        segmented_mode_ = other.segmented_mode_;
        segment_open_ = other.segment_open_;
        launched_ = other.launched_;
        positions_uploaded_ = other.positions_uploaded_;
        closed_ = other.closed_;
        other.cache_ = nullptr;
        other.allocation_ = nullptr;
        other.publication_ = nullptr;
        other.closed_ = true;
    }
    return *this;
}

void PagedKVTransaction::require_open(const char* operation) const {
    if (cache_ == nullptr || allocation_ == nullptr || publication_ == nullptr || closed_) {
        throw std::logic_error(std::string("cannot ") + operation +
                               " through a closed FP8-K/INT4-V transaction");
    }
}

void PagedKVTransaction::require_new_layer(std::uint32_t layer, const char* operation) const {
    if (layer >= cache_->layers() || (layers_launched_ & (1ULL << layer)) != 0U) {
        throw std::logic_error(std::string("cannot ") + operation +
                               " for a missing or already-launched FP8-K/INT4-V layer");
    }
}

void PagedKVTransaction::prepare_launch(hipStream_t stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("FP8-K/INT4-V transaction requires an explicit HIP stream");
    }
    if (!launched_) {
        if (segmented_mode_) {
            hipLaunchKernelGGL(initialize_segmented_cursor, dim3(1), dim3(1), 0, stream,
                               device_base_frontier_, minimum_base_frontier_,
                               maximum_base_frontier_, cache_->max_context(), workspace_.cursor,
                               workspace_.status);
            HIP_CHECK(hipGetLastError());
        } else {
            HIP_CHECK(hipMemsetAsync(workspace_.status, 0, sizeof(std::uint32_t), stream));
        }
        stream_ = stream;
    } else if (stream_ != stream) {
        throw std::invalid_argument("FP8-K/INT4-V transaction layers must use one ordered stream");
    }
    if (!positions_uploaded_) {
        if (positions_.empty() || positions_.size() > workspace_.position_capacity) {
            throw std::logic_error("FP8-K/INT4-V transaction positions are not device-bound");
        }
        HIP_CHECK(hipMemcpyAsync(workspace_.positions, positions_.data(),
                                 positions_.size_bytes(), hipMemcpyHostToDevice, stream));
        positions_uploaded_ = true;
    }
}

void PagedKVTransaction::begin_device_segment(const std::int32_t* device_positions,
                                              std::size_t maximum_position_count,
                                              const std::int32_t* device_position_count) {
    require_open("begin append segment");
    if (!segmented_mode_ || segment_open_ || layers_launched_ != 0U ||
        device_positions == nullptr || maximum_position_count == 0U ||
        maximum_position_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::logic_error("FP8-K/INT4-V append segment is invalid");
    }
    device_positions_ = device_positions;
    position_count_ = maximum_position_count;
    device_position_count_ = device_position_count;
    positions_uploaded_ = true;
    segment_open_ = true;
}

void PagedKVTransaction::launch_append_layer(std::uint32_t layer,
                                             const hip_bfloat16* keys,
                                             const hip_bfloat16* values,
                                             hipStream_t stream) {
    require_open("launch append");
    if (!append_mode_) {
        throw std::logic_error("cannot append through a compaction transaction");
    }
    if (segmented_mode_ && !segment_open_) {
        throw std::logic_error("cannot append outside an FP8-K/INT4-V segment");
    }
    require_new_layer(layer, "launch append");
    if (keys == nullptr || values == nullptr) {
        throw std::invalid_argument("FP8-K/INT4-V append requires represented device K and V");
    }
    const auto binding = device_positions_ == nullptr
                             ? cache_->bind_append(*allocation_, static_cast<std::int32_t>(layer),
                                                   positions_)
                             : Fp8KInt4VAppendBinding{
                                   .cache = cache_->layer_view(static_cast<std::int32_t>(layer),
                                                               *allocation_),
                                   .logical_positions = {},
                                   .mapped_pages = allocation_->mapped_page_count(),
                               };
    prepare_launch(stream);
    ops::r9700::kv::Fp8Int4KvAppendArgs args{
        .keys = keys,
        .values = values,
        .positions = device_positions_ == nullptr
                         ? workspace_.positions
                         : reinterpret_cast<const std::uint32_t*>(device_positions_),
        .tokens = position_count_,
        .token_count = device_position_count_,
        .first_position = publication_->valid_frontier,
        .first_position_device = segmented_mode_ ? workspace_.cursor : nullptr,
        .kv_heads = static_cast<std::uint32_t>(binding.cache.num_kv_heads),
        .page_table = device_table_row_ == nullptr
                          ? static_cast<const std::uint32_t*>(binding.cache.block_table.data)
                          : static_cast<const std::uint32_t*>(cache_->pool().block_tables().data),
        .page_table_pages = segmented_mode_ ? cache_->pool().logical_page_capacity()
                                             : binding.mapped_pages,
        .page_table_row = device_table_row_,
        .page_table_row_stride = cache_->pool().logical_page_capacity(),
        .page_table_row_count = static_cast<std::uint32_t>(cache_->pool().table_row_count()),
        .physical_tokens = static_cast<std::size_t>(cache_->spec_.page_group_count) *
                           static_cast<std::size_t>(kPagedKVPageSize),
        .key_codes = static_cast<std::uint8_t*>(binding.cache.key_fp8_pages.data),
        .value_codes = static_cast<std::uint8_t*>(binding.cache.value_int4_pages.data),
        .value_scale_bits =
            static_cast<std::uint16_t*>(binding.cache.value_scale_fp16_pages.data),
        .status = workspace_.status,
        .value_group = static_cast<std::uint32_t>(binding.cache.value_group),
        .key_layout = raw_layout(binding.cache.plane_layouts.key),
        .value_layout = raw_layout(binding.cache.plane_layouts.value),
        .value_scale_layout = raw_layout(binding.cache.plane_layouts.value_scale),
    };
    launched_ = true;
    HIP_CHECK(ops::r9700::kv::fp8_int4_kv_append(args, stream));
    layers_launched_ |= 1ULL << layer;
    published_frontier_ = segmented_mode_
                              ? visible_frontier_limit_
                              : publication_->valid_frontier +
                                    static_cast<std::uint32_t>(position_count_);
}

void PagedKVTransaction::end_device_segment(hipStream_t stream) {
    require_open("end append segment");
    const std::uint64_t expected_layers = cache_->layers() == 64U
                                              ? ~0ULL
                                              : ((1ULL << cache_->layers()) - 1ULL);
    if (!segmented_mode_ || !segment_open_ || !launched_ || stream_ != stream ||
        layers_launched_ != expected_layers) {
        throw std::logic_error("cannot close an incomplete FP8-K/INT4-V append segment");
    }
    hipLaunchKernelGGL(advance_segmented_cursor, dim3(1), dim3(1), 0, stream,
                       device_position_count_, static_cast<std::uint32_t>(position_count_),
                       cache_->max_context(), workspace_.cursor, workspace_.status);
    HIP_CHECK(hipGetLastError());
    layers_launched_ = 0U;
    segment_open_ = false;
}

void PagedKVTransaction::launch_compact_layer(
    std::uint32_t layer, std::uint32_t prefix,
    std::span<const std::uint32_t> selected_path, hipStream_t stream) {
    require_open("launch compaction");
    if (append_mode_) {
        throw std::logic_error("cannot compact through an append transaction");
    }
    require_new_layer(layer, "launch compaction");
    if (selected_path.empty() || prefix > publication_->valid_frontier) {
        throw std::invalid_argument("FP8-K/INT4-V compaction has an invalid prefix or path");
    }
    std::uint32_t previous = 0;
    for (std::size_t i = 0; i < selected_path.size(); ++i) {
        const std::uint64_t source = static_cast<std::uint64_t>(prefix) + selected_path[i];
        if (selected_path[i] < i || (i != 0 && selected_path[i] <= previous) ||
            source >= publication_->valid_frontier) {
            throw std::invalid_argument("FP8-K/INT4-V compaction path is not monotone");
        }
        previous = selected_path[i];
    }
    if (!launched_) {
        positions_ = selected_path;
        compact_prefix_ = prefix;
    } else if (prefix != compact_prefix_ ||
               !std::equal(positions_.begin(), positions_.end(), selected_path.begin(),
                           selected_path.end())) {
        throw std::invalid_argument("FP8-K/INT4-V compaction layers require one identical path");
    }
    const auto view = cache_->layer_view(static_cast<std::int32_t>(layer), *allocation_);
    prepare_launch(stream);
    ops::r9700::kv::Fp8Int4KvCompactArgs args{
        .prefix = prefix,
        .selected_path = workspace_.positions,
        .selected_count = positions_.size(),
        .page_table = static_cast<const std::uint32_t*>(view.block_table.data),
        .page_table_pages = allocation_->mapped_page_count(),
        .physical_tokens = static_cast<std::size_t>(cache_->spec_.page_group_count) *
                           static_cast<std::size_t>(kPagedKVPageSize),
        .kv_heads = static_cast<std::uint32_t>(view.num_kv_heads),
        .key_codes = static_cast<std::uint8_t*>(view.key_fp8_pages.data),
        .value_codes = static_cast<std::uint8_t*>(view.value_int4_pages.data),
        .value_scale_bits = static_cast<std::uint16_t*>(view.value_scale_fp16_pages.data),
        .status = workspace_.status,
        .value_group = static_cast<std::uint32_t>(view.value_group),
        .key_layout = raw_layout(view.plane_layouts.key),
        .value_layout = raw_layout(view.plane_layouts.value),
        .value_scale_layout = raw_layout(view.plane_layouts.value_scale),
    };
    launched_ = true;
    HIP_CHECK(ops::r9700::kv::fp8_int4_kv_compact(args, stream));
    layers_launched_ |= 1ULL << layer;
    published_frontier_ = prefix + static_cast<std::uint32_t>(positions_.size());
}

PagedKVLayerRead PagedKVTransaction::pending_layer_read(std::uint32_t layer) const {
    require_open("read pending append");
    if (!append_mode_ || !launched_ || stream_ == nullptr || layer >= cache_->layers() ||
        (layers_launched_ & (1ULL << layer)) == 0U || !publication_->healthy ||
        !publication_->transaction_open || published_frontier_ <= publication_->valid_frontier) {
        throw std::logic_error(
            "FP8-K/INT4-V pending layer is unavailable before its append launch");
    }
    const Tensor& pool_tables = cache_->pool().block_tables();
    return PagedKVLayerRead(
        cache_->layer_view(static_cast<std::int32_t>(layer), *allocation_), *allocation_,
        *publication_, publication_->valid_frontier, published_frontier_,
        segmented_mode_ ? cache_->pool().logical_page_capacity()
                        : allocation_->mapped_page_count(),
        allocation_->bound_row(),
        allocation_->mapping_generation(),
        publication_->transaction_generation, stream_, true, device_table_row_,
        device_table_row_ == nullptr
            ? nullptr
            : static_cast<const std::uint32_t*>(pool_tables.data),
        cache_->pool().logical_page_capacity(),
        static_cast<std::uint32_t>(cache_->pool().table_row_count()));
}

std::uint32_t PagedKVTransaction::commit() {
    require_open("commit");
    if (segmented_mode_) {
        throw std::logic_error("segmented FP8-K/INT4-V transactions require batched resolution");
    }
    const std::uint64_t expected_layers = cache_->layers() == 64U
                                              ? ~0ULL
                                              : ((1ULL << cache_->layers()) - 1ULL);
    if (!launched_ || layers_launched_ != expected_layers) {
        throw std::logic_error("cannot commit an incomplete FP8-K/INT4-V transaction");
    }
    std::uint32_t status = 0;
    std::int32_t committed_count = static_cast<std::int32_t>(position_count_);
    const hipError_t copy = hipMemcpyAsync(&status, workspace_.status, sizeof(status),
                                           hipMemcpyDeviceToHost, stream_);
    const hipError_t count_copy =
        copy == hipSuccess && device_position_count_ != nullptr
            ? hipMemcpyAsync(&committed_count, device_position_count_, sizeof(committed_count),
                             hipMemcpyDeviceToHost, stream_)
            : hipSuccess;
    const hipError_t sync = copy == hipSuccess && count_copy == hipSuccess
                                ? hipStreamSynchronize(stream_)
                                : hipSuccess;
    const bool invalid_count = committed_count < 0 ||
                               static_cast<std::size_t>(committed_count) > position_count_;
    if (copy != hipSuccess || count_copy != hipSuccess || sync != hipSuccess || status != 0U ||
        invalid_count) {
        finish(true);
        if (copy != hipSuccess) { HIP_CHECK(copy); }
        if (count_copy != hipSuccess) { HIP_CHECK(count_copy); }
        if (sync != hipSuccess) { HIP_CHECK(sync); }
        throw std::runtime_error(
            "FP8-K/INT4-V device mutation rejected before frontier publication");
    }
    if (device_position_count_ != nullptr) {
        published_frontier_ = publication_->valid_frontier +
                              static_cast<std::uint32_t>(committed_count);
    }
    const std::uint32_t result = published_frontier_;
    finish(false);
    return result;
}

void PagedKVTransaction::mark_graph_replay(hipStream_t stream) {
    require_open("mark graph replay");
    if (!segmented_mode_ || launched_ || segment_open_ || stream == nullptr ||
        !allocation_->valid() || allocation_->bound_row() != allocation_bound_row_ ||
        allocation_->mapping_generation() != allocation_mapping_generation_) {
        throw std::logic_error("FP8-K/INT4-V graph replay authority is invalid");
    }
    stream_ = stream;
    launched_ = true;
}

void PagedKVTransaction::enqueue_segmented_resolution(std::uint32_t* host_status,
                                                      std::uint32_t* host_cursor) {
    require_open("enqueue segmented resolution");
    if (!segmented_mode_ || !launched_ || segment_open_ || layers_launched_ != 0U ||
        stream_ == nullptr || host_status == nullptr || host_cursor == nullptr) {
        throw std::logic_error("FP8-K/INT4-V segmented resolution is invalid");
    }
    HIP_CHECK(hipMemcpyAsync(host_status, workspace_.status, sizeof(*host_status),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(host_cursor, workspace_.cursor, sizeof(*host_cursor),
                             hipMemcpyDeviceToHost, stream_));
}

void PagedKVTransaction::finish_segmented_resolution(std::uint32_t status,
                                                     std::uint32_t cursor,
                                                     std::uint32_t retained_frontier) {
    require_open("finish segmented resolution");
    const bool invalid = !segmented_mode_ || !launched_ || segment_open_ ||
                         layers_launched_ != 0U || status != 0U ||
                         !allocation_->valid() ||
                         allocation_->bound_row() != allocation_bound_row_ ||
                         allocation_->mapping_generation() != allocation_mapping_generation_ ||
                         cursor < initial_frontier_ || cursor > visible_frontier_limit_ ||
                         retained_frontier < initial_frontier_ || retained_frontier > cursor;
    if (invalid) {
        std::ostringstream message;
        message << "FP8-K/INT4-V segmented device mutation rejected before frontier publication"
                << " (status=" << status << ", cursor=" << cursor
                << ", initial=" << initial_frontier_
                << ", visible_limit=" << visible_frontier_limit_
                << ", retained=" << retained_frontier
                << ", segmented=" << segmented_mode_ << ", launched=" << launched_
                << ", segment_open=" << segment_open_
                << ", layers_pending=" << layers_launched_ << ')';
        finish(true);
        throw std::runtime_error(message.str());
    }
    published_frontier_ = retained_frontier;
    finish(false);
}

void PagedKVTransaction::close_captured_segmented() {
    require_open("close captured segmented transaction");
    if (!segmented_mode_ || !launched_ || segment_open_ || layers_launched_ != 0U) {
        throw std::logic_error("captured FP8-K/INT4-V segmented transaction is incomplete");
    }
    publication_->transaction_open = false;
    closed_ = true;
    cache_ = nullptr;
    allocation_ = nullptr;
    publication_ = nullptr;
}

void PagedKVTransaction::finish(bool poison) noexcept {
    if (!closed_ && publication_ != nullptr) {
        if (poison) {
            publication_->healthy = false;
        } else {
            publication_->valid_frontier = published_frontier_;
        }
        publication_->transaction_open = false;
    }
    closed_ = true;
    cache_ = nullptr;
    allocation_ = nullptr;
    publication_ = nullptr;
}

void PagedKVTransaction::abort() noexcept {
    if (closed_) { return; }
    if (launched_ && stream_ != nullptr) { (void)hipStreamSynchronize(stream_); }
    finish(launched_);
}

} // namespace ninfer::targets::qwen3
