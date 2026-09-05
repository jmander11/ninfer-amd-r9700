#include "core/fp8_int4_paged_kv_cache.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ninfer {
namespace {

void validate_spec(const Fp8KInt4VPagedKVSpec& spec) {
    if (spec.layer_count <= 0 || spec.head_dim <= 0 || spec.num_kv_heads <= 0) {
        throw std::invalid_argument("FP8-K/INT4-V cache requires positive layers, dimension, and heads");
    }
    if (spec.value_group != 16 && spec.value_group != 32) {
        throw std::invalid_argument("FP8-K/INT4-V cache V group must be 16 or 32");
    }
    if (spec.head_dim % spec.value_group != 0 || spec.head_dim % 2 != 0) {
        throw std::invalid_argument("FP8-K/INT4-V cache dimension must be even and divisible by V group");
    }
}

PagedKVPlaneOrder slab_order(Fp8KInt4VPlaneLayout layout) {
    if (layout != Fp8KInt4VPlaneLayout::FeatureFastestPageMajor &&
        layout != Fp8KInt4VPlaneLayout::TokenFastestHeadMajor) {
        throw std::invalid_argument("FP8-K/INT4-V cache has an invalid plane layout");
    }
    return layout == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor
               ? PagedKVPlaneOrder::PageMajor
               : PagedKVPlaneOrder::HeadMajor;
}

PagedKVIntraPageOrder intra_page_order(Fp8KInt4VPlaneLayout layout) {
    return layout == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor
               ? PagedKVIntraPageOrder::FeatureFastest
               : PagedKVIntraPageOrder::TokenFastest;
}

void check_plane(const PagedKVPool& pool, std::size_t index, DType dtype,
                 std::int32_t leading_extent, Fp8KInt4VPlaneLayout layout,
                 const Fp8KInt4VPagedKVSpec& spec,
                 const char* name) {
    const Tensor& plane = pool.plane(index);
    if (plane.dtype != dtype || pool.plane_order(index) != slab_order(layout) ||
        pool.plane_intra_page_order(index) != intra_page_order(layout)) {
        throw std::logic_error(std::string("FP8-K/INT4-V ") + name + " plane format mismatch");
    }
    const bool feature_fastest =
        intra_page_order(layout) == PagedKVIntraPageOrder::FeatureFastest;
    const bool page_major = slab_order(layout) == PagedKVPlaneOrder::PageMajor;
    const std::int32_t expected[] = {
        feature_fastest ? leading_extent : kPagedKVPageSize,
        feature_fastest ? kPagedKVPageSize : leading_extent,
        page_major ? spec.num_kv_heads : static_cast<std::int32_t>(spec.page_group_count),
        page_major ? static_cast<std::int32_t>(spec.page_group_count) : spec.num_kv_heads,
    };
    for (int axis = 0; axis < 4; ++axis) {
        if (plane.ne[axis] != expected[axis]) {
            throw std::logic_error(std::string("FP8-K/INT4-V ") + name +
                                   " plane geometry mismatch");
        }
    }
}

} // namespace

Fp8KInt4VSemanticFingerprint
fp8_k_int4_v_semantic_fingerprint(const Fp8KInt4VPagedKVSpec& spec) noexcept {
    return {.page_group_count = spec.page_group_count,
            .logical_page_capacity = spec.logical_page_capacity,
            .layer_count = spec.layer_count,
            .head_dim = spec.head_dim,
            .num_kv_heads = spec.num_kv_heads,
            .value_group = spec.value_group,
            .plane_layouts = spec.plane_layouts};
}

void require_fp8_k_int4_v_semantic_fingerprint(
    const Fp8KInt4VSemanticFingerprint& stored,
    const Fp8KInt4VSemanticFingerprint& expected) {
    if (stored != expected) {
        throw std::invalid_argument(
            "RAM cache fingerprint is incompatible with FP8-K/INT4-V storage");
    }
}

Fp8KInt4VPagedKVPoolLayout
plan_fp8_k_int4_v_paged_kv_pool(LayoutBuilder& builder, const Fp8KInt4VPagedKVSpec& spec) {
    validate_spec(spec);
    std::vector<PagedKVPlaneSpec> planes;
    planes.reserve(static_cast<std::size_t>(spec.layer_count) * 3U);
    for (std::int32_t layer = 0; layer < spec.layer_count; ++layer) {
        planes.push_back({DType::FP8_E4M3FN, spec.head_dim, spec.num_kv_heads, 256,
                          slab_order(spec.plane_layouts.key),
                          intra_page_order(spec.plane_layouts.key)});
        planes.push_back({DType::U8, spec.head_dim / 2, spec.num_kv_heads, 256,
                          slab_order(spec.plane_layouts.value),
                          intra_page_order(spec.plane_layouts.value)});
        planes.push_back({DType::FP16, spec.head_dim / spec.value_group, spec.num_kv_heads, 256,
                          slab_order(spec.plane_layouts.value_scale),
                          intra_page_order(spec.plane_layouts.value_scale)});
    }
    Fp8KInt4VPagedKVPoolLayout result;
    result.spec = spec;
    result.storage = plan_paged_kv_pool(builder, {.page_group_count = spec.page_group_count,
                                                   .logical_page_capacity = spec.logical_page_capacity,
                                                   .table_rows = spec.table_rows,
                                                   .planes = std::move(planes)});
    return result;
}

Fp8KInt4VPagedKVLayerView
fp8_k_int4_v_paged_kv_layer_view(const PagedKVPool& pool, const Fp8KInt4VPagedKVSpec& spec,
                                 std::int32_t layer_index, std::int32_t table_row) {
    validate_spec(spec);
    if (layer_index < 0 || layer_index >= spec.layer_count) {
        throw std::out_of_range("FP8-K/INT4-V layer index is out of range");
    }
    if (pool.page_group_count() != spec.page_group_count ||
        pool.logical_page_capacity() != spec.logical_page_capacity ||
        pool.table_row_count() != spec.table_rows ||
        pool.plane_count() != static_cast<std::size_t>(spec.layer_count) * 3U) {
        throw std::logic_error("FP8-K/INT4-V pool does not match its typed specification");
    }
    const std::size_t base = static_cast<std::size_t>(layer_index) * 3U;
    check_plane(pool, base, DType::FP8_E4M3FN, spec.head_dim, spec.plane_layouts.key, spec, "K");
    check_plane(pool, base + 1U, DType::U8, spec.head_dim / 2, spec.plane_layouts.value, spec,
                "V");
    check_plane(pool, base + 2U, DType::FP16, spec.head_dim / spec.value_group,
                spec.plane_layouts.value_scale, spec, "V scale");
    return {.key_fp8_pages = pool.plane(base),
            .value_int4_pages = pool.plane(base + 1U),
            .value_scale_fp16_pages = pool.plane(base + 2U),
            .block_table = pool.block_table_row(table_row),
            .head_dim = spec.head_dim,
            .num_kv_heads = spec.num_kv_heads,
            .value_group = spec.value_group,
            .plane_layouts = spec.plane_layouts};
}

Fp8KInt4VAppendBinding
bind_fp8_k_int4_v_append(const PagedKVAllocation& allocation, const PagedKVPool& pool,
                         const Fp8KInt4VPagedKVSpec& spec, std::int32_t layer_index,
                         std::span<const std::uint32_t> logical_positions) {
    if (!allocation.valid() || !allocation.belongs_to(pool)) {
        throw std::invalid_argument("FP8-K/INT4-V append requires an allocation from its pool");
    }
    if (allocation.bound_row() < 0) {
        throw std::logic_error("FP8-K/INT4-V append requires a bound block-table row");
    }
    std::unordered_set<std::uint32_t> seen;
    seen.reserve(logical_positions.size());
    const std::uint32_t mapped_pages = allocation.mapped_page_count();
    for (const std::uint32_t position : logical_positions) {
        const std::uint32_t page = position / static_cast<std::uint32_t>(kPagedKVPageSize);
        if (page >= mapped_pages || page >= spec.logical_page_capacity) {
            throw std::out_of_range("FP8-K/INT4-V append position is not materialized");
        }
        if (!seen.insert(position).second) {
            throw std::invalid_argument("FP8-K/INT4-V append positions must be distinct");
        }
    }
    return {.cache = fp8_k_int4_v_paged_kv_layer_view(pool, spec, layer_index,
                                                        allocation.bound_row()),
            .logical_positions = logical_positions,
            .mapped_pages = mapped_pages};
}

} // namespace ninfer
