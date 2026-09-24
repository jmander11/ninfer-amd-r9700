#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>
#include <span>

namespace ninfer {

// Physical storage for a K-FP8 / V-INT4 growing cache. K is one OCP E4M3FN byte per feature. V
// is two signed canonical INT4 codes per byte, with one IEEE binary16 scale per V group. There is
// intentionally no K scale, K mean, or optional fourth plane. Target code owns which model
// geometry and codec profile may instantiate this container.
enum class Fp8KInt4VPlaneLayout : std::uint8_t {
    FeatureFastestPageMajor,
    TokenFastestHeadMajor,
};

struct Fp8KInt4VPlaneLayouts {
    Fp8KInt4VPlaneLayout key = Fp8KInt4VPlaneLayout::FeatureFastestPageMajor;
    Fp8KInt4VPlaneLayout value = Fp8KInt4VPlaneLayout::FeatureFastestPageMajor;
    Fp8KInt4VPlaneLayout value_scale = Fp8KInt4VPlaneLayout::FeatureFastestPageMajor;

    friend bool operator==(const Fp8KInt4VPlaneLayouts&,
                           const Fp8KInt4VPlaneLayouts&) = default;
};

struct Fp8KInt4VPagedKVSpec {
    std::uint32_t page_group_count      = 0;
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
    std::int32_t layer_count            = 0;
    std::int32_t head_dim               = 0;
    std::int32_t num_kv_heads           = 0;
    std::int32_t value_group            = 0;
    Fp8KInt4VPlaneLayouts plane_layouts;
};

struct Fp8KInt4VPagedKVPoolLayout {
    Fp8KInt4VPagedKVSpec spec;
    PagedKVPoolLayout storage;
};

// RAM images bind this semantic identity before any physical byte is restored. Tensor extents
// alone cannot distinguish a future codec revision with the same storage widths.
struct Fp8KInt4VSemanticFingerprint {
    static constexpr std::uint32_t kFormatVersion = 1;
    std::uint32_t format_version = kFormatVersion;
    std::uint32_t page_size = kPagedKVPageSize;
    std::uint32_t page_group_count = 0;
    std::uint32_t logical_page_capacity = 0;
    std::int32_t layer_count = 0;
    std::int32_t head_dim = 0;
    std::int32_t num_kv_heads = 0;
    std::int32_t value_group = 0;
    Fp8KInt4VPlaneLayouts plane_layouts;

    friend bool operator==(const Fp8KInt4VSemanticFingerprint&,
                           const Fp8KInt4VSemanticFingerprint&) = default;
};

[[nodiscard]] Fp8KInt4VSemanticFingerprint
fp8_k_int4_v_semantic_fingerprint(const Fp8KInt4VPagedKVSpec& spec) noexcept;
void require_fp8_k_int4_v_semantic_fingerprint(
    const Fp8KInt4VSemanticFingerprint& stored,
    const Fp8KInt4VSemanticFingerprint& expected);

// A semantic view for one Qwen text or MTP layer. Every field is a required persistent plane;
// append status and position uniqueness are transaction inputs owned by the calling Op, not
// persistent cache planes. block_table is the allocation-owned row selected for this sequence.
struct Fp8KInt4VPagedKVLayerView {
    Tensor key_fp8_pages;
    Tensor value_int4_pages;
    Tensor value_scale_fp16_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    std::int32_t value_group  = 0;
    Fp8KInt4VPlaneLayouts plane_layouts;
};

// The host-side half of an append transaction. The target constructs this before launching the
// HIP codec, using the exact logical positions it will append. This makes page-table coverage and
// non-overlapping packed-byte writes a checked cache invariant instead of a kernel convention.
//
// The caller owns a zeroed device status word for the matching launch. It must inspect that word
// after the launch is complete and publish neither the valid frontier nor generated tokens unless
// it remains zero. The raw HIP codec ORs its rejection bits into that word; this binding contains
// only represented cache state and no runtime API handle.
struct Fp8KInt4VAppendBinding {
    Fp8KInt4VPagedKVLayerView cache;
    std::span<const std::uint32_t> logical_positions;
    std::uint32_t mapped_pages = 0;
};

[[nodiscard]] Fp8KInt4VPagedKVPoolLayout
plan_fp8_k_int4_v_paged_kv_pool(LayoutBuilder& builder, const Fp8KInt4VPagedKVSpec& spec);

// Binds exactly the three planes belonging to layer_index and one already-owned block-table row.
// This validates the physical geometry before a device Op receives raw pointers.
[[nodiscard]] Fp8KInt4VPagedKVLayerView
fp8_k_int4_v_paged_kv_layer_view(const PagedKVPool& pool, const Fp8KInt4VPagedKVSpec& spec,
                                 std::int32_t layer_index, std::int32_t table_row);

// Validates the cache-owned allocation mapping and distinct logical positions, then produces the
// device-Op binding. Positions may be unordered, but every logical page they name must already be
// materialized by allocation. Duplicate positions are rejected because packed INT4 V bytes and V
// scales are single-writer storage.
[[nodiscard]] Fp8KInt4VAppendBinding
bind_fp8_k_int4_v_append(const PagedKVAllocation& allocation, const PagedKVPool& pool,
                         const Fp8KInt4VPagedKVSpec& spec, std::int32_t layer_index,
                         std::span<const std::uint32_t> logical_positions);

} // namespace ninfer
