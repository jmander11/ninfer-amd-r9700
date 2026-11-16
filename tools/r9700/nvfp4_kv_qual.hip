// Qualification of the opt-in NVFP4 growing cache through its real typed-cache, transaction and
// full-attention boundaries, against the independent exact codec and FP64 attention oracle in
// tests/ops/nvfp4_kv_oracle.h.
//
// Covered: exact stored bytes of all four planes (ties, saturation, signed and zero groups,
// fragmented pages, mid-page appends), every status rejection, batched append, device-prefix and
// segmented append, compaction and rollback, the packed, mid-row, dense, fused and split-512 routes
// including causal row frontiers, packed-tree masks and device-selected rows, sequence batches on
// both loader-wave shapes, poison isolation, unwritten stale slots, the Q staging range, Device
// Graph replay with a changed mapping, and raw-argument rejection.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/nvfp4_kv_oracle.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/fp8_int4_kv_compact.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "targets/qwen3_8_27b/impl/r9700_full_attention.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q27    = ninfer::targets::qwen3_8_27b::detail;
namespace q3     = ninfer::targets::qwen3;
namespace kv     = ninfer::ops::r9700::kv;
namespace oracle = ninfer::test::nvfp4_kv_oracle;

constexpr std::size_t kHeadDim      = 256;
constexpr std::size_t kQueryHeads   = 24;
constexpr std::size_t kKvHeads      = 4;
constexpr std::size_t kQueriesPerKv = 6;
constexpr std::size_t kGroups       = 16;
constexpr std::size_t kPageSize     = 64;
constexpr std::size_t kRowElements  = kQueryHeads * kHeadDim;
constexpr std::size_t kTokenItems   = kKvHeads * kHeadDim;
constexpr double kAttentionScale    = 1.0 / 16.0;
// The staged routes weight V by FP16 probabilities (relative error about 2^-11). This fixture's V
// reaches 10.8, so their absolute criterion is 1e-3 of that peak; the FP32 fused routes keep 2e-4.
constexpr double kStagedTolerance = 1.1e-2;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

float represented_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

std::uint32_t mix(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    std::uint32_t hash = a * 747796405U + b * 2891336453U + c * 277803737U + 12345U;
    hash               = ((hash >> ((hash >> 28U) + 4U)) ^ hash) * 277803737U;
    return (hash >> 22U) ^ hash;
}

// Groups span about seven octaves so the E4M3 scales vary, as captured model K and V do.
std::vector<float> make_cache_plane(std::size_t tokens, std::uint32_t seed) {
    std::vector<float> plane(tokens * kTokenItems);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            for (std::size_t f = 0; f < kHeadDim; ++f) {
                const std::uint32_t group_hash =
                    mix(static_cast<std::uint32_t>(t), static_cast<std::uint32_t>(h) + seed,
                        static_cast<std::uint32_t>(f / 16U));
                const float octave = std::ldexp(1.0F, static_cast<int>(group_hash % 7U) - 4);
                const std::uint32_t hash =
                    mix(static_cast<std::uint32_t>(t) + seed, static_cast<std::uint32_t>(h),
                        static_cast<std::uint32_t>(f));
                plane[(t * kKvHeads + h) * kHeadDim + f] =
                    represented_bf16(octave * (static_cast<float>(hash % 201U) - 100.0F) / 37.0F);
            }
        }
    }
    return plane;
}

std::vector<float> make_queries(std::size_t rows, std::uint32_t seed = 11U) {
    std::vector<float> query(rows * kRowElements);
    for (std::size_t i = 0; i < query.size(); ++i) {
        const std::uint32_t hash = mix(static_cast<std::uint32_t>(i), seed, 3U);
        query[i] = represented_bf16((static_cast<float>(hash % 129U) - 64.0F) / 64.0F);
    }
    return query;
}

std::vector<hip_bfloat16> as_bf16(std::span<const float> represented) {
    std::vector<hip_bfloat16> result;
    result.reserve(represented.size());
    for (const float value : represented) { result.emplace_back(value); }
    return result;
}

std::vector<float> reorder_for_tokens(std::span<const float> plane, std::size_t first,
                                      std::size_t count) {
    return std::vector<float>(plane.begin() + static_cast<std::ptrdiff_t>(first * kTokenItems),
                              plane.begin() +
                                  static_cast<std::ptrdiff_t>((first + count) * kTokenItems));
}

// Independent physical layout: K and K scales token-fastest head-major, V and V scales
// feature-fastest page-major.
std::size_t key_offset(std::size_t physical, std::size_t head, std::size_t byte, std::size_t pages,
                       std::size_t leading) {
    return physical % kPageSize +
           kPageSize * (byte + leading * (physical / kPageSize + pages * head));
}

std::size_t value_offset(std::size_t physical, std::size_t head, std::size_t byte,
                         std::size_t leading) {
    return byte + leading * (physical % kPageSize +
                             kPageSize * (head + kKvHeads * (physical / kPageSize)));
}

struct Fixture {
    ninfer::DeviceContext& device;
    std::uint32_t physical_pages;
    std::uint32_t logical_pages;
    ninfer::Fp8KInt4VPagedKVSpec spec;
    std::unique_ptr<ninfer::DeviceBuffer> backing;
    std::unique_ptr<q3::PagedKVCache> cache;
    std::unique_ptr<ninfer::DeviceBuffer> positions;
    std::unique_ptr<ninfer::DeviceBuffer> status;

    Fixture(ninfer::DeviceContext& context, std::uint32_t physical, std::uint32_t logical,
            std::int32_t table_rows)
        : device(context), physical_pages(physical), logical_pages(logical),
          spec{.format                = ninfer::KvCacheFormat::Nvfp4,
               .page_group_count      = physical,
               .logical_page_capacity = logical,
               .table_rows            = table_rows,
               .layer_count           = 1,
               .head_dim              = static_cast<std::int32_t>(kHeadDim),
               .num_kv_heads          = static_cast<std::int32_t>(kKvHeads),
               .value_group           = 16,
               .plane_layouts = {.key   = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                                 .value = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                                 .value_scale =
                                     ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor}} {
        ninfer::LayoutBuilder builder;
        const q3::PagedKVCacheLayout layout{
            .storage     = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec),
            .layers      = 1U,
            .max_context = logical * static_cast<std::uint32_t>(kPageSize)};
        backing =
            std::make_unique<ninfer::DeviceBuffer>(builder.finish(256, "NVFP4 qualification"));
        cache = std::make_unique<q3::PagedKVCache>(
            ninfer::DeviceSpan{backing->data(), backing->size()}, layout);
        // Stale device memory: E2M1 codes of -6 and NaN E4M3 scales in every unwritten slot.
        for (std::size_t plane = 0; plane < cache->pool().plane_count(); ++plane) {
            HIP_CHECK(hipMemsetAsync(cache->pool().plane(plane).data, 0xFF,
                                     cache->pool().plane(plane).bytes(), device.stream));
        }
        positions = std::make_unique<ninfer::DeviceBuffer>(static_cast<std::size_t>(logical) *
                                                           kPageSize * sizeof(std::uint32_t));
        status    = std::make_unique<ninfer::DeviceBuffer>(sizeof(std::uint32_t));
    }

    [[nodiscard]] q3::PagedKVTransactionWorkspace workspace() const {
        return {.positions         = static_cast<std::uint32_t*>(positions->data()),
                .position_capacity = static_cast<std::size_t>(logical_pages) * kPageSize,
                .status            = static_cast<std::uint32_t*>(status->data())};
    }

    // A fragmented allocation: an occupied page separates the first two of its physical pages.
    [[nodiscard]] std::pair<ninfer::PagedKVAllocation, ninfer::PagedKVAllocation>
    fragmented(std::uint32_t pages, std::int32_t row) {
        auto first    = cache->pool().reserve(1U);
        auto occupied = cache->pool().reserve(1U);
        auto rest     = cache->pool().reserve(pages - 1U);
        first.materialize_pages(1U, device.stream);
        occupied.materialize_pages(1U, device.stream);
        rest.materialize_pages(pages - 1U, device.stream);
        first.release();
        rest.release();
        auto allocation = cache->pool().reserve(pages);
        allocation.materialize_pages(pages, device.stream);
        allocation.bind_row(row, device.stream);
        return {std::move(allocation), std::move(occupied)};
    }

    void append(ninfer::PagedKVAllocation& allocation, q3::PagedKVPublication& publication,
                std::span<const float> keys, std::span<const float> values, std::size_t first,
                std::size_t count) {
        std::vector<std::uint32_t> logical(count);
        std::iota(logical.begin(), logical.end(), static_cast<std::uint32_t>(first));
        const auto k = as_bf16(reorder_for_tokens(keys, first, count));
        const auto v = as_bf16(reorder_for_tokens(values, first, count));
        ninfer::DeviceBuffer dk(k.size() * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer dv(v.size() * sizeof(hip_bfloat16));
        dk.copy_from_host(k.data(), dk.size());
        dv.copy_from_host(v.data(), dv.size());
        auto transaction = cache->begin_append(allocation, publication, logical, workspace());
        transaction.launch_append_layer(0U, static_cast<const hip_bfloat16*>(dk.data()),
                                        static_cast<const hip_bfloat16*>(dv.data()), device.stream);
        if (transaction.commit() != first + count) { fail("append published a wrong frontier"); }
    }

    // Appends in chunks that begin mid-page and cross pages.
    void append_all(ninfer::PagedKVAllocation& allocation, q3::PagedKVPublication& publication,
                    std::span<const float> keys, std::span<const float> values) {
        const std::size_t tokens = keys.size() / kTokenItems;
        std::size_t position     = publication.valid_frontier;
        std::size_t chunk        = 7U;
        while (position < tokens) {
            const std::size_t count = std::min(chunk, tokens - position);
            append(allocation, publication, keys, values, position, count);
            position += count;
            chunk = chunk >= 2048U ? 2048U : chunk * 5U;
        }
    }

    [[nodiscard]] std::vector<std::uint8_t> read_plane(const ninfer::Tensor& plane) {
        device.synchronize();
        std::vector<std::uint8_t> bytes(plane.bytes());
        HIP_CHECK(hipMemcpy(bytes.data(), plane.data, bytes.size(), hipMemcpyDeviceToHost));
        return bytes;
    }

    // Every logical token's bytes in all four planes equal the independent encoding.
    void require_bytes(const ninfer::PagedKVAllocation& allocation, const oracle::Encoded& keys,
                       const oracle::Encoded& values, std::size_t tokens, const char* label) {
        const auto view        = cache->layer_view(0, allocation);
        const auto key_bytes   = read_plane(view.key_pages);
        const auto kscale      = read_plane(view.key_scale_pages);
        const auto value_bytes = read_plane(view.value_pages);
        const auto vscale      = read_plane(view.value_scale_pages);
        const auto pages       = allocation.page_ids();
        for (std::size_t t = 0; t < tokens; ++t) {
            const std::size_t physical =
                static_cast<std::size_t>(pages[t / kPageSize]) * kPageSize + t % kPageSize;
            for (std::size_t h = 0; h < kKvHeads; ++h) {
                const std::size_t item = t * kKvHeads + h;
                for (std::size_t byte = 0; byte < kHeadDim / 2U; ++byte) {
                    if (key_bytes[key_offset(physical, h, byte, physical_pages, 128U)] !=
                        keys.packed_codes[item * (kHeadDim / 2U) + byte]) {
                        fail(std::string(label) + ": K code byte differs at token " +
                             std::to_string(t));
                    }
                    if (value_bytes[value_offset(physical, h, byte, 128U)] !=
                        values.packed_codes[item * (kHeadDim / 2U) + byte]) {
                        fail(std::string(label) + ": V code byte differs at token " +
                             std::to_string(t));
                    }
                }
                for (std::size_t group = 0; group < kGroups; ++group) {
                    if (kscale[key_offset(physical, h, group, physical_pages, 16U)] !=
                        keys.scales[item * kGroups + group]) {
                        fail(std::string(label) + ": K scale differs at token " +
                             std::to_string(t));
                    }
                    if (vscale[value_offset(physical, h, group, 16U)] !=
                        values.scales[item * kGroups + group]) {
                        fail(std::string(label) + ": V scale differs at token " +
                             std::to_string(t));
                    }
                }
            }
        }
    }
};

struct Expected {
    std::vector<double> values;
};

// FP64 attention for causal row frontiers, optionally restricted by tree metadata.
std::vector<double> oracle_output(std::span<const float> queries, std::size_t rows,
                                  std::span<const std::int32_t> row_positions,
                                  const oracle::Encoded& keys, const oracle::Encoded& values,
                                  std::span<const std::int32_t> prefixes = {},
                                  std::span<const std::int32_t> masks    = {}) {
    std::vector<double> output(rows * kRowElements);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto visible = static_cast<std::size_t>(row_positions[row]) + 1U;
        std::vector<std::size_t> selected;
        selected.reserve(visible);
        for (std::size_t logical = 0; logical < visible; ++logical) {
            if (prefixes.empty() || logical < static_cast<std::size_t>(prefixes[row]) ||
                ((static_cast<std::uint32_t>(masks[row]) >>
                  (logical - static_cast<std::size_t>(prefixes[row]))) &
                 1U) != 0U) {
                selected.push_back(logical);
            }
        }
        for (std::size_t head = 0; head < kQueryHeads; ++head) {
            const std::size_t base = (row * kQueryHeads + head) * kHeadDim;
            const auto result =
                oracle::attention_fp64(queries.subspan(base, kHeadDim), selected, keys, values,
                                       kKvHeads, head / kQueriesPerKv, kAttentionScale);
            std::copy(result.begin(), result.end(),
                      output.begin() + static_cast<std::ptrdiff_t>(base));
        }
    }
    return output;
}

void compare(std::span<const float> actual, std::span<const double> expected, double tolerance,
             const std::string& label, double* maximum = nullptr) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double absolute = std::abs(static_cast<double>(actual[i]) - expected[i]);
        const double relative = absolute / std::max(std::abs(expected[i]), 1.0e-6);
        if (!std::isfinite(actual[i]) || (absolute > tolerance && relative > tolerance)) {
            fail(label + ": FP64 mismatch at " + std::to_string(i) + " actual=" +
                 std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
        }
        if (maximum != nullptr) { *maximum = std::max(*maximum, absolute); }
    }
}

struct LeafBuffers {
    ninfer::DeviceBuffer query, output, workspace, row_positions;

    LeafBuffers(std::size_t rows, std::size_t workspace_bytes)
        : query(rows * kRowElements * sizeof(hip_bfloat16)),
          output(rows * kRowElements * sizeof(float)),
          workspace(std::max<std::size_t>(workspace_bytes, 4096U)),
          row_positions(rows * sizeof(std::int32_t)) {}
};

q27::R9700FullAttentionArgs leaf_args(LeafBuffers& buffers, const q3::PagedKVLayerRead& read,
                                      std::size_t rows, std::size_t workspace_bytes) {
    const auto extent = static_cast<std::int32_t>(rows);
    return {.query = ninfer::Tensor(buffers.query.data(), ninfer::DType::BF16, {256, 24, extent}),
            .cache_read      = read,
            .row_positions   = static_cast<const std::int32_t*>(buffers.row_positions.data()),
            .workspace       = workspace_bytes == 0U ? nullptr : buffers.workspace.data(),
            .workspace_bytes = workspace_bytes,
            .output =
                ninfer::Tensor(buffers.output.data(), ninfer::DType::FP32, {256, 24, extent})};
}

// ---------------------------------------------------------------------------------- oracle/codec
void qualify_oracle_self() {
    constexpr std::array<std::pair<float, std::uint8_t>, 17> cases{{
        {0.0F, 0},
        {0.25F, 0},
        {0.26F, 1},
        {0.5F, 1},
        {0.75F, 2},
        {1.0F, 2},
        {1.25F, 2},
        {1.3F, 3},
        {1.75F, 4},
        {2.5F, 4},
        {2.6F, 5},
        {3.5F, 6},
        {5.0F, 6},
        {5.1F, 7},
        {6.0F, 7},
        {100.0F, 7},
        {-0.0F, 8},
    }};
    for (const auto& [quotient, code] : cases) {
        if (oracle::encode_e2m1(quotient) != code) { fail("E2M1 oracle self-check failed"); }
    }
    if (oracle::encode_e2m1(-0.75F) != 10U || oracle::decode_e2m1(15U) != -6.0) {
        fail("E2M1 oracle sign self-check failed");
    }
}

// Exact stored bytes across codec edge groups, fragmented pages and mid-page chunked appends.
void qualify_codec_bytes(ninfer::DeviceContext& device) {
    constexpr std::size_t tokens = 203U;
    auto keys                    = make_cache_plane(tokens, 1U);
    auto values                  = make_cache_plane(tokens, 2U);
    const auto set_group         = [](std::vector<float>& plane, std::size_t t, std::size_t h,
                                      std::size_t g, const std::array<float, 16>& group) {
        std::copy(group.begin(), group.end(),
                  plane.begin() +
                      static_cast<std::ptrdiff_t>((t * kKvHeads + h) * kHeadDim + g * 16U));
    };
    // max 6 gives scale 1: exact midpoints and both signs of each tie.
    const std::array<float, 16> ties{0.25F,  0.75F,  1.25F,  1.75F,  2.5F,  3.5F,  5.0F,  6.0F,
                                     -0.25F, -0.75F, -1.25F, -1.75F, -2.5F, -3.5F, -5.0F, -6.0F};
    const std::array<float, 16> signed_zero{-0.0F, 0.0F, 3.0F, -0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                            0.0F,  0.0F, 0.0F, 0.0F,  0.0F, 0.0F, 0.0F, 0.0F};
    std::array<float, 16> zero{};
    std::array<float, 16> tiny{};
    tiny.fill(represented_bf16(1.0e-3F)); // rounds to a zero scale: all-zero stored codes
    std::array<float, 16> small{};
    small.fill(represented_bf16(0.02F)); // an E4M3 subnormal scale
    std::array<float, 16> huge{};
    huge.fill(represented_bf16(1.0e4F)); // saturates the scale at 448, codes at 6
    huge[1] = represented_bf16(-3000.0F);
    set_group(keys, 3, 1, 2, ties);
    set_group(values, 3, 1, 2, ties);
    set_group(keys, 4, 0, 0, signed_zero);
    set_group(values, 4, 0, 0, signed_zero);
    set_group(keys, 64, 2, 15, zero);
    set_group(values, 64, 2, 15, zero);
    set_group(keys, 65, 3, 7, tiny);
    set_group(values, 65, 3, 7, tiny);
    set_group(keys, 127, 0, 3, small);
    set_group(values, 127, 0, 3, small);
    set_group(keys, 128, 1, 9, huge);
    set_group(values, 128, 1, 9, huge);

    Fixture fixture(device, 6U, 5U, 1);
    auto [allocation, occupied] = fixture.fragmented(5U, 0);
    q3::PagedKVPublication publication;
    fixture.append_all(allocation, publication, keys, values);
    const auto expected_keys   = oracle::encode(keys, tokens * kKvHeads, kHeadDim);
    const auto expected_values = oracle::encode(values, tokens * kKvHeads, kHeadDim);
    fixture.require_bytes(allocation, expected_keys, expected_values, tokens, "codec bytes");
    if (fixture.cache->fingerprint().format != ninfer::KvCacheFormat::Nvfp4) {
        fail("cache fingerprint lost its format");
    }
    std::cout << "codec_bytes tokens=" << tokens << " ties/signs/zero/subnormal/saturation=PASS\n";
}

// Rejected inputs set their status bit, poison the sequence and never move the frontier.
void qualify_status(ninfer::DeviceContext& device) {
    const auto run = [&](const char* label, float bad_key, float bad_value,
                         std::uint32_t expected) {
        constexpr std::size_t tokens = 3U;
        auto keys                    = make_cache_plane(tokens, 3U);
        auto values                  = make_cache_plane(tokens, 4U);
        keys[kTokenItems + 17U]      = bad_key;
        values[kTokenItems + 18U]    = bad_value;
        Fixture fixture(device, 2U, 2U, 1);
        auto allocation = fixture.cache->pool().reserve(2U);
        allocation.materialize_pages(2U, device.stream);
        allocation.bind_row(0, device.stream);
        q3::PagedKVPublication publication;
        std::vector<std::uint32_t> logical{0U, 1U, 2U};
        const auto k = as_bf16(keys);
        const auto v = as_bf16(values);
        ninfer::DeviceBuffer dk(k.size() * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer dv(v.size() * sizeof(hip_bfloat16));
        dk.copy_from_host(k.data(), dk.size());
        dv.copy_from_host(v.data(), dv.size());
        auto transaction =
            fixture.cache->begin_append(allocation, publication, logical, fixture.workspace());
        transaction.launch_append_layer(0U, static_cast<const hip_bfloat16*>(dk.data()),
                                        static_cast<const hip_bfloat16*>(dv.data()), device.stream);
        device.synchronize();
        std::uint32_t status = 0U;
        fixture.status->copy_to_host(&status, sizeof(status));
        bool rejected = false;
        try {
            (void)transaction.commit();
        } catch (const std::runtime_error&) { rejected = true; }
        if (status != expected || !rejected || publication.healthy ||
            publication.valid_frontier != 0U) {
            fail(std::string(label) + " did not reject its transaction, status " +
                 std::to_string(status));
        }
    };
    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    run("nonfinite K", kInf, 0.5F, kv::Fp8Int4KvAppendNonfiniteInput);
    run("NaN V", 0.5F, kNan, kv::Fp8Int4KvAppendNonfiniteInput);

    // Malformed device positions and table rows through a device-position transaction.
    constexpr std::size_t tokens = 2U;
    const auto keys              = as_bf16(make_cache_plane(tokens, 5U));
    const auto values            = as_bf16(make_cache_plane(tokens, 6U));
    ninfer::DeviceBuffer dk(keys.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer dv(values.size() * sizeof(hip_bfloat16));
    dk.copy_from_host(keys.data(), dk.size());
    dv.copy_from_host(values.data(), dv.size());
    const auto device_positions = [&](const char* label, std::array<std::int32_t, 2> positions,
                                      std::uint32_t expected) {
        Fixture fixture(device, 2U, 2U, 1);
        auto allocation = fixture.cache->pool().reserve(2U);
        allocation.materialize_pages(2U, device.stream);
        allocation.bind_row(0, device.stream);
        q3::PagedKVPublication publication;
        ninfer::DeviceBuffer panel(sizeof(positions));
        panel.copy_from_host(positions.data(), sizeof(positions));
        auto transaction = fixture.cache->begin_device_append(
            allocation, publication, static_cast<const std::int32_t*>(panel.data()), tokens,
            fixture.workspace());
        transaction.launch_append_layer(0U, static_cast<const hip_bfloat16*>(dk.data()),
                                        static_cast<const hip_bfloat16*>(dv.data()), device.stream);
        device.synchronize();
        std::uint32_t status = 0U;
        fixture.status->copy_to_host(&status, sizeof(status));
        if ((status & expected) == 0U) {
            fail(std::string(label) + " did not set its status, got " + std::to_string(status));
        }
        transaction.abort();
    };
    device_positions("gapped positions", {0, 5}, kv::Fp8Int4KvAppendInvalidPosition);
    std::cout << "status nonfinite/position/poison=PASS\n";
}

// The one-launch batch and every device-count form store the same bytes.
void qualify_append_forms(ninfer::DeviceContext& device) {
    constexpr std::size_t tokens = 70U;
    const auto keys_a            = make_cache_plane(tokens, 7U);
    const auto values_a          = make_cache_plane(tokens, 8U);
    const auto keys_b            = make_cache_plane(tokens, 9U);
    const auto values_b          = make_cache_plane(tokens, 10U);
    const auto expected_ka       = oracle::encode(keys_a, tokens * kKvHeads, kHeadDim);
    const auto expected_va       = oracle::encode(values_a, tokens * kKvHeads, kHeadDim);
    const auto expected_kb       = oracle::encode(keys_b, tokens * kKvHeads, kHeadDim);
    const auto expected_vb       = oracle::encode(values_b, tokens * kKvHeads, kHeadDim);
    {
        Fixture fixture(device, 6U, 2U, 2);
        auto a = fixture.cache->pool().reserve(2U);
        auto b = fixture.cache->pool().reserve(2U);
        a.materialize_pages(2U, device.stream);
        b.materialize_pages(2U, device.stream);
        a.bind_row(0, device.stream);
        b.bind_row(1, device.stream);
        q3::PagedKVPublication publication_a, publication_b;
        const auto k_a = as_bf16(keys_a), v_a = as_bf16(values_a);
        const auto k_b = as_bf16(keys_b), v_b = as_bf16(values_b);
        ninfer::DeviceBuffer dk_a(k_a.size() * 2U), dv_a(v_a.size() * 2U);
        ninfer::DeviceBuffer dk_b(k_b.size() * 2U), dv_b(v_b.size() * 2U);
        dk_a.copy_from_host(k_a.data(), dk_a.size());
        dv_a.copy_from_host(v_a.data(), dv_a.size());
        dk_b.copy_from_host(k_b.data(), dk_b.size());
        dv_b.copy_from_host(v_b.data(), dv_b.size());
        std::vector<std::uint32_t> logical(tokens);
        std::iota(logical.begin(), logical.end(), 0U);
        ninfer::DeviceBuffer positions_b(tokens * sizeof(std::uint32_t));
        ninfer::DeviceBuffer status_b(sizeof(std::uint32_t));
        auto transaction_a =
            fixture.cache->begin_append(a, publication_a, logical, fixture.workspace());
        auto transaction_b = fixture.cache->begin_append(
            b, publication_b, logical,
            {.positions         = static_cast<std::uint32_t*>(positions_b.data()),
             .position_capacity = tokens,
             .status            = static_cast<std::uint32_t*>(status_b.data())});
        std::array<q3::PagedKVTransaction*, 2> batch{&transaction_a, &transaction_b};
        std::array<const hip_bfloat16*, 2> batch_k{static_cast<const hip_bfloat16*>(dk_a.data()),
                                                   static_cast<const hip_bfloat16*>(dk_b.data())};
        std::array<const hip_bfloat16*, 2> batch_v{static_cast<const hip_bfloat16*>(dv_a.data()),
                                                   static_cast<const hip_bfloat16*>(dv_b.data())};
        q3::PagedKVTransaction::launch_append_layers(batch, 0U, batch_k, batch_v, device.stream);
        if (transaction_a.commit() != tokens || transaction_b.commit() != tokens) {
            fail("batched append published a wrong frontier");
        }
        fixture.require_bytes(a, expected_ka, expected_va, tokens, "batched append sequence 0");
        fixture.require_bytes(b, expected_kb, expected_vb, tokens, "batched append sequence 1");
    }
    {
        // Device prefix: the count selects what is stored and published, including zero.
        for (const std::int32_t count : {0, 17, 70}) {
            Fixture fixture(device, 2U, 2U, 1);
            auto allocation = fixture.cache->pool().reserve(2U);
            allocation.materialize_pages(2U, device.stream);
            allocation.bind_row(0, device.stream);
            q3::PagedKVPublication publication;
            const auto k = as_bf16(keys_a), v = as_bf16(values_a);
            ninfer::DeviceBuffer dk(k.size() * 2U), dv(v.size() * 2U);
            dk.copy_from_host(k.data(), dk.size());
            dv.copy_from_host(v.data(), dv.size());
            std::vector<std::int32_t> logical(tokens);
            std::iota(logical.begin(), logical.end(), 0);
            ninfer::DeviceBuffer panel(tokens * sizeof(std::int32_t));
            ninfer::DeviceBuffer device_count(sizeof(std::int32_t));
            panel.copy_from_host(logical.data(), panel.size());
            device_count.copy_from_host(&count, sizeof(count));
            auto transaction = fixture.cache->begin_device_prefix_append(
                allocation, publication, static_cast<const std::int32_t*>(panel.data()), tokens,
                static_cast<const std::int32_t*>(device_count.data()), fixture.workspace());
            transaction.launch_append_layer(0U, static_cast<const hip_bfloat16*>(dk.data()),
                                            static_cast<const hip_bfloat16*>(dv.data()),
                                            device.stream);
            if (transaction.commit() != static_cast<std::uint32_t>(count)) {
                fail("device-prefix append published a wrong frontier");
            }
            fixture.require_bytes(allocation, expected_ka, expected_va,
                                  static_cast<std::size_t>(count), "device-prefix append");
            const auto view       = fixture.cache->layer_view(0, allocation);
            const auto key_scales = fixture.read_plane(view.key_scale_pages);
            const auto pages      = allocation.page_ids();
            if (static_cast<std::size_t>(count) < tokens) {
                const std::size_t physical =
                    static_cast<std::size_t>(pages[static_cast<std::size_t>(count) / kPageSize]) *
                        kPageSize +
                    static_cast<std::size_t>(count) % kPageSize;
                if (key_scales[key_offset(physical, 0U, 0U, 2U, 16U)] != 0xFFU) {
                    fail("device-prefix append wrote past its active count");
                }
            }
        }
        // A malformed count is rejected.
        Fixture fixture(device, 2U, 2U, 1);
        auto allocation = fixture.cache->pool().reserve(2U);
        allocation.materialize_pages(2U, device.stream);
        allocation.bind_row(0, device.stream);
        q3::PagedKVPublication publication;
        const auto k = as_bf16(keys_a), v = as_bf16(values_a);
        ninfer::DeviceBuffer dk(k.size() * 2U), dv(v.size() * 2U);
        dk.copy_from_host(k.data(), dk.size());
        dv.copy_from_host(v.data(), dv.size());
        std::vector<std::int32_t> logical(tokens);
        std::iota(logical.begin(), logical.end(), 0);
        ninfer::DeviceBuffer panel(tokens * sizeof(std::int32_t));
        ninfer::DeviceBuffer device_count(sizeof(std::int32_t));
        panel.copy_from_host(logical.data(), panel.size());
        const std::int32_t bad_count = 71;
        device_count.copy_from_host(&bad_count, sizeof(bad_count));
        auto transaction = fixture.cache->begin_device_prefix_append(
            allocation, publication, static_cast<const std::int32_t*>(panel.data()), tokens,
            static_cast<const std::int32_t*>(device_count.data()), fixture.workspace());
        transaction.launch_append_layer(0U, static_cast<const hip_bfloat16*>(dk.data()),
                                        static_cast<const hip_bfloat16*>(dv.data()), device.stream);
        bool rejected = false;
        try {
            (void)transaction.commit();
        } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected || publication.healthy) { fail("over-width device count was committed"); }
    }
    {
        // Segmented MTP round: two segments advance one device cursor; the retained prefix
        // publishes.
        Fixture fixture(device, 2U, 2U, 1);
        auto allocation = fixture.cache->pool().reserve(2U);
        allocation.materialize_pages(2U, device.stream);
        allocation.bind_row(0, device.stream);
        q3::PagedKVPublication publication;
        ninfer::DeviceBuffer cursor(sizeof(std::uint32_t));
        ninfer::DeviceBuffer base(sizeof(std::int32_t));
        const std::int32_t zero = 0;
        base.copy_from_host(&zero, sizeof(zero));
        const auto k = as_bf16(keys_a), v = as_bf16(values_a);
        constexpr std::size_t first_count = 30U, second_count = 40U;
        ninfer::DeviceBuffer dk(k.size() * 2U), dv(v.size() * 2U);
        dk.copy_from_host(k.data(), dk.size());
        dv.copy_from_host(v.data(), dv.size());
        std::vector<std::int32_t> logical(tokens);
        std::iota(logical.begin(), logical.end(), 0);
        ninfer::DeviceBuffer panel(tokens * sizeof(std::int32_t));
        panel.copy_from_host(logical.data(), panel.size());
        auto workspace   = fixture.workspace();
        workspace.cursor = static_cast<std::uint32_t*>(cursor.data());
        auto transaction = fixture.cache->begin_device_segmented_append(
            allocation, publication, static_cast<const std::int32_t*>(base.data()), 0U, 0U,
            static_cast<std::uint32_t>(tokens), workspace);
        const auto* panel_data = static_cast<const std::int32_t*>(panel.data());
        const auto* key_data   = static_cast<const hip_bfloat16*>(dk.data());
        const auto* value_data = static_cast<const hip_bfloat16*>(dv.data());
        transaction.begin_device_segment(panel_data, first_count, nullptr);
        transaction.launch_append_layer(0U, key_data, value_data, device.stream);
        transaction.end_device_segment(device.stream);
        transaction.begin_device_segment(panel_data + first_count, second_count, nullptr);
        transaction.launch_append_layer(0U, key_data + first_count * kTokenItems,
                                        value_data + first_count * kTokenItems, device.stream);
        transaction.end_device_segment(device.stream);
        std::uint32_t status = 1U, resolved_cursor = 0U;
        transaction.enqueue_segmented_resolution(&status, &resolved_cursor);
        device.synchronize();
        if (status != 0U || resolved_cursor != tokens) {
            fail("segmented append resolved a wrong cursor");
        }
        transaction.finish_segmented_resolution(status, resolved_cursor, tokens);
        fixture.require_bytes(allocation, expected_ka, expected_va, tokens, "segmented append");
    }
    std::cout << "append_forms batched/device_prefix/invalid_count/segmented=PASS\n";
}

// Compaction copies all four planes exactly; rollback publishes a prefix without touching bytes.
void qualify_compaction(ninfer::DeviceContext& device) {
    constexpr std::size_t tokens = 150U;
    const auto keys              = make_cache_plane(tokens, 12U);
    const auto values            = make_cache_plane(tokens, 13U);
    Fixture fixture(device, 4U, 3U, 1);
    auto [allocation, occupied] = fixture.fragmented(3U, 0);
    q3::PagedKVPublication publication;
    fixture.append_all(allocation, publication, keys, values);
    constexpr std::uint32_t prefix = 60U;
    const std::vector<std::uint32_t> path{0U, 2U, 5U, 9U, 10U, 40U, 41U, 70U, 89U};
    auto compact_keys   = reorder_for_tokens(keys, 0U, prefix + path.size());
    auto compact_values = reorder_for_tokens(values, 0U, prefix + path.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
        std::copy_n(keys.begin() + static_cast<std::ptrdiff_t>((prefix + path[i]) * kTokenItems),
                    kTokenItems,
                    compact_keys.begin() + static_cast<std::ptrdiff_t>((prefix + i) * kTokenItems));
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>((prefix + path[i]) * kTokenItems),
                    kTokenItems,
                    compact_values.begin() +
                        static_cast<std::ptrdiff_t>((prefix + i) * kTokenItems));
    }
    auto transaction = fixture.cache->begin_compact(allocation, publication, fixture.workspace());
    transaction.launch_compact_layer(0U, prefix, path, device.stream);
    if (transaction.commit() != prefix + path.size()) {
        fail("compaction published a wrong frontier");
    }
    const std::size_t kept = prefix + path.size();
    const auto expected_k  = oracle::encode(compact_keys, kept * kKvHeads, kHeadDim);
    const auto expected_v  = oracle::encode(compact_values, kept * kKvHeads, kHeadDim);
    fixture.require_bytes(allocation, expected_k, expected_v, kept, "compaction");
    fixture.cache->truncate_publication(allocation, publication, prefix);
    if (publication.valid_frontier != prefix) { fail("rollback did not retain its prefix"); }
    fixture.require_bytes(allocation, expected_k, expected_v, prefix, "rollback");
    // Appending after rollback overwrites the suffix exactly.
    std::vector<float> resumed_k(compact_keys), resumed_v(compact_values);
    const auto extra_k = make_cache_plane(kept, 14U), extra_v = make_cache_plane(kept, 15U);
    std::copy(extra_k.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems), extra_k.end(),
              resumed_k.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems));
    std::copy(extra_v.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems), extra_v.end(),
              resumed_v.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems));
    fixture.append(allocation, publication, resumed_k, resumed_v, prefix, path.size());
    fixture.require_bytes(allocation, oracle::encode(resumed_k, kept * kKvHeads, kHeadDim),
                          oracle::encode(resumed_v, kept * kKvHeads, kHeadDim), kept,
                          "append after rollback");
    std::cout << "compaction copy/rollback/resume=PASS\n";
}

// ------------------------------------------------------------------------------- attention routes
struct RouteCache {
    std::size_t context;
    std::vector<float> keys, values;
    oracle::Encoded encoded_keys, encoded_values;
    Fixture fixture;
    ninfer::PagedKVAllocation allocation, occupied;
    q3::PagedKVPublication publication;

    RouteCache(ninfer::DeviceContext& device, std::size_t tokens, std::uint32_t seed)
        : context(tokens), keys(make_cache_plane(tokens, seed)),
          values(make_cache_plane(tokens, seed + 1U)),
          encoded_keys(oracle::encode(keys, tokens * kKvHeads, kHeadDim)),
          encoded_values(oracle::encode(values, tokens * kKvHeads, kHeadDim)),
          fixture(device, static_cast<std::uint32_t>((tokens + 63U) / 64U + 2U),
                  static_cast<std::uint32_t>((tokens + 63U) / 64U + 1U), 2) {
        auto allocated =
            fixture.fragmented(static_cast<std::uint32_t>((tokens + 63U) / 64U + 1U), 0);
        allocation = std::move(allocated.first);
        occupied   = std::move(allocated.second);
        fixture.append_all(allocation, publication, keys, values);
    }
};

void qualify_causal_routes(ninfer::DeviceContext& device) {
    double packed_max = 0.0, dense_max = 0.0, fused_max = 0.0;
    const auto cell = [&](RouteCache& cache, std::size_t rows, double tolerance, double& maximum) {
        const auto queries = make_queries(rows, static_cast<std::uint32_t>(rows * 3U + 1U));
        const auto bf16    = as_bf16(queries);
        std::vector<std::int32_t> positions(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            positions[r] = static_cast<std::int32_t>(cache.context - rows + r);
        }
        const std::size_t workspace_bytes = q27::r9700_full_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(rows), cache.context, false);
        LeafBuffers buffers(rows, workspace_bytes);
        buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
        buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
        const auto read =
            cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(
            leaf_args(buffers, read, rows, workspace_bytes), device.stream));
        device.synchronize();
        std::vector<float> actual(rows * kRowElements);
        buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        compare(actual,
                oracle_output(queries, rows, positions, cache.encoded_keys, cache.encoded_values),
                tolerance,
                "context " + std::to_string(cache.context) + " rows " + std::to_string(rows),
                &maximum);
    };
    {
        // Below the packed minimum: the fused route.
        RouteCache cache(device, 33U, 21U);
        for (const std::size_t rows : {1U, 3U, 8U, 33U}) {
            if (kv::use_packed_decode_attention(static_cast<std::uint32_t>(rows), cache.context,
                                                false)) {
                fail("context 33 unexpectedly selects packed decode");
            }
            cell(cache, rows, rows >= 9U ? kStagedTolerance : 2.0e-4,
                 rows >= 9U ? dense_max : fused_max);
        }
    }
    {
        RouteCache cache(device, 257U, 23U);
        for (const std::size_t rows : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
            cell(cache, rows, kStagedTolerance, packed_max);
        }
        for (const std::size_t rows : {9U, 17U, 127U, 128U, 129U, 257U}) {
            cell(cache, rows, kStagedTolerance, dense_max);
        }
    }
    {
        RouteCache cache(device, 2064U, 25U);
        for (const std::size_t rows : {1U, 6U, 8U}) {
            cell(cache, rows, kStagedTolerance, packed_max);
        }
        for (const std::size_t rows : {9U, 128U}) {
            cell(cache, rows, kStagedTolerance, dense_max);
        }
    }
    {
        RouteCache cache(device, 9000U, 27U);
        for (const std::size_t rows : {1U, 5U, 8U}) {
            cell(cache, rows, kStagedTolerance, packed_max);
        }
        if (!kv::use_mid_rows_attention(48U, cache.context, false)) {
            fail("context 9000 rows 48 does not select the mid-row route");
        }
        cell(cache, 48U, kStagedTolerance, dense_max);
    }
    std::cout << "causal_routes packed_max_abs=" << packed_max << " dense_mid_max_abs=" << dense_max
              << " fused_max_abs=" << fused_max << " PASS\n";
}

// Fused tree at short context, split-512 tree at long context, with rowwise and broadcast prefixes
// and device-selected rows whose inactive metadata is malformed.
void qualify_tree_routes(ninfer::DeviceContext& device) {
    double maximum = 0.0;
    for (const std::size_t context : {std::size_t{257U}, std::size_t{8200U}}) {
        RouteCache cache(device, context, static_cast<std::uint32_t>(context));
        constexpr std::size_t rows = 4U;
        const bool long_context    = kv::use_split512_attention(rows, context, true);
        if (long_context != (context >= 8192U)) { fail("tree route classifier changed"); }
        const auto queries = make_queries(rows, 41U);
        const auto bf16    = as_bf16(queries);
        const auto first   = static_cast<std::int32_t>(context) - 4;
        std::vector<std::int32_t> positions{first, first + 1, first + 2, first + 3};
        std::vector<std::int32_t> prefixes{first, first, first, first};
        std::vector<std::int32_t> masks{1, 3, 5, 13};
        const std::size_t workspace_bytes =
            q27::r9700_full_attention_workspace_capacity_bytes(rows, context, true);
        LeafBuffers buffers(rows, workspace_bytes);
        ninfer::DeviceBuffer d_masks(rows * sizeof(std::int32_t)),
            d_prefix(rows * sizeof(std::int32_t));
        ninfer::DeviceBuffer d_active(sizeof(std::int32_t));
        buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
        buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
        d_masks.copy_from_host(masks.data(), d_masks.size());
        d_prefix.copy_from_host(prefixes.data(), d_prefix.size());
        const auto read =
            cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
        auto args           = leaf_args(buffers, read, rows, workspace_bytes);
        args.ancestor_masks = static_cast<const std::int32_t*>(d_masks.data());
        args.prefix_lengths = static_cast<const std::int32_t*>(d_prefix.data());
        const auto expected = oracle_output(queries, rows, positions, cache.encoded_keys,
                                            cache.encoded_values, prefixes, masks);
        const auto run      = [&](const q27::R9700FullAttentionArgs& call) {
            HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(call, device.stream));
            device.synchronize();
            std::vector<float> actual(rows * kRowElements);
            buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            return actual;
        };
        compare(run(args), expected, 2.0e-4, "tree context " + std::to_string(context), &maximum);

        d_prefix.copy_from_host(prefixes.data(), sizeof(std::int32_t));
        auto broadcast                 = args;
        broadcast.prefix_length_stride = 0U;
        compare(run(broadcast), expected, 2.0e-4, "broadcast tree", &maximum);
        d_prefix.copy_from_host(prefixes.data(), d_prefix.size());

        // Two live rows; the tail's metadata is malformed and must not be read.
        std::vector<std::int32_t> partial_positions{first, first + 1, -1, -1};
        std::vector<std::int32_t> partial_prefixes{first, first, -1, -1};
        std::vector<std::int32_t> partial_masks{1, 3, 0, 0};
        const std::int32_t live = 2;
        buffers.row_positions.copy_from_host(partial_positions.data(), rows * sizeof(std::int32_t));
        d_prefix.copy_from_host(partial_prefixes.data(), d_prefix.size());
        d_masks.copy_from_host(partial_masks.data(), d_masks.size());
        d_active.copy_from_host(&live, sizeof(live));
        auto partial                = args;
        partial.active_query_rows   = static_cast<const std::int32_t*>(d_active.data());
        const auto partial_actual   = run(partial);
        const auto partial_expected = oracle_output(
            std::span<const float>(queries).first(2U * kRowElements), 2U, partial_positions,
            cache.encoded_keys, cache.encoded_values, partial_prefixes, partial_masks);
        compare(partial_actual, partial_expected, 2.0e-4, "partial tree", &maximum);
        if (!std::all_of(partial_actual.begin() + static_cast<std::ptrdiff_t>(2U * kRowElements),
                         partial_actual.end(),
                         [](float v) { return std::bit_cast<std::uint32_t>(v) == 0U; })) {
            fail("inactive tree tail is not exact positive zero");
        }
        // A bad device count poisons every row.
        const std::int32_t bad = 5;
        d_active.copy_from_host(&bad, sizeof(bad));
        const auto poisoned = run(partial);
        if (!std::all_of(poisoned.begin(), poisoned.end(), [](float v) { return std::isnan(v); })) {
            fail("invalid tree device count did not poison every row");
        }
    }
    std::cout << "tree_routes fused/split512 rowwise/broadcast/device_rows max_abs=" << maximum
              << " PASS\n";
}

// One device count over a fixed packed panel, and sequence batches on both loader-wave shapes.
void qualify_packed_forms(ninfer::DeviceContext& device) {
    RouteCache cache(device, 1000U, 51U);
    constexpr std::size_t rows = 6U;
    const auto queries         = make_queries(rows, 61U);
    const auto bf16            = as_bf16(queries);
    std::vector<std::int32_t> positions(rows);
    for (std::size_t r = 0; r < rows; ++r) {
        positions[r] = static_cast<std::int32_t>(cache.context - rows + r);
    }
    const std::size_t workspace_bytes =
        q27::r9700_full_attention_workspace_capacity_bytes(rows, cache.context, false);
    LeafBuffers buffers(rows, workspace_bytes);
    ninfer::DeviceBuffer d_active(sizeof(std::int32_t));
    buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
    const auto read =
        cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
    const auto expected =
        oracle_output(queries, rows, positions, cache.encoded_keys, cache.encoded_values);
    auto args              = leaf_args(buffers, read, rows, workspace_bytes);
    args.active_query_rows = static_cast<const std::int32_t*>(d_active.data());
    for (const std::int32_t count : {0, 3, 6, -1, 7, 3}) {
        auto panel       = positions;
        const bool valid = count >= 0 && count <= static_cast<std::int32_t>(rows);
        std::fill(panel.begin() + (valid ? count : 0), panel.end(), -777);
        buffers.row_positions.copy_from_host(panel.data(), rows * sizeof(std::int32_t));
        d_active.copy_from_host(&count, sizeof(count));
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
        device.synchronize();
        std::vector<float> actual(rows * kRowElements);
        buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!valid) {
                if (!std::isnan(actual[i])) { fail("invalid packed count did not poison"); }
            } else if (i >= static_cast<std::size_t>(count) * kRowElements) {
                if (std::bit_cast<std::uint32_t>(actual[i]) != 0U) {
                    fail("inactive packed row is not exact positive zero");
                }
            } else {
                const double e = std::abs(actual[i] - expected[i]);
                if (!std::isfinite(actual[i]) ||
                    (e > kStagedTolerance &&
                     e / std::max(std::abs(expected[i]), 1.0e-6) > kStagedTolerance)) {
                    fail("device-count packed row differs from the FP64 oracle");
                }
            }
        }
    }

    // Every batched sequence meets the FP64 oracle; its context partition depends on the batch
    // size, so it need not reproduce the one-sequence bytes. 16 sequences of this context exceed
    // 128 groups (five loader waves); a single sequence stays below (six).
    buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
    auto single = leaf_args(buffers, read, rows, workspace_bytes);
    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(single, device.stream));
    device.synchronize();
    std::vector<float> reference(rows * kRowElements);
    buffers.output.copy_to_host(reference.data(), reference.size() * sizeof(float));
    compare(reference, expected, kStagedTolerance, "packed reference");

    for (const std::size_t count : {std::size_t{2U}, std::size_t{16U}}) {
        std::vector<std::unique_ptr<LeafBuffers>> batch;
        std::vector<q27::R9700FullAttentionArgs> calls;
        batch.reserve(count);
        calls.reserve(count);
        const std::size_t stride =
            q27::r9700_full_attention_sequence_workspace_stride_bytes(rows, cache.context);
        for (std::size_t sequence = 0; sequence < count; ++sequence) {
            batch.push_back(std::make_unique<LeafBuffers>(rows, stride));
            batch.back()->query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
            batch.back()->row_positions.copy_from_host(positions.data(),
                                                       rows * sizeof(std::int32_t));
            calls.push_back(leaf_args(*batch.back(), read, rows, stride));
        }
        if (!q27::r9700_full_attention_sequences_supported(calls, device.stream)) {
            fail("packed sequence batch is not supported");
        }
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention_sequences(calls, device.stream));
        device.synchronize();
        for (auto& sequence : batch) {
            std::vector<float> actual(reference.size());
            sequence->output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            compare(actual, expected, kStagedTolerance, "batched packed sequence");
        }
    }
    std::cout << "packed_forms device_counts/sequence_batches six_and_five_loader_waves=PASS\n";
}

// A stale slot's NaN scale must not reach a row whose causal frontier stops before it, on every
// route; the last row's frontier sits mid-block.
void qualify_stale_slots(ninfer::DeviceContext& device) {
    constexpr std::size_t appended = 301U;
    RouteCache cache(device, appended, 71U);
    // Visible frontier past the appended keys: a pending-style read over mapped stale slots.
    for (const std::size_t rows :
         {std::size_t{1U}, std::size_t{4U}, std::size_t{9U}, std::size_t{128U}}) {
        const auto queries = make_queries(rows, 73U);
        const auto bf16    = as_bf16(queries);
        std::vector<std::int32_t> positions(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            positions[r] = static_cast<std::int32_t>(appended - rows + r);
        }
        const std::size_t workspace_bytes = q27::r9700_full_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(rows), appended, false);
        LeafBuffers buffers(rows, workspace_bytes);
        buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
        buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
        const auto read =
            cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(
            leaf_args(buffers, read, rows, workspace_bytes), device.stream));
        device.synchronize();
        std::vector<float> actual(rows * kRowElements);
        buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        compare(actual,
                oracle_output(queries, rows, positions, cache.encoded_keys, cache.encoded_values),
                kStagedTolerance, "stale slots rows " + std::to_string(rows));
    }
    std::cout << "stale_slots NaN_scales_past_frontier=PASS\n";
}

// FP16 Q staging: |Q| < 2^16 stays finite and accurate on staged routes; a row reaching 2^16
// poisons only itself there, while fused and split-512 keep represented Q and stay exact.
void qualify_query_range(ninfer::DeviceContext& device) {
    RouteCache cache(device, 300U, 81U);
    constexpr std::size_t rows = 2U;
    auto queries               = make_queries(rows, 83U);
    for (std::size_t f = 0; f < kHeadDim; ++f) { queries[f] = represented_bf16(1.0e-6F); }
    const std::vector<std::int32_t> positions{298, 299};
    const std::size_t workspace_bytes =
        q27::r9700_full_attention_workspace_capacity_bytes(rows, cache.context, false);
    LeafBuffers buffers(rows, workspace_bytes);
    buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
    const auto read =
        cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
    auto bf16 = as_bf16(queries);
    buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
    const auto args = leaf_args(buffers, read, rows, workspace_bytes);
    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
    device.synchronize();
    std::vector<float> actual(rows * kRowElements);
    buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
    compare(actual,
            oracle_output(queries, rows, positions, cache.encoded_keys, cache.encoded_values),
            kStagedTolerance, "small |Q| staged in FP16");

    queries[kRowElements + 40U] = 65536.0F; // row 1, head 0
    bf16                        = as_bf16(queries);
    buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
    device.synchronize();
    buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
    const auto expected =
        oracle_output(queries, rows, positions, cache.encoded_keys, cache.encoded_values);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const bool poisoned_pair = i >= kRowElements && i < kRowElements + kHeadDim;
        if (poisoned_pair) {
            if (!std::isnan(actual[i])) { fail("|Q| >= 2^16 did not poison its query head"); }
        } else if (std::isnan(actual[i]) ||
                   (std::abs(actual[i] - expected[i]) > kStagedTolerance &&
                    std::abs(actual[i] - expected[i]) / std::max(std::abs(expected[i]), 1.0e-6) >
                        kStagedTolerance)) {
            fail("|Q| >= 2^16 disturbed another query head");
        }
    }
    std::cout << "query_range small_q/overflow_poison=PASS\n";
}

// Invalid table rows and physical pages poison exactly the dependent rows on every raw route.
void qualify_poison_and_rejection(ninfer::DeviceContext& device) {
    RouteCache cache(device, 300U, 91U);
    const auto view             = cache.fixture.cache->layer_view(0, cache.allocation);
    constexpr std::size_t pages = 6U;
    std::vector<std::uint32_t> table(pages);
    const auto ids = cache.allocation.page_ids();
    for (std::size_t i = 0; i < pages; ++i) { table[i] = static_cast<std::uint32_t>(ids[i]); }
    table[2] = 0xfffffffeU; // keys 128..191
    ninfer::DeviceBuffer d_table(pages * sizeof(std::uint32_t));
    d_table.copy_from_host(table.data(), d_table.size());
    const auto run = [&](std::size_t rows, std::size_t row_context, const char* label) {
        const auto queries = make_queries(rows, 93U);
        const auto bf16    = as_bf16(queries);
        std::vector<std::int32_t> positions(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            positions[r] = static_cast<std::int32_t>(row_context - rows + r);
        }
        LeafBuffers buffers(rows, 1U << 20U);
        buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
        buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
        kv::Fp8Int4KvAttentionArgs raw{
            .query            = static_cast<const hip_bfloat16*>(buffers.query.data()),
            .key_codes        = static_cast<const std::uint8_t*>(view.key_pages.data),
            .value_codes      = static_cast<const std::uint8_t*>(view.value_pages.data),
            .page_table       = static_cast<const std::uint32_t*>(d_table.data()),
            .page_table_pages = static_cast<std::uint32_t>(pages),
            .context          = cache.context,
            .physical_tokens  = static_cast<std::size_t>(cache.fixture.physical_pages) * kPageSize,
            .query_heads      = 24U,
            .kv_heads         = 4U,
            .q_to_kv          = 6U,
            .value_group      = 16U,
            .query_rows       = static_cast<std::uint32_t>(rows),
            .row_positions    = static_cast<const std::int32_t*>(buffers.row_positions.data()),
            .attention_scale  = static_cast<float>(kAttentionScale),
            .key_layout       = kv::Fp8Int4KvPlaneLayout::TokenFastest,
            .value_layout     = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .score_workspace    = static_cast<float*>(buffers.workspace.data()),
            .output             = static_cast<float*>(buffers.output.data()),
            .codec              = kv::KvCodec::Nvfp4,
            .key_scale_codes    = static_cast<const std::uint8_t*>(view.key_scale_pages.data),
            .value_scale_codes  = static_cast<const std::uint8_t*>(view.value_scale_pages.data),
        };
        // Rejections before launch.
        auto bad            = raw;
        bad.key_scale_codes = nullptr;
        if (kv::fp8_int4_kv_attention_packed_decode(bad, device.stream) != hipErrorInvalidValue) {
            fail("packed route accepted a missing K-scale plane");
        }
        bad                   = raw;
        bad.value_scale_codes = nullptr;
        if (kv::fp8_int4_kv_attention_fused(bad, device.stream) != hipErrorInvalidValue) {
            fail("fused route accepted a missing V-scale plane");
        }
        bad             = raw;
        bad.value_group = 32U;
        if (kv::fp8_int4_kv_attention_fused(bad, device.stream) != hipErrorInvalidValue) {
            fail("fused route accepted NVFP4 G32");
        }
        bad            = raw;
        bad.key_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        if (kv::fp8_int4_kv_attention_fused(bad, device.stream) != hipErrorInvalidValue) {
            fail("fused route accepted another NVFP4 plane order");
        }
        if (kv::fp8_int4_kv_attention_wmma(raw, device.stream) != hipErrorInvalidValue) {
            fail("raw-WMMA tail accepted NVFP4 planes");
        }
        const hipError_t status =
            rows <= 8U ? kv::fp8_int4_kv_attention_packed_decode(raw, device.stream)
                       : (kv::use_mid_rows_attention(static_cast<std::uint32_t>(rows),
                                                     cache.context, false)
                              ? kv::fp8_int4_kv_attention_mid_rows(raw, device.stream)
                              : hipErrorInvalidValue);
        HIP_CHECK(status);
        device.synchronize();
        std::vector<float> actual(rows * kRowElements);
        buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        for (std::size_t row = 0; row < rows; ++row) {
            const bool sees_fault = static_cast<std::size_t>(positions[row]) >= 128U;
            for (std::size_t i = 0; i < kRowElements; ++i) {
                if (std::isnan(actual[row * kRowElements + i]) != sees_fault) {
                    fail(std::string(label) + ": page fault poisoned the wrong row");
                }
            }
        }
    };
    run(4U, 130U, "packed straddling fault");
    run(16U, 130U, "mid-row straddling fault");
    std::cout << "poison page_fault_rows/raw_rejection=PASS\n";
}

// Pending read through a device-selected pool table row: graph replay is bit-identical to eager and
// follows a changed mapping at fixed addresses; the leaf reads the row selected at replay time.
void qualify_graph(ninfer::DeviceContext& device) {
    constexpr std::size_t context = 800U;
    constexpr std::size_t rows    = 5U;
    const std::uint32_t pages     = 14U;
    Fixture fixture(device, pages * 2U, pages, 2);
    auto a = fixture.cache->pool().reserve(pages);
    auto b = fixture.cache->pool().reserve(pages);
    a.materialize_pages(pages, device.stream);
    b.materialize_pages(pages, device.stream);
    a.bind_row(0, device.stream);
    b.bind_row(1, device.stream);
    const auto keys_a = make_cache_plane(context, 101U), values_a = make_cache_plane(context, 102U);
    const auto keys_b = make_cache_plane(context, 103U), values_b = make_cache_plane(context, 104U);
    q3::PagedKVPublication publication_a, publication_b;
    fixture.append_all(a, publication_a, keys_a, values_a);
    fixture.append_all(b, publication_b, keys_b, values_b);

    const auto queries = make_queries(rows, 105U);
    const auto bf16    = as_bf16(queries);
    std::vector<std::int32_t> positions(rows);
    for (std::size_t r = 0; r < rows; ++r) {
        positions[r] = static_cast<std::int32_t>(context - rows + r);
    }
    const auto expected_a = oracle_output(queries, rows, positions,
                                          oracle::encode(keys_a, context * kKvHeads, kHeadDim),
                                          oracle::encode(values_a, context * kKvHeads, kHeadDim));
    const auto expected_b = oracle_output(queries, rows, positions,
                                          oracle::encode(keys_b, context * kKvHeads, kHeadDim),
                                          oracle::encode(values_b, context * kKvHeads, kHeadDim));

    // A zero-token segmented transaction: its pending read carries the device table-row selector.
    ninfer::DeviceBuffer d_row(sizeof(std::int32_t)), d_base(sizeof(std::int32_t));
    ninfer::DeviceBuffer d_cursor(sizeof(std::uint32_t)), d_panel(sizeof(std::int32_t) * 2U);
    ninfer::DeviceBuffer d_count(sizeof(std::int32_t));
    const std::int32_t zero = 0, base = static_cast<std::int32_t>(context);
    d_row.copy_from_host(&zero, sizeof(zero));
    d_base.copy_from_host(&base, sizeof(base));
    d_count.copy_from_host(&zero, sizeof(zero));
    auto workspace   = fixture.workspace();
    workspace.cursor = static_cast<std::uint32_t*>(d_cursor.data());
    auto transaction = fixture.cache->begin_device_segmented_append(
        a, publication_a, static_cast<const std::int32_t*>(d_base.data()),
        static_cast<std::uint32_t>(context), static_cast<std::uint32_t>(context),
        static_cast<std::uint32_t>(context + 2U), workspace,
        static_cast<const std::int32_t*>(d_row.data()));
    const auto k_tail = as_bf16(make_cache_plane(2U, 106U));
    ninfer::DeviceBuffer d_k(k_tail.size() * 2U), d_v(k_tail.size() * 2U);
    d_k.copy_from_host(k_tail.data(), d_k.size());
    d_v.copy_from_host(k_tail.data(), d_v.size());
    transaction.begin_device_segment(static_cast<const std::int32_t*>(d_panel.data()), 2U,
                                     static_cast<const std::int32_t*>(d_count.data()));
    transaction.launch_append_layer(0U, static_cast<const hip_bfloat16*>(d_k.data()),
                                    static_cast<const hip_bfloat16*>(d_v.data()), device.stream);
    const auto read = transaction.pending_layer_read(0U);
    transaction.end_device_segment(device.stream);

    const std::size_t workspace_bytes =
        q27::r9700_full_attention_workspace_capacity_bytes(rows, context + 2U, false);
    LeafBuffers buffers(rows, workspace_bytes);
    buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
    buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
    const auto args        = leaf_args(buffers, read, rows, workspace_bytes);
    const auto read_output = [&] {
        device.synchronize();
        std::vector<float> actual(rows * kRowElements);
        buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        return actual;
    };
    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
    const auto eager = read_output();
    compare(eager, expected_a, kStagedTolerance, "pending eager read");

    ninfer::DecodeGraphDefinition definition;
    definition.capture(device.stream, [&] {
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
    });
    ninfer::DecodeGraphExecutable graph;
    graph.instantiate(definition);
    HIP_CHECK(hipMemsetAsync(buffers.output.data(), 0xff, buffers.output.size(), device.stream));
    graph.launch(device.stream);
    if (read_output() != eager) { fail("graph replay differs from eager"); }

    const std::int32_t other = 1;
    d_row.copy_from_host(&other, sizeof(other));
    HIP_CHECK(hipMemsetAsync(buffers.output.data(), 0xff, buffers.output.size(), device.stream));
    graph.launch(device.stream);
    compare(read_output(), expected_b, kStagedTolerance, "graph replay after a mapping change");

    const std::int32_t invalid = 7;
    d_row.copy_from_host(&invalid, sizeof(invalid));
    graph.launch(device.stream);
    const auto poisoned = read_output();
    if (!std::all_of(poisoned.begin(), poisoned.end(), [](float v) { return std::isnan(v); })) {
        fail("graph replay with an invalid table row did not poison");
    }
    d_row.copy_from_host(&zero, sizeof(zero));
    std::uint32_t status = 1U, cursor = 0U;
    transaction.enqueue_segmented_resolution(&status, &cursor);
    device.synchronize();
    transaction.finish_segmented_resolution(status, cursor, static_cast<std::uint32_t>(context));
    std::cout << "graph pending_read/replay/mapping_change/invalid_row=PASS\n";
}

// Plan-level contract: byte ratio, fingerprints and spill/restore of all four planes.
void qualify_storage(ninfer::DeviceContext& device) {
    const auto spec_of = [](ninfer::KvCacheFormat format) {
        return ninfer::Fp8KInt4VPagedKVSpec{
            .format                = format,
            .page_group_count      = 8U,
            .logical_page_capacity = 8U,
            .table_rows            = 2,
            .layer_count           = 16,
            .head_dim              = 256,
            .num_kv_heads          = 4,
            .value_group           = 16,
            .plane_layouts = {.key   = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale =
                                  ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor}};
    };
    ninfer::LayoutBuilder a, b;
    const auto fp8 =
        ninfer::plan_fp8_k_int4_v_paged_kv_pool(a, spec_of(ninfer::KvCacheFormat::Fp8KInt4V));
    const auto nvfp4 =
        ninfer::plan_fp8_k_int4_v_paged_kv_pool(b, spec_of(ninfer::KvCacheFormat::Nvfp4));
    // 16 layers x 4 heads x 64 tokens x 8 pages x {416, 288} bytes per head-token.
    if (fp8.storage.payload_bytes() != 16U * 4U * 64U * 8U * 416U ||
        nvfp4.storage.payload_bytes() != 16U * 4U * 64U * 8U * 288U) {
        fail("format payload does not match its byte contract");
    }
    const auto fingerprint_a =
        ninfer::fp8_k_int4_v_semantic_fingerprint(spec_of(ninfer::KvCacheFormat::Fp8KInt4V));
    const auto fingerprint_b =
        ninfer::fp8_k_int4_v_semantic_fingerprint(spec_of(ninfer::KvCacheFormat::Nvfp4));
    bool rejected = false;
    try {
        ninfer::require_fp8_k_int4_v_semantic_fingerprint(fingerprint_a, fingerprint_b);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { fail("a format mismatch passed the semantic fingerprint"); }
    auto bad_group        = spec_of(ninfer::KvCacheFormat::Nvfp4);
    bad_group.value_group = 32;
    rejected              = false;
    ninfer::LayoutBuilder c;
    try {
        (void)ninfer::plan_fp8_k_int4_v_paged_kv_pool(c, bad_group);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { fail("NVFP4 G32 was planned"); }

    constexpr std::size_t tokens = 150U;
    const auto keys              = make_cache_plane(tokens, 111U);
    const auto values            = make_cache_plane(tokens, 112U);
    Fixture fixture(device, 4U, 3U, 1);
    auto [allocation, occupied] = fixture.fragmented(3U, 0);
    q3::PagedKVPublication publication;
    fixture.append_all(allocation, publication, keys, values);
    device.synchronize();
    auto& pool = fixture.cache->pool();
    std::vector<std::byte> image(
        ninfer::paged_kv_host_image_bytes(pool, allocation.mapped_page_count()));
    ninfer::pack_paged_kv_allocation_to_host(allocation, pool, image.data(), device.stream);
    device.synchronize();
    pool.zero_pages(allocation.page_ids(), device.stream);
    ninfer::unpack_paged_kv_allocation_from_host(allocation, pool, image.data(),
                                                 allocation.mapped_page_count(),
                                                 allocation.mapped_page_count(), device.stream);
    fixture.require_bytes(allocation, oracle::encode(keys, tokens * kKvHeads, kHeadDim),
                          oracle::encode(values, tokens * kKvHeads, kHeadDim), tokens,
                          "host spill/restore");
    std::cout << "storage payload/fingerprint/g32_rejection/spill_restore=PASS\n";
}

} // namespace

int main() {
    try {
        qualify_oracle_self();
        ninfer::DeviceContext device(0);
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            fail("selected physical device is not wave32 gfx1201");
        }
        qualify_storage(device);
        qualify_codec_bytes(device);
        qualify_status(device);
        qualify_append_forms(device);
        qualify_compaction(device);
        qualify_causal_routes(device);
        qualify_tree_routes(device);
        qualify_packed_forms(device);
        qualify_stale_slots(device);
        qualify_query_range(device);
        qualify_poison_and_rejection(device);
        qualify_graph(device);
        std::cout << "nvfp4_kv_qual PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nvfp4_kv_qual FAILED: " << error.what() << '\n';
        return 1;
    }
}
