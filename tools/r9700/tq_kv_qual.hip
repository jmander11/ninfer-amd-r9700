// Qualification of the opt-in TQ-K4-V3 growing cache through its real typed-cache, transaction and
// full-attention boundaries, against the independent FP64 oracle in tests/ops/tq_kv_oracle.h.
//
// Covered: the rotation Op (forward BF16 Q, inverse FP32 output, signed-zero and NaN handling);
// stored codes against an independent FP64 encoding (a code may differ only where the FP64
// level-domain magnitude lies within 1e-4 of a threshold, or a coordinate is within 1e-6 of zero
// for its sign) and every stored scale equal to FP16_rn(||x|| / ||c||) of the stored codes within
// one ulp, over fragmented pages and mid-page chunked appends; the zero vector; nonfinite status;
// batched append; compaction and rollback as exact byte copies; and the fused, packed (with
// device-selected rows and sequence batches), dense, mid-row and split-512 routes, including causal
// frontiers and packed-tree masks. Each route's input is the represented rotated BF16 Q that the
// leaf's rotation Op produced (that Op is separately checked against the FP64 rotation); the
// oracle computes FP64 attention over the decoded stored rotated K/V and rotates the result back
// in FP64, so route criteria cover only route arithmetic (FP16 staging on staged routes).

#include "core/arena.h"
#include "core/device.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "ops/r9700/kv/tq_rotation.h"
#include "ops/tq_kv_oracle.h"
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
namespace oracle = ninfer::test::tq_kv_oracle;

constexpr std::size_t kHeadDim      = 256;
constexpr std::size_t kQueryHeads   = 24;
constexpr std::size_t kKvHeads      = 4;
constexpr std::size_t kQueriesPerKv = 6;
constexpr std::size_t kPageSize     = 64;
constexpr std::size_t kRowElements  = kQueryHeads * kHeadDim;
constexpr std::size_t kTokenItems   = kKvHeads * kHeadDim;
constexpr double kAttentionScale    = 1.0 / 16.0;
// Absolute-or-relative criteria over represented rotated Q. This fixture's decoded V reaches about
// 10; staged routes' FP16 K/V/8 products and probabilities keep their error near 1e-3 of that peak,
// and the FP32 fused and split-512 routes near NVFP4's 2e-4.
constexpr double kStagedTolerance = 1.1e-2;
constexpr double kFusedTolerance  = 5.0e-4;

// Split-512 stages its QK keys as BF16 for BF16 WMMA. NVFP4 level*scale is exact there, but a TQ
// level times its binary16 scale is rounded once to BF16, an explicit staging cast of that route;
// its oracle therefore rounds each decoded rotated key to BF16 and keeps the fused criterion.

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

// Feature groups span seven octaves (outlier-like channels); every vector is nonzero.
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

// Independent physical layout: K nibbles and K scales token-fastest head-major; V bytes and V
// scales feature-fastest page-major rows.
std::size_t key_offset(std::size_t physical, std::size_t head, std::size_t byte, std::size_t pages,
                       std::size_t leading) {
    return physical % kPageSize +
           kPageSize * (byte + leading * (physical / kPageSize + pages * head));
}

std::size_t value_row(std::size_t physical, std::size_t head) {
    return physical % kPageSize + kPageSize * (head + kKvHeads * (physical / kPageSize));
}

// Decoded stored vectors (model basis, and rotated basis scale * c), [token * kKvHeads + head].
struct Decoded {
    std::vector<oracle::Vector> keys, values, rotated_keys, rotated_values, rotated_keys_bf16;
};

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
          spec{.format                = ninfer::KvCacheFormat::TqK4V3,
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
        backing = std::make_unique<ninfer::DeviceBuffer>(builder.finish(256, "TQ qualification"));
        cache   = std::make_unique<q3::PagedKVCache>(
            ninfer::DeviceSpan{backing->data(), backing->size()}, layout);
        // Stale device memory: all-ones codes and NaN scales in every unwritten slot.
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

    struct Planes {
        std::vector<std::uint8_t> keys, values;
        std::vector<std::uint16_t> key_scales, value_scales;
    };

    [[nodiscard]] Planes planes(const ninfer::PagedKVAllocation& allocation) {
        const auto view = cache->layer_view(0, allocation);
        Planes result{read_plane(view.key_pages), read_plane(view.value_pages), {}, {}};
        const auto ks = read_plane(view.key_scale_pages);
        const auto vs = read_plane(view.value_scale_pages);
        result.key_scales.resize(ks.size() / 2U);
        result.value_scales.resize(vs.size() / 2U);
        std::memcpy(result.key_scales.data(), ks.data(), ks.size());
        std::memcpy(result.value_scales.data(), vs.data(), vs.size());
        return result;
    }

    // One logical token-head's stored K bytes, V bytes and scales.
    struct Stored {
        std::array<std::uint8_t, 128> key{};
        std::array<std::uint8_t, 96> value{};
        std::uint16_t key_scale = 0U, value_scale = 0U;
    };

    [[nodiscard]] Stored stored(const Planes& p, const ninfer::PagedKVAllocation& allocation,
                                std::size_t token, std::size_t head) const {
        const auto pages = allocation.page_ids();
        const std::size_t physical =
            static_cast<std::size_t>(pages[token / kPageSize]) * kPageSize + token % kPageSize;
        Stored s;
        for (std::size_t byte = 0; byte < 128U; ++byte) {
            s.key[byte] = p.keys[key_offset(physical, head, byte, physical_pages, 128U)];
        }
        const std::size_t row = value_row(physical, head);
        for (std::size_t byte = 0; byte < 96U; ++byte) s.value[byte] = p.values[row * 96U + byte];
        s.key_scale   = p.key_scales[key_offset(physical, head, 0U, physical_pages, 1U)];
        s.value_scale = p.value_scales[row];
        return s;
    }

    // Checks every stored code and scale against the independent encoding and returns the
    // decoded model-basis vectors.
    [[nodiscard]] Decoded require_codes(const ninfer::PagedKVAllocation& allocation,
                                        std::span<const float> keys, std::span<const float> values,
                                        std::size_t tokens, const std::string& label,
                                        std::size_t* boundary_codes = nullptr) {
        const Planes p = planes(allocation);
        Decoded decoded;
        decoded.keys.resize(tokens * kKvHeads);
        decoded.values.resize(tokens * kKvHeads);
        decoded.rotated_keys.resize(tokens * kKvHeads);
        decoded.rotated_values.resize(tokens * kKvHeads);
        decoded.rotated_keys_bf16.resize(tokens * kKvHeads);
        const auto check = [&](std::span<const float> source, std::span<const std::uint8_t> bytes,
                               std::uint16_t scale, unsigned bits, std::size_t t) {
            const auto expected = oracle::encode(source, bits);
            std::array<std::uint8_t, kHeadDim> index{};
            std::array<bool, kHeadDim> negative{};
            if (bits == 4U) {
                oracle::key_codes(bytes, index, negative);
            } else {
                oracle::value_codes(bytes, index, negative);
            }
            const std::span<const float> thresholds =
                bits == 4U ? std::span<const float>(oracle::tq::kKeyThresholds)
                           : std::span<const float>(oracle::tq::kValueThresholds);
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                if (index[i] != expected.index[i]) {
                    const std::size_t low = std::min(index[i], expected.index[i]);
                    const bool adjacent =
                        std::max(index[i], expected.index[i]) == low + 1U &&
                        std::abs(expected.r[i] - thresholds[low]) <= 1.0e-4 * thresholds[low];
                    if (!adjacent) {
                        fail(label + ": " + (bits == 4U ? "K" : "V") + " code differs at token " +
                             std::to_string(t) + " feature " + std::to_string(i));
                    }
                    if (boundary_codes != nullptr) ++*boundary_codes;
                }
                if (negative[i] != expected.negative[i] && expected.r[i] > 1.0e-6) {
                    fail(label + ": sign differs at token " + std::to_string(t));
                }
            }
            const std::uint16_t wanted = oracle::scale_for_codes(source, index, bits);
            if (std::abs(static_cast<int>(wanted) - static_cast<int>(scale)) > 1) {
                fail(label + ": " + (bits == 4U ? "K" : "V") + " scale differs at token " +
                     std::to_string(t) + " stored " + std::to_string(scale) + " expected " +
                     std::to_string(wanted));
            }
        };
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t h = 0; h < kKvHeads; ++h) {
                const std::size_t item = t * kKvHeads + h;
                const Stored s         = stored(p, allocation, t, h);
                check(keys.subspan(item * kHeadDim, kHeadDim), s.key, s.key_scale, 4U, t);
                check(values.subspan(item * kHeadDim, kHeadDim), s.value, s.value_scale, 3U, t);
                decoded.keys[item]           = oracle::decode_key(s.key, s.key_scale);
                decoded.values[item]         = oracle::decode_value(s.value, s.value_scale);
                decoded.rotated_keys[item]   = oracle::decode_key_rotated(s.key, s.key_scale);
                decoded.rotated_values[item] = oracle::decode_value_rotated(s.value, s.value_scale);
                for (std::size_t i = 0; i < kHeadDim; ++i) {
                    decoded.rotated_keys_bf16[item][i] =
                        represented_bf16(static_cast<float>(decoded.rotated_keys[item][i]));
                }
            }
        }
        return decoded;
    }
};

// FP64 attention of represented rotated Q over the decoded rotated cache, rotated back.
std::vector<double> oracle_output(std::span<const float> rotated_queries, std::size_t rows,
                                  std::span<const std::int32_t> row_positions,
                                  const Decoded& decoded,
                                  std::span<const std::int32_t> prefixes = {},
                                  std::span<const std::int32_t> masks    = {},
                                  bool bf16_keys                         = false) {
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
            const auto rotated     = oracle::attention_fp64(
                rotated_queries.subspan(base, kHeadDim), selected,
                bf16_keys ? decoded.rotated_keys_bf16 : decoded.rotated_keys,
                decoded.rotated_values, kKvHeads, head / kQueriesPerKv, kAttentionScale);
            const auto result = oracle::unrotate(rotated);
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

// ------------------------------------------------------------------------------------ rotation
void qualify_rotation(ninfer::DeviceContext& device) {
    constexpr std::size_t vectors = 6U;
    std::vector<float> source(vectors * kHeadDim);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = represented_bf16(
            (static_cast<float>(mix(static_cast<std::uint32_t>(i), 5U, 9U) % 2001U) - 1000.0F) /
            97.0F);
    }
    std::fill_n(source.begin() + 2U * kHeadDim, kHeadDim, 0.0F); // a zero vector
    const auto bf16 = as_bf16(source);
    ninfer::DeviceBuffer query(bf16.size() * sizeof(hip_bfloat16));
    query.copy_from_host(bf16.data(), query.size());
    HIP_CHECK(
        kv::tq_rotate_query(static_cast<hip_bfloat16*>(query.data()), vectors, device.stream));
    device.synchronize();
    std::vector<hip_bfloat16> rotated(bf16.size());
    query.copy_to_host(rotated.data(), query.size());
    for (std::size_t v = 0; v < vectors; ++v) {
        std::array<double, kHeadDim> x{};
        for (std::size_t i = 0; i < kHeadDim; ++i) x[i] = source[v * kHeadDim + i];
        const auto y = oracle::rotate(x);
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            const double actual = static_cast<float>(rotated[v * kHeadDim + i]);
            if (std::abs(actual - y[i]) > std::abs(y[i]) * 0x1p-8 + 1.0e-4) {
                fail("rotated Q differs from the FP64 rotation");
            }
        }
    }
    // Inverse: FP32 in, FP32 out; +0 and -0 vectors become +0; NaN stays in its vector.
    std::vector<float> output(vectors * kHeadDim);
    for (std::size_t i = 0; i < output.size(); ++i) output[i] = source[i] * 0.37F;
    std::fill_n(output.begin() + 3U * kHeadDim, kHeadDim, -0.0F);
    output[4U * kHeadDim + 7U] = std::numeric_limits<float>::quiet_NaN();
    ninfer::DeviceBuffer d_output(output.size() * sizeof(float));
    d_output.copy_from_host(output.data(), d_output.size());
    HIP_CHECK(kv::tq_unrotate_output(static_cast<float*>(d_output.data()), vectors, device.stream));
    device.synchronize();
    std::vector<float> back(output.size());
    d_output.copy_to_host(back.data(), d_output.size());
    for (std::size_t v = 0; v < vectors; ++v) {
        std::array<double, kHeadDim> y{};
        for (std::size_t i = 0; i < kHeadDim; ++i) y[i] = output[v * kHeadDim + i];
        const auto x = oracle::unrotate(y);
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            const float actual = back[v * kHeadDim + i];
            if (v == 4U) {
                if (!std::isnan(actual)) fail("NaN did not poison its own vector");
                continue;
            }
            if ((v == 2U || v == 3U) && std::bit_cast<std::uint32_t>(actual) != 0U) {
                fail("a zero vector did not unrotate to +0");
            }
            if (std::abs(actual - x[i]) > 2.0e-5 * std::max(1.0, std::abs(x[i]))) {
                fail("unrotated output differs from the FP64 inverse");
            }
        }
    }
    if (!std::isfinite(back[5U * kHeadDim])) fail("NaN leaked into a neighbouring vector");
    // Batched form: three buffers of different lengths in one launch equal their single launches.
    std::array<ninfer::DeviceBuffer, 3> buffers{ninfer::DeviceBuffer(vectors * kHeadDim * 4U),
                                                ninfer::DeviceBuffer(vectors * kHeadDim * 4U),
                                                ninfer::DeviceBuffer(vectors * kHeadDim * 4U)};
    const std::array<std::size_t, 3> lengths{1U, 6U, 3U};
    std::array<float*, 3> pointers{};
    for (std::size_t b = 0; b < 3U; ++b) {
        buffers[b].copy_from_host(output.data(), vectors * kHeadDim * sizeof(float));
        pointers[b] = static_cast<float*>(buffers[b].data());
    }
    HIP_CHECK(kv::tq_unrotate_output(std::span<float* const>(pointers), lengths, device.stream));
    device.synchronize();
    for (std::size_t b = 0; b < 3U; ++b) {
        std::vector<float> batched(vectors * kHeadDim);
        buffers[b].copy_to_host(batched.data(), batched.size() * sizeof(float));
        for (std::size_t i = 0; i < batched.size(); ++i) {
            const float expected = i < lengths[b] * kHeadDim ? back[i] : output[i];
            if (std::bit_cast<std::uint32_t>(batched[i]) !=
                    std::bit_cast<std::uint32_t>(expected) &&
                !(std::isnan(batched[i]) && std::isnan(expected))) {
                fail("batched unrotation differs from its single launch");
            }
        }
    }
    // Batched forward at the full batch (kv::kTqRotationMaximumBatch buffers, lengths 1..6):
    // each equals the single launch over its prefix and leaves its tail untouched.
    constexpr std::size_t batch = kv::kTqRotationMaximumBatch;
    std::vector<ninfer::DeviceBuffer> query_buffers;
    query_buffers.reserve(batch);
    std::array<hip_bfloat16*, batch> query_pointers{};
    std::array<std::size_t, batch> query_lengths{};
    for (std::size_t b = 0; b < batch; ++b) {
        query_buffers.emplace_back(bf16.size() * sizeof(hip_bfloat16));
        query_buffers[b].copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
        query_pointers[b] = static_cast<hip_bfloat16*>(query_buffers[b].data());
        query_lengths[b]  = 1U + (b * 5U) % vectors;
    }
    HIP_CHECK(kv::tq_rotate_query(std::span<hip_bfloat16* const>(query_pointers), query_lengths,
                                  device.stream));
    device.synchronize();
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<hip_bfloat16> batched(bf16.size());
        query_buffers[b].copy_to_host(batched.data(), batched.size() * sizeof(hip_bfloat16));
        for (std::size_t i = 0; i < batched.size(); ++i) {
            const hip_bfloat16 expected = i < query_lengths[b] * kHeadDim ? rotated[i] : bf16[i];
            if (batched[i].data != expected.data)
                fail("batched rotation differs from its single launch");
        }
    }
    std::cout << "rotation forward/inverse/signed_zero/nan/batched=PASS\n";
}

// ------------------------------------------------------------------------------------ codec
void qualify_codec(ninfer::DeviceContext& device) {
    constexpr std::size_t tokens = 203U;
    auto keys                    = make_cache_plane(tokens, 1U);
    auto values                  = make_cache_plane(tokens, 2U);
    // Token 5 head 1: a zero K and V vector; token 6: one dominant outlier feature.
    std::fill_n(keys.begin() + (5U * kKvHeads + 1U) * kHeadDim, kHeadDim, 0.0F);
    std::fill_n(values.begin() + (5U * kKvHeads + 1U) * kHeadDim, kHeadDim, 0.0F);
    keys[(6U * kKvHeads) * kHeadDim + 17U]   = 120.0F;
    values[(6U * kKvHeads) * kHeadDim + 33U] = -96.0F;
    Fixture fixture(device, 6U, 4U, 1);
    auto [allocation, occupied] = fixture.fragmented(4U, 0);
    q3::PagedKVPublication publication;
    fixture.append_all(allocation, publication, keys, values);
    std::size_t boundary = 0U;
    const Decoded decoded =
        fixture.require_codes(allocation, keys, values, tokens, "codec", &boundary);
    const auto zero = fixture.stored(fixture.planes(allocation), allocation, 5U, 1U);
    if (zero.key_scale != 0U || zero.value_scale != 0U ||
        std::any_of(zero.key.begin(), zero.key.end(), [](std::uint8_t b) { return b != 0U; }) ||
        std::any_of(zero.value.begin(), zero.value.end(), [](std::uint8_t b) { return b != 0U; })) {
        fail("zero vector did not store zero codes and scales");
    }
    // Represented error: the decoded vectors are near their sources (MSE ratio below 1.5x the
    // Gaussian design point), and the norm correction preserves the norm within FP16.
    double key_error = 0.0, key_energy = 0.0, value_error = 0.0, value_energy = 0.0;
    for (std::size_t item = 0; item < tokens * kKvHeads; ++item) {
        double norm2 = 0.0, decoded2 = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            const double k = keys[item * kHeadDim + i], v = values[item * kHeadDim + i];
            key_error += (decoded.keys[item][i] - k) * (decoded.keys[item][i] - k);
            value_error += (decoded.values[item][i] - v) * (decoded.values[item][i] - v);
            key_energy += k * k;
            value_energy += v * v;
            norm2 += k * k;
            decoded2 += decoded.keys[item][i] * decoded.keys[item][i];
        }
        if (norm2 > 0.0 && std::abs(std::sqrt(decoded2 / norm2) - 1.0) > 1.0e-3) {
            fail("norm correction did not preserve a K norm");
        }
    }
    const double key_mse = key_error / key_energy, value_mse = value_error / value_energy;
    if (!(key_mse < 0.015) || !(value_mse < 0.05)) {
        fail("represented error exceeds the codec design point");
    }
    std::cout << "codec codes/scales/zero/norm K_rel_mse=" << key_mse << " V_rel_mse=" << value_mse
              << " threshold_ties=" << boundary << " PASS\n";
}

void qualify_status(ninfer::DeviceContext& device) {
    const auto run = [&](const char* label, float bad_key, float bad_value) {
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
        if (status != kv::Fp8Int4KvAppendNonfiniteInput || !rejected || publication.healthy ||
            publication.valid_frontier != 0U) {
            fail(std::string(label) + " did not reject its transaction, status " +
                 std::to_string(status));
        }
    };
    run("nonfinite K", std::numeric_limits<float>::infinity(), 0.5F);
    run("NaN V", 0.5F, std::numeric_limits<float>::quiet_NaN());
    std::cout << "status nonfinite=PASS\n";
}

// Batched append of two sequences stores the same codes as their one-sequence encodings.
void qualify_batched_append(ninfer::DeviceContext& device) {
    constexpr std::size_t tokens = 70U;
    const auto keys_a = make_cache_plane(tokens, 7U), values_a = make_cache_plane(tokens, 8U);
    const auto keys_b = make_cache_plane(tokens, 9U), values_b = make_cache_plane(tokens, 10U);
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
    auto transaction_b =
        fixture.cache->begin_append(b, publication_b, logical,
                                    {.positions = static_cast<std::uint32_t*>(positions_b.data()),
                                     .position_capacity = tokens,
                                     .status = static_cast<std::uint32_t*>(status_b.data())});
    std::array<q3::PagedKVTransaction*, 2> batch{&transaction_a, &transaction_b};
    std::array<const hip_bfloat16*, 2> batch_k{static_cast<const hip_bfloat16*>(dk_a.data()),
                                               static_cast<const hip_bfloat16*>(dk_b.data())};
    std::array<const hip_bfloat16*, 2> batch_v{static_cast<const hip_bfloat16*>(dv_a.data()),
                                               static_cast<const hip_bfloat16*>(dv_b.data())};
    q3::PagedKVTransaction::launch_append_layers(batch, 0U, batch_k, batch_v, device.stream);
    if (transaction_a.commit() != tokens || transaction_b.commit() != tokens) {
        fail("batched append published a wrong frontier");
    }
    (void)fixture.require_codes(a, keys_a, values_a, tokens, "batched sequence 0");
    (void)fixture.require_codes(b, keys_b, values_b, tokens, "batched sequence 1");
    std::cout << "batched_append=PASS\n";
}

// Compaction copies every plane byte exactly; rollback keeps the prefix bytes; resume overwrites.
void qualify_compaction(ninfer::DeviceContext& device) {
    constexpr std::size_t tokens = 150U;
    const auto keys              = make_cache_plane(tokens, 12U);
    const auto values            = make_cache_plane(tokens, 13U);
    Fixture fixture(device, 4U, 3U, 1);
    auto [allocation, occupied] = fixture.fragmented(3U, 0);
    q3::PagedKVPublication publication;
    fixture.append_all(allocation, publication, keys, values);
    const auto before = fixture.planes(allocation);
    std::vector<Fixture::Stored> original;
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            original.push_back(fixture.stored(before, allocation, t, h));
        }
    }
    constexpr std::uint32_t prefix = 60U;
    const std::vector<std::uint32_t> path{0U, 2U, 5U, 9U, 10U, 40U, 41U, 70U, 89U};
    auto transaction = fixture.cache->begin_compact(allocation, publication, fixture.workspace());
    transaction.launch_compact_layer(0U, prefix, path, device.stream);
    if (transaction.commit() != prefix + path.size()) fail("compaction published a wrong frontier");
    const auto after = fixture.planes(allocation);
    const auto same  = [](const Fixture::Stored& x, const Fixture::Stored& y) {
        return x.key == y.key && x.value == y.value && x.key_scale == y.key_scale &&
               x.value_scale == y.value_scale;
    };
    for (std::size_t i = 0; i < prefix + path.size(); ++i) {
        const std::size_t source = i < prefix ? i : prefix + path[i - prefix];
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            if (!same(fixture.stored(after, allocation, i, h), original[source * kKvHeads + h])) {
                fail("compaction did not copy token " + std::to_string(i) + " exactly");
            }
        }
    }
    fixture.cache->truncate_publication(allocation, publication, prefix);
    if (publication.valid_frontier != prefix) fail("rollback did not retain its prefix");
    std::vector<float> resumed_k = reorder_for_tokens(keys, 0U, prefix + path.size());
    std::vector<float> resumed_v = reorder_for_tokens(values, 0U, prefix + path.size());
    const auto extra_k           = make_cache_plane(prefix + path.size(), 14U);
    const auto extra_v           = make_cache_plane(prefix + path.size(), 15U);
    std::copy(extra_k.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems), extra_k.end(),
              resumed_k.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems));
    std::copy(extra_v.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems), extra_v.end(),
              resumed_v.begin() + static_cast<std::ptrdiff_t>(prefix * kTokenItems));
    fixture.append(allocation, publication, resumed_k, resumed_v, prefix, path.size());
    (void)fixture.require_codes(allocation, resumed_k, resumed_v, prefix + path.size(),
                                "append after rollback");
    std::cout << "compaction copy/rollback/resume=PASS\n";
}

// ------------------------------------------------------------------------------ attention routes
struct RouteCache {
    std::size_t context;
    std::vector<float> keys, values;
    Fixture fixture;
    ninfer::PagedKVAllocation allocation, occupied;
    q3::PagedKVPublication publication;
    Decoded decoded;

    RouteCache(ninfer::DeviceContext& device, std::size_t tokens, std::uint32_t seed)
        : context(tokens), keys(make_cache_plane(tokens, seed)),
          values(make_cache_plane(tokens, seed + 1U)),
          fixture(device, static_cast<std::uint32_t>((tokens + 63U) / 64U + 2U),
                  static_cast<std::uint32_t>((tokens + 63U) / 64U + 1U), 2) {
        auto allocated =
            fixture.fragmented(static_cast<std::uint32_t>((tokens + 63U) / 64U + 1U), 0);
        allocation = std::move(allocated.first);
        occupied   = std::move(allocated.second);
        fixture.append_all(allocation, publication, keys, values);
        decoded = fixture.require_codes(allocation, keys, values, tokens,
                                        "route cache " + std::to_string(tokens));
    }
};

// The represented rotated Q the leaf left in place, as FP32.
std::vector<float> rotated_query(LeafBuffers& buffers, std::size_t rows) {
    std::vector<hip_bfloat16> bits(rows * kRowElements);
    buffers.query.copy_to_host(bits.data(), bits.size() * sizeof(hip_bfloat16));
    std::vector<float> result(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) result[i] = static_cast<float>(bits[i]);
    return result;
}

struct LeafRun {
    std::vector<float> output, rotated;
};

LeafRun run_leaf(ninfer::DeviceContext& device, LeafBuffers& buffers,
                 const q27::R9700FullAttentionArgs& args, std::size_t rows,
                 std::span<const hip_bfloat16> query) {
    // The leaf rotates Q in place, so every call restores the represented model-basis Q.
    buffers.query.copy_from_host(query.data(), query.size() * sizeof(hip_bfloat16));
    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
    device.synchronize();
    LeafRun run{std::vector<float>(rows * kRowElements), rotated_query(buffers, rows)};
    buffers.output.copy_to_host(run.output.data(), run.output.size() * sizeof(float));
    return run;
}

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
        buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
        const auto read =
            cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
        const auto run =
            run_leaf(device, buffers, leaf_args(buffers, read, rows, workspace_bytes), rows, bf16);
        compare(run.output, oracle_output(run.rotated, rows, positions, cache.decoded), tolerance,
                "context " + std::to_string(cache.context) + " rows " + std::to_string(rows),
                &maximum);
    };
    {
        RouteCache cache(device, 33U, 21U); // below the packed minimum: fused, then dense
        for (const std::size_t rows : {1U, 3U, 8U, 33U}) {
            if (kv::use_packed_decode_attention(static_cast<std::uint32_t>(rows), cache.context,
                                                false)) {
                fail("context 33 unexpectedly selects packed decode");
            }
            cell(cache, rows, rows >= 9U ? kStagedTolerance : kFusedTolerance,
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
        for (const std::size_t rows : {1U, 6U, 8U}) cell(cache, rows, kStagedTolerance, packed_max);
        for (const std::size_t rows : {9U, 128U}) cell(cache, rows, kStagedTolerance, dense_max);
    }
    {
        RouteCache cache(device, 9000U, 27U);
        for (const std::size_t rows : {1U, 5U, 8U}) cell(cache, rows, kStagedTolerance, packed_max);
        if (!kv::use_mid_rows_attention(48U, cache.context, false)) {
            fail("context 9000 rows 48 does not select the mid-row route");
        }
        cell(cache, 48U, kStagedTolerance, dense_max);
    }
    std::cout << "causal_routes packed_max_abs=" << packed_max << " dense_mid_max_abs=" << dense_max
              << " fused_max_abs=" << fused_max << " PASS\n";
}

// Fused tree at short context, split-512 tree at long context, rowwise and broadcast prefixes, and
// device-selected rows whose inactive metadata is malformed.
void qualify_tree_routes(ninfer::DeviceContext& device) {
    double maximum = 0.0;
    for (const std::size_t context : {std::size_t{257U}, std::size_t{8200U}}) {
        RouteCache cache(device, context, static_cast<std::uint32_t>(context));
        constexpr std::size_t rows = 4U;
        if (kv::use_split512_attention(rows, context, true) != (context >= 8192U)) {
            fail("tree route classifier changed");
        }
        const auto queries  = make_queries(rows, 41U);
        const auto bf16     = as_bf16(queries);
        const auto first    = static_cast<std::int32_t>(context) - 4;
        const bool split512 = context >= 8192U;
        std::vector<std::int32_t> positions{first, first + 1, first + 2, first + 3};
        std::vector<std::int32_t> prefixes{first, first, first, first};
        std::vector<std::int32_t> masks{1, 3, 5, 13};
        const std::size_t workspace_bytes =
            q27::r9700_full_attention_workspace_capacity_bytes(rows, context, true);
        LeafBuffers buffers(rows, workspace_bytes);
        ninfer::DeviceBuffer d_masks(rows * sizeof(std::int32_t)),
            d_prefix(rows * sizeof(std::int32_t)), d_active(sizeof(std::int32_t));
        buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
        d_masks.copy_from_host(masks.data(), d_masks.size());
        d_prefix.copy_from_host(prefixes.data(), d_prefix.size());
        const auto read =
            cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
        auto args           = leaf_args(buffers, read, rows, workspace_bytes);
        args.ancestor_masks = static_cast<const std::int32_t*>(d_masks.data());
        args.prefix_lengths = static_cast<const std::int32_t*>(d_prefix.data());
        const auto tree     = run_leaf(device, buffers, args, rows, bf16);
        const auto expected =
            oracle_output(tree.rotated, rows, positions, cache.decoded, prefixes, masks, split512);
        compare(tree.output, expected, kFusedTolerance, "tree context " + std::to_string(context),
                &maximum);
        auto broadcast                 = args;
        broadcast.prefix_length_stride = 0U;
        compare(run_leaf(device, buffers, broadcast, rows, bf16).output, expected, kFusedTolerance,
                "broadcast tree", &maximum);

        std::vector<std::int32_t> partial_positions{first, first + 1, -1, -1};
        std::vector<std::int32_t> partial_prefixes{first, first, -1, -1};
        std::vector<std::int32_t> partial_masks{1, 3, 0, 0};
        const std::int32_t live = 2;
        buffers.row_positions.copy_from_host(partial_positions.data(), rows * sizeof(std::int32_t));
        d_prefix.copy_from_host(partial_prefixes.data(), d_prefix.size());
        d_masks.copy_from_host(partial_masks.data(), d_masks.size());
        d_active.copy_from_host(&live, sizeof(live));
        auto partial              = args;
        partial.active_query_rows = static_cast<const std::int32_t*>(d_active.data());
        const auto partial_run    = run_leaf(device, buffers, partial, rows, bf16);
        const auto& actual        = partial_run.output;
        compare(std::span<const float>(actual).first(2U * kRowElements),
                oracle_output(std::span<const float>(partial_run.rotated).first(2U * kRowElements),
                              2U, partial_positions, cache.decoded, partial_prefixes, partial_masks,
                              split512),
                kFusedTolerance, "partial tree", &maximum);
        if (!std::all_of(actual.begin() + static_cast<std::ptrdiff_t>(2U * kRowElements),
                         actual.end(),
                         [](float v) { return std::bit_cast<std::uint32_t>(v) == 0U; })) {
            fail("inactive tree tail is not exact positive zero");
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
    const auto read =
        cache.fixture.cache->published_layer_read(cache.allocation, cache.publication, 0U);
    auto args = leaf_args(buffers, read, rows, workspace_bytes);
    buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
    const auto reference =
        run_leaf(device, buffers, leaf_args(buffers, read, rows, workspace_bytes), rows, bf16);
    const auto expected = oracle_output(reference.rotated, rows, positions, cache.decoded);
    compare(reference.output, expected, kStagedTolerance, "packed reference");
    args.active_query_rows = static_cast<const std::int32_t*>(d_active.data());
    for (const std::int32_t count : {0, 3, 6, -1, 7, 3}) {
        auto panel       = positions;
        const bool valid = count >= 0 && count <= static_cast<std::int32_t>(rows);
        std::fill(panel.begin() + (valid ? count : 0), panel.end(), -777);
        buffers.row_positions.copy_from_host(panel.data(), rows * sizeof(std::int32_t));
        d_active.copy_from_host(&count, sizeof(count));
        const auto actual = run_leaf(device, buffers, args, rows, bf16).output;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!valid) {
                if (!std::isnan(actual[i])) fail("invalid packed count did not poison");
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
    buffers.row_positions.copy_from_host(positions.data(), rows * sizeof(std::int32_t));
    {
        // Caller-owned basis changes (the schedule's form): rotate Q, attend with
        // tq_rotated_basis, rotate the output back. Same Ops, so bit-identical to the leaf-owned
        // call.
        buffers.query.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
        HIP_CHECK(kv::tq_rotate_query(static_cast<hip_bfloat16*>(buffers.query.data()),
                                      rows * kQueryHeads, device.stream));
        auto owned             = leaf_args(buffers, read, rows, workspace_bytes);
        owned.tq_rotated_basis = true;
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(owned, device.stream));
        HIP_CHECK(kv::tq_unrotate_output(static_cast<float*>(buffers.output.data()),
                                         rows * kQueryHeads, device.stream));
        device.synchronize();
        std::vector<float> actual(rows * kRowElements);
        buffers.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::bit_cast<std::uint32_t>(actual[i]) !=
                std::bit_cast<std::uint32_t>(reference.output[i])) {
                fail("caller-owned TQ basis changes differ from the leaf-owned call");
            }
        }
    }
    for (const std::size_t count : {std::size_t{2U}, std::size_t{16U}}) {
        std::vector<std::unique_ptr<LeafBuffers>> batch;
        std::vector<q27::R9700FullAttentionArgs> calls;
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
            std::vector<float> actual(rows * kRowElements);
            sequence->output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            compare(actual, expected, kStagedTolerance, "batched packed sequence");
        }
    }
    std::cout << "packed_forms device_counts/sequence_batches=PASS\n";
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            fail("selected physical device is not wave32 gfx1201");
        }
        qualify_rotation(device);
        qualify_codec(device);
        qualify_status(device);
        qualify_batched_append(device);
        qualify_compaction(device);
        qualify_causal_routes(device);
        qualify_tree_routes(device);
        qualify_packed_forms(device);
        std::cout << "tq_kv_qual PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tq_kv_qual FAILED: " << error.what() << '\n';
        return 1;
    }
}
