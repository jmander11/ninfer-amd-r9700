#pragma once

// Independent host oracle for the R9700 XAttention qualification route. It implements the paper's
// inverse-stride antidiagonal estimator directly in FP64 over represented BF16 Q and decoded FP8 K,
// then evaluates full FP64 softmax/PV over exactly the retained 64-token logical cache pages. It
// does not share production indexing, reductions, scratch layout, or ranking code.

#include "fp8_int4_kv_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::test::xattention_oracle {

inline constexpr std::size_t kPageSize = 64U;
inline constexpr std::size_t kFindBlockSize = 128U;
inline constexpr std::size_t kQueryBlockRows = 128U;

struct Geometry {
    std::size_t head_dim{};
    std::size_t query_heads{};
    std::size_t kv_heads{};

    [[nodiscard]] std::size_t q_to_kv() const {
        if (head_dim == 0U || query_heads == 0U || kv_heads == 0U ||
            query_heads % kv_heads != 0U) {
            throw std::invalid_argument("XAttention oracle geometry is invalid");
        }
        return query_heads / kv_heads;
    }
};

struct KeepSets {
    std::size_t query_blocks{};
    std::size_t query_heads{};
    std::vector<std::vector<std::uint16_t>> pages; // [query_head * query_blocks + block]

    [[nodiscard]] const std::vector<std::uint16_t>& at(std::size_t query_head,
                                                        std::size_t query_block) const {
        if (query_head >= query_heads || query_block >= query_blocks ||
            pages.size() != query_heads * query_blocks) {
            throw std::out_of_range("XAttention keep-set index is invalid");
        }
        return pages[query_head * query_blocks + query_block];
    }
};

using PlaneLogits = std::vector<std::vector<std::pair<std::size_t, double>>>;

inline std::vector<std::uint16_t> select_blocks(std::span<const double> block_mass,
                                               double tau, std::size_t recent_first,
                                               std::size_t recent_last) {
    if (block_mass.empty() || !(tau > 0.0 && tau <= 1.0) || !std::isfinite(tau) ||
        recent_first > recent_last || recent_last >= block_mass.size() ||
        block_mass.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument("XAttention page selection inputs are invalid");
    }
    std::vector<std::size_t> ranking(block_mass.size());
    std::iota(ranking.begin(), ranking.end(), std::size_t{0});
    std::stable_sort(ranking.begin(), ranking.end(), [&](std::size_t left,
                                                         std::size_t right) {
        if (block_mass[left] != block_mass[right]) return block_mass[left] > block_mass[right];
        return left < right;
    });
    const double total = std::accumulate(block_mass.begin(), block_mass.end(), 0.0);
    if (!std::isfinite(total) || total < 0.0 ||
        std::any_of(block_mass.begin(), block_mass.end(),
                    [](double mass) { return !std::isfinite(mass) || mass < 0.0; })) {
        throw std::invalid_argument("XAttention page mass is invalid");
    }
    const double required = tau * total;
    std::vector<bool> selected(block_mass.size(), false);
    selected[0] = true;
    for (std::size_t page = recent_first; page <= recent_last; ++page) selected[page] = true;
    double retained = 0.0;
    for (std::size_t block = 0U; block < block_mass.size(); ++block) {
        if (selected[block]) retained += block_mass[block];
    }
    for (std::size_t page : ranking) {
        if (retained >= required || !(block_mass[page] > 0.0)) break;
        if (selected[page]) continue;
        selected[page] = true;
        retained += block_mass[page];
    }
    std::vector<std::uint16_t> keep;
    for (std::size_t page = 0U; page < block_mass.size(); ++page) {
        if (selected[page]) keep.push_back(static_cast<std::uint16_t>(page));
    }
    return keep;
}

// Algorithm 1 applies softmax independently to every reshaped query plane before the
// antidiagonal probabilities are accumulated into physical cache pages. Keeping this as a
// separately testable host step prevents the tempting but incorrect accumulation of exp(logit -
// plane_max), whose magnitude depends on each plane's unrelated softmax denominator.
inline std::vector<double> normalized_block_mass(const PlaneLogits& planes,
                                                std::size_t find_blocks,
                                                std::size_t stride) {
    if (find_blocks == 0U || stride == 0U) {
        throw std::invalid_argument("XAttention page-mass geometry is invalid");
    }
    std::vector<double> page_mass(find_blocks, 0.0);
    for (const auto& logits : planes) {
        if (logits.empty()) continue;
        const double maximum = std::max_element(
            logits.begin(), logits.end(),
            [](const auto& left, const auto& right) { return left.second < right.second; })
                                   ->second;
        double denominator = 0.0;
        for (const auto& [key_group, logit] : logits) {
            if (key_group * stride / kFindBlockSize >= find_blocks || !std::isfinite(logit)) {
                throw std::invalid_argument("XAttention plane logits are invalid");
            }
            denominator += std::exp(logit - maximum);
        }
        if (!std::isfinite(denominator) || !(denominator > 0.0)) {
            throw std::runtime_error("XAttention estimator softmax is invalid");
        }
        for (const auto& [key_group, logit] : logits) {
            page_mass[(key_group * stride) / kFindBlockSize] +=
                std::exp(logit - maximum) / denominator;
        }
    }
    return page_mass;
}

inline KeepSets antidiagonal_keep_sets(
    std::span<const float> represented_query,
    const fp8_int4_kv_oracle::Fp8Keys& keys,
    std::span<const std::int32_t> row_positions, Geometry geometry,
    std::size_t stride, double tau, double attention_scale) {
    const std::size_t q_to_kv = geometry.q_to_kv();
    const std::size_t rows = row_positions.size();
    if (rows < 2U || (stride != 8U && stride != 16U) || !(tau > 0.0 && tau <= 1.0) ||
        !std::isfinite(tau) || !std::isfinite(attention_scale) || !(attention_scale > 0.0) ||
        represented_query.size() != rows * geometry.query_heads * geometry.head_dim ||
        keys.dimension != geometry.head_dim || keys.tokens % geometry.kv_heads != 0U) {
        throw std::invalid_argument("XAttention oracle inputs are incompatible");
    }
    const std::size_t context = keys.tokens / geometry.kv_heads;
    const std::size_t query_blocks = (rows + kQueryBlockRows - 1U) / kQueryBlockRows;
    for (std::size_t row = 0U; row < rows; ++row) {
        const std::int64_t expected = static_cast<std::int64_t>(row_positions.front()) +
                                      static_cast<std::int64_t>(row);
        if (row_positions.front() < 0 || expected >= static_cast<std::int64_t>(context) ||
            row_positions[row] != expected) {
            throw std::invalid_argument("XAttention rows must be a contiguous causal suffix");
        }
    }
    KeepSets result{query_blocks, geometry.query_heads,
                    std::vector<std::vector<std::uint16_t>>(geometry.query_heads * query_blocks)};

    for (std::size_t block = 0U; block < query_blocks; ++block) {
        const std::size_t query_start = block * kQueryBlockRows;
        const std::size_t block_rows = std::min(kQueryBlockRows, rows - query_start);
        const std::int32_t first = row_positions[query_start];
        const std::size_t last = static_cast<std::size_t>(first) + block_rows - 1U;
        const std::size_t key_pages = last / kPageSize + 1U;
        if (key_pages > std::numeric_limits<std::uint16_t>::max()) {
            throw std::overflow_error("XAttention page ID exceeds U16");
        }

        for (std::size_t query_head = 0U; query_head < geometry.query_heads; ++query_head) {
            auto& keep = result.pages[query_head * query_blocks + block];
            // Algorithm 1 uses one global stride origin for Q and K. A product chunk whose first
            // query is not stride aligned takes the exact dense fallback; locally reanchoring Q
            // would make selection depend on chunk partitioning.
            if (tau == 1.0 || static_cast<std::size_t>(first) % kFindBlockSize != 0U) {
                keep.resize(key_pages);
                std::iota(keep.begin(), keep.end(), std::uint16_t{0});
                continue;
            }
            const std::size_t kv_head = query_head / q_to_kv;
            const std::size_t planes = (block_rows + stride - 1U) / stride;
            const std::size_t key_groups = (last + stride) / stride;
            PlaneLogits plane_logits;
            plane_logits.reserve(planes);
            for (std::size_t plane = 0U; plane < planes; ++plane) {
                const std::size_t query_max = static_cast<std::size_t>(first) +
                    std::min(plane * stride + stride - 1U, block_rows - 1U);
                auto& logits = plane_logits.emplace_back();
                logits.reserve(key_groups);
                for (std::size_t key_group = 0U; key_group < key_groups; ++key_group) {
                    if (key_group * stride > query_max) continue;
                    double dot = 0.0;
                    for (std::size_t s = 0U; s < stride; ++s) {
                        const std::size_t relative = stride - 1U - s + plane * stride;
                        if (relative >= block_rows) continue;
                        const std::size_t logical_key = key_group * stride + s;
                        if (logical_key > query_max) continue;
                        const std::size_t query_base =
                            ((query_start + relative) * geometry.query_heads + query_head) *
                            geometry.head_dim;
                        const std::size_t represented_key =
                            logical_key * geometry.kv_heads + kv_head;
                        for (std::size_t feature = 0U; feature < geometry.head_dim; ++feature) {
                            const float q = represented_query[query_base + feature];
                            if (!std::isfinite(q)) {
                                throw std::invalid_argument("XAttention query must be finite");
                            }
                            dot += static_cast<double>(q) * keys.at(represented_key, feature);
                        }
                    }
                    logits.emplace_back(key_group,
                                        dot * attention_scale / static_cast<double>(stride));
                }
            }
            const std::size_t find_blocks = last / kFindBlockSize + 1U;
            const std::vector<double> block_mass =
                normalized_block_mass(plane_logits, find_blocks, stride);
            const std::size_t recent_first = static_cast<std::size_t>(first) / kFindBlockSize;
            const std::size_t recent_last = last / kFindBlockSize;
            const std::vector<std::uint16_t> blocks =
                select_blocks(block_mass, tau, recent_first, recent_last);
            for (std::uint16_t block_id : blocks) {
                const std::size_t first_page = static_cast<std::size_t>(block_id) * 2U;
                keep.push_back(static_cast<std::uint16_t>(first_page));
                if (first_page + 1U < key_pages) {
                    keep.push_back(static_cast<std::uint16_t>(first_page + 1U));
                }
            }
        }
    }
    return result;
}

inline std::vector<double> sparse_attention_fp64(
    std::span<const float> represented_query,
    const fp8_int4_kv_oracle::Fp8Keys& keys,
    const fp8_int4_kv_oracle::Int4Values& values,
    std::span<const std::int32_t> row_positions, Geometry geometry,
    const KeepSets& keep_sets, double attention_scale) {
    const std::size_t q_to_kv = geometry.q_to_kv();
    const std::size_t rows = row_positions.size();
    const std::size_t context = keys.tokens / geometry.kv_heads;
    if (values.tokens != keys.tokens || values.dimension != geometry.head_dim ||
        represented_query.size() != rows * geometry.query_heads * geometry.head_dim ||
        keep_sets.query_blocks != (rows + kQueryBlockRows - 1U) / kQueryBlockRows ||
        keep_sets.query_heads != geometry.query_heads || !std::isfinite(attention_scale) ||
        !(attention_scale > 0.0)) {
        throw std::invalid_argument("XAttention FP64 oracle inputs are incompatible");
    }
    if (keep_sets.pages.size() != keep_sets.query_heads * keep_sets.query_blocks ||
        std::any_of(represented_query.begin(), represented_query.end(),
                    [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("XAttention FP64 oracle inputs are malformed");
    }
    std::vector<double> output(represented_query.size(), 0.0);
    for (std::size_t row = 0U; row < rows; ++row) {
        const std::int32_t position = row_positions[row];
        const std::int64_t expected = static_cast<std::int64_t>(row_positions.front()) +
                                      static_cast<std::int64_t>(row);
        if (row_positions.front() < 0 || position != expected || position < 0 ||
            static_cast<std::size_t>(position) >= context) {
            throw std::invalid_argument("XAttention row position is invalid");
        }
        for (std::size_t query_head = 0U; query_head < geometry.query_heads; ++query_head) {
            const std::size_t kv_head = query_head / q_to_kv;
            const auto& pages = keep_sets.at(query_head, row / kQueryBlockRows);
            if (pages.empty() || !std::is_sorted(pages.begin(), pages.end()) ||
                std::adjacent_find(pages.begin(), pages.end()) != pages.end() ||
                static_cast<std::size_t>(pages.back()) * kPageSize >= context) {
                throw std::invalid_argument("XAttention retained pages are malformed");
            }
            std::vector<double> scores;
            std::vector<std::size_t> logical_tokens;
            for (std::uint16_t page : pages) {
                const std::size_t first = static_cast<std::size_t>(page) * kPageSize;
                if (first > static_cast<std::size_t>(position)) break;
                const std::size_t end = std::min(first + kPageSize,
                                                 static_cast<std::size_t>(position) + 1U);
                for (std::size_t logical = first; logical < end; ++logical) {
                    double dot = 0.0;
                    const std::size_t query_base =
                        (row * geometry.query_heads + query_head) * geometry.head_dim;
                    const std::size_t represented_key = logical * geometry.kv_heads + kv_head;
                    for (std::size_t feature = 0U; feature < geometry.head_dim; ++feature) {
                        dot += static_cast<double>(represented_query[query_base + feature]) *
                               keys.at(represented_key, feature);
                    }
                    scores.push_back(dot * attention_scale);
                    logical_tokens.push_back(logical);
                }
            }
            if (scores.empty()) throw std::runtime_error("XAttention retained no causal tokens");
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            const std::size_t output_base =
                (row * geometry.query_heads + query_head) * geometry.head_dim;
            for (std::size_t index = 0U; index < scores.size(); ++index) {
                const double probability = scores[index] / denominator;
                const std::size_t represented_value =
                    logical_tokens[index] * geometry.kv_heads + kv_head;
                for (std::size_t feature = 0U; feature < geometry.head_dim; ++feature) {
                    output[output_base + feature] +=
                        probability * values.at(represented_value, feature);
                }
            }
        }
    }
    return output;
}

} // namespace ninfer::test::xattention_oracle
