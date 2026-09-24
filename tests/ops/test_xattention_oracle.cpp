#include "ops/xattention_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace xo = ninfer::test::xattention_oracle;
namespace codec = ninfer::test::fp8_int4_kv_oracle;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const char* message) {
    try {
        function();
        expect(false, message);
    } catch (const Exception&) {
    } catch (...) {
        expect(false, message);
    }
}

struct Fixture {
    xo::Geometry geometry{16U, 6U, 1U};
    std::size_t context = 257U;
    std::size_t rows = 129U;
    std::vector<float> query;
    codec::Fp8Keys keys;
    codec::Int4Values values;
    std::vector<std::int32_t> positions;

    Fixture() {
        query.resize(rows * geometry.query_heads * geometry.head_dim, 0.0F);
        std::vector<float> key_source(context * geometry.kv_heads * geometry.head_dim, 0.0F);
        std::vector<float> value_source(key_source.size(), 0.0F);
        positions.resize(rows);
        for (std::size_t row = 0U; row < rows; ++row) {
            positions[row] = static_cast<std::int32_t>(128U + row);
            for (std::size_t head = 0U; head < geometry.query_heads; ++head) {
                query[(row * geometry.query_heads + head) * geometry.head_dim] =
                    head == 0U ? 1.0F : 0.25F;
            }
        }
        // Page one is the estimator hot page for Q head zero; pages zero and two are still forced
        // by sink and ragged-block recency rules where applicable.
        for (std::size_t token = 64U; token < 128U; ++token) {
            key_source[token * geometry.head_dim] = 8.0F;
        }
        for (std::size_t token = 0U; token < context; ++token) {
            value_source[token * geometry.head_dim] = static_cast<float>(token % 7U) - 3.0F;
        }
        keys = codec::encode_keys(key_source, context * geometry.kv_heads, geometry.head_dim);
        values = codec::encode_values(value_source, context * geometry.kv_heads,
                                      geometry.head_dim, 16U);
    }
};

void test_dense_identity_and_deterministic_order() {
    Fixture fixture;
    const xo::KeepSets keep = xo::antidiagonal_keep_sets(
        fixture.query, fixture.keys, fixture.positions, fixture.geometry, 16U, 1.0, 0.25);
    expect(keep.query_blocks == 2U, "ragged 129-row prefill must make two query blocks");
    for (std::size_t head = 0U; head < fixture.geometry.query_heads; ++head) {
        expect(keep.at(head, 0U) == std::vector<std::uint16_t>({0U, 1U, 2U, 3U}),
               "tau one must retain every causal page for the first block");
        expect(keep.at(head, 1U) == std::vector<std::uint16_t>({0U, 1U, 2U, 3U, 4U}),
               "tau one must retain the causal tail page in increasing order");
    }
    const std::vector<double> sparse = xo::sparse_attention_fp64(
        fixture.query, fixture.keys, fixture.values, fixture.positions, fixture.geometry, keep,
        0.25);
    expect(sparse.size() == fixture.query.size() &&
               std::all_of(sparse.begin(), sparse.end(), [](double value) {
                   return std::isfinite(value);
               }),
           "tau-one sparse oracle must produce complete finite output");
}

void test_sparse_sink_hot_and_recent_retention() {
    Fixture fixture;
    const xo::KeepSets first = xo::antidiagonal_keep_sets(
        fixture.query, fixture.keys, fixture.positions, fixture.geometry, 16U, 0.50, 0.25);
    const xo::KeepSets second = xo::antidiagonal_keep_sets(
        fixture.query, fixture.keys, fixture.positions, fixture.geometry, 16U, 0.50, 0.25);
    expect(first.pages == second.pages, "XAttention keep ordering must be deterministic");
    const auto& main = first.at(0U, 0U);
    expect(std::find(main.begin(), main.end(), 0U) != main.end(),
           "XAttention must retain the sink page");
    expect(std::find(main.begin(), main.end(), 1U) != main.end(),
           "XAttention must retain the hot/recent page");
    const auto& ragged = first.at(0U, 1U);
    expect(std::find(ragged.begin(), ragged.end(), 4U) != ragged.end(),
           "XAttention must retain the first page of a ragged query tail block");
}

void test_unaligned_chunk_uses_dense_fallback() {
    Fixture fixture;
    // 64 is stride aligned for both qualified S values but not aligned to paper B128.
    for (std::int32_t& position : fixture.positions) position -= 64;
    const xo::KeepSets keep = xo::antidiagonal_keep_sets(
        fixture.query, fixture.keys, fixture.positions, fixture.geometry, 16U, 0.5, 0.25);
    for (std::size_t head = 0U; head < fixture.geometry.query_heads; ++head) {
        expect(keep.at(head, 0U) == std::vector<std::uint16_t>({0U, 1U, 2U}),
               "unaligned first query block must use the full causal page list");
        expect(keep.at(head, 1U) == std::vector<std::uint16_t>({0U, 1U, 2U, 3U}),
               "unaligned ragged query block must remain on dense fallback");
    }
}

void test_short_initial_partial_plane_retains_mandatory_page() {
    Fixture fixture;
    fixture.rows = 9U;
    fixture.query.resize(fixture.rows * fixture.geometry.query_heads * fixture.geometry.head_dim);
    fixture.positions.resize(fixture.rows);
    std::iota(fixture.positions.begin(), fixture.positions.end(), std::int32_t{0});
    const xo::KeepSets keep = xo::antidiagonal_keep_sets(
        fixture.query, fixture.keys, fixture.positions, fixture.geometry, 16U, 0.5, 0.25);
    expect(keep.query_blocks == 1U,
           "short initial prefill must use one partial estimator plane");
    for (std::size_t head = 0U; head < fixture.geometry.query_heads; ++head) {
        expect(keep.at(head, 0U) == std::vector<std::uint16_t>({0U}),
               "partial plane with no complete key group must retain the mandatory causal page");
    }
    const std::vector<double> output = xo::sparse_attention_fp64(
        fixture.query, fixture.keys, fixture.values, fixture.positions, fixture.geometry, keep,
        0.25);
    expect(std::all_of(output.begin(), output.end(), [](double value) {
               return std::isfinite(value);
           }),
           "short initial partial-plane attention must remain finite");
}

void test_each_plane_is_softmax_normalized_before_page_accumulation() {
    // Plane zero has two candidates and strongly favors page one. Plane one has one page-one
    // candidate but four equally strong page-two candidates. Its larger denominator must not give
    // it four times the voting mass. The old unnormalized exp(logit - plane_max) accumulation
    // incorrectly ranked page two first for this case.
    const xo::PlaneLogits logits{
        {{8U, 10.0}, {16U, 0.0}},
        {{8U, -1.3862943611198906}, {16U, 0.0}, {17U, 0.0}, {18U, 0.0}, {19U, 0.0}},
    };
    const std::vector<double> normalized = xo::normalized_block_mass(logits, 3U, 16U);
    expect(normalized[1] > normalized[2],
           "per-plane softmax must rank page one above page two");
    expect(std::abs(std::accumulate(normalized.begin(), normalized.end(), 0.0) - 2.0) < 1e-12,
           "each nonempty estimator plane must contribute unit probability mass");

    std::vector<double> buggy(3U, 0.0);
    for (const auto& plane : logits) {
        const double maximum = std::max_element(
            plane.begin(), plane.end(),
            [](const auto& left, const auto& right) { return left.second < right.second; })
                                   ->second;
        for (const auto& [key_group, logit] : plane) {
            buggy[(key_group * 16U) / xo::kFindBlockSize] += std::exp(logit - maximum);
        }
    }
    expect(buggy[2] > buggy[1],
           "regression fixture must distinguish normalized from unnormalized plane masses");
    const auto minimum_half_mass_set = [](const std::vector<double>& mass) {
        std::vector<std::size_t> rank(mass.size());
        std::iota(rank.begin(), rank.end(), std::size_t{0});
        std::stable_sort(rank.begin(), rank.end(), [&](std::size_t left, std::size_t right) {
            if (mass[left] != mass[right]) return mass[left] > mass[right];
            return left < right;
        });
        const double required =
            0.5 * std::accumulate(mass.begin(), mass.end(), 0.0);
        double retained = 0.0;
        std::vector<std::size_t> keep;
        for (const std::size_t page : rank) {
            if (retained >= required || !(mass[page] > 0.0)) break;
            keep.push_back(page);
            retained += mass[page];
        }
        return keep;
    };
    expect(minimum_half_mass_set(normalized) == std::vector<std::size_t>({1U}),
           "normalized tau-half selection must retain page one");
    expect(minimum_half_mass_set(buggy) == std::vector<std::size_t>({2U}),
           "unnormalized tau-half selection must retain a different page");
}

void test_mandatory_pages_count_toward_tau_budget() {
    const std::vector<double> mass{0.45, 0.10, 0.05, 0.40};
    expect(xo::select_blocks(mass, 0.50, 3U, 3U) ==
               std::vector<std::uint16_t>({0U, 3U}),
           "sink and recent page mass must count toward the tau budget");
    expect(xo::select_blocks(mass, 0.90, 3U, 3U) ==
               std::vector<std::uint16_t>({0U, 1U, 3U}),
           "selection must add only enough nonmandatory mass to cross tau");
    const std::vector<double> uniform(10U, 1.0);
    expect(xo::select_blocks(uniform, 0.90, 9U, 9U) ==
               std::vector<std::uint16_t>({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 9U}),
           "tau 0.9 must retain the minimum set reaching 90 percent of uniform mass");
}

void test_b128_selection_expands_only_existing_b64_pages() {
    const xo::Geometry geometry{16U, 1U, 1U};
    constexpr std::size_t context = 130U;
    constexpr std::size_t rows = 2U;
    std::vector<float> query(rows * geometry.head_dim, 0.0F);
    std::vector<float> key_source(context * geometry.head_dim, 0.0F);
    const codec::Fp8Keys keys = codec::encode_keys(key_source, context, geometry.head_dim);
    const std::vector<std::int32_t> positions{128, 129};

    const xo::KeepSets keep = xo::antidiagonal_keep_sets(
        query, keys, positions, geometry, 16U, 0.9, 0.25);
    expect(keep.at(0U, 0U) == std::vector<std::uint16_t>({0U, 1U, 2U}),
           "B128 selections must expand to paired B64 pages without emitting a nonexistent "
           "odd-tail page");
}

void test_partial_key_group_contributes_zero_padded_logit() {
    const xo::Geometry geometry{16U, 1U, 1U};
    constexpr std::size_t context = 393U;
    constexpr std::size_t rows = 9U;
    std::vector<float> query(rows * geometry.head_dim, 0.0F);
    query[8U * geometry.head_dim] = 1.0F;
    std::vector<float> key_source(context * geometry.head_dim, 0.0F);
    // For S16, Q relative row 8 pairs with s=7. The last real K group begins at 384 and
    // is only nine tokens wide, so token 391 is the decisive mixed real/padded term.
    key_source[391U * geometry.head_dim] = 448.0F;
    const codec::Fp8Keys keys = codec::encode_keys(key_source, context, geometry.head_dim);
    std::vector<std::int32_t> positions(rows);
    std::iota(positions.begin(), positions.end(), std::int32_t{384});
    const xo::KeepSets keep = xo::antidiagonal_keep_sets(
        query, keys, positions, geometry, 16U, 0.9, 0.25);
    expect(keep.at(0U, 0U) == std::vector<std::uint16_t>({0U, 1U, 6U}),
           "partial final K group must contribute its mixed real/zero-padded logit");
}

void test_malformed_and_nonfinite_rejection() {
    Fixture fixture;
    fixture.positions[9] += 1;
    expect_throws<std::invalid_argument>(
        [&] {
            (void)xo::antidiagonal_keep_sets(fixture.query, fixture.keys, fixture.positions,
                                             fixture.geometry, 16U, 0.9, 0.25);
        },
        "XAttention oracle must reject a position gap");
    fixture.positions[9] -= 1;
    fixture.positions[64] += 1;
    expect_throws<std::invalid_argument>(
        [&] {
            (void)xo::antidiagonal_keep_sets(fixture.query, fixture.keys, fixture.positions,
                                             fixture.geometry, 16U, 0.9, 0.25);
        },
        "XAttention oracle must reject a query-block-boundary position gap");
    fixture.positions[64] -= 1;
    fixture.query[0] = std::numeric_limits<float>::quiet_NaN();
    expect_throws<std::invalid_argument>(
        [&] {
            (void)xo::antidiagonal_keep_sets(fixture.query, fixture.keys, fixture.positions,
                                             fixture.geometry, 16U, 0.9, 0.25);
        },
        "XAttention oracle must reject a nonfinite represented query");
    expect_throws<std::invalid_argument>(
        [&] {
            (void)xo::antidiagonal_keep_sets(fixture.query, fixture.keys, fixture.positions,
                                             fixture.geometry, 4U, 0.9, 0.25);
        },
        "XAttention oracle must reject an unqualified stride");

    fixture.query[0] = 0.0F;
    xo::KeepSets keep = xo::antidiagonal_keep_sets(
        fixture.query, fixture.keys, fixture.positions, fixture.geometry, 16U, 0.9, 0.25);
    fixture.query.back() = std::numeric_limits<float>::infinity();
    expect_throws<std::invalid_argument>(
        [&] {
            (void)xo::sparse_attention_fp64(fixture.query, fixture.keys, fixture.values,
                                            fixture.positions, fixture.geometry, keep, 0.25);
        },
        "XAttention FP64 attention oracle must reject a nonfinite query");
    fixture.query.back() = 0.0F;
    keep.pages.front() = {1U, 0U};
    expect_throws<std::invalid_argument>(
        [&] {
            (void)xo::sparse_attention_fp64(fixture.query, fixture.keys, fixture.values,
                                            fixture.positions, fixture.geometry, keep, 0.25);
        },
        "XAttention FP64 attention oracle must reject unordered retained pages");
}

} // namespace

int main() {
    test_dense_identity_and_deterministic_order();
    test_sparse_sink_hot_and_recent_retention();
    test_unaligned_chunk_uses_dense_fallback();
    test_short_initial_partial_plane_retains_mandatory_page();
    test_each_plane_is_softmax_normalized_before_page_accumulation();
    test_mandatory_pages_count_toward_tau_budget();
    test_b128_selection_expands_only_existing_b64_pages();
    test_partial_key_group_contributes_zero_padded_logit();
    test_malformed_and_nonfinite_rejection();
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " XAttention host keep-set and retained-page FP64 oracle\n";
    return failures == 0 ? 0 : 1;
}
