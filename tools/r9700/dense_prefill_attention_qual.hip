#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace kv = ninfer::ops::r9700::kv;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;

constexpr std::uint32_t kD = 256U;
constexpr std::uint32_t kHq = 24U;
constexpr std::uint32_t kHkv = 4U;
constexpr std::uint32_t kPage = 64U;
constexpr std::uint32_t kMaxP = 4096U;
constexpr std::uint32_t kPhysicalPages = kMaxP / kPage + 5U;
constexpr std::size_t kPhysicalTokens = kPhysicalPages * kPage;
constexpr float kScale = 1.0F / 16.0F;
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

#define HIP_CHECK(call) do { const hipError_t e_ = (call); if (e_ != hipSuccess) \
    fail(std::string(#call) + ": " + hipGetErrorString(e_)); } while (false)

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&pointer_, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (pointer_ != nullptr) (void)hipFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0U;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    ~Event() { (void)hipEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    hipEvent_t get() const noexcept { return event_; }
private:
    hipEvent_t event_ = nullptr;
};

std::size_t key_offset(std::size_t physical, std::uint32_t head, std::uint32_t feature) {
    const std::size_t page = physical / kPage, in_page = physical % kPage;
    return in_page + kPage * (feature + kD * (page + kPhysicalPages * head));
}

std::size_t value_offset(std::size_t physical, std::uint32_t head, std::uint32_t feature) {
    return feature / 2U + (kD / 2U) *
        (physical % kPage + kPage * (head + kHkv * (physical / kPage)));
}

std::size_t scale_offset(std::size_t physical, std::uint32_t head, std::uint32_t feature,
                         std::uint32_t group) {
    const std::uint32_t groups = kD / group;
    return feature / group + groups *
        (physical % kPage + kPage * (head + kHkv * (physical / kPage)));
}

std::vector<std::uint32_t> make_pages() {
    std::vector<std::uint32_t> pages(kMaxP / kPage);
    for (std::uint32_t page = 0U; page < pages.size(); ++page) {
        pages[page] = (page * 11U + 3U) % kPhysicalPages;
    }
    return pages;
}

struct Fixture {
    std::uint32_t group;
    std::vector<std::uint32_t> pages;
    std::vector<std::uint8_t> keys;
    std::vector<std::uint8_t> values;
    std::vector<std::uint16_t> scales;
};

Fixture make_fixture(std::uint32_t group) {
    Fixture result{group, make_pages(),
                   std::vector<std::uint8_t>(kPhysicalTokens * kHkv * kD),
                   std::vector<std::uint8_t>(kPhysicalTokens * kHkv * kD / 2U),
                   std::vector<std::uint16_t>(kPhysicalTokens * kHkv * (kD / group))};
    constexpr std::array<std::uint8_t, 10> key_words{
        0x08U, 0x18U, 0x20U, 0x28U, 0x30U, 0x88U, 0x98U, 0xa0U, 0xa8U, 0xb0U};
    constexpr std::array<std::uint16_t, 6> scale_words{
        0x2c00U, 0x3000U, 0x3400U, 0x3800U, 0x3a00U, 0x3c00U};
    for (std::size_t physical = 0U; physical < kPhysicalTokens; ++physical) {
        for (std::uint32_t head = 0U; head < kHkv; ++head) {
            for (std::uint32_t feature = 0U; feature < kD; ++feature) {
                result.keys[key_offset(physical, head, feature)] =
                    key_words[(physical * 5U + head * 7U + feature * 3U) % key_words.size()];
            }
            for (std::uint32_t feature = 0U; feature < kD; feature += 2U) {
                const int lo = static_cast<int>((physical + head * 3U + feature) % 15U) - 7;
                const int hi = static_cast<int>((physical * 3U + head + feature) % 15U) - 7;
                result.values[value_offset(physical, head, feature)] =
                    static_cast<std::uint8_t>((lo & 15) | ((hi & 15) << 4));
            }
            for (std::uint32_t feature = 0U; feature < kD; feature += group) {
                result.scales[scale_offset(physical, head, feature, group)] =
                    scale_words[(physical + head + feature / group) % scale_words.size()];
            }
        }
    }
    return result;
}

std::vector<hip_bfloat16> make_queries(std::uint32_t rows) {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(rows) * kHq * kD);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t head = 0U; head < kHq; ++head) {
            for (std::uint32_t feature = 0U; feature < kD; ++feature) {
                const int word = static_cast<int>((row * 13U + head * 7U + feature * 5U) % 63U) - 31;
                result[(static_cast<std::size_t>(row) * kHq + head) * kD + feature] =
                    hip_bfloat16(static_cast<float>(word) / 128.0F);
            }
        }
    }
    return result;
}

std::vector<std::int32_t> make_positions(std::uint32_t rows) {
    std::vector<std::int32_t> result(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        result[row] = static_cast<std::int32_t>((row * 37U + 19U) % rows);
    }
    return result;
}

hipError_t launch_selected_dense(const kv::Fp8Int4KvAttentionArgs& args,
                                 void* workspace, std::size_t workspace_bytes,
                                 hipStream_t stream) {
    return kv::fp8_int4_kv_attention_dense_prefill_full_score(
        args, workspace, workspace_bytes, stream);
}

double expected(const Fixture& fixture, const std::vector<hip_bfloat16>& queries,
                const std::vector<std::int32_t>& positions, std::uint32_t row,
                std::uint32_t query_head, std::uint32_t feature) {
    const std::uint32_t kv_head = query_head / 6U;
    const std::size_t visible = static_cast<std::size_t>(positions[row]) + 1U;
    std::vector<double> weights(visible);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t logical = 0U; logical < visible; ++logical) {
        const std::size_t physical =
            static_cast<std::size_t>(fixture.pages[logical / kPage]) * kPage + logical % kPage;
        double dot = 0.0;
        for (std::uint32_t dimension = 0U; dimension < kD; ++dimension) {
            dot += static_cast<double>(static_cast<float>(queries[
                       (static_cast<std::size_t>(row) * kHq + query_head) * kD + dimension])) *
                   oracle::decode_e4m3fn(fixture.keys[key_offset(physical, kv_head, dimension)]);
        }
        weights[logical] = dot * static_cast<double>(kScale);
        maximum = std::max(maximum, weights[logical]);
    }
    double denominator = 0.0;
    for (double& weight : weights) {
        weight = std::exp(weight - maximum);
        denominator += weight;
    }
    double numerator = 0.0;
    for (std::size_t logical = 0U; logical < visible; ++logical) {
        const std::size_t physical =
            static_cast<std::size_t>(fixture.pages[logical / kPage]) * kPage + logical % kPage;
        const std::uint8_t packed = fixture.values[value_offset(physical, kv_head, feature)];
        const std::uint8_t nibble = (packed >> ((feature & 1U) * 4U)) & 15U;
        const int code = nibble >= 8U ? static_cast<int>(nibble) - 16 : nibble;
        const double value = static_cast<double>(code) * oracle::detail::fp16_to_fp32(
            fixture.scales[scale_offset(physical, kv_head, feature, fixture.group)]);
        numerator += weights[logical] * value;
    }
    return numerator / denominator;
}

double expected_score(const Fixture& fixture, const std::vector<hip_bfloat16>& queries,
                      std::uint32_t row, std::uint32_t query_head,
                      std::uint32_t logical) {
    const std::uint32_t kv_head = query_head / 6U;
    const std::size_t physical =
        static_cast<std::size_t>(fixture.pages[logical / kPage]) * kPage + logical % kPage;
    double dot = 0.0;
    for (std::uint32_t dimension = 0U; dimension < kD; ++dimension) {
        dot += static_cast<double>(static_cast<float>(queries[
                   (static_cast<std::size_t>(row) * kHq + query_head) * kD + dimension])) *
               oracle::decode_e4m3fn(fixture.keys[key_offset(physical, kv_head, dimension)]);
    }
    return dot * static_cast<double>(kScale);
}

void host_predicate_regression() {
    if (kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(127U) != 0U ||
        kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(4097U) != 0U ||
        kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(2048U) !=
            402849792U) {
        fail("full-score workspace capacity contract is incorrect");
    }
    kv::Fp8Int4KvAttentionArgs args{};
    args.query = reinterpret_cast<const hip_bfloat16*>(1U);
    args.key_codes = reinterpret_cast<const std::uint8_t*>(1U);
    args.value_codes = reinterpret_cast<const std::uint8_t*>(1U);
    args.value_scale_bits = reinterpret_cast<const std::uint16_t*>(1U);
    args.page_table = reinterpret_cast<const std::uint32_t*>(1U);
    args.page_table_pages = 32U;
    args.context = 2048U;
    args.physical_tokens = 2048U;
    args.query_heads = kHq; args.kv_heads = kHkv; args.q_to_kv = 6U;
    args.value_group = 16U; args.query_rows = 2048U;
    args.row_positions = reinterpret_cast<const std::int32_t*>(1U);
    args.attention_scale = kScale;
    args.key_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest;
    args.value_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    args.value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    args.output = reinterpret_cast<float*>(1U);
    auto rejected = [&](const kv::Fp8Int4KvAttentionArgs& candidate, const char* label) {
        if (launch_selected_dense(
                candidate, reinterpret_cast<void*>(4096U),
                kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(
                    candidate.query_rows), nullptr) != hipErrorInvalidValue) fail(label);
    };
    auto bad = args; bad.score_workspace = reinterpret_cast<float*>(1U);
    rejected(bad, "score workspace was accepted");
    bad = args; bad.key_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    rejected(bad, "wrong K layout was accepted");
    bad = args; bad.query_rows = 4097U;
    bad.context = 4097U;
    rejected(bad, "oversized prefill was accepted");
    bad = args; bad.ancestor_masks = reinterpret_cast<const std::int32_t*>(1U);
    bad.prefix_lengths = reinterpret_cast<const std::int32_t*>(1U);
    rejected(bad, "tree metadata was accepted");
    bad = args; bad.context = 2049U;
    rejected(bad, "later-prefix context was accepted");
    const std::size_t required =
        kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(args.query_rows);
    if (kv::fp8_int4_kv_attention_dense_prefill_full_score(
            args, reinterpret_cast<void*>(4096U), required - 1U, nullptr) !=
        hipErrorInvalidValue) {
        fail("undersized full-score workspace was accepted");
    }
}

void run_case(std::uint32_t rows, std::uint32_t group) {
    const Fixture fixture = make_fixture(group);
    const auto queries = make_queries(rows);
    const auto positions = make_positions(rows);
    DeviceBuffer<hip_bfloat16> d_queries(queries.size());
    DeviceBuffer<std::uint8_t> d_keys(fixture.keys.size());
    DeviceBuffer<std::uint8_t> d_values(fixture.values.size());
    DeviceBuffer<std::uint16_t> d_scales(fixture.scales.size());
    DeviceBuffer<std::uint32_t> d_pages(fixture.pages.size());
    DeviceBuffer<std::int32_t> d_positions(positions.size());
    DeviceBuffer<float> d_output(static_cast<std::size_t>(rows) * kHq * kD);
    DeviceBuffer<std::uint8_t> d_workspace(
        kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(rows));
    HIP_CHECK(hipMemcpy(d_queries.get(), queries.data(), d_queries.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_keys.get(), fixture.keys.data(), d_keys.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values.get(), fixture.values.data(), d_values.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales.get(), fixture.scales.data(), d_scales.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pages.get(), fixture.pages.data(), d_pages.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_positions.get(), positions.data(), d_positions.bytes(), hipMemcpyHostToDevice));
    kv::Fp8Int4KvAttentionArgs args{
        .query=d_queries.get(), .key_codes=d_keys.get(), .value_codes=d_values.get(),
        .value_scale_bits=d_scales.get(), .page_table=d_pages.get(),
        .page_table_pages=static_cast<std::uint32_t>(fixture.pages.size()),
        .context=rows, .physical_tokens=kPhysicalTokens, .query_heads=kHq, .kv_heads=kHkv,
        .q_to_kv=6U, .value_group=group, .query_rows=rows,
        .row_positions=d_positions.get(), .attention_scale=kScale,
        .key_layout=kv::Fp8Int4KvPlaneLayout::TokenFastest,
        .value_layout=kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .value_scale_layout=kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .output=d_output.get()};
    HIP_CHECK(launch_selected_dense(args, d_workspace.get(),
                                    d_workspace.bytes(), nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> output(static_cast<std::size_t>(rows) * kHq * kD);
    HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
    const float* scores = reinterpret_cast<const float*>(d_workspace.get());
    for (const auto row : {0U, rows / 3U, rows - 1U}) {
        const std::uint32_t visible = static_cast<std::uint32_t>(positions[row]) + 1U;
        for (const auto head : {0U, 7U, 23U}) {
            for (const auto logical : {0U, visible / 2U, visible - 1U}) {
                float actual = 0.0F;
                const std::size_t index =
                    (static_cast<std::size_t>(row) * kHq + head) * rows + logical;
                HIP_CHECK(hipMemcpy(&actual, scores + index, sizeof(actual), hipMemcpyDeviceToHost));
                const double reference = expected_score(fixture, queries, row, head, logical);
                const double error = std::abs(static_cast<double>(actual) - reference);
                if (!std::isfinite(reference) || !std::isfinite(actual) || !std::isfinite(error) ||
                    (error > 2.0e-3 &&
                     error / std::max(std::abs(reference), 1.0e-6) > 2.0e-3)) {
                    fail("production full-score QK differs from independent FP64 oracle");
                }
            }
        }
    }
    const std::array<std::uint32_t, 3> sampled_rows{0U, rows / 3U, rows - 1U};
    constexpr std::array<std::uint32_t, 3> sampled_heads{0U, 7U, 23U};
    constexpr std::array<std::uint32_t, 3> sampled_features{0U, 17U, 255U};
    for (const auto row : sampled_rows) for (const auto head : sampled_heads)
        for (const auto feature : sampled_features) {
            const double reference = expected(fixture, queries, positions, row, head, feature);
            const float actual = output[(static_cast<std::size_t>(row) * kHq + head) * kD + feature];
            const double error = std::abs(static_cast<double>(actual) - reference);
            if (!std::isfinite(reference) || !std::isfinite(actual) ||
                !std::isfinite(error) ||
                (error > 2.0e-3 &&
                 error / std::max(std::abs(reference), 1.0e-6) > 2.0e-3)) {
                fail("dense attention differs from independent FP64 oracle");
            }
        }
    if (rows == 129U) {
        const std::vector<float> baseline_output = output;
        DeviceBuffer<std::int32_t> d_active(1U);
        std::int32_t active = 127;
        HIP_CHECK(hipMemcpy(d_active.get(), &active, sizeof(active), hipMemcpyHostToDevice));
        args.active_query_rows = d_active.get();
        HIP_CHECK(hipMemset(d_output.get(), 0xff, d_output.bytes()));
        HIP_CHECK(launch_selected_dense(args, d_workspace.get(),
                                        d_workspace.bytes(), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        const std::size_t inactive_begin = static_cast<std::size_t>(active) * kHq * kD;
        if (!std::equal(output.begin(),
                        output.begin() + static_cast<std::ptrdiff_t>(inactive_begin),
                        baseline_output.begin())) {
            fail("device active-row prefix was not rewritten exactly");
        }
        if (!std::all_of(output.begin() + static_cast<std::ptrdiff_t>(inactive_begin), output.end(),
                         [](float value) { return value == 0.0F && !std::signbit(value); })) {
            fail("device active-row tail is not exact positive zero");
        }
        active = 0;
        HIP_CHECK(hipMemcpy(d_active.get(), &active, sizeof(active), hipMemcpyHostToDevice));
        auto inactive_args = args;
        inactive_args.query = reinterpret_cast<const hip_bfloat16*>(1U);
        inactive_args.key_codes = reinterpret_cast<const std::uint8_t*>(1U);
        inactive_args.value_codes = reinterpret_cast<const std::uint8_t*>(1U);
        inactive_args.value_scale_bits = reinterpret_cast<const std::uint16_t*>(1U);
        inactive_args.page_table = reinterpret_cast<const std::uint32_t*>(1U);
        inactive_args.page_table_row = reinterpret_cast<const std::int32_t*>(1U);
        inactive_args.page_table_row_stride = args.page_table_pages;
        inactive_args.page_table_row_count = 1U;
        inactive_args.row_positions = reinterpret_cast<const std::int32_t*>(1U);
        HIP_CHECK(launch_selected_dense(inactive_args, d_workspace.get(),
                                        d_workspace.bytes(), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(),
                         [](float value) { return value == 0.0F && !std::signbit(value); })) {
            fail("zero active-row state touched an inactive input or did not write positive zero");
        }
        active = -1;
        HIP_CHECK(hipMemcpy(d_active.get(), &active, sizeof(active), hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, d_workspace.get(),
                                        d_workspace.bytes(), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
            fail("invalid device active-row status did not poison every row");
        }

        args.active_query_rows = nullptr;
        auto malformed_positions = positions;
        malformed_positions[5] = -1;
        HIP_CHECK(hipMemcpy(d_positions.get(), malformed_positions.data(), d_positions.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, d_workspace.get(),
                                        d_workspace.bytes(), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        const std::size_t bad_row = 5U * kHq * kD;
        if (!std::all_of(output.begin() + static_cast<std::ptrdiff_t>(bad_row),
                         output.begin() + static_cast<std::ptrdiff_t>(bad_row + kHq * kD),
                         [](float value) { return std::isnan(value); })) {
            fail("invalid causal row did not poison exactly that output row");
        }

        HIP_CHECK(hipMemcpy(d_positions.get(), positions.data(), d_positions.bytes(),
                            hipMemcpyHostToDevice));
        DeviceBuffer<std::int32_t> d_page_row(1U);
        std::vector<std::uint32_t> page_rows(fixture.pages.size() * 2U,
                                             std::numeric_limits<std::uint32_t>::max());
        std::copy(fixture.pages.begin(), fixture.pages.end(),
                  page_rows.begin() + static_cast<std::ptrdiff_t>(fixture.pages.size()));
        DeviceBuffer<std::uint32_t> d_page_rows(page_rows.size());
        HIP_CHECK(hipMemcpy(d_page_rows.get(), page_rows.data(), d_page_rows.bytes(),
                            hipMemcpyHostToDevice));
        const std::int32_t selected_page_row = 1;
        HIP_CHECK(hipMemcpy(d_page_row.get(), &selected_page_row, sizeof(selected_page_row),
                            hipMemcpyHostToDevice));
        args.page_table = d_page_rows.get();
        args.page_table_row = d_page_row.get();
        args.page_table_row_stride = static_cast<std::uint32_t>(fixture.pages.size());
        args.page_table_row_count = 2U;
        HIP_CHECK(launch_selected_dense(args, d_workspace.get(),
                                        d_workspace.bytes(), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        for (const auto row : {0U, 64U, 128U}) {
            const double reference = expected(fixture, queries, positions, row, 23U, 255U);
            const float actual = output[(static_cast<std::size_t>(row) * kHq + 23U) * kD + 255U];
            const double error = std::abs(static_cast<double>(actual) - reference);
            if (error > 2.0e-3 &&
                error / std::max(std::abs(reference), 1.0e-6) > 2.0e-3) {
                fail("nonzero page-table row did not select the exact fragmented page sequence");
            }
        }

        args.page_table_row_count = 1U;
        HIP_CHECK(launch_selected_dense(args, d_workspace.get(),
                                        d_workspace.bytes(), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
            fail("invalid page-table row did not poison every active output row");
        }

        args.page_table_row_count = 2U;
        page_rows[fixture.pages.size()] = static_cast<std::uint32_t>(kPhysicalPages);
        HIP_CHECK(hipMemcpy(d_page_rows.get(), page_rows.data(), d_page_rows.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, d_workspace.get(),
                                        d_workspace.bytes(), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
            fail("invalid physical page did not poison every dependent output row");
        }
    }
}

} // namespace

int main() {
    try {
        host_predicate_regression();
        for (const std::uint32_t group : {16U, 32U}) {
            for (const std::uint32_t rows : {128U, 129U, 512U, 1024U, 2048U, 4096U}) {
                run_case(rows, group);
            }
        }
        std::cout << "production Bk16/Bk32 dense G16/G32 FP64 qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
