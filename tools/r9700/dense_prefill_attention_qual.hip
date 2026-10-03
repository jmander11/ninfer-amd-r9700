#include "qualification_device_guard.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
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
constexpr float kScale = 1.0F / 16.0F;
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

#define HIP_CHECK(call) do { const hipError_t e_ = (call); if (e_ != hipSuccess) \
    fail(std::string(#call) + ": " + hipGetErrorString(e_)); } while (false)

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&pointer_, (count + 32U) * sizeof(T)));
        HIP_CHECK(hipMemset(pointer_, 0xa5, (count + 32U) * sizeof(T)));
    }
    ~DeviceBuffer() { if (pointer_ != nullptr) (void)hipFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return pointer_ + 16U; }
    const T* get() const { return pointer_ + 16U; }
    std::size_t bytes() const { return count_ * sizeof(T); }
    void check_canaries() const {
        std::array<unsigned char, 16U * sizeof(T)> guard{};
        for (const T* address : {pointer_, pointer_ + 16U + count_}) {
            HIP_CHECK(hipMemcpy(guard.data(), address, guard.size(), hipMemcpyDeviceToHost));
            if (!std::all_of(guard.begin(), guard.end(), [](auto b) { return b == 0xa5; }))
                fail("dense attention buffer canary changed");
        }
    }
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

std::size_t key_offset(std::size_t physical, std::uint32_t head, std::uint32_t feature,
                       std::size_t physical_pages) {
    const std::size_t page = physical / kPage, in_page = physical % kPage;
    return in_page + kPage * (feature + kD * (page + physical_pages * head));
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

std::vector<std::uint32_t> make_pages(std::uint32_t context) {
    std::vector<std::uint32_t> pages((context + kPage - 1U) / kPage);
    const auto physical_pages = pages.size() + 5U;
    for (std::uint32_t page = 0U; page < pages.size(); ++page) {
        pages[page] = (page * (physical_pages - 1U) + 3U) % physical_pages;
    }
    return pages;
}

struct Fixture {
    std::uint32_t group;
    std::size_t physical_tokens;
    std::vector<std::uint32_t> pages;
    std::vector<std::uint8_t> keys;
    std::vector<std::uint8_t> values;
    std::vector<std::uint16_t> scales;
};

std::uint32_t fixture_word(std::size_t physical, std::uint32_t head,
                           std::uint32_t feature, std::uint32_t salt) {
    std::uint32_t word = static_cast<std::uint32_t>(physical) * 0x9e3779b9U ^
        head * 0x85ebca6bU ^ feature * 0xc2b2ae35U ^ salt;
    word ^= word >> 16U; word *= 0x7feb352dU;
    word ^= word >> 15U; word *= 0x846ca68bU;
    return word ^ (word >> 16U);
}

Fixture make_fixture(std::uint32_t group, std::uint32_t context) {
    const std::size_t physical_tokens = ((context + kPage - 1U) / kPage + 5U) * kPage;
    Fixture result{group, physical_tokens, make_pages(context),
                   std::vector<std::uint8_t>(physical_tokens * kHkv * kD),
                   std::vector<std::uint8_t>(physical_tokens * kHkv * kD / 2U),
                   std::vector<std::uint16_t>(physical_tokens * kHkv * (kD / group))};
    constexpr std::array<std::uint8_t, 10> key_words{
        0x08U, 0x18U, 0x20U, 0x28U, 0x30U, 0x88U, 0x98U, 0xa0U, 0xa8U, 0xb0U};
    constexpr std::array<std::uint16_t, 12> scale_words{
        0x2c00U, 0x3000U, 0x3400U, 0x3800U, 0x3a00U, 0x3c00U,
        0x2c01U, 0x3001U, 0x3555U, 0x37ffU, 0x3a01U, 0x3c01U};
    for (std::size_t physical = 0U; physical < physical_tokens; ++physical) {
        for (std::uint32_t head = 0U; head < kHkv; ++head) {
            for (std::uint32_t feature = 0U; feature < kD; ++feature) {
                result.keys[key_offset(physical, head, feature, physical_tokens / kPage)] =
                    key_words[fixture_word(physical, head, feature, 1U) % key_words.size()];
            }
            for (std::uint32_t feature = 0U; feature < kD; feature += 2U) {
                const int lo = static_cast<int>(fixture_word(physical, head, feature, 2U) % 16U) - 8;
                const int hi = static_cast<int>(fixture_word(physical, head, feature + 1U, 2U) % 16U) - 8;
                result.values[value_offset(physical, head, feature)] =
                    static_cast<std::uint8_t>((lo & 15) | ((hi & 15) << 4));
            }
            for (std::uint32_t feature = 0U; feature < kD; feature += group) {
                result.scales[scale_offset(physical, head, feature, group)] =
                    scale_words[fixture_word(physical, head, feature / group, 3U) % scale_words.size()];
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

// Host-fixed rows the mid-row predicate admits take the production split route; device-counted
// rows stay on the unsplit dense route, as in the leaf.
hipError_t launch_selected_dense(const kv::Fp8Int4KvAttentionArgs& args, hipStream_t stream) {
    if (kv::use_mid_rows_attention(args.query_rows, args.context,
                                   args.active_query_rows != nullptr))
        return kv::fp8_int4_kv_attention_mid_rows(args, stream);
    auto dense = args;
    dense.score_workspace = nullptr;
    return kv::fp8_int4_kv_attention_dense_prefill(dense, stream);
}

std::array<double, kD> expected_vector(const Fixture& fixture, const std::vector<hip_bfloat16>& queries,
                const std::vector<std::int32_t>& positions, std::uint32_t row,
                std::uint32_t query_head) {
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
                   oracle::decode_e4m3fn(fixture.keys[key_offset(physical, kv_head, dimension, fixture.physical_tokens / kPage)]);
        }
        weights[logical] = dot * static_cast<double>(kScale);
        maximum = std::max(maximum, weights[logical]);
    }
    double denominator = 0.0;
    for (double& weight : weights) {
        weight = std::exp(weight - maximum);
        denominator += weight;
    }
    std::array<double, kD> numerator{};
    for (std::size_t logical = 0U; logical < visible; ++logical) {
        const std::size_t physical =
            static_cast<std::size_t>(fixture.pages[logical / kPage]) * kPage + logical % kPage;
        for (std::uint32_t feature = 0U; feature < kD; ++feature) {
        const std::uint8_t packed = fixture.values[value_offset(physical, kv_head, feature)];
        const std::uint8_t nibble = (packed >> ((feature & 1U) * 4U)) & 15U;
        const int code = nibble >= 8U ? static_cast<int>(nibble) - 16 : nibble;
        const double value = static_cast<double>(code) * oracle::detail::fp16_to_fp32(
            fixture.scales[scale_offset(physical, kv_head, feature, fixture.group)]);
        numerator[feature] += weights[logical] * value;
        }
    }
    for (double& value : numerator) value /= denominator;
    return numerator;
}

double expected(const Fixture& fixture, const std::vector<hip_bfloat16>& queries,
                const std::vector<std::int32_t>& positions, std::uint32_t row,
                std::uint32_t head, std::uint32_t feature) {
    return expected_vector(fixture, queries, positions, row, head)[feature];
}

void host_predicate_regression() {
    for (const std::uint32_t group : {16U, 32U}) {
        for (const bool split : {false, true}) {
            kv::DensePrefillWmmaResources resources{};
            HIP_CHECK(kv::fp8_int4_kv_attention_dense_prefill_resources(group, split, &resources));
            std::cerr << "dense-prefill G" << group << (split ? " split" : "")
                      << " vgpr=" << resources.registers << " lds=" << resources.static_shared_bytes
                      << " scratch=" << resources.local_bytes << '\n';
            if (resources.local_bytes != 0 || resources.static_shared_bytes > 65536 ||
                resources.registers > 240)
                fail("dense prefill tile spills or exceeds the LDS or 240-VGPR budget");
        }
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
        if (kv::fp8_int4_kv_attention_dense_prefill(candidate, nullptr) != hipErrorInvalidValue)
            fail(label);
    };
    auto rejected_mid = [&](const kv::Fp8Int4KvAttentionArgs& candidate, const char* label) {
        if (kv::fp8_int4_kv_attention_mid_rows(candidate, nullptr) != hipErrorInvalidValue)
            fail(label);
    };
    auto bad = args; bad.score_workspace = reinterpret_cast<float*>(1U);
    rejected(bad, "score workspace was accepted");
    bad = args; bad.key_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    rejected(bad, "wrong K layout was accepted");
    bad = args; bad.query_rows = 8193U;
    bad.context = 8193U;
    rejected(bad, "oversized prefill was accepted");
    bad = args; bad.ancestor_masks = reinterpret_cast<const std::int32_t*>(1U);
    bad.prefix_lengths = reinterpret_cast<const std::int32_t*>(1U);
    rejected(bad, "tree metadata was accepted");
    bad = args; bad.context = 2047U;
    rejected(bad, "context shorter than query rows was accepted");

    auto mid = args;
    mid.query_rows = 19U;
    mid.score_workspace = reinterpret_cast<float*>(1U);
    bad = mid; bad.score_workspace = nullptr;
    rejected_mid(bad, "mid-row launch without partial workspace was accepted");
    bad = mid; bad.active_query_rows = reinterpret_cast<const std::int32_t*>(1U);
    rejected_mid(bad, "mid-row launch with device-selected rows was accepted");
    bad = mid; bad.ancestor_masks = reinterpret_cast<const std::int32_t*>(1U);
    rejected_mid(bad, "mid-row launch with tree metadata was accepted");
    bad = mid; bad.query_rows = 8U;
    rejected_mid(bad, "packed-decode row count took the mid-row route");
    bad = mid; bad.context = 18U;
    rejected_mid(bad, "mid-row context shorter than query rows was accepted");
    if (kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(2048U, 8U) != 0U ||
        kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(2048U, 1024U) != 0U ||
        kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(2048U, 9U) == 0U ||
        kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(2048U, 1023U) == 0U)
        fail("mid-row workspace predicate changed");
    for (std::uint32_t rows = kv::kMidRowsMinimumRows; rows <= kv::kMidRowsMaximumRows; ++rows)
        if (kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(262144U, rows) >
            std::size_t{2048U} * 24U * 258U * sizeof(float))
            fail("mid-row workspace exceeds its partial-slot bound");
}
void run_case(std::uint32_t rows, std::uint32_t group, std::uint32_t context,
              std::uint16_t uniform_scale = 0U, std::uint32_t timed_iterations = 0U) {
    Fixture fixture = make_fixture(group, context);
    if (uniform_scale != 0U) std::fill(fixture.scales.begin(), fixture.scales.end(), uniform_scale);
    const auto queries = make_queries(rows);
    auto positions = make_positions(rows);
    if (context > rows)
        for (std::uint32_t row = 0; row < rows; ++row) positions[row] = context - rows + row;
    DeviceBuffer<hip_bfloat16> d_queries(queries.size());
    DeviceBuffer<std::uint8_t> d_keys(fixture.keys.size());
    DeviceBuffer<std::uint8_t> d_values(fixture.values.size());
    DeviceBuffer<std::uint16_t> d_scales(fixture.scales.size());
    DeviceBuffer<std::uint32_t> d_pages(fixture.pages.size());
    DeviceBuffer<std::int32_t> d_positions(positions.size());
    DeviceBuffer<float> d_output(static_cast<std::size_t>(rows) * kHq * kD);
    const std::size_t partial_bytes = kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(context, rows);
    DeviceBuffer<float> d_partial(std::max<std::size_t>(partial_bytes / sizeof(float), 1U));
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
        .context=context, .physical_tokens=fixture.physical_tokens, .query_heads=kHq, .kv_heads=kHkv,
        .q_to_kv=6U, .value_group=group, .query_rows=rows,
        .row_positions=d_positions.get(), .attention_scale=kScale,
        .key_layout=kv::Fp8Int4KvPlaneLayout::TokenFastest,
        .value_layout=kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .value_scale_layout=kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .output=d_output.get()};
    if (kv::use_mid_rows_attention(rows, context, false)) args.score_workspace = d_partial.get();
    HIP_CHECK(launch_selected_dense(args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> output(static_cast<std::size_t>(rows) * kHq * kD);
    HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
    const std::array<std::uint32_t, 5> sampled_rows{
        0U, rows / 3U, std::min(rows / 2U + 15U, rows - 1U), std::min(rows - 16U, rows - 1U),
        rows - 1U};
    constexpr std::array<std::uint32_t, 4> sampled_heads{0U, 7U, 13U, 23U};
    for (const auto row : sampled_rows) for (const auto head : sampled_heads) {
        const auto references = expected_vector(fixture, queries, positions, row, head);
        for (std::uint32_t feature = 0U; feature < kD; ++feature) {
            const double reference = references[feature];
            const float actual = output[(static_cast<std::size_t>(row) * kHq + head) * kD + feature];
            const double error = std::abs(static_cast<double>(actual) - reference);
            if (!std::isfinite(reference) || !std::isfinite(actual) ||
                !std::isfinite(error) ||
                (error > 2.0e-3 &&
                 error / std::max(std::abs(reference), 1.0e-6) > 2.0e-3)) {
                fail("dense attention differs from independent FP64 oracle");
            }
        }
    }
    ninfer::r9700_qualification::device_guard().check("after-oracle-before-timing");
    if (timed_iterations != 0U) {
        // Unprofiled event timing after the oracle pass: median of individually timed launches.
        for (unsigned warm = 0; warm < 3; ++warm) HIP_CHECK(launch_selected_dense(args, nullptr));
        std::vector<float> samples;
        Event start, stop;
        for (std::uint32_t iteration = 0U; iteration < timed_iterations; ++iteration) {
            HIP_CHECK(hipEventRecord(start.get(), nullptr));
            HIP_CHECK(launch_selected_dense(args, nullptr));
            HIP_CHECK(hipEventRecord(stop.get(), nullptr));
            HIP_CHECK(hipEventSynchronize(stop.get()));
            float milliseconds = 0.0F;
            HIP_CHECK(hipEventElapsedTime(&milliseconds, start.get(), stop.get()));
            samples.push_back(milliseconds);
        }
        ninfer::r9700_qualification::device_guard().check("after-timing");
        std::sort(samples.begin(), samples.end());
        double visible = 0.0;
        for (const auto position : positions) visible += static_cast<double>(position) + 1.0;
        const double median = samples[samples.size() / 2U];
        std::cout << "timing rows=" << rows << " context=" << context << " group=" << group
                  << " median_ms=" << median << " min_ms=" << samples.front()
                  << " useful_tflops=" << 4.0 * kHq * kD * visible / (median * 1.0e9) << "\n";
    }
    if (rows == 19U && context == 22281U) {
        // Mid-row poisoning is per chunk: an invalid position poisons exactly its row, and a
        // missing physical page at key 22272 poisons exactly the rows whose context passes it.
        auto malformed_positions = positions;
        malformed_positions[5] = -1;
        HIP_CHECK(hipMemcpy(d_positions.get(), malformed_positions.data(), d_positions.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> poisoned(output.size());
        HIP_CHECK(hipMemcpy(poisoned.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto begin = poisoned.begin() + static_cast<std::ptrdiff_t>(row * kHq * kD);
            const bool all_nan = std::all_of(begin, begin + kHq * kD,
                                             [](float value) { return std::isnan(value); });
            const bool same = std::equal(begin, begin + kHq * kD,
                                         output.begin() + static_cast<std::ptrdiff_t>(row * kHq * kD));
            if (row == 5U ? !all_nan : !same)
                fail("mid-row invalid position did not poison exactly its row");
        }
        HIP_CHECK(hipMemcpy(d_positions.get(), positions.data(), d_positions.bytes(),
                            hipMemcpyHostToDevice));
        auto faulted_pages = fixture.pages;
        faulted_pages[22272U / kPage] = static_cast<std::uint32_t>(fixture.physical_tokens / kPage);
        HIP_CHECK(hipMemcpy(d_pages.get(), faulted_pages.data(), d_pages.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(poisoned.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto begin = poisoned.begin() + static_cast<std::ptrdiff_t>(row * kHq * kD);
            const bool past_fault = static_cast<std::uint32_t>(positions[row]) + 1U > 22272U;
            const bool all_nan = std::all_of(begin, begin + kHq * kD,
                                             [](float value) { return std::isnan(value); });
            const bool same = std::equal(begin, begin + kHq * kD,
                                         output.begin() + static_cast<std::ptrdiff_t>(row * kHq * kD));
            if (past_fault ? !all_nan : !same)
                fail("mid-row page fault did not poison exactly the rows past it");
        }
        HIP_CHECK(hipMemcpy(d_pages.get(), fixture.pages.data(), d_pages.bytes(),
                            hipMemcpyHostToDevice));
    }
    if (rows == 1537U) {
        hipStream_t stream{};
        hipGraph_t graph{};
        hipGraphExec_t executable{};
        HIP_CHECK(hipStreamCreate(&stream));
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        HIP_CHECK(launch_selected_dense(args, stream));
        HIP_CHECK(hipStreamEndCapture(stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (unsigned replay = 0; replay < 2; ++replay) {
            HIP_CHECK(hipMemsetAsync(d_output.get(), 0xff, d_output.bytes(), stream));
            HIP_CHECK(hipGraphLaunch(executable, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            std::vector<float> replay_output(output.size());
            HIP_CHECK(hipMemcpy(replay_output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
            if (replay_output != output) fail("panel Device Graph replay differs from eager output");
        }
        HIP_CHECK(hipGraphExecDestroy(executable));
        HIP_CHECK(hipGraphDestroy(graph));
        HIP_CHECK(hipStreamDestroy(stream));
    }
    if (rows == 129U || rows == 1537U) {
        // Device-counted rows take the unsplit dense route, so its full-count output is the
        // exact baseline for the active-row prefix.
        DeviceBuffer<std::int32_t> d_active(1U);
        std::int32_t active = static_cast<std::int32_t>(rows);
        HIP_CHECK(hipMemcpy(d_active.get(), &active, sizeof(active), hipMemcpyHostToDevice));
        args.active_query_rows = d_active.get();
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> baseline_output(output.size());
        HIP_CHECK(hipMemcpy(baseline_output.data(), d_output.get(), d_output.bytes(),
                            hipMemcpyDeviceToHost));
        active = rows == 129U ? 127 : static_cast<std::int32_t>(rows / 2U + 3U);
        HIP_CHECK(hipMemcpy(d_active.get(), &active, sizeof(active), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_output.get(), 0xff, d_output.bytes()));
        HIP_CHECK(launch_selected_dense(args, nullptr));
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
        HIP_CHECK(launch_selected_dense(inactive_args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(),
                         [](float value) { return value == 0.0F && !std::signbit(value); })) {
            fail("zero active-row state touched an inactive input or did not write positive zero");
        }
        active = -1;
        HIP_CHECK(hipMemcpy(d_active.get(), &active, sizeof(active), hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
            fail("invalid device active-row status did not poison every row");
        }
        active = static_cast<std::int32_t>(rows + 1U);
        HIP_CHECK(hipMemcpy(d_active.get(), &active, sizeof(active), hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); }))
            fail("active count above whole-call rows did not poison every panel");

        args.active_query_rows = nullptr;
        auto malformed_positions = positions;
        malformed_positions[5] = -1;
        HIP_CHECK(hipMemcpy(d_positions.get(), malformed_positions.data(), d_positions.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, nullptr));
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
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        for (const auto row : {0U, 64U, 128U}) {
            const double reference = expected(fixture, queries, positions, row, 23U, 255U);
            const float actual = output[(static_cast<std::size_t>(row) * kHq + 23U) * kD + 255U];
            const double error = std::abs(static_cast<double>(actual) - reference);
            if (!std::isfinite(actual) || !std::isfinite(reference) || !std::isfinite(error) ||
                (error > 2.0e-3 &&
                 error / std::max(std::abs(reference), 1.0e-6) > 2.0e-3)) {
                fail("nonzero page-table row did not select the exact fragmented page sequence");
            }
        }

        args.page_table_row_count = 1U;
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
            fail("invalid page-table row did not poison every active output row");
        }

        args.page_table_row_count = 2U;
        page_rows[fixture.pages.size()] = static_cast<std::uint32_t>(fixture.physical_tokens / kPage);
        HIP_CHECK(hipMemcpy(d_page_rows.get(), page_rows.data(), d_page_rows.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(launch_selected_dense(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
            fail("invalid physical page did not poison every dependent output row");
        }
    }
    d_output.check_canaries();
    d_queries.check_canaries();
    d_positions.check_canaries();
    d_keys.check_canaries();
    d_values.check_canaries();
    d_scales.check_canaries();
    std::cout << "rows=" << rows << " context=" << context << " group=" << group
              << " oracle=FP64 complete-sampled-vectors PASS\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool timed = argc == 6 && std::string(argv[1]) == "--time";
        if (argc != 1 && argc != 4 && !timed)
            fail("usage: dense_prefill_attention_qual [rows context value-group | --time rows context value-group iterations]");
        std::uint32_t cli_rows = 0, cli_context = 0, cli_group = 0, iterations = 0;
        if (argc != 1) {
            const int first = timed ? 2 : 1;
            cli_rows = static_cast<std::uint32_t>(std::stoul(argv[first]));
            cli_context = static_cast<std::uint32_t>(std::stoul(argv[first + 1]));
            cli_group = static_cast<std::uint32_t>(std::stoul(argv[first + 2]));
            if (timed) iterations = static_cast<std::uint32_t>(std::stoul(argv[5]));
            if ((!kv::use_dense_prefill_attention(cli_rows, cli_context) &&
                 !kv::use_mid_rows_attention(cli_rows, cli_context, false)) ||
                (cli_group != 16U && cli_group != 32U) || (timed && iterations == 0U))
                fail("invalid dense attention qualification/timing shape");
        }
        host_predicate_regression();
        ninfer::r9700_qualification::device_guard().select();
        std::cerr << "dense-prefill key-block=32 production\n";
        if (argc != 1) {
            run_case(cli_rows, cli_group, cli_context, 0U, iterations);
            return 0;
        }
        for (const std::uint32_t group : {16U, 32U}) {
            for (const std::uint32_t rows : {128U, 129U, 512U, 1024U, 2048U, 4096U}) {
                run_case(rows, group, rows);
            }
            run_case(1024U, group, 8192U);
            run_case(1537U, group, 8192U);
            run_case(8192U, group, 8192U);
            run_case(1025U, group, 32768U);
            run_case(8192U, group, 32768U);
            run_case(128U, group, 262144U);
            // Key-block and page boundaries of the fused 32-key schedule.
            for (const std::uint32_t context : {2047U, 2048U, 2049U, 4095U, 4096U, 4097U,
                                                12287U, 12288U, 12289U, 32767U, 32769U})
                run_case(129U, group, context);
            run_case(1537U, group, 12288U);
            run_case(1537U, group, 32768U);
            run_case(2048U, group, 65536U);
            run_case(128U, group, 131072U);
            run_case(512U, group, 131071U);
            run_case(512U, group, 131072U);
            run_case(512U, group, 131073U);
            // The private FP16 PV operands use V/8: cover both finite scale extremes.
            run_case(128U, group, 32768U, 0x7bffU);
            run_case(128U, group, 32768U, 0x0001U);
            // Mid-row split route: row tiles 1..4, fresh and appended contexts, chunk and page
            // boundaries, and a final chunk that starts inside the query rows (rows whose
            // context ends before it see only masked keys there).
            for (const std::uint32_t rows : {9U, 16U, 19U, 31U, 32U, 33U, 64U, 100U, 127U}) {
                run_case(rows, group, rows);
                run_case(rows, group, 2049U);
                run_case(rows, group, 22229U);
            }
            for (const std::uint32_t context : {255U, 256U, 257U, 4095U, 4097U, 32768U, 131072U})
                run_case(19U, group, context);
            run_case(127U, group, 262144U);
            run_case(19U, group, 22281U);   // poisoning: rows straddle the page at key 22272
            run_case(19U, group, 1793U);    // last chunk starts at 1792: 18 rows fully masked there
            run_case(19U, group, 20817U);
            run_case(64U, group, 2078U);
            run_case(127U, group, 797U);
            run_case(127U, group, 32768U, 0x7bffU);
            run_case(19U, group, 32768U, 0x0001U);
            // Split route above 127 rows (appended tool results and turns): wave-model chunking
            // of 5..31 row tiles, fresh and long contexts, and the dense fallback at one chunk.
            for (const std::uint32_t rows : {128U, 129U, 192U, 255U, 384U, 767U, 768U}) {
                run_case(rows, group, rows);
                run_case(rows, group, 22229U);
                run_case(rows, group, 131072U);
            }
            run_case(192U, group, 32768U, 0x7bffU);
            run_case(384U, group, 262144U);
        }
        std::cout << "production dense and mid-row G16/G32 FP64 qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
