#include "core/device.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace kv = ninfer::ops::r9700::kv;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;
using Json = nlohmann::json;

constexpr std::uint32_t kHeadDim = 256U;
constexpr std::uint32_t kQueryHeads = 24U;
constexpr std::uint32_t kKvHeads = 4U;
constexpr std::uint32_t kQueriesPerKv = 6U;
constexpr std::uint32_t kValueGroup = 16U;
constexpr std::uint32_t kGroups = kHeadDim / kValueGroup;
constexpr std::uint32_t kPageSize = 64U;
constexpr std::uint32_t kW1 = 1U;
constexpr std::uint32_t kW5 = 5U;
constexpr std::size_t kW1Context = 130U;
constexpr std::size_t kW5Context = 134U;
constexpr std::size_t kPhysicalPages = 4U;
constexpr std::size_t kPhysicalTokens = kPhysicalPages * kPageSize;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr std::uint32_t kGuard = 0x7fc01234U;
constexpr int kTrials = 7;
constexpr int kTimingIterations = 50;
static_assert(kQueryHeads * kW1Context * sizeof(float) == 12480U);
static_assert(kQueryHeads * kW5Context * sizeof(float) == 12864U);

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        fail(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

float represented_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t count) : count_(count) {
        hip_check(hipMalloc(&data_, count * sizeof(T)), "hipMalloc");
    }
    ~DeviceArray() { if (data_) (void)hipFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* get() const { return static_cast<T*>(data_); }
    std::size_t size() const { return count_; }
private:
    void* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
void upload(DeviceArray<T>& destination, const std::vector<T>& source) {
    if (destination.size() != source.size()) fail("upload extent mismatch");
    hip_check(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice), "hipMemcpy H2D");
}

template <typename T>
std::vector<T> download(const DeviceArray<T>& source) {
    std::vector<T> result(source.size());
    hip_check(hipMemcpy(result.data(), source.get(), result.size() * sizeof(T),
                        hipMemcpyDeviceToHost), "hipMemcpy D2H");
    return result;
}

std::size_t physical_token(std::size_t logical,
                           const std::array<std::uint32_t, 3>& pages) {
    return static_cast<std::size_t>(pages.at(logical / kPageSize)) * kPageSize +
           logical % kPageSize;
}

std::size_t key_offset(std::size_t token, std::uint32_t head, std::uint32_t feature) {
    return token % kPageSize + kPageSize *
        (static_cast<std::size_t>(feature) + kHeadDim * (token / kPageSize +
                                                         kPhysicalPages * head));
}

std::size_t value_offset(std::size_t token, std::uint32_t head, std::uint32_t feature) {
    return feature / 2U + (kHeadDim / 2U) *
        (token % kPageSize + kPageSize * (head + kKvHeads * (token / kPageSize)));
}

std::size_t scale_offset(std::size_t token, std::uint32_t head, std::uint32_t feature) {
    return feature / kValueGroup + kGroups *
        (token % kPageSize + kPageSize * (head + kKvHeads * (token / kPageSize)));
}

struct Fixture {
    std::array<std::uint32_t, 3> pages{2U, 0U, 3U};
    std::vector<float> queries;
    oracle::Fp8Keys logical_keys;
    oracle::Int4Values logical_values;
    std::vector<std::uint8_t> physical_keys;
    std::vector<std::uint8_t> physical_values;
    std::vector<std::uint16_t> physical_scales;
};

Fixture make_fixture() {
    std::vector<float> query(kW5 * kQueryHeads * kHeadDim);
    for (std::uint32_t row = 0; row < kW5; ++row) {
        for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
            for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
                const int value = static_cast<int>((row * 37U + head * 19U +
                                                    feature * 13U) % 113U) - 56;
                query[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim +
                      feature] = represented_bf16(static_cast<float>(value) / 32.0F);
            }
        }
    }
    std::vector<float> keys(kW5Context * kKvHeads * kHeadDim);
    std::vector<float> values(keys.size());
    for (std::size_t token = 0; token < kW5Context; ++token) {
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
                const std::size_t index = (token * kKvHeads + head) * kHeadDim + feature;
                const int key = static_cast<int>((token * 29U + head * 17U +
                                                  feature * 7U) % 101U) - 50;
                const int value = static_cast<int>((token * 11U + head * 31U +
                                                    feature * 23U) % 127U) - 63;
                keys[index] = represented_bf16(static_cast<float>(key) / 16.0F);
                values[index] = represented_bf16(static_cast<float>(value) / 40.0F);
            }
        }
    }
    Fixture fixture{
        .queries = std::move(query),
        .logical_keys = oracle::encode_keys(keys, kW5Context * kKvHeads, kHeadDim),
        .logical_values = oracle::encode_values(values, kW5Context * kKvHeads,
                                                 kHeadDim, kValueGroup),
        .physical_keys = std::vector<std::uint8_t>(kPhysicalTokens * kKvHeads * kHeadDim),
        .physical_values = std::vector<std::uint8_t>(kPhysicalTokens * kKvHeads * kHeadDim / 2U),
        .physical_scales = std::vector<std::uint16_t>(kPhysicalTokens * kKvHeads * kGroups)};
    for (std::size_t logical = 0; logical < kW5Context; ++logical) {
        const std::size_t physical = physical_token(logical, fixture.pages);
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            const std::size_t represented = logical * kKvHeads + head;
            for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
                fixture.physical_keys[key_offset(physical, head, feature)] =
                    fixture.logical_keys.codes[represented * kHeadDim + feature];
                const std::uint8_t packed = fixture.logical_values.packed_codes[
                    represented * (kHeadDim / 2U) + feature / 2U];
                std::uint8_t& destination =
                    fixture.physical_values[value_offset(physical, head, feature)];
                if ((feature & 1U) == 0U) destination = packed;
                fixture.physical_scales[scale_offset(physical, head, feature)] =
                    fixture.logical_values.fp16_scales[represented * kGroups +
                                                        feature / kValueGroup];
            }
        }
    }
    return fixture;
}

std::vector<double> attention_oracle(const Fixture& fixture,
                                     const std::vector<float>& represented_queries,
                                     std::uint32_t rows,
                                     const std::vector<std::int32_t>& positions) {
    if (represented_queries.size() < static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim) {
        fail("attention oracle query extent is too small");
    }
    std::vector<double> result(static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t visible = static_cast<std::size_t>(positions.at(row)) + 1U;
        for (std::uint32_t qh = 0; qh < kQueryHeads; ++qh) {
            const std::uint32_t kvh = qh / kQueriesPerKv;
            oracle::Fp8Keys keys{visible, kHeadDim,
                                 std::vector<std::uint8_t>(visible * kHeadDim)};
            oracle::Int4Values values{visible, kHeadDim, kValueGroup,
                                      std::vector<std::uint8_t>(visible * kHeadDim / 2U),
                                      std::vector<std::uint16_t>(visible * kGroups)};
            for (std::size_t token = 0; token < visible; ++token) {
                const std::size_t represented = token * kKvHeads + kvh;
                std::copy_n(fixture.logical_keys.codes.begin() +
                                static_cast<std::ptrdiff_t>(represented * kHeadDim),
                            kHeadDim, keys.codes.begin() +
                                          static_cast<std::ptrdiff_t>(token * kHeadDim));
                std::copy_n(fixture.logical_values.packed_codes.begin() +
                                static_cast<std::ptrdiff_t>(represented * kHeadDim / 2U),
                            kHeadDim / 2U, values.packed_codes.begin() +
                                                  static_cast<std::ptrdiff_t>(token * kHeadDim / 2U));
                std::copy_n(fixture.logical_values.fp16_scales.begin() +
                                static_cast<std::ptrdiff_t>(represented * kGroups),
                            kGroups, values.fp16_scales.begin() +
                                         static_cast<std::ptrdiff_t>(token * kGroups));
            }
            const std::size_t base = (static_cast<std::size_t>(row) * kQueryHeads + qh) * kHeadDim;
            const std::vector<double> head = oracle::attention_fp64(
                std::span<const float>(represented_queries.data() + base, kHeadDim), keys, values,
                static_cast<double>(kAttentionScale));
            std::copy(head.begin(), head.end(), result.begin() + static_cast<std::ptrdiff_t>(base));
        }
    }
    return result;
}

struct Error {
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    std::size_t maximum_absolute_index = 0U;
    double maximum_absolute_actual = 0.0;
    double maximum_absolute_expected = 0.0;
};

Error require_oracle(const std::vector<float>& actual, const std::vector<double>& expected,
                     double limit, const char* label) {
    if (actual.size() != expected.size()) fail(std::string(label) + " extent mismatch");
    Error error;
    std::size_t violating_index = actual.size();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) fail(std::string(label) + " produced nonfinite output");
        const double absolute = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double relative = absolute / std::max(std::abs(expected[index]), 1.0e-6);
        if (absolute > error.maximum_absolute) {
            error.maximum_absolute = absolute;
            error.maximum_absolute_index = index;
            error.maximum_absolute_actual = actual[index];
            error.maximum_absolute_expected = expected[index];
        }
        error.maximum_relative = std::max(error.maximum_relative, relative);
        if (violating_index == actual.size() && absolute > limit && relative > limit) {
            violating_index = index;
        }
    }
    if (violating_index != actual.size()) {
        std::ostringstream message;
        message << std::setprecision(std::numeric_limits<double>::max_digits10) << label
                << " exceeds FP64 oracle criterion: max_abs=" << error.maximum_absolute
                << " max_rel=" << error.maximum_relative
                << " first_violation=" << violating_index
                << " worst_index=" << error.maximum_absolute_index
                << " actual=" << error.maximum_absolute_actual
                << " expected=" << error.maximum_absolute_expected;
        fail(message.str());
    }
    return error;
}

Error require_profile_bound(const std::vector<float>& actual,
                            const std::vector<double>& public_expected,
                            const std::vector<double>& profile_expected,
                            double implementation_limit, const char* label) {
    if (actual.size() != public_expected.size() || actual.size() != profile_expected.size()) {
        fail(std::string(label) + " extent mismatch");
    }
    Error error;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double absolute =
            std::abs(static_cast<double>(actual[index]) - public_expected[index]);
        const double relative = absolute / std::max(std::abs(public_expected[index]), 1.0e-6);
        if (absolute > error.maximum_absolute) {
            error.maximum_absolute = absolute;
            error.maximum_absolute_index = index;
            error.maximum_absolute_actual = actual[index];
            error.maximum_absolute_expected = public_expected[index];
        }
        error.maximum_relative = std::max(error.maximum_relative, relative);
        const double represented_profile_delta =
            std::abs(profile_expected[index] - public_expected[index]);
        const double implementation_allowance =
            implementation_limit * std::max(1.0, std::abs(profile_expected[index]));
        if (absolute > represented_profile_delta + implementation_allowance) {
            std::ostringstream message;
            message << std::setprecision(std::numeric_limits<double>::max_digits10) << label
                    << " exceeds BF16-public oracle plus explicit FP8-Q profile bound at index="
                    << index << " actual=" << actual[index]
                    << " public_expected=" << public_expected[index]
                    << " fp8_q_expected=" << profile_expected[index]
                    << " public_abs=" << absolute
                    << " profile_delta=" << represented_profile_delta
                    << " implementation_allowance=" << implementation_allowance;
            fail(message.str());
        }
    }
    return error;
}

void fill_guarded(DeviceArray<float>& buffer, std::size_t payload) {
    std::vector<std::uint32_t> words(payload + 2U, kGuard);
    hip_check(hipMemcpy(buffer.get(), words.data(), words.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice), "initialize guarded output");
}

std::vector<float> read_guarded(const DeviceArray<float>& buffer, std::size_t payload,
                                const char* label) {
    const std::vector<float> all = download(buffer);
    if (std::bit_cast<std::uint32_t>(all.front()) != kGuard ||
        std::bit_cast<std::uint32_t>(all.back()) != kGuard) {
        fail(std::string(label) + " overwrote an output canary");
    }
    std::vector<float> result(all.begin() + 1, all.begin() + 1 +
                                                   static_cast<std::ptrdiff_t>(payload));
    if (std::any_of(result.begin(), result.end(), [](float value) {
            return std::bit_cast<std::uint32_t>(value) == kGuard;
        })) {
        fail(std::string(label) + " did not write its complete output");
    }
    return result;
}

double time_ms(hipStream_t stream, const auto& launch) {
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipEventCreate(&begin), "hipEventCreate begin");
    hip_check(hipEventCreate(&end), "hipEventCreate end");
    for (int index = 0; index < 3; ++index) hip_check(launch(), "timing warmup");
    hip_check(hipEventRecord(begin, stream), "hipEventRecord begin");
    for (int index = 0; index < kTimingIterations; ++index) hip_check(launch(), "timing launch");
    hip_check(hipEventRecord(end, stream), "hipEventRecord end");
    hip_check(hipEventSynchronize(end), "hipEventSynchronize");
    float elapsed = 0.0F;
    hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime");
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    return static_cast<double>(elapsed) / kTimingIterations;
}

double median(std::array<double, kTrials> values) {
    std::sort(values.begin(), values.end());
    return values[kTrials / 2];
}

} // namespace

int main() {
    try {
        if (!kv::use_fp8_qk_wmma(kW1, kW1Context) ||
            kv::use_fp8_qk_wmma(kW5, kW5Context)) {
            fail("production attention classifier does not retain W1-WMMA/W5-fused selection");
        }
        ninfer::DeviceContext device(0);
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            fail("physical device is not wave32 gfx1201");
        }
        if (std::string(device.props.name) != "AMD Radeon AI PRO R9700") {
            fail("physical device is not AMD Radeon AI PRO R9700");
        }
        int selected_device = -1;
        hip_check(hipGetDevice(&selected_device), "hipGetDevice");
        std::array<char, 32> pci{};
        hip_check(hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), selected_device),
                  "hipDeviceGetPCIBusId");
        if (selected_device != 0 || std::string(pci.data()) != "0000:13:00.0") {
            fail("physical device is not device0 at PCI 0000:13:00.0");
        }
        const Fixture fixture = make_fixture();
        const std::vector<hip_bfloat16> queries = [&] {
            std::vector<hip_bfloat16> result;
            result.reserve(fixture.queries.size());
            for (float value : fixture.queries) result.emplace_back(value);
            return result;
        }();
        const std::vector<std::uint32_t> pages(fixture.pages.begin(), fixture.pages.end());
        const std::vector<std::int32_t> w1_positions{129};
        const std::vector<std::int32_t> w5_positions{129, 130, 131, 132, 133};

        DeviceArray<hip_bfloat16> d_query(queries.size()); upload(d_query, queries);
        DeviceArray<std::uint8_t> d_keys(fixture.physical_keys.size()); upload(d_keys, fixture.physical_keys);
        DeviceArray<std::uint8_t> d_values(fixture.physical_values.size()); upload(d_values, fixture.physical_values);
        DeviceArray<std::uint16_t> d_scales(fixture.physical_scales.size()); upload(d_scales, fixture.physical_scales);
        DeviceArray<std::uint32_t> d_pages(pages.size()); upload(d_pages, pages);
        DeviceArray<std::int32_t> d_w1_positions(w1_positions.size()); upload(d_w1_positions, w1_positions);
        DeviceArray<std::int32_t> d_w5_positions(w5_positions.size()); upload(d_w5_positions, w5_positions);
        const std::size_t row_elements = kQueryHeads * kHeadDim;
        DeviceArray<float> d_w1_output(row_elements + 2U);
        DeviceArray<float> d_w1_fused_output(row_elements + 2U);
        DeviceArray<float> d_w5_wmma_output(kW5 * row_elements + 2U);
        DeviceArray<float> d_w5_fused_output(kW5 * row_elements + 2U);
        DeviceArray<float> d_w1_scores(kQueryHeads * kW1Context + 2U);
        DeviceArray<float> d_w5_scores(kQueryHeads * kW5Context + 2U);

        const auto args = [&](std::uint32_t rows, std::size_t context,
                              const std::int32_t* positions, float* scores, float* output) {
            return kv::Fp8Int4KvAttentionArgs{
                .query = d_query.get(), .key_codes = d_keys.get(), .value_codes = d_values.get(),
                .value_scale_bits = d_scales.get(), .page_table = d_pages.get(),
                .page_table_pages = static_cast<std::uint32_t>(pages.size()), .context = context,
                .physical_tokens = kPhysicalTokens, .query_heads = kQueryHeads,
                .kv_heads = kKvHeads, .q_to_kv = kQueriesPerKv, .value_group = kValueGroup,
                .query_rows = rows, .row_positions = positions, .attention_scale = kAttentionScale,
                .key_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest,
                .value_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                .score_workspace = scores, .output = output};
        };
        auto w1 = args(kW1, kW1Context, d_w1_positions.get(), d_w1_scores.get() + 1,
                       d_w1_output.get() + 1);
        auto w1_fused = w1;
        w1_fused.score_workspace = nullptr;
        w1_fused.output = d_w1_fused_output.get() + 1;
        auto w5_wmma = args(kW5, kW5Context, d_w5_positions.get(), d_w5_scores.get() + 1,
                            d_w5_wmma_output.get() + 1);
        auto w5_fused = w5_wmma;
        w5_fused.score_workspace = nullptr;
        w5_fused.output = d_w5_fused_output.get() + 1;

        auto missing_workspace = w5_wmma;
        missing_workspace.score_workspace = nullptr;
        if (kv::fp8_int4_kv_attention_wmma(missing_workspace, device.stream) !=
            hipErrorInvalidValue) fail("WMMA accepted a missing exact score workspace");
        auto malformed = w5_fused;
        malformed.query_rows = 0U;
        if (kv::fp8_int4_kv_attention_fused(malformed, device.stream) != hipErrorInvalidValue) {
            fail("fused route accepted zero rows");
        }

        fill_guarded(d_w1_output, row_elements);
        fill_guarded(d_w1_fused_output, row_elements);
        fill_guarded(d_w5_wmma_output, kW5 * row_elements);
        fill_guarded(d_w5_fused_output, kW5 * row_elements);
        fill_guarded(d_w1_scores, kQueryHeads * kW1Context);
        fill_guarded(d_w5_scores, kQueryHeads * kW5Context);
        hip_check(kv::fp8_int4_kv_attention_wmma(w1, device.stream), "W1 WMMA");
        hip_check(kv::fp8_int4_kv_attention_fused(w1_fused, device.stream), "W1 fused");
        hip_check(kv::fp8_int4_kv_attention_wmma(w5_wmma, device.stream), "W5 serial WMMA");
        hip_check(kv::fp8_int4_kv_attention_fused(w5_fused, device.stream), "W5 fused");
        device.synchronize();
        const std::vector<float> actual_w1 = read_guarded(d_w1_output, row_elements, "W1 WMMA");
        const std::vector<float> actual_w1_fused =
            read_guarded(d_w1_fused_output, row_elements, "W1 fused");
        const std::vector<float> actual_w5_wmma =
            read_guarded(d_w5_wmma_output, kW5 * row_elements, "W5 serial WMMA");
        const std::vector<float> actual_w5_fused =
            read_guarded(d_w5_fused_output, kW5 * row_elements, "W5 fused");
        (void)read_guarded(d_w1_scores, kQueryHeads * kW1Context, "W1 score workspace");
        (void)read_guarded(d_w5_scores, kQueryHeads * kW5Context, "W5 score workspace");

        if (!std::equal(actual_w1.begin(), actual_w1.end(), actual_w5_wmma.begin(),
                        [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("W1 and W5 serial-WMMA selected rows are not bit-exact");
        }
        if (!std::equal(actual_w1_fused.begin(), actual_w1_fused.end(), actual_w5_fused.begin(),
                        [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("W1 and W5 fused selected rows are not bit-exact");
        }
        std::size_t fused_mismatches = 0U;
        double fused_pair_maximum_absolute = 0.0;
        for (std::size_t index = 0; index < row_elements; ++index) {
            fused_mismatches += std::bit_cast<std::uint32_t>(actual_w1[index]) !=
                                std::bit_cast<std::uint32_t>(actual_w5_fused[index]);
            fused_pair_maximum_absolute = std::max(
                fused_pair_maximum_absolute,
                std::abs(static_cast<double>(actual_w1[index]) - actual_w5_fused[index]));
        }
        if (fused_mismatches == 0U) fail("fixture did not expose the fused/WMMA arithmetic split");

        std::vector<float> fp8_profile_queries(fixture.queries.size());
        std::transform(fixture.queries.begin(), fixture.queries.end(),
                       fp8_profile_queries.begin(), [](float value) {
                           return static_cast<float>(
                               oracle::decode_e4m3fn(oracle::encode_e4m3fn(value)));
                       });
        const std::vector<double> oracle_w1 =
            attention_oracle(fixture, fixture.queries, kW1, w1_positions);
        const std::vector<double> oracle_w5 =
            attention_oracle(fixture, fixture.queries, kW5, w5_positions);
        const std::vector<double> fp8_profile_oracle_w1 =
            attention_oracle(fixture, fp8_profile_queries, kW1, w1_positions);
        const std::vector<double> fp8_profile_oracle_w5 =
            attention_oracle(fixture, fp8_profile_queries, kW5, w5_positions);
        constexpr double kImplementationLimit = 2.0e-4;
        const Error w1_profile_error = require_oracle(actual_w1, fp8_profile_oracle_w1,
                                                       kImplementationLimit,
                                                       "W1 WMMA FP8-Q profile");
        const Error w1_error = require_profile_bound(actual_w1, oracle_w1,
                                                      fp8_profile_oracle_w1,
                                                      kImplementationLimit, "W1 WMMA");
        const Error w1_fused_error = require_oracle(actual_w1_fused, oracle_w1,
                                                     kImplementationLimit,
                                                     "W1 fused");
        const Error w5_wmma_profile_error = require_oracle(
            actual_w5_wmma, fp8_profile_oracle_w5, kImplementationLimit,
            "W5 serial WMMA FP8-Q profile");
        const Error w5_wmma_error = require_profile_bound(
            actual_w5_wmma, oracle_w5, fp8_profile_oracle_w5, kImplementationLimit,
            "W5 serial WMMA");
        const Error w5_fused_error = require_oracle(actual_w5_fused, oracle_w5,
                                                     kImplementationLimit,
                                                     "W5 fused");

        const auto launch_wmma = [&] {
            return kv::fp8_int4_kv_attention_wmma(w5_wmma, device.stream);
        };
        const auto launch_fused = [&] {
            return kv::fp8_int4_kv_attention_fused(w5_fused, device.stream);
        };
        std::array<double, kTrials> wmma_samples{};
        std::array<double, kTrials> fused_samples{};
        std::array<double, kTrials> w1_wmma_samples{};
        std::array<double, kTrials> w1_fused_samples{};
        for (int trial = 0; trial < kTrials; ++trial) {
            if ((trial & 1) == 0) {
                w1_wmma_samples[trial] = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_wmma(w1, device.stream);
                });
                w1_fused_samples[trial] = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_fused(w1_fused, device.stream);
                });
                wmma_samples[trial] = time_ms(device.stream, launch_wmma);
                fused_samples[trial] = time_ms(device.stream, launch_fused);
            } else {
                fused_samples[trial] = time_ms(device.stream, launch_fused);
                wmma_samples[trial] = time_ms(device.stream, launch_wmma);
                w1_fused_samples[trial] = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_fused(w1_fused, device.stream);
                });
                w1_wmma_samples[trial] = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_wmma(w1, device.stream);
                });
            }
        }
        const Json report{
            {"schema_version", 1},
            {"qualification", "dflash_w1_w5_attention_route_discriminator"},
            {"hardware", {{"device", 0}, {"name", device.props.name},
                           {"architecture", "gfx1201"}, {"wave_size", 32},
                           {"pci_bus_id", pci.data()}}},
            {"geometry", {{"query_heads", 24}, {"kv_heads", 4}, {"head_dim", 256},
                          {"value_group", 16}, {"w1_context", 130}, {"w5_context", 134},
                          {"w1_score_workspace_bytes", 12480},
                          {"w5_serial_score_workspace_bytes", 12864}}},
            {"w1_w5_wmma_row0_bit_exact", true},
            {"w1_w5_fused_row0_bit_exact", true},
            {"production_selected_w1_w5_row0_bit_exact", false},
            {"production_selected_w1_w5_row0_mismatches", fused_mismatches},
            {"production_selected_w1_w5_row0_max_abs", fused_pair_maximum_absolute},
            {"oracle", {{"w1_wmma_max_abs", w1_error.maximum_absolute},
                        {"w1_wmma_max_rel", w1_error.maximum_relative},
                        {"w1_wmma_fp8_q_profile_max_abs", w1_profile_error.maximum_absolute},
                        {"w1_wmma_fp8_q_profile_max_rel", w1_profile_error.maximum_relative},
                        {"w1_fused_max_abs", w1_fused_error.maximum_absolute},
                        {"w1_fused_max_rel", w1_fused_error.maximum_relative},
                        {"w5_wmma_max_abs", w5_wmma_error.maximum_absolute},
                        {"w5_wmma_max_rel", w5_wmma_error.maximum_relative},
                        {"w5_wmma_fp8_q_profile_max_abs",
                         w5_wmma_profile_error.maximum_absolute},
                        {"w5_wmma_fp8_q_profile_max_rel",
                         w5_wmma_profile_error.maximum_relative},
                        {"w5_fused_max_abs", w5_fused_error.maximum_absolute},
                        {"w5_fused_max_rel", w5_fused_error.maximum_relative},
                        {"wmma_implementation_limit", kImplementationLimit},
                        {"wmma_public_criterion",
                         "pointwise_abs<=abs(fp8_q_oracle-bf16_oracle)+"
                         "2e-4*max(1,abs(fp8_q_oracle))"}}},
            {"timing", {{"evidence_eligible", false}, {"diagnostic_only", true},
                        {"iterations", kTimingIterations}, {"trials", kTrials},
                        {"w1_wmma_median_ms", median(w1_wmma_samples)},
                        {"w1_fused_median_ms", median(w1_fused_samples)},
                        {"w5_serial_wmma_median_ms", median(wmma_samples)},
                        {"w5_fused_median_ms", median(fused_samples)},
                        {"w1_wmma_samples_ms", w1_wmma_samples},
                        {"w1_fused_samples_ms", w1_fused_samples},
                        {"w5_serial_wmma_samples_ms", wmma_samples},
                        {"w5_fused_samples_ms", fused_samples}}},
            {"production_routing_changed", false},
            {"production_route_authorized", false}};
        const std::string serialized = report.dump(2);
        const Json parsed = Json::parse(serialized);
        if (parsed != report || parsed.at("timing").at("evidence_eligible") != false ||
            parsed.at("production_route_authorized") != false ||
            parsed.at("production_selected_w1_w5_row0_mismatches") != fused_mismatches) {
            fail("serialized report failed its exact JSON schema/round-trip contract");
        }
        std::cout << serialized << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash_attention_route_discriminator: " << error.what() << '\n';
        return 1;
    }
}
