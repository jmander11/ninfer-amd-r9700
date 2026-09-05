#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kKAlignment = 128;
constexpr int kTimingRounds = 5;
constexpr int kLdsScopeTrials = 7;
constexpr const char* kPowerProfilePath =
    "/sys/class/drm/card2/device/power_dpm_force_performance_level";
constexpr double kMaximumPerTupleRatio = 1.01;
struct Shape {
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t count;
    const char* role;
};

constexpr Shape kMixedW8Shapes[] = {
    {1152, 1152, 27, "vision_attention_output"},
    {1152, 1536, 1, "vision_patch_embedding"},
    {1152, 4304, 27, "vision_mlp_fc2_tail"},
    {4608, 4608, 1, "vision_merger_fc1"},
    {5120, 4608, 1, "vision_merger_fc2"},
    {7168, 5120, 16, "text_attention_gate_value"},
    {12288, 5120, 48, "text_gdn_value_z"},
    {14336, 5120, 1, "mtp_attention_qkgv"},
    {34816, 5120, 1, "mtp_mlp_gate_up"},
    {248320, 5120, 2, "vocabulary"},
    {5120, 6144, 65, "text_attention_gdn_output"},
    {5120, 10240, 1, "mtp_input_projection"},
    {5120, 17408, 65, "text_mlp_down"},
};

struct Result {
    Shape shape{};
    std::uint32_t tokens = 0;
    std::size_t workspace_bytes = 0;
    double maximum_formula_error = 0.0;
    double maximum_source_error = 0.0;
    std::vector<double> a8_samples_ms;
    std::vector<double> exact_samples_ms;
    double a8_median_ms = 0.0;
    double exact_median_ms = 0.0;
};

struct PrefillResult {
    Shape shape{};
    std::uint32_t tokens = 0;
    bool exhaustive_oracle = false;
    std::uint32_t oracle_sample_count = 0;
    std::uint64_t maximum_bf16_steps = 0;
    std::vector<double> one_wave_samples_ms;
    std::vector<double> cooperative_cta_samples_ms;
    double one_wave_median_ms = 0.0;
    double cooperative_cta_median_ms = 0.0;
};

struct LdsScopeTiming {
    std::vector<double> incumbent_forward_ms;
    std::vector<double> challenger_forward_ms;
    std::vector<double> incumbent_reverse_ms;
    std::vector<double> challenger_reverse_ms;
    std::vector<double> incumbent_balanced_ms;
    std::vector<double> challenger_balanced_ms;
    double incumbent_median_ms = 0.0;
    double challenger_median_ms = 0.0;
};

struct LdsScopeEvidence {
    PrefillResult numerical{};
    LdsScopeTiming timing{};
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream out;
    out << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(out.str());
}
#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count != 0U) HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) (void)hipFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const noexcept { return data_; }
    std::size_t count() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    ~Event() { (void)hipEventDestroy(event_); }
    hipEvent_t get() const noexcept { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

std::uint32_t padded_columns(std::uint32_t columns) {
    return (columns + kKAlignment - 1U) / kKAlignment * kKAlignment;
}

std::uint16_t half_bits(float value) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}

float half_value(std::uint16_t bits) {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

int weight_code(std::uint32_t row, std::uint32_t column, std::uint32_t columns) {
    if (column >= columns) return 0;
    const std::uint32_t group = column / kGroup;
    const std::uint32_t lane = column % kGroup;
    return static_cast<int>((row * 17U + group * 13U + lane * 19U) % 255U) - 127;
}

std::uint16_t weight_scale_bits(std::uint32_t row, std::uint32_t group) {
    return half_bits(static_cast<float>(1U + ((row * 11U + group * 7U) % 31U)) / 4096.0F);
}

__global__ void initialize_weight_codes(std::int8_t* codes, std::uint32_t rows,
                                        std::uint32_t columns,
                                        std::uint32_t padded_columns_value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = static_cast<std::size_t>(rows) * padded_columns_value;
    if (index >= elements) return;
    const std::uint32_t row = static_cast<std::uint32_t>(index / padded_columns_value);
    const std::uint32_t column = static_cast<std::uint32_t>(index % padded_columns_value);
    int code = 0;
    if (column < columns) {
        const std::uint32_t group = column / kGroup;
        const std::uint32_t lane = column % kGroup;
        code = static_cast<int>((row * 17U + group * 13U + lane * 19U) % 255U) - 127;
    }
    codes[index] = static_cast<std::int8_t>(code);
}

__global__ void initialize_weight_scales(std::uint16_t* scales, std::uint32_t rows,
                                         std::uint32_t groups) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = static_cast<std::size_t>(rows) * groups;
    if (index >= elements) return;
    const std::uint32_t row = static_cast<std::uint32_t>(index / groups);
    const std::uint32_t group = static_cast<std::uint32_t>(index % groups);
    const float scale = static_cast<float>(1U + ((row * 11U + group * 7U) % 31U)) / 4096.0F;
    scales[index] = __half_as_ushort(__float2half_rn(scale));
}

__global__ void poison_i8_padded_tail(std::int8_t* codes, std::uint32_t tokens,
                                      std::uint32_t columns,
                                      std::uint32_t padded_columns_value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = static_cast<std::size_t>(tokens) * padded_columns_value;
    if (index >= elements) return;
    const std::uint32_t column = static_cast<std::uint32_t>(index % padded_columns_value);
    if (column >= columns) codes[index] = 1;
}

std::vector<hip_bfloat16> make_input(std::uint32_t tokens, std::uint32_t columns) {
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((token * 97U + column * 29U) % 509U) - 254;
            input[static_cast<std::size_t>(token) * columns + column] =
                hip_bfloat16(static_cast<float>(pattern) / 256.0F);
        }
    }
    if (tokens != 0U && columns >= 96U) {
        for (std::uint32_t lane = 0; lane < 32U; ++lane) input[lane] = hip_bfloat16(0.0F);
        hip_bfloat16 tiny{};
        tiny.data = 1U;
        for (std::uint32_t lane = 0; lane < 32U; ++lane) {
            input[32U + lane] = lane & 1U ? tiny : -tiny;
        }
        constexpr float ties[] = {-127.0F, -126.5F, -1.5F, -0.5F,
                                  0.5F, 1.5F, 126.5F, 127.0F};
        for (std::uint32_t lane = 0; lane < 32U; ++lane) {
            input[64U + lane] = hip_bfloat16(ties[lane % 8U]);
        }
    }
    return input;
}

struct HostActivation {
    std::vector<std::int8_t> codes;
    std::vector<std::uint16_t> scales;
};

HostActivation host_quantize(const std::vector<hip_bfloat16>& input, std::uint32_t tokens,
                             std::uint32_t columns, std::uint32_t padded) {
    const std::uint32_t groups = padded / kGroup;
    HostActivation output{std::vector<std::int8_t>(static_cast<std::size_t>(tokens) * padded),
                          std::vector<std::uint16_t>(static_cast<std::size_t>(tokens) * groups)};
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            float maximum = 0.0F;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const float value = column < columns
                                        ? static_cast<float>(input[
                                              static_cast<std::size_t>(token) * columns + column])
                                        : 0.0F;
                if (!std::isfinite(value)) fail("host codec input is nonfinite");
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint16_t bits = maximum == 0.0F ? 0U : half_bits(maximum / 127.0F);
            if (maximum != 0.0F && bits == 0U) bits = 1U;
            const float scale = half_value(bits);
            if (maximum != 0.0F && !std::isfinite(scale)) fail("host codec scale overflow");
            output.scales[static_cast<std::size_t>(token) * groups + group] = bits;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                int code = 0;
                if (column < columns && bits != 0U) {
                    const float value = static_cast<float>(
                        input[static_cast<std::size_t>(token) * columns + column]);
                    code = std::clamp(static_cast<int>(std::nearbyint(value / scale)), -127, 127);
                }
                output.codes[static_cast<std::size_t>(token) * padded + column] =
                    static_cast<std::int8_t>(code);
            }
        }
    }
    return output;
}

hipDeviceProp_t select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    hipDeviceProp_t properties{};
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t candidate{};
        HIP_CHECK(hipGetDeviceProperties(&candidate, ordinal));
        if (candidate.integrated == 0 && std::string_view(candidate.gcnArchName) == "gfx1201" &&
            candidate.warpSize == 32) {
            if (selected >= 0) fail("expected exactly one discrete wave32 gfx1201 device");
            selected = ordinal;
            properties = candidate;
        }
    }
    if (selected < 0) fail("no discrete wave32 gfx1201 device found");
    HIP_CHECK(hipSetDevice(selected));
    return properties;
}

template <typename Fn>
double time_once(Fn&& launch) {
    Event begin;
    Event end;
    HIP_CHECK(hipEventRecord(begin.get()));
    launch();
    HIP_CHECK(hipEventRecord(end.get()));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, begin.get(), end.get()));
    return milliseconds;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

double represented_oracle(const HostActivation& activation,
                          const std::vector<hip_bfloat16>& source,
                          std::uint32_t token, std::uint32_t row,
                          std::uint32_t columns, std::uint32_t padded,
                          bool quantized_activation) {
    const std::uint32_t groups = padded / kGroup;
    double total = 0.0;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const double weight_scale = half_value(weight_scale_bits(row, group));
        if (quantized_activation) {
            std::int32_t dot = 0;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                dot += static_cast<int>(activation.codes[
                           static_cast<std::size_t>(token) * padded + column]) *
                       weight_code(row, column, columns);
            }
            const double activation_scale = half_value(
                activation.scales[static_cast<std::size_t>(token) * groups + group]);
            total = std::fma(static_cast<double>(dot), activation_scale * weight_scale, total);
        } else {
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                if (column >= columns) continue;
                const double value = static_cast<float>(
                    source[static_cast<std::size_t>(token) * columns + column]);
                const double weight = static_cast<double>(weight_code(row, column, columns)) *
                                      weight_scale;
                total = std::fma(value, weight, total);
            }
        }
    }
    return static_cast<float>(hip_bfloat16(static_cast<float>(total)));
}

void initialize_weights(DeviceBuffer<std::int8_t>& codes,
                        DeviceBuffer<std::uint16_t>& scales, const Shape& shape,
                        std::uint32_t padded) {
    constexpr std::uint32_t threads = 256;
    const std::size_t code_count = static_cast<std::size_t>(shape.rows) * padded;
    const std::size_t scale_count = static_cast<std::size_t>(shape.rows) * (padded / kGroup);
    hipLaunchKernelGGL(initialize_weight_codes,
                       dim3((code_count + threads - 1U) / threads), dim3(threads), 0, 0,
                       codes.get(), shape.rows, shape.columns, padded);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(initialize_weight_scales,
                       dim3((scale_count + threads - 1U) / threads), dim3(threads), 0, 0,
                       scales.get(), shape.rows, padded / kGroup);
    HIP_CHECK(hipGetLastError());
}

Result qualify_width(const Shape& shape, std::uint32_t tokens,
                     DeviceBuffer<std::int8_t>& weights,
                     DeviceBuffer<std::uint16_t>& weight_scales) {
    const std::uint32_t padded = padded_columns(shape.columns);
    const std::vector<hip_bfloat16> input = make_input(tokens, shape.columns);
    const HostActivation host = host_quantize(input, tokens, shape.columns, padded);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(tokens) * shape.rows);
    DeviceBuffer<std::uint8_t> d_workspace(
        ninfer::ops::r9700::linear::a8w8g32_activation_workspace_capacity_bytes(
            tokens, shape.columns));
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    ninfer::ops::r9700::linear::A8G32ActivationWorkspace workspace{};
    HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_bind_activation_workspace(
        d_workspace.get(), d_workspace.bytes(), tokens, shape.columns, &workspace));
    HIP_CHECK(ninfer::ops::r9700::linear::a8g32_quantize_activation(
        {d_input.get(), workspace}, nullptr));
    std::vector<std::int8_t> actual_codes(host.codes.size());
    std::vector<std::uint16_t> actual_scales(host.scales.size());
    std::uint32_t status = std::numeric_limits<std::uint32_t>::max();
    HIP_CHECK(hipMemcpy(actual_codes.data(), workspace.codes, workspace.code_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_scales.data(), workspace.scales, workspace.scale_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
    if (status != 0U || actual_codes != host.codes || actual_scales != host.scales) {
        fail("production A8G32 codec differs from independent host bytes");
    }

    const auto candidate_args = ninfer::ops::r9700::linear::A8W8G32CandidateArgs{
        d_input.get(), weights.get(), weights.bytes(), weight_scales.get(),
        weight_scales.bytes(), d_workspace.get(), d_workspace.bytes(), d_output.get(),
        tokens, shape.rows, shape.columns, padded};
    const auto exact_args = ninfer::ops::r9700::linear::W8G32LinearArgs{
        d_input.get(), weights.get(), weights.bytes(), weight_scales.get(),
        weight_scales.bytes(), d_output.get(), tokens, shape.rows, shape.columns, padded};
    HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_linear_candidate(candidate_args, nullptr));
    std::vector<hip_bfloat16> output(d_output.count());
    HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));

    double maximum_formula_error = 0.0;
    double maximum_source_error = 0.0;
    const std::uint32_t token_samples[] = {0U, tokens / 2U, tokens - 1U};
    const std::uint32_t row_samples[] = {0U, 1U, shape.rows / 2U, shape.rows - 1U};
    for (const std::uint32_t token : token_samples) {
        for (const std::uint32_t row : row_samples) {
            const double actual = static_cast<float>(
                output[static_cast<std::size_t>(token) * shape.rows + row]);
            const double represented = represented_oracle(
                host, input, token, row, shape.columns, padded, true);
            const double original = represented_oracle(
                host, input, token, row, shape.columns, padded, false);
            const double error = std::abs(actual - represented);
            maximum_formula_error = std::max(maximum_formula_error, error);
            maximum_source_error = std::max(maximum_source_error, std::abs(actual - original));
            if (error > std::max(0.03125, std::abs(represented) * 0.0079)) {
                std::ostringstream message;
                message << "FP64 formula mismatch shape=[" << shape.rows << ',' << shape.columns
                        << "] T=" << tokens << " token=" << token << " row=" << row
                        << " actual=" << actual << " expected=" << represented;
                fail(message.str());
            }
        }
    }

    HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_linear_candidate(candidate_args, nullptr));
    HIP_CHECK(ninfer::ops::r9700::linear::w8g32_linear(exact_args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    Result result{};
    result.shape = shape;
    result.tokens = tokens;
    result.workspace_bytes = d_workspace.bytes();
    result.maximum_formula_error = maximum_formula_error;
    result.maximum_source_error = maximum_source_error;
    for (int round = 0; round < kTimingRounds; ++round) {
        const auto time_a8 = [&] {
            return time_once([&] {
                HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_linear_candidate(
                    candidate_args, nullptr));
            });
        };
        const auto time_exact = [&] {
            return time_once([&] {
                HIP_CHECK(ninfer::ops::r9700::linear::w8g32_linear(exact_args, nullptr));
            });
        };
        if ((round & 1) == 0) {
            result.a8_samples_ms.push_back(time_a8());
            result.exact_samples_ms.push_back(time_exact());
        } else {
            result.exact_samples_ms.push_back(time_exact());
            result.a8_samples_ms.push_back(time_a8());
        }
    }
    result.a8_median_ms = median(result.a8_samples_ms);
    result.exact_median_ms = median(result.exact_samples_ms);
    return result;
}

PrefillResult qualify_prefill_width(const Shape& shape, std::uint32_t tokens,
                                    bool exhaustive_oracle, bool collect_timing,
                                    bool compare_lds_scope = false,
                                    LdsScopeTiming* lds_scope_timing = nullptr) {
    const std::uint32_t padded = padded_columns(shape.columns);
    const std::uint32_t groups = padded / kGroup;
    const std::vector<hip_bfloat16> input = make_input(tokens, shape.columns);
    const HostActivation host = host_quantize(input, tokens, shape.columns, padded);
    std::vector<std::int8_t> stored_weight_codes(
        static_cast<std::size_t>(shape.rows) * padded);
    std::vector<std::uint16_t> stored_weight_scales(
        static_cast<std::size_t>(shape.rows) * groups);
    std::array<std::vector<std::int8_t>, 255> code_row_patterns;
    std::array<std::vector<std::uint16_t>, 31> scale_row_patterns;
    for (std::uint32_t row_pattern = 0; row_pattern < code_row_patterns.size();
         ++row_pattern) {
        auto& pattern = code_row_patterns[row_pattern];
        pattern.resize(padded);
        for (std::uint32_t column = 0; column < padded; ++column) {
            pattern[column] = column < shape.columns
                                  ? static_cast<std::int8_t>(weight_code(
                                        row_pattern, column, shape.columns))
                                  : static_cast<std::int8_t>(73);
        }
    }
    for (std::uint32_t row_pattern = 0; row_pattern < scale_row_patterns.size();
         ++row_pattern) {
        auto& pattern = scale_row_patterns[row_pattern];
        pattern.resize(groups);
        for (std::uint32_t group = 0; group < groups; ++group) {
            pattern[group] = weight_scale_bits(row_pattern, group);
        }
    }
    for (std::uint32_t row = 0; row < shape.rows; ++row) {
        std::copy(code_row_patterns[row % 255U].begin(),
                  code_row_patterns[row % 255U].end(),
                  stored_weight_codes.begin() + static_cast<std::size_t>(row) * padded);
        std::copy(scale_row_patterns[row % 31U].begin(),
                  scale_row_patterns[row % 31U].end(),
                  stored_weight_scales.begin() + static_cast<std::size_t>(row) * groups);
    }
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::int8_t> weights(static_cast<std::size_t>(shape.rows) * padded);
    DeviceBuffer<std::uint16_t> weight_scales(
        static_cast<std::size_t>(shape.rows) * groups);
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(tokens) * shape.rows);
    DeviceBuffer<std::uint8_t> d_workspace(
        ninfer::ops::r9700::linear::a8w8g32_activation_workspace_capacity_bytes(
            tokens, shape.columns));
    HIP_CHECK(hipMemcpy(weights.get(), stored_weight_codes.data(),
                        stored_weight_codes.size() * sizeof(std::int8_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(weight_scales.get(), stored_weight_scales.data(),
                        stored_weight_scales.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    ninfer::ops::r9700::linear::A8G32ActivationWorkspace workspace{};
    HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_bind_activation_workspace(
        d_workspace.get(), d_workspace.bytes(), tokens, shape.columns, &workspace));
    HIP_CHECK(ninfer::ops::r9700::linear::a8g32_quantize_activation(
        {d_input.get(), workspace}, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    const ninfer::ops::r9700::linear::A8W8G32LinearArgs args{
        workspace.codes, workspace.code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, weights.get(), weights.bytes(), weight_scales.get(),
        weight_scales.bytes(), d_output.get(), tokens, shape.rows, shape.columns, padded};
    const auto one_wave = [&] {
        return ninfer::ops::r9700::linear::a8w8g32_linear_wmma32(args, nullptr);
    };
    const auto cooperative = [&] {
        return ninfer::ops::r9700::linear::a8w8g32_linear_prefill_cta(args, nullptr);
    };
    const auto global_inv_incumbent = [&] {
        return ninfer::ops::r9700::linear::
            a8w8g32_linear_prefill_cta_global_inv_qualification(args, nullptr);
    };
    if (exhaustive_oracle && !collect_timing) {
        auto misaligned = args;
        ++misaligned.activation_codes;
        if (ninfer::ops::r9700::linear::a8w8g32_linear_prefill_cta(
                misaligned, nullptr) != hipErrorInvalidValue) {
            fail("A8W8 prefill CTA accepted a non-dword-aligned packed code plane");
        }
        if (compare_lds_scope &&
            ninfer::ops::r9700::linear::
                a8w8g32_linear_prefill_cta_global_inv_qualification(
                    misaligned, nullptr) != hipErrorInvalidValue) {
            fail("A8W8 LDS-scope challenger accepted a non-dword-aligned code plane");
        }
        constexpr std::uint32_t bad_status = 1U;
        HIP_CHECK(hipMemcpy(workspace.status, &bad_status, sizeof(bad_status),
                            hipMemcpyHostToDevice));
        HIP_CHECK(cooperative());
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<hip_bfloat16> poisoned(d_output.count());
        HIP_CHECK(hipMemcpy(poisoned.data(), d_output.get(), d_output.bytes(),
                            hipMemcpyDeviceToHost));
        if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
                return value.data == 0x7fc1U;
            })) {
            fail("A8W8 prefill CTA did not poison every T>128 output for nonzero status");
        }
        if (compare_lds_scope) {
            HIP_CHECK(global_inv_incumbent());
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(poisoned.data(), d_output.get(), d_output.bytes(),
                                hipMemcpyDeviceToHost));
            if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
                    return value.data == 0x7fc1U;
                })) {
                fail("A8W8 LDS-scope challenger did not preserve status poisoning");
            }
        }
        HIP_CHECK(hipMemset(workspace.status, 0, sizeof(std::uint32_t)));
        const std::size_t activation_elements =
            static_cast<std::size_t>(tokens) * padded;
        poison_i8_padded_tail<<<(activation_elements + 255U) / 256U, 256>>>(
            workspace.codes, tokens, shape.columns, padded);
        HIP_CHECK(hipGetLastError());
    }
    PrefillResult result{};
    result.shape = shape;
    result.tokens = tokens;
    result.exhaustive_oracle = exhaustive_oracle;
    if (collect_timing) {
        HIP_CHECK(cooperative());
        HIP_CHECK(lds_scope_timing == nullptr ? one_wave() : global_inv_incumbent());
        HIP_CHECK(hipDeviceSynchronize());
        if (lds_scope_timing != nullptr) {
            if (!compare_lds_scope) fail("LDS-scope timing requires challenger route");
            for (int round = 0; round < kLdsScopeTrials; ++round) {
                lds_scope_timing->incumbent_forward_ms.push_back(
                    time_once([&] { HIP_CHECK(global_inv_incumbent()); }));
                lds_scope_timing->challenger_forward_ms.push_back(
                    time_once([&] { HIP_CHECK(cooperative()); }));
                lds_scope_timing->challenger_reverse_ms.push_back(
                    time_once([&] { HIP_CHECK(cooperative()); }));
                lds_scope_timing->incumbent_reverse_ms.push_back(
                    time_once([&] { HIP_CHECK(global_inv_incumbent()); }));
                lds_scope_timing->incumbent_balanced_ms.push_back(
                    (lds_scope_timing->incumbent_forward_ms.back() +
                     lds_scope_timing->incumbent_reverse_ms.back()) / 2.0);
                lds_scope_timing->challenger_balanced_ms.push_back(
                    (lds_scope_timing->challenger_forward_ms.back() +
                     lds_scope_timing->challenger_reverse_ms.back()) / 2.0);
            }
            lds_scope_timing->incumbent_median_ms =
                median(lds_scope_timing->incumbent_balanced_ms);
            lds_scope_timing->challenger_median_ms =
                median(lds_scope_timing->challenger_balanced_ms);
            result.one_wave_samples_ms = lds_scope_timing->incumbent_balanced_ms;
            result.cooperative_cta_samples_ms = lds_scope_timing->challenger_balanced_ms;
            result.one_wave_median_ms = lds_scope_timing->incumbent_median_ms;
            result.cooperative_cta_median_ms = lds_scope_timing->challenger_median_ms;
            return result;
        }
        for (int round = 0; round < kTimingRounds; ++round) {
            if ((round & 1) == 0) {
                result.one_wave_samples_ms.push_back(time_once([&] { HIP_CHECK(one_wave()); }));
                result.cooperative_cta_samples_ms.push_back(time_once([&] { HIP_CHECK(cooperative()); }));
            } else {
                result.cooperative_cta_samples_ms.push_back(time_once([&] { HIP_CHECK(cooperative()); }));
                result.one_wave_samples_ms.push_back(time_once([&] { HIP_CHECK(one_wave()); }));
            }
        }
        result.one_wave_median_ms = median(result.one_wave_samples_ms);
        result.cooperative_cta_median_ms = median(result.cooperative_cta_samples_ms);
        return result;
    }
    HIP_CHECK(cooperative());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<hip_bfloat16> output(d_output.count());
    HIP_CHECK(hipMemcpy(output.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
    if (compare_lds_scope) {
        HIP_CHECK(global_inv_incumbent());
        HIP_CHECK(hipDeviceSynchronize());
        constexpr std::size_t kComparisonElements = 1U << 20U;
        std::vector<hip_bfloat16> challenger(
            std::min(kComparisonElements, output.size()));
        for (std::size_t begin = 0; begin < output.size(); begin += challenger.size()) {
            const std::size_t count = std::min(challenger.size(), output.size() - begin);
            HIP_CHECK(hipMemcpy(challenger.data(), d_output.get() + begin,
                                count * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
            for (std::size_t offset = 0; offset < count; ++offset) {
                if (challenger[offset].data != output[begin + offset].data) {
                    fail("A8W8 LDS-scope challenger is not bit-exact with production CTA");
                }
                output[begin + offset] = challenger[offset];
            }
        }
    }
    const std::uint32_t sampled_tokens[] = {0U, 31U, 32U, 63U, 64U, tokens / 3U,
                                            tokens / 2U, tokens - 1U};
    const std::uint32_t sampled_rows[] = {0U, 1U, 15U, 16U, shape.rows / 3U,
                                          shape.rows / 2U, shape.rows - 2U,
                                          shape.rows - 1U};
    const auto check = [&](std::uint32_t token, std::uint32_t row) {
        ++result.oracle_sample_count;
        double sum = 0.0;
        for (std::uint32_t group = 0; group < groups; ++group) {
            std::int32_t dot = 0;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                if (column >= shape.columns) continue;
                dot += static_cast<int>(host.codes[
                           static_cast<std::size_t>(token) * padded + column]) *
                       static_cast<int>(stored_weight_codes[
                           static_cast<std::size_t>(row) * padded + column]);
            }
            const double activation_scale = half_value(
                host.scales[static_cast<std::size_t>(token) * groups + group]);
            const double weight_scale = half_value(
                stored_weight_scales[static_cast<std::size_t>(row) * groups + group]);
            sum = std::fma(static_cast<double>(dot),
                           activation_scale * weight_scale, sum);
        }
        const hip_bfloat16 expected(static_cast<float>(sum));
        const hip_bfloat16 actual =
            output[static_cast<std::size_t>(token) * shape.rows + row];
        const std::uint64_t steps = actual.data > expected.data
                                        ? actual.data - expected.data
                                        : expected.data - actual.data;
        result.maximum_bf16_steps = std::max(result.maximum_bf16_steps, steps);
        if (steps > 2U) {
            std::ostringstream message;
            message << "A8W8 prefill CTA formula mismatch shape=[" << shape.rows << ','
                    << shape.columns << "] T=" << tokens << " token=" << token
                    << " row=" << row << " bf16_steps=" << steps;
            fail(message.str());
        }
    };
    if (exhaustive_oracle) {
        for (std::uint32_t token = 0; token < tokens; ++token) {
            for (std::uint32_t row = 0; row < shape.rows; ++row) check(token, row);
        }
    } else {
        for (std::uint32_t token : sampled_tokens) {
            for (std::uint32_t row : sampled_rows) check(token, row);
        }
    }
    return result;
}

void qualify_malformed_and_nonfinite() {
    constexpr Shape shape{17, 4304, 1, "tail_contract"};
    constexpr std::uint32_t tokens = 3;
    const std::uint32_t padded = padded_columns(shape.columns);
    DeviceBuffer<hip_bfloat16> input(static_cast<std::size_t>(tokens) * shape.columns);
    DeviceBuffer<hip_bfloat16> output(static_cast<std::size_t>(tokens) * shape.rows);
    DeviceBuffer<std::int8_t> weights(static_cast<std::size_t>(shape.rows) * padded);
    DeviceBuffer<std::uint16_t> weight_scales(
        static_cast<std::size_t>(shape.rows) * (padded / kGroup));
    initialize_weights(weights, weight_scales, shape, padded);
    const std::size_t bytes =
        ninfer::ops::r9700::linear::a8w8g32_activation_workspace_capacity_bytes(
            tokens, shape.columns);
    DeviceBuffer<std::uint8_t> storage(bytes);
    ninfer::ops::r9700::linear::A8G32ActivationWorkspace workspace{};
    if (ninfer::ops::r9700::linear::a8w8g32_bind_activation_workspace(
            storage.get(), bytes - 1U, tokens, shape.columns, &workspace) !=
        hipErrorInvalidValue) {
        fail("short A8G32 workspace was accepted");
    }
    HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_bind_activation_workspace(
        storage.get(), bytes, tokens, shape.columns, &workspace));
    auto args = ninfer::ops::r9700::linear::A8W8G32CandidateArgs{
        input.get(), weights.get(), weights.bytes(), weight_scales.get(), weight_scales.bytes(),
        storage.get(), storage.bytes(), output.get(), tokens, shape.rows, shape.columns, padded};
    args.padded_columns -= kKAlignment;
    if (ninfer::ops::r9700::linear::a8w8g32_linear_candidate(args, nullptr) !=
        hipErrorInvalidValue) {
        fail("noncanonical W8 padded K was accepted");
    }
    args.padded_columns = padded;
    args.output = reinterpret_cast<hip_bfloat16*>(weights.get());
    if (ninfer::ops::r9700::linear::a8w8g32_linear_candidate(args, nullptr) !=
        hipErrorInvalidValue) {
        fail("output/weight alias was accepted");
    }

    std::vector<hip_bfloat16> invalid(static_cast<std::size_t>(tokens) * shape.columns,
                                      hip_bfloat16(0.25F));
    invalid[shape.columns + 7U].data = 0x7f80U;
    HIP_CHECK(hipMemcpy(input.get(), invalid.data(), input.bytes(), hipMemcpyHostToDevice));
    args.output = output.get();
    HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_linear_candidate(args, nullptr));
    std::vector<hip_bfloat16> poisoned(output.count());
    HIP_CHECK(hipMemcpy(poisoned.data(), output.get(), output.bytes(), hipMemcpyDeviceToHost));
    if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
            return value.data == 0x7fc1U;
        })) {
        fail("nonfinite activation did not asynchronously poison every output");
    }
}

void qualify_tensor_dispatch() {
    static_assert(ninfer::ops::r9700::linear::kW8ActivationBits == 8);
    constexpr Shape shape{1152, 4304, 27, "vision_mlp_fc2_tail"};
    const std::uint32_t padded = padded_columns(shape.columns);
    DeviceBuffer<std::int8_t> weights(static_cast<std::size_t>(shape.rows) * padded);
    DeviceBuffer<std::uint16_t> scales(
        static_cast<std::size_t>(shape.rows) * (padded / kGroup));
    initialize_weights(weights, scales, shape, padded);
    ninfer::Weight weight{};
    weight.payload = weights.get();
    weight.payload_bytes = weights.bytes() + scales.bytes();
    weight.qdata = weights.get();
    weight.qdata_bytes = weights.bytes();
    weight.scales = scales.get();
    weight.scale_bytes = scales.bytes();
    weight.qtype = ninfer::QType::W8G32_F16S;
    weight.group_size = weight.group = kGroup;
    weight.layout = ninfer::QuantLayout::RowSplit;
    weight.scale_dtype = ninfer::DType::FP16;
    weight.ndim = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = shape.rows;
    weight.k = weight.shape[1] = shape.columns;
    weight.padded_shape[1] = padded;
    for (const std::uint32_t tokens : {63U, 64U}) {
        const std::vector<hip_bfloat16> host = make_input(tokens, shape.columns);
        DeviceBuffer<hip_bfloat16> input(host.size());
        DeviceBuffer<hip_bfloat16> actual(static_cast<std::size_t>(tokens) * shape.rows);
        DeviceBuffer<hip_bfloat16> expected(actual.count());
        HIP_CHECK(hipMemcpy(input.get(), host.data(), input.bytes(), hipMemcpyHostToDevice));
        ninfer::Tensor x(input.get(), ninfer::DType::BF16,
                         {static_cast<std::int32_t>(shape.columns),
                          static_cast<std::int32_t>(tokens)});
        ninfer::Tensor y(actual.get(), ninfer::DType::BF16,
                         {static_cast<std::int32_t>(shape.rows),
                          static_cast<std::int32_t>(tokens)});
        const std::size_t capacity = ninfer::ops::linear_workspace_capacity_bytes(
            ninfer::QType::W8G32_F16S, tokens, shape.columns);
        ninfer::WorkspaceArena arena(std::max<std::size_t>(capacity, 256U));
        ninfer::ops::linear(x, weight, y, arena, nullptr);
        if (tokens == 63U) {
            HIP_CHECK(ninfer::ops::r9700::linear::w8g32_linear(
                {input.get(), weights.get(), weights.bytes(), scales.get(), scales.bytes(),
                 expected.get(), tokens, shape.rows, shape.columns, padded}, nullptr));
            if (capacity == 0U || arena.peak_used() != 0U) {
                fail("A8 evaluator did not retain exact route below the shape crossover");
            }
        } else {
            DeviceBuffer<std::uint8_t> workspace(capacity);
            HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_linear_candidate(
                {input.get(), weights.get(), weights.bytes(), scales.get(), scales.bytes(),
                 workspace.get(), workspace.bytes(), expected.get(), tokens, shape.rows,
                 shape.columns, padded}, nullptr));
            if (capacity == 0U || arena.used() != 0U || arena.peak_used() != capacity) {
                fail("A8 Tensor path did not allocate and rewind crossover workspace");
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<hip_bfloat16> actual_host(actual.count());
        std::vector<hip_bfloat16> expected_host(expected.count());
        HIP_CHECK(hipMemcpy(actual_host.data(), actual.get(), actual.bytes(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(expected_host.data(), expected.get(), expected.bytes(),
                            hipMemcpyDeviceToHost));
        if (!std::equal(actual_host.begin(), actual_host.end(), expected_host.begin(),
                        [](hip_bfloat16 lhs, hip_bfloat16 rhs) { return lhs.data == rhs.data; })) {
            fail("compile-selected Tensor W8 crossover differs from its raw route");
        }
    }
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') out << '\\';
        out << static_cast<char>(character);
    }
    return out.str();
}

void write_samples(std::ostream& out, const std::vector<double>& samples) {
    out << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0U) out << ", ";
        out << samples[index];
    }
    out << ']';
}

std::string shell_quote(std::string_view value) {
    std::string out("'");
    for (const char character : value) {
        if (character == '\'') out += "'\\''";
        else out += character;
    }
    out += '\'';
    return out;
}

std::string sha256_file(const std::filesystem::path& path) {
    const std::string command = "sha256sum -- " + shell_quote(path.string());
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) fail("cannot launch sha256sum");
    std::array<char, 128> buffer{};
    const std::size_t size = fread(buffer.data(), 1U, buffer.size() - 1U, pipe);
    const int status = pclose(pipe);
    const std::string value(buffer.data(), size);
    if (status != 0 || value.size() < 65U || value[64] != ' ') {
        fail("sha256sum failed for " + path.string());
    }
    const std::string digest = value.substr(0U, 64U);
    if (!std::all_of(digest.begin(), digest.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) fail("invalid SHA-256 for " + path.string());
    return digest;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string value;
    std::getline(input, value);
    if (!input && value.empty()) fail("cannot read " + path.string());
    return value;
}

void require_fresh_output(const std::filesystem::path& path) {
    struct stat status {};
    errno = 0;
    if (::lstat(path.c_str(), &status) == 0 || errno != ENOENT) {
        fail("output path must be fresh and must not be a dangling alias");
    }
    if (path.parent_path().empty() || !std::filesystem::is_directory(path.parent_path())) {
        fail("output parent directory must already exist");
    }
}

struct SourceIdentity {
    const char* role;
    std::filesystem::path path;
    std::string sha256;
};

void write_lds_scope_report(
    const std::filesystem::path& output, const std::filesystem::path& executable,
    std::string_view executable_sha256, const std::filesystem::path& source_root,
    const std::array<SourceIdentity, 4>& sources, const hipDeviceProp_t& device,
    int runtime_version, int driver_version,
    const ninfer::ops::r9700::linear::A8W8G32KernelResources& resources,
    const std::vector<LdsScopeEvidence>& evidence) {
    const auto weight = [](const Shape& shape) -> std::uint32_t {
        if (shape.rows == 7168U && shape.columns == 5120U) return 16U;
        if (shape.rows == 12288U && shape.columns == 5120U) return 48U;
        if (shape.rows == 5120U && shape.columns == 17408U) return 65U;
        return 65U;
    };
    double weighted_incumbent = 0.0;
    double weighted_challenger = 0.0;
    for (const auto& row : evidence) {
        const double ratio = row.timing.challenger_median_ms / row.timing.incumbent_median_ms;
        if (!(ratio <= kMaximumPerTupleRatio)) fail("material LDS-scope tuple regression");
        if (row.numerical.tokens == 2048U) {
            weighted_incumbent += weight(row.numerical.shape) * row.timing.incumbent_median_ms;
            weighted_challenger += weight(row.numerical.shape) * row.timing.challenger_median_ms;
        }
    }
    if (!(weighted_challenger < weighted_incumbent)) {
        fail("LDS-scope challenger is not faster for weighted T2048 Text execution");
    }
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n  \"artifact_type\": \"ninfer_r9700_prefill_cta_lds_scope_ab\",\n"
        << "  \"schema_version\": 1,\n  \"status\": \"passed\",\n"
        << "  \"recipe\": \"w8\",\n  \"disposition\": \"qualification_only_unpromoted\",\n"
        << "  \"timing\": {\"method\": \"unprofiled HIP events\", "
           "\"warmup_iterations_per_route\": 1, \"balanced_forward_reverse_pairs\": 7, "
           "\"balanced_sample\": \"mean of same-route forward and reverse measurements\", "
           "\"token_extents\": [1024,2048,4096,8192]},\n"
        << "  \"numerical_qualification\": {\"completed_before_timing\": true, "
           "\"incumbent_challenger_bit_exact\": true, "
           "\"independent_oracle\": \"FP64 complete represented quantized formula\", "
           "\"maximum_bf16_steps_allowed\": 2, "
           "\"tail_status_and_alignment_gate\": \"passed\"},\n"
        << "  \"power_profile\": {\"path\": \"" << kPowerProfilePath
        << "\", \"required\": \"auto\", \"before\": \"auto\", \"after\": \"auto\"},\n"
        << "  \"hardware\": {\"device\": \"" << json_escape(device.name)
        << "\", \"architecture\": \"" << json_escape(device.gcnArchName)
        << "\", \"wave_size\": " << device.warpSize << ", \"vram_bytes\": "
        << device.totalGlobalMem << ", \"hip_runtime_version\": " << runtime_version
        << ", \"hip_driver_version\": " << driver_version << "},\n"
        << "  \"toolchain\": {\"compiler\": \"" << json_escape(__VERSION__)
        << "\", \"offload_architecture\": \"gfx1201\"},\n"
        << "  \"resources\": {\"incumbent_registers\": "
        << resources.prefill_cta_global_inv_qualification_registers
        << ", \"challenger_registers\": " << resources.prefill_cta_registers
        << ", \"incumbent_static_shared_bytes\": "
        << resources.prefill_cta_global_inv_qualification_static_shared_bytes
        << ", \"challenger_static_shared_bytes\": "
        << resources.prefill_cta_static_shared_bytes
        << ", \"incumbent_local_bytes\": "
        << resources.prefill_cta_global_inv_qualification_local_bytes
        << ", \"challenger_local_bytes\": " << resources.prefill_cta_local_bytes
        << "},\n  \"executable\": {\"path\": \"" << json_escape(executable.string())
        << "\", \"sha256\": \"" << executable_sha256 << "\"},\n"
        << "  \"source_root\": \"" << json_escape(source_root.string()) << "\",\n"
        << "  \"sources\": [\n";
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const auto& source = sources[index];
        out << "    {\"role\":\"" << source.role << "\",\"path\":\""
            << json_escape(source.path.string()) << "\",\"sha256\":\""
            << source.sha256 << "\"}" << (index + 1U == sources.size() ? "\n" : ",\n");
    }
    out << "  ],\n  \"cases\": [\n";
    for (std::size_t index = 0; index < evidence.size(); ++index) {
        const auto& row = evidence[index];
        out << "    {\"rows\":" << row.numerical.shape.rows
            << ",\"columns\":" << row.numerical.shape.columns
            << ",\"tokens\":" << row.numerical.tokens
            << ",\"bit_exact\":true,\"maximum_bf16_steps\":"
            << row.numerical.maximum_bf16_steps;
        const auto emit = [&](const char* name, const std::vector<double>& values) {
            out << ",\"" << name << "\":";
            write_samples(out, values);
        };
        emit("incumbent_forward_ms", row.timing.incumbent_forward_ms);
        emit("incumbent_reverse_ms", row.timing.incumbent_reverse_ms);
        emit("incumbent_balanced_ms", row.timing.incumbent_balanced_ms);
        emit("challenger_forward_ms", row.timing.challenger_forward_ms);
        emit("challenger_reverse_ms", row.timing.challenger_reverse_ms);
        emit("challenger_balanced_ms", row.timing.challenger_balanced_ms);
        out << ",\"incumbent_median_ms\":" << row.timing.incumbent_median_ms
            << ",\"challenger_median_ms\":" << row.timing.challenger_median_ms
            << ",\"challenger_over_incumbent\":"
            << row.timing.challenger_median_ms / row.timing.incumbent_median_ms << '}'
            << (index + 1U == evidence.size() ? "\n" : ",\n");
    }
    out << "  ],\n  \"decision\": {\"token_extent\": 2048, \"weights\": ["
        << "{\"rows\":7168,\"columns\":5120,\"count\":16},"
        << "{\"rows\":12288,\"columns\":5120,\"count\":48},"
        << "{\"rows\":5120,\"columns\":17408,\"count\":65},"
        << "{\"rows\":5120,\"columns\":6144,\"count\":65}],"
        << "\"weighted_incumbent_ms\":" << weighted_incumbent
        << ",\"weighted_challenger_ms\":" << weighted_challenger
        << ",\"challenger_over_incumbent\":" << weighted_challenger / weighted_incumbent
        << ",\"maximum_per_tuple_ratio\":1.01,"
        << "\"no_material_per_tuple_regression\":true,"
        << "\"weighted_challenger_faster\":true}\n}\n";
    const std::string bytes = out.str();
    const int descriptor = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("cannot exclusively create output report");
    std::size_t offset = 0U;
    while (offset < bytes.size()) {
        const ssize_t written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (written <= 0) { (void)::close(descriptor); (void)::unlink(output.c_str()); fail("cannot write report"); }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(descriptor) != 0) { (void)::unlink(output.c_str()); fail("cannot close report"); }
}

void write_json(const std::filesystem::path& path, const hipDeviceProp_t& device,
                const std::vector<Result>& results,
                const ninfer::ops::r9700::linear::A8W8G32KernelResources& resources,
                bool passed, std::string_view failure) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out) fail("cannot open W8A8 evidence path");
    out << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer.r9700.w8a8g32_linear_qualification.v3\",\n"
        << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
        << "  \"failure\": \"" << json_escape(failure) << "\",\n"
        << "  \"artifact_profiles\": [\"qwen3.8-27b/r9700-int-candidate\", "
           "\"qwen3.8-27b/r9700-w8g32-mse-eval\", "
           "\"qwen3.8-27b/r9700-q4-w8-n16k16-eval\", "
           "\"qwen3.8-27b/r9700-q4-w8-mse-n16k16-eval\"],\n"
        << "  \"w8_activation_bits\": 8,\n"
        << "  \"dispatch_thresholds\": ["
           "{\"shape\":[1152,1152],\"a8_from_tokens\":64},"
           "{\"shape\":[1152,1536],\"a8_from_tokens\":64},"
           "{\"shape\":[1152,4304],\"a8_from_tokens\":64},"
           "{\"shape\":[4608,4608],\"a8_from_tokens\":4},"
           "{\"shape\":[5120,4608],\"a8_from_tokens\":4},"
           "{\"shape\":[7168,5120],\"a8_from_tokens\":3},"
           "{\"shape\":[12288,5120],\"a8_from_tokens\":3},"
           "{\"shape\":[14336,5120],\"a8_from_tokens\":3},"
           "{\"shape\":[34816,5120],\"a8_from_tokens\":32},"
           "{\"shape\":[248320,5120],\"a8_from_tokens\":4},"
           "{\"shape\":[5120,6144],\"a8_from_tokens\":4},"
           "{\"shape\":[5120,10240],\"a8_from_tokens\":4},"
           "{\"shape\":[5120,17408],\"a8_from_tokens\":3}],\n"
        << "  \"mixed_w8_tensor_count\": 256,\n"
        << "  \"unique_shapes\": 13,\n"
        << "  \"oracle\": {\"activation_codec\": \"exact_all_codes_and_scales\", "
           "\"formula\": \"FP64_A8G32_times_decoded_W8G32_then_BF16\", "
           "\"source_comparison\": \"represented_BF16_times_decoded_W8G32\"},\n"
        << "  \"malformed_alias_tail_nonfinite\": true,\n"
        << "  \"tensor_workspace_shape_crossover\": true,\n"
        << "  \"timing_method\": {\"rounds\": 5, "
           "\"order\": \"alternating_a8_exact_then_exact_a8\", "
           "\"statistic\": \"per-route median HIP event milliseconds\"},\n"
        << "  \"resources\": {\"quantize_vgpr\": " << resources.quantize_registers
        << ", \"quantize_lds_bytes\": " << resources.quantize_static_shared_bytes
        << ", \"quantize_scratch_bytes\": " << resources.quantize_local_bytes
        << ", \"wmma_vgpr\": " << resources.wmma_registers
        << ", \"wmma_lds_bytes\": " << resources.wmma_static_shared_bytes
        << ", \"wmma_scratch_bytes\": " << resources.wmma_local_bytes << "},\n"
        << "  \"hardware\": {\"device\": \"" << json_escape(device.name)
        << "\", \"architecture\": \"" << json_escape(device.gcnArchName)
        << "\", \"wave_size\": " << device.warpSize << "},\n"
        << "  \"toolchain\": {\"compiler\": \"" << json_escape(__VERSION__)
        << "\", \"offload_architecture\": \"gfx1201\"},\n"
        << "  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const Result& result = results[index];
        out << "    {\"shape\": [" << result.shape.rows << ", " << result.shape.columns
            << "], \"inventory_count\": " << result.shape.count << ", \"role\": \""
            << result.shape.role << "\", \"tokens\": " << result.tokens
            << ", \"workspace_bytes\": " << result.workspace_bytes
            << ", \"maximum_formula_error\": " << result.maximum_formula_error
            << ", \"maximum_source_error\": " << result.maximum_source_error
            << ", \"a8_samples_ms\": ";
        write_samples(out, result.a8_samples_ms);
        out << ", \"exact_samples_ms\": ";
        write_samples(out, result.exact_samples_ms);
        out << ", \"a8_median_ms\": " << result.a8_median_ms
            << ", \"exact_median_ms\": " << result.exact_median_ms
            << ", \"speedup\": " << result.exact_median_ms / result.a8_median_ms << '}'
            << (index + 1U == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    out.close();
    if (!out) fail("cannot finish W8A8 evidence path");
}

void write_tensor_json(const std::filesystem::path& path, const hipDeviceProp_t& device,
                       const ninfer::ops::r9700::linear::A8W8G32KernelResources& resources,
                       bool passed, std::string_view failure) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out) fail("cannot open W8A8 Tensor evidence path");
    out << "{\n  \"schema\": \"ninfer.r9700.w8a8g32_tensor_dispatch.v1\",\n"
        << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
        << "  \"failure\": \"" << json_escape(failure) << "\",\n"
        << "  \"w8_activation_bits\": 8,\n"
        << "  \"shape\": [1152, 4304],\n"
        << "  \"exact_bf16_tokens\": 63,\n"
        << "  \"a8_tokens\": 64,\n"
        << "  \"checks\": [\"compile_selected_Tensor_route\", "
           "\"caller_owned_WorkspaceArena\", \"exact_bf16_boundary\", "
           "\"a8_candidate_boundary\", \"bit_exact_route_parity\", "
           "\"malformed_alias_tail\", \"async_nonfinite_nan\"],\n"
        << "  \"resources\": {\"quantize_vgpr\": " << resources.quantize_registers
        << ", \"quantize_lds_bytes\": " << resources.quantize_static_shared_bytes
        << ", \"quantize_scratch_bytes\": " << resources.quantize_local_bytes
        << ", \"wmma_vgpr\": " << resources.wmma_registers
        << ", \"wmma_lds_bytes\": " << resources.wmma_static_shared_bytes
        << ", \"wmma_scratch_bytes\": " << resources.wmma_local_bytes << "},\n"
        << "  \"hardware\": {\"device\": \"" << json_escape(device.name)
        << "\", \"architecture\": \"" << json_escape(device.gcnArchName)
        << "\", \"wave_size\": " << device.warpSize << "},\n"
        << "  \"toolchain\": {\"compiler\": \"" << json_escape(__VERSION__)
        << "\", \"offload_architecture\": \"gfx1201\"}\n}\n";
    out.close();
    if (!out) fail("cannot finish W8A8 Tensor evidence path");
}

void write_mtp_bulk_json(
    const std::filesystem::path& path, std::uint32_t tokens,
    const std::filesystem::path& executable, std::string_view executable_sha256,
    const std::array<SourceIdentity, 4>& sources,
    const hipDeviceProp_t& device,
    const ninfer::ops::r9700::linear::A8W8G32KernelResources& resources,
    const std::array<PrefillResult, 2>& evidence) {
    require_fresh_output(path);
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n  \"artifact_type\": \"ninfer_r9700_mtp_bulk_w8_operator_qualification\",\n"
        << "  \"schema_version\": 1,\n  \"status\": \"passed\",\n"
        << "  \"tokens\": " << tokens << ",\n  \"concurrency\": 1,\n"
        << "  \"power_profile\": {\"path\": \"" << kPowerProfilePath
        << "\", \"value\": \"auto\"},\n"
        << "  \"timing\": {\"method\": \"alternating HIP events\", "
           "\"rounds_per_route\": 5, \"scope\": \"post-quantization linear leaf\"},\n"
        << "  \"candidate\": {\"activation\": \"signed A8G32\", "
           "\"weight\": \"W8G32_F16S\", "
           "\"kernel\": \"a8w8g32_linear_prefill_cta_kernel\", "
           "\"opcode\": \"v_wmma_i32_16x16x16_iu8\"},\n"
        << "  \"baseline\": {\"activation\": \"signed A8G32\", "
        "\"weight\": \"W8G32_F16S\", "
           "\"route\": \"a8w8g32_linear_wmma32\"},\n"
        << "  \"resources\": {\"vgpr\": " << resources.prefill_cta_registers
        << ", \"lds_bytes\": " << resources.prefill_cta_static_shared_bytes
        << ", \"scratch_bytes\": " << resources.prefill_cta_local_bytes << "},\n"
        << "  \"hardware\": {\"device\": \"" << json_escape(device.name)
        << "\", \"architecture\": \"" << json_escape(device.gcnArchName)
        << "\", \"wave_size\": " << device.warpSize << "},\n"
        << "  \"executable\": {\"path\": \"" << json_escape(executable.string())
        << "\", \"sha256\": \"" << executable_sha256 << "\"},\n"
        << "  \"sources\": [\n";
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const SourceIdentity& source = sources[index];
        out << "    {\"role\":\"" << source.role << "\",\"path\":\""
            << json_escape(source.path.string()) << "\",\"sha256\":\""
            << source.sha256 << "\"}" << (index + 1U == sources.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"shapes\": [\n";
    for (std::size_t index = 0; index < evidence.size(); ++index) {
        const PrefillResult& row = evidence[index];
        if (row.tokens != tokens || row.maximum_bf16_steps > 2U ||
            row.oracle_sample_count != 64U ||
            !(row.cooperative_cta_median_ms < row.one_wave_median_ms)) {
            fail("MTP-bulk W8 qualification gate failed");
        }
        if (index != 0U) out << ",\n";
        out << "    {\"rows\": " << row.shape.rows
            << ", \"columns\": " << row.shape.columns
            << ", \"dispatches_per_full_chunk\": " << row.shape.count
            << ", \"oracle_max_bf16_steps\": " << row.maximum_bf16_steps
            << ", \"oracle\": \"independent FP64 represented A8G32-times-W8G32 formula\""
            << ", \"oracle_sample_count\": " << row.oracle_sample_count
            << ", \"candidate_median_ms\": " << row.cooperative_cta_median_ms
            << ", \"baseline_median_ms\": " << row.one_wave_median_ms
            << ", \"candidate_faster\": true}";
    }
    out << "\n  ]\n}\n";
    const std::string bytes = out.str();
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("cannot exclusively create MTP-bulk W8 evidence path");
    std::size_t offset = 0U;
    while (offset < bytes.size()) {
        const ssize_t written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (written <= 0) {
            (void)::close(descriptor); (void)::unlink(path.c_str());
            fail("cannot write MTP-bulk W8 evidence path");
        }
        offset += static_cast<std::size_t>(written);
    }
    const bool sync_failed = ::fsync(descriptor) != 0;
    const bool close_failed = ::close(descriptor) != 0;
    if (sync_failed || close_failed) {
        (void)::unlink(path.c_str()); fail("cannot finish MTP-bulk W8 evidence path");
    }
}

} // namespace

int main(int argc, char** argv) {
    const bool tensor_only = argc == 4 && std::string_view(argv[1]) == "--tensor-only" &&
                             std::string_view(argv[2]) == "--out-json";
    const bool prefill_regression =
        argc == 2 && std::string_view(argv[1]) == "--prefill-cta-regression";
    const bool lds_scope_regression = argc == 2 &&
        std::string_view(argv[1]) == "--prefill-cta-lds-scope-regression";
    const bool lds_scope_benchmark = argc == 4 &&
        std::string_view(argv[1]) == "--prefill-cta-lds-scope-benchmark" &&
        std::string_view(argv[2]) == "--out-json";
    const bool mtp_bulk = argc == 5 &&
        std::string_view(argv[1]) == "--mtp-bulk-selected-chunk" &&
        std::string_view(argv[3]) == "--out-json";
    if (!tensor_only && !prefill_regression && !lds_scope_regression &&
        !lds_scope_benchmark && !mtp_bulk &&
        (argc != 3 || std::string_view(argv[1]) != "--out-json")) {
        std::cerr << "usage: w8a8_wmma_linear_qual "
                     "{[--tensor-only] --out-json PATH|--prefill-cta-regression|"
                     "--prefill-cta-lds-scope-regression|"
                     "--prefill-cta-lds-scope-benchmark --out-json FRESH.json|"
                     "--mtp-bulk-selected-chunk TOKENS --out-json FRESH.json}\n";
        return 2;
    }
    std::filesystem::path output_path;
    if (!prefill_regression && !lds_scope_regression && !lds_scope_benchmark && !mtp_bulk) {
        output_path = argv[tensor_only ? 3 : 2];
    }
    hipDeviceProp_t device{};
    std::vector<Result> results;
    std::vector<PrefillResult> prefill_results;
    ninfer::ops::r9700::linear::A8W8G32KernelResources resources{};
    try {
        std::filesystem::path executable;
        std::filesystem::path source_root;
        std::string executable_sha256;
        std::array<SourceIdentity, 4> source_identities{};
        if (lds_scope_benchmark || mtp_bulk) {
            output_path = argv[3];
            require_fresh_output(output_path);
            if (read_text(kPowerProfilePath) != "auto") {
                fail("LDS-scope physical A/B requires auto power before device construction");
            }
            executable = std::filesystem::canonical("/proc/self/exe");
            source_root = std::filesystem::canonical(NINFER_SOURCE_DIR);
            executable_sha256 = sha256_file(executable);
            const std::array<std::pair<const char*, const char*>, 4> source_specs{{
                {"qualifier", "tools/r9700/w8a8_wmma_linear_qual.hip"},
                {"contract_header", "src/ops/r9700/linear/r9700_linear.h"},
                {"kernel", "src/ops/r9700/linear/r9700_linear.hip"},
                {"dispatch_profile", "src/ops/r9700/linear/r9700_w8_activation_profile.h"},
            }};
            for (std::size_t index = 0; index < source_specs.size(); ++index) {
                const auto path = std::filesystem::canonical(source_root / source_specs[index].second);
                source_identities[index] =
                    SourceIdentity{source_specs[index].first, path, sha256_file(path)};
            }
        }
        if (mtp_bulk) {
            output_path = argv[4];
            require_fresh_output(output_path);
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(argv[2], &end, 10);
            if (end == argv[2] || *end != '\0' || parsed < 1024UL || parsed > 8192UL ||
                (parsed & (parsed - 1UL)) != 0UL) {
                fail("selected MTP-bulk chunk must be one of 1024, 2048, 4096, or 8192");
            }
            if (read_text(kPowerProfilePath) != "auto") {
                fail("MTP-bulk physical qualification requires auto power");
            }
            const std::uint32_t tokens = static_cast<std::uint32_t>(parsed);
            device = select_r9700();
            HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_kernel_resources(&resources));
            if (resources.prefill_cta_registers > 96 ||
                resources.prefill_cta_static_shared_bytes != 4352 ||
                resources.prefill_cta_local_bytes != 0) {
                fail("MTP-bulk W8 production CTA resource gate failed");
            }
            const std::array<Shape, 2> shapes{{
                Shape{5120, 10240, 1, "mtp_input_projection"},
                Shape{1024, 5120, 2, "mtp_attention_kv"},
            }};
            std::array<PrefillResult, 2> evidence{};
            for (std::size_t index = 0; index < shapes.size(); ++index) {
                const PrefillResult numerical =
                    qualify_prefill_width(shapes[index], tokens, false, false);
                evidence[index] = qualify_prefill_width(shapes[index], tokens, false, true);
                evidence[index].maximum_bf16_steps = numerical.maximum_bf16_steps;
                evidence[index].oracle_sample_count = numerical.oracle_sample_count;
            }
            if (read_text(kPowerProfilePath) != "auto") {
                fail("MTP-bulk physical qualification did not retain auto power");
            }
            if (sha256_file(executable) != executable_sha256) {
                fail("MTP-bulk qualifier executable changed during qualification");
            }
            for (const SourceIdentity& source : source_identities) {
                if (sha256_file(source.path) != source.sha256) {
                    fail("MTP-bulk qualifier source changed during qualification");
                }
            }
            write_mtp_bulk_json(output_path, tokens, executable, executable_sha256,
                                source_identities, device, resources, evidence);
            std::cout << "MTP-bulk W8 selected-chunk qualification: PASS\n";
            return 0;
        }
        device = select_r9700();
        int runtime_version = 0;
        int driver_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK(hipDriverGetVersion(&driver_version));
        if (prefill_regression || lds_scope_regression || lds_scope_benchmark) {
            HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_kernel_resources(&resources));
            if (resources.prefill_cta_registers > 96 ||
                resources.prefill_cta_static_shared_bytes != 4352 ||
                resources.prefill_cta_local_bytes != 0) {
                fail("A8W8 prefill CTA resource gate requires <=96 VGPR, exactly 4352B "
                     "LDS, and no scratch");
            }
            if ((lds_scope_regression || lds_scope_benchmark) &&
                (resources.prefill_cta_global_inv_qualification_registers > 96 ||
                 resources.prefill_cta_global_inv_qualification_static_shared_bytes != 4352 ||
                 resources.prefill_cta_global_inv_qualification_local_bytes != 0)) {
                fail("A8W8 LDS-scope CTA requires <=96 VGPR, exactly 4352B LDS, no scratch");
            }
            const std::array<Shape, 4> prefill_shapes{
                     Shape{7168, 5120, 16, "text_attention_gate_value"},
                     Shape{12288, 5120, 48, "text_gdn_value_z"},
                     Shape{5120, 17408, 65, "text_mlp_down"},
                     Shape{5120, 6144, 65, "text_attention_gdn_output"}};
            const PrefillResult tail_qualification = qualify_prefill_width(
                Shape{65, 193, 0, "exact_tail"}, 129, true, false,
                lds_scope_regression || lds_scope_benchmark);
            std::vector<LdsScopeEvidence> evidence;
            for (const Shape shape : prefill_shapes) {
                for (const std::uint32_t tokens : {1024U, 2048U, 4096U, 8192U}) {
                    const PrefillResult result =
                        qualify_prefill_width(shape, tokens, false, false,
                                              lds_scope_regression || lds_scope_benchmark);
                    if (result.maximum_bf16_steps > 2U) {
                        fail("A8W8 cooperative prefill CTA regression differs from oracle");
                    }
                    if (lds_scope_benchmark) evidence.push_back({result, {}});
                }
            }
            if (tail_qualification.maximum_bf16_steps > 2U) fail("tail qualification evidence differs");
            if (lds_scope_benchmark) {
                for (LdsScopeEvidence& row : evidence) {
                    (void)qualify_prefill_width(row.numerical.shape, row.numerical.tokens,
                                                false, true, true, &row.timing);
                }
                if (read_text(kPowerProfilePath) != "auto") {
                    fail("LDS-scope physical A/B did not retain auto power");
                }
                if (sha256_file(executable) != executable_sha256) {
                    fail("qualification executable changed during physical A/B");
                }
                for (const SourceIdentity& source : source_identities) {
                    if (sha256_file(source.path) != source.sha256) {
                        fail("qualification source changed during physical A/B");
                    }
                }
                write_lds_scope_report(output_path, executable, executable_sha256, source_root,
                                       source_identities, device, runtime_version, driver_version,
                                       resources, evidence);
                std::cout << "A8W8 LDS-scope physical A/B: PASS\n";
                return 0;
            }
            std::cout << (lds_scope_regression
                              ? "A8W8 LDS-scope/production CTA bit-exact regression: PASS\n"
                              : "A8W8 cooperative prefill CTA regression: PASS\n");
            return 0;
        }
        qualify_malformed_and_nonfinite();
        qualify_tensor_dispatch();
        if (tensor_only) {
            HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_kernel_resources(&resources));
            write_tensor_json(output_path, device, resources, true, {});
            std::cout << "w8a8_wmma_tensor_dispatch: PASS\n";
            return 0;
        }
        for (const Shape& shape : kMixedW8Shapes) {
            const std::uint32_t padded = padded_columns(shape.columns);
            DeviceBuffer<std::int8_t> weights(
                static_cast<std::size_t>(shape.rows) * padded);
            DeviceBuffer<std::uint16_t> scales(
                static_cast<std::size_t>(shape.rows) * (padded / kGroup));
            initialize_weights(weights, scales, shape, padded);
            std::vector<std::uint32_t> widths{3U, 4U, 5U, 8U, 16U, 32U, 64U};
            if (shape.rows == 7168U && shape.columns == 5120U) {
                widths = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 16U, 32U, 64U, 128U};
            }
            for (const std::uint32_t tokens : widths) {
                Result result = qualify_width(shape, tokens, weights, scales);
                std::cout << std::fixed << std::setprecision(5)
                          << "shape=[" << shape.rows << ',' << shape.columns << "] T=" << tokens
                          << " fp64_error=" << result.maximum_formula_error
                          << " source_error=" << result.maximum_source_error
                          << " a8_ms=" << result.a8_median_ms
                          << " exact_ms=" << result.exact_median_ms
                          << " speedup=" << result.exact_median_ms / result.a8_median_ms << '\n';
                results.push_back(std::move(result));
            }
        }
        HIP_CHECK(ninfer::ops::r9700::linear::a8w8g32_kernel_resources(&resources));
        write_json(output_path, device, results, resources, true, {});
        std::cout << "w8a8_wmma_linear_qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        try {
            if (tensor_only) {
                write_tensor_json(output_path, device, resources, false, error.what());
            } else if (!prefill_regression && !lds_scope_regression &&
                       !lds_scope_benchmark) {
                write_json(output_path, device, results, resources, false, error.what());
            }
        } catch (...) {
        }
        std::cerr << "w8a8_wmma_linear_qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
