#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "ops/r9700/linear/r9700_linear.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

namespace linear = ninfer::ops::r9700::linear;

constexpr std::uint32_t kGroup = 64;
constexpr std::size_t kTrials = 7;
constexpr const char* kPowerProfilePath =
    "/sys/class/drm/card2/device/power_dpm_force_performance_level";
constexpr double kMaximumPerTupleRatio = 1.01;
constexpr std::uint32_t kMaxTokens = 128;
constexpr std::array<std::uint32_t, 10> kTokens{1, 2, 3, 4, 5, 6, 7, 8, 32, 128};
struct Shape {
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t all_q4_count;
    std::uint32_t mixed_q4_count;
    std::uint32_t dflash_q4_count = 0;
};

constexpr std::array<Shape, 22> kShapes{{
    {1152, 1152, 27, 0}, {1152, 1536, 1, 0}, {1152, 4304, 27, 0},
    {3456, 1152, 27, 27}, {4096, 5120, 48, 48}, {4304, 1152, 27, 27},
    {4608, 4608, 1, 0}, {5120, 4608, 1, 0}, {5120, 6144, 65, 0},
    {5120, 10240, 1, 0}, {5120, 17408, 65, 0, 5}, {7168, 5120, 32, 16},
    {12288, 5120, 48, 0}, {14336, 5120, 1, 0}, {34816, 5120, 65, 64, 5},
    {131072, 5120, 1, 1}, {248320, 5120, 2, 0},
    {5120, 25600, 0, 0, 1}, {6144, 5120, 0, 0, 5},
    {5120, 4096, 0, 0, 5}, {1280, 5120, 0, 0, 10},
    {256, 5120, 0, 0, 1},
}};

constexpr std::uint32_t dflash_matrix_count() {
    std::uint32_t count = 0;
    for (const Shape& shape : kShapes) count += shape.dflash_q4_count;
    return count;
}
static_assert(dflash_matrix_count() == 32U);

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
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        HIP_CHECK(hipMalloc(&pointer_, elements * sizeof(T)));
    }
    ~DeviceBuffer() { (void)hipFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
    std::size_t bytes() const { return elements_ * sizeof(T); }

private:
    T* pointer_ = nullptr;
    std::size_t elements_ = 0;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    ~Event() { (void)hipEventDestroy(event_); }
    hipEvent_t get() const { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

std::uint16_t half_bits(float value) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}

float half_value(std::uint16_t bits) {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

void set_nibble(std::vector<std::uint8_t>& packed, std::size_t index, int code) {
    const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
    const std::uint8_t mask = static_cast<std::uint8_t>(0x0fU << shift);
    packed[index >> 1U] = static_cast<std::uint8_t>(
        (packed[index >> 1U] & static_cast<std::uint8_t>(~mask)) |
        ((static_cast<unsigned>(code) & 0x0fU) << shift));
}

int signed_nibble(const std::vector<std::uint8_t>& packed, std::size_t index) {
    const unsigned code = (packed[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU;
    return code >= 8U ? static_cast<int>(code) - 16 : static_cast<int>(code);
}

int unsigned_nibble(const std::vector<std::uint8_t>& packed, std::size_t index) {
    return static_cast<int>((packed[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU);
}

struct HostActivation {
    std::vector<std::uint8_t> low;
    std::vector<std::uint8_t> high;
    std::vector<std::uint16_t> scales;
};

HostActivation quantize_host(const std::vector<hip_bfloat16>& input, std::uint32_t tokens,
                             std::uint32_t columns) {
    const std::uint32_t padded = (columns + 127U) / 128U * 128U;
    const std::uint32_t groups = padded / kGroup;
    HostActivation result{std::vector<std::uint8_t>(static_cast<std::size_t>(tokens) * padded / 2U),
                          std::vector<std::uint8_t>(static_cast<std::size_t>(tokens) * padded / 2U),
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
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint16_t bits = maximum == 0.0F ? 0U : half_bits(maximum / 127.0F);
            if (maximum != 0.0F && bits == 0U) bits = 1U;
            const float scale = half_value(bits);
            result.scales[static_cast<std::size_t>(token) * groups + group] = bits;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const std::size_t index =
                    static_cast<std::size_t>(token) * padded + column;
                const int code = bits == 0U ? 0 : std::clamp(static_cast<int>(std::nearbyint(
                    (column < columns ? static_cast<float>(input[
                         static_cast<std::size_t>(token) * columns + column]) : 0.0F) / scale)),
                    -127, 127);
                const int low = code & 15;
                const int high = (code - low) / 16;
                set_nibble(result.low, index, low);
                set_nibble(result.high, index, high);
            }
        }
    }
    return result;
}

std::vector<hip_bfloat16> make_input(std::uint32_t columns,
                                     std::uint32_t maximum_tokens = kMaxTokens) {
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(maximum_tokens) * columns,
                                    hip_bfloat16(0.0F));
    for (std::uint32_t token = 0; token < maximum_tokens; ++token) {
        const std::uint32_t groups = ((columns + 127U) / 128U * 128U) / kGroup;
        for (std::uint32_t group = 0; group < groups; ++group) {
            const float amplitude =
                0.5F + static_cast<float>((token * 7U + group * 3U) & 15U) / 16.0F;
            const std::array<float, 4> values{amplitude, -amplitude,
                                              amplitude * 0.5F, -amplitude * 0.25F};
            for (std::uint32_t slot = 0; slot < values.size(); ++slot) {
                const std::uint32_t lane = (group + slot * 16U) & 63U;
                const std::uint32_t column = group * kGroup + lane;
                if (column < columns) {
                    input[static_cast<std::size_t>(token) * columns + column] =
                        hip_bfloat16(values[slot]);
                }
            }
        }
    }
    return input;
}

struct PrefillResult {
    Shape shape{};
    std::uint32_t tokens = 0;
    bool exhaustive_oracle = false;
    double one_wave_ms = 0.0;
    double cooperative_cta_ms = 0.0;
    std::uint64_t maximum_bf16_steps = 0;
    std::array<double, kTrials> one_wave_samples{};
    std::array<double, kTrials> cooperative_cta_samples{};
};

struct LdsScopeTiming {
    std::array<double, kTrials> incumbent_forward{};
    std::array<double, kTrials> challenger_forward{};
    std::array<double, kTrials> incumbent_reverse{};
    std::array<double, kTrials> challenger_reverse{};
    std::array<double, kTrials> incumbent_balanced{};
    std::array<double, kTrials> challenger_balanced{};
    double incumbent_median = 0.0;
    double challenger_median = 0.0;
};

struct LdsScopeEvidence {
    PrefillResult numerical{};
    LdsScopeTiming timing{};
};

__global__ __launch_bounds__(64) void baseline_quantize_kernel(
    const hip_bfloat16* input, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scales, std::uint32_t* status, std::uint32_t columns,
    std::uint32_t padded_columns) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t column = group * kGroup + lane;
    float value = column < columns
                      ? static_cast<float>(input[static_cast<std::size_t>(token) * columns + column])
                      : 0.0F;
    if (!isfinite(value)) {
        atomicOr(status, 1U);
        value = 0.0F;
    }
    __shared__ float maxima[kGroup];
    __shared__ std::uint16_t scale_bits;
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (std::uint32_t width = kGroup / 2U; width != 0U; width >>= 1U) {
        if (lane < width) maxima[lane] = fmaxf(maxima[lane], maxima[lane + width]);
        __syncthreads();
    }
    if (lane == 0U) {
        std::uint16_t bits = maxima[0] == 0.0F
                                 ? 0U
                                 : __half_as_ushort(__float2half_rn(maxima[0] / 127.0F));
        if (maxima[0] != 0.0F && bits == 0U) bits = 1U;
        scale_bits = bits;
        scales[static_cast<std::size_t>(token) * (padded_columns / kGroup) + group] = bits;
    }
    __syncthreads();
    if (lane < kGroup / 2U) {
        const std::uint32_t column0 = group * kGroup + lane * 2U;
        const std::uint32_t column1 = column0 + 1U;
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int code0 = scale == 0.0F || column0 >= columns ? 0 : max(-127, min(127,
            __float2int_rn(static_cast<float>(input[
                static_cast<std::size_t>(token) * columns + column0]) / scale)));
        const int code1 = scale == 0.0F || column1 >= columns ? 0 : max(-127, min(127,
            __float2int_rn(static_cast<float>(input[
                static_cast<std::size_t>(token) * columns + column1]) / scale)));
        const int low0 = code0 & 15;
        const int low1 = code1 & 15;
        const int high0 = (code0 - low0) / 16;
        const int high1 = (code1 - low1) / 16;
        const std::size_t packed =
            (static_cast<std::size_t>(token) * padded_columns + column0) / 2U;
        low_codes[packed] = static_cast<std::uint8_t>(low0 | (low1 << 4));
        high_codes[packed] = static_cast<std::uint8_t>((high0 & 15) | ((high1 & 15) << 4));
    }
}

__global__ void fill_u16(std::uint16_t* values, std::size_t count, std::uint16_t bits) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = bits;
}

int prefill_weight_code(std::uint32_t row, std::uint32_t column) {
    return static_cast<int>((row * 3U + column * 5U + 1U) & 15U) - 8;
}

std::uint16_t prefill_weight_scale_bits(std::uint32_t row, std::uint32_t group) {
    return static_cast<std::uint16_t>(0x1c00U + ((row + group * 3U) & 3U) * 0x0400U);
}

__global__ void poison_q4_padded_tail(std::uint8_t* low, std::uint8_t* high,
                                      std::uint32_t tokens, std::uint32_t columns,
                                      std::uint32_t padded_columns) {
    const std::size_t packed = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t packed_count = static_cast<std::size_t>(tokens) * padded_columns / 2U;
    if (packed >= packed_count) return;
    const std::uint32_t column = static_cast<std::uint32_t>((packed * 2U) % padded_columns);
    if (column >= columns) {
        low[packed] = 0x11U;
        high[packed] = 0x11U;
    } else if (column + 1U >= columns) {
        low[packed] = static_cast<std::uint8_t>((low[packed] & 0x0fU) | 0x10U);
        high[packed] = static_cast<std::uint8_t>((high[packed] & 0x0fU) | 0x10U);
    }
}

template <typename Launch>
double event_ms(unsigned iterations, Launch&& launch, unsigned warmups = 2U) {
    for (unsigned warmup = 0; warmup < warmups; ++warmup) HIP_CHECK(launch());
    HIP_CHECK(hipDeviceSynchronize());
    Event begin;
    Event end;
    HIP_CHECK(hipEventRecord(begin.get()));
    for (unsigned iteration = 0; iteration < iterations; ++iteration) HIP_CHECK(launch());
    HIP_CHECK(hipEventRecord(end.get()));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float elapsed = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed, begin.get(), end.get()));
    return static_cast<double>(elapsed) / iterations;
}

double median(std::array<double, kTrials> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

std::uint64_t bf16_steps(hip_bfloat16 lhs, hip_bfloat16 rhs) {
    return lhs.data > rhs.data ? lhs.data - rhs.data : rhs.data - lhs.data;
}

struct Result {
    Shape shape{};
    std::uint32_t tokens = 0;
    double baseline_ms = 0.0;
    double selected_ms = 0.0;
    std::uint64_t maximum_bf16_steps = 0;
    std::array<double, kTrials> baseline_samples{};
    std::array<double, kTrials> selected_samples{};
};

void write_samples(std::ostream& out, const std::array<double, kTrials>& values);

std::vector<Result> run_shape(Shape shape) {
    const std::uint32_t padded_columns = (shape.columns + 127U) / 128U * 128U;
    const std::uint32_t groups = padded_columns / kGroup;
    const std::size_t weight_elements =
        static_cast<std::size_t>(shape.rows) * padded_columns;
    const std::size_t weight_scale_count = static_cast<std::size_t>(shape.rows) * groups;
    const std::vector<hip_bfloat16> input = make_input(shape.columns);
    const HostActivation expected = quantize_host(input, kMaxTokens, shape.columns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> d_weight_codes(weight_elements / 2U);
    DeviceBuffer<std::uint16_t> d_weight_scales(weight_scale_count);
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(kMaxTokens) * shape.rows);
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(kMaxTokens, shape.columns);
    DeviceBuffer<std::uint8_t> d_workspace(workspace_bytes);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_weight_codes.get(), 0x97, d_weight_codes.bytes()));
    constexpr std::uint16_t weight_scale_bits = 0x2000U; // exactly 0.0078125
    fill_u16<<<(weight_scale_count + 255U) / 256U, 256>>>(
        d_weight_scales.get(), weight_scale_count, weight_scale_bits);
    HIP_CHECK(hipGetLastError());

    std::vector<Result> results;
    for (std::uint32_t tokens : kTokens) {
        const std::size_t active_workspace_bytes =
            linear::a8q4g64_activation_workspace_capacity_bytes(tokens, shape.columns);
        linear::A8G64ActivationWorkspace workspace{};
        HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
            d_workspace.get(), active_workspace_bytes, tokens, shape.columns, &workspace));
        const linear::A8Q4G64LinearArgs wmma{
            workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
            workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
            workspace.status, d_weight_codes.get(), d_weight_codes.bytes(),
            d_weight_scales.get(), d_weight_scales.bytes(), d_output.get(), tokens,
            shape.rows, shape.columns, padded_columns};
        const linear::A8Q4G64CandidateArgs selected{
            d_input.get(), d_weight_codes.get(), d_weight_codes.bytes(),
            d_weight_scales.get(), d_weight_scales.bytes(), d_workspace.get(),
            active_workspace_bytes, d_output.get(), tokens, shape.rows, shape.columns,
            padded_columns};
        const auto launch_baseline = [&] {
            hipError_t status = hipMemsetAsync(workspace.status, 0, sizeof(std::uint32_t));
            if (status != hipSuccess) return status;
            const dim3 grid(groups, tokens);
            hipLaunchKernelGGL(baseline_quantize_kernel, grid, dim3(kGroup), 0, nullptr,
                               d_input.get(), workspace.low_codes, workspace.high_codes,
                               workspace.scales, workspace.status, shape.columns,
                               padded_columns);
            status = hipGetLastError();
            return status == hipSuccess ? linear::a8q4g64_linear_wmma32(wmma, nullptr) : status;
        };
        const auto launch_selected = [&] {
            return linear::a8q4g64_linear_candidate(selected, nullptr);
        };

        std::array<double, kTrials> baseline_samples{};
        std::array<double, kTrials> selected_samples{};
        const unsigned iterations = tokens <= 8U ? 12U : (tokens == 32U ? 5U : 2U);
        for (std::size_t trial = 0; trial < kTrials; ++trial) {
            if ((trial & 1U) == 0U) {
                baseline_samples[trial] = event_ms(iterations, launch_baseline);
                selected_samples[trial] = event_ms(iterations, launch_selected);
            } else {
                selected_samples[trial] = event_ms(iterations, launch_selected);
                baseline_samples[trial] = event_ms(iterations, launch_baseline);
            }
        }

        HIP_CHECK(launch_selected());
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<hip_bfloat16> output(static_cast<std::size_t>(tokens) * shape.rows);
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), output.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        std::uint64_t maximum_steps = 0;
        const std::array<std::uint32_t, 8> rows{
            0U, 1U, 15U, 16U, shape.rows / 3U, shape.rows / 2U,
            shape.rows - 2U, shape.rows - 1U};
        const std::array<std::uint32_t, 4> token_samples{
            0U, tokens / 3U, tokens / 2U, tokens - 1U};
        for (std::uint32_t token : token_samples) {
            for (std::uint32_t row : rows) {
                double sum = 0.0;
                for (std::uint32_t group = 0; group < groups; ++group) {
                    std::int32_t integer_dot = 0;
                    for (std::uint32_t slot = 0; slot < 4U; ++slot) {
                        const std::uint32_t lane = (group + slot * 16U) & 63U;
                        const std::size_t index = static_cast<std::size_t>(token) * padded_columns +
                                                  group * kGroup + lane;
                        const int activation_code = unsigned_nibble(expected.low, index) +
                                                    16 * signed_nibble(expected.high, index);
                        const int weight_code = ((group * kGroup + lane) & 1U) == 0U ? 7 : -7;
                        integer_dot += activation_code * weight_code;
                    }
                    const double activation_scale = half_value(
                        expected.scales[static_cast<std::size_t>(token) * groups + group]);
                    sum = std::fma(static_cast<double>(integer_dot),
                                   activation_scale * half_value(weight_scale_bits), sum);
                }
                const hip_bfloat16 oracle(static_cast<float>(sum));
                const hip_bfloat16 actual =
                    output[static_cast<std::size_t>(token) * shape.rows + row];
                maximum_steps = std::max(maximum_steps, bf16_steps(actual, oracle));
                if (bf16_steps(actual, oracle) > 2U) {
                    std::ostringstream message;
                    message << "oracle mismatch N=" << shape.rows << " K=" << shape.columns
                            << " T=" << tokens << " row=" << row << " token=" << token;
                    fail(message.str());
                }
            }
        }

        std::vector<std::uint8_t> low(workspace.low_code_bytes);
        std::vector<std::uint8_t> high(workspace.high_code_bytes);
        std::vector<std::uint16_t> scales(workspace.scale_bytes / sizeof(std::uint16_t));
        std::uint32_t status = ~0U;
        HIP_CHECK(hipMemcpy(low.data(), workspace.low_codes, low.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(high.data(), workspace.high_codes, high.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(scales.data(), workspace.scales, workspace.scale_bytes,
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
        const std::size_t plane_prefix = static_cast<std::size_t>(tokens) * padded_columns / 2U;
        const std::size_t scale_prefix = static_cast<std::size_t>(tokens) * groups;
        if (status != 0U || !std::equal(low.begin(), low.end(), expected.low.begin()) ||
            !std::equal(high.begin(), high.end(), expected.high.begin()) ||
            !std::equal(scales.begin(), scales.end(), expected.scales.begin()) ||
            low.size() != plane_prefix || scales.size() != scale_prefix) {
            fail("exact A8 activation codec mismatch in shape sweep");
        }

        results.push_back(Result{shape, tokens, median(baseline_samples),
                                 median(selected_samples), maximum_steps,
                                 baseline_samples, selected_samples});
    }
    return results;
}

PrefillResult run_prefill_case(Shape shape, std::uint32_t tokens,
                               bool exhaustive_oracle, bool collect_timing,
                               bool compare_lds_scope = false,
                               LdsScopeTiming* lds_scope_timing = nullptr,
                               bool compare_n128 = false,
                               bool compare_m128n128 = false,
                               bool compare_pingpong = false,
                               bool compare_scalar_base = false) {
    if (static_cast<unsigned>(compare_lds_scope) + static_cast<unsigned>(compare_n128) +
            static_cast<unsigned>(compare_m128n128) +
            static_cast<unsigned>(compare_pingpong) +
            static_cast<unsigned>(compare_scalar_base) > 1U) {
        fail("prefill comparison routes are exclusive");
    }
    const std::uint32_t padded_columns = (shape.columns + 127U) / 128U * 128U;
    const std::uint32_t groups = padded_columns / kGroup;
    const std::size_t weight_elements =
        static_cast<std::size_t>(shape.rows) * padded_columns;
    const std::size_t weight_scale_count = static_cast<std::size_t>(shape.rows) * groups;
    const std::vector<hip_bfloat16> input = make_input(shape.columns, tokens);
    const HostActivation expected = quantize_host(input, tokens, shape.columns);
    std::vector<std::uint8_t> logical_weight_codes(weight_elements / 2U);
    std::vector<std::uint16_t> logical_weight_scales(weight_scale_count);
    std::vector<std::uint8_t> stored_weight_codes(weight_elements / 2U);
    std::vector<std::uint16_t> stored_weight_scales(weight_scale_count);
    std::array<std::vector<std::uint8_t>, 16> code_row_patterns;
    std::array<std::vector<std::uint16_t>, 4> scale_row_patterns;
    for (std::uint32_t row_pattern = 0; row_pattern < code_row_patterns.size();
         ++row_pattern) {
        auto& pattern = code_row_patterns[row_pattern];
        pattern.resize(padded_columns / 2U);
        for (std::uint32_t column = 0; column < padded_columns; column += 2U) {
            const int low = prefill_weight_code(row_pattern, column);
            const int high = prefill_weight_code(row_pattern, column + 1U);
            pattern[column / 2U] =
                static_cast<std::uint8_t>((low & 15) | ((high & 15) << 4));
        }
    }
    for (std::uint32_t row_pattern = 0; row_pattern < scale_row_patterns.size();
         ++row_pattern) {
        auto& pattern = scale_row_patterns[row_pattern];
        pattern.resize(groups);
        for (std::uint32_t group = 0; group < groups; ++group) {
            pattern[group] = prefill_weight_scale_bits(row_pattern, group);
        }
    }
    for (std::uint32_t row = 0; row < shape.rows; ++row) {
        std::copy(code_row_patterns[row & 15U].begin(),
                  code_row_patterns[row & 15U].end(),
                  logical_weight_codes.begin() +
                      static_cast<std::size_t>(row) * padded_columns / 2U);
        std::copy(scale_row_patterns[row & 3U].begin(),
                  scale_row_patterns[row & 3U].end(),
                  logical_weight_scales.begin() + static_cast<std::size_t>(row) * groups);
        for (std::uint32_t group = 0; group < groups; ++group) {
            const std::size_t stored_scale =
                ((static_cast<std::size_t>(row) >> 4U) * groups + group) * 16U +
                (row & 15U);
            stored_weight_scales[stored_scale] =
                logical_weight_scales[static_cast<std::size_t>(row) * groups + group];
            for (std::uint32_t pair = 0; pair < 4U; ++pair) {
                const std::size_t stored_word =
                    (((static_cast<std::size_t>(row) >> 4U) * groups + group) * 4U + pair) *
                        16U +
                    (row & 15U);
                const std::size_t logical_word =
                    (static_cast<std::size_t>(row) * padded_columns + group * kGroup +
                     pair * 16U) /
                    2U;
                std::copy_n(logical_weight_codes.begin() + logical_word, 8U,
                            stored_weight_codes.begin() + stored_word * 8U);
            }
        }
    }
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> d_weight_codes(weight_elements / 2U);
    DeviceBuffer<std::uint16_t> d_weight_scales(weight_scale_count);
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(tokens) * shape.rows);
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, shape.columns);
    DeviceBuffer<std::uint8_t> d_workspace(workspace_bytes);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_codes.get(), stored_weight_codes.data(),
                        stored_weight_codes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_scales.get(), stored_weight_scales.data(),
                        stored_weight_scales.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    linear::A8G64ActivationWorkspace workspace{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
        d_workspace.get(), workspace_bytes, tokens, shape.columns, &workspace));
    HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    const linear::A8Q4G64LinearArgs args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, d_weight_codes.get(), d_weight_codes.bytes(),
        d_weight_scales.get(), d_weight_scales.bytes(), d_output.get(), tokens,
        shape.rows, shape.columns, padded_columns};
    const auto launch_one_wave = [&] {
        return linear::a8q4g64_linear_wmma32(args, nullptr);
    };
    const auto launch_cooperative = [&] {
        return linear::a8q4g64_linear_prefill_cta(args, nullptr);
    };
    const auto launch_m64n128_regression = [&] {
        return linear::a8q4g64_linear_prefill_cta_m64n128_regression(args, nullptr);
    };
    const auto launch_global_inv_incumbent = [&] {
        return linear::a8q4g64_linear_prefill_cta_global_inv_qualification(args, nullptr);
    };
    const auto launch_m64n64_regression = [&] {
        return linear::a8q4g64_linear_prefill_cta_m64n64_regression(args, nullptr);
    };
    const auto launch_m128n128 = [&] {
        return linear::a8q4g64_linear_prefill_cta_m128n128_qualification(args, nullptr);
    };
    const auto launch_pingpong = [&] {
        return linear::a8q4g64_linear_prefill_cta(args, nullptr);
    };
    const auto launch_scalar_base_args = [&](const linear::A8Q4G64LinearArgs& value) {
#if defined(NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION)
        return linear::a8q4g64_linear_prefill_cta_scalar_base_qualification(value, nullptr);
#else
        (void)value;
        return hipErrorNotSupported;
#endif
    };
    const auto launch_scalar_base = [&] {
        return launch_scalar_base_args(args);
    };
    const auto launch_comparison = [&] {
        if (compare_scalar_base) return launch_scalar_base();
        if (compare_pingpong) return launch_pingpong();
        if (compare_m128n128) return launch_m128n128();
        return compare_n128 ? launch_m64n64_regression() : launch_global_inv_incumbent();
    };
    if (exhaustive_oracle && !collect_timing) {
        linear::A8Q4G64LinearArgs misaligned = args;
        ++misaligned.activation_low_codes;
        if (linear::a8q4g64_linear_prefill_cta(misaligned, nullptr) !=
            hipErrorInvalidValue) {
            fail("A8Q4 prefill CTA accepted a non-dword-aligned packed code plane");
        }
        if ((compare_lds_scope || compare_n128 || compare_m128n128 || compare_pingpong ||
             compare_scalar_base) &&
            (compare_scalar_base
                 ? launch_scalar_base_args(misaligned)
             : compare_pingpong
                 ? linear::a8q4g64_linear_prefill_cta(misaligned, nullptr)
                 : compare_m128n128
                 ? linear::a8q4g64_linear_prefill_cta_m128n128_qualification(
                       misaligned, nullptr)
                 : compare_n128
                 ? linear::a8q4g64_linear_prefill_cta_m64n64_regression(misaligned, nullptr)
                 : linear::a8q4g64_linear_prefill_cta_global_inv_qualification(
                       misaligned, nullptr)) != hipErrorInvalidValue) {
            fail("A8Q4 comparison route accepted a non-dword-aligned code plane");
        }
        const std::uint32_t bad_status = linear::Q4G64ActivationNonfinite;
        HIP_CHECK(hipMemcpy(workspace.status, &bad_status, sizeof(bad_status),
                            hipMemcpyHostToDevice));
        HIP_CHECK(compare_lds_scope ? launch_m64n64_regression()
                  : (compare_m128n128 || compare_pingpong)
                      ? launch_m64n128_regression()
                      : launch_cooperative());
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<hip_bfloat16> poisoned(static_cast<std::size_t>(tokens) * shape.rows);
        HIP_CHECK(hipMemcpy(poisoned.data(), d_output.get(),
                            poisoned.size() * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
        if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
                return value.data == 0x7fc1U;
            })) {
            fail("A8Q4 prefill CTA did not poison every T>128 output for nonzero status");
        }
        if (compare_lds_scope || compare_n128 || compare_m128n128 || compare_pingpong ||
            compare_scalar_base) {
            HIP_CHECK(hipMemset(d_output.get(), 0, d_output.bytes()));
            HIP_CHECK(launch_comparison());
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(poisoned.data(), d_output.get(),
                                poisoned.size() * sizeof(hip_bfloat16),
                                hipMemcpyDeviceToHost));
            if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
                    return value.data == 0x7fc1U;
                })) {
                fail("A8Q4 comparison route did not preserve status poisoning");
            }
        }
        HIP_CHECK(hipMemset(workspace.status, 0, sizeof(std::uint32_t)));
        const std::size_t packed_count =
            static_cast<std::size_t>(tokens) * padded_columns / 2U;
        poison_q4_padded_tail<<<(packed_count + 255U) / 256U, 256>>>(
            workspace.low_codes, workspace.high_codes, tokens, shape.columns,
            padded_columns);
        HIP_CHECK(hipGetLastError());
    }
    std::array<double, kTrials> one_wave_samples{};
    std::array<double, kTrials> cooperative_samples{};
    if (collect_timing) {
        if (lds_scope_timing != nullptr && !compare_lds_scope && !compare_n128 &&
            !compare_m128n128 && !compare_pingpong && !compare_scalar_base) {
            fail("prefill timing requires an explicit comparison route");
        }
        HIP_CHECK(launch_cooperative());
        HIP_CHECK(lds_scope_timing == nullptr ? launch_one_wave() : launch_comparison());
        HIP_CHECK(hipDeviceSynchronize());
        if (lds_scope_timing != nullptr) {
            for (std::size_t trial = 0; trial < kTrials; ++trial) {
                lds_scope_timing->incumbent_forward[trial] =
                    (compare_m128n128 || compare_pingpong || compare_scalar_base)
                    ? event_ms(1U, launch_m64n128_regression, 0U)
                    : compare_n128 ? event_ms(1U, launch_m64n64_regression, 0U)
                                   : event_ms(1U, launch_global_inv_incumbent, 0U);
                lds_scope_timing->challenger_forward[trial] = compare_scalar_base
                    ? event_ms(1U, launch_scalar_base, 0U)
                    : compare_pingpong
                    ? event_ms(1U, launch_pingpong, 0U)
                    : compare_n128
                    ? event_ms(1U, launch_m64n128_regression, 0U)
                    : compare_m128n128 ? event_ms(1U, launch_m128n128, 0U)
                    : event_ms(1U, launch_m64n64_regression, 0U);
                lds_scope_timing->challenger_reverse[trial] = compare_scalar_base
                    ? event_ms(1U, launch_scalar_base, 0U)
                    : compare_pingpong
                    ? event_ms(1U, launch_pingpong, 0U)
                    : compare_n128
                    ? event_ms(1U, launch_m64n128_regression, 0U)
                    : compare_m128n128 ? event_ms(1U, launch_m128n128, 0U)
                    : event_ms(1U, launch_m64n64_regression, 0U);
                lds_scope_timing->incumbent_reverse[trial] =
                    (compare_m128n128 || compare_pingpong || compare_scalar_base)
                    ? event_ms(1U, launch_m64n128_regression, 0U)
                    : compare_n128 ? event_ms(1U, launch_m64n64_regression, 0U)
                                   : event_ms(1U, launch_global_inv_incumbent, 0U);
                lds_scope_timing->incumbent_balanced[trial] =
                    (lds_scope_timing->incumbent_forward[trial] +
                     lds_scope_timing->incumbent_reverse[trial]) / 2.0;
                lds_scope_timing->challenger_balanced[trial] =
                    (lds_scope_timing->challenger_forward[trial] +
                     lds_scope_timing->challenger_reverse[trial]) / 2.0;
            }
            lds_scope_timing->incumbent_median =
                median(lds_scope_timing->incumbent_balanced);
            lds_scope_timing->challenger_median =
                median(lds_scope_timing->challenger_balanced);
            return PrefillResult{shape, tokens, exhaustive_oracle,
                                 lds_scope_timing->incumbent_median,
                                 lds_scope_timing->challenger_median, 0U,
                                 lds_scope_timing->incumbent_balanced,
                                 lds_scope_timing->challenger_balanced};
        }
        for (std::size_t trial = 0; trial < kTrials; ++trial) {
            if ((trial & 1U) == 0U) {
                one_wave_samples[trial] = event_ms(1U, launch_one_wave, 0U);
                cooperative_samples[trial] = event_ms(1U, launch_cooperative, 0U);
            } else {
                cooperative_samples[trial] = event_ms(1U, launch_cooperative, 0U);
                one_wave_samples[trial] = event_ms(1U, launch_one_wave, 0U);
            }
        }
        return PrefillResult{shape, tokens, exhaustive_oracle,
                             median(one_wave_samples), median(cooperative_samples), 0U,
                             one_wave_samples, cooperative_samples};
    }
    HIP_CHECK(compare_lds_scope ? launch_m64n64_regression()
              : (compare_m128n128 || compare_pingpong)
                  ? launch_m64n128_regression()
                  : launch_cooperative());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<hip_bfloat16> output(static_cast<std::size_t>(tokens) * shape.rows);
    HIP_CHECK(hipMemcpy(output.data(), d_output.get(), output.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    if (compare_lds_scope || compare_n128 || compare_m128n128 || compare_pingpong ||
        compare_scalar_base) {
        constexpr std::size_t kComparisonElements = 1U << 20U;
        std::vector<hip_bfloat16> challenger(
            std::min(kComparisonElements, output.size()));
        for (std::size_t begin = 0; begin < output.size(); begin += challenger.size()) {
            const std::size_t count = std::min(challenger.size(), output.size() - begin);
            for (std::size_t offset = 0; offset < count; ++offset) {
                challenger[offset].data =
                    static_cast<std::uint16_t>(output[begin + offset].data ^ 0xffffU);
            }
            HIP_CHECK(hipMemcpy(d_output.get() + begin, challenger.data(),
                                count * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
        }
        HIP_CHECK(launch_comparison());
        HIP_CHECK(hipDeviceSynchronize());
        for (std::size_t begin = 0; begin < output.size(); begin += challenger.size()) {
            const std::size_t count = std::min(challenger.size(), output.size() - begin);
            HIP_CHECK(hipMemcpy(challenger.data(), d_output.get() + begin,
                                count * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
            for (std::size_t offset = 0; offset < count; ++offset) {
                if (challenger[offset].data != output[begin + offset].data) {
                    fail("A8Q4 comparison route is not bit-exact with production CTA");
                }
                output[begin + offset] = challenger[offset];
            }
        }
    }

    const std::array<std::uint32_t, 8> sampled_rows{
        0U, 1U, 15U, 16U, shape.rows / 3U, shape.rows / 2U,
        shape.rows - 2U, shape.rows - 1U};
    const std::array<std::uint32_t, 8> sampled_tokens{
        0U, 31U, 32U, 63U, 64U, tokens / 3U, tokens / 2U, tokens - 1U};
    std::uint64_t maximum_steps = 0;
    const auto check_output = [&](std::uint32_t token, std::uint32_t row) {
        double sum = 0.0;
        for (std::uint32_t group = 0; group < groups; ++group) {
            std::int32_t integer_dot = 0;
            const std::size_t activation_base =
                static_cast<std::size_t>(token) * padded_columns + group * kGroup;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const int activation_code = unsigned_nibble(
                    expected.low, activation_base + lane) + 16 * signed_nibble(
                    expected.high, activation_base + lane);
                const int weight_code = signed_nibble(
                    logical_weight_codes,
                    static_cast<std::size_t>(row) * padded_columns + group * kGroup + lane);
                integer_dot += activation_code * weight_code;
            }
            const double activation_scale = half_value(
                expected.scales[static_cast<std::size_t>(token) * groups + group]);
            const double weight_scale = half_value(
                logical_weight_scales[static_cast<std::size_t>(row) * groups + group]);
            sum = std::fma(static_cast<double>(integer_dot),
                           activation_scale * weight_scale, sum);
        }
        const hip_bfloat16 oracle(static_cast<float>(sum));
        const hip_bfloat16 actual =
            output[static_cast<std::size_t>(token) * shape.rows + row];
        const std::uint64_t steps = bf16_steps(actual, oracle);
        maximum_steps = std::max(maximum_steps, steps);
        if (steps > 2U) {
            std::ostringstream message;
            message << "prefill CTA oracle mismatch N=" << shape.rows
                    << " K=" << shape.columns << " T=" << tokens
                    << " row=" << row << " token=" << token
                    << " bf16_steps=" << steps;
            fail(message.str());
        }
    };
    if (exhaustive_oracle) {
        for (std::uint32_t token = 0; token < tokens; ++token) {
            for (std::uint32_t row = 0; row < shape.rows; ++row) {
                check_output(token, row);
            }
        }
    } else {
        for (std::uint32_t token : sampled_tokens) {
            for (std::uint32_t row : sampled_rows) check_output(token, row);
        }
    }
    return PrefillResult{shape, tokens, exhaustive_oracle,
                         median(one_wave_samples), median(cooperative_samples),
                         maximum_steps, one_wave_samples, cooperative_samples};
}

void write_samples(std::ostream& out, const std::array<double, kTrials>& values) {
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) out << ',';
        out << values[index];
    }
    out << ']';
}

void write_json(const std::filesystem::path& path, const std::vector<Result>& results,
                const hipDeviceProp_t& device, int runtime_version) {
    std::ofstream out(path);
    if (!out) fail("cannot open output JSON");
    out << std::setprecision(17)
        << "{\n  \"schema_version\": 2,\n"
        << "  \"qualification\": \"a8q4_base_and_dflash2_shape_inventory\",\n"
        << "  \"pass\": true,\n"
        << "  \"hardware\": {\"architecture\": \"" << device.gcnArchName
        << "\", \"wave_size\": " << device.warpSize
        << ", \"vram_bytes\": " << device.totalGlobalMem
        << ", \"hip_runtime_version\": " << runtime_version << "},\n"
        << "  \"contract\": {\"activation\": \"BF16->A8G64_F16S\", "
           "\"weight\": \"Q4G64_F16S row-split\", \"output\": \"BF16\"},\n"
        << "  \"oracle\": {\"kind\": \"independent FP64 complete quantized formula\", "
           "\"sample_rows_per_shape\": 8, \"sample_tokens_per_extent\": 4, "
           "\"maximum_bf16_steps_allowed\": 2},\n"
        << "  \"baseline\": \"64-thread/LDS A8 quantize plus selected WMMA\",\n"
        << "  \"selected\": \"wave32 shuffle A8 quantize plus selected WMMA\",\n"
        << "  \"timing\": {\"method\": \"unprofiled HIP events, 7 interleaved trials, median\", "
           "\"tokens\": [1,2,3,4,5,6,7,8,32,128]},\n"
        << "  \"shapes\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const Result& result = results[index];
        out << "    {\"rows\":" << result.shape.rows
            << ",\"columns\":" << result.shape.columns
            << ",\"all_q4_count\":" << result.shape.all_q4_count
            << ",\"mixed_q4_count\":" << result.shape.mixed_q4_count
            << ",\"dflash_q4_count\":" << result.shape.dflash_q4_count
            << ",\"tokens\":" << result.tokens
            << ",\"maximum_bf16_steps\":" << result.maximum_bf16_steps
            << ",\"baseline_ms\":" << result.baseline_ms
            << ",\"selected_ms\":" << result.selected_ms
            << ",\"selected_over_baseline\":" << result.selected_ms / result.baseline_ms
            << ",\"baseline_samples_ms\":";
        write_samples(out, result.baseline_samples);
        out << ",\"selected_samples_ms\":";
        write_samples(out, result.selected_samples);
        out << '}' << (index + 1U == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    if (!out) fail("failed writing output JSON");
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const char character : value) {
        if (character == '\\' || character == '"') out << '\\';
        out << character;
    }
    return out.str();
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

bool write_lds_scope_report(const std::filesystem::path& output,
                            const std::filesystem::path& executable,
                            std::string_view executable_sha256,
                            const std::filesystem::path& source_root,
                            const std::array<SourceIdentity, 4>& sources,
                            const hipDeviceProp_t& device, int runtime_version,
                            int driver_version,
                            const linear::A8Q4G64KernelResources& resources,
                            const std::vector<LdsScopeEvidence>& evidence,
                            bool n128, bool m128n128, bool pingpong) {
    if (evidence.empty()) fail("physical A/B report requires nonempty evidence");
    double weighted_incumbent = 0.0;
    double weighted_challenger = 0.0;
    std::vector<std::size_t> failing_tuples;
    const auto weight = [](const Shape& shape) -> std::uint32_t {
        if (shape.rows == 7168U && shape.columns == 5120U) return 32U;
        if (shape.rows == 4096U && shape.columns == 5120U) return 48U;
        if (shape.rows == 12288U && shape.columns == 5120U) return 48U;
        if (shape.rows == 5120U && shape.columns == 6144U) return 65U;
        if (shape.rows == 34816U && shape.columns == 5120U) return 65U;
        if (shape.rows == 5120U && shape.columns == 17408U) return 65U;
        if (shape.rows == 5120U && shape.columns == 10240U) return 1U;
        return 0U;
    };
    for (std::size_t index = 0; index < evidence.size(); ++index) {
        const LdsScopeEvidence& row = evidence[index];
        const double ratio = row.timing.challenger_median / row.timing.incumbent_median;
        if (!(ratio <= (pingpong ? 1.0 : kMaximumPerTupleRatio))) {
            failing_tuples.push_back(index);
        }
        if (row.numerical.tokens == 2048U) {
            const std::uint32_t calls = weight(row.numerical.shape);
            weighted_incumbent += calls * row.timing.incumbent_median;
            weighted_challenger += calls * row.timing.challenger_median;
        }
    }
    const bool weighted_raw_faster = weighted_challenger < weighted_incumbent;
    const bool weighted_faster = pingpong
        ? weighted_challenger * 1.5 <= weighted_incumbent
        : weighted_challenger < weighted_incumbent;
    const bool required_saving =
        !m128n128 || weighted_incumbent - weighted_challenger >= 150.0;
    const bool accepted = failing_tuples.empty() && weighted_faster && required_saving;
    if (!m128n128 && !pingpong && !failing_tuples.empty()) {
        fail("material prefill CTA tuple regression");
    }
    if (!m128n128 && !pingpong && !weighted_faster) {
        fail("prefill CTA challenger is not faster for weighted T2048 Text execution");
    }
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n  \"artifact_type\": \""
        << (pingpong ? "ninfer_r9700_prefill_cta_pingpong_ab"
            : m128n128 ? "ninfer_r9700_prefill_cta_m128n128_ab"
            : n128 ? "ninfer_r9700_prefill_cta_n128_ab"
                 : "ninfer_r9700_prefill_cta_lds_scope_ab") << "\",\n"
        << "  \"schema_version\": 1,\n  \"status\": \""
        << (accepted ? "passed" : "rejected") << "\",\n"
        << "  \"recipe\": \"" << (pingpong ? "q4-pingpong"
                                      : m128n128 ? "q4-m128n128"
                                      : n128 ? "q4-n128" : "q4") << "\",\n"
        << "  \"disposition\": \""
        << (accepted ? "qualification_only_unpromoted"
                     : "qualification_rejected_production_unchanged") << "\",\n"
        << "  \"production_state\": {\"dispatch_changed\": false, "
           "\"selected_route\": \"M64xN128-pingpong\"},\n"
        << "  \"timing\": {\"method\": \"unprofiled HIP events\", "
           "\"warmup_iterations_per_route\": 1, \"balanced_forward_reverse_pairs\": 7, "
           "\"balanced_sample\": \"mean of same-route forward and reverse measurements\", "
           "\"token_extents\": [1024,2048,4096,8192]},\n"
        << "  \"numerical_qualification\": {\"completed_before_timing\": true, "
           "\"incumbent_challenger_bit_exact\": true, "
           "\"active_output_coverage\": \"full rewrite and bit parity after non-incumbent poison\", "
           "\"independent_oracle\": \"FP64 represented quantized formula at sampled coordinates\", "
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
        << ((m128n128 || pingpong)
                ? resources.prefill_cta_m64n128_regression_registers
            : n128 ? resources.prefill_cta_m64n64_regression_registers
                 : resources.prefill_cta_global_inv_qualification_registers)
        << ", \"challenger_registers\": "
        << (pingpong ? resources.prefill_cta_registers
            : m128n128 ? resources.prefill_cta_m128n128_qualification_registers
            : n128 ? resources.prefill_cta_registers
                 : resources.prefill_cta_m64n64_regression_registers)
        << ", \"incumbent_static_shared_bytes\": "
        << ((m128n128 || pingpong)
                ? resources.prefill_cta_m64n128_regression_static_shared_bytes
            : n128 ? resources.prefill_cta_m64n64_regression_static_shared_bytes
                 : resources.prefill_cta_global_inv_qualification_static_shared_bytes)
        << ", \"challenger_static_shared_bytes\": "
        << (pingpong ? resources.prefill_cta_static_shared_bytes
            : m128n128 ? resources.prefill_cta_m128n128_qualification_static_shared_bytes
            : n128 ? resources.prefill_cta_static_shared_bytes
                 : resources.prefill_cta_m64n64_regression_static_shared_bytes)
        << ", \"incumbent_local_bytes\": "
        << ((m128n128 || pingpong)
                ? resources.prefill_cta_m64n128_regression_local_bytes
            : n128 ? resources.prefill_cta_m64n64_regression_local_bytes
                 : resources.prefill_cta_global_inv_qualification_local_bytes)
        << ", \"challenger_local_bytes\": "
        << (pingpong ? resources.prefill_cta_local_bytes
            : m128n128 ? resources.prefill_cta_m128n128_qualification_local_bytes
            : n128 ? resources.prefill_cta_local_bytes
                 : resources.prefill_cta_m64n64_regression_local_bytes)
        << ", \"challenger_max_threads_per_block\": "
        << (pingpong ? 512
            : m128n128
                ? resources.prefill_cta_m128n128_qualification_max_threads_per_block
                : 512)
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
        const auto emit = [&](const char* name, const auto& values) {
            out << ",\"" << name << "\":";
            write_samples(out, values);
        };
        emit("incumbent_forward_ms", row.timing.incumbent_forward);
        emit("incumbent_reverse_ms", row.timing.incumbent_reverse);
        emit("incumbent_balanced_ms", row.timing.incumbent_balanced);
        emit("challenger_forward_ms", row.timing.challenger_forward);
        emit("challenger_reverse_ms", row.timing.challenger_reverse);
        emit("challenger_balanced_ms", row.timing.challenger_balanced);
        out << ",\"incumbent_median_ms\":" << row.timing.incumbent_median
            << ",\"challenger_median_ms\":" << row.timing.challenger_median
            << ",\"challenger_over_incumbent\":"
            << row.timing.challenger_median / row.timing.incumbent_median << '}'
            << (index + 1U == evidence.size() ? "\n" : ",\n");
    }
    out << "  ],\n  \"failing_tuples\": [";
    for (std::size_t position = 0; position < failing_tuples.size(); ++position) {
        const auto& row = evidence[failing_tuples[position]];
        out << "{\"rows\":" << row.numerical.shape.rows
            << ",\"columns\":" << row.numerical.shape.columns
            << ",\"tokens\":" << row.numerical.tokens
            << ",\"challenger_over_incumbent\":"
            << row.timing.challenger_median / row.timing.incumbent_median
            << ",\"reason\":\"challenger_over_incumbent_exceeds_"
            << (pingpong ? "1.0" : "1.01") << "\"}"
            << (position + 1U == failing_tuples.size() ? "" : ",");
    }
    out << "],\n  \"decision\": {\"token_extent\": 2048, \"weights\": [";
    out << "{\"rows\":7168,\"columns\":5120,\"count\":32},"
        << "{\"rows\":4096,\"columns\":5120,\"count\":48},"
        << "{\"rows\":12288,\"columns\":5120,\"count\":48},"
        << "{\"rows\":5120,\"columns\":6144,\"count\":65},"
        << "{\"rows\":34816,\"columns\":5120,\"count\":65},"
        << "{\"rows\":5120,\"columns\":17408,\"count\":65},"
        << "{\"rows\":5120,\"columns\":10240,\"count\":1},"
        << "{\"rows\":1024,\"columns\":5120,\"count\":0}";
    out << "],"
        << "\"weighted_incumbent_ms\":" << weighted_incumbent
        << ",\"weighted_challenger_ms\":" << weighted_challenger
        << ",\"challenger_over_incumbent\":"
        << weighted_challenger / weighted_incumbent
        << ",\"maximum_per_tuple_ratio\":"
        << (pingpong ? 1.0 : 1.01) << ","
        << "\"no_material_per_tuple_regression\":"
        << (failing_tuples.empty() ? "true" : "false") << ","
        << "\"minimum_weighted_saving_ms\":" << (m128n128 ? 150 : 0) << ","
        << "\"minimum_weighted_speedup\":"
        << (pingpong ? 1.5 : 1.0) << ","
        << "\"weighted_saving_ms\":" << (weighted_incumbent - weighted_challenger) << ","
        << "\"weighted_challenger_faster\":"
        << (weighted_raw_faster ? "true" : "false") << ","
        << "\"required_speedup_met\":" << (weighted_faster ? "true" : "false")
        << ",\"accepted\":" << (accepted ? "true" : "false") << "}\n}\n";
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
    return accepted;
}

} // namespace

#ifndef NINFER_A8Q4_QUAL_NO_MAIN
int main(int argc, char** argv) {
    const bool prefill_regression =
        argc == 2 && std::string_view(argv[1]) == "--prefill-cta-regression";
    const bool lds_scope_regression =
        argc == 2 &&
        std::string_view(argv[1]) == "--prefill-cta-lds-scope-regression";
    const bool lds_scope_benchmark =
        argc == 4 &&
        std::string_view(argv[1]) == "--prefill-cta-lds-scope-benchmark" &&
        std::string_view(argv[2]) == "--out-json";
    const bool n128_regression =
        argc == 2 && std::string_view(argv[1]) == "--prefill-cta-m64n64-regression";
    constexpr bool n128_benchmark = false;
    const bool m128n128_regression =
        argc == 2 && std::string_view(argv[1]) == "--prefill-cta-m128n128-regression";
    const bool m128n128_benchmark =
        argc == 4 && std::string_view(argv[1]) == "--prefill-cta-m128n128-benchmark" &&
        std::string_view(argv[2]) == "--out-json";
    const bool pingpong_regression =
        argc == 2 && std::string_view(argv[1]) == "--prefill-cta-pingpong-regression";
    const bool pingpong_benchmark =
        argc == 4 && std::string_view(argv[1]) == "--prefill-cta-pingpong-benchmark" &&
        std::string_view(argv[2]) == "--out-json";
#if defined(NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION)
    const bool scalar_base_regression =
        argc == 2 && std::string_view(argv[1]) ==
                         "--prefill-cta-scalar-base-regression";
#else
    constexpr bool scalar_base_regression = false;
#endif
    const bool shape_sweep = argc == 3 && std::string_view(argv[1]) == "--out-json";
    if (!prefill_regression && !lds_scope_regression && !lds_scope_benchmark &&
        !n128_regression && !n128_benchmark && !m128n128_regression &&
        !m128n128_benchmark && !pingpong_regression && !pingpong_benchmark &&
        !scalar_base_regression && !shape_sweep) {
        std::cerr << "usage: a8q4_shape_sweep_qual "
                     "--out-json PATH | --prefill-cta-regression | "
                     "--prefill-cta-lds-scope-regression | "
                     "--prefill-cta-lds-scope-benchmark --out-json FRESH.json | "
                     "--prefill-cta-m64n64-regression | "
                     "--prefill-cta-m128n128-regression | "
                     "--prefill-cta-m128n128-benchmark --out-json FRESH.json | "
                     "--prefill-cta-pingpong-regression | "
                     "--prefill-cta-pingpong-benchmark --out-json FRESH.json"
#if defined(NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION)
                     " | --prefill-cta-scalar-base-regression"
#endif
                     "\n";
        return 2;
    }
    try {
        std::filesystem::path benchmark_output;
        std::filesystem::path executable;
        std::filesystem::path source_root;
        std::string executable_sha256;
        std::array<SourceIdentity, 4> source_identities{};
        if (lds_scope_benchmark || n128_benchmark || m128n128_benchmark ||
            pingpong_benchmark) {
            benchmark_output = argv[3];
            require_fresh_output(benchmark_output);
            if (read_text(kPowerProfilePath) != "auto") {
                fail("prefill CTA physical A/B requires auto power before device construction");
            }
            executable = std::filesystem::canonical("/proc/self/exe");
            source_root = std::filesystem::canonical(NINFER_SOURCE_DIR);
            executable_sha256 = sha256_file(executable);
            const std::array<std::pair<const char*, const char*>, 4> source_specs{{
                {"qualifier", "tools/r9700/a8q4_shape_sweep_qual.hip"},
                {"contract_header", "src/ops/r9700/linear/r9700_linear.h"},
                {"kernel", "src/ops/r9700/linear/r9700_linear.hip"},
                {"dispatch_profile", "src/ops/r9700/linear/r9700_q4_activation_profile.h"},
            }};
            for (std::size_t index = 0; index < source_specs.size(); ++index) {
                const auto path = std::filesystem::canonical(source_root / source_specs[index].second);
                source_identities[index] =
                    SourceIdentity{source_specs[index].first, path, sha256_file(path)};
            }
        }
        int count = 0;
        HIP_CHECK(hipGetDeviceCount(&count));
        hipDeviceProp_t device{};
        int selected = -1;
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            hipDeviceProp_t candidate{};
            HIP_CHECK(hipGetDeviceProperties(&candidate, ordinal));
            if (candidate.integrated == 0 &&
                std::string_view(candidate.gcnArchName) == "gfx1201" &&
                candidate.warpSize == 32) {
                if (selected != -1) fail("multiple discrete gfx1201 devices are visible");
                selected = ordinal;
                device = candidate;
            }
        }
        if (selected == -1) fail("shape sweep requires one discrete gfx1201 wave32 R9700");
        HIP_CHECK(hipSetDevice(selected));
        int runtime_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        int driver_version = 0;
        HIP_CHECK(hipDriverGetVersion(&driver_version));
        if (prefill_regression || lds_scope_regression || lds_scope_benchmark ||
            n128_regression || n128_benchmark || m128n128_regression ||
            m128n128_benchmark || pingpong_regression || pingpong_benchmark ||
            scalar_base_regression) {
            linear::A8Q4G64KernelResources resources{};
            HIP_CHECK(linear::a8q4g64_kernel_resources(&resources));
            if (resources.prefill_cta_registers > 96 ||
                resources.prefill_cta_static_shared_bytes != 17152 ||
                resources.prefill_cta_local_bytes != 0) {
                fail("prefill CTA resource gate requires <=96 VGPR, exactly 17152B LDS, "
                     "and no scratch");
            }
            if (resources.prefill_cta_m64n128_regression_registers > 96 ||
                resources.prefill_cta_m64n128_regression_static_shared_bytes != 8576 ||
                resources.prefill_cta_m64n128_regression_local_bytes != 0 ||
                resources.prefill_cta_m64n128_regression_max_threads_per_block < 512) {
                fail("M64xN128 regression CTA requires <=96 VGPR, exactly 8576B LDS, "
                     "no scratch, and a 512-thread launch bound");
            }
            if ((lds_scope_regression || lds_scope_benchmark) &&
                (resources.prefill_cta_global_inv_qualification_registers > 96 ||
                 resources.prefill_cta_global_inv_qualification_static_shared_bytes != 6400 ||
                 resources.prefill_cta_global_inv_qualification_local_bytes != 0)) {
                fail("LDS-scope CTA requires <=96 VGPR, exactly 6400B LDS, no scratch");
            }
            if (resources.prefill_cta_m64n64_regression_registers > 96 ||
                resources.prefill_cta_m64n64_regression_static_shared_bytes != 6400 ||
                resources.prefill_cta_m64n64_regression_local_bytes != 0) {
                fail("M64xN64 regression CTA requires <=96 VGPR, exactly 6400B LDS, no scratch");
            }
            if ((m128n128_regression || m128n128_benchmark) &&
                (resources.prefill_cta_m128n128_qualification_registers > 96 ||
                 resources.prefill_cta_m128n128_qualification_static_shared_bytes != 12800 ||
                 resources.prefill_cta_m128n128_qualification_local_bytes != 0 ||
                 resources.prefill_cta_m128n128_qualification_max_threads_per_block < 1024)) {
                fail("M128xN128 CTA requires <=96 VGPR, exactly 12800B LDS, no scratch");
            }
            if ((pingpong_regression || pingpong_benchmark) &&
                (resources.prefill_cta_registers > 96 ||
                 resources.prefill_cta_static_shared_bytes != 17152 ||
                 resources.prefill_cta_local_bytes != 0)) {
                fail("ping/pong CTA requires <=96 VGPR, exactly 17152B LDS, no scratch, "
                     "and a 512-thread launch bound");
            }
            const std::vector<Shape> prefill_shapes{
                Shape{7168, 5120, 0, 0}, Shape{4096, 5120, 0, 0},
                Shape{12288, 5120, 0, 0}, Shape{5120, 6144, 0, 0},
                Shape{34816, 5120, 0, 0}, Shape{5120, 17408, 0, 0},
                Shape{5120, 10240, 0, 0}, Shape{1024, 5120, 0, 0}};
            std::vector<PrefillResult> tail_qualifications;
            if (pingpong_regression || pingpong_benchmark || scalar_base_regression) {
                // Exercise the actual predicate-free fast kernel (including
                // nonzero status poisoning) under the complete FP64 oracle.
                tail_qualifications.push_back(run_prefill_case(
                    Shape{128U, 256U, 0, 0}, 192U, true, false,
                    false, nullptr, false, false,
                    pingpong_regression || pingpong_benchmark, scalar_base_regression));
            }
#if defined(NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION)
            if (scalar_base_regression) {
                // These valid off-inventory tuples must delegate through the
                // unchanged production wrapper and remain bit-identical. They
                // separately exercise an unqualified extent and logical-K tail.
                tail_qualifications.push_back(run_prefill_case(
                    Shape{128U, 256U, 0, 0}, 2047U, false, false,
                    false, nullptr, false, false, false, true));
                tail_qualifications.push_back(run_prefill_case(
                    Shape{1024U, 5056U, 0, 0}, 1024U, false, false,
                    false, nullptr, false, false, false, true));
            }
#endif
            const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> tail_shapes{{
                // N16/K16 storage requires N divisible by 16. These retain
                // non-full N128 tiles plus odd T and logical-K tails.
                {64U, 129U}, {128U, 129U}, {144U, 129U},
            }};
            for (const auto [tail_rows, tail_tokens] : tail_shapes) {
                tail_qualifications.push_back(run_prefill_case(
                    Shape{tail_rows, 193, 0, 0}, tail_tokens, true, false,
                    lds_scope_regression || lds_scope_benchmark, nullptr,
                    n128_regression || n128_benchmark,
                    m128n128_regression || m128n128_benchmark,
                    pingpong_regression || pingpong_benchmark,
                    scalar_base_regression));
            }
            std::vector<LdsScopeEvidence> evidence;
            for (const Shape shape : prefill_shapes) {
                for (const std::uint32_t tokens : {1024U, 2048U, 4096U, 8192U}) {
                    const PrefillResult result = run_prefill_case(
                        shape, tokens, false, false,
                        lds_scope_regression || lds_scope_benchmark, nullptr,
                        n128_regression || n128_benchmark,
                        m128n128_regression || m128n128_benchmark,
                        pingpong_regression || pingpong_benchmark,
                        scalar_base_regression);
                    if (result.maximum_bf16_steps > 2U) {
                        fail("prefill CTA numerical regression differs");
                    }
                    if (lds_scope_benchmark || n128_benchmark || m128n128_benchmark ||
                        pingpong_benchmark) {
                        evidence.push_back({result, {}});
                    }
                }
            }
            for (const PrefillResult& tail : tail_qualifications) {
                if (tail.maximum_bf16_steps > 2U) fail("tail qualification evidence differs");
            }
            if (lds_scope_benchmark || n128_benchmark || m128n128_benchmark ||
                pingpong_benchmark) {
                for (LdsScopeEvidence& row : evidence) {
                    (void)run_prefill_case(row.numerical.shape, row.numerical.tokens,
                                           false, true, lds_scope_benchmark, &row.timing,
                                           n128_benchmark, m128n128_benchmark,
                                           pingpong_benchmark);
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
                const bool accepted = write_lds_scope_report(
                    benchmark_output, executable, executable_sha256, source_root,
                    source_identities, device, runtime_version, driver_version, resources,
                    evidence, n128_benchmark, m128n128_benchmark, pingpong_benchmark);
                std::cout << (pingpong_benchmark
                                  ? accepted ? "A8Q4 ping/pong physical A/B: PASS\n"
                                             : "A8Q4 ping/pong physical A/B: REJECTED\n"
                              : m128n128_benchmark
                                  ? accepted ? "A8Q4 M128xN128 physical A/B: PASS\n"
                                             : "A8Q4 M128xN128 physical A/B: REJECTED\n"
                                  : n128_benchmark ? "A8Q4 N128 physical A/B: PASS\n"
                                             : "A8Q4 LDS-scope physical A/B: PASS\n");
                return accepted ? 0 : 1;
            }
            std::cout << (pingpong_regression
                              ? "A8Q4 ping/pong/production CTA bit-exact regression: PASS ("
                          : m128n128_regression
                              ? "A8Q4 M128xN128/production CTA bit-exact regression: PASS ("
                              : n128_regression
                              ? "A8Q4 M64xN64/production N128 CTA bit-exact regression: PASS ("
                              : lds_scope_regression
                              ? "A8Q4 LDS-scope/production CTA bit-exact regression: PASS ("
                          : scalar_base_regression
                              ? "A8Q4 scalar-base product-wrapper regression: PASS ("
                              : "A8Q4 cooperative prefill CTA regression: PASS (")
                      << "exact tail case and six Text plus two MTP shapes at four token extents)\n";
            return 0;
        }
        std::vector<Result> results;
        for (Shape shape : kShapes) {
            std::cout << "shape N=" << shape.rows << " K=" << shape.columns << std::endl;
            std::vector<Result> shape_results = run_shape(shape);
            results.insert(results.end(), shape_results.begin(), shape_results.end());
        }
        std::filesystem::create_directories(std::filesystem::path(argv[2]).parent_path());
        write_json(argv[2], results, device, runtime_version);
        std::cout << "A8Q4 real artifact shape sweep: PASS (" << kShapes.size()
                  << " shapes, " << results.size() << " extents)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "A8Q4 real artifact shape sweep: FAIL: " << error.what() << '\n';
        return 1;
    }
}
#endif
