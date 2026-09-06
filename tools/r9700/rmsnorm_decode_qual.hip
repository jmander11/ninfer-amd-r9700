#include <hip/hip_runtime.h>

#include "ops/r9700/eager/eager_ops.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::uint32_t kFeatures = 5120U;
constexpr float kModelEps = 1.0e-6F;
constexpr std::size_t kSamples = 9U;
constexpr std::uint32_t kTimingIterations = 32U;
constexpr std::string_view kPowerPath =
    "/sys/class/drm/card2/device/power_dpm_force_performance_level";

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream message;
    message << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(message.str());
}
#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

template <class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() { (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

std::uint16_t bf16_bits(float value) {
    std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    if ((word & 0x7f800000U) != 0x7f800000U) {
        word += 0x7fffU + ((word >> 16U) & 1U);
    } else if ((word & 0xffffU) != 0U) {
        word |= 0x10000U;
    }
    return static_cast<std::uint16_t>(word >> 16U);
}
hip_bfloat16 bf16(float value) {
    hip_bfloat16 result;
    result.data = bf16_bits(value);
    return result;
}
std::uint32_t ordered_bf16(std::uint16_t bits) {
    return (bits & 0x8000U) != 0U ? static_cast<std::uint32_t>(0xffffU - bits)
                                  : static_cast<std::uint32_t>(bits) + 0x8000U;
}
std::uint32_t bf16_steps(std::uint16_t lhs, std::uint16_t rhs) {
    const std::uint32_t a = ordered_bf16(lhs);
    const std::uint32_t b = ordered_bf16(rhs);
    return a > b ? a - b : b - a;
}

template <class T> void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() < source.size()) fail("upload extent mismatch");
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}
template <class T> std::vector<T> download(const DeviceBuffer<T>& source, std::size_t count) {
    if (source.size() < count) fail("download extent mismatch");
    std::vector<T> result(count);
    HIP_CHECK(hipMemcpy(result.data(), source.get(), count * sizeof(T), hipMemcpyDeviceToHost));
    return result;
}

void select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (!properties.integrated && std::string_view(properties.gcnArchName) == "gfx1201") {
            if (selected >= 0) fail("expected exactly one discrete gfx1201 device");
            selected = ordinal;
        }
    }
    if (selected < 0) fail("no discrete gfx1201 device");
    HIP_CHECK(hipSetDevice(selected));
}

std::vector<hip_bfloat16> make_input(std::uint32_t rows, std::uint32_t fixture) {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(rows) * kFeatures);
    for (std::size_t index = 0; index < result.size(); ++index) {
        float value = 0.0F;
        if (fixture == 0U) {
            const auto code = static_cast<std::int32_t>((index * 37U + 53U) % 509U) - 254;
            value = static_cast<float>(code) / 127.0F;
        } else if (fixture == 1U) {
            static constexpr std::array<float, 8> values{
                0.0F, -0.0F, 0x1p-14F, -0x1p-14F, 0.03125F, -0.5F, 8.0F, -31.75F};
            value = values[index % values.size()];
        }
        result[index] = bf16(value);
    }
    return result;
}
std::vector<hip_bfloat16> make_weight() {
    std::vector<hip_bfloat16> result(kFeatures);
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto code = static_cast<std::int32_t>((index * 29U + 17U) % 257U) - 128;
        result[index] = bf16(static_cast<float>(code) / 509.0F);
    }
    return result;
}

struct Error {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    std::uint32_t max_bf16_steps = 0;
};
Error require_oracle(const std::vector<hip_bfloat16>& actual,
                     const std::vector<hip_bfloat16>& input,
                     const std::vector<hip_bfloat16>& weight, std::uint32_t rows,
                     double eps, bool unit_offset) {
    if (actual.size() != input.size()) fail("oracle extent mismatch");
    Error error;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kFeatures;
        double sumsq = 0.0;
        for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
            const double value = static_cast<float>(input[base + feature]);
            sumsq += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sumsq / kFeatures + eps);
        for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
            const std::size_t index = base + feature;
            const double gain = static_cast<float>(weight[feature]) +
                                (unit_offset ? 1.0 : 0.0);
            const double expected = static_cast<float>(input[index]) * inverse * gain;
            const double observed = static_cast<float>(actual[index]);
            const double delta = observed - expected;
            error.max_abs = std::max(error.max_abs, std::abs(delta));
            squared_error += delta * delta;
            squared_reference += expected * expected;
            error.max_bf16_steps = std::max(
                error.max_bf16_steps, bf16_steps(actual[index].data,
                                                  bf16_bits(static_cast<float>(expected))));
        }
    }
    error.relative_l2 = std::sqrt(squared_error / std::max(1.0e-30, squared_reference));
    if (error.max_bf16_steps > 2U || error.relative_l2 > 5.0e-3 || error.max_abs > 0.05) {
        std::ostringstream message;
        message << "candidate exceeds FP64 oracle: steps=" << error.max_bf16_steps
                << " rel_l2=" << error.relative_l2 << " max_abs=" << error.max_abs;
        fail(message.str());
    }
    return error;
}

void launch_candidate(const DeviceBuffer<hip_bfloat16>& input,
                      const DeviceBuffer<hip_bfloat16>& weight,
                      DeviceBuffer<hip_bfloat16>& output, std::uint32_t rows, float eps,
                      bool unit_offset, hipStream_t stream) {
    HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_qualification(
        input.get(), weight.get(), output.get(), rows, eps, unit_offset, stream));
}

void qualify_numerical(hipStream_t stream) {
    const auto weight = make_weight();
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    upload(d_weight, weight);
    std::uint32_t maximum_steps = 0U;
    double maximum_abs = 0.0;
    double maximum_relative_l2 = 0.0;
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U}) {
        for (const std::uint32_t fixture : {0U, 1U, 2U}) {
            const auto input = make_input(rows, fixture);
            DeviceBuffer<hip_bfloat16> d_input(input.size()), d_output(input.size());
            upload(d_input, input);
            for (const bool unit_offset : {false, true}) {
                for (const float eps : {1.0e-8F, kModelEps, 1.0e-4F}) {
                    HIP_CHECK(hipMemsetAsync(d_output.get(), 0x55,
                                             input.size() * sizeof(hip_bfloat16), stream));
                    launch_candidate(d_input, d_weight, d_output, rows, eps, unit_offset, stream);
                    HIP_CHECK(hipStreamSynchronize(stream));
                    const Error error = require_oracle(download(d_output, input.size()), input,
                                                       weight, rows, eps, unit_offset);
                    maximum_steps = std::max(maximum_steps, error.max_bf16_steps);
                    maximum_abs = std::max(maximum_abs, error.max_abs);
                    maximum_relative_l2 = std::max(maximum_relative_l2, error.relative_l2);
                }
            }
        }
    }

    const auto graph_input = make_input(4U, 0U);
    DeviceBuffer<hip_bfloat16> d_input(graph_input.size()), d_output(graph_input.size());
    upload(d_input, graph_input);
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    launch_candidate(d_input, d_weight, d_output, 4U, kModelEps, true, stream);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) HIP_CHECK(hipGraphLaunch(executable, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    (void)require_oracle(download(d_output, graph_input.size()), graph_input, weight, 4U,
                         kModelEps, true);
    HIP_CHECK(hipGraphExecDestroy(executable));
    HIP_CHECK(hipGraphDestroy(graph));

    using ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_qualification;
    if (rmsnorm_k5120_rows4_qualification(nullptr, d_weight.get(), d_output.get(), 1U,
                                           kModelEps, true, stream) != hipErrorInvalidValue ||
        rmsnorm_k5120_rows4_qualification(d_input.get(), d_weight.get(), d_output.get(), 0U,
                                           kModelEps, true, stream) != hipErrorInvalidValue ||
        rmsnorm_k5120_rows4_qualification(d_input.get(), d_weight.get(), d_output.get(), 5U,
                                           kModelEps, true, stream) != hipErrorInvalidValue ||
        rmsnorm_k5120_rows4_qualification(d_input.get(), d_weight.get(), d_output.get(), 1U,
                                           std::numeric_limits<float>::quiet_NaN(), true,
                                           stream) != hipErrorInvalidValue) {
        fail("qualification boundary accepted malformed input");
    }
    std::cout << "r9700 RMSNorm K5120 rows1..4 qualification: PASS max_bf16_steps="
              << maximum_steps << " max_abs=" << maximum_abs
              << " max_relative_l2=" << maximum_relative_l2
              << " graph_replays=2 fallback_boundary=true\n";
}

template <class Launch> float time_launch(hipStream_t stream, Launch&& launch) {
    hipEvent_t begin = nullptr, end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin, stream));
    for (std::uint32_t iteration = 0; iteration < kTimingIterations; ++iteration) {
        HIP_CHECK(launch());
    }
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    return milliseconds / static_cast<float>(kTimingIterations);
}

void require_auto_and_fresh(const std::filesystem::path& path) {
    std::ifstream power{std::string(kPowerPath)};
    std::string profile;
    if (!power || !std::getline(power, profile) || profile != "auto") {
        fail("physical timing requires card2 performance profile auto");
    }
    std::error_code error;
    if (std::filesystem::symlink_status(path, error).type() !=
            std::filesystem::file_type::not_found ||
        (error && error != std::errc::no_such_file_or_directory)) {
        fail("benchmark output must be a fresh path");
    }
}

void benchmark(hipStream_t stream, const std::filesystem::path& path) {
    require_auto_and_fresh(path);
    const auto input = make_input(4U, 0U);
    const auto weight = make_weight();
    DeviceBuffer<hip_bfloat16> d_input(input.size()), d_weight(weight.size()), d_output(input.size());
    upload(d_input, input);
    upload(d_weight, weight);
    struct Row { std::uint32_t rows; std::array<float, kSamples> incumbent; std::array<float, kSamples> candidate; };
    std::array<Row, 4> rows{{{1U, {}, {}}, {2U, {}, {}}, {3U, {}, {}}, {4U, {}, {}}}};
    for (int warmup = 0; warmup < 3; ++warmup) {
        launch_candidate(d_input, d_weight, d_output, 4U, kModelEps, true, stream);
        HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
            d_input.get(), d_weight.get(), d_output.get(), kFeatures, 4U, kModelEps, true, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    for (Row& row : rows) {
        for (std::size_t sample = 0; sample < kSamples; ++sample) {
            auto incumbent = [&] {
                return ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
                    d_input.get(), d_weight.get(), d_output.get(), kFeatures, row.rows,
                    kModelEps, true, stream);
            };
            auto candidate = [&] {
                return ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_qualification(
                    d_input.get(), d_weight.get(), d_output.get(), row.rows,
                    kModelEps, true, stream);
            };
            if ((sample & 1U) == 0U) {
                row.incumbent[sample] = time_launch(stream, incumbent);
                row.candidate[sample] = time_launch(stream, candidate);
            } else {
                row.candidate[sample] = time_launch(stream, candidate);
                row.incumbent[sample] = time_launch(stream, incumbent);
            }
        }
    }
    auto median = [](std::array<float, kSamples> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2U];
    };
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) fail("could not create benchmark report");
    output << std::setprecision(9)
           << "{\n  \"schema\": \"ninfer.r9700.rmsnorm_k5120_rows4_qualification.v1\",\n"
           << "  \"target\": \"gfx1201\",\n  \"power_profile\": \"auto\",\n"
           << "  \"features\": 5120,\n  \"ordinary_calls_per_round\": 129,\n"
           << "  \"timing_iterations_per_sample\": " << kTimingIterations << ",\n"
           << "  \"rows\": [\n";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const Row& row = rows[index];
        output << "    {\"rows\": " << row.rows
               << ", \"incumbent_median_ms\": " << median(row.incumbent)
               << ", \"candidate_median_ms\": " << median(row.candidate) << "}";
        output << (index + 1U == rows.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    if (!output) fail("could not write benchmark report");
}
} // namespace

int main(int argc, char** argv) {
    try {
        select_r9700();
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        qualify_numerical(stream);
        if (argc == 4 && std::string_view(argv[1]) == "--benchmark" &&
            std::string_view(argv[2]) == "--out-json") {
            benchmark(stream, argv[3]);
        } else if (argc != 1) {
            fail("usage: rmsnorm_decode_qual [--benchmark --out-json PATH]");
        }
        HIP_CHECK(hipStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700 RMSNorm K5120 rows1..4 qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
