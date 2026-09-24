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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr std::uint32_t kFeatures = 256U;
constexpr float kEps = 1.0e-6F;
constexpr std::size_t kSamples = 7U;
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

template <typename T> class DeviceBuffer {
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
std::vector<hip_bfloat16> fixture(std::size_t count, std::uint32_t seed) {
    std::vector<hip_bfloat16> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto code = static_cast<std::int32_t>((index * 37U + seed * 19U) % 509U) - 254;
        values[index] = bf16(static_cast<float>(code) / 127.0F);
    }
    return values;
}
template <typename T> void upload(DeviceBuffer<T>& output, const std::vector<T>& input) {
    if (output.size() != input.size()) fail("upload extent mismatch");
    HIP_CHECK(hipMemcpy(output.get(), input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
}
template <typename T> std::vector<T> download(const DeviceBuffer<T>& input, std::size_t count) {
    std::vector<T> output(count);
    HIP_CHECK(hipMemcpy(output.data(), input.get(), count * sizeof(T), hipMemcpyDeviceToHost));
    return output;
}
void require_auto() {
    std::ifstream input{std::string(kPowerPath)};
    std::string value;
    if (!input || !std::getline(input, value) || value != "auto") {
        fail("physical timing requires card2 power profile auto");
    }
}
void require_fresh(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (status.type() != std::filesystem::file_type::not_found ||
        (error && error != std::errc::no_such_file_or_directory)) {
        fail("report path exists or cannot be inspected");
    }
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

void qualify_numerical(hipStream_t stream) {
    constexpr std::uint32_t maximum_rows = 129U;
    const auto input = fixture(static_cast<std::size_t>(maximum_rows) * kFeatures, 7U);
    const auto weight = fixture(kFeatures, 11U);
    DeviceBuffer<hip_bfloat16> d_input(input.size()), d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_incumbent(input.size()), d_candidate(input.size());
    upload(d_input, input);
    upload(d_weight, weight);
    for (const bool unit_offset : {false, true}) {
        for (const float eps : {kEps, 1.0e-8F, 1.0e-4F}) {
            for (const std::uint32_t rows : {1U, 2U, 7U, 8U, 9U, 31U, 32U, 33U, 128U, 129U}) {
                const std::size_t elements = static_cast<std::size_t>(rows) * kFeatures;
                HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
                    d_input.get(), d_weight.get(), d_incumbent.get(), kFeatures, rows, eps,
                    unit_offset, stream));
                HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_k256_token8_qualification(
                    d_input.get(), d_weight.get(), d_candidate.get(), rows, eps, unit_offset,
                    stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                const auto incumbent = download(d_incumbent, elements);
                const auto candidate = download(d_candidate, elements);
                for (std::size_t index = 0; index < elements; ++index) {
                    if (candidate[index].data != incumbent[index].data) {
                        fail("candidate differs from incumbent at element " +
                             std::to_string(index));
                    }
                }
                for (std::uint32_t row = 0; row < rows; ++row) {
                    const std::size_t base = static_cast<std::size_t>(row) * kFeatures;
                    double sumsq = 0.0;
                    for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
                        const double value = static_cast<float>(input[base + feature]);
                        sumsq += value * value;
                    }
                    const double inverse = 1.0 / std::sqrt(sumsq / kFeatures + eps);
                    for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
                        const double gain = static_cast<float>(weight[feature]) +
                                            (unit_offset ? 1.0 : 0.0);
                        const double expected = static_cast<float>(input[base + feature]) *
                                                inverse * gain;
                        const double observed = static_cast<float>(candidate[base + feature]);
                        const double absolute = std::abs(observed - expected);
                        const double relative = absolute /
                                                std::max(1.0e-6, std::abs(expected));
                        if (absolute > 4.0e-2 && relative > 4.0e-2) {
                            fail("candidate exceeds FP64 criterion");
                        }
                    }
                }
            }
        }
    }
    if (ninfer::ops::r9700::eager::rmsnorm_k256_token8_qualification(
            nullptr, d_weight.get(), d_candidate.get(), 1U, kEps, false, stream) !=
            hipErrorInvalidValue ||
        ninfer::ops::r9700::eager::rmsnorm_k256_token8_qualification(
            d_input.get(), d_weight.get(), d_candidate.get(), 0U, kEps, false, stream) !=
            hipErrorInvalidValue) {
        fail("candidate accepted malformed input");
    }
}

template <typename Launch> float timed(hipStream_t stream, Launch&& launch) {
    hipEvent_t start = nullptr, stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    HIP_CHECK(launch());
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float value = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&value, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return value;
}
float median(std::array<float, kSamples> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}
void write_samples(std::ostringstream& out, const std::array<float, kSamples>& values) {
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) out << ',';
        out << values[index];
    }
    out << ']';
}
void benchmark(hipStream_t stream, const std::filesystem::path& path) {
    constexpr std::uint32_t maximum_rows = 49152U;
    const auto input = fixture(static_cast<std::size_t>(maximum_rows) * kFeatures, 101U);
    const auto weight = fixture(kFeatures, 103U);
    DeviceBuffer<hip_bfloat16> d_input(input.size()), d_weight(weight.size()), d_output(input.size());
    upload(d_input, input);
    upload(d_weight, weight);
    struct Row {
        std::uint32_t rows;
        std::array<float, kSamples> incumbent{};
        std::array<float, kSamples> candidate{};
        float incumbent_median = 0.0F;
        float candidate_median = 0.0F;
    };
    std::vector<Row> results;
    for (const std::uint32_t rows : {128U, 512U, 2048U, 8192U, 49152U}) {
        auto incumbent = [&] { return ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
            d_input.get(), d_weight.get(), d_output.get(), kFeatures, rows, kEps, true, stream); };
        auto candidate = [&] { return ninfer::ops::r9700::eager::rmsnorm_k256_token8_qualification(
            d_input.get(), d_weight.get(), d_output.get(), rows, kEps, true, stream); };
        HIP_CHECK(incumbent()); HIP_CHECK(candidate()); HIP_CHECK(hipStreamSynchronize(stream));
        Row result{.rows = rows};
        for (std::size_t sample = 0; sample < kSamples; ++sample) {
            if ((sample & 1U) == 0U) {
                result.incumbent[sample] = timed(stream, incumbent);
                result.candidate[sample] = timed(stream, candidate);
            } else {
                result.candidate[sample] = timed(stream, candidate);
                result.incumbent[sample] = timed(stream, incumbent);
            }
        }
        result.incumbent_median = median(result.incumbent);
        result.candidate_median = median(result.candidate);
        if (result.candidate_median > result.incumbent_median) fail("candidate timing regression");
        if (rows == 2048U && result.incumbent_median / result.candidate_median < 1.5F) {
            fail("candidate misses P2048 1.5x gate");
        }
        results.push_back(result);
    }
    require_auto();
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer.r9700.rmsnorm_k256_token8_qualification.v1\",\n"
        << "  \"architecture\": \"gfx1201\",\n"
        << "  \"disposition\": \"qualification_only_unpromoted\",\n"
        << "  \"power_profile\": {\"required\": \"auto\", \"before\": \"auto\", "
           "\"after\": \"auto\"},\n"
        << "  \"protocol\": {\"warmup_per_route\": 1, \"samples_per_route\": 7, "
           "\"order\": \"balanced_forward_reverse\"},\n"
        << "  \"gates\": {\"p2048_minimum_speedup\": 1.5, "
           "\"no_t128_plus_regression\": true},\n"
        << "  \"numerical\": {\"incumbent_bf16_bit_exact\": true, "
           "\"fp64_oracle\": true},\n  \"rows\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const Row& row = results[index];
        out << "    {\"rows\": " << row.rows << ", \"incumbent_samples_ms\": ";
        write_samples(out, row.incumbent);
        out << ", \"candidate_samples_ms\": "; write_samples(out, row.candidate);
        out << ", \"incumbent_median_ms\": " << row.incumbent_median
            << ", \"candidate_median_ms\": " << row.candidate_median
            << ", \"speedup\": " << row.incumbent_median / row.candidate_median << '}'
            << (index + 1U == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    const std::string payload = out.str();
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) fail("refusing to overwrite or could not create report");
    std::size_t written = 0;
    while (written < payload.size()) {
        const ssize_t count = ::write(fd, payload.data() + written, payload.size() - written);
        if (count <= 0) { (void)::close(fd); fail("could not finish report"); }
        written += static_cast<std::size_t>(count);
    }
    if (::close(fd) != 0) fail("could not close report");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string_view(argv[1]) != "--benchmark" ||
            std::string_view(argv[2]) != "--out-json") {
            fail("usage: rmsnorm_k256_prefill_qual --benchmark --out-json PATH");
        }
        const std::filesystem::path path = argv[3];
        require_fresh(path); require_auto(); select_r9700();
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        qualify_numerical(stream); benchmark(stream, path);
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "r9700 RMSNorm K256 token8 qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700 RMSNorm K256 token8 qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
