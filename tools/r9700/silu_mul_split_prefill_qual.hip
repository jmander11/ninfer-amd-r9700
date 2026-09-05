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

constexpr std::uint32_t kFeatures = 17408U;
constexpr std::uint32_t kPackedStride = 34816U;
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

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { (void)hipFree(data_); }
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
        const auto code = static_cast<std::int32_t>((index * 43U + seed * 29U) % 509U) - 254;
        values[index] = bf16(static_cast<float>(code) / 73.0F);
    }
    return values;
}

template <typename T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() != source.size()) fail("upload extent mismatch");
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}

template <typename T>
std::vector<T> download(const DeviceBuffer<T>& source, std::size_t count) {
    if (count > source.size()) fail("download extent mismatch");
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
    const bool absent = status.type() == std::filesystem::file_type::not_found &&
                        (!error || error == std::errc::no_such_file_or_directory);
    if (!absent) fail("report path exists or cannot be inspected");
}

void qualify_numerical(hipStream_t stream) {
    constexpr std::uint32_t maximum_rows = 128U;
    const auto packed = fixture(static_cast<std::size_t>(maximum_rows) * kPackedStride, 17U);
    DeviceBuffer<hip_bfloat16> d_packed(packed.size());
    DeviceBuffer<hip_bfloat16> d_incumbent(static_cast<std::size_t>(maximum_rows) * kFeatures);
    DeviceBuffer<hip_bfloat16> d_candidate(d_incumbent.size());
    upload(d_packed, packed);
    for (const std::uint32_t rows : {1U, 2U, 3U, 7U, 8U, 9U, 17U, 128U}) {
        const std::size_t elements = static_cast<std::size_t>(rows) * kFeatures;
        const hip_bfloat16* gate = d_packed.get();
        const hip_bfloat16* up = d_packed.get() + kFeatures;
        HIP_CHECK(ninfer::ops::r9700::eager::silu_mul_split17408_incumbent_qualification(
            gate, up, d_incumbent.get(), rows, stream));
        HIP_CHECK(ninfer::ops::r9700::eager::silu_mul_split17408_2d_qualification(
            gate, up, d_candidate.get(), rows, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto incumbent = download(d_incumbent, elements);
        const auto candidate = download(d_candidate, elements);
        for (std::size_t index = 0; index < elements; ++index) {
            if (candidate[index].data != incumbent[index].data) {
                fail("candidate differs from incumbent at element " + std::to_string(index));
            }
            const std::uint32_t row = static_cast<std::uint32_t>(index / kFeatures);
            const std::uint32_t feature = static_cast<std::uint32_t>(index % kFeatures);
            const std::size_t offset = static_cast<std::size_t>(row) * kPackedStride + feature;
            const double gate_value = static_cast<float>(packed[offset]);
            const double up_value = static_cast<float>(packed[offset + kFeatures]);
            const double expected = gate_value / (1.0 + std::exp(-gate_value)) * up_value;
            const double observed = static_cast<float>(candidate[index]);
            const double absolute = std::abs(observed - expected);
            const double relative = absolute / std::max(1.0e-6, std::abs(expected));
            if (absolute > 4.0e-2 && relative > 4.0e-2) {
                fail("candidate exceeds independent FP64 criterion at element " +
                     std::to_string(index));
            }
        }
    }
    if (ninfer::ops::r9700::eager::silu_mul_split17408_2d_qualification(
            nullptr, d_packed.get(), d_candidate.get(), 1U, stream) != hipErrorInvalidValue ||
        ninfer::ops::r9700::eager::silu_mul_split17408_2d_qualification(
            d_packed.get(), d_packed.get(), d_candidate.get(), 0U, stream) !=
            hipErrorInvalidValue) {
        fail("candidate accepted malformed input");
    }
}

template <typename Launch>
float timed(hipStream_t stream, Launch&& launch) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    HIP_CHECK(launch());
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return milliseconds;
}

float median(std::array<float, kSamples> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

void samples_json(std::ostringstream& out, const std::array<float, kSamples>& values) {
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) out << ',';
        out << values[index];
    }
    out << ']';
}

void benchmark(hipStream_t stream, const std::filesystem::path& path) {
    constexpr std::uint32_t maximum_rows = 4096U;
    const auto packed = fixture(static_cast<std::size_t>(maximum_rows) * kPackedStride, 101U);
    DeviceBuffer<hip_bfloat16> d_packed(packed.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(maximum_rows) * kFeatures);
    upload(d_packed, packed);
    struct Row {
        std::uint32_t tokens;
        std::array<float, kSamples> incumbent{};
        std::array<float, kSamples> candidate{};
        float incumbent_median = 0.0F;
        float candidate_median = 0.0F;
    };
    std::vector<Row> results;
    for (const std::uint32_t rows : {128U, 512U, 2048U, 4096U}) {
        auto incumbent = [&] {
            return ninfer::ops::r9700::eager::silu_mul_split17408_incumbent_qualification(
                d_packed.get(), d_packed.get() + kFeatures, d_output.get(), rows, stream);
        };
        auto candidate = [&] {
            return ninfer::ops::r9700::eager::silu_mul_split17408_2d_qualification(
                d_packed.get(), d_packed.get() + kFeatures, d_output.get(), rows, stream);
        };
        HIP_CHECK(incumbent());
        HIP_CHECK(candidate());
        HIP_CHECK(hipStreamSynchronize(stream));
        Row result{.tokens = rows};
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
        results.push_back(result);
    }
    require_auto();
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer.r9700.silu_mul_split17408_2d_qualification.v1\",\n"
        << "  \"architecture\": \"gfx1201\",\n"
        << "  \"disposition\": \"qualification_only_unpromoted\",\n"
        << "  \"power_profile\": {\"required\": \"auto\", \"before\": \"auto\", "
           "\"after\": \"auto\"},\n"
        << "  \"protocol\": {\"warmup_per_route\": 1, \"samples_per_route\": 7, "
           "\"order\": \"balanced_forward_reverse\"},\n"
        << "  \"numerical\": {\"incumbent_bf16_bit_exact\": true, "
           "\"fp64_oracle\": true},\n  \"rows\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const Row& row = results[index];
        out << "    {\"tokens\": " << row.tokens << ", \"incumbent_samples_ms\": ";
        samples_json(out, row.incumbent);
        out << ", \"candidate_samples_ms\": ";
        samples_json(out, row.candidate);
        out << ", \"incumbent_median_ms\": " << row.incumbent_median
            << ", \"candidate_median_ms\": " << row.candidate_median
            << ", \"speedup\": " << row.incumbent_median / row.candidate_median << '}'
            << (index + 1U == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    const std::string payload = out.str();
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("refusing to overwrite or could not create report");
    std::size_t written = 0;
    while (written < payload.size()) {
        const ssize_t count = ::write(descriptor, payload.data() + written,
                                      payload.size() - written);
        if (count <= 0) {
            (void)::close(descriptor);
            fail("could not finish report");
        }
        written += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) fail("could not close report");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string_view(argv[1]) != "--benchmark" ||
            std::string_view(argv[2]) != "--out-json") {
            fail("usage: silu_mul_split_prefill_qual --benchmark --out-json PATH");
        }
        const std::filesystem::path path = argv[3];
        require_fresh(path);
        require_auto();
        select_r9700();
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        qualify_numerical(stream);
        benchmark(stream, path);
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "r9700 SiLU multiply split17408 2D qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700 SiLU multiply split17408 2D qualification: FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
