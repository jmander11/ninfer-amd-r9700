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

constexpr std::uint32_t kFeatures = 6144;
constexpr float kModelEps = 1.0e-6F;
constexpr std::size_t kSamples = 7;

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

std::uint16_t bf16_rne_bits(float value) {
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
    result.data = bf16_rne_bits(value);
    return result;
}

std::vector<hip_bfloat16> fixture(std::size_t count, std::uint32_t seed, float divisor) {
    std::vector<hip_bfloat16> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto wave = static_cast<std::int32_t>((index * 37U + seed * 19U) % 509U) - 254;
        values[index] = bf16(static_cast<float>(wave) / divisor);
    }
    if (count >= 8U) {
        values[0] = bf16(0.0F);
        values[1] = bf16(-0.0F);
        values[2] = bf16(0.00006103515625F);
        values[3] = bf16(-0.00006103515625F);
        values[4] = bf16(31.75F);
        values[5] = bf16(-31.75F);
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
    int devices = 0;
    HIP_CHECK(hipGetDeviceCount(&devices));
    int selected = -1;
    for (int ordinal = 0; ordinal < devices; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (!properties.integrated && std::string_view(properties.gcnArchName) == "gfx1201") {
            if (selected >= 0) fail("expected one discrete gfx1201 device");
            selected = ordinal;
        }
    }
    if (selected < 0) fail("no discrete gfx1201 device");
    HIP_CHECK(hipSetDevice(selected));
}

void require_fresh_output_path(const std::filesystem::path& output_path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(output_path, error);
    const bool absent = status.type() == std::filesystem::file_type::not_found &&
                        (!error || error == std::errc::no_such_file_or_directory);
    if (!absent) fail("benchmark report path already exists or cannot be inspected");
}

std::vector<double> fp64_oracle(const std::vector<hip_bfloat16>& input,
                                const std::vector<hip_bfloat16>& weight,
                                const std::vector<hip_bfloat16>& gate,
                                std::uint32_t rows, double eps) {
    std::vector<double> output(static_cast<std::size_t>(rows) * kFeatures);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kFeatures;
        double sumsq = 0.0;
        for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
            const double value = static_cast<float>(input[base + feature]);
            sumsq += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sumsq / kFeatures + eps);
        for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
            const double x = static_cast<float>(input[base + feature]);
            const double z = static_cast<float>(gate[base + feature]);
            const double silu = z / (1.0 + std::exp(-z));
            output[base + feature] = x * inverse * static_cast<float>(weight[feature]) * silu;
        }
    }
    return output;
}

void require_exact(const std::vector<hip_bfloat16>& actual,
                   const std::vector<hip_bfloat16>& expected) {
    if (actual.size() != expected.size()) fail("exact comparison extent mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].data != expected[index].data) {
            fail("candidate differs from incumbent at BF16 element " + std::to_string(index));
        }
    }
}

void require_oracle(const std::vector<hip_bfloat16>& actual,
                    const std::vector<double>& expected) {
    if (actual.size() != expected.size()) fail("oracle comparison extent mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double observed = static_cast<float>(actual[index]);
        const double absolute = std::abs(observed - expected[index]);
        const double relative = absolute / std::max(1.0e-6, std::abs(expected[index]));
        if (absolute > 4.0e-2 && relative > 4.0e-2) {
            fail("candidate exceeds FP64 gated RMSNorm criterion at element " +
                 std::to_string(index));
        }
    }
}

void qualify_numerical(hipStream_t stream) {
    constexpr std::uint32_t maximum_rows = 2048;
    const auto input = fixture(static_cast<std::size_t>(maximum_rows) * kFeatures, 7U, 127.0F);
    const auto weight = fixture(kFeatures, 11U, 509.0F);
    const auto gate = fixture(input.size(), 13U, 191.0F);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_gate(gate.size());
    DeviceBuffer<hip_bfloat16> d_incumbent(input.size());
    DeviceBuffer<hip_bfloat16> d_candidate(input.size());
    upload(d_input, input);
    upload(d_weight, weight);
    upload(d_gate, gate);
    constexpr std::array<std::uint32_t, 11> row_cases{
        1, 2, 7, 8, 9, 16, 32, 64, 128, 512, 2048};
    for (const float eps : {kModelEps, 1.0e-8F, 1.0e-4F}) {
        for (const std::uint32_t rows : row_cases) {
            if (eps != kModelEps && rows > 128U) continue;
            const std::size_t elements = static_cast<std::size_t>(rows) * kFeatures;
            HIP_CHECK(ninfer::ops::r9700::eager::gated_rmsnorm_incumbent_qualification(
                d_input.get(), d_weight.get(), d_gate.get(), d_incumbent.get(), kFeatures,
                rows, eps, stream));
            HIP_CHECK(ninfer::ops::r9700::eager::gated_rmsnorm_k6144_token8_qualification(
                d_input.get(), d_weight.get(), d_gate.get(), d_candidate.get(), rows, eps,
                stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            const auto candidate = download(d_candidate, elements);
            require_exact(candidate, download(d_incumbent, elements));
            if (rows <= 128U || rows == maximum_rows) {
                require_oracle(candidate, fp64_oracle(input, weight, gate, rows, eps));
            }
        }
    }
    if (ninfer::ops::r9700::eager::gated_rmsnorm_k6144_token8_qualification(
            nullptr, d_weight.get(), d_gate.get(), d_candidate.get(), 1, kModelEps, stream) !=
            hipErrorInvalidValue ||
        ninfer::ops::r9700::eager::gated_rmsnorm_k6144_token8_qualification(
            d_input.get(), d_weight.get(), d_gate.get(), d_candidate.get(), 0, kModelEps,
            stream) != hipErrorInvalidValue ||
        ninfer::ops::r9700::eager::gated_rmsnorm_k6144_token8_qualification(
            d_input.get(), d_weight.get(), d_gate.get(), d_candidate.get(), 1, 0.0F,
            stream) != hipErrorInvalidValue) {
        fail("qualification boundary accepted malformed input");
    }

    // The direct candidate preserves the incumbent boundary's supported input/output alias.
    // Lane zero completes the feature-ordered reduction before its wave writes the row.
    constexpr std::uint32_t alias_rows = 9;
    const std::size_t alias_elements = static_cast<std::size_t>(alias_rows) * kFeatures;
    upload(d_input, input);
    HIP_CHECK(ninfer::ops::r9700::eager::gated_rmsnorm_k6144_token8_qualification(
        d_input.get(), d_weight.get(), d_gate.get(), d_input.get(), alias_rows, kModelEps,
        stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    upload(d_candidate, input);
    HIP_CHECK(ninfer::ops::r9700::eager::gated_rmsnorm_incumbent_qualification(
        d_candidate.get(), d_weight.get(), d_gate.get(), d_incumbent.get(), kFeatures,
        alias_rows, kModelEps, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    require_exact(download(d_input, alias_elements), download(d_incumbent, alias_elements));
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
    float sample = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&sample, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return sample;
}

float median(std::array<float, kSamples> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

void write_samples(std::ostringstream& out, const std::array<float, kSamples>& samples) {
    out << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) out << ',';
        out << samples[index];
    }
    out << ']';
}

void benchmark(hipStream_t stream, const std::filesystem::path& output_path) {
    constexpr std::uint32_t maximum_rows = 8192;
    const auto input = fixture(static_cast<std::size_t>(maximum_rows) * kFeatures, 101U, 127.0F);
    const auto weight = fixture(kFeatures, 103U, 509.0F);
    const auto gate = fixture(input.size(), 107U, 191.0F);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_gate(gate.size());
    DeviceBuffer<hip_bfloat16> d_output(input.size());
    upload(d_input, input);
    upload(d_weight, weight);
    upload(d_gate, gate);
    struct Row {
        std::uint32_t tokens;
        std::array<float, kSamples> incumbent{};
        std::array<float, kSamples> candidate{};
        float incumbent_median = 0.0F;
        float candidate_median = 0.0F;
    };
    std::vector<Row> rows_out;
    for (const std::uint32_t rows : {64U, 128U, 512U, 2048U, 8192U}) {
        auto incumbent = [&] {
            return ninfer::ops::r9700::eager::gated_rmsnorm_incumbent_qualification(
                d_input.get(), d_weight.get(), d_gate.get(), d_output.get(), kFeatures, rows,
                kModelEps, stream);
        };
        auto candidate = [&] {
            return ninfer::ops::r9700::eager::gated_rmsnorm_k6144_token8_qualification(
                d_input.get(), d_weight.get(), d_gate.get(), d_output.get(), rows, kModelEps,
                stream);
        };
        HIP_CHECK(incumbent());
        HIP_CHECK(candidate());
        HIP_CHECK(hipStreamSynchronize(stream));
        Row row{.tokens = rows};
        for (std::size_t sample = 0; sample < kSamples; ++sample) {
            if ((sample & 1U) == 0U) {
                row.incumbent[sample] = timed(stream, incumbent);
                row.candidate[sample] = timed(stream, candidate);
            } else {
                row.candidate[sample] = timed(stream, candidate);
                row.incumbent[sample] = timed(stream, incumbent);
            }
        }
        row.incumbent_median = median(row.incumbent);
        row.candidate_median = median(row.candidate);
        if (rows >= 128U && row.candidate_median > row.incumbent_median) {
            fail("candidate regressed at a required T128+ timing point");
        }
        if (rows == 2048U && row.incumbent_median / row.candidate_median < 1.5F) {
            fail("candidate misses the P2048 1.5x admission gate");
        }
        rows_out.push_back(row);
    }

    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer.r9700.gated_rmsnorm_k6144_token8_qualification.v1\",\n"
        << "  \"architecture\": \"gfx1201\",\n"
        << "  \"disposition\": \"qualification_only_unpromoted\",\n"
        << "  \"protocol\": {\"warmup_per_route\": 1, \"samples_per_route\": 7, "
           "\"order\": \"balanced_forward_reverse\"},\n"
        << "  \"gates\": {\"p2048_minimum_speedup\": 1.5, "
           "\"no_t128_plus_regression\": true},\n"
        << "  \"numerical\": {\"incumbent_bf16_bit_exact\": true, "
           "\"fp64_oracle\": true},\n  \"rows\": [\n";
    for (std::size_t index = 0; index < rows_out.size(); ++index) {
        const Row& row = rows_out[index];
        out << "    {\"tokens\": " << row.tokens << ", \"incumbent_samples_ms\": ";
        write_samples(out, row.incumbent);
        out << ", \"candidate_samples_ms\": ";
        write_samples(out, row.candidate);
        out << ", \"incumbent_median_ms\": " << row.incumbent_median
            << ", \"candidate_median_ms\": " << row.candidate_median
            << ", \"speedup\": " << row.incumbent_median / row.candidate_median << '}'
            << (index + 1U == rows_out.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    const std::string payload = out.str();
    const int descriptor = ::open(output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                  0644);
    if (descriptor < 0) fail("refusing to overwrite or could not create benchmark report");
    std::size_t written = 0;
    while (written < payload.size()) {
        const ssize_t count = ::write(descriptor, payload.data() + written,
                                      payload.size() - written);
        if (count <= 0) {
            (void)::close(descriptor);
            fail("could not finish benchmark report");
        }
        written += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) fail("could not close benchmark report");
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool run_benchmark = false;
        std::filesystem::path output_path;
        if (argc == 3 && std::string_view(argv[1]) == "--check-output-path") {
            require_fresh_output_path(argv[2]);
            std::cout << "r9700 gated RMSNorm report path preflight: PASS\n";
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--benchmark" &&
            std::string_view(argv[2]) == "--out-json") {
            run_benchmark = true;
            output_path = argv[3];
            require_fresh_output_path(output_path);
        } else if (argc != 1) {
            fail("usage: gated_rmsnorm_prefill_qual [--benchmark --out-json PATH | "
                 "--check-output-path PATH]");
        }
        select_r9700();
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        qualify_numerical(stream);
        if (run_benchmark) benchmark(stream, output_path);
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "r9700 gated RMSNorm K6144 token8 qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700 gated RMSNorm K6144 token8 qualification: FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
