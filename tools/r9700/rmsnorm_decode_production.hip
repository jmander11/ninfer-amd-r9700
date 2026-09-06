#include <hip/hip_runtime.h>

#include "ops/r9700/eager/eager_ops.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
constexpr std::size_t kGuardElements = 32U;
constexpr std::uint16_t kGuardBits = 0x5555U;

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

struct Error {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    std::uint32_t max_bf16_steps = 0;
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

std::uint64_t round_ties_even(double value) {
    const double base = std::floor(value);
    const double tail = value - base;
    auto integer = static_cast<std::uint64_t>(base);
    if (tail > 0.5 || (tail == 0.5 && (integer & 1U) != 0U)) ++integer;
    return integer;
}

std::uint16_t bf16_bits_direct(double value) {
    if (!std::isfinite(value)) fail("nonfinite FP64 oracle value");
    const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
    const double magnitude = std::abs(value);
    if (magnitude == 0.0) return sign;
    int exponent = 0;
    const double fraction = std::frexp(magnitude, &exponent);
    int biased = exponent - 1 + 127;
    if (biased <= 0) {
        const auto mantissa = round_ties_even(std::ldexp(magnitude, 133));
        if (mantissa == 0U) return sign;
        if (mantissa >= 128U) return static_cast<std::uint16_t>(sign | 0x0080U);
        return static_cast<std::uint16_t>(sign | mantissa);
    }
    auto significand = round_ties_even(std::ldexp(fraction, 8));
    if (significand == 256U) {
        significand = 128U;
        ++biased;
    }
    if (biased >= 255) return static_cast<std::uint16_t>(sign | 0x7f80U);
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint16_t>(biased) << 7U) |
        static_cast<std::uint16_t>(significand - 128U));
}

template <class T> void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() < source.size()) fail("upload extent mismatch");
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}

template <class T> std::vector<T> download(const DeviceBuffer<T>& source) {
    std::vector<T> result(source.size());
    HIP_CHECK(hipMemcpy(result.data(), source.get(), result.size() * sizeof(T),
                        hipMemcpyDeviceToHost));
    return result;
}

void poison_output(DeviceBuffer<hip_bfloat16>& storage, hipStream_t stream) {
    HIP_CHECK(hipMemsetAsync(storage.get(), 0x55, storage.size() * sizeof(hip_bfloat16), stream));
}

std::vector<hip_bfloat16> checked_output(const DeviceBuffer<hip_bfloat16>& storage,
                                         std::size_t logical_elements) {
    const auto raw = download(storage);
    if (storage.size() != logical_elements + 2U * kGuardElements) {
        fail("guarded output extent mismatch");
    }
    for (std::size_t index = 0; index < kGuardElements; ++index) {
        if (raw[index].data != kGuardBits ||
            raw[kGuardElements + logical_elements + index].data != kGuardBits) {
            fail("RMSNorm output guard was clobbered");
        }
    }
    return {raw.begin() + static_cast<std::ptrdiff_t>(kGuardElements),
            raw.begin() + static_cast<std::ptrdiff_t>(kGuardElements + logical_elements)};
}

std::vector<std::uint16_t> output_bits(const DeviceBuffer<hip_bfloat16>& storage) {
    const auto raw = download(storage);
    std::vector<std::uint16_t> bits;
    bits.reserve(raw.size());
    for (const hip_bfloat16 value : raw) bits.push_back(value.data);
    return bits;
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
    if (selected != 0) fail("the sole discrete gfx1201 device must be HIP ordinal 0");
    HIP_CHECK(hipSetDevice(selected));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, selected));
    if (std::string_view(properties.name).find("R9700") == std::string_view::npos) {
        fail("HIP device 0 is not an AMD Radeon AI PRO R9700");
    }
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
        } else {
            const auto code = static_cast<std::int32_t>((index * 97U + 11U) % 8191U) - 4095;
            value = static_cast<float>(code) / 64.0F;
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
            if (!std::isfinite(expected) || !std::isfinite(observed)) {
                fail("nonfinite RMSNorm oracle value");
            }
            const double delta = observed - expected;
            error.max_abs = std::max(error.max_abs, std::abs(delta));
            squared_error += delta * delta;
            squared_reference += expected * expected;
            error.max_bf16_steps = std::max(
                error.max_bf16_steps,
                bf16_steps(actual[index].data, bf16_bits_direct(expected)));
        }
    }
    error.relative_l2 = std::sqrt(squared_error / std::max(1.0e-30, squared_reference));
    if (error.max_bf16_steps > 2U) {
        std::ostringstream message;
        message << "production route exceeds FP64 oracle: steps=" << error.max_bf16_steps
                << " rel_l2=" << error.relative_l2 << " max_abs=" << error.max_abs;
        fail(message.str());
    }
    return error;
}

void launch_production(const DeviceBuffer<hip_bfloat16>& input,
                       const DeviceBuffer<hip_bfloat16>& weight,
                       DeviceBuffer<hip_bfloat16>& output, std::uint32_t rows, float eps,
                       bool unit_offset, hipStream_t stream) {
    HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_bf16(
        input.get(), weight.get(), output.get() + kGuardElements, kFeatures, rows, eps,
        unit_offset, stream));
}

void run_regression(hipStream_t stream) {
    const auto weight = make_weight();
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    upload(d_weight, weight);
    Error maximum;
    std::uint32_t cases = 0;
    constexpr std::uint32_t maximum_rows = 6U;
    for (std::uint32_t rows = 1U; rows <= maximum_rows; ++rows) {
        for (const std::uint32_t fixture : {0U, 1U, 2U}) {
            const auto input = make_input(rows, fixture);
            DeviceBuffer<hip_bfloat16> d_input(input.size());
            DeviceBuffer<hip_bfloat16> d_output(input.size() + 2U * kGuardElements);
            upload(d_input, input);
            for (const bool unit_offset : {false, true}) {
                for (const float eps : {1.0e-8F, kModelEps, 1.0e-4F}) {
                    poison_output(d_output, stream);
                    launch_production(d_input, d_weight, d_output, rows, eps, unit_offset, stream);
                    HIP_CHECK(hipStreamSynchronize(stream));
                    const Error error = require_oracle(
                        checked_output(d_output, input.size()), input, weight, rows, eps,
                        unit_offset);
                    maximum.max_abs = std::max(maximum.max_abs, error.max_abs);
                    maximum.relative_l2 = std::max(maximum.relative_l2, error.relative_l2);
                    maximum.max_bf16_steps =
                        std::max(maximum.max_bf16_steps, error.max_bf16_steps);
                    ++cases;
                }
            }
        }
    }

    for (const std::uint32_t graph_rows : {5U, 6U}) {
        const auto graph_input = make_input(graph_rows, 0U);
        DeviceBuffer<hip_bfloat16> d_input(graph_input.size());
        DeviceBuffer<hip_bfloat16> d_output(graph_input.size() + 2U * kGuardElements);
        upload(d_input, graph_input);
        poison_output(d_output, stream);
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        launch_production(d_input, d_weight, d_output, graph_rows, kModelEps, true, stream);
        HIP_CHECK(hipStreamEndCapture(stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) HIP_CHECK(hipGraphLaunch(executable, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        (void)require_oracle(checked_output(d_output, graph_input.size()), graph_input, weight,
                             graph_rows, kModelEps, true);
        HIP_CHECK(hipGraphExecDestroy(executable));
        HIP_CHECK(hipGraphDestroy(graph));
    }

    const auto malformed_input = make_input(6U, 0U);
    DeviceBuffer<hip_bfloat16> d_input(malformed_input.size());
    DeviceBuffer<hip_bfloat16> d_output(malformed_input.size() + 2U * kGuardElements);
    upload(d_input, malformed_input);
    auto reject_without_write = [&](const hip_bfloat16* in, const hip_bfloat16* gain,
                                    hip_bfloat16* out, std::uint32_t features,
                                    std::uint32_t rows, float eps, hipStream_t test_stream) {
        poison_output(d_output, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto before = output_bits(d_output);
        if (ninfer::ops::r9700::eager::rmsnorm_bf16(
                in, gain, out, features, rows, eps, true, test_stream) != hipErrorInvalidValue) {
            fail("production boundary accepted malformed input");
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        if (output_bits(d_output) != before) fail("malformed input changed guarded output");
    };
    hip_bfloat16* const out = d_output.get() + kGuardElements;
    reject_without_write(nullptr, d_weight.get(), out, kFeatures, 1U, kModelEps, stream);
    reject_without_write(d_input.get(), nullptr, out, kFeatures, 1U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), nullptr, kFeatures, 1U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 0U, 1U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, kFeatures, 0U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, kFeatures, 1U, 0.0F, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, kFeatures, 1U, -kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, kFeatures, 1U,
                         std::numeric_limits<float>::quiet_NaN(), stream);
    reject_without_write(d_input.get(), d_weight.get(), out, kFeatures, 1U,
                         std::numeric_limits<float>::infinity(), stream);
    reject_without_write(d_input.get(), d_weight.get(), out, kFeatures, 1U, kModelEps, nullptr);

    std::cout << "r9700 production RMSNorm K5120 rows1.." << maximum_rows
              << ": PASS cases=" << cases
              << " max_bf16_steps=" << maximum.max_bf16_steps
              << " graph_rows=5,6 graph_replays_per_row=2 output_canaries=true malformed_cases=10"
              << " dflash_rows56_candidate="
              << (ninfer::ops::r9700::eager::kDFlashRmsnormRows56CandidateEnabled ? "true" :
                                                                                     "false")
              << '\n';
}
} // namespace

int main(int argc, char**) {
    try {
        if (argc != 1) fail("usage: rmsnorm_decode_production");
        select_r9700();
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        run_regression(stream);
        HIP_CHECK(hipStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700 production RMSNorm K5120: FAIL: " << error.what() << '\n';
        return 1;
    }
}
