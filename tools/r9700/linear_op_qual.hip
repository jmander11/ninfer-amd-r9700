#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "ops/r9700/linear/r9700_linear.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::ops::r9700::linear::Bf16LinearArgs;
using ninfer::ops::r9700::linear::W8G32LinearArgs;

constexpr std::uint32_t kDecodeMaximumTokens = 8;
constexpr std::uint32_t kPrefillWidths[] = {16, 32, 64, 128};
constexpr std::uint32_t kRealRows = 7168;     // Qwen3.8 full-attention query_key [7168,5120]
constexpr std::uint32_t kRealColumns = 5120;
constexpr std::uint32_t kOutputHeadRows = 248320;
constexpr std::uint32_t kOutputHeadValidRows = 248077;
constexpr std::uint32_t kDraftHeadRows = 131072;
constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kKAlignment = 128;
constexpr std::uint32_t kPlaneAlignment = 256;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream out;
    out << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(out.str());
}

#define HIP_CHECK(expr) hip_check((expr), #expr, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) HIP_CHECK(hipMalloc(&data_, count_ * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    [[nodiscard]] T* get() const { return data_; }

private:
    std::size_t count_ = 0;
    T* data_ = nullptr;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() { (void)hipEventDestroy(event_); }
    [[nodiscard]] hipEvent_t get() const { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

std::string read_file_line(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "r");
    if (file == nullptr) return {};
    char buffer[128]{};
    const char* result = std::fgets(buffer, sizeof(buffer), file);
    std::fclose(file);
    if (result == nullptr) return {};
    std::string value(buffer);
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
    return value;
}

std::string normalized_pci(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (value.size() == 7 && value[2] == ':' && value[5] == '.') value = "0000:" + value;
    return value;
}

void select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    const char* requested_env = std::getenv("NINFER_R9700_PCI_BUS_ID");
    const std::string requested = requested_env == nullptr ? "" : normalized_pci(requested_env);
    std::vector<int> matches;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        char pci_buffer[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_buffer, sizeof(pci_buffer), ordinal));
        const std::string pci = normalized_pci(pci_buffer);
        const std::string sysfs = "/sys/bus/pci/devices/" + pci;
        if (properties.integrated == 0 && std::string_view(properties.gcnArchName) == "gfx1201" &&
            read_file_line(sysfs + "/vendor") == "0x1002" &&
            read_file_line(sysfs + "/device") == "0x7551" &&
            (requested.empty() || requested == pci)) {
            matches.push_back(ordinal);
        }
    }
    if (matches.size() != 1) fail("expected exactly one discrete 1002:7551 gfx1201 R9700");
    HIP_CHECK(hipSetDevice(matches.front()));
    hipDeviceProp_t selected{};
    HIP_CHECK(hipGetDeviceProperties(&selected, matches.front()));
    if (selected.warpSize != 32) fail("R9700 linear qualifier requires wave32");
    std::cout << "device: arch=" << selected.gcnArchName << " wave=" << selected.warpSize
              << " vram_mib=" << selected.totalGlobalMem / (1024 * 1024) << '\n';
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

std::uint16_t half_bits(float value) {
    const _Float16 half = static_cast<_Float16>(value);
    return std::bit_cast<std::uint16_t>(half);
}

float half_from_bits(std::uint16_t bits) {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

__global__ void initialize_product_head_w8(std::int8_t* codes, std::uint16_t* scales,
                                           std::uint32_t rows, std::uint32_t padded_columns,
                                           std::uint32_t contender, std::uint32_t winner) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const std::uint32_t groups = padded_columns / kGroup;
    const std::size_t code_base = static_cast<std::size_t>(row) * padded_columns;
    const std::size_t scale_base = static_cast<std::size_t>(row) * groups;
    codes[code_base] = static_cast<std::int8_t>(row == contender || row == winner ? 1 : -1);
    codes[code_base + 31U] = static_cast<std::int8_t>(static_cast<int>(row % 5U) - 2);
    codes[code_base + 32U] = static_cast<std::int8_t>(static_cast<int>(row % 7U) - 3);
    codes[code_base + padded_columns - 1U] =
        static_cast<std::int8_t>(static_cast<int>(row % 3U) - 1);
    scales[scale_base] = row == winner ? 0x3c08U : 0x3c00U; // 1.0078125 or 1.0
    scales[scale_base + 1U] = 0x3400U;                       // 0.25
    scales[scale_base + groups - 1U] = 0x2c00U;              // 0.0625
}

struct W8Payload {
    std::vector<std::uint8_t> bytes;
    std::uint32_t padded_columns = 0;
    std::size_t scale_offset = 0;
};

W8Payload encode_w8g32_independent(const std::vector<hip_bfloat16>& weights,
                                    std::uint32_t rows, std::uint32_t columns) {
    if (weights.size() != static_cast<std::size_t>(rows) * columns) fail("W8 source shape mismatch");
    const std::uint32_t padded = static_cast<std::uint32_t>(align_up(columns, kKAlignment));
    const std::uint32_t groups = padded / kGroup;
    const std::size_t base_bytes = static_cast<std::size_t>(rows) * padded;
    const std::size_t scale_offset = align_up(base_bytes, kPlaneAlignment);
    W8Payload out{std::vector<std::uint8_t>(scale_offset + static_cast<std::size_t>(rows) * groups * 2U),
                  padded, scale_offset};
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            float maximum = 0.0F;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const float value = column < columns
                                        ? static_cast<float>(weights[static_cast<std::size_t>(row) * columns + column])
                                        : 0.0F;
                if (!std::isfinite(value)) fail("W8 source is nonfinite");
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint16_t scale_bits = half_bits(maximum / 127.0F);
            if (maximum != 0.0F && scale_bits == 0U) scale_bits = 1U;
            const float scale = half_from_bits(scale_bits);
            if (maximum != 0.0F && (!std::isfinite(scale) || scale <= 0.0F)) {
                fail("W8 source scale is unrepresentable");
            }
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const float value = column < columns
                                        ? static_cast<float>(weights[static_cast<std::size_t>(row) * columns + column])
                                        : 0.0F;
                const int code = scale_bits == 0U ? 0 : static_cast<int>(std::nearbyint(value / scale));
                out.bytes[static_cast<std::size_t>(row) * padded + column] =
                    static_cast<std::uint8_t>(std::clamp(code, -127, 127));
            }
            const std::size_t scale_index = static_cast<std::size_t>(row) * groups + group;
            std::memcpy(out.bytes.data() + scale_offset + scale_index * 2U, &scale_bits, sizeof(scale_bits));
        }
    }
    return out;
}

float decode_w8_weight(const W8Payload& packed, std::uint32_t row, std::uint32_t column) {
    const std::uint32_t groups = packed.padded_columns / kGroup;
    const std::int8_t code = static_cast<std::int8_t>(packed.bytes[static_cast<std::size_t>(row) * packed.padded_columns + column]);
    std::uint16_t scale_bits = 0;
    const std::size_t scale_index = static_cast<std::size_t>(row) * groups + column / kGroup;
    std::memcpy(&scale_bits, packed.bytes.data() + packed.scale_offset + scale_index * 2U, sizeof(scale_bits));
    return static_cast<float>(code) * half_from_bits(scale_bits);
}

W8G32LinearArgs w8_args(const hip_bfloat16* input, const std::uint8_t* payload,
                        hip_bfloat16* output, const W8Payload& encoded,
                        std::uint32_t tokens, std::uint32_t rows,
                        std::uint32_t columns) {
    const std::size_t code_bytes = static_cast<std::size_t>(rows) * encoded.padded_columns;
    const std::size_t scale_bytes = encoded.bytes.size() - encoded.scale_offset;
    return W8G32LinearArgs{
        .input = input,
        .codes = reinterpret_cast<const std::int8_t*>(payload),
        .code_bytes = code_bytes,
        .scales = reinterpret_cast<const std::uint16_t*>(payload + encoded.scale_offset),
        .scale_bytes = scale_bytes,
        .output = output,
        .tokens = tokens,
        .rows = rows,
        .columns = columns,
        .padded_columns = encoded.padded_columns,
    };
}

void test_w8_converter_codec_fixture() {
    // This is an exact published scalar-codec fixture: signed RNE-even ties
    // in the first G32 group, then a nonzero K-padded second group. It keeps
    // the C++ qualifier's raw payload contract tied to the converter authority
    // without importing Python or Torch into a ROCm device test.
    std::vector<hip_bfloat16> weights(33, hip_bfloat16(0.0F));
    const float first[] = {-127.0F, -126.5F, -1.5F, -0.5F, 0.5F, 1.5F, 126.5F, 127.0F};
    for (std::uint32_t column = 0; column < 8U; ++column) weights[column] = hip_bfloat16(first[column]);
    weights[32] = hip_bfloat16(127.0F);
    const W8Payload packed = encode_w8g32_independent(weights, 1, 33);
    if (packed.bytes.size() != 264U || packed.padded_columns != 128U || packed.scale_offset != 256U) {
        fail("W8 converter fixture layout mismatch");
    }
    const std::uint8_t expected_codes[] = {0x81U, 0x82U, 0xfeU, 0x00U, 0x00U, 0x02U, 0x7eU, 0x7fU};
    if (!std::equal(std::begin(expected_codes), std::end(expected_codes), packed.bytes.begin()) ||
        packed.bytes[32] != 0x7fU) {
        fail("W8 converter fixture code bytes mismatch");
    }
    const std::uint16_t expected_scales[] = {0x3c00U, 0x3c00U, 0x0000U, 0x0000U};
    for (std::size_t index = 0; index < std::size(expected_scales); ++index) {
        std::uint16_t actual = 0;
        std::memcpy(&actual, packed.bytes.data() + packed.scale_offset + index * 2U, sizeof(actual));
        if (actual != expected_scales[index]) fail("W8 converter fixture scale bytes mismatch");
    }
    std::cout << "codec: w8g32 converter fixture PASS\n";
}

std::vector<hip_bfloat16> make_input(std::uint32_t tokens, std::uint32_t columns) {
    std::vector<hip_bfloat16> out(static_cast<std::size_t>(tokens) * columns);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((token * 17U + column * 13U) % 31U) - 15;
            out[static_cast<std::size_t>(token) * columns + column] =
                hip_bfloat16(static_cast<float>(pattern) / 128.0F);
        }
    }
    return out;
}

std::vector<hip_bfloat16> make_weights(std::uint32_t rows, std::uint32_t columns, bool edges) {
    std::vector<hip_bfloat16> out(static_cast<std::size_t>(rows) * columns);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 19U + column * 7U) % 47U) - 23;
            out[static_cast<std::size_t>(row) * columns + column] =
                hip_bfloat16(static_cast<float>(pattern) / 256.0F);
        }
    }
    if (edges && rows >= 1U && columns >= 8U) {
        const float values[] = {-127.0F, -126.5F, -1.5F, -0.5F, 0.5F, 1.5F, 126.5F, 127.0F};
        for (std::uint32_t column = 0; column < 8U; ++column) out[column] = hip_bfloat16(values[column]);
    }
    return out;
}

std::vector<hip_bfloat16> oracle_fp64(const std::vector<hip_bfloat16>& input,
                                       const std::vector<hip_bfloat16>& bf16_weights,
                                       const W8Payload* w8, std::uint32_t tokens,
                                       std::uint32_t rows, std::uint32_t columns) {
    std::vector<hip_bfloat16> out(static_cast<std::size_t>(tokens) * rows);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (std::uint32_t column = 0; column < columns; ++column) {
                const double x = static_cast<float>(input[static_cast<std::size_t>(token) * columns + column]);
                const double w = w8 == nullptr
                                     ? static_cast<float>(bf16_weights[static_cast<std::size_t>(row) * columns + column])
                                     : decode_w8_weight(*w8, row, column);
                sum = std::fma(x, w, sum);
            }
            out[static_cast<std::size_t>(token) * rows + row] = hip_bfloat16(static_cast<float>(sum));
        }
    }
    return out;
}

double check_output(const std::string& label, const std::vector<hip_bfloat16>& actual,
                    const std::vector<hip_bfloat16>& expected, double tolerance) {
    if (actual.size() != expected.size()) fail(label + ": output extent mismatch");
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = std::abs(static_cast<double>(static_cast<float>(actual[index])) -
                                      static_cast<double>(static_cast<float>(expected[index])));
        maximum = std::max(maximum, error);
        if (error > tolerance) {
            std::ostringstream out;
            out << label << ": FP64 oracle mismatch at " << index << ", error=" << error
                << " tolerance=" << tolerance;
            fail(out.str());
        }
    }
    return maximum;
}

double check_output_samples(const std::string& label, const std::vector<hip_bfloat16>& actual,
                            const std::vector<hip_bfloat16>& input,
                            const std::vector<hip_bfloat16>& bf16_weights,
                            const W8Payload* w8, std::uint32_t tokens, std::uint32_t rows,
                            std::uint32_t columns, double tolerance) {
    if (actual.size() != static_cast<std::size_t>(tokens) * rows) fail(label + ": output extent mismatch");
    const std::uint32_t token_samples[] = {0U, tokens / 2U, tokens - 1U};
    const std::uint32_t row_samples[] = {0U, rows / 2U, rows - 1U};
    double maximum = 0.0;
    for (const std::uint32_t token : token_samples) {
        for (const std::uint32_t row : row_samples) {
            double sum = 0.0;
            for (std::uint32_t column = 0; column < columns; ++column) {
                const double x = static_cast<float>(input[static_cast<std::size_t>(token) * columns + column]);
                const double weight = w8 == nullptr
                                          ? static_cast<float>(bf16_weights[static_cast<std::size_t>(row) * columns + column])
                                          : decode_w8_weight(*w8, row, column);
                sum = std::fma(x, weight, sum);
            }
            const hip_bfloat16 expected = hip_bfloat16(static_cast<float>(sum));
            const double error = std::abs(static_cast<double>(static_cast<float>(
                actual[static_cast<std::size_t>(token) * rows + row])) -
                                          static_cast<double>(static_cast<float>(expected)));
            maximum = std::max(maximum, error);
            if (error > tolerance) {
                std::ostringstream out;
                out << label << ": FP64 oracle sample mismatch token=" << token << " row=" << row
                    << " error=" << error << " tolerance=" << tolerance;
                fail(out.str());
            }
        }
    }
    return maximum;
}

template <typename Launch>
double time_ms(int iterations, Launch&& launch) {
    for (int warmup = 0; warmup < 2; ++warmup) launch();
    HIP_CHECK(hipDeviceSynchronize());
    Event start;
    Event stop;
    HIP_CHECK(hipEventRecord(start.get()));
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    HIP_CHECK(hipEventRecord(stop.get()));
    HIP_CHECK(hipEventSynchronize(stop.get()));
    float total = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&total, start.get(), stop.get()));
    return static_cast<double>(total) / iterations;
}

void run_correctness_case(const std::string& label, std::uint32_t tokens, std::uint32_t rows,
                          std::uint32_t columns, bool edges) {
    const auto input = make_input(tokens, columns);
    const auto weights = make_weights(rows, columns, edges);
    const W8Payload w8 = encode_w8g32_independent(weights, rows, columns);
    const auto bf16_expected = oracle_fp64(input, weights, nullptr, tokens, rows, columns);
    const auto w8_expected = oracle_fp64(input, weights, &w8, tokens, rows, columns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_weights(weights.size());
    DeviceBuffer<std::uint8_t> d_w8(w8.bytes.size());
    DeviceBuffer<hip_bfloat16> d_output(std::max(bf16_expected.size(), w8_expected.size()));
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights.get(), weights.data(), weights.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w8.get(), w8.bytes.data(), w8.bytes.size(), hipMemcpyHostToDevice));
    const Bf16LinearArgs bf16{d_input.get(), d_weights.get(), d_output.get(), tokens, rows, columns};
    const W8G32LinearArgs w8_launch =
        w8_args(d_input.get(), d_w8.get(), d_output.get(), w8, tokens, rows, columns);
    auto check_route = [&](const std::string& route, auto launch,
                           const std::vector<hip_bfloat16>& expected) {
        HIP_CHECK(launch());
        std::vector<hip_bfloat16> actual(expected.size());
        HIP_CHECK(hipMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        // The public result boundary is BF16. The only permitted discrepancy
        // from the sequential FP64 oracle is one low-order BF16 unit caused by
        // the GPU's parallel FP32 reduction association.
        return check_output(label + " " + route, actual, expected, 0.0625);
    };
    const double bf16_baseline = check_route("bf16-baseline", [&] {
        return ninfer::ops::r9700::linear::bf16_linear_baseline(bf16, nullptr);
    }, bf16_expected);
    const double w8_baseline = check_route("w8g32-baseline", [&] {
        return ninfer::ops::r9700::linear::w8g32_linear_baseline(w8_launch, nullptr);
    }, w8_expected);
    const double bf16_selected = check_route("bf16-selected", [&] {
        return ninfer::ops::r9700::linear::bf16_linear(bf16, nullptr);
    }, bf16_expected);
    const double w8_selected = check_route("w8g32-selected", [&] {
        return ninfer::ops::r9700::linear::w8g32_linear(w8_launch, nullptr);
    }, w8_expected);
    if (tokens <= kDecodeMaximumTokens) {
        const double bf16_wave8 = check_route("bf16-wave8", [&] {
            return ninfer::ops::r9700::linear::bf16_linear_wave8(bf16, nullptr);
        }, bf16_expected);
        const double w8_wave8 = check_route("w8g32-wave8", [&] {
            return ninfer::ops::r9700::linear::w8g32_linear_wave8(w8_launch, nullptr);
        }, w8_expected);
        std::cout << "oracle: " << label << " PASS max_abs=[bf16_base=" << bf16_baseline
                  << ",bf16_wave8=" << bf16_wave8 << ",w8_base=" << w8_baseline
                  << ",w8_wave8=" << w8_wave8 << ",bf16_selected=" << bf16_selected
                  << ",w8_selected=" << w8_selected << "]\n";
        return;
    }
    const double bf16_wave16 = check_route("bf16-wave16", [&] {
        return ninfer::ops::r9700::linear::bf16_linear_wave16(bf16, nullptr);
    }, bf16_expected);
    const double bf16_wmma16 = check_route("bf16-wmma16", [&] {
        return ninfer::ops::r9700::linear::bf16_linear_wmma16(bf16, nullptr);
    }, bf16_expected);
    const double w8_wave16 = check_route("w8g32-wave16", [&] {
        return ninfer::ops::r9700::linear::w8g32_linear_wave16(w8_launch, nullptr);
    }, w8_expected);
    std::cout << "oracle: " << label << " PASS max_abs=[bf16_base=" << bf16_baseline
              << ",bf16_wave16=" << bf16_wave16 << ",bf16_wmma16=" << bf16_wmma16
              << ",w8_base=" << w8_baseline << ",w8_wave16=" << w8_wave16
              << ",bf16_selected=" << bf16_selected << ",w8_selected=" << w8_selected
              << "]\n";
}

void run_public_tensor_case(std::uint32_t tokens) {
    constexpr std::uint32_t rows = 17;
    constexpr std::uint32_t columns = 64;
    const auto input = make_input(tokens, columns);
    const auto weights = make_weights(rows, columns, true);
    const W8Payload w8 = encode_w8g32_independent(weights, rows, columns);
    const auto bf16_expected = oracle_fp64(input, weights, nullptr, tokens, rows, columns);
    const auto w8_expected = oracle_fp64(input, weights, &w8, tokens, rows, columns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_weights(weights.size());
    DeviceBuffer<std::uint8_t> d_w8(w8.bytes.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(tokens) * rows);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights.get(), weights.data(), weights.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w8.get(), w8.bytes.data(), w8.bytes.size(), hipMemcpyHostToDevice));

    ninfer::Tensor x(d_input.get(), ninfer::DType::BF16,
                     {static_cast<std::int32_t>(columns), static_cast<std::int32_t>(tokens)});
    ninfer::Tensor output(d_output.get(), ninfer::DType::BF16,
                          {static_cast<std::int32_t>(rows), static_cast<std::int32_t>(tokens)});
    ninfer::Weight bf16_weight{};
    bf16_weight.payload = d_weights.get();
    bf16_weight.payload_bytes = weights.size() * sizeof(hip_bfloat16);
    bf16_weight.qdata = d_weights.get();
    bf16_weight.qdata_bytes = bf16_weight.payload_bytes;
    bf16_weight.qtype = ninfer::QType::BF16_CTRL;
    bf16_weight.layout = ninfer::QuantLayout::Contiguous;
    bf16_weight.n = rows;
    bf16_weight.k = columns;
    bf16_weight.ndim = 2;
    bf16_weight.shape[0] = rows;
    bf16_weight.shape[1] = columns;
    bf16_weight.padded_shape[0] = rows;
    bf16_weight.padded_shape[1] = columns;

    const std::uint64_t code_bytes = static_cast<std::uint64_t>(rows) * w8.padded_columns;
    const std::uint64_t scale_bytes = w8.bytes.size() - w8.scale_offset;
    ninfer::Weight w8_weight{};
    w8_weight.payload = d_w8.get();
    w8_weight.payload_bytes = w8.bytes.size();
    w8_weight.qdata = d_w8.get();
    w8_weight.qdata_bytes = code_bytes;
    w8_weight.scales = d_w8.get() + w8.scale_offset;
    w8_weight.scale_bytes = scale_bytes;
    w8_weight.qtype = ninfer::QType::W8G32_F16S;
    w8_weight.group_size = kGroup;
    w8_weight.n = rows;
    w8_weight.k = columns;
    w8_weight.group = kGroup;
    w8_weight.layout = ninfer::QuantLayout::RowSplit;
    w8_weight.scale_dtype = ninfer::DType::FP16;
    w8_weight.ndim = 2;
    w8_weight.shape[0] = rows;
    w8_weight.shape[1] = columns;
    w8_weight.padded_shape[0] = rows;
    w8_weight.padded_shape[1] = w8.padded_columns;

    auto run_and_check = [&](const std::string& label, const ninfer::Weight& weight,
                             const std::vector<hip_bfloat16>& expected) {
        ninfer::ops::linear(x, weight, output, nullptr);
        std::vector<hip_bfloat16> actual(expected.size());
        HIP_CHECK(hipMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        (void)check_output(label, actual, expected, 0.0625);
    };
    run_and_check("public BF16 T=" + std::to_string(tokens), bf16_weight, bf16_expected);
    run_and_check("public W8 T=" + std::to_string(tokens), w8_weight, w8_expected);

    constexpr std::uint32_t first_row = 1;
    constexpr std::uint32_t sliced_rows = 3;
    const std::uint64_t scale_row_bytes = (w8.padded_columns / kGroup) * sizeof(std::uint16_t);
    ninfer::Weight sliced = w8_weight;
    sliced.qdata = d_w8.get() + static_cast<std::uint64_t>(first_row) * w8.padded_columns;
    sliced.qdata_bytes = static_cast<std::uint64_t>(sliced_rows) * w8.padded_columns;
    sliced.scales = d_w8.get() + w8.scale_offset + first_row * scale_row_bytes;
    sliced.scale_bytes = sliced_rows * scale_row_bytes;
    sliced.n = sliced_rows;
    sliced.shape[0] = sliced_rows;
    sliced.padded_shape[0] = sliced_rows;
    ninfer::Tensor sliced_output(
        d_output.get(), ninfer::DType::BF16,
        {static_cast<std::int32_t>(sliced_rows), static_cast<std::int32_t>(tokens)});
    ninfer::ops::linear(x, sliced, sliced_output, nullptr);
    std::vector<hip_bfloat16> sliced_actual(static_cast<std::size_t>(tokens) * sliced_rows);
    HIP_CHECK(hipMemcpy(sliced_actual.data(), d_output.get(),
                        sliced_actual.size() * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
    std::vector<hip_bfloat16> sliced_expected(sliced_actual.size());
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t row = 0; row < sliced_rows; ++row) {
            sliced_expected[static_cast<std::size_t>(token) * sliced_rows + row] =
                w8_expected[static_cast<std::size_t>(token) * rows + first_row + row];
        }
    }
    (void)check_output("public W8 row-view T=" + std::to_string(tokens), sliced_actual,
                       sliced_expected, 0.0625);

    ninfer::Weight malformed = w8_weight;
    --malformed.scale_bytes;
    try {
        ninfer::ops::linear(x, malformed, output, nullptr);
        fail("public linear accepted a short W8 scale plane");
    } catch (const std::invalid_argument&) {
    }
    std::cout << "public-linear: T=" << tokens << " BF16/W8/row-view/rejection PASS\n";
}

std::int8_t product_head_code(std::uint32_t row, std::uint32_t column,
                              std::uint32_t contender, std::uint32_t winner) {
    if (column == 0U) {
        return static_cast<std::int8_t>(row == contender || row == winner ? 1 : -1);
    }
    if (column == 31U) {
        return static_cast<std::int8_t>(static_cast<int>(row % 5U) - 2);
    }
    if (column == 32U) {
        return static_cast<std::int8_t>(static_cast<int>(row % 7U) - 3);
    }
    if (column == kRealColumns - 1U) {
        return static_cast<std::int8_t>(static_cast<int>(row % 3U) - 1);
    }
    return 0;
}

std::uint16_t product_head_scale_bits(std::uint32_t row, std::uint32_t group,
                                      std::uint32_t winner) {
    if (group == 0U) return row == winner ? 0x3c08U : 0x3c00U;
    if (group == 1U) return 0x3400U;
    if (group == kRealColumns / kGroup - 1U) return 0x2c00U;
    return 0U;
}

std::vector<hip_bfloat16> product_head_input(std::uint32_t tokens) {
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * kRealColumns,
                                    hip_bfloat16(0.0F));
    input[0] = hip_bfloat16(1.0F);
    if (tokens >= 2U) {
        input[kRealColumns + 31U] = hip_bfloat16(0.5F);
        input[kRealColumns + 32U] = hip_bfloat16(-0.25F);
        input[2U * kRealColumns - 1U] = hip_bfloat16(1.0F);
    }
    if (tokens >= 3U) {
        input[2U * kRealColumns] = hip_bfloat16(-0.5F);
        input[2U * kRealColumns + 31U] = hip_bfloat16(0.25F);
        input[2U * kRealColumns + 32U] = hip_bfloat16(0.125F);
        input[3U * kRealColumns - 1U] = hip_bfloat16(-0.5F);
    }
    return input;
}

std::vector<hip_bfloat16> product_head_oracle(const std::vector<hip_bfloat16>& input,
                                               std::uint32_t tokens, std::uint32_t rows,
                                               std::uint32_t contender,
                                               std::uint32_t winner) {
    constexpr std::array<std::uint32_t, 4> nonzero_columns{0U, 31U, 32U,
                                                           kRealColumns - 1U};
    std::vector<hip_bfloat16> expected(static_cast<std::size_t>(tokens) * rows);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (const std::uint32_t column : nonzero_columns) {
                const double x = static_cast<float>(
                    input[static_cast<std::size_t>(token) * kRealColumns + column]);
                const double code = product_head_code(row, column, contender, winner);
                const double scale = half_from_bits(
                    product_head_scale_bits(row, column / kGroup, winner));
                sum = std::fma(x, code * scale, sum);
            }
            expected[static_cast<std::size_t>(token) * rows + row] =
                hip_bfloat16(static_cast<float>(sum));
        }
    }
    return expected;
}

void require_product_head_exact(const std::string& label,
                                const std::vector<hip_bfloat16>& actual,
                                const std::vector<hip_bfloat16>& expected) {
    if (actual.size() != expected.size()) fail(label + ": output extent mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].data != expected[index].data) {
            std::ostringstream out;
            out << label << ": exact BF16 oracle mismatch at " << index << " actual=0x"
                << std::hex << actual[index].data << " expected=0x" << expected[index].data;
            fail(out.str());
        }
    }
}

void run_product_head_shape(const char* label, std::uint32_t rows,
                            std::uint32_t valid_rows, bool check_argmax) {
    constexpr std::uint32_t maximum_tokens = 3;
    constexpr std::uint32_t padded_columns = kRealColumns;
    constexpr std::uint32_t groups = padded_columns / kGroup;
    const std::uint32_t winner = valid_rows - 1U;
    const std::uint32_t contender = winner - 1U;
    const std::size_t code_elements = static_cast<std::size_t>(rows) * padded_columns;
    const std::size_t scale_elements = static_cast<std::size_t>(rows) * groups;
    const auto input = product_head_input(maximum_tokens);
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::int8_t> d_codes(code_elements);
    DeviceBuffer<std::uint16_t> d_scales(scale_elements);
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(rows) * maximum_tokens);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemsetAsync(d_codes.get(), 0, code_elements, stream));
    HIP_CHECK(hipMemsetAsync(d_scales.get(), 0, scale_elements * sizeof(std::uint16_t), stream));
    constexpr std::uint32_t threads = 256;
    hipLaunchKernelGGL(initialize_product_head_w8, dim3((rows + threads - 1U) / threads),
                       dim3(threads), 0, stream, d_codes.get(), d_scales.get(), rows,
                       padded_columns, contender, winner);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    ninfer::Weight weight{};
    weight.payload = d_codes.get();
    weight.payload_bytes = code_elements + scale_elements * sizeof(std::uint16_t);
    weight.qdata = d_codes.get();
    weight.qdata_bytes = code_elements;
    weight.scales = d_scales.get();
    weight.scale_bytes = scale_elements * sizeof(std::uint16_t);
    weight.qtype = ninfer::QType::W8G32_F16S;
    weight.group_size = weight.group = kGroup;
    weight.layout = ninfer::QuantLayout::RowSplit;
    weight.scale_dtype = ninfer::DType::FP16;
    weight.ndim = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = static_cast<std::int32_t>(rows);
    weight.k = weight.shape[1] = weight.padded_shape[1] =
        static_cast<std::int32_t>(kRealColumns);

    for (const std::uint32_t tokens : {1U, maximum_tokens}) {
        ninfer::Tensor x(d_input.get(), ninfer::DType::BF16,
                         {static_cast<std::int32_t>(kRealColumns),
                          static_cast<std::int32_t>(tokens)});
        ninfer::Tensor output(d_output.get(), ninfer::DType::BF16,
                              {static_cast<std::int32_t>(rows),
                               static_cast<std::int32_t>(tokens)});
        const auto expected = product_head_oracle(input, tokens, rows, contender, winner);
        ninfer::ops::linear(x, weight, output, stream);
        ninfer::ops::linear(x, weight, output, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        Event start;
        Event stop;
        HIP_CHECK(hipEventRecord(start.get(), stream));
        ninfer::ops::linear(x, weight, output, stream);
        HIP_CHECK(hipEventRecord(stop.get(), stream));
        HIP_CHECK(hipEventSynchronize(stop.get()));
        float milliseconds = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start.get(), stop.get()));
        std::vector<hip_bfloat16> actual(expected.size());
        HIP_CHECK(hipMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        require_product_head_exact(std::string(label) + " T=" + std::to_string(tokens),
                                   actual, expected);

        if (check_argmax) {
            DeviceBuffer<std::int32_t> d_tokens(tokens);
            ninfer::Tensor selected(d_tokens.get(), ninfer::DType::I32,
                                    {static_cast<std::int32_t>(tokens)});
            ninfer::ops::argmax(output, selected, static_cast<std::int32_t>(valid_rows), stream);
            HIP_CHECK(hipStreamSynchronize(stream));
            std::vector<std::int32_t> actual_tokens(tokens);
            HIP_CHECK(hipMemcpy(actual_tokens.data(), d_tokens.get(),
                                actual_tokens.size() * sizeof(std::int32_t),
                                hipMemcpyDeviceToHost));
            for (std::uint32_t token = 0; token < tokens; ++token) {
                std::uint32_t expected_token = 0;
                float maximum = static_cast<float>(expected[static_cast<std::size_t>(token) * rows]);
                for (std::uint32_t row = 1; row < valid_rows; ++row) {
                    const float value = static_cast<float>(
                        expected[static_cast<std::size_t>(token) * rows + row]);
                    if (value > maximum) {
                        maximum = value;
                        expected_token = row;
                    }
                }
                if (actual_tokens[token] != static_cast<std::int32_t>(expected_token)) {
                    fail(std::string(label) + ": composed argmax differs from exact oracle");
                }
            }
            const float winning = static_cast<float>(expected[winner]);
            const float runner_up = static_cast<float>(expected[contender]);
            if (actual_tokens[0] != static_cast<std::int32_t>(winner) ||
                winning - runner_up != 0.0078125F) {
                fail(std::string(label) + ": near-tie argmax fixture is not one BF16 ULP");
            }
        }
        std::cout << "w8-product-head: " << label << " shape=[" << rows << ','
                  << kRealColumns << "] T=" << tokens
                  << " exact_sparse_fp64_to_bf16=PASS"
                  << (check_argmax ? " exact_argmax=PASS near_tie_ulp=1" : "")
                  << " kernel_ms=" << milliseconds << '\n';
    }

    ninfer::Tensor x(d_input.get(), ninfer::DType::BF16,
                     {static_cast<std::int32_t>(kRealColumns), 1});
    ninfer::Tensor output(d_output.get(), ninfer::DType::BF16,
                          {static_cast<std::int32_t>(rows), 1});
    ninfer::Weight malformed = weight;
    --malformed.qdata_bytes;
    try {
        ninfer::ops::linear(x, malformed, output, stream);
        fail(std::string(label) + ": accepted short W8 code plane");
    } catch (const std::invalid_argument&) {
    }
    malformed = weight;
    --malformed.scale_bytes;
    try {
        ninfer::ops::linear(x, malformed, output, stream);
        fail(std::string(label) + ": accepted short W8 scale plane");
    } catch (const std::invalid_argument&) {
    }
    malformed = weight;
    malformed.padded_shape[1] += kKAlignment;
    try {
        ninfer::ops::linear(x, malformed, output, stream);
        fail(std::string(label) + ": accepted incorrect W8 K padding");
    } catch (const std::invalid_argument&) {
    }

    const double resident_mib = static_cast<double>(
        code_elements + scale_elements * sizeof(std::uint16_t) +
        input.size() * sizeof(hip_bfloat16) +
        static_cast<std::size_t>(rows) * maximum_tokens * sizeof(hip_bfloat16)) /
        static_cast<double>(1U << 20U);
    std::cout << "w8-product-head: " << label << " code/scale/padding rejection=PASS"
              << " peak_fixture_mib=" << resident_mib << '\n';
    HIP_CHECK(hipStreamDestroy(stream));
}

void run_prefill_correctness_case(std::uint32_t tokens, bool full_oracle) {
    // All widths share represented source data and the one uploaded complete
    // real matrix. That keeps the oracle gate focused on arithmetic rather
    // than repeatedly spending host time repacking an identical 36.7M-weight
    // candidate payload.
    static const auto input = make_input(128, kRealColumns);
    static const auto weights = make_weights(kRealRows, kRealColumns, false);
    static const W8Payload w8 = encode_w8g32_independent(weights, kRealRows, kRealColumns);
    static DeviceBuffer<hip_bfloat16> d_input(input.size());
    static DeviceBuffer<hip_bfloat16> d_weights(weights.size());
    static DeviceBuffer<std::uint8_t> d_w8(w8.bytes.size());
    static DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(128) * kRealRows);
    static bool uploaded = false;
    if (!uploaded) {
        HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_weights.get(), weights.data(), weights.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w8.get(), w8.bytes.data(), w8.bytes.size(), hipMemcpyHostToDevice));
        uploaded = true;
    }
    // The complete T=16 real shape is checked once. Wider real-shape runs
    // independently recompute three spread tokens by three spread rows in
    // FP64, retaining the full [N,K] device workload without making an
    // impractically large host oracle the timing bottleneck.
    const auto bf16_expected = full_oracle
                                   ? oracle_fp64(input, weights, nullptr, tokens, kRealRows, kRealColumns)
                                   : std::vector<hip_bfloat16>{};
    const auto w8_expected = full_oracle
                                 ? oracle_fp64(input, weights, &w8, tokens, kRealRows, kRealColumns)
                                 : std::vector<hip_bfloat16>{};
    const Bf16LinearArgs bf16{d_input.get(), d_weights.get(), d_output.get(), tokens, kRealRows, kRealColumns};
    const W8G32LinearArgs w8_launch = w8_args(d_input.get(), d_w8.get(), d_output.get(), w8,
                                               tokens, kRealRows, kRealColumns);
    auto check_route = [&](const std::string& route, auto launch, const W8Payload* codec,
                           const std::vector<hip_bfloat16>& expected) {
        HIP_CHECK(launch());
        std::vector<hip_bfloat16> actual(static_cast<std::size_t>(tokens) * kRealRows);
        HIP_CHECK(hipMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        return full_oracle
                   ? check_output(route, actual, expected, 0.0625)
                   : check_output_samples(route, actual, input, weights, codec, tokens, kRealRows,
                                          kRealColumns, 0.0625);
    };
    const double bf16_base = check_route("prefill bf16-baseline T=" + std::to_string(tokens), [&] {
        return ninfer::ops::r9700::linear::bf16_linear_baseline(bf16, nullptr);
    }, nullptr, bf16_expected);
    const double bf16_wave16 = check_route("prefill bf16-wave16 T=" + std::to_string(tokens), [&] {
        return ninfer::ops::r9700::linear::bf16_linear_wave16(bf16, nullptr);
    }, nullptr, bf16_expected);
    const double bf16_wmma = check_route("prefill bf16-wmma16 T=" + std::to_string(tokens), [&] {
        return ninfer::ops::r9700::linear::bf16_linear_wmma16(bf16, nullptr);
    }, nullptr, bf16_expected);
    const double bf16_selected = check_route("prefill bf16-selected T=" + std::to_string(tokens), [&] {
        return ninfer::ops::r9700::linear::bf16_linear_prefill(bf16, nullptr);
    }, nullptr, bf16_expected);
    const double w8_base = check_route("prefill w8g32-baseline T=" + std::to_string(tokens), [&] {
        return ninfer::ops::r9700::linear::w8g32_linear_baseline(w8_launch, nullptr);
    }, &w8, w8_expected);
    const double w8_wave16 = check_route("prefill w8g32-wave16 T=" + std::to_string(tokens), [&] {
        return ninfer::ops::r9700::linear::w8g32_linear_wave16(w8_launch, nullptr);
    }, &w8, w8_expected);
    const double w8_selected = check_route("prefill w8g32-selected T=" + std::to_string(tokens), [&] {
        return ninfer::ops::r9700::linear::w8g32_linear_prefill(w8_launch, nullptr);
    }, &w8, w8_expected);
    std::cout << "oracle: qwen3.8-full-attention-query-key prefill T=" << tokens
              << (full_oracle ? " full" : " sampled") << " PASS max_abs=[bf16_base=" << bf16_base
              << ",bf16_wave16=" << bf16_wave16 << ",bf16_wmma16=" << bf16_wmma
              << ",bf16_selected=" << bf16_selected << ",w8_base=" << w8_base
              << ",w8_wave16=" << w8_wave16 << ",w8_selected=" << w8_selected << "]\n";
}

void run_timings(int iterations) {
    const auto input = make_input(kDecodeMaximumTokens, kRealColumns);
    const auto weights = make_weights(kRealRows, kRealColumns, false);
    const W8Payload w8 = encode_w8g32_independent(weights, kRealRows, kRealColumns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_weights(weights.size());
    DeviceBuffer<std::uint8_t> d_w8(w8.bytes.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(kDecodeMaximumTokens) * kRealRows);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights.get(), weights.data(), weights.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w8.get(), w8.bytes.data(), w8.bytes.size(), hipMemcpyHostToDevice));
    for (std::uint32_t tokens = 1; tokens <= kDecodeMaximumTokens; ++tokens) {
        const Bf16LinearArgs bf16{d_input.get(), d_weights.get(), d_output.get(), tokens, kRealRows, kRealColumns};
        const W8G32LinearArgs w8_launch = w8_args(
            d_input.get(), d_w8.get(), d_output.get(), w8, tokens, kRealRows, kRealColumns);
        const double bf16_base_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::bf16_linear_baseline(bf16, nullptr));
        });
        const double bf16_wave8_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::bf16_linear_wave8(bf16, nullptr));
        });
        const double w8_base_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::w8g32_linear_baseline(w8_launch, nullptr));
        });
        const double w8_wave8_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::w8g32_linear_wave8(w8_launch, nullptr));
        });
        const double operations = 2.0 * tokens * kRealRows * kRealColumns;
        std::cout << std::fixed << std::setprecision(3)
                  << "timing: shape=[" << kRealRows << ',' << kRealColumns << "] T=" << tokens
                  << " bf16_base_ms=" << bf16_base_ms
                  << " bf16_wave8_ms=" << bf16_wave8_ms
                  << " w8_base_ms=" << w8_base_ms
                  << " w8_wave8_ms=" << w8_wave8_ms
                  << " best_tflops=" << operations /
                         (std::min({bf16_base_ms, bf16_wave8_ms, w8_base_ms, w8_wave8_ms}) * 1.0e9)
                  << '\n';
    }
}

void run_prefill_timings(int iterations) {
    constexpr std::uint32_t maximum_tokens = 128;
    const auto input = make_input(maximum_tokens, kRealColumns);
    const auto weights = make_weights(kRealRows, kRealColumns, false);
    const W8Payload w8 = encode_w8g32_independent(weights, kRealRows, kRealColumns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_weights(weights.size());
    DeviceBuffer<std::uint8_t> d_w8(w8.bytes.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(maximum_tokens) * kRealRows);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights.get(), weights.data(), weights.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w8.get(), w8.bytes.data(), w8.bytes.size(), hipMemcpyHostToDevice));
    for (const std::uint32_t tokens : kPrefillWidths) {
        const Bf16LinearArgs bf16{d_input.get(), d_weights.get(), d_output.get(), tokens, kRealRows, kRealColumns};
        const W8G32LinearArgs w8_launch = w8_args(
            d_input.get(), d_w8.get(), d_output.get(), w8, tokens, kRealRows, kRealColumns);
        const double bf16_base_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::bf16_linear_baseline(bf16, nullptr));
        });
        const double bf16_wave16_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::bf16_linear_wave16(bf16, nullptr));
        });
        const double bf16_wmma_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::bf16_linear_wmma16(bf16, nullptr));
        });
        const double w8_base_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::w8g32_linear_baseline(w8_launch, nullptr));
        });
        const double w8_wave16_ms = time_ms(iterations, [&] {
            HIP_CHECK(ninfer::ops::r9700::linear::w8g32_linear_wave16(w8_launch, nullptr));
        });
        const double operations = 2.0 * tokens * kRealRows * kRealColumns;
        std::cout << std::fixed << std::setprecision(3)
                  << "timing: prefill shape=[" << kRealRows << ',' << kRealColumns << "] T=" << tokens
                  << " bf16_decode_baseline_ms=" << bf16_base_ms
                  << " bf16_wave16_ms=" << bf16_wave16_ms
                  << " bf16_wmma16_ms=" << bf16_wmma_ms
                  << " w8_decode_baseline_ms=" << w8_base_ms
                  << " w8_wave16_ms=" << w8_wave16_ms
                  << " bf16_wmma_tflops=" << operations / (bf16_wmma_ms * 1.0e9)
                  << " w8_wave16_tflops=" << operations / (w8_wave16_ms * 1.0e9) << '\n';
    }
}

int parse_iterations(int argc, char** argv) {
    int iterations = 3;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--iterations" && index + 1 < argc) {
            const long parsed = std::strtol(argv[++index], nullptr, 10);
            if (parsed <= 0 || parsed > 1000) fail("--iterations must be in [1,1000]");
            iterations = static_cast<int>(parsed);
        } else if (argument == "--help") {
            std::cout << "usage: linear_op_qual [--iterations N]\n";
            std::exit(0);
        } else {
            fail("unknown or incomplete option: " + std::string(argument));
        }
    }
    return iterations;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int iterations = parse_iterations(argc, argv);
        select_r9700();
        std::cout << "toolchain: HIP " << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR << '.'
                  << HIP_VERSION_PATCH << " compile_target=gfx1201\n";
        // Raw API rejections have no state and must be deterministic before valid launch checks.
        if (ninfer::ops::r9700::linear::bf16_linear({}, nullptr) != hipErrorInvalidValue ||
            ninfer::ops::r9700::linear::w8g32_linear({}, nullptr) != hipErrorInvalidValue) {
            fail("linear raw argument rejection failed");
        }
        test_w8_converter_codec_fixture();
        run_correctness_case("synthetic-edges", 3, 5, 33, true);
        for (const std::uint32_t tokens : {9U, 15U, 17U}) {
            run_correctness_case("synthetic-prefill-tail-T" + std::to_string(tokens), tokens,
                                 17, 64, true);
        }
        run_public_tensor_case(3);
        run_public_tensor_case(17);
        run_correctness_case("qwen3.8-full-attention-query-key", 1, kRealRows, kRealColumns, false);
        run_correctness_case("qwen3.8-full-attention-query-key", 8, kRealRows, kRealColumns, false);
        for (const std::uint32_t tokens : kPrefillWidths) {
            run_prefill_correctness_case(tokens, tokens == 16U);
        }
        run_product_head_shape("qwen3.8-output-head", kOutputHeadRows,
                               kOutputHeadValidRows, true);
        run_product_head_shape("qwen3.8-draft-head", kDraftHeadRows,
                               kDraftHeadRows, false);
        run_timings(iterations);
        run_prefill_timings(iterations);
        std::cout << "linear_op_qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "linear_op_qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
