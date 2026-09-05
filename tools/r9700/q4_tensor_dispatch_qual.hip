#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"

#include <algorithm>
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

namespace {

constexpr std::uint32_t kTokens = 3;
constexpr std::uint32_t kRows = 17;
constexpr std::uint32_t kColumns = 192;
constexpr std::uint32_t kPaddedColumns = 256;
constexpr std::uint32_t kGroup = 64;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream out;
    out << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(out.str());
}

#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char character : value) {
        switch (character) {
        case '\"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (character < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(character) << std::dec;
            } else {
                out << static_cast<char>(character);
            }
        }
    }
    return out.str();
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

std::uint16_t half_bits(float value) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}

float half_value(std::uint16_t bits) {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

void set_q4(std::vector<std::uint8_t>& packed, std::size_t index, int code) {
    const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
    const std::uint8_t mask = static_cast<std::uint8_t>(0x0fU << shift);
    packed[index >> 1U] = static_cast<std::uint8_t>(
        (packed[index >> 1U] & static_cast<std::uint8_t>(~mask)) |
        ((static_cast<unsigned>(code) & 0x0fU) << shift));
}

int get_q4(const std::vector<std::uint8_t>& packed, std::size_t index) {
    const unsigned nibble = (packed[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

struct HostActivation {
    std::vector<int> codes;
    std::vector<std::uint16_t> scales;
};

HostActivation quantize_activation(const std::vector<hip_bfloat16>& input) {
    constexpr int maximum_code =
        ninfer::ops::r9700::linear::kQ4ActivationBits == 8 ? 127 : 7;
    constexpr int minimum_code =
        ninfer::ops::r9700::linear::kQ4ActivationBits == 8 ? -127 : -8;
    HostActivation result{
        std::vector<int>(static_cast<std::size_t>(kTokens) * kPaddedColumns),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kTokens) *
                                   (kPaddedColumns / kGroup))};
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        for (std::uint32_t group = 0; group < kPaddedColumns / kGroup; ++group) {
            float maximum = 0.0F;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const float value = column < kColumns
                                        ? static_cast<float>(input[
                                              static_cast<std::size_t>(token) * kColumns + column])
                                        : 0.0F;
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint16_t scale_bits =
                maximum == 0.0F ? 0U : half_bits(maximum / maximum_code);
            if (maximum != 0.0F && scale_bits == 0U) scale_bits = 1U;
            const float scale = half_value(scale_bits);
            result.scales[static_cast<std::size_t>(token) *
                              (kPaddedColumns / kGroup) +
                          group] = scale_bits;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const float value = column < kColumns
                                        ? static_cast<float>(input[
                                              static_cast<std::size_t>(token) * kColumns + column])
                                        : 0.0F;
                result.codes[static_cast<std::size_t>(token) * kPaddedColumns + column] =
                    scale_bits == 0U
                        ? 0
                        : std::clamp(static_cast<int>(std::nearbyint(value / scale)),
                                     minimum_code, maximum_code);
            }
        }
    }
    return result;
}

struct HostWeights {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
};

HostWeights make_weights() {
    HostWeights result{
        std::vector<std::uint8_t>(static_cast<std::size_t>(kRows) * kPaddedColumns / 2U),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kRows) *
                                   (kPaddedColumns / kGroup))};
    for (std::uint32_t row = 0; row < kRows; ++row) {
        for (std::uint32_t group = 0; group < kPaddedColumns / kGroup; ++group) {
            const bool padding = group * kGroup >= kColumns;
            result.scales[static_cast<std::size_t>(row) * (kPaddedColumns / kGroup) + group] =
                padding ? 0U : half_bits((1.0F + static_cast<float>((row + group) % 4U)) /
                                         128.0F);
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const int code = padding
                                     ? 0
                                     : static_cast<int>((row * 11U + group * 5U +
                                                         lane * 7U) &
                                                        15U) -
                                           8;
                set_q4(result.codes,
                       static_cast<std::size_t>(row) * kPaddedColumns + group * kGroup + lane,
                       code);
            }
        }
    }
    return result;
}

std::vector<hip_bfloat16> make_input() {
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(kTokens) * kColumns);
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        for (std::uint32_t column = 0; column < kColumns; ++column) {
            const int code = static_cast<int>((token * 17U + column * 13U) % 65U) - 32;
            input[static_cast<std::size_t>(token) * kColumns + column] =
                hip_bfloat16(static_cast<float>(code) / 32.0F);
        }
    }
    return input;
}

std::vector<hip_bfloat16> oracle(const HostActivation& activation,
                                 const HostWeights& weights) {
    std::vector<hip_bfloat16> output(static_cast<std::size_t>(kTokens) * kRows);
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        for (std::uint32_t row = 0; row < kRows; ++row) {
            double sum = 0.0;
            for (std::uint32_t group = 0; group < kPaddedColumns / kGroup; ++group) {
                std::int32_t dot = 0;
                for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                    const std::size_t activation_index =
                        static_cast<std::size_t>(token) * kPaddedColumns + group * kGroup + lane;
                    const std::size_t weight_index =
                        static_cast<std::size_t>(row) * kPaddedColumns + group * kGroup + lane;
                    dot += activation.codes[activation_index] * get_q4(weights.codes, weight_index);
                }
                const double activation_scale = half_value(
                    activation.scales[static_cast<std::size_t>(token) *
                                          (kPaddedColumns / kGroup) +
                                      group]);
                const double weight_scale = half_value(
                    weights.scales[static_cast<std::size_t>(row) *
                                       (kPaddedColumns / kGroup) +
                                   group]);
                sum = std::fma(static_cast<double>(dot), activation_scale * weight_scale, sum);
            }
            output[static_cast<std::size_t>(token) * kRows + row] =
                hip_bfloat16(static_cast<float>(sum));
        }
    }
    return output;
}

void write_json(const std::filesystem::path& path, bool passed, const std::string& failure,
                const hipDeviceProp_t& device, std::size_t workspace_bytes,
                bool workspace_rewound, std::uint64_t maximum_bf16_steps,
                bool status_poisoned) {
    const auto parent = path.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(path, std::ios::trunc);
    if (!out) fail("cannot open Tensor-dispatch evidence path");
    out << "{\n"
        << "  \"schema\": \"ninfer.r9700.q4_tensor_dispatch_qualification.v1\",\n"
        << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
        << "  \"failure\": \"" << json_escape(failure) << "\",\n"
        << "  \"artifact_profiles\": [\"qwen3.8-27b/r9700-q4g64-n16k16-eval\", "
           "\"qwen3.8-27b/r9700-q4-w8-n16k16-eval\", "
           "\"qwen3.8-27b/r9700-q4-w8-mse-n16k16-eval\"],\n"
        << "  \"q4_activation_bits\": "
        << ninfer::ops::r9700::linear::kQ4ActivationBits << ",\n"
        << "  \"shape\": {\"tokens\": " << kTokens << ", \"rows\": " << kRows
        << ", \"columns\": " << kColumns << ", \"padded_columns\": "
        << kPaddedColumns << "},\n"
        << "  \"workspace_bytes\": " << workspace_bytes << ",\n"
        << "  \"workspace_rewound_after_dispatch\": "
        << (workspace_rewound ? "true" : "false") << ",\n"
        << "  \"maximum_bf16_steps\": " << maximum_bf16_steps << ",\n"
        << "  \"asynchronous_status_nan\": " << (status_poisoned ? "true" : "false")
        << ",\n"
        << "  \"hardware\": {\"architecture\": \"" << json_escape(device.gcnArchName)
        << "\", \"wave_size\": " << device.warpSize << "},\n"
        << "  \"toolchain\": {\"compiler\": \"" << json_escape(__VERSION__)
        << "\", \"offload_architecture\": \"gfx1201\"}\n"
        << "}\n";
    out.close();
    if (!out) fail("cannot finish Tensor-dispatch evidence path");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[1]) != "--out-json") {
        std::cerr << "usage: q4_tensor_dispatch_qual --out-json PATH\n";
        return 2;
    }
    const std::filesystem::path path(argv[2]);
    hipDeviceProp_t device{};
    std::size_t workspace_bytes = 0;
    bool workspace_rewound = false;
    std::uint64_t maximum_bf16_steps = 0;
    bool status_poisoned = false;
    try {
        device = select_r9700();
        const std::vector<hip_bfloat16> input = make_input();
        const HostWeights weights = make_weights();
        const HostActivation activation = quantize_activation(input);
        const std::vector<hip_bfloat16> expected = oracle(activation, weights);
        ninfer::DeviceBuffer d_input(input.size() * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer d_weight_codes(weights.codes.size());
        ninfer::DeviceBuffer d_weight_scales(weights.scales.size() * sizeof(std::uint16_t));
        ninfer::DeviceBuffer d_output(expected.size() * sizeof(hip_bfloat16));
        d_input.copy_from_host(input.data(), d_input.size());
        d_weight_codes.copy_from_host(weights.codes.data(), d_weight_codes.size());
        d_weight_scales.copy_from_host(weights.scales.data(), d_weight_scales.size());

        ninfer::Tensor x(d_input.data(), ninfer::DType::BF16,
                         {static_cast<std::int32_t>(kColumns),
                          static_cast<std::int32_t>(kTokens)});
        ninfer::Tensor output(d_output.data(), ninfer::DType::BF16,
                              {static_cast<std::int32_t>(kRows),
                               static_cast<std::int32_t>(kTokens)});
        ninfer::Weight weight{};
        weight.payload = d_weight_codes.data();
        weight.payload_bytes = weights.codes.size() +
                               weights.scales.size() * sizeof(std::uint16_t);
        weight.qdata = d_weight_codes.data();
        weight.qdata_bytes = weights.codes.size();
        weight.scales = d_weight_scales.data();
        weight.scale_bytes = weights.scales.size() * sizeof(std::uint16_t);
        weight.qtype = ninfer::QType::Q4G64_F16S;
        weight.group_size = weight.group = kGroup;
        weight.layout = ninfer::QuantLayout::RowSplit;
        weight.scale_dtype = ninfer::DType::FP16;
        weight.ndim = 2;
        weight.n = weight.shape[0] = weight.padded_shape[0] = kRows;
        weight.k = weight.shape[1] = kColumns;
        weight.padded_shape[1] = kPaddedColumns;

        workspace_bytes = ninfer::ops::linear_workspace_capacity_bytes(
            ninfer::QType::Q4G64_F16S, kTokens, kColumns);
        const std::size_t expected_workspace =
            ninfer::ops::r9700::linear::kQ4ActivationBits == 8
                ? ninfer::ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
                      kTokens, kColumns)
                : ninfer::ops::r9700::linear::q4g64_activation_workspace_capacity_bytes(
                      kTokens, kColumns);
        if (workspace_bytes != expected_workspace) fail("compile-selected workspace mismatch");
        ninfer::WorkspaceArena workspace(workspace_bytes);
        ninfer::ops::linear(x, weight, output, workspace, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        if (workspace.used() != 0U || workspace.peak_used() != workspace_bytes) {
            fail("Tensor Linear did not allocate and rewind the selected workspace exactly");
        }
        workspace_rewound = true;
        std::vector<hip_bfloat16> actual(expected.size());
        d_output.copy_to_host(actual.data(), d_output.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const std::uint64_t steps = actual[index].data > expected[index].data
                                            ? actual[index].data - expected[index].data
                                            : expected[index].data - actual[index].data;
            maximum_bf16_steps = std::max(maximum_bf16_steps, steps);
            if (steps > 2U) fail("compile-selected Tensor output differs from FP64 oracle");
        }

        std::vector<hip_bfloat16> invalid = input;
        invalid[0].data = 0x7f80U;
        d_input.copy_from_host(invalid.data(), d_input.size());
        ninfer::ops::linear(x, weight, output, workspace, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        d_output.copy_to_host(actual.data(), d_output.size());
        status_poisoned = std::all_of(actual.begin(), actual.end(), [](hip_bfloat16 value) {
            return value.data == 0x7fc1U;
        });
        if (!status_poisoned || workspace.used() != 0U) {
            fail("compile-selected Tensor path silently consumed nonfinite activation");
        }
        write_json(path, true, {}, device, workspace_bytes, workspace_rewound,
                   maximum_bf16_steps,
                   status_poisoned);
        std::cout << "Q4 Tensor/WorkspaceArena A"
                  << ninfer::ops::r9700::linear::kQ4ActivationBits
                  << ": PASS workspace_bytes=" << workspace_bytes
                  << " max_bf16_steps=" << maximum_bf16_steps << '\n';
        return 0;
    } catch (const std::exception& error) {
        try {
            write_json(path, false, error.what(), device, workspace_bytes, workspace_rewound,
                       maximum_bf16_steps, status_poisoned);
        } catch (...) {
        }
        std::cerr << "Q4 Tensor/WorkspaceArena qualifier: FAIL: " << error.what() << '\n';
        return 1;
    }
}
