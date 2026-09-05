#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "ops/r9700/linear/r9700_linear.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

namespace linear = ninfer::ops::r9700::linear;

constexpr std::uint32_t kRows = 7168;
constexpr std::uint32_t kColumns = 5120;
constexpr std::uint32_t kQ4Group = 64;
constexpr std::uint32_t kKAlignment = 128;
constexpr std::uint32_t kW8Group = 32;
constexpr std::uint32_t kMaximumTokens = 128;
constexpr std::size_t kTimingTrials = 5;
constexpr std::size_t kTimingRoutes = 6;
constexpr std::array<std::uint32_t, 12> kTokenCases{1, 2, 3, 4, 5, 6, 7, 8,
                                                    16, 32, 64, 128};
constexpr std::uint32_t kActiveLanesPerGroup = 4;

constexpr std::uint32_t active_lane(std::uint32_t group, std::uint32_t slot) {
    return (group + slot * (kQ4Group / kActiveLanesPerGroup)) % kQ4Group;
}

struct DeviceEvidence {
    std::string architecture;
    std::string pci_bus_id;
    std::uint32_t wave_size = 0;
    std::uint64_t vram_bytes = 0;
    int hip_runtime_version = 0;
};

struct OracleMetrics {
    double maximum_error = 0.0;
    double maximum_source_error = 0.0;
    double maximum_source_bound = 0.0;
    std::uint64_t maximum_bf16_steps = 0;
};

struct ShapeEvidence {
    std::uint32_t tokens = 0;
    OracleMetrics a4_oracle{};
    OracleMetrics a8_oracle{};
    double a4_complete_ms = 0.0;
    double a4_wmma_ms = 0.0;
    double a4_vector_complete_ms = 0.0;
    double a8_complete_ms = 0.0;
    double a8_wmma_ms = 0.0;
    double w8_current_ms = 0.0;
    std::array<double, kTimingTrials> a4_complete_samples{};
    std::array<double, kTimingTrials> a4_wmma_samples{};
    std::array<double, kTimingTrials> a4_vector_complete_samples{};
    std::array<double, kTimingTrials> a8_complete_samples{};
    std::array<double, kTimingTrials> a8_wmma_samples{};
    std::array<double, kTimingTrials> w8_current_samples{};
};

struct CheckEvidence {
    bool a4_codec_edges = false;
    bool a8_codec_edges = false;
    bool asynchronous_status_poisoning = false;
    bool real_shapes = false;
};

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
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (character < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(character) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(character);
            }
        }
    }
    return out.str();
}

void write_samples(std::ostream& out,
                   const std::array<double, kTimingTrials>& samples) {
    out << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0U) out << ", ";
        out << samples[index];
    }
    out << ']';
}

void write_evidence(const std::filesystem::path& path, bool passed,
                    const std::string& failure, const DeviceEvidence& device,
                    const linear::A8Q4G64KernelResources& resources,
                    const CheckEvidence& checks,
                    const std::vector<ShapeEvidence>& shapes) {
    if (path.empty()) fail("qualification evidence path is empty");
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(path, std::ios::trunc);
    if (!out) fail("could not open qualification evidence path: " + path.string());
    out << std::setprecision(17)
        << "{\n"
        << "  \"schema\": \"ninfer.r9700.linear_qualification.v1\",\n"
        << "  \"qualification\": \"q4g64_a4_a8_wmma\",\n"
        << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
        << "  \"failure\": \"" << json_escape(failure) << "\",\n"
        << "  \"artifact_profiles\": [\n"
        << "    \"qwen3.8-27b/r9700-q4g64-n16k16-eval\",\n"
        << "    \"qwen3.8-27b/r9700-q4-w8-n16k16-eval\",\n"
        << "    \"qwen3.8-27b/r9700-q4-w8-mse-n16k16-eval\"\n"
        << "  ],\n"
        << "  \"persistent_weight_format\": \"Q4G64_F16S\",\n"
        << "  \"q4_activation_bits\": [4, 8],\n"
        << "  \"q4_activation_bits_qualified\": [4, 8],\n"
        << "  \"q4_activation_group\": 64,\n"
        << "  \"real_shape_k_lane_coverage\": "
           "\"four rotating nonzero lanes per G64; all 64 lane positions covered\",\n"
        << "  \"output_dtype\": \"BF16\",\n"
        << "  \"hardware\": {\n"
        << "    \"architecture\": \"" << json_escape(device.architecture) << "\",\n"
        << "    \"pci_bus_id\": \"" << json_escape(device.pci_bus_id) << "\",\n"
        << "    \"wave_size\": " << device.wave_size << ",\n"
        << "    \"vram_bytes\": " << device.vram_bytes << "\n"
        << "  },\n"
        << "  \"toolchain\": {\n"
        << "    \"compiler\": \"" << json_escape(__VERSION__) << "\",\n"
        << "    \"hip_runtime_version\": " << device.hip_runtime_version << ",\n"
        << "    \"offload_architecture\": \"gfx1201\"\n"
        << "  },\n"
        << "  \"isa\": {\n"
        << "    \"instruction\": \"v_wmma_i32_16x16x32_iu4\",\n"
        << "    \"low_nibble_signedness\": [0, 1, 0],\n"
        << "    \"high_nibble_signedness\": [1, 1, 0],\n"
        << "    \"integer_recombination\": \"low_dot + 16 * high_dot\"\n"
        << "  },\n"
        << "  \"resources\": {\n"
        << "    \"a8_quantize_registers\": " << resources.quantize_registers << ",\n"
        << "    \"a8_quantize_static_shared_bytes\": "
        << resources.quantize_static_shared_bytes << ",\n"
        << "    \"a8_quantize_local_bytes\": " << resources.quantize_local_bytes << ",\n"
        << "    \"a8q4_wmma_registers\": " << resources.wmma_registers << ",\n"
        << "    \"a8q4_wmma_static_shared_bytes\": "
        << resources.wmma_static_shared_bytes << ",\n"
        << "    \"a8q4_wmma_local_bytes\": " << resources.wmma_local_bytes << "\n"
        << "  },\n"
        << "  \"benchmark_methodology\": {\n"
        << "    \"timer\": \"HIP events on the default ordered stream\",\n"
        << "    \"warmups_per_route_sample\": 2,\n"
        << "    \"trial_rounds\": 5,\n"
        << "    \"route_order\": \"rotating start; alternating forward/reverse per round\",\n"
        << "    \"route_order_by_round\": [\n"
        << "      [\"a4_complete\", \"a4_wmma\", \"a4_vector_complete\", "
           "\"a8_complete\", \"a8_wmma\", \"w8_current\"],\n"
        << "      [\"a4_wmma\", \"a4_complete\", \"w8_current\", \"a8_wmma\", "
           "\"a8_complete\", \"a4_vector_complete\"],\n"
        << "      [\"a4_vector_complete\", \"a8_complete\", \"a8_wmma\", "
           "\"w8_current\", \"a4_complete\", \"a4_wmma\"],\n"
        << "      [\"a8_complete\", \"a4_vector_complete\", \"a4_wmma\", "
           "\"a4_complete\", \"w8_current\", \"a8_wmma\"],\n"
        << "      [\"a8_wmma\", \"w8_current\", \"a4_complete\", \"a4_wmma\", "
           "\"a4_vector_complete\", \"a8_complete\"]\n"
        << "    ],\n"
        << "    \"reported_statistic\": \"median of five per-route samples\"\n"
        << "  },\n"
        << "  \"checks\": {\n"
        << "    \"a4_codec_edges_tail_alias_nonfinite\": "
        << (checks.a4_codec_edges ? "true" : "false") << ",\n"
        << "    \"a8_codec_edges_tail_alias_nonfinite\": "
        << (checks.a8_codec_edges ? "true" : "false") << ",\n"
        << "    \"asynchronous_status_to_nan_output\": "
        << (checks.asynchronous_status_poisoning ? "true" : "false") << ",\n"
        << "    \"real_shape_exact_activation_and_fp64_formula\": "
        << (checks.real_shapes ? "true" : "false") << "\n"
        << "  },\n"
        << "  \"shapes\": [\n";
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        const ShapeEvidence& shape = shapes[index];
        out << "    {\n"
            << "      \"tokens\": " << shape.tokens << ",\n"
            << "      \"rows\": " << kRows << ",\n"
            << "      \"columns\": " << kColumns << ",\n"
            << "      \"a4_oracle\": {\"maximum_error\": "
            << shape.a4_oracle.maximum_error << ", \"maximum_bf16_steps\": "
            << shape.a4_oracle.maximum_bf16_steps
            << ", \"maximum_source_error\": " << shape.a4_oracle.maximum_source_error
            << ", \"maximum_source_bound\": " << shape.a4_oracle.maximum_source_bound
            << "},\n"
            << "      \"a8_oracle\": {\"maximum_error\": "
            << shape.a8_oracle.maximum_error << ", \"maximum_bf16_steps\": "
            << shape.a8_oracle.maximum_bf16_steps
            << ", \"maximum_source_error\": " << shape.a8_oracle.maximum_source_error
            << ", \"maximum_source_bound\": " << shape.a8_oracle.maximum_source_bound
            << "},\n"
            << "      \"timing_ms\": {\"a4_complete\": " << shape.a4_complete_ms
            << ", \"a4_wmma\": " << shape.a4_wmma_ms
            << ", \"a4_vector_complete\": " << shape.a4_vector_complete_ms
            << ", \"a8_complete\": " << shape.a8_complete_ms
            << ", \"a8_wmma\": " << shape.a8_wmma_ms
            << ", \"w8_current\": " << shape.w8_current_ms << "},\n"
            << "      \"timing_samples_ms\": {\n"
            << "        \"a4_complete\": ";
        write_samples(out, shape.a4_complete_samples);
        out << ",\n        \"a4_wmma\": ";
        write_samples(out, shape.a4_wmma_samples);
        out << ",\n        \"a4_vector_complete\": ";
        write_samples(out, shape.a4_vector_complete_samples);
        out << ",\n        \"a8_complete\": ";
        write_samples(out, shape.a8_complete_samples);
        out << ",\n        \"a8_wmma\": ";
        write_samples(out, shape.a8_wmma_samples);
        out << ",\n        \"w8_current\": ";
        write_samples(out, shape.w8_current_samples);
        out << "\n      },\n"
            << "      \"ratios\": {\"a8_over_a4\": "
            << shape.a8_complete_ms / shape.a4_complete_ms
            << ", \"a8_over_w8\": " << shape.a8_complete_ms / shape.w8_current_ms
            << ", \"a4_over_w8\": " << shape.a4_complete_ms / shape.w8_current_ms
            << "}\n"
            << "    }" << (index + 1U == shapes.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    out.close();
    if (!out) fail("could not finish qualification evidence: " + path.string());
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count != 0U) HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const { return data_; }
    [[nodiscard]] std::size_t bytes() const { return count_ * sizeof(T); }

private:
    std::size_t count_ = 0;
    T* data_ = nullptr;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    ~Event() { (void)hipEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    [[nodiscard]] hipEvent_t get() const { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

std::string read_line(const std::string& path) {
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

std::string normalize_pci(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (value.size() == 7U && value[2] == ':' && value[5] == '.') value = "0000:" + value;
    return value;
}

DeviceEvidence select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    const char* requested_env = std::getenv("NINFER_R9700_PCI_BUS_ID");
    const std::string requested = requested_env == nullptr ? "" : normalize_pci(requested_env);
    std::vector<int> matches;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        char pci_buffer[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_buffer, sizeof(pci_buffer), ordinal));
        const std::string pci = normalize_pci(pci_buffer);
        const std::string sysfs = "/sys/bus/pci/devices/" + pci;
        if (properties.integrated == 0 && std::string_view(properties.gcnArchName) == "gfx1201" &&
            read_line(sysfs + "/vendor") == "0x1002" &&
            read_line(sysfs + "/device") == "0x7551" &&
            (requested.empty() || requested == pci)) matches.push_back(ordinal);
    }
    if (matches.size() != 1U) fail("expected exactly one discrete 1002:7551 gfx1201 R9700");
    HIP_CHECK(hipSetDevice(matches.front()));
    hipDeviceProp_t selected{};
    HIP_CHECK(hipGetDeviceProperties(&selected, matches.front()));
    char selected_pci_buffer[32]{};
    HIP_CHECK(hipDeviceGetPCIBusId(selected_pci_buffer, sizeof(selected_pci_buffer),
                                   matches.front()));
    if (selected.warpSize != 32) fail("Q4G64 qualifier requires wave32");
    int hip_runtime_version = 0;
    HIP_CHECK(hipRuntimeGetVersion(&hip_runtime_version));
    std::cout << "device: arch=" << selected.gcnArchName << " wave=" << selected.warpSize
              << " vram_mib=" << selected.totalGlobalMem / (1024U * 1024U) << '\n';
    return DeviceEvidence{selected.gcnArchName, normalize_pci(selected_pci_buffer),
                          static_cast<std::uint32_t>(selected.warpSize),
                          static_cast<std::uint64_t>(selected.totalGlobalMem),
                          hip_runtime_version};
}

std::uint16_t half_bits(float value) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}

float half_value(std::uint16_t bits) {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

void set_q4(std::vector<std::uint8_t>& packed, std::size_t index, int code) {
    if (code < -8 || code > 7) fail("host Q4 code out of range");
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

void set_u4(std::vector<std::uint8_t>& packed, std::size_t index, int code) {
    if (code < 0 || code > 15) fail("host U4 code out of range");
    const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
    const std::uint8_t mask = static_cast<std::uint8_t>(0x0fU << shift);
    packed[index >> 1U] = static_cast<std::uint8_t>(
        (packed[index >> 1U] & static_cast<std::uint8_t>(~mask)) |
        (static_cast<unsigned>(code) << shift));
}

int get_u4(const std::vector<std::uint8_t>& packed, std::size_t index) {
    return static_cast<int>((packed[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU);
}

struct HostActivation {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
    std::uint32_t padded_columns = 0;
};

struct HostA8Activation {
    std::vector<std::uint8_t> low_codes;
    std::vector<std::uint8_t> high_codes;
    std::vector<std::uint16_t> scales;
    std::uint32_t padded_columns = 0;
};

HostActivation quantize_host(const std::vector<hip_bfloat16>& input, std::uint32_t tokens,
                             std::uint32_t columns) {
    const std::uint32_t padded = (columns + kKAlignment - 1U) / kKAlignment * kKAlignment;
    const std::uint32_t groups = padded / kQ4Group;
    HostActivation result{std::vector<std::uint8_t>(static_cast<std::size_t>(tokens) * padded / 2U),
                          std::vector<std::uint16_t>(static_cast<std::size_t>(tokens) * groups), padded};
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            float maximum = 0.0F;
            for (std::uint32_t lane = 0; lane < kQ4Group; ++lane) {
                const std::uint32_t column = group * kQ4Group + lane;
                const float value = column < columns
                                        ? static_cast<float>(input[static_cast<std::size_t>(token) * columns + column])
                                        : 0.0F;
                if (!std::isfinite(value)) fail("host quantizer requires finite represented input");
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint16_t scale_bits = maximum == 0.0F ? 0U : half_bits(maximum / 7.0F);
            if (maximum != 0.0F && scale_bits == 0U) scale_bits = 1U;
            const float scale = half_value(scale_bits);
            if (maximum != 0.0F && (!std::isfinite(scale) || scale <= 0.0F)) {
                fail("host Q4 scale is not finite positive");
            }
            result.scales[static_cast<std::size_t>(token) * groups + group] = scale_bits;
            for (std::uint32_t lane = 0; lane < kQ4Group; ++lane) {
                const std::uint32_t column = group * kQ4Group + lane;
                const float value = column < columns
                                        ? static_cast<float>(input[static_cast<std::size_t>(token) * columns + column])
                                        : 0.0F;
                const int code = scale_bits == 0U
                                     ? 0
                                     : std::clamp(static_cast<int>(std::nearbyint(value / scale)), -8, 7);
                set_q4(result.codes, static_cast<std::size_t>(token) * padded + column, code);
            }
        }
    }
    return result;
}

HostA8Activation quantize_a8_host(const std::vector<hip_bfloat16>& input,
                                  std::uint32_t tokens, std::uint32_t columns) {
    const std::uint32_t padded = (columns + kKAlignment - 1U) / kKAlignment * kKAlignment;
    const std::uint32_t groups = padded / kQ4Group;
    const std::size_t plane_bytes = static_cast<std::size_t>(tokens) * padded / 2U;
    HostA8Activation result{std::vector<std::uint8_t>(plane_bytes),
                            std::vector<std::uint8_t>(plane_bytes),
                            std::vector<std::uint16_t>(static_cast<std::size_t>(tokens) * groups),
                            padded};
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            float maximum = 0.0F;
            for (std::uint32_t lane = 0; lane < kQ4Group; ++lane) {
                const std::uint32_t column = group * kQ4Group + lane;
                const float value = column < columns
                                        ? static_cast<float>(input[static_cast<std::size_t>(token) * columns + column])
                                        : 0.0F;
                if (!std::isfinite(value)) fail("host A8 quantizer requires finite represented input");
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint16_t scale_bits = maximum == 0.0F ? 0U : half_bits(maximum / 127.0F);
            if (maximum != 0.0F && scale_bits == 0U) scale_bits = 1U;
            const float scale = half_value(scale_bits);
            if (maximum != 0.0F && (!std::isfinite(scale) || scale <= 0.0F)) {
                fail("host A8 scale is not finite positive");
            }
            result.scales[static_cast<std::size_t>(token) * groups + group] = scale_bits;
            for (std::uint32_t lane = 0; lane < kQ4Group; ++lane) {
                const std::uint32_t column = group * kQ4Group + lane;
                const float value = column < columns
                                        ? static_cast<float>(input[static_cast<std::size_t>(token) * columns + column])
                                        : 0.0F;
                const int code = scale_bits == 0U
                                     ? 0
                                     : std::clamp(static_cast<int>(std::nearbyint(value / scale)),
                                                  -127, 127);
                const int low = code & 0x0f;
                const int high = (code - low) / 16;
                const std::size_t index = static_cast<std::size_t>(token) * padded + column;
                set_u4(result.low_codes, index, low);
                set_q4(result.high_codes, index, high);
                if (low + 16 * high != code) fail("host A8 nibble decomposition is not exact");
            }
        }
    }
    return result;
}

std::vector<hip_bfloat16> make_real_input() {
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(kMaximumTokens) * kColumns,
                                    hip_bfloat16(0.0F));
    std::array<bool, kQ4Group> covered{};
    for (std::uint32_t token = 0; token < kMaximumTokens; ++token) {
        for (std::uint32_t group = 0; group < kColumns / kQ4Group; ++group) {
            const float amplitude = 0.5F + static_cast<float>((token * 7U + group * 3U) % 16U) / 16.0F;
            const std::size_t base = static_cast<std::size_t>(token) * kColumns + group * kQ4Group;
            const std::array<float, kActiveLanesPerGroup> values{
                amplitude, -amplitude, amplitude * 0.5F, -amplitude * 0.25F};
            for (std::uint32_t slot = 0; slot < kActiveLanesPerGroup; ++slot) {
                const std::uint32_t lane = active_lane(group, slot);
                input[base + lane] = hip_bfloat16(values[slot]);
                covered[lane] = true;
            }
        }
    }
    if (!std::all_of(covered.begin(), covered.end(), [](bool value) { return value; })) {
        fail("real-shape rotating fixture does not cover every G64 lane");
    }
    return input;
}

struct HostWeights {
    std::vector<std::uint8_t> q4_codes; // independent logical row-major oracle image
    std::vector<std::uint16_t> q4_scales; // independent logical row-major oracle image
    std::vector<std::uint8_t> q4_codes_n16;
    std::vector<std::uint16_t> q4_scales_n16;
    std::vector<std::int8_t> w8_codes;
    std::vector<std::uint16_t> w8_scales;
};

HostWeights make_real_weights() {
    const std::size_t elements = static_cast<std::size_t>(kRows) * kColumns;
    HostWeights weights{std::vector<std::uint8_t>(elements / 2U),
                        std::vector<std::uint16_t>(static_cast<std::size_t>(kRows) *
                                                   (kColumns / kQ4Group)),
                        std::vector<std::uint8_t>(elements / 2U),
                        std::vector<std::uint16_t>(static_cast<std::size_t>(kRows) *
                                                   (kColumns / kQ4Group)),
                        std::vector<std::int8_t>(elements),
                        std::vector<std::uint16_t>(static_cast<std::size_t>(kRows) *
                                                   (kColumns / kW8Group))};
    constexpr std::size_t q4_groups = kColumns / kQ4Group;
    for (std::uint32_t row = 0; row < kRows; ++row) {
        for (std::uint32_t group = 0; group < kColumns / kQ4Group; ++group) {
            const std::uint16_t scale = half_bits(0.00390625F *
                static_cast<float>(1U + ((row * 5U + group * 3U) % 8U)));
            weights.q4_scales[static_cast<std::size_t>(row) * (kColumns / kQ4Group) + group] = scale;
            const std::size_t stored_scale =
                ((static_cast<std::size_t>(row) >> 4U) * q4_groups + group) * 16U +
                (row & 15U);
            weights.q4_scales_n16[stored_scale] = scale;
            for (std::uint32_t lane = 0; lane < kQ4Group; ++lane) {
                const int code = static_cast<int>((row * 3U + group * 5U + lane * 7U + 1U) & 15U) - 8;
                set_q4(weights.q4_codes, static_cast<std::size_t>(row) * kColumns +
                                             group * kQ4Group + lane, code);
                const std::size_t pair = lane / 16U;
                const std::size_t within = lane % 16U;
                const std::size_t stored_word =
                    (((static_cast<std::size_t>(row) >> 4U) * q4_groups + group) * 4U +
                     pair) *
                        16U +
                    (row & 15U);
                set_q4(weights.q4_codes_n16, stored_word * 16U + within, code);
            }
        }
        for (std::uint32_t group = 0; group < kColumns / kW8Group; ++group) {
            weights.w8_scales[static_cast<std::size_t>(row) * (kColumns / kW8Group) + group] =
                half_bits(0.00048828125F * static_cast<float>(1U + ((row + group) % 8U)));
            for (std::uint32_t lane = 0; lane < kW8Group; ++lane) {
                weights.w8_codes[static_cast<std::size_t>(row) * kColumns + group * kW8Group + lane] =
                    static_cast<std::int8_t>(static_cast<int>((row * 11U + group * 13U + lane * 17U) % 255U) - 127);
            }
        }
    }
    return weights;
}

void test_codec_edges() {
    constexpr std::uint32_t tokens = 2;
    constexpr std::uint32_t columns = 67;
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns,
                                    hip_bfloat16(0.0F));
    const std::array<float, 12> values{-7.0F, -6.5F, -1.5F, -0.5F, 0.5F, 1.5F,
                                       6.5F, 7.0F, -0.0F, 0.0F, 3.5F, -3.5F};
    for (std::size_t index = 0; index < values.size(); ++index) input[index] = hip_bfloat16(values[index]);
    input[columns + 64U] = hip_bfloat16(1.0F);
    input[columns + 65U] = hip_bfloat16(-1.0F);
    input[columns + 66U] = hip_bfloat16(0.5F);
    const HostActivation expected = quantize_host(input, tokens, columns);
    const std::size_t workspace_bytes =
        linear::q4g64_activation_workspace_capacity_bytes(tokens, columns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> storage(workspace_bytes);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    linear::Q4G64ActivationWorkspace workspace{};
    HIP_CHECK(linear::q4g64_bind_activation_workspace(storage.get(), workspace_bytes, tokens,
                                                       columns, &workspace));
    HIP_CHECK(linear::q4g64_quantize_activation({d_input.get(), workspace}, nullptr));
    std::vector<std::uint8_t> actual_codes(expected.codes.size());
    std::vector<std::uint16_t> actual_scales(expected.scales.size());
    std::uint32_t status = ~0U;
    HIP_CHECK(hipMemcpy(actual_codes.data(), workspace.codes, actual_codes.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_scales.data(), workspace.scales,
                        actual_scales.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
    if (status != linear::Q4G64ActivationOk || actual_codes != expected.codes ||
        actual_scales != expected.scales) fail("Q4G64 exact activation codec mismatch");

    if (linear::q4g64_activation_workspace_capacity_bytes(0, columns) != 0U ||
        linear::q4g64_activation_workspace_capacity_bytes(tokens, 0) != 0U ||
        linear::q4g64_bind_activation_workspace(storage.get(), workspace_bytes - 1U, tokens,
                                                 columns, &workspace) != hipErrorInvalidValue) {
        fail("Q4G64 workspace malformed-shape rejection failed");
    }
    linear::Q4G64ActivationWorkspace alias = workspace;
    alias.codes = reinterpret_cast<std::uint8_t*>(d_input.get());
    if (linear::q4g64_quantize_activation({d_input.get(), alias}, nullptr) != hipErrorInvalidValue) {
        fail("Q4G64 activation alias rejection failed");
    }

    input[3] = hip_bfloat16(std::numeric_limits<float>::infinity());
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(linear::q4g64_quantize_activation({d_input.get(), workspace}, nullptr));
    HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
    if ((status & linear::Q4G64ActivationNonfinite) == 0U) fail("nonfinite input status missing");
    // Largest finite represented BF16. Construct the bits directly so host
    // float->BF16 conversion cannot turn the fixture into infinity first.
    input[3].data = 0x7f7fU;
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(linear::q4g64_quantize_activation({d_input.get(), workspace}, nullptr));
    HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
    if ((status & linear::Q4G64ActivationScaleOverflow) == 0U) fail("scale overflow status missing");
    // K=192 is group-complete but not K128-complete. The row-split artifact
    // contract must retain one entirely zero G64 padding group through K=256.
    constexpr std::uint32_t aligned_group_tail_columns = 192;
    std::vector<hip_bfloat16> tail_input(aligned_group_tail_columns, hip_bfloat16(0.0F));
    tail_input[191] = hip_bfloat16(-2.0F);
    const HostActivation tail_expected = quantize_host(tail_input, 1, aligned_group_tail_columns);
    const std::size_t tail_workspace_bytes =
        linear::q4g64_activation_workspace_capacity_bytes(1, aligned_group_tail_columns);
    DeviceBuffer<hip_bfloat16> d_tail_input(tail_input.size());
    DeviceBuffer<std::uint8_t> tail_storage(tail_workspace_bytes);
    HIP_CHECK(hipMemcpy(d_tail_input.get(), tail_input.data(), tail_input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    linear::Q4G64ActivationWorkspace tail_workspace{};
    HIP_CHECK(linear::q4g64_bind_activation_workspace(
        tail_storage.get(), tail_workspace_bytes, 1, aligned_group_tail_columns, &tail_workspace));
    HIP_CHECK(linear::q4g64_quantize_activation({d_tail_input.get(), tail_workspace}, nullptr));
    std::vector<std::uint8_t> tail_codes(tail_workspace.code_bytes);
    std::vector<std::uint16_t> tail_scales(tail_workspace.scale_bytes / 2U);
    HIP_CHECK(hipMemcpy(tail_codes.data(), tail_workspace.codes, tail_codes.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(tail_scales.data(), tail_workspace.scales, tail_workspace.scale_bytes,
                        hipMemcpyDeviceToHost));
    if (tail_workspace.padded_columns != 256U || tail_codes != tail_expected.codes ||
        tail_scales != tail_expected.scales || tail_scales.back() != 0U) {
        fail("Q4G64 K192 to K256 row-padding image mismatch");
    }
    std::cout << "codec: exact finite/RNE/partial-G64/K192-to-K256/status/alias checks PASS\n";
}

void test_a8_codec_edges() {
    constexpr std::uint32_t tokens = 2;
    constexpr std::uint32_t columns = 67;
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns,
                                    hip_bfloat16(0.0F));
    const std::array<float, 12> values{-127.0F, -126.5F, -2.5F, -1.5F, -0.5F, 0.5F,
                                       1.5F, 2.5F, 126.5F, 127.0F, -0.0F, 0.0F};
    for (std::size_t index = 0; index < values.size(); ++index) {
        input[index] = hip_bfloat16(values[index]);
    }
    input[columns + 64U].data = 0x0001U;
    input[columns + 65U].data = 0x8001U;
    const HostA8Activation expected = quantize_a8_host(input, tokens, columns);
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> storage(workspace_bytes);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    linear::A8G64ActivationWorkspace workspace{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(storage.get(), workspace_bytes,
                                                         tokens, columns, &workspace));
    HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));
    std::vector<std::uint8_t> actual_low(expected.low_codes.size());
    std::vector<std::uint8_t> actual_high(expected.high_codes.size());
    std::vector<std::uint16_t> actual_scales(expected.scales.size());
    std::uint32_t status = ~0U;
    HIP_CHECK(hipMemcpy(actual_low.data(), workspace.low_codes, actual_low.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_high.data(), workspace.high_codes, actual_high.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_scales.data(), workspace.scales, workspace.scale_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
    if (status != linear::Q4G64ActivationOk || actual_low != expected.low_codes ||
        actual_high != expected.high_codes || actual_scales != expected.scales) {
        fail("A8G64 exact activation codec mismatch");
    }

    if (linear::a8q4g64_activation_workspace_capacity_bytes(0, columns) != 0U ||
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, 0) != 0U ||
        linear::a8q4g64_bind_activation_workspace(storage.get(), workspace_bytes - 1U,
                                                   tokens, columns, &workspace) !=
            hipErrorInvalidValue) {
        fail("A8G64 workspace malformed-shape rejection failed");
    }
    linear::A8G64ActivationWorkspace alias = workspace;
    alias.high_codes = alias.low_codes;
    if (linear::a8g64_quantize_activation({d_input.get(), alias}, nullptr) !=
        hipErrorInvalidValue) {
        fail("A8G64 internal-plane alias rejection failed");
    }
    alias = workspace;
    alias.low_codes = reinterpret_cast<std::uint8_t*>(d_input.get());
    if (linear::a8g64_quantize_activation({d_input.get(), alias}, nullptr) !=
        hipErrorInvalidValue) {
        fail("A8G64 input/workspace alias rejection failed");
    }

    input[3] = hip_bfloat16(std::numeric_limits<float>::infinity());
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));
    HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
    if ((status & linear::Q4G64ActivationNonfinite) == 0U) {
        fail("A8G64 nonfinite input status missing");
    }
    input[3].data = 0x7f7fU;
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));
    HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
    if ((status & linear::Q4G64ActivationScaleOverflow) == 0U) {
        fail("A8G64 scale overflow status missing");
    }

    constexpr std::uint32_t tail_columns = 192;
    std::vector<hip_bfloat16> tail_input(tail_columns, hip_bfloat16(0.0F));
    tail_input[191] = hip_bfloat16(-2.0F);
    const HostA8Activation tail_expected = quantize_a8_host(tail_input, 1, tail_columns);
    const std::size_t tail_workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(1, tail_columns);
    DeviceBuffer<hip_bfloat16> d_tail_input(tail_input.size());
    DeviceBuffer<std::uint8_t> tail_storage(tail_workspace_bytes);
    HIP_CHECK(hipMemcpy(d_tail_input.get(), tail_input.data(),
                        tail_input.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    linear::A8G64ActivationWorkspace tail_workspace{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
        tail_storage.get(), tail_workspace_bytes, 1, tail_columns, &tail_workspace));
    HIP_CHECK(linear::a8g64_quantize_activation({d_tail_input.get(), tail_workspace}, nullptr));
    std::vector<std::uint8_t> tail_low(tail_workspace.low_code_bytes);
    std::vector<std::uint8_t> tail_high(tail_workspace.high_code_bytes);
    std::vector<std::uint16_t> tail_scales(tail_workspace.scale_bytes / 2U);
    HIP_CHECK(hipMemcpy(tail_low.data(), tail_workspace.low_codes, tail_low.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(tail_high.data(), tail_workspace.high_codes, tail_high.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(tail_scales.data(), tail_workspace.scales,
                        tail_workspace.scale_bytes, hipMemcpyDeviceToHost));
    if (tail_workspace.padded_columns != 256U || tail_low != tail_expected.low_codes ||
        tail_high != tail_expected.high_codes || tail_scales != tail_expected.scales ||
        tail_scales.back() != 0U) {
        fail("A8G64 K192 to K256 row-padding image mismatch");
    }
    std::cout << "A8 codec: exact signed-code/nibble/RNE/FP16-floor/tail/status/alias checks PASS\n";
}

void test_asynchronous_status_poisoning() {
    constexpr std::uint32_t tokens = 1;
    constexpr std::uint32_t rows = 16;
    constexpr std::uint32_t columns = 128;
    constexpr std::uint32_t groups = columns / kQ4Group;
    std::vector<hip_bfloat16> input(columns, hip_bfloat16(0.0F));
    std::vector<std::uint8_t> weight_codes(
        static_cast<std::size_t>(rows) * columns / 2U, 0U);
    std::vector<std::uint16_t> weight_scales(
        static_cast<std::size_t>(rows) * groups, half_bits(1.0F));
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> d_weight_codes(weight_codes.size());
    DeviceBuffer<std::uint16_t> d_weight_scales(weight_scales.size());
    DeviceBuffer<hip_bfloat16> d_output(rows);
    const std::size_t a4_workspace_bytes =
        linear::q4g64_activation_workspace_capacity_bytes(tokens, columns);
    const std::size_t a8_workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    DeviceBuffer<std::uint8_t> d_a4_workspace(a4_workspace_bytes);
    DeviceBuffer<std::uint8_t> d_a8_workspace(a8_workspace_bytes);
    HIP_CHECK(hipMemcpy(d_weight_codes.get(), weight_codes.data(), weight_codes.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_scales.get(), weight_scales.data(),
                        weight_scales.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    linear::Q4G64ActivationWorkspace a4_workspace{};
    linear::A8G64ActivationWorkspace a8_workspace{};
    HIP_CHECK(linear::q4g64_bind_activation_workspace(
        d_a4_workspace.get(), a4_workspace_bytes, tokens, columns, &a4_workspace));
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
        d_a8_workspace.get(), a8_workspace_bytes, tokens, columns, &a8_workspace));
    const linear::Q4G64CandidateArgs a4{
        d_input.get(), d_weight_codes.get(), d_weight_codes.bytes(),
        d_weight_scales.get(), d_weight_scales.bytes(), d_a4_workspace.get(),
        a4_workspace_bytes, d_output.get(), tokens, rows, columns, columns};
    const linear::A8Q4G64CandidateArgs a8{
        d_input.get(), d_weight_codes.get(), d_weight_codes.bytes(),
        d_weight_scales.get(), d_weight_scales.bytes(), d_a8_workspace.get(),
        a8_workspace_bytes, d_output.get(), tokens, rows, columns, columns};

    const auto require_poisoned = [&](const char* profile, const std::uint32_t* status_pointer,
                                      std::uint32_t expected_status) {
        std::uint32_t status = 0U;
        std::vector<hip_bfloat16> output(rows);
        HIP_CHECK(hipMemcpy(&status, status_pointer, sizeof(status), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(output.data(), d_output.get(), output.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        if ((status & expected_status) == 0U ||
            !std::all_of(output.begin(), output.end(), [](hip_bfloat16 value) {
                return value.data == 0x7fc1U;
            })) {
            fail(std::string(profile) + " silently consumed invalid activation codes");
        }
    };
    for (const std::uint16_t source_bits : {std::uint16_t{0x7f80U},
                                             std::uint16_t{0x7f7fU}}) {
        std::fill(input.begin(), input.end(), hip_bfloat16(0.0F));
        input[0].data = source_bits;
        HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                            hipMemcpyHostToDevice));
        const std::uint32_t expected_status =
            source_bits == 0x7f80U ? linear::Q4G64ActivationNonfinite
                                   : linear::Q4G64ActivationScaleOverflow;
        HIP_CHECK(linear::q4g64_linear_candidate(a4, nullptr));
        require_poisoned("A4Q4", a4_workspace.status, expected_status);
        HIP_CHECK(linear::a8q4g64_linear_candidate(a8, nullptr));
        require_poisoned("A8Q4", a8_workspace.status, expected_status);
    }
    std::cout << "status: A4/A8 nonfinite and scale-overflow outputs are exact BF16 NaN PASS\n";
}

linear::Q4G64LinearArgs q4_args(const linear::Q4G64ActivationWorkspace& workspace,
                                const DeviceBuffer<std::uint8_t>& weight_codes,
                                const DeviceBuffer<std::uint16_t>& weight_scales,
                                hip_bfloat16* output, std::uint32_t tokens) {
    return linear::Q4G64LinearArgs{workspace.codes, workspace.code_bytes, workspace.scales,
                                   workspace.scale_bytes, workspace.status,
                                   weight_codes.get(), weight_codes.bytes(),
                                   weight_scales.get(), weight_scales.bytes(), output, tokens, kRows,
                                   kColumns, kColumns};
}

linear::A8Q4G64LinearArgs a8q4_args(
    const linear::A8G64ActivationWorkspace& workspace,
    const DeviceBuffer<std::uint8_t>& weight_codes,
    const DeviceBuffer<std::uint16_t>& weight_scales,
    hip_bfloat16* output, std::uint32_t tokens) {
    return linear::A8Q4G64LinearArgs{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status,
        weight_codes.get(), weight_codes.bytes(), weight_scales.get(),
        weight_scales.bytes(), output, tokens, kRows, kColumns, kColumns};
}

OracleMetrics check_real_oracle(std::uint32_t tokens,
                                const std::vector<hip_bfloat16>& input,
                                const HostActivation& activation,
                                const HostWeights& weights,
                                const std::vector<hip_bfloat16>& actual) {
    const std::uint32_t groups = kColumns / kQ4Group;
    double maximum_error = 0.0;
    double maximum_source_error = 0.0;
    double maximum_source_bound = 0.0;
    std::uint64_t maximum_bf16_steps = 0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t row = 0; row < kRows; ++row) {
            double sum = 0.0;
            double source_sum = 0.0;
            double source_quantization_bound = 0.0;
            for (std::uint32_t group = 0; group < groups; ++group) {
                std::int32_t integer_dot = 0;
                const std::size_t a_base = static_cast<std::size_t>(token) * kColumns + group * kQ4Group;
                const std::size_t w_base = static_cast<std::size_t>(row) * kColumns + group * kQ4Group;
                const double activation_scale = half_value(
                    activation.scales[static_cast<std::size_t>(token) * groups + group]);
                const double weight_scale = half_value(
                    weights.q4_scales[static_cast<std::size_t>(row) * groups + group]);
                for (std::uint32_t slot = 0; slot < kActiveLanesPerGroup; ++slot) {
                    const std::uint32_t lane = active_lane(group, slot);
                    const int activation_code = get_q4(activation.codes, a_base + lane);
                    const int weight_code = get_q4(weights.q4_codes, w_base + lane);
                    integer_dot += activation_code * weight_code;
                    const double represented_input = static_cast<float>(
                        input[static_cast<std::size_t>(token) * kColumns + group * kQ4Group + lane]);
                    const double decoded_weight = static_cast<double>(weight_code) * weight_scale;
                    source_sum = std::fma(represented_input, decoded_weight, source_sum);
                    source_quantization_bound +=
                        std::abs(represented_input - activation_code * activation_scale) *
                        std::abs(decoded_weight);
                }
                const double scale = activation_scale * weight_scale;
                sum = std::fma(static_cast<double>(integer_dot), scale, sum);
            }
            const hip_bfloat16 expected(static_cast<float>(sum));
            const hip_bfloat16 source_expected(static_cast<float>(source_sum));
            const hip_bfloat16 got = actual[static_cast<std::size_t>(token) * kRows + row];
            const double error = std::abs(static_cast<double>(static_cast<float>(got)) -
                                          static_cast<double>(static_cast<float>(expected)));
            maximum_error = std::max(maximum_error, error);
            const double source_error = std::abs(static_cast<double>(static_cast<float>(got)) -
                                                 static_cast<double>(static_cast<float>(source_expected)));
            const double source_rounding_allowance =
                std::max(0.015625, std::abs(static_cast<double>(static_cast<float>(source_expected))) / 64.0);
            const double source_bound = source_quantization_bound + source_rounding_allowance;
            maximum_source_error = std::max(maximum_source_error, source_error);
            maximum_source_bound = std::max(maximum_source_bound, source_bound);
            if (source_error > source_bound) {
                std::ostringstream out;
                out << "Q4G64 represented-BF16 x decoded-Q4 bound exceeded T=" << tokens
                    << " token=" << token << " row=" << row << " error=" << source_error
                    << " quantization_bound=" << source_quantization_bound
                    << " allowance=" << source_rounding_allowance;
                fail(out.str());
            }
            const std::uint64_t steps = got.data > expected.data ? got.data - expected.data
                                                                  : expected.data - got.data;
            maximum_bf16_steps = std::max(maximum_bf16_steps, steps);
            if (steps > 2U) {
                std::ostringstream out;
                out << "Q4G64 full FP64 formula mismatch T=" << tokens << " token=" << token
                    << " row=" << row << " got=" << static_cast<float>(got)
                    << " expected=" << static_cast<float>(expected) << " bf16_steps=" << steps;
                fail(out.str());
            }
        }
    }
    std::cout << "oracle: [" << tokens << ',' << kRows << ',' << kColumns
              << "] all_outputs PASS max_abs=" << maximum_error
              << " max_bf16_steps=" << maximum_bf16_steps
              << " source_formula_max_abs=" << maximum_source_error
              << " source_bound_max=" << maximum_source_bound << '\n';
    return OracleMetrics{maximum_error, maximum_source_error, maximum_source_bound,
                         maximum_bf16_steps};
}

OracleMetrics check_a8_real_oracle(std::uint32_t tokens,
                                   const std::vector<hip_bfloat16>& input,
                                   const HostA8Activation& activation,
                                   const HostWeights& weights,
                                   const std::vector<hip_bfloat16>& actual) {
    const std::uint32_t groups = kColumns / kQ4Group;
    double maximum_error = 0.0;
    double maximum_source_error = 0.0;
    double maximum_source_bound = 0.0;
    std::uint64_t maximum_bf16_steps = 0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t row = 0; row < kRows; ++row) {
            double sum = 0.0;
            double source_sum = 0.0;
            double source_quantization_bound = 0.0;
            for (std::uint32_t group = 0; group < groups; ++group) {
                std::int32_t low_dot = 0;
                std::int32_t high_dot = 0;
                const std::size_t a_base =
                    static_cast<std::size_t>(token) * kColumns + group * kQ4Group;
                const std::size_t w_base =
                    static_cast<std::size_t>(row) * kColumns + group * kQ4Group;
                const double activation_scale = half_value(
                    activation.scales[static_cast<std::size_t>(token) * groups + group]);
                const double weight_scale = half_value(
                    weights.q4_scales[static_cast<std::size_t>(row) * groups + group]);
                for (std::uint32_t slot = 0; slot < kActiveLanesPerGroup; ++slot) {
                    const std::uint32_t lane = active_lane(group, slot);
                    const int low = get_u4(activation.low_codes, a_base + lane);
                    const int high = get_q4(activation.high_codes, a_base + lane);
                    const int weight_code = get_q4(weights.q4_codes, w_base + lane);
                    low_dot += low * weight_code;
                    high_dot += high * weight_code;
                    const int activation_code = low + 16 * high;
                    if (activation_code < -127 || activation_code > 127) {
                        fail("A8 host decode left signed-code domain");
                    }
                    const double represented_input = static_cast<float>(
                        input[static_cast<std::size_t>(token) * kColumns +
                              group * kQ4Group + lane]);
                    const double decoded_weight = static_cast<double>(weight_code) * weight_scale;
                    source_sum = std::fma(represented_input, decoded_weight, source_sum);
                    source_quantization_bound +=
                        std::abs(represented_input - activation_code * activation_scale) *
                        std::abs(decoded_weight);
                }
                const std::int32_t integer_dot = low_dot + 16 * high_dot;
                sum = std::fma(static_cast<double>(integer_dot),
                               activation_scale * weight_scale, sum);
            }
            const hip_bfloat16 expected(static_cast<float>(sum));
            const hip_bfloat16 source_expected(static_cast<float>(source_sum));
            const hip_bfloat16 got = actual[static_cast<std::size_t>(token) * kRows + row];
            const double error = std::abs(static_cast<double>(static_cast<float>(got)) -
                                          static_cast<double>(static_cast<float>(expected)));
            maximum_error = std::max(maximum_error, error);
            const double source_error =
                std::abs(static_cast<double>(static_cast<float>(got)) -
                         static_cast<double>(static_cast<float>(source_expected)));
            const double source_rounding_allowance =
                std::max(0.015625,
                         std::abs(static_cast<double>(static_cast<float>(source_expected))) /
                             64.0);
            const double source_bound = source_quantization_bound + source_rounding_allowance;
            maximum_source_error = std::max(maximum_source_error, source_error);
            maximum_source_bound = std::max(maximum_source_bound, source_bound);
            if (source_error > source_bound) {
                std::ostringstream out;
                out << "A8Q4 represented-BF16 x decoded-Q4 bound exceeded T=" << tokens
                    << " token=" << token << " row=" << row << " error=" << source_error
                    << " quantization_bound=" << source_quantization_bound
                    << " allowance=" << source_rounding_allowance;
                fail(out.str());
            }
            const std::uint64_t steps = got.data > expected.data ? got.data - expected.data
                                                                  : expected.data - got.data;
            maximum_bf16_steps = std::max(maximum_bf16_steps, steps);
            if (steps > 2U) {
                std::ostringstream out;
                out << "A8Q4 full FP64 formula mismatch T=" << tokens << " token=" << token
                    << " row=" << row << " got=" << static_cast<float>(got)
                    << " expected=" << static_cast<float>(expected)
                    << " bf16_steps=" << steps;
                fail(out.str());
            }
        }
    }
    std::cout << "A8 oracle: [" << tokens << ',' << kRows << ',' << kColumns
              << "] all_outputs PASS max_abs=" << maximum_error
              << " max_bf16_steps=" << maximum_bf16_steps
              << " source_formula_max_abs=" << maximum_source_error
              << " source_bound_max=" << maximum_source_bound << '\n';
    return OracleMetrics{maximum_error, maximum_source_error, maximum_source_bound,
                         maximum_bf16_steps};
}

template <typename Launch>
double event_ms(unsigned iterations, Launch&& launch) {
    for (unsigned warmup = 0; warmup < 2U; ++warmup) HIP_CHECK(launch());
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

double sample_median(const std::array<double, kTimingTrials>& samples) {
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    return ordered[ordered.size() / 2U];
}

std::vector<ShapeEvidence> run_real_shapes() {
    std::vector<ShapeEvidence> evidence;
    const std::vector<hip_bfloat16> input = make_real_input();
    const HostActivation activation = quantize_host(input, kMaximumTokens, kColumns);
    const HostA8Activation a8_activation =
        quantize_a8_host(input, kMaximumTokens, kColumns);
    const HostWeights weights = make_real_weights();
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> d_q4_codes(weights.q4_codes_n16.size());
    DeviceBuffer<std::uint16_t> d_q4_scales(weights.q4_scales_n16.size());
    DeviceBuffer<std::int8_t> d_w8_codes(weights.w8_codes.size());
    DeviceBuffer<std::uint16_t> d_w8_scales(weights.w8_scales.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(kMaximumTokens) * kRows);
    const std::size_t max_workspace_bytes =
        linear::q4g64_activation_workspace_capacity_bytes(kMaximumTokens, kColumns);
    DeviceBuffer<std::uint8_t> d_workspace(max_workspace_bytes);
    const std::size_t max_a8_workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(kMaximumTokens, kColumns);
    DeviceBuffer<std::uint8_t> d_a8_workspace(max_a8_workspace_bytes);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q4_codes.get(), weights.q4_codes_n16.data(),
                        weights.q4_codes_n16.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q4_scales.get(), weights.q4_scales_n16.data(),
                        weights.q4_scales_n16.size() * 2U,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w8_codes.get(), weights.w8_codes.data(), weights.w8_codes.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w8_scales.get(), weights.w8_scales.data(), weights.w8_scales.size() * 2U,
                        hipMemcpyHostToDevice));

    for (const std::uint32_t tokens : kTokenCases) {
        const std::size_t workspace_bytes =
            linear::q4g64_activation_workspace_capacity_bytes(tokens, kColumns);
        linear::Q4G64ActivationWorkspace workspace{};
        HIP_CHECK(linear::q4g64_bind_activation_workspace(d_workspace.get(), workspace_bytes,
                                                           tokens, kColumns, &workspace));
        HIP_CHECK(linear::q4g64_quantize_activation({d_input.get(), workspace}, nullptr));
        std::vector<std::uint8_t> actual_codes(workspace.code_bytes);
        std::vector<std::uint16_t> actual_scales(workspace.scale_bytes / 2U);
        std::uint32_t status = ~0U;
        HIP_CHECK(hipMemcpy(actual_codes.data(), workspace.codes, actual_codes.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(actual_scales.data(), workspace.scales, workspace.scale_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&status, workspace.status, sizeof(status), hipMemcpyDeviceToHost));
        const std::size_t code_prefix = static_cast<std::size_t>(tokens) * kColumns / 2U;
        const std::size_t scale_prefix = static_cast<std::size_t>(tokens) * (kColumns / kQ4Group);
        if (status != 0U || !std::equal(actual_codes.begin(), actual_codes.end(), activation.codes.begin()) ||
            actual_codes.size() != code_prefix ||
            !std::equal(actual_scales.begin(), actual_scales.end(), activation.scales.begin()) ||
            actual_scales.size() != scale_prefix) fail("real-shape exact activation image mismatch");

        const std::size_t a8_workspace_bytes =
            linear::a8q4g64_activation_workspace_capacity_bytes(tokens, kColumns);
        linear::A8G64ActivationWorkspace a8_workspace{};
        HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
            d_a8_workspace.get(), a8_workspace_bytes, tokens, kColumns, &a8_workspace));
        HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), a8_workspace}, nullptr));
        std::vector<std::uint8_t> actual_low(a8_workspace.low_code_bytes);
        std::vector<std::uint8_t> actual_high(a8_workspace.high_code_bytes);
        std::vector<std::uint16_t> actual_a8_scales(a8_workspace.scale_bytes / 2U);
        status = ~0U;
        HIP_CHECK(hipMemcpy(actual_low.data(), a8_workspace.low_codes, actual_low.size(),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(actual_high.data(), a8_workspace.high_codes, actual_high.size(),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(actual_a8_scales.data(), a8_workspace.scales,
                            a8_workspace.scale_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&status, a8_workspace.status, sizeof(status), hipMemcpyDeviceToHost));
        if (status != 0U ||
            !std::equal(actual_low.begin(), actual_low.end(), a8_activation.low_codes.begin()) ||
            !std::equal(actual_high.begin(), actual_high.end(),
                        a8_activation.high_codes.begin()) ||
            !std::equal(actual_a8_scales.begin(), actual_a8_scales.end(),
                        a8_activation.scales.begin()) ||
            actual_low.size() != code_prefix || actual_high.size() != code_prefix ||
            actual_a8_scales.size() != scale_prefix) {
            fail("real-shape exact A8 two-plane activation image mismatch");
        }

        const linear::Q4G64LinearArgs q4 = q4_args(workspace, d_q4_codes, d_q4_scales,
                                                    d_output.get(), tokens);
        HIP_CHECK(linear::q4g64_linear_wmma32(q4, nullptr));
        std::vector<hip_bfloat16> actual(static_cast<std::size_t>(tokens) * kRows);
        HIP_CHECK(hipMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        const OracleMetrics a4_oracle =
            check_real_oracle(tokens, input, activation, weights, actual);

        const linear::A8Q4G64LinearArgs a8q4 =
            a8q4_args(a8_workspace, d_q4_codes, d_q4_scales, d_output.get(), tokens);
        HIP_CHECK(linear::a8q4g64_linear_wmma32(a8q4, nullptr));
        HIP_CHECK(hipMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        const OracleMetrics a8_oracle =
            check_a8_real_oracle(tokens, input, a8_activation, weights, actual);

        linear::Q4G64LinearArgs malformed = q4;
        --malformed.weight_scale_bytes;
        if (linear::q4g64_linear_wmma32(malformed, nullptr) != hipErrorInvalidValue) {
            fail("short Q4 weight scale plane accepted");
        }
        malformed = q4;
        malformed.output = reinterpret_cast<hip_bfloat16*>(workspace.codes);
        if (linear::q4g64_linear_wmma32(malformed, nullptr) != hipErrorInvalidValue) {
            fail("Q4 output alias accepted");
        }
        linear::A8Q4G64LinearArgs malformed_a8 = a8q4;
        --malformed_a8.activation_high_code_bytes;
        if (linear::a8q4g64_linear_wmma32(malformed_a8, nullptr) != hipErrorInvalidValue) {
            fail("short A8 high-nibble plane accepted");
        }
        malformed_a8 = a8q4;
        malformed_a8.activation_high_codes = malformed_a8.activation_low_codes;
        if (linear::a8q4g64_linear_wmma32(malformed_a8, nullptr) != hipErrorInvalidValue) {
            fail("aliased A8 nibble planes accepted");
        }

        const linear::Q4G64CandidateArgs candidate{
            d_input.get(), d_q4_codes.get(), d_q4_codes.bytes(), d_q4_scales.get(),
            d_q4_scales.bytes(), d_workspace.get(), workspace_bytes, d_output.get(), tokens,
            kRows, kColumns, kColumns};
        linear::Q4G64CandidateArgs candidate_alias = candidate;
        candidate_alias.output = const_cast<hip_bfloat16*>(candidate.input);
        if (linear::q4g64_linear_candidate(candidate_alias, nullptr) != hipErrorInvalidValue) {
            fail("complete Q4 candidate accepted input/output alias");
        }
        candidate_alias = candidate;
        candidate_alias.activation_workspace = candidate_alias.output;
        if (linear::q4g64_linear_candidate(candidate_alias, nullptr) != hipErrorInvalidValue) {
            fail("complete Q4 candidate accepted workspace/output alias");
        }
        const linear::A8Q4G64CandidateArgs a8_candidate{
            d_input.get(), d_q4_codes.get(), d_q4_codes.bytes(), d_q4_scales.get(),
            d_q4_scales.bytes(), d_a8_workspace.get(), a8_workspace_bytes, d_output.get(),
            tokens, kRows, kColumns, kColumns};
        linear::A8Q4G64CandidateArgs a8_candidate_alias = a8_candidate;
        a8_candidate_alias.output = const_cast<hip_bfloat16*>(a8_candidate.input);
        if (linear::a8q4g64_linear_candidate(a8_candidate_alias, nullptr) !=
            hipErrorInvalidValue) {
            fail("complete A8Q4 candidate accepted input/output alias");
        }
        a8_candidate_alias = a8_candidate;
        a8_candidate_alias.activation_workspace = a8_candidate_alias.output;
        if (linear::a8q4g64_linear_candidate(a8_candidate_alias, nullptr) !=
            hipErrorInvalidValue) {
            fail("complete A8Q4 candidate accepted workspace/output alias");
        }
        const linear::W8G32LinearArgs w8{
            d_input.get(), d_w8_codes.get(), d_w8_codes.bytes(), d_w8_scales.get(),
            d_w8_scales.bytes(), d_output.get(), tokens, kRows, kColumns, kColumns};
        const unsigned q4_iterations = tokens <= 8U ? 20U : std::max(2U, 64U / tokens);
        const unsigned vector_iterations = tokens <= 8U ? 2U : 1U;
        const unsigned w8_iterations = q4_iterations;
        enum TimingRoute : std::size_t {
            A4Complete,
            A4Wmma,
            A4VectorComplete,
            A8Complete,
            A8Wmma,
            W8Current,
        };
        std::array<std::array<double, kTimingTrials>, kTimingRoutes> samples{};
        const auto launch_route = [&](std::size_t route) {
            switch (route) {
            case A4Complete:
                return linear::q4g64_linear_candidate(candidate, nullptr);
            case A4Wmma:
                return linear::q4g64_linear_wmma32(q4, nullptr);
            case A4VectorComplete: {
                const hipError_t quantized =
                    linear::q4g64_quantize_activation({d_input.get(), workspace}, nullptr);
                return quantized == hipSuccess
                           ? linear::q4g64_linear_vector(q4, nullptr)
                           : quantized;
            }
            case A8Complete:
                return linear::a8q4g64_linear_candidate(a8_candidate, nullptr);
            case A8Wmma:
                return linear::a8q4g64_linear_wmma32(a8q4, nullptr);
            case W8Current:
                return linear::w8g32_linear(w8, nullptr);
            default:
                return hipErrorInvalidValue;
            }
        };
        for (std::size_t round = 0; round < kTimingTrials; ++round) {
            for (std::size_t slot = 0; slot < kTimingRoutes; ++slot) {
                const std::size_t route = round % 2U == 0U
                                              ? (round + slot) % kTimingRoutes
                                              : (round + kTimingRoutes - slot) % kTimingRoutes;
                const unsigned iterations = route == A4VectorComplete
                                                ? vector_iterations
                                                : (route == W8Current ? w8_iterations
                                                                      : q4_iterations);
                samples[route][round] = event_ms(iterations, [&] {
                    return launch_route(route);
                });
            }
        }
        const double q4_complete = sample_median(samples[A4Complete]);
        const double q4_wmma = sample_median(samples[A4Wmma]);
        const double q4_vector = sample_median(samples[A4VectorComplete]);
        const double a8q4_complete = sample_median(samples[A8Complete]);
        const double a8q4_wmma = sample_median(samples[A8Wmma]);
        const double w8_selected = sample_median(samples[W8Current]);
        std::cout << std::fixed << std::setprecision(5)
                  << "timing: T=" << tokens << " q4_complete_ms=" << q4_complete
                  << " q4_wmma_only_ms=" << q4_wmma << " q4_vector_complete_ms=" << q4_vector
                  << " a8q4_complete_ms=" << a8q4_complete
                  << " a8q4_wmma_only_ms=" << a8q4_wmma
                  << " w8_current_ms=" << w8_selected
                  << " a8q4_over_a4q4=" << a8q4_complete / q4_complete
                  << " a8q4_over_w8=" << a8q4_complete / w8_selected
                  << " q4_over_w8=" << q4_complete / w8_selected
                  << " vector_over_wmma=" << q4_vector / q4_complete << '\n';
        evidence.push_back(ShapeEvidence{tokens, a4_oracle, a8_oracle, q4_complete,
                                         q4_wmma, q4_vector, a8q4_complete,
                                         a8q4_wmma, w8_selected,
                                         samples[A4Complete], samples[A4Wmma],
                                         samples[A4VectorComplete], samples[A8Complete],
                                         samples[A8Wmma], samples[W8Current]});
    }
    return evidence;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[1]) != "--out-json") {
        std::cerr << "usage: q4g64_linear_qual --out-json PATH\n";
        return 2;
    }
    const std::filesystem::path evidence_path(argv[2]);
    DeviceEvidence device{};
    linear::A8Q4G64KernelResources resources{};
    CheckEvidence checks{};
    std::vector<ShapeEvidence> shapes;
    try {
        device = select_r9700();
        HIP_CHECK(linear::a8q4g64_kernel_resources(&resources));
        test_codec_edges();
        checks.a4_codec_edges = true;
        test_a8_codec_edges();
        checks.a8_codec_edges = true;
        test_asynchronous_status_poisoning();
        checks.asynchronous_status_poisoning = true;
        shapes = run_real_shapes();
        checks.real_shapes = true;
        write_evidence(evidence_path, true, {}, device, resources, checks, shapes);
        std::cout << "Q4G64 linear candidate: PASS (evaluation dispatch qualified; W8 dispatch unchanged)\n";
        return 0;
    } catch (const std::exception& error) {
        try {
            write_evidence(evidence_path, false, error.what(), device, resources, checks, shapes);
        } catch (const std::exception& report_error) {
            std::cerr << "qualification evidence write failed: " << report_error.what() << '\n';
        }
        std::cerr << "Q4G64 linear candidate: FAIL: " << error.what() << '\n';
        return 1;
    }
}
