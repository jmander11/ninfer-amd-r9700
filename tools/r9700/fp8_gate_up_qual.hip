#include "ops/r9700/linear/linear_execution.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ops = ninfer::ops;
namespace linear = ninfer::ops::r9700::linear;

#ifndef NINFER_QUAL_SCHEMA
#define NINFER_QUAL_SCHEMA "ninfer.r9700.fp8_gate_up_qualification.v1"
#endif
#ifndef NINFER_QUAL_ARTIFACT_TYPE
#define NINFER_QUAL_ARTIFACT_TYPE "ninfer_r9700_fp8_gate_up_comparison"
#endif
#ifndef NINFER_QUAL_SCHEMA_VERSION
#define NINFER_QUAL_SCHEMA_VERSION 2
#endif
#ifndef NINFER_QUAL_HAS_ID
#define NINFER_QUAL_HAS_ID 0
#endif
#ifndef NINFER_QUALIFICATION_ID
#define NINFER_QUALIFICATION_ID ""
#endif
#ifndef NINFER_ORACLE_MIDDLE_ROW
#define NINFER_ORACLE_MIDDLE_ROW 12345
#endif
#ifndef NINFER_ORACLE_LAST_ROW
#define NINFER_ORACLE_LAST_ROW 34815
#endif
#ifndef NINFER_FP8_FIXED_PROJECTION_ROWS
#define NINFER_FP8_FIXED_PROJECTION_ROWS 34816
#endif

constexpr std::string_view kRequiredDeviceName = "AMD Radeon AI PRO R9700";
constexpr std::string_view kRequiredArchitecture = "gfx1201";
constexpr std::string_view kRequiredVendor = "0x1002";
constexpr std::string_view kRequiredDevice = "0x7551";
constexpr std::array<std::size_t, 9> kOracleTokens{0U, 0U, 0U, 17U, 17U, 17U,
                                                   2047U, 2047U, 2047U};
constexpr std::array<std::size_t, 9> kOracleRows{
    0U, NINFER_ORACLE_MIDDLE_ROW, NINFER_ORACLE_LAST_ROW,
    0U, NINFER_ORACLE_MIDDLE_ROW, NINFER_ORACLE_LAST_ROW,
    0U, NINFER_ORACLE_MIDDLE_ROW, NINFER_ORACLE_LAST_ROW};

struct DeviceIdentity {
    hipDeviceProp_t properties{};
    std::string pci_bus_id;
    std::string power_path;
    int runtime_version = 0;
    int driver_version = 0;
};

void hip_check(hipError_t status, const char* expression) {
    if (status == hipSuccess) return;
    throw std::runtime_error(std::string(expression) + ": " + hipGetErrorString(status));
}

#define HIP_CHECK(expression) hip_check((expression), #expression)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) (void)hipFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return elements_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t elements_ = 0;
};

class Stream {
public:
    Stream() { HIP_CHECK(hipStreamCreate(&stream_)); }
    ~Stream() {
        if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
    }
    [[nodiscard]] hipStream_t get() const noexcept { return stream_; }

private:
    hipStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    ~Event() {
        if (event_ != nullptr) (void)hipEventDestroy(event_);
    }
    [[nodiscard]] hipEvent_t get() const noexcept { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string trim(std::string value) {
    const std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1U);
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) return {};
    std::string value;
    std::getline(input, value);
    return trim(value);
}

std::string normalize_pci(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (value.size() == 7U && value[2] == ':' && value[5] == '.') value = "0000:" + value;
    return value;
}

DeviceIdentity require_device_zero_auto() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    require(count > 0, "device 0 is unavailable");
    HIP_CHECK(hipSetDevice(0));
    int current = -1;
    HIP_CHECK(hipGetDevice(&current));
    require(current == 0, "qualifier did not bind HIP device 0");
    DeviceIdentity identity;
    HIP_CHECK(hipGetDeviceProperties(&identity.properties, 0));
    char pci[32]{};
    HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
    identity.pci_bus_id = normalize_pci(pci);
    const std::string sysfs = "/sys/bus/pci/devices/" + identity.pci_bus_id;
    identity.power_path = sysfs + "/power_dpm_force_performance_level";
    require(std::string_view(identity.properties.name) == kRequiredDeviceName &&
                std::string_view(identity.properties.gcnArchName) == kRequiredArchitecture &&
                identity.properties.integrated == 0 && identity.properties.warpSize == 32 &&
                read_text(sysfs + "/vendor") == kRequiredVendor &&
                read_text(sysfs + "/device") == kRequiredDevice,
            "device 0 is not the discrete 1002:7551 Radeon AI PRO R9700 gfx1201 wave32 target");
    require(read_text(identity.power_path) == "auto",
            "FP8/Q4 matched timing requires device-0 DPM profile auto before allocation");
    HIP_CHECK(hipRuntimeGetVersion(&identity.runtime_version));
    HIP_CHECK(hipDriverGetVersion(&identity.driver_version));
    return identity;
}

std::string json_string(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const char character : value) {
        if (character == '"' || character == '\\') output << '\\' << character;
        else if (character == '\n') output << "\\n";
        else output << character;
    }
    output << '"';
    return output.str();
}

std::string algorithm_hex(const std::array<std::uint8_t, 16>& fingerprint) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : fingerprint) output << std::setw(2) << unsigned{byte};
    return output.str();
}

float decode_e4m3(std::uint8_t bits) {
    const unsigned magnitude = bits & 0x7fU;
    const unsigned exponent = magnitude >> 3U;
    const unsigned mantissa = magnitude & 7U;
    require(!(exponent == 15U && mantissa == 7U), "oracle encountered E4M3 NaN");
    float value = exponent == 0U
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                     static_cast<int>(exponent) - 7);
    return (bits & 0x80U) != 0U ? -value : value;
}

int decode_i4(const std::uint8_t* packed, std::size_t element) {
    const unsigned code = (packed[element / 2U] >> ((element & 1U) * 4U)) & 15U;
    return code >= 8U ? static_cast<int>(code) - 16 : static_cast<int>(code);
}

int decode_u4(const std::uint8_t* packed, std::size_t element) {
    return (packed[element / 2U] >> ((element & 1U) * 4U)) & 15U;
}

std::string executable_path() {
    std::array<char, 4096> path{};
    const ssize_t size = ::readlink("/proc/self/exe", path.data(), path.size() - 1U);
    require(size > 0 && static_cast<std::size_t>(size) < path.size(),
            "cannot resolve qualifier executable");
    return std::string(path.data(), static_cast<std::size_t>(size));
}

std::string executable_sha256(const std::string& path) {
    const std::string line = read_text(path + ".sha256");
    require(line.size() >= 65U && line[64] == ' ', "missing qualifier SHA-256 sidecar");
    const std::string digest = line.substr(0, 64);
    require(std::all_of(digest.begin(), digest.end(), [](unsigned char character) {
        return std::isdigit(character) || (character >= 'a' && character <= 'f');
    }), "invalid qualifier SHA-256 sidecar");
    return digest;
}

void publish_exclusive(const std::string& path, const std::string& payload) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) throw std::runtime_error("cannot exclusively create report: " + path);
    std::size_t offset = 0;
    while (offset < payload.size()) {
        const ssize_t written = ::write(descriptor, payload.data() + offset, payload.size() - offset);
        if (written <= 0) {
            ::close(descriptor);
            throw std::runtime_error("cannot write retained report: " + path);
        }
        offset += static_cast<std::size_t>(written);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0)
        throw std::runtime_error("cannot finalize retained report: " + path);
}

void require_launch(ops::LinearExecution::LaunchStatus status) {
    if (status.ok()) return;
    std::ostringstream message;
    message << "gate_up launch failed: HIP=" << static_cast<int>(status.hip)
            << ", hipBLASLt=" << static_cast<int>(status.hipblaslt);
    throw std::runtime_error(message.str());
}

} // namespace

int main(int argc, char** argv) try {
    require(argc == 3 && std::string_view(argv[1]) == "--output",
            "usage: ninfer_r9700_fp8_gate_up_qual --output NEW_REPORT.json");
    const std::string report_path = argv[2];
    require(!report_path.empty(), "retained report path is empty");
    constexpr std::size_t tokens  = 2048U;
    constexpr std::size_t rows    = NINFER_FP8_FIXED_PROJECTION_ROWS;
    constexpr std::size_t columns = 5120U;
    constexpr std::size_t code_bytes = rows * columns;
    constexpr std::size_t scale_bytes = rows * sizeof(float);
    constexpr std::size_t q4_code_bytes = rows * columns / 2U;
    constexpr std::size_t q4_scale_bytes = rows * (columns / 64U) * sizeof(std::uint16_t);
    constexpr std::size_t output_elements = tokens * rows;

    const DeviceIdentity device = require_device_zero_auto();

    DeviceBuffer<std::byte> weight_storage(code_bytes + scale_bytes);
    const std::size_t activation_bytes =
        ops::LinearExecution::activation_workspace_capacity_bytes(tokens, columns);
    require(activation_bytes != 0U, "fixed FP8 activation workspace geometry failed");
    DeviceBuffer<std::byte> activation_storage(activation_bytes);
    DeviceBuffer<std::byte> matmul_workspace(
        ops::LinearExecution::kMaximumMatmulWorkspaceBytes);
    DeviceBuffer<std::uint8_t> q4_weight_codes(q4_code_bytes);
    DeviceBuffer<std::uint16_t> q4_weight_scales(q4_scale_bytes / sizeof(std::uint16_t));
    const std::size_t q4_activation_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    require(q4_activation_bytes != 0U, "fixed A8Q4 activation workspace geometry failed");
    DeviceBuffer<std::byte> q4_activation_storage(q4_activation_bytes);
    DeviceBuffer<hip_bfloat16> input(tokens * columns);
    DeviceBuffer<hip_bfloat16> fp8_output(output_elements);
    DeviceBuffer<hip_bfloat16> q4_output(output_elements);
    HIP_CHECK(hipMemset(weight_storage.get(), 0, weight_storage.bytes()));
    HIP_CHECK(hipMemset(fp8_output.get(), 0, fp8_output.bytes()));
    HIP_CHECK(hipMemset(q4_output.get(), 0, q4_output.bytes()));
    constexpr std::array<std::uint8_t, 4> fp8_codes{0x38U, 0x40U, 0x30U, 0xB8U};
    std::vector<std::uint8_t> fp8_weight_codes(code_bytes);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t k = 0; k < columns; ++k)
            fp8_weight_codes[row * columns + k] = fp8_codes[((k / 16U) + row) & 3U];
    HIP_CHECK(hipMemcpy(weight_storage.get(), fp8_weight_codes.data(), code_bytes,
                        hipMemcpyHostToDevice));
    std::vector<float> weight_scales(rows);
    for (std::size_t row = 0; row < rows; ++row)
        weight_scales[row] = std::ldexp(1.0F, static_cast<int>(row & 3U) - 1);
    HIP_CHECK(hipMemcpy(weight_storage.get() + code_bytes, weight_scales.data(), scale_bytes,
                        hipMemcpyHostToDevice));
    std::vector<std::uint8_t> q4_codes(q4_code_bytes);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t k = 0; k < columns; k += 2U) {
            const int low = static_cast<int>((k / 8U + row) % 15U) - 7;
            const int high = static_cast<int>(((k + 1U) / 8U + row) % 15U) - 7;
            q4_codes[(row * columns + k) / 2U] =
                static_cast<std::uint8_t>((low & 15) | ((high & 15) << 4));
        }
    }
    HIP_CHECK(hipMemcpy(q4_weight_codes.get(), q4_codes.data(), q4_weight_codes.bytes(),
                        hipMemcpyHostToDevice));
    std::vector<std::uint16_t> q4_scales(q4_weight_scales.bytes() / sizeof(std::uint16_t));
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t group = 0; group < columns / 64U; ++group)
            q4_scales[row * (columns / 64U) + group] = std::bit_cast<std::uint16_t>(
                _Float16(std::ldexp(1.0F, static_cast<int>((row + group) % 3U) - 1)));
    HIP_CHECK(hipMemcpy(q4_weight_scales.get(), q4_scales.data(), q4_weight_scales.bytes(),
                        hipMemcpyHostToDevice));

    ninfer::Weight weight{};
    weight.payload          = weight_storage.get();
    weight.payload_bytes    = weight_storage.bytes();
    weight.qtype            = ninfer::QType::F8E4M3_ROW_F32S;
    weight.qdata            = weight_storage.get();
    weight.qdata_bytes      = code_bytes;
    weight.scales           = weight_storage.get() + code_bytes;
    weight.scale_bytes      = scale_bytes;
    weight.n                = static_cast<std::int32_t>(rows);
    weight.k                = static_cast<std::int32_t>(columns);
    weight.layout           = ninfer::QuantLayout::RowScaled;
    weight.scale_dtype      = ninfer::DType::FP32;
    weight.scale_ne[0]      = static_cast<std::int32_t>(rows);
    weight.scale_nb[0]      = sizeof(float);
    weight.ndim             = 2;
    weight.shape[0]         = static_cast<std::int32_t>(rows);
    weight.shape[1]         = static_cast<std::int32_t>(columns);
    weight.padded_shape[0]  = static_cast<std::int32_t>(rows);
    weight.padded_shape[1]  = static_cast<std::int32_t>(columns);

    ninfer::Weight malformed_weight = weight;
    malformed_weight.layout = ninfer::QuantLayout::RowSplit;
    bool malformed_rejected = false;
    try {
        ops::LinearExecution rejected(
            malformed_weight, activation_storage.get(), activation_storage.bytes(),
            matmul_workspace.get(), matmul_workspace.bytes());
    } catch (const std::invalid_argument&) {
        malformed_rejected = true;
    }
    require(malformed_rejected, "non-row-scaled gate_up Weight was accepted");

    bool undersized_rejected = false;
    try {
        ops::LinearExecution undersized(
            weight, activation_storage.get(), activation_storage.bytes() - 1U,
            matmul_workspace.get(), matmul_workspace.bytes());
        (void)undersized.prepare(tokens);
    } catch (const std::invalid_argument&) {
        undersized_rejected = true;
    }
    require(undersized_rejected, "undersized FP8 activation workspace was accepted");

    ops::LinearExecution execution(
        weight, activation_storage.get(), activation_storage.bytes(), matmul_workspace.get(),
        matmul_workspace.bytes());
    require(execution.rows() == rows && execution.columns() == columns,
            "prepared Linear execution did not derive the bound Weight geometry");
    require(!execution.run(tokens, input.get(), fp8_output.get(), nullptr).ok(),
            "unprepared FP8 token width was accepted");
    const ops::LinearExecution::PreparedProfile& fp8_profile = execution.prepare(tokens);
    require(execution.prepared_profile(tokens) == &fp8_profile,
            "prepared FP8 profile lookup is not stable");
    const linear::Fp8ActivationWorkspace* fp8_activation =
        execution.activation_workspace(tokens);
    require(fp8_activation != nullptr, "prepared FP8 activation workspace is unavailable");
    Stream stream;
    constexpr std::array<float, 8> input_pattern{1.0F, -0.5F, 0.25F, -0.125F,
                                                  0.75F, -0.375F, 0.0625F, -1.0F};
    std::vector<hip_bfloat16> represented_input(tokens * columns);
    for (std::size_t token = 0; token < tokens; ++token) {
        const float factor = std::ldexp(1.0F, static_cast<int>(token % 3U) - 1);
        for (std::size_t k = 0; k < columns; ++k)
            represented_input[token * columns + k] =
                hip_bfloat16(factor * input_pattern[(k + token) & 7U]);
    }
    HIP_CHECK(hipMemcpyAsync(input.get(), represented_input.data(), input.bytes(),
                             hipMemcpyHostToDevice, stream.get()));
    linear::A8G64ActivationWorkspace q4_activation{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
        q4_activation_storage.get(), q4_activation_storage.bytes(), tokens, columns,
        &q4_activation));
    const linear::A8Q4G64CandidateArgs q4_candidate{
        input.get(), q4_weight_codes.get(), q4_weight_codes.bytes(), q4_weight_scales.get(),
        q4_weight_scales.bytes(), q4_activation_storage.get(), q4_activation_storage.bytes(),
        q4_output.get(), static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(columns), static_cast<std::uint32_t>(columns)};

    HIP_CHECK(hipMemsetAsync(q4_output.get(), 0xA5, q4_output.bytes(), stream.get()));
    linear::A8Q4G64CandidateArgs rejected_q4 = q4_candidate;
    --rejected_q4.weight_scale_bytes;
    require(linear::a8q4g64_linear_candidate(rejected_q4, stream.get()) == hipErrorInvalidValue,
            "malformed current-production A8Q4 control was accepted");
    std::array<std::uint16_t, 2> no_clobber{};
    HIP_CHECK(hipMemcpyAsync(&no_clobber[0], q4_output.get(), sizeof(no_clobber[0]),
                             hipMemcpyDeviceToHost, stream.get()));
    HIP_CHECK(hipMemcpyAsync(&no_clobber[1], q4_output.get() + output_elements - 1U,
                             sizeof(no_clobber[1]), hipMemcpyDeviceToHost, stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    require(no_clobber[0] == 0xA5A5U && no_clobber[1] == 0xA5A5U,
            "rejected A8Q4 control clobbered its output");
    require(!execution.run(tokens, input.get(), input.get(), stream.get()).ok(),
            "aliased FP8 input/output was accepted");
    std::uint16_t input_after_rejection = 0U;
    HIP_CHECK(hipMemcpy(&input_after_rejection, input.get(), sizeof(input_after_rejection),
                        hipMemcpyDeviceToHost));
    require(input_after_rejection ==
                std::bit_cast<std::uint16_t>(represented_input.front()),
            "rejected FP8 input/output alias clobbered its input");

    require_launch(execution.run(tokens, input.get(), fp8_output.get(), stream.get()));
    HIP_CHECK(linear::a8q4g64_linear_candidate(q4_candidate, stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));

    std::array<std::uint16_t, kOracleTokens.size()> fp8_probes{};
    std::array<std::uint16_t, kOracleTokens.size()> q4_probes{};
    for (std::size_t probe = 0; probe < kOracleTokens.size(); ++probe) {
        const std::size_t element = kOracleTokens[probe] * rows + kOracleRows[probe];
        HIP_CHECK(hipMemcpy(&fp8_probes[probe], fp8_output.get() + element,
                            sizeof(fp8_probes[probe]), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&q4_probes[probe], q4_output.get() + element,
                            sizeof(q4_probes[probe]), hipMemcpyDeviceToHost));
    }
    std::vector<std::uint8_t> fp8_activation_codes(tokens * columns);
    std::vector<float> fp8_activation_scales(tokens);
    HIP_CHECK(hipMemcpy(fp8_activation_codes.data(), fp8_activation->codes,
                        fp8_activation_codes.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(fp8_activation_scales.data(), fp8_activation->scales,
                        fp8_activation_scales.size() * sizeof(float), hipMemcpyDeviceToHost));
    static_assert(sizeof(hip_bfloat16) == sizeof(std::uint16_t));
    std::uint32_t fp8_max_steps = 0;
    const auto bf16_steps = [](std::uint16_t actual, std::uint16_t expected) {
        return actual > expected ? actual - expected : expected - actual;
    };
    for (std::size_t probe = 0; probe < kOracleTokens.size(); ++probe) {
        const std::size_t token = kOracleTokens[probe], row = kOracleRows[probe];
        double dot = 0.0;
        for (std::size_t k = 0; k < columns; ++k)
            dot += static_cast<double>(decode_e4m3(fp8_activation_codes[token * columns + k])) *
                   fp8_activation_scales[token] *
                   decode_e4m3(fp8_weight_codes[row * columns + k]) * weight_scales[row];
        const auto expected = std::bit_cast<std::uint16_t>(hip_bfloat16(static_cast<float>(dot)));
        fp8_max_steps = std::max(fp8_max_steps,
            static_cast<std::uint32_t>(bf16_steps(fp8_probes[probe], expected)));
    }
    require(fp8_max_steps <= 1U,
            "FP8 axis-sensitive probes differ from represented-format FP64 oracle");
    std::uint32_t device_status = ~0U;
    HIP_CHECK(hipMemcpy(&device_status, fp8_activation->status,
                        sizeof(device_status), hipMemcpyDeviceToHost));
    require(device_status == linear::Fp8ActivationOk,
            "finite gate_up input set the nonfinite device status");
    std::uint32_t q4_device_status = ~0U;
    HIP_CHECK(hipMemcpy(&q4_device_status, q4_activation.status, sizeof(q4_device_status),
                        hipMemcpyDeviceToHost));
    require(q4_device_status == linear::Q4G64ActivationOk,
            "finite A8Q4 control input set the device status");

    std::vector<std::uint8_t> q4_activation_low(q4_activation.low_code_bytes);
    std::vector<std::uint8_t> q4_activation_high(q4_activation.high_code_bytes);
    std::vector<std::uint16_t> q4_activation_scales(q4_activation.scale_bytes / 2U);
    HIP_CHECK(hipMemcpy(q4_activation_low.data(), q4_activation.low_codes,
                        q4_activation.low_code_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(q4_activation_high.data(), q4_activation.high_codes,
                        q4_activation.high_code_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(q4_activation_scales.data(), q4_activation.scales,
                        q4_activation.scale_bytes, hipMemcpyDeviceToHost));
    std::uint32_t q4_max_steps = 0;
    for (std::size_t probe = 0; probe < kOracleTokens.size(); ++probe) {
        const std::size_t token = kOracleTokens[probe], row = kOracleRows[probe];
        double dot = 0.0;
        for (std::size_t k = 0; k < columns; ++k) {
            const std::size_t a = token * columns + k, w = row * columns + k;
            const int activation = decode_u4(q4_activation_low.data(), a) +
                                   16 * decode_i4(q4_activation_high.data(), a);
            const int weight_code = decode_i4(q4_codes.data(), w);
            const std::size_t group = k / 64U;
            const float a_scale = static_cast<float>(std::bit_cast<_Float16>(
                q4_activation_scales[token * (columns / 64U) + group]));
            const float w_scale = static_cast<float>(std::bit_cast<_Float16>(
                q4_scales[row * (columns / 64U) + group]));
            dot += static_cast<double>(activation * weight_code) * a_scale * w_scale;
        }
        const auto expected = std::bit_cast<std::uint16_t>(hip_bfloat16(static_cast<float>(dot)));
        const std::uint32_t steps = bf16_steps(q4_probes[probe], expected);
        if (steps > 2U) {
            std::ostringstream detail;
            detail << "A8Q4 represented-format oracle mismatch probe=" << probe
                   << " token=" << token << " row=" << row
                   << " actual=" << static_cast<float>(std::bit_cast<hip_bfloat16>(q4_probes[probe]))
                   << " expected=" << static_cast<float>(std::bit_cast<hip_bfloat16>(expected))
                   << " fp64=" << dot << " bf16_steps=" << steps;
            throw std::runtime_error(detail.str());
        }
        q4_max_steps = std::max(q4_max_steps, steps);
    }
    require(q4_max_steps <= 2U,
            "A8Q4 axis-sensitive probes differ from represented-format FP64 oracle");

    constexpr std::size_t trials = 14;
    std::array<float, trials> fp8_milliseconds{};
    std::array<float, trials> q4_milliseconds{};
    Event begin;
    Event end;
    const auto time_fp8 = [&](std::size_t trial) {
        HIP_CHECK(hipEventRecord(begin.get(), stream.get()));
        require_launch(execution.run(tokens, input.get(), fp8_output.get(), stream.get()));
        HIP_CHECK(hipEventRecord(end.get(), stream.get()));
        HIP_CHECK(hipEventSynchronize(end.get()));
        HIP_CHECK(hipEventElapsedTime(&fp8_milliseconds[trial], begin.get(), end.get()));
    };
    const auto time_q4 = [&](std::size_t trial) {
        HIP_CHECK(hipEventRecord(begin.get(), stream.get()));
        HIP_CHECK(linear::a8q4g64_linear_candidate(q4_candidate, stream.get()));
        HIP_CHECK(hipEventRecord(end.get(), stream.get()));
        HIP_CHECK(hipEventSynchronize(end.get()));
        HIP_CHECK(hipEventElapsedTime(&q4_milliseconds[trial], begin.get(), end.get()));
    };
    for (std::size_t pair = 0; pair < trials / 2U; ++pair) {
        time_fp8(2U * pair); time_q4(2U * pair);
        time_q4(2U * pair + 1U); time_fp8(2U * pair + 1U);
    }
    auto fp8_sorted_milliseconds = fp8_milliseconds;
    auto q4_sorted_milliseconds = q4_milliseconds;
    std::sort(fp8_sorted_milliseconds.begin(), fp8_sorted_milliseconds.end());
    std::sort(q4_sorted_milliseconds.begin(), q4_sorted_milliseconds.end());

    const hip_bfloat16 infinity(std::numeric_limits<float>::infinity());
    HIP_CHECK(hipMemcpyAsync(input.get(), &infinity, sizeof(infinity), hipMemcpyHostToDevice,
                             stream.get()));
    require_launch(execution.run(tokens, input.get(), fp8_output.get(), stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    HIP_CHECK(hipMemcpy(&device_status, fp8_activation->status,
                        sizeof(device_status), hipMemcpyDeviceToHost));
    std::array<std::uint16_t, 2> fp8_endpoints{};
    HIP_CHECK(hipMemcpy(&fp8_endpoints[0], fp8_output.get(), sizeof(fp8_endpoints[0]),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&fp8_endpoints[1], fp8_output.get() + output_elements - 1U,
                        sizeof(fp8_endpoints[1]), hipMemcpyDeviceToHost));
    require(device_status == linear::Fp8ActivationNonfinite && fp8_endpoints[0] == 0x7FC0U &&
                fp8_endpoints[1] == 0x7FC0U,
            "post-matmul device status consumer did not canonically poison nonfinite output");
    HIP_CHECK(linear::a8q4g64_linear_candidate(q4_candidate, stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    HIP_CHECK(hipMemcpy(&q4_device_status, q4_activation.status, sizeof(q4_device_status),
                        hipMemcpyDeviceToHost));
    std::array<std::uint16_t, 2> q4_endpoints{};
    HIP_CHECK(hipMemcpy(&q4_endpoints[0], q4_output.get(), sizeof(q4_endpoints[0]),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&q4_endpoints[1], q4_output.get() + output_elements - 1U,
                        sizeof(q4_endpoints[1]), hipMemcpyDeviceToHost));
    require((q4_device_status & linear::Q4G64ActivationNonfinite) != 0U &&
                q4_endpoints[0] == 0x7FC1U && q4_endpoints[1] == 0x7FC1U,
            "current-production A8Q4 did not publish canonical rejection output");

    const double operations = 2.0 * static_cast<double>(tokens) * rows * columns;
    const double fp8_median_ms = 0.5 * (fp8_sorted_milliseconds[trials / 2U - 1U] +
                                       fp8_sorted_milliseconds[trials / 2U]);
    const double q4_median_ms  = 0.5 * (q4_sorted_milliseconds[trials / 2U - 1U] +
                                       q4_sorted_milliseconds[trials / 2U]);
    const double fp8_over_q4_speedup = q4_median_ms / fp8_median_ms;
    require(read_text(device.power_path) == "auto",
            "device-0 DPM profile changed from auto during FP8/Q4 matched timing");
    const std::string algorithm_fingerprint =
        algorithm_hex(fp8_profile.selected_algorithm_fingerprint);
    const std::string self_path = executable_path();
    std::ostringstream report;
    report << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"schema\": " << json_string(NINFER_QUAL_SCHEMA) << ",\n"
              << "  \"artifact_type\": " << json_string(NINFER_QUAL_ARTIFACT_TYPE) << ",\n"
              << "  \"schema_version\": " << NINFER_QUAL_SCHEMA_VERSION << ",\n";
#if NINFER_QUAL_HAS_ID
    report << "  \"qualification_id\": " << json_string(NINFER_QUALIFICATION_ID) << ",\n";
#endif
    report
              << "  \"provenance\": {\"source_root\": " << json_string(NINFER_SOURCE_ROOT)
              << ", \"source_sha256\": " << json_string(NINFER_QUAL_SOURCE_SHA256)
              << ", \"executable_path\": " << json_string(self_path)
              << ", \"executable_sha256\": " << json_string(executable_sha256(self_path))
              << "},\n"
              << "  \"hardware\": {\"ordinal\": 0, \"device\": "
              << json_string(device.properties.name) << ", \"pci_bus_id\": "
              << json_string(device.pci_bus_id)
              << ", \"vendor_id\": \"0x1002\", \"device_id\": \"0x7551\", "
              << "\"architecture\": \"gfx1201\", \"wave_size\": 32, "
              << "\"compute_units\": " << device.properties.multiProcessorCount
              << ", \"l2_bytes\": " << device.properties.l2CacheSize
              << ", \"vram_bytes\": " << device.properties.totalGlobalMem << "},\n"
              << "  \"power_profile\": {\"path\": " << json_string(device.power_path)
              << ", \"required\": \"auto\", \"before\": \"auto\", "
              << "\"after\": \"auto\"},\n"
              << "  \"toolchain\": {\"hip_runtime_version\": " << device.runtime_version
              << ", \"hip_driver_version\": " << device.driver_version
              << ", \"hipblaslt_version\": " << execution.hipblaslt_version() << "},\n"
              << "  \"shape\": {\"tokens\": " << tokens << ", \"rows\": " << rows
              << ", \"columns\": " << columns << "},\n"
              << "  \"fp8_profile\": \"E4M3-outer-vec32f-hipBLASLt-top-supported\",\n"
              << "  \"q4_control\": \"A8Q4G64-m64n128-pingpong-production\",\n"
              << "  \"selected_workspace_bytes\": "
              << fp8_profile.selected_matmul_workspace_bytes << ",\n"
              << "  \"algorithm\": {\"heuristic_requested\": 8, \"heuristic_returned\": "
              << fp8_profile.heuristic_result_count << ", \"selected_rank\": "
              << fp8_profile.selected_heuristic_rank << ", \"opaque_data_hex\": "
              << json_string(algorithm_fingerprint) << ", \"selected_workspace_bytes\": "
              << fp8_profile.selected_matmul_workspace_bytes
              << ", \"algorithm_max_workspace_bytes\": "
              << fp8_profile.selected_algorithm_max_workspace_bytes
              << ", \"waves_count\": " << fp8_profile.selected_waves_count
              << ", \"compute\": \"HIPBLAS_COMPUTE_32F\", "
              << "\"scale_type\": \"HIP_R_32F\", \"a\": "
              << "\"HIP_R_8F_E4M3-col[K,N]-ldK-opT-outer_vec32f\", \"b\": "
              << "\"HIP_R_8F_E4M3-col[K,T]-ldK-outer_vec32f\", \"d\": "
              << "\"HIP_R_16BF-col[N,T]-ldN\"},\n"
              << "  \"selected_waves_count\": " << fp8_profile.selected_waves_count << ",\n"
              << "  \"fp8_complete_median_ms\": " << fp8_median_ms << ",\n"
              << "  \"q4_complete_median_ms\": " << q4_median_ms << ",\n"
              << "  \"fp8_complete_ms_samples\": [";
    for (std::size_t i = 0; i < trials; ++i) report << (i ? ", " : "") << fp8_milliseconds[i];
    report << "],\n  \"q4_complete_ms_samples\": [";
    for (std::size_t i = 0; i < trials; ++i) report << (i ? ", " : "") << q4_milliseconds[i];
    report << "],\n"
              << "  \"trial_order\": \"7 paired repetitions, each fp8,q4 then q4,fp8\",\n"
              << "  \"fp8_complete_tflops\": "
              << operations / (fp8_median_ms * 1.0e9) << ",\n"
              << "  \"q4_complete_tflops_equivalent\": "
              << operations / (q4_median_ms * 1.0e9) << ",\n"
              << "  \"fp8_over_q4_speedup\": " << fp8_over_q4_speedup << ",\n"
              << "  \"current_q4_implementation_underperforms_fp8\": "
              << (fp8_over_q4_speedup > 1.0 ? "true" : "false") << ",\n"
              << "  \"current_q4_implementation_poor_vs_fp8\": "
              << (fp8_over_q4_speedup > 1.0 ? "true" : "false") << ",\n"
              << "  \"direct_weight_binding\": true,\n"
              << "  \"represented_format_axis_oracle\": {\"probe_count\": 9, "
              << "\"tokens\": [0, 17, 2047], \"rows\": [0, "
              << NINFER_ORACLE_MIDDLE_ROW << ", " << NINFER_ORACLE_LAST_ROW << "], "
              << "\"fp8_max_bf16_steps\": " << fp8_max_steps
              << ", \"q4_max_bf16_steps\": " << q4_max_steps << "},\n"
              << "  \"outer_vector_scales\": true,\n"
              << "  \"nonfinite_status_poisoning\": true,\n"
              << "  \"q4_nonfinite_status_poisoning\": true,\n"
              << "  \"balanced_interleaved_pairs\": 7,\n"
              << "  \"no_clobber_rejection\": true,\n"
              << "  \"pass\": true\n"
              << "}\n";
    publish_exclusive(report_path, report.str());
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "fp8_gate_up_qual: " << error.what() << '\n';
    return EXIT_FAILURE;
}
