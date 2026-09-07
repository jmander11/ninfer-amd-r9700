#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ops/r9700/linear/linear_execution.h"
#include "targets/qwen3_8_27b/impl/load/fp8_hybrid_selection.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace artifact = ninfer::artifact;
namespace hybrid = ninfer::targets::qwen3_8_27b::detail::fp8_hybrid;
namespace linear = ninfer::ops::r9700::linear;

constexpr std::uint32_t kColumns = 5120;
constexpr std::uint32_t kPrefix = 128;
constexpr std::uint32_t kExtended = 129;
constexpr std::array<std::string_view, 2> kNames{
    "text/layers/0/mlp/gate_up",
    "text/layers/0/gdn/query_key",
};
constexpr std::array<std::uint32_t, 2> kRows{34816, 4096};
constexpr std::array<std::uint32_t, 4> kProbeTokens{0, 63, 127, 128};
constexpr std::array<std::uint8_t, 16> kQualifiedCommonAlgorithm{
    0xe2U, 0xe0U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
    0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U};
constexpr std::array<std::string_view, 8> kExpectedGateUpCommonCatalog{
    "dbe00100000000000000000000000000", "dce00100000000000000000000000000",
    "dde00100000000000000000000000000", "dee00100000000000000000000000000",
    "dfe00100000000000000000000000000", "e0e00100000000000000000000000000",
    "e1e00100000000000000000000000000", "e2e00100000000000000000000000000"};
constexpr std::string_view kPciBusId = "0000:13:00.0";
constexpr std::string_view kPowerPath =
    "/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level";

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}
struct Stream {
    hipStream_t value = nullptr;
    Stream() { HIP_CHECK(hipStreamCreate(&value)); }
    ~Stream() { if (value != nullptr) (void)hipStreamDestroy(value); }
};

struct Event {
    hipEvent_t value = nullptr;
    Event() { HIP_CHECK(hipEventCreate(&value)); }
    ~Event() { if (value != nullptr) (void)hipEventDestroy(value); }
};

template <class T>
struct DeviceArray {
    T* value = nullptr;
    std::size_t count = 0;
    explicit DeviceArray(std::size_t elements) : count(elements) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&value), count * sizeof(T)));
    }
    ~DeviceArray() { if (value != nullptr) (void)hipFree(value); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

std::string algorithm_hex(const std::array<std::uint8_t, 16>& value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::uint8_t byte : value) output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

std::string json_string(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (char character : value) {
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character != '\r') output << character;
    }
    output << '"';
    return output.str();
}

std::string shell_quote(std::string_view value) {
    std::string output{"'"};
    for (char character : value) {
        if (character == '\'') output += "'\\''";
        else output += character;
    }
    output += '\'';
    return output;
}

std::string sha256_file(const std::filesystem::path& path) {
    const std::string command = "sha256sum -- " + shell_quote(path.string());
    FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) fail("cannot launch sha256sum");
    std::array<char, 256> line{};
    const bool read = std::fgets(line.data(), static_cast<int>(line.size()), pipe) != nullptr;
    const int status = ::pclose(pipe);
    const std::string digest(line.data(), read ? 64U : 0U);
    require(read && status == 0 && digest.size() == 64U &&
                std::all_of(digest.begin(), digest.end(), [](unsigned char character) {
                    return std::isdigit(character) || (character >= 'a' && character <= 'f');
                }),
            "sha256sum failed or returned an invalid digest");
    return digest;
}

std::string read_trimmed(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(input.good(), "cannot read required system identity: " + path.string());
    std::string value;
    std::getline(input, value);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

void write_new(const std::filesystem::path& path, const std::string& contents) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("refusing non-new report path");
    std::size_t offset = 0;
    while (offset != contents.size()) {
        const ssize_t written = ::write(descriptor, contents.data() + offset,
                                        contents.size() - offset);
        if (written <= 0) {
            const int saved = errno;
            (void)::close(descriptor);
            (void)::unlink(path.c_str());
            fail("report write failed with errno " + std::to_string(saved));
        }
        offset += static_cast<std::size_t>(written);
    }
    const int sync_status = ::fsync(descriptor);
    const int close_status = ::close(descriptor);
    if (sync_status != 0 || close_status != 0) {
        (void)::unlink(path.c_str());
        fail("report finalization failed");
    }
}

float decode_e4m3(std::uint8_t bits) {
    const unsigned magnitude = bits & 0x7fU;
    const unsigned exponent = magnitude >> 3U;
    const unsigned mantissa = magnitude & 7U;
    require(!(exponent == 15U && mantissa == 7U), "oracle encountered E4M3 NaN");
    const float value = exponent == 0U
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                     static_cast<int>(exponent) - 7);
    return (bits & 0x80U) != 0U ? -value : value;
}

std::uint64_t round_ties_even_nonnegative(double value) {
    require(std::isfinite(value) && value >= 0.0,
            "direct BF16 rounding received an invalid magnitude");
    const double lower_value = std::floor(value);
    const auto lower = static_cast<std::uint64_t>(lower_value);
    const double remainder = value - lower_value;
    return lower + (remainder > 0.5 ||
                    (remainder == 0.5 && (lower & 1U) != 0U) ? 1U : 0U);
}

std::uint16_t round_fp64_to_bf16(double value) {
    require(!std::isnan(value), "direct BF16 oracle cannot round NaN");
    const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
    const double magnitude = std::abs(value);
    if (magnitude == 0.0) return sign;
    if (std::isinf(magnitude)) return sign | 0x7f80U;

    int exponent = 0;
    (void)std::frexp(magnitude, &exponent);
    int unbiased = exponent - 1;
    if (unbiased < -126) {
        const std::uint64_t significand =
            round_ties_even_nonnegative(std::ldexp(magnitude, 133));
        return static_cast<std::uint16_t>(sign | std::min<std::uint64_t>(significand, 128U));
    }
    std::uint64_t significand =
        round_ties_even_nonnegative(std::ldexp(magnitude, 7 - unbiased));
    if (significand == 256U) {
        significand = 128U;
        ++unbiased;
    }
    if (unbiased >= 128) return sign | 0x7f80U;
    require(significand >= 128U && significand < 256U,
            "direct BF16 oracle derived an invalid normal significand");
    const auto biased = static_cast<std::uint16_t>(unbiased + 127);
    return static_cast<std::uint16_t>(sign | (biased << 7U) |
                                      static_cast<std::uint16_t>(significand - 128U));
}

void check_direct_bf16_rounding_edges() {
    constexpr double even_midpoint = 1.0 + 0x1p-8;
    constexpr double above_midpoint = even_midpoint + 0x1p-30;
    constexpr double odd_midpoint = 1.0 + 3.0 * 0x1p-8;
    require(round_fp64_to_bf16(even_midpoint) == 0x3f80U &&
                round_fp64_to_bf16(above_midpoint) == 0x3f81U &&
                round_fp64_to_bf16(odd_midpoint) == 0x3f82U &&
                round_fp64_to_bf16(-above_midpoint) == 0xbf81U,
            "direct FP64-to-BF16 ties-even edge regression");
    require(std::bit_cast<std::uint16_t>(hip_bfloat16(static_cast<float>(above_midpoint))) ==
                0x3f80U,
            "double-round discriminator edge no longer distinguishes the old oracle");
}

std::vector<hip_bfloat16> make_input() {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(kExtended) * kColumns);
    for (std::uint32_t token = 0; token < kExtended; ++token) {
        for (std::uint32_t column = 0; column < kColumns; ++column) {
            std::uint32_t bits = 0x9e3779b9U * (token + 1U) ^ 0x85ebca6bU * (column + 17U);
            bits ^= bits >> 16U;
            const int signed_value = static_cast<int>(bits % 4093U) - 2046;
            const float scale = std::ldexp(1.0F, static_cast<int>((token + column) % 7U) - 12);
            result[static_cast<std::size_t>(token) * kColumns + column] =
                hip_bfloat16(static_cast<float>(signed_value) * scale);
        }
        // Fix a finite, unique per-row extremum so the quantization scale is well-conditioned.
        result[static_cast<std::size_t>(token) * kColumns + token % kColumns] =
            hip_bfloat16((token & 1U) == 0U ? 31.5F : -31.5F);
    }
    return result;
}

struct BoundWeight {
    const artifact::TensorDescriptor& descriptor;
    artifact::RowScaledGeometry geometry;
    artifact::PayloadSpan payload;
    DeviceArray<std::byte> device;
    ninfer::Weight weight{};

    BoundWeight(const artifact::Reader& reader, std::string_view name, std::uint32_t rows)
        : descriptor([&]() -> const artifact::TensorDescriptor& {
              const artifact::ObjectDescriptor* object = reader.find(name);
              if (object == nullptr) fail("artifact lacks required FP8 matrix: " + std::string(name));
              const auto* tensor = std::get_if<artifact::TensorDescriptor>(object);
              if (tensor == nullptr) fail("required FP8 matrix is not a tensor");
              return *tensor;
          }()),
          geometry(artifact::row_scaled_geometry(descriptor.format, descriptor.shape)),
          payload(reader.payload(descriptor)),
          device(static_cast<std::size_t>(descriptor.bytes)) {
        require(hybrid::contains(name), "matrix is outside the production FP8 selection");
        require(descriptor.name == name && descriptor.shape == std::vector<std::uint64_t>{rows, kColumns} &&
                    descriptor.format == artifact::NumericFormat::F8E4M3_ROW_F32S &&
                    descriptor.layout == artifact::StorageLayout::RowScaledK128V1 &&
                    descriptor.bytes == geometry.encoded_bytes &&
                    payload.data.size() == descriptor.bytes && geometry.rows == rows &&
                    geometry.columns == kColumns && geometry.padded_columns == kColumns,
                "production FP8 matrix descriptor/geometry differs");
        HIP_CHECK(hipMemcpy(device.value, payload.data.data(), payload.data.size(),
                            hipMemcpyHostToDevice));
        weight.payload = device.value;
        weight.payload_bytes = descriptor.bytes;
        weight.qtype = ninfer::QType::F8E4M3_ROW_F32S;
        weight.qdata = device.value;
        weight.qdata_bytes = geometry.code_plane_bytes;
        weight.scales = device.value + geometry.scale_plane_offset;
        weight.scale_bytes = geometry.scale_plane_bytes;
        weight.n = static_cast<std::int32_t>(rows);
        weight.k = static_cast<std::int32_t>(kColumns);
        weight.layout = ninfer::QuantLayout::RowScaled;
        weight.scale_dtype = ninfer::DType::FP32;
        weight.scale_ne[0] = static_cast<std::int32_t>(rows);
        weight.scale_nb[0] = sizeof(float);
        weight.ndim = 2;
        weight.shape[0] = static_cast<std::int32_t>(rows);
        weight.shape[1] = static_cast<std::int32_t>(kColumns);
        weight.padded_shape[0] = static_cast<std::int32_t>(rows);
        weight.padded_shape[1] = static_cast<std::int32_t>(kColumns);
    }
};

struct Run {
    std::vector<std::uint8_t> codes;
    std::vector<float> scales;
    std::vector<std::uint16_t> output;
    std::uint32_t status = ~0U;
};

Run run(ninfer::ops::LinearExecution& execution, std::uint32_t tokens,
        const DeviceArray<hip_bfloat16>& input, std::uint32_t rows, hipStream_t stream,
        bool use_default = false,
        const std::array<std::uint8_t, 16>* fingerprint = nullptr) {
    DeviceArray<hip_bfloat16> output(static_cast<std::size_t>(tokens) * rows);
    HIP_CHECK(hipMemset(output.value, 0xff, output.count * sizeof(hip_bfloat16)));
    const auto status = fingerprint != nullptr
        ? execution.run_qualified_algorithm(tokens, *fingerprint, input.value, output.value, stream)
        : use_default
            ? execution.run_default_heuristic(tokens, input.value, output.value, stream)
            : execution.run(tokens, input.value, output.value, stream);
    require(status.ok(), "LinearExecution launch failed");
    HIP_CHECK(hipStreamSynchronize(stream));
    const linear::Fp8ActivationWorkspace* activation = execution.activation_workspace(tokens);
    require(activation != nullptr && activation->tokens == tokens &&
                activation->columns == kColumns && activation->padded_columns == kColumns,
            "prepared activation workspace differs");
    Run result;
    result.codes.resize(activation->code_bytes);
    result.scales.resize(tokens);
    result.output.resize(static_cast<std::size_t>(tokens) * rows);
    HIP_CHECK(hipMemcpy(result.codes.data(), activation->codes, result.codes.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(result.scales.data(), activation->scales,
                        result.scales.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&result.status, activation->status, sizeof(result.status),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(result.output.data(), output.value,
                        result.output.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    require(result.status == linear::Fp8ActivationOk, "finite fixture set FP8 status");
    require(std::none_of(result.output.begin(), result.output.end(), [](std::uint16_t bits) {
                return (bits & 0x7f80U) == 0x7f80U;
            }),
            "LinearExecution produced a nonfinite BF16 output");
    return result;
}

std::uint32_t steps(std::uint16_t lhs, std::uint16_t rhs) {
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

struct CellResult {
    std::string name;
    std::uint32_t rows = 0;
    ninfer::ops::LinearExecution::PreparedProfile p128{};
    ninfer::ops::LinearExecution::PreparedProfile p129{};
    std::uint32_t oracle_probes = 0;
    std::uint32_t oracle_max_steps = 0;
    double oracle_max_bound_ratio = 0.0;
    std::uint64_t output_prefix_mismatches = 0;
    std::uint64_t first_output_mismatch = 0;
    std::uint16_t first_p128_bits = 0;
    std::uint16_t first_p129_bits = 0;
    std::uint64_t default_output_prefix_mismatches = 0;
    std::array<float, 14> p128_default_ms{};
    std::array<float, 14> p128_selected_ms{};
    std::array<float, 14> p129_default_ms{};
    std::array<float, 14> p129_selected_ms{};
    struct CommonResult {
        std::array<std::uint8_t, 16> fingerprint{};
        std::uint64_t output_prefix_mismatches = 0;
        std::uint32_t oracle_probes = 0;
        std::uint32_t oracle_max_steps = 0;
        double oracle_max_bound_ratio = 0.0;
        std::array<float, 14> p128_default_ms{};
        std::array<float, 14> p128_common_ms{};
        std::array<float, 14> p129_default_ms{};
        std::array<float, 14> p129_common_ms{};
    };
    std::vector<CommonResult> common_algorithms;
};

void time_pair(ninfer::ops::LinearExecution& execution, std::uint32_t tokens,
               const DeviceArray<hip_bfloat16>& input, DeviceArray<hip_bfloat16>& output,
               hipStream_t stream, std::array<float, 14>& defaults,
               std::array<float, 14>& selected,
               const std::array<std::uint8_t, 16>* fingerprint = nullptr) {
    Event begin;
    Event end;
    const auto launch = [&](bool use_default) {
        const auto status = !use_default && fingerprint != nullptr
            ? execution.run_qualified_algorithm(tokens, *fingerprint, input.value,
                                                output.value, stream)
            : use_default
            ? execution.run_default_heuristic(tokens, input.value, output.value, stream)
            : execution.run(tokens, input.value, output.value, stream);
        require(status.ok(), "timed LinearExecution launch failed");
    };
    launch(true);
    launch(false);
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto sample = [&](bool use_default, float& milliseconds) {
        HIP_CHECK(hipEventRecord(begin.value, stream));
        launch(use_default);
        HIP_CHECK(hipEventRecord(end.value, stream));
        HIP_CHECK(hipEventSynchronize(end.value));
        HIP_CHECK(hipEventElapsedTime(&milliseconds, begin.value, end.value));
    };
    for (std::size_t pair = 0; pair < 7; ++pair) {
        sample(true, defaults[2 * pair]);
        sample(false, selected[2 * pair]);
        sample(false, selected[2 * pair + 1]);
        sample(true, defaults[2 * pair + 1]);
    }
    require(std::all_of(defaults.begin(), defaults.end(), [](float value) {
                return std::isfinite(value) && value > 0.0F;
            }) &&
                std::all_of(selected.begin(), selected.end(), [](float value) {
                    return std::isfinite(value) && value > 0.0F;
                }),
            "balanced timing produced a nonpositive/nonfinite sample");
}

double median(std::array<float, 14> values) {
    std::sort(values.begin(), values.end());
    return (static_cast<double>(values[6]) + static_cast<double>(values[7])) / 2.0;
}

void check_oracle(const BoundWeight& bound, std::uint32_t rows,
                  const Run& r128, const Run& r129,
                  std::uint32_t& probe_count, std::uint32_t& maximum_steps,
                  double& maximum_bound_ratio) {
    const auto codes = reinterpret_cast<const std::uint8_t*>(bound.payload.data.data());
    const auto scales = reinterpret_cast<const float*>(bound.payload.data.data() +
                                                        bound.geometry.scale_plane_offset);
    const std::array<std::uint32_t, 4> probe_rows{0U, 17U, rows / 2U, rows - 1U};
    for (std::uint32_t token : kProbeTokens) {
        for (std::uint32_t row : probe_rows) {
            double dot = 0.0;
            double absolute_sum = 0.0;
            for (std::uint32_t column = 0; column < kColumns; ++column) {
                const double product =
                    static_cast<double>(decode_e4m3(
                        r129.codes[static_cast<std::size_t>(token) * kColumns + column])) *
                    static_cast<double>(r129.scales[token]) *
                    static_cast<double>(decode_e4m3(
                        codes[static_cast<std::size_t>(row) * kColumns + column])) *
                    static_cast<double>(scales[row]);
                dot += product;
                absolute_sum += std::abs(product);
            }
            const std::uint16_t expected = round_fp64_to_bf16(dot);
            const std::size_t index = static_cast<std::size_t>(token) * rows + row;
            const double unit_roundoff = std::ldexp(1.0, -24);
            const double gamma = kColumns * unit_roundoff /
                                 (1.0 - kColumns * unit_roundoff);
            const double expected_value =
                static_cast<float>(std::bit_cast<hip_bfloat16>(expected));
            const double bound_value =
                std::abs(expected_value - dot) + gamma * absolute_sum + 1.0e-30;
            const auto check_actual = [&](std::uint16_t actual) {
                maximum_steps = std::max(maximum_steps, steps(actual, expected));
                const double actual_value =
                    static_cast<float>(std::bit_cast<hip_bfloat16>(actual));
                maximum_bound_ratio = std::max(
                    maximum_bound_ratio, std::abs(actual_value - dot) / bound_value);
                ++probe_count;
            };
            check_actual(r129.output[index]);
            if (token < kPrefix) check_actual(r128.output[index]);
        }
    }
}

CellResult evaluate(const artifact::Reader& reader, std::string_view name, std::uint32_t rows,
                    const std::vector<hip_bfloat16>& represented, hipStream_t stream) {
    BoundWeight bound(reader, name, rows);
    const std::size_t activation_bytes =
        ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(kExtended, kColumns);
    DeviceArray<std::byte> activation(activation_bytes);
    DeviceArray<std::byte> matmul(ninfer::ops::LinearExecution::kMaximumMatmulWorkspaceBytes);
    DeviceArray<hip_bfloat16> input128(static_cast<std::size_t>(kPrefix) * kColumns);
    DeviceArray<hip_bfloat16> input129(static_cast<std::size_t>(kExtended) * kColumns);
    HIP_CHECK(hipMemcpy(input128.value, represented.data(), input128.count * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(input129.value, represented.data(), input129.count * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));

    ninfer::ops::LinearExecution execution(bound.weight, activation.value, activation.count,
                                           matmul.value, matmul.count);
    const auto& p128 = execution.prepare(kPrefix);
    const auto& p129 = execution.prepare(kExtended);
#if NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE
    if (rows == 34816U) {
        require(p128.selected_algorithm_fingerprint == kQualifiedCommonAlgorithm &&
                    p129.selected_algorithm_fingerprint == kQualifiedCommonAlgorithm,
                "selected profiles differ from qualified common FP8 fingerprint");
    }
#endif
    const Run r128 = run(execution, kPrefix, input128, rows, stream);
    const Run r129 = run(execution, kExtended, input129, rows, stream);
    const Run default128 = run(execution, kPrefix, input128, rows, stream, true);
    const Run default129 = run(execution, kExtended, input129, rows, stream, true);
    require(r128.codes == std::vector<std::uint8_t>(r129.codes.begin(),
                                                     r129.codes.begin() + r128.codes.size()),
            "T128/T129 activation FP8 code prefix differs");
    require(std::equal(r128.scales.begin(), r128.scales.end(), r129.scales.begin(),
                       [](float lhs, float rhs) {
                           return std::bit_cast<std::uint32_t>(lhs) ==
                                  std::bit_cast<std::uint32_t>(rhs);
                       }),
            "T128/T129 activation scale prefix differs");
    CellResult result{std::string(name), rows, p128, p129};
    for (std::size_t index = 0; index < r128.output.size(); ++index) {
        if (r128.output[index] == r129.output[index]) continue;
        if (result.output_prefix_mismatches == 0U) {
            result.first_output_mismatch = index;
            result.first_p128_bits = r128.output[index];
            result.first_p129_bits = r129.output[index];
        }
        ++result.output_prefix_mismatches;
    }
    for (std::size_t index = 0; index < default128.output.size(); ++index) {
        result.default_output_prefix_mismatches +=
            default128.output[index] != default129.output[index] ? 1U : 0U;
    }
#if NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE
    require(result.output_prefix_mismatches == 0U,
            "selected common FP8 algorithm did not preserve exact BF16 output prefix");
    if (rows == 34816U) {
        require(p128.selected_algorithm_fingerprint == p129.selected_algorithm_fingerprint,
                "selected T128/T129 algorithm fingerprint differs");
    }
#endif
    check_oracle(bound, rows, r128, r129, result.oracle_probes,
                 result.oracle_max_steps, result.oracle_max_bound_ratio);
    require(result.oracle_max_bound_ratio <= 1.05,
            "selected real-weight output differs from represented-operation FP64 oracle");
    DeviceArray<hip_bfloat16> timed_output(static_cast<std::size_t>(kExtended) * rows);
    time_pair(execution, kPrefix, input128, timed_output, stream,
              result.p128_default_ms, result.p128_selected_ms);
    time_pair(execution, kExtended, input129, timed_output, stream,
              result.p129_default_ms, result.p129_selected_ms);
    if (rows == 34816U) {
        require(p128.heuristic_result_count == 8 && p129.heuristic_result_count == 8 &&
                    p128.viable_algorithms.size() == 8U &&
                    p129.viable_algorithms.size() == 8U,
                "fixed eight-entry gate/up heuristic catalogs differ");
        std::vector<std::array<std::uint8_t, 16>> common;
        for (const auto& lhs : p128.viable_algorithms) {
            const auto same = [&](const auto& rhs) { return lhs.fingerprint == rhs.fingerprint; };
            require(std::count_if(p128.viable_algorithms.begin(),
                                  p128.viable_algorithms.end(), same) == 1 &&
                        std::count_if(p129.viable_algorithms.begin(),
                                      p129.viable_algorithms.end(), same) == 1,
                    "gate/up heuristic fingerprint catalog is not unique/common");
            common.push_back(lhs.fingerprint);
        }
        require(common.size() == 8U,
                "gate/up T128/T129 common heuristic catalog is not exactly eight");
        std::vector<std::string> catalog;
        for (const auto& fingerprint : common) catalog.push_back(algorithm_hex(fingerprint));
        std::sort(catalog.begin(), catalog.end());
        require(std::equal(catalog.begin(), catalog.end(),
                           kExpectedGateUpCommonCatalog.begin(),
                           kExpectedGateUpCommonCatalog.end()),
                "gate/up common heuristic fingerprint catalog differs");
        for (const auto& fingerprint : common) {
            const Run common128 = run(execution, kPrefix, input128, rows, stream,
                                      false, &fingerprint);
            const Run common129 = run(execution, kExtended, input129, rows, stream,
                                      false, &fingerprint);
            require(common128.codes ==
                        std::vector<std::uint8_t>(common129.codes.begin(),
                                                  common129.codes.begin() +
                                                      common128.codes.size()) &&
                        std::equal(common128.scales.begin(), common128.scales.end(),
                                   common129.scales.begin(), [](float lhs, float rhs) {
                                       return std::bit_cast<std::uint32_t>(lhs) ==
                                              std::bit_cast<std::uint32_t>(rhs);
                                   }),
                    "common algorithm activation prefix differs");
            CellResult::CommonResult common_result;
            common_result.fingerprint = fingerprint;
            for (std::size_t index = 0; index < common128.output.size(); ++index) {
                common_result.output_prefix_mismatches +=
                    common128.output[index] != common129.output[index] ? 1U : 0U;
            }
            check_oracle(bound, rows, common128, common129,
                         common_result.oracle_probes, common_result.oracle_max_steps,
                         common_result.oracle_max_bound_ratio);
            time_pair(execution, kPrefix, input128, timed_output, stream,
                      common_result.p128_default_ms, common_result.p128_common_ms,
                      &fingerprint);
            time_pair(execution, kExtended, input129, timed_output, stream,
                      common_result.p129_default_ms, common_result.p129_common_ms,
                      &fingerprint);
            result.common_algorithms.push_back(std::move(common_result));
        }
    }
    return result;
}

void print_profile(std::ostringstream& output,
                   const ninfer::ops::LinearExecution::PreparedProfile& profile) {
    output << "{\"tokens\":" << profile.tokens
           << ",\"activation_workspace_bytes\":" << profile.activation_workspace_bytes
           << ",\"selected_matmul_workspace_bytes\":"
           << profile.selected_matmul_workspace_bytes
           << ",\"algorithm_max_workspace_bytes\":"
           << profile.selected_algorithm_max_workspace_bytes
           << ",\"selected_heuristic_rank\":" << profile.selected_heuristic_rank
           << ",\"heuristic_result_count\":" << profile.heuristic_result_count
           << ",\"waves_count\":" << profile.selected_waves_count
           << ",\"algorithm_fingerprint\":"
           << json_string(algorithm_hex(profile.selected_algorithm_fingerprint))
           << ",\"viable_algorithms\":[";
    for (std::size_t index = 0; index < profile.viable_algorithms.size(); ++index) {
        const auto& algorithm = profile.viable_algorithms[index];
        output << (index == 0U ? "" : ",")
               << "{\"heuristic_rank\":" << algorithm.heuristic_rank
               << ",\"matmul_workspace_bytes\":" << algorithm.matmul_workspace_bytes
               << ",\"algorithm_max_workspace_bytes\":"
               << algorithm.algorithm_max_workspace_bytes
               << ",\"waves_count\":" << algorithm.waves_count
               << ",\"algorithm_fingerprint\":"
               << json_string(algorithm_hex(algorithm.fingerprint)) << '}';
    }
    output << "]}";
}

void print_samples(std::ostringstream& output, const std::array<float, 14>& samples) {
    output << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        output << (index == 0U ? "" : ",") << samples[index];
    }
    output << ']';
}

void print_timing(std::ostringstream& output, std::uint32_t tokens,
                  const std::array<float, 14>& defaults,
                  const std::array<float, 14>& selected) {
    const double default_median = median(defaults);
    const double selected_median = median(selected);
    output << "{\"tokens\":" << tokens << ",\"default_ms\":";
    print_samples(output, defaults);
    output << ",\"selected_ms\":";
    print_samples(output, selected);
    output << ",\"default_median_ms\":" << default_median
           << ",\"selected_median_ms\":" << selected_median
           << ",\"selected_over_default\":"
           << (default_median == 0.0 ? 0.0 : selected_median / default_median) << '}';
}

void print_cell(std::ostringstream& report, const CellResult& cell) {
    report << "{\"weight_name\":" << json_string(cell.name)
           << ",\"rows\":" << cell.rows << ",\"columns\":5120,"
           << "\"activation_code_prefix_exact\":true,"
           << "\"activation_scale_prefix_exact\":true,"
           << "\"bf16_output_prefix_exact\":"
           << (cell.output_prefix_mismatches == 0U ? "true" : "false") << ','
           << "\"bf16_output_prefix_mismatch_count\":"
           << cell.output_prefix_mismatches << ','
           << "\"default_bf16_output_prefix_mismatch_count\":"
           << cell.default_output_prefix_mismatches << ','
           << "\"first_output_mismatch\":";
    if (cell.output_prefix_mismatches == 0U) {
        report << "null,";
    } else {
        report << "{\"flat_index\":" << cell.first_output_mismatch
               << ",\"token\":" << cell.first_output_mismatch / cell.rows
               << ",\"row\":" << cell.first_output_mismatch % cell.rows
               << ",\"p128_bits\":" << cell.first_p128_bits
               << ",\"p129_bits\":" << cell.first_p129_bits << "},";
    }
    report << "\"represented_fp64_oracle\":{\"probe_count\":" << cell.oracle_probes
           << ",\"maximum_bf16_steps\":" << cell.oracle_max_steps
           << ",\"maximum_error_bound_ratio\":" << cell.oracle_max_bound_ratio
           << ",\"criterion\":\"abs_error <= BF16_rounding_error + gamma_5120*sum_abs; ratio <= 1.05\"},\"profiles\":[";
    print_profile(report, cell.p128);
    report << ',';
    print_profile(report, cell.p129);
    report << "],\"common_viable_algorithm_fingerprints\":[";
    bool first_common = true;
    for (const auto& lhs : cell.p128.viable_algorithms) {
        const bool common = std::any_of(
            cell.p129.viable_algorithms.begin(), cell.p129.viable_algorithms.end(),
            [&](const auto& rhs) { return lhs.fingerprint == rhs.fingerprint; });
        if (!common) continue;
        report << (first_common ? "" : ",") << json_string(algorithm_hex(lhs.fingerprint));
        first_common = false;
    }
    report << "],\"balanced_timings\":[";
    print_timing(report, kPrefix, cell.p128_default_ms, cell.p128_selected_ms);
    report << ',';
    print_timing(report, kExtended, cell.p129_default_ms, cell.p129_selected_ms);
    report << "],\"common_algorithm_trials\":[";
    for (std::size_t index = 0; index < cell.common_algorithms.size(); ++index) {
        const auto& trial = cell.common_algorithms[index];
        report << (index == 0U ? "" : ",")
               << "{\"algorithm_fingerprint\":"
               << json_string(algorithm_hex(trial.fingerprint))
               << ",\"bf16_output_prefix_exact\":"
               << (trial.output_prefix_mismatches == 0U ? "true" : "false")
               << ",\"bf16_output_prefix_mismatch_count\":"
               << trial.output_prefix_mismatches
               << ",\"eligible\":"
               << (trial.output_prefix_mismatches == 0U &&
                           trial.oracle_max_bound_ratio <= 1.05
                       ? "true" : "false")
               << ",\"represented_fp64_oracle\":{\"probe_count\":"
               << trial.oracle_probes << ",\"maximum_bf16_steps\":"
               << trial.oracle_max_steps << ",\"maximum_error_bound_ratio\":"
               << trial.oracle_max_bound_ratio << "},\"balanced_timings\":[";
        print_timing(report, kPrefix, trial.p128_default_ms, trial.p128_common_ms);
        report << ',';
        print_timing(report, kExtended, trial.p129_default_ms, trial.p129_common_ms);
        report << "]}";
    }
    report << "]}";
}

} // namespace

int main(int argc, char** argv) try {
    check_direct_bf16_rounding_edges();
    if (argc == 2 && std::string_view(argv[1]) == "--rounding-self-test") {
        constexpr double midpoint = 1.0 + 0x1p-8;
        constexpr double above = midpoint + 0x1p-30;
        std::cout << "{\"direct_midpoint_bits\":" << round_fp64_to_bf16(midpoint)
                  << ",\"direct_above_midpoint_bits\":" << round_fp64_to_bf16(above)
                  << ",\"legacy_double_rounded_bits\":"
                  << std::bit_cast<std::uint16_t>(hip_bfloat16(static_cast<float>(above)))
                  << "}\n";
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--describe") {
        std::cout << "{\"artifact_type\":\"ninfer_r9700_fp8_linear_prefix_discriminator\","
                     "\"schema_version\":2,\"device\":\"gfx1201\",\"tokens\":[128,129],"
                     "\"shapes\":[[34816,5120],[4096,5120]],"
                     "\"requires_real_production_fp8_artifact\":true,"
                     "\"qualified_common_algorithm_fingerprint\":"
                  << json_string(algorithm_hex(kQualifiedCommonAlgorithm)) << ','
                  << "\"qualified_selection\":\"preserve_t128_default_select_t129_viable\","
                  << "\"oracle_rounding\":\"direct_fp64_to_bf16_ties_even\","
                     "\"functional_output_prepoison\":\"bf16_nan_full_scan\","
                     "\"common_algorithm_candidate_selector\":"
                  << NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE << "}\n";
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--format-self-test") {
        CellResult cell;
        cell.name = std::string(kNames[0]);
        cell.rows = kRows[0];
        cell.p128.tokens = kPrefix;
        cell.p129.tokens = kExtended;
        cell.oracle_probes = 28;
        cell.oracle_max_bound_ratio = 0.5;
        std::ostringstream output;
        output << "{\"cell\":";
        print_cell(output, cell);
        output << "}\n";
        std::cout << output.str();
        return 0;
    }
    require(argc == 5 && std::string_view(argv[1]) == "--artifact" &&
                std::string_view(argv[3]) == "--output",
            "usage: fp8_linear_prefix_discriminator --artifact MODEL.ninfer --output NEW.json");
    const std::filesystem::path artifact_path = argv[2];
    const std::filesystem::path output_path = argv[4];
    require(artifact_path.is_absolute() && output_path.is_absolute() &&
                !output_path.empty() && !std::filesystem::exists(output_path) &&
                !std::filesystem::is_symlink(output_path),
            "artifact/output path contract differs");

    int device = -1;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    require(device == 0 && std::string_view(properties.name) == "AMD Radeon AI PRO R9700" &&
                std::string_view(properties.gcnArchName) == "gfx1201" &&
                properties.warpSize == 32,
            "discriminator requires device0 R9700 gfx1201 wave32");
    char pci_bus_id[32]{};
    HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));
    require(std::string_view(pci_bus_id) == kPciBusId,
            "discriminator requires exact R9700 PCI identity");
    require(read_trimmed(kPowerPath) == "auto",
            "discriminator requires auto performance level before execution");

    artifact::Reader reader(artifact_path);
    require(reader.identity().model_id == "qwen3.8-27b",
            "artifact model identity differs");
    const std::vector<hip_bfloat16> represented = make_input();
    const std::uintmax_t artifact_bytes = std::filesystem::file_size(artifact_path);
    const std::string artifact_sha256 = sha256_file(artifact_path);
    Stream stream;
    std::array<CellResult, 2> cells{
        evaluate(reader, kNames[0], kRows[0], represented, stream.value),
        evaluate(reader, kNames[1], kRows[1], represented, stream.value),
    };
    require(read_trimmed(kPowerPath) == "auto",
            "discriminator requires auto performance level after execution");

    std::ostringstream report;
    report << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n  \"artifact_type\": \"ninfer_r9700_fp8_linear_prefix_discriminator_result\",\n"
           << "  \"schema_version\": 2,\n  \"diagnostic_only\": true,\n"
           << "  \"timing_evidence_eligible\": false,\n"
           << "  \"production_routing_authorized\": false,\n"
           << "  \"common_algorithm_candidate_selector\": "
           << NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE << ",\n"
           << "  \"qualified_common_algorithm_fingerprint\": "
           << json_string(algorithm_hex(kQualifiedCommonAlgorithm)) << ",\n"
           << "  \"gate_up_common_catalog_contract\": [";
    for (std::size_t index = 0; index < kExpectedGateUpCommonCatalog.size(); ++index) {
        report << (index == 0U ? "" : ",")
               << json_string(kExpectedGateUpCommonCatalog[index]);
    }
    report << "],\n"
           << "  \"artifact\": {\"path\":" << json_string(artifact_path.string())
           << ",\"bytes\":" << artifact_bytes
           << ",\"sha256\":" << json_string(artifact_sha256)
           << ",\"model_id\":" << json_string(reader.identity().model_id)
           << ",\"weights_id\":" << json_string(reader.identity().weights_id) << "},\n"
           << "  \"device\": {\"ordinal\":0,\"name\":" << json_string(properties.name)
           << ",\"architecture\":\"gfx1201\",\"wave_size\":32,\"pci_bus_id\":"
           << json_string(pci_bus_id) << ",\"performance_level\":\"auto\"},\n"
           << "  \"input_contract\": \"one identical represented-BF16 128-column prefix plus one appended column; independent per-token E4M3 outer scale\",\n"
           << "  \"timing_contract\": \"one warmup per route then seven ABBA pairs per width; full activation quantize plus hipBLASLt matmul and status consumer\",\n"
           << "  \"cells\": [\n";
    for (std::size_t index = 0; index < cells.size(); ++index) {
        const CellResult& cell = cells[index];
        report << "    ";
        print_cell(report, cell);
        report << (index + 1 == cells.size() ? "\n" : ",\n");
    }
    const bool output_exact = std::all_of(cells.begin(), cells.end(), [](const CellResult& cell) {
        return cell.output_prefix_mismatches == 0U;
    });
    report << "  ],\n  \"conclusion\": "
           << json_string(output_exact
                  ? "activation and BF16 output prefixes are exact for both real production row-FP8 matrices"
                  : "activation prefixes are exact but at least one hipBLASLt BF16 output prefix differs")
           << "\n}\n";
    write_new(output_path, report.str());
    return 0;
} catch (const std::exception& error) {
    std::cerr << "fp8_linear_prefix_discriminator: " << error.what() << '\n';
    return 1;
}
