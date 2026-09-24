#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/fp8_int4_kv_xattention.h"
#include "ops/xattention_oracle.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace kv = ninfer::ops::r9700::kv;
namespace codec = ninfer::test::fp8_int4_kv_oracle;
namespace xo = ninfer::test::xattention_oracle;

namespace {

constexpr std::size_t kHeadDim = 256U;
constexpr std::size_t kQueryHeads = 24U;
constexpr std::size_t kKvHeads = 4U;
constexpr std::size_t kQueryRows = 65U;
constexpr std::size_t kContext = 321U;
constexpr std::size_t kPageSize = 64U;
constexpr std::size_t kPhysicalPages = 6U;
constexpr std::size_t kPhysicalTokens = kPhysicalPages * kPageSize;
#ifndef NINFER_R9700_KV_VALUE_GROUP
#define NINFER_R9700_KV_VALUE_GROUP 16
#endif
static_assert(NINFER_R9700_KV_VALUE_GROUP == 16 || NINFER_R9700_KV_VALUE_GROUP == 32);
constexpr std::size_t kValueGroup = NINFER_R9700_KV_VALUE_GROUP;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr std::size_t kBenchmarkHotPagePeriod = 16U;
constexpr double kOracleAbsoluteTolerance = 3.0e-4;
constexpr double kOracleRelativeTolerance = 3.0e-4;
constexpr std::string_view kR9700PowerProfilePath =
    "/sys/class/drm/card2/device/power_dpm_force_performance_level";

#ifndef NINFER_SOURCE_DIR
#error "NINFER_SOURCE_DIR must bind XAttention production-scale qualification provenance"
#endif

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::string require_auto_power_profile(std::string_view path) {
    std::ifstream input{std::string(path)};
    if (!input) fail("cannot read R9700 power profile: " + std::string(path));
    std::string value;
    std::getline(input, value);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    value = first == std::string::npos ? std::string{} : value.substr(first);
    if (value != "auto") {
        fail("performance qualification requires R9700 power profile 'auto', found '" +
             value + "'");
    }
    return value;
}

std::string shell_quote(std::string_view value) {
    std::string result("'");
    for (char character : value) {
        if (character == '\'') result += "'\\''";
        else result += character;
    }
    result += '\'';
    return result;
}

std::string sha256_file(const std::filesystem::path& path) {
    const std::string command = "sha256sum -- " + shell_quote(path.string());
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) fail("cannot launch sha256sum for " + path.string());
    std::array<char, 256> buffer{};
    const char* line = std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe);
    const int status = pclose(pipe);
    if (line == nullptr || status != 0) fail("cannot hash " + path.string());
    const std::string output(buffer.data());
    const std::size_t separator = output.find_first_of(" \t");
    const std::string digest = output.substr(0U, separator);
    if (digest.size() != 64U || !std::all_of(digest.begin(), digest.end(), [](char value) {
            return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
        })) {
        fail("sha256sum returned an invalid digest for " + path.string());
    }
    return digest;
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (unsigned char character : value) {
        switch (character) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
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

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream out;
    out << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(out.str());
}

#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const { return data_; }
    [[nodiscard]] std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_{};
};

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.bytes() != source.size() * sizeof(T)) fail("device copy shape mismatch");
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), destination.bytes(), hipMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copy_from_device(const DeviceBuffer<T>& source) {
    std::vector<T> result(source.bytes() / sizeof(T));
    HIP_CHECK(hipMemcpy(result.data(), source.get(), source.bytes(), hipMemcpyDeviceToHost));
    return result;
}

float represented_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t absolute = bits & 0x7fffffffU;
    if ((absolute & 0x7f800000U) != 0x7f800000U) {
        bits += 0x7fffU + ((bits >> 16U) & 1U);
    } else if ((bits & 0xffffU) != 0U) {
        bits |= 0x10000U;
    }
    return std::bit_cast<float>(bits & 0xffff0000U);
}

hip_bfloat16 as_bf16(float value) {
    hip_bfloat16 result;
    result.data = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(
                                                represented_bf16(value)) >>
                                            16U);
    return result;
}

std::uint16_t half_bits(float value) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}

float production_fixture_scale(std::size_t physical, std::size_t head,
                               std::size_t value_group_index) {
    const std::size_t physical_page = physical / kPageSize;
    const std::size_t in_page = physical % kPageSize;
    return 0.5F + 0.0625F * static_cast<float>(
        (in_page + 3U * physical_page + 5U * head) % 8U) +
        0.015625F * static_cast<float>(value_group_index % 8U);
}

void require_attention_close(const std::vector<float>& actual,
                             const std::vector<float>& expected,
                             std::size_t elements, const char* message) {
    if (actual.size() < elements || expected.size() < elements) fail("comparison shape mismatch");
    for (std::size_t index = 0U; index < elements; ++index) {
        const double absolute = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double relative =
            absolute / std::max(1.0, std::abs(static_cast<double>(expected[index])));
        if (!std::isfinite(actual[index]) ||
            (absolute > kOracleAbsoluteTolerance && relative > kOracleRelativeTolerance)) {
            fail(message);
        }
    }
}

hipDeviceProp_t select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device));
        if (std::string(properties.gcnArchName).starts_with("gfx1201") &&
            properties.warpSize == 32) {
            if (selected >= 0) fail("multiple gfx1201 devices require explicit isolation");
            selected = device;
        }
    }
    if (selected < 0) fail("no wave32 gfx1201 device found");
    HIP_CHECK(hipSetDevice(selected));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, selected));
    return properties;
}

std::size_t key_offset(std::size_t physical, std::size_t head, std::size_t feature) {
    const std::size_t page = physical / kPageSize;
    const std::size_t in_page = physical % kPageSize;
    return in_page + kPageSize * (feature + kHeadDim * (page + kPhysicalPages * head));
}

std::size_t benchmark_key_offset(std::size_t physical, std::size_t physical_pages,
                                 std::size_t head, std::size_t feature) {
    const std::size_t page = physical / kPageSize;
    const std::size_t in_page = physical % kPageSize;
    return in_page + kPageSize *
                         (feature + kHeadDim * (page + physical_pages * head));
}

std::size_t value_offset(std::size_t physical, std::size_t head, std::size_t feature) {
    const std::size_t page = physical / kPageSize;
    const std::size_t in_page = physical % kPageSize;
    return feature / 2U + (kHeadDim / 2U) *
                               (in_page + kPageSize * (head + kKvHeads * page));
}

std::size_t scale_offset(std::size_t physical, std::size_t head, std::size_t feature) {
    const std::size_t page = physical / kPageSize;
    const std::size_t in_page = physical % kPageSize;
    return feature / kValueGroup + (kHeadDim / kValueGroup) *
                                        (in_page + kPageSize * (head + kKvHeads * page));
}

struct Fixture {
    std::vector<float> query;
    std::vector<hip_bfloat16> query_bf16;
    codec::Fp8Keys logical_keys;
    codec::Int4Values logical_values;
    std::vector<std::uint8_t> physical_keys;
    std::vector<std::uint8_t> physical_values;
    std::vector<std::uint16_t> physical_scales;
    std::vector<std::uint32_t> page_table{3U, 0U, 5U, 1U, 4U, 2U};
    std::vector<std::int32_t> positions;

    Fixture() {
        query.resize(kQueryRows * kQueryHeads * kHeadDim);
        query_bf16.resize(query.size());
        positions.resize(kQueryRows);
        std::vector<float> key_source(kContext * kKvHeads * kHeadDim);
        std::vector<float> value_source(key_source.size());
        for (std::size_t row = 0U; row < kQueryRows; ++row) {
            positions[row] = static_cast<std::int32_t>(256U + row);
            for (std::size_t head = 0U; head < kQueryHeads; ++head) {
                for (std::size_t feature = 0U; feature < kHeadDim; ++feature) {
                    const int code = static_cast<int>((row * 7U + head * 13U + feature * 17U) % 61U) - 30;
                    const float value = represented_bf16(static_cast<float>(code) / 64.0F);
                    const std::size_t index = (row * kQueryHeads + head) * kHeadDim + feature;
                    query[index] = value;
                    query_bf16[index] = as_bf16(value);
                }
            }
        }
        for (std::size_t token = 0U; token < kContext; ++token) {
            for (std::size_t head = 0U; head < kKvHeads; ++head) {
                for (std::size_t feature = 0U; feature < kHeadDim; ++feature) {
                    const std::size_t index = (token * kKvHeads + head) * kHeadDim + feature;
                    const int kc = static_cast<int>((token * 11U + head * 19U + feature * 5U) % 73U) - 36;
                    const int vc = static_cast<int>((token * 3U + head * 23U + feature * 7U) % 43U) - 21;
                    key_source[index] = represented_bf16(static_cast<float>(kc) / 32.0F);
                    value_source[index] = represented_bf16(static_cast<float>(vc) / 24.0F);
                }
            }
        }
        // Make page two a decisive antidiagonal candidate without changing sink/recent coverage.
        for (std::size_t token = 128U; token < 192U; ++token) {
            for (std::size_t feature = 0U; feature < 32U; ++feature) {
                key_source[(token * kKvHeads) * kHeadDim + feature] = 6.0F;
            }
        }
        logical_keys = codec::encode_keys(key_source, kContext * kKvHeads, kHeadDim);
        logical_values = codec::encode_values(value_source, kContext * kKvHeads, kHeadDim,
                                              kValueGroup);
        physical_keys.assign(kPhysicalTokens * kKvHeads * kHeadDim, 0U);
        physical_values.assign(kPhysicalTokens * kKvHeads * kHeadDim / 2U, 0U);
        physical_scales.assign(kPhysicalTokens * kKvHeads * (kHeadDim / kValueGroup), 0U);
        for (std::size_t logical = 0U; logical < kContext; ++logical) {
            const std::size_t physical = static_cast<std::size_t>(page_table[logical / kPageSize]) *
                                             kPageSize +
                                         logical % kPageSize;
            for (std::size_t head = 0U; head < kKvHeads; ++head) {
                const std::size_t represented = logical * kKvHeads + head;
                for (std::size_t feature = 0U; feature < kHeadDim; ++feature) {
                    physical_keys[key_offset(physical, head, feature)] =
                        logical_keys.codes[represented * kHeadDim + feature];
                    const std::size_t source_byte = represented * (kHeadDim / 2U) + feature / 2U;
                    const std::uint8_t nibble = (feature & 1U) == 0U
                                                    ? logical_values.packed_codes[source_byte] & 0x0fU
                                                    : logical_values.packed_codes[source_byte] >> 4U;
                    std::uint8_t& destination = physical_values[value_offset(physical, head, feature)];
                    if ((feature & 1U) == 0U) destination = nibble;
                    else destination = static_cast<std::uint8_t>(destination | (nibble << 4U));
                    physical_scales[scale_offset(physical, head, feature)] =
                        logical_values.fp16_scales[
                            represented * (kHeadDim / kValueGroup) + feature / kValueGroup];
                }
            }
        }
    }
};

struct DeviceFixture {
    DeviceBuffer<hip_bfloat16> query{kQueryRows * kQueryHeads * kHeadDim};
    DeviceBuffer<std::uint8_t> keys{kPhysicalTokens * kKvHeads * kHeadDim};
    DeviceBuffer<std::uint8_t> values{kPhysicalTokens * kKvHeads * kHeadDim / 2U};
    DeviceBuffer<std::uint16_t> scales{kPhysicalTokens * kKvHeads * (kHeadDim / kValueGroup)};
    DeviceBuffer<std::uint32_t> pages{6U};
    DeviceBuffer<std::int32_t> positions{kQueryRows};
    DeviceBuffer<float> output{kQueryRows * kQueryHeads * kHeadDim};
    DeviceBuffer<float> dense{kQueryRows * kQueryHeads * kHeadDim};
    DeviceBuffer<std::uint8_t> workspace;

    explicit DeviceFixture(const Fixture& host, std::size_t workspace_bytes)
        : workspace(workspace_bytes + 512U) {
        copy_to_device(query, host.query_bf16);
        copy_to_device(keys, host.physical_keys);
        copy_to_device(values, host.physical_values);
        copy_to_device(scales, host.physical_scales);
        copy_to_device(pages, host.page_table);
        copy_to_device(positions, host.positions);
        HIP_CHECK(hipMemset(workspace.get(), 0xa5, workspace.bytes()));
    }
};

kv::Fp8Int4KvAttentionArgs raw_args(DeviceFixture& device) {
    return kv::Fp8Int4KvAttentionArgs{
        .query = device.query.get(),
        .key_codes = device.keys.get(),
        .value_codes = device.values.get(),
        .value_scale_bits = device.scales.get(),
        .page_table = device.pages.get(),
        .page_table_pages = 6U,
        .context = kContext,
        .physical_tokens = kPhysicalTokens,
        .query_heads = kQueryHeads,
        .kv_heads = kKvHeads,
        .q_to_kv = kQueryHeads / kKvHeads,
        .value_group = kValueGroup,
        .query_rows = kQueryRows,
        .row_positions = device.positions.get(),
        .attention_scale = kAttentionScale,
        .key_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest,
        .value_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .output = device.output.get(),
    };
}

struct OracleEvidence {
    double maximum_absolute = 0.0;
    double maximum_normalized_error = 0.0;
};

OracleEvidence qualify_route() {
    Fixture host;
    const std::size_t workspace_bytes = kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
        kQueryHeads, kKvHeads, kQueryRows, kContext);
    if (workspace_bytes == 0U) fail("XAttention workspace capacity rejected real geometry");
    DeviceFixture device(host, workspace_bytes);
    auto raw = raw_args(device);
    std::uint8_t* workspace = device.workspace.get() + 256U;
    HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
        kv::Fp8Int4KvXAttentionArgs{raw, workspace, workspace_bytes}, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    const xo::Geometry geometry{kHeadDim, kQueryHeads, kKvHeads};
    const xo::KeepSets keep = xo::antidiagonal_keep_sets(
        host.query, host.logical_keys, host.positions, geometry, kv::kXAttentionStride,
        static_cast<double>(kv::kXAttentionTau), static_cast<double>(kAttentionScale));
    const xo::KeepSets& expected_keep = keep;
    const auto bound = kv::bind_fp8_int4_kv_xattention_workspace(
        workspace, workspace_bytes, kQueryHeads, kKvHeads, kQueryRows, kContext);
    const std::size_t slots = kQueryHeads * bound.query_blocks;
    std::vector<std::int32_t> counts(slots);
    std::vector<std::uint16_t> ids(slots * bound.keep_stride);
    HIP_CHECK(hipMemcpy(counts.data(), bound.keep_count, counts.size() * sizeof(counts[0]),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ids.data(), bound.rank_and_keep, ids.size() * sizeof(ids[0]),
                        hipMemcpyDeviceToHost));
    for (std::size_t head = 0U; head < kQueryHeads; ++head) {
        for (std::size_t block = 0U; block < bound.query_blocks; ++block) {
            const std::size_t slot = head * bound.query_blocks + block;
            const auto& expected = expected_keep.at(head, block);
            if (counts[slot] != static_cast<std::int32_t>(expected.size()) ||
                !std::equal(expected.begin(), expected.end(),
                            ids.begin() + static_cast<std::ptrdiff_t>(slot * bound.keep_stride))) {
                fail("device XAttention keep list differs from independent host oracle");
            }
        }
    }
    const std::vector<double> expected = xo::sparse_attention_fp64(
        host.query, host.logical_keys, host.logical_values, host.positions, geometry, keep,
        static_cast<double>(kAttentionScale));
    const std::vector<float> actual = copy_from_device(device.output);
    double maximum_absolute = 0.0;
    double maximum_normalized_error = 0.0;
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        const double absolute = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double relative = absolute / std::max(1.0, std::abs(expected[index]));
        maximum_absolute = std::max(maximum_absolute, absolute);
        maximum_normalized_error = std::max(
            maximum_normalized_error,
            std::min(absolute / kOracleAbsoluteTolerance,
                     relative / kOracleRelativeTolerance));
        if (!std::isfinite(actual[index]) ||
            (absolute > kOracleAbsoluteTolerance && relative > kOracleRelativeTolerance)) {
            fail("XAttention sparse output exceeds independent FP64 oracle tolerance");
        }
    }

    if constexpr (kv::kXAttentionTauPermille == 1000U) {
        raw.output = device.dense.get();
        HIP_CHECK(kv::fp8_int4_kv_attention_fused(raw, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        const std::vector<float> dense = copy_from_device(device.dense);
        if (dense != actual) fail("tau-one XAttention is not bit-identical to dense attention");
    }

    // A chunk whose absolute first position is not B128 aligned must not locally reanchor Q
    // against globally anchored K. It emits the dense page list and remains numerically equivalent
    // to dense attention, independent of the compiled sparse threshold. The B16 WMMA consumer is
    // intentionally free to use a different FP32 reduction association.
    std::vector<std::int32_t> unaligned_positions(kQueryRows);
    for (std::size_t row = 0U; row < kQueryRows; ++row) {
        unaligned_positions[row] = static_cast<std::int32_t>(255U + row);
    }
    copy_to_device(device.positions, unaligned_positions);
    raw = raw_args(device);
    HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
        kv::Fp8Int4KvXAttentionArgs{raw, workspace, workspace_bytes}, nullptr));
    raw.output = device.dense.get();
    HIP_CHECK(kv::fp8_int4_kv_attention_fused(raw, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> unaligned_sparse = copy_from_device(device.output);
    const std::vector<float> unaligned_dense = copy_from_device(device.dense);
    require_attention_close(unaligned_sparse, unaligned_dense, unaligned_sparse.size(),
                            "unaligned XAttention fallback exceeds dense tolerance");
    HIP_CHECK(hipMemcpy(counts.data(), bound.keep_count, counts.size() * sizeof(counts[0]),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ids.data(), bound.rank_and_keep, ids.size() * sizeof(ids[0]),
                        hipMemcpyDeviceToHost));
    for (std::size_t slot = 0U; slot < slots; ++slot) {
        const std::vector<std::uint16_t> expected_pages{0U, 1U, 2U, 3U, 4U};
        if (counts[slot] != static_cast<std::int32_t>(expected_pages.size()) ||
            !std::equal(expected_pages.begin(), expected_pages.end(),
                        ids.begin() + static_cast<std::ptrdiff_t>(slot * bound.keep_stride))) {
            fail("unaligned XAttention fallback did not emit every causal page");
        }
    }

    // An aligned initial prefill shorter than the estimator stride still includes its partial,
    // zero-padded K group. The mandatory B128 block expands to page zero and must produce exact
    // dense attention instead of poisoning the block.
    constexpr std::uint32_t short_rows = 2U;
    std::vector<std::int32_t> short_positions(kQueryRows);
    std::iota(short_positions.begin(), short_positions.end(), std::int32_t{0});
    copy_to_device(device.positions, short_positions);
    const std::size_t short_workspace_bytes =
        kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
            kQueryHeads, kKvHeads, short_rows, kContext);
    DeviceBuffer<std::uint8_t> short_workspace(short_workspace_bytes);
    raw = raw_args(device);
    raw.query_rows = short_rows;
    HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
        kv::Fp8Int4KvXAttentionArgs{raw, short_workspace.get(), short_workspace_bytes}, nullptr));
    raw.output = device.dense.get();
    HIP_CHECK(kv::fp8_int4_kv_attention_fused(raw, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> short_sparse = copy_from_device(device.output);
    const std::vector<float> short_dense = copy_from_device(device.dense);
    const std::size_t short_elements =
        static_cast<std::size_t>(short_rows) * kQueryHeads * kHeadDim;
    require_attention_close(short_sparse, short_dense, short_elements,
                            "short initial partial-plane XAttention exceeds dense tolerance");
    const kv::XAttentionWorkspaceView short_bound =
        kv::bind_fp8_int4_kv_xattention_workspace(
            short_workspace.get(), short_workspace_bytes, kQueryHeads, kKvHeads, short_rows,
            kContext);
    std::vector<std::int32_t> short_counts(kQueryHeads);
    HIP_CHECK(hipMemcpy(short_counts.data(), short_bound.keep_count,
                        short_counts.size() * sizeof(short_counts[0]), hipMemcpyDeviceToHost));
    if (!std::all_of(short_counts.begin(), short_counts.end(),
                     [](std::int32_t count) { return count == 1; })) {
        fail("short initial partial-plane XAttention did not retain exactly page zero");
    }

    std::vector<std::uint8_t> guard(256U);
    HIP_CHECK(hipMemcpy(guard.data(), device.workspace.get(), guard.size(), hipMemcpyDeviceToHost));
    if (!std::all_of(guard.begin(), guard.end(), [](std::uint8_t byte) { return byte == 0xa5U; })) {
        fail("XAttention wrote before caller workspace");
    }
    HIP_CHECK(hipMemcpy(guard.data(), device.workspace.get() + 256U + workspace_bytes, guard.size(),
                        hipMemcpyDeviceToHost));
    if (!std::all_of(guard.begin(), guard.end(), [](std::uint8_t byte) { return byte == 0xa5U; })) {
        fail("XAttention wrote after caller workspace");
    }
    std::cout << "oracle: D256/Hq24/Hkv4 rows=65 context=" << kContext
              << " value_group=" << kValueGroup << " stride="
              << kv::kXAttentionStride << " tau=" << kv::kXAttentionTau
              << " max_abs=" << maximum_absolute
              << " max_normalized_error=" << maximum_normalized_error << '\n';
    return {maximum_absolute, maximum_normalized_error};
}

void qualify_rejection() {
    Fixture host;
    const std::size_t bytes = kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
        kQueryHeads, kKvHeads, kQueryRows, kContext);
    DeviceFixture device(host, bytes);
    auto raw = raw_args(device);
    std::uint8_t* workspace = device.workspace.get() + 256U;
    if (kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
            16777216U, kKvHeads, std::numeric_limits<std::uint32_t>::max(), 131073U) != 0U) {
        fail("XAttention capacity did not reject overflowing byte products");
    }
    if (kv::fp8_int4_kv_xattention_prefill(
            kv::Fp8Int4KvXAttentionArgs{raw, workspace + 1U, bytes}, nullptr) !=
        hipErrorInvalidValue) {
        fail("XAttention accepted a misaligned workspace");
    }
    if (kv::fp8_int4_kv_xattention_prefill(
            kv::Fp8Int4KvXAttentionArgs{raw, workspace, bytes - 1U}, nullptr) !=
        hipErrorInvalidValue) {
        fail("XAttention accepted undersized workspace");
    }
    raw.query_rows = 1U;
    if (kv::fp8_int4_kv_xattention_prefill(
            kv::Fp8Int4KvXAttentionArgs{raw, workspace, bytes}, nullptr) != hipErrorInvalidValue) {
        fail("XAttention leaked onto ordinary decode");
    }
    raw = raw_args(device);
    raw.ancestor_masks = device.positions.get();
    raw.prefix_lengths = device.positions.get();
    if (kv::fp8_int4_kv_xattention_prefill(
            kv::Fp8Int4KvXAttentionArgs{raw, workspace, bytes}, nullptr) != hipErrorInvalidValue) {
        fail("XAttention leaked onto tree/speculative verification");
    }

    auto malformed_positions = host.positions;
    malformed_positions[17] += 1;
    copy_to_device(device.positions, malformed_positions);
    raw = raw_args(device);
    HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
        kv::Fp8Int4KvXAttentionArgs{raw, workspace, bytes}, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> malformed = copy_from_device(device.output);
    if (!std::isnan(malformed[17U * kQueryHeads * kHeadDim])) {
        fail("malformed prefill position did not poison its query block");
    }

    malformed_positions = host.positions;
    malformed_positions[64] += 1;
    copy_to_device(device.positions, malformed_positions);
    raw = raw_args(device);
    HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
        kv::Fp8Int4KvXAttentionArgs{raw, workspace, bytes}, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> boundary_gap = copy_from_device(device.output);
    if (!std::isnan(boundary_gap[64U * kQueryHeads * kHeadDim])) {
        fail("query-block-boundary position gap did not poison its query block");
    }

    copy_to_device(device.positions, host.positions);
    host.query_bf16[0] = as_bf16(std::numeric_limits<float>::quiet_NaN());
    copy_to_device(device.query, host.query_bf16);
    raw = raw_args(device);
    HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
        kv::Fp8Int4KvXAttentionArgs{raw, workspace, bytes}, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> nonfinite = copy_from_device(device.output);
    if (!std::isnan(nonfinite[0])) fail("nonfinite ranker query did not poison its head block");
}

double median(std::vector<float> samples) {
    if (samples.empty()) fail("cannot summarize an empty timing sample");
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2U;
    return samples.size() % 2U != 0U
               ? static_cast<double>(samples[middle])
               : 0.5 * (static_cast<double>(samples[middle - 1U]) + samples[middle]);
}

struct BenchmarkEvidence {
    std::size_t context = 0U;
    std::uint32_t rows = 0U;
    std::uint32_t iterations = 0U;
    std::size_t workspace_bytes = 0U;
    double mean_keep_fraction = 0.0;
    std::vector<float> sparse_ms;
    std::vector<float> rank_ms;
    std::vector<float> consumer_ms;
    std::vector<float> dense_ms;
};

struct ProductionScaleEvidence {
    std::size_t context = 0U;
    std::uint32_t rows = 0U;
    std::uint32_t iterations = 0U;
    std::uint32_t rotating_replicas = 0U;
    std::size_t workspace_bytes_per_replica = 0U;
    double mean_keep_fraction = 0.0;
    double oracle_relative_l2 = 0.0;
    double oracle_max_abs_error = 0.0;
    std::vector<float> sparse_ms;
    std::vector<float> rank_ms;
    std::vector<float> consumer_ms;
};

struct ProductionReplica {
    ProductionReplica(std::size_t query_elements, std::size_t key_elements,
                      std::size_t value_elements, std::size_t scale_elements,
                      std::size_t workspace_bytes)
        : query(query_elements), keys(key_elements), values(value_elements),
          scales(scale_elements), output(query_elements), workspace(workspace_bytes) {}

    DeviceBuffer<hip_bfloat16> query;
    DeviceBuffer<std::uint8_t> keys;
    DeviceBuffer<std::uint8_t> values;
    DeviceBuffer<std::uint16_t> scales;
    DeviceBuffer<float> output;
    DeviceBuffer<std::uint8_t> workspace;
};

BenchmarkEvidence benchmark_context(std::size_t context, std::uint32_t rows,
                                    std::uint32_t iterations) {
    if (context < rows || context % kPageSize != 0U || rows < 2U || iterations == 0U) {
        fail("benchmark geometry is invalid");
    }
    const std::size_t pages = context / kPageSize;
    const std::size_t physical_tokens = context;
    const std::size_t query_elements =
        static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim;
    DeviceBuffer<hip_bfloat16> query(query_elements);
    DeviceBuffer<std::uint8_t> keys(physical_tokens * kKvHeads * kHeadDim);
    DeviceBuffer<std::uint8_t> values(physical_tokens * kKvHeads * kHeadDim / 2U);
    DeviceBuffer<std::uint16_t> scales(
        physical_tokens * kKvHeads * (kHeadDim / kValueGroup));
    DeviceBuffer<std::uint32_t> page_table(pages);
    DeviceBuffer<std::int32_t> positions(rows);
    DeviceBuffer<float> sparse_output(query_elements);
    DeviceBuffer<float> dense_output(query_elements);
    const std::size_t workspace_bytes = kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
        kQueryHeads, kKvHeads, rows, context);
    if (workspace_bytes == 0U) fail("benchmark workspace capacity rejected");
    DeviceBuffer<std::uint8_t> workspace(workspace_bytes);

    std::vector<std::uint32_t> host_pages(pages);
    std::iota(host_pages.begin(), host_pages.end(), 0U);
    std::vector<std::int32_t> host_positions(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        host_positions[row] = static_cast<std::int32_t>(context - rows + row);
    }
    copy_to_device(page_table, host_pages);
    copy_to_device(positions, host_positions);
    // A zero Q/K timing corpus has a uniform estimator distribution. At tau=.9 it therefore
    // forces about 90% of pages to be retained by construction and cannot measure the sparse
    // route at the concentrated distributions XAttention is designed to exploit. Use a fixed,
    // represented structured corpus: all-one Q, positive K on every sixteenth page, negative K
    // elsewhere. The retained fraction is still measured and reported; model quality remains a
    // separate admission gate.
    std::vector<hip_bfloat16> host_query(query_elements, as_bf16(1.0F));
    std::vector<std::uint8_t> host_keys(keys.bytes(), 0xb8U); // E4M3FN -1
    if (codec::decode_e4m3fn(0x38U) != 1.0 || codec::decode_e4m3fn(0xb8U) != -1.0) {
        fail("benchmark E4M3FN fixture codes are invalid");
    }
    for (std::size_t page = 0U; page < pages; page += kBenchmarkHotPagePeriod) {
        for (std::size_t token = page * kPageSize;
             token < (page + 1U) * kPageSize; ++token) {
            for (std::size_t head = 0U; head < kKvHeads; ++head) {
                for (std::size_t feature = 0U; feature < kHeadDim; ++feature) {
                    host_keys[benchmark_key_offset(token, pages, head, feature)] =
                        0x38U; // E4M3FN +1
                }
            }
        }
    }
    copy_to_device(query, host_query);
    copy_to_device(keys, host_keys);
    HIP_CHECK(hipMemset(values.get(), 0, values.bytes()));
    HIP_CHECK(hipMemset(scales.get(), 0, scales.bytes()));

    kv::Fp8Int4KvAttentionArgs args{
        .query = query.get(),
        .key_codes = keys.get(),
        .value_codes = values.get(),
        .value_scale_bits = scales.get(),
        .page_table = page_table.get(),
        .page_table_pages = static_cast<std::uint32_t>(pages),
        .context = context,
        .physical_tokens = physical_tokens,
        .query_heads = kQueryHeads,
        .kv_heads = kKvHeads,
        .q_to_kv = kQueryHeads / kKvHeads,
        .value_group = kValueGroup,
        .query_rows = rows,
        .row_positions = positions.get(),
        .attention_scale = kAttentionScale,
        .key_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest,
        .value_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .output = sparse_output.get(),
    };
    const auto run_sparse = [&] {
        args.output = sparse_output.get();
        HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
            kv::Fp8Int4KvXAttentionArgs{args, workspace.get(), workspace_bytes}, nullptr));
    };
    const auto run_rank = [&] {
        args.output = sparse_output.get();
        HIP_CHECK(kv::fp8_int4_kv_xattention_rank_qualification(
            kv::Fp8Int4KvXAttentionArgs{args, workspace.get(), workspace_bytes}, nullptr));
    };
    const auto run_consumer = [&] {
        args.output = sparse_output.get();
        HIP_CHECK(kv::fp8_int4_kv_xattention_consumer_qualification(
            kv::Fp8Int4KvXAttentionArgs{args, workspace.get(), workspace_bytes}, nullptr));
    };
    const auto run_dense = [&] {
        args.output = dense_output.get();
        HIP_CHECK(kv::fp8_int4_kv_attention_fused(args, nullptr));
    };
    for (std::uint32_t warmup = 0U; warmup < 3U; ++warmup) {
        run_sparse();
        run_rank();
        run_consumer();
        run_dense();
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> sparse_ms;
    std::vector<float> rank_ms;
    std::vector<float> consumer_ms;
    std::vector<float> dense_ms;
    sparse_ms.reserve(iterations);
    rank_ms.reserve(iterations);
    consumer_ms.reserve(iterations);
    dense_ms.reserve(iterations);
    const auto measure = [&](const auto& launch, std::vector<float>& samples) {
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        samples.push_back(elapsed);
    };
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        switch (iteration & 3U) {
        case 0U:
            measure(run_sparse, sparse_ms);
            measure(run_rank, rank_ms);
            run_rank();
            measure(run_consumer, consumer_ms);
            measure(run_dense, dense_ms);
            break;
        case 1U:
            measure(run_dense, dense_ms);
            run_rank();
            measure(run_consumer, consumer_ms);
            measure(run_rank, rank_ms);
            measure(run_sparse, sparse_ms);
            break;
        case 2U:
            measure(run_rank, rank_ms);
            measure(run_sparse, sparse_ms);
            measure(run_dense, dense_ms);
            run_rank();
            measure(run_consumer, consumer_ms);
            break;
        default:
            run_rank();
            measure(run_consumer, consumer_ms);
            measure(run_dense, dense_ms);
            measure(run_sparse, sparse_ms);
            measure(run_rank, rank_ms);
            break;
        }
    }
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    const kv::XAttentionWorkspaceView bound = kv::bind_fp8_int4_kv_xattention_workspace(
        workspace.get(), workspace_bytes, kQueryHeads, kKvHeads, rows, context);
    const std::size_t slots = static_cast<std::size_t>(kQueryHeads) * bound.query_blocks;
    std::vector<std::int32_t> keep_counts(slots);
    HIP_CHECK(hipMemcpy(keep_counts.data(), bound.keep_count,
                        keep_counts.size() * sizeof(keep_counts[0]), hipMemcpyDeviceToHost));
    double mean_keep_fraction = 0.0;
    for (std::size_t slot = 0U; slot < keep_counts.size(); ++slot) {
        const std::int32_t count = keep_counts[slot];
        const std::size_t query_block = slot % bound.query_blocks;
        const std::size_t block_end_row =
            std::min<std::size_t>((query_block + 1U) * kv::kXAttentionQueryBlockRows, rows) - 1U;
        const std::size_t causal_pages =
            (context - rows + block_end_row) / kPageSize + 1U;
        if (count <= 0 || static_cast<std::size_t>(count) > causal_pages) {
            fail("benchmark produced an invalid keep count");
        }
        mean_keep_fraction +=
            static_cast<double>(count) / static_cast<double>(causal_pages);
    }
    mean_keep_fraction /= static_cast<double>(keep_counts.size());
    const double sparse_median = median(sparse_ms);
    const double rank_median = median(rank_ms);
    const double consumer_median = median(consumer_ms);
    const double dense_median = median(dense_ms);
    std::cout << "timing: context=" << context << " rows=" << rows
              << " iterations=" << iterations << " stride=" << kv::kXAttentionStride
              << " tau=" << kv::kXAttentionTau << " sparse_median_ms=" << sparse_median
              << " rank_median_ms=" << rank_median
              << " consumer_median_ms=" << consumer_median
              << " dense_median_ms=" << dense_median
              << " speedup=" << dense_median / sparse_median
              << " mean_keep_fraction=" << mean_keep_fraction
              << " workspace_bytes=" << workspace_bytes << '\n';
    return BenchmarkEvidence{context, rows, iterations, workspace_bytes, mean_keep_fraction,
                             std::move(sparse_ms), std::move(rank_ms),
                             std::move(consumer_ms), std::move(dense_ms)};
}

ProductionScaleEvidence benchmark_production_scale(std::uint32_t iterations) {
    constexpr std::size_t context = 8192U;
    constexpr std::uint32_t rows = 4096U;
    constexpr std::uint32_t replicas = 4U;
    if (iterations == 0U || iterations > 20U) {
        fail("production-scale iterations must be in [1,20]");
    }
    const std::size_t pages = context / kPageSize;
    const std::size_t query_elements =
        static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim;
    const std::size_t key_elements = context * kKvHeads * kHeadDim;
    const std::size_t value_elements = context * kKvHeads * kHeadDim / 2U;
    const std::size_t scale_elements =
        context * kKvHeads * (kHeadDim / kValueGroup);
    const std::size_t workspace_bytes = kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
        kQueryHeads, kKvHeads, rows, context);
    if (workspace_bytes == 0U) fail("production-scale workspace capacity rejected");

    DeviceBuffer<std::uint32_t> page_table(pages);
    DeviceBuffer<std::int32_t> positions(rows);
    std::vector<std::uint32_t> host_pages(pages);
    for (std::size_t logical_page = 0U; logical_page < pages; ++logical_page) {
        host_pages[logical_page] =
            static_cast<std::uint32_t>((17U * logical_page + 3U) % pages);
    }
    std::vector<std::int32_t> host_positions(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        host_positions[row] = static_cast<std::int32_t>(context - rows + row);
    }
    copy_to_device(page_table, host_pages);
    copy_to_device(positions, host_positions);

    // Four address-distinct replicas prevent the one-block qualifier from turning into a
    // same-buffer hot-cache microbenchmark.  The represented V plane is deliberately nonzero and
    // signed: every even feature has code +1 and every odd feature has code -2.  Exact FP16 scales
    // vary by physical token and KV head, while the nonidentity page table forces the consumer to
    // translate logical pages.  This makes the oracle sensitive to nibble, token, page, head,
    // scale-group, and page-table addressing instead of accepting a constant plane after a bad
    // load.
    const std::vector<hip_bfloat16> host_query(query_elements, as_bf16(1.0F));
    const std::vector<std::uint8_t> host_values(value_elements, 0xe1U);
    std::vector<std::uint16_t> host_scales(scale_elements);
    for (std::size_t physical = 0U; physical < context; ++physical) {
        for (std::size_t head = 0U; head < kKvHeads; ++head) {
            for (std::size_t feature = 0U; feature < kHeadDim; feature += kValueGroup) {
                host_scales[scale_offset(physical, head, feature)] = half_bits(
                    production_fixture_scale(physical, head, feature / kValueGroup));
            }
        }
    }
    std::vector<std::unique_ptr<ProductionReplica>> buffers;
    buffers.reserve(replicas);
    for (std::uint32_t replica = 0U; replica < replicas; ++replica) {
        auto storage = std::make_unique<ProductionReplica>(
            query_elements, key_elements, value_elements, scale_elements, workspace_bytes);
        std::vector<std::uint8_t> host_keys(key_elements, 0xb8U); // E4M3FN -1
        const std::size_t phase = replica % kBenchmarkHotPagePeriod;
        for (std::size_t logical_page = phase; logical_page < pages;
             logical_page += kBenchmarkHotPagePeriod) {
            const std::size_t physical_page = host_pages[logical_page];
            for (std::size_t token = physical_page * kPageSize;
                 token < (physical_page + 1U) * kPageSize; ++token) {
                for (std::size_t head = 0U; head < kKvHeads; ++head) {
                    for (std::size_t feature = 0U; feature < kHeadDim; ++feature) {
                        host_keys[benchmark_key_offset(token, pages, head, feature)] = 0x38U;
                    }
                }
            }
        }
        copy_to_device(storage->query, host_query);
        copy_to_device(storage->keys, host_keys);
        copy_to_device(storage->values, host_values);
        copy_to_device(storage->scales, host_scales);
        buffers.push_back(std::move(storage));
    }

    const auto args_for = [&](std::uint32_t replica) {
        ProductionReplica& storage = *buffers[replica];
        return kv::Fp8Int4KvAttentionArgs{
            .query = storage.query.get(),
            .key_codes = storage.keys.get(),
            .value_codes = storage.values.get(),
            .value_scale_bits = storage.scales.get(),
            .page_table = page_table.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages),
            .context = context,
            .physical_tokens = context,
            .query_heads = kQueryHeads,
            .kv_heads = kKvHeads,
            .q_to_kv = kQueryHeads / kKvHeads,
            .value_group = kValueGroup,
            .query_rows = rows,
            .row_positions = positions.get(),
            .attention_scale = kAttentionScale,
            .key_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest,
            .value_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .output = storage.output.get(),
        };
    };
    const auto run_sparse = [&](std::uint32_t replica) {
        auto args = args_for(replica);
        HIP_CHECK(kv::fp8_int4_kv_xattention_prefill(
            kv::Fp8Int4KvXAttentionArgs{
                args, buffers[replica]->workspace.get(), workspace_bytes}, nullptr));
    };
    const auto run_rank = [&](std::uint32_t replica) {
        auto args = args_for(replica);
        HIP_CHECK(kv::fp8_int4_kv_xattention_rank_qualification(
            kv::Fp8Int4KvXAttentionArgs{
                args, buffers[replica]->workspace.get(), workspace_bytes}, nullptr));
    };
    const auto run_consumer = [&](std::uint32_t replica) {
        auto args = args_for(replica);
        HIP_CHECK(kv::fp8_int4_kv_xattention_consumer_qualification(
            kv::Fp8Int4KvXAttentionArgs{
                args, buffers[replica]->workspace.get(), workspace_bytes}, nullptr));
    };

    for (std::uint32_t replica = 0U; replica < replicas; ++replica) run_sparse(replica);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> sparse_ms;
    std::vector<float> rank_ms;
    std::vector<float> consumer_ms;
    sparse_ms.reserve(iterations);
    rank_ms.reserve(iterations);
    consumer_ms.reserve(iterations);
    const auto measure = [&](const auto& launch, std::vector<float>& samples) {
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        samples.push_back(elapsed);
    };
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const std::uint32_t sparse_replica = (3U * iteration) % replicas;
        const std::uint32_t rank_replica = (3U * iteration + 1U) % replicas;
        const std::uint32_t consumer_replica = (3U * iteration + 2U) % replicas;
        measure([&] { run_sparse(sparse_replica); }, sparse_ms);
        measure([&] { run_rank(rank_replica); }, rank_ms);
        run_rank(consumer_replica);
        measure([&] { run_consumer(consumer_replica); }, consumer_ms);
    }
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    // Validate every represented output element, not a sampled prefix.  Replica zero has score
    // +16 on every sixteenth logical page and -16 elsewhere.  Reconstruct its exact retained
    // attention from the selected logical page list, the causal frontier, the nonidentity page
    // translation, signed INT4 codes, and represented FP16 scales.
    run_rank(0U);
    run_consumer(0U);
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> output = copy_from_device(buffers[0U]->output);
    const kv::XAttentionWorkspaceView oracle_work =
        kv::bind_fp8_int4_kv_xattention_workspace(
            buffers[0U]->workspace.get(), workspace_bytes, kQueryHeads, kKvHeads, rows,
            context);
    const std::size_t oracle_slots =
        static_cast<std::size_t>(kQueryHeads) * oracle_work.query_blocks;
    std::vector<std::uint16_t> oracle_pages(oracle_slots * oracle_work.keep_stride);
    std::vector<std::int32_t> oracle_counts(oracle_slots);
    HIP_CHECK(hipMemcpy(oracle_pages.data(), oracle_work.rank_and_keep,
                        oracle_pages.size() * sizeof(oracle_pages[0]), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(oracle_counts.data(), oracle_work.keep_count,
                        oracle_counts.size() * sizeof(oracle_counts[0]), hipMemcpyDeviceToHost));
    constexpr double cold_weight = 1.2664165549094176e-14; // exp(-32), relative to +16
    double oracle_error_squared = 0.0;
    double oracle_reference_squared = 0.0;
    double oracle_max_abs_error = 0.0;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const std::uint32_t position = static_cast<std::uint32_t>(host_positions[row]);
        const std::size_t query_block = row / kv::kXAttentionQueryBlockRows;
        for (std::size_t query_head = 0U; query_head < kQueryHeads; ++query_head) {
            const std::size_t slot = query_head * oracle_work.query_blocks + query_block;
            const std::int32_t count = oracle_counts[slot];
            if (count <= 0 || static_cast<std::uint32_t>(count) > oracle_work.keep_stride) {
                fail("production-scale oracle received an invalid keep count");
            }
            const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
            double denominator = 0.0;
            double scale_numerator = 0.0;
            for (std::int32_t kept = 0; kept < count; ++kept) {
                const std::size_t logical_page =
                    oracle_pages[slot * oracle_work.keep_stride + kept];
                const std::size_t first = logical_page * kPageSize;
                if (first > position) break;
                const std::size_t end = std::min(first + kPageSize,
                                                 static_cast<std::size_t>(position) + 1U);
                const double weight = logical_page % kBenchmarkHotPagePeriod == 0U
                                          ? 1.0 : cold_weight;
                const std::size_t physical_page = host_pages[logical_page];
                for (std::size_t logical = first; logical < end; ++logical) {
                    const std::size_t physical =
                        physical_page * kPageSize + logical % kPageSize;
                    denominator += weight;
                    scale_numerator += weight *
                        static_cast<double>(production_fixture_scale(physical, kv_head, 0U));
                }
            }
            if (!(denominator > 0.0)) fail("production-scale oracle has empty support");
            const float represented_scale_mean =
                static_cast<float>(scale_numerator / denominator);
            const std::size_t output_base =
                (static_cast<std::size_t>(row) * kQueryHeads + query_head) * kHeadDim;
            for (std::size_t feature = 0U; feature < kHeadDim; ++feature) {
                const float code = (feature & 1U) == 0U ? 1.0F : -2.0F;
                const float group_adjustment = 0.015625F * static_cast<float>(
                    (feature / kValueGroup) % 8U);
                const float expected = code * (represented_scale_mean + group_adjustment);
                const float actual = output[output_base + feature];
                if (!std::isfinite(actual))
                    fail("production-scale V output is nonfinite");
                const double error = static_cast<double>(actual) - expected;
                oracle_error_squared += error * error;
                oracle_reference_squared += static_cast<double>(expected) * expected;
                oracle_max_abs_error = std::max(oracle_max_abs_error, std::abs(error));
            }
        }
    }
    const double oracle_relative_l2 =
        std::sqrt(oracle_error_squared / oracle_reference_squared);
    if (!(oracle_relative_l2 <= 3.0e-4) || !(oracle_max_abs_error <= 2.0e-3)) {
        fail("production-scale V output exceeds normwise or gross pointwise oracle tolerance");
    }

    double mean_keep_fraction = 0.0;
    std::size_t keep_observations = 0U;
    for (std::uint32_t replica = 0U; replica < replicas; ++replica) {
        run_rank(replica);
        HIP_CHECK(hipDeviceSynchronize());
        const kv::XAttentionWorkspaceView bound = kv::bind_fp8_int4_kv_xattention_workspace(
            buffers[replica]->workspace.get(), workspace_bytes, kQueryHeads, kKvHeads, rows,
            context);
        const std::size_t slots = static_cast<std::size_t>(kQueryHeads) * bound.query_blocks;
        std::vector<std::int32_t> keep_counts(slots);
        HIP_CHECK(hipMemcpy(keep_counts.data(), bound.keep_count,
                            keep_counts.size() * sizeof(keep_counts[0]), hipMemcpyDeviceToHost));
        for (std::size_t slot = 0U; slot < slots; ++slot) {
            const std::size_t query_block = slot % bound.query_blocks;
            const std::size_t final_row =
                std::min<std::size_t>((query_block + 1U) * kv::kXAttentionQueryBlockRows,
                                      rows) - 1U;
            const std::size_t causal_pages = (context - rows + final_row) / kPageSize + 1U;
            const std::int32_t count = keep_counts[slot];
            if (count <= 0 || static_cast<std::size_t>(count) > causal_pages) {
                fail("production-scale benchmark produced an invalid keep count");
            }
            mean_keep_fraction += static_cast<double>(count) /
                                  static_cast<double>(causal_pages);
            ++keep_observations;
        }
    }
    mean_keep_fraction /= static_cast<double>(keep_observations);
    std::cout << "production timing: context=" << context << " rows=" << rows
              << " replicas=" << replicas << " iterations=" << iterations
              << " sparse_median_ms=" << median(sparse_ms)
              << " rank_median_ms=" << median(rank_ms)
              << " consumer_median_ms=" << median(consumer_ms)
              << " mean_keep_fraction=" << mean_keep_fraction
              << " workspace_bytes_per_replica=" << workspace_bytes << '\n';
    return ProductionScaleEvidence{context, rows, iterations, replicas, workspace_bytes,
                                   mean_keep_fraction, oracle_relative_l2,
                                   oracle_max_abs_error, std::move(sparse_ms),
                                   std::move(rank_ms), std::move(consumer_ms)};
}

void write_samples(std::ostream& out, const std::vector<float>& samples) {
    out << '[';
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        if (index != 0U) out << ',';
        out << samples[index];
    }
    out << ']';
}

struct SourceIdentity {
    std::string role;
    std::filesystem::path path;
    std::string sha256;
};

struct AdmissionIdentity {
    std::filesystem::path source_root;
    std::filesystem::path executable;
    std::string executable_sha256;
    std::vector<SourceIdentity> sources;
    std::string device_name;
    std::string architecture;
    int wave_size = 0;
    int runtime_version = 0;
    int driver_version = 0;
};

void write_evidence(const std::string& path, const OracleEvidence& oracle,
                    const std::vector<BenchmarkEvidence>& evidence,
                    const ProductionScaleEvidence* production,
                    std::string_view power_profile,
                    const AdmissionIdentity* identity) {
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer_r9700_xattention_qualification\",\n"
        << "  \"schema_version\": " << (production == nullptr ? 5 : 6) << ",\n"
        << "  \"architecture\": \"gfx1201\",\n"
        << "  \"disposition\": \"qualification-candidate\",\n"
        << "  \"power_profile\": {\"path\": \"" << kR9700PowerProfilePath
        << "\", \"value\": \"" << power_profile << "\", \"required\": \"auto\"},\n"
        << "  \"physical_oracle\": {\n"
        << "    \"status\": \"passed\",\n"
        << "    \"max_abs_error\": " << oracle.maximum_absolute << ",\n"
        << "    \"max_normalized_error\": " << oracle.maximum_normalized_error << ",\n"
        << "    \"absolute_tolerance\": " << kOracleAbsoluteTolerance << ",\n"
        << "    \"relative_tolerance\": " << kOracleRelativeTolerance << ",\n"
        << "    \"criterion\": \"per-element-absolute-or-relative\"\n"
        << "  },\n"
        << "  \"timing_fixture\": \"structured-concentrated-e4m3fn-v1\",\n"
        << "  \"hot_page_period\": " << kBenchmarkHotPagePeriod << ",\n"
        << "  \"sample_order\": \"rotating-interleaved\",\n"
        << "  \"stage_samples_are_independent\": true,\n"
        << "  \"rank_stage\": \"fp8-k-pack-plus-two-pass-bf16-wmma-plus-b128-select\",\n"
        << "  \"consumer_stage\": "
           "\"b16-query-b16-key-bf16-wmma-online-fp32-softmax-direct-int4-v\",\n"
        << "  \"value_group\": " << kValueGroup << ",\n"
        << "  \"stride\": " << kv::kXAttentionStride << ",\n"
        << "  \"tau_permille\": " << kv::kXAttentionTauPermille << ",\n";
    if (production != nullptr) {
        if (identity == nullptr) fail("production-scale report identity is missing");
        out << "  \"physical_qualification\": {\"status\": \"passed\", "
               "\"scope\": \"full-numerical-rejection-before-timing\"},\n"
            << "  \"device\": {\"name\": \"" << json_escape(identity->device_name)
            << "\", \"architecture\": \"" << json_escape(identity->architecture)
            << "\", \"wave_size\": " << identity->wave_size
            << ", \"runtime_version\": " << identity->runtime_version
            << ", \"driver_version\": " << identity->driver_version << "},\n"
            << "  \"executable\": {\"path\": \""
            << json_escape(identity->executable.string()) << "\", \"sha256\": \""
            << identity->executable_sha256 << "\"},\n"
            << "  \"source_root\": \"" << json_escape(identity->source_root.string())
            << "\",\n"
            << "  \"sources\": [\n";
        for (std::size_t index = 0U; index < identity->sources.size(); ++index) {
            const SourceIdentity& source = identity->sources[index];
            out << "    {\"role\": \"" << json_escape(source.role)
                << "\", \"path\": \"" << json_escape(source.path.string())
                << "\", \"sha256\": \"" << source.sha256 << "\"}"
                << (index + 1U == identity->sources.size() ? "\n" : ",\n");
        }
        out << "  ],\n";
    }
    out << "  \"measurements\": [\n";
    for (std::size_t index = 0U; index < evidence.size(); ++index) {
        const BenchmarkEvidence& row = evidence[index];
        const double sparse_median = median(row.sparse_ms);
        const double dense_median = median(row.dense_ms);
        out << "    {\n"
            << "      \"context\": " << row.context << ",\n"
            << "      \"query_rows\": " << row.rows << ",\n"
            << "      \"iterations\": " << row.iterations << ",\n"
            << "      \"workspace_bytes\": " << row.workspace_bytes << ",\n"
            << "      \"mean_keep_fraction\": " << row.mean_keep_fraction << ",\n"
            << "      \"sparse_median_ms\": " << sparse_median << ",\n"
            << "      \"rank_median_ms\": " << median(row.rank_ms) << ",\n"
            << "      \"consumer_median_ms\": " << median(row.consumer_ms) << ",\n"
            << "      \"dense_median_ms\": " << dense_median << ",\n"
            << "      \"dense_over_sparse\": " << dense_median / sparse_median << ",\n"
            << "      \"sparse_samples_ms\": ";
        write_samples(out, row.sparse_ms);
        out << ",\n      \"rank_samples_ms\": ";
        write_samples(out, row.rank_ms);
        out << ",\n      \"consumer_samples_ms\": ";
        write_samples(out, row.consumer_ms);
        out << ",\n      \"dense_samples_ms\": ";
        write_samples(out, row.dense_ms);
        out << "\n    }" << (index + 1U == evidence.size() ? "\n" : ",\n");
    }
    out << "  ]";
    if (production != nullptr) {
        out << ",\n  \"production_scale\": {\n"
            << "    \"fixture\": \"rotating-address-concentrated-e4m3fn-nonzero-int4-v1\",\n"
            << "    \"context\": " << production->context << ",\n"
            << "    \"query_rows\": " << production->rows << ",\n"
            << "    \"iterations\": " << production->iterations << ",\n"
            << "    \"rotating_replicas\": " << production->rotating_replicas << ",\n"
            << "    \"workspace_bytes_per_replica\": "
            << production->workspace_bytes_per_replica << ",\n"
            << "    \"value_fixture\": \"signed-int4-feature-1-minus2-token-head-varying-fp16-scale\",\n"
            << "    \"output_oracle\": \"all-elements-fp64-softmax-pv-from-device-keep-list\",\n"
            << "    \"oracle_relative_l2\": " << production->oracle_relative_l2 << ",\n"
            << "    \"oracle_relative_l2_tolerance\": 0.0003,\n"
            << "    \"oracle_max_abs_error\": " << production->oracle_max_abs_error << ",\n"
            << "    \"oracle_gross_max_abs_tolerance\": 0.002,\n"
            << "    \"mean_keep_fraction\": " << production->mean_keep_fraction << ",\n"
            << "    \"sparse_median_ms\": " << median(production->sparse_ms) << ",\n"
            << "    \"rank_median_ms\": " << median(production->rank_ms) << ",\n"
            << "    \"consumer_median_ms\": " << median(production->consumer_ms) << ",\n"
            << "    \"sparse_samples_ms\": ";
        write_samples(out, production->sparse_ms);
        out << ",\n    \"rank_samples_ms\": ";
        write_samples(out, production->rank_ms);
        out << ",\n    \"consumer_samples_ms\": ";
        write_samples(out, production->consumer_ms);
        out << "\n  }";
    }
    out << "\n}\n";
    if (!out) fail("failed to serialize --out-json report");
    const std::string bytes = out.str();
    if (production == nullptr) {
        std::ofstream legacy(path);
        if (!legacy) fail("cannot open --out-json path: " + path);
        legacy << bytes;
        if (!legacy) fail("failed to write --out-json path: " + path);
        return;
    }
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) {
        fail("cannot exclusively create --out-json path " + path + ": " +
             std::strerror(errno));
    }
    std::size_t offset = 0U;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count <= 0) {
            const int error = errno;
            (void)::close(descriptor);
            (void)::unlink(path.c_str());
            fail("failed to write --out-json path " + path + ": " + std::strerror(error));
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        const int error = errno;
        (void)::unlink(path.c_str());
        fail("failed to close --out-json path " + path + ": " + std::strerror(error));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool benchmark = false;
        bool production_scale = false;
        std::uint32_t iterations = 10U;
        std::string out_json;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--benchmark") {
                benchmark = true;
            } else if (argument == "--production-scale") {
                production_scale = true;
            } else if (argument == "--iterations" && index + 1 < argc) {
                const unsigned long parsed = std::stoul(argv[++index]);
                if (parsed == 0UL || parsed > 1000UL) fail("invalid --iterations value");
                iterations = static_cast<std::uint32_t>(parsed);
            } else if (argument == "--out-json" && index + 1 < argc) {
                out_json = argv[++index];
            } else {
                fail("usage: xattention_prefill_qual [--benchmark] [--production-scale] [--iterations N] "
                     "[--out-json PATH]");
            }
        }
        if (!out_json.empty() && !benchmark) fail("--out-json requires --benchmark");
        if (production_scale && !benchmark) fail("--production-scale requires --benchmark");
        if (production_scale && out_json.empty()) {
            fail("--production-scale admission requires --out-json");
        }
        if (production_scale && iterations > 20U) {
            fail("production-scale iterations must be in [1,20]");
        }
        const std::string power_profile = benchmark
            ? require_auto_power_profile(kR9700PowerProfilePath) : std::string{};
        AdmissionIdentity identity;
        if (production_scale) {
            const std::filesystem::path output =
                std::filesystem::absolute(out_json).lexically_normal();
            if (std::filesystem::exists(output)) {
                fail("--out-json path already exists: " + output.string());
            }
            if (!std::filesystem::is_directory(output.parent_path())) {
                fail("--out-json parent directory does not exist: " +
                     output.parent_path().string());
            }
            out_json = output.string();
            identity.executable = std::filesystem::canonical("/proc/self/exe");
            identity.executable_sha256 = sha256_file(identity.executable);
            const std::filesystem::path root = std::filesystem::canonical(NINFER_SOURCE_DIR);
            identity.source_root = root;
            const auto add_source = [&](std::string role, const std::filesystem::path& relative) {
                const std::filesystem::path source = std::filesystem::canonical(root / relative);
                identity.sources.push_back(
                    {std::move(role), source, sha256_file(source)});
            };
            add_source("qualifier", "tools/r9700/xattention_prefill_qual.hip");
            add_source("contract_header", "src/ops/r9700/kv/fp8_int4_kv_xattention.h");
            add_source("kernel", "src/ops/r9700/kv/fp8_int4_kv_xattention.hip");
            add_source("profile", "src/ops/r9700/kv/r9700_xattention_profile.h");
            add_source("dense_control_kernel", "src/ops/r9700/kv/fp8_int4_kv_attention.hip");
            add_source("independent_oracle", "tests/ops/xattention_oracle.h");
            add_source("oracle_codec", "tests/ops/fp8_int4_kv_oracle.h");
        }
        const hipDeviceProp_t device = select_r9700();
        if (production_scale) {
            identity.device_name = device.name;
            identity.architecture = device.gcnArchName;
            identity.wave_size = device.warpSize;
            HIP_CHECK(hipRuntimeGetVersion(&identity.runtime_version));
            HIP_CHECK(hipDriverGetVersion(&identity.driver_version));
        }
        const OracleEvidence oracle = qualify_route();
        qualify_rejection();
        if (production_scale &&
            require_auto_power_profile(kR9700PowerProfilePath) != power_profile) {
            fail("R9700 power profile changed during numerical qualification");
        }
        if (benchmark) {
            std::vector<BenchmarkEvidence> evidence;
            evidence.push_back(benchmark_context(8192U, 128U, iterations));
            evidence.push_back(benchmark_context(32768U, 128U, iterations));
            std::unique_ptr<ProductionScaleEvidence> production;
            if (production_scale) {
                production = std::make_unique<ProductionScaleEvidence>(
                    benchmark_production_scale(iterations));
                if (require_auto_power_profile(kR9700PowerProfilePath) != power_profile) {
                    fail("R9700 power profile changed during timing");
                }
            }
            if (!out_json.empty()) {
                write_evidence(out_json, oracle, evidence, production.get(),
                               power_profile, production_scale ? &identity : nullptr);
            }
        }
        std::cout << "OK gfx1201 XAttention prefill evaluator\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
