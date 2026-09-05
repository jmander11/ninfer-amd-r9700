#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

namespace kv = ninfer::ops::r9700::kv;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kQueryHeads = 24;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kQueriesPerKv = 6;
constexpr std::size_t kPageSize = 64;
constexpr std::size_t kLargestContext = 32768;
constexpr std::size_t kMappedPages = (kLargestContext + kPageSize - 1U) / kPageSize;
constexpr std::size_t kPhysicalPages = kMappedPages + 3U;
constexpr std::size_t kPhysicalTokens = kPhysicalPages * kPageSize;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr std::string_view kR9700PowerProfilePath =
    "/sys/class/drm/card2/device/power_dpm_force_performance_level";

#ifndef NINFER_SOURCE_DIR
#error "NINFER_SOURCE_DIR must bind split-512 qualification provenance"
#endif

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::string trim(std::string value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

std::string require_auto_power_profile() {
    std::ifstream input{std::string(kR9700PowerProfilePath)};
    if (!input) fail("cannot read R9700 power profile: " + std::string(kR9700PowerProfilePath));
    std::string value;
    std::getline(input, value);
    value = trim(std::move(value));
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

float represented_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

std::vector<std::uint32_t> make_page_table() {
    std::vector<std::uint32_t> result(kMappedPages);
    // A bijection into a larger pool deliberately makes every logical page transition
    // non-contiguous. The qualifier owns these exact stored page-table words.
    for (std::size_t page = 0; page < kMappedPages; ++page) {
        result[page] = static_cast<std::uint32_t>((page * 7U + 2U) % kPhysicalPages);
    }
    if (std::adjacent_find(result.begin(), result.end(), [](std::uint32_t a, std::uint32_t b) {
            return a == b || a + 1U == b;
        }) != result.end()) {
        fail("page-table fixture is not fragmented");
    }
    return result;
}

std::size_t key_offset(std::size_t token, std::size_t head, std::size_t feature) {
    return token % kPageSize +
           kPageSize * (feature + kHeadDim * (token / kPageSize + kPhysicalPages * head));
}

std::size_t value_offset(std::size_t token, std::size_t head, std::size_t byte) {
    return byte + (kHeadDim / 2U) *
                      (token % kPageSize +
                       kPageSize * (head + kKvHeads * (token / kPageSize)));
}

std::size_t scale_offset(std::size_t token, std::size_t head, std::size_t group,
                         std::size_t groups) {
    return group + groups *
                       (token % kPageSize +
                        kPageSize * (head + kKvHeads * (token / kPageSize)));
}

int decode_int4(std::uint8_t nibble) {
    nibble &= 0x0fU;
    return (nibble & 0x08U) == 0U ? static_cast<int>(nibble)
                                  : static_cast<int>(nibble) - 16;
}

struct StoredFixture {
    std::uint32_t value_group{};
    std::vector<std::uint32_t> page_table;
    std::vector<std::uint8_t> keys;
    std::vector<std::uint8_t> values;
    std::vector<std::uint16_t> scales;
};

StoredFixture make_stored_fixture(std::uint32_t value_group) {
    if (value_group != 16U && value_group != 32U) { fail("unsupported fixture group"); }
    const std::size_t groups = kHeadDim / value_group;
    StoredFixture fixture{
        value_group,
        make_page_table(),
        std::vector<std::uint8_t>(kPhysicalTokens * kKvHeads * kHeadDim, 0U),
        std::vector<std::uint8_t>(kPhysicalTokens * kKvHeads * kHeadDim / 2U, 0U),
        std::vector<std::uint16_t>(kPhysicalTokens * kKvHeads * groups, 0U)};

    // All three cache planes are authored directly as deterministic stored bytes. The FP8
    // patterns are finite OCP E4M3FN values, INT4 excludes the reserved -8 code, and scale words
    // are ordinary positive binary16 encodings. No production codec participates in this path.
    constexpr std::array<std::uint8_t, 12> key_words{
        0x00U, 0x18U, 0x20U, 0x28U, 0x30U, 0x38U,
        0x98U, 0xa0U, 0xa8U, 0xb0U, 0xb8U, 0x08U};
    constexpr std::array<std::uint16_t, 8> scale_words{
        0x2800U, 0x2c00U, 0x3000U, 0x3400U, 0x3800U, 0x3a00U, 0x3c00U, 0x3e00U};
    for (std::size_t logical = 0; logical < kLargestContext; ++logical) {
        const std::size_t physical =
            static_cast<std::size_t>(fixture.page_table[logical / kPageSize]) * kPageSize +
            logical % kPageSize;
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                fixture.keys[key_offset(physical, head, feature)] =
                    key_words[(logical * 5U + head * 7U + feature * 3U) % key_words.size()];
            }
            for (std::size_t group = 0; group < groups; ++group) {
                fixture.scales[scale_offset(physical, head, group, groups)] =
                    scale_words[(logical * 3U + head * 5U + group) % scale_words.size()];
            }
            for (std::size_t byte = 0; byte < kHeadDim / 2U; ++byte) {
                const int low = static_cast<int>((logical * 11U + head * 3U + byte * 5U) % 15U) - 7;
                const int high =
                    static_cast<int>((logical * 13U + head * 7U + byte * 2U + 1U) % 15U) - 7;
                const std::uint8_t packed =
                    static_cast<std::uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
                fixture.values[value_offset(physical, head, byte)] = packed;
            }
        }
    }
    return fixture;
}

std::vector<float> make_queries(std::size_t rows, std::uint32_t salt = 0U) {
    std::vector<float> result(rows * kQueryHeads * kHeadDim);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t head = 0; head < kQueryHeads; ++head) {
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                const int word = static_cast<int>(
                                     (row * 17U + head * 13U + feature * 7U + salt) % 63U) -
                                 31;
                result[(row * kQueryHeads + head) * kHeadDim + feature] =
                    represented_bf16(static_cast<float>(word) / 128.0F);
            }
        }
    }
    return result;
}

std::vector<hip_bfloat16> as_bf16(const std::vector<float>& values) {
    std::vector<hip_bfloat16> result;
    result.reserve(values.size());
    for (float value : values) { result.emplace_back(value); }
    return result;
}

std::vector<double> attention_oracle(
    const StoredFixture& fixture, const std::vector<float>& queries, std::size_t rows,
    std::size_t context, const std::vector<std::int32_t>* row_positions = nullptr,
    const std::vector<std::int32_t>* ancestor_masks = nullptr,
    const std::vector<std::int32_t>* prefix_lengths = nullptr,
    std::size_t active_rows = std::numeric_limits<std::size_t>::max(),
    bool fp8_query = false) {
    if (active_rows == std::numeric_limits<std::size_t>::max()) { active_rows = rows; }
    const std::size_t groups = kHeadDim / fixture.value_group;
    std::vector<double> result(rows * kQueryHeads * kHeadDim, 0.0);
    std::vector<double> scores;
    for (std::size_t row = 0; row < active_rows; ++row) {
        const std::size_t visible = row_positions == nullptr
                                        ? context
                                        : static_cast<std::size_t>((*row_positions)[row]) + 1U;
        std::vector<std::size_t> selected;
        selected.reserve(visible);
        for (std::size_t token = 0; token < visible; ++token) {
            bool include = true;
            if (ancestor_masks != nullptr) {
                const std::size_t prefix = static_cast<std::size_t>((*prefix_lengths)[row]);
                include = token < prefix ||
                          (((static_cast<std::uint32_t>((*ancestor_masks)[row]) >>
                             static_cast<std::uint32_t>(token - prefix)) &
                            1U) != 0U);
            }
            if (include) { selected.push_back(token); }
        }
        for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
            const std::size_t kv_head = query_head / kQueriesPerKv;
            std::array<double, kHeadDim> query_values{};
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                const float represented_query =
                    queries[(row * kQueryHeads + query_head) * kHeadDim + feature];
                query_values[feature] = fp8_query
                    ? oracle::decode_e4m3fn(oracle::encode_e4m3fn(represented_query))
                    : static_cast<double>(represented_query);
            }
            scores.assign(selected.size(), 0.0);
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t compact = 0; compact < selected.size(); ++compact) {
                const std::size_t logical = selected[compact];
                const std::size_t physical =
                    static_cast<std::size_t>(fixture.page_table[logical / kPageSize]) * kPageSize +
                    logical % kPageSize;
                double dot = 0.0;
                for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                    dot += query_values[feature] *
                           oracle::decode_e4m3fn(
                               fixture.keys[key_offset(physical, kv_head, feature)]);
                }
                scores[compact] = dot * static_cast<double>(kAttentionScale);
                maximum = std::max(maximum, scores[compact]);
            }
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                double numerator = 0.0;
                for (std::size_t compact = 0; compact < selected.size(); ++compact) {
                    const std::size_t logical = selected[compact];
                    const std::size_t physical =
                        static_cast<std::size_t>(fixture.page_table[logical / kPageSize]) *
                            kPageSize +
                        logical % kPageSize;
                    const std::uint8_t packed = fixture.values[value_offset(
                        physical, kv_head, feature / 2U)];
                    const std::uint8_t nibble =
                        (feature & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
                    const double scale = static_cast<double>(oracle::detail::fp16_to_fp32(
                        fixture.scales[scale_offset(physical, kv_head,
                                                    feature / fixture.value_group, groups)]));
                    numerator += scores[compact] * static_cast<double>(decode_int4(nibble)) * scale;
                }
                result[(row * kQueryHeads + query_head) * kHeadDim + feature] =
                    numerator / denominator;
            }
        }
    }
    return result;
}

struct SampledResult {
    std::size_t row{};
    std::size_t head{};
    std::size_t feature{};
    double value{};
};

std::vector<SampledResult> sampled_attention_oracle(
    const StoredFixture& fixture, const std::vector<float>& queries, std::size_t rows,
    std::size_t context, bool fp8_query) {
    constexpr std::array<std::size_t, 3> sampled_heads{0U, 6U, 23U};
    constexpr std::array<std::size_t, 5> sampled_features{0U, 15U, 16U, 127U, 255U};
    const std::array<std::size_t, 2> sampled_rows{0U, rows - 1U};
    const std::size_t groups = kHeadDim / fixture.value_group;
    std::vector<SampledResult> result;
    std::vector<double> probabilities(context);
    for (std::size_t row_index = 0; row_index < (rows == 1U ? 1U : sampled_rows.size());
         ++row_index) {
        const std::size_t row = sampled_rows[row_index];
        for (std::size_t query_head : sampled_heads) {
            const std::size_t kv_head = query_head / kQueriesPerKv;
            std::array<double, kHeadDim> query_values{};
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                const float represented_query =
                    queries[(row * kQueryHeads + query_head) * kHeadDim + feature];
                query_values[feature] = fp8_query
                    ? oracle::decode_e4m3fn(oracle::encode_e4m3fn(represented_query))
                    : static_cast<double>(represented_query);
            }
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t logical = 0; logical < context; ++logical) {
                const std::size_t physical =
                    static_cast<std::size_t>(fixture.page_table[logical / kPageSize]) * kPageSize +
                    logical % kPageSize;
                double dot = 0.0;
                for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                    dot += query_values[feature] * oracle::decode_e4m3fn(
                                                       fixture.keys[key_offset(
                                                           physical, kv_head, feature)]);
                }
                probabilities[logical] = dot * static_cast<double>(kAttentionScale);
                maximum = std::max(maximum, probabilities[logical]);
            }
            double denominator = 0.0;
            for (double& probability : probabilities) {
                probability = std::exp(probability - maximum);
                denominator += probability;
            }
            for (std::size_t feature : sampled_features) {
                double numerator = 0.0;
                for (std::size_t logical = 0; logical < context; ++logical) {
                    const std::size_t physical =
                        static_cast<std::size_t>(fixture.page_table[logical / kPageSize]) *
                            kPageSize +
                        logical % kPageSize;
                    const std::uint8_t packed = fixture.values[value_offset(
                        physical, kv_head, feature / 2U)];
                    const std::uint8_t nibble =
                        (feature & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
                    const double scale = static_cast<double>(oracle::detail::fp16_to_fp32(
                        fixture.scales[scale_offset(physical, kv_head,
                                                    feature / fixture.value_group, groups)]));
                    numerator += probabilities[logical] *
                                 static_cast<double>(decode_int4(nibble)) * scale;
                }
                result.push_back({row, query_head, feature, numerator / denominator});
            }
        }
    }
    return result;
}

void compare(const std::vector<float>& actual, const std::vector<double>& expected,
             const std::string& label,
             std::size_t active_elements = std::numeric_limits<std::size_t>::max(),
             double tolerance = 5.0e-4) {
    if (actual.size() != expected.size()) { fail(label + ": output extent differs"); }
    if (active_elements == std::numeric_limits<std::size_t>::max()) {
        active_elements = actual.size();
    }
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    for (std::size_t index = 0; index < active_elements; ++index) {
        const double absolute = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double relative = absolute / std::max(std::abs(expected[index]), 1.0e-6);
        maximum_absolute = std::max(maximum_absolute, absolute);
        maximum_relative = std::max(maximum_relative, relative);
        if (absolute > tolerance && relative > tolerance) {
            std::ostringstream out;
            out << label << " differs from FP64 oracle at " << index
                << ": actual=" << actual[index] << " expected=" << expected[index]
                << " abs=" << absolute << " rel=" << relative;
            fail(out.str());
        }
    }
    if (!std::all_of(actual.begin() + static_cast<std::ptrdiff_t>(active_elements), actual.end(),
                     [](float value) { return std::bit_cast<std::uint32_t>(value) == 0U; })) {
        fail(label + ": inactive tail is not exact positive zero");
    }
    std::cout << label << " max_abs=" << maximum_absolute
              << " max_rel=" << maximum_relative << '\n';
}

void require_invalid(hipError_t status, const char* label) {
    if (status != hipErrorInvalidValue) { fail(std::string(label) + " was not rejected"); }
}

struct DeviceFixture {
    ninfer::DeviceBuffer keys;
    ninfer::DeviceBuffer values;
    ninfer::DeviceBuffer scales;
    ninfer::DeviceBuffer pages;
    ninfer::DeviceBuffer page_row;
    ninfer::DeviceBuffer queries;
    ninfer::DeviceBuffer output;
    ninfer::DeviceBuffer positions;
    ninfer::DeviceBuffer masks;
    ninfer::DeviceBuffer prefixes;
    ninfer::DeviceBuffer active;
    ninfer::DeviceBuffer workspace;

    DeviceFixture(const StoredFixture& fixture, std::size_t workspace_bytes)
        : keys(fixture.keys.size()), values(fixture.values.size()),
          scales(fixture.scales.size() * sizeof(std::uint16_t)),
          pages(2U * fixture.page_table.size() * sizeof(std::uint32_t)),
          page_row(sizeof(std::int32_t)),
          queries(4U * kQueryHeads * kHeadDim * sizeof(hip_bfloat16)),
          output(4U * kQueryHeads * kHeadDim * sizeof(float)),
          positions(4U * sizeof(std::int32_t)), masks(4U * sizeof(std::int32_t)),
          prefixes(4U * sizeof(std::int32_t)), active(sizeof(std::int32_t)),
          workspace(workspace_bytes) {
        keys.copy_from_host(fixture.keys.data(), fixture.keys.size());
        values.copy_from_host(fixture.values.data(), fixture.values.size());
        scales.copy_from_host(fixture.scales.data(),
                              fixture.scales.size() * sizeof(std::uint16_t));
        std::vector<std::uint32_t> page_rows;
        page_rows.reserve(2U * fixture.page_table.size());
        page_rows.insert(page_rows.end(), fixture.page_table.begin(), fixture.page_table.end());
        page_rows.insert(page_rows.end(), fixture.page_table.begin(), fixture.page_table.end());
        pages.copy_from_host(page_rows.data(), page_rows.size() * sizeof(std::uint32_t));
    }
};

kv::Fp8Int4KvAttentionArgs make_args(DeviceFixture& device, const StoredFixture& fixture,
                                     std::uint32_t rows, std::size_t context) {
    return kv::Fp8Int4KvAttentionArgs{
        .query = static_cast<const hip_bfloat16*>(device.queries.data()),
        .key_codes = static_cast<const std::uint8_t*>(device.keys.data()),
        .value_codes = static_cast<const std::uint8_t*>(device.values.data()),
        .value_scale_bits = static_cast<const std::uint16_t*>(device.scales.data()),
        .page_table = static_cast<const std::uint32_t*>(device.pages.data()),
        .page_table_pages = static_cast<std::uint32_t>(kMappedPages),
        .context = context,
        .physical_tokens = kPhysicalTokens,
        .query_heads = static_cast<std::uint32_t>(kQueryHeads),
        .kv_heads = static_cast<std::uint32_t>(kKvHeads),
        .q_to_kv = static_cast<std::uint32_t>(kQueriesPerKv),
        .value_group = fixture.value_group,
        .query_rows = rows,
        .attention_scale = kAttentionScale,
        .key_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest,
        .value_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .output = static_cast<float*>(device.output.data())};
}

bool all_nan(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isnan(value); });
}

struct WorkspaceContractEvidence {
    std::size_t maximum_context = 0U;
    std::size_t t1_maximum_bytes = 0U;
    std::size_t t4_maximum_bytes = 0U;
    std::size_t maximum_plus_one_context = 0U;
    std::size_t t1_maximum_plus_one_bytes = 0U;
    std::size_t t4_maximum_plus_one_bytes = 0U;
    std::size_t overflow_context = 0U;
    std::size_t t1_overflow_bytes = 0U;
    std::size_t t4_overflow_bytes = 0U;
};

WorkspaceContractEvidence qualify_workspace_contract() {
    constexpr std::size_t maximum_context = 262144U;
    constexpr std::size_t expected_t1 = 37'847'040U;
    constexpr std::size_t expected_t4 = 151'388'160U;
    const std::size_t actual_t1 =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(1U, maximum_context);
    const std::size_t actual_t4 =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, maximum_context);
    if (actual_t1 != expected_t1 || actual_t4 != expected_t4) {
        fail("split-512 native maximum workspace sizes changed");
    }
    const std::size_t maximum_plus_one = maximum_context + 1U;
    const std::size_t t1_maximum_plus_one =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(1U, maximum_plus_one);
    const std::size_t t4_maximum_plus_one =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, maximum_plus_one);
    if (t1_maximum_plus_one != 0U || t4_maximum_plus_one != 0U) {
        fail("split-512 accepted maximum context plus one");
    }
    const std::size_t overflow_context = std::numeric_limits<std::size_t>::max();
    const std::size_t t1_overflow =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(1U, overflow_context);
    const std::size_t t4_overflow =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, overflow_context);
    if (t1_overflow != 0U || t4_overflow != 0U) {
        fail("split-512 workspace planner did not reject overflow edge");
    }
    return {maximum_context, actual_t1, actual_t4, maximum_plus_one,
            t1_maximum_plus_one, t4_maximum_plus_one, overflow_context,
            t1_overflow, t4_overflow};
}

void qualify_matrix(ninfer::DeviceContext& gpu, const StoredFixture& fixture,
                    DeviceFixture& device) {
    constexpr std::array<std::size_t, 8> contexts{1U, 63U, 64U, 65U,
                                                  511U, 512U, 513U, 1025U};
    for (std::size_t context : contexts) {
        for (std::uint32_t rows : {1U, 4U}) {
            const std::vector<float> queries = make_queries(rows, static_cast<std::uint32_t>(context));
            const std::vector<hip_bfloat16> bf16 = as_bf16(queries);
            device.queries.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
            kv::Fp8Int4KvAttentionArgs args = make_args(device, fixture, rows, context);
            std::vector<std::int32_t> positions(rows, static_cast<std::int32_t>(context - 1U));
            if (rows == 4U) {
                device.positions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
                args.row_positions = static_cast<const std::int32_t*>(device.positions.data());
            }
            const std::size_t workspace_bytes =
                kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(rows, context);
            HIP_CHECK(kv::fp8_int4_kv_attention_split512(
                args, device.workspace.data(), workspace_bytes, gpu.stream));
            gpu.synchronize();
            std::vector<float> actual(rows * kQueryHeads * kHeadDim);
            device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            const std::vector<double> expected = attention_oracle(
                fixture, queries, rows, context, rows == 4U ? &positions : nullptr,
                nullptr, nullptr, std::numeric_limits<std::size_t>::max(), rows == 1U);
            compare(actual, expected,
                    "G" + std::to_string(fixture.value_group) + " T=" +
                        std::to_string(rows) + " C=" + std::to_string(context),
                    std::numeric_limits<std::size_t>::max(),
                    rows == 1U ? 5.0e-3 : 5.0e-4);
        }
    }
}

void qualify_real_context_samples(ninfer::DeviceContext& gpu, const StoredFixture& fixture,
                                  DeviceFixture& device) {
    constexpr std::array<std::size_t, 2> contexts{8192U, 32768U};
    for (std::size_t context : contexts) {
        for (std::uint32_t rows : {1U, 4U}) {
            const std::vector<float> queries =
                make_queries(rows, static_cast<std::uint32_t>(context / 64U));
            const std::vector<hip_bfloat16> bf16 = as_bf16(queries);
            device.queries.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
            kv::Fp8Int4KvAttentionArgs args = make_args(device, fixture, rows, context);
            std::vector<std::int32_t> positions(rows, static_cast<std::int32_t>(context - 1U));
            if (rows == 4U) {
                device.positions.copy_from_host(positions.data(),
                                                positions.size() * sizeof(std::int32_t));
                args.row_positions = static_cast<const std::int32_t*>(device.positions.data());
            }
            const std::size_t workspace_bytes =
                kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(rows, context);
            HIP_CHECK(kv::fp8_int4_kv_attention_split512(
                args, device.workspace.data(), workspace_bytes, gpu.stream));
            gpu.synchronize();
            std::vector<float> actual(rows * kQueryHeads * kHeadDim);
            device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            const std::vector<SampledResult> expected = sampled_attention_oracle(
                fixture, queries, rows, context, rows == 1U);
            double maximum_absolute = 0.0;
            double maximum_relative = 0.0;
            for (const SampledResult& sample : expected) {
                const std::size_t index =
                    (sample.row * kQueryHeads + sample.head) * kHeadDim + sample.feature;
                const double absolute =
                    std::abs(static_cast<double>(actual[index]) - sample.value);
                const double relative = absolute / std::max(std::abs(sample.value), 1.0e-6);
                maximum_absolute = std::max(maximum_absolute, absolute);
                maximum_relative = std::max(maximum_relative, relative);
                const double tolerance = rows == 1U ? 5.0e-3 : 5.0e-4;
                if (absolute > tolerance && relative > tolerance) {
                    std::ostringstream out;
                    out << "sampled G" << fixture.value_group << " T=" << rows
                        << " C=" << context << " differs at row/head/feature=" << sample.row
                        << '/' << sample.head << '/' << sample.feature << ": actual="
                        << actual[index] << " expected=" << sample.value << " abs=" << absolute
                        << " rel=" << relative;
                    fail(out.str());
                }
            }
            std::cout << "sampled G" << fixture.value_group << " T=" << rows
                      << " C=" << context << " max_abs=" << maximum_absolute
                      << " max_rel=" << maximum_relative << '\n';
        }
    }
}

void qualify_page_rows_and_t1_positions(ninfer::DeviceContext& gpu,
                                        const StoredFixture& fixture,
                                        DeviceFixture& device) {
    constexpr std::size_t context = 513U;
    const std::size_t row_elements = kQueryHeads * kHeadDim;
    std::vector<std::uint32_t> page_rows;
    page_rows.reserve(2U * fixture.page_table.size());
    page_rows.insert(page_rows.end(), fixture.page_table.begin(), fixture.page_table.end());
    page_rows.insert(page_rows.end(), fixture.page_table.begin(), fixture.page_table.end());
    device.pages.copy_from_host(page_rows.data(), page_rows.size() * sizeof(std::uint32_t));

    // Both score producers and the INT4-V consumer must apply the same selected page-table row.
    // Row one deliberately duplicates the oracle-owned fragmented mapping, so selecting it is
    // observably nonzero metadata without changing the represented mathematical inputs.
    for (std::uint32_t rows : {1U, 4U}) {
        const std::vector<float> queries = make_queries(rows, 211U + rows);
        const std::vector<hip_bfloat16> bf16 = as_bf16(queries);
        device.queries.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
        std::vector<std::int32_t> positions(rows, static_cast<std::int32_t>(context - 1U));
        if (rows == 4U) {
            device.positions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        }
        std::int32_t selected_row = 1;
        device.page_row.copy_from_host(&selected_row, sizeof(selected_row));
        kv::Fp8Int4KvAttentionArgs args = make_args(device, fixture, rows, context);
        args.page_table_row = static_cast<const std::int32_t*>(device.page_row.data());
        args.page_table_row_stride = static_cast<std::uint32_t>(kMappedPages);
        args.page_table_row_count = 2U;
        if (rows == 4U) {
            args.row_positions = static_cast<const std::int32_t*>(device.positions.data());
        }
        const std::size_t workspace_bytes =
            kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(rows, context);
        HIP_CHECK(kv::fp8_int4_kv_attention_split512(
            args, device.workspace.data(), workspace_bytes, gpu.stream));
        gpu.synchronize();
        std::vector<float> actual(rows * row_elements);
        device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        const std::vector<double> expected = attention_oracle(
            fixture, queries, rows, context, rows == 4U ? &positions : nullptr,
            nullptr, nullptr, std::numeric_limits<std::size_t>::max(), rows == 1U);
        compare(actual, expected,
                "page-row=1 G" + std::to_string(fixture.value_group) +
                    " T=" + std::to_string(rows),
                std::numeric_limits<std::size_t>::max(), rows == 1U ? 5.0e-3 : 5.0e-4);

        for (std::int32_t malformed_row : {-1, 2}) {
            selected_row = malformed_row;
            device.page_row.copy_from_host(&selected_row, sizeof(selected_row));
            HIP_CHECK(kv::fp8_int4_kv_attention_split512(
                args, device.workspace.data(), workspace_bytes, gpu.stream));
            gpu.synchronize();
            device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            if (!all_nan(actual)) { fail("malformed page-table row did not poison output"); }
        }

        selected_row = 1;
        device.page_row.copy_from_host(&selected_row, sizeof(selected_row));
        page_rows[kMappedPages] = static_cast<std::uint32_t>(kPhysicalPages);
        device.pages.copy_from_host(page_rows.data(), page_rows.size() * sizeof(std::uint32_t));
        HIP_CHECK(kv::fp8_int4_kv_attention_split512(
            args, device.workspace.data(), workspace_bytes, gpu.stream));
        gpu.synchronize();
        device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        if (!all_nan(actual)) { fail("out-of-range physical page did not poison output"); }
        page_rows[kMappedPages] = fixture.page_table[0];
        device.pages.copy_from_host(page_rows.data(), page_rows.size() * sizeof(std::uint32_t));
    }

    // T=1 keeps its FP8-Q implementation profile, but visibility is still governed by the
    // represented row position consumed by the split softmax/PV stage.
    const std::vector<float> queries = make_queries(1U, 313U);
    const std::vector<hip_bfloat16> bf16 = as_bf16(queries);
    device.queries.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
    kv::Fp8Int4KvAttentionArgs args = make_args(device, fixture, 1U, context);
    args.row_positions = static_cast<const std::int32_t*>(device.positions.data());
    const std::size_t workspace_bytes =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(1U, context);
    std::vector<float> actual(row_elements);
    std::int32_t position = 127;
    device.positions.copy_from_host(&position, sizeof(position));
    HIP_CHECK(kv::fp8_int4_kv_attention_split512(
        args, device.workspace.data(), workspace_bytes, gpu.stream));
    gpu.synchronize();
    device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
    const std::vector<std::int32_t> shortened{position};
    const std::vector<double> expected = attention_oracle(
        fixture, queries, 1U, context, &shortened, nullptr, nullptr,
        std::numeric_limits<std::size_t>::max(), true);
    compare(actual, expected, "shortened T=1 G" + std::to_string(fixture.value_group),
            std::numeric_limits<std::size_t>::max(), 5.0e-3);
    for (std::int32_t invalid_position :
         {-1, static_cast<std::int32_t>(context)}) {
        position = invalid_position;
        device.positions.copy_from_host(&position, sizeof(position));
        HIP_CHECK(kv::fp8_int4_kv_attention_split512(
            args, device.workspace.data(), workspace_bytes, gpu.stream));
        gpu.synchronize();
        device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        if (!all_nan(actual)) { fail("invalid T=1 row position did not poison output"); }
    }
}

void qualify_graph(ninfer::DeviceContext& gpu, const StoredFixture& fixture,
                   DeviceFixture& device) {
    constexpr std::size_t context = 513U;
    constexpr std::uint32_t rows = 4U;
    // T=1 retains its qualified FP8-Q score profile. Capture it independently, then mutate the
    // contents behind the fixed query address and require the replay to observe the new values.
    {
        std::vector<float> t1_queries = make_queries(1U, 37U);
        std::vector<hip_bfloat16> t1_bf16 = as_bf16(t1_queries);
        device.queries.copy_from_host(t1_bf16.data(), t1_bf16.size() * sizeof(hip_bfloat16));
        kv::Fp8Int4KvAttentionArgs t1_args = make_args(device, fixture, 1U, context);
        const std::size_t t1_workspace_bytes =
            kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(1U, context);
        ninfer::DecodeGraphDefinition t1_definition;
        t1_definition.capture(gpu.stream, [&] {
            HIP_CHECK(kv::fp8_int4_kv_attention_split512(
                t1_args, device.workspace.data(), t1_workspace_bytes, gpu.stream));
        });
        ninfer::DecodeGraphExecutable t1_executable;
        t1_executable.instantiate(t1_definition);
        t1_executable.upload(gpu.stream);
        t1_queries = make_queries(1U, 113U);
        t1_bf16 = as_bf16(t1_queries);
        device.queries.copy_from_host(t1_bf16.data(), t1_bf16.size() * sizeof(hip_bfloat16));
        t1_executable.launch(gpu.stream);
        gpu.synchronize();
        std::vector<float> t1_actual(kQueryHeads * kHeadDim);
        device.output.copy_to_host(t1_actual.data(), t1_actual.size() * sizeof(float));
        const std::vector<double> t1_expected = attention_oracle(
            fixture, t1_queries, 1U, context, nullptr, nullptr, nullptr,
            std::numeric_limits<std::size_t>::max(), true);
        compare(t1_actual, t1_expected,
                "graph G" + std::to_string(fixture.value_group) + " T=1 replay",
                std::numeric_limits<std::size_t>::max(), 5.0e-3);
    }

    const std::vector<float> queries = make_queries(rows, 91U);
    const std::vector<hip_bfloat16> bf16 = as_bf16(queries);
    const std::vector<std::int32_t> positions{509, 510, 511, 512};
    device.queries.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
    device.positions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    kv::Fp8Int4KvAttentionArgs args = make_args(device, fixture, rows, context);
    args.row_positions = static_cast<const std::int32_t*>(device.positions.data());
    args.active_query_rows = static_cast<const std::int32_t*>(device.active.data());
    const std::size_t workspace_bytes =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(rows, context);

    const void* const stable_workspace = device.workspace.data();
    const float* const stable_output = args.output;
    ninfer::DecodeGraphDefinition definition;
    definition.capture(gpu.stream, [&] {
        HIP_CHECK(kv::fp8_int4_kv_attention_split512(
            args, device.workspace.data(), workspace_bytes, gpu.stream));
    });
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.upload(gpu.stream);
    if (stable_workspace != device.workspace.data() || stable_output != args.output) {
        fail("graph capture changed caller-owned addresses");
    }

    const std::size_t row_elements = kQueryHeads * kHeadDim;
    for (std::int32_t active : {0, 2, 4, -1, 5, 2}) {
        device.active.copy_from_host(&active, sizeof(active));
        executable.launch(gpu.stream);
        gpu.synchronize();
        std::vector<float> actual(rows * row_elements);
        device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
        if (active < 0 || active > static_cast<std::int32_t>(rows)) {
            if (!std::all_of(actual.begin(), actual.end(),
                             [](float value) { return std::isnan(value); })) {
                fail("invalid graph active-row count did not poison every output");
            }
            continue;
        }
        const std::size_t active_rows = static_cast<std::size_t>(active);
        const std::vector<double> expected = attention_oracle(
            fixture, queries, rows, context, &positions, nullptr, nullptr, active_rows);
        compare(actual, expected,
                "graph G" + std::to_string(fixture.value_group) + " active=" +
                    std::to_string(active),
                active_rows * row_elements);
    }

    // Packed-tree metadata crosses the 512-token split boundary and excludes siblings from the
    // final suffix while retaining the same query, output, metadata, and workspace allocations.
    const std::vector<std::int32_t> tree_prefixes{508, 508, 508, 508};
    const std::vector<std::int32_t> tree_masks{3, 7, 23, 31};
    device.prefixes.copy_from_host(tree_prefixes.data(), tree_prefixes.size() * sizeof(std::int32_t));
    device.masks.copy_from_host(tree_masks.data(), tree_masks.size() * sizeof(std::int32_t));
    args.prefix_lengths = static_cast<const std::int32_t*>(device.prefixes.data());
    args.ancestor_masks = static_cast<const std::int32_t*>(device.masks.data());
    args.active_query_rows = nullptr;
    HIP_CHECK(kv::fp8_int4_kv_attention_split512(
        args, device.workspace.data(), workspace_bytes, gpu.stream));
    gpu.synchronize();
    std::vector<float> actual(rows * row_elements);
    device.output.copy_to_host(actual.data(), actual.size() * sizeof(float));
    const std::vector<double> expected = attention_oracle(
        fixture, queries, rows, context, &positions, &tree_masks, &tree_prefixes);
    compare(actual, expected, "tree G" + std::to_string(fixture.value_group));

    // Device metadata errors are row-local. Valid neighbors must still compute normally while
    // negative/out-of-context positions and malformed tree prefixes poison their complete rows.
    const auto require_rowwise_invalid = [&](const std::vector<std::int32_t>& test_positions,
                                             const std::vector<std::int32_t>* test_masks,
                                             const std::vector<std::int32_t>* test_prefixes,
                                             const std::array<bool, rows>& invalid,
                                             const char* label) {
        device.positions.copy_from_host(test_positions.data(),
                                        test_positions.size() * sizeof(std::int32_t));
        kv::Fp8Int4KvAttentionArgs invalid_args = make_args(device, fixture, rows, context);
        invalid_args.row_positions = static_cast<const std::int32_t*>(device.positions.data());
        std::vector<std::int32_t> safe_positions(rows, 512);
        std::vector<std::int32_t> safe_masks;
        std::vector<std::int32_t> safe_prefixes;
        if (test_masks != nullptr) {
            device.masks.copy_from_host(test_masks->data(), test_masks->size() * sizeof(std::int32_t));
            device.prefixes.copy_from_host(test_prefixes->data(),
                                            test_prefixes->size() * sizeof(std::int32_t));
            invalid_args.ancestor_masks = static_cast<const std::int32_t*>(device.masks.data());
            invalid_args.prefix_lengths = static_cast<const std::int32_t*>(device.prefixes.data());
            safe_masks = *test_masks;
            safe_prefixes.assign(rows, 508);
        }
        HIP_CHECK(kv::fp8_int4_kv_attention_split512(
            invalid_args, device.workspace.data(), workspace_bytes, gpu.stream));
        gpu.synchronize();
        std::vector<float> rowwise(rows * row_elements);
        device.output.copy_to_host(rowwise.data(), rowwise.size() * sizeof(float));
        const std::vector<double> safe_expected = attention_oracle(
            fixture, queries, rows, context, &safe_positions,
            test_masks == nullptr ? nullptr : &safe_masks,
            test_prefixes == nullptr ? nullptr : &safe_prefixes);
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t begin = row * row_elements;
            if (invalid[row]) {
                if (!std::all_of(rowwise.begin() + static_cast<std::ptrdiff_t>(begin),
                                 rowwise.begin() + static_cast<std::ptrdiff_t>(begin + row_elements),
                                 [](float value) { return std::isnan(value); })) {
                    fail(std::string(label) + ": malformed row was not fully poisoned");
                }
            } else {
                for (std::size_t index = begin; index < begin + row_elements; ++index) {
                    const double absolute =
                        std::abs(static_cast<double>(rowwise[index]) - safe_expected[index]);
                    const double relative =
                        absolute / std::max(std::abs(safe_expected[index]), 1.0e-6);
                    if (absolute > 5.0e-4 && relative > 5.0e-4) {
                        fail(std::string(label) + ": valid neighboring row differs from oracle");
                    }
                }
            }
        }
    };
    require_rowwise_invalid({512, -1, 513, 512}, nullptr, nullptr,
                            {false, true, true, false}, "invalid causal positions");
    const std::vector<std::int32_t> invalid_tree_masks{31, 31, 31, 31};
    const std::vector<std::int32_t> invalid_tree_prefixes{508, -1, 514, 508};
    require_rowwise_invalid({512, 512, 512, 512}, &invalid_tree_masks,
                            &invalid_tree_prefixes, {false, true, true, false},
                            "invalid tree prefixes");
}

void qualify_rejections(ninfer::DeviceContext& gpu, const StoredFixture& fixture,
                        DeviceFixture& device) {
    constexpr std::size_t context = 513U;
    for (std::uint32_t rows : {1U, 4U}) {
        const std::size_t exact =
            kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(rows, context);
        kv::Fp8Int4KvAttentionArgs exact_args = make_args(device, fixture, rows, context);
        require_invalid(kv::fp8_int4_kv_attention_split512(
                            exact_args, device.workspace.data(), exact - 1U, gpu.stream),
                        rows == 1U ? "undersized T=1 split workspace"
                                   : "undersized T=4 split workspace");
        require_invalid(kv::fp8_int4_kv_attention_split512(
                            exact_args, device.workspace.data(), exact + 1U, gpu.stream),
                        rows == 1U ? "oversized T=1 split workspace"
                                   : "oversized T=4 split workspace");
    }
    const std::size_t workspace_bytes =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, context);
    kv::Fp8Int4KvAttentionArgs args = make_args(device, fixture, 4U, context);
    args.query_rows = 2U;
    require_invalid(kv::fp8_int4_kv_attention_split512(
                        args, device.workspace.data(), workspace_bytes, gpu.stream),
                    "unsupported T=2");
    args.query_rows = 1U;
    args.active_query_rows = static_cast<const std::int32_t*>(device.active.data());
    require_invalid(kv::fp8_int4_kv_attention_split512(
                        args, device.workspace.data(), workspace_bytes, gpu.stream),
                    "T=1 device-active metadata");
    args.active_query_rows = nullptr;
    args.value_group = 8U;
    require_invalid(kv::fp8_int4_kv_attention_split512(
                        args, device.workspace.data(), workspace_bytes, gpu.stream),
                    "unsupported value group");
    args.value_group = fixture.value_group;
    args.key_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    require_invalid(kv::fp8_int4_kv_attention_split512(
                        args, device.workspace.data(), workspace_bytes, gpu.stream),
                    "unsupported key layout");
}

struct BenchmarkEvidence {
    std::uint32_t value_group{};
    std::uint32_t query_rows{};
    std::uint32_t active_rows{};
    std::size_t context{};
    std::uint32_t iterations{};
    std::size_t split_workspace_bytes{};
    std::string incumbent_route;
    std::vector<float> incumbent_samples_ms;
    std::vector<float> split_samples_ms;
};

double mean(const std::vector<float>& values) {
    if (values.empty()) fail("cannot summarize empty timing samples");
    const double result = std::accumulate(values.begin(), values.end(), 0.0) /
                          static_cast<double>(values.size());
    if (!std::isfinite(result) || !(result > 0.0)) fail("timing mean is not finite and positive");
    return result;
}

BenchmarkEvidence benchmark_leaf(ninfer::DeviceContext& gpu, const StoredFixture& fixture,
                                  DeviceFixture& device, std::size_t context,
                                  std::uint32_t rows, std::uint32_t active_rows,
                                  std::uint32_t iterations) {
    if ((rows == 1U && active_rows != 1U) ||
        (rows == 4U && (active_rows == 0U || active_rows > 4U))) {
        fail("invalid split-512 timing shape");
    }
    const std::vector<float> queries =
        make_queries(rows, static_cast<std::uint32_t>(context / kPageSize + active_rows));
    const std::vector<hip_bfloat16> bf16 = as_bf16(queries);
    device.queries.copy_from_host(bf16.data(), bf16.size() * sizeof(hip_bfloat16));
    kv::Fp8Int4KvAttentionArgs args = make_args(device, fixture, rows, context);
    std::vector<std::int32_t> positions;
    std::int32_t represented_active = static_cast<std::int32_t>(active_rows);
    if (rows == 4U) {
        positions.resize(rows);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            positions[row] = static_cast<std::int32_t>(context - rows + row);
        }
        device.positions.copy_from_host(positions.data(), positions.size() * sizeof(positions[0]));
        device.active.copy_from_host(&represented_active, sizeof(represented_active));
        args.row_positions = static_cast<const std::int32_t*>(device.positions.data());
        args.active_query_rows = static_cast<const std::int32_t*>(device.active.data());
    }
    const std::size_t split_workspace_bytes =
        kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(rows, context);
    if (split_workspace_bytes == 0U || split_workspace_bytes > device.workspace.size()) {
        fail("split-512 timing workspace is unavailable");
    }
    const bool incumbent_wmma = rows == 1U;
    if (incumbent_wmma) {
        const std::size_t score_bytes = kQueryHeads * context * sizeof(float);
        if (score_bytes > device.workspace.size()) fail("incumbent score workspace is unavailable");
        args.score_workspace = static_cast<float*>(device.workspace.data());
    }
    const auto launch_incumbent = [&] {
        const hipError_t status = incumbent_wmma
            ? kv::fp8_int4_kv_attention_wmma(args, gpu.stream)
            : kv::fp8_int4_kv_attention_fused(args, gpu.stream);
        HIP_CHECK(status);
    };
    const auto launch_split = [&] {
        HIP_CHECK(kv::fp8_int4_kv_attention_split512(
            args, device.workspace.data(), split_workspace_bytes, gpu.stream));
    };
    for (std::uint32_t warmup = 0U; warmup < 3U; ++warmup) {
        if ((warmup & 1U) == 0U) {
            launch_incumbent();
            launch_split();
        } else {
            launch_split();
            launch_incumbent();
        }
    }
    gpu.synchronize();

    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    BenchmarkEvidence evidence{
        fixture.value_group, rows, active_rows, context, iterations, split_workspace_bytes,
        incumbent_wmma ? "fp8-q-wmma-score-plus-fp32-softmax-int4-pv"
                       : "bf16-q-score-streaming-online-fp32-softmax-int4-pv",
        {}, {}};
    evidence.incumbent_samples_ms.reserve(iterations);
    evidence.split_samples_ms.reserve(iterations);
    const auto measure = [&](const auto& launch, std::vector<float>& samples) {
        HIP_CHECK(hipEventRecord(begin, gpu.stream));
        launch();
        HIP_CHECK(hipEventRecord(end, gpu.stream));
        HIP_CHECK(hipEventSynchronize(end));
        float elapsed = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&elapsed, begin, end));
        if (!std::isfinite(elapsed) || !(elapsed > 0.0F)) {
            fail("split-512 timing sample is not finite and positive");
        }
        samples.push_back(elapsed);
    };
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        if ((iteration & 1U) == 0U) {
            measure(launch_incumbent, evidence.incumbent_samples_ms);
            measure(launch_split, evidence.split_samples_ms);
        } else {
            measure(launch_split, evidence.split_samples_ms);
            measure(launch_incumbent, evidence.incumbent_samples_ms);
        }
    }
    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipEventDestroy(begin));
    const double incumbent_mean = mean(evidence.incumbent_samples_ms);
    const double split_mean = mean(evidence.split_samples_ms);
    std::cout << "timing G" << fixture.value_group << " T=" << rows
              << " active=" << active_rows << " C=" << context
              << " incumbent_mean_ms=" << incumbent_mean
              << " split_mean_ms=" << split_mean
              << " incumbent_over_split=" << incumbent_mean / split_mean << '\n';
    return evidence;
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

void write_benchmark_report(const std::filesystem::path& output,
                            const std::filesystem::path& executable,
                            std::string_view executable_sha256,
                            const std::filesystem::path& source_root,
                            const std::vector<SourceIdentity>& sources,
                            const WorkspaceContractEvidence& workspace_contract,
                            std::string_view power_profile,
                            std::string_view device_name, std::string_view device_architecture,
                            int wave_size, int runtime_version,
                            int driver_version,
                            const std::vector<BenchmarkEvidence>& measurements) {
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer_r9700_split512_attention_qualification\",\n"
        << "  \"schema_version\": 2,\n"
        << "  \"architecture\": \"gfx1201\",\n"
        << "  \"disposition\": \"unpromoted-qualification-candidate\",\n"
        << "  \"physical_qualification\": {\"status\": \"passed\", "
           "\"scope\": \"full-numerical-rejection-device-graph\"},\n"
        << "  \"power_profile\": {\"path\": \"" << json_escape(kR9700PowerProfilePath)
        << "\", \"value\": \"" << json_escape(power_profile)
        << "\", \"required\": \"auto\"},\n"
        << "  \"device\": {\"name\": \"" << json_escape(device_name)
        << "\", \"architecture\": \"" << json_escape(device_architecture)
        << "\", \"wave_size\": " << wave_size
        << ", \"runtime_version\": " << runtime_version
        << ", \"driver_version\": " << driver_version << "},\n"
        << "  \"executable\": {\"path\": \"" << json_escape(executable.string())
        << "\", \"sha256\": \"" << executable_sha256 << "\"},\n"
        << "  \"source_root\": \"" << json_escape(source_root.string()) << "\",\n"
        << "  \"sources\": [\n";
    for (std::size_t index = 0U; index < sources.size(); ++index) {
        out << "    {\"role\": \"" << json_escape(sources[index].role)
            << "\", \"path\": \"" << json_escape(sources[index].path.string())
            << "\", \"sha256\": \"" << sources[index].sha256 << "\"}"
            << (index + 1U == sources.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"workspace_contract\": {\n"
        << "    \"status\": \"passed\",\n"
        << "    \"maximum_context\": " << workspace_contract.maximum_context << ",\n"
        << "    \"t1_maximum_bytes\": " << workspace_contract.t1_maximum_bytes << ",\n"
        << "    \"t4_maximum_bytes\": " << workspace_contract.t4_maximum_bytes << ",\n"
        << "    \"maximum_plus_one_context\": "
        << workspace_contract.maximum_plus_one_context << ",\n"
        << "    \"t1_maximum_plus_one_bytes\": "
        << workspace_contract.t1_maximum_plus_one_bytes << ",\n"
        << "    \"t4_maximum_plus_one_bytes\": "
        << workspace_contract.t4_maximum_plus_one_bytes << ",\n"
        << "    \"overflow_context\": " << workspace_contract.overflow_context << ",\n"
        << "    \"t1_overflow_bytes\": " << workspace_contract.t1_overflow_bytes << ",\n"
        << "    \"t4_overflow_bytes\": " << workspace_contract.t4_overflow_bytes << "\n"
        << "  },\n"
        << "  \"sample_order\": \"alternating-interleaved\",\n"
        << "  \"warmup_iterations_per_route\": 3,\n"
        << "  \"incumbent_scope\": \"complete-production-semantic-attention-leaf\",\n"
        << "  \"split_scope\": \"complete-qk-plus-partial-softmax-pv-plus-merge-leaf\",\n"
        << "  \"measurements\": [\n";
    for (std::size_t index = 0U; index < measurements.size(); ++index) {
        const BenchmarkEvidence& row = measurements[index];
        const double incumbent_mean = mean(row.incumbent_samples_ms);
        const double split_mean = mean(row.split_samples_ms);
        out << "    {\n"
            << "      \"value_group\": " << row.value_group << ",\n"
            << "      \"context\": " << row.context << ",\n"
            << "      \"query_rows\": " << row.query_rows << ",\n"
            << "      \"active_rows\": " << row.active_rows << ",\n"
            << "      \"iterations\": " << row.iterations << ",\n"
            << "      \"split_workspace_bytes\": " << row.split_workspace_bytes << ",\n"
            << "      \"incumbent_route\": \"" << row.incumbent_route << "\",\n"
            << "      \"incumbent_mean_ms\": " << incumbent_mean << ",\n"
            << "      \"split_mean_ms\": " << split_mean << ",\n"
            << "      \"incumbent_over_split\": " << incumbent_mean / split_mean << ",\n"
            << "      \"incumbent_samples_ms\": ";
        write_samples(out, row.incumbent_samples_ms);
        out << ",\n      \"split_samples_ms\": ";
        write_samples(out, row.split_samples_ms);
        out << "\n    }" << (index + 1U == measurements.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    const std::string bytes = out.str();
    const int descriptor =
        ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) {
        fail("cannot exclusively create --out-json path " + output.string() + ": " +
             std::strerror(errno));
    }
    std::size_t offset = 0U;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count <= 0) {
            const int error = errno;
            (void)::close(descriptor);
            (void)::unlink(output.c_str());
            fail("failed to write --out-json path " + output.string() + ": " +
                 std::strerror(error));
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        const int error = errno;
        (void)::unlink(output.c_str());
        fail("failed to close --out-json path " + output.string() + ": " +
             std::strerror(error));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool benchmark = false;
        bool iterations_supplied = false;
        std::uint32_t iterations = 0U;
        std::filesystem::path output;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--workspace-only") {
                if (argc != 2) {
                    fail("--workspace-only cannot be combined with other options");
                }
                (void)qualify_workspace_contract();
                std::cout << "split-512 workspace contract passed\n";
                return 0;
            }
            if (argument == "--help") {
                if (argc != 2) fail("--help cannot be combined with other options");
                std::cout << "usage: split512_attention_qual [--workspace-only] | "
                             "--benchmark --iterations 20 --out-json PATH\n";
                return 0;
            }
            if (argument == "--benchmark") {
                if (benchmark) fail("duplicate --benchmark");
                benchmark = true;
            } else if (argument == "--iterations") {
                if (iterations_supplied || index + 1 >= argc) fail("invalid --iterations option");
                const std::string_view value(argv[++index]);
                unsigned parsed = 0U;
                const auto [end, error] =
                    std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (error != std::errc{} || end != value.data() + value.size() ||
                    parsed != 20U) {
                    fail("--iterations must be exactly 20 for admission");
                }
                iterations = parsed;
                iterations_supplied = true;
            } else if (argument == "--out-json") {
                if (!output.empty() || index + 1 >= argc) fail("invalid --out-json option");
                output = argv[++index];
                if (output.empty()) fail("--out-json path must not be empty");
            } else {
                fail("usage: split512_attention_qual [--workspace-only] | "
                     "--benchmark --iterations 20 --out-json PATH");
            }
        }
        if (benchmark != iterations_supplied || benchmark != !output.empty()) {
            fail("--benchmark, --iterations N, and --out-json PATH are required together");
        }
        if (argc != 1 && !benchmark) {
            fail("usage: split512_attention_qual [--workspace-only] | "
                 "--benchmark --iterations 20 --out-json PATH");
        }

        const WorkspaceContractEvidence workspace_contract = qualify_workspace_contract();
        std::string power_profile;
        std::filesystem::path executable;
        std::filesystem::path source_root;
        std::vector<SourceIdentity> sources;
        std::string executable_sha256;
        if (benchmark) {
            power_profile = require_auto_power_profile();
            output = std::filesystem::absolute(output).lexically_normal();
            if (std::filesystem::exists(std::filesystem::symlink_status(output))) {
                fail("--out-json path already exists: " + output.string());
            }
            const std::filesystem::path parent = output.parent_path();
            if (parent.empty() || !std::filesystem::is_directory(parent)) {
                fail("--out-json parent directory does not exist: " + parent.string());
            }
            executable = std::filesystem::canonical("/proc/self/exe");
            executable_sha256 = sha256_file(executable);
            source_root = std::filesystem::canonical(NINFER_SOURCE_DIR);
            for (const auto& [role, relative] : {
                     std::pair{"qualifier", "tools/r9700/split512_attention_qual.hip"},
                     std::pair{"static_validator", "tools/r9700/check_split512_static.py"},
                     std::pair{"independent_oracle", "tests/ops/fp8_int4_kv_oracle.h"},
                     std::pair{"contract_header", "src/ops/r9700/kv/fp8_int4_kv_attention.h"},
                     std::pair{"kernel", "src/ops/r9700/kv/fp8_int4_kv_attention.hip"}}) {
                const std::filesystem::path path =
                    std::filesystem::canonical(source_root / relative);
                sources.push_back({role, path, sha256_file(path)});
            }
        }
        ninfer::DeviceContext gpu(0);
        if (!gpu.is_gfx1201() || gpu.props.warpSize != 32) {
            fail("split-512 qualifier requires wave32 gfx1201");
        }
        const std::size_t maximum_workspace =
            kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, kLargestContext);
        if (maximum_workspace == 0U) { fail("split-512 workspace planner returned zero"); }
        for (std::uint32_t group : {16U, 32U}) {
            const StoredFixture fixture = make_stored_fixture(group);
            DeviceFixture device(fixture, maximum_workspace);
            qualify_matrix(gpu, fixture, device);
            qualify_real_context_samples(gpu, fixture, device);
            qualify_page_rows_and_t1_positions(gpu, fixture, device);
            qualify_graph(gpu, fixture, device);
            qualify_rejections(gpu, fixture, device);
        }
        if (benchmark) {
            if (require_auto_power_profile() != power_profile) {
                fail("R9700 power profile changed during qualification");
            }
            std::vector<BenchmarkEvidence> measurements;
            measurements.reserve(20U);
            for (std::uint32_t group : {16U, 32U}) {
                const StoredFixture fixture = make_stored_fixture(group);
                DeviceFixture device(fixture, maximum_workspace);
                for (std::size_t context : {8192U, 32768U}) {
                    measurements.push_back(
                        benchmark_leaf(gpu, fixture, device, context, 1U, 1U, iterations));
                    for (std::uint32_t active_rows : {1U, 2U, 3U, 4U}) {
                        measurements.push_back(benchmark_leaf(
                            gpu, fixture, device, context, 4U, active_rows, iterations));
                    }
                }
            }
            if (require_auto_power_profile() != power_profile) {
                fail("R9700 power profile changed during timing");
            }
            int runtime_version = 0;
            int driver_version = 0;
            HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
            HIP_CHECK(hipDriverGetVersion(&driver_version));
            write_benchmark_report(output, executable, executable_sha256, source_root, sources,
                                   workspace_contract, power_profile, gpu.props.name,
                                   gpu.props.gcnArchName, gpu.props.warpSize, runtime_version,
                                   driver_version, measurements);
        }
        std::cout << "split-512 attention qualification passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "split-512 attention qualification failed: " << error.what() << '\n';
        return 1;
    }
}
