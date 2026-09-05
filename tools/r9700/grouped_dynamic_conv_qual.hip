#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/grouped_dynamic_conv.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::DeviceBuffer;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::WorkspaceArena;

constexpr int kD = ninfer::ops::kGroupedDynamicConvHidden;
constexpr int kGroups = ninfer::ops::kGroupedDynamicConvGroups;
constexpr int kProjection = ninfer::ops::kGroupedDynamicConvProjRows;
constexpr int kW8Group = 32;
constexpr int kQ4Group = 64;

enum class WeightKind : std::uint8_t { BF16, W8, Q4 };

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

template <class T>
DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer result(values.size() * sizeof(T));
    result.copy_from_host(values.data(), result.size());
    return result;
}

float patterned(int first, int second, int modulus, float divisor) {
    return static_cast<float>((first * 17 + second * 29) % modulus - modulus / 2) / divisor;
}

std::uint16_t half_bits(float value) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}

float half_value(std::uint16_t bits) {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

struct ProjectionWeight {
    std::vector<hip_bfloat16> bf16;
    std::vector<std::int8_t> codes;
    std::vector<std::uint8_t> q4_codes;
    std::vector<std::uint8_t> q4_codes_row;
    std::vector<std::uint16_t> scales;
    std::vector<std::uint16_t> q4_scales_row;
    DeviceBuffer d_primary;
    DeviceBuffer d_scales;
    Weight view;
    WeightKind kind = WeightKind::BF16;
};

void set_nibble(std::vector<std::uint8_t>& packed, std::size_t index, int code) {
    const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
    packed[index >> 1U] |= static_cast<std::uint8_t>((code & 0x0f) << shift);
}

ProjectionWeight make_weight(WeightKind kind) {
    ProjectionWeight result;
    result.kind = kind;
    if (kind == WeightKind::W8) {
        result.codes.resize(static_cast<std::size_t>(kProjection) * kD);
        result.scales.resize(static_cast<std::size_t>(kProjection) * (kD / kW8Group));
        for (int row = 0; row < kProjection; ++row) {
            for (int group = 0; group < kD / kW8Group; ++group) {
                const float scale = static_cast<float>(1 + (row + 3 * group) % 5) / 2048.0F;
                result.scales[static_cast<std::size_t>(row) * (kD / kW8Group) + group] =
                    half_bits(scale);
                for (int lane = 0; lane < kW8Group; ++lane) {
                    const int code = (row * 13 + group * 7 + lane * 5) % 31 - 15;
                    result.codes[static_cast<std::size_t>(row) * kD +
                                 group * kW8Group + lane] = static_cast<std::int8_t>(code);
                }
            }
        }
        result.d_primary = upload(result.codes);
        result.d_scales = upload(result.scales);
        result.view.payload = result.d_primary.data();
        result.view.payload_bytes = result.d_primary.size() + result.d_scales.size();
        result.view.qtype = QType::W8G32_F16S;
        result.view.group_size = kW8Group;
        result.view.qdata = result.d_primary.data();
        result.view.qdata_bytes = result.d_primary.size();
        result.view.scales = result.d_scales.data();
        result.view.scale_bytes = result.d_scales.size();
        result.view.group = kW8Group;
        result.view.layout = QuantLayout::RowSplit;
        result.view.scale_dtype = DType::FP16;
        result.view.scale_ne[0] = kD / kW8Group;
        result.view.scale_ne[1] = kProjection;
        result.view.scale_nb[0] = sizeof(std::uint16_t);
        result.view.scale_nb[1] = (kD / kW8Group) * sizeof(std::uint16_t);
    } else if (kind == WeightKind::Q4) {
        result.q4_codes.resize(static_cast<std::size_t>(kProjection) * kD / 2U);
        result.q4_codes_row.resize(result.q4_codes.size());
        result.scales.resize(static_cast<std::size_t>(kProjection) * (kD / kQ4Group));
        result.q4_scales_row.resize(result.scales.size());
        constexpr std::size_t groups = kD / kQ4Group;
        for (int row = 0; row < kProjection; ++row) {
            for (int group = 0; group < kD / kQ4Group; ++group) {
                const float scale = static_cast<float>(1 + (row + 3 * group) % 5) / 1024.0F;
                const std::size_t logical_scale = static_cast<std::size_t>(row) * groups + group;
                const std::size_t n16_scale =
                    ((static_cast<std::size_t>(row) >> 4U) * groups + group) * 16U +
                    (static_cast<std::size_t>(row) & 15U);
                result.q4_scales_row[logical_scale] = half_bits(scale);
                result.scales[n16_scale] = half_bits(scale);
                for (int lane = 0; lane < kQ4Group; ++lane) {
                    const int code = (row * 13 + group * 7 + lane * 5) % 15 - 7;
                    const std::size_t logical =
                        static_cast<std::size_t>(row) * kD + group * kQ4Group + lane;
                    set_nibble(result.q4_codes_row, logical, code);
                    const std::size_t pair = static_cast<std::size_t>(lane) / 16U;
                    const std::size_t within = static_cast<std::size_t>(lane) % 16U;
                    const std::size_t word =
                        (((static_cast<std::size_t>(row) >> 4U) * groups + group) * 4U +
                         pair) * 16U + (static_cast<std::size_t>(row) & 15U);
                    set_nibble(result.q4_codes, word * 16U + within, code);
                }
            }
        }
        result.d_primary = upload(result.q4_codes);
        result.d_scales = upload(result.scales);
        result.view.payload = result.d_primary.data();
        result.view.payload_bytes = result.d_primary.size() + result.d_scales.size();
        result.view.qtype = QType::Q4G64_F16S;
        result.view.group_size = kQ4Group;
        result.view.qdata = result.d_primary.data();
        result.view.qdata_bytes = result.d_primary.size();
        result.view.scales = result.d_scales.data();
        result.view.scale_bytes = result.d_scales.size();
        result.view.group = kQ4Group;
        result.view.layout = QuantLayout::Q4N16K16;
        result.view.scale_dtype = DType::FP16;
        result.view.scale_ne[0] = kD / kQ4Group;
        result.view.scale_ne[1] = kProjection;
        result.view.scale_nb[0] = sizeof(std::uint16_t);
        result.view.scale_nb[1] = (kD / kQ4Group) * sizeof(std::uint16_t);
    } else {
        result.bf16.resize(static_cast<std::size_t>(kProjection) * kD);
        for (int row = 0; row < kProjection; ++row) {
            for (int column = 0; column < kD; ++column) {
                result.bf16[static_cast<std::size_t>(row) * kD + column] =
                    hip_bfloat16(patterned(row, column, 41, 512.0F));
            }
        }
        result.d_primary = upload(result.bf16);
        result.view.payload = result.d_primary.data();
        result.view.payload_bytes = result.d_primary.size();
        result.view.qtype = QType::BF16_CTRL;
        result.view.qdata = result.d_primary.data();
        result.view.qdata_bytes = result.d_primary.size();
        result.view.layout = QuantLayout::Contiguous;
    }
    result.view.ndim = 2;
    result.view.n = kProjection;
    result.view.k = kD;
    result.view.shape[0] = kProjection;
    result.view.shape[1] = kD;
    result.view.padded_shape[0] = kProjection;
    result.view.padded_shape[1] = kD;
    return result;
}

double decode_weight(const ProjectionWeight& weight, int row, int column) {
    if (weight.kind == WeightKind::BF16) {
        return static_cast<float>(weight.bf16[static_cast<std::size_t>(row) * kD + column]);
    }
    if (weight.kind == WeightKind::Q4) {
        const std::size_t index = static_cast<std::size_t>(row) * kD + column;
        const unsigned nibble =
            (weight.q4_codes_row[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU;
        const int code = nibble >= 8U ? static_cast<int>(nibble) - 16
                                      : static_cast<int>(nibble);
        const std::uint16_t scale =
            weight.q4_scales_row[static_cast<std::size_t>(row) * (kD / kQ4Group) +
                                 column / kQ4Group];
        return static_cast<double>(code) * half_value(scale);
    }
    const std::int8_t code = weight.codes[static_cast<std::size_t>(row) * kD + column];
    const std::uint16_t scale =
        weight.scales[static_cast<std::size_t>(row) * (kD / kW8Group) + column / kW8Group];
    return static_cast<double>(code) * half_value(scale);
}

std::size_t hidden_index(int channel, int token, int batch, int tokens) {
    return static_cast<std::size_t>(channel) + static_cast<std::size_t>(kD) *
        (static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * batch);
}

std::size_t dynamic_index(int group, int offset, int token, int batch, int tokens) {
    return static_cast<std::size_t>(group) + static_cast<std::size_t>(kGroups) *
        (static_cast<std::size_t>(offset) + 2U *
         (static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * batch));
}

struct Error {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    float milliseconds = 0.0F;
};

Error compare(const std::vector<hip_bfloat16>& actual, const std::vector<double>& expected,
              double absolute_limit, double relative_l2_limit, const char* label) {
    if (actual.size() != expected.size()) fail(std::string(label) + ": extent mismatch");
    double squared_error = 0.0;
    double squared_reference = 0.0;
    Error result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double got = static_cast<float>(actual[index]);
        const double error = got - expected[index];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        squared_error += error * error;
        squared_reference += expected[index] * expected[index];
    }
    result.relative_l2 = std::sqrt(squared_error / std::max(1.0e-30, squared_reference));
    if (result.max_abs > absolute_limit || result.relative_l2 > relative_l2_limit) {
        fail(std::string(label) + ": FP64 oracle mismatch max_abs=" +
             std::to_string(result.max_abs) + " rel_l2=" +
             std::to_string(result.relative_l2));
    }
    return result;
}

Error run_prepare(ninfer::DeviceContext& device, WeightKind kind, int tokens, int batch) {
    ProjectionWeight weight = make_weight(kind);
    const int columns = tokens * batch;
    std::vector<hip_bfloat16> hidden(static_cast<std::size_t>(kD) * columns);
    std::vector<hip_bfloat16> base(static_cast<std::size_t>(kD) * 4);
    for (int column = 0; column < columns; ++column) {
        for (int d = 0; d < kD; ++d) {
            hidden[static_cast<std::size_t>(column) * kD + d] =
                hip_bfloat16(patterned(column, d, 37, 256.0F));
        }
    }
    for (int phase = 0; phase < 2; ++phase) {
        for (int offset = 0; offset < 2; ++offset) {
            for (int d = 0; d < kD; ++d) {
                base[static_cast<std::size_t>(phase * 2 + offset) * kD + d] =
                    hip_bfloat16(patterned(phase * 2 + offset, d, 23, 1024.0F));
            }
        }
    }

    std::vector<double> projection(static_cast<std::size_t>(kProjection) * columns);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < kProjection; ++row) {
            double sum = 0.0;
            for (int d = 0; d < kD; ++d) {
                sum = std::fma(static_cast<double>(static_cast<float>(
                                   hidden[static_cast<std::size_t>(column) * kD + d])),
                               decode_weight(weight, row, d), sum);
            }
            projection[static_cast<std::size_t>(column) * kProjection + row] = sum;
        }
    }
    std::vector<double> expected(static_cast<std::size_t>(kD) * columns);
    std::vector<double> expected_finish(static_cast<std::size_t>(kGroups) * 2 * columns);
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < tokens; ++token) {
            const int column = b * tokens + token;
            for (int group = 0; group < kGroups; ++group) {
                expected_finish[dynamic_index(group, 0, token, b, tokens)] =
                    projection[static_cast<std::size_t>(column) * kProjection + 640 + group];
                expected_finish[dynamic_index(group, 1, token, b, tokens)] =
                    projection[static_cast<std::size_t>(column) * kProjection + 960 + group];
            }
            for (int d = 0; d < kD; ++d) {
                const int group = d / 16;
                const double current = static_cast<float>(hidden[hidden_index(d, token, b, tokens)]);
                const double previous = token == 0
                    ? 0.0
                    : static_cast<float>(hidden[hidden_index(d, token - 1, b, tokens)]);
                const double base0 = static_cast<float>(base[d]);
                const double base1 = static_cast<float>(base[kD + d]);
                const double dynamic0 =
                    projection[static_cast<std::size_t>(column) * kProjection + group];
                const double dynamic1 =
                    projection[static_cast<std::size_t>(column) * kProjection + kGroups + group];
                expected[hidden_index(d, token, b, tokens)] =
                    (base0 + dynamic0) * current + (base1 + dynamic1) * previous;
            }
        }
    }

    auto d_hidden = upload(hidden);
    auto d_base = upload(base);
    DeviceBuffer d_prepared(hidden.size() * sizeof(hip_bfloat16));
    DeviceBuffer d_finish(expected_finish.size() * sizeof(hip_bfloat16));
    const std::size_t workspace_bytes =
        ninfer::ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(
            weight.view.qtype, tokens, tokens, batch);
    WorkspaceArena workspace(workspace_bytes);
    Tensor hidden_tensor(d_hidden.data(), DType::BF16, {kD, tokens, batch});
    Tensor base_tensor(d_base.data(), DType::BF16, {kD, 2, 2});
    Tensor prepared_tensor(d_prepared.data(), DType::BF16, {kD, tokens, batch});
    Tensor finish_tensor(d_finish.data(), DType::BF16, {kGroups, 2, tokens, batch});
    if (kind == WeightKind::Q4) {
        Weight retired = weight.view;
        retired.layout = QuantLayout::RowSplit;
        try {
            ninfer::ops::grouped_dynamic_conv_prepare(hidden_tensor, base_tensor, retired,
                                                       prepared_tensor, finish_tensor, workspace,
                                                       device.stream);
            fail("grouped_dynamic_conv accepted the retired Q4 RowSplit layout");
        } catch (const std::invalid_argument&) {
        }
    }
    ninfer::ops::grouped_dynamic_conv_prepare(hidden_tensor, base_tensor, weight.view,
                                               prepared_tensor, finish_tensor, workspace,
                                               device.stream);
    device.synchronize();
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        fail("grouped_dynamic_conv workspace ownership mismatch");
    }
    std::vector<hip_bfloat16> actual(hidden.size());
    std::vector<hip_bfloat16> actual_finish(expected_finish.size());
    d_prepared.copy_to_host(actual.data(), d_prepared.size());
    d_finish.copy_to_host(actual_finish.data(), d_finish.size());
    Error result = compare(actual, expected, 0.04, 0.02, "prepare");
    (void)compare(actual_finish, expected_finish, 0.04, 0.02, "finish_dynamic");
    constexpr int iterations = 10;
    ninfer::DeviceEventTimer timer(device);
    timer.start();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        ninfer::ops::grouped_dynamic_conv_prepare(hidden_tensor, base_tensor, weight.view,
                                                   prepared_tensor, finish_tensor, workspace,
                                                   device.stream);
    }
    result.milliseconds = timer.stop_ms() / iterations;
    return result;
}

Error run_finish(ninfer::DeviceContext& device) {
    constexpr int tokens = 4;
    constexpr int batch = 2;
    const int columns = tokens * batch;
    std::vector<hip_bfloat16> hidden(static_cast<std::size_t>(kD) * columns);
    std::vector<hip_bfloat16> base(static_cast<std::size_t>(kD) * 4);
    std::vector<hip_bfloat16> dynamic(static_cast<std::size_t>(kGroups) * 2 * columns);
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = hip_bfloat16(patterned(static_cast<int>(i / kD), static_cast<int>(i % kD), 31, 256.0F));
    for (std::size_t i = 0; i < base.size(); ++i)
        base[i] = hip_bfloat16(patterned(static_cast<int>(i / kD), static_cast<int>(i % kD), 19, 512.0F));
    for (std::size_t i = 0; i < dynamic.size(); ++i)
        dynamic[i] = hip_bfloat16(patterned(static_cast<int>(i / kGroups), static_cast<int>(i % kGroups), 17, 256.0F));
    std::vector<double> expected(hidden.size());
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < tokens; ++token) {
            for (int d = 0; d < kD; ++d) {
                const int group = d / 16;
                const double x0 = static_cast<float>(hidden[hidden_index(d, token, b, tokens)]);
                const double x1 = token == 0 ? 0.0 :
                    static_cast<float>(hidden[hidden_index(d, token - 1, b, tokens)]);
                expected[hidden_index(d, token, b, tokens)] =
                    (static_cast<float>(base[2 * kD + d]) +
                     static_cast<float>(dynamic[dynamic_index(group, 0, token, b, tokens)])) * x0 +
                    (static_cast<float>(base[3 * kD + d]) +
                     static_cast<float>(dynamic[dynamic_index(group, 1, token, b, tokens)])) * x1;
            }
        }
    }
    auto d_hidden = upload(hidden);
    auto d_base = upload(base);
    auto d_dynamic = upload(dynamic);
    DeviceBuffer d_output(hidden.size() * sizeof(hip_bfloat16));
    Tensor hidden_tensor(d_hidden.data(), DType::BF16, {kD, tokens, batch});
    Tensor base_tensor(d_base.data(), DType::BF16, {kD, 2, 2});
    Tensor dynamic_tensor(d_dynamic.data(), DType::BF16, {kGroups, 2, tokens, batch});
    Tensor output_tensor(d_output.data(), DType::BF16, {kD, tokens, batch});
    ninfer::ops::grouped_dynamic_conv_finish(hidden_tensor, base_tensor, dynamic_tensor,
                                              output_tensor, device.stream);
    device.synchronize();
    std::vector<hip_bfloat16> actual(hidden.size());
    d_output.copy_to_host(actual.data(), d_output.size());
    return compare(actual, expected, 0.001, 0.005, "finish");
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        const Error bf16 = run_prepare(device, WeightKind::BF16, 1, 1);
        const Error w8 = run_prepare(device, WeightKind::W8, 2, 2);
        const Error q4 = run_prepare(device, WeightKind::Q4, 2, 2);
        const Error finish = run_finish(device);
        std::cout << "grouped_dynamic_conv: PASS gfx1201 D5120/G320 BF16+W8+Q4 T1/B1 "
                     "T2/B2 caller-owned-arena max_abs="
                  << std::max({bf16.max_abs, w8.max_abs, q4.max_abs, finish.max_abs})
                  << " rel_l2=" << std::max({bf16.relative_l2, w8.relative_l2,
                                              q4.relative_l2, finish.relative_l2})
                  << " ms_bf16_t1=" << bf16.milliseconds
                  << " ms_w8_t2b2=" << w8.milliseconds
                  << " ms_q4_t2b2=" << q4.milliseconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "grouped_dynamic_conv: FAIL: " << error.what() << '\n';
        return 1;
    }
}
