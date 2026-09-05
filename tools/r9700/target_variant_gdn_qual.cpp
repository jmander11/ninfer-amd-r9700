#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::DeviceBuffer;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::WorkspaceArena;
using Variant = ninfer::targets::qwen3_8_27b::detail::Variant;

static_assert(Variant::gdn_input_projection_prefill_p2048_selected(
    ninfer::targets::qwen3::TextPhase::Prefill, 2048));
static_assert(!Variant::gdn_input_projection_prefill_p2048_selected(
    ninfer::targets::qwen3::TextPhase::Prefill, 1024));
static_assert(!Variant::gdn_input_projection_prefill_p2048_selected(
    ninfer::targets::qwen3::TextPhase::Prefill, 4096));
static_assert(!Variant::gdn_input_projection_prefill_p2048_selected(
    ninfer::targets::qwen3::TextPhase::Verify, 2048));
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows = 2048;
constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kChannels = kQueryRows + kKeyRows + kValueRows;
constexpr std::int32_t kValueZRows = 2 * kValueRows;
constexpr std::int32_t kGroup = 32;
constexpr std::uint16_t kScaleBits = 0x3000U; // IEEE FP16 0.125

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::uint16_t bf16_rne_bits(float value) {
    std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t absolute = word & 0x7fffffffU;
    if ((absolute & 0x7f800000U) != 0x7f800000U) {
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

float bf16_float(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

std::pair<int, float> represented_a8g32(const std::vector<hip_bfloat16>& input,
                                       std::size_t token, std::int32_t column) {
    const std::int32_t group_base = column / kGroup * kGroup;
    float maximum = 0.0F;
    for (std::int32_t lane = 0; lane < kGroup; ++lane) {
        maximum = std::max(maximum, std::abs(bf16_float(
            input[token * kHidden + static_cast<std::size_t>(group_base + lane)])));
    }
    if (maximum == 0.0F) return {0, 0.0F};
    _Float16 half_scale = static_cast<_Float16>(maximum / 127.0F);
    std::uint16_t scale_bits = std::bit_cast<std::uint16_t>(half_scale);
    if (scale_bits == 0U) scale_bits = 1U;
    const float scale = static_cast<float>(std::bit_cast<_Float16>(scale_bits));
    const float value = bf16_float(input[token * kHidden + static_cast<std::size_t>(column)]);
    const int code = std::clamp(static_cast<int>(std::nearbyint(value / scale)), -127, 127);
    return {code, scale};
}

template <typename T>
void upload(DeviceBuffer& buffer, const std::vector<T>& source) {
    if (buffer.size() != source.size() * sizeof(T)) { fail("upload extent mismatch"); }
    buffer.copy_from_host(source.data(), buffer.size());
}

template <typename T>
std::vector<T> download(const DeviceBuffer& buffer) {
    if (buffer.size() % sizeof(T) != 0) { fail("download extent mismatch"); }
    std::vector<T> result(buffer.size() / sizeof(T));
    buffer.copy_to_host(result.data(), buffer.size());
    return result;
}

struct SparseW8 {
    SparseW8(std::int32_t rows_in, std::int32_t columns_in, std::int32_t tokens_in,
             std::uint32_t seed)
        : rows(rows_in), columns(columns_in), tokens(tokens_in),
          codes(static_cast<std::size_t>(rows) * columns),
          scales(static_cast<std::size_t>(rows) * (columns / kGroup) * sizeof(std::uint16_t)),
          selected_column(static_cast<std::size_t>(rows)),
          selected_code(static_cast<std::size_t>(rows)) {
        if (rows <= 0 || columns <= 0 || tokens <= 0 || columns % 128 != 0) {
            fail("invalid sparse W8 fixture geometry");
        }
        std::vector<std::int8_t> host_codes(static_cast<std::size_t>(rows) * columns, 0);
        std::vector<std::uint16_t> host_scales(static_cast<std::size_t>(rows) *
                                                   (columns / kGroup),
                                               0);
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t column =
                static_cast<std::int32_t>((static_cast<std::uint64_t>(row) * 37U + seed * 19U) %
                                          static_cast<std::uint32_t>(columns));
            const std::int8_t code = static_cast<std::int8_t>(
                (row & 1) == 0 ? 1 + row % 7 : -(1 + row % 7));
            selected_column[static_cast<std::size_t>(row)] = column;
            selected_code[static_cast<std::size_t>(row)] = code;
            host_codes[static_cast<std::size_t>(row) * columns + column] = code;
            host_scales[static_cast<std::size_t>(row) * (columns / kGroup) + column / kGroup] =
                kScaleBits;
        }
        upload(codes, host_codes);
        upload(scales, host_scales);

        view.qtype = QType::W8G32_F16S;
        view.layout = QuantLayout::RowSplit;
        view.scale_dtype = DType::FP16;
        view.group = kGroup;
        view.group_size = kGroup;
        view.ndim = 2;
        view.n = rows;
        view.k = columns;
        view.shape[0] = rows;
        view.shape[1] = columns;
        view.padded_shape[0] = rows;
        view.padded_shape[1] = columns;
        view.qdata = codes.data();
        view.qdata_bytes = codes.size();
        view.scales = scales.data();
        view.scale_bytes = scales.size();
        view.payload = codes.data();
        view.payload_bytes = codes.size() + scales.size();
    }

    hip_bfloat16 project(const std::vector<hip_bfloat16>& input, std::size_t column,
                         std::int32_t row) const {
        const std::int32_t input_row = selected_column[static_cast<std::size_t>(row)];
        const float code = static_cast<float>(selected_code[static_cast<std::size_t>(row)]);
        if (ninfer::ops::r9700::linear::use_a8w8(static_cast<std::uint32_t>(tokens),
                                                 static_cast<std::uint32_t>(rows),
                                                 static_cast<std::uint32_t>(columns))) {
            const auto [activation_code, activation_scale] =
                represented_a8g32(input, column, input_row);
            return bf16(std::fma(static_cast<float>(activation_code) * code,
                                 activation_scale * 0.125F, 0.0F));
        }
        const float source = bf16_float(input[column * kHidden + input_row]);
        return bf16(std::fma(source, code * 0.125F, 0.0F));
    }

    std::int32_t rows;
    std::int32_t columns;
    std::int32_t tokens;
    DeviceBuffer codes;
    DeviceBuffer scales;
    std::vector<std::int32_t> selected_column;
    std::vector<std::int8_t> selected_code;
    Weight view{};
};

void require_exact(const std::vector<hip_bfloat16>& actual,
                   const std::vector<hip_bfloat16>& expected, const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].data != expected[index].data) {
            std::ostringstream message;
            message << label << " mismatch at " << index << " actual=0x" << std::hex
                    << actual[index].data << " expected=0x" << expected[index].data;
            fail(message.str());
        }
    }
}

void require_close(const std::vector<hip_bfloat16>& actual, const std::vector<double>& expected,
                   const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double observed = bf16_float(actual[index]);
        const double absolute = std::abs(observed - expected[index]);
        const double relative = absolute / std::max(1.0e-6, std::abs(expected[index]));
        if (absolute > 3.0e-2 && relative > 3.0e-2) {
            std::ostringstream message;
            message << label << " mismatch at " << index << " actual=" << observed
                    << " expected=" << expected[index];
            fail(message.str());
        }
    }
}

struct Outputs {
    explicit Outputs(std::int32_t width, std::int32_t batch)
        : query(static_cast<std::size_t>(kQueryRows) * width * batch * sizeof(hip_bfloat16)),
          key(static_cast<std::size_t>(kKeyRows) * width * batch * sizeof(hip_bfloat16)),
          value(static_cast<std::size_t>(kValueRows) * width * batch * sizeof(hip_bfloat16)),
          gate(static_cast<std::size_t>(kValueRows) * width * batch * sizeof(hip_bfloat16)),
          query_view(query.data(), DType::BF16, {kQueryRows, width, batch}),
          key_view(key.data(), DType::BF16, {kKeyRows, width, batch}),
          value_view(value.data(), DType::BF16, {kValueRows, width, batch}),
          gate_view(gate.data(), DType::BF16, {kValueRows, width, batch}) {}

    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer gate;
    Tensor query_view;
    Tensor key_view;
    Tensor value_view;
    Tensor gate_view;
};

struct OracleOutputs {
    std::vector<double> query;
    std::vector<double> key;
    std::vector<double> value;
    std::vector<hip_bfloat16> gate;
};

OracleOutputs oracle_conv(const std::vector<hip_bfloat16>& input, const SparseW8& query_key,
                          const SparseW8& value_z,
                          const std::vector<hip_bfloat16>& conv_weight,
                          const std::vector<hip_bfloat16>& state,
                          const std::vector<std::int32_t>& valid,
                          const std::vector<std::int32_t>& initial_slots,
                          const std::vector<std::int32_t>* parents, std::int32_t width,
                          std::int32_t batch, std::vector<hip_bfloat16>* snapshots,
                          const std::vector<std::int32_t>* snapshot_bases,
                          std::vector<hip_bfloat16>* record) {
    const std::size_t columns = static_cast<std::size_t>(width) * batch;
    OracleOutputs output{
        std::vector<double>(columns * kQueryRows, 0.0),
        std::vector<double>(columns * kKeyRows, 0.0),
        std::vector<double>(columns * kValueRows, 0.0),
        std::vector<hip_bfloat16>(columns * kValueRows),
    };
    if (snapshots != nullptr) { *snapshots = state; }
    if (record != nullptr) { record->assign(columns * kChannels, bf16(0.0F)); }

    const auto projected = [&](std::size_t column, std::int32_t channel) {
        return channel < kQueryRows + kKeyRows
            ? query_key.project(input, column, channel)
            : value_z.project(input, column, channel - kQueryRows - kKeyRows);
    };
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t channel = 0; channel < kChannels; ++channel) {
            const std::size_t state_base =
                static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(b)]) * 3U *
                kChannels;
            const float checkpoint0 = bf16_float(state[state_base + channel]);
            const float checkpoint1 = bf16_float(state[state_base + kChannels + channel]);
            const float checkpoint2 =
                bf16_float(state[state_base + 2U * kChannels + channel]);
            std::vector<float> saved0(static_cast<std::size_t>(width));
            std::vector<float> saved1(static_cast<std::size_t>(width));
            std::vector<float> saved2(static_cast<std::size_t>(width));
            float sequential0 = checkpoint0;
            float sequential1 = checkpoint1;
            float sequential2 = checkpoint2;
            for (std::int32_t token = 0; token < width; ++token) {
                const std::size_t column = static_cast<std::size_t>(b) * width + token;
                if (channel < kValueRows) {
                    output.gate[column * kValueRows + channel] =
                        value_z.project(input, column, kValueRows + channel);
                }
                if (token >= valid[static_cast<std::size_t>(b)]) { continue; }
                float h0 = sequential0;
                float h1 = sequential1;
                float h2 = sequential2;
                if (parents != nullptr) {
                    const std::int32_t parent = (*parents)[column];
                    h0 = parent < 0 ? checkpoint0 : saved0[static_cast<std::size_t>(parent)];
                    h1 = parent < 0 ? checkpoint1 : saved1[static_cast<std::size_t>(parent)];
                    h2 = parent < 0 ? checkpoint2 : saved2[static_cast<std::size_t>(parent)];
                }
                const hip_bfloat16 p = projected(column, channel);
                if (record != nullptr) { (*record)[column * kChannels + channel] = p; }
                const float current = bf16_float(p);
                float sum = std::fma(bf16_float(conv_weight[channel]), h0, 0.0F);
                sum = std::fma(bf16_float(conv_weight[kChannels + channel]), h1, sum);
                sum = std::fma(bf16_float(conv_weight[2 * kChannels + channel]), h2, sum);
                sum = std::fma(bf16_float(conv_weight[3 * kChannels + channel]), current,
                               sum);
                const double result = static_cast<double>(sum) /
                    (1.0 + std::exp(-static_cast<double>(sum)));
                if (channel < kQueryRows) {
                    output.query[column * kQueryRows + channel] = result;
                } else if (channel < kQueryRows + kKeyRows) {
                    output.key[column * kKeyRows + channel - kQueryRows] = result;
                } else {
                    output.value[column * kValueRows + channel - kQueryRows - kKeyRows] = result;
                }
                saved0[static_cast<std::size_t>(token)] = h1;
                saved1[static_cast<std::size_t>(token)] = h2;
                saved2[static_cast<std::size_t>(token)] = current;
                sequential0 = h1;
                sequential1 = h2;
                sequential2 = current;
                if (snapshots != nullptr && snapshot_bases != nullptr) {
                    const std::size_t destination =
                        (static_cast<std::size_t>((*snapshot_bases)[static_cast<std::size_t>(b)]) +
                         token) * 3U * kChannels;
                    (*snapshots)[destination + channel] = bf16(h1);
                    (*snapshots)[destination + kChannels + channel] = bf16(h2);
                    (*snapshots)[destination + 2U * kChannels + channel] = bf16(current);
                }
            }
        }
    }
    return output;
}

void compare_outputs(const Outputs& actual, const OracleOutputs& expected, const char* prefix) {
    require_close(download<hip_bfloat16>(actual.query), expected.query,
                  (std::string(prefix) + " query").c_str());
    require_close(download<hip_bfloat16>(actual.key), expected.key,
                  (std::string(prefix) + " key").c_str());
    require_close(download<hip_bfloat16>(actual.value), expected.value,
                  (std::string(prefix) + " value").c_str());
    require_exact(download<hip_bfloat16>(actual.gate), expected.gate,
                  (std::string(prefix) + " gate").c_str());
}

void qualify(hipStream_t stream) {
    constexpr std::int32_t width = 4;
    constexpr std::int32_t batch = 2;
    constexpr std::int32_t tokens = width * batch;
    constexpr std::int32_t state_slots = 16;
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * kHidden);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const auto wave = static_cast<std::int32_t>((index * 31U + 17U) % 97U) - 48;
        input[index] = bf16(static_cast<float>(wave == 0 ? 1 : wave) / 64.0F);
    }
    SparseW8 query_key(kQueryRows + kKeyRows, kHidden, tokens, 3U);
    SparseW8 value_z(kValueZRows, kHidden, tokens, 7U);
    Variant::GdnProjectionWeights weights{};
    weights.input_projection.query_key = query_key.view;
    weights.input_projection.value_z = value_z.view;

    std::vector<hip_bfloat16> conv_weight(static_cast<std::size_t>(kChannels) * 4U);
    for (std::int32_t channel = 0; channel < kChannels; ++channel) {
        conv_weight[channel] = bf16(0.25F);
        conv_weight[kChannels + channel] = bf16(-0.125F);
        conv_weight[2 * kChannels + channel] = bf16(0.0625F);
        conv_weight[3 * kChannels + channel] = bf16(0.5F);
    }
    std::vector<hip_bfloat16> state(static_cast<std::size_t>(state_slots) * 3U * kChannels);
    for (std::size_t index = 0; index < state.size(); ++index) {
        const auto wave = static_cast<std::int32_t>((index * 13U + 5U) % 37U) - 18;
        state[index] = bf16(static_cast<float>(wave) / 64.0F);
    }
    const std::vector<std::int32_t> valid{4, 2};
    const std::vector<std::int32_t> initial{0, 1};
    const std::vector<std::int32_t> bases{4, 8};
    const std::vector<std::int32_t> parents{-1, 0, 0, 2, -1, 0, 0, 0};

    DeviceBuffer d_input(input.size() * sizeof(hip_bfloat16));
    DeviceBuffer d_conv_weight(conv_weight.size() * sizeof(hip_bfloat16));
    DeviceBuffer d_state(state.size() * sizeof(hip_bfloat16));
    DeviceBuffer d_valid(valid.size() * sizeof(std::int32_t));
    DeviceBuffer d_initial(initial.size() * sizeof(std::int32_t));
    DeviceBuffer d_bases(bases.size() * sizeof(std::int32_t));
    DeviceBuffer d_parents(parents.size() * sizeof(std::int32_t));
    upload(d_input, input);
    upload(d_conv_weight, conv_weight);
    upload(d_valid, valid);
    upload(d_initial, initial);
    upload(d_bases, bases);
    upload(d_parents, parents);
    Tensor hidden(d_input.data(), DType::BF16, {kHidden, width, batch});
    Tensor conv(d_conv_weight.data(), DType::BF16, {kChannels, 4});
    Tensor states(d_state.data(), DType::BF16, {kChannels, 3, state_slots});
    Tensor valid_view(d_valid.data(), DType::I32, {batch});
    Tensor initial_view(d_initial.data(), DType::I32, {batch});
    Tensor bases_view(d_bases.data(), DType::I32, {batch});
    Tensor parent_view(d_parents.data(), DType::I32, {width, batch});

    Outputs snapshot(width, batch);
    std::vector<hip_bfloat16> expected_snapshots;
    const OracleOutputs snapshot_oracle =
        oracle_conv(input, query_key, value_z, conv_weight, state, valid, initial, nullptr, width,
                    batch, &expected_snapshots, &bases, nullptr);
    upload(d_state, state);
    const std::size_t fallback_linear_bytes = ninfer::ops::linear_workspace_capacity_bytes(
        QType::W8G32_F16S, tokens, kHidden);
    const std::size_t snapshot_leaf_bytes =
        Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
        Variant::WeightsProfile::R9700W8G32Candidate,
        ninfer::targets::qwen3::TextPhase::Verify, batch, width, width);
    // Production supplies this activation storage through ExecutionState. The direct leaf
    // qualifier intentionally exercises the null-execution fallback, so its local arena owns it.
    const std::size_t snapshot_bytes = snapshot_leaf_bytes + fallback_linear_bytes;
    WorkspaceArena snapshot_workspace(snapshot_bytes);
    Variant::gdn_input_projection_snapshot(
        hidden, weights, conv, states, valid_view, initial_view, bases_view, snapshot.query_view,
        snapshot.key_view, snapshot.value_view, snapshot.gate_view,
        ninfer::targets::qwen3::TextPhase::Verify, snapshot_workspace, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    compare_outputs(snapshot, snapshot_oracle, "Variant snapshot");
    require_exact(download<hip_bfloat16>(d_state), expected_snapshots,
                  "Variant snapshot state publication");
    if (snapshot_workspace.peak_used() != snapshot_bytes) {
        fail("Variant snapshot workspace query does not match observed peak");
    }

    Outputs record_outputs(width, batch);
    DeviceBuffer d_record(static_cast<std::size_t>(tokens) * kChannels * sizeof(hip_bfloat16));
    Tensor record(d_record.data(), DType::BF16, {kChannels, width, batch});
    std::vector<hip_bfloat16> expected_record;
    const OracleOutputs record_oracle =
        oracle_conv(input, query_key, value_z, conv_weight, state, valid, initial, &parents, width,
                    batch, nullptr, nullptr, &expected_record);
    upload(d_state, state);
    const std::size_t record_leaf_bytes =
        Variant::gdn_input_projection_record_workspace_capacity_bytes(
        Variant::WeightsProfile::R9700W8G32Candidate,
        ninfer::targets::qwen3::TextPhase::Verify, batch, width, width);
    const std::size_t record_bytes = record_leaf_bytes + fallback_linear_bytes;
    WorkspaceArena record_workspace(record_bytes);
    Variant::gdn_input_projection_record(
        hidden, weights, conv, states, valid_view, initial_view, record,
        record_outputs.query_view, record_outputs.key_view, record_outputs.value_view,
        record_outputs.gate_view, ninfer::targets::qwen3::TextPhase::Verify, record_workspace,
        stream, &parent_view);
    HIP_CHECK(hipStreamSynchronize(stream));
    compare_outputs(record_outputs, record_oracle, "Variant record");
    require_exact(download<hip_bfloat16>(d_record), expected_record,
                  "Variant represented projection record");
    require_exact(download<hip_bfloat16>(d_state), state,
                  "Variant replay checkpoint immutability");
    if (record_workspace.peak_used() != record_bytes) {
        fail("Variant record workspace query does not match observed peak");
    }

    bool rejected = false;
    try {
        (void)Variant::gdn_input_projection_record_workspace_capacity_bytes(
            Variant::WeightsProfile::R9700W8G32Candidate,
            ninfer::targets::qwen3::TextPhase::Verify, batch, 1, 1);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { fail("Variant record workspace admitted width one"); }
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        qualify(device.stream);
        std::cout << "r9700_target_variant_gdn: PASS W8G32 B=2 W=4 snapshot/tree-record\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700_target_variant_gdn: FAIL: " << error.what() << '\n';
        return 1;
    }
}
