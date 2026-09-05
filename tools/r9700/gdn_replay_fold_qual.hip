#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ninfer/ops/gdn_replay.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kLayers = 48;
constexpr int kCapacity = 2;
constexpr int kWidth = 4;
constexpr int kConvChannels = 10240;
constexpr int kQkHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kStateDim = 128;
constexpr int kSlots = 2;
constexpr int kOracleLayer = 17;

double g_max_state_abs = 0.0;
double g_max_state_relative_l2 = 0.0;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (std::string(properties.gcnArchName).starts_with("gfx1201") &&
            properties.warpSize == 32) {
            if (selected >= 0) { fail("multiple wave32 gfx1201 devices found"); }
            selected = ordinal;
        }
    }
    if (selected < 0) { fail("no wave32 gfx1201 device found"); }
    return selected;
}

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

std::uint16_t bf16_bits(float value) {
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
    result.data = bf16_bits(value);
    return result;
}

float fp32(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

float pattern(std::uint32_t seed, float scale) {
    const int centered = static_cast<int>(mix(seed) % 2001U) - 1000;
    return static_cast<float>(centered) * scale / 1000.0F;
}

std::size_t record_column(int layer, int row, int token) {
    return (static_cast<std::size_t>(layer) * kCapacity + row) * kWidth + token;
}

std::size_t key_index(int layer, int row, int token, int head, int dim) {
    return (record_column(layer, row, token) * kQkHeads + head) * kStateDim + dim;
}

std::size_t value_index(int layer, int row, int token, int head, int dim) {
    return (record_column(layer, row, token) * kValueHeads + head) * kStateDim + dim;
}

std::size_t gate_index(int layer, int row, int token, int head, int component) {
    return (record_column(layer, row, token) * kValueHeads + head) * 2 + component;
}

std::size_t conv_index(int layer, int row, int token, int channel) {
    return record_column(layer, row, token) * kConvChannels + channel;
}

std::vector<float> initial_state(int row) {
    std::vector<float> state(static_cast<std::size_t>(kValueHeads) * kStateDim * kStateDim);
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] = pattern(0x10007U + static_cast<std::uint32_t>(row) * 131U +
                                   static_cast<std::uint32_t>(index),
                               0.006F);
    }
    return state;
}

std::vector<hip_bfloat16> initial_conv(int row) {
    std::vector<hip_bfloat16> state(static_cast<std::size_t>(3) * kConvChannels);
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] = bf16(pattern(0x30011U + static_cast<std::uint32_t>(row) * 149U +
                                        static_cast<std::uint32_t>(index),
                                    0.05F));
    }
    return state;
}

struct Fixture {
    std::vector<hip_bfloat16> conv;
    std::vector<hip_bfloat16> key;
    std::vector<hip_bfloat16> value;
    std::vector<float> gate;
};

Fixture make_fixture(const ninfer::GdnReplayRecords& records) {
    Fixture fixture{
        std::vector<hip_bfloat16>(static_cast<std::size_t>(records.conv.numel()), bf16(0.0F)),
        std::vector<hip_bfloat16>(static_cast<std::size_t>(records.key.numel()), bf16(0.0F)),
        std::vector<hip_bfloat16>(static_cast<std::size_t>(records.value.numel()), bf16(0.0F)),
        std::vector<float>(static_cast<std::size_t>(records.gate.numel()), 0.0F),
    };
    for (int row = 0; row < kCapacity; ++row) {
        for (int token = 0; token < kWidth; ++token) {
            for (int channel = 0; channel < kConvChannels; ++channel) {
                fixture.conv[conv_index(kOracleLayer, row, token, channel)] =
                    bf16(pattern(0x50021U + row * 181U + token * 17U + channel, 0.08F));
            }
            for (int head = 0; head < kQkHeads; ++head) {
                for (int dim = 0; dim < kStateDim; ++dim) {
                    fixture.key[key_index(kOracleLayer, row, token, head, dim)] =
                        bf16(pattern(0x70001U + row * 193U + token * 19U + head * 7U + dim,
                                     0.09F));
                }
            }
            for (int head = 0; head < kValueHeads; ++head) {
                for (int dim = 0; dim < kStateDim; ++dim) {
                    fixture.value[value_index(kOracleLayer, row, token, head, dim)] =
                        bf16(pattern(0x90031U + row * 211U + token * 23U + head * 11U + dim,
                                     0.07F));
                }
                fixture.gate[gate_index(kOracleLayer, row, token, head, 0)] =
                    -0.03F - static_cast<float>(mix(0xb001U + row * 31U + token * 13U + head) %
                                                700U) /
                                 1000.0F;
                fixture.gate[gate_index(kOracleLayer, row, token, head, 1)] =
                    0.15F + static_cast<float>(mix(0xd007U + row * 37U + token * 17U + head) %
                                               700U) /
                                1000.0F;
            }
        }
    }
    return fixture;
}

std::vector<double> recurrent_oracle(const Fixture& fixture, int row,
                                     const std::vector<int>& sequence,
                                     const std::vector<float>& initial) {
    std::vector<double> state(initial.begin(), initial.end());
    std::array<double, static_cast<std::size_t>(kQkHeads) * kStateDim> normalized{};
    for (int token : sequence) {
        for (int qk_head = 0; qk_head < kQkHeads; ++qk_head) {
            double norm2 = 0.0;
            for (int dim = 0; dim < kStateDim; ++dim) {
                const double value = fp32(fixture.key[key_index(kOracleLayer, row, token,
                                                                qk_head, dim)]);
                normalized[static_cast<std::size_t>(qk_head) * kStateDim + dim] = value;
                norm2 += value * value;
            }
            const double inverse = 1.0 / std::sqrt(norm2 + 1.0e-6);
            for (int dim = 0; dim < kStateDim; ++dim) {
                normalized[static_cast<std::size_t>(qk_head) * kStateDim + dim] *= inverse;
            }
        }
        for (int value_head = 0; value_head < kValueHeads; ++value_head) {
            const int qk_head = value_head / 3;
            const double alpha = std::exp(static_cast<double>(
                fixture.gate[gate_index(kOracleLayer, row, token, value_head, 0)]));
            const double beta =
                fixture.gate[gate_index(kOracleLayer, row, token, value_head, 1)];
            for (int value_dim = 0; value_dim < kStateDim; ++value_dim) {
                const std::size_t state_base =
                    (static_cast<std::size_t>(value_head) * kStateDim + value_dim) * kStateDim;
                double state_key = 0.0;
                for (int key_dim = 0; key_dim < kStateDim; ++key_dim) {
                    state_key += state[state_base + key_dim] *
                                 normalized[static_cast<std::size_t>(qk_head) * kStateDim +
                                            key_dim];
                }
                const double represented_value = fp32(
                    fixture.value[value_index(kOracleLayer, row, token, value_head, value_dim)]);
                const double delta = beta * (represented_value - alpha * state_key);
                for (int key_dim = 0; key_dim < kStateDim; ++key_dim) {
                    double& element = state[state_base + key_dim];
                    element = alpha * element +
                              delta * normalized[static_cast<std::size_t>(qk_head) * kStateDim +
                                                 key_dim];
                }
            }
        }
    }
    return state;
}

std::vector<hip_bfloat16> conv_oracle(const Fixture& fixture, int row,
                                      const std::vector<int>& sequence,
                                      std::vector<hip_bfloat16> state) {
    for (int token : sequence) {
        for (int channel = 0; channel < kConvChannels; ++channel) {
            state[channel] = state[kConvChannels + channel];
            state[kConvChannels + channel] = state[2 * kConvChannels + channel];
            state[2 * kConvChannels + channel] =
                fixture.conv[conv_index(kOracleLayer, row, token, channel)];
        }
    }
    return state;
}

void require_state_close(const std::vector<float>& actual, const std::vector<double>& expected,
                         const std::string& label) {
    if (actual.size() != expected.size()) { fail(label + ": size mismatch"); }
    double maximum = 0.0;
    double error2 = 0.0;
    double oracle2 = 0.0;
    std::size_t maximum_index = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = static_cast<double>(actual[index]) - expected[index];
        if (std::abs(error) > maximum) {
            maximum = std::abs(error);
            maximum_index = index;
        }
        error2 += error * error;
        oracle2 += expected[index] * expected[index];
    }
    const double relative = std::sqrt(error2 / std::max(oracle2, 1.0e-30));
    g_max_state_abs = std::max(g_max_state_abs, maximum);
    g_max_state_relative_l2 = std::max(g_max_state_relative_l2, relative);
    if (maximum > 1.0e-5 || relative > 3.0e-3) {
        fail(label + ": max_abs=" + std::to_string(maximum) +
             " rel_l2=" + std::to_string(relative) +
             " index=" + std::to_string(maximum_index));
    }
}

void require_bf16_exact(const std::vector<hip_bfloat16>& actual,
                        const std::vector<hip_bfloat16>& expected, const std::string& label) {
    if (actual.size() != expected.size()) { fail(label + ": size mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].data != expected[index].data) {
            fail(label + ": bit mismatch at " + std::to_string(index));
        }
    }
}

template <class T>
std::vector<T> copy_tensor(const ninfer::Tensor& tensor) {
    std::vector<T> result(static_cast<std::size_t>(tensor.numel()));
    HIP_CHECK(hipMemcpy(result.data(), tensor.data, result.size() * sizeof(T),
                        hipMemcpyDeviceToHost));
    return result;
}

template <class Callable>
void require_invalid(Callable&& callable, const std::string& label) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    }
    fail(label + ": expected invalid_argument");
}

void qualify(hipStream_t stream) {
    ninfer::LayoutBuilder record_builder;
    const ninfer::GdnReplayRecordLayout record_layout = ninfer::plan_gdn_replay_records(
        record_builder,
        {.layers = kLayers,
         .record_capacity = kCapacity,
         .width = kWidth,
         .conv_channels = kConvChannels,
         .qk_heads = kQkHeads,
         .value_heads = kValueHeads,
         .key_dim = kStateDim,
         .value_dim = kStateDim});
    ninfer::DeviceBuffer record_storage(record_builder.finish(256));
    ninfer::GdnReplayRecords records({record_storage.data(), record_storage.size()}, record_layout);
    const Fixture fixture = make_fixture(records);
    HIP_CHECK(hipMemcpy(records.conv.data, fixture.conv.data(), records.conv.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.key.data, fixture.key.data(), records.key.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.value.data, fixture.value.data(), records.value.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.gate.data, fixture.gate.data(), records.gate.bytes(),
                        hipMemcpyHostToDevice));

    ninfer::LayoutBuilder state_builder;
    const ninfer::LinearAttentionStatePoolLayout state_layout =
        ninfer::plan_linear_attention_state_pool(
            state_builder,
            {.layers = kLayers,
             .conv_channels = kConvChannels,
             .conv_width = 3,
             .value_heads = kValueHeads,
             .value_head_dim = kStateDim,
             .key_head_dim = kStateDim,
             .slot_count = kSlots,
             .conv_dtype = ninfer::DType::BF16});
    ninfer::DeviceBuffer state_storage(state_builder.finish(256));
    state_storage.fill(0);
    ninfer::LinearAttentionStatePool states({state_storage.data(), state_storage.size()},
                                             state_layout);

    const std::array<std::vector<float>, kCapacity> initial_recurrent{initial_state(0),
                                                                       initial_state(1)};
    const std::array<std::vector<hip_bfloat16>, kCapacity> initial_history{initial_conv(0),
                                                                            initial_conv(1)};
    // Physical row zero deliberately updates slot one; row one updates slot zero. This proves
    // record-row identity is not inferred from the destination state slot.
    const std::array<int, kCapacity> slots{1, 0};
    for (int row = 0; row < kCapacity; ++row) {
        HIP_CHECK(hipMemcpyAsync(states.recurrent_slot(kOracleLayer, slots[row]).data,
                                 initial_recurrent[row].data(),
                                 initial_recurrent[row].size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(states.conv_slot(kOracleLayer, slots[row]).data,
                                 initial_history[row].data(),
                                 initial_history[row].size() * sizeof(hip_bfloat16),
                                 hipMemcpyHostToDevice, stream));
    }

    ninfer::ops::GdnReplayFoldRow prefix{.linear_state_slot = slots[0],
                                         .commit_columns = kWidth};
    ninfer::ops::GdnReplayFoldRow path{};
    path.linear_state_slot = slots[1];
    path.commit_columns = 1; // Ignored in path mode.
    path.path = {3, 1, 2};
    path.path_length = 3;
    const std::array rows{prefix, path};
    ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), rows, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    const std::array<std::vector<int>, kCapacity> sequences{std::vector<int>{0, 1, 2, 3},
                                                             std::vector<int>{3, 1, 2}};
    for (int row = 0; row < kCapacity; ++row) {
        const std::vector<float> actual_state =
            copy_tensor<float>(states.recurrent_slot(kOracleLayer, slots[row]));
        require_state_close(actual_state,
                            recurrent_oracle(fixture, row, sequences[row], initial_recurrent[row]),
                            "row " + std::to_string(row) + " independent FP64 recurrence");
        const std::vector<hip_bfloat16> actual_conv =
            copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, slots[row]));
        require_bf16_exact(actual_conv,
                           conv_oracle(fixture, row, sequences[row], initial_history[row]),
                           "row " + std::to_string(row) + " convolution tail");
    }

    // A layer with all-zero represented records and initial state remains exact zero, proving the
    // launch covers the all-layer domain without bleeding the oracle layer's record outer index.
    const auto untouched_layer = copy_tensor<float>(states.recurrent_slot(kOracleLayer + 1, 0));
    if (std::any_of(untouched_layer.begin(), untouched_layer.end(),
                    [](float value) { return std::bit_cast<std::uint32_t>(value) != 0U; })) {
        fail("record outer index bled into a neighboring layer");
    }

    const auto state_before_noop0 = copy_tensor<float>(states.recurrent_slot(kOracleLayer, 0));
    const auto state_before_noop1 = copy_tensor<float>(states.recurrent_slot(kOracleLayer, 1));
    const auto conv_before_noop0 = copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 0));
    const auto conv_before_noop1 = copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 1));
    ninfer::ops::GdnReplayFoldRow zero_prefix{.linear_state_slot = 0, .commit_columns = 0};
    ninfer::ops::GdnReplayFoldRow zero_path{.linear_state_slot = 1, .commit_columns = kWidth};
    zero_path.path_length = 0;
    const std::array noops{zero_prefix, zero_path};
    ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), noops, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (copy_tensor<float>(states.recurrent_slot(kOracleLayer, 0)) != state_before_noop0 ||
        copy_tensor<float>(states.recurrent_slot(kOracleLayer, 1)) != state_before_noop1) {
        fail("zero-extent replay changed recurrent state");
    }
    require_bf16_exact(copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 0)),
                       conv_before_noop0, "zero-prefix convolution no-op");
    require_bf16_exact(copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 1)),
                       conv_before_noop1, "zero-path convolution no-op");

    std::array duplicate{prefix, path};
    duplicate[1].linear_state_slot = duplicate[0].linear_state_slot;
    require_invalid(
        [&] { ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), duplicate, stream); },
        "duplicate destination slots");
    std::array invalid_path{path};
    invalid_path[0].path[0] = kWidth;
    require_invalid(
        [&] { ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), invalid_path, stream); },
        "out-of-range path index");

    require_bf16_exact(copy_tensor<hip_bfloat16>(records.conv), fixture.conv,
                       "read-only convolution records");
    require_bf16_exact(copy_tensor<hip_bfloat16>(records.key), fixture.key,
                       "read-only key records");
    require_bf16_exact(copy_tensor<hip_bfloat16>(records.value), fixture.value,
                       "read-only value records");
    if (copy_tensor<float>(records.gate) != fixture.gate) {
        fail("read-only gate records changed");
    }
}

} // namespace

int main() {
    try {
        HIP_CHECK(hipSetDevice(select_r9700()));
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        try {
            qualify(stream);
            HIP_CHECK(hipStreamDestroy(stream));
        } catch (...) {
            (void)hipStreamDestroy(stream);
            throw;
        }
        std::cout << "r9700_gdn_replay_fold: PASS gfx1201 wave32 Qwen3.8 L48/Hq16/Hv48 "
                     "B=2 W=4 prefix/path/no-op/read-only max_state_abs="
                  << g_max_state_abs << " max_state_rel_l2=" << g_max_state_relative_l2 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700_gdn_replay_fold: FAIL " << error.what() << '\n';
        return 1;
    }
}
