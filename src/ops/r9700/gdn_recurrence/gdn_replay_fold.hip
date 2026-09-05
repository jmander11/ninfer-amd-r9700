// Native wave32 implementation of the all-layer ReplaySSM fold boundary.
#include "ops/r9700/gdn_recurrence/gdn_replay_fold_internal.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

constexpr int kStateDim = 128;
constexpr int kWaveSize = 32;
constexpr int kValuesPerWave = 4;
constexpr int kWavesPerBlock = 4;
constexpr int kValuesPerBlock = kValuesPerWave * kWavesPerBlock;
constexpr int kKeysPerLane = kStateDim / kWaveSize;
constexpr float kL2NormEpsilon = 1.0e-6F;

static_assert(kKeysPerLane == 4);
static_assert(kStateDim % kValuesPerBlock == 0);
static_assert(sizeof(GdnReplayFoldKernelRow) == 80);
static_assert(alignof(GdnReplayFoldKernelRow) == 8);
static_assert(sizeof(GdnReplayFoldKernelRows) == 640);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

template <int Layers, int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kLayers = Layers;
    static constexpr int kQkHeads = QkHeads;
    static constexpr int kValueHeads = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(ValueHeads % QkHeads == 0);
    static_assert(ConvChannels % kStateDim == 0);
};

using FoldGeometry48x48 = FoldGeometry<48, 16, 48, 10240>;

__device__ __forceinline__ float wave_sum(float value) {
#pragma unroll
    for (int offset = kWaveSize / 2; offset != 0; offset >>= 1) {
        value += __shfl_down(value, offset, kWaveSize);
    }
    return __shfl(value, 0, kWaveSize);
}

__device__ __forceinline__ void load_state_lane(float (&lane_state)[kValuesPerWave][kKeysPerLane],
                                                 const float* base, int value_base, int key_base) {
#pragma unroll
    for (int value = 0; value < kValuesPerWave; ++value) {
#pragma unroll
        for (int key = 0; key < kKeysPerLane; ++key) {
            lane_state[value][key] =
                base[static_cast<std::int64_t>(value_base + value) * kStateDim + key_base + key];
        }
    }
}

__device__ __forceinline__ void store_state_lane(
    const float (&lane_state)[kValuesPerWave][kKeysPerLane], float* base, int value_base,
    int key_base) {
#pragma unroll
    for (int value = 0; value < kValuesPerWave; ++value) {
#pragma unroll
        for (int key = 0; key < kKeysPerLane; ++key) {
            base[static_cast<std::int64_t>(value_base + value) * kStateDim + key_base + key] =
                lane_state[value][key];
        }
    }
}

__device__ __forceinline__ void load_normalized_key(float (&key)[kKeysPerLane],
                                                     const hip_bfloat16* base, int key_base) {
    float norm2 = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerLane; ++item) {
        key[item] = static_cast<float>(base[key_base + item]);
        norm2 = fmaf(key[item], key[item], norm2);
    }
    const float inverse_norm = rsqrtf(wave_sum(norm2) + kL2NormEpsilon);
#pragma unroll
    for (int item = 0; item < kKeysPerLane; ++item) { key[item] *= inverse_norm; }
}

__device__ __forceinline__ void apply_transition(
    float (&lane_state)[kValuesPerWave][kKeysPerLane], const float (&key)[kKeysPerLane],
    float local_value, float alpha, float beta) {
#pragma unroll
    for (int value = 0; value < kValuesPerWave; ++value) {
        float state_key = 0.0F;
#pragma unroll
        for (int item = 0; item < kKeysPerLane; ++item) {
            state_key = fmaf(lane_state[value][item], key[item], state_key);
        }
        state_key = wave_sum(state_key);
        const float represented_value = __shfl(local_value, value, kWaveSize);
        const float delta = beta * (represented_value - alpha * state_key);
#pragma unroll
        for (int item = 0; item < kKeysPerLane; ++item) {
            lane_state[value][item] =
                fmaf(delta, key[item], alpha * lane_state[value][item]);
        }
    }
}

template <class Geometry>
struct FoldAccess {
    const hip_bfloat16* key_record;
    const hip_bfloat16* value_record;
    const float* gate_record;
    const hip_bfloat16* conv_record;
    float* recurrent_layer0;
    hip_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;

    __device__ __forceinline__ std::int32_t sequence_column(std::int32_t batch,
                                                            std::int32_t step) const {
        const std::int32_t path_length = rows.row[batch].path_length;
        return path_length > 0 ? rows.row[batch].path[step] : step;
    }

    __device__ __forceinline__ std::int32_t active_columns(std::int32_t batch) const {
        const std::int32_t path_length = rows.row[batch].path_length;
        return path_length >= 0 ? path_length : rows.row[batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_outer(std::int32_t layer,
                                                         std::int32_t batch) const {
        return static_cast<std::int64_t>(layer) * record_capacity + batch;
    }

    __device__ __forceinline__ float* state_base(std::int32_t layer, std::int32_t batch,
                                                  std::int32_t value_head) const {
        constexpr std::int64_t kSlotStride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[batch].linear_state_slot) * kSlotStride +
               static_cast<std::int64_t>(value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const hip_bfloat16* key_base(std::int32_t layer,
                                                            std::int32_t batch,
                                                            std::int32_t value_head,
                                                            std::int32_t step) const {
        constexpr std::int32_t kGroup = Geometry::kValueHeads / Geometry::kQkHeads;
        const std::int32_t qk_head = value_head / kGroup;
        const std::int64_t column =
            record_outer(layer, batch) * width + sequence_column(batch, step);
        return key_record + (column * Geometry::kQkHeads + qk_head) * kStateDim;
    }

    __device__ __forceinline__ const hip_bfloat16* value_base(std::int32_t layer,
                                                              std::int32_t batch,
                                                              std::int32_t value_head,
                                                              std::int32_t step) const {
        const std::int64_t column =
            record_outer(layer, batch) * width + sequence_column(batch, step);
        return value_record + (column * Geometry::kValueHeads + value_head) * kStateDim;
    }

    __device__ __forceinline__ void gate(std::int32_t layer, std::int32_t batch,
                                         std::int32_t value_head, std::int32_t step, float& g,
                                         float& beta) const {
        const std::int64_t column =
            record_outer(layer, batch) * width + sequence_column(batch, step);
        const std::int64_t pair = (column * Geometry::kValueHeads + value_head) * 2;
        g = gate_record[pair];
        beta = gate_record[pair + 1];
    }

    __device__ __forceinline__ void publish_conv_history(std::int32_t layer,
                                                         std::int32_t batch,
                                                         std::int32_t value_head,
                                                         std::int32_t state_tile,
                                                         std::int32_t commit) const {
        const std::int32_t channel_block = value_head * 8 + state_tile;
        if (channel_block >= Geometry::kConvChannels / kStateDim) { return; }

        const std::int32_t thread = static_cast<std::int32_t>(threadIdx.y) * kWaveSize +
                                    static_cast<std::int32_t>(threadIdx.x);
        const std::int32_t channel = channel_block * kStateDim + thread;
        hip_bfloat16* history =
            conv_layer0 + static_cast<std::int64_t>(layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[batch].linear_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        const hip_bfloat16* record =
            conv_record + record_outer(layer, batch) * width * Geometry::kConvChannels + channel;

        const std::int32_t i0 = sequence_column(batch, commit <= 2 ? 0 : commit - 3);
        const std::int32_t i1 = sequence_column(batch, commit == 1 ? 0 :
                                                       (commit == 2 ? 0 : commit - 2));
        const std::int32_t i2 = sequence_column(batch, commit == 1 ? 0 :
                                                       (commit == 2 ? 1 : commit - 1));
        hip_bfloat16 h0;
        hip_bfloat16 h1;
        hip_bfloat16 h2;
        if (commit == 1) {
            h0 = history[Geometry::kConvChannels];
            h1 = history[2LL * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(i2) * Geometry::kConvChannels];
        } else if (commit == 2) {
            h0 = history[2LL * Geometry::kConvChannels];
            h1 = record[static_cast<std::int64_t>(i1) * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(i2) * Geometry::kConvChannels];
        } else {
            h0 = record[static_cast<std::int64_t>(i0) * Geometry::kConvChannels];
            h1 = record[static_cast<std::int64_t>(i1) * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(i2) * Geometry::kConvChannels];
        }
        history[0] = h0;
        history[Geometry::kConvChannels] = h1;
        history[2LL * Geometry::kConvChannels] = h2;
    }
};

template <class Geometry>
__global__ __launch_bounds__(kWaveSize * kWavesPerBlock, 2)
void replay_fold_kernel(FoldAccess<Geometry> access) {
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t layer_tile = static_cast<std::int32_t>(blockIdx.z);
    const std::int32_t layer = layer_tile / (kStateDim / kValuesPerBlock);
    const std::int32_t state_tile = layer_tile % (kStateDim / kValuesPerBlock);
    const std::int32_t value_head = static_cast<std::int32_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int value_base = state_tile * kValuesPerBlock + wave * kValuesPerWave;
    const int key_base = lane * kKeysPerLane;
    const std::int32_t columns = access.active_columns(batch);
    if (columns == 0) { return; }

    float* recurrent = access.state_base(layer, batch, value_head);
    float lane_state[kValuesPerWave][kKeysPerLane];
    load_state_lane(lane_state, recurrent, value_base, key_base);

    for (std::int32_t step = 0; step < columns; ++step) {
        float key[kKeysPerLane];
        load_normalized_key(key, access.key_base(layer, batch, value_head, step), key_base);
        float local_value = 0.0F;
        if (lane < kValuesPerWave) {
            local_value = static_cast<float>(
                access.value_base(layer, batch, value_head, step)[value_base + lane]);
        }
        float g;
        float beta;
        access.gate(layer, batch, value_head, step, g, beta);
        apply_transition(lane_state, key, local_value, expf(g), beta);
    }

    store_state_lane(lane_state, recurrent, value_base, key_base);
    access.publish_conv_history(layer, batch, value_head, state_tile, columns);
}

template <class Geometry>
void launch_fixed(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                  const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                  hipStream_t stream) {
    const FoldAccess<Geometry> access{
        static_cast<const hip_bfloat16*>(records.key.data),
        static_cast<const hip_bfloat16*>(records.value.data),
        static_cast<const float*>(records.gate.data),
        static_cast<const hip_bfloat16*>(records.conv.data),
        static_cast<float*>(states.recurrent_layer0.data),
        static_cast<hip_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(float)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(hip_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers *
                                          (kStateDim / kValuesPerBlock)));
    const dim3 block(kWaveSize, kWavesPerBlock, 1);
    hipLaunchKernelGGL((replay_fold_kernel<Geometry>), grid, block, 0, stream, access);
    HIP_CHECK(hipGetLastError());
}

} // namespace

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        hipStream_t stream) {
    if (records.spec.layers == FoldGeometry48x48::kLayers &&
        records.spec.qk_heads == FoldGeometry48x48::kQkHeads &&
        records.spec.value_heads == FoldGeometry48x48::kValueHeads &&
        records.spec.conv_channels == FoldGeometry48x48::kConvChannels) {
        launch_fixed<FoldGeometry48x48>(records, states, rows, active_rows, stream);
        return;
    }
    throw std::invalid_argument("GDN replay fold launcher received an unregistered geometry");
}

} // namespace ninfer::ops::detail::gated_delta_net
