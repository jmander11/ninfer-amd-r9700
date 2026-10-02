#pragma once

// Device-side ReplaySSM fold of one (layer, row, value head, 16-row state tile) item, shared by the
// all-layer fold kernel, the per-layer deferred fold kernel and the GDN front kernel that folds a
// layer ahead of its verification round (ops/gdn_replay.h). One item is 128 threads; every caller
// runs the same instruction sequence per item, so the folded state and convolution history are
// bitwise those of the all-layer fold.
#include <hip/hip_bfloat16.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::fold {

constexpr int kStateDim = 128;
constexpr int kWaveSize = 32;
constexpr int kItemThreads = 128;
constexpr int kThreadsPerValue = 8;
constexpr int kValuesPerItem = kItemThreads / kThreadsPerValue;
constexpr int kKeysPerLane = kStateDim / kThreadsPerValue;
constexpr int kStateTiles = kStateDim / kValuesPerItem;
constexpr float kL2NormEpsilon = 1.0e-6F;
// The sole Qwen3.8-27B geometry.
constexpr int kQkHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kConvChannels = 10240;
constexpr int kItemsPerLayerRow = kValueHeads * kStateTiles;

static_assert(kKeysPerLane == 16 && kStateTiles == 8);

// Record planes (outer index layer * record_capacity + record row, `width` columns each).
struct Planes {
    const hip_bfloat16* key   = nullptr;
    const hip_bfloat16* value = nullptr;
    const float* gate         = nullptr;
    const hip_bfloat16* conv  = nullptr;
    std::int32_t record_capacity = 0;
    std::int32_t width           = 0;
};

// One row's fold: its record row, absolute state slot and committed columns, in packed order or
// through `path` when path_length > 0.
struct Row {
    std::int32_t record_row  = 0;
    std::int32_t slot        = 0;
    std::int32_t columns     = 0;
    std::int32_t path_length = -1;
    const std::int32_t* path = nullptr;
};

#if defined(__HIPCC__)
__device__ __forceinline__ std::int32_t sequence_column(const Row& row, std::int32_t step) {
    return row.path_length > 0 ? row.path[step] : step;
}

__device__ __forceinline__ std::int64_t record_column(const Planes& planes, std::int32_t layer,
                                                      const Row& row, std::int32_t step) {
    return (static_cast<std::int64_t>(layer) * planes.record_capacity + row.record_row) *
               planes.width +
           sequence_column(row, step);
}

__device__ __forceinline__ float group_sum(float value) {
    value += __shfl_xor(value, 4, kWaveSize);
    value += __shfl_xor(value, 2, kWaveSize);
    value += __shfl_xor(value, 1, kWaveSize);
    return value;
}

// Convolution history of the item's 128 channels: tail_3(old history || committed conv records).
__device__ __forceinline__ void publish_conv_history(const Planes& planes, std::int32_t layer,
                                                     const Row& row, hip_bfloat16* conv_layer,
                                                     std::int32_t value_head,
                                                     std::int32_t state_tile, int tid) {
    const std::int32_t channel_block = value_head * kStateTiles + state_tile;
    if (channel_block >= kConvChannels / kStateDim) return;
    const std::int32_t commit = row.columns;
    const std::int32_t channel = channel_block * kStateDim + tid;
    hip_bfloat16* history =
        conv_layer + static_cast<std::int64_t>(row.slot) * (3LL * kConvChannels) + channel;
    const hip_bfloat16* record =
        planes.conv +
        (static_cast<std::int64_t>(layer) * planes.record_capacity + row.record_row) *
            planes.width * kConvChannels +
        channel;
    const std::int32_t i0 = sequence_column(row, commit <= 2 ? 0 : commit - 3);
    const std::int32_t i1 = sequence_column(row, commit == 1 ? 0 : (commit == 2 ? 0 : commit - 2));
    const std::int32_t i2 = sequence_column(row, commit == 1 ? 0 : (commit == 2 ? 1 : commit - 1));
    hip_bfloat16 h0;
    hip_bfloat16 h1;
    hip_bfloat16 h2;
    if (commit == 1) {
        h0 = history[kConvChannels];
        h1 = history[2LL * kConvChannels];
        h2 = record[static_cast<std::int64_t>(i2) * kConvChannels];
    } else if (commit == 2) {
        h0 = history[2LL * kConvChannels];
        h1 = record[static_cast<std::int64_t>(i1) * kConvChannels];
        h2 = record[static_cast<std::int64_t>(i2) * kConvChannels];
    } else {
        h0 = record[static_cast<std::int64_t>(i0) * kConvChannels];
        h1 = record[static_cast<std::int64_t>(i1) * kConvChannels];
        h2 = record[static_cast<std::int64_t>(i2) * kConvChannels];
    }
    history[0] = h0;
    history[kConvChannels] = h1;
    history[2LL * kConvChannels] = h2;
}

// Folds one item (thread `tid` of 128) of a row with row.columns > 0: state tiles
// [state_tile, state_tile + Tiles) of one value head, which share its normalized keys. `key` and
// `reduction` are this item's 128-float LDS arrays; `sync` is the CTA barrier. An inactive item
// (a CTA's spare 128-thread slot) touches no memory but executes every barrier of the same row.
// Every state element and history channel takes the same arithmetic for any Tiles.
template <int Tiles = 1, class Sync>
__device__ __forceinline__ void fold_item(const Planes& planes, std::int32_t layer, const Row& row,
                                          float* recurrent_layer, hip_bfloat16* conv_layer,
                                          std::int32_t value_head, std::int32_t state_tile,
                                          int tid, bool active, float* key, float* reduction,
                                          Sync&& sync) {
    constexpr std::int32_t kGroup = kValueHeads / kQkHeads;
    const int lane = tid % kWaveSize;
    const int sublane = lane % kThreadsPerValue;
    const auto value_row = [&](int tile) {
        return (state_tile + tile) * kValuesPerItem + tid / kThreadsPerValue;
    };
    constexpr std::int64_t kSlotStride = static_cast<std::int64_t>(kValueHeads) * kStateDim * kStateDim;
    float* recurrent = recurrent_layer + static_cast<std::int64_t>(row.slot) * kSlotStride +
                       static_cast<std::int64_t>(value_head) * kStateDim * kStateDim;
    float lane_state[Tiles][kKeysPerLane];
    if (active) {
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile)
#pragma unroll
            for (int item = 0; item < kKeysPerLane; ++item)
                lane_state[tile][item] =
                    recurrent[value_row(tile) * kStateDim + sublane + item * kThreadsPerValue];
    }
    for (std::int32_t step = 0; step < row.columns; ++step) {
        // Match snapshot/record's represented-key normalization exactly. The item's 128 threads
        // cooperate, independently of batch or path.
        const std::int64_t column = record_column(planes, layer, row, step);
        if (active) {
            key[tid] = static_cast<float>(
                planes.key[(column * kQkHeads + value_head / kGroup) * kStateDim + tid]);
            reduction[tid] = key[tid] * key[tid];
        }
        sync();
        for (int stride = kStateDim / 2; stride != 0; stride >>= 1) {
            if (active && tid < stride) reduction[tid] += reduction[tid + stride];
            sync();
        }
        if (active) key[tid] *= rsqrtf(reduction[0] + kL2NormEpsilon);
        sync();
        if (active) {
            const std::int64_t pair = (column * kValueHeads + value_head) * 2;
            const float g = planes.gate[pair];
            const float beta = planes.gate[pair + 1];
            const float alpha = expf(g);
#pragma unroll
            for (int tile = 0; tile < Tiles; ++tile) {
                const float value = static_cast<float>(
                    planes.value[(column * kValueHeads + value_head) * kStateDim + value_row(tile)]);
                float state_key = 0.0F;
#pragma unroll
                for (int item = 0; item < kKeysPerLane; ++item)
                    state_key = fmaf(lane_state[tile][item], key[sublane + item * kThreadsPerValue],
                                     state_key);
                state_key = group_sum(state_key);
                const float delta = beta * (value - alpha * state_key);
#pragma unroll
                for (int item = 0; item < kKeysPerLane; ++item)
                    lane_state[tile][item] = fmaf(delta, key[sublane + item * kThreadsPerValue],
                                                  alpha * lane_state[tile][item]);
            }
        }
        sync();
    }
    if (!active) return;
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
        for (int item = 0; item < kKeysPerLane; ++item)
            recurrent[value_row(tile) * kStateDim + sublane + item * kThreadsPerValue] =
                lane_state[tile][item];
        publish_conv_history(planes, layer, row, conv_layer, value_head, state_tile + tile, tid);
    }
}

#endif

// Deferred fold rows as the device reads them (ops::GdnDeferredFoldRows layout).
struct DeferredRow {
    std::int32_t slot;
    std::int32_t columns;
    std::int32_t record_row;
    std::int32_t reserved;
};

// One layer's deferred fold for `batch` device rows.
struct LayerArgs {
    Planes planes{};
    std::int32_t layer = 0;
    float* recurrent_layer = nullptr;
    hip_bfloat16* conv_layer = nullptr;
    const DeferredRow* rows = nullptr;
    std::int32_t batch = 0;
};

// CTAs of `threads` threads folding a layer with `tiles` state tiles per 128-thread item: per
// row, ceil(items / (threads / 128)) CTAs.
constexpr std::uint32_t layer_ctas_per_row(std::uint32_t threads, std::uint32_t tiles = 1U) {
    const std::uint32_t items_per_cta = threads / kItemThreads;
    const std::uint32_t items = kItemsPerLayerRow / tiles;
    return (items + items_per_cta - 1U) / items_per_cta;
}

#if defined(__HIPCC__)
// CTA `cta` of the layer fold grid (rows-major) run on CTA context `context` of `Threads`
// threads (persistent_cta.h), LDS key/reduction arrays of Threads/128 items.
template <std::uint32_t Threads, int Tiles = 1, class Context>
__device__ __forceinline__ void fold_layer_cta(const Context& context, const LayerArgs& args,
                                               std::uint32_t cta,
                                               float (&key)[Threads / kItemThreads][kStateDim],
                                               float (&reduction)[Threads / kItemThreads][kStateDim]) {
    static_assert(kStateTiles % Tiles == 0);
    constexpr std::uint32_t kItems = Threads / kItemThreads;
    constexpr std::uint32_t kCtasPerRow = layer_ctas_per_row(Threads, Tiles);
    constexpr std::uint32_t kRowItems = kItemsPerLayerRow / Tiles;
    constexpr std::int32_t kHeadItems = kStateTiles / Tiles;
    const std::uint32_t batch = cta / kCtasPerRow;
    const DeferredRow source = args.rows[batch];
    if (source.columns <= 0) return;  // CTA-uniform
    const Row row{.record_row = source.record_row, .slot = source.slot, .columns = source.columns};
    const std::uint32_t sub = context.thread() / kItemThreads;
    const std::uint32_t item = (cta % kCtasPerRow) * kItems + sub;
    const bool active = item < kRowItems;
    const std::int32_t value_head = active ? static_cast<std::int32_t>(item) / kHeadItems : 0;
    const std::int32_t state_tile =
        active ? static_cast<std::int32_t>(item) % kHeadItems * Tiles : 0;
    fold_item<Tiles>(args.planes, args.layer, row, args.recurrent_layer, args.conv_layer,
                     value_head, state_tile, static_cast<int>(context.thread() % kItemThreads),
                     active, key[sub], reduction[sub], [&] { context.sync(); });
}

#endif

} // namespace ninfer::ops::detail::gated_delta_net::fold
