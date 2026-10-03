#pragma once

// GDN verification recurrence with replay records (ops::gated_delta_net_replay_record), shared by
// its launcher (gated_delta_net.hip) and the persistent decode kernel.
#include "core/cache_warm.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn_recurrence/gdn_verify_profile.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::recurrence {

inline constexpr int kDim                  = 128;
inline constexpr int kQHeads               = 16;
inline constexpr int kValueHeads           = 48;
inline constexpr int kGroup                = kValueHeads / kQHeads;
inline constexpr int kThreadsPerRow        = 8;
inline constexpr int kBlock                = kDim * kThreadsPerRow;
inline constexpr int kOrdinaryRowTiles     = 4;
inline constexpr int kOrdinaryRowsPerBlock = kDim / kOrdinaryRowTiles;
inline constexpr int kOrdinaryBlock        = kOrdinaryRowsPerBlock * kThreadsPerRow;
inline constexpr int kKeysPerThread        = kDim / kThreadsPerRow;
inline constexpr int kMaximumVerifyWidth   = 16;
inline constexpr int kMaximumPrefillWidth  = 262144;
inline constexpr int kMaximumBatch         = static_cast<int>(kMaximumConcurrency);
inline constexpr float kEpsilon            = 1.0e-6F;

__device__ __forceinline__ float group_sum(float value) {
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

__device__ __forceinline__ float wave_sum(float value) {
    value += __shfl_xor(value, 16, 32);
    value += __shfl_xor(value, 8, 32);
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

__device__ __forceinline__ void load_state_row(const float* state, std::size_t base, int value_row,
                                               int sublane, float (&local)[kKeysPerThread]) {
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        local[item] = state[base + static_cast<std::size_t>(value_row) * kDim + sublane +
                            item * kThreadsPerRow];
    }
}

__device__ __forceinline__ void store_state_row(float* state, std::size_t base, int value_row,
                                                int sublane, const float (&local)[kKeysPerThread]) {
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        state[base + static_cast<std::size_t>(value_row) * kDim + sublane + item * kThreadsPerRow] =
            local[item];
    }
}

__device__ __forceinline__ void transition(float (&local)[kKeysPerThread], const float* staged_q,
                                           const float* staged_k, float value, float alpha,
                                           float beta, float scale, int sublane,
                                           hip_bfloat16* output) {
    float state_key = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        const int key = sublane + item * kThreadsPerRow;
        state_key     = fmaf(local[item], staged_k[key], state_key);
    }
    state_key         = group_sum(state_key);
    const float delta = beta * (value - alpha * state_key);
    float state_query = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        const int key = sublane + item * kThreadsPerRow;
        local[item]   = fmaf(delta, staged_k[key], alpha * local[item]);
        state_query   = fmaf(local[item], staged_q[key], state_query);
    }
    state_query = group_sum(state_query);
    if (sublane == 0) { *output = hip_bfloat16(state_query * scale); }
}

// Normalizes one token's q/k head with a single wave into caller LDS rows. The FP32 arithmetic
// is exactly the block routine's: stage_qk<true>'s stride-64,32,16,... tree (lane items
// {i, i+64} and {i+32, i+96} pair first, then the stride-32 add and the in-wave strides), or
// stage_qk_ordinary's lane FMA chain for the wave-QK profile. The replay fold keeps that tree.
__device__ __forceinline__ void stage_token_qk_wave(const hip_bfloat16* q, const hip_bfloat16* k,
                                                    std::size_t base, float* staged_q,
                                                    float* staged_k, int lane) {
    float qv[4], kv[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        qv[item] = static_cast<float>(q[base + lane + item * 32]);
        kv[item] = static_cast<float>(k[base + lane + item * 32]);
    }
    float q_scale = 0.0F, k_scale = 0.0F;
    if constexpr (r9700::gdn_recurrence::kVerifyWaveQkCandidate) {
        float q_sum = 0.0F, k_sum = 0.0F;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            q_sum = fmaf(qv[item], qv[item], q_sum);
            k_sum = fmaf(kv[item], kv[item], k_sum);
        }
        q_scale = rsqrtf(wave_sum(q_sum) + kEpsilon);
        k_scale = rsqrtf(wave_sum(k_sum) + kEpsilon);
    } else {
        float qs[4], ks[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            qs[item] = __fmul_rn(qv[item], qv[item]);
            ks[item] = __fmul_rn(kv[item], kv[item]);
            // HIP rounded intrinsics alone still permit contraction here.
            asm volatile("" : "+v"(qs[item]), "+v"(ks[item]));
        }
        float qsum = __fadd_rn(__fadd_rn(qs[0], qs[2]), __fadd_rn(qs[1], qs[3]));
        float ksum = __fadd_rn(__fadd_rn(ks[0], ks[2]), __fadd_rn(ks[1], ks[3]));
#pragma unroll
        for (int stride = 16; stride != 0; stride >>= 1) {
            qsum = __fadd_rn(qsum, __shfl_down(qsum, stride, 32));
            ksum = __fadd_rn(ksum, __shfl_down(ksum, stride, 32));
        }
        q_scale = rsqrtf(__shfl(qsum, 0, 32) + kEpsilon);
        k_scale = rsqrtf(__shfl(ksum, 0, 32) + kEpsilon);
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        staged_q[lane + item * 32] = qv[item] * q_scale;
        staged_k[lane + item * 32] = kv[item] * k_scale;
    }
}

// Value rows evolve independently, so each (batch, head) is split into kOrdinaryRowTiles CTAs.
// Every token's normalized q/k, controls and this tile's values are staged in LDS up front (one
// wave per token), leaving the sequential recurrence with LDS reads and lane shuffles only. Row
// tile zero alone publishes the replay records.
struct RecordShared {
    float staged_q[kMaximumVerifyWidth][kDim];
    float staged_k[kMaximumVerifyWidth][kDim];
    float staged_v[kMaximumVerifyWidth][kOrdinaryRowsPerBlock];
    float controls[kMaximumVerifyWidth][2];
};

template <bool Tree, class Cta>
__device__ __forceinline__ void
record_body(const Cta& cta, RecordShared& shared, const hip_bfloat16* q, const hip_bfloat16* k,
            const hip_bfloat16* v, const float* g, const float* beta, const float* states,
            const std::int32_t* valid_columns, const std::int32_t* initial_slots,
            const std::int32_t* parent_index, float* tree_states, hip_bfloat16* key_record,
            hip_bfloat16* value_record, float* gate_record, hip_bfloat16* output, int width,
            int slots, float scale, const CacheWarm& warm, unsigned work_ctas) {
    const std::uint32_t block = cta.block().x;
    if (block >= work_ctas) {
        warm_cache(warm, block - work_ctas, cta.grid().x - work_ctas, cta.thread(), cta.threads());
        return;
    }
    constexpr int kWaves                = kOrdinaryBlock / 32;
    const int combined                  = static_cast<int>(block) / kOrdinaryRowTiles;
    const int row_tile                  = static_cast<int>(block) % kOrdinaryRowTiles;
    const int batch                     = combined / kValueHeads;
    const int value_head                = combined % kValueHeads;
    const int tid                       = static_cast<int>(cta.thread());
    const int lane                      = tid % 32;
    const int wave                      = tid / 32;
    const int tile_row                  = tid / kThreadsPerRow;
    const int value_row                 = row_tile * kOrdinaryRowsPerBlock + tile_row;
    const int sublane                   = tid % kThreadsPerRow;
    const int q_head                    = value_head / kGroup;
    const int valid                     = valid_columns == nullptr ? width : valid_columns[batch];
    const int initial                   = initial_slots[batch];
    const std::size_t state_head        = static_cast<std::size_t>(value_head) * kDim * kDim;
    const std::size_t slot_stride       = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::size_t tree_token_stride = slot_stride;
    const std::size_t tree_batch_stride = static_cast<std::size_t>(width) * slot_stride;
    auto& staged_q                      = shared.staged_q;
    auto& staged_k                      = shared.staged_k;
    auto& staged_v                      = shared.staged_v;
    auto& controls                      = shared.controls;
    float local[kKeysPerThread];
    if (initial < 0 || initial >= slots || valid < 1 || valid > width) { return; }
    load_state_row(states, static_cast<std::size_t>(initial) * slot_stride + state_head, value_row,
                   sublane, local);
    for (int token = wave; token < valid; token += kWaves) {
        const std::size_t column  = static_cast<std::size_t>(batch) * width + token;
        const std::size_t qk_base = (column * kQHeads + q_head) * kDim;
        const std::size_t vh      = column * kValueHeads + value_head;
        stage_token_qk_wave(q, k, qk_base, staged_q[token], staged_k[token], lane);
        staged_v[token][lane] =
            static_cast<float>(v[vh * kDim + row_tile * kOrdinaryRowsPerBlock + lane]);
        if (lane == 0) {
            controls[token][0] = expf(g[vh]);
            controls[token][1] = beta[vh];
        }
        if (row_tile == 0) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index                 = lane + item * 32;
                value_record[vh * kDim + index] = v[vh * kDim + index];
                if (value_head % kGroup == 0) key_record[qk_base + index] = k[qk_base + index];
            }
            if (lane == 0) {
                gate_record[vh * 2]     = g[vh];
                gate_record[vh * 2 + 1] = beta[vh];
            }
        }
    }
    cta.sync();
    for (int token = 0; token < valid; ++token) {
        const std::size_t column = static_cast<std::size_t>(batch) * width + token;
        const std::size_t vh     = column * kValueHeads + value_head;
        if constexpr (Tree) {
            const int parent = parent_index[column];
            if ((token == 0 && parent != -1) || (token > 0 && (parent < 0 || parent >= token))) {
                if (sublane == 0) { output[vh * kDim + value_row] = hip_bfloat16(NAN); }
                return;
            }
            if (parent < 0) {
                load_state_row(states, static_cast<std::size_t>(initial) * slot_stride + state_head,
                               value_row, sublane, local);
            } else {
                const std::size_t parent_base =
                    static_cast<std::size_t>(batch) * tree_batch_stride +
                    static_cast<std::size_t>(parent) * tree_token_stride + state_head;
                load_state_row(tree_states, parent_base, value_row, sublane, local);
            }
        }
        transition(local, staged_q[token], staged_k[token], staged_v[token][tile_row],
                   controls[token][0], controls[token][1], scale, sublane,
                   output + vh * kDim + value_row);
        if constexpr (Tree) {
            const std::size_t child_base = static_cast<std::size_t>(batch) * tree_batch_stride +
                                           static_cast<std::size_t>(token) * tree_token_stride +
                                           state_head;
            // A child reloads exactly the elements this thread stored for its parent.
            store_state_row(tree_states, child_base, value_row, sublane, local);
        }
    }
    for (int token = valid; token < width; ++token) {
        if (sublane == 0) {
            const std::size_t vh =
                (static_cast<std::size_t>(batch) * width + token) * kValueHeads + value_head;
            output[vh * kDim + value_row] = hip_bfloat16(0.0F);
        }
    }
}

// The ordinary record (no tree) with one CTA of Threads threads per (batch, head) instead of
// kOrdinaryRowTiles: each token is staged once, and every 8-thread row slot carries value rows
// slot, slot + Threads / 8, ... through the recurrence, interleaving their independent chains.
// Each row's arithmetic is record_body's. Grid: work_ctas / kOrdinaryRowTiles CTAs.
template <int Threads, int MaxWidth>
struct RecordHeadShared {
    float staged_q[MaxWidth][kDim];
    float staged_k[MaxWidth][kDim];
    float staged_v[MaxWidth][kDim];
    float controls[MaxWidth][2];
};

template <int Threads, int MaxWidth, class Cta>
__device__ __forceinline__ void
record_head_body(const Cta& cta, RecordHeadShared<Threads, MaxWidth>& shared, const hip_bfloat16* q,
                 const hip_bfloat16* k, const hip_bfloat16* v, const float* g, const float* beta,
                 const float* states, const std::int32_t* valid_columns,
                 const std::int32_t* initial_slots, const std::int32_t* /*parent_index*/,
                 float* /*tree_states*/, hip_bfloat16* key_record, hip_bfloat16* value_record,
                 float* gate_record, hip_bfloat16* output, int width, int slots, float scale,
                 const CacheWarm& /*warm*/, unsigned /*work_ctas*/) {
    static_assert(Threads % 32 == 0 && MaxWidth >= 1 && MaxWidth <= Threads / 32);
    constexpr int kSlots     = Threads / kThreadsPerRow;
    constexpr int kRowPasses = (kDim + kSlots - 1) / kSlots;
    const int combined       = static_cast<int>(cta.block().x);
    const int batch          = combined / kValueHeads;
    const int value_head     = combined % kValueHeads;
    const int tid            = static_cast<int>(cta.thread());
    const int lane           = tid % 32;
    const int wave           = tid / 32;
    const int slot           = tid / kThreadsPerRow;
    const int sublane        = tid % kThreadsPerRow;
    const int q_head         = value_head / kGroup;
    const int valid          = valid_columns == nullptr ? width : valid_columns[batch];
    const int initial        = initial_slots[batch];
    if (initial < 0 || initial >= slots || valid < 1 || valid > width || width > MaxWidth) {
        return;
    }
    const std::size_t state_head  = static_cast<std::size_t>(value_head) * kDim * kDim;
    const std::size_t slot_stride = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const auto row_of             = [&](int pass) { return slot + pass * kSlots; };
    float local[kRowPasses][kKeysPerThread];
#pragma unroll
    for (int pass = 0; pass < kRowPasses; ++pass)
        if (row_of(pass) < kDim)
            load_state_row(states, static_cast<std::size_t>(initial) * slot_stride + state_head,
                           row_of(pass), sublane, local[pass]);
    // Wave w stages token w (MaxWidth <= waves).
    if (wave < valid) {
        const int token           = wave;
        const std::size_t column  = static_cast<std::size_t>(batch) * width + token;
        const std::size_t qk_base = (column * kQHeads + q_head) * kDim;
        const std::size_t vh      = column * kValueHeads + value_head;
        stage_token_qk_wave(q, k, qk_base, shared.staged_q[token], shared.staged_k[token], lane);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int index                 = lane + item * 32;
            const hip_bfloat16 value        = v[vh * kDim + index];
            shared.staged_v[token][index]   = static_cast<float>(value);
            value_record[vh * kDim + index] = value;
            if (value_head % kGroup == 0) key_record[qk_base + index] = k[qk_base + index];
        }
        if (lane == 0) {
            shared.controls[token][0] = expf(g[vh]);
            shared.controls[token][1] = beta[vh];
            gate_record[vh * 2]       = g[vh];
            gate_record[vh * 2 + 1]   = beta[vh];
        }
    }
    cta.sync();
    for (int token = 0; token < valid; ++token) {
        const std::size_t vh =
            (static_cast<std::size_t>(batch) * width + token) * kValueHeads + value_head;
#pragma unroll
        for (int pass = 0; pass < kRowPasses; ++pass) {
            const int row = row_of(pass);
            if (row < kDim)
                transition(local[pass], shared.staged_q[token], shared.staged_k[token],
                           shared.staged_v[token][row], shared.controls[token][0],
                           shared.controls[token][1], scale, sublane, output + vh * kDim + row);
        }
    }
    for (int token = valid; token < width; ++token) {
        const std::size_t vh =
            (static_cast<std::size_t>(batch) * width + token) * kValueHeads + value_head;
#pragma unroll
        for (int pass = 0; pass < kRowPasses; ++pass)
            if (row_of(pass) < kDim && sublane == 0)
                output[vh * kDim + row_of(pass)] = hip_bfloat16(0.0F);
    }
}

template <bool Tree>
__global__ __launch_bounds__(kOrdinaryBlock, 1) void record_kernel(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v, const float* g,
    const float* beta, const float* states, const std::int32_t* valid_columns,
    const std::int32_t* initial_slots, const std::int32_t* parent_index, float* tree_states,
    hip_bfloat16* key_record, hip_bfloat16* value_record, float* gate_record, hip_bfloat16* output,
    int width, int slots, float scale, CacheWarm warm, unsigned work_ctas) {
    __shared__ RecordShared shared;
    record_body<Tree>(r9700::persistent::LaunchCta{}, shared, q, k, v, g, beta, states,
                      valid_columns, initial_slots, parent_index, tree_states, key_record,
                      value_record, gate_record, output, width, slots, scale, warm, work_ctas);
}

} // namespace ninfer::ops::detail::gated_delta_net::recurrence
