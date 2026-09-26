#include "ops/r9700/dflash/swa_launch.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim = 128;
constexpr int kQHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kGroup = 4;
constexpr int kWave = 32;
constexpr int kBlock = kGroup * kWave;

__device__ __forceinline__ float wave_sum(float value) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, kWave);
    }
    return __shfl(value, 0, kWave);
}

__device__ __forceinline__ void consume_key(const float (&query)[4], const hip_bfloat16* key,
                                            const hip_bfloat16* value, float scale, float& maximum,
                                            float& denominator, float (&accumulator)[4]) {
    const int lane = static_cast<int>(threadIdx.x) & (kWave - 1);
    float dot = 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        dot = fmaf(query[item], static_cast<float>(key[lane + item * kWave]), dot);
    }
    const float score = wave_sum(dot) * scale;
    const float next_maximum = fmaxf(maximum, score);
    const float alpha = isinf(maximum) ? 0.0F : expf(maximum - next_maximum);
    const float probability = expf(score - next_maximum);
    denominator = denominator * alpha + probability;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        accumulator[item] = accumulator[item] * alpha +
                            probability * static_cast<float>(value[lane + item * kWave]);
    }
    maximum = next_maximum;
}

__global__ __launch_bounds__(kBlock, 2) void swa_direct_kernel(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ query_k,
    const hip_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const hip_bfloat16* __restrict__ context_k,
    const hip_bfloat16* __restrict__ context_v, hip_bfloat16* __restrict__ output, int tokens,
    int batch_size, int padded_capacity, int window, int lane_capacity, int minimum_context,
    int maximum_context, float scale) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int wave = static_cast<int>(threadIdx.x) / kWave;
    const int lane = static_cast<int>(threadIdx.x) & (kWave - 1);
    const int q_head = kv_head * kGroup + wave;
    const std::int64_t q_base = static_cast<std::int64_t>(kHeadDim) *
        (q_head + kQHeads * (token + tokens * batch));

    const int valid = valid_columns[batch];
    const int cache_lane = lanes[batch];
    const int context_length = positions[tokens * batch];
    const int query_position = positions[token + tokens * batch];
    const bool active = token < valid && valid >= 0 && valid <= tokens && cache_lane >= 0 &&
                        cache_lane < lane_capacity && context_length >= minimum_context &&
                        context_length <= maximum_context && query_position >= context_length;
    if (!active) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            output[q_base + lane + item * kWave] = hip_bfloat16(0.0F);
        }
        return;
    }

    float query[4];
    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        query[item] = static_cast<float>(q[q_base + lane + item * kWave]);
    }
    float maximum = -INFINITY;
    float denominator = 0.0F;
    const int context_begin = max(0, query_position - (window - 1));
    const std::int64_t context_lane_base = static_cast<std::int64_t>(kHeadDim) *
        padded_capacity * kKvHeads * cache_lane;
    for (int position = context_begin; position < context_length; ++position) {
        const int slot = position & (window - 1);
        const std::int64_t key_base = context_lane_base + static_cast<std::int64_t>(kHeadDim) *
            (slot + padded_capacity * kv_head);
        consume_key(query, context_k + key_base, context_v + key_base, scale, maximum,
                    denominator, accumulator);
    }
    for (int key_token = 0; key_token < valid; ++key_token) {
        const std::int64_t key_base = static_cast<std::int64_t>(kHeadDim) *
            (kv_head + kKvHeads * (key_token + tokens * batch));
        consume_key(query, query_k + key_base, query_v + key_base, scale, maximum, denominator,
                    accumulator);
    }
    const float inverse = denominator > 0.0F ? 1.0F / denominator : 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        output[q_base + lane + item * kWave] = hip_bfloat16(accumulator[item] * inverse);
    }
}

__device__ __forceinline__ std::int64_t partial_index(int d, int q_head, int token, int split,
                                                       int batch, int tokens, int splits) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kHeadDim) *
        (q_head + kQHeads * (token + tokens * (split + splits * batch)));
}

__device__ __forceinline__ std::int64_t statistic_index(int q_head, int token, int split,
                                                         int batch, int tokens, int splits) {
    return q_head + static_cast<std::int64_t>(kQHeads) *
        (token + tokens * (split + splits * batch));
}

// Split-context partials with BF16 WMMA. One CTA per (KV head, split, batch) and one wave per
// sixteen (token, query head) rows (row r: token r / 4, query head 4 * kv_head + r % 4), so every
// context K/V block is read once per KV head instead of once per token and query head. For each
// 16-key block, S^T = K Q^T (eight BF16 WMMAs over the head dimension, keys straight from the
// cyclic cache) leaves each lane's eight scores of its query row in the A layout of P; the online
// Softmax runs in the log2 domain and O += P V (BF16 probabilities, BF16 V staged transposed in
// LDS, FP32 accumulation). Each row masks keys outside its own sliding window; the CTA covers the
// union of the rows' windows split evenly, and the last split also consumes the query block's own
// keys. Partials keep the reduce kernel's layout.
using SwaBf16x8 = short __attribute__((ext_vector_type(8)));
using SwaF32x8 = float __attribute__((ext_vector_type(8)));
constexpr int kSwaWmmaWaves = 4;
constexpr int kSwaWmmaThreads = kSwaWmmaWaves * kWave;
constexpr int kSwaValuePitch = 16 + 8;

__device__ __forceinline__ std::uint16_t swa_bf16_bits(float value) {
    return static_cast<hip_bfloat16>(value).data;
}

__global__ __launch_bounds__(kSwaWmmaThreads) void swa_partial_kernel(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ query_k,
    const hip_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const hip_bfloat16* __restrict__ context_k,
    const hip_bfloat16* __restrict__ context_v, hip_bfloat16* __restrict__ partial_acc,
    float* __restrict__ partial_m, float* __restrict__ partial_l, int tokens, int padded_capacity,
    int window, int lane_capacity, int minimum_context, int maximum_context, int splits,
    float scale) {
    __shared__ std::uint16_t staged_values[kHeadDim * kSwaValuePitch];
    const int kv_head = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / kWave, lane = tid % kWave;
    const int axis = lane % 16, half = lane / 16;
    const int valid = valid_columns[batch];
    const int cache_lane = lanes[batch];
    const int context_length = positions[tokens * batch];
    const bool sequence_active = valid >= 0 && valid <= tokens && cache_lane >= 0 &&
                                 cache_lane < lane_capacity &&
                                 context_length >= minimum_context &&
                                 context_length <= maximum_context;
    // This lane's query row (the B/P column): token and query head.
    const int row = wave * 16 + axis;
    const int token = row / kGroup;
    const int q_head = kv_head * kGroup + row % kGroup;
    const bool row_exists = token < tokens;
    const int query_position = row_exists ? positions[token + tokens * batch] : 0;
    const bool row_active = sequence_active && row_exists && token < valid &&
                            query_position >= context_length;
    const int row_begin = max(0, query_position - (window - 1));
    // The CTA's key range: the union of the active rows' windows, split evenly.
    int union_begin = context_length;
    for (int t = 0; t < valid && t < tokens; ++t)
        union_begin = min(union_begin, max(0, positions[t + tokens * batch] - (window - 1)));
    const int union_count = sequence_active ? max(0, context_length - union_begin) : 0;
    const int split_begin = union_begin +
        static_cast<int>((static_cast<std::int64_t>(union_count) * split) / splits);
    const int split_end = union_begin +
        static_cast<int>((static_cast<std::int64_t>(union_count) * (split + 1)) / splits);
    const bool wave_active = wave * 16 < tokens * kGroup;

    SwaBf16x8 query[kHeadDim / 16];
#pragma unroll
    for (int tile = 0; tile < kHeadDim / 16; ++tile) {
        query[tile] = row_active
            ? *reinterpret_cast<const SwaBf16x8*>(
                  q + static_cast<std::int64_t>(kHeadDim) *
                          (q_head + kQHeads * (token + tokens * batch)) +
                  tile * 16 + half * 8)
            : SwaBf16x8{};
    }
    SwaF32x8 numerator[kHeadDim / 16];
#pragma unroll
    for (int tile = 0; tile < kHeadDim / 16; ++tile) numerator[tile] = SwaF32x8{};
    float reference = -INFINITY;
    float denominator = 0.0F;
    const float scale_log2 = scale * 1.4426950408889634F;
    const std::int64_t context_lane_base = static_cast<std::int64_t>(kHeadDim) *
        padded_capacity * kKvHeads * cache_lane;
    // `key_at(i)` addresses key i (0..15) of a block; `key_valid(i)` masks it for this row.
    const auto block = [&](auto key_at, auto key_valid, bool stage_valid) {
        // Stage V^T: 128 threads, each one key-feature-octet of the sixteen keys.
        __syncthreads();
        {
            const int key = tid / 8, octet = tid % 8;
            const hip_bfloat16* source = key_at(key, true);
            const uint4 packed = stage_valid && source != nullptr
                ? *reinterpret_cast<const uint4*>(source + octet * 16)
                : uint4{};
            const uint4 packed_high = stage_valid && source != nullptr
                ? *reinterpret_cast<const uint4*>(source + octet * 16 + 8)
                : uint4{};
            const std::uint32_t words[8] = {packed.x, packed.y, packed.z, packed.w,
                                            packed_high.x, packed_high.y, packed_high.z,
                                            packed_high.w};
#pragma unroll
            for (int item = 0; item < 16; ++item) {
                const std::uint32_t word = words[item / 2];
                staged_values[(octet * 16 + item) * kSwaValuePitch + key] =
                    static_cast<std::uint16_t>((item & 1) ? word >> 16 : word & 0xffffU);
            }
        }
        __syncthreads();
        if (!wave_active) return;
        SwaF32x8 scores{};
        const hip_bfloat16* key_row = key_at(axis, false);
#pragma unroll
        for (int tile = 0; tile < kHeadDim / 16; ++tile) {
            const SwaBf16x8 keys = key_row != nullptr
                ? *reinterpret_cast<const SwaBf16x8*>(key_row + tile * 16 + half * 8)
                : SwaBf16x8{};
            scores = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(keys, query[tile], scores);
        }
        float block_maximum = -INFINITY;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            float score = scores[item];
            if (!row_active || !key_valid(half * 8 + item)) score = -INFINITY;
            scores[item] = score;
            block_maximum = fmaxf(block_maximum, score);
        }
        block_maximum *= scale_log2;
        block_maximum = fmaxf(block_maximum, __shfl_xor(block_maximum, 16, 32));
        const bool raise = block_maximum > reference + 8.0F;
        const float next = raise ? block_maximum : reference;
        const float rescale = raise ? __builtin_amdgcn_exp2f(reference - next) : 1.0F;
        const float origin = next == -INFINITY ? 0.0F : next;
        if (raise) denominator *= rescale;
        SwaBf16x8 probabilities;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const float probability =
                __builtin_amdgcn_exp2f(__builtin_fmaf(scores[item], scale_log2, -origin));
            const std::uint16_t bits = swa_bf16_bits(probability);
            probabilities[item] = static_cast<short>(bits);
            // The denominator sums the represented BF16 probabilities, like the numerator.
            denominator += __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
        }
        if (__builtin_amdgcn_ballot_w32(raise) != 0U) {
            float row_rescale[8];
#pragma unroll
            for (int item = 0; item < 8; ++item)
                row_rescale[item] = __shfl(rescale, half * 8 + item, 32);
#pragma unroll
            for (int tile = 0; tile < kHeadDim / 16; ++tile)
#pragma unroll
                for (int item = 0; item < 8; ++item) numerator[tile][item] *= row_rescale[item];
        }
        reference = next;
#pragma unroll
        for (int tile = 0; tile < kHeadDim / 16; ++tile) {
            const SwaBf16x8 values = *reinterpret_cast<const SwaBf16x8*>(
                &staged_values[(tile * 16 + axis) * kSwaValuePitch + half * 8]);
            numerator[tile] =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(probabilities, values,
                                                                  numerator[tile]);
        }
    };
    // Context blocks of this split. Key rows and value rows share the slot address.
    for (int first = split_begin; first < split_end; first += 16) {
        block(
            [&](int key, bool values) -> const hip_bfloat16* {
                const int position = first + key;
                if (position >= split_end) return nullptr;
                const std::int64_t base = context_lane_base + static_cast<std::int64_t>(kHeadDim) *
                    ((position & (window - 1)) + padded_capacity * kv_head);
                return (values ? context_v : context_k) + base;
            },
            [&](int key) {
                const int position = first + key;
                return position < split_end && position >= row_begin &&
                       position < context_length;
            },
            true);
    }
    // The query block's own keys (every valid block token) in the last split.
    if (split == splits - 1 && sequence_active) {
        block(
            [&](int key, bool values) -> const hip_bfloat16* {
                if (key >= valid) return nullptr;
                const std::int64_t base = static_cast<std::int64_t>(kHeadDim) *
                    (kv_head + kKvHeads * (key + tokens * batch));
                return (values ? query_v : query_k) + base;
            },
            [&](int key) { return key < valid; },
            true);
    }
    if (!wave_active) return;
    const float row_denominator = denominator + __shfl_xor(denominator, 16, 32);
    // Numerator rows are half * 8 + item of this wave's tile; the lane holds feature axis.
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int out_row = wave * 16 + half * 8 + item;
        const int out_token = out_row / kGroup;
        if (out_token >= tokens) continue;
        const int out_head = kv_head * kGroup + out_row % kGroup;
#pragma unroll
        for (int tile = 0; tile < kHeadDim / 16; ++tile)
            partial_acc[partial_index(tile * 16 + axis, out_head, out_token, split, batch, tokens,
                                      splits)] = hip_bfloat16(numerator[tile][item]);
    }
    if (half == 0 && row_exists) {
        const std::int64_t stats = statistic_index(q_head, token, split, batch, tokens, splits);
        // The reduce kernel merges natural-exponent maxima.
        partial_m[stats] = reference == -INFINITY ? -INFINITY : reference / 1.4426950408889634F;
        partial_l[stats] = row_active ? row_denominator : 0.0F;
    }
}

__global__ __launch_bounds__(kHeadDim, 2) void swa_reduce_kernel(
    const hip_bfloat16* __restrict__ partial_acc, const float* __restrict__ partial_m,
    const float* __restrict__ partial_l, const std::int32_t* __restrict__ valid_columns,
    hip_bfloat16* __restrict__ output, int tokens, int splits) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int d = static_cast<int>(threadIdx.x);
    const std::int64_t output_index = static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kHeadDim) *
            (q_head + kQHeads * (token + tokens * batch));
    if (token >= valid_columns[batch]) {
        output[output_index] = hip_bfloat16(0.0F);
        return;
    }
    float maximum = -INFINITY;
    for (int split = 0; split < splits; ++split) {
        maximum = fmaxf(maximum, partial_m[statistic_index(q_head, token, split, batch, tokens,
                                                           splits)]);
    }
    float denominator = 0.0F;
    float numerator = 0.0F;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t stats = statistic_index(q_head, token, split, batch, tokens, splits);
        const float local_m = partial_m[stats];
        const float factor = isinf(local_m) ? 0.0F : expf(local_m - maximum);
        denominator += factor * partial_l[stats];
        numerator += factor * static_cast<float>(
            partial_acc[partial_index(d, q_head, token, split, batch, tokens, splits)]);
    }
    output[output_index] = hip_bfloat16(denominator > 0.0F ? numerator / denominator : 0.0F);
}

} // namespace

void swa_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                const Tensor& positions, const Tensor& valid_columns, const Tensor& lanes,
                float scale, const CyclicKVCacheLayerView& context,
                SwaContextExecutionEnvelope envelope, SwaLaunchPlan plan,
                const Tensor& partial_acc, const Tensor& partial_m, const Tensor& partial_l,
                Tensor& out, hipStream_t stream) {
    const dim3 grid(kKvHeads, static_cast<unsigned int>(q.ne[2]),
                    static_cast<unsigned int>(q.ne[3]));
    if (plan.direct) {
        swa_direct_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(q.data),
            static_cast<const hip_bfloat16*>(query_k.data),
            static_cast<const hip_bfloat16*>(query_v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(lanes.data),
            static_cast<const hip_bfloat16*>(context.k.data),
            static_cast<const hip_bfloat16*>(context.v.data),
            static_cast<hip_bfloat16*>(out.data), q.ne[2], q.ne[3],
            static_cast<int>(context.padded_capacity), static_cast<int>(context.capacity),
            context.lane_capacity, static_cast<int>(envelope.min_context),
            static_cast<int>(envelope.max_context), scale);
        HIP_CHECK(hipGetLastError());
        return;
    }

    const dim3 partial_grid(kKvHeads, static_cast<unsigned int>(plan.splits),
                            static_cast<unsigned int>(q.ne[3]));
    swa_partial_kernel<<<partial_grid, kSwaWmmaThreads, 0, stream>>>(
        static_cast<const hip_bfloat16*>(q.data),
        static_cast<const hip_bfloat16*>(query_k.data),
        static_cast<const hip_bfloat16*>(query_v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<const hip_bfloat16*>(context.k.data),
        static_cast<const hip_bfloat16*>(context.v.data),
        static_cast<hip_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
        static_cast<float*>(partial_l.data), q.ne[2],
        static_cast<int>(context.padded_capacity), static_cast<int>(context.capacity),
        context.lane_capacity, static_cast<int>(envelope.min_context),
        static_cast<int>(envelope.max_context), plan.splits, scale);
    HIP_CHECK(hipGetLastError());
    const dim3 reduce_grid(kQHeads, static_cast<unsigned int>(q.ne[2]),
                           static_cast<unsigned int>(q.ne[3]));
    swa_reduce_kernel<<<reduce_grid, kHeadDim, 0, stream>>>(
        static_cast<const hip_bfloat16*>(partial_acc.data),
        static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<hip_bfloat16*>(out.data), q.ne[2], plan.splits);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
