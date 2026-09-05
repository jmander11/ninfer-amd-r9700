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

__global__ __launch_bounds__(kBlock, 2) void swa_partial_kernel(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ query_k,
    const hip_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const hip_bfloat16* __restrict__ context_k,
    const hip_bfloat16* __restrict__ context_v, hip_bfloat16* __restrict__ partial_acc,
    float* __restrict__ partial_m, float* __restrict__ partial_l, int tokens, int padded_capacity,
    int window, int lane_capacity, int minimum_context, int maximum_context, int splits,
    float scale) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int split_batch = static_cast<int>(blockIdx.z);
    const int split = split_batch % splits;
    const int batch = split_batch / splits;
    const int wave = static_cast<int>(threadIdx.x) / kWave;
    const int lane = static_cast<int>(threadIdx.x) & (kWave - 1);
    const int q_head = kv_head * kGroup + wave;
    const int valid = valid_columns[batch];
    const int cache_lane = lanes[batch];
    const int context_length = positions[tokens * batch];
    const int query_position = positions[token + tokens * batch];
    const bool active = token < valid && valid >= 0 && valid <= tokens && cache_lane >= 0 &&
                        cache_lane < lane_capacity && context_length >= minimum_context &&
                        context_length <= maximum_context && query_position >= context_length;

    float query[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const std::int64_t q_base = static_cast<std::int64_t>(kHeadDim) *
        (q_head + kQHeads * (token + tokens * batch));
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        if (active) query[item] = static_cast<float>(q[q_base + lane + item * kWave]);
    }
    float maximum = -INFINITY;
    float denominator = 0.0F;
    if (active) {
        const int context_begin = max(0, query_position - (window - 1));
        const int context_count = max(0, context_length - context_begin);
        const int split_begin = context_begin +
            static_cast<int>((static_cast<std::int64_t>(context_count) * split) / splits);
        const int split_end = context_begin +
            static_cast<int>((static_cast<std::int64_t>(context_count) * (split + 1)) / splits);
        const std::int64_t context_lane_base = static_cast<std::int64_t>(kHeadDim) *
            padded_capacity * kKvHeads * cache_lane;
        for (int position = split_begin; position < split_end; ++position) {
            const int slot = position & (window - 1);
            const std::int64_t key_base = context_lane_base +
                static_cast<std::int64_t>(kHeadDim) * (slot + padded_capacity * kv_head);
            consume_key(query, context_k + key_base, context_v + key_base, scale, maximum,
                        denominator, accumulator);
        }
        if (split == splits - 1) {
            for (int key_token = 0; key_token < valid; ++key_token) {
                const std::int64_t key_base = static_cast<std::int64_t>(kHeadDim) *
                    (kv_head + kKvHeads * (key_token + tokens * batch));
                consume_key(query, query_k + key_base, query_v + key_base, scale, maximum,
                            denominator, accumulator);
            }
        }
    }

    const std::int64_t stats = statistic_index(q_head, token, split, batch, tokens, splits);
    if (lane == 0) {
        partial_m[stats] = maximum;
        partial_l[stats] = denominator;
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        partial_acc[partial_index(lane + item * kWave, q_head, token, split, batch, tokens,
                                  splits)] = hip_bfloat16(accumulator[item]);
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

    const dim3 partial_grid(kKvHeads, static_cast<unsigned int>(q.ne[2]),
                            static_cast<unsigned int>(q.ne[3] * plan.splits));
    swa_partial_kernel<<<partial_grid, kBlock, 0, stream>>>(
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
