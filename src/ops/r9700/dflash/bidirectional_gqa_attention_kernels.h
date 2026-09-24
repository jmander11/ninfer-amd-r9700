#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::r9700 {

inline constexpr int kBidirectionalHeadDim = 128;
inline constexpr int kBidirectionalQHeads = 32;
inline constexpr int kBidirectionalKvHeads = 8;
inline constexpr int kBidirectionalGroup = 4;
inline constexpr int kBidirectionalPage = 64;
inline constexpr int kBidirectionalWave = 32;
inline constexpr int kBidirectionalBlock = kBidirectionalGroup * kBidirectionalWave;

__device__ __forceinline__ float bidirectional_wave_sum(float value) {
#pragma unroll
    for (int offset = kBidirectionalWave / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, kBidirectionalWave);
    }
    return __shfl(value, 0, kBidirectionalWave);
}

__device__ __forceinline__ std::int64_t bidirectional_q_index(int d, int q_head, int token,
                                                              int batch, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kBidirectionalQHeads) *
                    (static_cast<std::int64_t>(token) +
                     static_cast<std::int64_t>(tokens) * batch));
}

__device__ __forceinline__ std::int64_t bidirectional_query_kv_index(
    int d, int kv_head, int token, int batch, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(kBidirectionalKvHeads) *
                    (static_cast<std::int64_t>(token) +
                     static_cast<std::int64_t>(tokens) * batch));
}

__device__ __forceinline__ std::int64_t bidirectional_context_index(
    int d, int offset, int physical_page, int kv_head, int physical_pages) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalHeadDim) *
               (static_cast<std::int64_t>(offset) +
                static_cast<std::int64_t>(kBidirectionalPage) *
                    (static_cast<std::int64_t>(physical_page) +
                     static_cast<std::int64_t>(physical_pages) * kv_head));
}

__launch_bounds__(kBidirectionalBlock) __global__ void bidirectional_gqa_bf16_kernel(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ query_k,
    const hip_bfloat16* __restrict__ query_v,
    const std::int32_t* __restrict__ context_lengths,
    const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ table_rows,
    const hip_bfloat16* __restrict__ context_k,
    const hip_bfloat16* __restrict__ context_v,
    const std::int32_t* __restrict__ block_tables, hip_bfloat16* __restrict__ out,
    int tokens, int physical_pages, int logical_pages, int table_row_count,
    int min_context, int max_context, float scale) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int wave = static_cast<int>(threadIdx.x) / kBidirectionalWave;
    const int lane = static_cast<int>(threadIdx.x) % kBidirectionalWave;
    const int q_head = kv_head * kBidirectionalGroup + wave;
    if (kv_head >= kBidirectionalKvHeads || token >= tokens || wave >= kBidirectionalGroup) {
        return;
    }

    const int length = context_lengths[batch];
    const int valid = valid_columns[batch];
    const int table_row = table_rows[batch];
    const bool active = token < valid && valid >= 1 && valid <= tokens &&
                        length >= min_context && length <= max_context &&
                        length <= logical_pages * kBidirectionalPage && table_row >= 0 &&
                        table_row < table_row_count;

    float q_lane[4];
    float numerator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * kBidirectionalWave;
        q_lane[item] = active
                           ? static_cast<float>(q[bidirectional_q_index(d, q_head, token, batch,
                                                                       tokens)])
                           : 0.0F;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0F;
    const int key_count = active ? length + valid : 0;
    for (int key = 0; key < key_count; ++key) {
        const bool context_key = key < length;
        const int logical_page = context_key ? key / kBidirectionalPage : 0;
        const int offset = context_key ? key % kBidirectionalPage : 0;
        const int query_token = context_key ? 0 : key - length;
        const int physical_page =
            context_key ? block_tables[logical_page + logical_pages * table_row] : 0;
        const bool physical_live = !context_key ||
                                   (physical_page >= 0 && physical_page < physical_pages);

        float dot = 0.0F;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * kBidirectionalWave;
            float key_value = 0.0F;
            if (physical_live) {
                const std::int64_t index =
                    context_key
                        ? bidirectional_context_index(d, offset, physical_page, kv_head,
                                                      physical_pages)
                        : bidirectional_query_kv_index(d, kv_head, query_token, batch, tokens);
                key_value = static_cast<float>((context_key ? context_k : query_k)[index]);
            }
            dot += q_lane[item] * key_value;
        }
        const float score = bidirectional_wave_sum(dot) * scale;
        const float next_max = fmaxf(row_max, score);
        const float old_scale = row_max == -INFINITY ? 0.0F : expf(row_max - next_max);
        const float probability = physical_live ? expf(score - next_max) : 0.0F;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * kBidirectionalWave;
            float value = 0.0F;
            if (physical_live) {
                const std::int64_t index =
                    context_key
                        ? bidirectional_context_index(d, offset, physical_page, kv_head,
                                                      physical_pages)
                        : bidirectional_query_kv_index(d, kv_head, query_token, batch, tokens);
                value = static_cast<float>((context_key ? context_v : query_v)[index]);
            }
            numerator[item] = numerator[item] * old_scale + probability * value;
        }
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
    }

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * kBidirectionalWave;
        const float value = active && row_sum > 0.0F ? numerator[item] / row_sum : 0.0F;
        out[bidirectional_q_index(d, q_head, token, batch, tokens)] =
            static_cast<hip_bfloat16>(value);
    }
}

} // namespace ninfer::ops::r9700
