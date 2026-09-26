#pragma once

// Implements: include/ninfer/ops/argmax.h
// Configured contiguous BF16 argmax: one wave32 CTA per column, exact lower-id tie break.
// Suppression and eligibility apply before the reduction; no counts or sampling state mutate.

#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <climits>
#include <cmath>

#include "ninfer/ops/sampling.h"

namespace ninfer::ops {

inline constexpr int kArgmaxBlock          = 512;

__device__ __forceinline__ bool argmax_better(float value, std::int32_t index, float best_value,
                                              std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ bool argmax_token_suppressed(std::int32_t token,
                                                        const SamplingConfig* config,
                                                        std::int32_t column) {
    if (config == nullptr) { return false; }
    if (config->allowed_token_words) {
        const auto* words = config->allowed_token_words +
            static_cast<std::int64_t>(column) * config->allowed_token_column_stride;
        if (!(words[token >> 5] & (std::uint32_t{1} << (token & 31)))) { return true; }
    }
    const int count = min(config->suppressed_token_count,
                          SamplingConfig::kMaximumSuppressedTokens);
    for (int i = 0; i < count; ++i) {
        if (config->suppressed_tokens[i] == token) { return true; }
    }
    return false;
}

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_kernel(const hip_bfloat16* logits, std::int32_t* out, std::int32_t valid_rows,
                       std::int32_t physical_rows, const SamplingConfig* configs,
                       std::int32_t columns_per_config) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const SamplingConfig* config =
        configs == nullptr ? nullptr : configs + t / columns_per_config;

    float best_value        = -INFINITY;
    std::int32_t best_index = INT32_MAX;
    const std::int32_t column = config ? t % columns_per_config : 0;
    const auto consider = [&](std::int32_t v, float value) {
        if (argmax_better(value, v, best_value, best_index) &&
            !argmax_token_suppressed(v, config, column)) {
            best_value = value;
            best_index = v;
        }
    };
    // 16-byte rows when the column base is aligned: eight logits per load, several in flight.
    std::int32_t scalar_begin = 0;
    if (physical_rows % 8 == 0 && reinterpret_cast<std::uintptr_t>(logits) % 16U == 0U) {
        const auto* vectors = reinterpret_cast<const uint4*>(logits + base);
        const std::int32_t vector_count = valid_rows / 8;
#pragma unroll 4
        for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < vector_count;
             i += kArgmaxBlock) {
            const uint4 packed = vectors[i];
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                consider(i * 8 + 2 * w, __uint_as_float(words[w] << 16));
                consider(i * 8 + 2 * w + 1, __uint_as_float(words[w] & 0xffff0000U));
            }
        }
        scalar_begin = vector_count * 8;
    }
    for (std::int32_t v = scalar_begin + static_cast<std::int32_t>(threadIdx.x); v < valid_rows;
         v += kArgmaxBlock) {
        consider(v, static_cast<float>(logits[base + v]));
    }

    __shared__ float values[kArgmaxBlock];
    __shared__ std::int32_t indices[kArgmaxBlock];
    values[threadIdx.x]  = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value        = values[threadIdx.x + stride];
            const std::int32_t other_index = indices[threadIdx.x + stride];
            if (argmax_better(other_value, other_index, values[threadIdx.x],
                              indices[threadIdx.x])) {
                values[threadIdx.x]  = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[t] = indices[0]; }
}

} // namespace ninfer::ops
