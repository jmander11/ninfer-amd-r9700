#include "ops/r9700/gdn_recurrence/gdn_prefill_scale_sidecar.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {
namespace {

constexpr int kDim = 128;
constexpr int kQHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadsPerQ = kValueHeads / kQHeads;
constexpr int kThreadsPerRow = 8;
constexpr int kKeysPerThread = kDim / kThreadsPerRow;
constexpr int kRowTiles = 4;
constexpr int kRowsPerBlock = kDim / kRowTiles;
constexpr int kBlock = kRowsPerBlock * kThreadsPerRow;
constexpr int kWavesPerPrepareBlock = kBlock / 32;
constexpr int kWidth = 2048;
constexpr float kEpsilon = 1.0e-6F;

__device__ __forceinline__ float wave_sum(float value) {
    value += __shfl_xor(value, 16, 32);
    value += __shfl_xor(value, 8, 32);
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

__device__ __forceinline__ float group_sum(float value) {
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

__global__ __launch_bounds__(kBlock) void norm_sidecar_prepare_kernel(
    const hip_bfloat16* query, const hip_bfloat16* key, float* query_inverse,
    float* key_inverse) {
    const int pair = static_cast<int>(blockIdx.x) * kWavesPerPrepareBlock +
        static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int token = pair / kQHeads;
    const int head = pair % kQHeads;
    const std::size_t base =
        (static_cast<std::size_t>(token) * kQHeads + head) * kDim;
    float query_sum = 0.0F;
    float key_sum = 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int feature = lane + item * 32;
        const float query_value = static_cast<float>(query[base + feature]);
        const float key_value = static_cast<float>(key[base + feature]);
        query_sum = fmaf(query_value, query_value, query_sum);
        key_sum = fmaf(key_value, key_value, key_sum);
    }
    query_sum = wave_sum(query_sum);
    key_sum = wave_sum(key_sum);
    if (lane == 0) {
        query_inverse[pair] = rsqrtf(query_sum + kEpsilon);
        key_inverse[pair] = rsqrtf(key_sum + kEpsilon);
    }
}

__device__ __forceinline__ void load_state_row(const float* state, std::size_t base,
                                                int value_row, int sublane,
                                                float (&local)[kKeysPerThread]) {
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        local[item] = state[base + static_cast<std::size_t>(value_row) * kDim + sublane +
                            item * kThreadsPerRow];
    }
}

__device__ __forceinline__ void store_state_row(float* state, std::size_t base,
                                                 int value_row, int sublane,
                                                 const float (&local)[kKeysPerThread]) {
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        state[base + static_cast<std::size_t>(value_row) * kDim + sublane +
              item * kThreadsPerRow] = local[item];
    }
}

__device__ __forceinline__ void transition(float (&local)[kKeysPerThread],
                                            const float* staged_query,
                                            const float* staged_key, float value,
                                            float alpha, float beta, float scale,
                                            int sublane, hip_bfloat16* output) {
    float state_key = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        const int feature = sublane + item * kThreadsPerRow;
        state_key = fmaf(local[item], staged_key[feature], state_key);
    }
    state_key = group_sum(state_key);
    const float delta = beta * (value - alpha * state_key);
    float state_query = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        const int feature = sublane + item * kThreadsPerRow;
        local[item] = fmaf(delta, staged_key[feature], alpha * local[item]);
        state_query = fmaf(local[item], staged_query[feature], state_query);
    }
    state_query = group_sum(state_query);
    if (sublane == 0) { *output = hip_bfloat16(state_query * scale); }
}

__global__ __launch_bounds__(kBlock, 1) void scale_sidecar_recurrence_kernel(
    const hip_bfloat16* query, const hip_bfloat16* key, const hip_bfloat16* value,
    const float* g, const float* beta, const float* query_inverse,
    const float* key_inverse, const float* state_in, float* state_out,
    hip_bfloat16* output, float scale) {
    const int value_head = static_cast<int>(blockIdx.x) / kRowTiles;
    const int row_tile = static_cast<int>(blockIdx.x) % kRowTiles;
    const int tid = static_cast<int>(threadIdx.x);
    const int value_row = row_tile * kRowsPerBlock + tid / kThreadsPerRow;
    const int sublane = tid % kThreadsPerRow;
    const int query_head = value_head / kHeadsPerQ;
    __shared__ float staged_query[kDim];
    __shared__ float staged_key[kDim];
    __shared__ float controls[2];
    float local[kKeysPerThread];
    const std::size_t state_base = static_cast<std::size_t>(value_head) * kDim * kDim;
    load_state_row(state_in, state_base, value_row, sublane, local);
    __syncthreads();
    for (int token = 0; token < kWidth; ++token) {
        const std::size_t pair = static_cast<std::size_t>(token) * kQHeads + query_head;
        const std::size_t query_base = pair * kDim;
        const std::size_t value_head_index =
            static_cast<std::size_t>(token) * kValueHeads + value_head;
        if (tid == 0) {
            controls[0] = expf(g[value_head_index]);
            controls[1] = beta[value_head_index];
        }
        if (tid < kDim) {
            staged_query[tid] =
                static_cast<float>(query[query_base + tid]) * query_inverse[pair];
            staged_key[tid] = static_cast<float>(key[query_base + tid]) * key_inverse[pair];
        }
        __syncthreads();
        float represented_value = 0.0F;
        if (sublane == 0) {
            represented_value = static_cast<float>(
                value[value_head_index * kDim + value_row]);
        }
        represented_value =
            __shfl(represented_value, (tid & 31) & -kThreadsPerRow, 32);
        transition(local, staged_query, staged_key, represented_value, controls[0], controls[1],
                   scale, sublane, output + value_head_index * kDim + value_row);
        __syncthreads();
    }
    store_state_row(state_out, state_base, value_row, sublane, local);
}

} // namespace

std::size_t prefill_scale_sidecar_workspace_bytes(std::int32_t width) noexcept {
    return width > 0 ? static_cast<std::size_t>(width) * 2U * kQHeads * sizeof(float) : 0U;
}

hipError_t prefill_scale_sidecar_prepare(const hip_bfloat16* query,
                                         const hip_bfloat16* key, float* sidecar,
                                         std::int32_t width, hipStream_t stream) noexcept {
    if (query == nullptr || key == nullptr || sidecar == nullptr || width != kWidth ||
        stream == nullptr) {
        return hipErrorInvalidValue;
    }
    float* const key_inverse = sidecar + static_cast<std::size_t>(kWidth) * kQHeads;
    constexpr int kPairs = kWidth * kQHeads;
    static_assert(kPairs % kWavesPerPrepareBlock == 0);
    hipLaunchKernelGGL(norm_sidecar_prepare_kernel,
                       dim3(kPairs / kWavesPerPrepareBlock), dim3(kBlock), 0, stream,
                       query, key, sidecar, key_inverse);
    return hipGetLastError();
}

hipError_t prefill_scale_sidecar_recurrence(
    const hip_bfloat16* query, const hip_bfloat16* key, const hip_bfloat16* value,
    const float* g, const float* beta, const float* state_in, float* state_out,
    hip_bfloat16* output, const float* sidecar, std::int32_t width, float scale,
    hipStream_t stream) noexcept {
    if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
        beta == nullptr || state_in == nullptr || state_out == nullptr || output == nullptr ||
        sidecar == nullptr || width != kWidth || !std::isfinite(scale) || stream == nullptr) {
        return hipErrorInvalidValue;
    }
    const float* const key_inverse = sidecar + static_cast<std::size_t>(kWidth) * kQHeads;
    hipLaunchKernelGGL(scale_sidecar_recurrence_kernel, dim3(kValueHeads * kRowTiles),
                       dim3(kBlock), 0, stream, query, key, value, g, beta, sidecar,
                       key_inverse, state_in, state_out, output, scale);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::gdn_recurrence
