#include "ops/r9700/gdn_recurrence/gdn_grouped_heads_qualification.h"

#include <hip/hip_bfloat16.h>

#include <cstddef>

namespace ninfer::ops::r9700::gdn_recurrence {
namespace {

constexpr int kDim = 128;
constexpr int kQHeads = 16;
constexpr int kValueHeadsPerQ = 3;
constexpr int kValueHeads = kQHeads * kValueHeadsPerQ;
constexpr int kRowTiles = 4;
constexpr int kRowsPerBlock = kDim / kRowTiles;
constexpr int kThreadsPerRow = 8;
constexpr int kKeysPerThread = kDim / kThreadsPerRow;
constexpr int kThreads = kRowsPerBlock * kThreadsPerRow;
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

__device__ __forceinline__ void workgroup_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__global__ __launch_bounds__(kThreads, 1)
void grouped_heads_normalized_kernel(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    const float* g, const float* beta, const float* state_in, float* state_out,
    hip_bfloat16* output, int width, float scale) {
    const int q_head = static_cast<int>(blockIdx.x) / kRowTiles;
    const int row_tile = static_cast<int>(blockIdx.x) % kRowTiles;
    const int tid = static_cast<int>(threadIdx.x);
    const int value_row = row_tile * kRowsPerBlock + tid / kThreadsPerRow;
    const int sublane = tid % kThreadsPerRow;
    __shared__ float staged_q[kDim];
    __shared__ float staged_k[kDim];
    __shared__ float norm_scales[2];
    __shared__ float controls[2 * kValueHeadsPerQ];
    float local[kValueHeadsPerQ][kKeysPerThread];

#pragma unroll
    for (int head_offset = 0; head_offset < kValueHeadsPerQ; ++head_offset) {
        const int value_head = q_head * kValueHeadsPerQ + head_offset;
        const std::size_t state_base =
            static_cast<std::size_t>(value_head) * kDim * kDim +
            static_cast<std::size_t>(value_row) * kDim + sublane;
#pragma unroll
        for (int item = 0; item < kKeysPerThread; ++item) {
            local[head_offset][item] = state_in[state_base + item * kThreadsPerRow];
        }
    }
    workgroup_barrier();

    for (int token = 0; token < width; ++token) {
        const std::size_t qk_base =
            (static_cast<std::size_t>(token) * kQHeads + q_head) * kDim;
        if (tid < 32) {
            float q_sum = 0.0F;
            float k_sum = 0.0F;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = tid + item * 32;
                const float q_value = static_cast<float>(q[qk_base + index]);
                const float k_value = static_cast<float>(k[qk_base + index]);
                staged_q[index] = q_value;
                staged_k[index] = k_value;
                q_sum = fmaf(q_value, q_value, q_sum);
                k_sum = fmaf(k_value, k_value, k_sum);
            }
            q_sum = wave_sum(q_sum);
            k_sum = wave_sum(k_sum);
            if (tid == 0) {
                norm_scales[0] = rsqrtf(q_sum + kEpsilon);
                norm_scales[1] = rsqrtf(k_sum + kEpsilon);
            }
        }
        if (tid < kValueHeadsPerQ) {
            const int value_head = q_head * kValueHeadsPerQ + tid;
            const std::size_t vh =
                static_cast<std::size_t>(token) * kValueHeads + value_head;
            controls[2 * tid] = expf(g[vh]);
            controls[2 * tid + 1] = beta[vh];
        }
        workgroup_barrier();
        if (tid < kDim) {
            staged_q[tid] *= norm_scales[0];
            staged_k[tid] *= norm_scales[1];
        }
        workgroup_barrier();

#pragma unroll
        for (int head_offset = 0; head_offset < kValueHeadsPerQ; ++head_offset) {
            const int value_head = q_head * kValueHeadsPerQ + head_offset;
            const std::size_t vh =
                static_cast<std::size_t>(token) * kValueHeads + value_head;
            float value = sublane == 0
                ? static_cast<float>(v[vh * kDim + value_row]) : 0.0F;
            value = __shfl(value, (tid & 31) & -kThreadsPerRow, 32);
            float state_key = 0.0F;
#pragma unroll
            for (int item = 0; item < kKeysPerThread; ++item) {
                const int key = sublane + item * kThreadsPerRow;
                state_key = fmaf(local[head_offset][item], staged_k[key], state_key);
            }
            state_key = group_sum(state_key);
            const float alpha = controls[2 * head_offset];
            const float delta = controls[2 * head_offset + 1] *
                                (value - alpha * state_key);
            float state_query = 0.0F;
#pragma unroll
            for (int item = 0; item < kKeysPerThread; ++item) {
                const int key = sublane + item * kThreadsPerRow;
                local[head_offset][item] = fmaf(
                    delta, staged_k[key], alpha * local[head_offset][item]);
                state_query = fmaf(local[head_offset][item], staged_q[key], state_query);
            }
            state_query = group_sum(state_query);
            if (sublane == 0) {
                output[vh * kDim + value_row] = hip_bfloat16(state_query * scale);
            }
        }
        workgroup_barrier();
    }

#pragma unroll
    for (int head_offset = 0; head_offset < kValueHeadsPerQ; ++head_offset) {
        const int value_head = q_head * kValueHeadsPerQ + head_offset;
        const std::size_t state_base =
            static_cast<std::size_t>(value_head) * kDim * kDim +
            static_cast<std::size_t>(value_row) * kDim + sublane;
#pragma unroll
        for (int item = 0; item < kKeysPerThread; ++item) {
            state_out[state_base + item * kThreadsPerRow] = local[head_offset][item];
        }
    }
}

} // namespace

hipError_t grouped_heads_qualification(
    const void* q, const void* k, const void* v, const float* g, const float* beta,
    const float* state_in, float* state_out, void* output, int width, float scale,
    hipStream_t stream) noexcept {
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr ||
        state_in == nullptr || state_out == nullptr || output == nullptr || width < 1 ||
        !__builtin_isfinite(scale)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(grouped_heads_normalized_kernel, dim3(kQHeads * kRowTiles),
                       dim3(kThreads), 0, stream,
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v), g, beta, state_in, state_out,
                       static_cast<hip_bfloat16*>(output), width, scale);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::gdn_recurrence
