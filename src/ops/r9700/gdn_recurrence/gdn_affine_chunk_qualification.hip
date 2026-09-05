#include "ops/r9700/gdn_recurrence/gdn_affine_chunk_qualification.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::r9700::gdn_recurrence {
namespace {

constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kQHeads = 16;
constexpr std::uint32_t kValueHeads = 48;
constexpr std::uint32_t kHeadGroup = 3;
constexpr std::uint32_t kRowTiles = 4;
constexpr std::uint32_t kRowsPerBlock = kDim / kRowTiles;
constexpr std::uint32_t kThreadsPerRow = 8;
constexpr std::uint32_t kKeysPerThread = kDim / kThreadsPerRow;
constexpr std::uint32_t kBlock = kRowsPerBlock * kThreadsPerRow;
constexpr float kEpsilon = 1.0e-6F;

struct Workspace {
    float* transforms = nullptr;
    float* additions = nullptr;
    std::uint32_t chunks = 0;
};

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

__device__ __forceinline__ void stage_normalized_k(const hip_bfloat16* k,
                                                    std::size_t base,
                                                    float* staged_k,
                                                    float* inverse_slot) {
    const std::uint32_t tid = threadIdx.x;
    if (tid < 32U) {
        float sum = 0.0F;
#pragma unroll
        for (std::uint32_t item = 0; item < 4U; ++item) {
            const std::uint32_t index = tid + item * 32U;
            const float value = static_cast<float>(k[base + index]);
            staged_k[index] = value;
            sum = fmaf(value, value, sum);
        }
        sum = wave_sum(sum);
        if (tid == 0U) *inverse_slot = rsqrtf(sum + kEpsilon);
    }
    __syncthreads();
    if (tid < kDim) staged_k[tid] *= *inverse_slot;
    __syncthreads();
}

__device__ __forceinline__ void stage_normalized_qk(
    const hip_bfloat16* q, const hip_bfloat16* k, std::size_t base,
    float* staged_q, float* staged_k, float* inverses) {
    const std::uint32_t tid = threadIdx.x;
    if (tid < 32U) {
        float qsum = 0.0F;
        float ksum = 0.0F;
#pragma unroll
        for (std::uint32_t item = 0; item < 4U; ++item) {
            const std::uint32_t index = tid + item * 32U;
            const float qvalue = static_cast<float>(q[base + index]);
            const float kvalue = static_cast<float>(k[base + index]);
            staged_q[index] = qvalue;
            staged_k[index] = kvalue;
            qsum = fmaf(qvalue, qvalue, qsum);
            ksum = fmaf(kvalue, kvalue, ksum);
        }
        qsum = wave_sum(qsum);
        ksum = wave_sum(ksum);
        if (tid == 0U) {
            inverses[0] = rsqrtf(qsum + kEpsilon);
            inverses[1] = rsqrtf(ksum + kEpsilon);
        }
    }
    __syncthreads();
    if (tid < kDim) {
        staged_q[tid] *= inverses[0];
        staged_k[tid] *= inverses[1];
    }
    __syncthreads();
}

__global__ __launch_bounds__(kBlock, 1) void affine_chunk_build_qualification_kernel(
    const hip_bfloat16* k, const hip_bfloat16* v, const float* g,
    const float* beta, float* transforms, float* additions,
    std::uint32_t tokens, std::uint32_t chunk_size, std::uint32_t chunks) {
    const std::uint32_t row_tile = blockIdx.x;
    const std::uint32_t chunk_index = blockIdx.y;
    const std::uint32_t value_head = blockIdx.z;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t row = row_tile * kRowsPerBlock + tid / kThreadsPerRow;
    const std::uint32_t sublane = tid % kThreadsPerRow;
    const std::uint32_t q_head = value_head / kHeadGroup;
    const std::size_t matrix_base =
        (static_cast<std::size_t>(value_head) * chunks + chunk_index) * kDim * kDim;
    float local_a[kKeysPerThread];
    float local_b[kKeysPerThread];
#pragma unroll
    for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
        const std::uint32_t column = sublane + item * kThreadsPerRow;
        local_a[item] = row == column ? 1.0F : 0.0F;
        local_b[item] = 0.0F;
    }
    __shared__ float staged_k[kDim];
    __shared__ float controls[3];
    const std::uint32_t token_begin = chunk_index * chunk_size;
    const std::uint32_t token_end = min(tokens, token_begin + chunk_size);
    for (std::uint32_t token = token_begin; token < token_end; ++token) {
        const std::size_t qk_base =
            (static_cast<std::size_t>(token) * kQHeads + q_head) * kDim;
        const std::size_t vh = static_cast<std::size_t>(token) * kValueHeads + value_head;
        if (tid == 0U) {
            controls[0] = expf(g[vh]);
            controls[1] = beta[vh];
        }
        stage_normalized_k(k, qk_base, staged_k, controls + 2);
        float dot_a = 0.0F;
        float dot_b = 0.0F;
#pragma unroll
        for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
            const std::uint32_t column = sublane + item * kThreadsPerRow;
            dot_a = fmaf(local_a[item], staged_k[column], dot_a);
            dot_b = fmaf(local_b[item], staged_k[column], dot_b);
        }
        dot_a = group_sum(dot_a);
        dot_b = group_sum(dot_b);
        float value = 0.0F;
        if (sublane == 0U) value = static_cast<float>(v[vh * kDim + row]);
        const int source_lane =
            (static_cast<int>(tid) & 31) & -static_cast<int>(kThreadsPerRow);
        value = __shfl(value, source_lane, 32);
        const float alpha = controls[0];
        const float beta_value = controls[1];
        const float delta_a = -alpha * beta_value * dot_a;
        const float delta_b = beta_value * (value - alpha * dot_b);
#pragma unroll
        for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
            const std::uint32_t column = sublane + item * kThreadsPerRow;
            local_a[item] = fmaf(delta_a, staged_k[column], alpha * local_a[item]);
            local_b[item] = fmaf(delta_b, staged_k[column], alpha * local_b[item]);
        }
        __syncthreads();
    }
#pragma unroll
    for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
        const std::uint32_t column = sublane + item * kThreadsPerRow;
        transforms[matrix_base + static_cast<std::size_t>(row) * kDim + column] = local_a[item];
        additions[matrix_base + static_cast<std::size_t>(row) * kDim + column] = local_b[item];
    }
}

__global__ __launch_bounds__(kBlock, 1) void affine_chunk_boundaries_qualification_kernel(
    const float* state_in, float* state_out, const float* transforms,
    float* additions_and_boundaries, std::uint32_t chunks) {
    const std::uint32_t row_tile = blockIdx.x;
    const std::uint32_t value_head = blockIdx.y;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t row = row_tile * kRowsPerBlock + tid / kThreadsPerRow;
    const std::uint32_t sublane = tid % kThreadsPerRow;
    const std::size_t state_base = static_cast<std::size_t>(value_head) * kDim * kDim;
    __shared__ float state_tile[kRowsPerBlock * kDim];
#pragma unroll
    for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
        const std::uint32_t column = sublane + item * kThreadsPerRow;
        state_tile[row % kRowsPerBlock * kDim + column] =
            state_in[state_base + static_cast<std::size_t>(row) * kDim + column];
    }
    __syncthreads();
    for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t matrix_base =
            (static_cast<std::size_t>(value_head) * chunks + chunk) * kDim * kDim;
        float next[kKeysPerThread];
#pragma unroll
        for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
            const std::uint32_t column = sublane + item * kThreadsPerRow;
            float sum = 0.0F;
            for (std::uint32_t source = 0; source < kDim; ++source) {
                sum = fmaf(state_tile[row % kRowsPerBlock * kDim + source],
                           transforms[matrix_base + static_cast<std::size_t>(source) * kDim + column],
                           sum);
            }
            const std::size_t element = matrix_base + static_cast<std::size_t>(row) * kDim + column;
            const float addition = additions_and_boundaries[element];
            additions_and_boundaries[element] =
                state_tile[row % kRowsPerBlock * kDim + column];
            next[item] = sum + addition;
        }
        __syncthreads();
#pragma unroll
        for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
            const std::uint32_t column = sublane + item * kThreadsPerRow;
            state_tile[row % kRowsPerBlock * kDim + column] = next[item];
        }
        __syncthreads();
    }
#pragma unroll
    for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
        const std::uint32_t column = sublane + item * kThreadsPerRow;
        state_out[state_base + static_cast<std::size_t>(row) * kDim + column] =
            state_tile[row % kRowsPerBlock * kDim + column];
    }
}

__global__ __launch_bounds__(kBlock, 1) void affine_chunk_replay_qualification_kernel(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    const float* g, const float* beta, const float* boundaries,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t chunk_size,
    std::uint32_t chunks, float scale) {
    const std::uint32_t row_tile = blockIdx.x;
    const std::uint32_t chunk_index = blockIdx.y;
    const std::uint32_t value_head = blockIdx.z;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t row = row_tile * kRowsPerBlock + tid / kThreadsPerRow;
    const std::uint32_t sublane = tid % kThreadsPerRow;
    const std::uint32_t q_head = value_head / kHeadGroup;
    const std::size_t matrix_base =
        (static_cast<std::size_t>(value_head) * chunks + chunk_index) * kDim * kDim;
    float local[kKeysPerThread];
#pragma unroll
    for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
        const std::uint32_t column = sublane + item * kThreadsPerRow;
        local[item] = boundaries[matrix_base + static_cast<std::size_t>(row) * kDim + column];
    }
    __shared__ float staged_q[kDim];
    __shared__ float staged_k[kDim];
    __shared__ float controls[4];
    const std::uint32_t token_begin = chunk_index * chunk_size;
    const std::uint32_t token_end = min(tokens, token_begin + chunk_size);
    for (std::uint32_t token = token_begin; token < token_end; ++token) {
        const std::size_t qk_base =
            (static_cast<std::size_t>(token) * kQHeads + q_head) * kDim;
        const std::size_t vh = static_cast<std::size_t>(token) * kValueHeads + value_head;
        if (tid == 0U) {
            controls[0] = expf(g[vh]);
            controls[1] = beta[vh];
        }
        stage_normalized_qk(q, k, qk_base, staged_q, staged_k, controls + 2);
        float state_key = 0.0F;
#pragma unroll
        for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
            const std::uint32_t column = sublane + item * kThreadsPerRow;
            state_key = fmaf(local[item], staged_k[column], state_key);
        }
        state_key = group_sum(state_key);
        float value = 0.0F;
        if (sublane == 0U) value = static_cast<float>(v[vh * kDim + row]);
        const int source_lane =
            (static_cast<int>(tid) & 31) & -static_cast<int>(kThreadsPerRow);
        value = __shfl(value, source_lane, 32);
        const float alpha = controls[0];
        const float delta = controls[1] * (value - alpha * state_key);
        float state_query = 0.0F;
#pragma unroll
        for (std::uint32_t item = 0; item < kKeysPerThread; ++item) {
            const std::uint32_t column = sublane + item * kThreadsPerRow;
            local[item] = fmaf(delta, staged_k[column], alpha * local[item]);
            state_query = fmaf(local[item], staged_q[column], state_query);
        }
        state_query = group_sum(state_query);
        if (sublane == 0U) output[vh * kDim + row] = hip_bfloat16(state_query * scale);
        __syncthreads();
    }
}

bool bind_workspace(void* storage, std::size_t bytes, std::uint32_t tokens,
                    std::uint32_t chunk, Workspace* out) {
    const std::size_t required = affine_chunk_workspace_capacity_bytes(tokens, chunk);
    if (storage == nullptr || out == nullptr || bytes != required || required == 0U ||
        reinterpret_cast<std::uintptr_t>(storage) % alignof(float) != 0U) return false;
    const std::uint32_t chunks = (tokens + chunk - 1U) / chunk;
    const std::size_t matrix_elements =
        static_cast<std::size_t>(kValueHeads) * chunks * kDim * kDim;
    auto* base = static_cast<float*>(storage);
    *out = Workspace{base, base + matrix_elements, chunks};
    return true;
}

} // namespace

std::size_t affine_chunk_workspace_capacity_bytes(std::uint32_t tokens,
                                                   std::uint32_t chunk) noexcept {
    if (tokens == 0U || (chunk != 32U && chunk != 64U && chunk != 128U)) return 0U;
    const std::uint64_t chunks = (static_cast<std::uint64_t>(tokens) + chunk - 1U) / chunk;
    const std::uint64_t elements = 2U * kValueHeads * chunks * kDim * kDim;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) return 0U;
    return static_cast<std::size_t>(elements) * sizeof(float);
}

hipError_t affine_chunk_qualification(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    const float* g, const float* beta, const float* state_in, float* state_out,
    hip_bfloat16* output, void* workspace, std::size_t workspace_bytes,
    std::uint32_t tokens, std::uint32_t chunk, float scale,
    hipStream_t stream) noexcept {
    Workspace bound{};
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr ||
        state_in == nullptr || state_out == nullptr || output == nullptr || stream == nullptr ||
        !std::isfinite(scale) || !bind_workspace(workspace, workspace_bytes, tokens, chunk, &bound)) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(affine_chunk_build_qualification_kernel,
                       dim3(kRowTiles, bound.chunks, kValueHeads), dim3(kBlock), 0, stream,
                       k, v, g, beta, bound.transforms, bound.additions, tokens, chunk,
                       bound.chunks);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(affine_chunk_boundaries_qualification_kernel,
                       dim3(kRowTiles, kValueHeads), dim3(kBlock), 0, stream,
                       state_in, state_out, bound.transforms, bound.additions, bound.chunks);
    status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(affine_chunk_replay_qualification_kernel,
                       dim3(kRowTiles, bound.chunks, kValueHeads), dim3(kBlock), 0, stream,
                       q, k, v, g, beta, bound.additions, output, tokens, chunk,
                       bound.chunks, scale);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::gdn_recurrence
