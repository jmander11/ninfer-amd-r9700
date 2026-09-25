#include "ops/r9700/gdn_recurrence/gdn_prefill_staged.h"

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
constexpr int kRowsPerWave = 32 / kThreadsPerRow;
constexpr int kWavesPerHead = kDim / kRowsPerWave;
constexpr int kRecurrenceWaves = 8;
constexpr int kStageWaves = 8;
constexpr float kEpsilon = 1.0e-6F;
constexpr std::size_t kStagedPairFloats = 2U * kDim;

__device__ __forceinline__ float wave_sum(float value) {
    value += __shfl_xor(value, 16, 32);
    value += __shfl_xor(value, 8, 32);
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

// value + value[lane ^ 4], [lane ^ 2], [lane ^ 1] within each eight-lane row group, as DPP
// row_xmask additions.
__device__ __forceinline__ float group_sum(float value) {
    value += __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x164, 0xf, 0xf, false));
    value += __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x162, 0xf, 0xf, false));
    value += __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x161, 0xf, 0xf, false));
    return value;
}

// One wave per (token, Q/K head): FP32 sum of squares over four lane-strided items, reciprocal
// norm, and each normalized value in the recurrence lane order [sublane][item], q then k.
__global__ __launch_bounds__(kStageWaves * 32) void stage_qk_kernel(
    const hip_bfloat16* query, const hip_bfloat16* key, float* staged, int pairs) {
    const int pair = static_cast<int>(blockIdx.x) * kStageWaves + static_cast<int>(threadIdx.x) / 32;
    if (pair >= pairs) return;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const std::size_t base = static_cast<std::size_t>(pair) * kDim;
    float query_values[4], key_values[4];
    float query_sum = 0.0F, key_sum = 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int feature = lane + item * 32;
        query_values[item] = static_cast<float>(query[base + feature]);
        key_values[item] = static_cast<float>(key[base + feature]);
        query_sum = fmaf(query_values[item], query_values[item], query_sum);
        key_sum = fmaf(key_values[item], key_values[item], key_sum);
    }
    const float query_inverse = rsqrtf(wave_sum(query_sum) + kEpsilon);
    const float key_inverse = rsqrtf(wave_sum(key_sum) + kEpsilon);
    float* target = staged + static_cast<std::size_t>(pair) * kStagedPairFloats;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int feature = lane + item * 32;
        const int slot = (feature % kThreadsPerRow) * kKeysPerThread + feature / kThreadsPerRow;
        target[slot] = query_values[item] * query_inverse;
        target[kDim + slot] = key_values[item] * key_inverse;
    }
}

__global__ __launch_bounds__(256) void decay_kernel(const float* g, float* alpha, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) alpha[index] = expf(g[index]);
}

struct StagedToken {
    float4 query[4], key[4];
    float value, alpha, beta;
};

__device__ __forceinline__ StagedToken load_token(
    const float* staged, const float* alpha, const hip_bfloat16* value, const float* beta,
    int token, int query_head, int value_head, int value_row, int sublane) {
    StagedToken data;
    const float4* source = reinterpret_cast<const float4*>(
        staged + (static_cast<std::size_t>(token) * kQHeads + query_head) * kStagedPairFloats +
        sublane * kKeysPerThread);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        data.query[item] = source[item];
        data.key[item] = source[kDim / 4 + item];
    }
    const std::size_t value_head_index = static_cast<std::size_t>(token) * kValueHeads + value_head;
    data.value = static_cast<float>(value[value_head_index * kDim + value_row]);
    data.alpha = alpha[value_head_index];
    data.beta = beta[value_head_index];
    return data;
}

__device__ __forceinline__ void transition(float (&local)[kKeysPerThread], const StagedToken& data,
                                           float scale, int sublane, hip_bfloat16* output) {
    const float keys[kKeysPerThread] = {
        data.key[0].x, data.key[0].y, data.key[0].z, data.key[0].w,
        data.key[1].x, data.key[1].y, data.key[1].z, data.key[1].w,
        data.key[2].x, data.key[2].y, data.key[2].z, data.key[2].w,
        data.key[3].x, data.key[3].y, data.key[3].z, data.key[3].w};
    const float queries[kKeysPerThread] = {
        data.query[0].x, data.query[0].y, data.query[0].z, data.query[0].w,
        data.query[1].x, data.query[1].y, data.query[1].z, data.query[1].w,
        data.query[2].x, data.query[2].y, data.query[2].z, data.query[2].w,
        data.query[3].x, data.query[3].y, data.query[3].z, data.query[3].w};
    float state_key = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        state_key = fmaf(local[item], keys[item], state_key);
    }
    state_key = group_sum(state_key);
    const float delta = data.beta * (data.value - data.alpha * state_key);
    float state_query = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        local[item] = fmaf(delta, keys[item], data.alpha * local[item]);
        state_query = fmaf(local[item], queries[item], state_query);
    }
    state_query = group_sum(state_query);
    if (sublane == 0) { *output = hip_bfloat16(state_query * scale); }
}

// Each wave owns four state rows of one value head (eight lanes per row, sixteen keys per lane)
// and walks all tokens without workgroup barriers, prefetching the next token's staged inputs.
__global__ __launch_bounds__(kRecurrenceWaves * 32) void recurrence_kernel(
    const float* staged, const float* alpha, const hip_bfloat16* value, const float* beta,
    const float* state_in, float* state_out, hip_bfloat16* output, int width, float scale) {
    const int wave = static_cast<int>(blockIdx.x) * kRecurrenceWaves +
                     static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int value_head = wave / kWavesPerHead;
    const int value_row = (wave % kWavesPerHead) * kRowsPerWave + lane / kThreadsPerRow;
    const int sublane = lane % kThreadsPerRow;
    const int query_head = value_head / kHeadsPerQ;
    const std::size_t row_base = (static_cast<std::size_t>(value_head) * kDim + value_row) * kDim;
    float local[kKeysPerThread];
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        local[item] = state_in[row_base + sublane + item * kThreadsPerRow];
    }
    // Two register buffers alternate without copies; an odd width finishes with one step.
    StagedToken even = load_token(staged, alpha, value, beta, 0, query_head, value_head,
                                  value_row, sublane);
    int token = 0;
    for (; token + 1 < width; token += 2) {
        const StagedToken odd = load_token(staged, alpha, value, beta, token + 1, query_head,
                                           value_head, value_row, sublane);
        transition(local, even, scale, sublane,
                   output + (static_cast<std::size_t>(token) * kValueHeads + value_head) * kDim +
                       value_row);
        if (token + 2 < width) {
            even = load_token(staged, alpha, value, beta, token + 2, query_head, value_head,
                              value_row, sublane);
        }
        transition(local, odd, scale, sublane,
                   output + (static_cast<std::size_t>(token + 1) * kValueHeads + value_head) *
                                kDim + value_row);
    }
    if (token < width) {
        transition(local, even, scale, sublane,
                   output + (static_cast<std::size_t>(token) * kValueHeads + value_head) * kDim +
                       value_row);
    }
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        state_out[row_base + sublane + item * kThreadsPerRow] = local[item];
    }
}

} // namespace

std::size_t prefill_staged_workspace_bytes(std::int32_t width) noexcept {
    if (width < kStagedMinimumWidth || width > kStagedMaximumWidth) return 0U;
    const auto tokens = static_cast<std::size_t>(width);
    return (tokens * kQHeads * kStagedPairFloats + tokens * kValueHeads) * sizeof(float);
}

hipError_t prefill_staged(const hip_bfloat16* query, const hip_bfloat16* key,
                          const hip_bfloat16* value, const float* g, const float* beta,
                          const float* state_in, float* state_out, hip_bfloat16* output,
                          void* workspace, std::size_t workspace_bytes, std::int32_t width,
                          float scale, hipStream_t stream) noexcept {
    const std::size_t required = prefill_staged_workspace_bytes(width);
    if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
        beta == nullptr || state_in == nullptr || state_out == nullptr || output == nullptr ||
        workspace == nullptr || required == 0U || workspace_bytes < required ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(float4) != 0U ||
        !std::isfinite(scale) || stream == nullptr) {
        return hipErrorInvalidValue;
    }
    auto* staged = static_cast<float*>(workspace);
    float* alpha = staged + static_cast<std::size_t>(width) * kQHeads * kStagedPairFloats;
    const int pairs = width * kQHeads;
    hipLaunchKernelGGL(stage_qk_kernel, dim3((pairs + kStageWaves - 1) / kStageWaves),
                       dim3(kStageWaves * 32), 0, stream, query, key, staged, pairs);
    const int controls = width * kValueHeads;
    hipLaunchKernelGGL(decay_kernel, dim3((controls + 255) / 256), dim3(256), 0, stream, g,
                       alpha, controls);
    hipLaunchKernelGGL(recurrence_kernel, dim3(kValueHeads * kWavesPerHead / kRecurrenceWaves),
                       dim3(kRecurrenceWaves * 32), 0, stream, staged, alpha, value, beta,
                       state_in, state_out, output, width, scale);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::gdn_recurrence
