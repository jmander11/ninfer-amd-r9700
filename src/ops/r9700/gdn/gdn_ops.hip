#include "ops/r9700/gdn/gdn_ops.h"

#include "ninfer/types.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::r9700::gdn {
namespace {

constexpr std::uint32_t kBlock = 256;
constexpr std::uint32_t kMaxKeyDim = 256;
constexpr std::uint32_t kProjectionQueryRows = 2048;
constexpr std::uint32_t kProjectionKeyRows = 2048;
constexpr std::uint32_t kProjectionValueRows = 6144;
constexpr std::uint32_t kProjectionValueZRows = 12288;
constexpr std::uint32_t kProjectionChannels =
    kProjectionQueryRows + kProjectionKeyRows + kProjectionValueRows;
constexpr std::uint32_t kProjectionPrefillTokens = 2048;
constexpr std::uint32_t kMaximumVerifyWidth = 16;
constexpr std::uint32_t kMaximumBatch = kMaximumConcurrency;
constexpr std::uint32_t kControlHeads = 48;
constexpr std::uint32_t kControlColumns = 5120;
constexpr std::uint32_t kControlT1MaximumTokens = 24;
// Wider launches: one CTA per 16 tokens, eight waves splitting K, six WMMA row tiles (a|b).
constexpr std::uint32_t kControlWideTokens = 16;
constexpr std::uint32_t kControlWideSplits = 8;
constexpr std::uint32_t kControlWideSplitColumns = kControlColumns / kControlWideSplits;
constexpr std::uint32_t kControlWideTiles = 2U * kControlHeads / 16U;
static_assert(kControlColumns % (kControlWideSplits * 16U) == 0U);

[[nodiscard]] bool valid_stream(hipStream_t stream) noexcept { return stream != nullptr; }

[[nodiscard]] bool valid_count(std::size_t count) noexcept {
    return count != 0 && count <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) *
                                      static_cast<std::size_t>(kBlock);
}

[[nodiscard]] dim3 linear_grid(std::size_t count) noexcept {
    return dim3(static_cast<std::uint32_t>((count + kBlock - 1U) / kBlock));
}

struct ByteRange {
    std::uintptr_t first;
    std::uintptr_t last;
};

template <typename T>
[[nodiscard]] ByteRange byte_range(const T* pointer, std::size_t elements) noexcept {
    const std::uintptr_t first = reinterpret_cast<std::uintptr_t>(pointer);
    return {first, first + elements * sizeof(T)};
}

[[nodiscard]] bool overlaps(ByteRange lhs, ByteRange rhs) noexcept {
    return lhs.first < rhs.last && rhs.first < lhs.last;
}

template <typename Launch>
[[nodiscard]] hipError_t checked_launch(bool valid, Launch&& launch) noexcept {
    if (!valid) { return hipErrorInvalidValue; }
    launch();
    return hipGetLastError();
}

__global__ void causal_conv1d_silu_kernel(const hip_bfloat16* input,
                                           const hip_bfloat16* weight,
                                           const hip_bfloat16* history_in,
                                           hip_bfloat16* history_out, hip_bfloat16* output,
                                           std::uint32_t channels, std::uint32_t tokens) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) { return; }
    hip_bfloat16 h0 = history_in[channel];
    hip_bfloat16 h1 = history_in[channels + channel];
    hip_bfloat16 h2 = history_in[2U * channels + channel];
    const float w0 = static_cast<float>(weight[channel]);
    const float w1 = static_cast<float>(weight[channels + channel]);
    const float w2 = static_cast<float>(weight[2U * channels + channel]);
    const float w3 = static_cast<float>(weight[3U * channels + channel]);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const hip_bfloat16 current = input[static_cast<std::size_t>(token) * channels + channel];
        float sum = fmaf(w0, static_cast<float>(h0), 0.0F);
        sum = fmaf(w1, static_cast<float>(h1), sum);
        sum = fmaf(w2, static_cast<float>(h2), sum);
        sum = fmaf(w3, static_cast<float>(current), sum);
        output[static_cast<std::size_t>(token) * channels + channel] =
            hip_bfloat16(sum / (1.0F + expf(-sum)));
        h0 = h1;
        h1 = h2;
        h2 = current;
    }
    history_out[channel] = h0;
    history_out[channels + channel] = h1;
    history_out[2U * channels + channel] = h2;
}

template <std::uint32_t TokenTile>
__global__ __launch_bounds__(kBlock) void causal_conv1d_silu_prefill_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight,
    const hip_bfloat16* history_in, hip_bfloat16* output,
    std::uint32_t channels, std::uint32_t tokens) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) { return; }
    const std::uint32_t token_begin = blockIdx.y * TokenTile;
    const float w0 = static_cast<float>(weight[channel]);
    const float w1 = static_cast<float>(weight[channels + channel]);
    const float w2 = static_cast<float>(weight[2U * channels + channel]);
    const float w3 = static_cast<float>(weight[3U * channels + channel]);
#pragma unroll
    for (std::uint32_t local = 0; local < TokenTile; ++local) {
        const std::uint32_t token = token_begin + local;
        if (token >= tokens) { break; }
        const auto represented = [&](std::int64_t source_token) {
            if (source_token < 0) {
                return history_in[static_cast<std::size_t>(source_token + 3) * channels + channel];
            }
            return input[static_cast<std::size_t>(source_token) * channels + channel];
        };
        const hip_bfloat16 h0 = represented(static_cast<std::int64_t>(token) - 3);
        const hip_bfloat16 h1 = represented(static_cast<std::int64_t>(token) - 2);
        const hip_bfloat16 h2 = represented(static_cast<std::int64_t>(token) - 1);
        const hip_bfloat16 current = represented(token);
        float sum = fmaf(w0, static_cast<float>(h0), 0.0F);
        sum = fmaf(w1, static_cast<float>(h1), sum);
        sum = fmaf(w2, static_cast<float>(h2), sum);
        sum = fmaf(w3, static_cast<float>(current), sum);
        output[static_cast<std::size_t>(token) * channels + channel] =
            hip_bfloat16(sum / (1.0F + expf(-sum)));
    }
}

__global__ __launch_bounds__(kBlock) void causal_conv1d_publish_short_history_qualification_kernel(
    const hip_bfloat16* input, const hip_bfloat16* history_in,
    hip_bfloat16* history_out, std::uint32_t channels, std::uint32_t tokens) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) { return; }
    hip_bfloat16 h0 = history_in[channel];
    hip_bfloat16 h1 = history_in[channels + channel];
    hip_bfloat16 h2 = history_in[2U * channels + channel];
    for (std::uint32_t token = 0; token < tokens; ++token) {
        h0 = h1;
        h1 = h2;
        h2 = input[static_cast<std::size_t>(token) * channels + channel];
    }
    history_out[channel] = h0;
    history_out[channels + channel] = h1;
    history_out[2U * channels + channel] = h2;
}

template <std::uint32_t TokenTile>
hipError_t launch_causal_conv1d_silu_prefill(
    const hip_bfloat16* input, const hip_bfloat16* weight,
    const hip_bfloat16* history_in, hip_bfloat16* history_out,
    hip_bfloat16* output, std::uint32_t channels, std::uint32_t tokens,
    hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(causal_conv1d_silu_prefill_kernel<TokenTile>),
                       dim3((channels + kBlock - 1U) / kBlock,
                            (tokens + TokenTile - 1U) / TokenTile),
                       dim3(kBlock), 0, stream, input, weight, history_in, output,
                       channels, tokens);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) { return status; }
    if (tokens >= 3U) {
        return hipMemcpyAsync(history_out,
                              input + static_cast<std::size_t>(tokens - 3U) * channels,
                              static_cast<std::size_t>(3U) * channels * sizeof(hip_bfloat16),
                              hipMemcpyDeviceToDevice, stream);
    } else {
        hipLaunchKernelGGL(causal_conv1d_publish_short_history_qualification_kernel,
                           linear_grid(channels), dim3(kBlock), 0, stream, input,
                           history_in, history_out, channels, tokens);
        return hipGetLastError();
    }
}

// value_stride is kProjectionValueZRows for the combined [value,z] rows and kProjectionValueRows
// for a separately projected value.
template <std::uint32_t ValueStride = kProjectionValueZRows>
__device__ __forceinline__ hip_bfloat16 projection_value(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z, std::size_t column,
    std::uint32_t channel) {
    if (channel < kProjectionQueryRows + kProjectionKeyRows) {
        return query_key[column * (kProjectionQueryRows + kProjectionKeyRows) + channel];
    }
    return value_z[column * ValueStride + (channel - kProjectionQueryRows - kProjectionKeyRows)];
}

__device__ __forceinline__ void publish_projection_output(
    hip_bfloat16 output, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    std::size_t column, std::uint32_t channel) {
    if (channel < kProjectionQueryRows) {
        query[column * kProjectionQueryRows + channel] = output;
    } else if (channel < kProjectionQueryRows + kProjectionKeyRows) {
        key[column * kProjectionKeyRows + channel - kProjectionQueryRows] = output;
    } else {
        value[column * kProjectionValueRows +
              channel - kProjectionQueryRows - kProjectionKeyRows] = output;
    }
}

template <std::uint32_t TokenTile>
__global__ __launch_bounds__(kBlock) void projection_conv_prefill_direct_scatter_kernel(
    const hip_bfloat16* query_key, const hip_bfloat16* projected_value,
    const hip_bfloat16* conv_weight, const hip_bfloat16* history_in,
    hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kProjectionChannels) { return; }
    const std::uint32_t token_begin = blockIdx.y * TokenTile;
    const float w0 = static_cast<float>(conv_weight[channel]);
    const float w1 = static_cast<float>(conv_weight[kProjectionChannels + channel]);
    const float w2 = static_cast<float>(conv_weight[2U * kProjectionChannels + channel]);
    const float w3 = static_cast<float>(conv_weight[3U * kProjectionChannels + channel]);
#pragma unroll
    for (std::uint32_t local = 0; local < TokenTile; ++local) {
        const std::uint32_t token = token_begin + local;
        const auto represented = [&](std::int32_t source_token) {
            if (source_token < 0) {
                return history_in[static_cast<std::size_t>(source_token + 3) *
                                      kProjectionChannels +
                                  channel];
            }
            return projection_value<kProjectionValueRows>(
                query_key, projected_value, static_cast<std::size_t>(source_token), channel);
        };
        const float h0 = static_cast<float>(represented(static_cast<std::int32_t>(token) - 3));
        const float h1 = static_cast<float>(represented(static_cast<std::int32_t>(token) - 2));
        const float h2 = static_cast<float>(represented(static_cast<std::int32_t>(token) - 1));
        const float current = static_cast<float>(represented(static_cast<std::int32_t>(token)));
        float sum = fmaf(w0, h0, 0.0F);
        sum = fmaf(w1, h1, sum);
        sum = fmaf(w2, h2, sum);
        sum = fmaf(w3, current, sum);
        publish_projection_output(hip_bfloat16(sum / (1.0F + expf(-sum))), query, key, value,
                                  token, channel);
    }
}

__global__ __launch_bounds__(kBlock) void projection_prefill_publish_history_kernel(
    const hip_bfloat16* query_key, const hip_bfloat16* projected_value,
    hip_bfloat16* history_out) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kProjectionChannels) { return; }
#pragma unroll
    for (std::uint32_t history_column = 0; history_column < 3U; ++history_column) {
        constexpr std::uint32_t kTailBegin = kProjectionPrefillTokens - 3U;
        history_out[static_cast<std::size_t>(history_column) * kProjectionChannels + channel] =
            projection_value<kProjectionValueRows>(query_key, projected_value,
                                                   kTailBegin + history_column, channel);
    }
}

__global__ void projection_conv_snapshot_kernel(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z,
    const hip_bfloat16* conv_weight, hip_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* snapshot_base_slots, hip_bfloat16* query, hip_bfloat16* key,
    hip_bfloat16* value, hip_bfloat16* z, std::uint32_t width,
    std::uint32_t state_slots) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t batch_index = blockIdx.y;
    if (channel >= kProjectionChannels) { return; }

    const std::int32_t valid_raw = valid_columns == nullptr
        ? static_cast<std::int32_t>(width)
        : valid_columns[batch_index];
    const std::uint32_t valid = valid_raw <= 0
        ? 0U
        : min(static_cast<std::uint32_t>(valid_raw), width);
    const std::int32_t initial_raw = initial_state_slots[batch_index];
    const std::int32_t base_raw = snapshot_base_slots[batch_index];
    const bool valid_initial = initial_raw >= 0 &&
        static_cast<std::uint32_t>(initial_raw) < state_slots;
    const bool valid_destination = base_raw >= 0 &&
        static_cast<std::uint64_t>(base_raw) + valid <= state_slots;

    const std::size_t initial_base = valid_initial
        ? static_cast<std::size_t>(initial_raw) * 3U * kProjectionChannels
        : 0U;
    float h0 = valid_initial
        ? static_cast<float>(conv_states[initial_base + channel]) : 0.0F;
    float h1 = valid_initial
        ? static_cast<float>(conv_states[initial_base + kProjectionChannels + channel]) : 0.0F;
    float h2 = valid_initial
        ? static_cast<float>(conv_states[initial_base + 2U * kProjectionChannels + channel]) : 0.0F;
    const float w0 = static_cast<float>(conv_weight[channel]);
    const float w1 = static_cast<float>(conv_weight[kProjectionChannels + channel]);
    const float w2 = static_cast<float>(conv_weight[2U * kProjectionChannels + channel]);
    const float w3 = static_cast<float>(conv_weight[3U * kProjectionChannels + channel]);

    for (std::uint32_t token = 0; token < width; ++token) {
        const std::size_t column = static_cast<std::size_t>(batch_index) * width + token;
        if (channel < kProjectionValueRows) {
            z[column * kProjectionValueRows + channel] =
                value_z[column * kProjectionValueZRows + kProjectionValueRows + channel];
        }
        if (token >= valid || !valid_initial || !valid_destination) {
            publish_projection_output(hip_bfloat16(0.0F), query, key, value, column, channel);
            continue;
        }
        const hip_bfloat16 represented = projection_value(query_key, value_z, column, channel);
        const float current = static_cast<float>(represented);
        float sum = fmaf(w0, h0, 0.0F);
        sum = fmaf(w1, h1, sum);
        sum = fmaf(w2, h2, sum);
        sum = fmaf(w3, current, sum);
        publish_projection_output(hip_bfloat16(sum / (1.0F + expf(-sum))), query, key, value,
                                  column, channel);
        h0 = h1;
        h1 = h2;
        h2 = current;
        const std::size_t destination_base =
            (static_cast<std::size_t>(base_raw) + token) * 3U * kProjectionChannels;
        conv_states[destination_base + channel] = hip_bfloat16(h0);
        conv_states[destination_base + kProjectionChannels + channel] = hip_bfloat16(h1);
        conv_states[destination_base + 2U * kProjectionChannels + channel] = hip_bfloat16(h2);
    }
}

__global__ void projection_conv_record_kernel(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z,
    const hip_bfloat16* conv_weight, const hip_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* parent_index, hip_bfloat16* conv_record, hip_bfloat16* query,
    hip_bfloat16* key, hip_bfloat16* value, hip_bfloat16* z, std::uint32_t width,
    std::uint32_t state_slots) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t batch_index = blockIdx.y;
    if (channel >= kProjectionChannels) { return; }

    const std::int32_t valid_raw = valid_columns == nullptr
        ? static_cast<std::int32_t>(width)
        : valid_columns[batch_index];
    const std::uint32_t valid = valid_raw <= 0
        ? 0U
        : min(static_cast<std::uint32_t>(valid_raw), width);
    const std::int32_t initial_raw = initial_state_slots[batch_index];
    const bool valid_initial = initial_raw >= 0 &&
        static_cast<std::uint32_t>(initial_raw) < state_slots;
    const std::size_t initial_base = valid_initial
        ? static_cast<std::size_t>(initial_raw) * 3U * kProjectionChannels
        : 0U;
    const float checkpoint0 = valid_initial
        ? static_cast<float>(conv_states[initial_base + channel]) : 0.0F;
    const float checkpoint1 = valid_initial
        ? static_cast<float>(conv_states[initial_base + kProjectionChannels + channel]) : 0.0F;
    const float checkpoint2 = valid_initial
        ? static_cast<float>(conv_states[initial_base + 2U * kProjectionChannels + channel]) : 0.0F;
    const float w0 = static_cast<float>(conv_weight[channel]);
    const float w1 = static_cast<float>(conv_weight[kProjectionChannels + channel]);
    const float w2 = static_cast<float>(conv_weight[2U * kProjectionChannels + channel]);
    const float w3 = static_cast<float>(conv_weight[3U * kProjectionChannels + channel]);
    float saved0[kMaximumVerifyWidth];
    float saved1[kMaximumVerifyWidth];
    float saved2[kMaximumVerifyWidth];

    for (std::uint32_t token = 0; token < width; ++token) {
        const std::size_t column = static_cast<std::size_t>(batch_index) * width + token;
        if (channel < kProjectionValueRows) {
            z[column * kProjectionValueRows + channel] =
                value_z[column * kProjectionValueZRows + kProjectionValueRows + channel];
        }
        if (token >= valid || !valid_initial) {
            conv_record[column * kProjectionChannels + channel] = hip_bfloat16(0.0F);
            publish_projection_output(hip_bfloat16(0.0F), query, key, value, column, channel);
            continue;
        }

        const std::int32_t parent = parent_index == nullptr ?
            (token == 0 ? -1 : static_cast<std::int32_t>(token - 1U)) : parent_index[column];
        if (parent >= static_cast<std::int32_t>(token)) {
            conv_record[column * kProjectionChannels + channel] = hip_bfloat16(0.0F);
            publish_projection_output(hip_bfloat16(0.0F), query, key, value, column, channel);
            continue;
        }
        float h0 = parent < 0 ? checkpoint0 : saved0[parent];
        float h1 = parent < 0 ? checkpoint1 : saved1[parent];
        float h2 = parent < 0 ? checkpoint2 : saved2[parent];
        const hip_bfloat16 represented = projection_value(query_key, value_z, column, channel);
        conv_record[column * kProjectionChannels + channel] = represented;
        const float current = static_cast<float>(represented);
        float sum = fmaf(w0, h0, 0.0F);
        sum = fmaf(w1, h1, sum);
        sum = fmaf(w2, h2, sum);
        sum = fmaf(w3, current, sum);
        publish_projection_output(hip_bfloat16(sum / (1.0F + expf(-sum))), query, key, value,
                                  column, channel);
        saved0[token] = h1;
        saved1[token] = h2;
        saved2[token] = current;
    }
}

__global__ void control_gates_kernel(const hip_bfloat16* a, const hip_bfloat16* b,
                                     const float* a_log, const float* dt_bias, float* g,
                                     float* beta, std::size_t count,
                                     std::uint32_t value_heads) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const std::uint32_t head = static_cast<std::uint32_t>(index % value_heads);
    const float av = static_cast<float>(a[index]) + dt_bias[head];
    const float softplus = av > 20.0F ? av : log1pf(expf(av));
    g[index] = -expf(a_log[head]) * softplus;
    const float bv = static_cast<float>(b[index]);
    beta[index] = 1.0F / (1.0F + expf(-bv));
}

__global__ __launch_bounds__(kBlock) void bf16_projected_control_t1_kernel(
    const hip_bfloat16* hidden, const hip_bfloat16* a_weight,
    const hip_bfloat16* b_weight, const float* a_log, const float* dt_bias,
    float* g, float* beta) {
    const std::uint32_t head = blockIdx.x;
    hidden += static_cast<std::size_t>(blockIdx.y) * kControlColumns;
    const std::size_t output = static_cast<std::size_t>(blockIdx.y) * kControlHeads + head;
    const std::size_t base = static_cast<std::size_t>(head) * kControlColumns;
    float a_partial = 0.0F;
    float b_partial = 0.0F;
    for (std::uint32_t k = threadIdx.x; k < kControlColumns; k += blockDim.x) {
        const float x = static_cast<float>(hidden[k]);
        a_partial = fmaf(x, static_cast<float>(a_weight[base + k]), a_partial);
        b_partial = fmaf(x, static_cast<float>(b_weight[base + k]), b_partial);
    }
    __shared__ float a_reduction[kBlock];
    __shared__ float b_reduction[kBlock];
    a_reduction[threadIdx.x] = a_partial;
    b_reduction[threadIdx.x] = b_partial;
    __syncthreads();
    for (std::uint32_t width = kBlock / 2U; width != 0U; width >>= 1U) {
        if (threadIdx.x < width) {
            a_reduction[threadIdx.x] += a_reduction[threadIdx.x + width];
            b_reduction[threadIdx.x] += b_reduction[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const hip_bfloat16 ar = static_cast<hip_bfloat16>(a_reduction[0]);
        const hip_bfloat16 br = static_cast<hip_bfloat16>(b_reduction[0]);
        const float av = static_cast<float>(ar) + dt_bias[head];
        const float softplus = av > 20.0F ? av : log1pf(expf(av));
        g[output] = -expf(a_log[head]) * softplus;
        const float bv = static_cast<float>(br);
        beta[output] = 1.0F / (1.0F + expf(-bv));
    }
}

using ControlBf16x8 = short __attribute__((ext_vector_type(8)));
using ControlF32x8 = float __attribute__((ext_vector_type(8)));

// The launcher admits only 16-byte aligned hidden and weight bases; rows are 10240 bytes.
__device__ __forceinline__ ControlBf16x8 load_control_bf16x8(const hip_bfloat16* source) {
    return *reinterpret_cast<const ControlBf16x8*>(source);
}

// Tokens are WMMA rows and the stacked [a;b] weight rows are WMMA columns, so one lane holds
// a[h,t] and b[h,t] for the same head in tiles i and i+3 and applies the control formula in place.
__global__ __launch_bounds__(kControlWideSplits * 32U) void bf16_projected_control_wide_kernel(
    const hip_bfloat16* hidden, const hip_bfloat16* a_weight, const hip_bfloat16* b_weight,
    const float* a_log, const float* dt_bias, float* g, float* beta, std::uint32_t tokens) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t split = threadIdx.x >> 5U;
    const std::uint32_t lane_row = lane & 15U;
    const std::uint32_t lane_half = lane >> 4U;
    const std::uint32_t token_base = blockIdx.x * kControlWideTokens;
    const std::uint32_t column = split * kControlWideSplitColumns + lane_half * 8U;
    // Rows past the token extent load the last valid token; their results are never stored.
    const hip_bfloat16* input = hidden +
        static_cast<std::size_t>(min(token_base + lane_row, tokens - 1U)) * kControlColumns +
        column;
    const hip_bfloat16* weight[kControlWideTiles];
#pragma unroll
    for (std::uint32_t tile = 0; tile < kControlWideTiles; ++tile) {
        const std::uint32_t head = (tile % 3U) * 16U + lane_row;
        weight[tile] = (tile < 3U ? a_weight : b_weight) +
            static_cast<std::size_t>(head) * kControlColumns + column;
    }
    ControlF32x8 accumulator[kControlWideTiles]{};
#pragma unroll 4
    for (std::uint32_t offset = 0; offset < kControlWideSplitColumns; offset += 16U) {
        const ControlBf16x8 x = load_control_bf16x8(input + offset);
#pragma unroll
        for (std::uint32_t tile = 0; tile < kControlWideTiles; ++tile) {
            accumulator[tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                x, load_control_bf16x8(weight[tile] + offset), accumulator[tile]);
        }
    }
    // Every wave publishes its K-split partials; all threads then reduce them in split order and
    // apply the control formula to three of the 48x16 (head, token) outputs each.
    __shared__ float partial[kControlWideSplits][kControlWideTiles][8][32];
#pragma unroll
    for (std::uint32_t tile = 0; tile < kControlWideTiles; ++tile) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            partial[split][tile][element][lane] = accumulator[tile][element];
        }
    }
    __syncthreads();
    for (std::uint32_t item = threadIdx.x; item < 3U * 8U * 32U;
         item += kControlWideSplits * 32U) {
        const std::uint32_t item_lane = item % 32U;
        const std::uint32_t element = (item / 32U) % 8U;
        const std::uint32_t tile = item / 256U;
        const std::uint32_t token = token_base + (item_lane >> 4U) * 8U + element;
        if (token >= tokens) continue;
        float a = 0.0F, b = 0.0F;
#pragma unroll
        for (std::uint32_t source = 0; source < kControlWideSplits; ++source) {
            a += partial[source][tile][element][item_lane];
            b += partial[source][tile + 3U][element][item_lane];
        }
        const std::uint32_t head = tile * 16U + (item_lane & 15U);
        const std::size_t output = static_cast<std::size_t>(token) * kControlHeads + head;
        const float av = static_cast<float>(hip_bfloat16(a)) + dt_bias[head];
        const float softplus = av > 20.0F ? av : log1pf(expf(av));
        g[output] = -expf(a_log[head]) * softplus;
        const float bv = static_cast<float>(hip_bfloat16(b));
        beta[output] = 1.0F / (1.0F + expf(-bv));
    }
}

__global__ void copy_fp32_kernel(const float* source, float* destination, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = source[index]; }
}

__global__ void recurrent_kernel(const hip_bfloat16* q, const hip_bfloat16* k,
                                 const hip_bfloat16* v, const float* g, const float* beta,
                                 const float* state_in, float* state_out, hip_bfloat16* output,
                                 std::uint32_t key_dim, std::uint32_t value_dim,
                                 std::uint32_t qk_heads, std::uint32_t value_heads,
                                 std::uint32_t tokens, float scale, float epsilon) {
    const std::uint32_t value_head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t qk_head = value_head / (value_heads / qk_heads);
    __shared__ float q_buffer[kMaxKeyDim];
    __shared__ float k_buffer[kMaxKeyDim];
    __shared__ float q_reduce[kMaxKeyDim];
    __shared__ float k_reduce[kMaxKeyDim];
    __shared__ float alpha;
    __shared__ float delta;

    const std::size_t head_state_base = static_cast<std::size_t>(value_head) * value_dim * key_dim;
    const std::size_t head_state_elements = static_cast<std::size_t>(value_dim) * key_dim;
    for (std::size_t index = tid; index < head_state_elements; index += blockDim.x) {
        state_out[head_state_base + index] = state_in[head_state_base + index];
    }
    __syncthreads();

    for (std::uint32_t token = 0; token < tokens; ++token) {
        const std::size_t qk_base =
            (static_cast<std::size_t>(token) * qk_heads + qk_head) * key_dim;
        float q_value = 0.0F;
        float k_value = 0.0F;
        if (tid < key_dim) {
            q_value = static_cast<float>(q[qk_base + tid]);
            k_value = static_cast<float>(k[qk_base + tid]);
        }
        q_buffer[tid] = q_value;
        k_buffer[tid] = k_value;
        q_reduce[tid] = q_value * q_value;
        k_reduce[tid] = k_value * k_value;
        __syncthreads();
        for (std::uint32_t stride = kBlock / 2U; stride != 0U; stride >>= 1U) {
            if (tid < stride) {
                q_reduce[tid] += q_reduce[tid + stride];
                k_reduce[tid] += k_reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid < key_dim) {
            q_buffer[tid] *= rsqrtf(q_reduce[0] + epsilon);
            k_buffer[tid] *= rsqrtf(k_reduce[0] + epsilon);
        }
        __syncthreads();

        if (tid == 0) {
            const std::size_t control_index = static_cast<std::size_t>(token) * value_heads + value_head;
            alpha = expf(g[control_index]);
        }
        __syncthreads();
        for (std::uint32_t value_row = 0; value_row < value_dim; ++value_row) {
            const std::size_t state_base = head_state_base + static_cast<std::size_t>(value_row) * key_dim;
            q_reduce[tid] = tid < key_dim ? state_out[state_base + tid] * k_buffer[tid] : 0.0F;
            __syncthreads();
            for (std::uint32_t stride = kBlock / 2U; stride != 0U; stride >>= 1U) {
                if (tid < stride) { q_reduce[tid] += q_reduce[tid + stride]; }
                __syncthreads();
            }
            if (tid == 0) {
                const float value = static_cast<float>(
                    v[(static_cast<std::size_t>(token) * value_heads + value_head) * value_dim + value_row]);
                const std::size_t control_index = static_cast<std::size_t>(token) * value_heads + value_head;
                delta = beta[control_index] * (value - alpha * q_reduce[0]);
            }
            __syncthreads();
            if (tid < key_dim) {
                state_out[state_base + tid] = alpha * state_out[state_base + tid] + delta * k_buffer[tid];
            }
            __syncthreads();

            q_reduce[tid] = tid < key_dim ? state_out[state_base + tid] * q_buffer[tid] : 0.0F;
            __syncthreads();
            for (std::uint32_t stride = kBlock / 2U; stride != 0U; stride >>= 1U) {
                if (tid < stride) { q_reduce[tid] += q_reduce[tid + stride]; }
                __syncthreads();
            }
            if (tid == 0) {
                output[(static_cast<std::size_t>(token) * value_heads + value_head) * value_dim + value_row] =
                    hip_bfloat16(q_reduce[0] * scale);
            }
            __syncthreads();
        }
    }
}

} // namespace

hipError_t causal_conv1d_silu_bf16(const hip_bfloat16* input, const hip_bfloat16* weight,
                                   const hip_bfloat16* history_in, hip_bfloat16* history_out,
                                   hip_bfloat16* output, std::uint32_t channels,
                                   std::uint32_t tokens, hipStream_t stream) noexcept {
    const bool valid = input != nullptr && weight != nullptr && history_in != nullptr &&
        history_out != nullptr && output != nullptr && channels != 0 && tokens != 0 &&
        valid_stream(stream);
    if (!valid) { return hipErrorInvalidValue; }
    if (tokens >= 64U) {
        return launch_causal_conv1d_silu_prefill<4U>(
            input, weight, history_in, history_out, output, channels, tokens, stream);
    }
    hipLaunchKernelGGL(causal_conv1d_silu_kernel, linear_grid(channels), dim3(kBlock), 0,
                       stream, input, weight, history_in, history_out, output, channels, tokens);
    return hipGetLastError();
}

hipError_t causal_conv1d_silu_incumbent_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight,
    const hip_bfloat16* history_in, hip_bfloat16* history_out,
    hip_bfloat16* output, std::uint32_t channels, std::uint32_t tokens,
    hipStream_t stream) noexcept {
    return checked_launch(input != nullptr && weight != nullptr && history_in != nullptr &&
                              history_out != nullptr && output != nullptr && channels != 0 &&
                              tokens != 0 && valid_stream(stream),
                          [&] {
                              hipLaunchKernelGGL(causal_conv1d_silu_kernel,
                                                 linear_grid(channels), dim3(kBlock), 0, stream,
                                                 input, weight, history_in, history_out, output,
                                                 channels, tokens);
                          });
}

hipError_t causal_conv1d_silu_prefill_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight,
    const hip_bfloat16* history_in, hip_bfloat16* history_out,
    hip_bfloat16* output, std::uint32_t channels, std::uint32_t tokens,
    std::uint32_t token_tile, hipStream_t stream) noexcept {
    const bool valid_tile = token_tile == 4U || token_tile == 8U ||
        token_tile == 16U || token_tile == 32U;
    if (input == nullptr || weight == nullptr || history_in == nullptr ||
        history_out == nullptr || output == nullptr || channels == 0 || tokens == 0 ||
        !valid_tile || !valid_stream(stream)) {
        return hipErrorInvalidValue;
    }
    switch (token_tile) {
    case 4U:
        return launch_causal_conv1d_silu_prefill<4U>(
            input, weight, history_in, history_out, output, channels, tokens, stream);
    case 8U:
        return launch_causal_conv1d_silu_prefill<8U>(
            input, weight, history_in, history_out, output, channels, tokens, stream);
    case 16U:
        return launch_causal_conv1d_silu_prefill<16U>(
            input, weight, history_in, history_out, output, channels, tokens, stream);
    case 32U:
        return launch_causal_conv1d_silu_prefill<32U>(
            input, weight, history_in, history_out, output, channels, tokens, stream);
    default: return hipErrorInvalidValue;
    }
}

hipError_t projection_conv_prefill_p2048_direct_scatter_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* projected_value,
    const hip_bfloat16* conv_weight, const hip_bfloat16* history_in,
    hip_bfloat16* history_out, hip_bfloat16* query, hip_bfloat16* key,
    hip_bfloat16* value, std::uint32_t tokens, hipStream_t stream) noexcept {
    if (query_key == nullptr || projected_value == nullptr || conv_weight == nullptr ||
        history_in == nullptr || history_out == nullptr || query == nullptr || key == nullptr ||
        value == nullptr || tokens != kProjectionPrefillTokens || !valid_stream(stream)) {
        return hipErrorInvalidValue;
    }
    constexpr std::size_t kQueryKeyElements =
        static_cast<std::size_t>(kProjectionPrefillTokens) *
        (kProjectionQueryRows + kProjectionKeyRows);
    constexpr std::size_t kConvolutionWeightElements =
        static_cast<std::size_t>(kProjectionChannels) * 4U;
    constexpr std::size_t kHistoryElements =
        static_cast<std::size_t>(kProjectionChannels) * 3U;
    constexpr std::size_t kQueryElements =
        static_cast<std::size_t>(kProjectionPrefillTokens) * kProjectionQueryRows;
    constexpr std::size_t kKeyElements =
        static_cast<std::size_t>(kProjectionPrefillTokens) * kProjectionKeyRows;
    constexpr std::size_t kValueElements =
        static_cast<std::size_t>(kProjectionPrefillTokens) * kProjectionValueRows;
    const std::array<ByteRange, 8> ranges{
        byte_range(query_key, kQueryKeyElements),
        byte_range(projected_value, kValueElements),
        byte_range(conv_weight, kConvolutionWeightElements),
        byte_range(history_in, kHistoryElements),
        byte_range(history_out, kHistoryElements),
        byte_range(query, kQueryElements),
        byte_range(key, kKeyElements),
        byte_range(value, kValueElements),
    };
    for (std::size_t lhs = 0; lhs < ranges.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < ranges.size(); ++rhs) {
            const bool exact_history_alias = lhs == 3U && rhs == 4U &&
                ranges[lhs].first == ranges[rhs].first;
            if (!exact_history_alias && overlaps(ranges[lhs], ranges[rhs])) {
                return hipErrorInvalidValue;
            }
        }
    }
    constexpr std::uint32_t kTokenTile = 4U;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(projection_conv_prefill_direct_scatter_kernel<kTokenTile>),
        dim3((kProjectionChannels + kBlock - 1U) / kBlock,
             kProjectionPrefillTokens / kTokenTile),
        dim3(kBlock), 0, stream, query_key, projected_value, conv_weight, history_in, query, key,
        value);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) { return status; }
    hipLaunchKernelGGL(projection_prefill_publish_history_kernel,
                       dim3((kProjectionChannels + kBlock - 1U) / kBlock), dim3(kBlock), 0,
                       stream, query_key, projected_value, history_out);
    return hipGetLastError();
}

hipError_t projection_conv_snapshot_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z,
    const hip_bfloat16* conv_weight, hip_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* snapshot_base_slots, hip_bfloat16* query, hip_bfloat16* key,
    hip_bfloat16* value, hip_bfloat16* z, std::uint32_t width, std::uint32_t batch,
    std::uint32_t state_slots, hipStream_t stream) noexcept {
    const bool valid_geometry = width != 0 && batch != 0 && batch <= kMaximumBatch &&
        (batch == 1 || width <= kMaximumVerifyWidth) && state_slots != 0;
    return checked_launch(
        query_key != nullptr && value_z != nullptr && conv_weight != nullptr &&
            conv_states != nullptr && initial_state_slots != nullptr &&
            snapshot_base_slots != nullptr && query != nullptr && key != nullptr &&
            value != nullptr && z != nullptr && valid_geometry && valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(projection_conv_snapshot_kernel,
                               dim3((kProjectionChannels + kBlock - 1U) / kBlock, batch),
                               dim3(kBlock), 0, stream, query_key, value_z, conv_weight,
                               conv_states, valid_columns, initial_state_slots,
                               snapshot_base_slots, query, key, value, z, width, state_slots);
        });
}

hipError_t projection_conv_record_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z,
    const hip_bfloat16* conv_weight, const hip_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* parent_index, hip_bfloat16* conv_record, hip_bfloat16* query,
    hip_bfloat16* key, hip_bfloat16* value, hip_bfloat16* z, std::uint32_t width,
    std::uint32_t batch, std::uint32_t state_slots, hipStream_t stream) noexcept {
    const bool valid_geometry = width >= 2U && width <= kMaximumVerifyWidth && batch != 0 &&
        batch <= kMaximumBatch && state_slots != 0;
    return checked_launch(
        query_key != nullptr && value_z != nullptr && conv_weight != nullptr &&
            conv_states != nullptr && initial_state_slots != nullptr && conv_record != nullptr &&
            query != nullptr && key != nullptr && value != nullptr && z != nullptr &&
            valid_geometry && valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(projection_conv_record_kernel,
                               dim3((kProjectionChannels + kBlock - 1U) / kBlock, batch),
                               dim3(kBlock), 0, stream, query_key, value_z, conv_weight,
                               conv_states, valid_columns, initial_state_slots, parent_index,
                               conv_record, query, key, value, z, width, state_slots);
        });
}

hipError_t control_gates_bf16(const hip_bfloat16* a, const hip_bfloat16* b, const float* a_log,
                              const float* dt_bias, float* g, float* beta,
                              std::uint32_t value_heads, std::uint32_t tokens,
                              hipStream_t stream) noexcept {
    const std::size_t count = static_cast<std::size_t>(value_heads) * tokens;
    return checked_launch(a != nullptr && b != nullptr && a_log != nullptr && dt_bias != nullptr &&
                              g != nullptr && beta != nullptr && value_heads != 0 && tokens != 0 &&
                              valid_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(control_gates_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, a, b, a_log, dt_bias, g, beta, count, value_heads); });
}

hipError_t bf16_projected_control(
    const hip_bfloat16* hidden, const hip_bfloat16* a_weight,
    const hip_bfloat16* b_weight, const float* a_log, const float* dt_bias,
    float* g, float* beta, std::uint32_t tokens, hipStream_t stream) noexcept {
    if (tokens < 1U) return hipErrorInvalidValue;
    const std::size_t hidden_bytes =
        static_cast<std::size_t>(kControlColumns) * tokens * sizeof(hip_bfloat16);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kControlHeads) * kControlColumns * sizeof(hip_bfloat16);
    constexpr std::size_t vector_bytes = kControlHeads * sizeof(float);
    const bool pointers_valid = hidden != nullptr && a_weight != nullptr && b_weight != nullptr &&
        a_log != nullptr && dt_bias != nullptr && g != nullptr && beta != nullptr &&
        valid_stream(stream);
    if (!pointers_valid) { return hipErrorInvalidValue; }
    const auto aligned16 = [](const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
    };
    if (tokens > kControlT1MaximumTokens &&
        (!aligned16(hidden) || !aligned16(a_weight) || !aligned16(b_weight))) {
        return hipErrorInvalidValue;
    }
    const std::array<ByteRange, 7> ranges{
        byte_range(hidden, hidden_bytes / sizeof(hip_bfloat16)),
        byte_range(a_weight, weight_bytes / sizeof(hip_bfloat16)),
        byte_range(b_weight, weight_bytes / sizeof(hip_bfloat16)),
        byte_range(a_log, vector_bytes / sizeof(float)),
        byte_range(dt_bias, vector_bytes / sizeof(float)),
        byte_range(g, tokens * vector_bytes / sizeof(float)),
        byte_range(beta, tokens * vector_bytes / sizeof(float)),
    };
    for (std::size_t first = 0; first < ranges.size(); ++first) {
        for (std::size_t second = first + 1; second < ranges.size(); ++second) {
            if (overlaps(ranges[first], ranges[second])) return hipErrorInvalidValue;
        }
    }
    return checked_launch(true, [&] {
        if (tokens <= kControlT1MaximumTokens) {
            hipLaunchKernelGGL(bf16_projected_control_t1_kernel, dim3(kControlHeads, tokens),
                               dim3(kBlock), 0, stream, hidden, a_weight, b_weight, a_log,
                               dt_bias, g, beta);
        } else {
            hipLaunchKernelGGL(bf16_projected_control_wide_kernel,
                               dim3((tokens + kControlWideTokens - 1U) / kControlWideTokens),
                               dim3(kControlWideSplits * 32U), 0, stream, hidden, a_weight,
                               b_weight, a_log, dt_bias, g, beta, tokens);
        }
    });
}

hipError_t copy_state_fp32(const float* source, float* destination, std::size_t count,
                           hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && destination != nullptr && source != destination &&
                              valid_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(copy_fp32_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, source, destination, count); });
}

hipError_t recurrent_bf16_fp32(const hip_bfloat16* q, const hip_bfloat16* k,
                               const hip_bfloat16* v, const float* g, const float* beta,
                               const float* state_in, float* state_out, hip_bfloat16* output,
                               std::uint32_t key_dim, std::uint32_t value_dim,
                               std::uint32_t qk_heads, std::uint32_t value_heads,
                               std::uint32_t tokens, float scale, float epsilon,
                               hipStream_t stream) noexcept {
    const bool valid_geometry = key_dim != 0 && key_dim <= kMaxKeyDim && value_dim != 0 &&
        qk_heads != 0 && value_heads != 0 && value_heads % qk_heads == 0 && tokens != 0;
    return checked_launch(q != nullptr && k != nullptr && v != nullptr && g != nullptr && beta != nullptr &&
                              state_in != nullptr && state_out != nullptr && output != nullptr && valid_geometry &&
                              std::isfinite(scale) && std::isfinite(epsilon) && epsilon > 0.0F &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(recurrent_kernel, dim3(value_heads), dim3(kBlock),
                                                   0, stream, q, k, v, g, beta, state_in, state_out, output,
                                                   key_dim, value_dim, qk_heads, value_heads, tokens, scale,
                                                   epsilon); });
}

} // namespace ninfer::ops::r9700::gdn
