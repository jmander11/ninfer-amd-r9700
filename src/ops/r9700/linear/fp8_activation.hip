#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_encode_impl.h"

#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::r9700::linear {
namespace {

using fp8_encode::encode_row8;
using fp8_encode::encode_rows;
using fp8_encode::unpack8;

constexpr std::uint32_t kWaveSize        = 32U;
constexpr std::uint32_t kBlockSize       = 256U;
constexpr std::uint32_t kWavesPerBlock   = kBlockSize / kWaveSize;
constexpr std::uint32_t kQuantizeResidentBlocks = 128U;
constexpr std::uint32_t kColumnAlignment = 128U;
constexpr std::size_t kPlaneAlignment    = 256U;
constexpr float kE4M3FiniteMaximum       = 448.0F;

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* out) noexcept {
    if (out == nullptr || (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* out) noexcept {
    if (out == nullptr || rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t* out) noexcept {
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) return false;
    *out = (value + alignment - 1U) / alignment * alignment;
    return true;
}

bool padded_columns(std::uint32_t columns, std::uint32_t* out) noexcept {
    if (out == nullptr || columns == 0U ||
        columns > std::numeric_limits<std::uint32_t>::max() - (kColumnAlignment - 1U)) {
        return false;
    }
    *out = (columns + kColumnAlignment - 1U) / kColumnAlignment * kColumnAlignment;
    return true;
}

struct WorkspaceGeometry {
    std::uint32_t padded      = 0;
    std::size_t code_bytes    = 0;
    std::size_t scale_offset = 0;
    std::size_t scale_bytes   = 0;
    std::size_t status_offset = 0;
    std::size_t total_bytes   = 0;
};

bool workspace_geometry(std::uint32_t tokens, std::uint32_t columns,
                        WorkspaceGeometry* out) noexcept {
    if (out == nullptr || tokens == 0U) return false;
    WorkspaceGeometry result;
    if (!padded_columns(columns, &result.padded) ||
        !checked_mul(tokens, result.padded, &result.code_bytes) ||
        !align_up(result.code_bytes, kPlaneAlignment, &result.scale_offset) ||
        !checked_mul(tokens, sizeof(float), &result.scale_bytes)) {
        return false;
    }
    std::size_t scale_end = 0;
    if (!checked_add(result.scale_offset, result.scale_bytes, &scale_end) ||
        !align_up(scale_end, kPlaneAlignment, &result.status_offset) ||
        !checked_add(result.status_offset, static_cast<std::size_t>(tokens) * sizeof(std::uint32_t),
                     &result.total_bytes)) {
        return false;
    }
    *out = result;
    return true;
}

bool range(const void* pointer, std::size_t bytes, std::uintptr_t* begin,
           std::uintptr_t* end) noexcept {
    if (pointer == nullptr || bytes == 0U || begin == nullptr || end == nullptr) return false;
    *begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (*begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    *end = *begin + bytes;
    return true;
}

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes, const void* rhs,
                    std::size_t rhs_bytes) noexcept {
    std::uintptr_t lhs_begin = 0, lhs_end = 0, rhs_begin = 0, rhs_end = 0;
    if (!range(lhs, lhs_bytes, &lhs_begin, &lhs_end) ||
        !range(rhs, rhs_bytes, &rhs_begin, &rhs_end)) {
        return true;
    }
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

std::size_t status_bytes(const Fp8ActivationWorkspace& workspace) noexcept {
    return static_cast<std::size_t>(workspace.tokens) * sizeof(std::uint32_t);
}

bool valid_workspace(const Fp8ActivationWorkspace& workspace) noexcept {
    WorkspaceGeometry geometry;
    return workspace_geometry(workspace.tokens, workspace.columns, &geometry) &&
           workspace.padded_columns == geometry.padded &&
           workspace.code_bytes == geometry.code_bytes &&
           workspace.scale_bytes == geometry.scale_bytes && workspace.codes != nullptr &&
           workspace.scales != nullptr && workspace.status != nullptr &&
           reinterpret_cast<std::uintptr_t>(workspace.codes) % kPlaneAlignment == 0U &&
           reinterpret_cast<std::uintptr_t>(workspace.scales) % alignof(float) == 0U &&
           reinterpret_cast<std::uintptr_t>(workspace.status) % alignof(std::uint32_t) == 0U &&
           !ranges_overlap(workspace.codes, workspace.code_bytes, workspace.scales,
                           workspace.scale_bytes) &&
           !ranges_overlap(workspace.codes, workspace.code_bytes, workspace.status,
                           status_bytes(workspace)) &&
           !ranges_overlap(workspace.scales, workspace.scale_bytes, workspace.status,
                           status_bytes(workspace));
}

__device__ __forceinline__ void quantize_activation_row(
    std::uint32_t token, const hip_bfloat16* input, std::uint8_t* codes, float* scales,
    std::uint32_t* status, std::uint32_t columns, std::uint32_t padded) {
    const std::uint32_t lane = threadIdx.x % kWaveSize;
    const std::uint32_t wave = threadIdx.x / kWaveSize;
    float local_maximum = 0.0F;
    unsigned local_bad  = 0U;
    const auto observe = [&](float value) {
        local_bad |= static_cast<unsigned>(!isfinite(value));
        if (isfinite(value)) local_maximum = fmaxf(local_maximum, fabsf(value));
    };
    // Eight-column 16-byte rows when every token row is aligned; the codec is unchanged.
    const bool vectorized = columns % 8U == 0U &&
        reinterpret_cast<std::uintptr_t>(input) % 16U == 0U;
    const hip_bfloat16* row = input + static_cast<std::size_t>(token) * columns;
    if (vectorized) {
        const auto* vectors = reinterpret_cast<const uint4*>(row);
#pragma unroll 4
        for (std::uint32_t i = threadIdx.x; i < columns / 8U; i += blockDim.x) {
            const uint4 packed = vectors[i];
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                observe(__uint_as_float(words[w] << 16));
                observe(__uint_as_float(words[w] & 0xffff0000U));
            }
        }
    } else {
        for (std::uint32_t column = threadIdx.x; column < columns; column += blockDim.x) {
            observe(static_cast<float>(row[column]));
        }
    }
#pragma unroll
    for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
        local_maximum = fmaxf(local_maximum, __shfl_down(local_maximum, delta, kWaveSize));
        local_bad |= __shfl_down(local_bad, delta, kWaveSize);
    }

    __shared__ float wave_maxima[kWavesPerBlock];
    __shared__ unsigned wave_bad[kWavesPerBlock];
    __shared__ float token_scale;
    __shared__ unsigned token_bad;
    if (lane == 0U) {
        wave_maxima[wave] = local_maximum;
        wave_bad[wave] = local_bad;
    }
    __syncthreads();

    if (wave == 0U) {
        float maximum = lane < kWavesPerBlock ? wave_maxima[lane] : 0.0F;
        unsigned bad  = lane < kWavesPerBlock ? wave_bad[lane] : 0U;
#pragma unroll
        for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
            maximum = fmaxf(maximum, __shfl_down(maximum, delta, kWaveSize));
            bad |= __shfl_down(bad, delta, kWaveSize);
        }
        if (lane == 0U) {
            token_bad = bad != 0U;
            token_scale = bad != 0U ? 0.0F : maximum / kE4M3FiniteMaximum;
            scales[token] = token_scale;
            status[token] = bad != 0U ? Fp8ActivationNonfinite : Fp8ActivationOk;
        }
    }
    __syncthreads();

    const auto encode = [&](float value) {
        return static_cast<std::uint8_t>(
            __hip_cvt_float_to_fp8(value / token_scale, __HIP_SATFINITE, __HIP_E4M3));
    };
    const bool encoded = token_bad == 0U && token_scale != 0.0F;
    std::uint8_t* code_row = codes + static_cast<std::size_t>(token) * padded;
    std::uint32_t tail = 0U;
    if (vectorized) {
        const auto* vectors = reinterpret_cast<const uint4*>(row);
        for (std::uint32_t i = threadIdx.x; i < columns / 8U; i += blockDim.x) {
            std::uint32_t packed_codes[2] = {0U, 0U};
            if (encoded) {
                const uint4 packed = vectors[i];
                const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const std::uint32_t pair =
                        static_cast<std::uint32_t>(encode(__uint_as_float(words[w] << 16))) |
                        (static_cast<std::uint32_t>(
                             encode(__uint_as_float(words[w] & 0xffff0000U))) << 8U);
                    packed_codes[w / 2] |= pair << (16U * (w % 2));
                }
            }
            *reinterpret_cast<uint2*>(code_row + static_cast<std::size_t>(i) * 8U) =
                uint2{packed_codes[0], packed_codes[1]};
        }
        tail = columns;
    }
    for (std::uint32_t column = tail + threadIdx.x; column < padded; column += blockDim.x) {
        code_row[column] = encoded && column < columns ? encode(static_cast<float>(row[column]))
                                                       : std::uint8_t{0U};
    }
}

// A resident grid strides over tokens: grids beyond about 2048 waves paid a ~30 us dispatch
// penalty on gfx1201 whatever the row width. Each token row is quantized independently.
__global__ __launch_bounds__(kBlockSize) void fp8_quantize_activation_kernel(
    const hip_bfloat16* input, std::uint8_t* codes, float* scales, std::uint32_t* status,
    std::uint32_t tokens, std::uint32_t columns, std::uint32_t padded) {
    for (std::uint32_t token = blockIdx.x; token < tokens; token += gridDim.x) {
        quantize_activation_row(token, input, codes, scales, status, columns, padded);
        __syncthreads();  // the next row reuses the block's reduction storage
    }
}

// K5120 RMSNorm with the eager row-CTA reduction: one 16-byte vector per thread of a 640-thread
// row, per-thread FMA sum of squares, wave butterfly, then the wave sums in ascending order.
constexpr std::uint32_t kNormFeatures = 5120U;
constexpr std::uint32_t kNormThreads = kNormFeatures / 8U;
constexpr std::uint32_t kNormWaves = kNormThreads / kWaveSize;

__device__ __forceinline__ float norm_vector_sumsq(const float (&values)[8]) {
    float sumsq = 0.0F;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumsq = fmaf(values[i], values[i], sumsq);
#pragma unroll
    for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U)
        sumsq += __shfl_xor(sumsq, width, kWaveSize);
    return sumsq;  // identical in every lane
}

__device__ __forceinline__ void norm_vector_apply(float (&values)[8], uint4 gains, float inverse,
                                                  bool unit_offset) {
    const std::uint32_t gain_words[4] = {gains.x, gains.y, gains.z, gains.w};
    const float offset = unit_offset ? 1.0F : 0.0F;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const float low_gain = __uint_as_float(gain_words[w] << 16) + offset;
        const float high_gain = __uint_as_float(gain_words[w] & 0xffff0000U) + offset;
        values[2 * w] = static_cast<float>(hip_bfloat16(values[2 * w] * inverse * low_gain));
        values[2 * w + 1] =
            static_cast<float>(hip_bfloat16(values[2 * w + 1] * inverse * high_gain));
    }
}

__global__ __launch_bounds__(kNormThreads) void fp8_quantize_normalized_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, float eps, bool unit_offset,
    std::uint8_t* codes, float* scales, std::uint32_t* status, hip_bfloat16* normalized) {
    const std::uint32_t lane = threadIdx.x % kWaveSize, wave = threadIdx.x / kWaveSize;
    const auto* rows = reinterpret_cast<const uint4*>(input);
    const auto* gains = reinterpret_cast<const uint4*>(weight);
    const std::uint32_t token = blockIdx.x;
    float values[8];
    unpack8(rows[static_cast<std::size_t>(token) * kNormThreads + threadIdx.x], values);
    const float sumsq = norm_vector_sumsq(values);
    __shared__ float wave_sums[kNormWaves];
    if (lane == 0U) wave_sums[wave] = sumsq;
    __syncthreads();
    float block_sum = 0.0F;
#pragma unroll
    for (std::uint32_t index = 0; index < kNormWaves; ++index) block_sum += wave_sums[index];
    const float inverse = rsqrtf(block_sum / static_cast<float>(kNormFeatures) + eps);
    norm_vector_apply(values, gains[threadIdx.x], inverse, unit_offset);
    if (normalized != nullptr) {
        // The values are BF16-represented: the high halves are their exact BF16 bits.
        std::uint32_t words[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            words[w] = (__float_as_uint(values[2 * w]) >> 16) |
                       (__float_as_uint(values[2 * w + 1]) & 0xffff0000U);
        }
        reinterpret_cast<uint4*>(normalized)[static_cast<std::size_t>(token) * kNormThreads +
                                             threadIdx.x] = uint4{words[0], words[1], words[2],
                                                                  words[3]};
    }
    encode_row8<kNormThreads>(values, threadIdx.x, token, codes, scales, kNormFeatures, status);
}

// Output gate over K = 8 * threads features.
__device__ __forceinline__ void gate_vector(const hip_bfloat16* gate, const float* attention,
                                            std::size_t offset, float (&values)[8]) {
    float gates[8];
    unpack8(*reinterpret_cast<const uint4*>(gate + offset), gates);
    const float4 low = *reinterpret_cast<const float4*>(attention + offset);
    const float4 high = *reinterpret_cast<const float4*>(attention + offset + 4U);
    const float x[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float represented = static_cast<float>(hip_bfloat16(x[i]));
        values[i] = static_cast<float>(hip_bfloat16(represented / (1.0F + expf(-gates[i]))));
    }
}

template <std::uint32_t Threads>
__global__ __launch_bounds__(Threads) void fp8_quantize_gated_kernel(
    const hip_bfloat16* gate, const float* attention, std::uint8_t* codes, float* scales,
    std::uint32_t* status) {
    constexpr std::uint32_t columns = 8U * Threads;
    const std::uint32_t token = blockIdx.x;
    float values[8];
    gate_vector(gate, attention, static_cast<std::size_t>(token) * columns + threadIdx.x * 8U,
                values);
    encode_row8<Threads>(values, threadIdx.x, token, codes, scales, columns, status);
}


// Gated per-head RMSNorm of the GDN recurrent output x [T, 48 x 128] with gate z:
// BF16(x * rsqrt(mean_head(x^2) + eps) * w * silu(z)); the 16 lanes of a DPP row hold one head
// (eight features each). One 768-thread CTA per token row.
constexpr std::uint32_t kGatedNormColumns = 6144U;
constexpr std::uint32_t kGatedNormThreads = kGatedNormColumns / 8U;
constexpr std::uint32_t kGatedNormHead = 128U;

__device__ __forceinline__ void gated_norm_vector(const hip_bfloat16* input,
                                                  const hip_bfloat16* gate,
                                                  const hip_bfloat16* weight, float eps,
                                                  std::uint32_t token, float (&values)[8]) {
    const std::size_t offset = static_cast<std::size_t>(token) * kGatedNormThreads + threadIdx.x;
    float x[8], z[8], w[8];
    unpack8(reinterpret_cast<const uint4*>(input)[offset], x);
    unpack8(reinterpret_cast<const uint4*>(gate)[offset], z);
    unpack8(reinterpret_cast<const uint4*>(weight)[threadIdx.x % (kGatedNormHead / 8U)], w);
    float sumsq = 0.0F;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumsq = fmaf(x[i], x[i], sumsq);
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(sumsq), 0x168, 0xf, 0xf, false));
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(sumsq), 0x164, 0xf, 0xf, false));
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(sumsq), 0x162, 0xf, 0xf, false));
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(sumsq), 0x161, 0xf, 0xf, false));
    const float inverse = rsqrtf(sumsq / static_cast<float>(kGatedNormHead) + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float silu = z[i] / (1.0F + expf(-z[i]));
        values[i] = static_cast<float>(hip_bfloat16(x[i] * inverse * w[i] * silu));
    }
}

__global__ __launch_bounds__(kGatedNormThreads) void fp8_quantize_gated_rmsnorm_kernel(
    const hip_bfloat16* input, const hip_bfloat16* gate, const hip_bfloat16* weight, float eps,
    std::uint8_t* codes, float* scales, std::uint32_t* status) {
    const std::uint32_t token = blockIdx.x;
    float values[8];
    gated_norm_vector(input, gate, weight, eps, token, values);
    encode_row8<kGatedNormThreads>(values, threadIdx.x, token, codes, scales, kGatedNormColumns,
                                   status);
}

constexpr std::uint32_t kGatedColumns = 6144U;

} // namespace

std::size_t fp8_activation_workspace_capacity_bytes(std::uint32_t tokens,
                                                    std::uint32_t columns) noexcept {
    WorkspaceGeometry geometry;
    return workspace_geometry(tokens, columns, &geometry) ? geometry.total_bytes : 0U;
}

hipError_t fp8_bind_activation_workspace(void* storage, std::size_t storage_bytes,
                                         std::uint32_t tokens, std::uint32_t columns,
                                         Fp8ActivationWorkspace* out) noexcept {
    WorkspaceGeometry geometry;
    if (storage == nullptr || out == nullptr ||
        reinterpret_cast<std::uintptr_t>(storage) % kPlaneAlignment != 0U ||
        !workspace_geometry(tokens, columns, &geometry) ||
        storage_bytes != geometry.total_bytes) {
        return hipErrorInvalidValue;
    }
    auto* bytes = static_cast<std::uint8_t*>(storage);
    *out = Fp8ActivationWorkspace{
        bytes,
        geometry.code_bytes,
        reinterpret_cast<float*>(bytes + geometry.scale_offset),
        geometry.scale_bytes,
        reinterpret_cast<std::uint32_t*>(bytes + geometry.status_offset),
        tokens,
        columns,
        geometry.padded,
    };
    return hipSuccess;
}

hipError_t fp8_quantize_activation(const Fp8ActivationQuantizeArgs& args,
                                   hipStream_t stream) noexcept {
    if (args.input == nullptr || !valid_workspace(args.workspace)) return hipErrorInvalidValue;
    std::size_t input_elements = 0, input_bytes = 0;
    if (!checked_mul(args.workspace.tokens, args.workspace.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.codes,
                       args.workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.scales,
                       args.workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.status, status_bytes(args.workspace))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(fp8_quantize_activation_kernel,
                       dim3(std::min(args.workspace.tokens, kQuantizeResidentBlocks)),
                       dim3(kBlockSize), 0, stream, args.input, args.workspace.codes,
                       args.workspace.scales, args.workspace.status, args.workspace.tokens,
                       args.workspace.columns, args.workspace.padded_columns);
    return hipGetLastError();
}

hipError_t fp8_quantize_normalized_activation(const Fp8NormalizedQuantizeArgs& args,
                                              hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t input_bytes = 0;
    if (args.input == nullptr || args.weight == nullptr || !valid_workspace(workspace) ||
        workspace.columns != kNormFeatures ||
        workspace.padded_columns != kNormFeatures || !std::isfinite(args.eps) ||
        !(args.eps > 0.0F) || reinterpret_cast<std::uintptr_t>(args.input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.weight) % 16U != 0U ||
        !checked_mul(static_cast<std::size_t>(workspace.tokens) * kNormFeatures,
                     sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.status, status_bytes(workspace))) {
        return hipErrorInvalidValue;
    }
    if (args.normalized != nullptr &&
        (reinterpret_cast<std::uintptr_t>(args.normalized) % 16U != 0U ||
         ranges_overlap(args.normalized, input_bytes, args.input, input_bytes) ||
         ranges_overlap(args.normalized, input_bytes, args.weight,
                        kNormFeatures * sizeof(hip_bfloat16)) ||
         ranges_overlap(args.normalized, input_bytes, workspace.codes, workspace.code_bytes) ||
         ranges_overlap(args.normalized, input_bytes, workspace.scales, workspace.scale_bytes) ||
         ranges_overlap(args.normalized, input_bytes, workspace.status, status_bytes(workspace)))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(fp8_quantize_normalized_kernel,
                       dim3(workspace.tokens), dim3(kNormThreads), 0,
                       stream, args.input, args.weight, args.eps, args.unit_offset,
                       workspace.codes, workspace.scales, workspace.status, args.normalized);
    return hipGetLastError();
}

hipError_t fp8_quantize_gated_activation(const Fp8GatedQuantizeArgs& args,
                                         hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t elements = 0, gate_bytes = 0, attention_bytes = 0;
    if (args.gate == nullptr || args.attention == nullptr || !valid_workspace(workspace) ||
        workspace.columns != kGatedColumns ||
        workspace.padded_columns != kGatedColumns ||
        reinterpret_cast<std::uintptr_t>(args.gate) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.attention) % 16U != 0U ||
        !checked_mul(workspace.tokens, kGatedColumns, &elements) ||
        !checked_mul(elements, sizeof(hip_bfloat16), &gate_bytes) ||
        !checked_mul(elements, sizeof(float), &attention_bytes) ||
        ranges_overlap(args.gate, gate_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.attention, attention_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.gate, gate_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.attention, attention_bytes, workspace.scales,
                       workspace.scale_bytes) ||
        ranges_overlap(args.gate, gate_bytes, workspace.status, status_bytes(workspace)) ||
        ranges_overlap(args.attention, attention_bytes, workspace.status,
                       status_bytes(workspace))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(fp8_quantize_gated_kernel<kGatedColumns / 8U>,
                       dim3(workspace.tokens), dim3(kGatedColumns / 8U),
                       0, stream, args.gate, args.attention, workspace.codes, workspace.scales,
                       workspace.status);
    return hipGetLastError();
}

hipError_t fp8_quantize_gated_rmsnorm_activation(const Fp8GatedRmsNormQuantizeArgs& args,
                                                 hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t elements = 0, input_bytes = 0;
    if (args.input == nullptr || args.gate == nullptr || args.weight == nullptr ||
        !valid_workspace(workspace) || workspace.columns != kGatedNormColumns ||
        workspace.padded_columns != kGatedNormColumns || !std::isfinite(args.eps) ||
        !(args.eps > 0.0F) || reinterpret_cast<std::uintptr_t>(args.input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.gate) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.weight) % 16U != 0U ||
        !checked_mul(workspace.tokens, kGatedNormColumns, &elements) ||
        !checked_mul(elements, sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.gate, input_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.gate, input_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.status, status_bytes(workspace)) ||
        ranges_overlap(args.gate, input_bytes, workspace.status, status_bytes(workspace))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(fp8_quantize_gated_rmsnorm_kernel,
                       dim3(workspace.tokens), dim3(kGatedNormThreads),
                       0, stream, args.input, args.gate, args.weight, args.eps, workspace.codes,
                       workspace.scales, workspace.status);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::linear
