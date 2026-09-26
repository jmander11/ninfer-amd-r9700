#include "ops/r9700/linear/fp8_activation.h"

#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::r9700::linear {
namespace {

constexpr std::uint32_t kWaveSize        = 32U;
constexpr std::uint32_t kBlockSize       = 1024U;
constexpr std::uint32_t kWavesPerBlock   = kBlockSize / kWaveSize;
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
        !checked_add(result.status_offset, sizeof(std::uint32_t), &result.total_bytes)) {
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
                           sizeof(std::uint32_t)) &&
           !ranges_overlap(workspace.scales, workspace.scale_bytes, workspace.status,
                           sizeof(std::uint32_t));
}

// Status CTA epilogue: the OR over the CTA's threads of `bad` is stored once. Status CTAs lay their
// threads out like a token CTA and loop over the tokens, so their loads stay independent.
template <std::uint32_t Waves>
__device__ __forceinline__ void publish_status(bool bad, std::uint32_t* status) {
    __shared__ unsigned wave_bad[Waves];
    const std::uint32_t lane = threadIdx.x % kWaveSize, wave = threadIdx.x / kWaveSize;
    const bool any = __builtin_amdgcn_ballot_w32(bad) != 0U;
    if (lane == 0U) wave_bad[wave] = any ? 1U : 0U;
    __syncthreads();
    if (threadIdx.x == 0U) {
        unsigned result = 0U;
        for (std::uint32_t index = 0; index < Waves; ++index) result |= wave_bad[index];
        *status = result != 0U ? Fp8ActivationNonfinite : Fp8ActivationOk;
    }
}

__global__ __launch_bounds__(kBlockSize) void fp8_quantize_activation_kernel(
    const hip_bfloat16* input, std::uint8_t* codes, float* scales, std::uint32_t* status,
    std::uint32_t columns, std::uint32_t padded, std::uint32_t status_cta) {
    const std::uint32_t token = blockIdx.x;
    const std::uint32_t lane = threadIdx.x % kWaveSize;
    const std::uint32_t wave = threadIdx.x / kWaveSize;
    if (status_cta != 0U && token == gridDim.x - 1U) {
        const std::size_t elements = static_cast<std::size_t>(token) * columns;
        const auto* bits = reinterpret_cast<const std::uint16_t*>(input);
        bool bad = false;
        if (elements % 8U == 0U && reinterpret_cast<std::uintptr_t>(input) % 16U == 0U) {
            const auto* vectors = reinterpret_cast<const uint4*>(input);
#pragma unroll 4
            for (std::size_t i = threadIdx.x; i < elements / 8U; i += blockDim.x) {
                const uint4 packed = vectors[i];
                const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (int w = 0; w < 4; ++w)
                    bad |= (words[w] & 0x7f80U) == 0x7f80U ||
                           (words[w] & 0x7f800000U) == 0x7f800000U;
            }
        } else {
            for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x)
                bad |= (bits[i] & 0x7f80U) == 0x7f80U;
        }
        publish_status<kWavesPerBlock>(bad, status);
        return;
    }
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
            if (bad != 0U && status_cta == 0U) {
                atomicOr(status, static_cast<std::uint32_t>(Fp8ActivationNonfinite));
            }
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

// Encodes one token row held as eight values per thread: the finite maximum and nonfinite flag
// are reduced over the CTA, then the row publishes scale max / 448 and E4M3 codes exactly as
// fp8_quantize_activation_kernel does for the same BF16 values.
template <std::uint32_t Threads>
__device__ __forceinline__ void encode_row8(const float (&values)[8], std::uint32_t vector,
                                            std::uint32_t token, std::uint8_t* codes,
                                            float* scales, std::uint32_t padded) {
    constexpr std::uint32_t waves = Threads / kWaveSize;
    const std::uint32_t lane = threadIdx.x % kWaveSize, wave = threadIdx.x / kWaveSize;
    float maximum = 0.0F;
    unsigned bad = 0U;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        bad |= static_cast<unsigned>(!isfinite(values[i]));
        if (isfinite(values[i])) maximum = fmaxf(maximum, fabsf(values[i]));
    }
#pragma unroll
    for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, delta, kWaveSize));
        bad |= __shfl_xor(bad, delta, kWaveSize);
    }
    __shared__ float wave_maxima[waves];
    __shared__ unsigned wave_bad[waves];
    if (lane == 0U) {
        wave_maxima[wave] = maximum;
        wave_bad[wave] = bad;
    }
    __syncthreads();
    float row_maximum = 0.0F;
    unsigned row_bad = 0U;
#pragma unroll
    for (std::uint32_t index = 0; index < waves; ++index) {
        row_maximum = fmaxf(row_maximum, wave_maxima[index]);
        row_bad |= wave_bad[index];
    }
    const float token_scale = row_bad != 0U ? 0.0F : row_maximum / kE4M3FiniteMaximum;
    if (threadIdx.x == 0U) scales[token] = token_scale;
    const bool encoded = row_bad == 0U && token_scale != 0.0F;
    std::uint32_t packed[2] = {0U, 0U};
    if (encoded) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t code = static_cast<std::uint8_t>(
                __hip_cvt_float_to_fp8(values[i] / token_scale, __HIP_SATFINITE, __HIP_E4M3));
            packed[i / 4] |= code << (8U * (i % 4));
        }
    }
    *reinterpret_cast<uint2*>(codes + static_cast<std::size_t>(token) * padded +
                              static_cast<std::size_t>(vector) * 8U) = uint2{packed[0], packed[1]};
}

__device__ __forceinline__ void unpack8(uint4 packed, float (&values)[8]) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        values[2 * w] = __uint_as_float(words[w] << 16);
        values[2 * w + 1] = __uint_as_float(words[w] & 0xffff0000U);
    }
}

// K5120 RMSNorm with the eager small-T reduction: one 16-byte vector per thread of a 640-thread
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
    std::uint8_t* codes, float* scales, std::uint32_t* status) {
    const std::uint32_t tokens = gridDim.x - 1U;
    const std::uint32_t lane = threadIdx.x % kWaveSize, wave = threadIdx.x / kWaveSize;
    const auto* rows = reinterpret_cast<const uint4*>(input);
    const auto* gains = reinterpret_cast<const uint4*>(weight);
    if (blockIdx.x == tokens) {
        // Replays the token CTAs' reductions (same thread-to-vector map and order), eight rows at
        // a time.
        constexpr std::uint32_t group = 8U;
        __shared__ float group_sums[group][kNormWaves];
        const uint4 gain = gains[threadIdx.x];
        bool bad = false;
        for (std::uint32_t first = 0; first < tokens; first += group) {
            float values[group][8];
#pragma unroll
            for (std::uint32_t g = 0; g < group; ++g) {
                const std::uint32_t token = min(first + g, tokens - 1U);
                unpack8(rows[static_cast<std::size_t>(token) * kNormThreads + threadIdx.x],
                        values[g]);
            }
#pragma unroll
            for (std::uint32_t g = 0; g < group; ++g) {
                const float sumsq = norm_vector_sumsq(values[g]);
                if (lane == 0U) group_sums[g][wave] = sumsq;
            }
            __syncthreads();
#pragma unroll
            for (std::uint32_t g = 0; g < group; ++g) {
                float block_sum = 0.0F;
#pragma unroll
                for (std::uint32_t index = 0; index < kNormWaves; ++index)
                    block_sum += group_sums[g][index];
                const float inverse =
                    rsqrtf(block_sum / static_cast<float>(kNormFeatures) + eps);
                norm_vector_apply(values[g], gain, inverse, unit_offset);
#pragma unroll
                for (int i = 0; i < 8; ++i) bad |= !isfinite(values[g][i]);
            }
            __syncthreads();
        }
        publish_status<kNormWaves>(bad, status);
        return;
    }
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
    encode_row8<kNormThreads>(values, threadIdx.x, token, codes, scales, kNormFeatures);
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
    const std::uint32_t tokens = gridDim.x - 1U;
    const std::uint32_t lane = threadIdx.x % kWaveSize;
    if (blockIdx.x == tokens) {
        bool bad = false;
#pragma unroll 4
        for (std::uint32_t token = 0; token < tokens; ++token) {
            float values[8];
            gate_vector(gate, attention,
                        static_cast<std::size_t>(token) * columns + threadIdx.x * 8U, values);
#pragma unroll
            for (int i = 0; i < 8; ++i) bad |= !isfinite(values[i]);
        }
        publish_status<Threads / kWaveSize>(bad, status);
        return;
    }
    const std::uint32_t token = blockIdx.x;
    float values[8];
    gate_vector(gate, attention, static_cast<std::size_t>(token) * columns + threadIdx.x * 8U,
                values);
    encode_row8<Threads>(values, threadIdx.x, token, codes, scales, columns);
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
        ranges_overlap(args.input, input_bytes, args.workspace.status, sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    const bool status_cta = args.workspace.tokens <= kFp8StatusCtaTokenLimit;
    if (!status_cta) {
        const hipError_t result =
            hipMemsetAsync(args.workspace.status, 0, sizeof(std::uint32_t), stream);
        if (result != hipSuccess) return result;
    }
    hipLaunchKernelGGL(fp8_quantize_activation_kernel,
                       dim3(args.workspace.tokens + (status_cta ? 1U : 0U)), dim3(kBlockSize), 0,
                       stream, args.input, args.workspace.codes, args.workspace.scales,
                       args.workspace.status, args.workspace.columns,
                       args.workspace.padded_columns, status_cta ? 1U : 0U);
    return hipGetLastError();
}

hipError_t fp8_quantize_normalized_activation(const Fp8NormalizedQuantizeArgs& args,
                                              hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t input_bytes = 0;
    if (args.input == nullptr || args.weight == nullptr || !valid_workspace(workspace) ||
        workspace.tokens > kFp8StatusCtaTokenLimit || workspace.columns != kNormFeatures ||
        workspace.padded_columns != kNormFeatures || !std::isfinite(args.eps) ||
        !(args.eps > 0.0F) || reinterpret_cast<std::uintptr_t>(args.input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.weight) % 16U != 0U ||
        !checked_mul(static_cast<std::size_t>(workspace.tokens) * kNormFeatures,
                     sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.status, sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(fp8_quantize_normalized_kernel, dim3(workspace.tokens + 1U),
                       dim3(kNormThreads), 0, stream, args.input, args.weight, args.eps,
                       args.unit_offset, workspace.codes, workspace.scales, workspace.status);
    return hipGetLastError();
}

hipError_t fp8_quantize_gated_activation(const Fp8GatedQuantizeArgs& args,
                                         hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t elements = 0, gate_bytes = 0, attention_bytes = 0;
    if (args.gate == nullptr || args.attention == nullptr || !valid_workspace(workspace) ||
        workspace.tokens > kFp8StatusCtaTokenLimit || workspace.columns != kGatedColumns ||
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
        ranges_overlap(args.gate, gate_bytes, workspace.status, sizeof(std::uint32_t)) ||
        ranges_overlap(args.attention, attention_bytes, workspace.status,
                       sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(fp8_quantize_gated_kernel<kGatedColumns / 8U>, dim3(workspace.tokens + 1U),
                       dim3(kGatedColumns / 8U), 0, stream, args.gate, args.attention,
                       workspace.codes, workspace.scales, workspace.status);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::linear
