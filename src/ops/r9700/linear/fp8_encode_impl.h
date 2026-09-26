#pragma once

#include "ops/r9700/linear/fp8_activation.h"

#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Device-side per-token E4M3 row encoder shared by the activation producers and fused fronts.
namespace ninfer::ops::r9700::linear::fp8_encode {

constexpr std::uint32_t kWaveSize = 32U;
constexpr float kE4M3FiniteMaximum = 448.0F;

// Encodes one token row held as V vectors of eight values per thread (vector i * Threads + thread):
// the finite maximum and nonfinite flag are reduced over the CTA, then the row publishes scale
// max / 448, its status word and E4M3 codes exactly as fp8_quantize_activation_kernel does for the
// same BF16 values.
template <std::uint32_t Threads, std::uint32_t V = 1U>
__device__ __forceinline__ void encode_rows(const float (&values)[V][8], std::uint32_t token,
                                            std::uint8_t* codes, float* scales,
                                            std::uint32_t padded, std::uint32_t* status) {
    constexpr std::uint32_t waves = Threads / kWaveSize;
    const std::uint32_t lane = threadIdx.x % kWaveSize, wave = threadIdx.x / kWaveSize;
    float maximum = 0.0F;
    unsigned bad = 0U;
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            bad |= static_cast<unsigned>(!isfinite(values[v][i]));
            if (isfinite(values[v][i])) maximum = fmaxf(maximum, fabsf(values[v][i]));
        }
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
    if (threadIdx.x == 0U) {
        scales[token] = token_scale;
        status[token] = row_bad != 0U ? Fp8ActivationNonfinite : Fp8ActivationOk;
    }
    const bool encoded = row_bad == 0U && token_scale != 0.0F;
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        std::uint32_t packed[2] = {0U, 0U};
        if (encoded) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const std::uint32_t code = static_cast<std::uint8_t>(__hip_cvt_float_to_fp8(
                    values[v][i] / token_scale, __HIP_SATFINITE, __HIP_E4M3));
                packed[i / 4] |= code << (8U * (i % 4));
            }
        }
        *reinterpret_cast<uint2*>(codes + static_cast<std::size_t>(token) * padded +
                                  (static_cast<std::size_t>(v) * Threads + threadIdx.x) * 8U) =
            uint2{packed[0], packed[1]};
    }
}

template <std::uint32_t Threads>
__device__ __forceinline__ void encode_row8(const float (&values)[8], std::uint32_t vector,
                                            std::uint32_t token, std::uint8_t* codes,
                                            float* scales, std::uint32_t padded,
                                            std::uint32_t* status) {
    (void)vector;  // vector == threadIdx.x for every single-vector producer
    const float (&rows)[1][8] = reinterpret_cast<const float (&)[1][8]>(values);
    encode_rows<Threads, 1U>(rows, token, codes, scales, padded, status);
}

__device__ __forceinline__ void unpack8(uint4 packed, float (&values)[8]) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        values[2 * w] = __uint_as_float(words[w] << 16);
        values[2 * w + 1] = __uint_as_float(words[w] & 0xffff0000U);
    }
}

} // namespace ninfer::ops::r9700::linear::fp8_encode
