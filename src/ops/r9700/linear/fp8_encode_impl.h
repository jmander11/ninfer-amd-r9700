#pragma once

#include "ops/r9700/linear/fp8_activation.h"

#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Device-side per-token E4M3 row encoder shared by the activation producers and fused fronts.
namespace ninfer::ops::r9700::linear::fp8_encode {

constexpr std::uint32_t kWaveSize  = 32U;
constexpr float kE4M3FiniteMaximum = 448.0F;

template <std::uint32_t Threads>
struct EncodeShared {
    float wave_maxima[Threads / kWaveSize];
    unsigned wave_bad[Threads / kWaveSize];
};

// Encodes one token row of 8 * Vectors columns held by a Threads-thread CTA context as V vectors
// of eight values per thread (vector v * Threads + thread; those at or past Vectors do not exist):
// the finite maximum and nonfinite flag are reduced over the CTA, then the row publishes scale
// max / 448, its status word and E4M3 codes exactly as fp8_quantize_activation_kernel does for the
// same BF16 values. The maximum and flag are order-independent, so any CTA width that holds the
// row encodes it identically.
template <std::uint32_t Threads, std::uint32_t V = 1U, std::uint32_t Vectors = Threads * V,
          class Cta>
__device__ __forceinline__ void encode_rows(const Cta& cta, EncodeShared<Threads>& shared,
                                            const float (&values)[V][8], std::uint32_t token,
                                            std::uint8_t* codes, float* scales,
                                            std::uint32_t padded, std::uint32_t* status) {
    static_assert(Threads % kWaveSize == 0U && Vectors <= Threads * V);
    const std::uint32_t thread = cta.thread();
    const std::uint32_t lane = thread % kWaveSize, wave = thread / kWaveSize;
    const auto exists = [&](std::uint32_t v) { return v * Threads + thread < Vectors; };
    float maximum     = 0.0F;
    unsigned bad      = 0U;
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        if (!exists(v)) continue;
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
    if (lane == 0U) {
        shared.wave_maxima[wave] = maximum;
        shared.wave_bad[wave]    = bad;
    }
    cta.sync();
    float row_maximum = 0.0F;
    unsigned row_bad  = 0U;
#pragma unroll
    for (std::uint32_t index = 0; index < Threads / kWaveSize; ++index) {
        row_maximum = fmaxf(row_maximum, shared.wave_maxima[index]);
        row_bad |= shared.wave_bad[index];
    }
    const float token_scale = row_bad != 0U ? 0.0F : row_maximum / kE4M3FiniteMaximum;
    if (thread == 0U) {
        scales[token] = token_scale;
        status[token] = row_bad != 0U ? Fp8ActivationNonfinite : Fp8ActivationOk;
    }
    const bool encoded = row_bad == 0U && token_scale != 0.0F;
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        if (!exists(v)) continue;
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
                                  (static_cast<std::size_t>(v) * Threads + thread) * 8U) =
            uint2{packed[0], packed[1]};
    }
}

__device__ __forceinline__ void unpack8(uint4 packed, float (&values)[8]) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        values[2 * w]     = __uint_as_float(words[w] << 16);
        values[2 * w + 1] = __uint_as_float(words[w] & 0xffff0000U);
    }
}

} // namespace ninfer::ops::r9700::linear::fp8_encode
