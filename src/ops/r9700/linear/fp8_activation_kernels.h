#pragma once

// Per-token E4M3 activation producers (fp8_activation.h), shared by their launchers and the
// persistent decode kernel. Row-wide producers take the CTA width as `Threads`: thread t of the
// context holds row vectors v * Threads + t, and every reduction keeps the per-vector arithmetic
// and the wave-sum order of the one-vector-per-thread launch (wave w of the launch is the wave
// holding vector 32 w), so any width that holds the row produces identical bytes.
#include "core/cache_warm.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_encode_impl.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear::fp8_activation_kernels {

using fp8_encode::encode_rows;
using fp8_encode::EncodeShared;
using fp8_encode::unpack8;

inline constexpr std::uint32_t kWaveSize      = 32U;
inline constexpr std::uint32_t kBlockSize     = 256U;
inline constexpr float kE4M3FiniteMaximum     = 448.0F;

// ---- Arbitrary-width rows, one CTA per row (kBlockSize threads at launch). The row maximum and
// nonfinite flag are order-independent and the codes elementwise, so any CTA width of at most
// kWaveSize waves encodes the row identically.
inline constexpr std::uint32_t kQuantizeMaximumWaves = kWaveSize;
struct QuantizeShared {
    float wave_maxima[kQuantizeMaximumWaves];
    unsigned wave_bad[kQuantizeMaximumWaves];
    float token_scale;
    unsigned token_bad;
};

template <class Cta>
__device__ __forceinline__ void quantize_activation_row(
    const Cta& cta, QuantizeShared& shared, std::uint32_t token, const hip_bfloat16* input,
    std::uint8_t* codes, float* scales, std::uint32_t* status, std::uint32_t columns,
    std::uint32_t padded) {
    const std::uint32_t thread = cta.thread(), threads = cta.threads();
    const std::uint32_t lane = thread % kWaveSize;
    const std::uint32_t wave = thread / kWaveSize;
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
        for (std::uint32_t i = thread; i < columns / 8U; i += threads) {
            const uint4 packed = vectors[i];
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                observe(__uint_as_float(words[w] << 16));
                observe(__uint_as_float(words[w] & 0xffff0000U));
            }
        }
    } else {
        for (std::uint32_t column = thread; column < columns; column += threads) {
            observe(static_cast<float>(row[column]));
        }
    }
#pragma unroll
    for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
        local_maximum = fmaxf(local_maximum, __shfl_down(local_maximum, delta, kWaveSize));
        local_bad |= __shfl_down(local_bad, delta, kWaveSize);
    }

    if (lane == 0U) {
        shared.wave_maxima[wave] = local_maximum;
        shared.wave_bad[wave] = local_bad;
    }
    cta.sync();

    if (wave == 0U) {
        const std::uint32_t waves = threads / kWaveSize;
        float maximum = lane < waves ? shared.wave_maxima[lane] : 0.0F;
        unsigned bad  = lane < waves ? shared.wave_bad[lane] : 0U;
#pragma unroll
        for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
            maximum = fmaxf(maximum, __shfl_down(maximum, delta, kWaveSize));
            bad |= __shfl_down(bad, delta, kWaveSize);
        }
        if (lane == 0U) {
            shared.token_bad = bad != 0U;
            shared.token_scale = bad != 0U ? 0.0F : maximum / kE4M3FiniteMaximum;
            scales[token] = shared.token_scale;
            status[token] = bad != 0U ? Fp8ActivationNonfinite : Fp8ActivationOk;
        }
    }
    cta.sync();

    const float token_scale = shared.token_scale;
    const auto encode = [&](float value) {
        return static_cast<std::uint8_t>(
            __hip_cvt_float_to_fp8(value / token_scale, __HIP_SATFINITE, __HIP_E4M3));
    };
    const bool encoded = shared.token_bad == 0U && token_scale != 0.0F;
    std::uint8_t* code_row = codes + static_cast<std::size_t>(token) * padded;
    std::uint32_t tail = 0U;
    if (vectorized) {
        const auto* vectors = reinterpret_cast<const uint4*>(row);
        for (std::uint32_t i = thread; i < columns / 8U; i += threads) {
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
    for (std::uint32_t column = tail + thread; column < padded; column += threads) {
        code_row[column] = encoded && column < columns ? encode(static_cast<float>(row[column]))
                                                       : std::uint8_t{0U};
    }
}

// A resident grid strides over tokens: grids beyond about 2048 waves paid a ~30 us dispatch
// penalty on gfx1201 whatever the row width. Each token row is quantized independently.
template <class Cta>
__device__ __forceinline__ void quantize_activation_body(
    const Cta& cta, QuantizeShared& shared, const hip_bfloat16* input, std::uint8_t* codes,
    float* scales, std::uint32_t* status, std::uint32_t tokens, std::uint32_t columns,
    std::uint32_t padded, const CacheWarm& warm, std::uint32_t work_ctas) {
    const std::uint32_t block = cta.block().x;
    if (block >= work_ctas) {
        warm_cache(warm, block - work_ctas, cta.grid().x - work_ctas, cta.thread(),
                   cta.threads());
        return;
    }
    for (std::uint32_t token = block; token < tokens; token += work_ctas) {
        quantize_activation_row(cta, shared, token, input, codes, scales, status, columns,
                                padded);
        cta.sync();  // the next row reuses the context's reduction storage
    }
}

__global__ __launch_bounds__(kBlockSize) void fp8_quantize_activation_kernel(
    const hip_bfloat16* input, std::uint8_t* codes, float* scales, std::uint32_t* status,
    std::uint32_t tokens, std::uint32_t columns, std::uint32_t padded, CacheWarm warm,
    std::uint32_t work_ctas);

// ---- K5120 RMSNorm with the eager row-CTA reduction: one 16-byte vector per launch thread of a
// 640-thread row, per-thread FMA sum of squares, wave butterfly, then the wave sums in ascending
// order.
inline constexpr std::uint32_t kNormFeatures = 5120U;
inline constexpr std::uint32_t kNormThreads = kNormFeatures / 8U;
inline constexpr std::uint32_t kNormWaves = kNormThreads / kWaveSize;

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

template <std::uint32_t Threads>
struct NormalizedShared {
    float wave_sums[kNormWaves];
    EncodeShared<Threads> encode;
};

template <std::uint32_t Threads, class Cta>
__device__ __forceinline__ void quantize_normalized_body(
    const Cta& cta, NormalizedShared<Threads>& shared, const hip_bfloat16* input,
    const hip_bfloat16* weight, float eps, bool unit_offset, std::uint8_t* codes, float* scales,
    std::uint32_t* status, hip_bfloat16* normalized, const CacheWarm& warm,
    std::uint32_t tokens) {
    static_assert(Threads % kWaveSize == 0U);
    constexpr std::uint32_t V = (kNormThreads + Threads - 1U) / Threads;
    const std::uint32_t block = cta.block().x;
    if (block >= tokens) {
        warm_cache(warm, block - tokens, cta.grid().x - tokens, cta.thread(), cta.threads());
        return;
    }
    const std::uint32_t thread = cta.thread();
    const std::uint32_t lane = thread % kWaveSize, wave = thread / kWaveSize;
    const auto exists = [&](std::uint32_t v) { return v * Threads + thread < kNormThreads; };
    const auto* rows = reinterpret_cast<const uint4*>(input);
    const auto* gains = reinterpret_cast<const uint4*>(weight);
    const std::uint32_t token = block;
    float values[V][8];
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        if (!exists(v)) continue;
        unpack8(rows[static_cast<std::size_t>(token) * kNormThreads + v * Threads + thread],
                values[v]);
        const float sumsq = norm_vector_sumsq(values[v]);
        if (lane == 0U) shared.wave_sums[v * (Threads / kWaveSize) + wave] = sumsq;
    }
    cta.sync();
    float block_sum = 0.0F;
#pragma unroll
    for (std::uint32_t index = 0; index < kNormWaves; ++index) block_sum += shared.wave_sums[index];
    const float inverse = rsqrtf(block_sum / static_cast<float>(kNormFeatures) + eps);
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        if (!exists(v)) continue;
        const std::uint32_t vector = v * Threads + thread;
        norm_vector_apply(values[v], gains[vector], inverse, unit_offset);
        if (normalized != nullptr) {
            // The values are BF16-represented: the high halves are their exact BF16 bits.
            std::uint32_t words[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                words[w] = (__float_as_uint(values[v][2 * w]) >> 16) |
                           (__float_as_uint(values[v][2 * w + 1]) & 0xffff0000U);
            }
            reinterpret_cast<uint4*>(normalized)[static_cast<std::size_t>(token) * kNormThreads +
                                                 vector] = uint4{words[0], words[1], words[2],
                                                                 words[3]};
        }
    }
    encode_rows<Threads, V, kNormThreads>(cta, shared.encode, values, token, codes, scales,
                                          kNormFeatures, status);
}

__global__ __launch_bounds__(kNormThreads) void fp8_quantize_normalized_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, float eps, bool unit_offset,
    std::uint8_t* codes, float* scales, std::uint32_t* status, hip_bfloat16* normalized,
    CacheWarm warm, std::uint32_t tokens);

// ---- Output gate over K = 8 * Vectors features.
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

template <std::uint32_t Vectors, std::uint32_t Threads, class Cta>
__device__ __forceinline__ void quantize_gated_body(
    const Cta& cta, EncodeShared<Threads>& shared, const hip_bfloat16* gate,
    const float* attention, std::uint8_t* codes, float* scales, std::uint32_t* status,
    const CacheWarm& warm, std::uint32_t tokens) {
    constexpr std::uint32_t V = (Vectors + Threads - 1U) / Threads;
    constexpr std::uint32_t columns = 8U * Vectors;
    const std::uint32_t block = cta.block().x;
    if (block >= tokens) {
        warm_cache(warm, block - tokens, cta.grid().x - tokens, cta.thread(), cta.threads());
        return;
    }
    const std::uint32_t token = block, thread = cta.thread();
    float values[V][8];
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        if (v * Threads + thread >= Vectors) continue;
        gate_vector(gate, attention,
                    static_cast<std::size_t>(token) * columns + (v * Threads + thread) * 8U,
                    values[v]);
    }
    encode_rows<Threads, V, Vectors>(cta, shared, values, token, codes, scales, columns, status);
}

template <std::uint32_t Threads>
__global__ __launch_bounds__(Threads) void fp8_quantize_gated_kernel(
    const hip_bfloat16* gate, const float* attention, std::uint8_t* codes, float* scales,
    std::uint32_t* status, CacheWarm warm, std::uint32_t tokens) {
    __shared__ EncodeShared<Threads> shared;
    quantize_gated_body<Threads, Threads>(persistent::LaunchCta{}, shared, gate, attention,
                                          codes, scales, status, warm, tokens);
}

// ---- Gated per-head RMSNorm of the GDN recurrent output x [T, 48 x 128] with gate z:
// BF16(x * rsqrt(mean_head(x^2) + eps) * w * silu(z)); the 16 lanes of a DPP row hold one head
// (eight features each), so vector j sits in lane j % 32 at any CTA width.
inline constexpr std::uint32_t kGatedNormColumns = 6144U;
inline constexpr std::uint32_t kGatedNormThreads = kGatedNormColumns / 8U;
inline constexpr std::uint32_t kGatedNormHead = 128U;

__device__ __forceinline__ void gated_norm_vector(const hip_bfloat16* input,
                                                  const hip_bfloat16* gate,
                                                  const hip_bfloat16* weight, float eps,
                                                  std::uint32_t token, std::uint32_t vector,
                                                  float (&values)[8]) {
    const std::size_t offset = static_cast<std::size_t>(token) * kGatedNormThreads + vector;
    float x[8], z[8], w[8];
    unpack8(reinterpret_cast<const uint4*>(input)[offset], x);
    unpack8(reinterpret_cast<const uint4*>(gate)[offset], z);
    unpack8(reinterpret_cast<const uint4*>(weight)[vector % (kGatedNormHead / 8U)], w);
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

template <std::uint32_t Threads, class Cta>
__device__ __forceinline__ void quantize_gated_rmsnorm_body(
    const Cta& cta, EncodeShared<Threads>& shared, const hip_bfloat16* input,
    const hip_bfloat16* gate, const hip_bfloat16* weight, float eps, std::uint8_t* codes,
    float* scales, std::uint32_t* status, const CacheWarm& warm, std::uint32_t tokens) {
    static_assert(Threads % kWaveSize == 0U);
    constexpr std::uint32_t V = (kGatedNormThreads + Threads - 1U) / Threads;
    const std::uint32_t block = cta.block().x;
    if (block >= tokens) {
        warm_cache(warm, block - tokens, cta.grid().x - tokens, cta.thread(), cta.threads());
        return;
    }
    const std::uint32_t token = block, thread = cta.thread();
    float values[V][8];
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        if (v * Threads + thread >= kGatedNormThreads) continue;
        gated_norm_vector(input, gate, weight, eps, token, v * Threads + thread, values[v]);
    }
    encode_rows<Threads, V, kGatedNormThreads>(cta, shared, values, token, codes, scales,
                                               kGatedNormColumns, status);
}

__global__ __launch_bounds__(kGatedNormThreads) void fp8_quantize_gated_rmsnorm_kernel(
    const hip_bfloat16* input, const hip_bfloat16* gate, const hip_bfloat16* weight, float eps,
    std::uint8_t* codes, float* scales, std::uint32_t* status, CacheWarm warm,
    std::uint32_t tokens);

} // namespace ninfer::ops::r9700::linear::fp8_activation_kernels
