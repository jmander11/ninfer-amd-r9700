#pragma once

// FP8LUT4 small-T projection kernel (T <= 16 per token tile; kTiles = 2 serves T 17..32): one CTA
// context per 16 rows of the first or the second projection. Shared by the Linear launcher
// (fp8lut4_linear.hip) and the persistent decode kernel.
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/fp8lut4_pipeline_impl.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear::fp8lut4_kernels {

using fp8lut4::decode8;
using fp8lut4::F32x8;
using fp8lut4::I32x2;
using U32x4 = std::uint32_t __attribute__((ext_vector_type(4)));

inline constexpr std::uint16_t kCanonicalBf16QuietNan = 0x7FC0U;

// Branch-free round-to-nearest-even to BF16 (identical to hip_bfloat16 for every non-NaN value;
// NaN becomes the canonical quiet NaN).
__device__ __forceinline__ std::uint16_t round_bf16(float value) {
    const std::uint32_t bits    = __float_as_uint(value);
    const std::uint32_t rounded = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U;
    return value != value ? kCanonicalBf16QuietNan : static_cast<std::uint16_t>(rounded);
}

__device__ __forceinline__ float bf16_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

// One launch projects `weight` (and, small T, `second` in CTAs past weight.rows / 16).
struct KernelArgs {
    Fp8Lut4Weight weight{};
    Fp8Lut4Output output{};
    Fp8Lut4Weight second{};
    Fp8Lut4Output second_output{};
    const std::uint8_t* activation = nullptr;
    const float* token_scales      = nullptr;
    const std::uint32_t* status    = nullptr;
    std::uint32_t tokens           = 0;
};

// Element (token, row) of one published output.
__device__ __forceinline__ std::uint16_t* output_at(const Fp8Lut4Weight& weight,
                                                    const Fp8Lut4Output& output,
                                                    std::uint32_t token, std::uint32_t row) {
    const std::uint32_t rows = weight.rows, split = output.split;
    const bool leading        = split == 0U || row < split;
    const std::uint32_t width = split == 0U ? rows : (leading ? split : rows - split);
    auto* base = reinterpret_cast<std::uint16_t*>(leading ? output.leading : output.trailing);
    return base + static_cast<std::size_t>(token) * width + (leading ? row : row - split);
}

__device__ __forceinline__ std::uint16_t* output_at(const KernelArgs& a, std::uint32_t token,
                                                    std::uint32_t row) {
    return output_at(a.weight, a.output, token, row);
}

// BF16(silu(g) * u) of the BF16-rounded gate and up projections. The hardware exponential and
// reciprocal (a few FP32 ulps before the BF16 rounding) keep the fully unrolled prefill epilogue
// compact; accurate expf/division sequences tripled the kernel's code and slowed its main loop.
__device__ __forceinline__ std::uint16_t silu_pair_value(float gate, float up) {
    const float g = bf16_float(round_bf16(gate));
    const float u = bf16_float(round_bf16(up));
    return round_bf16(g * __builtin_amdgcn_rcpf(1.0F + __expf(-g)) * u);
}

template <bool kSiluPair, std::uint32_t kSplit, std::uint32_t kTiles>
struct SmallTShared {
    uint2 table[256];
    float partial[kSplit][kTiles][8][32];
    float staged[kSiluPair ? 16 : 1][16U * kTiles];
};

template <bool kAccumulate, bool kSiluPair, std::uint32_t kSplit, std::uint32_t kSteps,
          std::uint32_t kTiles, class Cta>
__device__ __forceinline__ void
small_t_body(const Cta& cta, SmallTShared<kSiluPair, kSplit, kTiles>& shared, const KernelArgs& a) {
    const std::uint32_t first_blocks = a.weight.rows / 16U;
    const std::uint32_t block        = cta.block().x;
    const bool second                = block >= first_blocks;
    // By value: the CTA-uniform selection stays in scalar registers.
    const Fp8Lut4Weight weight   = second ? a.second : a.weight;
    const Fp8Lut4Output output   = second ? a.second_output : a.output;
    const std::uint32_t row_base = (second ? block - first_blocks : block) * 16U;
    fp8lut4::load_table(shared.table, cta.thread(), 32U * kSplit);
    const auto publish = [&](std::uint32_t token, std::uint32_t row, float value) {
        if constexpr (kSiluPair) {
            shared.staged[row - row_base][token] = value;
            return;
        }
        auto* out = persistent::global(output_at(weight, output, token, row));
        if (persistent::global(a.status)[token] != Fp8ActivationOk) {
            *out = kCanonicalBf16QuietNan;
        } else if constexpr (kAccumulate) {
            *out = round_bf16(bf16_float(*out) + bf16_float(round_bf16(value)));
        } else {
            *out = round_bf16(value);
        }
    };
    fp8lut4::small_t_rows<kSplit, kSteps, kTiles>(cta, weight, row_base, a.activation,
                                                  a.token_scales, a.tokens, shared.table,
                                                  shared.partial, publish);
    if constexpr (kSiluPair) {
        cta.sync();
        const std::uint32_t features = weight.rows / 2U;
        for (std::uint32_t index = cta.thread(); index < 8U * a.tokens; index += cta.threads()) {
            const std::uint32_t feature = index % 8U, token = index / 8U;
            auto* out = persistent::global(reinterpret_cast<std::uint16_t*>(output.leading)) +
                        static_cast<std::size_t>(token) * features + row_base / 2U + feature;
            *out      = persistent::global(a.status)[token] != Fp8ActivationOk
                            ? kCanonicalBf16QuietNan
                            : silu_pair_value(shared.staged[feature][token],
                                              shared.staged[feature + 8U][token]);
        }
    }
}

template <bool kAccumulate, bool kSiluPair, std::uint32_t kSplit, std::uint32_t kSteps,
          std::uint32_t kTiles>
__global__ __launch_bounds__(32U * kSplit) void fp8lut4_small_t_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ SmallTShared<kSiluPair, kSplit, kTiles> shared;
    small_t_body<kAccumulate, kSiluPair, kSplit, kSteps, kTiles>(persistent::LaunchCta{}, shared,
                                                                 a);
#else
    (void)a;
#endif
}

} // namespace ninfer::ops::r9700::linear::fp8lut4_kernels
