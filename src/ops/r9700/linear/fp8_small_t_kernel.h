#pragma once

// Row-scaled FP8 small-T projection kernel (fp8_small_t_linear.h), shared by its launcher and the
// persistent decode kernel.
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_small_t_linear.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear::fp8_small_t_kernels {

inline constexpr std::uint16_t kCanonicalBf16QuietNan = 0x7FC0U;

// One 128-thread CTA per sixteen weight rows, each wave accumulating one contiguous K quarter with
// native FP8 WMMA (rows are the M operand, tokens the N operand). Every lane streams 32 contiguous
// bytes of its row and token per 64-K step; the four WMMAs of a step pair those bytes identically,
// which only permutes the exact K sum. Four steps stay in flight behind the current four. The
// quarters are summed in order, then scaled by the weight-row and token scales and rounded to BF16.
// T 17..32 adds a second token tile (tokens 16 + axis) fed by the same weight bytes, with two steps
// in flight to hold the doubled activation registers.
inline constexpr std::uint32_t kSplit      = 4U;
inline constexpr std::uint32_t kSteps      = 4U;
inline constexpr std::uint32_t kKAlignment = 64U * kSplit * kSteps;
using I32x2                                = int __attribute__((ext_vector_type(2)));
using F32x8                                = float __attribute__((ext_vector_type(8)));

enum class Epilogue { Store, PairSplit, Residual };

struct KernelArgs {
    Fp8RowScaledWeight first{};
    Fp8RowScaledWeight second{}; // PairSplit: CTAs past first.rows / 16 project this weight
    const std::uint8_t* activation = nullptr;
    const float* token_scales      = nullptr;
    const std::uint32_t* status    = nullptr;
    std::uint32_t tokens           = 0;
    std::uint32_t columns          = 0;
    std::uint32_t split            = 0;
    hip_bfloat16* outputs[4]{}; // Store/Residual: [0]; PairSplit: first/second leading/trailing
};

template <std::uint32_t kTiles>
struct SmallTShared {
    float partial[kSplit][kTiles][8][32];
};

template <Epilogue Kind, std::uint32_t kTiles, class Cta>
__device__ __forceinline__ void small_t_body(const Cta& cta, SmallTShared<kTiles>& shared,
                                             const KernelArgs& a) {
    static_assert(kTiles == 1U || kTiles == 2U);
    constexpr std::uint32_t kInFlight = kTiles == 1U ? kSteps : 2U;
    auto& partial                     = shared.partial;
    const std::uint32_t lane = cta.thread() & 31U, wave = cta.thread() >> 5U;
    const std::uint32_t axis = lane & 15U, half = lane >> 4U;
    const std::uint32_t first_blocks = a.first.rows / 16U;
    const std::uint32_t block        = cta.block().x;
    const bool second                = Kind == Epilogue::PairSplit && block >= first_blocks;
    const Fp8RowScaledWeight& weight = second ? a.second : a.first;
    const std::uint32_t row_base     = (second ? block - first_blocks : block) * 16U;
    const std::uint32_t rows         = weight.rows;
    const std::uint32_t columns      = a.columns;
    // Publishes one rounded projection element (or the poison value) of token `token`.
    const auto publish = [&](std::uint32_t token, std::uint32_t row, bool poison, float value) {
        if constexpr (Kind == Epilogue::Store) {
            auto* out = reinterpret_cast<std::uint16_t*>(a.outputs[0]) +
                        static_cast<std::size_t>(token) * rows + row;
            *out      = poison ? kCanonicalBf16QuietNan : hip_bfloat16(value).data;
        } else if constexpr (Kind == Epilogue::PairSplit) {
            const bool leading        = row < a.split;
            hip_bfloat16* base        = a.outputs[(second ? 2U : 0U) + (leading ? 0U : 1U)];
            const std::uint32_t width = leading ? a.split : rows - a.split;
            auto* out = reinterpret_cast<std::uint16_t*>(base) +
                        static_cast<std::size_t>(token) * width + (leading ? row : row - a.split);
            *out      = poison ? kCanonicalBf16QuietNan : hip_bfloat16(value).data;
        } else {
            hip_bfloat16* out = a.outputs[0] + static_cast<std::size_t>(token) * rows + row;
            const float delta =
                poison ? __builtin_nanf("") : static_cast<float>(hip_bfloat16(value));
            *out = hip_bfloat16(static_cast<float>(*out) + delta);
        }
    };
    const std::uint32_t steps  = columns / 64U / kSplit;
    const std::uint32_t k_base = wave * steps * 64U + half * 32U;
    bool token[kTiles];
    const std::uint8_t* activation_row[kTiles];
#pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
        token[tile] = tile * 16U + axis < a.tokens;
        activation_row[tile] =
            a.activation +
            static_cast<std::size_t>(token[tile] ? tile * 16U + axis : 0U) * columns + k_base;
    }
    const std::uint8_t* weight_row =
        weight.codes + static_cast<std::size_t>(row_base + axis) * columns + k_base;

    struct Step {
        uint4 w0, w1;
        uint4 a[kTiles][2];
    };

    const auto load = [&](std::uint32_t step) {
        Step value;
        const uint4* w =
            reinterpret_cast<const uint4*>(weight_row + static_cast<std::size_t>(step) * 64U);
        value.w0 = w[0];
        value.w1 = w[1];
#pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
            if (token[tile]) {
                const uint4* x = reinterpret_cast<const uint4*>(
                    activation_row[tile] + static_cast<std::size_t>(step) * 64U);
                value.a[tile][0] = x[0];
                value.a[tile][1] = x[1];
            } else {
                value.a[tile][0] = uint4{};
                value.a[tile][1] = uint4{};
            }
        }
        return value;
    };
    F32x8 accumulator[kTiles]{};
    const auto consume = [&](const Step& value) {
        const std::uint32_t w[8]{value.w0.x, value.w0.y, value.w0.z, value.w0.w,
                                 value.w1.x, value.w1.y, value.w1.z, value.w1.w};
#pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
            const uint4* at = value.a[tile];
            const std::uint32_t x[8]{at[0].x, at[0].y, at[0].z, at[0].w,
                                     at[1].x, at[1].y, at[1].z, at[1].w};
#pragma unroll
            for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
                const I32x2 weight_operand{static_cast<int>(w[2U * chunk]),
                                           static_cast<int>(w[2U * chunk + 1U])};
                const I32x2 activation_operand{static_cast<int>(x[2U * chunk]),
                                               static_cast<int>(x[2U * chunk + 1U])};
                accumulator[tile] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    weight_operand, activation_operand, accumulator[tile]);
            }
        }
    };
    Step current[kInFlight];
#pragma unroll
    for (std::uint32_t index = 0; index < kInFlight; ++index) current[index] = load(index);
    for (std::uint32_t step = 0; step + kInFlight < steps; step += kInFlight) {
        Step next[kInFlight];
#pragma unroll
        for (std::uint32_t index = 0; index < kInFlight; ++index)
            next[index] = load(step + kInFlight + index);
#pragma unroll
        for (std::uint32_t index = 0; index < kInFlight; ++index) consume(current[index]);
#pragma unroll
        for (std::uint32_t index = 0; index < kInFlight; ++index) current[index] = next[index];
    }
#pragma unroll
    for (std::uint32_t index = 0; index < kInFlight; ++index) consume(current[index]);
#pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#pragma unroll
        for (std::uint32_t item = 0; item < 8U; ++item)
            partial[wave][tile][item][lane] = accumulator[tile][item];
    cta.sync();
    if (wave == 0U) {
#pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
            if (!token[tile]) continue;
            const std::uint32_t t   = tile * 16U + axis;
            const float token_scale = a.token_scales[t];
#pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item) {
                float sum = partial[0][tile][item][lane];
#pragma unroll
                for (std::uint32_t quarter = 1; quarter < kSplit; ++quarter)
                    sum += partial[quarter][tile][item][lane];
                const std::uint32_t row = row_base + half * 8U + item;
                publish(t, row, a.status[t] != Fp8ActivationOk,
                        sum * weight.scales[row] * token_scale);
            }
        }
    }
}

template <Epilogue Kind, std::uint32_t kTiles>
__global__ __launch_bounds__(32U * kSplit) void fp8_small_t_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ SmallTShared<kTiles> shared;
    small_t_body<Kind, kTiles>(persistent::LaunchCta{}, shared, a);
#else
    (void)a;
#endif
}

} // namespace ninfer::ops::r9700::linear::fp8_small_t_kernels
