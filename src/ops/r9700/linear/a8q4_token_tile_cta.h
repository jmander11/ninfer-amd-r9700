#pragma once
#include "ops/r9700/linear/r9700_linear.h"
#include <hip/hip_fp16.h>
#include <cstdint>

// Token-tile CTA family of the concurrent-DFlash wide widths T17..64 (selected by
// select_a8q4_small_batch_wide_route). The prefill CTA computes a 128-token tile, so at T<=64
// half of its WMMA and scale work is discarded, and at N5120 its 40 N128 CTAs occupy 40 of the
// 64 CUs; the per-M-tile small-batch bodies re-read activations per wave. A CTA here covers
// exactly ceil(T/16) token fragments with every staged G64 activation fragment shared by all of
// its row fragments through LDS, and narrows N or splits K inside the CTA to raise CU fill and
// the bytes in flight. Per output: exact INT32 G64 dots (the prefill CTA's high-chain origin and
// <<4 fold), one FP32 FMA with the exact FP16*FP16 scale product per group in ascending order
// inside each K slice, and the later slices' partial sums added by slice zero in ascending slice
// order before one BF16 cast.
namespace ninfer::ops::r9700::linear::detail {
using TokenTileI2 = int __attribute__((ext_vector_type(2)));
using TokenTileI8 = int __attribute__((ext_vector_type(8)));
using TokenTileF8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

// Mt token fragments per CTA, Tf of them per wave; Rf row fragments per wave and Rb row waves;
// Ks K slices of contiguous ascending G64 ranges; Pd G64 groups of global loads in flight.
template <unsigned Mt, unsigned Tf, unsigned Rf, unsigned Rb, unsigned Ks, unsigned Pd>
struct TokenTileConfig {
    static_assert(Mt >= 1 && Mt <= 4 && Tf >= 1 && Mt % Tf == 0 && Rf >= 1 && Rb >= 1 && Ks >= 1 &&
                  (Pd == 1 || Pd == 2) && (Rf * Rb == 1 || Rf * Rb % 2 == 0));
    static constexpr unsigned Tf_ = Tf, Rf_ = Rf, Ks_ = Ks, Pd_ = Pd;
    static constexpr unsigned TokenWaves   = Mt / Tf;
    static constexpr unsigned SliceWaves   = TokenWaves * Rb;
    static constexpr unsigned Waves        = SliceWaves * Ks;
    static constexpr unsigned Threads      = 32U * Waves;
    static constexpr unsigned SliceThreads = 32U * SliceWaves;
    static constexpr unsigned Tokens       = 16U * Mt;
    static constexpr unsigned RowTiles     = Rf * Rb;
    static constexpr unsigned Rows         = 16U * RowTiles;
    // Per slice and group: one unit is 16 bytes of each activation plane (32 K of one token), one
    // 16-byte quarter of a 512-byte Q4N16K16 tile-group, or one FP16 scale.
    static constexpr unsigned ActivationUnits = 2U * Tokens;
    static constexpr unsigned WeightUnits     = 32U * RowTiles;
    static constexpr unsigned ScaleUnits      = Tokens + Rows;
    static constexpr unsigned ActivationLoads =
        (ActivationUnits + SliceThreads - 1U) / SliceThreads;
    static constexpr unsigned WeightLoads  = (WeightUnits + SliceThreads - 1U) / SliceThreads;
    static constexpr unsigned ScaleLoads   = (ScaleUnits + SliceThreads - 1U) / SliceThreads;
    static constexpr unsigned Accumulators = Tf * Rf;
};

// [pair][row] u64 images. Arrays of at least 32 rows move odd pairs (the upper half-wave) by 16
// rows so the two half-waves read disjoint bank halves; a 16-row array already does.
template <unsigned Rows>
__device__ __forceinline__ unsigned token_tile_lds_index(unsigned pair, unsigned row) {
    if constexpr (Rows >= 32U)
        return pair * Rows + (row ^ ((pair & 1U) << 4U));
    else
        return pair * Rows + row;
}

template <unsigned Rows>
struct TokenTileStage {
    std::uint64_t activation_low[4U * 64U];
    std::uint64_t activation_high[4U * 64U];
    std::uint64_t weights[4U * Rows];
    // [0, 64) token scales, then [64, 64 + Rows) row scales.
    float scales[64U + Rows];
};

template <class S>
struct TokenTileLoad {
    uint4 activation_low[S::ActivationLoads], activation_high[S::ActivationLoads];
    uint4 weights[S::WeightLoads];
    // The aligned 32-bit word holding each FP16 scale and that scale's bit offset in it: a full
    // VGPR per in-flight scale, so two groups' scales never share one VGPR's halves (which would
    // make the older group's store wait for the newer group's loads).
    std::uint32_t scales[S::ScaleLoads], scale_shift[S::ScaleLoads];
};

// Branch-free per-thread load plan. Excess threads load unit (index mod count),
// Staging stores are restricted to the unit owner to avoid cross-wave LDS races.
// Byte offsets are from uniform plane bases (every plane is below 4 GiB); a scale unit
// reads the token or the row scale plane with that plane's per-group stride. Both scale planes
// are 4-byte aligned with an even scale count per token or row tile-group, so the aligned word
// holding a scale lies inside its plane.
template <class S>
struct TokenTilePlan {
    std::uint32_t activation[S::ActivationLoads], weight[S::WeightLoads], scale[S::ScaleLoads];
    const std::uint8_t* scale_base[S::ScaleLoads];
    std::uint32_t scale_stride[S::ScaleLoads];
    std::uint32_t activation_lds[S::ActivationLoads], weight_lds[S::WeightLoads],
        scale_lds[S::ScaleLoads];
};

template <class S>
__device__ __forceinline__ TokenTilePlan<S>
token_tile_plan(const std::uint16_t* activation_scales, const std::uint16_t* weight_scales,
                unsigned slice_thread, unsigned tokens, unsigned row_base, unsigned groups) {
    TokenTilePlan<S> plan;
    // Rows of a partial token tile stage the last real token: their products are never stored.
    const auto staged = [&](unsigned token) { return token < tokens ? token : tokens - 1U; };
#pragma unroll
    for (unsigned i = 0; i < S::ActivationLoads; ++i) {
        const unsigned unit    = (slice_thread + i * S::SliceThreads) % S::ActivationUnits;
        plan.activation[i]     = staged(unit >> 1U) * (groups * 32U) + (unit & 1U) * 16U;
        plan.activation_lds[i] = token_tile_lds_index<64>((unit & 1U) * 2U, unit >> 1U);
    }
#pragma unroll
    for (unsigned i = 0; i < S::WeightLoads; ++i) {
        const unsigned unit   = (slice_thread + i * S::SliceThreads) % S::WeightUnits;
        const unsigned within = unit & 31U;
        plan.weight[i]        = (row_base / 16U + (unit >> 5U)) * groups * 512U + within * 16U;
        plan.weight_lds[i] =
            token_tile_lds_index<S::Rows>(within >> 3U, (unit >> 5U) * 16U + (within & 7U) * 2U);
    }
#pragma unroll
    for (unsigned i = 0; i < S::ScaleLoads; ++i) {
        const unsigned unit = (slice_thread + i * S::SliceThreads) % S::ScaleUnits;
        const bool token    = unit < S::Tokens;
        const unsigned row  = row_base + unit - S::Tokens;
        plan.scale_base[i] =
            reinterpret_cast<const std::uint8_t*>(token ? activation_scales : weight_scales);
        plan.scale[i] =
            token ? staged(unit) * groups * 2U : ((row >> 4U) * groups * 16U + (row & 15U)) * 2U;
        plan.scale_stride[i] = token ? 2U : 32U;
        plan.scale_lds[i]    = token ? unit : 64U + unit - S::Tokens;
    }
    return plan;
}

template <class T>
__device__ __forceinline__ const T& token_tile_at(const void* base, std::uint32_t offset) {
    return *reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(base) + offset);
}

template <class S>
__device__ __forceinline__ TokenTileLoad<S>
token_tile_load(const std::uint8_t* low, const std::uint8_t* high, const std::uint8_t* codes,
                const TokenTilePlan<S>& plan, unsigned group) {
    TokenTileLoad<S> loaded;
    // Scales first: loads complete in order, so their conversion never waits for the code planes.
#pragma unroll
    for (unsigned i = 0; i < S::ScaleLoads; ++i) {
        const std::uint32_t offset = plan.scale[i] + group * plan.scale_stride[i];
        loaded.scales[i] =
            *reinterpret_cast<const std::uint32_t*>(plan.scale_base[i] + (offset & ~3U));
        loaded.scale_shift[i] = (offset & 2U) * 8U;
    }
#pragma unroll
    for (unsigned i = 0; i < S::ActivationLoads; ++i) {
        loaded.activation_low[i]  = token_tile_at<uint4>(low, plan.activation[i] + group * 32U);
        loaded.activation_high[i] = token_tile_at<uint4>(high, plan.activation[i] + group * 32U);
    }
#pragma unroll
    for (unsigned i = 0; i < S::WeightLoads; ++i)
        loaded.weights[i] = token_tile_at<uint4>(codes, plan.weight[i] + group * 512U);
    return loaded;
}

__device__ __forceinline__ std::uint64_t token_tile_u64(std::uint32_t low, std::uint32_t high) {
    return (static_cast<std::uint64_t>(high) << 32U) | low;
}

template <class S>
__device__ __forceinline__ void
token_tile_store(TokenTileStage<S::Rows>& stage, const TokenTileLoad<S>& loaded,
                 const TokenTilePlan<S>& plan, unsigned slice_thread) {
    // A unit's second K16 pair is odd: 64 rows up with the token swizzled by 16 (activations), or
    // the next row of the same pair (weights: the unit holds two adjacent rows of one pair).
#pragma unroll
    for (unsigned i = 0; i < S::ActivationLoads; ++i) {
        if (slice_thread + i * S::SliceThreads >= S::ActivationUnits) continue;
        const unsigned first = plan.activation_lds[i], second = (first + 64U) ^ 16U;
        const auto& lo                = loaded.activation_low[i];
        const auto& hi                = loaded.activation_high[i];
        stage.activation_low[first]   = token_tile_u64(lo.x, lo.y);
        stage.activation_low[second]  = token_tile_u64(lo.z, lo.w);
        stage.activation_high[first]  = token_tile_u64(hi.x, hi.y);
        stage.activation_high[second] = token_tile_u64(hi.z, hi.w);
    }
#pragma unroll
    for (unsigned i = 0; i < S::WeightLoads; ++i) {
        if (slice_thread + i * S::SliceThreads >= S::WeightUnits) continue;
        const auto& w                          = loaded.weights[i];
        stage.weights[plan.weight_lds[i]]      = token_tile_u64(w.x, w.y);
        stage.weights[plan.weight_lds[i] + 1U] = token_tile_u64(w.z, w.w);
    }
#pragma unroll
    for (unsigned i = 0; i < S::ScaleLoads; ++i) {
        if (slice_thread + i * S::SliceThreads >= S::ScaleUnits) continue;
        stage.scales[plan.scale_lds[i]] = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(loaded.scales[i] >> loaded.scale_shift[i])));
    }
}

__device__ __forceinline__ TokenTileI2 token_tile_fragment(std::uint64_t value) {
    return TokenTileI2{static_cast<int>(static_cast<std::uint32_t>(value)),
                       static_cast<int>(static_cast<std::uint32_t>(value >> 32U))};
}

// One G64 group of the wave's Tf x Rf fragments: the production prefill CTA's exact integer chain
// (high from 0x04B40000, <<4, low) and its FP32 scale product and FMA.
template <class S>
__device__ __forceinline__ void
token_tile_group(const TokenTileStage<S::Rows>& stage, unsigned wave_token, unsigned wave_row,
                 unsigned axis, unsigned lane_group, const TokenTileI8& high_origin,
                 TokenTileF8 (&totals)[S::Tf_][S::Rf_]) {
    constexpr unsigned Tf = S::Tf_, Rf = S::Rf_;
    TokenTileI2 low[2][Tf], high[2][Tf], weights[2][Rf];
#pragma unroll
    for (unsigned half = 0; half < 2U; ++half) {
        const unsigned pair = half * 2U + lane_group;
#pragma unroll
        for (unsigned f = 0; f < Tf; ++f) {
            const unsigned index = token_tile_lds_index<64>(pair, wave_token + f * 16U + axis);
            low[half][f]         = token_tile_fragment(stage.activation_low[index]);
            high[half][f]        = token_tile_fragment(stage.activation_high[index]);
        }
#pragma unroll
        for (unsigned r = 0; r < Rf; ++r)
            weights[half][r] = token_tile_fragment(
                stage.weights[token_tile_lds_index<S::Rows>(pair, wave_row + r * 16U + axis)]);
    }
    float activation_scale[Tf][8];
#pragma unroll
    for (unsigned f = 0; f < Tf; ++f)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e)
            activation_scale[f][e] = stage.scales[wave_token + f * 16U + lane_group * 8U + e];
    float weight_scale[Rf];
#pragma unroll
    for (unsigned r = 0; r < Rf; ++r)
        weight_scale[r] = stage.scales[64U + wave_row + r * 16U + axis];
#pragma unroll
    for (unsigned f = 0; f < Tf; ++f) {
#pragma unroll
        for (unsigned r = 0; r < Rf; ++r) {
            TokenTileI8 dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, high[0][f], true, weights[0][r], high_origin, false);
            dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, high[1][f], true,
                                                                   weights[1][r], dot, false);
            dot = dot << 4;
            dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, low[0][f], true,
                                                                   weights[0][r], dot, false);
            dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, low[1][f], true,
                                                                   weights[1][r], dot, false);
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e)
                totals[f][r][e] = fmaf(__int_as_float(dot[e]) - 12582912.0F,
                                       activation_scale[f][e] * weight_scale[r], totals[f][r][e]);
        }
    }
    // As in the production prefill CTA: each WMMA interleaved with four independent VALU.
#pragma unroll
    for (unsigned instruction = 0; instruction < 4U * Tf * Rf; ++instruction) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x002, 4, 0);
    }
}

__device__ __forceinline__ void token_tile_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <class S>
struct TokenTileShared {
    union {
        TokenTileStage<S::Rows> stage[S::Ks_][2];
        // Later slices' sums, [(slice-1, wave, accumulator)][element][lane]; one slot when Ks=1.
        float partial[S::Ks_ > 1U ? (S::Ks_ - 1U) * S::SliceWaves* S::Accumulators : 1U][8][32];
    };
};

// Accumulate publishes BF16(output + BF16(projection)) in place (projected-residual boundary).
// A nonzero activation status publishes the canonical BF16 quiet NaN for every output.
template <class S, bool Accumulate>
__global__ __launch_bounds__(S::Threads) void a8q4_token_tile_kernel(
    const std::uint8_t* low, const std::uint8_t* high, const std::uint16_t* activation_scales,
    const std::uint32_t* status, const std::uint8_t* codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows, std::uint32_t groups) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    constexpr unsigned Tf = S::Tf_, Rf = S::Rf_, Ks = S::Ks_, Pd = S::Pd_;
    __shared__ TokenTileShared<S> shared;
    const unsigned thread = threadIdx.x, lane = thread & 31U;
    const unsigned wave  = __builtin_amdgcn_readfirstlane(thread >> 5U);
    const unsigned slice = wave / S::SliceWaves, slice_wave = wave % S::SliceWaves;
    const unsigned slice_thread = slice_wave * 32U + lane;
    const unsigned axis = lane & 15U, lane_group = lane >> 4U;
    const unsigned row_base    = blockIdx.x * S::Rows;
    const unsigned wave_token  = (slice_wave % S::TokenWaves) * Tf * 16U;
    const unsigned wave_row    = (slice_wave / S::TokenWaves) * Rf * 16U;
    const auto for_each_output = [&](auto&& visit) {
#    pragma unroll
        for (unsigned f = 0; f < Tf; ++f)
#    pragma unroll
            for (unsigned r = 0; r < Rf; ++r)
#    pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const unsigned token = wave_token + f * 16U + lane_group * 8U + e;
                    if (token < tokens)
                        visit(f, r, e,
                              output[static_cast<std::size_t>(token) * rows + row_base + wave_row +
                                     r * 16U + axis]);
                }
    };
    if (*status != Q4G64ActivationOk) {
        if (slice != 0U) return;
        for_each_output(
            [](unsigned, unsigned, unsigned, hip_bfloat16& target) { target.data = 0x7fc1; });
        return;
    }
    const unsigned slice_groups = groups / Ks, first = slice * slice_groups;
    const TokenTilePlan<S> plan = token_tile_plan<S>(activation_scales, weight_scales, slice_thread,
                                                     tokens, row_base, groups);
    const auto load             = [&](unsigned group) {
        return token_tile_load<S>(low, high, codes, plan, first + group);
    };
    TokenTileStage<S::Rows>(&stage)[2] = shared.stage[slice];
    TokenTileF8 totals[Tf][Rf]{};
    TokenTileI8 high_origin;
#    pragma unroll
    for (unsigned e = 0; e < 8U; ++e) high_origin[e] = 0x04B40000;
    const auto compute = [&](unsigned bank) {
        token_tile_group<S>(stage[bank], wave_token, wave_row, axis, lane_group, high_origin,
                            totals);
    };
    token_tile_store<S>(stage[0], load(0U), plan, slice_thread);
    if constexpr (Pd == 1U) {
        token_tile_barrier();
        for (unsigned group = 0; group + 1U < slice_groups; ++group) {
            const auto next = load(group + 1U);
            __builtin_amdgcn_sched_barrier(0);
            compute(group & 1U);
            __builtin_amdgcn_sched_barrier(0);
            token_tile_store<S>(stage[(group + 1U) & 1U], next, plan, slice_thread);
            token_tile_barrier();
        }
        compute((slice_groups - 1U) & 1U);
    } else {
        // Two groups of loads in flight across each compute; slice_groups is even.
        auto pending = load(1U);
        token_tile_barrier();
        for (unsigned group = 0; group + 2U < slice_groups; group += 2U) {
            const auto even = load(group + 2U);
            __builtin_amdgcn_sched_barrier(0);
            compute(0U);
            __builtin_amdgcn_sched_barrier(0);
            token_tile_store<S>(stage[1], pending, plan, slice_thread);
            token_tile_barrier();
            pending = load(group + 3U);
            __builtin_amdgcn_sched_barrier(0);
            compute(1U);
            __builtin_amdgcn_sched_barrier(0);
            token_tile_store<S>(stage[0], even, plan, slice_thread);
            token_tile_barrier();
        }
        compute(0U);
        token_tile_store<S>(stage[1], pending, plan, slice_thread);
        token_tile_barrier();
        compute(1U);
    }
    if constexpr (Ks > 1U) {
        token_tile_barrier();
        if (slice != 0U) {
#    pragma unroll
            for (unsigned f = 0; f < Tf; ++f)
#    pragma unroll
                for (unsigned r = 0; r < Rf; ++r)
#    pragma unroll
                    for (unsigned e = 0; e < 8U; ++e)
                        shared
                            .partial[((slice - 1U) * S::SliceWaves + slice_wave) * S::Accumulators +
                                     f * Rf + r][e][lane] = totals[f][r][e];
        }
        token_tile_barrier();
        if (slice != 0U) return;
#    pragma unroll
        for (unsigned s = 0; s + 1U < Ks; ++s)
#    pragma unroll
            for (unsigned f = 0; f < Tf; ++f)
#    pragma unroll
                for (unsigned r = 0; r < Rf; ++r)
#    pragma unroll
                    for (unsigned e = 0; e < 8U; ++e)
                        totals[f][r][e] +=
                            shared.partial[(s * S::SliceWaves + slice_wave) * S::Accumulators +
                                           f * Rf + r][e][lane];
    }
    for_each_output([&](unsigned f, unsigned r, unsigned e, hip_bfloat16& target) {
        const hip_bfloat16 projection = static_cast<hip_bfloat16>(totals[f][r][e]);
        if constexpr (Accumulate)
            target = static_cast<hip_bfloat16>(static_cast<float>(target) +
                                               static_cast<float>(projection));
        else
            target = projection;
    });
#else
    (void)low;
    (void)high;
    (void)activation_scales;
    (void)status;
    (void)codes;
    (void)weight_scales;
    (void)output;
    (void)tokens;
    (void)rows;
    (void)groups;
#endif
}
} // namespace ninfer::ops::r9700::linear::detail
