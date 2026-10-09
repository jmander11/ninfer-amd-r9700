#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/fp8lut4_pipeline_impl.h"
#include "ops/r9700/linear/fp8lut4_small_t_kernel.h"
#include "ops/r9700/linear/fp8_small_t_linear.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::r9700::linear {
namespace {

using fp8lut4_kernels::bf16_float; // NOLINT(misc-unused-using-decls): device-pass use
using fp8lut4_kernels::decode8;    // NOLINT(misc-unused-using-decls): device-pass use
using fp8lut4_kernels::F32x8;
using fp8lut4_kernels::fp8lut4_small_t_kernel;
using fp8lut4_kernels::I32x2;
using fp8lut4_kernels::kCanonicalBf16QuietNan; // NOLINT(misc-unused-using-decls): device-pass use
using fp8lut4_kernels::KernelArgs;
using fp8lut4_kernels::output_at;       // NOLINT(misc-unused-using-decls): device-pass use
using fp8lut4_kernels::round_bf16;      // NOLINT(misc-unused-using-decls): device-pass use
using fp8lut4_kernels::silu_pair_value; // NOLINT(misc-unused-using-decls): device-pass use
using fp8lut4_kernels::U32x4;

// ---- Mid T (T 33..128, kTiles token tiles of 16): eight waves; wave w owns row block w %
// kRowBlocks (16 rows) over K split kSplit-ways (w / kRowBlocks). Each 64-column step, the waves of
// one K split stage that step's T x 64 activation bytes into LDS once for all their row blocks,
// instead of every 16-row CTA re-reading the whole activation (the small-T route's L2 cost past T
// 64). Weight fragments stream per wave exactly as small_t_rows (slot 16 half + axis of the N16 x
// K64 tile), paired with the identical activation bytes, so the K sum is the same permutation. The
// K splits combine in order through LDS and wave split 0 publishes; the SiLU pair's up rows sit
// in lane ^ 16 of the same fragment.
constexpr std::uint32_t kMidStride = 80U; // 64 bytes + 16: the 16 token rows hit distinct banks

template <bool kAccumulate, bool kSiluPair, std::uint32_t kRowBlocks, std::uint32_t kSplit,
          std::uint32_t kTiles, bool kRawFp8 = false>
__global__ __launch_bounds__(256) void fp8lut4_mid_t_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    static_assert(kRowBlocks * kSplit == 8U && kTiles >= 1U && kTiles <= 8U);
    constexpr std::uint32_t kTokens          = 16U * kTiles;
    constexpr std::uint32_t kGroupThreads    = 32U * kRowBlocks;
    constexpr std::uint32_t kUnits           = (kTokens * 4U + kGroupThreads - 1U) / kGroupThreads;
    constexpr std::uint32_t kActivationBytes = 2U * kSplit * kTokens * kMidStride;
    constexpr std::uint32_t kPartialBytes = kSplit > 1U ? kRowBlocks * kTiles * 8U * 32U * 4U : 0U;
    __shared__ uint2 table[256];
    __shared__ __attribute__((aligned(
        16))) std::uint8_t lds[kActivationBytes > kPartialBytes ? kActivationBytes : kPartialBytes];
    const std::uint32_t thread = threadIdx.x, lane = thread & 31U;
    const std::uint32_t wave  = __builtin_amdgcn_readfirstlane(thread >> 5U);
    const std::uint32_t block = wave % kRowBlocks, split = wave / kRowBlocks;
    const std::uint32_t axis = lane & 15U, half = lane >> 4U;
    const std::uint32_t row_base = (blockIdx.x * kRowBlocks + block) * 16U;
    const std::uint32_t columns = a.weight.columns, slabs = columns / 64U;
    const std::uint32_t steps = slabs / kSplit, first_slab = split * steps;
    const std::uint32_t tokens = a.tokens;
    if constexpr (!kRawFp8) fp8lut4::load_table(table, threadIdx.x, 256U);

    const std::size_t first_tile = static_cast<std::size_t>(row_base / 16U) * slabs + first_slab;
    // kRawFp8 (row-scaled E4M3 rows, fp8_row_scaled_prefill_linear): the same (row, 32 columns)
    // fragment is read where the row-major weight stores it.
    const std::uint8_t* raw_codes = a.weight.codes +
                                    static_cast<std::size_t>(row_base + axis) * columns +
                                    first_slab * 64U + half * 32U;
    const std::uint32_t slot      = half * 16U + axis;
    const std::uint8_t* codes     = a.weight.codes + first_tile * 512U + slot * 16U;
    const std::uint8_t* groups    = a.weight.groups + first_tile * 32U + slot;
    // Staging unit u of this split's group: token u / 4, 16-byte part u % 4 of the step.
    const std::uint32_t group_thread = thread % kGroupThreads;
    const auto stage_load            = [&](std::uint32_t step, uint4(&units)[kUnits]) {
#    pragma unroll
        for (std::uint32_t i = 0; i < kUnits; ++i) {
            const std::uint32_t unit  = group_thread + i * kGroupThreads;
            const std::uint32_t token = unit / 4U, part = unit % 4U;
            units[i] = token < tokens && unit < kTokens * 4U
                           ? *reinterpret_cast<const uint4*>(
                                 a.activation + static_cast<std::size_t>(token) * columns +
                                 static_cast<std::size_t>(first_slab + step) * 64U + part * 16U)
                           : uint4{};
        }
    };
    const auto stage_store = [&](const uint4(&units)[kUnits], std::uint32_t buffer) {
        std::uint8_t* base = lds + (buffer * kSplit + split) * kTokens * kMidStride;
#    pragma unroll
        for (std::uint32_t i = 0; i < kUnits; ++i) {
            const std::uint32_t unit = group_thread + i * kGroupThreads;
            if (unit < kTokens * 4U)
                *reinterpret_cast<uint4*>(base + (unit / 4U) * kMidStride + (unit % 4U) * 16U) =
                    units[i];
        }
    };

    struct Weight {
        uint4 w;
        uint4 raw;
        std::uint32_t g;
    };

    const auto weight_load = [&](std::uint32_t step) {
        Weight value{};
        if constexpr (kRawFp8) {
            const uint4* row =
                reinterpret_cast<const uint4*>(raw_codes + static_cast<std::size_t>(step) * 64U);
            value.w   = row[0];
            value.raw = row[1];
        } else {
            value.w =
                *reinterpret_cast<const uint4*>(codes + static_cast<std::size_t>(step) * 512U);
            value.g = groups[static_cast<std::size_t>(step) * 32U];
        }
        return value;
    };

    F32x8 accumulator[kTiles]{};
    Weight current = weight_load(0U);
    {
        uint4 units[kUnits];
        stage_load(0U, units);
        stage_store(units, 0U);
    }
    __syncthreads(); // codebook and the first activation step
    for (std::uint32_t step = 0; step < steps; ++step) {
        const bool more = step + 1U < steps;
        Weight next{};
        uint4 units[kUnits];
        if (more) {
            next = weight_load(step + 1U);
            stage_load(step + 1U, units);
        }
        std::uint32_t w[8];
        if constexpr (kRawFp8) {
            const std::uint32_t raw[8]{current.w.x,   current.w.y,   current.w.z,   current.w.w,
                                       current.raw.x, current.raw.y, current.raw.z, current.raw.w};
#    pragma unroll
            for (std::uint32_t i = 0; i < 8U; ++i) w[i] = raw[i];
        } else {
            const uint2 book = table[current.g];
            const uint2 w0 = decode8(current.w.x, book), w1 = decode8(current.w.y, book);
            const uint2 w2 = decode8(current.w.z, book), w3 = decode8(current.w.w, book);
            const std::uint32_t decoded[8]{w0.x, w0.y, w1.x, w1.y, w2.x, w2.y, w3.x, w3.y};
#    pragma unroll
            for (std::uint32_t i = 0; i < 8U; ++i) w[i] = decoded[i];
        }
        const std::uint8_t* activation =
            lds + ((step & 1U) * kSplit + split) * kTokens * kMidStride;
#    pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
            const uint4* xt = reinterpret_cast<const uint4*>(
                activation + (tile * 16U + axis) * kMidStride + half * 32U);
            const uint4 x0 = xt[0], x1 = xt[1];
            const std::uint32_t x[8]{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#    pragma unroll
            for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
                accumulator[tile] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    I32x2{static_cast<int>(w[2U * chunk]), static_cast<int>(w[2U * chunk + 1U])},
                    I32x2{static_cast<int>(x[2U * chunk]), static_cast<int>(x[2U * chunk + 1U])},
                    accumulator[tile]);
            }
        }
        if (more) {
            stage_store(units, (step + 1U) & 1U);
            current = next;
        }
        __syncthreads();
    }

    if constexpr (kSplit > 1U) {
        // The activation buffers are dead after the last barrier: split partials reuse them.
        float* partial = reinterpret_cast<float*>(lds);
        for (std::uint32_t s = 1; s < kSplit; ++s) {
            if (split == s) {
#    pragma unroll
                for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#    pragma unroll
                    for (std::uint32_t item = 0; item < 8U; ++item) {
                        float* p = partial + ((block * kTiles + tile) * 8U + item) * 32U + lane;
                        *p       = s == 1U ? accumulator[tile][item] : *p + accumulator[tile][item];
                    }
            }
            __syncthreads();
        }
        if (split != 0U) return;
#    pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#    pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item)
                accumulator[tile][item] +=
                    partial[((block * kTiles + tile) * 8U + item) * 32U + lane];
    }

#    pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
        const std::uint32_t token = tile * 16U + axis;
        const bool present        = token < tokens;
        const float token_scale   = present ? a.token_scales[token] : 0.0F;
        const bool poison         = present && a.status[token] != Fp8ActivationOk;
        float y[8];
#    pragma unroll
        for (std::uint32_t item = 0; item < 8U; ++item)
            y[item] = accumulator[tile][item] * a.weight.scales[row_base + half * 8U + item] *
                      token_scale;
        if constexpr (kSiluPair) {
            // Lane `half` 0 holds this block's eight gate rows, lane ^ 16 their up partners.
            float partner[8];
#    pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item)
                partner[item] = __shfl_xor(y[item], 16, 32);
            if (half == 0U && present) {
                auto* out = reinterpret_cast<std::uint16_t*>(a.output.leading) +
                            static_cast<std::size_t>(token) * (a.weight.rows / 2U) + row_base / 2U;
#    pragma unroll
                for (std::uint32_t item = 0; item < 8U; ++item)
                    out[item] =
                        poison ? kCanonicalBf16QuietNan : silu_pair_value(y[item], partner[item]);
            }
            continue;
        }
        if (!present) continue;
#    pragma unroll
        for (std::uint32_t item = 0; item < 8U; ++item) {
            std::uint16_t* out = output_at(a, token, row_base + half * 8U + item);
            if (poison) {
                *out = kCanonicalBf16QuietNan;
            } else if constexpr (kAccumulate) {
                *out = round_bf16(bf16_float(*out) + bf16_float(round_bf16(y[item])));
            } else {
                *out = round_bf16(y[item]);
            }
        }
    }
#else
    (void)a;
#endif
}

// A CTA barrier ordering only LDS: __syncthreads' workgroup fence also waits for every global
// load in flight, which would drain the deep mid-T prefetch at each step.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    // This wave's LDS accesses, explicitly (no builtin on gfx12; operand-free, so no load waits).
    // NOLINTNEXTLINE(portability-no-assembler): s_wait_dscnt, see the comment above
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ---- Deep mid-T (T 33..128, codebook weights; deep_shape selects it). The mid-T CTA (kRowBlocks
// blocks of 16 rows, K split kSplit-ways, each split's T x 64 activation bytes staged in LDS once
// per step) with kDepth 64-column steps of weight codes, group codes and staging units in flight
// per wave. Loads return in order: the mid-T kernel issues step s + 1's loads in step s and waits
// for all of them before its WMMAs (its conditional staging loads join with a full wait) and
// again in __syncthreads, one DRAM round trip per step. Here step s + 1's activation is loaded
// with its weight kDepth - 1 steps earlier, so waiting for it before the LDS store leaves steps
// s + 2 .. s + kDepth in flight. CTAs past weight.rows / (16 kRowBlocks) project `second` (a pair
// in one launch). The WMMA chain and the in-order K split combine are those of the mid-T kernel
// of the same kRowBlocks / kSplit; the epilogue publishes eight rows (or SiLU features) per lane
// with one 16-byte store (16-byte-aligned outputs). Barriers order LDS only (lds_barrier).
template <bool kAccumulate, bool kSiluPair, std::uint32_t kRowBlocks, std::uint32_t kSplit,
          std::uint32_t kTiles, std::uint32_t kDepth>
__global__ __launch_bounds__(256) void fp8lut4_mid_deep_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    static_assert(kRowBlocks * kSplit == 8U && kTiles >= 3U && kTiles <= 8U && kDepth >= 2U &&
                  kDepth <= 4U);
    constexpr std::uint32_t kTokens          = 16U * kTiles;
    constexpr std::uint32_t kGroupThreads    = 32U * kRowBlocks;
    constexpr std::uint32_t kUnits           = (kTokens * 4U + kGroupThreads - 1U) / kGroupThreads;
    constexpr std::uint32_t kActivationBytes = 2U * kSplit * kTokens * kMidStride;
    constexpr std::uint32_t kPartialBytes = kSplit > 1U ? kRowBlocks * kTiles * 8U * 32U * 4U : 0U;
    __shared__ uint2 table[256];
    __shared__ __attribute__((aligned(
        16))) std::uint8_t lds[kActivationBytes > kPartialBytes ? kActivationBytes : kPartialBytes];
    const std::uint32_t thread = threadIdx.x, lane = thread & 31U;
    const std::uint32_t wave  = __builtin_amdgcn_readfirstlane(thread >> 5U);
    const std::uint32_t block = wave % kRowBlocks, split = wave / kRowBlocks;
    const std::uint32_t axis = lane & 15U, half = lane >> 4U;
    const std::uint32_t first_ctas = a.weight.rows / (16U * kRowBlocks);
    const bool second              = blockIdx.x >= first_ctas;
    // By value: the CTA-uniform selection stays in scalar registers.
    const Fp8Lut4Weight weight = second ? a.second : a.weight;
    const Fp8Lut4Output output = second ? a.second_output : a.output;
    const std::uint32_t row_base =
        ((second ? blockIdx.x - first_ctas : blockIdx.x) * kRowBlocks + block) * 16U;
    const std::uint32_t columns = weight.columns, slabs = columns / 64U;
    const std::uint32_t steps = slabs / kSplit, first_slab = split * steps;
    const std::uint32_t tokens = a.tokens;
    fp8lut4::load_table(table, threadIdx.x, 256U);

    const std::size_t first_tile = static_cast<std::size_t>(row_base / 16U) * slabs + first_slab;
    const std::uint32_t slot     = half * 16U + axis;
    const std::uint8_t* codes    = weight.codes + first_tile * 512U + slot * 16U;
    const std::uint8_t* groups   = weight.groups + first_tile * 32U + slot;
    // Staging unit u of this split's group: token u / 4, 16-byte part u % 4 of the step. Every
    // load is unconditional (a conditional load joins its register with a wait for every load in
    // flight): absent tokens and units past the tile read the last token, whose WMMA columns are
    // never published.
    const std::uint32_t group_thread = thread % kGroupThreads;
    const std::uint8_t* unit_bytes[kUnits];
#    pragma unroll
    for (std::uint32_t i = 0; i < kUnits; ++i) {
        const std::uint32_t unit  = group_thread + i * kGroupThreads;
        const std::uint32_t token = min(unit / 4U, tokens - 1U);
        unit_bytes[i]             = a.activation + static_cast<std::size_t>(token) * columns +
                                    static_cast<std::size_t>(first_slab) * 64U + (unit % 4U) * 16U;
    }
    // The ring of kDepth steps in flight (slot = step % kDepth, a compile-time index everywhere),
    // in vector registers: uint4 struct copies become memcpys that keep the ring in private memory.
    U32x4 ring_w[kDepth];
    std::uint32_t ring_g[kDepth];
    U32x4 ring_x[kDepth][kUnits];
    const auto load = [&](std::uint32_t step, U32x4& w, std::uint32_t& g,
                          U32x4(&units)[kUnits]) __attribute__((always_inline)) {
        w = *reinterpret_cast<const U32x4*>(codes + static_cast<std::size_t>(step) * 512U);
        g = groups[static_cast<std::size_t>(step) * 32U];
#    pragma unroll
        for (std::uint32_t i = 0; i < kUnits; ++i)
            units[i] = *reinterpret_cast<const U32x4*>(unit_bytes[i] +
                                                       static_cast<std::size_t>(step) * 64U);
    };
    const auto stage_store = [&](const U32x4(&units)[kUnits],
                                 std::uint32_t buffer) __attribute__((always_inline)) {
        std::uint8_t* base = lds + (buffer * kSplit + split) * kTokens * kMidStride;
#    pragma unroll
        for (std::uint32_t i = 0; i < kUnits; ++i) {
            const std::uint32_t unit = group_thread + i * kGroupThreads;
            if (unit < kTokens * 4U)
                *reinterpret_cast<U32x4*>(base + (unit / 4U) * kMidStride + (unit % 4U) * 16U) =
                    units[i];
        }
    };

    F32x8 accumulator[kTiles]{};
    // Inlined everywhere: an outlined call would keep the ring in private memory.
    const auto run_step = [&]<std::uint32_t kSlot>(
                              std::uint32_t step) __attribute__((always_inline)) {
        const uint2 book      = table[ring_g[kSlot]];
        const U32x4 code_word = ring_w[kSlot];
        const uint2 w0 = decode8(code_word.x, book), w1 = decode8(code_word.y, book);
        const uint2 w2 = decode8(code_word.z, book), w3 = decode8(code_word.w, book);
        const std::uint32_t w[8]{w0.x, w0.y, w1.x, w1.y, w2.x, w2.y, w3.x, w3.y};
        // The slot is free: its weight is decoded and its activation was staged a step ago. The
        // last kDepth steps reload the final step (unused) instead of branching.
        load(step + kDepth < steps ? step + kDepth : steps - 1U, ring_w[kSlot], ring_g[kSlot],
             ring_x[kSlot]);
        const std::uint8_t* activation =
            lds + ((step & 1U) * kSplit + split) * kTokens * kMidStride;
#    pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
            const uint4* xt = reinterpret_cast<const uint4*>(
                activation + (tile * 16U + axis) * kMidStride + half * 32U);
            const uint4 x0 = xt[0], x1 = xt[1];
            const std::uint32_t x[8]{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#    pragma unroll
            for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
                accumulator[tile] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    I32x2{static_cast<int>(w[2U * chunk]), static_cast<int>(w[2U * chunk + 1U])},
                    I32x2{static_cast<int>(x[2U * chunk]), static_cast<int>(x[2U * chunk + 1U])},
                    accumulator[tile]);
            }
        }
        if (step + 1U < steps) stage_store(ring_x[(kSlot + 1U) % kDepth], (step + 1U) & 1U);
        lds_barrier();
    };
    using Slots = std::make_integer_sequence<std::uint32_t, kDepth>;

    // deep_fits guarantees steps >= kDepth.
    [&]<std::uint32_t... kSlot>(std::integer_sequence<std::uint32_t, kSlot...>)
        __attribute__((always_inline)) {
            (load(kSlot, ring_w[kSlot], ring_g[kSlot], ring_x[kSlot]), ...);
        }(Slots{});
    stage_store(ring_x[0], 0U);
    lds_barrier(); // codebook and the first activation step
    const std::uint32_t whole = steps - steps % kDepth;
    for (std::uint32_t base = 0; base < whole; base += kDepth) {
        [&]<std::uint32_t... kSlot>(std::integer_sequence<std::uint32_t, kSlot...>)
            __attribute__((always_inline)) {
                (run_step.template operator()<kSlot>(base + kSlot), ...);
            }(Slots{});
    }
    [&]<std::uint32_t... kSlot>(std::integer_sequence<std::uint32_t, kSlot...>)
        __attribute__((always_inline)) {
            ((whole + kSlot < steps ? run_step.template operator()<kSlot>(whole + kSlot) : void()),
             ...);
        }(Slots{});

    if constexpr (kSplit > 1U) {
        // The activation buffers are dead after the last barrier: split partials reuse them.
        float* partial = reinterpret_cast<float*>(lds);
        for (std::uint32_t s = 1; s < kSplit; ++s) {
            if (split == s) {
#    pragma unroll
                for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#    pragma unroll
                    for (std::uint32_t item = 0; item < 8U; ++item) {
                        float* p = partial + ((block * kTiles + tile) * 8U + item) * 32U + lane;
                        *p       = s == 1U ? accumulator[tile][item] : *p + accumulator[tile][item];
                    }
            }
            lds_barrier();
        }
        if (split != 0U) return;
#    pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#    pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item)
                accumulator[tile][item] +=
                    partial[((block * kTiles + tile) * 8U + item) * 32U + lane];
    }

    float row_scale[8];
#    pragma unroll
    for (std::uint32_t item = 0; item < 8U; ++item)
        row_scale[item] = weight.scales[row_base + half * 8U + item];
#    pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
        const std::uint32_t token = tile * 16U + axis;
        const bool present        = token < tokens;
        const float token_scale   = present ? a.token_scales[token] : 0.0F;
        const bool poison         = present && a.status[token] != Fp8ActivationOk;
        float y[8];
#    pragma unroll
        for (std::uint32_t item = 0; item < 8U; ++item)
            y[item] = accumulator[tile][item] * row_scale[item] * token_scale;
        std::uint16_t value[8];
        uint4* out = nullptr;
        if constexpr (kSiluPair) {
            // Lane `half` 0 holds this block's eight gate rows, lane ^ 16 their up partners.
            float partner[8];
#    pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item)
                partner[item] = __shfl_xor(y[item], 16, 32);
            if (half != 0U || !present) continue;
#    pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item)
                value[item] =
                    poison ? kCanonicalBf16QuietNan : silu_pair_value(y[item], partner[item]);
            out = reinterpret_cast<uint4*>(reinterpret_cast<std::uint16_t*>(output.leading) +
                                           static_cast<std::size_t>(token) * (weight.rows / 2U) +
                                           row_base / 2U);
        } else {
            if (!present) continue;
            // Eight consecutive rows of one side of the split (a multiple of 128).
            out = reinterpret_cast<uint4*>(output_at(weight, output, token, row_base + half * 8U));
            uint4 prior{};
            if constexpr (kAccumulate) prior = *out;
            const std::uint32_t prior_words[4]{prior.x, prior.y, prior.z, prior.w};
#    pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item) {
                if (poison) {
                    value[item] = kCanonicalBf16QuietNan;
                } else if constexpr (kAccumulate) {
                    const std::uint16_t prior_bits =
                        static_cast<std::uint16_t>(prior_words[item / 2U] >> (16U * (item & 1U)));
                    value[item] =
                        round_bf16(bf16_float(prior_bits) + bf16_float(round_bf16(y[item])));
                } else {
                    value[item] = round_bf16(y[item]);
                }
            }
        }
        *out = uint4{value[0] | (static_cast<std::uint32_t>(value[1]) << 16U),
                     value[2] | (static_cast<std::uint32_t>(value[3]) << 16U),
                     value[4] | (static_cast<std::uint32_t>(value[5]) << 16U),
                     value[6] | (static_cast<std::uint32_t>(value[7]) << 16U)};
    }
#else
    (void)a;
#endif
}

// ---- One token: a GEMV of 32 rows per CTA (one per lane) and four waves each owning a K quarter
// with four steps in flight. Each lane decodes its row's two 32-column groups per 64-column step
// (slots r % 16 and 16 + r % 16 of the N16 x K64 tile) and accumulates FP8 dot4 products against
// the broadcast activation; the quarters combine in order. No WMMA columns are wasted on absent
// tokens.
constexpr std::uint32_t kGemvSplit = 4U, kGemvSteps = 4U;

template <bool kAccumulate, bool kSiluPair>
__global__ __launch_bounds__(32U * kGemvSplit) void fp8lut4_t1_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ uint2 table[256];
    __shared__ float partial[kGemvSplit][32];
    const std::uint32_t first_blocks = a.weight.rows / 32U;
    const bool second                = blockIdx.x >= first_blocks;
    const Fp8Lut4Weight weight       = second ? a.second : a.weight;
    const Fp8Lut4Output output       = second ? a.second_output : a.output;
    const std::uint32_t row_base     = (second ? blockIdx.x - first_blocks : blockIdx.x) * 32U;
    fp8lut4::load_table(table, threadIdx.x, 32U * kGemvSplit);
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const std::uint32_t row     = row_base + lane;
    const std::uint32_t columns = weight.columns, steps_total = columns / 64U;
    const std::uint32_t steps    = steps_total / kGemvSplit;
    const std::size_t first_tile = static_cast<std::size_t>(row / 16U) * steps_total + wave * steps;
    const std::uint8_t* codes    = weight.codes + first_tile * 512U + (row % 16U) * 16U;
    const std::uint8_t* groups   = weight.groups + first_tile * 32U + row % 16U;
    const std::uint8_t* activation = a.activation + static_cast<std::size_t>(wave) * steps * 64U;

    struct Batch {
        uint4 c[kGemvSteps][2];
        std::uint32_t g[kGemvSteps][2];
        uint4 x[kGemvSteps][4];
    };

    const auto load = [&](std::uint32_t step) {
        Batch value;
#    pragma unroll
        for (std::uint32_t i = 0; i < kGemvSteps; ++i) {
            const std::uint8_t* tile  = codes + static_cast<std::size_t>(step + i) * 512U;
            value.c[i][0]             = *reinterpret_cast<const uint4*>(tile);
            value.c[i][1]             = *reinterpret_cast<const uint4*>(tile + 256U);
            const std::uint8_t* group = groups + static_cast<std::size_t>(step + i) * 32U;
            value.g[i][0]             = group[0];
            value.g[i][1]             = group[16];
            const uint4* x            = reinterpret_cast<const uint4*>(
                activation + static_cast<std::size_t>(step + i) * 64U);
            value.x[i][0] = x[0];
            value.x[i][1] = x[1];
            value.x[i][2] = x[2];
            value.x[i][3] = x[3];
        }
        return value;
    };
    float sum          = 0.0F;
    const auto consume = [&](const Batch& value) {
#    pragma unroll
        for (std::uint32_t i = 0; i < kGemvSteps; ++i) {
#    pragma unroll
            for (std::uint32_t half = 0; half < 2U; ++half) {
                const uint2 book = table[value.g[i][half]];
                const uint4 word = value.c[i][half];
                const uint2 w0 = decode8(word.x, book), w1 = decode8(word.y, book);
                const uint2 w2 = decode8(word.z, book), w3 = decode8(word.w, book);
                const uint4 x0 = value.x[i][2U * half], x1 = value.x[i][2U * half + 1U];
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w0.x),
                                                        static_cast<int>(x0.x), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w0.y),
                                                        static_cast<int>(x0.y), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w1.x),
                                                        static_cast<int>(x0.z), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w1.y),
                                                        static_cast<int>(x0.w), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w2.x),
                                                        static_cast<int>(x1.x), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w2.y),
                                                        static_cast<int>(x1.y), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w3.x),
                                                        static_cast<int>(x1.z), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w3.y),
                                                        static_cast<int>(x1.w), sum);
            }
        }
    };
    Batch current = load(0U);
    __syncthreads(); // codebook
    for (std::uint32_t step = kGemvSteps; step < steps; step += kGemvSteps) {
        const Batch next = load(step);
        consume(current);
        current = next;
    }
    consume(current);
    partial[wave][lane] = sum;
    __syncthreads();
    if (wave != 0U) return;
    float total = partial[0][lane];
#    pragma unroll
    for (std::uint32_t split = 1; split < kGemvSplit; ++split) total += partial[split][lane];
    const float y     = total * weight.scales[row] * a.token_scales[0];
    const bool poison = a.status[0] != Fp8ActivationOk;
    if constexpr (kSiluPair) {
        // Rows 16 b + i (gate) and 16 b + 8 + i (up) sit in lanes l and l ^ 8.
        const float partner = __shfl_xor(y, 8, 32);
        if ((lane & 8U) == 0U) {
            auto* out = reinterpret_cast<std::uint16_t*>(output.leading) + row_base / 2U +
                        (lane / 16U) * 8U + lane % 8U;
            *out      = poison ? kCanonicalBf16QuietNan : silu_pair_value(y, partner);
        }
    } else {
        std::uint16_t* out = output_at(weight, output, 0U, row);
        if (poison) {
            *out = kCanonicalBf16QuietNan;
        } else if constexpr (kAccumulate) {
            *out = round_bf16(bf16_float(*out) + bf16_float(round_bf16(y)));
        } else {
            *out = round_bf16(y);
        }
    }
#else
    (void)a;
#endif
}

// ---- Prefill: a 256-token x 128-row CTA of eight waves (4 along tokens x 2 along rows), each
// owning 4x4 WMMA fragments. Per 64-column slab every thread decodes one 32-column group of one
// weight row into the double-buffered LDS tile; the waves read the activation fragments straight
// from global memory, 32 contiguous bytes per lane, paired with the identical permutation of the
// decoded weight bytes. Scales apply once in the epilogue.
//
// The next slab is decoded and staged before the last of the four 16-column chunks, so its
// codebook read and decode VALU interleave with WMMAs instead of running between the last WMMA
// and the barrier (6-8% at T2048). The kernel is held to 240 VGPRs (six waves per SIMD), so three
// CTAs share a WGP instead of two (6-10% at T2048). kRawFp8 serves row-scaled E4M3 weights
// (row-major, fp8_row_scaled_prefill_linear): the staging copies each (row, half) slot's 32 raw
// bytes, loaded where they are stored, and the codebook is not loaded.
//
// The CTA shape is (kTokenGroups x 64 tokens) x (8 / kTokenGroups x kWaveRows rows): 512x64
// decodes each weight slab once per 512 tokens, 256x128 serves widths whose 512-token tiles
// would pad more, and 256x64 fills the device when few 128-row CTAs exist (prefill_shape). Only
// the first 2 x rows threads stage. Every output is one wave's accumulation over K in the same
// slab and chunk order, so every shape publishes identical bits.
constexpr std::uint32_t kStageChunk    = 3U;
constexpr std::uint32_t kPrefillTokens = 256U;
constexpr std::uint32_t kPrefillRows   = 128U;
constexpr std::uint32_t kSlabStride    = 72U; // 64 bytes + 8: two-way LDS conflicts at most

template <bool kAccumulate, bool kSiluPair, bool kRawFp8 = false, std::uint32_t kTokenGroups = 4U,
          std::uint32_t kWaveRows = 64U>
__global__ __launch_bounds__(256)
    __attribute__((amdgpu_waves_per_eu(6))) void fp8lut4_prefill_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    static_assert((kTokenGroups == 4U && (kWaveRows == 64U || kWaveRows == 32U)) ||
                  (kTokenGroups == 8U && kWaveRows == 64U));
    constexpr std::uint32_t kRows = 8U / kTokenGroups * kWaveRows, kFragments = kWaveRows / 16U;
    __shared__ uint2 table[256];
    __shared__ __attribute__((aligned(16))) std::uint8_t slab[2][kRows * kSlabStride];
    const std::uint32_t thread = threadIdx.x, lane = thread & 31U;
    const std::uint32_t wave = __builtin_amdgcn_readfirstlane(thread >> 5U);
    const std::uint32_t axis = lane & 15U, half = lane >> 4U;
    const std::uint32_t wave_tokens = (wave % kTokenGroups) * 64U;
    const std::uint32_t wave_rows   = (wave / kTokenGroups) * kWaveRows;
    // Wave-uniform: the first 2 kRows threads stage one (row, half) slot each.
    const bool stager              = thread < 2U * kRows;
    const std::uint32_t token_base = blockIdx.x * kTokenGroups * 64U;
    const std::uint32_t row_base   = blockIdx.y * kRows;
    const std::uint32_t columns    = a.weight.columns;
    const std::uint32_t tokens     = a.tokens;
    if constexpr (!kRawFp8) fp8lut4::load_table(table, threadIdx.x, 256U);

    // Weight staging: thread -> (row, 32-column half of the slab).
    const std::uint32_t stage_row = (stager ? thread : 0U) >> 1U, stage_half = thread & 1U;
    // N16 x K64 weight tiles: the staged (row, half) slot of slab 0; one tile per slab.
    const std::size_t stage_tile =
        static_cast<std::size_t>((row_base + stage_row) / 16U) * (columns / 64U);
    const std::uint32_t stage_slot = stage_half * 16U + stage_row % 16U;
    // Raw E4M3 rows: the same (row, half) slot read straight from the row-major weight.
    const std::uint8_t* stage_codes =
        kRawFp8 ? a.weight.codes + static_cast<std::size_t>(row_base + stage_row) * columns +
                      stage_half * 32U
                : a.weight.codes + stage_tile * 512U + stage_slot * 16U;
    const std::uint8_t* stage_groups = a.weight.groups + stage_tile * 32U + stage_slot;
    const std::uint32_t stage_offset = stage_row * kSlabStride + stage_half * 32U;
    // Activation fragments: token rows clamp to the last token (their results are never stored).
    // The epilogue's token scales and status words load here, landing during the main loop.
    const std::uint8_t* fragment[4];
    float fragment_scale[4];
    bool fragment_poison[4];
#    pragma unroll
    for (std::uint32_t i = 0; i < 4U; ++i) {
        std::uint32_t token = token_base + wave_tokens + i * 16U + axis;
        token               = token < tokens ? token : tokens - 1U;
        fragment[i]         = a.activation + static_cast<std::size_t>(token) * columns + half * 32U;
        fragment_scale[i]   = a.token_scales[token];
        fragment_poison[i]  = a.status[token] != Fp8ActivationOk;
    }
    // Wave-uniform count of 16-token fragments holding at least one real token: the rest of a
    // partial tile neither loads activations nor issues WMMAs (their results are never stored).
    const std::uint32_t first_token = token_base + wave_tokens;
    const std::uint32_t live        = __builtin_amdgcn_readfirstlane(
        first_token >= tokens ? 0U : min(4U, (tokens - first_token + 15U) / 16U));

    struct Load {
        uint4 w;
        std::uint32_t g;
        uint4 x[4][2];
    };

    const auto load = [&](std::uint32_t slab_index) {
        Load value;
        if constexpr (kRawFp8) {
            (void)slab_index; // raw rows are copied at staging time
        } else if (stager) {
            value.w = *reinterpret_cast<const uint4*>(stage_codes +
                                                      static_cast<std::size_t>(slab_index) * 512U);
            value.g = stage_groups[static_cast<std::size_t>(slab_index) * 32U];
        }
#    pragma unroll
        for (std::uint32_t i = 0; i < 4U; ++i) {
            if (i >= live) break;
            const uint4* x = reinterpret_cast<const uint4*>(
                fragment[i] + static_cast<std::size_t>(slab_index) * 64U);
            value.x[i][0] = x[0];
            value.x[i][1] = x[1];
        }
        return value;
    };
    // Raw E4M3 rows are copied through loop-local registers (never part of the carried Load).
    const auto load_raw = [&](std::uint32_t slab_index, uint4(&raw)[2]) {
        if (!stager) return;
        const uint4* source = reinterpret_cast<const uint4*>(
            stage_codes + static_cast<std::size_t>(slab_index) * 64U);
        raw[0] = source[0];
        raw[1] = source[1];
    };
    const auto store_raw = [&](const uint4(&raw)[2], std::uint32_t buffer) {
        if (!stager) return;
        uint4* target = reinterpret_cast<uint4*>(&slab[buffer][stage_offset]);
        target[0]     = raw[0];
        target[1]     = raw[1];
    };
    const auto stage = [&](const Load& value, std::uint32_t buffer) {
        if (!stager) return;
        const uint2 book = table[value.g];
        const uint2 w0 = decode8(value.w.x, book), w1 = decode8(value.w.y, book);
        const uint2 w2 = decode8(value.w.z, book), w3 = decode8(value.w.w, book);
        uint4* target = reinterpret_cast<uint4*>(&slab[buffer][stage_offset]);
        target[0]     = uint4{w0.x, w0.y, w1.x, w1.y};
        target[1]     = uint4{w2.x, w2.y, w3.x, w3.y};
    };

    F32x8 accumulator[kFragments][4]{};
    const std::uint32_t slabs = columns / 64U;
    Load current              = load(0U);
    __syncthreads(); // table
    if constexpr (kRawFp8) {
        uint4 raw[2];
        load_raw(0U, raw);
        store_raw(raw, 0U);
    } else {
        stage(current, 0U);
    }
    __syncthreads();
    for (std::uint32_t s = 0; s < slabs; ++s) {
        const bool more = s + 1U < slabs;
        Load next;
        if (more) next = load(s + 1U);
        const std::uint8_t* tile = slab[s & 1U];
#    pragma unroll
        for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
            if constexpr (kRawFp8) {
                // Loaded where it is stored: a register lifetime across chunks spills at 240.
                if (chunk == kStageChunk && more) {
                    uint4 raw[2];
                    load_raw(s + 1U, raw);
                    store_raw(raw, (s + 1U) & 1U);
                }
            } else {
                if (chunk == kStageChunk && more) stage(next, (s + 1U) & 1U);
            }
            I32x2 weight[kFragments];
#    pragma unroll
            for (std::uint32_t j = 0; j < kFragments; ++j) {
                const uint2 w = *reinterpret_cast<const uint2*>(
                    tile + (wave_rows + j * 16U + axis) * kSlabStride + half * 32U + chunk * 8U);
                weight[j] = I32x2{static_cast<int>(w.x), static_cast<int>(w.y)};
            }
#    pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i) {
                if (i >= live) break;
                const std::uint32_t lo =
                    chunk < 2U ? (chunk == 0U ? current.x[i][0].x : current.x[i][0].z)
                               : (chunk == 2U ? current.x[i][1].x : current.x[i][1].z);
                const std::uint32_t hi =
                    chunk < 2U ? (chunk == 0U ? current.x[i][0].y : current.x[i][0].w)
                               : (chunk == 2U ? current.x[i][1].y : current.x[i][1].w);
                const I32x2 x{static_cast<int>(lo), static_cast<int>(hi)};
#    pragma unroll
                for (std::uint32_t j = 0; j < kFragments; ++j)
                    accumulator[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        weight[j], x, accumulator[j][i]);
            }
        }
        if (more) current = next;
        __syncthreads();
    }

    const float* row_scales = a.weight.scales + row_base + wave_rows + half * 8U;
#    pragma unroll
    for (std::uint32_t i = 0; i < 4U; ++i) {
        const std::uint32_t token = token_base + wave_tokens + i * 16U + axis;
        if (token >= tokens) continue;
        const float token_scale = fragment_scale[i];
        const bool poison       = fragment_poison[i];
#    pragma unroll
        for (std::uint32_t j = 0; j < kFragments; ++j) {
            const std::uint32_t row = row_base + wave_rows + j * 16U + half * 8U;
            if constexpr (kSiluPair) {
                // Lane `half` 0 holds this tile's eight gate rows, lane ^ 16 their up partners.
                float y[8], partner[8];
#    pragma unroll
                for (std::uint32_t e = 0; e < 8U; ++e) {
                    y[e]       = accumulator[j][i][e] * row_scales[j * 16U + e] * token_scale;
                    partner[e] = __shfl_xor(y[e], 16, 32);
                }
                if (half == 0U) {
                    std::uint16_t value[8];
#    pragma unroll
                    for (std::uint32_t e = 0; e < 8U; ++e)
                        value[e] =
                            poison ? kCanonicalBf16QuietNan : silu_pair_value(y[e], partner[e]);
                    *reinterpret_cast<uint4*>(
                        reinterpret_cast<std::uint16_t*>(a.output.leading) +
                        static_cast<std::size_t>(token) * (a.weight.rows / 2U) + row / 2U) =
                        uint4{value[0] | (static_cast<std::uint32_t>(value[1]) << 16U),
                              value[2] | (static_cast<std::uint32_t>(value[3]) << 16U),
                              value[4] | (static_cast<std::uint32_t>(value[5]) << 16U),
                              value[6] | (static_cast<std::uint32_t>(value[7]) << 16U)};
                }
                continue;
            }
            uint4* out = reinterpret_cast<uint4*>(output_at(a, token, row));
            std::uint16_t value[8];
            uint4 prior{};
            if constexpr (kAccumulate) prior = *out;
            const std::uint32_t prior_words[4]{prior.x, prior.y, prior.z, prior.w};
#    pragma unroll
            for (std::uint32_t e = 0; e < 8U; ++e) {
                const float y = accumulator[j][i][e] * row_scales[j * 16U + e] * token_scale;
                if (poison) {
                    value[e] = kCanonicalBf16QuietNan;
                } else if constexpr (kAccumulate) {
                    const std::uint16_t base =
                        static_cast<std::uint16_t>(prior_words[e / 2U] >> (16U * (e & 1U)));
                    value[e] = round_bf16(bf16_float(base) + bf16_float(round_bf16(y)));
                } else {
                    value[e] = round_bf16(y);
                }
            }
            *out = uint4{value[0] | (static_cast<std::uint32_t>(value[1]) << 16U),
                         value[2] | (static_cast<std::uint32_t>(value[3]) << 16U),
                         value[4] | (static_cast<std::uint32_t>(value[5]) << 16U),
                         value[6] | (static_cast<std::uint32_t>(value[7]) << 16U)};
        }
    }
#else
    (void)a;
#endif
}

template <std::uint32_t kSplit, std::uint32_t kSteps, std::uint32_t kTiles>
void launch_small_depth(const KernelArgs& args, hipStream_t stream) {
    const dim3 grid((args.weight.rows + args.second.rows) / 16U), block(32U * kSplit);
    if (args.output.silu_pair)
        hipLaunchKernelGGL((fp8lut4_small_t_kernel<false, true, kSplit, kSteps, kTiles>), grid,
                           block, 0, stream, args);
    else if (args.output.accumulate)
        hipLaunchKernelGGL((fp8lut4_small_t_kernel<true, false, kSplit, kSteps, kTiles>), grid,
                           block, 0, stream, args);
    else
        hipLaunchKernelGGL((fp8lut4_small_t_kernel<false, false, kSplit, kSteps, kTiles>), grid,
                           block, 0, stream, args);
}

hipError_t launch_small(const KernelArgs& args, hipStream_t stream) {
    if (args.tokens == 1U) {
        const dim3 grid((args.weight.rows + args.second.rows) / 32U), block(32U * kGemvSplit);
        if (args.output.silu_pair)
            hipLaunchKernelGGL((fp8lut4_t1_kernel<false, true>), grid, block, 0, stream, args);
        else if (args.output.accumulate)
            hipLaunchKernelGGL((fp8lut4_t1_kernel<true, false>), grid, block, 0, stream, args);
        else
            hipLaunchKernelGGL((fp8lut4_t1_kernel<false, false>), grid, block, 0, stream, args);
    } else if (args.tokens <= fp8lut4::kSmallTokens) {
        launch_small_depth<fp8lut4::kVerifySplit, fp8lut4::kVerifySteps, 1U>(args, stream);
    } else {
        launch_small_depth<fp8lut4::kVerifySplit, fp8lut4::kVerifySteps, 2U>(args, stream);
    }
    return hipGetLastError();
}

template <std::uint32_t kRowBlocks, std::uint32_t kSplit, std::uint32_t kTiles>
void launch_mid_tiles(const KernelArgs& args, hipStream_t stream) {
    const dim3 grid(args.weight.rows / (16U * kRowBlocks)), block(256U);
    if (args.weight.groups == nullptr) // row-scaled E4M3 rows: plain output only
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<false, false, kRowBlocks, kSplit, kTiles, true>),
                           grid, block, 0, stream, args);
    else if (args.output.silu_pair)
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<false, true, kRowBlocks, kSplit, kTiles>), grid,
                           block, 0, stream, args);
    else if (args.output.accumulate)
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<true, false, kRowBlocks, kSplit, kTiles>), grid,
                           block, 0, stream, args);
    else
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<false, false, kRowBlocks, kSplit, kTiles>), grid,
                           block, 0, stream, args);
}

template <std::uint32_t kRowBlocks, std::uint32_t kSplit>
void launch_mid_shape(const KernelArgs& args, hipStream_t stream) {
    switch ((args.tokens + 15U) / 16U) {
    case 3U:
        launch_mid_tiles<kRowBlocks, kSplit, 3U>(args, stream);
        break;
    case 4U:
        launch_mid_tiles<kRowBlocks, kSplit, 4U>(args, stream);
        break;
    case 5U:
        launch_mid_tiles<kRowBlocks, kSplit, 5U>(args, stream);
        break;
    case 6U:
        launch_mid_tiles<kRowBlocks, kSplit, 6U>(args, stream);
        break;
    case 7U:
        launch_mid_tiles<kRowBlocks, kSplit, 7U>(args, stream);
        break;
    default:
        launch_mid_tiles<kRowBlocks, kSplit, 8U>(args, stream);
        break;
    }
}

// T 33..128 (between the two-tile small route and the 256-token prefill tile). Wide outputs at
// T >= 96 keep the whole K per wave (8 row blocks); otherwise two K halves double the CTAs, which
// the 5120-row outputs need to fill the device (qualification timings, 2026-09-29).
constexpr std::uint32_t kMidMinimumTokens = 33U, kMidMaximumTokens = 128U;

bool mid_whole_k(std::uint32_t tokens, std::uint32_t rows) noexcept {
    return tokens >= 96U && rows >= 12288U && rows % 128U == 0U;
}

bool mid_supported(std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns) noexcept {
    return tokens >= kMidMinimumTokens && tokens <= kMidMaximumTokens &&
           (mid_whole_k(tokens, rows) || (rows % 64U == 0U && (columns / 64U) % 2U == 0U));
}

void launch_mid(const KernelArgs& args, hipStream_t stream) {
    if (mid_whole_k(args.tokens, args.weight.rows)) {
        launch_mid_shape<8U, 1U>(args, stream);
    } else {
        launch_mid_shape<4U, 2U>(args, stream);
    }
}

template <bool kAccumulate, bool kSiluPair, std::uint32_t kRowBlocks, std::uint32_t kSplit,
          std::uint32_t kDepth, std::uint32_t kTiles>
void launch_deep_as(const KernelArgs& args, hipStream_t stream) {
    const dim3 grid((args.weight.rows + args.second.rows) / (16U * kRowBlocks)), block(256U);
    hipLaunchKernelGGL(
        (fp8lut4_mid_deep_kernel<kAccumulate, kSiluPair, kRowBlocks, kSplit, kTiles, kDepth>), grid,
        block, 0, stream, args);
}

// ceil(T / 16) token tiles, instantiated for kMinTiles..kMaxTiles (the widths its shapes select).
template <bool kAccumulate, bool kSiluPair, std::uint32_t kRowBlocks, std::uint32_t kSplit,
          std::uint32_t kDepth, std::uint32_t kMinTiles, std::uint32_t kMaxTiles>
void launch_deep_tiles(const KernelArgs& args, hipStream_t stream) {
    const std::uint32_t tiles = (args.tokens + 15U) / 16U;
    [&]<std::uint32_t... kOffset>(std::integer_sequence<std::uint32_t, kOffset...>) {
        ((tiles == kMinTiles + kOffset ? launch_deep_as<kAccumulate, kSiluPair, kRowBlocks, kSplit,
                                                        kDepth, kMinTiles + kOffset>(args, stream)
                                       : void()),
         ...);
    }(std::make_integer_sequence<std::uint32_t, kMaxTiles - kMinTiles + 1U>{});
}

// Plain or residual output (one projection or a fused pair).
template <std::uint32_t kRowBlocks, std::uint32_t kSplit, std::uint32_t kDepth,
          std::uint32_t kMinTiles, std::uint32_t kMaxTiles>
void launch_deep_plain(const KernelArgs& args, hipStream_t stream) {
    if (args.output.accumulate)
        launch_deep_tiles<true, false, kRowBlocks, kSplit, kDepth, kMinTiles, kMaxTiles>(args,
                                                                                         stream);
    else
        launch_deep_tiles<false, false, kRowBlocks, kSplit, kDepth, kMinTiles, kMaxTiles>(args,
                                                                                          stream);
}

// Deep mid-T selection at T 33..128 (codebook weights, 16-byte-aligned outputs), from interleaved
// eager medians against the mid-T kernel on the R9700 (fp8lut4_linear_qual candidate sweeps of
// 4 x 2 / 8 x 1 row blocks / K splits at depth 2..4, 2 x 2 and 4 x 1 four-wave CTAs, and the
// prefill CTA; 2026-09-30 T 33..64, 2026-10-01 T 40..128). The prefill CTA takes 1.2-2.4x the
// mid-T time below T 129.
// - pairs of >= 14336 combined rows run fused in one launch: 4 x 2, depth 2 (attention N7168 +
//   N7168 K5120: 0.66-0.73x, T64 96 -> 70 us, T128 171 -> 115 us), or 8 x 1 from 16384 rows (GDN
//   N4096 + N12288 K5120), depth 2 at T 33..64 (0.66-0.73x, T64 105 -> 73 us) and depth 4 at
//   T 65..128 (0.70-0.75x, T128 161 -> 120 us);
// - N5120 outputs (MLP down K17408, attention/GDN output K6144), plain or residual: 4 x 2, depth 2
//   (0.78-0.90x; K17408 T64 106 -> 89 us, T128 166 -> 149 us; K6144 T128 71 -> 58 us);
// - SiLU gate/up N34816: 4 x 2, depth 4 at T 49..64 (0.95-0.97x; mid-T is as fast below) and
//   depth 2 at T 65..128 (0.86-0.92x, T128 294 -> 269 us);
// - output head (>= 131072 rows) at T 49..64: 8 x 1, depth 2 (0.97x, T64 1244 -> 1206 us).
// Everything else keeps the mid-T kernel.
enum class DeepShape { None, Rows64Depth2, Rows64Depth4, Rows128Depth2, Rows128Depth4 };

constexpr std::uint32_t kDeepMaximumTokens = 128U, kDeepFourTileTokens = 49U;
constexpr std::uint32_t kDeepShallowMaximumTokens = 64U; // depth and head boundary
constexpr std::uint32_t kDeepPairRows = 14336U, kDeepWidePairRows = 16384U;
constexpr std::uint32_t kDeepNarrowRows = 5120U, kDeepHeadRows = 131072U;

bool deep_fits(DeepShape shape, std::uint32_t rows, std::uint32_t columns) noexcept {
    const bool wide = shape == DeepShape::Rows128Depth2 || shape == DeepShape::Rows128Depth4;
    const std::uint32_t row_blocks = wide ? 8U : 4U, split = 8U / row_blocks;
    const std::uint32_t depth =
        shape == DeepShape::Rows64Depth4 || shape == DeepShape::Rows128Depth4 ? 4U : 2U;
    return rows % (16U * row_blocks) == 0U && (columns / 64U) % split == 0U &&
           columns / 64U / split >= depth;
}

bool aligned16(const Fp8Lut4Output& output) noexcept {
    return reinterpret_cast<std::uintptr_t>(output.leading) % 16U == 0U &&
           reinterpret_cast<std::uintptr_t>(output.trailing) % 16U == 0U;
}

bool deep_width(const KernelArgs& args) noexcept {
    return args.tokens >= kMidMinimumTokens && args.tokens <= kDeepMaximumTokens &&
           args.weight.groups != nullptr && aligned16(args.output);
}

DeepShape deep_shape(const KernelArgs& args) noexcept {
    if (!deep_width(args)) return DeepShape::None;
    const std::uint32_t rows = args.weight.rows;
    DeepShape shape          = DeepShape::None;
    const bool shallow       = args.tokens <= kDeepShallowMaximumTokens;
    if (args.output.silu_pair) {
        if (!shallow)
            shape = DeepShape::Rows64Depth2;
        else if (args.tokens >= kDeepFourTileTokens)
            shape = DeepShape::Rows64Depth4;
    } else if (rows == kDeepNarrowRows) {
        shape = DeepShape::Rows64Depth2;
    } else if (rows >= kDeepHeadRows && args.tokens >= kDeepFourTileTokens && shallow) {
        shape = DeepShape::Rows128Depth2;
    }
    return shape != DeepShape::None && deep_fits(shape, rows, args.weight.columns)
               ? shape
               : DeepShape::None;
}

// Both projections of `args` (first and second) in one launch.
DeepShape deep_pair_shape(const KernelArgs& args) noexcept {
    KernelArgs second        = args;
    second.weight            = args.second;
    second.output            = args.second_output;
    const std::uint32_t rows = args.weight.rows + args.second.rows;
    if (!deep_width(args) || !deep_width(second) || rows < kDeepPairRows) return DeepShape::None;
    const DeepShape shape = rows < kDeepWidePairRows                   ? DeepShape::Rows64Depth2
                            : args.tokens <= kDeepShallowMaximumTokens ? DeepShape::Rows128Depth2
                                                                       : DeepShape::Rows128Depth4;
    return deep_fits(shape, args.weight.rows, args.weight.columns) &&
                   deep_fits(shape, args.second.rows, args.second.columns)
               ? shape
               : DeepShape::None;
}

void launch_deep(const KernelArgs& args, DeepShape shape, hipStream_t stream) {
    switch (shape) {
    case DeepShape::Rows64Depth2: // pairs and N5120 at T 33..128, SiLU at T 65..128
        if (args.output.silu_pair)
            launch_deep_tiles<false, true, 4U, 2U, 2U, 5U, 8U>(args, stream);
        else
            launch_deep_plain<4U, 2U, 2U, 3U, 8U>(args, stream);
        break;
    case DeepShape::Rows64Depth4: // SiLU pair at T 49..64
        launch_deep_as<false, true, 4U, 2U, 4U, 4U>(args, stream);
        break;
    case DeepShape::Rows128Depth2:
        launch_deep_plain<8U, 1U, 2U, 3U, 4U>(args, stream);
        break;
    case DeepShape::Rows128Depth4:
        launch_deep_plain<8U, 1U, 4U, 5U, 8U>(args, stream);
        break;
    case DeepShape::None:
        break;
    }
}

enum class PrefillShape { T256R128, T256R64, T512R64 };

// Three 256x128 CTAs are resident per WGP (32 WGPs, 96 CTAs). Selected by whole-model prefill
// cost on the R9700 (interleaved runs, T 256..2048):
// - T <= 256: 64-row CTAs when a launch has fewer than 96 128-row CTAs (N5120/7168/4096 16-20%
//   faster per launch);
// - wide N (>= 96 128-row CTAs per tile): 512-token tiles wherever they add no padding over
//   256-token tiles (T2048 588 -> 583 ms);
// - narrow N: one 512-token tile for T 257..511 (no partial second 256-token tile; T288 129 ->
//   113 ms), otherwise 256x128 (512-token tiles double the re-reads of their large K17408
//   activation images and lose 15-35% per launch at T2048 in the model).
constexpr std::uint32_t kPrefillResidentCtas = 96U;

PrefillShape prefill_shape(std::uint32_t tokens, std::uint32_t rows) noexcept {
    const std::uint32_t tiles256 = (tokens + kPrefillTokens - 1U) / kPrefillTokens;
    const std::uint32_t row_ctas = rows / kPrefillRows;
    if (tokens <= kPrefillTokens) {
        return row_ctas < kPrefillResidentCtas ? PrefillShape::T256R64 : PrefillShape::T256R128;
    }
    if (row_ctas >= kPrefillResidentCtas) {
        return (tokens + 511U) / 512U * 2U == tiles256 ? PrefillShape::T512R64
                                                       : PrefillShape::T256R128;
    }
    return tokens < 512U ? PrefillShape::T512R64 : PrefillShape::T256R128;
}

template <bool kAccumulate, bool kSiluPair, bool kRawFp8, std::uint32_t kTokenGroups,
          std::uint32_t kWaveRows>
void launch_prefill_as(const KernelArgs& args, hipStream_t stream) {
    constexpr std::uint32_t kTileTokens = kTokenGroups * 64U;
    constexpr std::uint32_t kRows       = 8U / kTokenGroups * kWaveRows;
    hipLaunchKernelGGL(
        (fp8lut4_prefill_kernel<kAccumulate, kSiluPair, kRawFp8, kTokenGroups, kWaveRows>),
        dim3((args.tokens + kTileTokens - 1U) / kTileTokens, args.weight.rows / kRows), dim3(256),
        0, stream, args);
}

// Tokens [start, start + tokens) of `args` as a launch of its own: the activation rows, token
// scales and status words, and every output row of those tokens.
KernelArgs token_span(const KernelArgs& args, std::uint32_t start, std::uint32_t tokens) noexcept {
    KernelArgs part          = args;
    const std::uint32_t rows = args.weight.rows, split = args.output.split;
    const std::size_t lead = args.output.silu_pair ? rows / 2U : split != 0U ? split : rows;
    part.activation += static_cast<std::size_t>(start) * args.weight.columns;
    part.token_scales += start;
    part.status += start;
    part.output.leading += static_cast<std::size_t>(start) * lead;
    if (split != 0U) part.output.trailing += static_cast<std::size_t>(start) * (rows - split);
    part.tokens = tokens;
    return part;
}

// One launch per token span. A launch's CTAs walk every token tile of a row block together, so the
// resident CTAs keep the launch's whole activation image hot; once it outgrows the cache, every
// row block re-streams it from DRAM. Per-token cost on the R9700 (fp8lut4_linear_qual
// --time-cell, one launch): wide N34816 K5120 1.75 us at T2048, 2.75 at T2560, 4.36 at T4096,
// 5.55 at T8192; N12288 0.62 -> 1.90; narrow N5120 K17408 flat to T4096 (0.84-0.87), 1.49 at
// T8192. Wide outputs (>= 96 128-row CTAs per tile) launch 2048-token spans, narrow ones 4096
// (2048 cost them 2-6% at T2560..3072): every width now holds its T2048 per-token cost.
constexpr std::uint32_t kPrefillWideSpanTokens = 2048U, kPrefillNarrowSpanTokens = 4096U;

template <bool kAccumulate, bool kSiluPair, bool kRawFp8>
void launch_prefill_shape(const KernelArgs& args, hipStream_t stream) {
    const std::uint32_t span = args.weight.rows / kPrefillRows >= kPrefillResidentCtas
                                   ? kPrefillWideSpanTokens
                                   : kPrefillNarrowSpanTokens;
    for (std::uint32_t start = 0; start < args.tokens; start += span) {
        const KernelArgs part = token_span(args, start, std::min(span, args.tokens - start));
        switch (prefill_shape(part.tokens, part.weight.rows)) {
        case PrefillShape::T256R128:
            launch_prefill_as<kAccumulate, kSiluPair, kRawFp8, 4U, 64U>(part, stream);
            break;
        case PrefillShape::T256R64:
            launch_prefill_as<kAccumulate, kSiluPair, kRawFp8, 4U, 32U>(part, stream);
            break;
        case PrefillShape::T512R64:
            launch_prefill_as<kAccumulate, kSiluPair, kRawFp8, 8U, 64U>(part, stream);
            break;
        }
    }
}

hipError_t launch_prefill(const KernelArgs& args, hipStream_t stream) {
    if (args.weight.groups == nullptr) { // row-scaled E4M3 rows (raw staging), no SiLU pair
        if (args.output.accumulate)
            launch_prefill_shape<true, false, true>(args, stream);
        else
            launch_prefill_shape<false, false, true>(args, stream);
    } else if (args.output.silu_pair) {
        launch_prefill_shape<false, true, false>(args, stream);
    } else if (args.output.accumulate) {
        launch_prefill_shape<true, false, false>(args, stream);
    } else {
        launch_prefill_shape<false, false, false>(args, stream);
    }
    return hipGetLastError();
}

// Row-scaled E4M3 rows (null groups) run only the prefill CTA, which has the split and
// accumulate epilogues; narrower widths use the row-scaled small-T and mid-T routes.
bool valid_raw(const Fp8Lut4Weight& weight, const Fp8Lut4Output& output,
               const Fp8ActivationWorkspace& activation) noexcept {
    return weight.groups != nullptr || (!output.silu_pair && activation.tokens > kMidMaximumTokens);
}

bool valid_target(const Fp8Lut4Weight& weight, const Fp8Lut4Output& output,
                  const Fp8ActivationWorkspace& activation) noexcept {
    return weight.codes != nullptr && valid_raw(weight, output, activation) &&
           weight.scales != nullptr && output.leading != nullptr &&
           (output.split == 0U || (output.trailing != nullptr && output.split % 128U == 0U &&
                                   output.split < weight.rows)) &&
           (!output.silu_pair || (output.split == 0U && !output.accumulate)) &&
           activation.padded_columns == weight.columns &&
           fp8lut4_linear_supported(activation.tokens, weight.rows, weight.columns);
}

KernelArgs kernel_args(const Fp8Lut4Weight& weight, const Fp8Lut4Output& output,
                       const Fp8ActivationWorkspace& activation) noexcept {
    KernelArgs args;
    args.weight       = weight;
    args.output       = output;
    args.activation   = activation.codes;
    args.token_scales = activation.scales;
    args.status       = activation.status;
    args.tokens       = activation.tokens;
    return args;
}

} // namespace

bool fp8_row_scaled_prefill_supported(std::uint32_t tokens, std::uint32_t rows,
                                      std::uint32_t columns,
                                      std::uint32_t padded_columns) noexcept {
    return tokens > kFp8SmallTokenLimit && rows != 0U && rows % kPrefillRows == 0U &&
           columns != 0U && columns % 128U == 0U && padded_columns == columns;
}

hipError_t fp8_row_scaled_prefill_linear(const Fp8RowScaledWeight& weight,
                                         const Fp8ActivationWorkspace& activation,
                                         hip_bfloat16* output, hipStream_t stream) noexcept {
    if (weight.codes == nullptr || weight.scales == nullptr || output == nullptr ||
        activation.codes == nullptr || activation.scales == nullptr ||
        activation.status == nullptr ||
        !fp8_row_scaled_prefill_supported(activation.tokens, weight.rows, activation.columns,
                                          activation.padded_columns)) {
        return hipErrorInvalidValue;
    }
    KernelArgs args;
    args.weight =
        Fp8Lut4Weight{weight.codes, nullptr, weight.scales, weight.rows, activation.padded_columns};
    args.output       = Fp8Lut4Output{.leading = output};
    args.activation   = activation.codes;
    args.token_scales = activation.scales;
    args.status       = activation.status;
    args.tokens       = activation.tokens;
    if (mid_supported(args.tokens, args.weight.rows, args.weight.columns)) {
        launch_mid(args, stream);
        return hipGetLastError();
    }
    launch_prefill_shape<false, false, true>(args, stream);
    return hipGetLastError();
}

const std::array<std::array<std::uint8_t, 8>, 256>& fp8lut4_magnitude_table() noexcept {
    return fp8lut4::kTable;
}

namespace {
bool small_route(std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns) noexcept {
    if (tokens == 1U) return rows % 32U == 0U && (columns / 64U) % 16U == 0U;
    return tokens <= fp8lut4::kSmallTwoTileTokens && rows % 16U == 0U && (columns / 64U) % 8U == 0U;
}
} // namespace

bool fp8_row_scaled_projection_supported(std::uint32_t tokens, std::uint32_t rows,
                                         std::uint32_t columns) noexcept {
    return tokens > kMidMaximumTokens && rows != 0U && rows % kPrefillRows == 0U && columns != 0U &&
           columns % 128U == 0U;
}

bool fp8lut4_linear_supported(std::uint32_t tokens, std::uint32_t rows,
                              std::uint32_t columns) noexcept {
    if (tokens == 0U || rows == 0U || columns == 0U || columns % 128U != 0U) return false;
    if (tokens <= fp8lut4::kSmallTokens) return small_route(tokens, rows, columns);
    return small_route(tokens, rows, columns) || rows % kPrefillRows == 0U;
}

hipError_t fp8lut4_linear(const Fp8Lut4Weight& weight, const Fp8ActivationWorkspace& activation,
                          const Fp8Lut4Output& output, hipStream_t stream) noexcept {
    if (activation.codes == nullptr || activation.scales == nullptr ||
        activation.status == nullptr || !valid_target(weight, output, activation)) {
        return hipErrorInvalidValue;
    }
    const KernelArgs args = kernel_args(weight, output, activation);
    if (const DeepShape shape = deep_shape(args); shape != DeepShape::None) {
        launch_deep(args, shape, stream);
        return hipGetLastError();
    }
    if (mid_supported(activation.tokens, weight.rows, weight.columns)) {
        launch_mid(args, stream);
        return hipGetLastError();
    }
    return small_route(activation.tokens, weight.rows, weight.columns)
               ? launch_small(args, stream)
               : launch_prefill(args, stream);
}

hipError_t fp8lut4_linear_pair(const Fp8Lut4Weight& first, const Fp8Lut4Output& first_output,
                               const Fp8Lut4Weight& second, const Fp8Lut4Output& second_output,
                               const Fp8ActivationWorkspace& activation,
                               hipStream_t stream) noexcept {
    if (activation.codes == nullptr || activation.scales == nullptr ||
        activation.status == nullptr || !valid_target(first, first_output, activation) ||
        !valid_target(second, second_output, activation) ||
        first_output.accumulate != second_output.accumulate || first_output.silu_pair ||
        second_output.silu_pair) {
        return hipErrorInvalidValue;
    }
    {
        KernelArgs args    = kernel_args(first, first_output, activation);
        args.second        = second;
        args.second_output = second_output;
        if (const DeepShape shape = deep_pair_shape(args); shape != DeepShape::None) {
            launch_deep(args, shape, stream);
            return hipGetLastError();
        }
    }
    if (mid_supported(activation.tokens, first.rows, first.columns) ||
        !small_route(activation.tokens, first.rows, first.columns) ||
        !small_route(activation.tokens, second.rows, second.columns)) {
        const hipError_t result = fp8lut4_linear(first, activation, first_output, stream);
        return result != hipSuccess ? result
                                    : fp8lut4_linear(second, activation, second_output, stream);
    }
    KernelArgs args    = kernel_args(first, first_output, activation);
    args.second        = second;
    args.second_output = second_output;
    return launch_small(args, stream);
}

} // namespace ninfer::ops::r9700::linear
