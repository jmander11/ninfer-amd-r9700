#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/fp8lut4_pipeline_impl.h"
#include "ops/r9700/linear/fp8_small_t_linear.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear {
namespace {

using fp8lut4::decode8;
using fp8lut4::F32x8;
using fp8lut4::I32x2;

constexpr std::uint16_t kCanonicalBf16QuietNan = 0x7FC0U;

// Branch-free round-to-nearest-even to BF16 (identical to hip_bfloat16 for every non-NaN value;
// NaN becomes the canonical quiet NaN).
__device__ __forceinline__ std::uint16_t round_bf16(float value) {
    const std::uint32_t bits = __float_as_uint(value);
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
    const float* token_scales = nullptr;
    const std::uint32_t* status = nullptr;
    std::uint32_t tokens = 0;
};

// Element (token, row) of one published output.
__device__ __forceinline__ std::uint16_t* output_at(const Fp8Lut4Weight& weight,
                                                    const Fp8Lut4Output& output, std::uint32_t token,
                                                    std::uint32_t row) {
    const std::uint32_t rows = weight.rows, split = output.split;
    const bool leading = split == 0U || row < split;
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

// ---- Small T (T <= 16 per token tile; kTiles = 2 serves T 17..32): one CTA per 16 rows of the
// first or the second projection.
template <bool kAccumulate, bool kSiluPair, std::uint32_t kSplit, std::uint32_t kSteps,
          std::uint32_t kTiles>
__global__ __launch_bounds__(32U * kSplit) void fp8lut4_small_t_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ uint2 table[256];
    __shared__ float partial[kSplit][kTiles][8][32];
    __shared__ float staged[kSiluPair ? 16 : 1][16U * kTiles];
    const std::uint32_t first_blocks = a.weight.rows / 16U;
    const bool second = blockIdx.x >= first_blocks;
    // By value: the CTA-uniform selection stays in scalar registers.
    const Fp8Lut4Weight weight = second ? a.second : a.weight;
    const Fp8Lut4Output output = second ? a.second_output : a.output;
    const std::uint32_t row_base = (second ? blockIdx.x - first_blocks : blockIdx.x) * 16U;
    fp8lut4::load_table(table, 32U * kSplit);
    const auto publish = [&](std::uint32_t token, std::uint32_t row, float value) {
            if constexpr (kSiluPair) {
                staged[row - row_base][token] = value;
                return;
            }
            std::uint16_t* out = output_at(weight, output, token, row);
            if (a.status[token] != Fp8ActivationOk) {
                *out = kCanonicalBf16QuietNan;
            } else if constexpr (kAccumulate) {
                *out = round_bf16(bf16_float(*out) + bf16_float(round_bf16(value)));
            } else {
                *out = round_bf16(value);
            }
        };
    fp8lut4::small_t_rows<kSplit, kSteps, kTiles>(
        weight, row_base, a.activation, a.token_scales, a.tokens, table, partial, publish);
    if constexpr (kSiluPair) {
        __syncthreads();
        const std::uint32_t features = weight.rows / 2U;
        for (std::uint32_t index = threadIdx.x; index < 8U * a.tokens; index += blockDim.x) {
            const std::uint32_t feature = index % 8U, token = index / 8U;
            auto* out = reinterpret_cast<std::uint16_t*>(output.leading) +
                        static_cast<std::size_t>(token) * features + row_base / 2U + feature;
            *out = a.status[token] != Fp8ActivationOk
                ? kCanonicalBf16QuietNan
                : silu_pair_value(staged[feature][token], staged[feature + 8U][token]);
        }
    }
#else
    (void)a;
#endif
}

// ---- Mid T (T 33..128, kTiles token tiles of 16): eight waves; wave w owns row block w % kRowBlocks
// (16 rows) over K split kSplit-ways (w / kRowBlocks). Each 64-column step, the waves of one K
// split stage that step's T x 64 activation bytes into LDS once for all their row blocks, instead
// of every 16-row CTA re-reading the whole activation (the small-T route's L2 cost past T 64).
// Weight fragments stream per wave exactly as small_t_rows (slot 16 half + axis of the N16 x K64
// tile), paired with the identical activation bytes, so the K sum is the same permutation. The
// K splits combine in order through LDS and wave split 0 publishes; the SiLU pair's up rows sit
// in lane ^ 16 of the same fragment.
constexpr std::uint32_t kMidStride = 80U;  // 64 bytes + 16: the 16 token rows hit distinct banks

template <bool kAccumulate, bool kSiluPair, std::uint32_t kRowBlocks, std::uint32_t kSplit,
          std::uint32_t kTiles, bool kRawFp8 = false>
__global__ __launch_bounds__(256) void fp8lut4_mid_t_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    static_assert(kRowBlocks * kSplit == 8U && kTiles >= 1U && kTiles <= 8U);
    constexpr std::uint32_t kTokens = 16U * kTiles;
    constexpr std::uint32_t kGroupThreads = 32U * kRowBlocks;
    constexpr std::uint32_t kUnits = (kTokens * 4U + kGroupThreads - 1U) / kGroupThreads;
    constexpr std::uint32_t kActivationBytes = 2U * kSplit * kTokens * kMidStride;
    constexpr std::uint32_t kPartialBytes =
        kSplit > 1U ? kRowBlocks * kTiles * 8U * 32U * 4U : 0U;
    __shared__ uint2 table[256];
    __shared__ __attribute__((aligned(16)))
    std::uint8_t lds[kActivationBytes > kPartialBytes ? kActivationBytes : kPartialBytes];
    const std::uint32_t thread = threadIdx.x, lane = thread & 31U;
    const std::uint32_t wave = __builtin_amdgcn_readfirstlane(thread >> 5U);
    const std::uint32_t block = wave % kRowBlocks, split = wave / kRowBlocks;
    const std::uint32_t axis = lane & 15U, half = lane >> 4U;
    const std::uint32_t row_base = (blockIdx.x * kRowBlocks + block) * 16U;
    const std::uint32_t columns = a.weight.columns, slabs = columns / 64U;
    const std::uint32_t steps = slabs / kSplit, first_slab = split * steps;
    const std::uint32_t tokens = a.tokens;
    if constexpr (!kRawFp8) fp8lut4::load_table(table, 256U);

    const std::size_t first_tile = static_cast<std::size_t>(row_base / 16U) * slabs + first_slab;
    // kRawFp8 (row-scaled E4M3 rows, fp8_row_scaled_prefill_linear): the same (row, 32 columns)
    // fragment is read where the row-major weight stores it.
    const std::uint8_t* raw_codes = a.weight.codes +
        static_cast<std::size_t>(row_base + axis) * columns + first_slab * 64U + half * 32U;
    const std::uint32_t slot = half * 16U + axis;
    const std::uint8_t* codes = a.weight.codes + first_tile * 512U + slot * 16U;
    const std::uint8_t* groups = a.weight.groups + first_tile * 32U + slot;
    // Staging unit u of this split's group: token u / 4, 16-byte part u % 4 of the step.
    const std::uint32_t group_thread = thread % kGroupThreads;
    const auto stage_load = [&](std::uint32_t step, uint4 (&units)[kUnits]) {
#pragma unroll
        for (std::uint32_t i = 0; i < kUnits; ++i) {
            const std::uint32_t unit = group_thread + i * kGroupThreads;
            const std::uint32_t token = unit / 4U, part = unit % 4U;
            units[i] = token < tokens && unit < kTokens * 4U
                ? *reinterpret_cast<const uint4*>(a.activation + static_cast<std::size_t>(token) * columns +
                                                  static_cast<std::size_t>(first_slab + step) * 64U + part * 16U)
                : uint4{};
        }
    };
    const auto stage_store = [&](const uint4 (&units)[kUnits], std::uint32_t buffer) {
        std::uint8_t* base = lds + (buffer * kSplit + split) * kTokens * kMidStride;
#pragma unroll
        for (std::uint32_t i = 0; i < kUnits; ++i) {
            const std::uint32_t unit = group_thread + i * kGroupThreads;
            if (unit < kTokens * 4U)
                *reinterpret_cast<uint4*>(base + (unit / 4U) * kMidStride + (unit % 4U) * 16U) = units[i];
        }
    };
    struct Weight { uint4 w; uint4 raw; std::uint32_t g; };
    const auto weight_load = [&](std::uint32_t step) {
        Weight value{};
        if constexpr (kRawFp8) {
            const uint4* row = reinterpret_cast<const uint4*>(raw_codes + static_cast<std::size_t>(step) * 64U);
            value.w = row[0];
            value.raw = row[1];
        } else {
            value.w = *reinterpret_cast<const uint4*>(codes + static_cast<std::size_t>(step) * 512U);
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
    __syncthreads();  // codebook and the first activation step
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
            const std::uint32_t raw[8]{current.w.x, current.w.y, current.w.z, current.w.w,
                                       current.raw.x, current.raw.y, current.raw.z, current.raw.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 8U; ++i) w[i] = raw[i];
        } else {
            const uint2 book = table[current.g];
            const uint2 w0 = decode8(current.w.x, book), w1 = decode8(current.w.y, book);
            const uint2 w2 = decode8(current.w.z, book), w3 = decode8(current.w.w, book);
            const std::uint32_t decoded[8]{w0.x, w0.y, w1.x, w1.y, w2.x, w2.y, w3.x, w3.y};
#pragma unroll
            for (std::uint32_t i = 0; i < 8U; ++i) w[i] = decoded[i];
        }
        const std::uint8_t* activation = lds + ((step & 1U) * kSplit + split) * kTokens * kMidStride;
#pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
            const uint4* xt = reinterpret_cast<const uint4*>(
                activation + (tile * 16U + axis) * kMidStride + half * 32U);
            const uint4 x0 = xt[0], x1 = xt[1];
            const std::uint32_t x[8]{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
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
#pragma unroll
                for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#pragma unroll
                    for (std::uint32_t item = 0; item < 8U; ++item) {
                        float* p = partial + ((block * kTiles + tile) * 8U + item) * 32U + lane;
                        *p = s == 1U ? accumulator[tile][item] : *p + accumulator[tile][item];
                    }
            }
            __syncthreads();
        }
        if (split != 0U) return;
#pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item)
                accumulator[tile][item] += partial[((block * kTiles + tile) * 8U + item) * 32U + lane];
    }

#pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
        const std::uint32_t token = tile * 16U + axis;
        const bool present = token < tokens;
        const float token_scale = present ? a.token_scales[token] : 0.0F;
        const bool poison = present && a.status[token] != Fp8ActivationOk;
        float y[8];
#pragma unroll
        for (std::uint32_t item = 0; item < 8U; ++item)
            y[item] = accumulator[tile][item] * a.weight.scales[row_base + half * 8U + item] * token_scale;
        if constexpr (kSiluPair) {
            // Lane `half` 0 holds this block's eight gate rows, lane ^ 16 their up partners.
            float partner[8];
#pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item) partner[item] = __shfl_xor(y[item], 16, 32);
            if (half == 0U && present) {
                auto* out = reinterpret_cast<std::uint16_t*>(a.output.leading) +
                            static_cast<std::size_t>(token) * (a.weight.rows / 2U) + row_base / 2U;
#pragma unroll
                for (std::uint32_t item = 0; item < 8U; ++item)
                    out[item] = poison ? kCanonicalBf16QuietNan : silu_pair_value(y[item], partner[item]);
            }
            continue;
        }
        if (!present) continue;
#pragma unroll
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
    const bool second = blockIdx.x >= first_blocks;
    const Fp8Lut4Weight weight = second ? a.second : a.weight;
    const Fp8Lut4Output output = second ? a.second_output : a.output;
    const std::uint32_t row_base = (second ? blockIdx.x - first_blocks : blockIdx.x) * 32U;
    fp8lut4::load_table(table, 32U * kGemvSplit);
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const std::uint32_t row = row_base + lane;
    const std::uint32_t columns = weight.columns, steps_total = columns / 64U;
    const std::uint32_t steps = steps_total / kGemvSplit;
    const std::size_t first_tile = static_cast<std::size_t>(row / 16U) * steps_total + wave * steps;
    const std::uint8_t* codes = weight.codes + first_tile * 512U + (row % 16U) * 16U;
    const std::uint8_t* groups = weight.groups + first_tile * 32U + row % 16U;
    const std::uint8_t* activation = a.activation + static_cast<std::size_t>(wave) * steps * 64U;
    struct Batch { uint4 c[kGemvSteps][2]; std::uint32_t g[kGemvSteps][2]; uint4 x[kGemvSteps][4]; };
    const auto load = [&](std::uint32_t step) {
        Batch value;
#pragma unroll
        for (std::uint32_t i = 0; i < kGemvSteps; ++i) {
            const std::uint8_t* tile = codes + static_cast<std::size_t>(step + i) * 512U;
            value.c[i][0] = *reinterpret_cast<const uint4*>(tile);
            value.c[i][1] = *reinterpret_cast<const uint4*>(tile + 256U);
            const std::uint8_t* group = groups + static_cast<std::size_t>(step + i) * 32U;
            value.g[i][0] = group[0];
            value.g[i][1] = group[16];
            const uint4* x = reinterpret_cast<const uint4*>(activation + static_cast<std::size_t>(step + i) * 64U);
            value.x[i][0] = x[0];
            value.x[i][1] = x[1];
            value.x[i][2] = x[2];
            value.x[i][3] = x[3];
        }
        return value;
    };
    float sum = 0.0F;
    const auto consume = [&](const Batch& value) {
#pragma unroll
        for (std::uint32_t i = 0; i < kGemvSteps; ++i) {
#pragma unroll
            for (std::uint32_t half = 0; half < 2U; ++half) {
                const uint2 book = table[value.g[i][half]];
                const uint4 word = value.c[i][half];
                const uint2 w0 = decode8(word.x, book), w1 = decode8(word.y, book);
                const uint2 w2 = decode8(word.z, book), w3 = decode8(word.w, book);
                const uint4 x0 = value.x[i][2U * half], x1 = value.x[i][2U * half + 1U];
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w0.x), static_cast<int>(x0.x), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w0.y), static_cast<int>(x0.y), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w1.x), static_cast<int>(x0.z), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w1.y), static_cast<int>(x0.w), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w2.x), static_cast<int>(x1.x), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w2.y), static_cast<int>(x1.y), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w3.x), static_cast<int>(x1.z), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w3.y), static_cast<int>(x1.w), sum);
            }
        }
    };
    Batch current = load(0U);
    __syncthreads();  // codebook
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
#pragma unroll
    for (std::uint32_t split = 1; split < kGemvSplit; ++split) total += partial[split][lane];
    const float y = total * weight.scales[row] * a.token_scales[0];
    const bool poison = a.status[0] != Fp8ActivationOk;
    if constexpr (kSiluPair) {
        // Rows 16 b + i (gate) and 16 b + 8 + i (up) sit in lanes l and l ^ 8.
        const float partner = __shfl_xor(y, 8, 32);
        if ((lane & 8U) == 0U) {
            auto* out = reinterpret_cast<std::uint16_t*>(output.leading) + row_base / 2U +
                        (lane / 16U) * 8U + lane % 8U;
            *out = poison ? kCanonicalBf16QuietNan : silu_pair_value(y, partner);
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
constexpr std::uint32_t kStageChunk = 3U;
constexpr std::uint32_t kPrefillTokens = 256U;
constexpr std::uint32_t kPrefillRows = 128U;
constexpr std::uint32_t kSlabStride = 72U;  // 64 bytes + 8: two-way LDS conflicts at most

template <bool kAccumulate, bool kSiluPair, bool kRawFp8 = false, std::uint32_t kTokenGroups = 4U,
          std::uint32_t kWaveRows = 64U>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6)))
void fp8lut4_prefill_kernel(KernelArgs a) {
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
    const std::uint32_t wave_rows = (wave / kTokenGroups) * kWaveRows;
    // Wave-uniform: the first 2 kRows threads stage one (row, half) slot each.
    const bool stager = thread < 2U * kRows;
    const std::uint32_t token_base = blockIdx.x * kTokenGroups * 64U;
    const std::uint32_t row_base = blockIdx.y * kRows;
    const std::uint32_t columns = a.weight.columns;
    const std::uint32_t tokens = a.tokens;
    if constexpr (!kRawFp8) fp8lut4::load_table(table, 256U);

    // Weight staging: thread -> (row, 32-column half of the slab).
    const std::uint32_t stage_row = (stager ? thread : 0U) >> 1U, stage_half = thread & 1U;
    // N16 x K64 weight tiles: the staged (row, half) slot of slab 0; one tile per slab.
    const std::size_t stage_tile =
        static_cast<std::size_t>((row_base + stage_row) / 16U) * (columns / 64U);
    const std::uint32_t stage_slot = stage_half * 16U + stage_row % 16U;
    // Raw E4M3 rows: the same (row, half) slot read straight from the row-major weight.
    const std::uint8_t* stage_codes = kRawFp8
        ? a.weight.codes + static_cast<std::size_t>(row_base + stage_row) * columns + stage_half * 32U
        : a.weight.codes + stage_tile * 512U + stage_slot * 16U;
    const std::uint8_t* stage_groups = a.weight.groups + stage_tile * 32U + stage_slot;
    const std::uint32_t stage_offset = stage_row * kSlabStride + stage_half * 32U;
    // Activation fragments: token rows clamp to the last token (their results are never stored).
    // The epilogue's token scales and status words load here, landing during the main loop.
    const std::uint8_t* fragment[4];
    float fragment_scale[4];
    bool fragment_poison[4];
#pragma unroll
    for (std::uint32_t i = 0; i < 4U; ++i) {
        std::uint32_t token = token_base + wave_tokens + i * 16U + axis;
        token = token < tokens ? token : tokens - 1U;
        fragment[i] = a.activation + static_cast<std::size_t>(token) * columns + half * 32U;
        fragment_scale[i] = a.token_scales[token];
        fragment_poison[i] = a.status[token] != Fp8ActivationOk;
    }
    // Wave-uniform count of 16-token fragments holding at least one real token: the rest of a
    // partial tile neither loads activations nor issues WMMAs (their results are never stored).
    const std::uint32_t first_token = token_base + wave_tokens;
    const std::uint32_t live = __builtin_amdgcn_readfirstlane(
        first_token >= tokens ? 0U : min(4U, (tokens - first_token + 15U) / 16U));
    struct Load { uint4 w; std::uint32_t g; uint4 x[4][2]; };
    const auto load = [&](std::uint32_t slab_index) {
        Load value;
        if constexpr (kRawFp8) {
            (void)slab_index;  // raw rows are copied at staging time
        } else if (stager) {
            value.w = *reinterpret_cast<const uint4*>(stage_codes + static_cast<std::size_t>(slab_index) * 512U);
            value.g = stage_groups[static_cast<std::size_t>(slab_index) * 32U];
        }
#pragma unroll
        for (std::uint32_t i = 0; i < 4U; ++i) {
            if (i >= live) break;
            const uint4* x = reinterpret_cast<const uint4*>(fragment[i] +
                                                            static_cast<std::size_t>(slab_index) * 64U);
            value.x[i][0] = x[0];
            value.x[i][1] = x[1];
        }
        return value;
    };
    // Raw E4M3 rows are copied through loop-local registers (never part of the carried Load).
    const auto load_raw = [&](std::uint32_t slab_index, uint4 (&raw)[2]) {
        if (!stager) return;
        const uint4* source = reinterpret_cast<const uint4*>(
            stage_codes + static_cast<std::size_t>(slab_index) * 64U);
        raw[0] = source[0];
        raw[1] = source[1];
    };
    const auto store_raw = [&](const uint4 (&raw)[2], std::uint32_t buffer) {
        if (!stager) return;
        uint4* target = reinterpret_cast<uint4*>(&slab[buffer][stage_offset]);
        target[0] = raw[0];
        target[1] = raw[1];
    };
    const auto stage = [&](const Load& value, std::uint32_t buffer) {
        if (!stager) return;
        const uint2 book = table[value.g];
        const uint2 w0 = decode8(value.w.x, book), w1 = decode8(value.w.y, book);
        const uint2 w2 = decode8(value.w.z, book), w3 = decode8(value.w.w, book);
        uint4* target = reinterpret_cast<uint4*>(&slab[buffer][stage_offset]);
        target[0] = uint4{w0.x, w0.y, w1.x, w1.y};
        target[1] = uint4{w2.x, w2.y, w3.x, w3.y};
    };

    F32x8 accumulator[kFragments][4]{};
    const std::uint32_t slabs = columns / 64U;
    Load current = load(0U);
    __syncthreads();  // table
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
#pragma unroll
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
#pragma unroll
            for (std::uint32_t j = 0; j < kFragments; ++j) {
                const uint2 w = *reinterpret_cast<const uint2*>(
                    tile + (wave_rows + j * 16U + axis) * kSlabStride + half * 32U + chunk * 8U);
                weight[j] = I32x2{static_cast<int>(w.x), static_cast<int>(w.y)};
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i) {
                if (i >= live) break;
                const std::uint32_t lo = chunk < 2U ? (chunk == 0U ? current.x[i][0].x : current.x[i][0].z)
                                                    : (chunk == 2U ? current.x[i][1].x : current.x[i][1].z);
                const std::uint32_t hi = chunk < 2U ? (chunk == 0U ? current.x[i][0].y : current.x[i][0].w)
                                                    : (chunk == 2U ? current.x[i][1].y : current.x[i][1].w);
                const I32x2 x{static_cast<int>(lo), static_cast<int>(hi)};
#pragma unroll
                for (std::uint32_t j = 0; j < kFragments; ++j)
                    accumulator[j][i] =
                        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(weight[j], x, accumulator[j][i]);
            }
        }
        if (more) current = next;
        __syncthreads();
    }

    const float* row_scales = a.weight.scales + row_base + wave_rows + half * 8U;
#pragma unroll
    for (std::uint32_t i = 0; i < 4U; ++i) {
        const std::uint32_t token = token_base + wave_tokens + i * 16U + axis;
        if (token >= tokens) continue;
        const float token_scale = fragment_scale[i];
        const bool poison = fragment_poison[i];
#pragma unroll
        for (std::uint32_t j = 0; j < kFragments; ++j) {
            const std::uint32_t row = row_base + wave_rows + j * 16U + half * 8U;
            if constexpr (kSiluPair) {
                // Lane `half` 0 holds this tile's eight gate rows, lane ^ 16 their up partners.
                float y[8], partner[8];
#pragma unroll
                for (std::uint32_t e = 0; e < 8U; ++e) {
                    y[e] = accumulator[j][i][e] * row_scales[j * 16U + e] * token_scale;
                    partner[e] = __shfl_xor(y[e], 16, 32);
                }
                if (half == 0U) {
                    std::uint16_t value[8];
#pragma unroll
                    for (std::uint32_t e = 0; e < 8U; ++e)
                        value[e] = poison ? kCanonicalBf16QuietNan : silu_pair_value(y[e], partner[e]);
                    *reinterpret_cast<uint4*>(reinterpret_cast<std::uint16_t*>(a.output.leading) +
                                              static_cast<std::size_t>(token) * (a.weight.rows / 2U) +
                                              row / 2U) =
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
#pragma unroll
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
        hipLaunchKernelGGL((fp8lut4_small_t_kernel<false, true, kSplit, kSteps, kTiles>), grid, block, 0, stream, args);
    else if (args.output.accumulate)
        hipLaunchKernelGGL((fp8lut4_small_t_kernel<true, false, kSplit, kSteps, kTiles>), grid, block, 0, stream, args);
    else
        hipLaunchKernelGGL((fp8lut4_small_t_kernel<false, false, kSplit, kSteps, kTiles>), grid, block, 0, stream, args);
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
    if (args.weight.groups == nullptr)  // row-scaled E4M3 rows: plain output only
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<false, false, kRowBlocks, kSplit, kTiles, true>), grid, block, 0, stream, args);
    else if (args.output.silu_pair)
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<false, true, kRowBlocks, kSplit, kTiles>), grid, block, 0, stream, args);
    else if (args.output.accumulate)
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<true, false, kRowBlocks, kSplit, kTiles>), grid, block, 0, stream, args);
    else
        hipLaunchKernelGGL((fp8lut4_mid_t_kernel<false, false, kRowBlocks, kSplit, kTiles>), grid, block, 0, stream, args);
}

template <std::uint32_t kRowBlocks, std::uint32_t kSplit>
void launch_mid_shape(const KernelArgs& args, hipStream_t stream) {
    switch ((args.tokens + 15U) / 16U) {
    case 3U: launch_mid_tiles<kRowBlocks, kSplit, 3U>(args, stream); break;
    case 4U: launch_mid_tiles<kRowBlocks, kSplit, 4U>(args, stream); break;
    case 5U: launch_mid_tiles<kRowBlocks, kSplit, 5U>(args, stream); break;
    case 6U: launch_mid_tiles<kRowBlocks, kSplit, 6U>(args, stream); break;
    case 7U: launch_mid_tiles<kRowBlocks, kSplit, 7U>(args, stream); break;
    default: launch_mid_tiles<kRowBlocks, kSplit, 8U>(args, stream); break;
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
    constexpr std::uint32_t kRows = 8U / kTokenGroups * kWaveRows;
    hipLaunchKernelGGL(
        (fp8lut4_prefill_kernel<kAccumulate, kSiluPair, kRawFp8, kTokenGroups, kWaveRows>),
        dim3((args.tokens + kTileTokens - 1U) / kTileTokens, args.weight.rows / kRows), dim3(256),
        0, stream, args);
}

template <bool kAccumulate, bool kSiluPair, bool kRawFp8>
void launch_prefill_shape(const KernelArgs& args, hipStream_t stream) {
    switch (prefill_shape(args.tokens, args.weight.rows)) {
    case PrefillShape::T256R128:
        launch_prefill_as<kAccumulate, kSiluPair, kRawFp8, 4U, 64U>(args, stream); break;
    case PrefillShape::T256R64:
        launch_prefill_as<kAccumulate, kSiluPair, kRawFp8, 4U, 32U>(args, stream); break;
    case PrefillShape::T512R64:
        launch_prefill_as<kAccumulate, kSiluPair, kRawFp8, 8U, 64U>(args, stream); break;
    }
}

hipError_t launch_prefill(const KernelArgs& args, hipStream_t stream) {
    if (args.weight.groups == nullptr) {  // row-scaled E4M3 rows (raw staging), no SiLU pair
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
    return weight.groups != nullptr ||
           (!output.silu_pair && activation.tokens > kMidMaximumTokens);
}

bool valid_target(const Fp8Lut4Weight& weight, const Fp8Lut4Output& output,
                  const Fp8ActivationWorkspace& activation) noexcept {
    return weight.codes != nullptr && valid_raw(weight, output, activation) &&
           weight.scales != nullptr &&
           output.leading != nullptr &&
           (output.split == 0U || (output.trailing != nullptr && output.split % 128U == 0U &&
                                   output.split < weight.rows)) &&
           (!output.silu_pair || (output.split == 0U && !output.accumulate)) &&
           activation.padded_columns == weight.columns &&
           fp8lut4_linear_supported(activation.tokens, weight.rows, weight.columns);
}

KernelArgs kernel_args(const Fp8Lut4Weight& weight, const Fp8Lut4Output& output,
                       const Fp8ActivationWorkspace& activation) noexcept {
    KernelArgs args;
    args.weight = weight;
    args.output = output;
    args.activation = activation.codes;
    args.token_scales = activation.scales;
    args.status = activation.status;
    args.tokens = activation.tokens;
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
    args.weight = Fp8Lut4Weight{weight.codes, nullptr, weight.scales, weight.rows,
                                activation.padded_columns};
    args.output = Fp8Lut4Output{.leading = output};
    args.activation = activation.codes;
    args.token_scales = activation.scales;
    args.status = activation.status;
    args.tokens = activation.tokens;
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
    return tokens <= fp8lut4::kSmallTwoTileTokens && rows % 16U == 0U &&
           (columns / 64U) % 8U == 0U;
}
}  // namespace

bool fp8_row_scaled_projection_supported(std::uint32_t tokens, std::uint32_t rows,
                                         std::uint32_t columns) noexcept {
    return tokens > kMidMaximumTokens && rows != 0U && rows % kPrefillRows == 0U &&
           columns != 0U && columns % 128U == 0U;
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
    if (mid_supported(activation.tokens, weight.rows, weight.columns)) {
        launch_mid(args, stream);
        return hipGetLastError();
    }
    return small_route(activation.tokens, weight.rows, weight.columns)
               ? launch_small(args, stream) : launch_prefill(args, stream);
}

hipError_t fp8lut4_linear_pair(const Fp8Lut4Weight& first, const Fp8Lut4Output& first_output,
                           const Fp8Lut4Weight& second, const Fp8Lut4Output& second_output,
                           const Fp8ActivationWorkspace& activation, hipStream_t stream) noexcept {
    if (activation.codes == nullptr || activation.scales == nullptr ||
        activation.status == nullptr || !valid_target(first, first_output, activation) ||
        !valid_target(second, second_output, activation) ||
        first_output.accumulate != second_output.accumulate || first_output.silu_pair ||
        second_output.silu_pair) {
        return hipErrorInvalidValue;
    }
    if (mid_supported(activation.tokens, first.rows, first.columns) ||
        !small_route(activation.tokens, first.rows, first.columns) ||
        !small_route(activation.tokens, second.rows, second.columns)) {
        const hipError_t result = fp8lut4_linear(first, activation, first_output, stream);
        return result != hipSuccess ? result : fp8lut4_linear(second, activation, second_output, stream);
    }
    KernelArgs args = kernel_args(first, first_output, activation);
    args.second = second;
    args.second_output = second_output;
    return launch_small(args, stream);
}

} // namespace ninfer::ops::r9700::linear
