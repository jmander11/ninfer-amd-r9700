#include "ops/r9700/linear/cb4_linear.h"
#include "ops/r9700/linear/cb4_pipeline_impl.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear {
namespace {

using cb4::decode8;
using cb4::F32x8;
using cb4::I32x2;

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
    Cb4Weight weight{};
    Cb4Output output{};
    Cb4Weight second{};
    Cb4Output second_output{};
    const std::uint8_t* activation = nullptr;
    const float* token_scales = nullptr;
    const std::uint32_t* status = nullptr;
    std::uint32_t tokens = 0;
};

// Element (token, row) of one published output.
__device__ __forceinline__ std::uint16_t* output_at(const Cb4Weight& weight,
                                                    const Cb4Output& output, std::uint32_t token,
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

// ---- Small T (T <= 16): one CTA per 16 rows of the first or the second projection.
template <bool kAccumulate, bool kSiluPair, std::uint32_t kSplit, std::uint32_t kSteps>
__global__ __launch_bounds__(32U * kSplit) void cb4_small_t_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ uint2 table[256];
    __shared__ float partial[kSplit][8][32];
    __shared__ float staged[kSiluPair ? 16 : 1][cb4::kSmallTokens];
    const std::uint32_t first_blocks = a.weight.rows / 16U;
    const bool second = blockIdx.x >= first_blocks;
    // By value: the CTA-uniform selection stays in scalar registers.
    const Cb4Weight weight = second ? a.second : a.weight;
    const Cb4Output output = second ? a.second_output : a.output;
    const std::uint32_t row_base = (second ? blockIdx.x - first_blocks : blockIdx.x) * 16U;
    cb4::load_table(table, 32U * kSplit);
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
    cb4::small_t_rows<kSplit, kSteps>(
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

// ---- One token: a GEMV of 32 rows per CTA (one per lane) and four waves each owning a K quarter
// with four steps in flight. Each lane decodes its row's two 32-column groups per 64-column step
// (slots r % 16 and 16 + r % 16 of the N16 x K64 tile) and accumulates FP8 dot4 products against
// the broadcast activation; the quarters combine in order. No WMMA columns are wasted on absent
// tokens.
constexpr std::uint32_t kGemvSplit = 4U, kGemvSteps = 4U;

template <bool kAccumulate, bool kSiluPair>
__global__ __launch_bounds__(32U * kGemvSplit) void cb4_t1_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ uint2 table[256];
    __shared__ float partial[kGemvSplit][32];
    const std::uint32_t first_blocks = a.weight.rows / 32U;
    const bool second = blockIdx.x >= first_blocks;
    const Cb4Weight weight = second ? a.second : a.weight;
    const Cb4Output output = second ? a.second_output : a.output;
    const std::uint32_t row_base = (second ? blockIdx.x - first_blocks : blockIdx.x) * 32U;
    cb4::load_table(table, 32U * kGemvSplit);
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
constexpr std::uint32_t kPrefillTokens = 256U;
constexpr std::uint32_t kPrefillRows = 128U;
constexpr std::uint32_t kSlabStride = 72U;  // 64 bytes + 8: two-way LDS conflicts at most

template <bool kAccumulate, bool kSiluPair>
__global__ __launch_bounds__(256) void cb4_prefill_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ uint2 table[256];
    __shared__ __attribute__((aligned(16))) std::uint8_t slab[2][kPrefillRows * kSlabStride];
    const std::uint32_t thread = threadIdx.x, lane = thread & 31U;
    const std::uint32_t wave = __builtin_amdgcn_readfirstlane(thread >> 5U);
    const std::uint32_t axis = lane & 15U, half = lane >> 4U;
    const std::uint32_t wave_tokens = (wave & 3U) * 64U, wave_rows = (wave >> 2U) * 64U;
    const std::uint32_t token_base = blockIdx.x * kPrefillTokens;
    const std::uint32_t row_base = blockIdx.y * kPrefillRows;
    const std::uint32_t columns = a.weight.columns;
    const std::uint32_t tokens = a.tokens;
    cb4::load_table(table, 256U);

    // Weight staging: thread -> (row, 32-column half of the slab).
    const std::uint32_t stage_row = thread >> 1U, stage_half = thread & 1U;
    // N16 x K64 weight tiles: the staged (row, half) slot of slab 0; one tile per slab.
    const std::size_t stage_tile =
        static_cast<std::size_t>((row_base + stage_row) / 16U) * (columns / 64U);
    const std::uint32_t stage_slot = stage_half * 16U + stage_row % 16U;
    const std::uint8_t* stage_codes = a.weight.codes + stage_tile * 512U + stage_slot * 16U;
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
    struct Load { uint4 w; std::uint32_t g; uint4 x[4][2]; };
    const auto load = [&](std::uint32_t slab_index) {
        Load value;
        value.w = *reinterpret_cast<const uint4*>(stage_codes + static_cast<std::size_t>(slab_index) * 512U);
        value.g = stage_groups[static_cast<std::size_t>(slab_index) * 32U];
#pragma unroll
        for (std::uint32_t i = 0; i < 4U; ++i) {
            const uint4* x = reinterpret_cast<const uint4*>(fragment[i] +
                                                            static_cast<std::size_t>(slab_index) * 64U);
            value.x[i][0] = x[0];
            value.x[i][1] = x[1];
        }
        return value;
    };
    const auto stage = [&](const Load& value, std::uint32_t buffer) {
        const uint2 book = table[value.g];
        const uint2 w0 = decode8(value.w.x, book), w1 = decode8(value.w.y, book);
        const uint2 w2 = decode8(value.w.z, book), w3 = decode8(value.w.w, book);
        uint4* target = reinterpret_cast<uint4*>(&slab[buffer][stage_offset]);
        target[0] = uint4{w0.x, w0.y, w1.x, w1.y};
        target[1] = uint4{w2.x, w2.y, w3.x, w3.y};
    };

    F32x8 accumulator[4][4]{};
    const std::uint32_t slabs = columns / 64U;
    Load current = load(0U);
    __syncthreads();  // table
    stage(current, 0U);
    __syncthreads();
    for (std::uint32_t s = 0; s < slabs; ++s) {
        const bool more = s + 1U < slabs;
        Load next;
        if (more) next = load(s + 1U);
        const std::uint8_t* tile = slab[s & 1U];
#pragma unroll
        for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
            I32x2 weight[4];
#pragma unroll
            for (std::uint32_t j = 0; j < 4U; ++j) {
                const uint2 w = *reinterpret_cast<const uint2*>(
                    tile + (wave_rows + j * 16U + axis) * kSlabStride + half * 32U + chunk * 8U);
                weight[j] = I32x2{static_cast<int>(w.x), static_cast<int>(w.y)};
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i) {
                const std::uint32_t lo = chunk < 2U ? (chunk == 0U ? current.x[i][0].x : current.x[i][0].z)
                                                    : (chunk == 2U ? current.x[i][1].x : current.x[i][1].z);
                const std::uint32_t hi = chunk < 2U ? (chunk == 0U ? current.x[i][0].y : current.x[i][0].w)
                                                    : (chunk == 2U ? current.x[i][1].y : current.x[i][1].w);
                const I32x2 x{static_cast<int>(lo), static_cast<int>(hi)};
#pragma unroll
                for (std::uint32_t j = 0; j < 4U; ++j)
                    accumulator[j][i] =
                        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(weight[j], x, accumulator[j][i]);
            }
        }
        if (more) {
            stage(next, (s + 1U) & 1U);
            current = next;
        }
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
        for (std::uint32_t j = 0; j < 4U; ++j) {
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

template <std::uint32_t kSplit, std::uint32_t kSteps>
void launch_small_depth(const KernelArgs& args, hipStream_t stream) {
    const dim3 grid((args.weight.rows + args.second.rows) / 16U), block(32U * kSplit);
    if (args.output.silu_pair)
        hipLaunchKernelGGL((cb4_small_t_kernel<false, true, kSplit, kSteps>), grid, block, 0, stream, args);
    else if (args.output.accumulate)
        hipLaunchKernelGGL((cb4_small_t_kernel<true, false, kSplit, kSteps>), grid, block, 0, stream, args);
    else
        hipLaunchKernelGGL((cb4_small_t_kernel<false, false, kSplit, kSteps>), grid, block, 0, stream, args);
}

hipError_t launch_small(const KernelArgs& args, hipStream_t stream) {
    if (args.tokens == 1U) {
        const dim3 grid((args.weight.rows + args.second.rows) / 32U), block(32U * kGemvSplit);
        if (args.output.silu_pair)
            hipLaunchKernelGGL((cb4_t1_kernel<false, true>), grid, block, 0, stream, args);
        else if (args.output.accumulate)
            hipLaunchKernelGGL((cb4_t1_kernel<true, false>), grid, block, 0, stream, args);
        else
            hipLaunchKernelGGL((cb4_t1_kernel<false, false>), grid, block, 0, stream, args);
    } else {
        launch_small_depth<cb4::kVerifySplit, cb4::kVerifySteps>(args, stream);
    }
    return hipGetLastError();
}

hipError_t launch_prefill(const KernelArgs& args, hipStream_t stream) {
    const dim3 grid((args.tokens + kPrefillTokens - 1U) / kPrefillTokens,
                    args.weight.rows / kPrefillRows);
    if (args.output.silu_pair)
        hipLaunchKernelGGL((cb4_prefill_kernel<false, true>), grid, dim3(256), 0, stream, args);
    else if (args.output.accumulate)
        hipLaunchKernelGGL((cb4_prefill_kernel<true, false>), grid, dim3(256), 0, stream, args);
    else
        hipLaunchKernelGGL((cb4_prefill_kernel<false, false>), grid, dim3(256), 0, stream, args);
    return hipGetLastError();
}

bool valid_target(const Cb4Weight& weight, const Cb4Output& output,
                  const Fp8ActivationWorkspace& activation) noexcept {
    return weight.codes != nullptr && weight.groups != nullptr && weight.scales != nullptr &&
           output.leading != nullptr &&
           (output.split == 0U || (output.trailing != nullptr && output.split % 128U == 0U &&
                                   output.split < weight.rows)) &&
           (!output.silu_pair || (output.split == 0U && !output.accumulate)) &&
           activation.padded_columns == weight.columns &&
           cb4_linear_supported(activation.tokens, weight.rows, weight.columns);
}

KernelArgs kernel_args(const Cb4Weight& weight, const Cb4Output& output,
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

const std::array<std::array<std::uint8_t, 8>, 256>& cb4_magnitude_table() noexcept {
    return cb4::kTable;
}

bool cb4_linear_supported(std::uint32_t tokens, std::uint32_t rows,
                          std::uint32_t columns) noexcept {
    if (tokens == 0U || rows == 0U || columns == 0U || columns % 128U != 0U) return false;
    if (tokens <= cb4::kSmallTokens)
        return (tokens == 1U ? rows % 32U == 0U && (columns / 64U) % 16U == 0U
                             : rows % 16U == 0U && (columns / 64U) % 8U == 0U);
    return rows % kPrefillRows == 0U;
}

hipError_t cb4_linear(const Cb4Weight& weight, const Fp8ActivationWorkspace& activation,
                      const Cb4Output& output, hipStream_t stream) noexcept {
    if (activation.codes == nullptr || activation.scales == nullptr ||
        activation.status == nullptr || !valid_target(weight, output, activation)) {
        return hipErrorInvalidValue;
    }
    const KernelArgs args = kernel_args(weight, output, activation);
    return activation.tokens <= cb4::kSmallTokens ? launch_small(args, stream)
                                                  : launch_prefill(args, stream);
}

hipError_t cb4_linear_pair(const Cb4Weight& first, const Cb4Output& first_output,
                           const Cb4Weight& second, const Cb4Output& second_output,
                           const Fp8ActivationWorkspace& activation, hipStream_t stream) noexcept {
    if (activation.codes == nullptr || activation.scales == nullptr ||
        activation.status == nullptr || !valid_target(first, first_output, activation) ||
        !valid_target(second, second_output, activation) ||
        first_output.accumulate != second_output.accumulate || first_output.silu_pair ||
        second_output.silu_pair) {
        return hipErrorInvalidValue;
    }
    if (activation.tokens > cb4::kSmallTokens) {
        const hipError_t result = cb4_linear(first, activation, first_output, stream);
        return result != hipSuccess ? result : cb4_linear(second, activation, second_output, stream);
    }
    KernelArgs args = kernel_args(first, first_output, activation);
    args.second = second;
    args.second_output = second_output;
    return launch_small(args, stream);
}

} // namespace ninfer::ops::r9700::linear
