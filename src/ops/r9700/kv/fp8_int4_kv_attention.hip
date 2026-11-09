#include "fp8_int4_kv_attention.h"
#include "r9700_attention_profile.h"

#include <hip/hip_fp8.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace ninfer::ops::r9700::kv {
namespace {
inline constexpr std::uint32_t kHeadDim        = 256;
inline constexpr std::uint32_t kPageSize       = 64;
inline constexpr std::uint32_t kTokensPerBlock = 128;
// gfx1201 exposes at most 65,535 workgroups in grid Y. Split only the row dimension; every launch
// still covers all query heads and preserves ordered-stream output independence.
inline constexpr std::uint32_t kMaximumGridY = 65535;
using I32x2                                  = int __attribute__((ext_vector_type(2)));
using I16x8                                  = short __attribute__((ext_vector_type(8)));
using F32x8                                  = float __attribute__((ext_vector_type(8)));
inline constexpr std::uint32_t kSplitTokens  = 512;

[[nodiscard]] constexpr bool valid_plane_layout(Fp8Int4KvPlaneLayout layout) noexcept {
    return layout == Fp8Int4KvPlaneLayout::FeatureFastest ||
           layout == Fp8Int4KvPlaneLayout::TokenFastest;
}

// The codec's persistent planes are bound; Nvfp4 additionally has its one fixed plane order and
// G16.
[[nodiscard]] constexpr bool valid_codec_planes(const Fp8Int4KvAttentionArgs& a) noexcept {
    if (a.codec == KvCodec::Fp8Int4) return a.value_scale_bits != nullptr;
    return a.codec == KvCodec::Nvfp4 && a.key_scale_codes != nullptr &&
           a.value_scale_codes != nullptr && a.value_group == 16U &&
           a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest &&
           a.value_layout == Fp8Int4KvPlaneLayout::FeatureFastest &&
           a.value_scale_layout == Fp8Int4KvPlaneLayout::FeatureFastest;
}

// One E2M1 magnitude table as the high byte of FP16 (exact; the low byte is zero).
__device__ __forceinline__ std::uint32_t e2m1_f16_high_bytes(std::uint32_t codes) {
    return __builtin_amdgcn_perm(0x46444240U, 0x3e3c3800U, codes & 0x07070707U) |
           ((codes & 0x08080808U) << 4U);
}

// The same table divided by eight, for the FP16 V/8 PV staging.
__device__ __forceinline__ std::uint32_t e2m1_over8_f16_high_bytes(std::uint32_t codes) {
    return __builtin_amdgcn_perm(0x3a383634U, 0x32302c00U, codes & 0x07070707U) |
           ((codes & 0x08080808U) << 4U);
}

__device__ __forceinline__ float e2m1_float(std::uint32_t code) {
    return __half2float(
        __ushort_as_half(static_cast<unsigned short>((e2m1_f16_high_bytes(code) & 0xffU) << 8U)));
}

__device__ __forceinline__ float e4m3_float(std::uint32_t code) {
    return __half2float(__half(__hip_cvt_fp8_to_halfraw(code, __HIP_E4M3)));
}

template <bool TokenFastest>
__device__ std::size_t key_offset(std::size_t token, std::uint32_t head, std::uint32_t feature,
                                  std::uint32_t heads, std::size_t physical_tokens) {
    const std::size_t page = token / kPageSize, in_page = token % kPageSize;
    const std::size_t pages = physical_tokens / kPageSize;
    if constexpr (TokenFastest) {
        return in_page +
               kPageSize * (static_cast<std::size_t>(feature) + kHeadDim * (page + pages * head));
    }
    return feature +
           kHeadDim * (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
}

template <bool TokenFastest>
__global__ void
qk_vector_kernel(const hip_bfloat16* query, const std::uint8_t* key_codes,
                 const std::uint32_t* page_table, const std::int32_t* page_table_row,
                 std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count,
                 std::size_t context, std::size_t physical_tokens, std::uint32_t kv_heads,
                 std::uint32_t q_to_kv, float attention_scale, float* scores) {
    const std::uint32_t query_head = blockIdx.y;
    const std::size_t logical =
        static_cast<std::size_t>(blockIdx.x) * kTokensPerBlock + threadIdx.x;
    if (logical >= context) return;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            scores[static_cast<std::size_t>(query_head) * context + logical] = __builtin_nanf("");
            return;
        }
        page_table += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    const std::size_t physical =
        static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize + logical % kPageSize;
    const std::uint32_t kv_head = query_head / q_to_kv;
    if (physical >= physical_tokens || kv_head >= kv_heads) {
        scores[static_cast<std::size_t>(query_head) * context + logical] = __builtin_nanf("");
        return;
    }
    float sum = 0.0F;
#pragma unroll
    for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
        const float q =
            static_cast<float>(query[static_cast<std::size_t>(query_head) * kHeadDim + feature]);
        const std::uint8_t code = key_codes[key_offset<TokenFastest>(physical, kv_head, feature,
                                                                     kv_heads, physical_tokens)];
        sum = fmaf(q, __half2float(__half(__hip_cvt_fp8_to_halfraw(code, __HIP_E4M3))), sum);
    }
    scores[static_cast<std::size_t>(query_head) * context + logical] = sum * attention_scale;
}

__device__ __attribute__((unused)) int pack_fp8x4(const std::uint8_t* bytes) {
    return static_cast<int>(static_cast<std::uint32_t>(bytes[0]) |
                            (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                            (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                            (static_cast<std::uint32_t>(bytes[3]) << 24U));
}

template <bool TokenFastest>
__global__ void qk_wmma_kernel(const hip_bfloat16* query, const std::uint8_t* key_codes,
                               const std::uint32_t* page_table, const std::int32_t* page_table_row,
                               std::uint32_t page_table_row_stride,
                               std::uint32_t page_table_row_count, std::size_t context,
                               std::size_t physical_tokens, std::uint32_t q_to_kv,
                               float attention_scale, float* scores) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane    = threadIdx.x & 31U;
    const std::uint32_t k_half  = lane >> 4U;
    const std::uint32_t axis    = lane & 15U;
    const std::uint32_t kv_head = blockIdx.y;
    const std::size_t tile      = static_cast<std::size_t>(blockIdx.x) * 16U;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            for (std::uint32_t j = 0; j < 8; ++j) {
                const std::uint32_t row   = (lane >> 4U) * 8U + j;
                const std::size_t logical = tile + axis;
                if (row < q_to_kv && logical < context) {
                    const std::uint32_t query_head = kv_head * q_to_kv + row;
                    scores[static_cast<std::size_t>(query_head) * context + logical] =
                        __builtin_nanf("");
                }
            }
            return;
        }
        page_table += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    F32x8 accum = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint32_t chunk = 0; chunk < kHeadDim / 16U; ++chunk) {
        std::uint8_t a_bytes[8];
        std::uint8_t b_bytes[8];
        const std::uint32_t query_head = kv_head * q_to_kv + axis;
        for (std::uint32_t j = 0; j < 8; ++j) {
            const std::uint32_t feature = chunk * 16U + k_half * 8U + j;
            a_bytes[j] =
                query_head < gridDim.y * q_to_kv
                    ? __hip_cvt_float_to_fp8(
                          static_cast<float>(
                              query[static_cast<std::size_t>(query_head) * kHeadDim + feature]),
                          __HIP_SATFINITE, __HIP_E4M3)
                    : 0U;
            const std::size_t logical = tile + axis;
            if (logical < context) {
                const std::size_t physical =
                    static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
                    logical % kPageSize;
                b_bytes[j] = physical < physical_tokens
                                 ? key_codes[key_offset<TokenFastest>(physical, kv_head, feature,
                                                                      gridDim.y, physical_tokens)]
                                 : 0U;
            } else {
                b_bytes[j] = 0U;
            }
        }
        const I32x2 a = {pack_fp8x4(a_bytes), pack_fp8x4(a_bytes + 4)};
        const I32x2 b = {pack_fp8x4(b_bytes), pack_fp8x4(b_bytes + 4)};
        accum         = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, accum);
    }
    for (std::uint32_t j = 0; j < 8; ++j) {
        const std::uint32_t row        = (lane >> 4U) * 8U + j;
        const std::uint32_t query_head = kv_head * q_to_kv + row;
        const std::size_t logical      = tile + axis;
        if (row < q_to_kv && logical < context) {
            const std::size_t physical =
                static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
                logical % kPageSize;
            scores[static_cast<std::size_t>(query_head) * context + logical] =
                physical < physical_tokens ? accum[j] * attention_scale : __builtin_nanf("");
        }
    }
#else
    (void)query;
    (void)key_codes;
    (void)page_table;
    (void)page_table_row;
    (void)page_table_row_stride;
    (void)page_table_row_count;
    (void)context;
    (void)physical_tokens;
    (void)q_to_kv;
    (void)attention_scale;
    (void)scores;
#endif
}

} // namespace

hipError_t fp8_kv_qk_vector(const Fp8KvQkArgs& a, hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.page_table || !a.page_table_pages || !a.context ||
        !a.physical_tokens || a.physical_tokens % kPageSize || !a.query_heads || !a.kv_heads ||
        !a.q_to_kv || a.query_heads != a.kv_heads * a.q_to_kv || !a.scores ||
        !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        !valid_plane_layout(a.key_layout) ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize)
        return hipErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>((a.context + kTokensPerBlock - 1U) / kTokensPerBlock),
                    a.query_heads);
    if (a.key_layout == Fp8Int4KvPlaneLayout::FeatureFastest) {
        hipLaunchKernelGGL((qk_vector_kernel<false>), grid, dim3(kTokensPerBlock), 0, stream,
                           a.query, a.key_codes, a.page_table, a.page_table_row,
                           a.page_table_row_stride, a.page_table_row_count, a.context,
                           a.physical_tokens, a.kv_heads, a.q_to_kv, a.attention_scale, a.scores);
    } else if (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest) {
        hipLaunchKernelGGL((qk_vector_kernel<true>), grid, dim3(kTokensPerBlock), 0, stream,
                           a.query, a.key_codes, a.page_table, a.page_table_row,
                           a.page_table_row_stride, a.page_table_row_count, a.context,
                           a.physical_tokens, a.kv_heads, a.q_to_kv, a.attention_scale, a.scores);
    } else
        return hipErrorInvalidValue;
    return hipGetLastError();
}

hipError_t fp8_kv_qk_wmma(const Fp8KvQkArgs& a, hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.page_table || !a.page_table_pages || !a.context ||
        !a.physical_tokens || a.physical_tokens % kPageSize || !a.query_heads || !a.kv_heads ||
        !a.q_to_kv || a.q_to_kv != 6U || a.query_heads != a.kv_heads * a.q_to_kv || !a.scores ||
        !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        !valid_plane_layout(a.key_layout) ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize)
        return hipErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>((a.context + 15U) / 16U), a.kv_heads);
    if (a.key_layout == Fp8Int4KvPlaneLayout::FeatureFastest) {
        hipLaunchKernelGGL((qk_wmma_kernel<false>), grid, dim3(32), 0, stream, a.query, a.key_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens, a.q_to_kv,
                           a.attention_scale, a.scores);
    } else if (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest) {
        hipLaunchKernelGGL((qk_wmma_kernel<true>), grid, dim3(32), 0, stream, a.query, a.key_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens, a.q_to_kv,
                           a.attention_scale, a.scores);
    } else
        return hipErrorInvalidValue;
    return hipGetLastError();
}

template <std::uint32_t Group, bool VToken, bool SToken>
__device__ std::size_t int4_offset(std::size_t token, std::uint32_t head, std::uint32_t feature,
                                   std::uint32_t heads, std::size_t physical_tokens, bool scale) {
    const std::size_t page = token / kPageSize, in_page = token % kPageSize;
    const std::size_t pages        = physical_tokens / kPageSize;
    const std::uint32_t leading    = scale ? kHeadDim / Group : kHeadDim / 2U;
    const std::uint32_t coordinate = scale ? feature / Group : feature / 2U;
    const bool token_fastest       = scale ? SToken : VToken;
    return token_fastest
               ? in_page + kPageSize * (static_cast<std::size_t>(coordinate) +
                                        leading * (page + pages * head))
               : coordinate + leading * (in_page + kPageSize * (static_cast<std::size_t>(head) +
                                                                heads * page));
}

template <std::uint32_t Group, bool VToken, bool SToken>
__device__ __forceinline__ void
pv_vector_body(const float* probabilities, const std::uint8_t* values, const std::uint16_t* scales,
               const std::uint32_t* pages, const std::int32_t* page_table_row,
               std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count,
               std::size_t context, std::size_t physical_tokens, std::uint32_t q_to_kv,
               float* output) {
    // Each feature retains its independent serial FP32 FMA chain.
    const std::uint32_t qh = blockIdx.x, feature = blockIdx.y * blockDim.x + threadIdx.x,
                        kvh = qh / q_to_kv;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            output[static_cast<std::size_t>(qh) * kHeadDim + feature] = __builtin_nanf("");
            return;
        }
        pages += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    float sum                     = 0.0F;
    constexpr unsigned kLoadAhead = 32U;
    for (std::size_t page_base = 0; page_base < context; page_base += kPageSize) {
        const std::size_t physical_base =
            static_cast<std::size_t>(pages[page_base / kPageSize]) * kPageSize;
        if (physical_base >= physical_tokens) {
            output[static_cast<std::size_t>(qh) * kHeadDim + feature] = __builtin_nanf("");
            return;
        }
        const std::size_t end = min(context, page_base + kPageSize);
        std::size_t base      = page_base;
        for (; base + kLoadAhead <= end; base += kLoadAhead) {
            float probability[kLoadAhead];
            std::uint8_t packed[kLoadAhead];
            std::uint16_t bits[kLoadAhead];
// Stage independent reads ahead of the ordered accumulation. Unlike split-K,
// this exposes memory-level parallelism without changing any FP32 FMA chain.
#pragma unroll
            for (unsigned i = 0; i < kLoadAhead; ++i) {
                const std::size_t logical  = base + i;
                const std::size_t physical = physical_base + logical - page_base;
                packed[i]                  = values[int4_offset<Group, VToken, SToken>(
                    physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, false)];
                bits[i]                    = scales[int4_offset<Group, VToken, SToken>(
                    physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, true)];
                probability[i] = probabilities[static_cast<std::size_t>(qh) * context + logical];
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (unsigned i = 0; i < kLoadAhead; ++i) {
                int code = static_cast<int>((packed[i] >> ((feature & 1U) * 4U)) & 0x0fU);
                code     = code >= 8 ? code - 16 : code;
                // Masked cache slots may hold nonfinite scale bits. Zero probability must
                // contribute zero, rather than letting fmaf(0, NaN, sum) poison the row.
                const float value =
                    probability[i] != 0.0F
                        ? static_cast<float>(code) * __half2float(__ushort_as_half(bits[i]))
                        : 0.0F;
                sum = fmaf(probability[i], value, sum);
            }
        }
        for (; base < end; ++base) {
            const std::size_t physical = physical_base + base - page_base;
            const std::uint8_t packed  = values[int4_offset<Group, VToken, SToken>(
                physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, false)];
            int code                 = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
            code                     = code >= 8 ? code - 16 : code;
            const std::uint16_t bits = scales[int4_offset<Group, VToken, SToken>(
                physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, true)];
            const float probability  = probabilities[static_cast<std::size_t>(qh) * context + base];
            const float value = probability != 0.0F ? static_cast<float>(code) *
                                                          __half2float(__ushort_as_half(bits))
                                                    : 0.0F;
            sum               = fmaf(probability, value, sum);
        }
    }
    output[static_cast<std::size_t>(qh) * kHeadDim + feature] = sum;
}

template <std::uint32_t Group, bool VToken, bool SToken>
__global__ void
pv_vector_kernel(const float* probabilities, const std::uint8_t* values,
                 const std::uint16_t* scales, const std::uint32_t* pages,
                 const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
                 std::uint32_t page_table_row_count, std::size_t context,
                 std::size_t physical_tokens, std::uint32_t q_to_kv, float* output) {
    pv_vector_body<Group, VToken, SToken>(probabilities, values, scales, pages, page_table_row,
                                          page_table_row_stride, page_table_row_count, context,
                                          physical_tokens, q_to_kv, output);
}

template <std::uint32_t G, bool V, bool S>
hipError_t pv_launch(const Int4KvPvArgs& a, hipStream_t stream) {
    hipLaunchKernelGGL((pv_vector_kernel<G, V, S>), dim3(a.query_heads, kHeadDim / 32U), dim3(32),
                       0, stream, a.probabilities, a.value_codes, a.value_scale_bits, a.page_table,
                       a.page_table_row, a.page_table_row_stride, a.page_table_row_count, a.context,
                       a.physical_tokens, a.q_to_kv, a.output);
    return hipGetLastError();
}

template <std::uint32_t G>
hipError_t pv_dispatch(const Int4KvPvArgs& a, hipStream_t s) {
    const unsigned m = (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
                       (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U);
    switch (m) {
    case 0:
        return pv_launch<G, false, false>(a, s);
    case 1:
        return pv_launch<G, true, false>(a, s);
    case 2:
        return pv_launch<G, false, true>(a, s);
    default:
        return pv_launch<G, true, true>(a, s);
    }
}

hipError_t int4_kv_pv_vector(const Int4KvPvArgs& a, hipStream_t stream) noexcept {
    if (!a.probabilities || !a.value_codes || !a.value_scale_bits || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens || a.physical_tokens % kPageSize ||
        !a.query_heads || !a.kv_heads || !a.q_to_kv || a.query_heads != a.kv_heads * a.q_to_kv ||
        !a.output || !valid_plane_layout(a.value_layout) ||
        !valid_plane_layout(a.value_scale_layout) ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize)
        return hipErrorInvalidValue;
    return a.value_group == 16   ? pv_dispatch<16>(a, stream)
           : a.value_group == 32 ? pv_dispatch<32>(a, stream)
                                 : hipErrorInvalidValue;
}

// Per-element plane access of the score-streaming routes.
struct Fp8Int4ElementCodec {
    using ValueScale = std::uint16_t;

    struct Value {
        float code, scale;
    };

    [[nodiscard]] static const ValueScale* value_scales(const Fp8Int4KvAttentionArgs& a) {
        return a.value_scale_bits;
    }

    template <bool KToken>
    __device__ static __forceinline__ float
    key(const std::uint8_t* keys, const std::uint8_t*, std::size_t physical, std::uint32_t kvh,
        std::uint32_t feature, std::uint32_t heads, std::size_t physical_tokens) {
        return __half2float(__half(__hip_cvt_fp8_to_halfraw(
            keys[key_offset<KToken>(physical, kvh, feature, heads, physical_tokens)], __HIP_E4M3)));
    }

    template <std::uint32_t Group, bool VToken, bool SToken>
    __device__ static __forceinline__ Value value(const std::uint8_t* values,
                                                  const ValueScale* scales, std::size_t physical,
                                                  std::uint32_t kvh, std::uint32_t feature,
                                                  std::uint32_t heads,
                                                  std::size_t physical_tokens) {
        const std::uint8_t packed = values[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, heads, physical_tokens, false)];
        int code                  = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
        code                      = code >= 8 ? code - 16 : code;
        const std::uint16_t bits  = scales[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, heads, physical_tokens, true)];
        return {static_cast<float>(code), __half2float(__ushort_as_half(bits))};
    }
};

struct Nvfp4ElementCodec {
    using ValueScale = std::uint8_t;

    struct Value {
        float code, scale;
    };

    [[nodiscard]] static const ValueScale* value_scales(const Fp8Int4KvAttentionArgs& a) {
        return a.value_scale_codes;
    }

    template <bool KToken>
    __device__ static __forceinline__ float
    key(const std::uint8_t* keys, const std::uint8_t* key_scales, std::size_t physical,
        std::uint32_t kvh, std::uint32_t feature, std::uint32_t, std::size_t physical_tokens) {
        static_assert(KToken);
        const std::size_t in_page   = physical % kPageSize;
        const std::size_t head_page = physical / kPageSize + physical_tokens / kPageSize * kvh;
        const std::uint32_t packed  = keys[in_page + kPageSize * (feature / 2U + 128U * head_page)];
        const std::uint32_t scale =
            key_scales[in_page + kPageSize * (feature / 16U + 16U * head_page)];
        return e2m1_float((packed >> ((feature & 1U) * 4U)) & 0x0fU) * e4m3_float(scale);
    }

    template <std::uint32_t Group, bool VToken, bool SToken>
    __device__ static __forceinline__ Value value(const std::uint8_t* values,
                                                  const ValueScale* scales, std::size_t physical,
                                                  std::uint32_t kvh, std::uint32_t feature,
                                                  std::uint32_t heads,
                                                  std::size_t physical_tokens) {
        static_assert(Group == 16U && !VToken && !SToken);
        const std::uint32_t packed = values[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, heads, physical_tokens, false)];
        const std::uint32_t scale  = scales[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, heads, physical_tokens, true)];
        return {e2m1_float((packed >> ((feature & 1U) * 4U)) & 0x0fU), e4m3_float(scale)};
    }
};

struct FusedAttentionShared {
    float wave_reductions[kHeadDim / 32U];
    float running_max;
    float denominator;
    float score;
    float rescale;
    float probability_numerator;
};

template <class Codec, std::uint32_t Group, bool KToken, bool VToken, bool SToken, bool TreeMask,
          bool DeviceActiveRows>
__device__ __forceinline__ void fused_attention_body(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint8_t* key_scales,
    const std::uint8_t* values, const typename Codec::ValueScale* scales,
    const std::uint32_t* pages, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count, std::size_t context,
    std::size_t physical_tokens, std::uint32_t q_to_kv, const std::int32_t* row_positions,
    const std::int32_t* ancestor_masks, const std::int32_t* prefix_lengths,
    std::uint32_t prefix_length_stride, const std::int32_t* active_query_rows,
    std::uint32_t query_rows, std::uint32_t row_offset, float attention_scale, float* output,
    FusedAttentionShared& shared) {
    const std::uint32_t row     = row_offset + blockIdx.y;
    const std::uint32_t qh      = blockIdx.x;
    const std::uint32_t feature = threadIdx.x;
    const std::size_t row_base  = (static_cast<std::size_t>(row) * gridDim.x + qh) * kHeadDim;
    // The count is stream-ordered device state and uniform across every block in this launch.
    // Resolve it before touching row-specific query/metadata/cache inputs so an inactive fixed-
    // width tail is a true semantic zero rather than merely a masked result.
    if constexpr (DeviceActiveRows) {
        const std::int32_t represented_active_rows = active_query_rows[0];
        if (represented_active_rows < 0 ||
            static_cast<std::uint32_t>(represented_active_rows) > query_rows) {
            output[row_base + feature] = __builtin_nanf("");
            return;
        }
        if (row >= static_cast<std::uint32_t>(represented_active_rows)) {
            output[row_base + feature] = 0.0F;
            return;
        }
    }
    if (page_table_row != nullptr) {
        const std::int32_t represented_table_row = *page_table_row;
        if (represented_table_row < 0 ||
            static_cast<std::uint32_t>(represented_table_row) >= page_table_row_count) {
            output[row_base + feature] = __builtin_nanf("");
            return;
        }
        pages += static_cast<std::size_t>(represented_table_row) * page_table_row_stride;
    }
    const std::uint32_t kvh         = qh / q_to_kv;
    const std::int32_t row_position = row_positions == nullptr ? -1 : row_positions[row];
    const bool invalid_row =
        row_positions != nullptr &&
        (row_position < 0 || static_cast<std::size_t>(row_position) >= context);
    const std::size_t row_context = row_positions == nullptr || invalid_row
                                        ? context
                                        : static_cast<std::size_t>(row_position) + 1U;
    std::size_t prefix            = 0;
    std::size_t suffix            = 0;
    std::uint32_t ancestor_mask   = 0U;
    bool invalid_tree             = false;
    if constexpr (TreeMask) {
        const std::int32_t represented_prefix =
            prefix_lengths[static_cast<std::size_t>(row) * prefix_length_stride];
        ancestor_mask = static_cast<std::uint32_t>(ancestor_masks[row]);
        invalid_tree  = represented_prefix < 0;
        if (!invalid_tree) {
            prefix       = static_cast<std::size_t>(represented_prefix);
            invalid_tree = prefix > row_context;
        }
        if (!invalid_tree) {
            suffix       = row_context - prefix;
            invalid_tree = suffix >= 32U;
        }
        if (!invalid_tree && prefix == 0U) {
            const std::uint32_t available =
                suffix == 0U ? 0U : ((1U << static_cast<std::uint32_t>(suffix)) - 1U);
            invalid_tree = (ancestor_mask & available) == 0U;
        }
    }
    // D256 maps to eight wave32s.  Reduce within each wave in registers, then let the first
    // wave merge eight partials.  This replaces the previous full-block binary tree and its
    // synchronization at every level while preserving FP32 score arithmetic.
    if (feature == 0) {
        shared.running_max = -INFINITY;
        shared.denominator = 0.0F;
    }
    const float q   = static_cast<float>(query[row_base + feature]);
    float numerator = 0.0F;
    __syncthreads();
    // A corrupt transaction frontier is made visibly invalid, rather than becoming an empty
    // reduction or a page-table out-of-bounds access.  Validity is normally established by the
    // typed cache owner before this raw kernel is called.
    if (invalid_row || row_context == 0 || row_context > context || invalid_tree) {
        output[row_base + feature] = __builtin_nanf("");
        return;
    }
    for (std::size_t logical = 0; logical < row_context; ++logical) {
        if constexpr (TreeMask) {
            if (logical >= prefix &&
                (ancestor_mask & (1U << static_cast<std::uint32_t>(logical - prefix))) == 0U) {
                continue;
            }
        }
        const std::size_t physical =
            static_cast<std::size_t>(pages[logical / kPageSize]) * kPageSize + logical % kPageSize;
        if (physical >= physical_tokens) {
            output[row_base + feature] = __builtin_nanf("");
            return;
        }
        float partial = q * Codec::template key<KToken>(keys, key_scales, physical, kvh, feature,
                                                        gridDim.x / q_to_kv, physical_tokens);
        for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
            partial += __shfl_down(partial, stride, 32);
        }
        if ((feature & 31U) == 0U) { shared.wave_reductions[feature / 32U] = partial; }
        __syncthreads();
        if (feature < kHeadDim / 32U) {
            partial = shared.wave_reductions[feature];
            for (std::uint32_t stride = 4U; stride != 0U; stride >>= 1U) {
                partial += __shfl_down(partial, stride, 8);
            }
        }
        if (feature == 0) {
            shared.score                 = partial * attention_scale;
            const float next_max         = fmaxf(shared.running_max, shared.score);
            shared.rescale               = __expf(shared.running_max - next_max);
            shared.probability_numerator = __expf(shared.score - next_max);
            shared.denominator = shared.denominator * shared.rescale + shared.probability_numerator;
            shared.running_max = next_max;
        }
        __syncthreads();
        const auto element = Codec::template value<Group, VToken, SToken>(
            values, scales, physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens);
        numerator = numerator * shared.rescale +
                    shared.probability_numerator * element.code * element.scale;
        __syncthreads();
    }
    output[row_base + feature] = numerator / shared.denominator;
}

template <class Codec, std::uint32_t Group, bool KToken, bool VToken, bool SToken,
          bool DeviceActiveRows>
__global__ void fused_attention_causal_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint8_t* key_scales,
    const std::uint8_t* values, const typename Codec::ValueScale* scales,
    const std::uint32_t* pages, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count, std::size_t context,
    std::size_t physical_tokens, std::uint32_t q_to_kv, const std::int32_t* row_positions,
    const std::int32_t* active_query_rows, std::uint32_t query_rows, std::uint32_t row_offset,
    float attention_scale, float* output) {
    __shared__ FusedAttentionShared shared;
    fused_attention_body<Codec, Group, KToken, VToken, SToken, false, DeviceActiveRows>(
        query, keys, key_scales, values, scales, pages, page_table_row, page_table_row_stride,
        page_table_row_count, context, physical_tokens, q_to_kv, row_positions, nullptr, nullptr,
        1U, active_query_rows, query_rows, row_offset, attention_scale, output, shared);
}

template <class Codec, std::uint32_t Group, bool KToken, bool VToken, bool SToken,
          bool DeviceActiveRows>
__global__ void fused_attention_tree_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint8_t* key_scales,
    const std::uint8_t* values, const typename Codec::ValueScale* scales,
    const std::uint32_t* pages, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count, std::size_t context,
    std::size_t physical_tokens, std::uint32_t q_to_kv, const std::int32_t* row_positions,
    const std::int32_t* ancestor_masks, const std::int32_t* prefix_lengths,
    std::uint32_t prefix_length_stride, const std::int32_t* active_query_rows,
    std::uint32_t query_rows, std::uint32_t row_offset, float attention_scale, float* output) {
    __shared__ FusedAttentionShared shared;
    fused_attention_body<Codec, Group, KToken, VToken, SToken, true, DeviceActiveRows>(
        query, keys, key_scales, values, scales, pages, page_table_row, page_table_row_stride,
        page_table_row_count, context, physical_tokens, q_to_kv, row_positions, ancestor_masks,
        prefix_lengths, prefix_length_stride, active_query_rows, query_rows, row_offset,
        attention_scale, output, shared);
}

template <class Codec, std::uint32_t G, bool K, bool V, bool S>
hipError_t fused_attention_launch(const Fp8Int4KvAttentionArgs& a, hipStream_t stream) {
    for (std::uint32_t row_offset = 0; row_offset < a.query_rows;) {
        const std::uint32_t rows =
            a.query_rows - row_offset < kMaximumGridY ? a.query_rows - row_offset : kMaximumGridY;
        if (a.ancestor_masks != nullptr) {
            if (a.active_query_rows != nullptr) {
                hipLaunchKernelGGL((fused_attention_tree_kernel<Codec, G, K, V, S, true>),
                                   dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query,
                                   a.key_codes, a.key_scale_codes, a.value_codes,
                                   Codec::value_scales(a), a.page_table, a.page_table_row,
                                   a.page_table_row_stride, a.page_table_row_count, a.context,
                                   a.physical_tokens, a.q_to_kv, a.row_positions, a.ancestor_masks,
                                   a.prefix_lengths, a.prefix_length_stride, a.active_query_rows,
                                   a.query_rows, row_offset, a.attention_scale, a.output);
            } else {
                hipLaunchKernelGGL((fused_attention_tree_kernel<Codec, G, K, V, S, false>),
                                   dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query,
                                   a.key_codes, a.key_scale_codes, a.value_codes,
                                   Codec::value_scales(a), a.page_table, a.page_table_row,
                                   a.page_table_row_stride, a.page_table_row_count, a.context,
                                   a.physical_tokens, a.q_to_kv, a.row_positions, a.ancestor_masks,
                                   a.prefix_lengths, a.prefix_length_stride, nullptr, a.query_rows,
                                   row_offset, a.attention_scale, a.output);
            }
        } else {
            if (a.active_query_rows != nullptr) {
                hipLaunchKernelGGL(
                    (fused_attention_causal_kernel<Codec, G, K, V, S, true>),
                    dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query, a.key_codes,
                    a.key_scale_codes, a.value_codes, Codec::value_scales(a), a.page_table,
                    a.page_table_row, a.page_table_row_stride, a.page_table_row_count, a.context,
                    a.physical_tokens, a.q_to_kv, a.row_positions, a.active_query_rows,
                    a.query_rows, row_offset, a.attention_scale, a.output);
            } else {
                hipLaunchKernelGGL((fused_attention_causal_kernel<Codec, G, K, V, S, false>),
                                   dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query,
                                   a.key_codes, a.key_scale_codes, a.value_codes,
                                   Codec::value_scales(a), a.page_table, a.page_table_row,
                                   a.page_table_row_stride, a.page_table_row_count, a.context,
                                   a.physical_tokens, a.q_to_kv, a.row_positions, nullptr,
                                   a.query_rows, row_offset, a.attention_scale, a.output);
            }
        }
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) { return error; }
        row_offset += rows;
    }
    return hipSuccess;
}

template <std::uint32_t G>
hipError_t fused_attention_dispatch(const Fp8Int4KvAttentionArgs& a, hipStream_t stream) {
    const unsigned layout = (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
                            (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U) |
                            (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 4U : 0U);
    switch (layout) {
    case 0:
        return fused_attention_launch<Fp8Int4ElementCodec, G, false, false, false>(a, stream);
    case 1:
        return fused_attention_launch<Fp8Int4ElementCodec, G, true, false, false>(a, stream);
    case 2:
        return fused_attention_launch<Fp8Int4ElementCodec, G, false, true, false>(a, stream);
    case 3:
        return fused_attention_launch<Fp8Int4ElementCodec, G, true, true, false>(a, stream);
    case 4:
        return fused_attention_launch<Fp8Int4ElementCodec, G, false, false, true>(a, stream);
    case 5:
        return fused_attention_launch<Fp8Int4ElementCodec, G, true, false, true>(a, stream);
    case 6:
        return fused_attention_launch<Fp8Int4ElementCodec, G, false, true, true>(a, stream);
    default:
        return fused_attention_launch<Fp8Int4ElementCodec, G, true, true, true>(a, stream);
    }
}

// Fused causal dense prefill. One CTA owns one KV head and 32 query rows; each of
// its twelve waves owns one GQA query head and sixteen rows over all 256 features.
// Every 32-key block is decoded once into LDS: FP8 K becomes exact BF16, and signed
// INT4 times its FP16 scale becomes FP16 V/8 (codes staged as exact n/8 and scaled
// with one RNE product, as in the retired score-panel PV). S^T = K Q^T puts one
// query row per lane, which is the P operand layout of O += P V, so probabilities
// never leave registers. The FP32 Softmax origin is raised only when a block exceeds
// it by 2^8, keeping FP16 probabilities finite. Probabilities are packed to FP16
// once and the denominator sums exactly those FP16 values in FP32, so numerator and
// denominator use identical weights.
// The kernel is held to 240 VGPRs (six waves per SIMD) so two 12-wave CTAs share a
// WGP: one CTA's staging and barriers overlap the other's WMMA. At 255 VGPRs only one
// fits and staging is fully exposed (9-14% slower at T2048, 8K-64K context).
inline constexpr std::uint32_t kDensePrefillRows       = 32U;
inline constexpr std::uint32_t kDensePrefillKeys       = 32U;
inline constexpr std::uint32_t kDensePrefillWaves      = 12U;
inline constexpr std::uint32_t kDensePrefillThreads    = kDensePrefillWaves * 32U;
inline constexpr std::uint32_t kDensePrefillKeyPitch   = kHeadDim + 8U;
inline constexpr std::uint32_t kDensePrefillValuePitch = kDensePrefillKeys + 8U;
using DenseF16x8 = _Float16 __attribute__((ext_vector_type(8)));
using DenseF16x2 = _Float16 __attribute__((ext_vector_type(2)));

struct DensePrefillShared {
    std::uint16_t keys[kDensePrefillKeys * kDensePrefillKeyPitch]; // BF16 [key][feature]
    std::uint16_t values[kHeadDim * kDensePrefillValuePitch];      // FP16 V/8 [feature][key]
    std::uint32_t row_context[kDensePrefillRows];
    std::uint32_t invalid_rows, inactive_rows, maximum_context, page_fault_from;
    std::uint32_t full_context[6]; // per sixteen-row wave tile, rows with context only
    std::size_t table_offset;
};

// Split verification stages 16-key blocks into two LDS buffers: loaders convert block n + 1 into
// one while the compute waves consume block n from the other.
inline constexpr std::uint32_t kDenseVerifyKeys       = 16U;
inline constexpr std::uint32_t kDenseVerifyValuePitch = kDenseVerifyKeys + 8U;

struct DenseVerifyShared {
    std::uint16_t keys[2][kDenseVerifyKeys * kDensePrefillKeyPitch];
    std::uint16_t values[2][kHeadDim * kDenseVerifyValuePitch];
    std::uint32_t row_context[kDensePrefillRows];
    std::uint32_t invalid_rows, inactive_rows, maximum_context, page_fault_from;
    std::uint32_t full_context[6];
    std::uint32_t block_valid[2];               // the buffer's page is resident
    std::uint32_t plan_chunk_keys, plan_chunks; // the sequence's partition
    std::size_t table_offset;
};

// Split-context verification: one CTA per KV head and context chunk writes, for every row and
// query head, its unnormalized FP32 numerator [256], log2-domain Softmax origin and FP32
// denominator; dense_verify_merge_kernel combines the chunks.
inline constexpr std::uint32_t kDenseVerifyPartialStride    = kHeadDim + 2U;
inline constexpr std::uint32_t kDenseVerifyMaximumChunks    = 64U;
inline constexpr std::uint32_t kDenseVerifyMinimumChunkKeys = 256U;

// Largest packed-decode partition of a `context`-key envelope; it sizes the partial workspace,
// whose word after the last partial slot holds the chunk count the split chose for the merge.
__host__ __device__ constexpr std::uint32_t dense_verify_chunk_capacity(std::size_t context) {
    const std::size_t chunks =
        (context + kDenseVerifyMinimumChunkKeys - 1U) / kDenseVerifyMinimumChunkKeys;
    return static_cast<std::uint32_t>(
        chunks < kDenseVerifyMaximumChunks ? chunks : kDenseVerifyMaximumChunks);
}

__host__ __device__ constexpr std::size_t dense_verify_plan_word(std::size_t context,
                                                                 std::uint32_t rows) {
    return static_cast<std::size_t>(dense_verify_chunk_capacity(context)) * rows * 24U *
           kDenseVerifyPartialStride;
}

__device__ __forceinline__ void dense_prefill_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// One CTA's tile setup (thread 0): per-row causal context, invalid/inactive rows, the page-table
// row, and for every sixteen-lane wave tile the smallest context among its rows (blocks below it
// need no masking). Packed tiles hold (row, query head) pairs p -> row p / 6.
template <class Shared>
__device__ __forceinline__ void
dense_attention_setup(Shared& shared, const Fp8Int4KvAttentionArgs& a, std::uint32_t row_base,
                      std::uint32_t tile_rows, bool packed) {
    const std::int32_t active = a.active_query_rows != nullptr
                                    ? *a.active_query_rows
                                    : static_cast<std::int32_t>(a.query_rows);
    const bool invalid        = active < 0 || static_cast<std::uint32_t>(active) > a.query_rows;
    shared.invalid_rows       = 0U;
    shared.inactive_rows      = 0U;
    shared.maximum_context    = 0U;
    shared.table_offset       = 0U;
    shared.page_fault_from    = 0xffffffffU;
    for (std::uint32_t row = 0U; row < kDensePrefillRows; ++row) {
        shared.row_context[row] = 0U;
        if (row >= tile_rows) continue;
        if (invalid)
            shared.invalid_rows |= 1U << row;
        else if (row_base + row >= static_cast<std::uint32_t>(active))
            shared.inactive_rows |= 1U << row;
        else {
            const std::int32_t position = a.row_positions[row_base + row];
            if (position < 0 || static_cast<std::size_t>(position) >= a.context)
                shared.invalid_rows |= 1U << row;
            else {
                shared.row_context[row] = static_cast<std::uint32_t>(position) + 1U;
                shared.maximum_context  = max(shared.maximum_context, shared.row_context[row]);
            }
        }
    }
    if (!invalid && shared.maximum_context != 0U && a.page_table_row != nullptr) {
        const std::int32_t table_row = a.page_table_row[0];
        if (table_row < 0 || static_cast<std::uint32_t>(table_row) >= a.page_table_row_count) {
            shared.invalid_rows |=
                (tile_rows == 32U ? ~0U : (1U << tile_rows) - 1U) & ~shared.inactive_rows;
            for (std::uint32_t row = 0U; row < kDensePrefillRows; ++row)
                shared.row_context[row] = 0U;
            shared.maximum_context = 0U;
        } else
            shared.table_offset = static_cast<std::size_t>(table_row) * a.page_table_row_stride;
    }
    for (std::uint32_t wave_tile = 0U; wave_tile < 6U; ++wave_tile) {
        std::uint32_t full = 0xffffffffU;
        for (std::uint32_t lane_row = 0U; lane_row < 16U; ++lane_row) {
            const std::uint32_t row =
                packed ? (wave_tile * 16U + lane_row) / 6U : wave_tile * 16U + lane_row;
            if (row >= kDensePrefillRows) continue;
            const std::uint32_t context = shared.row_context[row];
            if (context != 0U) full = min(full, context);
        }
        shared.full_context[wave_tile] = full;
    }
}

// Loader items of a Keys-key block: 128 * Keys / 16 key items (one feature pair over sixteen keys)
// followed by 4 * Keys value items (two tokens by 32 features), numbered separately below. Loads
// and LDS staging are separate so a loader can hold raw words while a block is consumed.
struct DenseKeyItem {
    uint2 first[2], second[2];
};

struct DenseValueItem {
    std::uint32_t even[4], odd[4], scales[4];
};

__device__ __forceinline__ DenseKeyItem dense_load_key(const Fp8Int4KvAttentionArgs& a,
                                                       std::uint32_t loader, std::size_t page,
                                                       std::size_t physical_pages,
                                                       std::size_t in_page, std::uint32_t kv_head) {
    const std::uint32_t feature = (loader % 128U) * 2U, key_half = loader / 128U;
    const std::uint8_t* source = a.key_codes + in_page + key_half * 16U +
                                 kPageSize * (static_cast<std::size_t>(feature) +
                                              kHeadDim * (page + physical_pages * kv_head));
    DenseKeyItem item;
#pragma unroll
    for (std::uint32_t part = 0U; part < 2U; ++part) {
        item.first[part]  = *reinterpret_cast<const uint2*>(source + part * 8U);
        item.second[part] = *reinterpret_cast<const uint2*>(source + kPageSize + part * 8U);
    }
    return item;
}

// One feature pair over sixteen keys; E4M3FN values are exact in BF16.
__device__ __forceinline__ void dense_store_key(std::uint16_t* keys, std::uint32_t loader,
                                                const DenseKeyItem& item) {
    const std::uint32_t feature = (loader % 128U) * 2U, key_half = loader / 128U;
#pragma unroll
    for (std::uint32_t part = 0U; part < 2U; ++part) {
        const std::uint32_t low[2]  = {item.first[part].x, item.first[part].y};
        const std::uint32_t high[2] = {item.second[part].x, item.second[part].y};
#pragma unroll
        for (std::uint32_t word = 0U; word < 2U; ++word) {
#pragma unroll
            for (std::uint32_t pair = 0U; pair < 2U; ++pair) {
                const auto even =
                    pair == 0U ? __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(low[word]), false)
                               : __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(low[word]), true);
                const auto odd =
                    pair == 0U
                        ? __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(high[word]), false)
                        : __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(high[word]), true);
#pragma unroll
                for (std::uint32_t element = 0U; element < 2U; ++element) {
                    const std::uint32_t key =
                        key_half * 16U + part * 8U + word * 4U + pair * 2U + element;
                    *reinterpret_cast<std::uint32_t*>(
                        &keys[key * kDensePrefillKeyPitch + feature]) =
                        (__float_as_uint(even[element]) >> 16U) |
                        (__float_as_uint(odd[element]) & 0xffff0000U);
                }
            }
        }
    }
}

// Slots of the block at or past the tile's largest causal context (`written` keys of the block
// precede it) may be unwritten (the visible frontier can run past the appended keys): their
// scales are taken as zero, so a masked key's zero probability meets a zero value instead of
// whatever the unwritten codes and scale hold (an FP16 NaN scale poisons the row through 0 * NaN).
template <std::uint32_t Group, std::uint32_t Keys>
__device__ __forceinline__ DenseValueItem dense_load_value(const Fp8Int4KvAttentionArgs& a,
                                                           std::uint32_t item_index,
                                                           std::size_t page, std::size_t in_page,
                                                           std::uint32_t kv_head,
                                                           std::uint32_t written) {
    const std::uint32_t token_pair = item_index % (Keys / 2U), chunk = item_index / (Keys / 2U);
    const std::uint32_t even_mask = 2U * token_pair < written ? 0xFFFFU : 0U;
    const std::uint32_t odd_mask  = 2U * token_pair + 1U < written ? 0xFFFF0000U : 0U;
    const std::size_t row         = in_page + 2U * token_pair + kPageSize * (kv_head + 4U * page);
    DenseValueItem item;
#pragma unroll
    for (std::uint32_t word = 0U; word < 4U; ++word) {
        item.even[word] = *reinterpret_cast<const std::uint32_t*>(
            a.value_codes + chunk * 16U + word * 4U + (kHeadDim / 2U) * row);
        item.odd[word] = *reinterpret_cast<const std::uint32_t*>(
            a.value_codes + chunk * 16U + word * 4U + (kHeadDim / 2U) * (row + 1U));
        const std::uint32_t group = (chunk * 32U + word * 8U) / Group;
        item.scales[word] =
            static_cast<std::uint32_t>(a.value_scale_bits[group + (kHeadDim / Group) * row]) |
            (static_cast<std::uint32_t>(a.value_scale_bits[group + (kHeadDim / Group) * (row + 1U)])
             << 16U);
        item.scales[word] &= even_mask | odd_mask;
    }
    return item;
}

// Two tokens by 32 features. A signed nibble n is staged as FP16 1024+(n^8), mapped exactly to
// n/8 by one FMA, then rounded once by the FP16 scale product.
template <std::uint32_t Keys>
__device__ __forceinline__ void dense_store_value(std::uint16_t* values, std::uint32_t item_index,
                                                  const DenseValueItem& item) {
    constexpr std::uint32_t pitch  = Keys + 8U;
    const std::uint32_t token_pair = item_index % (Keys / 2U), chunk = item_index / (Keys / 2U);
#pragma unroll
    for (std::uint32_t word = 0U; word < 4U; ++word) {
        const std::uint32_t feature = chunk * 32U + word * 8U;
        const DenseF16x2 scale      = __builtin_bit_cast(DenseF16x2, item.scales[word]);
        const std::uint32_t biased0 = item.even[word] ^ 0x88888888U;
        const std::uint32_t biased1 = item.odd[word] ^ 0x88888888U;
        const std::uint32_t low0 = biased0 & 0x0f0f0f0fU, high0 = (biased0 >> 4U) & 0x0f0f0f0fU;
        const std::uint32_t low1 = biased1 & 0x0f0f0f0fU, high1 = (biased1 >> 4U) & 0x0f0f0f0fU;
#pragma unroll
        for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
            // Low half from the even token, high half from the odd token.
            const std::uint32_t select =
                byte | (0x0cU << 8U) | ((byte + 4U) << 16U) | (0x0cU << 24U);
            const std::uint32_t lower = __builtin_amdgcn_perm(low1, low0, select) | 0x64006400U;
            const std::uint32_t upper = __builtin_amdgcn_perm(high1, high0, select) | 0x64006400U;
            const DenseF16x2 lower_value =
                __builtin_elementwise_fma(__builtin_bit_cast(DenseF16x2, lower),
                                          DenseF16x2{0.125f16, 0.125f16},
                                          DenseF16x2{-129.0f16, -129.0f16}) *
                scale;
            const DenseF16x2 upper_value =
                __builtin_elementwise_fma(__builtin_bit_cast(DenseF16x2, upper),
                                          DenseF16x2{0.125f16, 0.125f16},
                                          DenseF16x2{-129.0f16, -129.0f16}) *
                scale;
            const std::uint32_t target = feature + 2U * byte;
            *reinterpret_cast<DenseF16x2*>(&values[target * pitch + 2U * token_pair]) = lower_value;
            *reinterpret_cast<DenseF16x2*>(&values[(target + 1U) * pitch + 2U * token_pair]) =
                upper_value;
        }
    }
}

// Loads this lane's represented query row. A nonfinite element makes every score of the row
// nonfinite; any nonfinite visible score poisons the row through a NaN FP32 denominator.
__device__ __forceinline__ bool dense_load_query(const Fp8Int4KvAttentionArgs& a, bool has_context,
                                                 std::size_t query_row, std::uint32_t query_head,
                                                 std::uint32_t half, I16x8 (&query)[16]) {
    bool nonfinite = false;
#pragma unroll
    for (std::uint32_t tile_k = 0U; tile_k < 16U; ++tile_k) {
        query[tile_k] = has_context ? *reinterpret_cast<const I16x8*>(
                                          a.query + (query_row * 24U + query_head) * kHeadDim +
                                          tile_k * 16U + half * 8U)
                                    : I16x8{};
#pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item)
            nonfinite |= (static_cast<std::uint16_t>(query[tile_k][item]) & 0x7f80U) == 0x7f80U;
    }
    return nonfinite;
}

// The dense routes' staging policy: raw loads, LDS conversion, the Q operand and the QK WMMA type.
struct Fp8Int4DenseCodec {
    using KeyItem   = DenseKeyItem;
    using ValueItem = DenseValueItem;

    __device__ static __forceinline__ KeyItem load_key(const Fp8Int4KvAttentionArgs& a,
                                                       std::uint32_t loader, std::size_t page,
                                                       std::size_t physical_pages,
                                                       std::size_t in_page, std::uint32_t kv_head) {
        return dense_load_key(a, loader, page, physical_pages, in_page, kv_head);
    }

    __device__ static __forceinline__ void store_key(std::uint16_t* keys, std::uint32_t loader,
                                                     const KeyItem& item) {
        dense_store_key(keys, loader, item);
    }

    template <std::uint32_t Group, std::uint32_t Keys>
    __device__ static __forceinline__ ValueItem load_value(const Fp8Int4KvAttentionArgs& a,
                                                           std::uint32_t item_index,
                                                           std::size_t page, std::size_t in_page,
                                                           std::uint32_t kv_head,
                                                           std::uint32_t written) {
        return dense_load_value<Group, Keys>(a, item_index, page, in_page, kv_head, written);
    }

    template <std::uint32_t Keys>
    __device__ static __forceinline__ void
    store_value(std::uint16_t* values, std::uint32_t item_index, const ValueItem& item) {
        dense_store_value<Keys>(values, item_index, item);
    }

    __device__ static __forceinline__ bool load_query(const Fp8Int4KvAttentionArgs& a,
                                                      bool has_context, std::size_t query_row,
                                                      std::uint32_t query_head, std::uint32_t half,
                                                      I16x8 (&query)[16]) {
        return dense_load_query(a, has_context, query_row, query_head, half, query);
    }

    __device__ static __forceinline__ F32x8 qk(I16x8 keys, I16x8 query, F32x8 scores) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(keys, query, scores);
    }
};

// NVFP4 staging: every E2M1*E4M3 key and V/8 is exact in FP16, and packed FP16 tables and products
// replace per-element FP32 conversion. Scales load once per item and are masked afterwards, so an
// unwritten slot's byte never reaches the staged value.
struct Nvfp4DenseCodec {
    struct KeyItem {
        uint2 codes[2], scales[2];
    };

    struct ValueItem {
        std::uint32_t even[4], odd[4], scales[2];
    };

    __device__ static __forceinline__ KeyItem load_key(const Fp8Int4KvAttentionArgs& a,
                                                       std::uint32_t loader, std::size_t page,
                                                       std::size_t physical_pages,
                                                       std::size_t in_page, std::uint32_t kv_head) {
        const std::uint32_t byte = loader % 128U, key_half = loader / 128U;
        const std::size_t head_page = page + physical_pages * kv_head;
        const std::size_t token     = in_page + key_half * 16U;
        const std::uint8_t* codes =
            a.key_codes + token + kPageSize * (static_cast<std::size_t>(byte) + 128U * head_page);
        const std::uint8_t* scales =
            a.key_scale_codes + token +
            kPageSize * (static_cast<std::size_t>(byte / 8U) + 16U * head_page);
        KeyItem item;
#pragma unroll
        for (std::uint32_t part = 0U; part < 2U; ++part) {
            item.codes[part]  = *reinterpret_cast<const uint2*>(codes + part * 8U);
            item.scales[part] = *reinterpret_cast<const uint2*>(scales + part * 8U);
        }
        return item;
    }

    __device__ static __forceinline__ void store_key(std::uint16_t* keys, std::uint32_t loader,
                                                     const KeyItem& item) {
        const std::uint32_t feature = (loader % 128U) * 2U, key_half = loader / 128U;
#pragma unroll
        for (std::uint32_t part = 0U; part < 2U; ++part) {
#pragma unroll
            for (std::uint32_t word = 0U; word < 2U; ++word) {
                const std::uint32_t codes  = word == 0U ? item.codes[part].x : item.codes[part].y;
                const std::uint32_t scales = word == 0U ? item.scales[part].x : item.scales[part].y;
                const std::uint32_t low    = e2m1_f16_high_bytes(codes & 0x0f0f0f0fU);
                const std::uint32_t high   = e2m1_f16_high_bytes((codes >> 4U) & 0x0f0f0f0fU);
                const auto scale0 =
                    __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(scales), false);
                const auto scale1 = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(scales), true);
#pragma unroll
                for (std::uint32_t token = 0U; token < 4U; ++token) {
                    const float scale = token < 2U ? scale0[token] : scale1[token - 2U];
                    const std::uint32_t select =
                        0x0cU | (token << 8U) | (0x0cU << 16U) | ((token + 4U) << 24U);
                    const DenseF16x2 pair =
                        __builtin_bit_cast(DenseF16x2, __builtin_amdgcn_perm(high, low, select)) *
                        __builtin_bit_cast(DenseF16x2, __builtin_amdgcn_cvt_pkrtz(scale, scale));
                    const std::uint32_t key = key_half * 16U + part * 8U + word * 4U + token;
                    *reinterpret_cast<DenseF16x2*>(&keys[key * kDensePrefillKeyPitch + feature]) =
                        pair;
                }
            }
        }
    }

    template <std::uint32_t Group, std::uint32_t Keys>
    __device__ static __forceinline__ ValueItem load_value(const Fp8Int4KvAttentionArgs& a,
                                                           std::uint32_t item_index,
                                                           std::size_t page, std::size_t in_page,
                                                           std::uint32_t kv_head,
                                                           std::uint32_t written) {
        static_assert(Group == 16U);
        const std::uint32_t token_pair = item_index % (Keys / 2U), chunk = item_index / (Keys / 2U);
        const std::size_t row = in_page + 2U * token_pair + kPageSize * (kv_head + 4U * page);
        ValueItem item;
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
            item.even[word] = *reinterpret_cast<const std::uint32_t*>(a.value_codes + row * 128U +
                                                                      chunk * 16U + word * 4U);
            item.odd[word]  = *reinterpret_cast<const std::uint32_t*>(
                a.value_codes + (row + 1U) * 128U + chunk * 16U + word * 4U);
        }
        // This chunk's two G16 scales of each token, as one packed load.
        std::uint32_t even =
            *reinterpret_cast<const std::uint16_t*>(a.value_scale_codes + row * 16U + chunk * 2U);
        std::uint32_t odd = *reinterpret_cast<const std::uint16_t*>(a.value_scale_codes +
                                                                    (row + 1U) * 16U + chunk * 2U);
        even &= 2U * token_pair < written ? 0xFFFFU : 0U;
        odd &= 2U * token_pair + 1U < written ? 0xFFFFU : 0U;
        item.scales[0] = (even & 0xFFU) | ((odd & 0xFFU) << 8U);
        item.scales[1] = (even >> 8U) | (odd & 0xFF00U);
        return item;
    }

    template <std::uint32_t Keys>
    __device__ static __forceinline__ void
    store_value(std::uint16_t* values, std::uint32_t item_index, const ValueItem& item) {
        constexpr std::uint32_t pitch  = Keys + 8U;
        const std::uint32_t token_pair = item_index % (Keys / 2U), chunk = item_index / (Keys / 2U);
        DenseF16x2 scales[2];
#pragma unroll
        for (std::uint32_t group = 0U; group < 2U; ++group) {
            const auto decoded =
                __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(item.scales[group]), false);
            scales[group] =
                __builtin_bit_cast(DenseF16x2, __builtin_amdgcn_cvt_pkrtz(decoded[0], decoded[1]));
        }
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
            const std::uint32_t low0 = e2m1_over8_f16_high_bytes(item.even[word] & 0x0f0f0f0fU);
            const std::uint32_t high0 =
                e2m1_over8_f16_high_bytes((item.even[word] >> 4U) & 0x0f0f0f0fU);
            const std::uint32_t low1 = e2m1_over8_f16_high_bytes(item.odd[word] & 0x0f0f0f0fU);
            const std::uint32_t high1 =
                e2m1_over8_f16_high_bytes((item.odd[word] >> 4U) & 0x0f0f0f0fU);
#pragma unroll
            for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
                // Low half: even token's high byte; high half: odd token's. Low bytes zero.
                const std::uint32_t select =
                    0x0cU | (byte << 8U) | (0x0cU << 16U) | ((byte + 4U) << 24U);
                const DenseF16x2 lower =
                    __builtin_bit_cast(DenseF16x2, __builtin_amdgcn_perm(low1, low0, select)) *
                    scales[word / 2U];
                const DenseF16x2 upper =
                    __builtin_bit_cast(DenseF16x2, __builtin_amdgcn_perm(high1, high0, select)) *
                    scales[word / 2U];
                const std::uint32_t target = chunk * 32U + word * 8U + byte * 2U;
                *reinterpret_cast<DenseF16x2*>(&values[target * pitch + token_pair * 2U]) = lower;
                *reinterpret_cast<DenseF16x2*>(&values[(target + 1U) * pitch + token_pair * 2U]) =
                    upper;
            }
        }
    }

    // Q validated on its represented bits (|Q| >= 2^16 includes every nonfinite value), then packed
    // to FP16 with round-toward-zero.
    __device__ static __forceinline__ bool load_query(const Fp8Int4KvAttentionArgs& a,
                                                      bool has_context, std::size_t query_row,
                                                      std::uint32_t query_head, std::uint32_t half,
                                                      I16x8 (&query)[16]) {
        bool out_of_range = false;
#pragma unroll
        for (std::uint32_t tile_k = 0U; tile_k < 16U; ++tile_k) {
            query[tile_k] = has_context ? *reinterpret_cast<const I16x8*>(
                                              a.query + (query_row * 24U + query_head) * kHeadDim +
                                              tile_k * 16U + half * 8U)
                                        : I16x8{};
#pragma unroll
            for (std::uint32_t item = 0U; item < 8U; ++item)
                out_of_range |=
                    (static_cast<std::uint16_t>(query[tile_k][item]) & 0x7f80U) >= 0x4780U;
#pragma unroll
            for (std::uint32_t item = 0U; item < 8U; item += 2U) {
                const auto packed = __builtin_amdgcn_cvt_pkrtz(
                    __uint_as_float(
                        static_cast<std::uint32_t>(static_cast<std::uint16_t>(query[tile_k][item]))
                        << 16U),
                    __uint_as_float(static_cast<std::uint32_t>(
                                        static_cast<std::uint16_t>(query[tile_k][item + 1U]))
                                    << 16U));
                const auto bits          = __builtin_bit_cast(std::uint32_t, packed);
                query[tile_k][item]      = static_cast<short>(bits);
                query[tile_k][item + 1U] = static_cast<short>(bits >> 16U);
            }
        }
        return out_of_range;
    }

    __device__ static __forceinline__ F32x8 qk(I16x8 keys, I16x8 query, F32x8 scores) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(keys, query, scores);
    }
};

// One staged block of 16 * KeyTiles keys of the online Softmax for this wave's sixteen lane rows:
// S^T = K Q^T (KeyTiles key tiles x 16 feature tiles of BF16 WMMA), masking, the log2-domain origin
// update, FP16 probabilities, and O += P V (16 feature tiles x KeyTiles key tiles of FP16 WMMA).
// RowContext is this lane's causal context, or a pointer to its LDS entry: the prefill kernel
// re-reads it on masked blocks only, so it does not occupy a VGPR across the block loop.
template <class Codec, std::uint32_t KeyTiles, class RowContext>
__device__ __forceinline__ void
dense_attention_block(const std::uint16_t* staged_keys, const std::uint16_t* staged_values,
                      std::uint32_t key_base, bool page_valid, std::uint32_t full_context,
                      RowContext row_context_source, std::uint32_t axis, std::uint32_t half,
                      float scale_log2, const I16x8 (&query)[16], F32x8 (&numerator)[16],
                      float& reference, float& denominator) {
    constexpr std::uint32_t value_pitch = 16U * KeyTiles + 8U;
    F32x8 scores[KeyTiles]              = {};
    if (page_valid) {
#pragma unroll
        for (std::uint32_t tile_k = 0U; tile_k < 16U; ++tile_k) {
#pragma unroll
            for (std::uint32_t key_tile = 0U; key_tile < KeyTiles; ++key_tile) {
                const I16x8 keys = *reinterpret_cast<const I16x8*>(
                    &staged_keys[(key_tile * 16U + axis) * kDensePrefillKeyPitch + tile_k * 16U +
                                 half * 8U]);
                scores[key_tile] = Codec::qk(keys, query[tile_k], scores[key_tile]);
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // Blocks wholly visible to every row with context skip masking. A nonfinite score of
    // such a block reaches the FP32 denominator and poisons its row there.
    float block_maximum = -INFINITY;
    if (page_valid && key_base + 16U * KeyTiles <= full_context) {
#pragma unroll
        for (std::uint32_t key_tile = 0U; key_tile < KeyTiles; ++key_tile)
#pragma unroll
            for (std::uint32_t item = 0U; item < 8U; ++item)
                block_maximum = fmaxf(block_maximum, scores[key_tile][item]);
    } else {
        std::uint32_t row_context;
        if constexpr (std::is_pointer_v<RowContext>) {
            RowContext entry = row_context_source;
            // NOLINTNEXTLINE(portability-no-assembler): register-only scheduling barrier
            asm volatile("" : "+v"(entry));
            row_context = *entry;
        } else {
            row_context = row_context_source;
        }
        bool poisoned = false;
#pragma unroll
        for (std::uint32_t key_tile = 0U; key_tile < KeyTiles; ++key_tile) {
#pragma unroll
            for (std::uint32_t item = 0U; item < 8U; ++item) {
                float score = scores[key_tile][item];
                if (key_base + key_tile * 16U + half * 8U + item < row_context) {
                    poisoned |= !page_valid || !(fabsf(score) < INFINITY);
                } else {
                    score = -INFINITY;
                }
                scores[key_tile][item] = score;
                block_maximum          = fmaxf(block_maximum, score);
            }
        }
        if (poisoned) denominator = __builtin_nanf("");
    }
    block_maximum *= scale_log2;
    block_maximum       = fmaxf(block_maximum, __shfl_xor(block_maximum, 16, 32));
    const bool raise    = block_maximum > reference + 8.0F;
    const float next    = raise ? block_maximum : reference;
    const float rescale = raise ? __builtin_amdgcn_exp2f(reference - next) : 1.0F;
    const float origin  = next == -INFINITY ? 0.0F : next;
    if (raise) denominator *= rescale;
    DenseF16x8 probabilities[KeyTiles];
#pragma unroll
    for (std::uint32_t key_tile = 0U; key_tile < KeyTiles; ++key_tile) {
#pragma unroll
        for (std::uint32_t item = 0U; item < 8U; item += 2U) {
            const auto packed = __builtin_amdgcn_cvt_pkrtz(
                __builtin_amdgcn_exp2f(__builtin_fmaf(scores[key_tile][item], scale_log2, -origin)),
                __builtin_amdgcn_exp2f(
                    __builtin_fmaf(scores[key_tile][item + 1U], scale_log2, -origin)));
            probabilities[key_tile][item]      = packed[0];
            probabilities[key_tile][item + 1U] = packed[1];
            denominator = __builtin_amdgcn_fdot2(__builtin_bit_cast(DenseF16x2, packed),
                                                 DenseF16x2{1.0f16, 1.0f16}, denominator, false);
        }
    }
    if (__builtin_amdgcn_ballot_w32(raise) != 0U) {
        float row_rescale[8];
#pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item)
            row_rescale[item] = __shfl(rescale, static_cast<int>(half * 8U + item), 32);
#pragma unroll
        for (std::uint32_t tile_f = 0U; tile_f < 16U; ++tile_f)
#pragma unroll
            for (std::uint32_t item = 0U; item < 8U; ++item)
                numerator[tile_f][item] *= row_rescale[item];
    }
    reference = next;
    __builtin_amdgcn_sched_barrier(0);
    if (page_valid) {
#pragma unroll
        for (std::uint32_t tile_f = 0U; tile_f < 16U; ++tile_f) {
#pragma unroll
            for (std::uint32_t key_tile = 0U; key_tile < KeyTiles; ++key_tile) {
                const DenseF16x8 values = *reinterpret_cast<const DenseF16x8*>(
                    &staged_values[(tile_f * 16U + axis) * value_pitch + key_tile * 16U +
                                   half * 8U]);
                numerator[tile_f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    probabilities[key_tile], values, numerator[tile_f]);
            }
        }
    }
}

// One CTA per KV head and 32-row tile. Split launches add a context-chunk axis: each CTA covers
// keys [chunk * chunk_keys, +chunk_keys) and writes the dense-verify partial layout (unnormalized
// FP32 numerator, log2-domain origin, FP32 denominator) for dense_verify_merge_kernel; a row whose
// causal context ends before the chunk keeps origin -inf and zero numerator and denominator.
template <class Codec, std::uint32_t Group, bool Split>
__device__ __forceinline__ void dense_prefill_body(const Fp8Int4KvAttentionArgs& a,
                                                   std::uint32_t chunk_keys, float* partial) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ DensePrefillShared shared;
    const std::uint32_t tid = threadIdx.x, lane = tid % 32U;
    // Wave-uniform values live in scalar registers; the 192 persistent Q/O VGPRs leave 48 of the
    // 240-VGPR (six waves per SIMD) budget.
    const std::uint32_t wave = __builtin_amdgcn_readfirstlane(tid / 32U);
    const std::uint32_t axis = lane % 16U, half = lane / 16U;
    const std::uint32_t head = wave % 6U, row_tile = wave / 6U;
    const std::uint32_t kv_head = blockIdx.x;
    // Launch the longest causal tiles first.
    const std::uint32_t tile         = gridDim.y - 1U - blockIdx.y;
    const std::uint32_t row_base     = tile * kDensePrefillRows;
    const std::uint32_t tile_rows    = min(kDensePrefillRows, a.query_rows - row_base);
    const std::uint32_t query_head   = kv_head * 6U + head;
    const std::size_t physical_pages = a.physical_tokens / kPageSize;

    if (tid == 0U) dense_attention_setup(shared, a, row_base, tile_rows, false);
    dense_prefill_barrier();
    const std::uint32_t maximum_context = __builtin_amdgcn_readfirstlane(shared.maximum_context);
    const std::uint32_t lane_row        = row_tile * 16U + axis;
    const std::uint32_t row_context     = shared.row_context[lane_row];
    const std::uint32_t full_context =
        __builtin_amdgcn_readfirstlane(shared.full_context[row_tile]);
    const std::size_t table_offset =
        static_cast<std::size_t>(
            __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(shared.table_offset))) |
        (static_cast<std::size_t>(
             __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(shared.table_offset >> 32U)))
         << 32U);

    // Chunk bounds are wave-uniform scalars beside the 192 persistent Q/O VGPRs.
    const std::uint32_t key_begin =
        Split ? __builtin_amdgcn_readfirstlane(blockIdx.z * chunk_keys) : 0U;
    const std::uint32_t key_end =
        Split ? __builtin_amdgcn_readfirstlane(min(maximum_context, key_begin + chunk_keys))
              : maximum_context;
    if constexpr (Split) {
        // A chunk past every row's causal context contributes nothing: origin -inf, zero
        // numerator and denominator (NaN for an invalid row, as its other chunks write).
        if (key_begin >= key_end) {
            if (lane_row < tile_rows) {
                float* const out =
                    partial +
                    ((static_cast<std::size_t>(blockIdx.z) * a.query_rows + row_base + lane_row) *
                         24U +
                     query_head) *
                        kDenseVerifyPartialStride;
                for (std::uint32_t feature = half * 128U; feature < half * 128U + 128U; ++feature)
                    out[feature] = 0.0F;
                if (half == 0U) {
                    out[kHeadDim] = -INFINITY;
                    out[kHeadDim + 1U] =
                        ((shared.invalid_rows >> lane_row) & 1U) != 0U ? __builtin_nanf("") : 0.0F;
                }
            }
            return;
        }
    }

    I16x8 query[16];
    const bool nonfinite =
        Codec::load_query(a, row_context != 0U, row_base + lane_row, query_head, half, query);
    F32x8 numerator[16];
#    pragma unroll
    for (std::uint32_t tile_f = 0U; tile_f < 16U; ++tile_f) numerator[tile_f] = F32x8{};
    float reference = -INFINITY; // log2-domain Softmax origin of this lane's query row
    // This lane's keys only; the half-wave pair is summed at the end. A nonfinite query is applied
    // after the loop, where its lane mask costs no VGPR.
    float denominator      = 0.0F;
    const float scale_log2 = a.attention_scale * 1.4426950408889634F;

    for (std::uint32_t key_base = key_begin; key_base < key_end; key_base += kDensePrefillKeys) {
        const std::size_t page = a.page_table[table_offset + key_base / kPageSize];
        const bool page_valid  = page < physical_pages;
        if (!page_valid) {
            if (tid == 0U) shared.page_fault_from = min(shared.page_fault_from, key_base);
        } else {
            const std::size_t in_page = key_base % kPageSize;
            // Recompute per-thread decode coordinates each block instead of holding them live
            // beside the 192 persistent Q/O registers.
            std::uint32_t loader = tid;
            asm volatile("" : "+v"(loader));
            if (loader < 256U)
                Codec::store_key(
                    shared.keys, loader,
                    Codec::load_key(a, loader, page, physical_pages, in_page, kv_head));
            else
                Codec::template store_value<kDensePrefillKeys>(
                    shared.values, loader - 256U,
                    Codec::template load_value<Group, kDensePrefillKeys>(
                        a, loader - 256U, page, in_page, kv_head, maximum_context - key_base));
        }
        dense_prefill_barrier();
        // Phase boundaries bound register live ranges at the 240-VGPR budget.
        __builtin_amdgcn_sched_barrier(0);
        dense_attention_block<Codec, 2U>(shared.keys, shared.values, key_base, page_valid,
                                         full_context, &shared.row_context[lane_row], axis, half,
                                         scale_log2, query, numerator, reference, denominator);
        dense_prefill_barrier();
    }

    if (nonfinite) denominator = __builtin_nanf("");
    const float row_denominator         = denominator + __shfl_xor(denominator, 16, 32);
    const std::uint32_t page_fault_from = shared.page_fault_from;
    if constexpr (Split) {
        // Lane coordinates are recomputed rather than held live across the key loop.
        std::uint32_t thread = threadIdx.x;
        asm volatile("" : "+v"(thread));
        const std::uint32_t out_axis = thread % 16U, out_half = (thread % 32U) / 16U;
        const std::uint32_t out_row = row_tile * 16U + out_axis;
        float* const head_partial =
            partial +
            ((static_cast<std::size_t>(blockIdx.z) * a.query_rows + row_base) * 24U + query_head) *
                kDenseVerifyPartialStride;
        const auto slot = [&](std::uint32_t row) {
            return head_partial + static_cast<std::size_t>(row) * 24U * kDenseVerifyPartialStride;
        };
#    pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t row = row_tile * 16U + out_half * 8U + item;
            if (row >= tile_rows) continue;
            float* numerator_out = slot(row) + out_axis;
#    pragma unroll
            for (std::uint32_t tile_f = 0U; tile_f < 16U; ++tile_f)
                numerator_out[tile_f * 16U] = numerator[tile_f][item];
        }
        // Lane `axis` of the first half owns row `lane_row`'s origin and denominator.
        if (out_half == 0U && out_row < tile_rows) {
            const bool invalid = ((shared.invalid_rows >> out_row) & 1U) != 0U ||
                                 shared.row_context[out_row] > page_fault_from;
            float* out         = slot(out_row);
            out[kHeadDim]      = reference;
            out[kHeadDim + 1U] = invalid ? __builtin_nanf("") : row_denominator;
        }
        return;
    }
#    pragma unroll
    for (std::uint32_t item = 0U; item < 8U; ++item) {
        const std::uint32_t row = row_tile * 16U + half * 8U + item;
        const float total       = __shfl(row_denominator, static_cast<int>(half * 8U + item), 32);
        if (row >= tile_rows) continue;
        const bool inactive = ((shared.inactive_rows >> row) & 1U) != 0U;
        const bool invalid  = ((shared.invalid_rows >> row) & 1U) != 0U || !(total == total) ||
                              shared.row_context[row] > page_fault_from;
        // Restore the exact V/8 staging factor in the final FP32 normalization.
        const float normalize = 8.0F / total;
        float* output = a.output +
                        (static_cast<std::size_t>(row_base + row) * 24U + query_head) * kHeadDim +
                        axis;
#    pragma unroll
        for (std::uint32_t tile_f = 0U; tile_f < 16U; ++tile_f)
            output[tile_f * 16U] = inactive  ? 0.0F
                                   : invalid ? __builtin_nanf("")
                                             : numerator[tile_f][item] * normalize;
    }
#else
    (void)a;
    (void)chunk_keys;
    (void)partial;
#endif
}

template <class Codec, std::uint32_t Group>
__global__ __launch_bounds__(kDensePrefillThreads)
    __attribute__((amdgpu_waves_per_eu(6))) void dense_prefill_kernel(Fp8Int4KvAttentionArgs a) {
    dense_prefill_body<Codec, Group, false>(a, 0U, nullptr);
}

template <class Codec, std::uint32_t Group>
__global__ __launch_bounds__(kDensePrefillThreads)
    __attribute__((amdgpu_waves_per_eu(6))) void dense_prefill_split_kernel(
        Fp8Int4KvAttentionArgs a, std::uint32_t chunk_keys, float* partial) {
    dense_prefill_body<Codec, Group, true>(a, chunk_keys, partial);
}

// Split-context packed decode (1..6 rows): one CTA per KV head and context chunk of
// kDenseVerifyThreads. The six query heads' (row, head) pairs are packed densely into sixteen-lane
// tiles (pair p = 16 * wave + lane row is row p / 6 of query head p % 6), so waves 0..2 compute
// every pair with the dense-prefill block arithmetic over 16-key blocks. Waves 3..5 are loaders
// that convert block n + 1 into the other LDS buffer while the compute waves consume block n, so
// staging overlaps compute and each block costs one barrier.
inline constexpr std::uint32_t kDenseVerifyComputeWaves = 3U;
inline constexpr std::uint32_t kDenseVerifyThreads      = 2U * kDenseVerifyComputeWaves * 32U;
inline constexpr std::uint32_t kDenseVerifyLoaders      = kDenseVerifyComputeWaves * 32U;
inline constexpr std::uint32_t kDenseVerifyKeyItems     = 128U * kDenseVerifyKeys / 16U;
static_assert(kDenseVerifyComputeWaves * 16U >= 8U * 6U);
// Two loader items per thread: a key item, then a key (wave 3) or value (waves 4..5) item.
static_assert(kDenseVerifyKeyItems + 4U * kDenseVerifyKeys == 2U * kDenseVerifyLoaders);
static_assert(kDenseVerifyKeyItems - kDenseVerifyLoaders == 32U);

// Context partition of one packed-decode sequence: `chunks` consecutive ranges of `chunk_keys`
// keys (a multiple of the 32-key tile) covering [0, live). Each partition holds at least
// kDenseVerifyMinimumChunkKeys keys, so short contexts never split finely.
struct DenseVerifyPlan {
    std::uint32_t chunk_keys = kDensePrefillKeys;
    std::uint32_t chunks     = 1U;
};

// Measured gfx1201 partition targets for one sequence of a `batch`-sequence launch (index
// batch - 1), by row class (1..4 and 5..8 query rows). A sequence whose live frontier is at most
// `short_live` uses `short_chunks`, otherwise `long_chunks`; a zero `long_chunks` keeps the
// frontier-envelope partition for every frontier of an envelope above `short_live` (the host
// launches that sequence with its fixed envelope plan). The batch grid fills the 64 compute units
// with about 4 * batch * chunks ~ 64 workgroups where that tiles evenly; wide rows at long
// frontiers with large batches keep the envelope partition, which the per-batch counts did not
// beat (docs/performance.md, 2026-11-08 packed attention partitions).
struct DenseVerifyTarget {
    std::uint8_t short_chunks;
    std::uint8_t long_chunks;
    std::uint16_t short_live;
};

inline constexpr DenseVerifyTarget kDenseVerifyTargets[2][kPackedDecodeMaximumSequences] = {
    {{16U, 16U, 8192U},
     {8U, 8U, 8192U},
     {5U, 5U, 8192U},
     {4U, 4U, 8192U},
     {3U, 3U, 8192U},
     {8U, 8U, 8192U},
     {2U, 16U, 8192U},
     {2U, 2U, 8192U},
     {5U, 16U, 8192U},
     {3U, 8U, 8192U},
     {6U, 12U, 8192U},
     {4U, 8U, 8192U},
     {5U, 16U, 8192U},
     {1U, 12U, 8192U},
     {1U, 12U, 8192U},
     {1U, 1U, 8192U}},
    {{16U, 16U, 8192U},
     {8U, 8U, 8192U},
     {5U, 5U, 8192U},
     {4U, 4U, 8192U},
     {3U, 3U, 8192U},
     {5U, 8U, 8192U},
     {2U, 0U, 8192U},
     {2U, 2U, 8192U},
     {5U, 0U, 8192U},
     {3U, 8U, 8192U},
     {4U, 0U, 8192U},
     {4U, 8U, 16384U},
     {5U, 0U, 8192U},
     {1U, 0U, 8192U},
     {1U, 0U, 8192U},
     {1U, 0U, 16384U}},
};

__host__ __device__ constexpr std::uint32_t dense_verify_target_chunks(DenseVerifyTarget target,
                                                                       std::uint32_t live) {
    return live <= target.short_live ? target.short_chunks : target.long_chunks;
}

__host__ __device__ constexpr DenseVerifyPlan
dense_verify_plan(std::uint32_t live, std::uint32_t target, std::uint32_t upper) {
    if (live == 0U) return {};
    std::uint32_t chunks =
        (live + kDenseVerifyMinimumChunkKeys - 1U) / kDenseVerifyMinimumChunkKeys;
    chunks                   = target < chunks ? target : chunks;
    chunks                   = upper < chunks ? upper : chunks;
    chunks                   = chunks == 0U ? 1U : chunks;
    const std::uint32_t keys = ((live + chunks - 1U) / chunks + kDensePrefillKeys - 1U) /
                               kDensePrefillKeys * kDensePrefillKeys;
    return {keys, (live + keys - 1U) / keys};
}

// Visible keys of a sequence's active rows: the largest valid causal position plus one, zero for
// an out-of-range active-row count (every row poisoned). An invalid table row only makes this an
// over-estimate, so the partition still covers every key the setup admits.
__device__ __forceinline__ std::uint32_t
dense_verify_live_context(const Fp8Int4KvAttentionArgs& a) {
    std::uint32_t rows = a.query_rows;
    if (a.active_query_rows != nullptr) {
        const std::int32_t active = *a.active_query_rows;
        if (active < 0 || static_cast<std::uint32_t>(active) > rows) return 0U;
        rows = static_cast<std::uint32_t>(active);
    }
    std::uint32_t live = 0U;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const std::int32_t position = a.row_positions[row];
        if (position >= 0 && static_cast<std::size_t>(position) < a.context)
            live = max(live, static_cast<std::uint32_t>(position) + 1U);
    }
    return __builtin_amdgcn_readfirstlane(live);
}

// Kernel arguments of one packed launch over the sequences of a compact decode batch. Grid y is
// the sequence: its own arguments (page table or table row, frontier, positions, output) and its
// own partial workspace (score_workspace) come from this kernarg array. `chunks` is the host
// upper bound of a sequence's context partition (the plan at its frontier envelope, which sizes
// the grid); the mid-row split reuses it as its exact merge count. Slots at and above the
// sequence count are never read.
struct DenseVerifySequences {
    Fp8Int4KvAttentionArgs sequence[kPackedDecodeMaximumSequences];
    std::uint32_t chunks[kPackedDecodeMaximumSequences];
    // Chunk size of the frontier-envelope partition, used where the target table keeps it.
    std::uint32_t envelope_chunk_keys[kPackedDecodeMaximumSequences];
    // kDenseVerifyTargets entry of the sequence's batch size and row class, passed as an argument
    // so no workgroup waits on a table load after its setup.
    DenseVerifyTarget target[kPackedDecodeMaximumSequences];
    // Bit s: sequence s keeps its envelope partition (envelope_chunk_keys, chunks) at every live
    // frontier; a launch whose sequences all keep it runs without live-frontier code.
    std::uint32_t envelope_plan_mask;
};

static_assert(sizeof(DenseVerifySequences) <= 4096U,
              "dense verify sequences must fit the kernarg segment");

__device__ __forceinline__ bool dense_verify_envelope_plan(const DenseVerifySequences& sequences,
                                                           std::uint32_t sequence) {
    return ((sequences.envelope_plan_mask >> sequence) & 1U) != 0U;
}

// Partition of sequence `sequence` at its live frontier. A zero long target makes the host keep
// the envelope plan, so any target reached here is nonzero.
__device__ __forceinline__ DenseVerifyPlan
dense_verify_sequence_plan(const DenseVerifySequences& sequences, std::uint32_t sequence) {
    if (dense_verify_envelope_plan(sequences, sequence))
        return {sequences.envelope_chunk_keys[sequence], sequences.chunks[sequence]};
    const std::uint32_t live = dense_verify_live_context(sequences.sequence[sequence]);
    return dense_verify_plan(live, dense_verify_target_chunks(sequences.target[sequence], live),
                             sequences.chunks[sequence]);
}

// NVFP4 loader waves: its keys cost more than its values, so the K+K wave above would be the
// critical path. Every loader stages one key item and five waves add the 64 value items to the
// first loaders; six waves give values their own wave, the faster shape while a launch leaves
// compute units free.
inline constexpr std::uint32_t kNvfp4MaximumGroupsForSixLoaderWaves = 128U;

// LivePlan false: every sequence keeps its envelope partition (chunk size and count from the
// kernel arguments), with no live-frontier code on the workgroup's critical path.
template <class Codec, std::uint32_t Group, std::uint32_t LoaderWaves, bool LivePlan>
__global__ __launch_bounds__((kDenseVerifyComputeWaves + LoaderWaves) *
                             32U) void dense_verify_kernel(DenseVerifySequences sequences) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    // Interleave sequences within each chunk so a long final sequence starts promptly.
    const std::uint32_t sequence = blockIdx.y, chunk = blockIdx.z;
    // The host bound covers every live partition.
    if (chunk >= sequences.chunks[sequence]) return;
    std::uint32_t chunk_keys        = sequences.envelope_chunk_keys[sequence];
    const Fp8Int4KvAttentionArgs& a = sequences.sequence[sequence];
    float* const partial            = a.score_workspace;
    __shared__ DenseVerifyShared shared;
    const std::uint32_t tid = threadIdx.x, lane = tid % 32U;
    const std::uint32_t wave = __builtin_amdgcn_readfirstlane(tid / 32U);
    const std::uint32_t axis = lane % 16U, half = lane / 16U;
    const std::uint32_t kv_head      = blockIdx.x;
    const std::uint32_t tile_rows    = a.query_rows;
    const std::size_t physical_pages = a.physical_tokens / kPageSize;

    if constexpr (LivePlan) {
        // The second wave forms the partition (frontier read and integer divisions) while
        // thread 0 runs the serial tile setup, so neither latency delays the other.
        if (tid == 0U) {
            dense_attention_setup(shared, a, 0U, tile_rows, true);
        } else if (tid == 32U) {
            const DenseVerifyPlan plan = dense_verify_sequence_plan(sequences, sequence);
            shared.plan_chunk_keys     = plan.chunk_keys;
            shared.plan_chunks         = plan.chunks;
            if (kv_head == 0U && chunk == 0U)
                reinterpret_cast<std::uint32_t*>(
                    partial)[dense_verify_plan_word(a.context, a.query_rows)] = plan.chunks;
        }
        dense_prefill_barrier();
        const auto chunks =
            static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(shared.plan_chunks));
        if (chunk >= chunks) return;
        chunk_keys =
            static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(shared.plan_chunk_keys));
    } else {
        if (tid == 0U) dense_attention_setup(shared, a, 0U, tile_rows, true);
        dense_prefill_barrier();
    }
    const std::uint32_t maximum_context = __builtin_amdgcn_readfirstlane(shared.maximum_context);
    const std::size_t table_offset =
        static_cast<std::size_t>(
            __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(shared.table_offset))) |
        (static_cast<std::size_t>(
             __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(shared.table_offset >> 32U)))
         << 32U);
    const std::uint32_t key_begin = chunk * chunk_keys;
    const std::uint32_t key_end   = min(maximum_context, key_begin + chunk_keys);
    const std::uint32_t blocks =
        key_end > key_begin ? (key_end - key_begin + kDenseVerifyKeys - 1U) / kDenseVerifyKeys : 0U;

    if (wave >= kDenseVerifyComputeWaves) {
        const std::uint32_t loader = tid - kDenseVerifyLoaders;
        if constexpr (LoaderWaves != 3U) {
            const bool has_key             = loader < kDenseVerifyKeyItems;
            const std::uint32_t value_item = loader >= kDenseVerifyKeyItems
                                                 ? loader - kDenseVerifyKeyItems
                                                 : (LoaderWaves == 5U ? loader + 32U : 64U);
            const bool has_value           = value_item < 4U * kDenseVerifyKeys;
            typename Codec::KeyItem key;
            typename Codec::ValueItem value;
            const auto stage = [&](std::uint32_t block) {
                const std::uint32_t buffer   = block % 2U;
                const std::uint32_t key_base = key_begin + block * kDenseVerifyKeys;
                const std::size_t page       = a.page_table[table_offset + key_base / kPageSize];
                const bool valid             = page < physical_pages;
                if (valid) {
                    const std::size_t in_page = key_base % kPageSize;
                    if (has_key)
                        key = Codec::load_key(a, loader, page, physical_pages, in_page, kv_head);
                    if (has_value)
                        value = Codec::template load_value<Group, kDenseVerifyKeys>(
                            a, value_item, page, in_page, kv_head, maximum_context - key_base);
                    if (has_key) Codec::store_key(shared.keys[buffer], loader, key);
                    if (has_value)
                        Codec::template store_value<kDenseVerifyKeys>(shared.values[buffer],
                                                                      value_item, value);
                } else if (loader == 0U) {
                    shared.page_fault_from = min(shared.page_fault_from, key_base);
                }
                if (loader == 0U) shared.block_valid[buffer] = valid ? 1U : 0U;
            };
            if (blocks != 0U) stage(0U);
            for (std::uint32_t block = 0U; block < blocks; ++block) {
                dense_prefill_barrier();
                if (block + 1U < blocks) stage(block + 1U);
            }
            return;
        }
        const bool second_is_key = loader + kDenseVerifyLoaders < kDenseVerifyKeyItems;
        DenseKeyItem keys[2];
        DenseValueItem value;
        // Stage block `block` into buffer block % 2: its loads, then the conversion and stores.
        const auto stage = [&](std::uint32_t block) {
            const std::uint32_t buffer   = block % 2U;
            const std::uint32_t key_base = key_begin + block * kDenseVerifyKeys;
            const std::size_t page       = a.page_table[table_offset + key_base / kPageSize];
            const bool valid             = page < physical_pages;
            if (valid) {
                const std::size_t in_page = key_base % kPageSize;
                keys[0] = dense_load_key(a, loader, page, physical_pages, in_page, kv_head);
                if (second_is_key)
                    keys[1] = dense_load_key(a, loader + kDenseVerifyLoaders, page, physical_pages,
                                             in_page, kv_head);
                else
                    value = dense_load_value<Group, kDenseVerifyKeys>(
                        a, loader + kDenseVerifyLoaders - kDenseVerifyKeyItems, page, in_page,
                        kv_head, maximum_context - key_base);
                dense_store_key(shared.keys[buffer], loader, keys[0]);
                if (second_is_key)
                    dense_store_key(shared.keys[buffer], loader + kDenseVerifyLoaders, keys[1]);
                else
                    dense_store_value<kDenseVerifyKeys>(
                        shared.values[buffer], loader + kDenseVerifyLoaders - kDenseVerifyKeyItems,
                        value);
            } else if (loader == 0U) {
                shared.page_fault_from = min(shared.page_fault_from, key_base);
            }
            if (loader == 0U) shared.block_valid[buffer] = valid ? 1U : 0U;
        };
        if (blocks != 0U) stage(0U);
        for (std::uint32_t block = 0U; block < blocks; ++block) {
            // Block `block` is staged; the compute waves are done with block - 1's buffer.
            dense_prefill_barrier();
            if (block + 1U < blocks) stage(block + 1U);
        }
        return;
    }

    // Compute waves: this lane's (row, query head) pair.
    const std::uint32_t pair         = wave * 16U + axis;
    const std::uint32_t lane_row     = pair / 6U;
    const std::uint32_t query_head   = kv_head * 6U + pair % 6U;
    const bool computes              = wave * 16U < 6U * tile_rows;
    const std::uint32_t row_context  = lane_row < tile_rows ? shared.row_context[lane_row] : 0U;
    const std::uint32_t full_context = __builtin_amdgcn_readfirstlane(shared.full_context[wave]);
    I16x8 query[16];
    const bool nonfinite =
        Codec::load_query(a, row_context != 0U, lane_row, query_head, half, query);
    F32x8 numerator[16];
#    pragma unroll
    for (std::uint32_t tile_f = 0U; tile_f < 16U; ++tile_f) numerator[tile_f] = F32x8{};
    float reference        = -INFINITY;
    float denominator      = nonfinite ? __builtin_nanf("") : 0.0F;
    const float scale_log2 = a.attention_scale * 1.4426950408889634F;
    for (std::uint32_t block = 0U; block < blocks; ++block) {
        dense_prefill_barrier();
        __builtin_amdgcn_sched_barrier(0);
        const std::uint32_t buffer = block % 2U;
        if (computes)
            dense_attention_block<Codec, 1U>(
                shared.keys[buffer], shared.values[buffer], key_begin + block * kDenseVerifyKeys,
                shared.block_valid[buffer] != 0U, full_context, row_context, axis, half, scale_log2,
                query, numerator, reference, denominator);
    }
    // The loaders' page faults are complete: their last stores preceded the final barrier.
    if (!computes) return;

    const float row_denominator         = denominator + __shfl_xor(denominator, 16, 32);
    const std::uint32_t page_fault_from = shared.page_fault_from;
    const auto slot                     = [&](std::uint32_t row, std::uint32_t pair_head) {
        return partial + ((static_cast<std::size_t>(chunk) * a.query_rows + row) * 24U +
                          kv_head * 6U + pair_head) *
                             kDenseVerifyPartialStride;
    };
#    pragma unroll
    for (std::uint32_t item = 0U; item < 8U; ++item) {
        const std::uint32_t out_pair = wave * 16U + half * 8U + item;
        const std::uint32_t row      = out_pair / 6U;
        if (row >= tile_rows) continue;
        float* numerator_out = slot(row, out_pair % 6U) + axis;
#    pragma unroll
        for (std::uint32_t tile_f = 0U; tile_f < 16U; ++tile_f)
            numerator_out[tile_f * 16U] = numerator[tile_f][item];
    }
    // Lane `axis` of the first half owns pair `pair`'s origin and denominator.
    if (half == 0U && lane_row < tile_rows) {
        const bool invalid = ((shared.invalid_rows >> lane_row) & 1U) != 0U ||
                             shared.row_context[lane_row] > page_fault_from;
        float* out         = slot(lane_row, pair % 6U);
        out[kHeadDim]      = reference;
        out[kHeadDim + 1U] = invalid ? __builtin_nanf("") : row_denominator;
    }
#else
    (void)sequences;
#endif
}

// Stable FP32 merge of the split partials: O = sum_c n_c 2^(m_c - M), L = sum_c l_c 2^(m_c - M),
// output = O * (8 / L), restoring the V/8 staging factor as the one-pass kernel does. Grid z is
// the sequence and grid y covers the batch's widest row panel; a narrower sequence's surplus rows
// exit. LivePlan reads the chunk count the split recorded in the workspace's plan word (clamped to
// the host bound); otherwise `chunks` is the host-fixed count.
template <bool LivePlan>
__global__
__launch_bounds__(kHeadDim) void dense_verify_merge_kernel(DenseVerifySequences sequences) {
    const Fp8Int4KvAttentionArgs& a = sequences.sequence[blockIdx.z];
    const std::uint32_t head = blockIdx.x, row = blockIdx.y, feature = threadIdx.x;
    if (row >= a.query_rows) return;
    const float* const partial = a.score_workspace;
    const auto slot            = [&](std::uint32_t chunk) {
        return partial + ((static_cast<std::size_t>(chunk) * a.query_rows + row) * 24U + head) *
                             kDenseVerifyPartialStride;
    };
    // Chunk origins and denominators up to the host bound (inside the sized workspace) load
    // together with the chunk count the live split recorded; chunks past that count are discarded.
    const std::uint32_t upper = sequences.chunks[blockIdx.z];
    float chunk_origin = -INFINITY, chunk_denominator = 0.0F;
    if (feature < upper) {
        chunk_origin      = slot(feature)[kHeadDim];
        chunk_denominator = slot(feature)[kHeadDim + 1U];
    }
    std::uint32_t chunks = upper;
    if constexpr (LivePlan) {
        const std::uint32_t recorded =
            __builtin_amdgcn_readfirstlane(reinterpret_cast<const std::uint32_t*>(
                partial)[dense_verify_plan_word(a.context, a.query_rows)]);
        chunks = recorded < upper ? recorded : upper;
    }
    float* output = a.output + (static_cast<std::size_t>(row) * 24U + head) * kHeadDim + feature;
    if (a.active_query_rows != nullptr) {
        const std::int32_t active = *a.active_query_rows;
        if (active < 0 || static_cast<std::uint32_t>(active) > a.query_rows) {
            *output = __builtin_nanf("");
            return;
        }
        if (row >= static_cast<std::uint32_t>(active)) {
            *output = 0.0F;
            return;
        }
    }
    // Chunk origins and denominators are read once, in parallel; each chunk's weight is formed
    // once. Every thread then accumulates in ascending chunk order exactly as a serial merge.
    __shared__ float origins[kDenseVerifyMaximumChunks], denominators[kDenseVerifyMaximumChunks];
    __shared__ float weights[kDenseVerifyMaximumChunks];
    if (feature < chunks) {
        origins[feature]      = chunk_origin;
        denominators[feature] = chunk_denominator;
    }
    __syncthreads();
    float origin = -INFINITY;
    for (std::uint32_t chunk = 0U; chunk < chunks; ++chunk) origin = fmaxf(origin, origins[chunk]);
    if (feature < chunks) {
        weights[feature] = origins[feature] == -INFINITY
                               ? 0.0F
                               : __builtin_amdgcn_exp2f(origins[feature] - origin);
    }
    __syncthreads();
    float numerator = 0.0F, denominator = 0.0F;
#pragma unroll 8
    for (std::uint32_t chunk = 0U; chunk < chunks; ++chunk) {
        const float weight = weights[chunk];
        // A NaN denominator (invalid row, poisoned score or page fault) poisons the row.
        // NOLINTNEXTLINE(misc-redundant-expression): self-inequality is the NaN test
        denominator = denominators[chunk] != denominators[chunk]
                          ? __builtin_nanf("")
                          : fmaf(denominators[chunk], weight, denominator);
        numerator   = fmaf(slot(chunk)[feature], weight, numerator);
    }
    *output = numerator * (8.0F / denominator);
}

__global__ void softmax_wmma_scores_in_place_kernel(float* scores, std::size_t context,
                                                    const std::int32_t* row_positions,
                                                    std::uint32_t row) {
    const std::uint32_t qh          = blockIdx.x;
    const std::uint32_t thread      = threadIdx.x;
    const std::uint32_t lane        = thread & 31U;
    const std::uint32_t wave        = thread >> 5U;
    const std::int32_t row_position = row_positions == nullptr ? -1 : row_positions[row];
    const bool invalid_row =
        row_positions != nullptr &&
        (row_position < 0 || static_cast<std::size_t>(row_position) >= context);
    const std::size_t row_context = row_positions == nullptr || invalid_row
                                        ? context
                                        : static_cast<std::size_t>(row_position) + 1U;
    float* head_scores            = scores + static_cast<std::size_t>(qh) * context;
    __shared__ float wave_values[kHeadDim / 32U];
    __shared__ std::uint32_t wave_invalid[kHeadDim / 32U];
    __shared__ float block_max;
    __shared__ float denominator;
    __shared__ std::uint32_t invalid;

    float local_max = -INFINITY;
    std::uint32_t local_invalid =
        invalid_row || row_context == 0 || row_context > context ? 1U : 0U;
    if (local_invalid == 0U) {
        for (std::size_t logical = thread; logical < row_context; logical += blockDim.x) {
            const float score = head_scores[logical];
            if (!isfinite(score))
                local_invalid = 1U;
            else
                local_max = fmaxf(local_max, score);
        }
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride, 32));
    }
    const std::uint32_t invalid_lanes = __ballot(local_invalid != 0U);
    if (lane == 0U) {
        wave_values[wave]  = local_max;
        wave_invalid[wave] = invalid_lanes != 0U ? 1U : 0U;
    }
    __syncthreads();
    if (thread == 0U) {
        float maximum     = wave_values[0];
        std::uint32_t bad = wave_invalid[0];
        for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
            maximum = fmaxf(maximum, wave_values[index]);
            bad |= wave_invalid[index];
        }
        block_max = maximum;
        invalid   = bad;
    }
    __syncthreads();
    if (invalid != 0U) {
        for (std::size_t logical = thread; logical < context; logical += blockDim.x) {
            head_scores[logical] = __builtin_nanf("");
        }
        return;
    }

    float local_sum = 0.0F;
    for (std::size_t logical = thread; logical < row_context; logical += blockDim.x) {
        local_sum += __expf(head_scores[logical] - block_max);
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_sum += __shfl_down(local_sum, stride, 32);
    }
    if (lane == 0U) wave_values[wave] = local_sum;
    __syncthreads();
    if (thread == 0U) {
        float sum = 0.0F;
        for (std::uint32_t index = 0U; index < kHeadDim / 32U; ++index) {
            sum += wave_values[index];
        }
        denominator = sum;
    }
    __syncthreads();
    for (std::size_t logical = thread; logical < row_context; logical += blockDim.x) {
        head_scores[logical] = __expf(head_scores[logical] - block_max) / denominator;
    }
    for (std::size_t logical = row_context + thread; logical < context; logical += blockDim.x) {
        head_scores[logical] = 0.0F;
    }
}

template <std::uint32_t G, bool V, bool S>
hipError_t softmax_pv_from_wmma_scores_launch(const Fp8Int4KvAttentionArgs& a, std::uint32_t row,
                                              hipStream_t stream) {
    hipLaunchKernelGGL(softmax_wmma_scores_in_place_kernel, dim3(a.query_heads), dim3(kHeadDim), 0,
                       stream, a.score_workspace, a.context, a.row_positions, row);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    const Int4KvPvArgs pv{
        .probabilities         = a.score_workspace,
        .value_codes           = a.value_codes,
        .value_scale_bits      = a.value_scale_bits,
        .page_table            = a.page_table,
        .page_table_pages      = a.page_table_pages,
        .page_table_row        = a.page_table_row,
        .page_table_row_stride = a.page_table_row_stride,
        .page_table_row_count  = a.page_table_row_count,
        .context               = a.context,
        .physical_tokens       = a.physical_tokens,
        .query_heads           = a.query_heads,
        .kv_heads              = a.kv_heads,
        .q_to_kv               = a.q_to_kv,
        .value_group           = G,
        .value_layout          = a.value_layout,
        .value_scale_layout    = a.value_scale_layout,
        .output = a.output + static_cast<std::size_t>(row) * a.query_heads * kHeadDim,
    };
    return pv_launch<G, V, S>(pv, stream);
}

template <std::uint32_t G>
hipError_t softmax_pv_from_wmma_scores_dispatch(const Fp8Int4KvAttentionArgs& a, std::uint32_t row,
                                                hipStream_t stream) {
    const unsigned layout = (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
                            (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U);
    switch (layout) {
    case 0:
        return softmax_pv_from_wmma_scores_launch<G, false, false>(a, row, stream);
    case 1:
        return softmax_pv_from_wmma_scores_launch<G, true, false>(a, row, stream);
    case 2:
        return softmax_pv_from_wmma_scores_launch<G, false, true>(a, row, stream);
    default:
        return softmax_pv_from_wmma_scores_launch<G, true, true>(a, row, stream);
    }
}

hipError_t fp8_int4_kv_attention_fused(const Fp8Int4KvAttentionArgs& a,
                                       hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.value_codes || !valid_codec_planes(a) || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens || a.physical_tokens % kPageSize ||
        !a.query_heads || !a.kv_heads || !a.q_to_kv || a.query_heads != a.kv_heads * a.q_to_kv ||
        !a.query_rows || !a.output || !std::isfinite(a.attention_scale) ||
        !(a.attention_scale > 0.0F) || !valid_plane_layout(a.key_layout) ||
        !valid_plane_layout(a.value_layout) || !valid_plane_layout(a.value_scale_layout) ||
        (a.ancestor_masks == nullptr) != (a.prefix_lengths == nullptr) ||
        (a.ancestor_masks != nullptr && a.row_positions == nullptr) ||
        a.prefix_length_stride > 1U ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize) {
        return hipErrorInvalidValue;
    }
    if (a.codec == KvCodec::Nvfp4) {
        return fused_attention_launch<Nvfp4ElementCodec, 16U, true, false, false>(a, stream);
    }
    return a.value_group == 16   ? fused_attention_dispatch<16>(a, stream)
           : a.value_group == 32 ? fused_attention_dispatch<32>(a, stream)
                                 : hipErrorInvalidValue;
}

hipError_t fp8_int4_kv_attention_dense_prefill(const Fp8Int4KvAttentionArgs& a,
                                               hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.value_codes || !valid_codec_planes(a) || !a.page_table ||
        !a.page_table_pages || !a.context ||
        a.context > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        !a.physical_tokens || a.physical_tokens % kPageSize || a.query_heads != 24U ||
        a.kv_heads != 4U || a.q_to_kv != 6U ||
        !use_dense_prefill_attention(a.query_rows, a.context) || !a.row_positions ||
        a.ancestor_masks || a.prefix_lengths || a.prefix_length_stride != 1U || a.score_workspace ||
        !a.output || !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        a.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
        a.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        a.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize ||
        (a.value_group != 16U && a.value_group != 32U)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid(4U, (a.query_rows + kDensePrefillRows - 1U) / kDensePrefillRows);
    if (a.codec == KvCodec::Nvfp4) {
        hipLaunchKernelGGL((dense_prefill_kernel<Nvfp4DenseCodec, 16U>), grid,
                           dim3(kDensePrefillThreads), 0, stream, a);
    } else if (a.value_group == 16U) {
        hipLaunchKernelGGL((dense_prefill_kernel<Fp8Int4DenseCodec, 16U>), grid,
                           dim3(kDensePrefillThreads), 0, stream, a);
    } else {
        hipLaunchKernelGGL((dense_prefill_kernel<Fp8Int4DenseCodec, 32U>), grid,
                           dim3(kDensePrefillThreads), 0, stream, a);
    }
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t
fp8_int4_kv_attention_dense_prefill_resources(std::uint32_t value_group, bool split,
                                              DensePrefillWmmaResources* resources) noexcept {
    if (resources == nullptr || (value_group != 16U && value_group != 32U)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    hipFuncAttributes attributes{};
    const void* kernel =
        split ? (value_group == 16U ? reinterpret_cast<const void*>(
                                          dense_prefill_split_kernel<Fp8Int4DenseCodec, 16U>)
                                    : reinterpret_cast<const void*>(
                                          dense_prefill_split_kernel<Fp8Int4DenseCodec, 32U>))
              : (value_group == 16U
                     ? reinterpret_cast<const void*>(dense_prefill_kernel<Fp8Int4DenseCodec, 16U>)
                     : reinterpret_cast<const void*>(dense_prefill_kernel<Fp8Int4DenseCodec, 32U>));
    const hipError_t error = hipFuncGetAttributes(&attributes, kernel);
    if (error != hipSuccess) return error;
    *resources =
        DensePrefillWmmaResources{attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
                                  static_cast<int>(attributes.localSizeBytes)};
    return hipSuccess;
#else
    return hipErrorNotSupported;
#endif
}

hipError_t fp8_int4_kv_attention_wmma(const Fp8Int4KvAttentionArgs& a,
                                      hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.value_codes || !a.value_scale_bits ||
        a.codec != KvCodec::Fp8Int4 || !a.page_table || !a.page_table_pages || !a.context ||
        !a.physical_tokens || a.physical_tokens % kPageSize || !a.query_heads || !a.kv_heads ||
        a.q_to_kv != 6U || a.query_heads != a.kv_heads * a.q_to_kv || !a.query_rows ||
        !a.score_workspace || !a.output || !std::isfinite(a.attention_scale) ||
        !(a.attention_scale > 0.0F) ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        !valid_plane_layout(a.key_layout) || !valid_plane_layout(a.value_layout) ||
        !valid_plane_layout(a.value_scale_layout) || a.ancestor_masks != nullptr ||
        a.prefix_lengths != nullptr || a.active_query_rows != nullptr ||
        a.prefix_length_stride > 1U ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize ||
        (a.value_group != 16U && a.value_group != 32U)) {
        return hipErrorInvalidValue;
    }
    for (std::uint32_t row = 0; row < a.query_rows; ++row) {
        const Fp8KvQkArgs qk{
            .query            = a.query + static_cast<std::size_t>(row) * a.query_heads * kHeadDim,
            .key_codes        = a.key_codes,
            .page_table       = a.page_table,
            .page_table_pages = a.page_table_pages,
            .page_table_row   = a.page_table_row,
            .page_table_row_stride = a.page_table_row_stride,
            .page_table_row_count  = a.page_table_row_count,
            .context               = a.context,
            .physical_tokens       = a.physical_tokens,
            .query_heads           = a.query_heads,
            .kv_heads              = a.kv_heads,
            .q_to_kv               = a.q_to_kv,
            .attention_scale       = a.attention_scale,
            .key_layout            = a.key_layout,
            .scores                = a.score_workspace,
        };
        hipError_t error = fp8_kv_qk_wmma(qk, stream);
        if (error != hipSuccess) { return error; }
        error = a.value_group == 16U ? softmax_pv_from_wmma_scores_dispatch<16>(a, row, stream)
                                     : softmax_pv_from_wmma_scores_dispatch<32>(a, row, stream);
        if (error != hipSuccess) { return error; }
    }
    return hipSuccess;
}

namespace {

// Context chunks per KV head are bounded by ceil(context / 256), capped at 64.
// The bound is monotonic in the context, so the workspace sized at a frontier envelope covers
// every smaller context.
std::uint32_t dense_verify_chunk_bound(std::size_t context) noexcept {
    return dense_verify_chunk_capacity(context);
}

std::uint32_t dense_verify_chunk_keys(std::size_t context, std::uint32_t rows) noexcept {
    // Match the captured frontier ranges: the 128K envelope needs more parallelism near 64K.
    const std::size_t maximum_chunks =
        rows >= 5U ? (context <= 65536U ? 16U : (context > 131072U ? 32U : 64U))
                   : kDenseVerifyMaximumChunks;
    const std::size_t bound =
        std::min<std::size_t>(dense_verify_chunk_bound(context), maximum_chunks);
    const std::size_t keys = (context + bound - 1U) / bound;
    return static_cast<std::uint32_t>((keys + kDensePrefillKeys - 1U) / kDensePrefillKeys *
                                      kDensePrefillKeys);
}

struct MidRowsSplit {
    std::uint32_t chunks     = 0U;
    std::uint32_t chunk_keys = 0U;
};

MidRowsSplit mid_rows_split(std::size_t context, std::uint32_t rows) noexcept {
    const std::size_t chunks = mid_rows_chunks(context, rows);
    const std::size_t keys   = (context + chunks - 1U) / chunks;
    const auto chunk_keys    = static_cast<std::uint32_t>((keys + kDensePrefillKeys - 1U) /
                                                          kDensePrefillKeys * kDensePrefillKeys);
    return {static_cast<std::uint32_t>((context + chunk_keys - 1U) / chunk_keys), chunk_keys};
}

} // namespace

std::size_t fp8_int4_kv_attention_mid_rows_workspace_bytes(std::size_t context,
                                                           std::uint32_t rows) noexcept {
    // Sized for every row count the route may take at this context, independent of whether the
    // wave model splits it, so a plan at a frontier envelope covers every smaller call.
    if (rows < kMidRowsMinimumRows || rows > kMidRowsMaximumRows || context < rows ||
        context > kDensePrefillMaximumContext)
        return 0U;
    return static_cast<std::size_t>(mid_rows_chunk_limit(context, rows)) * rows * 24U *
           kDenseVerifyPartialStride * sizeof(float);
}

std::size_t
fp8_int4_kv_attention_mid_rows_workspace_capacity_bytes(std::size_t context,
                                                        std::uint32_t max_rows) noexcept {
    const std::uint32_t rows = std::min(max_rows, kMidRowsMaximumRows);
    if (rows < kMidRowsMinimumRows || context < kMidRowsMinimumRows ||
        context > kDensePrefillMaximumContext)
        return 0U;
    // Each call needs rows x min(64, 2048 / rows, ceil(context / 256)) slots, which is at most
    // min(2048, max_rows x min(64, ceil(context / 256))).
    const std::size_t per_row =
        std::min<std::size_t>(kMidRowsMaximumChunks,
                              (context + kMidRowsMinimumChunkKeys - 1U) / kMidRowsMinimumChunkKeys);
    const std::size_t slots =
        std::min<std::size_t>(kMidRowsPartialRows, static_cast<std::size_t>(rows) * per_row);
    return slots * 24U * kDenseVerifyPartialStride * sizeof(float);
}

hipError_t fp8_int4_kv_attention_mid_rows(const Fp8Int4KvAttentionArgs& a,
                                          hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.value_codes || !valid_codec_planes(a) || !a.page_table ||
        !a.page_table_pages || !a.context ||
        a.context > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        !a.physical_tokens || a.physical_tokens % kPageSize || a.query_heads != 24U ||
        a.kv_heads != 4U || a.q_to_kv != 6U ||
        !use_mid_rows_attention(a.query_rows, a.context, false) || !a.row_positions ||
        a.ancestor_masks || a.prefix_lengths || a.prefix_length_stride > 1U ||
        a.active_query_rows || !a.score_workspace || !a.output ||
        !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        a.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
        a.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        a.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize ||
        (a.value_group != 16U && a.value_group != 32U)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const MidRowsSplit split = mid_rows_split(a.context, a.query_rows);
    const dim3 grid(4U, (a.query_rows + kDensePrefillRows - 1U) / kDensePrefillRows, split.chunks);
    if (a.codec == KvCodec::Nvfp4) {
        hipLaunchKernelGGL((dense_prefill_split_kernel<Nvfp4DenseCodec, 16U>), grid,
                           dim3(kDensePrefillThreads), 0, stream, a, split.chunk_keys,
                           a.score_workspace);
    } else if (a.value_group == 16U) {
        hipLaunchKernelGGL((dense_prefill_split_kernel<Fp8Int4DenseCodec, 16U>), grid,
                           dim3(kDensePrefillThreads), 0, stream, a, split.chunk_keys,
                           a.score_workspace);
    } else {
        hipLaunchKernelGGL((dense_prefill_split_kernel<Fp8Int4DenseCodec, 32U>), grid,
                           dim3(kDensePrefillThreads), 0, stream, a, split.chunk_keys,
                           a.score_workspace);
    }
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    DenseVerifySequences merge{};
    merge.sequence[0] = a;
    merge.chunks[0]   = split.chunks;
    hipLaunchKernelGGL(dense_verify_merge_kernel<false>, dim3(24U, a.query_rows), dim3(kHeadDim), 0,
                       stream, merge);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

std::size_t fp8_int4_kv_attention_packed_decode_workspace_bytes(std::size_t context,
                                                                std::uint32_t rows) noexcept {
    if (context == 0U || rows == 0U ||
        context > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        return 0U;
    return (dense_verify_plan_word(context, rows) + 1U) * sizeof(float);
}

hipError_t fp8_int4_kv_attention_packed_decode(std::span<const Fp8Int4KvAttentionArgs> sequences,
                                               hipStream_t stream) noexcept {
    if (sequences.empty() || sequences.size() > kPackedDecodeMaximumSequences) {
        return hipErrorInvalidValue;
    }
    DenseVerifySequences batch{};
    std::uint32_t grid_chunks = 0U, grid_rows = 0U;
    for (std::size_t sequence = 0U; sequence < sequences.size(); ++sequence) {
        const Fp8Int4KvAttentionArgs& a = sequences[sequence];
        if (!a.query || !a.key_codes || !a.value_codes || !valid_codec_planes(a) ||
            a.codec != sequences.front().codec || !a.page_table || !a.page_table_pages ||
            !a.context ||
            a.context > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            !a.physical_tokens || a.physical_tokens % kPageSize || a.query_heads != 24U ||
            a.kv_heads != 4U || a.q_to_kv != 6U || (a.query_rows < 1U || a.query_rows > 8U) ||
            !a.row_positions || !a.score_workspace || !a.output ||
            !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
            (a.page_table_row != nullptr &&
             (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
            a.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
            a.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
            a.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
            a.ancestor_masks != nullptr || a.prefix_lengths != nullptr ||
            a.prefix_length_stride > 1U ||
            a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize || a.value_group != 16U) {
            return hipErrorInvalidValue;
        }
        batch.sequence[sequence]            = a;
        const auto envelope                 = static_cast<std::uint32_t>(a.context);
        const std::uint32_t envelope_keys   = dense_verify_chunk_keys(a.context, a.query_rows);
        batch.envelope_chunk_keys[sequence] = envelope_keys;
        // Largest partition any live frontier in [1, envelope] can select: the grid extent.
        const std::uint32_t bound = dense_verify_chunk_bound(a.context);
        const auto reachable      = [&](std::uint32_t target, std::uint32_t live) {
            const std::uint32_t minimum =
                (live + kDenseVerifyMinimumChunkKeys - 1U) / kDenseVerifyMinimumChunkKeys;
            return std::max(1U, std::min({target, minimum, bound}));
        };
        const DenseVerifyTarget target =
            kDenseVerifyTargets[a.query_rows >= 5U ? 1U : 0U][sequences.size() - 1U];
        batch.target[sequence] = target;
        const bool has_long    = envelope > target.short_live;
        // The sequence keeps the envelope partition when (a) its envelope reaches a zero long
        // target (the envelope plan was not beaten there), or (b) its envelope chunks already
        // have the minimum size and the target never binds below the envelope, where a live plan
        // would only rebalance the same number of at most 256-key chunks.
        const std::uint32_t smallest =
            has_long ? std::min(target.short_chunks, target.long_chunks) : target.short_chunks;
        if ((has_long && target.long_chunks == 0U) ||
            (envelope_keys == kDenseVerifyMinimumChunkKeys &&
             smallest * kDenseVerifyMinimumChunkKeys >= envelope)) {
            batch.envelope_plan_mask |= 1U << sequence;
            batch.chunks[sequence] = (envelope + envelope_keys - 1U) / envelope_keys;
        } else {
            std::uint32_t upper = reachable(target.short_chunks,
                                            std::min<std::uint32_t>(envelope, target.short_live));
            if (has_long) upper = std::max(upper, reachable(target.long_chunks, envelope));
            batch.chunks[sequence] = std::min(upper, bound);
        }
        grid_chunks = std::max(grid_chunks, batch.chunks[sequence]);
        grid_rows   = std::max(grid_rows, a.query_rows);
    }
#if defined(__HIPCC__)
    const auto count  = static_cast<std::uint32_t>(sequences.size());
    const auto launch = [&]<bool LivePlan>() {
        if (sequences.front().codec == KvCodec::Nvfp4) {
            if (4U * count * grid_chunks <= kNvfp4MaximumGroupsForSixLoaderWaves) {
                hipLaunchKernelGGL((dense_verify_kernel<Nvfp4DenseCodec, 16U, 6U, LivePlan>),
                                   dim3(4U, count, grid_chunks),
                                   dim3((kDenseVerifyComputeWaves + 6U) * 32U), 0, stream, batch);
            } else {
                hipLaunchKernelGGL((dense_verify_kernel<Nvfp4DenseCodec, 16U, 5U, LivePlan>),
                                   dim3(4U, count, grid_chunks),
                                   dim3((kDenseVerifyComputeWaves + 5U) * 32U), 0, stream, batch);
            }
        } else {
            hipLaunchKernelGGL((dense_verify_kernel<Fp8Int4DenseCodec, 16U, 3U, LivePlan>),
                               dim3(4U, count, grid_chunks), dim3(kDenseVerifyThreads), 0, stream,
                               batch);
        }
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
        hipLaunchKernelGGL(dense_verify_merge_kernel<LivePlan>, dim3(24U, grid_rows, count),
                           dim3(kHeadDim), 0, stream, batch);
        return hipGetLastError();
    };
    // Every sequence on its envelope plan: the launch is the fixed-partition one.
    if (batch.envelope_plan_mask == (1U << count) - 1U) return launch.template operator()<false>();
    return launch.template operator()<true>();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t fp8_int4_kv_attention_packed_decode(const Fp8Int4KvAttentionArgs& args,
                                               hipStream_t stream) noexcept {
    return fp8_int4_kv_attention_packed_decode(std::span<const Fp8Int4KvAttentionArgs>(&args, 1U),
                                               stream);
}

namespace {

struct Split512Workspace {
    float* scores      = nullptr;
    float* maxima      = nullptr;
    float* sums        = nullptr;
    float* numerators  = nullptr;
    std::size_t chunks = 0;
};

[[nodiscard]] bool checked_product(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0U && b > static_cast<std::size_t>(-1) / a) return false;
    out = a * b;
    return true;
}

[[nodiscard]] std::size_t split512_workspace_bytes(std::uint32_t rows,
                                                   std::size_t context) noexcept {
    constexpr std::size_t kMaximumSplitContext = 512U * kSplitTokens;
    if ((rows != 1U && rows != 4U) || context == 0U || context > kMaximumSplitContext) return 0U;
    const std::size_t chunks    = (context + kSplitTokens - 1U) / kSplitTokens;
    std::size_t heads_context   = 0U;
    std::size_t records         = 0U;
    std::size_t score_elements  = 0U;
    std::size_t record_elements = 0U;
    std::size_t elements        = 0U;
    if (!checked_product(rows, 24U, heads_context) ||
        !checked_product(heads_context, context, score_elements) ||
        !checked_product(heads_context, chunks, records) ||
        !checked_product(records, kHeadDim + 2U, record_elements) ||
        score_elements > static_cast<std::size_t>(-1) - record_elements) {
        return 0U;
    }
    elements          = score_elements + record_elements;
    std::size_t bytes = 0U;
    return checked_product(elements, sizeof(float), bytes) ? bytes : 0U;
}

[[nodiscard]] Split512Workspace split512_workspace(void* storage, std::uint32_t rows,
                                                   std::size_t context) noexcept {
    auto* base                       = static_cast<float*>(storage);
    const std::size_t chunks         = (context + kSplitTokens - 1U) / kSplitTokens;
    const std::size_t score_elements = static_cast<std::size_t>(rows) * 24U * context;
    const std::size_t records        = static_cast<std::size_t>(rows) * 24U * chunks;
    return Split512Workspace{
        .scores     = base,
        .maxima     = base + score_elements,
        .sums       = base + score_elements + records,
        .numerators = base + score_elements + records * 2U,
        .chunks     = chunks,
    };
}

template <class Codec, bool DeviceActiveRows>
__global__ void
split512_qk_bf16_t4_kernel(const hip_bfloat16* query, const std::uint8_t* keys,
                           const std::uint8_t* key_scales, const std::uint32_t* pages,
                           const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
                           std::uint32_t page_table_row_count, std::size_t context,
                           std::size_t physical_tokens, const std::int32_t* active_query_rows,
                           float attention_scale, float* scores) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    constexpr std::uint32_t kRows             = 4U;
    constexpr std::uint32_t kQueriesPerKvHead = 6U;
    constexpr std::uint32_t kQueryHeads       = 24U;
    const std::uint32_t thread                = threadIdx.x;
    const std::uint32_t lane                  = thread & 31U;
    const std::uint32_t wave                  = thread >> 5U;
    const std::uint32_t axis                  = lane & 15U;
    const std::uint32_t half                  = lane >> 4U;
    const std::uint32_t kv_head               = blockIdx.y;
    const std::size_t logical_base            = static_cast<std::size_t>(blockIdx.x) * 16U;
    std::int32_t active                       = static_cast<std::int32_t>(kRows);
    if constexpr (DeviceActiveRows) {
        active = active_query_rows[0];
        if (active < 0 || active > static_cast<std::int32_t>(kRows)) return;
        if (active == 0) return;
    }
    if (page_table_row != nullptr) {
        const std::int32_t represented = page_table_row[0];
        if (represented < 0 || static_cast<std::uint32_t>(represented) >= page_table_row_count) {
            for (std::uint32_t index = thread; index < kRows * kQueriesPerKvHead * 16U;
                 index += blockDim.x) {
                const std::uint32_t row        = index / (kQueriesPerKvHead * 16U);
                const std::uint32_t remainder  = index % (kQueriesPerKvHead * 16U);
                const std::uint32_t local_head = remainder / 16U;
                const std::size_t logical      = logical_base + remainder % 16U;
                if (row < static_cast<std::uint32_t>(active) && logical < context) {
                    scores[(static_cast<std::size_t>(row) * kQueryHeads +
                            kv_head * kQueriesPerKvHead + local_head) *
                               context +
                           logical] = __builtin_nanf("");
                }
            }
            return;
        }
        pages += static_cast<std::size_t>(represented) * page_table_row_stride;
    }

    // Two adjacent token BF16 values share one dword, so paired lanes broadcast one LDS read.
    // The ninth dword is a fragment-row pad: the two wave halves consume features eight apart,
    // and a stride of nine shifts those rows by eight banks instead of aliasing every bank.
    __shared__ std::uint32_t key_tile[kHeadDim][9];
    __shared__ std::uint32_t invalid;
    if (thread == 0U) invalid = 0U;
    __syncthreads();
    for (std::uint32_t flat = thread; flat < kHeadDim * 8U; flat += blockDim.x) {
        const std::uint32_t feature    = flat / 8U;
        const std::uint32_t pair       = flat % 8U;
        std::uint32_t represented_pair = 0U;
#    pragma unroll
        for (std::uint32_t item = 0U; item < 2U; ++item) {
            const std::uint32_t token = pair * 2U + item;
            const std::size_t logical = logical_base + token;
            hip_bfloat16 decoded(0.0F);
            if (logical < context) {
                const std::size_t physical =
                    static_cast<std::size_t>(pages[logical / kPageSize]) * kPageSize +
                    logical % kPageSize;
                if (physical >= physical_tokens) {
                    atomicExch(&invalid, 1U);
                } else {
                    decoded = hip_bfloat16(Codec::template key<true>(
                        keys, key_scales, physical, kv_head, feature, 4U, physical_tokens));
                }
            }
            represented_pair |= static_cast<std::uint32_t>(static_cast<std::uint16_t>(decoded.data))
                                << (item * 16U);
        }
        key_tile[feature][pair] = represented_pair;
    }
    __syncthreads();
    if (invalid != 0U) {
        for (std::uint32_t index = thread; index < kRows * kQueriesPerKvHead * 16U;
             index += blockDim.x) {
            const std::uint32_t row        = index / (kQueriesPerKvHead * 16U);
            const std::uint32_t remainder  = index % (kQueriesPerKvHead * 16U);
            const std::uint32_t local_head = remainder / 16U;
            const std::size_t logical      = logical_base + remainder % 16U;
            if (row < static_cast<std::uint32_t>(active) && logical < context) {
                scores[(static_cast<std::size_t>(row) * kQueryHeads + kv_head * kQueriesPerKvHead +
                        local_head) *
                           context +
                       logical] = __builtin_nanf("");
            }
        }
        return;
    }

    F32x8 accum{};
#    pragma unroll
    for (std::uint32_t feature_tile = 0U; feature_tile < kHeadDim / 16U; ++feature_tile) {
        I16x8 q_fragment{};
        I16x8 k_fragment{};
        const std::uint32_t flat_query = wave * 16U + axis;
#    pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t feature = feature_tile * 16U + half * 8U + item;
            if (flat_query < kRows * kQueriesPerKvHead) {
                const std::uint32_t row        = flat_query / kQueriesPerKvHead;
                const std::uint32_t local_head = flat_query % kQueriesPerKvHead;
                if (row < static_cast<std::uint32_t>(active)) {
                    q_fragment[item] =
                        static_cast<short>(query[(static_cast<std::size_t>(row) * kQueryHeads +
                                                  kv_head * kQueriesPerKvHead + local_head) *
                                                     kHeadDim +
                                                 feature]
                                               .data);
                }
            }
            const std::uint32_t pair = key_tile[feature][axis >> 1U];
            k_fragment[item]         = static_cast<short>(pair >> ((axis & 1U) * 16U));
        }
        accum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(q_fragment, k_fragment, accum);
    }
#    pragma unroll
    for (std::uint32_t item = 0U; item < 8U; ++item) {
        const std::uint32_t flat_query = wave * 16U + half * 8U + item;
        const std::uint32_t row        = flat_query / kQueriesPerKvHead;
        const std::uint32_t local_head = flat_query % kQueriesPerKvHead;
        const std::size_t logical      = logical_base + axis;
        if (flat_query < kRows * kQueriesPerKvHead && row < static_cast<std::uint32_t>(active) &&
            logical < context) {
            const std::uint32_t query_head = kv_head * kQueriesPerKvHead + local_head;
            scores[(static_cast<std::size_t>(row) * kQueryHeads + query_head) * context + logical] =
                accum[item] * attention_scale;
        }
    }
#else
    (void)query;
    (void)keys;
    (void)key_scales;
    (void)pages;
    (void)page_table_row;
    (void)page_table_row_stride;
    (void)page_table_row_count;
    (void)context;
    (void)physical_tokens;
    (void)active_query_rows;
    (void)attention_scale;
    (void)scores;
#endif
}

template <class Codec, std::uint32_t Group, bool TreeMask, bool DeviceActiveRows>
__global__ void split512_softmax_pv_partial_kernel(
    const float* scores, const std::uint8_t* values, const typename Codec::ValueScale* scales,
    const std::uint32_t* pages, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count, std::size_t context,
    std::size_t physical_tokens, const std::int32_t* row_positions,
    const std::int32_t* ancestor_masks, const std::int32_t* prefix_lengths,
    std::uint32_t prefix_length_stride, const std::int32_t* active_query_rows,
    std::uint32_t query_rows, std::size_t chunks, float* maxima, float* sums, float* numerators) {
    const std::uint32_t thread   = threadIdx.x;
    const std::uint32_t lane     = thread & 31U;
    const std::uint32_t wave     = thread >> 5U;
    const std::uint32_t row      = blockIdx.z;
    const std::uint32_t qh       = blockIdx.y;
    const std::size_t chunk      = blockIdx.x;
    const std::size_t record     = (static_cast<std::size_t>(row) * 24U + qh) * chunks + chunk;
    const std::size_t chunk_base = chunk * kSplitTokens;
    __shared__ float weights[kSplitTokens];
    __shared__ unsigned long long physical[kSplitTokens];
    __shared__ float wave_values[kHeadDim / 32U];
    __shared__ std::uint32_t wave_bad[kHeadDim / 32U];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ std::uint32_t invalid;

    std::int32_t active = static_cast<std::int32_t>(query_rows);
    if constexpr (DeviceActiveRows) {
        active = active_query_rows[0];
        if (active < 0 || static_cast<std::uint32_t>(active) > query_rows) {
            if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
            return;
        }
        if (row >= static_cast<std::uint32_t>(active)) return;
    }
    if (page_table_row != nullptr) {
        const std::int32_t represented = page_table_row[0];
        if (represented < 0 || static_cast<std::uint32_t>(represented) >= page_table_row_count) {
            if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
            return;
        }
        pages += static_cast<std::size_t>(represented) * page_table_row_stride;
    }
    const std::int32_t represented_position =
        row_positions == nullptr ? static_cast<std::int32_t>(context - 1U) : row_positions[row];
    bool bad_metadata =
        represented_position < 0 || static_cast<std::size_t>(represented_position) >= context;
    const std::size_t row_context =
        bad_metadata ? 0U : static_cast<std::size_t>(represented_position) + 1U;
    std::size_t prefix = 0U;
    std::uint32_t mask = 0U;
    if constexpr (TreeMask) {
        const std::int32_t represented_prefix =
            prefix_lengths[static_cast<std::size_t>(row) * prefix_length_stride];
        mask = static_cast<std::uint32_t>(ancestor_masks[row]);
        if (represented_prefix < 0) bad_metadata = true;
        if (!bad_metadata) {
            prefix = static_cast<std::size_t>(represented_prefix);
            if (prefix > row_context || row_context - prefix >= 32U)
                bad_metadata = true;
            else if (prefix == 0U) {
                const std::size_t suffix = row_context;
                const std::uint32_t available =
                    suffix == 0U ? 0U : ((1U << static_cast<std::uint32_t>(suffix)) - 1U);
                if ((mask & available) == 0U) bad_metadata = true;
            }
        }
    }
    if (bad_metadata) {
        if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
        return;
    }

    float local_max         = -INFINITY;
    std::uint32_t local_bad = 0U;
    for (std::uint32_t offset = thread; offset < kSplitTokens; offset += blockDim.x) {
        const std::size_t logical = chunk_base + offset;
        bool visible              = logical < row_context;
        if constexpr (TreeMask) {
            if (visible && logical >= prefix) {
                visible = (mask & (1U << static_cast<std::uint32_t>(logical - prefix))) != 0U;
            }
        }
        float value = -INFINITY;
        if (visible) {
            value = scores[(static_cast<std::size_t>(row) * 24U + qh) * context + logical];
            if (!isfinite(value))
                local_bad = 1U;
            else
                local_max = fmaxf(local_max, value);
        }
        weights[offset] = value;
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride, 32));
    }
    const std::uint32_t bad_lanes = __ballot(local_bad != 0U);
    if (lane == 0U) {
        wave_values[wave] = local_max;
        wave_bad[wave]    = bad_lanes != 0U ? 1U : 0U;
    }
    __syncthreads();
    if (thread == 0U) {
        float value       = wave_values[0];
        std::uint32_t bad = wave_bad[0];
        for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
            value = fmaxf(value, wave_values[index]);
            bad |= wave_bad[index];
        }
        maximum = value;
        invalid = bad;
    }
    __syncthreads();
    if (invalid != 0U) {
        if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
        return;
    }

    float local_sum = 0.0F;
    for (std::uint32_t offset = thread; offset < kSplitTokens; offset += blockDim.x) {
        const float score       = weights[offset];
        const float probability = isfinite(score) ? __expf(score - maximum) : 0.0F;
        weights[offset]         = probability;
        local_sum += probability;
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_sum += __shfl_down(local_sum, stride, 32);
    }
    if (lane == 0U) wave_values[wave] = local_sum;
    __syncthreads();
    if (thread == 0U) {
        float value = 0.0F;
        for (std::uint32_t index = 0U; index < kHeadDim / 32U; ++index) {
            value += wave_values[index];
        }
        denominator = value;
        invalid     = 0U;
    }
    __syncthreads();

    for (std::uint32_t offset = thread; offset < kSplitTokens; offset += blockDim.x) {
        const std::size_t logical      = chunk_base + offset;
        unsigned long long represented = static_cast<unsigned long long>(-1);
        if (weights[offset] != 0.0F && logical < row_context) {
            const std::size_t translated =
                static_cast<std::size_t>(pages[logical / kPageSize]) * kPageSize +
                logical % kPageSize;
            if (translated >= physical_tokens)
                atomicExch(&invalid, 1U);
            else
                represented = static_cast<unsigned long long>(translated);
        }
        physical[offset] = represented;
    }
    __syncthreads();
    if (invalid != 0U) {
        if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
        return;
    }

    const std::uint32_t kvh = qh / 6U;
    float numerator         = 0.0F;
    for (std::uint32_t offset = 0U; offset < kSplitTokens; ++offset) {
        const float probability = weights[offset];
        if (probability == 0.0F) continue;
        const std::size_t translated = static_cast<std::size_t>(physical[offset]);
        const auto element           = Codec::template value<Group, false, false>(
            values, scales, translated, kvh, thread, 4U, physical_tokens);
        numerator = fmaf(probability, element.code * element.scale, numerator);
    }
    if (thread == 0U) {
        maxima[record] = denominator == 0.0F ? -INFINITY : maximum;
        sums[record]   = denominator;
    }
    numerators[record * kHeadDim + thread] = numerator;
}

template <bool DeviceActiveRows>
__global__ void split512_merge_kernel(const float* maxima, const float* sums,
                                      const float* numerators,
                                      const std::int32_t* active_query_rows,
                                      std::uint32_t query_rows, std::size_t chunks, float* output) {
    const std::uint32_t thread    = threadIdx.x;
    const std::uint32_t lane      = thread & 31U;
    const std::uint32_t wave      = thread >> 5U;
    const std::uint32_t qh        = blockIdx.x;
    const std::uint32_t row       = blockIdx.y;
    const std::size_t first       = (static_cast<std::size_t>(row) * 24U + qh) * chunks;
    const std::size_t output_base = (static_cast<std::size_t>(row) * 24U + qh) * kHeadDim;
    __shared__ float coefficients[512];
    __shared__ float wave_values[kHeadDim / 32U];
    __shared__ std::uint32_t wave_bad[kHeadDim / 32U];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ std::uint32_t invalid;

    if constexpr (DeviceActiveRows) {
        const std::int32_t active = active_query_rows[0];
        if (active < 0 || static_cast<std::uint32_t>(active) > query_rows) {
            output[output_base + thread] = __builtin_nanf("");
            return;
        }
        if (row >= static_cast<std::uint32_t>(active)) {
            output[output_base + thread] = 0.0F;
            return;
        }
    }

    float local_max         = -INFINITY;
    std::uint32_t local_bad = 0U;
    for (std::size_t chunk = thread; chunk < chunks; chunk += blockDim.x) {
        const float chunk_max = maxima[first + chunk];
        const float chunk_sum = sums[first + chunk];
        if (isnan(chunk_max) || !isfinite(chunk_sum) || chunk_sum < 0.0F)
            local_bad = 1U;
        else if (chunk_sum != 0.0F)
            local_max = fmaxf(local_max, chunk_max);
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride, 32));
    }
    const std::uint32_t bad_lanes = __ballot(local_bad != 0U);
    if (lane == 0U) {
        wave_values[wave] = local_max;
        wave_bad[wave]    = bad_lanes != 0U ? 1U : 0U;
    }
    __syncthreads();
    if (thread == 0U) {
        float value       = wave_values[0];
        std::uint32_t bad = wave_bad[0];
        for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
            value = fmaxf(value, wave_values[index]);
            bad |= wave_bad[index];
        }
        maximum = value;
        invalid = bad;
    }
    __syncthreads();
    if (invalid != 0U || !isfinite(maximum)) {
        output[output_base + thread] = __builtin_nanf("");
        return;
    }
    float local_sum = 0.0F;
    for (std::size_t chunk = thread; chunk < chunks; chunk += blockDim.x) {
        const float coefficient =
            sums[first + chunk] == 0.0F ? 0.0F : __expf(maxima[first + chunk] - maximum);
        coefficients[chunk] = coefficient;
        local_sum += sums[first + chunk] * coefficient;
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_sum += __shfl_down(local_sum, stride, 32);
    }
    if (lane == 0U) wave_values[wave] = local_sum;
    __syncthreads();
    if (thread == 0U) {
        float value = 0.0F;
        for (std::uint32_t index = 0U; index < kHeadDim / 32U; ++index) {
            value += wave_values[index];
        }
        denominator = value;
    }
    __syncthreads();
    if (!(denominator > 0.0F) || !isfinite(denominator)) {
        output[output_base + thread] = __builtin_nanf("");
        return;
    }
    float numerator = 0.0F;
    for (std::size_t chunk = 0U; chunk < chunks; ++chunk) {
        numerator =
            fmaf(coefficients[chunk], numerators[(first + chunk) * kHeadDim + thread], numerator);
    }
    output[output_base + thread] = numerator / denominator;
}

template <class Codec, std::uint32_t Group, bool TreeMask, bool DeviceActiveRows>
hipError_t split512_partial_merge(const Fp8Int4KvAttentionArgs& a,
                                  const Split512Workspace& workspace, hipStream_t stream) {
    hipLaunchKernelGGL(
        (split512_softmax_pv_partial_kernel<Codec, Group, TreeMask, DeviceActiveRows>),
        dim3(static_cast<unsigned>(workspace.chunks), a.query_heads, a.query_rows), dim3(kHeadDim),
        0, stream, workspace.scores, a.value_codes, Codec::value_scales(a), a.page_table,
        a.page_table_row, a.page_table_row_stride, a.page_table_row_count, a.context,
        a.physical_tokens, a.row_positions, a.ancestor_masks, a.prefix_lengths,
        a.prefix_length_stride, a.active_query_rows, a.query_rows, workspace.chunks,
        workspace.maxima, workspace.sums, workspace.numerators);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    hipLaunchKernelGGL((split512_merge_kernel<DeviceActiveRows>), dim3(a.query_heads, a.query_rows),
                       dim3(kHeadDim), 0, stream, workspace.maxima, workspace.sums,
                       workspace.numerators, a.active_query_rows, a.query_rows, workspace.chunks,
                       a.output);
    return hipGetLastError();
}

template <class Codec, std::uint32_t Group>
hipError_t split512_metadata_dispatch(const Fp8Int4KvAttentionArgs& a,
                                      const Split512Workspace& workspace, hipStream_t stream) {
    if (a.ancestor_masks != nullptr) {
        return a.active_query_rows != nullptr
                   ? split512_partial_merge<Codec, Group, true, true>(a, workspace, stream)
                   : split512_partial_merge<Codec, Group, true, false>(a, workspace, stream);
    }
    return a.active_query_rows != nullptr
               ? split512_partial_merge<Codec, Group, false, true>(a, workspace, stream)
               : split512_partial_merge<Codec, Group, false, false>(a, workspace, stream);
}

template <class Codec>
hipError_t split512_qk_launch(const Fp8Int4KvAttentionArgs& a, const Split512Workspace& workspace,
                              hipStream_t stream) {
    if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((split512_qk_bf16_t4_kernel<Codec, true>),
                           dim3(static_cast<unsigned>((a.context + 15U) / 16U), a.kv_heads),
                           dim3(64), 0, stream, a.query, a.key_codes, a.key_scale_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens,
                           a.active_query_rows, a.attention_scale, workspace.scores);
    } else {
        hipLaunchKernelGGL((split512_qk_bf16_t4_kernel<Codec, false>),
                           dim3(static_cast<unsigned>((a.context + 15U) / 16U), a.kv_heads),
                           dim3(64), 0, stream, a.query, a.key_codes, a.key_scale_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens, nullptr,
                           a.attention_scale, workspace.scores);
    }
    return hipGetLastError();
}

} // namespace

std::size_t fp8_int4_kv_attention_split512_workspace_capacity_bytes(std::uint32_t query_rows,
                                                                    std::size_t context) noexcept {
    return split512_workspace_bytes(query_rows, context);
}

hipError_t fp8_int4_kv_attention_split512(const Fp8Int4KvAttentionArgs& a, void* storage,
                                          std::size_t storage_bytes, hipStream_t stream) noexcept {
    const std::size_t required = split512_workspace_bytes(a.query_rows, a.context);
    if (!a.query || !a.key_codes || !a.value_codes || !valid_codec_planes(a) || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens || a.physical_tokens % kPageSize ||
        a.query_heads != 24U || a.kv_heads != 4U || a.q_to_kv != 6U ||
        (a.query_rows != 1U && a.query_rows != 4U) ||
        (a.query_rows == 1U && a.codec != KvCodec::Fp8Int4) || !a.output ||
        !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        a.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
        a.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        a.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        (a.value_group != 16U && a.value_group != 32U) ||
        (a.ancestor_masks == nullptr) != (a.prefix_lengths == nullptr) ||
        (a.ancestor_masks != nullptr && a.row_positions == nullptr) ||
        a.prefix_length_stride > 1U ||
        (a.query_rows == 1U && (a.ancestor_masks != nullptr || a.active_query_rows != nullptr)) ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize || required == 0U ||
        storage == nullptr || storage_bytes != required ||
        reinterpret_cast<std::uintptr_t>(storage) % alignof(float) != 0U) {
        return hipErrorInvalidValue;
    }
    const Split512Workspace workspace = split512_workspace(storage, a.query_rows, a.context);
    hipError_t error                  = hipSuccess;
    if (a.query_rows == 1U) {
        const Fp8KvQkArgs qk{
            .query                 = a.query,
            .key_codes             = a.key_codes,
            .page_table            = a.page_table,
            .page_table_pages      = a.page_table_pages,
            .page_table_row        = a.page_table_row,
            .page_table_row_stride = a.page_table_row_stride,
            .page_table_row_count  = a.page_table_row_count,
            .context               = a.context,
            .physical_tokens       = a.physical_tokens,
            .query_heads           = a.query_heads,
            .kv_heads              = a.kv_heads,
            .q_to_kv               = a.q_to_kv,
            .attention_scale       = a.attention_scale,
            .key_layout            = a.key_layout,
            .scores                = workspace.scores,
        };
        error = fp8_kv_qk_wmma(qk, stream);
    } else if (a.codec == KvCodec::Nvfp4) {
        error = split512_qk_launch<Nvfp4ElementCodec>(a, workspace, stream);
    } else {
        error = split512_qk_launch<Fp8Int4ElementCodec>(a, workspace, stream);
    }
    if (error != hipSuccess) return error;
    if (a.codec == KvCodec::Nvfp4) {
        return split512_metadata_dispatch<Nvfp4ElementCodec, 16U>(a, workspace, stream);
    }
    return a.value_group == 16U
               ? split512_metadata_dispatch<Fp8Int4ElementCodec, 16U>(a, workspace, stream)
               : split512_metadata_dispatch<Fp8Int4ElementCodec, 32U>(a, workspace, stream);
}

} // namespace ninfer::ops::r9700::kv
