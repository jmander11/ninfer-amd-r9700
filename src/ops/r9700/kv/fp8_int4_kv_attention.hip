#include "fp8_int4_kv_attention.h"

#include <hip/hip_fp8.h>

#include <cmath>
#include <limits>

namespace ninfer::ops::r9700::kv {
namespace {
inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kPageSize = 64;
inline constexpr std::uint32_t kTokensPerBlock = 128;
// gfx1201 exposes at most 65,535 workgroups in grid Y. Split only the row dimension; every launch
// still covers all query heads and preserves ordered-stream output independence.
inline constexpr std::uint32_t kMaximumGridY = 65535;
using I32x2 = int __attribute__((ext_vector_type(2)));
using I16x8 = short __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));
inline constexpr std::uint32_t kSplitTokens = 512;

[[nodiscard]] constexpr bool valid_plane_layout(Fp8Int4KvPlaneLayout layout) noexcept {
    return layout == Fp8Int4KvPlaneLayout::FeatureFastest ||
           layout == Fp8Int4KvPlaneLayout::TokenFastest;
}

template <bool TokenFastest>
__device__ std::size_t key_offset(std::size_t token, std::uint32_t head,
                                  std::uint32_t feature, std::uint32_t heads,
                                  std::size_t physical_tokens) {
    const std::size_t page = token / kPageSize, in_page = token % kPageSize;
    const std::size_t pages = physical_tokens / kPageSize;
    if constexpr (TokenFastest) {
        return in_page + kPageSize *
                             (static_cast<std::size_t>(feature) + kHeadDim * (page + pages * head));
    }
    return feature + kHeadDim *
                         (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
}

template <bool TokenFastest>
__global__ void qk_vector_kernel(const hip_bfloat16* query, const std::uint8_t* key_codes,
                                 const std::uint32_t* page_table,
                                 const std::int32_t* page_table_row,
                                 std::uint32_t page_table_row_stride,
                                 std::uint32_t page_table_row_count, std::size_t context,
                                 std::size_t physical_tokens, std::uint32_t kv_heads,
                                 std::uint32_t q_to_kv, float attention_scale, float* scores) {
    const std::uint32_t query_head = blockIdx.y;
    const std::size_t logical = static_cast<std::size_t>(blockIdx.x) * kTokensPerBlock + threadIdx.x;
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
    const std::size_t physical = static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
                                 logical % kPageSize;
    const std::uint32_t kv_head = query_head / q_to_kv;
    if (physical >= physical_tokens || kv_head >= kv_heads) {
        scores[static_cast<std::size_t>(query_head) * context + logical] = __builtin_nanf("");
        return;
    }
    float sum = 0.0F;
#pragma unroll
    for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
        const float q = static_cast<float>(query[static_cast<std::size_t>(query_head) * kHeadDim + feature]);
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
                               const std::uint32_t* page_table,
                               const std::int32_t* page_table_row,
                               std::uint32_t page_table_row_stride,
                               std::uint32_t page_table_row_count, std::size_t context,
                               std::size_t physical_tokens, std::uint32_t q_to_kv,
                               float attention_scale, float* scores) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t k_half = lane >> 4U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t kv_head = blockIdx.y;
    const std::size_t tile = static_cast<std::size_t>(blockIdx.x) * 16U;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            for (std::uint32_t j = 0; j < 8; ++j) {
                const std::uint32_t row = (lane >> 4U) * 8U + j;
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
            a_bytes[j] = query_head < gridDim.y * q_to_kv
                             ? __hip_cvt_float_to_fp8(
                                   static_cast<float>(query[static_cast<std::size_t>(query_head) *
                                                             kHeadDim + feature]),
                                   __HIP_SATFINITE, __HIP_E4M3)
                             : 0U;
            const std::size_t logical = tile + axis;
            if (logical < context) {
                const std::size_t physical = static_cast<std::size_t>(page_table[logical / kPageSize]) *
                                             kPageSize + logical % kPageSize;
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
        accum = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, accum);
    }
    for (std::uint32_t j = 0; j < 8; ++j) {
        const std::uint32_t row = (lane >> 4U) * 8U + j;
        const std::uint32_t query_head = kv_head * q_to_kv + row;
        const std::size_t logical = tile + axis;
        if (row < q_to_kv && logical < context) {
            const std::size_t physical =
                static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
                logical % kPageSize;
            scores[static_cast<std::size_t>(query_head) * context + logical] =
                physical < physical_tokens ? accum[j] * attention_scale : __builtin_nanf("");
        }
    }
#else
    (void)query; (void)key_codes; (void)page_table; (void)page_table_row;
    (void)page_table_row_stride; (void)page_table_row_count; (void)context; (void)physical_tokens;
    (void)q_to_kv; (void)attention_scale; (void)scores;
#endif
}

template <bool TokenFastest>
__global__ void qk_wmma_batched_w5_kernel(
    const hip_bfloat16* query, const std::uint8_t* key_codes,
    const std::uint32_t* page_table, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count,
    std::size_t context, std::size_t physical_tokens, std::uint32_t kv_heads,
    std::uint32_t q_to_kv,
    float attention_scale, float* scores) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t k_half = lane >> 4U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t row = blockIdx.y / kv_heads;
    const std::uint32_t kv_head = blockIdx.y % kv_heads;
    const std::size_t tile = static_cast<std::size_t>(blockIdx.x) * 16U;
    float* row_scores = scores + static_cast<std::size_t>(row) * kv_heads * q_to_kv * context;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            for (std::uint32_t j = 0; j < 8; ++j) {
                const std::uint32_t qrow = (lane >> 4U) * 8U + j;
                const std::size_t logical = tile + axis;
                if (qrow < q_to_kv && logical < context) {
                    const std::uint32_t query_head = kv_head * q_to_kv + qrow;
                    row_scores[static_cast<std::size_t>(query_head) * context + logical] =
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
            a_bytes[j] = query_head < kv_heads * q_to_kv
                             ? __hip_cvt_float_to_fp8(
                                   static_cast<float>(query[
                                       (static_cast<std::size_t>(row) * kv_heads * q_to_kv +
                                        query_head) * kHeadDim + feature]),
                                   __HIP_SATFINITE, __HIP_E4M3)
                             : 0U;
            const std::size_t logical = tile + axis;
            if (logical < context) {
                const std::size_t physical =
                    static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
                    logical % kPageSize;
                b_bytes[j] = physical < physical_tokens
                                 ? key_codes[key_offset<TokenFastest>(
                                       physical, kv_head, feature, kv_heads, physical_tokens)]
                                 : 0U;
            } else {
                b_bytes[j] = 0U;
            }
        }
        const I32x2 a = {pack_fp8x4(a_bytes), pack_fp8x4(a_bytes + 4)};
        const I32x2 b = {pack_fp8x4(b_bytes), pack_fp8x4(b_bytes + 4)};
        accum = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, accum);
    }
    for (std::uint32_t j = 0; j < 8; ++j) {
        const std::uint32_t qrow = (lane >> 4U) * 8U + j;
        const std::uint32_t query_head = kv_head * q_to_kv + qrow;
        const std::size_t logical = tile + axis;
        if (qrow < q_to_kv && logical < context) {
            const std::size_t physical =
                static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
                logical % kPageSize;
            row_scores[static_cast<std::size_t>(query_head) * context + logical] =
                physical < physical_tokens ? accum[j] * attention_scale : __builtin_nanf("");
        }
    }
#else
    (void)query; (void)key_codes; (void)page_table; (void)page_table_row;
    (void)page_table_row_stride; (void)page_table_row_count; (void)context;
    (void)physical_tokens; (void)kv_heads; (void)q_to_kv; (void)attention_scale; (void)scores;
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
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize) return hipErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>((a.context + kTokensPerBlock - 1U) / kTokensPerBlock),
                    a.query_heads);
    if (a.key_layout == Fp8Int4KvPlaneLayout::FeatureFastest) {
        hipLaunchKernelGGL((qk_vector_kernel<false>), grid, dim3(kTokensPerBlock), 0, stream,
                           a.query,a.key_codes,a.page_table,a.page_table_row,
                           a.page_table_row_stride,a.page_table_row_count,a.context,
                           a.physical_tokens,a.kv_heads,a.q_to_kv,a.attention_scale,a.scores);
    } else if (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest) {
        hipLaunchKernelGGL((qk_vector_kernel<true>), grid, dim3(kTokensPerBlock), 0, stream,
                           a.query,a.key_codes,a.page_table,a.page_table_row,
                           a.page_table_row_stride,a.page_table_row_count,a.context,
                           a.physical_tokens,a.kv_heads,a.q_to_kv,a.attention_scale,a.scores);
    } else return hipErrorInvalidValue;
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
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize) return hipErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>((a.context + 15U) / 16U), a.kv_heads);
    if (a.key_layout == Fp8Int4KvPlaneLayout::FeatureFastest) {
        hipLaunchKernelGGL((qk_wmma_kernel<false>), grid, dim3(32), 0, stream,
                           a.query,a.key_codes,a.page_table,a.page_table_row,
                           a.page_table_row_stride,a.page_table_row_count,a.context,
                           a.physical_tokens,a.q_to_kv,a.attention_scale,a.scores);
    } else if (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest) {
        hipLaunchKernelGGL((qk_wmma_kernel<true>), grid, dim3(32), 0, stream,
                           a.query,a.key_codes,a.page_table,a.page_table_row,
                           a.page_table_row_stride,a.page_table_row_count,a.context,
                           a.physical_tokens,a.q_to_kv,a.attention_scale,a.scores);
    } else return hipErrorInvalidValue;
    return hipGetLastError();
}

template <std::uint32_t Group, bool VToken, bool SToken>
__device__ std::size_t int4_offset(std::size_t token, std::uint32_t head, std::uint32_t feature,
                                   std::uint32_t heads, std::size_t physical_tokens,
                                   bool scale) {
    const std::size_t page = token / kPageSize, in_page = token % kPageSize;
    const std::size_t pages = physical_tokens / kPageSize;
    const std::uint32_t leading = scale ? kHeadDim / Group : kHeadDim / 2U;
    const std::uint32_t coordinate = scale ? feature / Group : feature / 2U;
    const bool token_fastest = scale ? SToken : VToken;
    return token_fastest
               ? in_page + kPageSize * (static_cast<std::size_t>(coordinate) + leading * (page + pages * head))
               : coordinate + leading * (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
}

template <std::uint32_t Group, bool VToken, bool SToken>
__global__ void pv_vector_kernel(const float* probabilities, const std::uint8_t* values,
                                 const std::uint16_t* scales, const std::uint32_t* pages,
                                 const std::int32_t* page_table_row,
                                 std::uint32_t page_table_row_stride,
                                 std::uint32_t page_table_row_count,
                                 std::size_t context, std::size_t physical_tokens,
                                 std::uint32_t q_to_kv, float* output) {
    const std::uint32_t qh = blockIdx.x, feature = threadIdx.x, kvh = qh / q_to_kv;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            output[static_cast<std::size_t>(qh) * kHeadDim + feature] = __builtin_nanf("");
            return;
        }
        pages += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    float sum = 0.0F;
    for (std::size_t logical = 0; logical < context; ++logical) {
        const std::size_t physical = static_cast<std::size_t>(pages[logical / kPageSize]) * kPageSize + logical % kPageSize;
        // The typed cache transaction validates page ownership before A3. Keep this raw Op
        // memory-safe as well: a corrupt external table is an invalid invocation, not an OOB V
        // or scale dereference. A NaN makes accidental unchecked use conspicuous to its caller.
        if (physical >= physical_tokens) {
            output[static_cast<std::size_t>(qh) * kHeadDim + feature] = __builtin_nanf("");
            return;
        }
        const std::uint8_t packed = values[int4_offset<Group,VToken,SToken>(physical,kvh,feature,gridDim.x / q_to_kv,physical_tokens,false)];
        int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
        code = code >= 8 ? code - 16 : code;
        const std::uint16_t bits = scales[int4_offset<Group,VToken,SToken>(physical,kvh,feature,gridDim.x / q_to_kv,physical_tokens,true)];
        sum = fmaf(probabilities[static_cast<std::size_t>(qh) * context + logical],
                   static_cast<float>(code) * __half2float(__ushort_as_half(bits)), sum);
    }
    output[static_cast<std::size_t>(qh) * kHeadDim + feature] = sum;
}

template <std::uint32_t Group, bool VToken, bool SToken>
__global__ void pv_vector_batched_w5_kernel(
    const float* probabilities, const std::uint8_t* values, const std::uint16_t* scales,
    const std::uint32_t* pages, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count,
    std::size_t context, std::size_t physical_tokens, std::uint32_t q_to_kv,
    float* output) {
    const std::uint32_t qh = blockIdx.x;
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t feature = threadIdx.x;
    const std::uint32_t kvh = qh / q_to_kv;
    const std::size_t heads = gridDim.x;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            output[(static_cast<std::size_t>(row) * heads + qh) * kHeadDim + feature] =
                __builtin_nanf("");
            return;
        }
        pages += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    const float* row_probabilities =
        probabilities + static_cast<std::size_t>(row) * heads * context;
    float sum = 0.0F;
    for (std::size_t logical = 0; logical < context; ++logical) {
        const std::size_t physical =
            static_cast<std::size_t>(pages[logical / kPageSize]) * kPageSize +
            logical % kPageSize;
        if (physical >= physical_tokens) {
            output[(static_cast<std::size_t>(row) * heads + qh) * kHeadDim + feature] =
                __builtin_nanf("");
            return;
        }
        const std::uint8_t packed = values[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, false)];
        int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
        code = code >= 8 ? code - 16 : code;
        const std::uint16_t bits = scales[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, true)];
        sum = fmaf(row_probabilities[static_cast<std::size_t>(qh) * context + logical],
                   static_cast<float>(code) * __half2float(__ushort_as_half(bits)), sum);
    }
    output[(static_cast<std::size_t>(row) * heads + qh) * kHeadDim + feature] = sum;
}

template <std::uint32_t G, bool V, bool S> hipError_t pv_launch(const Int4KvPvArgs& a, hipStream_t stream) {
    hipLaunchKernelGGL((pv_vector_kernel<G,V,S>), dim3(a.query_heads), dim3(kHeadDim), 0, stream,
                       a.probabilities,a.value_codes,a.value_scale_bits,a.page_table,
                       a.page_table_row,a.page_table_row_stride,a.page_table_row_count,a.context,
                       a.physical_tokens,a.q_to_kv,a.output);
    return hipGetLastError();
}
template <std::uint32_t G> hipError_t pv_dispatch(const Int4KvPvArgs& a, hipStream_t s) {
    const unsigned m=(a.value_layout==Fp8Int4KvPlaneLayout::TokenFastest?1U:0U)|(a.value_scale_layout==Fp8Int4KvPlaneLayout::TokenFastest?2U:0U);
    switch(m) { case 0:return pv_launch<G,false,false>(a,s); case 1:return pv_launch<G,true,false>(a,s); case 2:return pv_launch<G,false,true>(a,s); default:return pv_launch<G,true,true>(a,s); }
}

hipError_t int4_kv_pv_vector(const Int4KvPvArgs& a, hipStream_t stream) noexcept {
    if (!a.probabilities || !a.value_codes || !a.value_scale_bits || !a.page_table || !a.page_table_pages || !a.context || !a.physical_tokens || a.physical_tokens%kPageSize || !a.query_heads || !a.kv_heads || !a.q_to_kv || a.query_heads != a.kv_heads*a.q_to_kv || !a.output || !valid_plane_layout(a.value_layout) || !valid_plane_layout(a.value_scale_layout) || (a.page_table_row != nullptr && (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) || a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize) return hipErrorInvalidValue;
    return a.value_group==16 ? pv_dispatch<16>(a,stream) : a.value_group==32 ? pv_dispatch<32>(a,stream) : hipErrorInvalidValue;
}

struct FusedAttentionShared {
    float wave_reductions[kHeadDim / 32U];
    float running_max;
    float denominator;
    float score;
    float rescale;
    float probability_numerator;
};

template <std::uint32_t Group, bool KToken, bool VToken, bool SToken, bool TreeMask,
          bool DeviceActiveRows>
__device__ __forceinline__ void fused_attention_body(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint16_t* scales, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context,
    std::size_t physical_tokens, std::uint32_t q_to_kv,
    const std::int32_t* row_positions, const std::int32_t* ancestor_masks,
    const std::int32_t* prefix_lengths, std::uint32_t prefix_length_stride,
    const std::int32_t* active_query_rows, std::uint32_t query_rows,
    std::uint32_t row_offset, float attention_scale, float* output,
    FusedAttentionShared& shared) {
    const std::uint32_t row = row_offset + blockIdx.y;
    const std::uint32_t qh = blockIdx.x;
    const std::uint32_t feature = threadIdx.x;
    const std::size_t row_base =
        (static_cast<std::size_t>(row) * gridDim.x + qh) * kHeadDim;
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
    const std::uint32_t kvh = qh / q_to_kv;
    const std::int32_t row_position =
        row_positions == nullptr ? -1 : row_positions[row];
    const bool invalid_row = row_positions != nullptr &&
                             (row_position < 0 ||
                              static_cast<std::size_t>(row_position) >= context);
    const std::size_t row_context =
        row_positions == nullptr || invalid_row
            ? context
            : static_cast<std::size_t>(row_position) + 1U;
    std::size_t prefix = 0;
    std::size_t suffix = 0;
    std::uint32_t ancestor_mask = 0U;
    bool invalid_tree = false;
    if constexpr (TreeMask) {
        const std::int32_t represented_prefix =
            prefix_lengths[static_cast<std::size_t>(row) * prefix_length_stride];
        ancestor_mask = static_cast<std::uint32_t>(ancestor_masks[row]);
        invalid_tree = represented_prefix < 0;
        if (!invalid_tree) {
            prefix = static_cast<std::size_t>(represented_prefix);
            invalid_tree = prefix > row_context;
        }
        if (!invalid_tree) {
            suffix = row_context - prefix;
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
    const float q = static_cast<float>(query[row_base + feature]);
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
                (ancestor_mask &
                 (1U << static_cast<std::uint32_t>(logical - prefix))) == 0U) {
                continue;
            }
        }
        const std::size_t physical = static_cast<std::size_t>(pages[logical / kPageSize]) *
                                         kPageSize + logical % kPageSize;
        if (physical >= physical_tokens) {
            output[row_base + feature] = __builtin_nanf("");
            return;
        }
        float partial = q * __half2float(__half(__hip_cvt_fp8_to_halfraw(
            keys[key_offset<KToken>(physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens)],
            __HIP_E4M3)));
        for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
            partial += __shfl_down(partial, stride, 32);
        }
        if ((feature & 31U) == 0U) {
            shared.wave_reductions[feature / 32U] = partial;
        }
        __syncthreads();
        if (feature < kHeadDim / 32U) {
            partial = shared.wave_reductions[feature];
            for (std::uint32_t stride = 4U; stride != 0U; stride >>= 1U) {
                partial += __shfl_down(partial, stride, 8);
            }
        }
        if (feature == 0) {
            shared.score = partial * attention_scale;
            const float next_max = fmaxf(shared.running_max, shared.score);
            shared.rescale = __expf(shared.running_max - next_max);
            shared.probability_numerator = __expf(shared.score - next_max);
            shared.denominator = shared.denominator * shared.rescale +
                                 shared.probability_numerator;
            shared.running_max = next_max;
        }
        __syncthreads();
        const std::uint8_t packed = values[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, false)];
        int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
        code = code >= 8 ? code - 16 : code;
        const std::uint16_t scale_bits = scales[int4_offset<Group, VToken, SToken>(
            physical, kvh, feature, gridDim.x / q_to_kv, physical_tokens, true)];
        numerator = numerator * shared.rescale +
                    shared.probability_numerator * static_cast<float>(code) *
                        __half2float(__ushort_as_half(scale_bits));
        __syncthreads();
    }
    output[row_base + feature] = numerator / shared.denominator;
}

template <std::uint32_t Group, bool KToken, bool VToken, bool SToken, bool DeviceActiveRows>
__global__ void fused_attention_causal_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint16_t* scales, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context,
    std::size_t physical_tokens, std::uint32_t q_to_kv,
    const std::int32_t* row_positions, const std::int32_t* active_query_rows,
    std::uint32_t query_rows, std::uint32_t row_offset, float attention_scale, float* output) {
    __shared__ FusedAttentionShared shared;
    fused_attention_body<Group, KToken, VToken, SToken, false, DeviceActiveRows>(
        query, keys, values, scales, pages, page_table_row, page_table_row_stride,
        page_table_row_count, context, physical_tokens, q_to_kv, row_positions,
        nullptr, nullptr, 1U, active_query_rows, query_rows, row_offset, attention_scale, output,
        shared);
}

template <std::uint32_t Group, bool KToken, bool VToken, bool SToken, bool DeviceActiveRows>
__global__ void fused_attention_tree_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint16_t* scales, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context,
    std::size_t physical_tokens, std::uint32_t q_to_kv,
    const std::int32_t* row_positions, const std::int32_t* ancestor_masks,
    const std::int32_t* prefix_lengths, std::uint32_t prefix_length_stride,
    const std::int32_t* active_query_rows, std::uint32_t query_rows,
    std::uint32_t row_offset, float attention_scale, float* output) {
    __shared__ FusedAttentionShared shared;
    fused_attention_body<Group, KToken, VToken, SToken, true, DeviceActiveRows>(
        query, keys, values, scales, pages, page_table_row, page_table_row_stride,
        page_table_row_count, context, physical_tokens, q_to_kv, row_positions,
        ancestor_masks, prefix_lengths, prefix_length_stride, active_query_rows, query_rows,
        row_offset, attention_scale, output, shared);
}

template <std::uint32_t G, bool K, bool V, bool S>
hipError_t fused_attention_launch(const Fp8Int4KvAttentionArgs& a, hipStream_t stream) {
    for (std::uint32_t row_offset = 0; row_offset < a.query_rows;) {
        const std::uint32_t rows =
            a.query_rows - row_offset < kMaximumGridY
                ? a.query_rows - row_offset
                : kMaximumGridY;
        if (a.ancestor_masks != nullptr) {
            if (a.active_query_rows != nullptr) {
                hipLaunchKernelGGL((fused_attention_tree_kernel<G, K, V, S, true>),
                                   dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query,
                                   a.key_codes, a.value_codes, a.value_scale_bits, a.page_table,
                                   a.page_table_row, a.page_table_row_stride,
                                   a.page_table_row_count,
                                   a.context, a.physical_tokens, a.q_to_kv, a.row_positions,
                                   a.ancestor_masks, a.prefix_lengths, a.prefix_length_stride,
                                   a.active_query_rows, a.query_rows, row_offset,
                                   a.attention_scale, a.output);
            } else {
                hipLaunchKernelGGL((fused_attention_tree_kernel<G, K, V, S, false>),
                                   dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query,
                                   a.key_codes, a.value_codes, a.value_scale_bits, a.page_table,
                                   a.page_table_row, a.page_table_row_stride,
                                   a.page_table_row_count,
                                   a.context, a.physical_tokens, a.q_to_kv, a.row_positions,
                                   a.ancestor_masks, a.prefix_lengths, a.prefix_length_stride,
                                   nullptr, a.query_rows, row_offset, a.attention_scale, a.output);
            }
        } else {
            if (a.active_query_rows != nullptr) {
                hipLaunchKernelGGL((fused_attention_causal_kernel<G, K, V, S, true>),
                                   dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query,
                                   a.key_codes, a.value_codes, a.value_scale_bits, a.page_table,
                                   a.page_table_row, a.page_table_row_stride,
                                   a.page_table_row_count,
                                   a.context, a.physical_tokens, a.q_to_kv, a.row_positions,
                                   a.active_query_rows, a.query_rows, row_offset,
                                   a.attention_scale, a.output);
            } else {
                hipLaunchKernelGGL((fused_attention_causal_kernel<G, K, V, S, false>),
                                   dim3(a.query_heads, rows), dim3(kHeadDim), 0, stream, a.query,
                                   a.key_codes, a.value_codes, a.value_scale_bits, a.page_table,
                                   a.page_table_row, a.page_table_row_stride,
                                   a.page_table_row_count,
                                   a.context, a.physical_tokens, a.q_to_kv, a.row_positions,
                                   nullptr, a.query_rows, row_offset, a.attention_scale, a.output);
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
    const unsigned layout =
        (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
        (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U) |
        (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 4U : 0U);
    switch (layout) {
    case 0: return fused_attention_launch<G, false, false, false>(a, stream);
    case 1: return fused_attention_launch<G, true, false, false>(a, stream);
    case 2: return fused_attention_launch<G, false, true, false>(a, stream);
    case 3: return fused_attention_launch<G, true, true, false>(a, stream);
    case 4: return fused_attention_launch<G, false, false, true>(a, stream);
    case 5: return fused_attention_launch<G, true, false, true>(a, stream);
    case 6: return fused_attention_launch<G, false, true, true>(a, stream);
    default: return fused_attention_launch<G, true, true, true>(a, stream);
    }
}

inline constexpr std::uint32_t kDensePrefillQueryTile = 16U;
inline constexpr std::uint32_t kDensePrefillThreads = 256U;
inline constexpr std::uint32_t kDenseFullScoreQkThreads = 192U;

struct DenseFullScoreQkShared {
    hip_bfloat16 keys[kHeadDim][16];
    std::uint32_t row_context[kDensePrefillQueryTile];
    std::uint32_t invalid_rows;
    std::uint32_t inactive_rows;
    std::uint32_t maximum_context;
    std::size_t physical_page_base;
    std::size_t table_offset;
    std::uint32_t page_valid;
};

template <bool DeviceActiveRows>
__global__ __launch_bounds__(kDenseFullScoreQkThreads)
void dense_full_score_qk_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context, std::size_t physical_tokens,
    const std::int32_t* row_positions, const std::int32_t* active_query_rows,
    std::uint32_t total_query_rows, float attention_scale, float* scores) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ DenseFullScoreQkShared shared;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave = tid >> 5U;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t half = lane >> 4U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t kv_head = blockIdx.x;
    const std::uint32_t query_base = blockIdx.y * kDensePrefillQueryTile;
    const std::uint32_t tile_rows = min(kDensePrefillQueryTile,
                                        total_query_rows - query_base);
    if (tid == 0U) {
        const std::int32_t active = DeviceActiveRows ? active_query_rows[0]
                                                     : static_cast<std::int32_t>(total_query_rows);
        shared.inactive_rows = 0U;
        shared.maximum_context = 0U;
        shared.table_offset = 0U;
        shared.invalid_rows = 0U;
        const bool invalid_active = active < 0 ||
            static_cast<std::uint32_t>(active) > total_query_rows;
        for (std::uint32_t row = 0U; row < tile_rows; ++row) {
            const std::uint32_t bit = 1U << row;
            const std::uint32_t global_row = query_base + row;
            if (invalid_active) {
                shared.invalid_rows |= bit;
                shared.row_context[row] = 0U;
            } else if (global_row >= static_cast<std::uint32_t>(active)) {
                shared.inactive_rows |= bit;
                shared.row_context[row] = 0U;
            } else {
                const std::int32_t position = row_positions[global_row];
                if (position < 0 || static_cast<std::size_t>(position) >= context) {
                    shared.invalid_rows |= bit;
                    shared.row_context[row] = 0U;
                } else {
                    shared.row_context[row] = static_cast<std::uint32_t>(position) + 1U;
                    shared.maximum_context = max(shared.maximum_context,
                                                 shared.row_context[row]);
                }
            }
        }
        if (!invalid_active && shared.maximum_context != 0U && page_table_row != nullptr) {
            const std::int32_t table_row = page_table_row[0];
            if (table_row < 0 || static_cast<std::uint32_t>(table_row) >= page_table_row_count) {
                const std::uint32_t mask = (1U << tile_rows) - 1U;
                shared.invalid_rows |= mask & ~shared.inactive_rows;
                shared.maximum_context = 0U;
            } else {
                shared.table_offset = static_cast<std::size_t>(table_row) * page_table_row_stride;
            }
        }
    }
    __syncthreads();
    const std::uint32_t key_base = blockIdx.z * 16U;
    if (key_base >= shared.maximum_context) return;
    if (tid == 0U) {
        const std::size_t physical_page = pages[shared.table_offset + key_base / kPageSize];
        shared.page_valid = physical_page < physical_tokens / kPageSize ? 1U : 0U;
        shared.physical_page_base = physical_page * kPageSize + key_base % kPageSize;
    }
    __syncthreads();
    for (std::uint32_t flat = tid; flat < kHeadDim * 16U; flat += blockDim.x) {
        const std::uint32_t feature = flat / 16U;
        const std::uint32_t token = flat % 16U;
        hip_bfloat16 decoded(__builtin_nanf(""));
        if (key_base + token < shared.maximum_context && shared.page_valid != 0U) {
            const std::uint8_t code = keys[key_offset<true>(
                shared.physical_page_base + token, kv_head, feature, 4U, physical_tokens)];
            if ((code & 0x7fU) != 0x7fU) {
                decoded = hip_bfloat16(__half2float(
                    __half(__hip_cvt_fp8_to_halfraw(code, __HIP_E4M3))));
            }
        }
        shared.keys[feature][token] = decoded;
    }
    __syncthreads();
    F32x8 score{};
#pragma unroll
    for (std::uint32_t feature_tile = 0U; feature_tile < kHeadDim / 16U;
         ++feature_tile) {
        I16x8 q_fragment{};
        I16x8 k_fragment{};
#pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t feature = feature_tile * 16U + half * 8U + item;
            if (axis < tile_rows && (shared.inactive_rows & (1U << axis)) == 0U &&
                (shared.invalid_rows & (1U << axis)) == 0U) {
                q_fragment[item] = static_cast<short>(query[
                    (static_cast<std::size_t>(query_base + axis) * 24U +
                     kv_head * 6U + wave) * kHeadDim + feature].data);
            }
            k_fragment[item] = static_cast<short>(shared.keys[feature][axis].data);
        }
        score = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
            q_fragment, k_fragment, score);
    }
#pragma unroll
    for (std::uint32_t item = 0U; item < 8U; ++item) {
        const std::uint32_t row = half * 8U + item;
        if (row >= tile_rows) continue;
        const std::uint32_t logical = key_base + axis;
        if (logical >= shared.row_context[row]) continue;
        scores[(static_cast<std::size_t>(query_base + row) * 24U +
                kv_head * 6U + wave) * context + logical] = score[item] * attention_scale;
    }
#else
    (void)query; (void)keys; (void)pages; (void)page_table_row;
    (void)page_table_row_stride; (void)page_table_row_count; (void)context;
    (void)physical_tokens; (void)row_positions; (void)active_query_rows;
    (void)total_query_rows; (void)attention_scale; (void)scores;
#endif
}

struct DenseFullScoreQkBk32Shared {
    hip_bfloat16 keys[kHeadDim][32];
    std::uint32_t row_context[kDensePrefillQueryTile];
    std::uint32_t invalid_rows;
    std::uint32_t inactive_rows;
    std::uint32_t maximum_context;
    std::size_t physical_page_base;
    std::size_t table_offset;
    std::uint32_t page_valid;
};

// Selected production Bq16/Bk32 QK route. Since both the page and key tile are aligned and
// 32 divides the fixed 64-token page, a key tile is always contained by one physical page.
// Each wave retains two independent FP32 accumulators so each individual score sees exactly
// the production feature-tile accumulation order while its query fragment feeds both K tiles.
template <bool DeviceActiveRows>
__global__ __launch_bounds__(kDenseFullScoreQkThreads)
void dense_full_score_qk_bk32_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context, std::size_t physical_tokens,
    const std::int32_t* row_positions, const std::int32_t* active_query_rows,
    std::uint32_t total_query_rows, float attention_scale, float* scores) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ DenseFullScoreQkBk32Shared shared;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave = tid >> 5U;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t half = lane >> 4U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t kv_head = blockIdx.x;
    const std::uint32_t query_base = blockIdx.y * kDensePrefillQueryTile;
    const std::uint32_t tile_rows = min(kDensePrefillQueryTile,
                                        total_query_rows - query_base);
    if (tid == 0U) {
        const std::int32_t active = DeviceActiveRows ? active_query_rows[0]
                                                     : static_cast<std::int32_t>(total_query_rows);
        shared.inactive_rows = 0U;
        shared.maximum_context = 0U;
        shared.table_offset = 0U;
        shared.invalid_rows = 0U;
        const bool invalid_active = active < 0 ||
            static_cast<std::uint32_t>(active) > total_query_rows;
        for (std::uint32_t row = 0U; row < tile_rows; ++row) {
            const std::uint32_t bit = 1U << row;
            const std::uint32_t global_row = query_base + row;
            if (invalid_active) {
                shared.invalid_rows |= bit;
                shared.row_context[row] = 0U;
            } else if (global_row >= static_cast<std::uint32_t>(active)) {
                shared.inactive_rows |= bit;
                shared.row_context[row] = 0U;
            } else {
                const std::int32_t position = row_positions[global_row];
                if (position < 0 || static_cast<std::size_t>(position) >= context) {
                    shared.invalid_rows |= bit;
                    shared.row_context[row] = 0U;
                } else {
                    shared.row_context[row] = static_cast<std::uint32_t>(position) + 1U;
                    shared.maximum_context = max(shared.maximum_context,
                                                 shared.row_context[row]);
                }
            }
        }
        if (!invalid_active && shared.maximum_context != 0U && page_table_row != nullptr) {
            const std::int32_t table_row = page_table_row[0];
            if (table_row < 0 || static_cast<std::uint32_t>(table_row) >= page_table_row_count) {
                const std::uint32_t mask = (1U << tile_rows) - 1U;
                shared.invalid_rows |= mask & ~shared.inactive_rows;
                shared.maximum_context = 0U;
            } else {
                shared.table_offset = static_cast<std::size_t>(table_row) * page_table_row_stride;
            }
        }
    }
    __syncthreads();
    const std::uint32_t key_base = blockIdx.z * 32U;
    if (key_base >= shared.maximum_context) return;
    if (tid == 0U) {
        const std::size_t physical_page = pages[shared.table_offset + key_base / kPageSize];
        shared.page_valid = physical_page < physical_tokens / kPageSize ? 1U : 0U;
        shared.physical_page_base = physical_page * kPageSize + key_base % kPageSize;
    }
    __syncthreads();
    for (std::uint32_t flat = tid; flat < kHeadDim * 32U; flat += blockDim.x) {
        const std::uint32_t feature = flat / 32U;
        const std::uint32_t token = flat % 32U;
        hip_bfloat16 decoded(__builtin_nanf(""));
        if (key_base + token < shared.maximum_context && shared.page_valid != 0U) {
            const std::uint8_t code = keys[key_offset<true>(
                shared.physical_page_base + token, kv_head, feature, 4U, physical_tokens)];
            if ((code & 0x7fU) != 0x7fU) {
                decoded = hip_bfloat16(__half2float(
                    __half(__hip_cvt_fp8_to_halfraw(code, __HIP_E4M3))));
            }
        }
        shared.keys[feature][token] = decoded;
    }
    __syncthreads();
    F32x8 score0{};
    F32x8 score1{};
#pragma unroll
    for (std::uint32_t feature_tile = 0U; feature_tile < kHeadDim / 16U;
         ++feature_tile) {
        I16x8 q_fragment{};
        I16x8 k_fragment0{};
        I16x8 k_fragment1{};
#pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t feature = feature_tile * 16U + half * 8U + item;
            if (axis < tile_rows && (shared.inactive_rows & (1U << axis)) == 0U &&
                (shared.invalid_rows & (1U << axis)) == 0U) {
                q_fragment[item] = static_cast<short>(query[
                    (static_cast<std::size_t>(query_base + axis) * 24U +
                     kv_head * 6U + wave) * kHeadDim + feature].data);
            }
            k_fragment0[item] = static_cast<short>(shared.keys[feature][axis].data);
            k_fragment1[item] = static_cast<short>(shared.keys[feature][16U + axis].data);
        }
        score0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
            q_fragment, k_fragment0, score0);
        score1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
            q_fragment, k_fragment1, score1);
    }
#pragma unroll
    for (std::uint32_t item = 0U; item < 8U; ++item) {
        const std::uint32_t row = half * 8U + item;
        if (row >= tile_rows) continue;
        const std::size_t vector =
            static_cast<std::size_t>(query_base + row) * 24U + kv_head * 6U + wave;
        const std::uint32_t logical0 = key_base + axis;
        if (logical0 < shared.row_context[row]) {
            scores[vector * context + logical0] = score0[item] * attention_scale;
        }
        const std::uint32_t logical1 = key_base + 16U + axis;
        if (logical1 < shared.row_context[row]) {
            scores[vector * context + logical1] = score1[item] * attention_scale;
        }
    }
#else
    (void)query; (void)keys; (void)pages; (void)page_table_row;
    (void)page_table_row_stride; (void)page_table_row_count; (void)context;
    (void)physical_tokens; (void)row_positions; (void)active_query_rows;
    (void)total_query_rows; (void)attention_scale; (void)scores;
#endif
}

template <bool DeviceActiveRows>
__global__ __launch_bounds__(kDensePrefillThreads)
void dense_full_score_maximum_kernel(
    const float* scores, std::size_t context, const std::int32_t* row_positions,
    const std::int32_t* active_query_rows, std::uint32_t query_rows, float* maxima) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t wave = tid >> 5U;
    const std::uint32_t vector = blockIdx.x;
    const std::uint32_t row = vector / 24U;
    __shared__ float wave_max[8];
    __shared__ std::uint32_t wave_bad[8];
    const std::int32_t active = DeviceActiveRows ? active_query_rows[0]
                                                 : static_cast<std::int32_t>(query_rows);
    const bool inactive = active >= 0 && static_cast<std::uint32_t>(active) <= query_rows &&
                          row >= static_cast<std::uint32_t>(active);
    const std::int32_t position = inactive ? -1 : row_positions[row];
    const bool invalid = active < 0 || static_cast<std::uint32_t>(active) > query_rows ||
                         (!inactive && (position < 0 || static_cast<std::size_t>(position) >= context));
    const std::size_t visible = invalid || inactive ? 0U : static_cast<std::size_t>(position) + 1U;
    float maximum = -INFINITY;
    std::uint32_t bad = invalid ? 1U : 0U;
    for (std::size_t logical = tid; logical < visible; logical += blockDim.x) {
        const float value = scores[static_cast<std::size_t>(vector) * context + logical];
        if (!isfinite(value)) bad = 1U;
        else maximum = fmaxf(maximum, value);
    }
    for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U)
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
    const std::uint32_t bad_lanes = __ballot(bad != 0U);
    if (lane == 0U) {
        wave_max[wave] = maximum;
        wave_bad[wave] = bad_lanes != 0U ? 1U : 0U;
    }
    __syncthreads();
    if (tid == 0U) {
        float result = wave_max[0];
        std::uint32_t result_bad = wave_bad[0];
        for (std::uint32_t item = 1U; item < 8U; ++item) {
            result = fmaxf(result, wave_max[item]);
            result_bad |= wave_bad[item];
        }
        maxima[vector] = inactive ? 0.0F : result_bad != 0U ? __builtin_nanf("") : result;
    }
}

template <std::uint32_t Group>
struct DenseFullScorePvShared {
    float probability[6][kDensePrefillQueryTile][16];
    float denominator[6][kDensePrefillQueryTile];
    std::uint8_t packed_values[16][kHeadDim / 2U];
    std::uint16_t value_scales[16][kHeadDim / Group];
    std::uint32_t row_context[kDensePrefillQueryTile];
    std::uint32_t invalid_rows[6];
    std::uint32_t inactive_rows;
    std::uint32_t maximum_context;
    std::size_t physical_base;
    std::size_t table_offset;
    std::uint32_t page_valid;
};

template <std::uint32_t Group, bool DeviceActiveRows>
__global__ __launch_bounds__(kDensePrefillThreads)
void dense_full_score_pv_kernel(
    const float* scores, const float* maxima, const std::uint8_t* values,
    const std::uint16_t* scales, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context, std::size_t physical_tokens,
    const std::int32_t* row_positions, const std::int32_t* active_query_rows,
    std::uint32_t total_query_rows, float* output) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ DenseFullScorePvShared<Group> shared;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t kv_head = blockIdx.x;
    const std::uint32_t query_base = blockIdx.y * kDensePrefillQueryTile;
    const std::uint32_t tile_rows = min(kDensePrefillQueryTile,
                                        total_query_rows - query_base);
    if (tid == 0U) {
        const std::int32_t active = DeviceActiveRows ? active_query_rows[0]
                                                     : static_cast<std::int32_t>(total_query_rows);
        shared.inactive_rows = 0U;
        shared.maximum_context = 0U;
        shared.table_offset = 0U;
        for (std::uint32_t head = 0U; head < 6U; ++head) shared.invalid_rows[head] = 0U;
        const bool invalid_active = active < 0 ||
            static_cast<std::uint32_t>(active) > total_query_rows;
        for (std::uint32_t row = 0U; row < tile_rows; ++row) {
            const std::uint32_t bit = 1U << row;
            const std::uint32_t global_row = query_base + row;
            if (invalid_active) {
                for (std::uint32_t head = 0U; head < 6U; ++head)
                    shared.invalid_rows[head] |= bit;
                shared.row_context[row] = 0U;
            } else if (global_row >= static_cast<std::uint32_t>(active)) {
                shared.inactive_rows |= bit;
                shared.row_context[row] = 0U;
            } else {
                const std::int32_t position = row_positions[global_row];
                if (position < 0 || static_cast<std::size_t>(position) >= context) {
                    for (std::uint32_t head = 0U; head < 6U; ++head)
                        shared.invalid_rows[head] |= bit;
                    shared.row_context[row] = 0U;
                } else {
                    shared.row_context[row] = static_cast<std::uint32_t>(position) + 1U;
                    shared.maximum_context = max(shared.maximum_context,
                                                 shared.row_context[row]);
                }
            }
        }
        if (!invalid_active && shared.maximum_context != 0U && page_table_row != nullptr) {
            const std::int32_t table_row = page_table_row[0];
            if (table_row < 0 || static_cast<std::uint32_t>(table_row) >= page_table_row_count) {
                const std::uint32_t mask = (1U << tile_rows) - 1U;
                for (std::uint32_t head = 0U; head < 6U; ++head)
                    shared.invalid_rows[head] |= mask & ~shared.inactive_rows;
                shared.maximum_context = 0U;
            } else shared.table_offset = static_cast<std::size_t>(table_row) * page_table_row_stride;
        }
        for (std::uint32_t head = 0U; head < 6U; ++head)
            for (std::uint32_t row = 0U; row < tile_rows; ++row)
                shared.denominator[head][row] = 0.0F;
    }
    float numerator[6][kDensePrefillQueryTile]{};
    __syncthreads();
    for (std::uint32_t key_base = 0U; key_base < shared.maximum_context; key_base += 16U) {
        if (tid == 0U) {
            const std::size_t physical_page = pages[shared.table_offset + key_base / kPageSize];
            shared.page_valid = physical_page < physical_tokens / kPageSize ? 1U : 0U;
            shared.physical_base = physical_page * kPageSize + key_base % kPageSize;
            if (shared.page_valid == 0U) {
                for (std::uint32_t row = 0U; row < tile_rows; ++row) {
                    if (shared.row_context[row] <= key_base) continue;
                    for (std::uint32_t head = 0U; head < 6U; ++head)
                        shared.invalid_rows[head] |= 1U << row;
                }
            }
        }
        __syncthreads();
        for (std::uint32_t flat = tid; flat < 16U * (kHeadDim / 2U); flat += blockDim.x) {
            const std::uint32_t token = flat / (kHeadDim / 2U);
            const std::uint32_t packed_feature = flat % (kHeadDim / 2U);
            shared.packed_values[token][packed_feature] =
                shared.page_valid != 0U && key_base + token < shared.maximum_context
                    ? values[int4_offset<Group, false, false>(
                          shared.physical_base + token, kv_head, packed_feature * 2U,
                          4U, physical_tokens, false)] : 0U;
        }
        for (std::uint32_t flat = tid; flat < 16U * (kHeadDim / Group); flat += blockDim.x) {
            const std::uint32_t token = flat / (kHeadDim / Group);
            const std::uint32_t value_group = flat % (kHeadDim / Group);
            shared.value_scales[token][value_group] =
                shared.page_valid != 0U && key_base + token < shared.maximum_context
                    ? scales[int4_offset<Group, false, false>(
                          shared.physical_base + token, kv_head, value_group * Group,
                          4U, physical_tokens, true)] : 0U;
        }
        for (std::uint32_t flat = tid; flat < 6U * kDensePrefillQueryTile * 16U;
             flat += blockDim.x) {
            const std::uint32_t token = flat % 16U;
            const std::uint32_t query_item = flat / 16U;
            const std::uint32_t row = query_item % kDensePrefillQueryTile;
            const std::uint32_t head = query_item / kDensePrefillQueryTile;
            float probability = 0.0F;
            if (row < tile_rows && key_base + token < shared.row_context[row] &&
                shared.page_valid != 0U &&
                (shared.invalid_rows[head] & (1U << row)) == 0U) {
                const std::size_t vector =
                    static_cast<std::size_t>(query_base + row) * 24U + kv_head * 6U + head;
                const float maximum = maxima[vector];
                const float score = scores[vector * context + key_base + token];
                if (!isfinite(maximum) || !isfinite(score))
                    atomicOr(&shared.invalid_rows[head], 1U << row);
                else probability = __expf(score - maximum);
            }
            shared.probability[head][row][token] = probability;
        }
        __syncthreads();
        if (tid < 6U * kDensePrefillQueryTile) {
            const std::uint32_t head = tid / kDensePrefillQueryTile;
            const std::uint32_t row = tid % kDensePrefillQueryTile;
            float sum = 0.0F;
            for (std::uint32_t token = 0U; token < 16U; ++token)
                sum += shared.probability[head][row][token];
            shared.denominator[head][row] += sum;
        }
#pragma unroll
        for (std::uint32_t token = 0U; token < 16U; ++token) {
            const std::uint8_t packed = shared.packed_values[token][tid >> 1U];
            int code = static_cast<int>((packed >> ((tid & 1U) * 4U)) & 0x0fU);
            code = code >= 8 ? code - 16 : code;
            const float represented_value = static_cast<float>(code) * __half2float(
                __ushort_as_half(shared.value_scales[token][tid / Group]));
#pragma unroll
            for (std::uint32_t head = 0U; head < 6U; ++head) {
#pragma unroll
                for (std::uint32_t row = 0U; row < kDensePrefillQueryTile; ++row) {
                    numerator[head][row] = fmaf(shared.probability[head][row][token],
                                                represented_value, numerator[head][row]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (std::uint32_t head = 0U; head < 6U; ++head) {
#pragma unroll
        for (std::uint32_t row = 0U; row < kDensePrefillQueryTile; ++row) {
            if (row >= tile_rows) continue;
            const std::uint32_t bit = 1U << row;
            const std::size_t output_index =
                (static_cast<std::size_t>(query_base + row) * 24U + kv_head * 6U + head) *
                    kHeadDim + tid;
            if ((shared.inactive_rows & bit) != 0U) output[output_index] = 0.0F;
            else if ((shared.invalid_rows[head] & bit) != 0U ||
                     shared.denominator[head][row] == 0.0F)
                output[output_index] = __builtin_nanf("");
            else output[output_index] = numerator[head][row] / shared.denominator[head][row];
        }
    }
#else
    (void)scores; (void)maxima; (void)values; (void)scales; (void)pages;
    (void)page_table_row; (void)page_table_row_stride; (void)page_table_row_count;
    (void)context; (void)physical_tokens; (void)row_positions; (void)active_query_rows;
    (void)total_query_rows; (void)output;
#endif
}

template <std::uint32_t Group>
hipError_t dense_full_score_launch(const Fp8Int4KvAttentionArgs& a, float* scores,
                                   float* maxima, hipStream_t stream) noexcept {
    const dim3 pv_grid(4U, (a.query_rows + kDensePrefillQueryTile - 1U) /
                              kDensePrefillQueryTile);
    const dim3 qk_grid(pv_grid.x, pv_grid.y, (a.context + 15U) / 16U);
    if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((dense_full_score_qk_kernel<true>), qk_grid,
                           dim3(kDenseFullScoreQkThreads), 0, stream, a.query, a.key_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens,
                           a.row_positions, a.active_query_rows, a.query_rows,
                           a.attention_scale, scores);
    } else {
        hipLaunchKernelGGL((dense_full_score_qk_kernel<false>), qk_grid,
                           dim3(kDenseFullScoreQkThreads), 0, stream, a.query, a.key_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens,
                           a.row_positions, nullptr, a.query_rows, a.attention_scale, scores);
    }
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    const dim3 maximum_grid(a.query_rows * 24U);
    if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((dense_full_score_maximum_kernel<true>), maximum_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, a.context,
                           a.row_positions, a.active_query_rows, a.query_rows, maxima);
    } else {
        hipLaunchKernelGGL((dense_full_score_maximum_kernel<false>), maximum_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, a.context,
                           a.row_positions, nullptr, a.query_rows, maxima);
    }
    error = hipGetLastError();
    if (error != hipSuccess) return error;
    if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((dense_full_score_pv_kernel<Group, true>), pv_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, maxima,
                           a.value_codes, a.value_scale_bits, a.page_table, a.page_table_row,
                           a.page_table_row_stride, a.page_table_row_count, a.context,
                           a.physical_tokens, a.row_positions, a.active_query_rows,
                           a.query_rows, a.output);
    } else {
        hipLaunchKernelGGL((dense_full_score_pv_kernel<Group, false>), pv_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, maxima,
                           a.value_codes, a.value_scale_bits, a.page_table, a.page_table_row,
                           a.page_table_row_stride, a.page_table_row_count, a.context,
                           a.physical_tokens, a.row_positions, nullptr, a.query_rows, a.output);
    }
    return hipGetLastError();
}

hipError_t dense_full_score_qk_bk32_launch(const Fp8Int4KvAttentionArgs& a, float* scores,
                                           hipStream_t stream) noexcept {
    const dim3 qk_grid(4U, (a.query_rows + kDensePrefillQueryTile - 1U) /
                              kDensePrefillQueryTile,
                       (a.context + 31U) / 32U);
    if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((dense_full_score_qk_bk32_kernel<true>), qk_grid,
                           dim3(kDenseFullScoreQkThreads), 0, stream, a.query, a.key_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens,
                           a.row_positions, a.active_query_rows, a.query_rows,
                           a.attention_scale, scores);
    } else {
        hipLaunchKernelGGL((dense_full_score_qk_bk32_kernel<false>), qk_grid,
                           dim3(kDenseFullScoreQkThreads), 0, stream, a.query, a.key_codes,
                           a.page_table, a.page_table_row, a.page_table_row_stride,
                           a.page_table_row_count, a.context, a.physical_tokens,
                           a.row_positions, nullptr, a.query_rows, a.attention_scale, scores);
    }
    return hipGetLastError();
}

template <std::uint32_t Group>
hipError_t dense_full_score_bk32_launch(
    const Fp8Int4KvAttentionArgs& a, float* scores, float* maxima,
    hipStream_t stream) noexcept {
    hipError_t error = dense_full_score_qk_bk32_launch(a, scores, stream);
    if (error != hipSuccess) return error;
    const dim3 pv_grid(4U, (a.query_rows + kDensePrefillQueryTile - 1U) /
                              kDensePrefillQueryTile);
    const dim3 maximum_grid(a.query_rows * 24U);
    if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((dense_full_score_maximum_kernel<true>), maximum_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, a.context,
                           a.row_positions, a.active_query_rows, a.query_rows, maxima);
    } else {
        hipLaunchKernelGGL((dense_full_score_maximum_kernel<false>), maximum_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, a.context,
                           a.row_positions, nullptr, a.query_rows, maxima);
    }
    error = hipGetLastError();
    if (error != hipSuccess) return error;
    if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((dense_full_score_pv_kernel<Group, true>), pv_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, maxima,
                           a.value_codes, a.value_scale_bits, a.page_table, a.page_table_row,
                           a.page_table_row_stride, a.page_table_row_count, a.context,
                           a.physical_tokens, a.row_positions, a.active_query_rows,
                           a.query_rows, a.output);
    } else {
        hipLaunchKernelGGL((dense_full_score_pv_kernel<Group, false>), pv_grid,
                           dim3(kDensePrefillThreads), 0, stream, scores, maxima,
                           a.value_codes, a.value_scale_bits, a.page_table, a.page_table_row,
                           a.page_table_row_stride, a.page_table_row_count, a.context,
                           a.physical_tokens, a.row_positions, nullptr, a.query_rows, a.output);
    }
    return hipGetLastError();
}

__global__ void softmax_wmma_scores_in_place_kernel(
    float* scores, std::size_t context, const std::int32_t* row_positions, std::uint32_t row) {
    const std::uint32_t qh = blockIdx.x;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::int32_t row_position =
        row_positions == nullptr ? -1 : row_positions[row];
    const bool invalid_row = row_positions != nullptr &&
                             (row_position < 0 ||
                              static_cast<std::size_t>(row_position) >= context);
    const std::size_t row_context =
        row_positions == nullptr || invalid_row
            ? context
            : static_cast<std::size_t>(row_position) + 1U;
    float* head_scores = scores + static_cast<std::size_t>(qh) * context;
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
            if (!isfinite(score)) local_invalid = 1U;
            else local_max = fmaxf(local_max, score);
        }
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride, 32));
    }
    const std::uint32_t invalid_lanes = __ballot(local_invalid != 0U);
    if (lane == 0U) {
        wave_values[wave] = local_max;
        wave_invalid[wave] = invalid_lanes != 0U ? 1U : 0U;
    }
    __syncthreads();
    if (thread == 0U) {
        float maximum = wave_values[0];
        std::uint32_t bad = wave_invalid[0];
        for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
            maximum = fmaxf(maximum, wave_values[index]);
            bad |= wave_invalid[index];
        }
        block_max = maximum;
        invalid = bad;
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

__global__ void softmax_wmma_scores_batched_w5_in_place_kernel(
    float* scores, std::size_t context, const std::int32_t* row_positions) {
    const std::uint32_t qh = blockIdx.x;
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::int32_t row_position = row_positions[row];
    const bool invalid_row = row_position < 0 || static_cast<std::size_t>(row_position) >= context;
    const std::size_t row_context = invalid_row ? context :
        static_cast<std::size_t>(row_position) + 1U;
    float* head_scores = scores +
        (static_cast<std::size_t>(row) * gridDim.x + qh) * context;
    __shared__ float wave_values[kHeadDim / 32U];
    __shared__ std::uint32_t wave_invalid[kHeadDim / 32U];
    __shared__ float block_max;
    __shared__ float denominator;
    __shared__ std::uint32_t invalid;

    float local_max = -INFINITY;
    std::uint32_t local_invalid = invalid_row || row_context == 0U || row_context > context;
    if (local_invalid == 0U) {
        for (std::size_t logical = thread; logical < row_context; logical += blockDim.x) {
            const float score = head_scores[logical];
            if (!isfinite(score)) local_invalid = 1U;
            else local_max = fmaxf(local_max, score);
        }
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride, 32));
    }
    const std::uint32_t invalid_lanes = __ballot(local_invalid != 0U);
    if (lane == 0U) {
        wave_values[wave] = local_max;
        wave_invalid[wave] = invalid_lanes != 0U;
    }
    __syncthreads();
    if (thread == 0U) {
        float maximum = wave_values[0];
        std::uint32_t bad = wave_invalid[0];
        for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
            maximum = fmaxf(maximum, wave_values[index]);
            bad |= wave_invalid[index];
        }
        block_max = maximum;
        invalid = bad;
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
hipError_t softmax_pv_from_wmma_scores_launch(const Fp8Int4KvAttentionArgs& a,
                                               std::uint32_t row, hipStream_t stream) {
    hipLaunchKernelGGL(softmax_wmma_scores_in_place_kernel, dim3(a.query_heads), dim3(kHeadDim),
                       0, stream, a.score_workspace, a.context, a.row_positions, row);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    const Int4KvPvArgs pv{
        .probabilities = a.score_workspace,
        .value_codes = a.value_codes,
        .value_scale_bits = a.value_scale_bits,
        .page_table = a.page_table,
        .page_table_pages = a.page_table_pages,
        .page_table_row = a.page_table_row,
        .page_table_row_stride = a.page_table_row_stride,
        .page_table_row_count = a.page_table_row_count,
        .context = a.context,
        .physical_tokens = a.physical_tokens,
        .query_heads = a.query_heads,
        .kv_heads = a.kv_heads,
        .q_to_kv = a.q_to_kv,
        .value_group = G,
        .value_layout = a.value_layout,
        .value_scale_layout = a.value_scale_layout,
        .output = a.output + static_cast<std::size_t>(row) * a.query_heads * kHeadDim,
    };
    return pv_launch<G, V, S>(pv, stream);
}

template <std::uint32_t G>
hipError_t softmax_pv_from_wmma_scores_dispatch(const Fp8Int4KvAttentionArgs& a,
                                                 std::uint32_t row, hipStream_t stream) {
    const unsigned layout =
        (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
        (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U);
    switch (layout) {
    case 0: return softmax_pv_from_wmma_scores_launch<G, false, false>(a, row, stream);
    case 1: return softmax_pv_from_wmma_scores_launch<G, true, false>(a, row, stream);
    case 2: return softmax_pv_from_wmma_scores_launch<G, false, true>(a, row, stream);
    default: return softmax_pv_from_wmma_scores_launch<G, true, true>(a, row, stream);
    }
}

hipError_t fp8_int4_kv_attention_fused(const Fp8Int4KvAttentionArgs& a,
                                        hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.value_codes || !a.value_scale_bits || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens ||
        a.physical_tokens % kPageSize || !a.query_heads || !a.kv_heads || !a.q_to_kv ||
        a.query_heads != a.kv_heads * a.q_to_kv || !a.query_rows || !a.output ||
        !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        !valid_plane_layout(a.key_layout) || !valid_plane_layout(a.value_layout) ||
        !valid_plane_layout(a.value_scale_layout) ||
        (a.ancestor_masks == nullptr) != (a.prefix_lengths == nullptr) ||
        (a.ancestor_masks != nullptr && a.row_positions == nullptr) ||
        a.prefix_length_stride > 1U ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages ||
          a.page_table_row_count == 0U)) ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize) {
        return hipErrorInvalidValue;
    }
    return a.value_group == 16 ? fused_attention_dispatch<16>(a, stream)
         : a.value_group == 32 ? fused_attention_dispatch<32>(a, stream)
                               : hipErrorInvalidValue;
}

std::size_t fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(
    std::uint32_t query_rows) noexcept {
    if (query_rows < 128U || query_rows > 4096U) return 0U;
    const std::size_t vectors = static_cast<std::size_t>(query_rows) * 24U;
    return (vectors * query_rows + vectors) * sizeof(float);
}

hipError_t fp8_int4_kv_attention_dense_prefill_full_score(
    const Fp8Int4KvAttentionArgs& a, void* workspace, std::size_t workspace_bytes,
    hipStream_t stream) noexcept {
    const std::size_t required =
        fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(a.query_rows);
    if (!a.query || !a.key_codes || !a.value_codes || !a.value_scale_bits ||
        !a.page_table || !a.page_table_pages || !a.context ||
        a.context > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        !a.physical_tokens || a.physical_tokens % kPageSize || a.query_heads != 24U ||
        a.kv_heads != 4U || a.q_to_kv != 6U || a.query_rows < 128U ||
        a.query_rows > 4096U || !a.row_positions || a.ancestor_masks || a.prefix_lengths ||
        a.prefix_length_stride != 1U || a.score_workspace || !a.output ||
        !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        a.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
        a.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        a.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages ||
          a.page_table_row_count == 0U)) ||
        a.context != a.query_rows || !workspace || required == 0U ||
        workspace_bytes < required ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(float) != 0U ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize ||
        (a.value_group != 16U && a.value_group != 32U)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    auto* scores = static_cast<float*>(workspace);
    float* maxima = scores + static_cast<std::size_t>(a.query_rows) * 24U * a.query_rows;
    if (a.query_rows >= 512U) {
        return a.value_group == 16U ? dense_full_score_bk32_launch<16U>(a, scores, maxima, stream)
                                    : dense_full_score_bk32_launch<32U>(a, scores, maxima, stream);
    }
    return a.value_group == 16U ? dense_full_score_launch<16U>(a, scores, maxima, stream)
                                : dense_full_score_launch<32U>(a, scores, maxima, stream);
#else
    (void)workspace;
    (void)workspace_bytes;
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t fp8_int4_kv_attention_dense_prefill_full_score_resources(
    std::uint32_t value_group, DensePrefillFullScoreStage stage,
    DensePrefillWmmaResources* resources) noexcept {
    if (resources == nullptr || (value_group != 16U && value_group != 32U)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const void* kernel = nullptr;
    switch (stage) {
    case DensePrefillFullScoreStage::QkBk16:
        kernel = reinterpret_cast<const void*>(dense_full_score_qk_kernel<false>);
        break;
    case DensePrefillFullScoreStage::Maximum:
        kernel = reinterpret_cast<const void*>(dense_full_score_maximum_kernel<false>);
        break;
    case DensePrefillFullScoreStage::Pv:
        kernel = value_group == 16U
            ? reinterpret_cast<const void*>(dense_full_score_pv_kernel<16U, false>)
            : reinterpret_cast<const void*>(dense_full_score_pv_kernel<32U, false>);
        break;
    case DensePrefillFullScoreStage::QkBk32:
        kernel = reinterpret_cast<const void*>(dense_full_score_qk_bk32_kernel<false>);
        break;
    default: return hipErrorInvalidValue;
    }
    hipFuncAttributes attributes{};
    const hipError_t error = hipFuncGetAttributes(&attributes, kernel);
    if (error != hipSuccess) return error;
    *resources = DensePrefillWmmaResources{
        attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
        static_cast<int>(attributes.localSizeBytes)};
    return hipSuccess;
#else
    return hipErrorNotSupported;
#endif
}

hipError_t fp8_int4_kv_attention_wmma(const Fp8Int4KvAttentionArgs& a,
                                       hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.value_codes || !a.value_scale_bits || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens ||
        a.physical_tokens % kPageSize || !a.query_heads || !a.kv_heads || a.q_to_kv != 6U ||
        a.query_heads != a.kv_heads * a.q_to_kv || !a.query_rows || !a.score_workspace ||
        !a.output || !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages ||
          a.page_table_row_count == 0U)) ||
        !valid_plane_layout(a.key_layout) || !valid_plane_layout(a.value_layout) ||
        !valid_plane_layout(a.value_scale_layout) ||
        a.ancestor_masks != nullptr || a.prefix_lengths != nullptr ||
        a.active_query_rows != nullptr ||
        a.prefix_length_stride > 1U ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize ||
        (a.value_group != 16U && a.value_group != 32U)) {
        return hipErrorInvalidValue;
    }
    for (std::uint32_t row = 0; row < a.query_rows; ++row) {
        const Fp8KvQkArgs qk{
            .query = a.query + static_cast<std::size_t>(row) * a.query_heads * kHeadDim,
            .key_codes = a.key_codes,
            .page_table = a.page_table,
            .page_table_pages = a.page_table_pages,
            .page_table_row = a.page_table_row,
            .page_table_row_stride = a.page_table_row_stride,
            .page_table_row_count = a.page_table_row_count,
            .context = a.context,
            .physical_tokens = a.physical_tokens,
            .query_heads = a.query_heads,
            .kv_heads = a.kv_heads,
            .q_to_kv = a.q_to_kv,
            .attention_scale = a.attention_scale,
            .key_layout = a.key_layout,
            .scores = a.score_workspace,
        };
        hipError_t error = fp8_kv_qk_wmma(qk, stream);
        if (error != hipSuccess) { return error; }
        error = a.value_group == 16U
                    ? softmax_pv_from_wmma_scores_dispatch<16>(a, row, stream)
                    : softmax_pv_from_wmma_scores_dispatch<32>(a, row, stream);
        if (error != hipSuccess) { return error; }
    }
    return hipSuccess;
}

std::size_t fp8_int4_kv_attention_wmma_batched_w5_workspace_bytes(
    std::size_t context) noexcept {
    constexpr std::size_t elements_per_token = 5U * 24U;
    return context != 0U && context <= std::numeric_limits<std::size_t>::max() /
                                        (elements_per_token * sizeof(float))
               ? context * elements_per_token * sizeof(float)
               : 0U;
}

template <std::uint32_t Group, bool VToken, bool SToken>
hipError_t batched_w5_pv_launch(const Fp8Int4KvAttentionArgs& a, hipStream_t stream) {
    hipLaunchKernelGGL((pv_vector_batched_w5_kernel<Group, VToken, SToken>),
                       dim3(a.query_heads, 5U), dim3(kHeadDim), 0, stream,
                       a.score_workspace, a.value_codes, a.value_scale_bits, a.page_table,
                       a.page_table_row, a.page_table_row_stride, a.page_table_row_count,
                       a.context, a.physical_tokens, a.q_to_kv, a.output);
    return hipGetLastError();
}

template <std::uint32_t Group>
hipError_t batched_w5_pv_dispatch(const Fp8Int4KvAttentionArgs& a, hipStream_t stream) {
    const unsigned layout =
        (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
        (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U);
    switch (layout) {
    case 0: return batched_w5_pv_launch<Group, false, false>(a, stream);
    case 1: return batched_w5_pv_launch<Group, true, false>(a, stream);
    case 2: return batched_w5_pv_launch<Group, false, true>(a, stream);
    default: return batched_w5_pv_launch<Group, true, true>(a, stream);
    }
}

hipError_t fp8_int4_kv_attention_wmma_batched_w5(
    const Fp8Int4KvAttentionArgs& a, hipStream_t stream) noexcept {
    if (!a.query || !a.key_codes || !a.value_codes || !a.value_scale_bits || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens ||
        a.physical_tokens % kPageSize || a.query_heads != 24U || a.kv_heads != 4U ||
        a.q_to_kv != 6U || a.query_rows != 5U || !a.row_positions || !a.score_workspace ||
        !a.output || !std::isfinite(a.attention_scale) || !(a.attention_scale > 0.0F) ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
        a.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        a.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
        a.ancestor_masks != nullptr ||
        a.prefix_lengths != nullptr || a.active_query_rows != nullptr ||
        a.prefix_length_stride > 1U ||
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize ||
        a.value_group != 16U) {
        return hipErrorInvalidValue;
    }
    const dim3 qk_grid(static_cast<unsigned>((a.context + 15U) / 16U),
                       a.query_rows * a.kv_heads);
    hipLaunchKernelGGL((qk_wmma_batched_w5_kernel<true>), qk_grid, dim3(32), 0, stream,
                       a.query, a.key_codes, a.page_table, a.page_table_row,
                       a.page_table_row_stride, a.page_table_row_count, a.context,
                       a.physical_tokens, a.kv_heads, a.q_to_kv, a.attention_scale,
                       a.score_workspace);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    hipLaunchKernelGGL(softmax_wmma_scores_batched_w5_in_place_kernel,
                       dim3(a.query_heads, a.query_rows), dim3(kHeadDim), 0, stream,
                       a.score_workspace, a.context, a.row_positions);
    error = hipGetLastError();
    if (error != hipSuccess) return error;
    return batched_w5_pv_launch<16U, false, false>(a, stream);
}

namespace {

struct Split512Workspace {
    float* scores = nullptr;
    float* maxima = nullptr;
    float* sums = nullptr;
    float* numerators = nullptr;
    std::size_t chunks = 0;
};

[[nodiscard]] bool checked_product(std::size_t a, std::size_t b,
                                   std::size_t& out) noexcept {
    if (a != 0U && b > static_cast<std::size_t>(-1) / a) return false;
    out = a * b;
    return true;
}

[[nodiscard]] std::size_t split512_workspace_bytes(std::uint32_t rows,
                                                    std::size_t context) noexcept {
    constexpr std::size_t kMaximumSplitContext = 512U * kSplitTokens;
    if ((rows != 1U && rows != 4U) || context == 0U ||
        context > kMaximumSplitContext) return 0U;
    const std::size_t chunks = (context + kSplitTokens - 1U) / kSplitTokens;
    std::size_t heads_context = 0U;
    std::size_t records = 0U;
    std::size_t score_elements = 0U;
    std::size_t record_elements = 0U;
    std::size_t elements = 0U;
    if (!checked_product(rows, 24U, heads_context) ||
        !checked_product(heads_context, context, score_elements) ||
        !checked_product(heads_context, chunks, records) ||
        !checked_product(records, kHeadDim + 2U, record_elements) ||
        score_elements > static_cast<std::size_t>(-1) - record_elements) {
        return 0U;
    }
    elements = score_elements + record_elements;
    std::size_t bytes = 0U;
    return checked_product(elements, sizeof(float), bytes) ? bytes : 0U;
}

[[nodiscard]] Split512Workspace split512_workspace(void* storage, std::uint32_t rows,
                                                    std::size_t context) noexcept {
    auto* base = static_cast<float*>(storage);
    const std::size_t chunks = (context + kSplitTokens - 1U) / kSplitTokens;
    const std::size_t score_elements = static_cast<std::size_t>(rows) * 24U * context;
    const std::size_t records = static_cast<std::size_t>(rows) * 24U * chunks;
    return Split512Workspace{
        .scores = base,
        .maxima = base + score_elements,
        .sums = base + score_elements + records,
        .numerators = base + score_elements + records * 2U,
        .chunks = chunks,
    };
}

template <bool DeviceActiveRows>
__global__ void split512_qk_bf16_t4_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context, std::size_t physical_tokens,
    const std::int32_t* active_query_rows, float attention_scale, float* scores) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    constexpr std::uint32_t kRows = 4U;
    constexpr std::uint32_t kQueriesPerKvHead = 6U;
    constexpr std::uint32_t kQueryHeads = 24U;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t half = lane >> 4U;
    const std::uint32_t kv_head = blockIdx.y;
    const std::size_t logical_base = static_cast<std::size_t>(blockIdx.x) * 16U;
    std::int32_t active = static_cast<std::int32_t>(kRows);
    if constexpr (DeviceActiveRows) {
        active = active_query_rows[0];
        if (active < 0 || active > static_cast<std::int32_t>(kRows)) return;
        if (active == 0) return;
    }
    if (page_table_row != nullptr) {
        const std::int32_t represented = page_table_row[0];
        if (represented < 0 ||
            static_cast<std::uint32_t>(represented) >= page_table_row_count) {
            for (std::uint32_t index = thread; index < kRows * kQueriesPerKvHead * 16U;
                 index += blockDim.x) {
                const std::uint32_t row = index / (kQueriesPerKvHead * 16U);
                const std::uint32_t remainder = index % (kQueriesPerKvHead * 16U);
                const std::uint32_t local_head = remainder / 16U;
                const std::size_t logical = logical_base + remainder % 16U;
                if (row < static_cast<std::uint32_t>(active) && logical < context) {
                    scores[(static_cast<std::size_t>(row) * kQueryHeads +
                            kv_head * kQueriesPerKvHead + local_head) * context + logical] =
                        __builtin_nanf("");
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
        const std::uint32_t feature = flat / 8U;
        const std::uint32_t pair = flat % 8U;
        std::uint32_t represented_pair = 0U;
#pragma unroll
        for (std::uint32_t item = 0U; item < 2U; ++item) {
            const std::uint32_t token = pair * 2U + item;
            const std::size_t logical = logical_base + token;
            hip_bfloat16 decoded(0.0F);
            if (logical < context) {
                const std::size_t physical = static_cast<std::size_t>(pages[logical / kPageSize]) *
                                                 kPageSize + logical % kPageSize;
                if (physical >= physical_tokens) {
                    atomicExch(&invalid, 1U);
                } else {
                    decoded = hip_bfloat16(__half2float(__half(__hip_cvt_fp8_to_halfraw(
                        keys[key_offset<true>(physical, kv_head, feature, 4U, physical_tokens)],
                        __HIP_E4M3))));
                }
            }
            represented_pair |= static_cast<std::uint32_t>(
                static_cast<std::uint16_t>(decoded.data)) << (item * 16U);
        }
        key_tile[feature][pair] = represented_pair;
    }
    __syncthreads();
    if (invalid != 0U) {
        for (std::uint32_t index = thread; index < kRows * kQueriesPerKvHead * 16U;
             index += blockDim.x) {
            const std::uint32_t row = index / (kQueriesPerKvHead * 16U);
            const std::uint32_t remainder = index % (kQueriesPerKvHead * 16U);
            const std::uint32_t local_head = remainder / 16U;
            const std::size_t logical = logical_base + remainder % 16U;
            if (row < static_cast<std::uint32_t>(active) && logical < context) {
                scores[(static_cast<std::size_t>(row) * kQueryHeads +
                        kv_head * kQueriesPerKvHead + local_head) * context + logical] =
                    __builtin_nanf("");
            }
        }
        return;
    }

    F32x8 accum{};
#pragma unroll
    for (std::uint32_t feature_tile = 0U; feature_tile < kHeadDim / 16U; ++feature_tile) {
        I16x8 q_fragment{};
        I16x8 k_fragment{};
        const std::uint32_t flat_query = wave * 16U + axis;
#pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t feature = feature_tile * 16U + half * 8U + item;
            if (flat_query < kRows * kQueriesPerKvHead) {
                const std::uint32_t row = flat_query / kQueriesPerKvHead;
                const std::uint32_t local_head = flat_query % kQueriesPerKvHead;
                if (row < static_cast<std::uint32_t>(active)) {
                    q_fragment[item] = static_cast<short>(query[
                        (static_cast<std::size_t>(row) * kQueryHeads +
                         kv_head * kQueriesPerKvHead + local_head) * kHeadDim + feature].data);
                }
            }
            const std::uint32_t pair = key_tile[feature][axis >> 1U];
            k_fragment[item] = static_cast<short>(pair >> ((axis & 1U) * 16U));
        }
        accum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
            q_fragment, k_fragment, accum);
    }
#pragma unroll
    for (std::uint32_t item = 0U; item < 8U; ++item) {
        const std::uint32_t flat_query = wave * 16U + half * 8U + item;
        const std::uint32_t row = flat_query / kQueriesPerKvHead;
        const std::uint32_t local_head = flat_query % kQueriesPerKvHead;
        const std::size_t logical = logical_base + axis;
        if (flat_query < kRows * kQueriesPerKvHead &&
            row < static_cast<std::uint32_t>(active) && logical < context) {
            const std::uint32_t query_head = kv_head * kQueriesPerKvHead + local_head;
            scores[(static_cast<std::size_t>(row) * kQueryHeads + query_head) * context + logical] =
                accum[item] * attention_scale;
        }
    }
#else
    (void)query; (void)keys; (void)pages; (void)page_table_row;
    (void)page_table_row_stride; (void)page_table_row_count; (void)context;
    (void)physical_tokens; (void)active_query_rows; (void)attention_scale; (void)scores;
#endif
}

template <std::uint32_t Group, bool TreeMask, bool DeviceActiveRows>
__global__ void split512_softmax_pv_partial_kernel(
    const float* scores, const std::uint8_t* values, const std::uint16_t* scales,
    const std::uint32_t* pages, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count,
    std::size_t context, std::size_t physical_tokens, const std::int32_t* row_positions,
    const std::int32_t* ancestor_masks, const std::int32_t* prefix_lengths,
    std::uint32_t prefix_length_stride, const std::int32_t* active_query_rows,
    std::uint32_t query_rows, std::size_t chunks, float* maxima, float* sums,
    float* numerators) {
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t row = blockIdx.z;
    const std::uint32_t qh = blockIdx.y;
    const std::size_t chunk = blockIdx.x;
    const std::size_t record = (static_cast<std::size_t>(row) * 24U + qh) * chunks + chunk;
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
        if (represented < 0 ||
            static_cast<std::uint32_t>(represented) >= page_table_row_count) {
            if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
            return;
        }
        pages += static_cast<std::size_t>(represented) * page_table_row_stride;
    }
    const std::int32_t represented_position = row_positions == nullptr ?
        static_cast<std::int32_t>(context - 1U) : row_positions[row];
    bool bad_metadata = represented_position < 0 ||
                        static_cast<std::size_t>(represented_position) >= context;
    const std::size_t row_context = bad_metadata ? 0U :
                                    static_cast<std::size_t>(represented_position) + 1U;
    std::size_t prefix = 0U;
    std::uint32_t mask = 0U;
    if constexpr (TreeMask) {
        const std::int32_t represented_prefix = prefix_lengths[
            static_cast<std::size_t>(row) * prefix_length_stride];
        mask = static_cast<std::uint32_t>(ancestor_masks[row]);
        if (represented_prefix < 0) bad_metadata = true;
        if (!bad_metadata) {
            prefix = static_cast<std::size_t>(represented_prefix);
            if (prefix > row_context || row_context - prefix >= 32U) bad_metadata = true;
            else if (prefix == 0U) {
                const std::size_t suffix = row_context;
                const std::uint32_t available = suffix == 0U ? 0U :
                    ((1U << static_cast<std::uint32_t>(suffix)) - 1U);
                if ((mask & available) == 0U) bad_metadata = true;
            }
        }
    }
    if (bad_metadata) {
        if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
        return;
    }

    float local_max = -INFINITY;
    std::uint32_t local_bad = 0U;
    for (std::uint32_t offset = thread; offset < kSplitTokens; offset += blockDim.x) {
        const std::size_t logical = chunk_base + offset;
        bool visible = logical < row_context;
        if constexpr (TreeMask) {
            if (visible && logical >= prefix) {
                visible = (mask & (1U << static_cast<std::uint32_t>(logical - prefix))) != 0U;
            }
        }
        float value = -INFINITY;
        if (visible) {
            value = scores[(static_cast<std::size_t>(row) * 24U + qh) * context + logical];
            if (!isfinite(value)) local_bad = 1U;
            else local_max = fmaxf(local_max, value);
        }
        weights[offset] = value;
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride, 32));
    }
    const std::uint32_t bad_lanes = __ballot(local_bad != 0U);
    if (lane == 0U) {
        wave_values[wave] = local_max;
        wave_bad[wave] = bad_lanes != 0U ? 1U : 0U;
    }
    __syncthreads();
    if (thread == 0U) {
        float value = wave_values[0];
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
        const float score = weights[offset];
        const float probability = isfinite(score) ? __expf(score - maximum) : 0.0F;
        weights[offset] = probability;
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
        invalid = 0U;
    }
    __syncthreads();

    for (std::uint32_t offset = thread; offset < kSplitTokens; offset += blockDim.x) {
        const std::size_t logical = chunk_base + offset;
        unsigned long long represented = static_cast<unsigned long long>(-1);
        if (weights[offset] != 0.0F && logical < row_context) {
            const std::size_t translated = static_cast<std::size_t>(pages[logical / kPageSize]) *
                                               kPageSize + logical % kPageSize;
            if (translated >= physical_tokens) atomicExch(&invalid, 1U);
            else represented = static_cast<unsigned long long>(translated);
        }
        physical[offset] = represented;
    }
    __syncthreads();
    if (invalid != 0U) {
        if (thread == 0U) maxima[record] = sums[record] = __builtin_nanf("");
        return;
    }

    const std::uint32_t kvh = qh / 6U;
    float numerator = 0.0F;
    for (std::uint32_t offset = 0U; offset < kSplitTokens; ++offset) {
        const float probability = weights[offset];
        if (probability == 0.0F) continue;
        const std::size_t translated = static_cast<std::size_t>(physical[offset]);
        const std::uint8_t packed = values[int4_offset<Group, false, false>(
            translated, kvh, thread, 4U, physical_tokens, false)];
        int code = static_cast<int>((packed >> ((thread & 1U) * 4U)) & 0x0fU);
        code = code >= 8 ? code - 16 : code;
        const std::uint16_t scale_bits = scales[int4_offset<Group, false, false>(
            translated, kvh, thread, 4U, physical_tokens, true)];
        numerator = fmaf(probability,
                         static_cast<float>(code) * __half2float(__ushort_as_half(scale_bits)),
                         numerator);
    }
    if (thread == 0U) {
        maxima[record] = denominator == 0.0F ? -INFINITY : maximum;
        sums[record] = denominator;
    }
    numerators[record * kHeadDim + thread] = numerator;
}

template <bool DeviceActiveRows>
__global__ void split512_merge_kernel(
    const float* maxima, const float* sums, const float* numerators,
    const std::int32_t* active_query_rows, std::uint32_t query_rows,
    std::size_t chunks, float* output) {
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t qh = blockIdx.x;
    const std::uint32_t row = blockIdx.y;
    const std::size_t first = (static_cast<std::size_t>(row) * 24U + qh) * chunks;
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

    float local_max = -INFINITY;
    std::uint32_t local_bad = 0U;
    for (std::size_t chunk = thread; chunk < chunks; chunk += blockDim.x) {
        const float chunk_max = maxima[first + chunk];
        const float chunk_sum = sums[first + chunk];
        if (isnan(chunk_max) || !isfinite(chunk_sum) || chunk_sum < 0.0F) local_bad = 1U;
        else if (chunk_sum != 0.0F) local_max = fmaxf(local_max, chunk_max);
    }
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride, 32));
    }
    const std::uint32_t bad_lanes = __ballot(local_bad != 0U);
    if (lane == 0U) {
        wave_values[wave] = local_max;
        wave_bad[wave] = bad_lanes != 0U ? 1U : 0U;
    }
    __syncthreads();
    if (thread == 0U) {
        float value = wave_values[0];
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
        const float coefficient = sums[first + chunk] == 0.0F ? 0.0F :
                                  __expf(maxima[first + chunk] - maximum);
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
        numerator = fmaf(coefficients[chunk],
                         numerators[(first + chunk) * kHeadDim + thread], numerator);
    }
    output[output_base + thread] = numerator / denominator;
}

template <std::uint32_t Group, bool TreeMask, bool DeviceActiveRows>
hipError_t split512_partial_merge(const Fp8Int4KvAttentionArgs& a,
                                  const Split512Workspace& workspace,
                                  hipStream_t stream) {
    hipLaunchKernelGGL((split512_softmax_pv_partial_kernel<Group, TreeMask, DeviceActiveRows>),
                       dim3(static_cast<unsigned>(workspace.chunks), a.query_heads, a.query_rows),
                       dim3(kHeadDim), 0, stream, workspace.scores, a.value_codes,
                       a.value_scale_bits, a.page_table, a.page_table_row,
                       a.page_table_row_stride, a.page_table_row_count, a.context,
                       a.physical_tokens, a.row_positions, a.ancestor_masks, a.prefix_lengths,
                       a.prefix_length_stride, a.active_query_rows, a.query_rows,
                       workspace.chunks, workspace.maxima, workspace.sums,
                       workspace.numerators);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    hipLaunchKernelGGL((split512_merge_kernel<DeviceActiveRows>),
                       dim3(a.query_heads, a.query_rows), dim3(kHeadDim), 0, stream,
                       workspace.maxima, workspace.sums, workspace.numerators,
                       a.active_query_rows, a.query_rows, workspace.chunks, a.output);
    return hipGetLastError();
}

template <std::uint32_t Group>
hipError_t split512_metadata_dispatch(const Fp8Int4KvAttentionArgs& a,
                                      const Split512Workspace& workspace,
                                      hipStream_t stream) {
    if (a.ancestor_masks != nullptr) {
        return a.active_query_rows != nullptr
            ? split512_partial_merge<Group, true, true>(a, workspace, stream)
            : split512_partial_merge<Group, true, false>(a, workspace, stream);
    }
    return a.active_query_rows != nullptr
        ? split512_partial_merge<Group, false, true>(a, workspace, stream)
        : split512_partial_merge<Group, false, false>(a, workspace, stream);
}

} // namespace

std::size_t fp8_int4_kv_attention_split512_workspace_capacity_bytes(
    std::uint32_t query_rows, std::size_t context) noexcept {
    return split512_workspace_bytes(query_rows, context);
}

hipError_t fp8_int4_kv_attention_split512(
    const Fp8Int4KvAttentionArgs& a, void* storage, std::size_t storage_bytes,
    hipStream_t stream) noexcept {
    const std::size_t required = split512_workspace_bytes(a.query_rows, a.context);
    if (!a.query || !a.key_codes || !a.value_codes || !a.value_scale_bits || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens ||
        a.physical_tokens % kPageSize || a.query_heads != 24U || a.kv_heads != 4U ||
        a.q_to_kv != 6U || (a.query_rows != 1U && a.query_rows != 4U) || !a.output ||
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
        a.page_table_pages < (a.context + kPageSize - 1U) / kPageSize ||
        required == 0U || storage == nullptr || storage_bytes != required ||
        reinterpret_cast<std::uintptr_t>(storage) % alignof(float) != 0U) {
        return hipErrorInvalidValue;
    }
    const Split512Workspace workspace = split512_workspace(storage, a.query_rows, a.context);
    hipError_t error = hipSuccess;
    if (a.query_rows == 1U) {
        const Fp8KvQkArgs qk{
            .query = a.query,
            .key_codes = a.key_codes,
            .page_table = a.page_table,
            .page_table_pages = a.page_table_pages,
            .page_table_row = a.page_table_row,
            .page_table_row_stride = a.page_table_row_stride,
            .page_table_row_count = a.page_table_row_count,
            .context = a.context,
            .physical_tokens = a.physical_tokens,
            .query_heads = a.query_heads,
            .kv_heads = a.kv_heads,
            .q_to_kv = a.q_to_kv,
            .attention_scale = a.attention_scale,
            .key_layout = a.key_layout,
            .scores = workspace.scores,
        };
        error = fp8_kv_qk_wmma(qk, stream);
    } else if (a.active_query_rows != nullptr) {
        hipLaunchKernelGGL((split512_qk_bf16_t4_kernel<true>),
                           dim3(static_cast<unsigned>((a.context + 15U) / 16U), a.kv_heads),
                           dim3(64), 0, stream, a.query, a.key_codes, a.page_table,
                           a.page_table_row, a.page_table_row_stride, a.page_table_row_count,
                           a.context, a.physical_tokens, a.active_query_rows,
                           a.attention_scale, workspace.scores);
        error = hipGetLastError();
    } else {
        hipLaunchKernelGGL((split512_qk_bf16_t4_kernel<false>),
                           dim3(static_cast<unsigned>((a.context + 15U) / 16U), a.kv_heads),
                           dim3(64), 0, stream, a.query, a.key_codes, a.page_table,
                           a.page_table_row, a.page_table_row_stride, a.page_table_row_count,
                           a.context, a.physical_tokens, nullptr, a.attention_scale,
                           workspace.scores);
        error = hipGetLastError();
    }
    if (error != hipSuccess) return error;
    return a.value_group == 16U
        ? split512_metadata_dispatch<16U>(a, workspace, stream)
        : split512_metadata_dispatch<32U>(a, workspace, stream);
}

} // namespace ninfer::ops::r9700::kv
