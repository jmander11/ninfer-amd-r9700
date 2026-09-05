#include "fp8_int4_kv_xattention.h"

#include <hip/hip_fp8.h>

#include <cmath>
#include <limits>

namespace ninfer::ops::r9700::kv {
namespace {

inline constexpr std::uint32_t kHeadDim = 256U;
inline constexpr std::uint32_t kThreads = 256U;
inline constexpr std::uint32_t kWaves = kThreads / 32U;
inline constexpr std::uint32_t kMaximumGridY = 65535U;
inline constexpr std::uint32_t kEstimatorPlanes =
    kXAttentionQueryBlockRows / kXAttentionStride;
using I16x8 = short __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));

[[nodiscard]] constexpr bool valid_plane_layout(Fp8Int4KvPlaneLayout layout) noexcept {
    return layout == Fp8Int4KvPlaneLayout::FeatureFastest ||
           layout == Fp8Int4KvPlaneLayout::TokenFastest;
}

[[nodiscard]] bool align256(std::size_t& value) noexcept {
    if (value > std::numeric_limits<std::size_t>::max() - 255U) return false;
    value = (value + 255U) & ~std::size_t{255U};
    return true;
}

[[nodiscard]] constexpr std::uint32_t ceil_div64(std::size_t value) noexcept {
    return static_cast<std::uint32_t>((value + kXAttentionPageSize - 1U) /
                                      kXAttentionPageSize);
}

[[nodiscard]] constexpr std::uint32_t next_power_of_two(std::uint32_t value) noexcept {
    if (value == 0U || value > 4096U) return 0U;
    --value;
    value |= value >> 1U;
    value |= value >> 2U;
    value |= value >> 4U;
    value |= value >> 8U;
    value |= value >> 16U;
    return value + 1U;
}

[[nodiscard]] bool checked_add(std::size_t& total, std::size_t value) noexcept {
    if (!align256(total)) return false;
    if (total > std::numeric_limits<std::size_t>::max() - value) return false;
    total += value;
    return true;
}

template <typename T>
[[nodiscard]] bool checked_add_elements(std::size_t& total, std::size_t elements) noexcept {
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) return false;
    return checked_add(total, elements * sizeof(T));
}

template <typename T>
T* take(std::uint8_t* base, std::size_t& offset, std::size_t elements) noexcept {
    (void)align256(offset);
    T* result = reinterpret_cast<T*>(base + offset);
    offset += elements * sizeof(T);
    return result;
}

template <bool TokenFastest>
__device__ std::size_t key_offset(std::size_t token, std::uint32_t head,
                                  std::uint32_t feature, std::uint32_t heads,
                                  std::size_t physical_tokens) {
    const std::size_t page = token / kXAttentionPageSize;
    const std::size_t in_page = token % kXAttentionPageSize;
    const std::size_t pages = physical_tokens / kXAttentionPageSize;
    if constexpr (TokenFastest) {
        return in_page + kXAttentionPageSize *
                             (static_cast<std::size_t>(feature) +
                              kHeadDim * (page + pages * head));
    }
    return feature + kHeadDim *
                         (in_page + kXAttentionPageSize *
                                        (static_cast<std::size_t>(head) + heads * page));
}

template <std::uint32_t Group, bool VToken, bool SToken>
__device__ std::size_t int4_offset(std::size_t token, std::uint32_t head,
                                   std::uint32_t feature, std::uint32_t heads,
                                   std::size_t physical_tokens, bool scale) {
    const std::size_t page = token / kXAttentionPageSize;
    const std::size_t in_page = token % kXAttentionPageSize;
    const std::size_t pages = physical_tokens / kXAttentionPageSize;
    const std::uint32_t leading = scale ? kHeadDim / Group : kHeadDim / 2U;
    const std::uint32_t coordinate = scale ? feature / Group : feature / 2U;
    const bool token_fastest = scale ? SToken : VToken;
    return token_fastest
               ? in_page + kXAttentionPageSize *
                               (static_cast<std::size_t>(coordinate) +
                                leading * (page + pages * head))
               : coordinate + leading *
                                  (in_page + kXAttentionPageSize *
                                                 (static_cast<std::size_t>(head) + heads * page));
}

__device__ float wave_sum(float value) {
    for (std::uint32_t delta = 16U; delta != 0U; delta >>= 1U) {
        value += __shfl_down(value, delta, 32);
    }
    return value;
}

template <std::uint32_t Stride, bool KToken>
__global__ void xattention_pack_keys_kernel(
    const std::uint8_t* keys, const std::uint32_t* page_table,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context, std::size_t physical_tokens,
    std::uint32_t kv_heads, hip_bfloat16* packed) {
    const std::uint32_t kv_head = blockIdx.x;
    if (kv_head >= kv_heads) return;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = page_table_row[0];
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            for (std::uint32_t element = threadIdx.x;
                 element < kXAttentionPageSize * kHeadDim; element += blockDim.x) {
                const std::size_t logical =
                    static_cast<std::size_t>(blockIdx.y) * kXAttentionPageSize +
                    element / kHeadDim;
                if (logical < context) {
                    packed[(static_cast<std::size_t>(kv_head) * context + logical) * kHeadDim +
                           element % kHeadDim] = hip_bfloat16(__builtin_nanf(""));
                }
            }
            return;
        }
        page_table += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    for (std::uint32_t element = threadIdx.x;
         element < kXAttentionPageSize * kHeadDim; element += blockDim.x) {
        const std::size_t logical = static_cast<std::size_t>(blockIdx.y) * kXAttentionPageSize +
                                    element / kHeadDim;
        if (logical >= context) continue;
        const std::uint32_t feature = element % kHeadDim;
        const std::size_t physical =
            static_cast<std::size_t>(page_table[logical / kXAttentionPageSize]) *
                kXAttentionPageSize + logical % kXAttentionPageSize;
        float value = __builtin_nanf("");
        if (physical < physical_tokens) {
            const std::uint8_t code = keys[key_offset<KToken>(
                physical, kv_head, feature, kv_heads, physical_tokens)];
            if ((code & 0x7fU) != 0x7fU) {
                value = __half2float(__half(__hip_cvt_fp8_to_halfraw(code, __HIP_E4M3)));
            }
        }
        packed[(static_cast<std::size_t>(kv_head) * context + logical) * kHeadDim + feature] =
            hip_bfloat16(value);
    }
}

template <std::uint32_t Stride>
__device__ F32x8 xattention_score_wmma_tile(
    const hip_bfloat16* query, const hip_bfloat16* packed_keys, std::size_t context,
    std::uint32_t query_heads, std::uint32_t query_head, std::uint32_t kv_head,
    std::uint32_t query_start, std::uint32_t rows, std::uint32_t query_max,
    std::uint32_t group_base, float attention_scale) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t k_half = lane >> 4U;
    F32x8 accum = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint32_t chunk = 0U; chunk < Stride * kHeadDim / 16U; ++chunk) {
        I16x8 a{};
        I16x8 b{};
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t flat = chunk * 16U + k_half * 8U + item;
            const std::uint32_t s = flat / kHeadDim;
            const std::uint32_t feature = flat % kHeadDim;
            const std::uint32_t relative = axis * Stride + Stride - 1U - s;
            if (relative < rows) {
                const std::size_t q_index =
                    (static_cast<std::size_t>(query_start + relative) * query_heads + query_head) *
                        kHeadDim + feature;
                a[item] = static_cast<short>(query[q_index].data);
            }
            const std::uint32_t group = group_base + axis;
            const std::size_t logical = static_cast<std::size_t>(group) * Stride + s;
            if (logical < context && logical <= query_max) {
                const std::size_t k_index =
                    (static_cast<std::size_t>(kv_head) * context + logical) * kHeadDim + feature;
                b[item] = static_cast<short>(packed_keys[k_index].data);
            }
        }
        accum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, accum);
    }
    return accum * (attention_scale / static_cast<float>(Stride));
#else
    (void)query; (void)packed_keys; (void)context; (void)query_heads; (void)query_head;
    (void)kv_head; (void)query_start; (void)rows; (void)query_max; (void)group_base;
    (void)attention_scale;
    return F32x8{};
#endif
}

template <std::uint32_t Stride>
__device__ float xattention_diagonal_score(
    const hip_bfloat16* query, const hip_bfloat16* packed_keys, std::size_t context,
    std::uint32_t query_heads, std::uint32_t query_head, std::uint32_t kv_head,
    std::uint32_t query_start, std::uint32_t rows, std::uint32_t first_position,
    std::uint32_t plane, float attention_scale) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t query_max = first_position +
        min(plane * Stride + Stride - 1U, rows - 1U);
    const std::uint32_t key_group = query_max / Stride;
    float dot = 0.0F;
    for (std::uint32_t flat = lane; flat < Stride * kHeadDim; flat += 32U) {
        const std::uint32_t s = flat / kHeadDim;
        const std::uint32_t feature = flat % kHeadDim;
        const std::uint32_t relative = plane * Stride + Stride - 1U - s;
        const std::size_t logical = static_cast<std::size_t>(key_group) * Stride + s;
        if (relative >= rows || logical >= context || logical > query_max) continue;
        const std::size_t q_index =
            (static_cast<std::size_t>(query_start + relative) * query_heads + query_head) *
                kHeadDim + feature;
        const std::size_t k_index =
            (static_cast<std::size_t>(kv_head) * context + logical) * kHeadDim + feature;
        dot += static_cast<float>(query[q_index]) * static_cast<float>(packed_keys[k_index]);
    }
    dot = wave_sum(dot);
    return __shfl(dot, 0, 32) * (attention_scale / static_cast<float>(Stride));
}

__device__ void atomic_max_finite(float* address, float value) {
    auto* bits = reinterpret_cast<unsigned*>(address);
    unsigned observed = atomicCAS(bits, 0U, 0U);
    while (__uint_as_float(observed) < value) {
        const unsigned prior = atomicCAS(bits, observed, __float_as_uint(value));
        if (prior == observed) break;
        observed = prior;
    }
}

template <std::uint32_t Stride>
__global__ void xattention_rank_kernel(
    const hip_bfloat16* query, const hip_bfloat16* packed_keys, std::uint32_t q_to_kv,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_count,
    std::size_t context, std::uint32_t query_heads,
    std::uint32_t query_rows, const std::int32_t* row_positions, float attention_scale,
    float* all_mass, float* all_plane_mass, std::uint16_t* all_ids,
    std::uint8_t* all_selected,
    std::int32_t* all_counts, std::uint32_t rank_capacity, std::uint32_t query_blocks) {
    const std::uint32_t query_head = blockIdx.x;
    const std::uint32_t query_block = blockIdx.y;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t wave = tid >> 5U;
    const std::size_t slot = static_cast<std::size_t>(query_head) * query_blocks + query_block;
    float* mass = all_mass + slot * rank_capacity;
    float* plane_mass = all_plane_mass + slot * kEstimatorPlanes * rank_capacity;
    const std::uint32_t keep_stride = ceil_div64(context);
    std::uint16_t* ids = all_ids + slot * keep_stride;
    std::uint8_t* selected = all_selected + slot * rank_capacity;
    __shared__ float plane_max[kEstimatorPlanes];
    __shared__ float plane_denominator[kEstimatorPlanes];
    __shared__ float diagonal_score[kEstimatorPlanes];
    __shared__ std::uint32_t invalid;

    if (tid == 0U) invalid = 0U;
    for (std::uint32_t page = tid; page < rank_capacity; page += blockDim.x) {
        mass[page] = 0.0F;
        ids[page] = static_cast<std::uint16_t>(page);
        selected[page] = 0U;
    }
    for (std::uint32_t index = tid; index < kEstimatorPlanes * rank_capacity;
         index += blockDim.x) plane_mass[index] = 0.0F;
    __syncthreads();

    if (query_head >= query_heads || query_block >= query_blocks) return;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = page_table_row[0];
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            if (tid == 0U) all_counts[slot] = -1;
            return;
        }
    }

    const std::uint32_t query_start = query_block * kXAttentionQueryBlockRows;
    const std::uint32_t rows = min(kXAttentionQueryBlockRows, query_rows - query_start);
    const std::int32_t first_position = row_positions[query_start];
    if (tid == 0U && query_start != 0U &&
        (first_position <= 0 || row_positions[query_start - 1U] != first_position - 1)) {
        atomicOr(&invalid, 1U);
    }
    for (std::uint32_t row = tid; row < rows; row += blockDim.x) {
        const std::int64_t expected = static_cast<std::int64_t>(first_position) + row;
        if (first_position < 0 || expected < 0 ||
            expected >= static_cast<std::int64_t>(context) ||
            row_positions[query_start + row] != expected) {
            atomicOr(&invalid, 1U);
        }
    }
    __syncthreads();
    const std::uint32_t last_position =
        first_position < 0 ? 0U : static_cast<std::uint32_t>(first_position) + rows - 1U;
    const std::uint32_t key_pages = last_position / kXAttentionPageSize + 1U;
    const std::uint32_t find_blocks = last_position / kXAttentionFindBlockSize + 1U;
    if (invalid != 0U || key_pages > keep_stride || find_blocks > rank_capacity) {
        if (tid == 0U) all_counts[slot] = -1;
        return;
    }

    // Tau one deliberately bypasses every estimator arithmetic path. Its ascending page list is
    // the dense identity and the consumer visits tokens in exactly the dense order.
    if constexpr (kXAttentionTauPermille == 1000U) {
        for (std::uint32_t page = tid; page < key_pages; page += blockDim.x) {
            ids[page] = static_cast<std::uint16_t>(page);
        }
        if (tid == 0U) all_counts[slot] = static_cast<std::int32_t>(key_pages);
        return;
    }
    // Algorithm 1's query and key slices share one global stride origin. Product prefill chunks
    // may begin at an arbitrary token, so an unaligned suffix cannot be reanchored locally without
    // changing the keep set solely because of chunk partitioning. Preserve exact attention by
    // emitting the full causal page list for that chunk.
    if (static_cast<std::uint32_t>(first_position) % kXAttentionFindBlockSize != 0U) {
        for (std::uint32_t page = tid; page < key_pages; page += blockDim.x) {
            ids[page] = static_cast<std::uint16_t>(page);
        }
        if (tid == 0U) all_counts[slot] = static_cast<std::int32_t>(key_pages);
        return;
    }

    // Algorithm 1 zero-pads short final reshaped Q/K rows instead of discarding them.
    const std::uint32_t planes = (rows + Stride - 1U) / Stride;
    const std::uint32_t key_groups = (last_position + Stride) / Stride;
    if (tid < planes) {
        plane_max[tid] = -INFINITY;
        plane_denominator[tid] = 0.0F;
    }
    __syncthreads();
    const std::uint32_t kv_head = query_head / q_to_kv;
    for (std::uint32_t plane_base = 0U; plane_base < planes; plane_base += kWaves) {
        const std::uint32_t plane = plane_base + wave;
        if (plane < planes) {
            const float score = xattention_diagonal_score<Stride>(
                query, packed_keys, context, query_heads, query_head, kv_head, query_start,
                rows, static_cast<std::uint32_t>(first_position), plane, attention_scale);
            if (lane == 0U) diagonal_score[plane] = score;
        }
    }
    __syncthreads();
    for (std::uint32_t group_base = wave * 16U; group_base < key_groups;
         group_base += kWaves * 16U) {
        const F32x8 scores = xattention_score_wmma_tile<Stride>(
            query, packed_keys, context, query_heads, query_head, kv_head, query_start, rows,
            last_position, group_base, attention_scale);
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t plane = (lane >> 4U) * 8U + item;
            const std::uint32_t key_group = group_base + (lane & 15U);
            const std::uint32_t query_max = static_cast<std::uint32_t>(first_position) +
                min(plane * Stride + Stride - 1U, rows - 1U);
            if (plane >= planes || key_group >= key_groups || key_group * Stride > query_max)
                continue;
            const float score = key_group == query_max / Stride
                ? diagonal_score[plane] : scores[item];
            if (!isfinite(score)) atomicOr(&invalid, 1U);
            else atomic_max_finite(&plane_max[plane], score);
        }
    }
    __syncthreads();
    for (std::uint32_t group_base = wave * 16U; group_base < key_groups;
         group_base += kWaves * 16U) {
        const F32x8 scores = xattention_score_wmma_tile<Stride>(
            query, packed_keys, context, query_heads, query_head, kv_head, query_start, rows,
            last_position, group_base, attention_scale);
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t plane = (lane >> 4U) * 8U + item;
            const std::uint32_t key_group = group_base + (lane & 15U);
            const std::uint32_t query_max = static_cast<std::uint32_t>(first_position) +
                min(plane * Stride + Stride - 1U, rows - 1U);
            if (plane >= planes || key_group >= key_groups || key_group * Stride > query_max)
                continue;
            const float score = key_group == query_max / Stride
                ? diagonal_score[plane] : scores[item];
            const float weight = __expf(score - plane_max[plane]);
            if (!isfinite(weight)) {
                atomicOr(&invalid, 1U);
            } else {
                atomicAdd(&plane_denominator[plane], weight);
                atomicAdd(&plane_mass[plane * rank_capacity +
                                      (key_group * Stride) / kXAttentionFindBlockSize], weight);
            }
        }
    }
    __syncthreads();
    for (std::uint32_t plane = tid; plane < planes; plane += blockDim.x) {
        if (!(plane_denominator[plane] > 0.0F) || !isfinite(plane_denominator[plane]))
            atomicOr(&invalid, 1U);
    }
    __syncthreads();
    for (std::uint32_t block = tid; block < find_blocks; block += blockDim.x) {
        float normalized = 0.0F;
        for (std::uint32_t plane = 0U; plane < planes; ++plane) {
            normalized += plane_mass[plane * rank_capacity + block] /
                          plane_denominator[plane];
        }
        mass[block] = normalized;
    }
    __syncthreads();
    if (invalid != 0U) {
        if (tid == 0U) all_counts[slot] = -1;
        return;
    }

    for (std::uint32_t page = tid; page < rank_capacity; page += blockDim.x) {
        if (page >= find_blocks) {
            mass[page] = -INFINITY;
            ids[page] = 0xffffU;
        }
    }
    __syncthreads();

    // Stable descending bitonic rank: equal masses choose the lower logical page. The final list
    // is subsequently compacted in ascending page order, independently of this ranking order.
    for (std::uint32_t width = 2U; width <= rank_capacity; width <<= 1U) {
        for (std::uint32_t distance = width >> 1U; distance != 0U; distance >>= 1U) {
            for (std::uint32_t index = tid; index < rank_capacity; index += blockDim.x) {
                const std::uint32_t other = index ^ distance;
                if (other <= index) continue;
                const bool descending = (index & width) == 0U;
                const bool index_better =
                    mass[index] > mass[other] ||
                    (mass[index] == mass[other] && ids[index] < ids[other]);
                if (index_better != descending) {
                    const float score = mass[index];
                    mass[index] = mass[other];
                    mass[other] = score;
                    const std::uint16_t id = ids[index];
                    ids[index] = ids[other];
                    ids[other] = id;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0U) {
        float total = 0.0F;
        for (std::uint32_t index = 0U; index < find_blocks; ++index) total += mass[index];
        const float required = total * kXAttentionTau;
        float retained = 0.0F;
        // The reference causal selector accounts for mandatory sink and self/recent blocks
        // inside the threshold budget.  Selecting to tau first and unioning them afterwards
        // needlessly over-retains pages, especially for diffuse heads.
        const std::uint32_t recent_first =
            static_cast<std::uint32_t>(first_position) / kXAttentionFindBlockSize;
        const std::uint32_t recent_last = last_position / kXAttentionFindBlockSize;
        for (std::uint32_t index = 0U; index < find_blocks; ++index) {
            const std::uint32_t page = ids[index];
            if (page == 0U || (page >= recent_first && page <= recent_last)) {
                if (selected[page] == 0U) {
                    selected[page] = 1U;
                    retained += mass[index];
                }
            }
        }
        for (std::uint32_t index = 0U; index < find_blocks && retained < required; ++index) {
            if (ids[index] == 0xffffU || !(mass[index] > 0.0F)) break;
            if (selected[ids[index]] != 0U) continue;
            selected[ids[index]] = 1U;
            retained += mass[index];
        }
        std::int32_t count = 0;
        for (std::uint32_t block = 0U; block < find_blocks; ++block) {
            if (selected[block] == 0U) continue;
            const std::uint32_t first_page = block * 2U;
            ids[count++] = static_cast<std::uint16_t>(first_page);
            if (first_page + 1U < key_pages) {
                ids[count++] = static_cast<std::uint16_t>(first_page + 1U);
            }
        }
        all_counts[slot] = count;
    }
}

struct SparseAttentionShared {
    float wave_reductions[kWaves];
    float running_max;
    float denominator;
    float score;
    float rescale;
    float probability_numerator;
};

inline constexpr std::uint32_t kConsumerQueryTile = 16U;
inline constexpr std::uint32_t kConsumerKeyTile = 16U;

struct FlashSparseAttentionShared {
    float probability[kConsumerQueryTile][kConsumerKeyTile];
    float running_max[kConsumerQueryTile];
    float denominator[kConsumerQueryTile];
    float alpha[kConsumerQueryTile];
    std::size_t physical[kConsumerKeyTile];
    std::uint32_t invalid;
};

[[maybe_unused]] __device__ __forceinline__ float subgroup_max16(float value) {
#pragma unroll
    for (std::uint32_t offset = 8U; offset != 0U; offset >>= 1U) {
        value = fmaxf(value, __shfl_xor(value, offset, 16));
    }
    return value;
}

[[maybe_unused]] __device__ __forceinline__ float subgroup_sum16(float value) {
#pragma unroll
    for (std::uint32_t offset = 8U; offset != 0U; offset >>= 1U) {
        value += __shfl_xor(value, offset, 16);
    }
    return value;
}

template <std::uint32_t Group, bool KToken, bool VToken, bool SToken>
__global__ void xattention_consumer_kernel(
    const hip_bfloat16* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint16_t* scales, const std::uint32_t* page_table,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context, std::size_t physical_tokens,
    std::uint32_t q_to_kv, std::uint32_t row_offset,
    const std::int32_t* row_positions, float attention_scale,
    const std::uint16_t* keep_pages, const std::int32_t* keep_count,
    std::uint32_t keep_stride, std::uint32_t query_blocks, float* output) {
    __shared__ SparseAttentionShared shared;
    const std::uint32_t row = row_offset + blockIdx.y;
    const std::uint32_t query_head = blockIdx.x;
    const std::uint32_t feature = threadIdx.x;
    const std::size_t row_base =
        (static_cast<std::size_t>(row) * gridDim.x + query_head) * kHeadDim;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = page_table_row[0];
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            output[row_base + feature] = __builtin_nanf("");
            return;
        }
        page_table += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    const std::int32_t position = row_positions[row];
    const std::uint32_t query_block = row / kXAttentionQueryBlockRows;
    const std::size_t slot = static_cast<std::size_t>(query_head) * query_blocks + query_block;
    const std::int32_t count = keep_count[slot];
    if (position < 0 || static_cast<std::size_t>(position) >= context || count <= 0 ||
        static_cast<std::uint32_t>(count) > keep_stride) {
        output[row_base + feature] = __builtin_nanf("");
        return;
    }
    const std::uint16_t* pages = keep_pages + slot * keep_stride;
    const std::uint32_t kv_head = query_head / q_to_kv;
    if (feature == 0U) {
        shared.running_max = -INFINITY;
        shared.denominator = 0.0F;
    }
    const float q = static_cast<float>(query[row_base + feature]);
    float numerator = 0.0F;
    __syncthreads();
    for (std::int32_t kept = 0; kept < count; ++kept) {
        const std::uint32_t logical_page = pages[kept];
        const std::uint32_t first = logical_page * kXAttentionPageSize;
        if (first > static_cast<std::uint32_t>(position)) break;
        const std::uint32_t end = min(first + kXAttentionPageSize,
                                      static_cast<std::uint32_t>(position) + 1U);
        for (std::uint32_t logical = first; logical < end; ++logical) {
            const std::size_t physical =
                static_cast<std::size_t>(page_table[logical_page]) * kXAttentionPageSize +
                logical % kXAttentionPageSize;
            if (physical >= physical_tokens) {
                output[row_base + feature] = __builtin_nanf("");
                return;
            }
            float partial = q * __half2float(__half(__hip_cvt_fp8_to_halfraw(
                keys[key_offset<KToken>(physical, kv_head, feature, gridDim.x / q_to_kv,
                                        physical_tokens)],
                __HIP_E4M3)));
            partial = wave_sum(partial);
            if ((feature & 31U) == 0U) shared.wave_reductions[feature / 32U] = partial;
            __syncthreads();
            if (feature < kWaves) {
                partial = shared.wave_reductions[feature];
                for (std::uint32_t delta = 4U; delta != 0U; delta >>= 1U) {
                    partial += __shfl_down(partial, delta, 8);
                }
            }
            if (feature == 0U) {
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
                physical, kv_head, feature, gridDim.x / q_to_kv, physical_tokens, false)];
            int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
            code = code >= 8 ? code - 16 : code;
            const std::uint16_t scale_bits = scales[int4_offset<Group, VToken, SToken>(
                physical, kv_head, feature, gridDim.x / q_to_kv, physical_tokens, true)];
            numerator = numerator * shared.rescale +
                        shared.probability_numerator * static_cast<float>(code) *
                            __half2float(__ushort_as_half(scale_bits));
            __syncthreads();
        }
    }
    output[row_base + feature] = numerator / shared.denominator;
}

// The selected tau<1 consumer is a B16-query sparse FlashAttention-style tile.  Wave zero forms
// each B16xB16 QK tile with native gfx1201 BF16 WMMA from the estimator's reusable logical K pack.
// All 256 lanes then own one output feature and sixteen FP32 row accumulators, so each direct
// INT4-V/FP16-scale load is reused across all B16 queries.  Online FP32 softmax retains increasing
// logical-key order and needs no global score workspace.  Tau one keeps the serial consumer because
// its qualification contract is bit identity with the dense token-order implementation.
template <std::uint32_t Group, bool VToken, bool SToken>
__global__ __launch_bounds__(kThreads) void xattention_flash_consumer_kernel(
    const hip_bfloat16* query, const hip_bfloat16* packed_keys,
    const std::uint8_t* values, const std::uint16_t* scales,
    const std::uint32_t* page_table, const std::int32_t* page_table_row,
    std::uint32_t page_table_row_stride, std::uint32_t page_table_row_count,
    std::size_t context, std::size_t physical_tokens, std::uint32_t kv_heads,
    std::uint32_t q_to_kv, std::uint32_t total_query_rows,
    const std::int32_t* row_positions, float attention_scale,
    const std::uint16_t* keep_pages, const std::int32_t* keep_count,
    std::uint32_t keep_stride, std::uint32_t query_blocks, float* output) {
    __shared__ FlashSparseAttentionShared shared;
    const std::uint32_t query_head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave = tid >> 5U;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t half = lane >> 4U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t query_base = blockIdx.y * kConsumerQueryTile;
    const std::uint32_t query_rows = min(kConsumerQueryTile, total_query_rows - query_base);
    const std::uint32_t feature = tid;
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = page_table_row[0];
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            for (std::uint32_t row = 0U; row < query_rows; ++row) {
                output[((static_cast<std::size_t>(query_base + row) * gridDim.x + query_head) *
                        kHeadDim) + feature] = __builtin_nanf("");
            }
            return;
        }
        page_table += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }
    const std::uint32_t query_block = query_base / kXAttentionQueryBlockRows;
    const std::size_t slot = static_cast<std::size_t>(query_head) * query_blocks + query_block;
    const std::int32_t count = keep_count[slot];
    const std::int32_t last_position = row_positions[query_base + query_rows - 1U];
    if (query_rows == 0U || last_position < 0 ||
        static_cast<std::size_t>(last_position) >= context || count <= 0 ||
        static_cast<std::uint32_t>(count) > keep_stride) {
        for (std::uint32_t row = 0U; row < query_rows; ++row) {
            output[((static_cast<std::size_t>(query_base + row) * gridDim.x + query_head) *
                    kHeadDim) + feature] = __builtin_nanf("");
        }
        return;
    }
    const std::uint16_t* pages = keep_pages + slot * keep_stride;
    const std::uint32_t kv_head = query_head / q_to_kv;
    if (tid < kConsumerQueryTile) {
        shared.running_max[tid] = -INFINITY;
        shared.denominator[tid] = 0.0F;
    }
    float numerator[kConsumerQueryTile]{};
    __syncthreads();
    for (std::int32_t kept = 0; kept < count; ++kept) {
        const std::uint32_t logical_page = pages[kept];
        const std::uint32_t first = logical_page * kXAttentionPageSize;
        if (first > static_cast<std::uint32_t>(last_position)) break;
        const std::uint32_t end = min(first + kXAttentionPageSize,
                                      static_cast<std::uint32_t>(last_position) + 1U);
        for (std::uint32_t key_base = first; key_base < end;
             key_base += kConsumerKeyTile) {
            const std::uint32_t key_rows = min(kConsumerKeyTile, end - key_base);
            if (tid == 0U) {
                shared.invalid = 0U;
                for (std::uint32_t item = 0U; item < key_rows; ++item) {
                    const std::uint32_t logical = key_base + item;
                    const std::size_t physical =
                        static_cast<std::size_t>(page_table[logical_page]) *
                            kXAttentionPageSize + logical % kXAttentionPageSize;
                    shared.physical[item] = physical;
                    if (physical >= physical_tokens) shared.invalid = 1U;
                }
            }
            __syncthreads();
            if (shared.invalid != 0U) {
                for (std::uint32_t row = 0U; row < query_rows; ++row) {
                    output[((static_cast<std::size_t>(query_base + row) * gridDim.x + query_head) *
                            kHeadDim) + feature] = __builtin_nanf("");
                }
                return;
            }

            F32x8 score{};
            if (wave == 0U) {
#pragma unroll
                for (std::uint32_t feature_tile = 0U; feature_tile < kHeadDim / 16U;
                     ++feature_tile) {
                    I16x8 q_fragment{};
                    I16x8 k_fragment{};
#pragma unroll
                    for (std::uint32_t item = 0U; item < 8U; ++item) {
                        const std::uint32_t dimension =
                            feature_tile * 16U + half * 8U + item;
                        const std::uint32_t query_row = query_base + axis;
                        if (axis < query_rows) {
                            q_fragment[item] = static_cast<short>(query[
                                (static_cast<std::size_t>(query_row) * gridDim.x + query_head) *
                                    kHeadDim + dimension].data);
                        }
                        if (axis < key_rows) {
                            k_fragment[item] = static_cast<short>(packed_keys[
                                (static_cast<std::size_t>(kv_head) * context + key_base + axis) *
                                    kHeadDim + dimension].data);
                        }
                    }
                    score = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                        q_fragment, k_fragment, score);
                }

#pragma unroll
                for (std::uint32_t item = 0U; item < 8U; ++item) {
                    const std::uint32_t query_row = half * 8U + item;
                    const bool valid = query_row < query_rows && axis < key_rows &&
                        key_base + axis <=
                            static_cast<std::uint32_t>(row_positions[query_base + query_row]);
                    const float scaled = valid ? score[item] * attention_scale : -INFINITY;
                    const float block_max = subgroup_max16(scaled);
                    const float next_max = fmaxf(shared.running_max[query_row], block_max);
                    const float alpha = isfinite(shared.running_max[query_row])
                                            ? __expf(shared.running_max[query_row] - next_max)
                                            : 0.0F;
                    const float probability = valid ? __expf(scaled - next_max) : 0.0F;
                    const float block_sum = subgroup_sum16(probability);
                    shared.probability[query_row][axis] = probability;
                    if (axis == 0U) {
                        shared.alpha[query_row] = alpha;
                        shared.denominator[query_row] =
                            shared.denominator[query_row] * alpha + block_sum;
                        shared.running_max[query_row] = next_max;
                    }
                }
            }
            __syncthreads();

#pragma unroll
            for (std::uint32_t row = 0U; row < kConsumerQueryTile; ++row) {
                numerator[row] *= shared.alpha[row];
            }
            for (std::uint32_t item = 0U; item < key_rows; ++item) {
                const std::size_t physical = shared.physical[item];
                const std::uint8_t packed = values[int4_offset<Group, VToken, SToken>(
                    physical, kv_head, feature, kv_heads, physical_tokens, false)];
                int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
                code = code >= 8 ? code - 16 : code;
                const std::uint16_t scale_bits = scales[int4_offset<Group, VToken, SToken>(
                    physical, kv_head, feature, kv_heads, physical_tokens, true)];
                const float represented_value = static_cast<float>(code) *
                    __half2float(__ushort_as_half(scale_bits));
#pragma unroll
                for (std::uint32_t row = 0U; row < kConsumerQueryTile; ++row) {
                    numerator[row] = fmaf(shared.probability[row][item], represented_value,
                                          numerator[row]);
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (std::uint32_t row = 0U; row < kConsumerQueryTile; ++row) {
        if (row < query_rows) {
            output[((static_cast<std::size_t>(query_base + row) * gridDim.x + query_head) *
                    kHeadDim) + feature] = numerator[row] / shared.denominator[row];
        }
    }
}

template <std::uint32_t Group, bool KToken, bool VToken, bool SToken>
hipError_t launch_consumer(const Fp8Int4KvAttentionArgs& a, const XAttentionWorkspaceView& work,
                           hipStream_t stream) {
    for (std::uint32_t row_offset = 0U; row_offset < a.query_rows;) {
        const std::uint32_t rows = min(kMaximumGridY, a.query_rows - row_offset);
        hipLaunchKernelGGL((xattention_consumer_kernel<Group, KToken, VToken, SToken>),
                           dim3(a.query_heads, rows), dim3(kThreads), 0, stream, a.query,
                           a.key_codes, a.value_codes, a.value_scale_bits, a.page_table,
                           a.page_table_row, a.page_table_row_stride, a.page_table_row_count,
                           a.context, a.physical_tokens, a.q_to_kv, row_offset,
                           a.row_positions, a.attention_scale, work.rank_and_keep,
                           work.keep_count, work.keep_stride, work.query_blocks, a.output);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
        row_offset += rows;
    }
    return hipSuccess;
}

template <std::uint32_t Group>
hipError_t dispatch_consumer(const Fp8Int4KvAttentionArgs& a,
                             const XAttentionWorkspaceView& work, hipStream_t stream) {
    const unsigned layout =
        (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
        (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U) |
        (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 4U : 0U);
    switch (layout) {
    case 0: return launch_consumer<Group, false, false, false>(a, work, stream);
    case 1: return launch_consumer<Group, true, false, false>(a, work, stream);
    case 2: return launch_consumer<Group, false, true, false>(a, work, stream);
    case 3: return launch_consumer<Group, true, true, false>(a, work, stream);
    case 4: return launch_consumer<Group, false, false, true>(a, work, stream);
    case 5: return launch_consumer<Group, true, false, true>(a, work, stream);
    case 6: return launch_consumer<Group, false, true, true>(a, work, stream);
    default: return launch_consumer<Group, true, true, true>(a, work, stream);
    }
}

template <std::uint32_t Group, bool VToken, bool SToken>
hipError_t launch_flash_consumer(const Fp8Int4KvAttentionArgs& a,
                                 const XAttentionWorkspaceView& work, hipStream_t stream) {
    const std::uint32_t query_tiles =
        (a.query_rows + kConsumerQueryTile - 1U) / kConsumerQueryTile;
    hipLaunchKernelGGL((xattention_flash_consumer_kernel<Group, VToken, SToken>),
                       dim3(a.query_heads, query_tiles), dim3(kThreads), 0, stream,
                       a.query, work.packed_keys, a.value_codes, a.value_scale_bits, a.page_table,
                       a.page_table_row, a.page_table_row_stride, a.page_table_row_count,
                       a.context, a.physical_tokens, a.kv_heads, a.q_to_kv, a.query_rows,
                       a.row_positions,
                       a.attention_scale, work.rank_and_keep, work.keep_count, work.keep_stride,
                       work.query_blocks, a.output);
    return hipGetLastError();
}

template <std::uint32_t Group>
hipError_t dispatch_flash_consumer(const Fp8Int4KvAttentionArgs& a,
                                   const XAttentionWorkspaceView& work,
                                   hipStream_t stream) {
    const unsigned layout =
        (a.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
        (a.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U);
    switch (layout) {
    case 0: return launch_flash_consumer<Group, false, false>(a, work, stream);
    case 1: return launch_flash_consumer<Group, true, false>(a, work, stream);
    case 2: return launch_flash_consumer<Group, false, true>(a, work, stream);
    default: return launch_flash_consumer<Group, true, true>(a, work, stream);
    }
}

template <std::uint32_t Stride>
hipError_t launch_rank(const Fp8Int4KvAttentionArgs& a, const XAttentionWorkspaceView& work,
                       hipStream_t stream) {
    if constexpr (kXAttentionTauPermille != 1000U) {
        const dim3 pack_grid(a.kv_heads, ceil_div64(a.context));
        if (a.key_layout == Fp8Int4KvPlaneLayout::TokenFastest) {
            hipLaunchKernelGGL((xattention_pack_keys_kernel<Stride, true>), pack_grid,
                               dim3(kThreads), 0, stream, a.key_codes, a.page_table,
                               a.page_table_row, a.page_table_row_stride, a.page_table_row_count,
                               a.context, a.physical_tokens, a.kv_heads, work.packed_keys);
        } else {
            hipLaunchKernelGGL((xattention_pack_keys_kernel<Stride, false>), pack_grid,
                               dim3(kThreads), 0, stream, a.key_codes, a.page_table,
                               a.page_table_row, a.page_table_row_stride, a.page_table_row_count,
                               a.context, a.physical_tokens, a.kv_heads, work.packed_keys);
        }
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
    }
    hipLaunchKernelGGL((xattention_rank_kernel<Stride>),
                       dim3(a.query_heads, work.query_blocks), dim3(kThreads), 0, stream,
                       a.query, work.packed_keys, a.q_to_kv, a.page_table_row,
                       a.page_table_row_count, a.context,
                       a.query_heads, a.query_rows, a.row_positions, a.attention_scale,
                       work.page_mass, work.plane_mass,
                       work.rank_and_keep, work.selected, work.keep_count, work.rank_capacity,
                       work.query_blocks);
    return hipGetLastError();
}

[[nodiscard]] bool validate_and_bind(const Fp8Int4KvXAttentionArgs& args,
                                     XAttentionWorkspaceView& work) noexcept {
    const auto& a = args.attention;
    if (!a.query || !a.key_codes || !a.value_codes || !a.value_scale_bits || !a.page_table ||
        !a.page_table_pages || !a.context || !a.physical_tokens ||
        a.physical_tokens % kXAttentionPageSize || !a.query_heads || !a.kv_heads ||
        !a.q_to_kv || a.query_heads != a.kv_heads * a.q_to_kv || a.query_rows < 2U ||
        !a.row_positions || !a.output || !std::isfinite(a.attention_scale) ||
        !(a.attention_scale > 0.0F) || !valid_plane_layout(a.key_layout) ||
        !valid_plane_layout(a.value_layout) || !valid_plane_layout(a.value_scale_layout) ||
        a.ancestor_masks != nullptr || a.prefix_lengths != nullptr ||
        a.active_query_rows != nullptr || a.score_workspace != nullptr ||
        (a.page_table_row != nullptr &&
         (a.page_table_row_stride < a.page_table_pages || a.page_table_row_count == 0U)) ||
        a.page_table_pages < ceil_div64(a.context) ||
        (a.value_group != 16U && a.value_group != 32U)) {
        return false;
    }
    work = bind_fp8_int4_kv_xattention_workspace(
        args.workspace, args.workspace_bytes, a.query_heads, a.kv_heads, a.query_rows, a.context);
    return work.packed_keys && work.page_mass && work.plane_mass && work.rank_and_keep && work.selected &&
           work.keep_count;
}

} // namespace

std::size_t fp8_int4_kv_xattention_workspace_capacity_bytes(
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t query_rows,
    std::size_t context) noexcept {
    if (query_heads == 0U || kv_heads == 0U || query_rows == 0U || context == 0U ||
        context > 262144U) return 0U;
    const std::uint32_t query_blocks = query_rows / kXAttentionQueryBlockRows +
        (query_rows % kXAttentionQueryBlockRows != 0U ? 1U : 0U);
    const std::uint32_t rank_capacity = next_power_of_two(
        (static_cast<std::uint32_t>(context) + kXAttentionFindBlockSize - 1U) /
        kXAttentionFindBlockSize);
    const std::uint32_t keep_stride = ceil_div64(context);
    if (rank_capacity == 0U) return 0U;
    const std::size_t slots = static_cast<std::size_t>(query_heads) * query_blocks;
    if (slots > std::numeric_limits<std::size_t>::max() / rank_capacity ||
        slots > std::numeric_limits<std::size_t>::max() / keep_stride) return 0U;
    const std::size_t rank_elements = slots * rank_capacity;
    if (rank_elements > std::numeric_limits<std::size_t>::max() / kEstimatorPlanes ||
        context > std::numeric_limits<std::size_t>::max() / kv_heads / kHeadDim) return 0U;
    const std::size_t plane_elements = rank_elements * kEstimatorPlanes;
    const std::size_t packed_elements = static_cast<std::size_t>(kv_heads) * context * kHeadDim;
    const std::size_t keep_elements = slots * keep_stride;
    std::size_t total = 0U;
    if (!checked_add_elements<hip_bfloat16>(total, packed_elements) ||
        !checked_add_elements<float>(total, rank_elements) ||
        !checked_add_elements<float>(total, plane_elements) ||
        !checked_add_elements<std::uint16_t>(total, keep_elements) ||
        !checked_add_elements<std::uint8_t>(total, rank_elements) ||
        !checked_add_elements<std::int32_t>(total, slots) || !align256(total)) {
        return 0U;
    }
    return total;
}

XAttentionWorkspaceView bind_fp8_int4_kv_xattention_workspace(
    void* workspace, std::size_t workspace_bytes, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t query_rows, std::size_t context) noexcept {
    XAttentionWorkspaceView result{};
    const std::size_t required = fp8_int4_kv_xattention_workspace_capacity_bytes(
        query_heads, kv_heads, query_rows, context);
    if (workspace == nullptr || reinterpret_cast<std::uintptr_t>(workspace) % 256U != 0U ||
        required == 0U || workspace_bytes != required) {
        return result;
    }
    result.query_blocks = query_rows / kXAttentionQueryBlockRows +
        (query_rows % kXAttentionQueryBlockRows != 0U ? 1U : 0U);
    result.rank_capacity = next_power_of_two(
        (static_cast<std::uint32_t>(context) + kXAttentionFindBlockSize - 1U) /
        kXAttentionFindBlockSize);
    result.keep_stride = ceil_div64(context);
    const std::size_t slots = static_cast<std::size_t>(query_heads) * result.query_blocks;
    const std::size_t rank_elements = slots * result.rank_capacity;
    const std::size_t plane_elements = rank_elements * kEstimatorPlanes;
    const std::size_t packed_elements = static_cast<std::size_t>(kv_heads) * context * kHeadDim;
    const std::size_t keep_elements = slots * result.keep_stride;
    auto* base = static_cast<std::uint8_t*>(workspace);
    std::size_t offset = 0U;
    result.packed_keys = take<hip_bfloat16>(base, offset, packed_elements);
    result.page_mass = take<float>(base, offset, rank_elements);
    result.plane_mass = take<float>(base, offset, plane_elements);
    result.rank_and_keep = take<std::uint16_t>(base, offset, keep_elements);
    result.selected = take<std::uint8_t>(base, offset, rank_elements);
    result.keep_count = take<std::int32_t>(base, offset, slots);
    return result;
}

hipError_t fp8_int4_kv_xattention_prefill(const Fp8Int4KvXAttentionArgs& args,
                                           hipStream_t stream) noexcept {
    hipError_t error = fp8_int4_kv_xattention_rank_qualification(args, stream);
    if (error != hipSuccess) return error;
    return fp8_int4_kv_xattention_consumer_qualification(args, stream);
}

hipError_t fp8_int4_kv_xattention_rank_qualification(
    const Fp8Int4KvXAttentionArgs& args, hipStream_t stream) noexcept {
    const auto& a = args.attention;
    XAttentionWorkspaceView work;
    if (!validate_and_bind(args, work)) return hipErrorInvalidValue;
    hipError_t error;
    if constexpr (kXAttentionStride == 8U) {
        error = launch_rank<8U>(a, work, stream);
    } else {
        error = launch_rank<16U>(a, work, stream);
    }
    return error;
}

hipError_t fp8_int4_kv_xattention_consumer_qualification(
    const Fp8Int4KvXAttentionArgs& args, hipStream_t stream) noexcept {
    const auto& a = args.attention;
    XAttentionWorkspaceView work;
    if (!validate_and_bind(args, work)) return hipErrorInvalidValue;
    if constexpr (kXAttentionTauPermille == 1000U) {
        return a.value_group == 16U ? dispatch_consumer<16U>(a, work, stream)
                                    : dispatch_consumer<32U>(a, work, stream);
    } else {
        return a.value_group == 16U ? dispatch_flash_consumer<16U>(a, work, stream)
                                    : dispatch_flash_consumer<32U>(a, work, stream);
    }
}

} // namespace ninfer::ops::r9700::kv
