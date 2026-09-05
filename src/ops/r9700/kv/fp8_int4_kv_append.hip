#include "fp8_int4_kv_append.h"

#include <hip/hip_fp8.h>

namespace ninfer::ops::r9700::kv {
namespace {

inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kPageSize = 64;

[[nodiscard]] constexpr bool valid_plane_layout(Fp8Int4KvPlaneLayout layout) noexcept {
    return layout == Fp8Int4KvPlaneLayout::FeatureFastest ||
           layout == Fp8Int4KvPlaneLayout::TokenFastest;
}

template <std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest,
          bool ScaleTokenFastest>
struct PlaneOffset {
    static constexpr std::uint32_t kGroups = kHeadDim / Group;
    static constexpr std::uint32_t kBytesPerGroup = Group / 2;

    __device__ static std::size_t key(std::size_t token, std::uint32_t head,
                                      std::uint32_t feature, std::uint32_t heads,
                                      std::size_t physical_tokens) {
        const std::size_t page = token / kPageSize;
        const std::size_t in_page = token % kPageSize;
        const std::size_t pages = physical_tokens / kPageSize;
        if constexpr (KeyTokenFastest) {
            return in_page + kPageSize *
                                 (static_cast<std::size_t>(feature) + kHeadDim *
                                                                      (page + pages * head));
        }
        return feature + kHeadDim *
                             (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
    }

    __device__ static std::size_t value(std::size_t token, std::uint32_t head,
                                        std::uint32_t group, std::uint32_t byte,
                                        std::uint32_t heads, std::size_t physical_tokens) {
        const std::size_t page = token / kPageSize;
        const std::size_t in_page = token % kPageSize;
        const std::size_t pages = physical_tokens / kPageSize;
        const std::size_t feature = static_cast<std::size_t>(group) * kBytesPerGroup + byte;
        if constexpr (ValueTokenFastest) {
            return in_page + kPageSize *
                                 (feature + (kGroups * kBytesPerGroup) * (page + pages * head));
        }
        return feature + (kGroups * kBytesPerGroup) *
                             (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
    }

    __device__ static std::size_t scale(std::size_t token, std::uint32_t head,
                                        std::uint32_t group, std::uint32_t heads,
                                        std::size_t physical_tokens) {
        const std::size_t page = token / kPageSize;
        const std::size_t in_page = token % kPageSize;
        const std::size_t pages = physical_tokens / kPageSize;
        if constexpr (ScaleTokenFastest) {
            return in_page + kPageSize *
                                 (static_cast<std::size_t>(group) + kGroups * (page + pages * head));
        }
        return group + kGroups *
                           (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
    }
};

__device__ std::size_t physical_token(std::size_t logical, const std::uint32_t* page_table) {
    return static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
           logical % kPageSize;
}

template <std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest,
          bool ScaleTokenFastest>
__global__ void append_kernel(const hip_bfloat16* keys,
                              const hip_bfloat16* values,
                              const std::uint32_t* positions,
                              std::size_t tokens,
                              const std::int32_t* token_count,
                              std::uint32_t first_position,
                              const std::uint32_t* first_position_device,
                              std::uint32_t kv_heads,
                              const std::uint32_t* page_table,
                              std::uint32_t page_table_pages,
                              const std::int32_t* page_table_row,
                              std::uint32_t page_table_row_stride,
                              std::uint32_t page_table_row_count,
                              std::size_t physical_tokens,
                              std::uint8_t* key_codes,
                              std::uint8_t* value_codes,
                              std::uint16_t* value_scale_bits,
                              std::uint32_t* status) {
    const std::size_t item = blockIdx.x;
    const std::size_t input_token = item / kv_heads;
    const std::uint32_t head = static_cast<std::uint32_t>(item % kv_heads);
    const std::int32_t device_count = token_count == nullptr ? 0 : *token_count;
    if (token_count != nullptr &&
        (device_count < 0 || static_cast<std::size_t>(device_count) > tokens)) {
        if (item == 0 && threadIdx.x == 0) {
            atomicOr(status, Fp8Int4KvAppendInvalidCount);
        }
        return;
    }
    const std::size_t active_count =
        token_count == nullptr ? tokens : static_cast<std::size_t>(device_count);
    if (input_token >= active_count) {
        return;
    }
    if (page_table_row != nullptr) {
        const std::int32_t represented_row = *page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= page_table_row_count) {
            atomicOr(status, Fp8Int4KvAppendInvalidAddress);
            return;
        }
        page_table += static_cast<std::size_t>(represented_row) * page_table_row_stride;
    }

    const std::uint32_t feature = threadIdx.x;
    const std::size_t source = (input_token * kv_heads + head) * kHeadDim + feature;
    const std::uint32_t logical_position = positions[input_token];
    const std::uint32_t represented_first =
        first_position_device == nullptr ? first_position : *first_position_device;
    if (input_token > static_cast<std::size_t>(UINT32_MAX - represented_first) ||
        logical_position != represented_first + static_cast<std::uint32_t>(input_token)) {
        atomicOr(status, Fp8Int4KvAppendInvalidPosition);
        return;
    }
    if (logical_position / kPageSize >= page_table_pages) {
        atomicOr(status, Fp8Int4KvAppendInvalidAddress);
        return;
    }
    const std::size_t destination = physical_token(logical_position, page_table);
    if (destination >= physical_tokens) {
        atomicOr(status, Fp8Int4KvAppendInvalidAddress);
        return;
    }

    const std::uint32_t lane_in_group = feature % Group;
    const std::uint32_t group = feature / Group;
    const float value = static_cast<float>(values[source]);
    const float key = static_cast<float>(keys[source]);
    if (__ballot(!isfinite(value) || !isfinite(key)) != 0) {
        if (lane_in_group == 0) {
            atomicOr(status, Fp8Int4KvAppendNonfiniteInput);
        }
        return;
    }
    float maximum = fabsf(value);
    for (std::uint32_t delta = Group / 2; delta != 0; delta >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down(maximum, delta, Group));
    }
    maximum = __shfl(maximum, 0, Group);
    const __half stored_scale = __float2half(maximum / 7.0F);
    const float scale = __half2float(stored_scale);
    if (!isfinite(scale)) {
        if (lane_in_group == 0) {
            atomicOr(status, Fp8Int4KvAppendScaleOverflow);
        }
        return;
    }
    int code = scale == 0.0F ? 0 : __float2int_rn(value / scale);
    code = code < -7 ? -7 : (code > 7 ? 7 : code);

    key_codes[PlaneOffset<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>::key(
        destination, head, feature, kv_heads, physical_tokens)] =
        __hip_cvt_float_to_fp8(key, __HIP_SATFINITE, __HIP_E4M3);

    if (lane_in_group == 0) {
        value_scale_bits[PlaneOffset<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>::scale(
            destination, head, group, kv_heads, physical_tokens)] = __half_as_ushort(stored_scale);
    }
    const int adjacent = __shfl(code, lane_in_group ^ 1U, Group);
    if ((lane_in_group & 1U) == 0) {
        value_codes[PlaneOffset<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>::value(
            destination, head, group, lane_in_group / 2, kv_heads, physical_tokens)] =
            static_cast<std::uint8_t>((code & 0x0f) | ((adjacent & 0x0f) << 4));
    }
}

template <std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest,
          bool ScaleTokenFastest>
hipError_t launch(const Fp8Int4KvAppendArgs& args, hipStream_t stream) {
    hipLaunchKernelGGL((append_kernel<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>),
                       dim3(static_cast<unsigned int>(args.tokens * args.kv_heads)),
                       dim3(kHeadDim),
                       0,
                       stream,
                       args.keys,
                       args.values,
                       args.positions,
                       args.tokens,
                       args.token_count,
                       args.first_position,
                       args.first_position_device,
                       args.kv_heads,
                       args.page_table,
                       args.page_table_pages,
                       args.page_table_row,
                       args.page_table_row_stride,
                       args.page_table_row_count,
                       args.physical_tokens,
                       args.key_codes,
                       args.value_codes,
                       args.value_scale_bits,
                       args.status);
    return hipGetLastError();
}

template <std::uint32_t Group>
hipError_t dispatch_layouts(const Fp8Int4KvAppendArgs& args, hipStream_t stream) {
    const unsigned mask =
        (args.key_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
        (args.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U) |
        (args.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 4U : 0U);
    switch (mask) {
    case 0: return launch<Group, false, false, false>(args, stream);
    case 1: return launch<Group, true, false, false>(args, stream);
    case 2: return launch<Group, false, true, false>(args, stream);
    case 3: return launch<Group, true, true, false>(args, stream);
    case 4: return launch<Group, false, false, true>(args, stream);
    case 5: return launch<Group, true, false, true>(args, stream);
    case 6: return launch<Group, false, true, true>(args, stream);
    case 7: return launch<Group, true, true, true>(args, stream);
    }
    return hipErrorInvalidValue;
}

} // namespace

hipError_t fp8_int4_kv_append(const Fp8Int4KvAppendArgs& args, hipStream_t stream) noexcept {
    if (args.keys == nullptr || args.values == nullptr || args.positions == nullptr ||
        args.tokens == 0 || args.kv_heads == 0 || args.page_table == nullptr ||
        args.page_table_pages == 0 || args.physical_tokens == 0 ||
        args.physical_tokens % kPageSize != 0 || args.key_codes == nullptr || args.value_codes == nullptr ||
        args.value_scale_bits == nullptr || args.tokens > static_cast<std::size_t>(UINT32_MAX) /
                                                          args.kv_heads ||
        (args.first_position_device == nullptr &&
         args.tokens - 1U > static_cast<std::size_t>(UINT32_MAX - args.first_position)) ||
        args.status == nullptr ||
        (args.page_table_row != nullptr &&
         (args.page_table_row_stride < args.page_table_pages ||
          args.page_table_row_count == 0U)) ||
        !valid_plane_layout(args.key_layout) || !valid_plane_layout(args.value_layout) ||
        !valid_plane_layout(args.value_scale_layout)) {
        return hipErrorInvalidValue;
    }

    if (args.value_group == 16) {
        return dispatch_layouts<16>(args, stream);
    }
    if (args.value_group == 32) {
        return dispatch_layouts<32>(args, stream);
    }
    return hipErrorInvalidValue;
}

} // namespace ninfer::ops::r9700::kv
