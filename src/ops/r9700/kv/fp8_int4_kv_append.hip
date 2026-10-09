#include "fp8_int4_kv_append.h"

#include <hip/hip_fp8.h>

#include <algorithm>

namespace ninfer::ops::r9700::kv {
namespace {

inline constexpr std::uint32_t kHeadDim  = 256;
inline constexpr std::uint32_t kPageSize = 64;

[[nodiscard]] constexpr bool valid_plane_layout(Fp8Int4KvPlaneLayout layout) noexcept {
    return layout == Fp8Int4KvPlaneLayout::FeatureFastest ||
           layout == Fp8Int4KvPlaneLayout::TokenFastest;
}

template <std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest, bool ScaleTokenFastest>
struct PlaneOffset {
    static constexpr std::uint32_t kGroups        = kHeadDim / Group;
    static constexpr std::uint32_t kBytesPerGroup = Group / 2;

    __device__ static std::size_t key(std::size_t token, std::uint32_t head, std::uint32_t feature,
                                      std::uint32_t heads, std::size_t physical_tokens) {
        const std::size_t page    = token / kPageSize;
        const std::size_t in_page = token % kPageSize;
        const std::size_t pages   = physical_tokens / kPageSize;
        if constexpr (KeyTokenFastest) {
            return in_page + kPageSize * (static_cast<std::size_t>(feature) +
                                          kHeadDim * (page + pages * head));
        }
        return feature +
               kHeadDim * (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
    }

    __device__ static std::size_t value(std::size_t token, std::uint32_t head, std::uint32_t group,
                                        std::uint32_t byte, std::uint32_t heads,
                                        std::size_t physical_tokens) {
        const std::size_t page    = token / kPageSize;
        const std::size_t in_page = token % kPageSize;
        const std::size_t pages   = physical_tokens / kPageSize;
        const std::size_t feature = static_cast<std::size_t>(group) * kBytesPerGroup + byte;
        if constexpr (ValueTokenFastest) {
            return in_page +
                   kPageSize * (feature + (kGroups * kBytesPerGroup) * (page + pages * head));
        }
        return feature +
               (kGroups * kBytesPerGroup) *
                   (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
    }

    __device__ static std::size_t scale(std::size_t token, std::uint32_t head, std::uint32_t group,
                                        std::uint32_t heads, std::size_t physical_tokens) {
        const std::size_t page    = token / kPageSize;
        const std::size_t in_page = token % kPageSize;
        const std::size_t pages   = physical_tokens / kPageSize;
        if constexpr (ScaleTokenFastest) {
            return in_page +
                   kPageSize * (static_cast<std::size_t>(group) + kGroups * (page + pages * head));
        }
        return group +
               kGroups * (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
    }
};

__device__ std::size_t physical_token(std::size_t logical, const std::uint32_t* page_table) {
    return static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
           logical % kPageSize;
}

// Kernel arguments of one append launch over the sequences of a compact decode batch. Grid y is
// the sequence, whose own arguments come from this kernarg array; slots at and above gridDim.y are
// never read.
struct AppendSequences {
    Fp8Int4KvAppendArgs sequence[kAppendMaximumSequences];
};

static_assert(sizeof(AppendSequences) <= 4096U, "append sequences must fit the kernarg segment");

template <std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest, bool ScaleTokenFastest>
__global__ void append_kernel(AppendSequences sequences) {
    const Fp8Int4KvAppendArgs& args       = sequences.sequence[blockIdx.y];
    const std::uint32_t kv_heads          = args.kv_heads;
    const std::size_t tokens              = args.tokens;
    const std::int32_t* const token_count = args.token_count;
    std::uint32_t* const status           = args.status;
    const std::size_t item                = blockIdx.x;
    // Grid x covers the batch's longest panel; a shorter sequence's surplus blocks exit.
    if (item >= tokens * kv_heads) { return; }
    const std::size_t input_token   = item / kv_heads;
    const std::uint32_t head        = static_cast<std::uint32_t>(item % kv_heads);
    const std::int32_t device_count = token_count == nullptr ? 0 : *token_count;
    if (token_count != nullptr &&
        (device_count < 0 || static_cast<std::size_t>(device_count) > tokens)) {
        if (item == 0 && threadIdx.x == 0) { atomicOr(status, Fp8Int4KvAppendInvalidCount); }
        return;
    }
    const std::size_t active_count =
        token_count == nullptr ? tokens : static_cast<std::size_t>(device_count);
    if (input_token >= active_count) { return; }
    const std::uint32_t* page_table = args.page_table;
    if (args.page_table_row != nullptr) {
        const std::int32_t represented_row = *args.page_table_row;
        if (represented_row < 0 ||
            static_cast<std::uint32_t>(represented_row) >= args.page_table_row_count) {
            atomicOr(status, Fp8Int4KvAppendInvalidAddress);
            return;
        }
        page_table += static_cast<std::size_t>(represented_row) * args.page_table_row_stride;
    }
    const hip_bfloat16* const keys    = args.keys;
    const hip_bfloat16* const values  = args.values;
    const std::size_t physical_tokens = args.physical_tokens;

    const std::uint32_t feature          = threadIdx.x;
    const std::size_t source             = (input_token * kv_heads + head) * kHeadDim + feature;
    const std::uint32_t logical_position = args.positions[input_token];
    const std::uint32_t represented_first =
        args.first_position_device == nullptr ? args.first_position : *args.first_position_device;
    if (input_token > static_cast<std::size_t>(UINT32_MAX - represented_first) ||
        logical_position != represented_first + static_cast<std::uint32_t>(input_token)) {
        atomicOr(status, Fp8Int4KvAppendInvalidPosition);
        return;
    }
    if (logical_position / kPageSize >= args.page_table_pages) {
        atomicOr(status, Fp8Int4KvAppendInvalidAddress);
        return;
    }
    const std::size_t destination = physical_token(logical_position, page_table);
    if (destination >= physical_tokens) {
        atomicOr(status, Fp8Int4KvAppendInvalidAddress);
        return;
    }

    const std::uint32_t lane_in_group = feature % Group;
    const std::uint32_t group         = feature / Group;
    const float value                 = static_cast<float>(values[source]);
    const float key                   = static_cast<float>(keys[source]);
    if (__ballot(!isfinite(value) || !isfinite(key)) != 0) {
        if (lane_in_group == 0) { atomicOr(status, Fp8Int4KvAppendNonfiniteInput); }
        return;
    }
    float maximum = fabsf(value);
    for (std::uint32_t delta = Group / 2; delta != 0; delta >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down(maximum, delta, Group));
    }
    maximum                   = __shfl(maximum, 0, Group);
    const __half stored_scale = __float2half(maximum / 7.0F);
    const float scale         = __half2float(stored_scale);
    if (!isfinite(scale)) {
        if (lane_in_group == 0) { atomicOr(status, Fp8Int4KvAppendScaleOverflow); }
        return;
    }
    int code = scale == 0.0F ? 0 : __float2int_rn(value / scale);
    code     = code < -7 ? -7 : (code > 7 ? 7 : code);

    args.key_codes[PlaneOffset<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>::key(
        destination, head, feature, kv_heads, physical_tokens)] =
        __hip_cvt_float_to_fp8(key, __HIP_SATFINITE, __HIP_E4M3);

    if (lane_in_group == 0) {
        args.value_scale_bits[PlaneOffset<Group, KeyTokenFastest, ValueTokenFastest,
                                          ScaleTokenFastest>::scale(destination, head, group,
                                                                    kv_heads, physical_tokens)] =
            __half_as_ushort(stored_scale);
    }
    const int adjacent = __shfl(code, lane_in_group ^ 1U, Group);
    if ((lane_in_group & 1U) == 0) {
        args.value_codes
            [PlaneOffset<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>::value(
                destination, head, group, lane_in_group / 2, kv_heads, physical_tokens)] =
            static_cast<std::uint8_t>((code & 0x0f) | ((adjacent & 0x0f) << 4));
    }
}

template <std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest, bool ScaleTokenFastest>
hipError_t launch(const AppendSequences& sequences, std::uint32_t count, std::size_t grid_items,
                  hipStream_t stream) {
    hipLaunchKernelGGL(
        (append_kernel<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>),
        dim3(static_cast<unsigned int>(grid_items), count), dim3(kHeadDim), 0, stream, sequences);
    return hipGetLastError();
}

template <std::uint32_t Group>
hipError_t dispatch_layouts(const Fp8Int4KvAppendArgs& args, const AppendSequences& sequences,
                            std::uint32_t count, std::size_t grid_items, hipStream_t stream) {
    const unsigned mask = (args.key_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 1U : 0U) |
                          (args.value_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 2U : 0U) |
                          (args.value_scale_layout == Fp8Int4KvPlaneLayout::TokenFastest ? 4U : 0U);
    switch (mask) {
    case 0:
        return launch<Group, false, false, false>(sequences, count, grid_items, stream);
    case 1:
        return launch<Group, true, false, false>(sequences, count, grid_items, stream);
    case 2:
        return launch<Group, false, true, false>(sequences, count, grid_items, stream);
    case 3:
        return launch<Group, true, true, false>(sequences, count, grid_items, stream);
    case 4:
        return launch<Group, false, false, true>(sequences, count, grid_items, stream);
    case 5:
        return launch<Group, true, false, true>(sequences, count, grid_items, stream);
    case 6:
        return launch<Group, false, true, true>(sequences, count, grid_items, stream);
    case 7:
        return launch<Group, true, true, true>(sequences, count, grid_items, stream);
    default:
        return hipErrorInvalidValue;
    }
}

[[nodiscard]] bool valid_append(const Fp8Int4KvAppendArgs& args) noexcept {
    return !(
        args.keys == nullptr || args.values == nullptr || args.positions == nullptr ||
        args.tokens == 0 || args.kv_heads == 0 || args.page_table == nullptr ||
        args.page_table_pages == 0 || args.physical_tokens == 0 ||
        args.physical_tokens % kPageSize != 0 || args.key_codes == nullptr ||
        args.value_codes == nullptr || args.value_scale_bits == nullptr ||
        args.tokens > static_cast<std::size_t>(UINT32_MAX) / args.kv_heads ||
        (args.first_position_device == nullptr &&
         args.tokens - 1U > static_cast<std::size_t>(UINT32_MAX - args.first_position)) ||
        args.status == nullptr ||
        (args.page_table_row != nullptr &&
         (args.page_table_row_stride < args.page_table_pages || args.page_table_row_count == 0U)) ||
        !valid_plane_layout(args.key_layout) || !valid_plane_layout(args.value_layout) ||
        !valid_plane_layout(args.value_scale_layout) ||
        (args.value_group != 16 && args.value_group != 32));
}

} // namespace

hipError_t fp8_int4_kv_append(std::span<const Fp8Int4KvAppendArgs> sequences,
                              hipStream_t stream) noexcept {
    if (sequences.empty() || sequences.size() > kAppendMaximumSequences) {
        return hipErrorInvalidValue;
    }
    const Fp8Int4KvAppendArgs& first = sequences.front();
    AppendSequences batch{};
    std::size_t grid_items = 0;
    for (std::size_t sequence = 0; sequence < sequences.size(); ++sequence) {
        const Fp8Int4KvAppendArgs& args = sequences[sequence];
        // The codec instantiation is shared: one value group and plane order per launch.
        if (!valid_append(args) || args.value_group != first.value_group ||
            args.key_layout != first.key_layout || args.value_layout != first.value_layout ||
            args.value_scale_layout != first.value_scale_layout) {
            return hipErrorInvalidValue;
        }
        batch.sequence[sequence] = args;
        grid_items               = std::max(grid_items, args.tokens * args.kv_heads);
    }
    const auto count = static_cast<std::uint32_t>(sequences.size());
    return first.value_group == 16 ? dispatch_layouts<16>(first, batch, count, grid_items, stream)
                                   : dispatch_layouts<32>(first, batch, count, grid_items, stream);
}

hipError_t fp8_int4_kv_append(const Fp8Int4KvAppendArgs& args, hipStream_t stream) noexcept {
    return fp8_int4_kv_append(std::span<const Fp8Int4KvAppendArgs>(&args, 1U), stream);
}

} // namespace ninfer::ops::r9700::kv
