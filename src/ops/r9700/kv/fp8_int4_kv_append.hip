#include "fp8_int4_kv_append.h"
#include "tq_k4v3_device.h"

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

// E2M1_rn of value / scale, where FP32 midpoints tie to the even code. Sign is kept in bit 3, so a
// negative value below the first threshold stores as signed zero.
__device__ __forceinline__ std::uint32_t nvfp4_code(float value, float scale) {
    if (scale == 0.0F) return 0U;
    const float x   = fabsf(__fdiv_rn(value, scale));
    std::uint32_t c = x > 0.25F ? 1U : 0U;
    c               = x >= 0.75F ? 2U : c;
    c               = x > 1.25F ? 3U : c;
    c               = x >= 1.75F ? 4U : c;
    c               = x > 2.5F ? 5U : c;
    c               = x >= 3.5F ? 6U : c;
    c               = x > 5.0F ? 7U : c;
    return c | (signbit(value) ? 8U : 0U);
}

__device__ __forceinline__ float nvfp4_group_maximum(float magnitude) {
    for (std::uint32_t delta = 8U; delta != 0U; delta >>= 1U) {
        magnitude = fmaxf(magnitude, __shfl_down(magnitude, delta, 16));
    }
    return __shfl(magnitude, 0, 16);
}

// Block-wide FP32 sum of one value per thread (256 threads, eight wave32 partials).
__device__ __forceinline__ float tq_block_sum(float value, float* partials) {
    for (std::uint32_t delta = 16U; delta != 0U; delta >>= 1U)
        value += __shfl_xor(value, delta, 32);
    if ((threadIdx.x & 31U) == 0U) partials[threadIdx.x / 32U] = value;
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (std::uint32_t wave = 0U; wave < kHeadDim / 32U; ++wave) total += partials[wave];
    __syncthreads();
    return total;
}

__device__ __forceinline__ std::uint32_t tq_wave_or(std::uint32_t bits) {
    for (std::uint32_t delta = 16U; delta != 0U; delta >>= 1U) bits |= __shfl_xor(bits, delta, 32);
    return bits;
}

// One rotated, norm-corrected TQ-K4-V3 head vector of K and of V (see tq_k4v3_codec.h). Thread t
// owns feature t; wave w owns the 32-feature V chunk w.
__device__ void tq_append_vector(const Fp8Int4KvAppendArgs& args, float key, float value,
                                 std::size_t destination, std::uint32_t head) {
    __shared__ float rotated[2][kHeadDim];
    __shared__ float partials[kHeadDim / 32U];
    const std::uint32_t feature = threadIdx.x;
    const float sign            = tq::negative_sign(feature) ? -1.0F : 1.0F;
    rotated[0][feature]         = key * sign;
    rotated[1][feature]         = value * sign;
    // Unnormalized Sylvester FWHT; the 1/16 is applied exactly afterwards.
    for (std::uint32_t h = 1U; h < kHeadDim; h <<= 1U) {
        __syncthreads();
        const float k0 = rotated[0][feature], k1 = rotated[0][feature ^ h];
        const float v0 = rotated[1][feature], v1 = rotated[1][feature ^ h];
        __syncthreads();
        rotated[0][feature] = (feature & h) != 0U ? k1 - k0 : k0 + k1;
        rotated[1][feature] = (feature & h) != 0U ? v1 - v0 : v0 + v1;
    }
    __syncthreads();
    const float key_rotated   = rotated[0][feature] * 0.0625F;
    const float value_rotated = rotated[1][feature] * 0.0625F;
    const float key_norm      = sqrtf(tq_block_sum(key * key, partials));
    const float value_norm    = sqrtf(tq_block_sum(value * value, partials));
    const float key_factor    = key_norm > 0.0F ? 16.0F * tq::kKeyAlpha / key_norm : 0.0F;
    const float value_factor  = value_norm > 0.0F ? 16.0F * tq::kValueAlpha / value_norm : 0.0F;
    const float key_r         = fabsf(key_rotated) * key_factor;
    const float value_r       = fabsf(value_rotated) * value_factor;
    std::uint32_t key_index = 0U, value_index = 0U;
#pragma unroll
    for (std::uint32_t i = 0U; i < 7U; ++i) key_index += key_r > tq::kKeyThresholds[i] ? 1U : 0U;
#pragma unroll
    for (std::uint32_t i = 0U; i < 3U; ++i)
        value_index += value_r > tq::kValueThresholds[i] ? 1U : 0U;
    const float key_level       = tq::key_level(key_index);
    const float value_level     = tq::value_level(value_index, 0U);
    const float key_code_norm   = sqrtf(tq_block_sum(key_level * key_level, partials));
    const float value_code_norm = sqrtf(tq_block_sum(value_level * value_level, partials));
    const __half key_scale      = __float2half_rn(key_norm / key_code_norm);
    const __half value_scale    = __float2half_rn(value_norm / value_code_norm);
    // Attention stages level * scale in binary16 (largest K level 2, V levels staged / 8), so a K
    // scale above 32752 would overflow on the staged routes: reject it like an infinite scale.
    if (!(__half2float(key_scale) <= 32752.0F) || !isfinite(__half2float(value_scale))) {
        if (feature == 0U) atomicOr(args.status, Fp8Int4KvAppendScaleOverflow);
        return;
    }
    const std::uint32_t key_code       = key_index | (key_rotated < 0.0F ? 8U : 0U);
    const std::uint32_t value_negative = value_rotated < 0.0F ? 1U : 0U;

    const std::size_t page           = destination / kPageSize;
    const std::size_t in_page        = destination % kPageSize;
    const std::size_t pages          = args.physical_tokens / kPageSize;
    const std::size_t head_page      = page + pages * head;
    const std::size_t value_row      = in_page + kPageSize * (head + args.kv_heads * page);
    const std::uint32_t key_adjacent = __shfl(key_code, (feature & 31U) ^ 1U, 32);
    if ((feature & 1U) == 0U) {
        args.key_codes[in_page + kPageSize * (feature / 2U + tq::kKeyBytes * head_page)] =
            static_cast<std::uint8_t>(key_code | (key_adjacent << 4U));
    }
    const std::uint32_t lane      = feature % 32U;
    const std::uint32_t lane_byte = 8U * (lane % 4U);
    const std::uint32_t magnitude = value_index << (lane_byte + 2U * ((lane % 16U) / 4U));
    const std::uint32_t word_a0   = tq_wave_or(lane < 16U ? magnitude : 0U);
    const std::uint32_t word_a1   = tq_wave_or(lane < 16U ? 0U : magnitude);
    const std::uint32_t word_b    = tq_wave_or(value_negative << (lane_byte + lane / 4U));
    if (lane == 0U) {
        auto* chunk = reinterpret_cast<std::uint32_t*>(
            args.value_codes + value_row * tq::kValueBytes + (feature / 32U) * tq::kValueChunk);
        chunk[0] = word_a0;
        chunk[1] = word_a1;
        chunk[2] = word_b;
    }
    if (feature == 0U) {
        args.key_scale_bits[in_page + kPageSize * head_page] = __half_as_ushort(key_scale);
        args.value_scale_bits[value_row]                     = __half_as_ushort(value_scale);
    }
}

// Kernel arguments of one append launch over the sequences of a compact decode batch. Grid y is
// the sequence, whose own arguments come from this kernarg array; slots at and above gridDim.y are
// never read.
struct AppendSequences {
    Fp8Int4KvAppendArgs sequence[kAppendMaximumSequences];
};

static_assert(sizeof(AppendSequences) <= 4096U, "append sequences must fit the kernarg segment");

template <KvCodec Codec, std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest,
          bool ScaleTokenFastest>
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
    if constexpr (Codec == KvCodec::TqK4V3) {
        // Every exit above is uniform over the block; this one must be too (the codec syncs).
        if (__syncthreads_or(!isfinite(value) || !isfinite(key)) != 0) {
            if (feature == 0U) atomicOr(status, Fp8Int4KvAppendNonfiniteInput);
            return;
        }
        tq_append_vector(args, key, value, destination, head);
        return;
    }
    if (__ballot(!isfinite(value) || !isfinite(key)) != 0) {
        if (lane_in_group == 0) { atomicOr(status, Fp8Int4KvAppendNonfiniteInput); }
        return;
    }
    if constexpr (Codec == KvCodec::Nvfp4) {
        const std::size_t page        = destination / kPageSize;
        const std::size_t in_page     = destination % kPageSize;
        const std::size_t pages       = physical_tokens / kPageSize;
        const std::uint32_t key_scale = __hip_cvt_float_to_fp8(
            __fdiv_rn(nvfp4_group_maximum(fabsf(key)), 6.0F), __HIP_SATFINITE, __HIP_E4M3);
        const std::uint32_t value_scale = __hip_cvt_float_to_fp8(
            __fdiv_rn(nvfp4_group_maximum(fabsf(value)), 6.0F), __HIP_SATFINITE, __HIP_E4M3);
        const std::uint32_t key_code =
            nvfp4_code(key, __half2float(__half(__hip_cvt_fp8_to_halfraw(key_scale, __HIP_E4M3))));
        const std::uint32_t value_code = nvfp4_code(
            value, __half2float(__half(__hip_cvt_fp8_to_halfraw(value_scale, __HIP_E4M3))));
        const std::uint32_t key_adjacent   = __shfl(key_code, lane_in_group ^ 1U, Group);
        const std::uint32_t value_adjacent = __shfl(value_code, lane_in_group ^ 1U, Group);
        const std::size_t head_page        = page + pages * head;
        const std::size_t value_row        = in_page + kPageSize * (head + kv_heads * page);
        if ((lane_in_group & 1U) == 0U) {
            args.key_codes[in_page + kPageSize * (feature / 2U + 128U * head_page)] =
                static_cast<std::uint8_t>(key_code | (key_adjacent << 4U));
            args.value_codes[feature / 2U + 128U * value_row] =
                static_cast<std::uint8_t>(value_code | (value_adjacent << 4U));
        }
        if (lane_in_group == 0U) {
            args.key_scale_codes[in_page + kPageSize * (group + 16U * head_page)] =
                static_cast<std::uint8_t>(key_scale);
            args.value_scale_codes[group + 16U * value_row] =
                static_cast<std::uint8_t>(value_scale);
        }
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

template <std::uint32_t Group, bool KeyTokenFastest, bool ValueTokenFastest, bool ScaleTokenFastest,
          KvCodec Codec = KvCodec::Fp8Int4>
hipError_t launch(const AppendSequences& sequences, std::uint32_t count, std::size_t grid_items,
                  hipStream_t stream) {
    hipLaunchKernelGGL(
        (append_kernel<Codec, Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>),
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
        args.value_codes == nullptr ||
        (args.codec == KvCodec::Fp8Int4 && args.value_scale_bits == nullptr) ||
        (args.codec == KvCodec::Nvfp4 &&
         (args.key_scale_codes == nullptr || args.value_scale_codes == nullptr ||
          args.value_group != 16 || args.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
          args.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
          args.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest)) ||
        (args.codec == KvCodec::TqK4V3 &&
         (args.key_scale_bits == nullptr || args.value_scale_bits == nullptr ||
          args.key_layout != Fp8Int4KvPlaneLayout::TokenFastest ||
          args.value_layout != Fp8Int4KvPlaneLayout::FeatureFastest ||
          args.value_scale_layout != Fp8Int4KvPlaneLayout::FeatureFastest)) ||
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
        if (!valid_append(args) || args.codec != first.codec ||
            args.value_group != first.value_group || args.key_layout != first.key_layout ||
            args.value_layout != first.value_layout ||
            args.value_scale_layout != first.value_scale_layout) {
            return hipErrorInvalidValue;
        }
        batch.sequence[sequence] = args;
        grid_items               = std::max(grid_items, args.tokens * args.kv_heads);
    }
    const auto count = static_cast<std::uint32_t>(sequences.size());
    if (first.codec == KvCodec::Nvfp4) {
        return launch<16U, true, false, false, KvCodec::Nvfp4>(batch, count, grid_items, stream);
    }
    if (first.codec == KvCodec::TqK4V3) {
        return launch<16U, true, false, false, KvCodec::TqK4V3>(batch, count, grid_items, stream);
    }
    return first.value_group == 16 ? dispatch_layouts<16>(first, batch, count, grid_items, stream)
                                   : dispatch_layouts<32>(first, batch, count, grid_items, stream);
}

hipError_t fp8_int4_kv_append(const Fp8Int4KvAppendArgs& args, hipStream_t stream) noexcept {
    return fp8_int4_kv_append(std::span<const Fp8Int4KvAppendArgs>(&args, 1U), stream);
}

} // namespace ninfer::ops::r9700::kv
