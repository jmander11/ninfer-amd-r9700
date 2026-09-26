#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/linear/a8q4_small_batch_projection.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"

#include <hip/hip_fp16.h>

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace ninfer::ops::r9700::linear {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kW8Group = 32;
constexpr std::uint32_t kKAlignment = 128;
constexpr std::uint32_t kWaveSize = 32;
constexpr std::uint32_t kWavesPerDecodeBlock = kThreads / kWaveSize;
constexpr std::uint32_t kWavesPerPrefillBlock = 16;
constexpr std::uint32_t kPrefillThreads = kWavesPerPrefillBlock * kWaveSize;
constexpr std::uint32_t kCooperativeTile = 256;
constexpr std::uint32_t kWmmaTile = 16;
constexpr std::uint32_t kQ4Group = 64;
constexpr std::uint32_t kA8Q4PrefillTokenTile = 64;
constexpr std::uint32_t kA8Q4PrefillRowTile = 64;
constexpr std::uint32_t kA8Q4PrefillWaves = 16;
constexpr std::uint32_t kA8Q4PrefillThreads = kA8Q4PrefillWaves * kWaveSize;
constexpr std::uint32_t kA8Q4M128PrefillTokenTile = 128;
constexpr std::uint32_t kA8Q4M128PrefillWaves = 32;
constexpr std::uint32_t kA8Q4M128PrefillThreads = kA8Q4M128PrefillWaves * kWaveSize;
constexpr std::uint32_t kA8Q4M128ProductionThreads = 8U * kWaveSize;
constexpr std::uint32_t kA8W8PrefillTokenTile = 64;
constexpr std::uint32_t kA8W8PrefillRowTile = 64;
constexpr std::uint32_t kA8W8PrefillWaves = 16;
constexpr std::uint32_t kA8W8PrefillThreads = kA8W8PrefillWaves * kWaveSize;
constexpr std::uint32_t kQ4KAlignment = 128;
constexpr std::size_t kQ4PlaneAlignment = 256;

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1U) / alignment * alignment;
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) noexcept {
    if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    *result = lhs * rhs;
    return true;
}

bool aligned_dword(const void* pointer) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignof(std::uint32_t) == 0U;
}

bool aligned_qword(const void* pointer) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignof(std::uint64_t) == 0U;
}

bool q4_padded_columns(std::uint32_t columns, std::uint32_t* padded) noexcept {
    if (columns == 0U || columns > std::numeric_limits<std::uint32_t>::max() -
                                      (kQ4KAlignment - 1U)) return false;
    *padded = (columns + kQ4KAlignment - 1U) / kQ4KAlignment * kQ4KAlignment;
    return true;
}

bool q4_layout_bytes(std::uint32_t rows, std::uint32_t columns,
                     std::uint32_t padded_columns, std::size_t* code_bytes,
                     std::size_t* scale_bytes) noexcept {
    std::uint32_t expected_padded = 0;
    if (rows == 0U || !q4_padded_columns(columns, &expected_padded) ||
        padded_columns != expected_padded) return false;
    std::size_t elements = 0;
    if (!checked_mul(rows, padded_columns, &elements)) return false;
    *code_bytes = elements / 2U;
    const std::size_t groups = padded_columns / kQ4Group;
    std::size_t scale_words = 0;
    return checked_mul(rows, groups, &scale_words) &&
           checked_mul(scale_words, sizeof(std::uint16_t), scale_bytes);
}

// Persistent Q4G64 weights are tiled for the native gfx1201 IU4 operand:
// [N/16][K/64][four K16 pairs][16 rows][packed-u64].  The scale plane uses
// the matching [N/16][K/64][16 rows] order.  This is a byte permutation of
// the logical [N,K] codes and [N,K/64] scales; no runtime repack exists.
__host__ __device__ constexpr std::size_t q4_n16k16_pair_index(
    std::uint32_t row, std::uint32_t group, std::uint32_t pair,
    std::uint32_t groups) noexcept {
    return (((static_cast<std::size_t>(row >> 4U) * groups + group) * 4U + pair) *
            16U + (row & 15U));
}

__host__ __device__ constexpr std::size_t q4_n16k16_scale_index(
    std::uint32_t row, std::uint32_t group, std::uint32_t groups) noexcept {
    return ((static_cast<std::size_t>(row >> 4U) * groups + group) * 16U +
            (row & 15U));
}

bool byte_interval(const void* pointer, std::size_t bytes, std::uintptr_t* begin,
                   std::uintptr_t* end) noexcept {
    if (pointer == nullptr || bytes == 0U) return false;
    *begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (*begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    *end = *begin + bytes;
    return true;
}

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes, const void* rhs,
                    std::size_t rhs_bytes) noexcept {
    std::uintptr_t lhs_begin = 0, lhs_end = 0, rhs_begin = 0, rhs_end = 0;
    if (!byte_interval(lhs, lhs_bytes, &lhs_begin, &lhs_end) ||
        !byte_interval(rhs, rhs_bytes, &rhs_begin, &rhs_end)) return true;
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool valid_q4_workspace(const Q4G64ActivationWorkspace& workspace) noexcept {
    std::size_t code_bytes = 0, scale_bytes = 0;
    if (workspace.status == nullptr ||
        !q4_layout_bytes(workspace.tokens, workspace.columns, workspace.padded_columns,
                         &code_bytes, &scale_bytes) ||
        workspace.codes == nullptr || workspace.scales == nullptr ||
        workspace.code_bytes != code_bytes || workspace.scale_bytes != scale_bytes) return false;
    return !ranges_overlap(workspace.codes, code_bytes, workspace.scales, scale_bytes) &&
           !ranges_overlap(workspace.codes, code_bytes, workspace.status, sizeof(std::uint32_t)) &&
           !ranges_overlap(workspace.scales, scale_bytes, workspace.status, sizeof(std::uint32_t));
}

bool valid_q4_linear_args(const Q4G64LinearArgs& args) noexcept {
    std::size_t activation_code_bytes = 0, activation_scale_bytes = 0;
    std::size_t weight_code_bytes = 0, weight_scale_bytes = 0;
    if (args.activation_codes == nullptr || args.activation_scales == nullptr ||
        args.activation_status == nullptr ||
        args.weight_codes == nullptr || args.weight_scales == nullptr || args.output == nullptr ||
        !aligned_dword(args.activation_codes) || !aligned_qword(args.weight_codes) ||
        args.rows % 16U != 0U ||
        !q4_layout_bytes(args.tokens, args.columns, args.padded_columns,
                         &activation_code_bytes, &activation_scale_bytes) ||
        !q4_layout_bytes(args.rows, args.columns, args.padded_columns,
                         &weight_code_bytes, &weight_scale_bytes) ||
        args.activation_code_bytes != activation_code_bytes ||
        args.activation_scale_bytes != activation_scale_bytes ||
        args.weight_code_bytes != weight_code_bytes ||
        args.weight_scale_bytes != weight_scale_bytes) return false;
    std::size_t output_elements = 0, output_bytes = 0;
    if (!checked_mul(args.tokens, args.rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes)) return false;
    const void* pointers[] = {args.activation_codes, args.activation_scales,
                              args.activation_status, args.weight_codes,
                              args.weight_scales};
    const std::size_t sizes[] = {activation_code_bytes, activation_scale_bytes,
                                 sizeof(std::uint32_t), weight_code_bytes,
                                 weight_scale_bytes};
    for (unsigned index = 0; index < 5U; ++index) {
        if (ranges_overlap(args.output, output_bytes, pointers[index], sizes[index])) return false;
        for (unsigned other = index + 1U; other < 5U; ++other) {
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other])) {
                return false;
            }
        }
    }
    return true;
}

bool valid_a8_workspace(const A8G64ActivationWorkspace& workspace) noexcept {
    std::size_t plane_bytes = 0, scale_bytes = 0;
    if (workspace.status == nullptr ||
        !q4_layout_bytes(workspace.tokens, workspace.columns, workspace.padded_columns,
                         &plane_bytes, &scale_bytes) ||
        workspace.low_codes == nullptr || workspace.high_codes == nullptr ||
        workspace.scales == nullptr || workspace.low_code_bytes != plane_bytes ||
        workspace.high_code_bytes != plane_bytes || workspace.scale_bytes != scale_bytes) {
        return false;
    }
    const void* pointers[] = {workspace.low_codes, workspace.high_codes, workspace.scales,
                              workspace.status};
    const std::size_t sizes[] = {plane_bytes, plane_bytes, scale_bytes,
                                 sizeof(std::uint32_t)};
    for (unsigned index = 0; index < 4U; ++index) {
        for (unsigned other = index + 1U; other < 4U; ++other) {
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other])) {
                return false;
            }
        }
    }
    return true;
}

bool valid_a8q4_linear_args(const A8Q4G64LinearArgs& args,
                             bool allow_accumulate = false) noexcept {
    std::size_t activation_plane_bytes = 0, activation_scale_bytes = 0;
    std::size_t weight_code_bytes = 0, weight_scale_bytes = 0;
    if ((args.accumulate_output && !allow_accumulate) ||
        args.activation_low_codes == nullptr || args.activation_high_codes == nullptr ||
        args.activation_scales == nullptr || args.activation_status == nullptr ||
        args.weight_codes == nullptr ||
        args.weight_scales == nullptr || args.output == nullptr ||
        !aligned_dword(args.activation_low_codes) ||
        !aligned_dword(args.activation_high_codes) || !aligned_qword(args.weight_codes) ||
        args.rows % 16U != 0U ||
        !q4_layout_bytes(args.tokens, args.columns, args.padded_columns,
                         &activation_plane_bytes, &activation_scale_bytes) ||
        !q4_layout_bytes(args.rows, args.columns, args.padded_columns,
                         &weight_code_bytes, &weight_scale_bytes) ||
        args.activation_low_code_bytes != activation_plane_bytes ||
        args.activation_high_code_bytes != activation_plane_bytes ||
        args.activation_scale_bytes != activation_scale_bytes ||
        args.weight_code_bytes != weight_code_bytes ||
        args.weight_scale_bytes != weight_scale_bytes) {
        return false;
    }
    std::size_t output_elements = 0, output_bytes = 0;
    if (!checked_mul(args.tokens, args.rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes)) return false;
    const void* pointers[] = {args.activation_low_codes, args.activation_high_codes,
                              args.activation_scales, args.activation_status,
                              args.weight_codes, args.weight_scales};
    const std::size_t sizes[] = {activation_plane_bytes, activation_plane_bytes,
                                 activation_scale_bytes, sizeof(std::uint32_t),
                                 weight_code_bytes, weight_scale_bytes};
    for (unsigned index = 0; index < 6U; ++index) {
        if (ranges_overlap(args.output, output_bytes, pointers[index], sizes[index])) return false;
        for (unsigned other = index + 1U; other < 6U; ++other) {
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other])) {
                return false;
            }
        }
    }
    return true;
}

bool dflash_small_t_qualification_shape(std::uint32_t rows,
                                        std::uint32_t columns) noexcept {
    return rows == 34816U && columns == 5120U;
}

bool dflash_small_t_qualification_tokens(std::uint32_t tokens) noexcept {
    switch (tokens) {
    case 4U: case 5U: case 6U: case 8U: case 10U: case 12U:
    case 15U: case 16U: case 18U: case 20U: case 24U: return true;
    default: return false;
    }
}

bool scalar_base_offsets_fit(std::uint32_t tokens, std::uint32_t rows,
                             std::uint32_t padded_columns) noexcept {
    if (tokens == 0U || rows == 0U || padded_columns == 0U ||
        padded_columns % kQ4Group != 0U) return false;
    using Wide = unsigned __int128;
    const Wide groups = padded_columns / kQ4Group;
    if (groups == 0U) return false;
    const Wide last_token = tokens - 1U;
    const Wide last_row = rows - 1U;
    const Wide last_group = groups - 1U;
    const Wide activation_code_last =
        last_token * (padded_columns / 2U) + last_group * 32U + 28U + 3U;
    const Wide weight_code_last =
        (((((last_row >> 4U) * groups + last_group) * 4U + 3U) * 16U +
           (last_row & 15U)) * 8U) + 7U;
    const Wide activation_scale_last =
        (last_token * groups + last_group) * sizeof(std::uint16_t) + 1U;
    const Wide weight_scale_last =
        (((last_row >> 4U) * groups + last_group) * 16U + (last_row & 15U)) *
            sizeof(std::uint16_t) + 1U;
    constexpr Wide maximum = std::numeric_limits<std::uint32_t>::max();
    return activation_code_last <= maximum && weight_code_last <= maximum &&
           activation_scale_last <= maximum && weight_scale_last <= maximum;
}

bool valid_decode(const hip_bfloat16* input, hip_bfloat16* output, std::uint32_t tokens,
                  std::uint32_t rows, std::uint32_t columns) noexcept {
    return input != nullptr && output != nullptr && tokens >= 1U && tokens <= 8U && rows != 0U &&
           columns != 0U;
}

bool valid_prefill(const hip_bfloat16* input, hip_bfloat16* output, std::uint32_t tokens,
                   std::uint32_t rows, std::uint32_t columns) noexcept {
    return input != nullptr && output != nullptr && tokens >= 9U && rows != 0U && columns != 0U;
}

bool valid_qualified_width(const hip_bfloat16* input, hip_bfloat16* output,
                           std::uint32_t tokens, std::uint32_t rows,
                           std::uint32_t columns) noexcept {
    return valid_decode(input, output, tokens, rows, columns) ||
           valid_prefill(input, output, tokens, rows, columns);
}

bool w8_layout_bytes(std::uint32_t rows, std::uint32_t columns,
                     std::uint32_t padded_columns, std::size_t* code_bytes,
                     std::size_t* scale_bytes) noexcept {
    const std::size_t expected_padded = align_up(columns, kKAlignment);
    if (expected_padded != padded_columns || expected_padded < columns ||
        rows > std::numeric_limits<std::size_t>::max() / expected_padded) {
        return false;
    }
    const std::size_t base_bytes = static_cast<std::size_t>(rows) * expected_padded;
    const std::size_t groups = expected_padded / kW8Group;
    if (groups == 0U || rows > std::numeric_limits<std::size_t>::max() /
                                      (groups * sizeof(std::uint16_t))) {
        return false;
    }
    *code_bytes = base_bytes;
    *scale_bytes = static_cast<std::size_t>(rows) * groups * sizeof(std::uint16_t);
    return true;
}

bool valid_a8w8_workspace(const A8G32ActivationWorkspace& workspace) noexcept {
    std::size_t code_bytes = 0, scale_bytes = 0;
    if (workspace.codes == nullptr || workspace.scales == nullptr ||
        workspace.status == nullptr ||
        !w8_layout_bytes(workspace.tokens, workspace.columns, workspace.padded_columns,
                         &code_bytes, &scale_bytes) ||
        workspace.code_bytes != code_bytes || workspace.scale_bytes != scale_bytes) {
        return false;
    }
    return !ranges_overlap(workspace.codes, code_bytes, workspace.scales, scale_bytes) &&
           !ranges_overlap(workspace.codes, code_bytes, workspace.status,
                           sizeof(std::uint32_t)) &&
           !ranges_overlap(workspace.scales, scale_bytes, workspace.status,
                           sizeof(std::uint32_t));
}

bool valid_a8w8_linear_args(const A8W8G32LinearArgs& args) noexcept {
    std::size_t activation_code_bytes = 0, activation_scale_bytes = 0;
    std::size_t weight_code_bytes = 0, weight_scale_bytes = 0;
    if (args.tokens == 0U || args.rows == 0U || args.columns == 0U ||
        args.activation_codes == nullptr || args.activation_scales == nullptr ||
        args.activation_status == nullptr || args.weight_codes == nullptr ||
        args.weight_scales == nullptr || args.output == nullptr ||
        !aligned_dword(args.activation_codes) || !aligned_dword(args.weight_codes) ||
        !w8_layout_bytes(args.tokens, args.columns, args.padded_columns,
                         &activation_code_bytes, &activation_scale_bytes) ||
        !w8_layout_bytes(args.rows, args.columns, args.padded_columns,
                         &weight_code_bytes, &weight_scale_bytes) ||
        args.activation_code_bytes != activation_code_bytes ||
        args.activation_scale_bytes != activation_scale_bytes ||
        args.weight_code_bytes != weight_code_bytes ||
        args.weight_scale_bytes != weight_scale_bytes) {
        return false;
    }
    std::size_t output_elements = 0, output_bytes = 0;
    if (!checked_mul(args.tokens, args.rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes)) return false;
    const void* pointers[] = {args.activation_codes, args.activation_scales,
                              args.activation_status, args.weight_codes,
                              args.weight_scales};
    const std::size_t sizes[] = {activation_code_bytes, activation_scale_bytes,
                                 sizeof(std::uint32_t), weight_code_bytes,
                                 weight_scale_bytes};
    for (unsigned index = 0; index < 5U; ++index) {
        if (ranges_overlap(args.output, output_bytes, pointers[index], sizes[index])) return false;
        for (unsigned other = index + 1U; other < 5U; ++other) {
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other])) {
                return false;
            }
        }
    }
    return true;
}

__global__ void bf16_linear_kernel(const hip_bfloat16* input, const hip_bfloat16* weights,
                                   hip_bfloat16* output, std::uint32_t columns,
                                   std::uint32_t rows) {
    const std::uint32_t column = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    if (column >= rows) return;
    float partial = 0.0F;
    const std::size_t x_base = static_cast<std::size_t>(token) * columns;
    const std::size_t w_base = static_cast<std::size_t>(column) * columns;
    for (std::uint32_t k = threadIdx.x; k < columns; k += blockDim.x) {
        partial = fmaf(static_cast<float>(input[x_base + k]),
                       static_cast<float>(weights[w_base + k]), partial);
    }
    __shared__ float reductions[kThreads];
    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t width = blockDim.x / 2U; width != 0U; width >>= 1U) {
        if (threadIdx.x < width) reductions[threadIdx.x] += reductions[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        output[static_cast<std::size_t>(token) * rows + column] =
            static_cast<hip_bfloat16>(reductions[0]);
    }
}

__global__ void w8g32_linear_kernel(const hip_bfloat16* input,
                                    const std::int8_t* codes,
                                    const std::uint16_t* scale_words,
                                    hip_bfloat16* output, std::uint32_t columns,
                                    std::uint32_t rows, std::size_t padded_columns) {
    const std::uint32_t column = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    if (column >= rows) return;
    float partial = 0.0F;
    const std::size_t x_base = static_cast<std::size_t>(token) * columns;
    const std::size_t w_base = static_cast<std::size_t>(column) * padded_columns;
    const std::size_t groups_per_row = padded_columns / kW8Group;
    for (std::uint32_t k = threadIdx.x; k < columns; k += blockDim.x) {
        const std::int32_t code = codes[w_base + k];
        const std::uint16_t scale_bits =
            scale_words[static_cast<std::size_t>(column) * groups_per_row + k / kW8Group];
        const float weight = static_cast<float>(code) * __half2float(__ushort_as_half(scale_bits));
        partial = fmaf(static_cast<float>(input[x_base + k]), weight, partial);
    }
    __shared__ float reductions[kThreads];
    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t width = blockDim.x / 2U; width != 0U; width >>= 1U) {
        if (threadIdx.x < width) reductions[threadIdx.x] += reductions[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        output[static_cast<std::size_t>(token) * rows + column] =
            static_cast<hip_bfloat16>(reductions[0]);
    }
}

__global__ __launch_bounds__(64) void q4g64_quantize_activation_kernel(
    const hip_bfloat16* input, std::uint8_t* codes, std::uint16_t* scale_words,
    std::uint32_t* status, std::uint32_t columns, std::uint32_t padded_columns) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t column = group * kQ4Group + lane;
    float value = 0.0F;
    bool finite = true;
    if (column < columns) {
        value = static_cast<float>(input[static_cast<std::size_t>(token) * columns + column]);
        finite = isfinite(value);
        if (!finite) {
            atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
            value = 0.0F;
        }
    }
    __shared__ float maxima[kQ4Group];
    __shared__ std::uint16_t represented_scale_bits;
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (std::uint32_t width = kQ4Group / 2U; width != 0U; width >>= 1U) {
        if (lane < width) maxima[lane] = fmaxf(maxima[lane], maxima[lane + width]);
        __syncthreads();
    }
    if (lane == 0U) {
        std::uint16_t bits = 0U;
        if (maxima[0] != 0.0F) {
            bits = __half_as_ushort(__float2half_rn(maxima[0] / 7.0F));
            if (bits == 0U) bits = 1U;
            if (!isfinite(__half2float(__ushort_as_half(bits)))) {
                atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
                bits = 0U;
            }
        }
        represented_scale_bits = bits;
        scale_words[static_cast<std::size_t>(token) * (padded_columns / kQ4Group) + group] = bits;
    }
    __syncthreads();
    if (lane < kQ4Group / 2U) {
        const std::uint32_t column0 = group * kQ4Group + lane * 2U;
        const std::uint32_t column1 = column0 + 1U;
        const float scale = __half2float(__ushort_as_half(represented_scale_bits));
        auto encode = [&](std::uint32_t source_column) {
            if (source_column >= columns || scale == 0.0F) return 0;
            const float source = static_cast<float>(
                input[static_cast<std::size_t>(token) * columns + source_column]);
            if (!isfinite(source)) return 0;
            return max(-8, min(7, __float2int_rn(source / scale)));
        };
        const int low = encode(column0);
        const int high = encode(column1);
        codes[(static_cast<std::size_t>(token) * padded_columns + column0) / 2U] =
            static_cast<std::uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
    }
}

__global__ __launch_bounds__(kWaveSize) void a8g64_quantize_activation_kernel(
    const hip_bfloat16* input, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words, std::uint32_t* status, std::uint32_t columns,
    std::uint32_t padded_columns) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t column0 = group * kQ4Group + lane * 2U;
    const std::uint32_t column1 = column0 + 1U;
    auto load_finite = [&](std::uint32_t column) {
        if (column >= columns) return 0.0F;
        float value = static_cast<float>(
            input[static_cast<std::size_t>(token) * columns + column]);
        if (!isfinite(value)) {
            atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
            value = 0.0F;
        }
        return value;
    };
    const float value0 = load_finite(column0);
    const float value1 = load_finite(column1);
    float maximum = fmaxf(fabsf(value0), fabsf(value1));
#pragma unroll
    for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down(maximum, width, kWaveSize));
    }
    std::uint16_t represented_scale_bits = 0U;
    if (lane == 0U) {
        if (maximum != 0.0F) {
            represented_scale_bits = __half_as_ushort(__float2half_rn(maximum / 127.0F));
            if (represented_scale_bits == 0U) represented_scale_bits = 1U;
            if (!isfinite(__half2float(__ushort_as_half(represented_scale_bits)))) {
                atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
                represented_scale_bits = 0U;
            }
        }
        scale_words[static_cast<std::size_t>(token) * (padded_columns / kQ4Group) + group] =
            represented_scale_bits;
    }
    represented_scale_bits = static_cast<std::uint16_t>(
        __shfl(static_cast<unsigned>(represented_scale_bits), 0, kWaveSize));
    const float scale = __half2float(__ushort_as_half(represented_scale_bits));
    const int code0 = column0 < columns && scale != 0.0F
                          ? max(-127, min(127, __float2int_rn(value0 / scale)))
                          : 0;
    const int code1 = column1 < columns && scale != 0.0F
                          ? max(-127, min(127, __float2int_rn(value1 / scale)))
                          : 0;
    const int low0 = code0 & 0x0f;
    const int low1 = code1 & 0x0f;
    const int high0 = (code0 - low0) / 16;
    const int high1 = (code1 - low1) / 16;
    const std::size_t packed =
        (static_cast<std::size_t>(token) * padded_columns + column0) / 2U;
    low_codes[packed] = static_cast<std::uint8_t>(low0 | (low1 << 4));
    high_codes[packed] = static_cast<std::uint8_t>((high0 & 0x0f) |
                                                   ((high1 & 0x0f) << 4));
}

// Small-T codec helpers. Eight adjacent lanes own one G64 group (one 16-byte BF16 vector each);
// the group maximum is a DPP row reduction.
__device__ __forceinline__ float a8g64_group_maximum(float maximum) {
    maximum = fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(maximum), 0x164, 0xf, 0xf, false)));
    maximum = fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(maximum), 0x162, 0xf, 0xf, false)));
    return fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(maximum), 0x161, 0xf, 0xf, false)));
}

// The G64 scale overflows exactly when some finite |x| of the group does: max/127 and the FP16
// conversion are monotone, so status passes need no group reduction. FP16(|x|/127) is infinite
// exactly when |x| >= 8321040 (checked over every finite FP32 value).
__device__ __forceinline__ bool a8g64_element_overflows(float x) {
    return fabsf(x) >= 8321040.0F;
}

// Encodes the eight finite values of one vector with its group's represented scale. Only lanes
// with (lane & 7) == 0 write the scale word.
__device__ __forceinline__ void a8g64_encode_vector(
    const float (&value)[8], float maximum, std::uint32_t lane, std::size_t vector,
    std::size_t scale_index, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words) {
    std::uint16_t scale_bits = 0U;
    if (maximum != 0.0F) {
        scale_bits = __half_as_ushort(__float2half_rn(maximum / 127.0F));
        if (scale_bits == 0U) scale_bits = 1U;
        if (!isfinite(__half2float(__ushort_as_half(scale_bits)))) scale_bits = 0U;
    }
    if ((lane & 7U) == 0U) scale_words[scale_index] = scale_bits;
    const float scale = __half2float(__ushort_as_half(scale_bits));
    std::uint32_t low_word = 0U, high_word = 0U;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const int code = scale != 0.0F
            ? max(-127, min(127, __float2int_rn(value[element] / scale)))
            : 0;
        const int low = code & 0x0f;
        const int high = (code - low) / 16;
        low_word |= static_cast<std::uint32_t>(low) << (element * 4U);
        high_word |= static_cast<std::uint32_t>(high & 0x0f) << (element * 4U);
    }
    reinterpret_cast<std::uint32_t*>(low_codes)[vector] = low_word;
    reinterpret_cast<std::uint32_t*>(high_codes)[vector] = high_word;
}

__device__ __forceinline__ float bf16_word_element(const std::uint32_t (&words)[4],
                                                   std::uint32_t element) {
    const std::uint32_t word = words[element >> 1U];
    return __uint_as_float((element & 1U) ? word & 0xffff0000U : word << 16U);
}

// Small-T form of the same codec: 512-thread CTAs with one 16-byte vector per thread, plus one
// final status CTA that scans every element for the nonfinite/overflow conditions and publishes
// the status word with a plain store. No CTA accumulates into the status word, so no reset launch
// is needed. Rows are unpadded, 16-byte aligned and a multiple of 64 columns.
constexpr std::uint32_t kSmallQuantizeThreads = 512U;
constexpr std::uint32_t kSmallQuantizeMaximumVectors = 8192U;

__global__ __launch_bounds__(kSmallQuantizeThreads) void a8g64_quantize_activation_small_kernel(
    const hip_bfloat16* input, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words, std::uint32_t* status, std::uint32_t tokens,
    std::uint32_t columns) {
    const std::uint32_t row_vectors = columns / 8U;
    const std::uint32_t total = tokens * row_vectors;
    const uint4* vectors = reinterpret_cast<const uint4*>(input);
    if (blockIdx.x == gridDim.x - 1U) {
        constexpr std::uint32_t kInFlight = 8U;
        bool nonfinite = false, overflow = false;
        for (std::uint32_t base = 0; base < total; base += kInFlight * kSmallQuantizeThreads) {
            uint4 loaded[kInFlight];
#pragma unroll
            for (std::uint32_t item = 0; item < kInFlight; ++item) {
                const std::uint32_t vector = base + item * kSmallQuantizeThreads + threadIdx.x;
                loaded[item] = vector < total ? vectors[vector] : uint4{};
            }
#pragma unroll
            for (std::uint32_t item = 0; item < kInFlight; ++item) {
                const std::uint32_t words[4]{loaded[item].x, loaded[item].y, loaded[item].z,
                                             loaded[item].w};
#pragma unroll
                for (std::uint32_t element = 0; element < 8U; ++element) {
                    const float x = bf16_word_element(words, element);
                    nonfinite |= !isfinite(x);
                    overflow |= isfinite(x) && a8g64_element_overflows(x);
                }
            }
        }
        const bool any_nonfinite = __syncthreads_or(nonfinite);
        const bool any_overflow = __syncthreads_or(overflow);
        if (threadIdx.x == 0U)
            *status = (any_nonfinite ? static_cast<std::uint32_t>(Q4G64ActivationNonfinite) : 0U) |
                      (any_overflow ? static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow)
                                    : 0U);
        return;
    }
    const std::uint32_t vector = blockIdx.x * kSmallQuantizeThreads + threadIdx.x;
    const bool valid = vector < total;  // whole G64 groups share validity
    const uint4 packed = valid ? vectors[vector] : uint4{};
    const std::uint32_t words[4]{packed.x, packed.y, packed.z, packed.w};
    float value[8];
    float maximum = 0.0F;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        float x = bf16_word_element(words, element);
        x = isfinite(x) ? x : 0.0F;
        value[element] = x;
        maximum = fmaxf(maximum, fabsf(x));
    }
    maximum = a8g64_group_maximum(maximum);
    if (!valid) return;
    const std::uint32_t token = vector / row_vectors, slot = vector % row_vectors;
    a8g64_encode_vector(value, maximum, threadIdx.x, vector,
                        static_cast<std::size_t>(token) * (columns / kQ4Group) + slot / 8U,
                        low_codes, high_codes, scale_words);
}

// One cooperative RMSNorm row followed by eight independent G64 codec waves.
// The BF16 conversion is the public normalization seam, before the exact codec.
__global__ __launch_bounds__(kThreads) void a8g64_normalized_prepare_t1_kernel(
    const hip_bfloat16* input, const hip_bfloat16* norm, float eps, bool unit_offset,
    std::uint8_t* low_codes, std::uint8_t* high_codes, std::uint16_t* scale_words,
    std::uint32_t* status) {
    constexpr std::uint32_t columns = 5120U;
    constexpr std::uint32_t waves = kThreads / kWaveSize;
    const std::uint32_t lane = threadIdx.x % kWaveSize;
    const std::uint32_t wave = threadIdx.x / kWaveSize;
    float sumsq = 0.0F;
#pragma unroll
    for (std::uint32_t column = threadIdx.x; column < columns; column += kThreads) {
        const float value = static_cast<float>(input[column]);
        sumsq = fmaf(value, value, sumsq);
    }
#pragma unroll
    for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U)
        sumsq += __shfl_down(sumsq, width, kWaveSize);
    __shared__ float wave_sums[waves];
    if (lane == 0U) wave_sums[wave] = sumsq;
    if (threadIdx.x == 0U) *status = 0U;
    __syncthreads();
    if (wave == 0U) {
        float block_sum = lane < waves ? wave_sums[lane] : 0.0F;
#pragma unroll
        for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U)
            block_sum += __shfl_down(block_sum, width, kWaveSize);
        if (lane == 0U) wave_sums[0] = rsqrtf(block_sum / static_cast<float>(columns) + eps);
    }
    __syncthreads();
    const float inverse = wave_sums[0];
    for (std::uint32_t group = wave; group < columns / kQ4Group; group += waves) {
        const std::uint32_t column = group * kQ4Group + lane * 2U;
        auto normalized = [&](std::uint32_t index) {
            const float gain = static_cast<float>(norm[index]) + (unit_offset ? 1.0F : 0.0F);
            float value = static_cast<float>(hip_bfloat16(
                static_cast<float>(input[index]) * inverse * gain));
            if (!isfinite(value)) {
                atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
                value = 0.0F;
            }
            return value;
        };
        const float value0 = normalized(column);
        const float value1 = normalized(column + 1U);
        float maximum = fmaxf(fabsf(value0), fabsf(value1));
#pragma unroll
        for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U)
            maximum = fmaxf(maximum, __shfl_down(maximum, width, kWaveSize));
        std::uint16_t scale_bits = 0U;
        if (lane == 0U) {
            if (maximum != 0.0F) {
                scale_bits = __half_as_ushort(__float2half_rn(maximum / 127.0F));
                if (scale_bits == 0U) scale_bits = 1U;
                if (!isfinite(__half2float(__ushort_as_half(scale_bits)))) {
                    atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
                    scale_bits = 0U;
                }
            }
            scale_words[group] = scale_bits;
        }
        scale_bits = static_cast<std::uint16_t>(
            __shfl(static_cast<unsigned>(scale_bits), 0, kWaveSize));
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int code0 = scale != 0.0F ? max(-127, min(127, __float2int_rn(value0 / scale))) : 0;
        const int code1 = scale != 0.0F ? max(-127, min(127, __float2int_rn(value1 / scale))) : 0;
        const int low0 = code0 & 0x0f;
        const int low1 = code1 & 0x0f;
        const int high0 = (code0 - low0) / 16;
        const int high1 = (code1 - low1) / 16;
        low_codes[column / 2U] = static_cast<std::uint8_t>(low0 | (low1 << 4));
        high_codes[column / 2U] = static_cast<std::uint8_t>((high0 & 0x0f) | ((high1 & 0x0f) << 4));
    }
}

// Prefill form of the normalized A8G64 preparation: one wave per K5120 token row. Each lane
// owns twenty 8-feature vectors (vector v covers features 8(32v+lane)..+7), so eight adjacent
// lanes hold one G64 group. The FP32 sum of squares is lane-serial then a butterfly; the BF16
// conversion is the public normalization seam before the exact codec.
constexpr std::uint32_t kNormalizedPrefillColumns = 5120U;
constexpr std::uint32_t kNormalizedPrefillVectors = kNormalizedPrefillColumns / (8U * kWaveSize);
constexpr std::uint32_t kNormalizedPrefillTokensPerBlock = 8U;

__global__ __launch_bounds__(kNormalizedPrefillTokensPerBlock * kWaveSize)
void a8g64_normalized_prepare_prefill_kernel(
    const hip_bfloat16* input, const hip_bfloat16* norm, float eps, bool unit_offset,
    std::uint32_t tokens, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words, std::uint32_t* status, hip_bfloat16* normalized_rows) {
    constexpr std::uint32_t columns = kNormalizedPrefillColumns;
    const std::uint32_t lane = threadIdx.x % kWaveSize;
    const std::uint32_t token =
        blockIdx.x * kNormalizedPrefillTokensPerBlock + threadIdx.x / kWaveSize;
    if (token >= tokens) return;
    const uint4* row =
        reinterpret_cast<const uint4*>(input + static_cast<std::size_t>(token) * columns);
    uint4 packed[kNormalizedPrefillVectors];
    float sumsq = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < kNormalizedPrefillVectors; ++vector) {
        packed[vector] = row[vector * kWaveSize + lane];
        const std::uint32_t words[4]{packed[vector].x, packed[vector].y, packed[vector].z,
                                     packed[vector].w};
#pragma unroll
        for (std::uint32_t word = 0; word < 4U; ++word) {
            const float low = __uint_as_float(words[word] << 16U);
            const float high = __uint_as_float(words[word] & 0xffff0000U);
            sumsq = fmaf(low, low, sumsq);
            sumsq = fmaf(high, high, sumsq);
        }
    }
#pragma unroll
    for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U)
        sumsq += __shfl_xor(sumsq, width, kWaveSize);
    const float inverse = rsqrtf(sumsq / static_cast<float>(columns) + eps);
    const float offset = unit_offset ? 1.0F : 0.0F;
    bool nonfinite = false, overflow = false;
    const uint4* gains = reinterpret_cast<const uint4*>(norm);
#pragma unroll
    for (std::uint32_t vector = 0; vector < kNormalizedPrefillVectors; ++vector) {
        const std::uint32_t slot = vector * kWaveSize + lane;
        const uint4 gain_words = gains[slot];
        const std::uint32_t words[4]{packed[vector].x, packed[vector].y, packed[vector].z,
                                     packed[vector].w};
        const std::uint32_t gain_packed[4]{gain_words.x, gain_words.y, gain_words.z,
                                           gain_words.w};
        float value[8];
        std::uint32_t seam[4]{};
        float maximum = 0.0F;
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t word = words[element >> 1U];
            const std::uint32_t gain_word = gain_packed[element >> 1U];
            const float x = __uint_as_float((element & 1U) ? word & 0xffff0000U : word << 16U);
            const float gain = __uint_as_float((element & 1U) ? gain_word & 0xffff0000U
                                                              : gain_word << 16U) + offset;
            const hip_bfloat16 rounded(x * inverse * gain);
            seam[element >> 1U] |= static_cast<std::uint32_t>(rounded.data)
                                   << ((element & 1U) * 16U);
            float normalized = static_cast<float>(rounded);
            nonfinite |= !isfinite(normalized);
            normalized = isfinite(normalized) ? normalized : 0.0F;
            value[element] = normalized;
            maximum = fmaxf(maximum, fabsf(normalized));
        }
        // Group maximum over the eight lanes of this vector slot (DPP row_xmask 4,2,1).
        maximum = fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(maximum), 0x164, 0xf, 0xf, false)));
        maximum = fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(maximum), 0x162, 0xf, 0xf, false)));
        maximum = fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(maximum), 0x161, 0xf, 0xf, false)));
        std::uint16_t scale_bits = 0U;
        if (maximum != 0.0F) {
            scale_bits = __half_as_ushort(__float2half_rn(maximum / 127.0F));
            if (scale_bits == 0U) scale_bits = 1U;
            if (!isfinite(__half2float(__ushort_as_half(scale_bits)))) {
                overflow = true;
                scale_bits = 0U;
            }
        }
        if ((lane & 7U) == 0U) {
            scale_words[static_cast<std::size_t>(token) * (columns / kQ4Group) + slot / 8U] =
                scale_bits;
        }
        const float scale = __half2float(__ushort_as_half(scale_bits));
        std::uint32_t low_word = 0U, high_word = 0U;
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const int code = scale != 0.0F
                ? max(-127, min(127, __float2int_rn(value[element] / scale)))
                : 0;
            const int low = code & 0x0f;
            const int high = (code - low) / 16;
            low_word |= static_cast<std::uint32_t>(low) << (element * 4U);
            high_word |= static_cast<std::uint32_t>(high & 0x0f) << (element * 4U);
        }
        const std::size_t packed_offset =
            static_cast<std::size_t>(token) * (columns / 8U) + slot;
        reinterpret_cast<std::uint32_t*>(low_codes)[packed_offset] = low_word;
        reinterpret_cast<std::uint32_t*>(high_codes)[packed_offset] = high_word;
        if (normalized_rows != nullptr) {
            reinterpret_cast<uint4*>(normalized_rows)[packed_offset] =
                make_uint4(seam[0], seam[1], seam[2], seam[3]);
        }
    }
    if (nonfinite) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
    if (overflow) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
}

// Small-T normalized preparation: one 640-thread CTA per K5120 row (one 16-byte vector per
// thread) plus one final status CTA. Per-vector FP32 sums of squares go to LDS and wave 0 reduces
// the row in a fixed order; the same BF16 seam and exact codec as the prefill preparation follow
// with eight adjacent lanes per G64 group. The status CTA repeats the normalization of every row
// at once without encoding, publishes the status word with a plain store and clears
// clear_status, so no reset launch is needed. normalized_rows, when non-null, receives the BF16
// seam rows; clear_status, when non-null, is another status word the status CTA clears.
constexpr std::uint32_t kNormalizedSmallRowVectors = kNormalizedPrefillColumns / 8U;
constexpr std::uint32_t kNormalizedSmallMaximumTokens = 12U;

// Sum of squares of one row from its per-vector partials, in the fixed order of every small route.
__device__ __forceinline__ float normalized_small_row_sum(const float* partial,
                                                          std::uint32_t lane) {
    float sum = 0.0F;
    for (std::uint32_t index = lane; index < kNormalizedSmallRowVectors; index += kWaveSize)
        sum += partial[index];
#pragma unroll
    for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U)
        sum += __shfl_xor(sum, width, kWaveSize);
    return sum;
}

__device__ __forceinline__ float normalized_small_vector_sumsq(uint4 packed) {
    const std::uint32_t words[4]{packed.x, packed.y, packed.z, packed.w};
    float sumsq = 0.0F;
#pragma unroll
    for (std::uint32_t word = 0; word < 4U; ++word) {
        const float low = __uint_as_float(words[word] << 16U);
        const float high = __uint_as_float(words[word] & 0xffff0000U);
        sumsq = fmaf(low, low, sumsq);
        sumsq = fmaf(high, high, sumsq);
    }
    return sumsq;
}

// BF16 seam values of one vector: seam words and their FP32 values (nonfinite kept).
__device__ __forceinline__ void normalized_small_vector_seam(
    uint4 packed, uint4 gain_words, float row_inverse, float offset, std::uint32_t (&seam)[4],
    float (&value)[8]) {
    const std::uint32_t words[4]{packed.x, packed.y, packed.z, packed.w};
    const std::uint32_t gain_packed[4]{gain_words.x, gain_words.y, gain_words.z, gain_words.w};
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const float x = bf16_word_element(words, element);
        const float gain = bf16_word_element(gain_packed, element) + offset;
        const hip_bfloat16 rounded(x * row_inverse * gain);
        seam[element >> 1U] |= static_cast<std::uint32_t>(rounded.data) << ((element & 1U) * 16U);
        value[element] = static_cast<float>(rounded);
    }
}

__global__ __launch_bounds__(kNormalizedSmallRowVectors) void a8g64_normalized_prepare_small_kernel(
    const hip_bfloat16* input, const hip_bfloat16* norm, float eps, bool unit_offset,
    std::uint32_t tokens, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words, std::uint32_t* status, hip_bfloat16* normalized_rows,
    std::uint32_t* clear_status) {
    __shared__ float partial[kNormalizedSmallMaximumTokens][kNormalizedSmallRowVectors];
    __shared__ float inverse[kNormalizedSmallMaximumTokens];
    constexpr std::uint32_t columns = kNormalizedPrefillColumns;
    const std::uint32_t slot = threadIdx.x;
    const std::uint32_t wave = threadIdx.x / kWaveSize, lane = threadIdx.x % kWaveSize;
    const float offset = unit_offset ? 1.0F : 0.0F;
    const uint4 gain_words = reinterpret_cast<const uint4*>(norm)[slot];
    const uint4* rows = reinterpret_cast<const uint4*>(input);
    if (blockIdx.x == tokens) {
        // Rows are predicated, never broken out of, so the unrolled arrays stay in registers.
        uint4 loaded[kNormalizedSmallMaximumTokens];
#pragma unroll
        for (std::uint32_t row = 0; row < kNormalizedSmallMaximumTokens; ++row)
            loaded[row] = row < tokens
                ? rows[static_cast<std::size_t>(row) * kNormalizedSmallRowVectors + slot]
                : uint4{};
#pragma unroll
        for (std::uint32_t row = 0; row < kNormalizedSmallMaximumTokens; ++row)
            if (row < tokens) partial[row][slot] = normalized_small_vector_sumsq(loaded[row]);
        __syncthreads();
        if (wave < tokens) {
            const float sum = normalized_small_row_sum(partial[wave], lane);
            if (lane == 0U) inverse[wave] = rsqrtf(sum / static_cast<float>(columns) + eps);
        }
        __syncthreads();
        bool nonfinite = false, overflow = false;
#pragma unroll
        for (std::uint32_t row = 0; row < kNormalizedSmallMaximumTokens; ++row) {
            if (row < tokens) {
                std::uint32_t seam[4]{};
                float value[8];
                normalized_small_vector_seam(loaded[row], gain_words, inverse[row], offset, seam,
                                             value);
#pragma unroll
                for (std::uint32_t element = 0; element < 8U; ++element) {
                    nonfinite |= !isfinite(value[element]);
                    overflow |= isfinite(value[element]) &&
                                a8g64_element_overflows(value[element]);
                }
            }
        }
        const bool any_nonfinite = __syncthreads_or(nonfinite);
        const bool any_overflow = __syncthreads_or(overflow);
        if (threadIdx.x == 0U) {
            *status = (any_nonfinite ? static_cast<std::uint32_t>(Q4G64ActivationNonfinite) : 0U) |
                      (any_overflow ? static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow)
                                    : 0U);
            // A later stage's status word owned by the same Op (see a8q4g64_normalized_mlp).
            if (clear_status != nullptr) *clear_status = 0U;
        }
        return;
    }
    const std::uint32_t token = blockIdx.x;
    const std::size_t vector = static_cast<std::size_t>(token) * kNormalizedSmallRowVectors + slot;
    const uint4 loaded = rows[vector];
    partial[0][slot] = normalized_small_vector_sumsq(loaded);
    __syncthreads();
    if (wave == 0U) {
        const float sum = normalized_small_row_sum(partial[0], lane);
        if (lane == 0U) inverse[0] = rsqrtf(sum / static_cast<float>(columns) + eps);
    }
    __syncthreads();
    std::uint32_t seam[4]{};
    float value[8];
    normalized_small_vector_seam(loaded, gain_words, inverse[0], offset, seam, value);
    float maximum = 0.0F;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        value[element] = isfinite(value[element]) ? value[element] : 0.0F;
        maximum = fmaxf(maximum, fabsf(value[element]));
    }
    maximum = a8g64_group_maximum(maximum);
    a8g64_encode_vector(value, maximum, lane, vector,
                        static_cast<std::size_t>(token) * (columns / kQ4Group) + slot / 8U,
                        low_codes, high_codes, scale_words);
    if (normalized_rows != nullptr)
        reinterpret_cast<uint4*>(normalized_rows)[vector] =
            make_uint4(seam[0], seam[1], seam[2], seam[3]);
}

// Gated per-head normalization of the Qwen3.8 GDN output into the A8G64 codec, one wave per
// token row of 48 heads x 128 features. Vector v of lane l covers features 8(32v+l)..+7, so the
// 16 lanes of each DPP row hold one head and eight adjacent lanes one G64 group. The value is
// BF16(x * rsqrt(mean_head(x^2) + eps) * w * silu(z)), the public gated-RMSNorm seam, before the
// exact codec.
constexpr std::uint32_t kGatedPrefillColumns = 6144U;
constexpr std::uint32_t kGatedPrefillHeadDim = 128U;
constexpr std::uint32_t kGatedPrefillVectors = kGatedPrefillColumns / (8U * kWaveSize);

// One 16-byte vector (slot = 32 * vector + lane of a token row) of the gated prepare. Callers
// keep each wave on 32 consecutive slots of one token so the DPP rows are whole heads/groups.
__device__ __forceinline__ uint4 a8g64_gated_load(const hip_bfloat16* rows, std::uint32_t token,
                                                  std::uint32_t slot) {
    return reinterpret_cast<const uint4*>(
        rows + static_cast<std::size_t>(token) * kGatedPrefillColumns)[slot];
}

// The eight BF16 gated-seam values of one vector (nonfinite values kept).
__device__ __forceinline__ void a8g64_gated_seam_values(
    uint4 value_words, uint4 gate_words, const hip_bfloat16* norm, float eps, std::uint32_t slot,
    float (&value)[8]) {
    const std::uint32_t lane = slot % kWaveSize;
    // Every head uses the same [128] weight: this lane always reads the same eight gains.
    const uint4 gain_words =
        reinterpret_cast<const uint4*>(norm)[lane % (kGatedPrefillHeadDim / 8U)];
    const std::uint32_t gain_packed[4]{gain_words.x, gain_words.y, gain_words.z, gain_words.w};
    const std::uint32_t x_packed[4]{value_words.x, value_words.y, value_words.z, value_words.w};
    const std::uint32_t z_packed[4]{gate_words.x, gate_words.y, gate_words.z, gate_words.w};
    float sumsq = 0.0F;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const float x = bf16_word_element(x_packed, element);
        sumsq = fmaf(x, x, sumsq);
    }
    // Sum over the 16 lanes (one head) of this DPP row.
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(sumsq), 0x168, 0xf, 0xf, false));
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(sumsq), 0x164, 0xf, 0xf, false));
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(sumsq), 0x162, 0xf, 0xf, false));
    sumsq += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(sumsq), 0x161, 0xf, 0xf, false));
    const float inverse = rsqrtf(sumsq / static_cast<float>(kGatedPrefillHeadDim) + eps);
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const float z = bf16_word_element(z_packed, element);
        const float silu = z / (1.0F + expf(-z));
        value[element] = static_cast<float>(hip_bfloat16(
            bf16_word_element(x_packed, element) * inverse *
            bf16_word_element(gain_packed, element) * silu));
    }
}

__device__ __forceinline__ void a8g64_gated_prepare_vector(
    uint4 value_words, uint4 gate_words, const hip_bfloat16* norm, float eps,
    std::uint32_t token, std::uint32_t slot, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words, bool& nonfinite, bool& overflow) {
    constexpr std::uint32_t columns = kGatedPrefillColumns;
    float value[8];
    a8g64_gated_seam_values(value_words, gate_words, norm, eps, slot, value);
    float maximum = 0.0F;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        nonfinite |= !isfinite(value[element]);
        value[element] = isfinite(value[element]) ? value[element] : 0.0F;
        maximum = fmaxf(maximum, fabsf(value[element]));
    }
    maximum = a8g64_group_maximum(maximum);
    overflow |= maximum != 0.0F &&
                !isfinite(__half2float(__float2half_rn(maximum / 127.0F)));
    a8g64_encode_vector(value, maximum, slot % kWaveSize,
                        static_cast<std::size_t>(token) * (columns / 8U) + slot,
                        static_cast<std::size_t>(token) * (columns / kQ4Group) + slot / 8U,
                        low_codes, high_codes, scale_words);
}

__global__ __launch_bounds__(kNormalizedPrefillTokensPerBlock * kWaveSize)
void a8g64_gated_normalized_prepare_prefill_kernel(
    const hip_bfloat16* input, const hip_bfloat16* norm, const hip_bfloat16* gate, float eps,
    std::uint32_t tokens, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words, std::uint32_t* status) {
    const std::uint32_t lane = threadIdx.x % kWaveSize;
    const std::uint32_t token =
        blockIdx.x * kNormalizedPrefillTokensPerBlock + threadIdx.x / kWaveSize;
    if (token >= tokens) return;
    bool nonfinite = false, overflow = false;
#pragma unroll 4
    for (std::uint32_t vector = 0; vector < kGatedPrefillVectors; ++vector) {
        const std::uint32_t slot = vector * kWaveSize + lane;
        a8g64_gated_prepare_vector(a8g64_gated_load(input, token, slot),
                                   a8g64_gated_load(gate, token, slot), norm, eps, token, slot,
                                   low_codes, high_codes, scale_words, nonfinite, overflow);
    }
    if (nonfinite) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
    if (overflow) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
}

// Small-T form: one 16-byte slot per thread over T x 768 slots (256-thread CTAs keep every wave
// inside one token row), the same per-vector arithmetic. The status word is reset by the
// launcher: a status CTA recomputing every seam value would be the critical path.
constexpr std::uint32_t kGatedSmallMaximumTokens = 8U;
constexpr std::uint32_t kGatedSmallThreads = 256U;
static_assert((kGatedPrefillColumns / 8U) % kGatedSmallThreads == 0U);

__global__ __launch_bounds__(kGatedSmallThreads)
void a8g64_gated_normalized_prepare_small_kernel(
    const hip_bfloat16* input, const hip_bfloat16* norm, const hip_bfloat16* gate, float eps,
    std::uint8_t* low_codes, std::uint8_t* high_codes, std::uint16_t* scale_words,
    std::uint32_t* status) {
    constexpr std::uint32_t row_slots = kGatedPrefillColumns / 8U;
    const std::uint32_t index = blockIdx.x * kGatedSmallThreads + threadIdx.x;
    const std::uint32_t token = index / row_slots, slot = index % row_slots;
    bool nonfinite = false, overflow = false;
    a8g64_gated_prepare_vector(a8g64_gated_load(input, token, slot),
                               a8g64_gated_load(gate, token, slot), norm, eps, token, slot,
                               low_codes, high_codes, scale_words, nonfinite, overflow);
    if (nonfinite) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
    if (overflow) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
}

__global__ __launch_bounds__(kWaveSize) void fused_silu_a8g64_quantize_kernel(
    const hip_bfloat16* gate_up, std::uint8_t* low_codes, std::uint8_t* high_codes,
    std::uint16_t* scale_words, std::uint32_t* status) {
    constexpr std::uint32_t columns = 17408U;
    constexpr std::uint32_t packed_stride = 2U * columns;
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t column0 = group * kQ4Group + lane * 2U;
    const std::uint32_t column1 = column0 + 1U;
    const std::size_t token_base = static_cast<std::size_t>(token) * packed_stride;
    auto rounded_activation = [&](std::uint32_t column) {
        const float gate = static_cast<float>(gate_up[token_base + column]);
        const float up = static_cast<float>(gate_up[token_base + columns + column]);
        const hip_bfloat16 represented(
            (gate / (1.0F + expf(-gate))) * up);
        float value = static_cast<float>(represented);
        if (!isfinite(value)) {
            atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
            value = 0.0F;
        }
        return value;
    };
    const float value0 = rounded_activation(column0);
    const float value1 = rounded_activation(column1);
    float maximum = fmaxf(fabsf(value0), fabsf(value1));
#pragma unroll
    for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down(maximum, width, kWaveSize));
    }
    std::uint16_t represented_scale_bits = 0U;
    if (lane == 0U) {
        if (maximum != 0.0F) {
            represented_scale_bits = __half_as_ushort(__float2half_rn(maximum / 127.0F));
            if (represented_scale_bits == 0U) represented_scale_bits = 1U;
            if (!isfinite(__half2float(__ushort_as_half(represented_scale_bits)))) {
                atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
                represented_scale_bits = 0U;
            }
        }
        scale_words[static_cast<std::size_t>(token) * (columns / kQ4Group) + group] =
            represented_scale_bits;
    }
    represented_scale_bits = static_cast<std::uint16_t>(
        __shfl(static_cast<unsigned>(represented_scale_bits), 0, kWaveSize));
    const float scale = __half2float(__ushort_as_half(represented_scale_bits));
    const int code0 = scale != 0.0F
                          ? max(-127, min(127, __float2int_rn(value0 / scale)))
                          : 0;
    const int code1 = scale != 0.0F
                          ? max(-127, min(127, __float2int_rn(value1 / scale)))
                          : 0;
    const int low0 = code0 & 0x0f;
    const int low1 = code1 & 0x0f;
    const int high0 = (code0 - low0) / 16;
    const int high1 = (code1 - low1) / 16;
    const std::size_t packed =
        (static_cast<std::size_t>(token) * columns + column0) / 2U;
    low_codes[packed] = static_cast<std::uint8_t>(low0 | (low1 << 4));
    high_codes[packed] = static_cast<std::uint8_t>((high0 & 0x0f) |
                                                   ((high1 & 0x0f) << 4));
}

__global__ __launch_bounds__(kWaveSize) void a8g32_quantize_activation_kernel(
    const hip_bfloat16* input, std::int8_t* codes, std::uint16_t* scale_words,
    std::uint32_t* status, std::uint32_t columns, std::uint32_t padded_columns) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t column = group * kW8Group + lane;
    float value = 0.0F;
    if (column < columns) {
        value = static_cast<float>(input[static_cast<std::size_t>(token) * columns + column]);
        if (!isfinite(value)) {
            atomicOr(status, static_cast<std::uint32_t>(A8G32ActivationNonfinite));
            value = 0.0F;
        }
    }
    float maximum = fabsf(value);
#pragma unroll
    for (std::uint32_t width = kWaveSize / 2U; width != 0U; width >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down(maximum, width, kWaveSize));
    }
    std::uint16_t represented_scale_bits = 0U;
    if (lane == 0U) {
        if (maximum != 0.0F) {
            represented_scale_bits = __half_as_ushort(__float2half_rn(maximum / 127.0F));
            if (represented_scale_bits == 0U) represented_scale_bits = 1U;
            if (!isfinite(__half2float(__ushort_as_half(represented_scale_bits)))) {
                atomicOr(status, static_cast<std::uint32_t>(A8G32ActivationScaleOverflow));
                represented_scale_bits = 0U;
            }
        }
        scale_words[static_cast<std::size_t>(token) * (padded_columns / kW8Group) + group] =
            represented_scale_bits;
    }
    represented_scale_bits = static_cast<std::uint16_t>(
        __shfl(static_cast<unsigned>(represented_scale_bits), 0, kWaveSize));
    const float scale = __half2float(__ushort_as_half(represented_scale_bits));
    const int code = column < columns && scale != 0.0F
                         ? max(-127, min(127, __float2int_rn(value / scale)))
                         : 0;
    codes[static_cast<std::size_t>(token) * padded_columns + column] =
        static_cast<std::int8_t>(code);
}

__device__ int q4_code(const std::uint8_t* codes, std::size_t index) {
    const unsigned nibble = (codes[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

__device__ int q4_n16k16_code(const std::uint8_t* codes, std::uint32_t row,
                              std::uint32_t column, std::uint32_t groups) {
    const std::uint32_t group = column / kQ4Group;
    const std::uint32_t within = column & (kQ4Group - 1U);
    const std::uint64_t pair = reinterpret_cast<const std::uint64_t*>(codes)[
        q4_n16k16_pair_index(row, group, within / 16U, groups)];
    const unsigned nibble = static_cast<unsigned>(pair >> ((within & 15U) * 4U)) & 15U;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

__device__ hip_bfloat16 conspicuous_bf16_nan() {
    hip_bfloat16 value;
    value.data = 0x7fc1U;
    return value;
}

__global__ void q4g64_linear_vector_kernel(
    const std::uint8_t* activation_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t padded_columns) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    if (row >= rows || token >= tokens) return;
    if (*activation_status != Q4G64ActivationOk) {
        if (threadIdx.x == 0U) {
            output[static_cast<std::size_t>(token) * rows + row] = conspicuous_bf16_nan();
        }
        return;
    }
    const std::size_t groups = padded_columns / kQ4Group;
    float partial = 0.0F;
    for (std::uint32_t group = threadIdx.x; group < groups; group += blockDim.x) {
        std::int32_t dot = 0;
        const std::size_t base = static_cast<std::size_t>(group) * kQ4Group;
        const std::size_t a_base = static_cast<std::size_t>(token) * padded_columns + base;
#pragma unroll
        for (std::uint32_t lane = 0; lane < kQ4Group; ++lane) {
            dot += q4_code(activation_codes, a_base + lane) *
                   q4_n16k16_code(weight_codes, row,
                                  static_cast<std::uint32_t>(base) + lane,
                                  static_cast<std::uint32_t>(groups));
        }
        const float a_scale = __half2float(__ushort_as_half(
            activation_scales[static_cast<std::size_t>(token) * groups + group]));
        const float w_scale = __half2float(__ushort_as_half(
            weight_scales[q4_n16k16_scale_index(
                row, group, static_cast<std::uint32_t>(groups))]));
        partial = fmaf(static_cast<float>(dot), a_scale * w_scale, partial);
    }
    __shared__ float reductions[kThreads];
    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t width = kThreads / 2U; width != 0U; width >>= 1U) {
        if (threadIdx.x < width) reductions[threadIdx.x] += reductions[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        output[static_cast<std::size_t>(token) * rows + row] =
            static_cast<hip_bfloat16>(reductions[0]);
    }
}

template <bool W8, std::uint32_t WavesPerBlock>
__global__ void wave8_linear_kernel(const hip_bfloat16* input, const void* weights,
                                    const std::uint16_t* w8_scales,
                                    hip_bfloat16* output, std::uint32_t columns,
                                    std::uint32_t rows, std::size_t padded_columns,
                                    std::uint32_t tokens) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t wave = threadIdx.x / kWaveSize;
    const std::uint32_t lane = threadIdx.x & (kWaveSize - 1U);
    const std::uint32_t token = blockIdx.y * WavesPerBlock + wave;
    if (row >= rows) return;
    __shared__ float weight_tile[kCooperativeTile];
    float partial = 0.0F;
    const auto* bf16_weights = static_cast<const hip_bfloat16*>(weights);
    const auto* codes = static_cast<const std::int8_t*>(weights);
    const std::size_t groups_per_row = padded_columns / kW8Group;
    for (std::uint32_t base = 0; base < columns; base += kCooperativeTile) {
        const std::uint32_t k = base + threadIdx.x;
        // The A16 prefill CTA has sixteen waves but this staging tile holds
        // only 256 values. Extra consumer waves must not write beyond it.
        if (threadIdx.x < kCooperativeTile && k < columns) {
            if constexpr (W8) {
                const std::int32_t code = static_cast<std::int8_t>(
                    codes[static_cast<std::size_t>(row) * padded_columns + k]);
                const std::uint16_t scale_bits =
                    w8_scales[static_cast<std::size_t>(row) * groups_per_row + k / kW8Group];
                weight_tile[threadIdx.x] =
                    static_cast<float>(code) * __half2float(__ushort_as_half(scale_bits));
            } else {
                weight_tile[threadIdx.x] =
                    static_cast<float>(bf16_weights[static_cast<std::size_t>(row) * columns + k]);
            }
        }
        __syncthreads();
        if (token < tokens) {
            const std::size_t input_base = static_cast<std::size_t>(token) * columns + base;
            for (std::uint32_t local = lane; local < kCooperativeTile && base + local < columns;
                 local += kWaveSize) {
                partial = fmaf(static_cast<float>(input[input_base + local]), weight_tile[local], partial);
            }
        }
        __syncthreads();
    }
    for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
        partial += __shfl_down(partial, delta, kWaveSize);
    }
    if (token < tokens && lane == 0U) {
        output[static_cast<std::size_t>(token) * rows + row] = static_cast<hip_bfloat16>(partial);
    }
}

// GFX12 wave32 WMMA's owned raw map is A[axis][8*half+element],
// B[8*half+element][axis], and C[8*half+element][axis]. This is the same
// asymmetric fragment map independently checked by the R9700 opcode probe.
// This is the exact BF16 input/weight contract; W8G32 cannot use it without
// an observable (and uncontracted) dequantization/rounding change.
#if defined(__HIPCC__)
using WmmaBf16Fragment = __attribute__((__vector_size__(8 * sizeof(short)))) short;
using WmmaF32Fragment = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using WmmaI32x2 = int __attribute__((ext_vector_type(2)));
using WmmaI32x8 = int __attribute__((ext_vector_type(8)));

__device__ short bf16_bits(hip_bfloat16 value) {
    return static_cast<short>(value.data);
}

__global__ void bf16_linear_wmma16_kernel(const hip_bfloat16* input,
                                           const hip_bfloat16* weights,
                                           hip_bfloat16* output,
                                           std::uint32_t columns,
                                           std::uint32_t rows,
                                           std::uint32_t tokens) {
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t token_base = blockIdx.y * kWmmaTile;
    const std::uint32_t row_base = blockIdx.x * kWmmaTile;
    const std::uint32_t lane_group = lane / 16U;
    const std::uint32_t lane_wrapped = lane & 15U;
    WmmaF32Fragment accum{};
    for (std::uint32_t base = 0; base < columns; base += kWmmaTile) {
        WmmaBf16Fragment a{};
        WmmaBf16Fragment b{};
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t k = base + lane_group * 8U + element;
            const std::uint32_t token = token_base + lane_wrapped;
            const std::uint32_t row = row_base + lane_wrapped;
            a[element] = token < tokens && k < columns
                             ? bf16_bits(input[static_cast<std::size_t>(token) * columns + k])
                             : 0;
            b[element] = row < rows && k < columns
                             ? bf16_bits(weights[static_cast<std::size_t>(row) * columns + k])
                             : 0;
        }
        accum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, accum);
    }
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const std::uint32_t token = token_base + lane_group * 8U + element;
        const std::uint32_t row = row_base + lane_wrapped;
        if (token < tokens && row < rows) {
            output[static_cast<std::size_t>(token) * rows + row] =
                static_cast<hip_bfloat16>(accum[element]);
        }
    }
}

__device__ __attribute__((unused)) int load_q4x8(const std::uint8_t* codes,
                                                  std::size_t element) {
    return *reinterpret_cast<const int*>(codes + element / 2U);
}

__device__ __forceinline__ std::uint64_t load_q4_n16k16_pair(
    const std::uint8_t* codes, std::uint32_t row, std::uint32_t group,
    std::uint32_t pair, std::uint32_t groups) {
    return reinterpret_cast<const std::uint64_t*>(codes)[
        q4_n16k16_pair_index(row, group, pair, groups)];
}

__device__ __forceinline__ int load_q4_n16k16_x8(
    const std::uint8_t* codes, std::uint32_t row, std::uint32_t group,
    std::uint32_t word, std::uint32_t groups) {
    const std::uint64_t pair = load_q4_n16k16_pair(
        codes, row, group, word >> 1U, groups);
    return static_cast<int>(word & 1U ? pair >> 32U : pair);
}

__device__ __forceinline__ __attribute__((unused)) std::uint32_t load_q4_n16k16_x8_logical(
    const std::uint8_t* codes, std::uint32_t row, std::uint32_t group,
    std::uint32_t word, std::uint32_t groups, std::uint32_t logical_columns) {
    const std::uint32_t packed_k = group * kQ4Group + word * 8U;
    if (packed_k >= logical_columns) return 0U;
    const std::uint32_t packed = static_cast<std::uint32_t>(
        load_q4_n16k16_x8(codes, row, group, word, groups));
    const std::uint32_t remaining = logical_columns - packed_k;
    return remaining >= 8U ? packed : packed & ((1U << (remaining * 4U)) - 1U);
}

__global__ __launch_bounds__(kWaveSize) void q4g64_linear_wmma32_kernel(
    const std::uint8_t* activation_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t token_base = blockIdx.y * kWmmaTile;
    const std::uint32_t row_base = blockIdx.x * kWmmaTile;
    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const std::uint32_t row = row_base + axis;
            if (token < tokens && row < rows) {
                output[static_cast<std::size_t>(token) * rows + row] =
                    conspicuous_bf16_nan();
            }
        }
        return;
    }
    const std::uint32_t groups = padded_columns / kQ4Group;
    WmmaF32Fragment total{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        WmmaI32x8 integer_dot{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            const std::uint32_t k_base = group * kQ4Group + half * 32U + lane_group * 16U;
            WmmaI32x2 a{};
            WmmaI32x2 b{};
            const std::uint32_t token = token_base + axis;
            const std::uint32_t row = row_base + axis;
            if (token < tokens) {
                const std::size_t element = static_cast<std::size_t>(token) * padded_columns + k_base;
                a[0] = load_q4x8(activation_codes, element);
                a[1] = load_q4x8(activation_codes, element + 8U);
            }
            if (row < rows) {
                const std::uint32_t word = (k_base & (kQ4Group - 1U)) / 8U;
                b[0] = load_q4_n16k16_x8(weight_codes, row, group, word, groups);
                b[1] = load_q4_n16k16_x8(weight_codes, row, group, word + 1U, groups);
            }
            integer_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a, true, b, integer_dot, false);
        }
        const std::uint32_t row = row_base + axis;
        const float w_scale = row < rows
                                  ? __half2float(__ushort_as_half(weight_scales[
                                        q4_n16k16_scale_index(row, group, groups)]))
                                  : 0.0F;
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const float a_scale = token < tokens
                                      ? __half2float(__ushort_as_half(activation_scales[
                                            static_cast<std::size_t>(token) * groups + group]))
                                      : 0.0F;
            total[element] = fmaf(static_cast<float>(integer_dot[element]),
                                  a_scale * w_scale, total[element]);
        }
    }
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const std::uint32_t token = token_base + lane_group * 8U + element;
        const std::uint32_t row = row_base + axis;
        if (token < tokens && row < rows) {
            output[static_cast<std::size_t>(token) * rows + row] =
                static_cast<hip_bfloat16>(total[element]);
        }
    }
#else
    (void)activation_codes; (void)activation_scales; (void)activation_status;
    (void)weight_codes;
    (void)weight_scales; (void)output; (void)tokens; (void)rows;
    (void)padded_columns;
#endif
}

__global__ __launch_bounds__(kWaveSize) void a8q4g64_linear_wmma32_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t token_base = blockIdx.y * kWmmaTile;
    const std::uint32_t row_base = blockIdx.x * kWmmaTile;
    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const std::uint32_t row = row_base + axis;
            if (token < tokens && row < rows) {
                output[static_cast<std::size_t>(token) * rows + row] =
                    conspicuous_bf16_nan();
            }
        }
        return;
    }
    const std::uint32_t groups = padded_columns / kQ4Group;
    WmmaF32Fragment total{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        WmmaI32x8 low_dot{};
        WmmaI32x8 high_dot{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            const std::uint32_t k_base = group * kQ4Group + half * 32U + lane_group * 16U;
            WmmaI32x2 a_low{};
            WmmaI32x2 a_high{};
            WmmaI32x2 b{};
            const std::uint32_t token = token_base + axis;
            const std::uint32_t row = row_base + axis;
            if (token < tokens) {
                const std::size_t element =
                    static_cast<std::size_t>(token) * padded_columns + k_base;
                a_low[0] = load_q4x8(activation_low_codes, element);
                a_low[1] = load_q4x8(activation_low_codes, element + 8U);
                a_high[0] = load_q4x8(activation_high_codes, element);
                a_high[1] = load_q4x8(activation_high_codes, element + 8U);
            }
            if (row < rows) {
                const std::uint32_t word = (k_base & (kQ4Group - 1U)) / 8U;
                b[0] = load_q4_n16k16_x8(weight_codes, row, group, word, groups);
                b[1] = load_q4_n16k16_x8(weight_codes, row, group, word + 1U, groups);
            }
            low_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a_low, true, b, low_dot, false);
            high_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_high, true, b, high_dot, false);
        }
        const std::uint32_t row = row_base + axis;
        const float w_scale = row < rows
                                  ? __half2float(__ushort_as_half(weight_scales[
                                        q4_n16k16_scale_index(row, group, groups)]))
                                  : 0.0F;
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const float a_scale = token < tokens
                                      ? __half2float(__ushort_as_half(activation_scales[
                                            static_cast<std::size_t>(token) * groups + group]))
                                      : 0.0F;
            const std::int32_t combined = low_dot[element] + 16 * high_dot[element];
            total[element] = fmaf(static_cast<float>(combined),
                                  a_scale * w_scale, total[element]);
        }
    }
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const std::uint32_t token = token_base + lane_group * 8U + element;
        const std::uint32_t row = row_base + axis;
        if (token < tokens && row < rows) {
            output[static_cast<std::size_t>(token) * rows + row] =
                static_cast<hip_bfloat16>(total[element]);
        }
    }
#else
    (void)activation_low_codes; (void)activation_high_codes; (void)activation_scales;
    (void)activation_status;
    (void)weight_codes; (void)weight_scales; (void)output; (void)tokens; (void)rows;
    (void)padded_columns;
#endif
}

__global__ __launch_bounds__(kThreads)
void a8q4g64_linear_decode_dot8_t1_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t rows, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    if (*activation_status != Q4G64ActivationOk) {
        output[row] = conspicuous_bf16_nan();
        return;
    }
    const std::uint32_t groups = padded_columns / kQ4Group;
    float total = 0.0F;
    for (std::uint32_t group = 0; group < groups; ++group) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (std::uint32_t word = 0; word < 8U; ++word) {
            const std::size_t element =
                static_cast<std::size_t>(group) * kQ4Group + word * 8U;
            const int low = load_q4x8(activation_low_codes, element);
            const int high = load_q4x8(activation_high_codes, element);
            const int weight =
                load_q4_n16k16_x8(weight_codes, row, group, word, groups);
            low_dot = __builtin_amdgcn_sudot8(
                false, low, true, weight, low_dot, false);
            high_dot = __builtin_amdgcn_sudot8(
                true, high, true, weight, high_dot, false);
        }
        const std::int32_t combined = low_dot + 16 * high_dot;
        const float activation_scale = __half2float(
            __ushort_as_half(activation_scales[group]));
        const float weight_scale = __half2float(__ushort_as_half(
            weight_scales[q4_n16k16_scale_index(row, group, groups)]));
        total = fmaf(static_cast<float>(combined),
                     activation_scale * weight_scale, total);
    }
    output[row] = static_cast<hip_bfloat16>(total);
#else
    (void)activation_low_codes; (void)activation_high_codes;
    (void)activation_scales; (void)activation_status; (void)weight_codes;
    (void)weight_scales; (void)output; (void)rows; (void)padded_columns;
#endif
}

__global__ __launch_bounds__(kThreads) void a8q4g64_projected_residual_t1_kernel(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* residual, std::uint32_t columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 5120U) return;
    if (*activation_status != Q4G64ActivationOk) {
        residual[row] = conspicuous_bf16_nan();
        return;
    }
    const std::uint32_t groups = columns / kQ4Group;
    float total = 0.0F;
    for (std::uint32_t group = 0U; group < groups; ++group) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (std::uint32_t word = 0U; word < 8U; ++word) {
            const std::size_t element =
                static_cast<std::size_t>(group) * kQ4Group + word * 8U;
            const int low = load_q4x8(activation_low, element);
            const int high = load_q4x8(activation_high, element);
            const int weight =
                load_q4_n16k16_x8(weight_codes, row, group, word, groups);
            low_dot = __builtin_amdgcn_sudot8(false, low, true, weight, low_dot, false);
            high_dot = __builtin_amdgcn_sudot8(true, high, true, weight, high_dot, false);
        }
        const int combined = low_dot + 16 * high_dot;
        const float activation_scale =
            __half2float(__ushort_as_half(activation_scales[group]));
        const float weight_scale = __half2float(__ushort_as_half(
            weight_scales[q4_n16k16_scale_index(row, group, groups)]));
        total = fmaf(static_cast<float>(combined),
                     activation_scale * weight_scale, total);
    }
    // This BF16 seam is part of the projected-residual boundary, even though
    // the intermediate projection never needs a global-memory allocation.
    const hip_bfloat16 delta = static_cast<hip_bfloat16>(total);
    residual[row] = hip_bfloat16(static_cast<float>(residual[row]) +
                                 static_cast<float>(delta));
#else
    (void)activation_low; (void)activation_high; (void)activation_scales;
    (void)activation_status; (void)weight_codes; (void)weight_scales;
    (void)residual; (void)columns;
#endif
}

__global__ __launch_bounds__(kThreads)
void a8q4g64_projected_residual_down_stage2_kernel(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* residual) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    constexpr std::uint32_t groups = 17408U / kQ4Group;
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 5120U) return;
    if (*activation_status != Q4G64ActivationOk) {
        residual[row] = conspicuous_bf16_nan();
        return;
    }
    float total = 0.0F;
    for (std::uint32_t base = 0U; base < groups; base += 2U) {
        int low[2][8], high[2][8], weight[2][8];
        std::uint16_t a_scale_bits[2], w_scale_bits[2];
#pragma unroll
        for (unsigned stage = 0U; stage < 2U; ++stage) {
            const std::uint32_t group = base + stage;
#pragma unroll
            for (unsigned word = 0U; word < 8U; ++word) {
                const std::size_t element =
                    static_cast<std::size_t>(group) * kQ4Group + word * 8U;
                low[stage][word] = load_q4x8(activation_low, element);
                high[stage][word] = load_q4x8(activation_high, element);
                weight[stage][word] =
                    load_q4_n16k16_x8(weight_codes, row, group, word, groups);
            }
            a_scale_bits[stage] = activation_scales[group];
            w_scale_bits[stage] =
                weight_scales[q4_n16k16_scale_index(row, group, groups)];
        }
        // Issue both raw groups before compute; never decode a future scale
        // before the current group's arithmetic. Each original dot/FMA chain
        // and the explicit BF16 residual seam remain unchanged.
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (unsigned stage = 0U; stage < 2U; ++stage) {
            int low_dot = 0;
            int high_dot = 0;
#pragma unroll
            for (unsigned word = 0U; word < 8U; ++word) {
                low_dot = __builtin_amdgcn_sudot8(
                    false, low[stage][word], true, weight[stage][word], low_dot, false);
                high_dot = __builtin_amdgcn_sudot8(
                    true, high[stage][word], true, weight[stage][word], high_dot, false);
            }
            const int combined = low_dot + 16 * high_dot;
            const float a_scale = __half2float(__ushort_as_half(a_scale_bits[stage]));
            const float w_scale = __half2float(__ushort_as_half(w_scale_bits[stage]));
            total = fmaf(static_cast<float>(combined), a_scale * w_scale, total);
        }
    }
    const hip_bfloat16 delta = static_cast<hip_bfloat16>(total);
    residual[row] = hip_bfloat16(static_cast<float>(residual[row]) +
                                 static_cast<float>(delta));
#else
    (void)activation_low; (void)activation_high; (void)activation_scales;
    (void)activation_status; (void)weight_codes; (void)weight_scales; (void)residual;
#endif
}

__global__ __launch_bounds__(kThreads)
void a8q4g64_gdn_pair_t1_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight0_codes, const std::uint16_t* weight0_scales,
    hip_bfloat16* output0,
    const std::uint8_t* weight1_codes, const std::uint16_t* weight1_scales,
    hip_bfloat16* output1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    constexpr std::uint32_t kRows0 = 4096U;
    constexpr std::uint32_t kRows1 = 12288U;
    constexpr std::uint32_t kColumns = 5120U;
    constexpr std::uint32_t kGroups = kColumns / kQ4Group;
    const std::uint32_t combined_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (combined_row >= kRows0 + kRows1) return;
    const bool second = combined_row >= kRows0;
    const std::uint32_t row = second ? combined_row - kRows0 : combined_row;
    hip_bfloat16* const output = second ? output1 : output0;
    if (*activation_status != Q4G64ActivationOk) {
        output[row] = conspicuous_bf16_nan();
        return;
    }
    const std::uint8_t* const weight_codes = second ? weight1_codes : weight0_codes;
    const std::uint16_t* const weight_scales = second ? weight1_scales : weight0_scales;
    float total = 0.0F;
    for (std::uint32_t group = 0; group < kGroups; ++group) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (std::uint32_t word = 0; word < 8U; ++word) {
            const std::size_t element =
                static_cast<std::size_t>(group) * kQ4Group + word * 8U;
            const int low = load_q4x8(activation_low_codes, element);
            const int high = load_q4x8(activation_high_codes, element);
            const int weight =
                load_q4_n16k16_x8(weight_codes, row, group, word, kGroups);
            low_dot = __builtin_amdgcn_sudot8(
                false, low, true, weight, low_dot, false);
            high_dot = __builtin_amdgcn_sudot8(
                true, high, true, weight, high_dot, false);
        }
        const std::int32_t combined = low_dot + 16 * high_dot;
        const float activation_scale = __half2float(
            __ushort_as_half(activation_scales[group]));
        const float weight_scale = __half2float(__ushort_as_half(
            weight_scales[q4_n16k16_scale_index(row, group, kGroups)]));
        total = fmaf(static_cast<float>(combined),
                     activation_scale * weight_scale, total);
    }
    output[row] = static_cast<hip_bfloat16>(total);
#else
    (void)activation_low_codes; (void)activation_high_codes;
    (void)activation_scales; (void)activation_status; (void)weight0_codes;
    (void)weight0_scales; (void)output0; (void)weight1_codes;
    (void)weight1_scales; (void)output1;
#endif
}

template <std::uint32_t Tokens>
__global__ __launch_bounds__(kThreads)
void a8q4g64_linear_dflash_small_t_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t rows, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    static_assert(Tokens == 4U || Tokens == 5U || Tokens == 6U || Tokens == 8U ||
                  Tokens == 10U || Tokens == 12U || Tokens == 15U || Tokens == 16U ||
                  Tokens == 18U || Tokens == 20U || Tokens == 24U);
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t token = 0; token < Tokens; ++token) {
            output[static_cast<std::size_t>(token) * rows + row] = conspicuous_bf16_nan();
        }
        return;
    }
    const std::uint32_t groups = padded_columns / kQ4Group;
    float totals[Tokens]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        int low_dots[Tokens]{};
        int high_dots[Tokens]{};
#pragma unroll
        for (std::uint32_t word = 0; word < 8U; ++word) {
            const int weight =
                load_q4_n16k16_x8(weight_codes, row, group, word, groups);
#pragma unroll
            for (std::uint32_t token = 0; token < Tokens; ++token) {
                const std::size_t element =
                    static_cast<std::size_t>(token) * padded_columns +
                    static_cast<std::size_t>(group) * kQ4Group + word * 8U;
                const int low = load_q4x8(activation_low_codes, element);
                const int high = load_q4x8(activation_high_codes, element);
                low_dots[token] = __builtin_amdgcn_sudot8(
                    false, low, true, weight, low_dots[token], false);
                high_dots[token] = __builtin_amdgcn_sudot8(
                    true, high, true, weight, high_dots[token], false);
            }
        }
        const float weight_scale = __half2float(__ushort_as_half(
            weight_scales[q4_n16k16_scale_index(row, group, groups)]));
#pragma unroll
        for (std::uint32_t token = 0; token < Tokens; ++token) {
            const std::int32_t combined = low_dots[token] + 16 * high_dots[token];
            const float activation_scale = __half2float(__ushort_as_half(
                activation_scales[static_cast<std::size_t>(token) * groups + group]));
            totals[token] = fmaf(static_cast<float>(combined),
                                 activation_scale * weight_scale, totals[token]);
        }
    }
#pragma unroll
    for (std::uint32_t token = 0; token < Tokens; ++token) {
        output[static_cast<std::size_t>(token) * rows + row] =
            static_cast<hip_bfloat16>(totals[token]);
    }
#else
    (void)activation_low_codes; (void)activation_high_codes;
    (void)activation_scales; (void)activation_status; (void)weight_codes;
    (void)weight_scales; (void)output; (void)rows; (void)padded_columns;
#endif
}

// Exact qualification-selected DFlash MLP-down cell. Keeping N, K, and T compile-time constants
// preserves the accepted instruction/resource shape; the public Linear boundary still owns all
// pointer, extent, overlap, codec-status, and stream semantics.
__global__ __launch_bounds__(kThreads)
void a8q4g64_linear_dflash_mlp_down_t5_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    constexpr std::uint32_t tokens = 5U;
    constexpr std::uint32_t rows = 5120U;
    constexpr std::uint32_t columns = 17408U;
    constexpr std::uint32_t groups = columns / kQ4Group;
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t token = 0; token < tokens; ++token) {
            output[static_cast<std::size_t>(token) * rows + row] = conspicuous_bf16_nan();
        }
        return;
    }
    float totals[tokens]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        int low_dots[tokens]{};
        int high_dots[tokens]{};
#pragma unroll
        for (std::uint32_t word = 0; word < 8U; ++word) {
            const int weight = load_q4_n16k16_x8(weight_codes, row, group, word, groups);
#pragma unroll
            for (std::uint32_t token = 0; token < tokens; ++token) {
                const std::size_t element = static_cast<std::size_t>(token) * columns +
                    static_cast<std::size_t>(group) * kQ4Group + word * 8U;
                low_dots[token] = __builtin_amdgcn_sudot8(
                    false, load_q4x8(activation_low_codes, element), true, weight,
                    low_dots[token], false);
                high_dots[token] = __builtin_amdgcn_sudot8(
                    true, load_q4x8(activation_high_codes, element), true, weight,
                    high_dots[token], false);
            }
        }
        const float weight_scale = __half2float(__ushort_as_half(
            weight_scales[q4_n16k16_scale_index(row, group, groups)]));
#pragma unroll
        for (std::uint32_t token = 0; token < tokens; ++token) {
            const float activation_scale = __half2float(__ushort_as_half(
                activation_scales[static_cast<std::size_t>(token) * groups + group]));
            totals[token] = fmaf(static_cast<float>(low_dots[token] + 16 * high_dots[token]),
                                 activation_scale * weight_scale, totals[token]);
        }
    }
#pragma unroll
    for (std::uint32_t token = 0; token < tokens; ++token) {
        output[static_cast<std::size_t>(token) * rows + row] =
            static_cast<hip_bfloat16>(totals[token]);
    }
#else
    (void)activation_low_codes; (void)activation_high_codes;
    (void)activation_scales; (void)activation_status; (void)weight_codes;
    (void)weight_scales; (void)output;
#endif
}

// A fragment-major LDS map avoids the 4-way conflicts produced by a logical
// [row][packed-K-word] staging image. Bit 4 separates the two WMMA lane groups
// onto disjoint banks while preserving contiguous rows within each group. The
// persistent N16/K16 source is mapped directly into this LDS image while loading.
__device__ __forceinline__ __attribute__((unused)) std::uint32_t a8q4_lds_word_index(
    std::uint32_t row, std::uint32_t word, std::uint32_t rows) {
    return word * rows + (row ^ (((word >> 1U) & 1U) << 4U));
}

__device__ __forceinline__ __attribute__((unused)) std::uint32_t load_q4x8_logical(
    const std::uint8_t* codes, std::size_t element, std::uint32_t logical_k,
    std::uint32_t packed_k) {
    if (packed_k >= logical_k) return 0U;
    const std::uint32_t packed = static_cast<std::uint32_t>(
        load_q4x8(codes, element));
    const std::uint32_t remaining = logical_k - packed_k;
    if (remaining >= 8U) return packed;
    return packed & ((1U << (remaining * 4U)) - 1U);
}

struct A4Q4PrefillN128Tile {
    std::uint32_t activations[8U * kA8Q4PrefillTokenTile];
    std::uint32_t weights[8U * 2U * kA8Q4PrefillRowTile];
    std::uint16_t activation_scales[kA8Q4PrefillTokenTile];
    std::uint16_t weight_scales[2U * kA8Q4PrefillRowTile];
};

static_assert(sizeof(A4Q4PrefillN128Tile) == 6528U);

template <bool LdsScopeOnly>
__device__ __forceinline__ void prefill_cta_barrier();

__global__ __launch_bounds__(kA8Q4PrefillThreads)
void q4g64_linear_prefill_cta_m64n128_kernel(
    const std::uint8_t* activation_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ A4Q4PrefillN128Tile tile;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & (kWaveSize - 1U);
    const std::uint32_t wave = thread / kWaveSize;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_token_tile = wave / 4U;
    const std::uint32_t wave_row_tile = wave & 3U;
    const std::uint32_t cta_token_base = blockIdx.y * kA8Q4PrefillTokenTile;
    const std::uint32_t cta_row_base = blockIdx.x * (2U * kA8Q4PrefillRowTile);
    const std::uint32_t token_base = cta_token_base + wave_token_tile * kWmmaTile;

    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t token = token_base + lane_group * 8U + element;
                const std::uint32_t row = cta_row_base + row_half * kA8Q4PrefillRowTile +
                                          wave_row_tile * kWmmaTile + axis;
                if (token < tokens && row < rows) {
                    output[static_cast<std::size_t>(token) * rows + row] =
                        conspicuous_bf16_nan();
                }
            }
        }
        return;
    }

    const std::uint32_t groups = padded_columns / kQ4Group;
    WmmaF32Fragment totals[2]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t word = thread & 7U;
        const std::uint32_t activation_row = thread / 8U;
        const std::uint32_t token = cta_token_base + activation_row;
        const std::uint32_t packed_k = group * kQ4Group + word * 8U;
        const std::size_t activation_element =
            static_cast<std::size_t>(token) * padded_columns + packed_k;
        tile.activations[a8q4_lds_word_index(
            activation_row, word, kA8Q4PrefillTokenTile)] = token < tokens
            ? load_q4x8_logical(activation_codes, activation_element, columns, packed_k)
            : 0U;
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
            const std::uint32_t weight_row = thread / 8U + row_half * kA8Q4PrefillRowTile;
            const std::uint32_t row = cta_row_base + weight_row;
            tile.weights[a8q4_lds_word_index(
                weight_row, word, 2U * kA8Q4PrefillRowTile)] = row < rows
                ? load_q4_n16k16_x8_logical(
                      weight_codes, row, group, word, groups, columns)
                : 0U;
        }
        if (thread < kA8Q4PrefillTokenTile) {
            const std::uint32_t scale_token = cta_token_base + thread;
            tile.activation_scales[thread] = scale_token < tokens
                ? activation_scales[static_cast<std::size_t>(scale_token) * groups + group]
                : 0U;
        }
        if (thread < 2U * kA8Q4PrefillRowTile) {
            const std::uint32_t scale_row = cta_row_base + thread;
            tile.weight_scales[thread] = scale_row < rows
                ? weight_scales[q4_n16k16_scale_index(scale_row, group, groups)]
                : 0U;
        }
        prefill_cta_barrier<true>();

        WmmaI32x8 dots[2]{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            WmmaI32x2 a{};
            WmmaI32x2 b[2]{};
            const std::uint32_t fragment_word = half * 4U + lane_group * 2U;
            const std::uint32_t activation_fragment_row =
                wave_token_tile * kWmmaTile + axis;
            const std::uint32_t a_index = a8q4_lds_word_index(
                activation_fragment_row, fragment_word, kA8Q4PrefillTokenTile);
            a[0] = static_cast<int>(tile.activations[a_index]);
            a[1] = static_cast<int>(tile.activations[a8q4_lds_word_index(
                activation_fragment_row, fragment_word + 1U,
                kA8Q4PrefillTokenTile)]);
#pragma unroll
            for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
                const std::uint32_t weight_fragment_row =
                    row_half * kA8Q4PrefillRowTile + wave_row_tile * kWmmaTile + axis;
                const std::uint32_t b_index = a8q4_lds_word_index(
                    weight_fragment_row, fragment_word, 2U * kA8Q4PrefillRowTile);
                b[row_half][0] = static_cast<int>(tile.weights[b_index]);
                b[row_half][1] = static_cast<int>(tile.weights[a8q4_lds_word_index(
                    weight_fragment_row, fragment_word + 1U,
                    2U * kA8Q4PrefillRowTile)]);
                dots[row_half] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    true, a, true, b[row_half], dots[row_half], false);
            }
        }
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
            const std::uint32_t fragment_row = row_half * kA8Q4PrefillRowTile +
                                               wave_row_tile * kWmmaTile + axis;
            const float w_scale = __half2float(__ushort_as_half(
                tile.weight_scales[fragment_row]));
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t fragment_token =
                    wave_token_tile * kWmmaTile + lane_group * 8U + element;
                const float a_scale = __half2float(__ushort_as_half(
                    tile.activation_scales[fragment_token]));
                totals[row_half][element] = fmaf(
                    static_cast<float>(dots[row_half][element]), a_scale * w_scale,
                    totals[row_half][element]);
            }
        }
        prefill_cta_barrier<true>();
    }

#pragma unroll
    for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const std::uint32_t row = cta_row_base + row_half * kA8Q4PrefillRowTile +
                                      wave_row_tile * kWmmaTile + axis;
            if (token < tokens && row < rows) {
                output[static_cast<std::size_t>(token) * rows + row] =
                    static_cast<hip_bfloat16>(totals[row_half][element]);
            }
        }
    }
#else
    (void)activation_codes; (void)activation_scales; (void)activation_status;
    (void)weight_codes; (void)weight_scales; (void)output; (void)tokens; (void)rows;
    (void)columns; (void)padded_columns;
#endif
}

struct A8Q4PrefillTile {
    std::uint32_t activation_low[8U * kA8Q4PrefillTokenTile];
    std::uint32_t activation_high[8U * kA8Q4PrefillTokenTile];
    std::uint32_t weights[8U * kA8Q4PrefillRowTile];
    std::uint16_t activation_scales[kA8Q4PrefillTokenTile];
    std::uint16_t weight_scales[kA8Q4PrefillRowTile];
};

struct A8Q4PrefillN128Tile {
    std::uint32_t activation_low[8U * kA8Q4PrefillTokenTile];
    std::uint32_t activation_high[8U * kA8Q4PrefillTokenTile];
    std::uint32_t weights[8U * 2U * kA8Q4PrefillRowTile];
    std::uint16_t activation_scales[kA8Q4PrefillTokenTile];
    std::uint16_t weight_scales[2U * kA8Q4PrefillRowTile];
};

struct A4Q4PrefillPingPongBank {
    std::uint32_t activation_low[8U * kA8Q4PrefillTokenTile];
    std::uint32_t weights[8U * 2U * kA8Q4PrefillRowTile];
    std::uint16_t activation_scales[kA8Q4PrefillTokenTile];
    std::uint16_t weight_scales[2U * kA8Q4PrefillRowTile];
};
struct A4Q4PrefillPingPongTile { A4Q4PrefillPingPongBank bank[2]; };
static_assert(sizeof(A4Q4PrefillPingPongTile) == 13056U);

static_assert(sizeof(A8Q4PrefillN128Tile) == 8576U);

struct A8Q4PrefillM128N128Tile {
    std::uint32_t activation_low[8U * kA8Q4M128PrefillTokenTile];
    std::uint32_t activation_high[8U * kA8Q4M128PrefillTokenTile];
    std::uint32_t weights[8U * 2U * kA8Q4PrefillRowTile];
    std::uint16_t activation_scales[kA8Q4M128PrefillTokenTile];
    std::uint16_t weight_scales[2U * kA8Q4PrefillRowTile];
};

template <bool LdsScopeOnly>
__device__ __forceinline__ void prefill_cta_barrier() {
    if constexpr (LdsScopeOnly) {
        // The qualified production barrier orders and exposes only workgroup LDS.
        // It deliberately avoids the superseded global cache invalidation.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier_signal(-1);
        __builtin_amdgcn_s_barrier_wait(-1);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else {
        __syncthreads();
    }
}

// Production prefill CTA: sixteen waves cooperatively stage one K64 group for a
// 64-token x 64-row output tile. Each staged activation fragment is consumed
// by four waves and each weight fragment by four. Persistent weights are read
// directly from Q4N16K16; only the LDS image is fragment-major.
template <bool LdsScopeOnly>
__device__ __forceinline__ void a8q4g64_linear_prefill_cta_body(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns, A8Q4PrefillTile& tile) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & (kWaveSize - 1U);
    const std::uint32_t wave = thread / kWaveSize;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_token_tile = wave / 4U;
    const std::uint32_t wave_row_tile = wave & 3U;
    const std::uint32_t cta_token_base = blockIdx.y * kA8Q4PrefillTokenTile;
    const std::uint32_t cta_row_base = blockIdx.x * kA8Q4PrefillRowTile;
    const std::uint32_t token_base = cta_token_base + wave_token_tile * kWmmaTile;
    const std::uint32_t row_base = cta_row_base + wave_row_tile * kWmmaTile;

    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const std::uint32_t row = row_base + axis;
            if (token < tokens && row < rows) {
                output[static_cast<std::size_t>(token) * rows + row] =
                    conspicuous_bf16_nan();
            }
        }
        return;
    }

    const std::uint32_t groups = padded_columns / kQ4Group;
    WmmaF32Fragment total{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t activation_row = thread / 8U;
        const std::uint32_t activation_word = thread & 7U;
        const std::uint32_t token = cta_token_base + activation_row;
        const std::size_t activation_element =
            static_cast<std::size_t>(token) * padded_columns + group * kQ4Group +
            activation_word * 8U;
        const std::uint32_t activation_lds = a8q4_lds_word_index(
            activation_row, activation_word, kA8Q4PrefillTokenTile);
        const std::uint32_t packed_k = group * kQ4Group + activation_word * 8U;
        tile.activation_low[activation_lds] = token < tokens
            ? load_q4x8_logical(activation_low_codes, activation_element, columns, packed_k)
            : 0U;
        tile.activation_high[activation_lds] = token < tokens
            ? load_q4x8_logical(activation_high_codes, activation_element, columns, packed_k)
            : 0U;

        const std::uint32_t weight_row = thread / 8U;
        const std::uint32_t weight_word = thread & 7U;
        const std::uint32_t row = cta_row_base + weight_row;
        tile.weights[a8q4_lds_word_index(weight_row, weight_word,
                                         kA8Q4PrefillRowTile)] = row < rows
            ? load_q4_n16k16_x8_logical(
                  weight_codes, row, group, weight_word, groups, columns)
            : 0U;
        if (thread < kA8Q4PrefillTokenTile) {
            const std::uint32_t scale_token = cta_token_base + thread;
            tile.activation_scales[thread] = scale_token < tokens
                ? activation_scales[static_cast<std::size_t>(scale_token) * groups + group]
                : 0U;
        }
        if (thread < kA8Q4PrefillRowTile) {
            const std::uint32_t scale_row = cta_row_base + thread;
            tile.weight_scales[thread] = scale_row < rows
                ? weight_scales[q4_n16k16_scale_index(scale_row, group, groups)]
                : 0U;
        }
        prefill_cta_barrier<LdsScopeOnly>();

        WmmaI32x8 low_dot{};
        WmmaI32x8 high_dot{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            WmmaI32x2 a_low{};
            WmmaI32x2 a_high{};
            WmmaI32x2 b{};
            const std::uint32_t word = half * 4U + lane_group * 2U;
            const std::uint32_t activation_fragment_row =
                wave_token_tile * kWmmaTile + axis;
            const std::uint32_t weight_fragment_row = wave_row_tile * kWmmaTile + axis;
            const std::uint32_t a_index = a8q4_lds_word_index(
                activation_fragment_row, word, kA8Q4PrefillTokenTile);
            const std::uint32_t b_index = a8q4_lds_word_index(
                weight_fragment_row, word, kA8Q4PrefillRowTile);
            a_low[0] = static_cast<int>(tile.activation_low[a_index]);
            a_low[1] = static_cast<int>(tile.activation_low[a8q4_lds_word_index(
                activation_fragment_row, word + 1U, kA8Q4PrefillTokenTile)]);
            a_high[0] = static_cast<int>(tile.activation_high[a_index]);
            a_high[1] = static_cast<int>(tile.activation_high[a8q4_lds_word_index(
                activation_fragment_row, word + 1U, kA8Q4PrefillTokenTile)]);
            b[0] = static_cast<int>(tile.weights[b_index]);
            b[1] = static_cast<int>(tile.weights[a8q4_lds_word_index(
                weight_fragment_row, word + 1U, kA8Q4PrefillRowTile)]);
            low_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a_low, true, b, low_dot, false);
            high_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_high, true, b, high_dot, false);
        }
        const float w_scale = __half2float(__ushort_as_half(
            tile.weight_scales[wave_row_tile * kWmmaTile + axis]));
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t fragment_token =
                wave_token_tile * kWmmaTile + lane_group * 8U + element;
            const float a_scale = __half2float(__ushort_as_half(
                tile.activation_scales[fragment_token]));
            const std::int32_t combined = low_dot[element] + 16 * high_dot[element];
            total[element] = fmaf(static_cast<float>(combined), a_scale * w_scale,
                                  total[element]);
        }
        prefill_cta_barrier<LdsScopeOnly>();
    }

#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const std::uint32_t token = token_base + lane_group * 8U + element;
        const std::uint32_t row = row_base + axis;
        if (token < tokens && row < rows) {
            output[static_cast<std::size_t>(token) * rows + row] =
                static_cast<hip_bfloat16>(total[element]);
        }
    }
#else
    (void)activation_low_codes; (void)activation_high_codes; (void)activation_scales;
    (void)activation_status; (void)weight_codes; (void)weight_scales; (void)output;
    (void)tokens; (void)rows; (void)columns; (void)padded_columns; (void)tile;
#endif
}

__global__ __launch_bounds__(kA8Q4PrefillThreads)
void a8q4g64_linear_prefill_cta_m64n64_regression_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns) {
    __shared__ A8Q4PrefillTile tile;
    a8q4g64_linear_prefill_cta_body<true>(
        activation_low_codes, activation_high_codes, activation_scales,
        activation_status, weight_codes, weight_scales, output, tokens, rows,
        columns, padded_columns, tile);
}

__global__ __launch_bounds__(kA8Q4PrefillThreads)
void a8q4g64_linear_prefill_cta_global_inv_qualification_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns) {
    __shared__ A8Q4PrefillTile tile;
    a8q4g64_linear_prefill_cta_body<false>(
        activation_low_codes, activation_high_codes, activation_scales,
        activation_status, weight_codes, weight_scales, output, tokens, rows,
        columns, padded_columns, tile);
}

__global__ __launch_bounds__(kA8Q4PrefillThreads)
void a8q4g64_linear_prefill_cta_m64n128_regression_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ A8Q4PrefillN128Tile tile;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & (kWaveSize - 1U);
    const std::uint32_t wave = thread / kWaveSize;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_token_tile = wave / 4U;
    const std::uint32_t wave_row_tile = wave & 3U;
    const std::uint32_t cta_token_base = blockIdx.y * kA8Q4PrefillTokenTile;
    const std::uint32_t cta_row_base = blockIdx.x * (2U * kA8Q4PrefillRowTile);
    const std::uint32_t token_base = cta_token_base + wave_token_tile * kWmmaTile;

    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t token = token_base + lane_group * 8U + element;
                const std::uint32_t row = cta_row_base + row_half * kA8Q4PrefillRowTile +
                                          wave_row_tile * kWmmaTile + axis;
                if (token < tokens && row < rows) {
                    output[static_cast<std::size_t>(token) * rows + row] =
                        conspicuous_bf16_nan();
                }
            }
        }
        return;
    }

    const std::uint32_t groups = padded_columns / kQ4Group;
    WmmaF32Fragment totals[2]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t activation_row = thread / 8U;
        const std::uint32_t word = thread & 7U;
        const std::uint32_t token = cta_token_base + activation_row;
        const std::size_t activation_element =
            static_cast<std::size_t>(token) * padded_columns + group * kQ4Group + word * 8U;
        const std::uint32_t activation_lds =
            a8q4_lds_word_index(activation_row, word, kA8Q4PrefillTokenTile);
        const std::uint32_t packed_k = group * kQ4Group + word * 8U;
        tile.activation_low[activation_lds] = token < tokens
            ? load_q4x8_logical(activation_low_codes, activation_element, columns, packed_k)
            : 0U;
        tile.activation_high[activation_lds] = token < tokens
            ? load_q4x8_logical(activation_high_codes, activation_element, columns, packed_k)
            : 0U;

#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
            const std::uint32_t weight_row = thread / 8U + row_half * kA8Q4PrefillRowTile;
            const std::uint32_t row = cta_row_base + weight_row;
            tile.weights[a8q4_lds_word_index(
                weight_row, word, 2U * kA8Q4PrefillRowTile)] = row < rows
                ? load_q4_n16k16_x8_logical(
                      weight_codes, row, group, word, groups, columns)
                : 0U;
        }
        if (thread < kA8Q4PrefillTokenTile) {
            const std::uint32_t scale_token = cta_token_base + thread;
            tile.activation_scales[thread] = scale_token < tokens
                ? activation_scales[static_cast<std::size_t>(scale_token) * groups + group]
                : 0U;
        }
        if (thread < 2U * kA8Q4PrefillRowTile) {
            const std::uint32_t scale_row = cta_row_base + thread;
            tile.weight_scales[thread] = scale_row < rows
                ? weight_scales[q4_n16k16_scale_index(scale_row, group, groups)]
                : 0U;
        }
        prefill_cta_barrier<true>();

        WmmaI32x8 low_dot[2]{};
        WmmaI32x8 high_dot[2]{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            WmmaI32x2 a_low{};
            WmmaI32x2 a_high{};
            WmmaI32x2 b[2]{};
            const std::uint32_t fragment_word = half * 4U + lane_group * 2U;
            const std::uint32_t activation_fragment_row =
                wave_token_tile * kWmmaTile + axis;
            const std::uint32_t a_index = a8q4_lds_word_index(
                activation_fragment_row, fragment_word, kA8Q4PrefillTokenTile);
            a_low[0] = static_cast<int>(tile.activation_low[a_index]);
            a_low[1] = static_cast<int>(tile.activation_low[a8q4_lds_word_index(
                activation_fragment_row, fragment_word + 1U, kA8Q4PrefillTokenTile)]);
            a_high[0] = static_cast<int>(tile.activation_high[a_index]);
            a_high[1] = static_cast<int>(tile.activation_high[a8q4_lds_word_index(
                activation_fragment_row, fragment_word + 1U, kA8Q4PrefillTokenTile)]);
#pragma unroll
            for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
                const std::uint32_t weight_fragment_row =
                    row_half * kA8Q4PrefillRowTile + wave_row_tile * kWmmaTile + axis;
                const std::uint32_t b_index = a8q4_lds_word_index(
                    weight_fragment_row, fragment_word, 2U * kA8Q4PrefillRowTile);
                b[row_half][0] = static_cast<int>(tile.weights[b_index]);
                b[row_half][1] = static_cast<int>(tile.weights[a8q4_lds_word_index(
                    weight_fragment_row, fragment_word + 1U,
                    2U * kA8Q4PrefillRowTile)]);
            }
            low_dot[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a_low, true, b[0], low_dot[0], false);
            high_dot[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_high, true, b[0], high_dot[0], false);
            low_dot[1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a_low, true, b[1], low_dot[1], false);
            high_dot[1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_high, true, b[1], high_dot[1], false);
        }
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
            const std::uint32_t fragment_row = row_half * kA8Q4PrefillRowTile +
                                               wave_row_tile * kWmmaTile + axis;
            const float w_scale = __half2float(__ushort_as_half(
                tile.weight_scales[fragment_row]));
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t fragment_token =
                    wave_token_tile * kWmmaTile + lane_group * 8U + element;
                const float a_scale = __half2float(__ushort_as_half(
                    tile.activation_scales[fragment_token]));
                const std::int32_t combined =
                    low_dot[row_half][element] + 16 * high_dot[row_half][element];
                totals[row_half][element] = fmaf(
                    static_cast<float>(combined), a_scale * w_scale,
                    totals[row_half][element]);
            }
        }
        prefill_cta_barrier<true>();
    }

#pragma unroll
    for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const std::uint32_t row = cta_row_base + row_half * kA8Q4PrefillRowTile +
                                      wave_row_tile * kWmmaTile + axis;
            if (token < tokens && row < rows) {
                output[static_cast<std::size_t>(token) * rows + row] =
                    static_cast<hip_bfloat16>(totals[row_half][element]);
            }
        }
    }
#else
    (void)activation_low_codes; (void)activation_high_codes; (void)activation_scales;
    (void)activation_status; (void)weight_codes; (void)weight_scales; (void)output;
    (void)tokens; (void)rows; (void)columns; (void)padded_columns;
#endif
}

// Selected software pipeline for exact qualified M64xN128 tuples. Immutable
// global pointers remain scalar bases while all byte cursors are host-proven
// zero-extendable U32 offsets.
[[maybe_unused]] __device__ __forceinline__ std::uint32_t scalar_base_load_u32(
    const std::uint8_t* base, std::uint32_t offset) {
    return *reinterpret_cast<const std::uint32_t*>(base + offset);
}

[[maybe_unused]] __device__ __forceinline__ std::uint64_t scalar_base_load_u64(
    const std::uint8_t* base, std::uint32_t offset) {
    return *reinterpret_cast<const std::uint64_t*>(base + offset);
}

[[maybe_unused]] __device__ __forceinline__ std::uint16_t scalar_base_load_u16(
    const std::uint16_t* base, std::uint32_t offset) {
    return *reinterpret_cast<const std::uint16_t*>(
        reinterpret_cast<const std::uint8_t*>(base) + offset);
}


// Production A8Q4G64 prefill: an M128xN128 CTA of eight waves, each owning 2x4
// WMMA fragments (32 tokens x 64 rows). Token tiles vary fastest in the grid so
// consecutive CTAs share one weight tile in L2. The signed high-nibble chain
// starts from 0x04B40000; after <<4 and the unsigned low-nibble chain the I32
// word is exactly 0x4B400000 + dot (|dot| <= 64*127*8 < 2^22), whose FP32 value
// minus 12582912 recovers dot without a conversion instruction. The FP32 scale
// product and FMA order equal the per-group incumbent formula.
struct A8Q4PrefillM128Stage {
    std::uint64_t activation_low[4U * kA8Q4M128PrefillTokenTile];
    std::uint64_t activation_high[4U * kA8Q4M128PrefillTokenTile];
    std::uint64_t weights[4U * kA8Q4M128PrefillTokenTile];
    float activation_scales[kA8Q4M128PrefillTokenTile];
    float weight_scales[kA8Q4M128PrefillTokenTile];
};
struct A8Q4PrefillM128PingPong { A8Q4PrefillM128Stage stage[2]; };
static_assert(sizeof(A8Q4PrefillM128PingPong) == 26624U);

// [pair][128 rows] u64 image; odd pairs (the upper half-wave) move by 16 rows so
// the two half-waves read disjoint bank halves.
[[maybe_unused]] __device__ __forceinline__ std::uint32_t a8q4_m128_lds_index(
    std::uint32_t pair, std::uint32_t row) {
    return pair * 128U + (row ^ ((pair & 1U) << 4U));
}

struct A8Q4PrefillM128Load {
    std::uint16_t scale;
    uint4 activation_low, activation_high, weights;
};

// Per-thread byte offsets from the uniform plane bases; every qualified plane is below 4 GiB,
// so loads use scalar-base plus 32-bit-offset addressing.
struct A8Q4PrefillM128Offsets {
    std::uint32_t activation, weight, scale;
    bool activation_scale;
};

template <class T>
__device__ __forceinline__ const T& a8q4_m128_at(const void* base, std::uint32_t offset) {
    return *reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(base) + offset);
}

[[maybe_unused]] __device__ __forceinline__ A8Q4PrefillM128Load a8q4_m128_load(
    const std::uint8_t* activation_low_codes, const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales, const std::uint8_t* weight_codes,
    const std::uint16_t* weight_scales, const A8Q4PrefillM128Offsets& offsets,
    std::uint32_t group) {
    A8Q4PrefillM128Load loaded;
    // Issue the scale first: loads complete in order, so its conversion never waits for the
    // prefetched code planes, which are consumed only after this group's WMMA work.
    loaded.scale = offsets.activation_scale
        ? a8q4_m128_at<std::uint16_t>(activation_scales, offsets.scale + group * 2U)
        : a8q4_m128_at<std::uint16_t>(weight_scales, offsets.scale + group * 32U);
    loaded.activation_low =
        a8q4_m128_at<uint4>(activation_low_codes, offsets.activation + group * 32U);
    loaded.activation_high =
        a8q4_m128_at<uint4>(activation_high_codes, offsets.activation + group * 32U);
    // One 16-row weight tile-group is 512 contiguous bytes: [pair][16 rows][u64].
    loaded.weights = a8q4_m128_at<uint4>(weight_codes, offsets.weight + group * 512U);
    return loaded;
}

[[maybe_unused]] __device__ __forceinline__ std::uint64_t a8q4_m128_u64(std::uint32_t low,
                                                                      std::uint32_t high) {
    return (static_cast<std::uint64_t>(high) << 32U) | low;
}

[[maybe_unused]] __device__ __forceinline__ void a8q4_m128_store(
    A8Q4PrefillM128Stage& stage, const A8Q4PrefillM128Load& loaded, std::uint32_t thread) {
    const std::uint32_t token = thread >> 1U;
    const std::uint32_t activation_pair = (thread & 1U) * 2U;
    stage.activation_low[a8q4_m128_lds_index(activation_pair, token)] =
        a8q4_m128_u64(loaded.activation_low.x, loaded.activation_low.y);
    stage.activation_low[a8q4_m128_lds_index(activation_pair + 1U, token)] =
        a8q4_m128_u64(loaded.activation_low.z, loaded.activation_low.w);
    stage.activation_high[a8q4_m128_lds_index(activation_pair, token)] =
        a8q4_m128_u64(loaded.activation_high.x, loaded.activation_high.y);
    stage.activation_high[a8q4_m128_lds_index(activation_pair + 1U, token)] =
        a8q4_m128_u64(loaded.activation_high.z, loaded.activation_high.w);
    const std::uint32_t within = thread & 31U;
    const std::uint32_t weight_pair = within >> 3U;
    const std::uint32_t row = (thread >> 5U) * 16U + (within & 7U) * 2U;
    stage.weights[a8q4_m128_lds_index(weight_pair, row)] =
        a8q4_m128_u64(loaded.weights.x, loaded.weights.y);
    stage.weights[a8q4_m128_lds_index(weight_pair, row + 1U)] =
        a8q4_m128_u64(loaded.weights.z, loaded.weights.w);
    const float scale = __half2float(__ushort_as_half(loaded.scale));
    if (thread < kA8Q4M128PrefillTokenTile) {
        stage.activation_scales[thread] = scale;
    } else {
        stage.weight_scales[thread - kA8Q4M128PrefillTokenTile] = scale;
    }
}

[[maybe_unused]] __device__ __forceinline__ WmmaI32x2 a8q4_m128_fragment(std::uint64_t value) {
    return WmmaI32x2{static_cast<int>(static_cast<std::uint32_t>(value)),
                     static_cast<int>(static_cast<std::uint32_t>(value >> 32U))};
}

// One K64 group of the wave's 2x4 fragments.
[[maybe_unused]] __device__ __forceinline__ void a8q4_m128_group(
    const A8Q4PrefillM128Stage& stage, std::uint32_t wave_token, std::uint32_t wave_row,
    std::uint32_t axis, std::uint32_t lane_group, const WmmaI32x8& high_origin,
    WmmaF32Fragment (&totals)[2][4]) {
    WmmaI32x2 low[2][2], high[2][2], weights[2][4];
#pragma unroll
    for (std::uint32_t half = 0; half < 2U; ++half) {
        const std::uint32_t pair = half * 2U + lane_group;
#pragma unroll
        for (std::uint32_t fragment = 0; fragment < 2U; ++fragment) {
            const std::uint32_t index =
                a8q4_m128_lds_index(pair, wave_token + fragment * 16U + axis);
            low[half][fragment] = a8q4_m128_fragment(stage.activation_low[index]);
            high[half][fragment] = a8q4_m128_fragment(stage.activation_high[index]);
        }
#pragma unroll
        for (std::uint32_t row_fragment = 0; row_fragment < 4U; ++row_fragment) {
            weights[half][row_fragment] = a8q4_m128_fragment(stage.weights[
                a8q4_m128_lds_index(pair, wave_row + row_fragment * 16U + axis)]);
        }
    }
    float activation_scale[2][8];
#pragma unroll
    for (std::uint32_t fragment = 0; fragment < 2U; ++fragment) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            activation_scale[fragment][element] = stage.activation_scales[
                wave_token + fragment * 16U + lane_group * 8U + element];
        }
    }
    float weight_scale[4];
#pragma unroll
    for (std::uint32_t row_fragment = 0; row_fragment < 4U; ++row_fragment) {
        weight_scale[row_fragment] = stage.weight_scales[wave_row + row_fragment * 16U + axis];
    }
#pragma unroll
    for (std::uint32_t fragment = 0; fragment < 2U; ++fragment) {
#pragma unroll
        for (std::uint32_t row_fragment = 0; row_fragment < 4U; ++row_fragment) {
            WmmaI32x8 dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, high[0][fragment], true, weights[0][row_fragment], high_origin, false);
            dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, high[1][fragment], true, weights[1][row_fragment], dot, false);
            dot = dot << 4;
            dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, low[0][fragment], true, weights[0][row_fragment], dot, false);
            dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, low[1][fragment], true, weights[1][row_fragment], dot, false);
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                totals[fragment][row_fragment][element] = fmaf(
                    __int_as_float(dot[element]) - 12582912.0F,
                    activation_scale[fragment][element] * weight_scale[row_fragment],
                    totals[fragment][row_fragment][element]);
            }
        }
    }
    // Interleave each WMMA with four independent VALU instructions of the scale epilogue.
#pragma unroll
    for (std::uint32_t instruction = 0; instruction < 32U; ++instruction) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x002, 4, 0);
    }
}

template <bool kAccumulate>
__global__ __launch_bounds__(kA8Q4M128ProductionThreads)
void a8q4g64_linear_prefill_cta_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t rows, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ A8Q4PrefillM128PingPong tiles;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & (kWaveSize - 1U);
    const std::uint32_t wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t token_base = blockIdx.x * kA8Q4M128PrefillTokenTile;
    const std::uint32_t row_base = blockIdx.y * kA8Q4M128PrefillTokenTile;
    const std::uint32_t wave_token = (wave & 3U) * 32U;
    const std::uint32_t wave_row = (wave >> 2U) * 64U;

    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t fragment = 0; fragment < 2U; ++fragment) {
#pragma unroll
            for (std::uint32_t row_fragment = 0; row_fragment < 4U; ++row_fragment) {
#pragma unroll
                for (std::uint32_t element = 0; element < 8U; ++element) {
                    const std::uint32_t token = token_base + wave_token + fragment * 16U +
                                                lane_group * 8U + element;
                    const std::uint32_t row =
                        row_base + wave_row + row_fragment * 16U + axis;
                    output[static_cast<std::size_t>(token) * rows + row] =
                        conspicuous_bf16_nan();
                }
            }
        }
        return;
    }

    const std::uint32_t groups = padded_columns / kQ4Group;
    A8Q4PrefillM128Offsets offsets;
    offsets.activation =
        (token_base + (thread >> 1U)) * (padded_columns / 2U) + (thread & 1U) * 16U;
    offsets.weight = (row_base / 16U + (thread >> 5U)) * groups * 512U + (thread & 31U) * 16U;
    offsets.activation_scale = thread < kA8Q4M128PrefillTokenTile;
    offsets.scale = offsets.activation_scale
        ? (token_base + thread) * groups * 2U
        : static_cast<std::uint32_t>(
              q4_n16k16_scale_index(row_base + thread - kA8Q4M128PrefillTokenTile, 0U, groups) *
              sizeof(std::uint16_t));

    WmmaF32Fragment totals[2][4]{};
    WmmaI32x8 high_origin;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) high_origin[element] = 0x04B40000;

    a8q4_m128_store(tiles.stage[0],
                    a8q4_m128_load(activation_low_codes, activation_high_codes,
                                   activation_scales, weight_codes, weight_scales, offsets, 0U),
                    thread);
    prefill_cta_barrier<true>();
    for (std::uint32_t group = 0; group + 1U < groups; ++group) {
        const A8Q4PrefillM128Load next =
            a8q4_m128_load(activation_low_codes, activation_high_codes, activation_scales,
                           weight_codes, weight_scales, offsets, group + 1U);
        a8q4_m128_group(tiles.stage[group & 1U], wave_token, wave_row, axis, lane_group,
                        high_origin, totals);
        a8q4_m128_store(tiles.stage[(group + 1U) & 1U], next, thread);
        prefill_cta_barrier<true>();
    }
    a8q4_m128_group(tiles.stage[(groups - 1U) & 1U], wave_token, wave_row, axis, lane_group,
                    high_origin, totals);

#pragma unroll
    for (std::uint32_t fragment = 0; fragment < 2U; ++fragment) {
#pragma unroll
        for (std::uint32_t row_fragment = 0; row_fragment < 4U; ++row_fragment) {
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t token = token_base + wave_token + fragment * 16U +
                                            lane_group * 8U + element;
                const std::uint32_t row = row_base + wave_row + row_fragment * 16U + axis;
                hip_bfloat16& target = output[static_cast<std::size_t>(token) * rows + row];
                const hip_bfloat16 projection =
                    static_cast<hip_bfloat16>(totals[fragment][row_fragment][element]);
                if constexpr (kAccumulate) {
                    target = static_cast<hip_bfloat16>(static_cast<float>(target) +
                                                       static_cast<float>(projection));
                } else {
                    target = projection;
                }
            }
        }
    }
#else
    (void)activation_low_codes; (void)activation_high_codes;
    (void)activation_scales; (void)activation_status; (void)weight_codes;
    (void)weight_scales; (void)output; (void)rows; (void)padded_columns;
#endif
}

__global__ __launch_bounds__(kA8Q4PrefillThreads)
void a4q4g64_linear_prefill_pingpong_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    (void)tokens;
    (void)columns;
    __shared__ A4Q4PrefillPingPongTile tiles;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & (kWaveSize - 1U);
    const std::uint32_t wave = thread / kWaveSize;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_token_tile = wave / 4U;
    const std::uint32_t wave_row_tile = wave & 3U;
    const std::uint32_t cta_token_base = blockIdx.y * kA8Q4PrefillTokenTile;
    const std::uint32_t cta_row_base = blockIdx.x * (2U * kA8Q4PrefillRowTile);
    const std::uint32_t token_base =
        cta_token_base + wave_token_tile * kWmmaTile;

    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t token = token_base + lane_group * 8U + element;
                const std::uint32_t row = cta_row_base +
                    row_half * kA8Q4PrefillRowTile + wave_row_tile * kWmmaTile + axis;
                output[static_cast<std::size_t>(token) * rows + row] =
                    conspicuous_bf16_nan();
            }
        }
        return;
    }

    const std::uint32_t groups = padded_columns / kQ4Group;
    const std::uint32_t activation_row = thread / 8U;
    const std::uint32_t word = thread & 7U;
    const std::uint32_t token = cta_token_base + activation_row;
    const std::uint32_t activation_lds =
        a8q4_lds_word_index(activation_row, word, kA8Q4PrefillTokenTile);
    const std::uint32_t load_tile = thread >> 6U;
    const std::uint32_t load_pair = (thread >> 4U) & 3U;
    const std::uint32_t load_lane = thread & 15U;
    const std::uint32_t load_row_local = load_tile * 16U + load_lane;
    const std::uint32_t load_row = cta_row_base + load_row_local;
    const std::uint32_t load_word = load_pair * 2U;

    std::uint32_t activation_offset =
        token * (padded_columns / 2U) + word * sizeof(std::uint32_t);
    std::uint32_t weight_offset = static_cast<std::uint32_t>(
        q4_n16k16_pair_index(load_row, 0U, load_pair, groups) *
        sizeof(std::uint64_t));
    std::uint32_t activation_scale_offset =
        (cta_token_base + thread) * groups * sizeof(std::uint16_t);
    std::uint32_t weight_scale_offset = static_cast<std::uint32_t>(
        q4_n16k16_scale_index(cta_row_base + thread, 0U, groups) *
        sizeof(std::uint16_t));

    tiles.bank[0].activation_low[activation_lds] =
        scalar_base_load_u32(activation_low_codes, activation_offset);
    const std::uint64_t first_weight =
        scalar_base_load_u64(weight_codes, weight_offset);
    tiles.bank[0].weights[a8q4_lds_word_index(
        load_row_local, load_word, 2U * kA8Q4PrefillRowTile)] =
        static_cast<std::uint32_t>(first_weight);
    tiles.bank[0].weights[a8q4_lds_word_index(
        load_row_local, load_word + 1U, 2U * kA8Q4PrefillRowTile)] =
        static_cast<std::uint32_t>(first_weight >> 32U);
    if (thread < kA8Q4PrefillTokenTile) {
        tiles.bank[0].activation_scales[thread] =
            scalar_base_load_u16(activation_scales, activation_scale_offset);
    }
    if (thread < 2U * kA8Q4PrefillRowTile) {
        tiles.bank[0].weight_scales[thread] =
            scalar_base_load_u16(weight_scales, weight_scale_offset);
    }
    prefill_cta_barrier<true>();

    WmmaF32Fragment totals[2]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        const bool has_next = group + 1U < groups;
        std::uint32_t next_activation_low = 0U;
        std::uint64_t next_weight = 0U;
        if (has_next) {
            activation_offset += kQ4Group / 2U;
            weight_offset += 4U * 16U * sizeof(std::uint64_t);
            next_activation_low =
                scalar_base_load_u32(activation_low_codes, activation_offset);
            next_weight = scalar_base_load_u64(weight_codes, weight_offset);
        }

        auto& current = tiles.bank[group & 1U];
        WmmaI32x8 low_dot[2]{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            WmmaI32x2 a_low{};
            WmmaI32x2 b[2]{};
            const std::uint32_t fragment_word = half * 4U + lane_group * 2U;
            const std::uint32_t activation_fragment_row =
                wave_token_tile * kWmmaTile + axis;
            const std::uint32_t a_index = a8q4_lds_word_index(
                activation_fragment_row, fragment_word, kA8Q4PrefillTokenTile);
            a_low[0] = static_cast<int>(current.activation_low[a_index]);
            a_low[1] = static_cast<int>(current.activation_low[a8q4_lds_word_index(
                activation_fragment_row, fragment_word + 1U,
                kA8Q4PrefillTokenTile)]);
#pragma unroll
            for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
                const std::uint32_t weight_fragment_row =
                    row_half * kA8Q4PrefillRowTile + wave_row_tile * kWmmaTile + axis;
                const std::uint32_t b_index = a8q4_lds_word_index(
                    weight_fragment_row, fragment_word,
                    2U * kA8Q4PrefillRowTile);
                b[row_half][0] = static_cast<int>(current.weights[b_index]);
                b[row_half][1] = static_cast<int>(current.weights[
                    a8q4_lds_word_index(weight_fragment_row, fragment_word + 1U,
                                           2U * kA8Q4PrefillRowTile)]);
            }
            low_dot[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_low, true, b[0], low_dot[0], false);
            low_dot[1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_low, true, b[1], low_dot[1], false);
        }
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
            const std::uint32_t fragment_row = row_half * kA8Q4PrefillRowTile +
                wave_row_tile * kWmmaTile + axis;
            const float weight_scale = __half2float(__ushort_as_half(
                current.weight_scales[fragment_row]));
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t fragment_token =
                    wave_token_tile * kWmmaTile + lane_group * 8U + element;
                const float activation_scale = __half2float(__ushort_as_half(
                    current.activation_scales[fragment_token]));
                const std::int32_t combined =
                    low_dot[row_half][element];
                totals[row_half][element] = fmaf(
                    static_cast<float>(combined), activation_scale * weight_scale,
                    totals[row_half][element]);
            }
        }

        if (has_next) {
            auto& next = tiles.bank[(group + 1U) & 1U];
            next.activation_low[activation_lds] = next_activation_low;
            next.weights[a8q4_lds_word_index(
                load_row_local, load_word, 2U * kA8Q4PrefillRowTile)] =
                static_cast<std::uint32_t>(next_weight);
            next.weights[a8q4_lds_word_index(
                load_row_local, load_word + 1U, 2U * kA8Q4PrefillRowTile)] =
                static_cast<std::uint32_t>(next_weight >> 32U);
            if (thread < kA8Q4PrefillTokenTile) {
                activation_scale_offset += sizeof(std::uint16_t);
                next.activation_scales[thread] = scalar_base_load_u16(
                    activation_scales, activation_scale_offset);
            }
            if (thread < 2U * kA8Q4PrefillRowTile) {
                weight_scale_offset += 16U * sizeof(std::uint16_t);
                next.weight_scales[thread] =
                    scalar_base_load_u16(weight_scales, weight_scale_offset);
            }
            prefill_cta_barrier<true>();
        }
    }

#pragma unroll
    for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const std::uint32_t row = cta_row_base +
                row_half * kA8Q4PrefillRowTile + wave_row_tile * kWmmaTile + axis;
            output[static_cast<std::size_t>(token) * rows + row] =
                static_cast<hip_bfloat16>(totals[row_half][element]);
        }
    }
#else
    (void)activation_low_codes;
    (void)activation_scales; (void)activation_status; (void)weight_codes;
    (void)weight_scales; (void)output; (void)tokens; (void)rows;
    (void)columns; (void)padded_columns;
#endif
}

__global__ __launch_bounds__(kA8Q4M128PrefillThreads)
void a8q4g64_linear_prefill_cta_m128n128_qualification_kernel(
    const std::uint8_t* activation_low_codes,
    const std::uint8_t* activation_high_codes,
    const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ A8Q4PrefillM128N128Tile tile;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & (kWaveSize - 1U);
    const std::uint32_t wave = thread / kWaveSize;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_token_tile = wave / 4U;
    const std::uint32_t wave_row_tile = wave & 3U;
    const std::uint32_t cta_token_base = blockIdx.y * kA8Q4M128PrefillTokenTile;
    const std::uint32_t cta_row_base = blockIdx.x * (2U * kA8Q4PrefillRowTile);
    const std::uint32_t token_base = cta_token_base + wave_token_tile * kWmmaTile;

    if (*activation_status != Q4G64ActivationOk) {
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t token = token_base + lane_group * 8U + element;
                const std::uint32_t row = cta_row_base + row_half * kA8Q4PrefillRowTile +
                                          wave_row_tile * kWmmaTile + axis;
                if (token < tokens && row < rows) {
                    output[static_cast<std::size_t>(token) * rows + row] =
                        conspicuous_bf16_nan();
                }
            }
        }
        return;
    }

    const std::uint32_t groups = padded_columns / kQ4Group;
    WmmaF32Fragment totals[2]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t activation_row = thread / 8U;
        const std::uint32_t word = thread & 7U;
        const std::uint32_t token = cta_token_base + activation_row;
        const std::uint32_t row = cta_row_base + activation_row;
        const std::uint32_t packed_k = group * kQ4Group + word * 8U;
        const std::size_t activation_element =
            static_cast<std::size_t>(token) * padded_columns + packed_k;
        const std::uint32_t activation_lds = a8q4_lds_word_index(
            activation_row, word, kA8Q4M128PrefillTokenTile);
        const std::uint32_t weight_lds = a8q4_lds_word_index(
            activation_row, word, 2U * kA8Q4PrefillRowTile);
        tile.activation_low[activation_lds] = token < tokens
            ? load_q4x8_logical(activation_low_codes, activation_element, columns, packed_k)
            : 0U;
        tile.activation_high[activation_lds] = token < tokens
            ? load_q4x8_logical(activation_high_codes, activation_element, columns, packed_k)
            : 0U;
        tile.weights[weight_lds] = row < rows
            ? load_q4_n16k16_x8_logical(
                  weight_codes, row, group, word, groups, columns)
            : 0U;
        if (thread < kA8Q4M128PrefillTokenTile) {
            const std::uint32_t scale_token = cta_token_base + thread;
            tile.activation_scales[thread] = scale_token < tokens
                ? activation_scales[static_cast<std::size_t>(scale_token) * groups + group]
                : 0U;
        }
        if (thread < 2U * kA8Q4PrefillRowTile) {
            const std::uint32_t scale_row = cta_row_base + thread;
            tile.weight_scales[thread] = scale_row < rows
                ? weight_scales[q4_n16k16_scale_index(scale_row, group, groups)]
                : 0U;
        }
        prefill_cta_barrier<true>();

        WmmaI32x8 low_dot[2]{};
        WmmaI32x8 high_dot[2]{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            WmmaI32x2 a_low{};
            WmmaI32x2 a_high{};
            WmmaI32x2 b[2]{};
            const std::uint32_t fragment_word = half * 4U + lane_group * 2U;
            const std::uint32_t activation_fragment_row =
                wave_token_tile * kWmmaTile + axis;
            const std::uint32_t a_index = a8q4_lds_word_index(
                activation_fragment_row, fragment_word, kA8Q4M128PrefillTokenTile);
            a_low[0] = static_cast<int>(tile.activation_low[a_index]);
            a_low[1] = static_cast<int>(tile.activation_low[a8q4_lds_word_index(
                activation_fragment_row, fragment_word + 1U,
                kA8Q4M128PrefillTokenTile)]);
            a_high[0] = static_cast<int>(tile.activation_high[a_index]);
            a_high[1] = static_cast<int>(tile.activation_high[a8q4_lds_word_index(
                activation_fragment_row, fragment_word + 1U,
                kA8Q4M128PrefillTokenTile)]);
#pragma unroll
            for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
                const std::uint32_t weight_fragment_row =
                    row_half * kA8Q4PrefillRowTile + wave_row_tile * kWmmaTile + axis;
                const std::uint32_t b_index = a8q4_lds_word_index(
                    weight_fragment_row, fragment_word, 2U * kA8Q4PrefillRowTile);
                b[row_half][0] = static_cast<int>(tile.weights[b_index]);
                b[row_half][1] = static_cast<int>(tile.weights[a8q4_lds_word_index(
                    weight_fragment_row, fragment_word + 1U,
                    2U * kA8Q4PrefillRowTile)]);
            }
            low_dot[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a_low, true, b[0], low_dot[0], false);
            high_dot[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_high, true, b[0], high_dot[0], false);
            low_dot[1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a_low, true, b[1], low_dot[1], false);
            high_dot[1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_high, true, b[1], high_dot[1], false);
        }
#pragma unroll
        for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
            const std::uint32_t fragment_row = row_half * kA8Q4PrefillRowTile +
                                               wave_row_tile * kWmmaTile + axis;
            const float w_scale = __half2float(__ushort_as_half(
                tile.weight_scales[fragment_row]));
#pragma unroll
            for (std::uint32_t element = 0; element < 8U; ++element) {
                const std::uint32_t fragment_token =
                    wave_token_tile * kWmmaTile + lane_group * 8U + element;
                const float a_scale = __half2float(__ushort_as_half(
                    tile.activation_scales[fragment_token]));
                const std::int32_t combined =
                    low_dot[row_half][element] + 16 * high_dot[row_half][element];
                totals[row_half][element] = fmaf(
                    static_cast<float>(combined), a_scale * w_scale,
                    totals[row_half][element]);
            }
        }
        prefill_cta_barrier<true>();
    }

#pragma unroll
    for (std::uint32_t row_half = 0; row_half < 2U; ++row_half) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = token_base + lane_group * 8U + element;
            const std::uint32_t row = cta_row_base + row_half * kA8Q4PrefillRowTile +
                                      wave_row_tile * kWmmaTile + axis;
            if (token < tokens && row < rows) {
                output[static_cast<std::size_t>(token) * rows + row] =
                    static_cast<hip_bfloat16>(totals[row_half][element]);
            }
        }
    }
#else
    (void)activation_low_codes; (void)activation_high_codes; (void)activation_scales;
    (void)activation_status; (void)weight_codes; (void)weight_scales; (void)output;
    (void)tokens; (void)rows; (void)columns; (void)padded_columns;
#endif
}

__device__ __attribute__((unused)) WmmaI32x2 load_i8x8(
    const std::int8_t* values, std::size_t begin, bool valid) {
    WmmaI32x2 packed{};
    if (valid) {
        // The public launcher requires dword-aligned code planes; every G32/K16
        // fragment offset is a multiple of eight. Do not require 8-byte base
        // alignment or read beyond this exact eight-byte fragment.
        const auto* words = reinterpret_cast<const std::int32_t*>(values + begin);
        packed[0] = words[0];
        packed[1] = words[1];
    }
    return packed;
}

__global__ __launch_bounds__(kWaveSize) void a8w8g32_linear_wmma32_kernel(
    const std::int8_t* activation_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status, const std::int8_t* weight_codes,
    const std::uint16_t* weight_scales, hip_bfloat16* output,
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t padded_columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t token_base = blockIdx.y * kWmmaTile;
    const std::uint32_t row_base = blockIdx.x * kWmmaTile;
    const std::uint32_t fragment_token_base = token_base + lane_group * 8U;
    const std::uint32_t row = row_base + axis;
    if (*activation_status != A8G32ActivationOk) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = fragment_token_base + element;
            if (token < tokens && row < rows) {
                output[static_cast<std::size_t>(token) * rows + row] = conspicuous_bf16_nan();
            }
        }
        return;
    }
    WmmaF32Fragment total{};
    const std::uint32_t groups = padded_columns / kW8Group;
    for (std::uint32_t group = 0; group < groups; ++group) {
        WmmaI32x8 integer_dot{};
#pragma unroll
        for (std::uint32_t tile = 0; tile < 2U; ++tile) {
            const std::uint32_t k_base =
                group * kW8Group + tile * kWmmaTile + lane_group * 8U;
            const std::uint32_t token = token_base + axis;
            const WmmaI32x2 a = load_i8x8(
                activation_codes,
                static_cast<std::size_t>(token) * padded_columns + k_base,
                token < tokens);
            const WmmaI32x2 b = load_i8x8(
                weight_codes, static_cast<std::size_t>(row) * padded_columns + k_base,
                row < rows);
            integer_dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                true, a, true, b, integer_dot, false);
        }
        const float weight_scale =
            row < rows
                ? __half2float(__ushort_as_half(
                      weight_scales[static_cast<std::size_t>(row) * groups + group]))
                : 0.0F;
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = fragment_token_base + element;
            const float activation_scale =
                token < tokens
                    ? __half2float(__ushort_as_half(
                          activation_scales[static_cast<std::size_t>(token) * groups + group]))
                    : 0.0F;
            total[element] = fmaf(static_cast<float>(integer_dot[element]),
                                  activation_scale * weight_scale, total[element]);
        }
    }
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const std::uint32_t token = fragment_token_base + element;
        if (token < tokens && row < rows) {
            output[static_cast<std::size_t>(token) * rows + row] =
                static_cast<hip_bfloat16>(total[element]);
        }
    }
#else
    (void)activation_codes;
    (void)activation_scales;
    (void)activation_status;
    (void)weight_codes;
    (void)weight_scales;
    (void)output;
    (void)tokens;
    (void)rows;
    (void)padded_columns;
#endif
}

__device__ __forceinline__ __attribute__((unused)) std::uint32_t a8w8_lds_word_index(
    std::uint32_t row, std::uint32_t word, std::uint32_t rows) {
    return word * rows + (row ^ (((word >> 1U) & 1U) << 4U));
}

__device__ __forceinline__ __attribute__((unused)) std::uint32_t load_i8x4_logical(
    const std::int8_t* codes, std::size_t element, std::uint32_t logical_k,
    std::uint32_t packed_k) {
    if (packed_k >= logical_k) return 0U;
    const std::uint32_t packed =
        *reinterpret_cast<const std::uint32_t*>(codes + element);
    const std::uint32_t remaining = logical_k - packed_k;
    if (remaining >= 4U) return packed;
    return packed & ((1U << (remaining * 8U)) - 1U);
}

struct A8W8PrefillTile {
    std::uint32_t activations[8U * kA8W8PrefillTokenTile];
    std::uint32_t weights[8U * kA8W8PrefillRowTile];
    std::uint16_t activation_scales[kA8W8PrefillTokenTile];
    std::uint16_t weight_scales[kA8W8PrefillRowTile];
};

// This is intentionally a separate W8 implementation rather than a Q4/W8
// template. Its G32 signed-I8 payload and two K16 IU8 operations are distinct
// from A8Q4's two-plane G64 decomposition even though both use a 16-wave tile.
template <bool LdsScopeOnly>
__device__ __forceinline__ void a8w8g32_linear_prefill_cta_body(
    const std::int8_t* activation_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status, const std::int8_t* weight_codes,
    const std::uint16_t* weight_scales, hip_bfloat16* output,
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns, A8W8PrefillTile& tile) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & (kWaveSize - 1U);
    const std::uint32_t wave = thread / kWaveSize;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_token_tile = wave / 4U;
    const std::uint32_t wave_row_tile = wave & 3U;
    const std::uint32_t cta_token_base = blockIdx.y * kA8W8PrefillTokenTile;
    const std::uint32_t cta_row_base = blockIdx.x * kA8W8PrefillRowTile;
    const std::uint32_t token_base = cta_token_base + wave_token_tile * kWmmaTile;
    const std::uint32_t row_base = cta_row_base + wave_row_tile * kWmmaTile;
    const std::uint32_t fragment_token_base = token_base + lane_group * 8U;
    const std::uint32_t output_row = row_base + axis;
    if (*activation_status != A8G32ActivationOk) {
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = fragment_token_base + element;
            if (token < tokens && output_row < rows) {
                output[static_cast<std::size_t>(token) * rows + output_row] =
                    conspicuous_bf16_nan();
            }
        }
        return;
    }

    const std::uint32_t groups = padded_columns / kW8Group;
    WmmaF32Fragment total{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t activation_row = thread / 8U;
        const std::uint32_t activation_word = thread & 7U;
        const std::uint32_t token = cta_token_base + activation_row;
        const std::size_t activation_element =
            static_cast<std::size_t>(token) * padded_columns + group * kW8Group +
            activation_word * 4U;
        const std::uint32_t packed_k = group * kW8Group + activation_word * 4U;
        tile.activations[a8w8_lds_word_index(
            activation_row, activation_word, kA8W8PrefillTokenTile)] = token < tokens
            ? load_i8x4_logical(activation_codes, activation_element, columns, packed_k)
            : 0U;
        const std::uint32_t weight_row = thread / 8U;
        const std::uint32_t weight_word = thread & 7U;
        const std::uint32_t row = cta_row_base + weight_row;
        const std::size_t weight_element =
            static_cast<std::size_t>(row) * padded_columns + group * kW8Group +
            weight_word * 4U;
        const std::uint32_t weight_packed_k = group * kW8Group + weight_word * 4U;
        tile.weights[a8w8_lds_word_index(weight_row, weight_word,
                                         kA8W8PrefillRowTile)] = row < rows
            ? load_i8x4_logical(weight_codes, weight_element, columns, weight_packed_k)
            : 0U;
        if (thread < kA8W8PrefillTokenTile) {
            const std::uint32_t scale_token = cta_token_base + thread;
            tile.activation_scales[thread] = scale_token < tokens
                ? activation_scales[static_cast<std::size_t>(scale_token) * groups + group]
                : 0U;
        }
        if (thread < kA8W8PrefillRowTile) {
            const std::uint32_t scale_row = cta_row_base + thread;
            tile.weight_scales[thread] = scale_row < rows
                ? weight_scales[static_cast<std::size_t>(scale_row) * groups + group]
                : 0U;
        }
        prefill_cta_barrier<LdsScopeOnly>();

        WmmaI32x8 integer_dot{};
#pragma unroll
        for (std::uint32_t k_tile = 0; k_tile < 2U; ++k_tile) {
            const std::uint32_t word = k_tile * 4U + lane_group * 2U;
            const std::uint32_t activation_fragment_row =
                wave_token_tile * kWmmaTile + axis;
            const std::uint32_t weight_fragment_row = wave_row_tile * kWmmaTile + axis;
            WmmaI32x2 a{};
            WmmaI32x2 b{};
            a[0] = static_cast<int>(tile.activations[a8w8_lds_word_index(
                activation_fragment_row, word, kA8W8PrefillTokenTile)]);
            a[1] = static_cast<int>(tile.activations[a8w8_lds_word_index(
                activation_fragment_row, word + 1U, kA8W8PrefillTokenTile)]);
            b[0] = static_cast<int>(tile.weights[a8w8_lds_word_index(
                weight_fragment_row, word, kA8W8PrefillRowTile)]);
            b[1] = static_cast<int>(tile.weights[a8w8_lds_word_index(
                weight_fragment_row, word + 1U, kA8W8PrefillRowTile)]);
            integer_dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                true, a, true, b, integer_dot, false);
        }
        const float weight_scale = __half2float(__ushort_as_half(
            tile.weight_scales[wave_row_tile * kWmmaTile + axis]));
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t fragment_token =
                wave_token_tile * kWmmaTile + lane_group * 8U + element;
            const float activation_scale = __half2float(__ushort_as_half(
                tile.activation_scales[fragment_token]));
            total[element] = fmaf(static_cast<float>(integer_dot[element]),
                                  activation_scale * weight_scale, total[element]);
        }
        prefill_cta_barrier<LdsScopeOnly>();
    }
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const std::uint32_t token = fragment_token_base + element;
        if (token < tokens && output_row < rows) {
            output[static_cast<std::size_t>(token) * rows + output_row] =
                static_cast<hip_bfloat16>(total[element]);
        }
    }
#else
    (void)activation_codes; (void)activation_scales; (void)activation_status;
    (void)weight_codes; (void)weight_scales; (void)output; (void)tokens; (void)rows;
    (void)columns; (void)padded_columns; (void)tile;
#endif
}

__global__ __launch_bounds__(kA8W8PrefillThreads) void a8w8g32_linear_prefill_cta_kernel(
    const std::int8_t* activation_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status, const std::int8_t* weight_codes,
    const std::uint16_t* weight_scales, hip_bfloat16* output,
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) {
    __shared__ A8W8PrefillTile tile;
    a8w8g32_linear_prefill_cta_body<true>(
        activation_codes, activation_scales, activation_status, weight_codes,
        weight_scales, output, tokens, rows, columns, padded_columns, tile);
}

__global__ __launch_bounds__(kA8W8PrefillThreads)
void a8w8g32_linear_prefill_cta_global_inv_qualification_kernel(
    const std::int8_t* activation_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* activation_status, const std::int8_t* weight_codes,
    const std::uint16_t* weight_scales, hip_bfloat16* output,
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) {
    __shared__ A8W8PrefillTile tile;
    a8w8g32_linear_prefill_cta_body<false>(
        activation_codes, activation_scales, activation_status, weight_codes,
        weight_scales, output, tokens, rows, columns, padded_columns, tile);
}

#endif

} // namespace

bool a8q4g64_scalar_base_u32_offsets_fit(
    std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t padded_columns) noexcept {
    return scalar_base_offsets_fit(tokens, rows, padded_columns);
}

std::size_t q4g64_activation_workspace_capacity_bytes(std::uint32_t tokens,
                                                       std::uint32_t columns) noexcept {
    std::uint32_t padded = 0;
    std::size_t code_bytes = 0, scale_bytes = 0;
    if (!q4_padded_columns(columns, &padded) ||
        !q4_layout_bytes(tokens, columns, padded, &code_bytes, &scale_bytes) ||
        code_bytes > std::numeric_limits<std::size_t>::max() - (kQ4PlaneAlignment - 1U)) {
        return 0U;
    }
    const std::size_t scale_offset = align_up(code_bytes, kQ4PlaneAlignment);
    if (scale_offset > std::numeric_limits<std::size_t>::max() - scale_bytes -
                           kQ4PlaneAlignment - sizeof(std::uint32_t)) return 0U;
    const std::size_t status_offset = align_up(scale_offset + scale_bytes, kQ4PlaneAlignment);
    return status_offset + sizeof(std::uint32_t);
}

hipError_t q4g64_bind_activation_workspace(
    void* storage, std::size_t storage_bytes, std::uint32_t tokens,
    std::uint32_t columns, Q4G64ActivationWorkspace* out) noexcept {
    const std::size_t required = q4g64_activation_workspace_capacity_bytes(tokens, columns);
    std::uint32_t padded = 0;
    std::size_t code_bytes = 0, scale_bytes = 0;
    if (storage == nullptr || out == nullptr || required == 0U || storage_bytes != required ||
        reinterpret_cast<std::uintptr_t>(storage) % alignof(std::uint32_t) != 0U ||
        !q4_padded_columns(columns, &padded) ||
        !q4_layout_bytes(tokens, columns, padded, &code_bytes, &scale_bytes)) {
        return hipErrorInvalidValue;
    }
    auto* bytes = static_cast<std::uint8_t*>(storage);
    const std::size_t scale_offset = align_up(code_bytes, kQ4PlaneAlignment);
    const std::size_t status_offset = align_up(scale_offset + scale_bytes, kQ4PlaneAlignment);
    *out = Q4G64ActivationWorkspace{bytes, code_bytes,
                                    reinterpret_cast<std::uint16_t*>(bytes + scale_offset),
                                    scale_bytes,
                                    reinterpret_cast<std::uint32_t*>(bytes + status_offset),
                                    tokens, columns, padded};
    return hipSuccess;
}

hipError_t q4g64_quantize_activation(const Q4G64ActivationQuantizeArgs& args,
                                     hipStream_t stream) noexcept {
    if (args.input == nullptr || !valid_q4_workspace(args.workspace)) return hipErrorInvalidValue;
    std::size_t input_elements = 0, input_bytes = 0;
    if (!checked_mul(args.workspace.tokens, args.workspace.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.codes, args.workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.scales, args.workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.status, sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    hipError_t result = hipMemsetAsync(args.workspace.status, 0, sizeof(std::uint32_t), stream);
    if (result != hipSuccess) return result;
    const dim3 grid(args.workspace.padded_columns / kQ4Group, args.workspace.tokens);
    hipLaunchKernelGGL(q4g64_quantize_activation_kernel, grid, dim3(kQ4Group), 0, stream,
                       args.input, args.workspace.codes, args.workspace.scales,
                       args.workspace.status, args.workspace.columns,
                       args.workspace.padded_columns);
    return hipGetLastError();
}

hipError_t q4g64_linear_wmma32(const Q4G64LinearArgs& args,
                               hipStream_t stream) noexcept {
    if (!valid_q4_linear_args(args)) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kWmmaTile - 1U) / kWmmaTile,
                    (args.tokens + kWmmaTile - 1U) / kWmmaTile);
    hipLaunchKernelGGL(q4g64_linear_wmma32_kernel, grid, dim3(kWaveSize), 0, stream,
                       args.activation_codes, args.activation_scales,
                       args.activation_status, args.weight_codes, args.weight_scales,
                       args.output, args.tokens, args.rows, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t q4g64_linear_vector(const Q4G64LinearArgs& args,
                               hipStream_t stream) noexcept {
    if (!valid_q4_linear_args(args)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(q4g64_linear_vector_kernel, dim3(args.rows, args.tokens),
                       dim3(kThreads), 0, stream, args.activation_codes,
                       args.activation_scales, args.activation_status, args.weight_codes,
                       args.weight_scales,
                       args.output, args.tokens, args.rows, args.padded_columns);
    return hipGetLastError();
}

hipError_t q4g64_linear_prefill_cta_m64n128(
    const Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_q4_linear_args(args) || args.tokens <= 128U) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    if (args.rows == 34816U && args.columns == 5120U &&
        args.padded_columns == args.columns && args.tokens % kA8Q4PrefillTokenTile == 0U &&
        scalar_base_offsets_fit(args.tokens, args.rows, args.padded_columns)) {
        hipLaunchKernelGGL(a4q4g64_linear_prefill_pingpong_kernel,
            dim3(args.rows / 128U, args.tokens / 64U), dim3(kA8Q4PrefillThreads), 0, stream,
            args.activation_codes, args.activation_scales, args.activation_status,
            args.weight_codes, args.weight_scales, args.output, args.tokens, args.rows,
            args.columns, args.padded_columns);
        return hipGetLastError();
    }
    const dim3 grid((args.rows + 2U * kA8Q4PrefillRowTile - 1U) /
                        (2U * kA8Q4PrefillRowTile),
                    (args.tokens + kA8Q4PrefillTokenTile - 1U) /
                        kA8Q4PrefillTokenTile);
    hipLaunchKernelGGL(q4g64_linear_prefill_cta_m64n128_kernel,
                       grid, dim3(kA8Q4PrefillThreads), 0, stream,
                       args.activation_codes, args.activation_scales,
                       args.activation_status, args.weight_codes, args.weight_scales,
                       args.output, args.tokens, args.rows, args.columns,
                       args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t q4g64_prefill_qualification_resources(
    Q4G64PrefillQualificationResources* resources, bool pingpong) noexcept {
    if (resources == nullptr) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    hipFuncAttributes attributes{};
    const hipError_t result = hipFuncGetAttributes(
        &attributes, pingpong ? reinterpret_cast<const void*>(a4q4g64_linear_prefill_pingpong_kernel)
            : reinterpret_cast<const void*>(q4g64_linear_prefill_cta_m64n128_kernel));
    if (result != hipSuccess) return result;
    *resources = Q4G64PrefillQualificationResources{
        attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
        static_cast<int>(attributes.localSizeBytes), attributes.maxThreadsPerBlock};
    return hipSuccess;
#else
    (void)pingpong;
    return hipErrorNotSupported;
#endif
}

hipError_t q4g64_linear_candidate(const Q4G64CandidateArgs& args,
                                  hipStream_t stream) noexcept {
    std::size_t input_elements = 0, input_bytes = 0, output_elements = 0, output_bytes = 0;
    if (args.input == nullptr ||
        !checked_mul(args.tokens, args.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        !checked_mul(args.tokens, args.rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes) ||
        ranges_overlap(args.input, input_bytes, args.output, output_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_scales, args.weight_scale_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.input, input_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.output, output_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_scales, args.weight_scale_bytes)) {
        return hipErrorInvalidValue;
    }
    Q4G64ActivationWorkspace workspace{};
    if (q4g64_bind_activation_workspace(args.activation_workspace,
                                        args.activation_workspace_bytes, args.tokens,
                                        args.columns, &workspace) != hipSuccess ||
        workspace.padded_columns != args.padded_columns) return hipErrorInvalidValue;
    const Q4G64LinearArgs linear_args{
        workspace.codes, workspace.code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status,
        args.weight_codes, args.weight_code_bytes, args.weight_scales,
        args.weight_scale_bytes, args.output, args.tokens, args.rows, args.columns,
        args.padded_columns};
    if (!valid_q4_linear_args(linear_args)) return hipErrorInvalidValue;
    const hipError_t quantized = q4g64_quantize_activation({args.input, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
    if (args.tokens > 128U && args.rows == 34816U && args.columns == 5120U &&
        args.padded_columns == args.columns) {
        return q4g64_linear_prefill_cta_m64n128(linear_args, stream);
    }
    return q4g64_linear_wmma32(linear_args, stream);
}

std::size_t a8q4g64_activation_workspace_capacity_bytes(std::uint32_t tokens,
                                                         std::uint32_t columns) noexcept {
    std::uint32_t padded = 0;
    std::size_t plane_bytes = 0, scale_bytes = 0;
    if (!q4_padded_columns(columns, &padded) ||
        !q4_layout_bytes(tokens, columns, padded, &plane_bytes, &scale_bytes) ||
        plane_bytes > std::numeric_limits<std::size_t>::max() -
                          (kQ4PlaneAlignment - 1U)) {
        return 0U;
    }
    const std::size_t high_offset = align_up(plane_bytes, kQ4PlaneAlignment);
    if (high_offset > std::numeric_limits<std::size_t>::max() - plane_bytes) return 0U;
    const std::size_t high_end = high_offset + plane_bytes;
    if (high_end > std::numeric_limits<std::size_t>::max() -
                       (kQ4PlaneAlignment - 1U)) return 0U;
    const std::size_t scale_offset =
        align_up(high_end, kQ4PlaneAlignment);
    if (scale_offset > std::numeric_limits<std::size_t>::max() - scale_bytes) return 0U;
    const std::size_t scale_end = scale_offset + scale_bytes;
    if (scale_end > std::numeric_limits<std::size_t>::max() -
                        (kQ4PlaneAlignment - 1U)) return 0U;
    const std::size_t status_offset = align_up(scale_end, kQ4PlaneAlignment);
    if (status_offset > std::numeric_limits<std::size_t>::max() - sizeof(std::uint32_t)) {
        return 0U;
    }
    return status_offset + sizeof(std::uint32_t);
}

hipError_t a8q4g64_bind_activation_workspace(
    void* storage, std::size_t storage_bytes, std::uint32_t tokens,
    std::uint32_t columns, A8G64ActivationWorkspace* out) noexcept {
    const std::size_t required =
        a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    std::uint32_t padded = 0;
    std::size_t plane_bytes = 0, scale_bytes = 0;
    if (storage == nullptr || out == nullptr || required == 0U || storage_bytes != required ||
        reinterpret_cast<std::uintptr_t>(storage) % alignof(std::uint32_t) != 0U ||
        !q4_padded_columns(columns, &padded) ||
        !q4_layout_bytes(tokens, columns, padded, &plane_bytes, &scale_bytes)) {
        return hipErrorInvalidValue;
    }
    auto* bytes = static_cast<std::uint8_t*>(storage);
    const std::size_t high_offset = align_up(plane_bytes, kQ4PlaneAlignment);
    const std::size_t scale_offset =
        align_up(high_offset + plane_bytes, kQ4PlaneAlignment);
    const std::size_t status_offset = align_up(scale_offset + scale_bytes, kQ4PlaneAlignment);
    *out = A8G64ActivationWorkspace{
        bytes, plane_bytes, bytes + high_offset, plane_bytes,
        reinterpret_cast<std::uint16_t*>(bytes + scale_offset), scale_bytes,
        reinterpret_cast<std::uint32_t*>(bytes + status_offset), tokens, columns, padded};
    return hipSuccess;
}

hipError_t a8g64_quantize_activation(const A8G64ActivationQuantizeArgs& args,
                                     hipStream_t stream) noexcept {
    if (args.input == nullptr || !valid_a8_workspace(args.workspace)) {
        return hipErrorInvalidValue;
    }
    std::size_t input_elements = 0, input_bytes = 0;
    if (!checked_mul(args.workspace.tokens, args.workspace.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.low_codes,
                       args.workspace.low_code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.high_codes,
                       args.workspace.high_code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.scales,
                       args.workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.status,
                       sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    const std::uint32_t columns = args.workspace.columns;
    if (columns == args.workspace.padded_columns && columns % kQ4Group == 0U &&
        reinterpret_cast<std::uintptr_t>(args.input) % 16U == 0U &&
        static_cast<std::size_t>(args.workspace.tokens) * (columns / 8U) <=
            kSmallQuantizeMaximumVectors) {
        const std::uint32_t vectors = args.workspace.tokens * (columns / 8U);
        hipLaunchKernelGGL(a8g64_quantize_activation_small_kernel,
                           dim3((vectors + kSmallQuantizeThreads - 1U) / kSmallQuantizeThreads + 1U),
                           dim3(kSmallQuantizeThreads), 0, stream, args.input,
                           args.workspace.low_codes, args.workspace.high_codes,
                           args.workspace.scales, args.workspace.status, args.workspace.tokens,
                           columns);
        return hipGetLastError();
    }
    hipError_t result = hipMemsetAsync(args.workspace.status, 0, sizeof(std::uint32_t), stream);
    if (result != hipSuccess) return result;
    const dim3 grid(args.workspace.padded_columns / kQ4Group, args.workspace.tokens);
    hipLaunchKernelGGL(a8g64_quantize_activation_kernel, grid, dim3(kWaveSize), 0, stream,
                       args.input, args.workspace.low_codes, args.workspace.high_codes,
                       args.workspace.scales, args.workspace.status, args.workspace.columns,
                       args.workspace.padded_columns);
    return hipGetLastError();
}

hipError_t a8q4g64_linear_wmma32(const A8Q4G64LinearArgs& args,
                                  hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args)) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kWmmaTile - 1U) / kWmmaTile,
                    (args.tokens + kWmmaTile - 1U) / kWmmaTile);
    hipLaunchKernelGGL(a8q4g64_linear_wmma32_kernel, grid, dim3(kWaveSize), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status, args.weight_codes,
                       args.weight_scales, args.output, args.tokens, args.rows,
                       args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_decode_dot8_t1(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) ||
        !use_a8q4_decode_dot8_t1(
            args.tokens, args.rows, args.columns, args.padded_columns)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kThreads - 1U) / kThreads);
    hipLaunchKernelGGL(a8q4g64_linear_decode_dot8_t1_kernel,
                       grid, dim3(kThreads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status,
                       args.weight_codes, args.weight_scales, args.output,
                       args.rows, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_gdn_pair_t1(
    const A8Q4G64GdnPairArgs& args, hipStream_t stream) noexcept {
    constexpr std::uint32_t kRows0 = 4096U;
    constexpr std::uint32_t kRows1 = 12288U;
    constexpr std::uint32_t kColumns = 5120U;
    constexpr std::uint32_t kPaddedColumns = 5120U;
    const std::size_t required_workspace =
        a8q4g64_activation_workspace_capacity_bytes(1U, kColumns);
    std::size_t input_bytes = 0U;
    if (args.input == nullptr || args.tokens != 1U || args.columns != kColumns ||
        args.activation_workspace == nullptr ||
        args.activation_workspace_bytes != required_workspace ||
        !checked_mul(kColumns, sizeof(hip_bfloat16), &input_bytes)) {
        return hipErrorInvalidValue;
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(
            args.activation_workspace, args.activation_workspace_bytes, 1U,
            kColumns, &workspace) != hipSuccess ||
        workspace.padded_columns != kPaddedColumns) {
        return hipErrorInvalidValue;
    }
    const A8Q4G64LinearArgs linear0{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, args.weight0_codes, args.weight0_code_bytes,
        args.weight0_scales, args.weight0_scale_bytes, args.output0, 1U, kRows0,
        kColumns, kPaddedColumns};
    const A8Q4G64LinearArgs linear1{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, args.weight1_codes, args.weight1_code_bytes,
        args.weight1_scales, args.weight1_scale_bytes, args.output1, 1U, kRows1,
        kColumns, kPaddedColumns};
    if (!valid_a8q4_linear_args(linear0) ||
        !valid_a8q4_linear_args(linear1)) {
        return hipErrorInvalidValue;
    }
    constexpr std::size_t kOutput0Bytes =
        static_cast<std::size_t>(kRows0) * sizeof(hip_bfloat16);
    constexpr std::size_t kOutput1Bytes =
        static_cast<std::size_t>(kRows1) * sizeof(hip_bfloat16);
    const void* pointers[] = {
        args.input, args.activation_workspace,
        args.weight0_codes, args.weight0_scales, args.output0,
        args.weight1_codes, args.weight1_scales, args.output1};
    const std::size_t sizes[] = {
        input_bytes, args.activation_workspace_bytes,
        args.weight0_code_bytes, args.weight0_scale_bytes, kOutput0Bytes,
        args.weight1_code_bytes, args.weight1_scale_bytes, kOutput1Bytes};
    for (unsigned index = 0U; index < 8U; ++index) {
        for (unsigned other = index + 1U; other < 8U; ++other) {
            if (ranges_overlap(pointers[index], sizes[index],
                               pointers[other], sizes[other])) {
                return hipErrorInvalidValue;
            }
        }
    }
    const hipError_t quantized =
        a8g64_quantize_activation({args.input, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
#if defined(__HIPCC__)
    constexpr std::uint32_t kRows = kRows0 + kRows1;
    const dim3 grid((kRows + kThreads - 1U) / kThreads);
    hipLaunchKernelGGL(
        a8q4g64_gdn_pair_t1_kernel, grid, dim3(kThreads), 0, stream,
        workspace.low_codes, workspace.high_codes, workspace.scales,
        workspace.status, args.weight0_codes, args.weight0_scales, args.output0,
        args.weight1_codes, args.weight1_scales, args.output1);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_gdn_pair_t1_resources(
    A8Q4G64GdnPairResources* resources) noexcept {
    if (resources == nullptr) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    hipFuncAttributes attributes{};
    const hipError_t result = hipFuncGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(a8q4g64_gdn_pair_t1_kernel));
    if (result != hipSuccess) return result;
    *resources = A8Q4G64GdnPairResources{
        attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
        static_cast<int>(attributes.localSizeBytes), attributes.maxThreadsPerBlock};
    return hipSuccess;
#else
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_decode_dot8_block_t1_qualification(
    const A8Q4G64LinearArgs& args, std::uint32_t threads,
    hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) ||
        !use_a8q4_decode_dot8_t1(
            args.tokens, args.rows, args.columns, args.padded_columns) ||
        (threads != 32U && threads != 64U && threads != 128U && threads != 256U)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + threads - 1U) / threads);
    hipLaunchKernelGGL(a8q4g64_linear_decode_dot8_t1_kernel,
                       grid, dim3(threads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status,
                       args.weight_codes, args.weight_scales, args.output,
                       args.rows, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_dflash_small_t_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) ||
        !dflash_small_t_qualification_tokens(args.tokens) ||
        !dflash_small_t_qualification_shape(args.rows, args.columns)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kThreads - 1U) / kThreads);
    switch (args.tokens) {
    case 4U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<4U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 5U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<5U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 6U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<6U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 8U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<8U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 10U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<10U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 12U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<12U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 15U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<15U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 16U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<16U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 18U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<18U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 20U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<20U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    case 24U:
        hipLaunchKernelGGL((a8q4g64_linear_dflash_small_t_kernel<24U>), grid,
                           dim3(kThreads), 0, stream, args.activation_low_codes,
                           args.activation_high_codes, args.activation_scales,
                           args.activation_status, args.weight_codes, args.weight_scales,
                           args.output, args.rows, args.padded_columns);
        break;
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_dflash_mlp_down_t5_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) ||
        !is_a8q4_dflash_mlp_down_t5_eligible(
            args.tokens, args.rows, args.columns, args.padded_columns)) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kThreads - 1U) / kThreads);
    hipLaunchKernelGGL(a8q4g64_linear_dflash_mlp_down_t5_kernel,
                       grid, dim3(kThreads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status,
                       args.weight_codes, args.weight_scales, args.output);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_prefill_cta_m64n128_regression(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) || args.tokens <= 128U) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + 2U * kA8Q4PrefillRowTile - 1U) /
                        (2U * kA8Q4PrefillRowTile),
                    (args.tokens + kA8Q4PrefillTokenTile - 1U) /
                        kA8Q4PrefillTokenTile);
    hipLaunchKernelGGL(a8q4g64_linear_prefill_cta_m64n128_regression_kernel, grid,
                       dim3(kA8Q4PrefillThreads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status, args.weight_codes,
                       args.weight_scales, args.output, args.tokens, args.rows,
                       args.columns, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_prefill_cta_global_inv_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) || args.tokens <= 128U) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kA8Q4PrefillRowTile - 1U) / kA8Q4PrefillRowTile,
                    (args.tokens + kA8Q4PrefillTokenTile - 1U) /
                        kA8Q4PrefillTokenTile);
    hipLaunchKernelGGL(a8q4g64_linear_prefill_cta_global_inv_qualification_kernel,
                       grid, dim3(kA8Q4PrefillThreads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status,
                       args.weight_codes, args.weight_scales, args.output, args.tokens,
                       args.rows, args.columns, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_prefill_cta_m64n64_regression(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) || args.tokens <= 128U) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kA8Q4PrefillRowTile - 1U) / kA8Q4PrefillRowTile,
                    (args.tokens + kA8Q4PrefillTokenTile - 1U) /
                        kA8Q4PrefillTokenTile);
    hipLaunchKernelGGL(a8q4g64_linear_prefill_cta_m64n64_regression_kernel,
                       grid, dim3(kA8Q4PrefillThreads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status,
                       args.weight_codes, args.weight_scales, args.output, args.tokens,
                       args.rows, args.columns, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_prefill_cta_m128n128_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args) || args.tokens <= 128U) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + 2U * kA8Q4PrefillRowTile - 1U) /
                        (2U * kA8Q4PrefillRowTile),
                    (args.tokens + kA8Q4M128PrefillTokenTile - 1U) /
                        kA8Q4M128PrefillTokenTile);
    hipLaunchKernelGGL(a8q4g64_linear_prefill_cta_m128n128_qualification_kernel,
                       grid, dim3(kA8Q4M128PrefillThreads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status,
                       args.weight_codes, args.weight_scales, args.output, args.tokens,
                       args.rows, args.columns, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_linear_prefill_cta(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8q4_linear_args(args, true) || args.tokens <= 128U) {
        return hipErrorInvalidValue;
    }
    // The M128xN128 kernel has no tile predicates and uses U32 byte offsets. The size_t-addressed
    // single-bank kernel remains the complete direct-call fallback for every
    // tuple outside the qualified inventory.
    if (!use_a8q4_prefill_cta(args.tokens, args.rows, args.columns) ||
        args.columns != args.padded_columns ||
        args.tokens % kA8Q4M128PrefillTokenTile != 0U ||
        args.rows % kA8Q4M128PrefillTokenTile != 0U ||
        !scalar_base_offsets_fit(args.tokens, args.rows, args.padded_columns)) {
        if (args.accumulate_output) return hipErrorInvalidValue;
        return a8q4g64_linear_prefill_cta_m64n128_regression(args, stream);
    }
#if defined(__HIPCC__)
    const dim3 grid(args.tokens / kA8Q4M128PrefillTokenTile,
                    args.rows / kA8Q4M128PrefillTokenTile);
    hipLaunchKernelGGL(args.accumulate_output ? a8q4g64_linear_prefill_cta_kernel<true>
                                              : a8q4g64_linear_prefill_cta_kernel<false>,
                       grid, dim3(kA8Q4M128ProductionThreads), 0, stream,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status,
                       args.weight_codes, args.weight_scales, args.output, args.rows,
                       args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t run_a8q4_projection(const A8Q4G64LinearArgs& linear_args,
                               hipStream_t stream) noexcept;

// K5120 normalized preparation: the per-row small route (its status CTA publishes the status) up to
// kNormalizedSmallMaximumTokens rows, otherwise the reset launch and one wave per row.
hipError_t launch_normalized_prepare(const hip_bfloat16* input, const hip_bfloat16* norm,
                                     float eps, bool unit_offset, std::uint32_t tokens,
                                     const A8G64ActivationWorkspace& workspace,
                                     hip_bfloat16* normalized, hipStream_t stream,
                                     std::uint32_t* clear_status = nullptr) noexcept {
#if defined(__HIPCC__)
    if (tokens <= kNormalizedSmallMaximumTokens) {
        hipLaunchKernelGGL(a8g64_normalized_prepare_small_kernel, dim3(tokens + 1U),
                           dim3(kNormalizedSmallRowVectors), 0, stream, input, norm, eps,
                           unit_offset, tokens, workspace.low_codes, workspace.high_codes,
                           workspace.scales, workspace.status, normalized, clear_status);
        return hipGetLastError();
    }
    if (clear_status != nullptr) return hipErrorInvalidValue;
    const hipError_t reset = hipMemsetAsync(workspace.status, 0, sizeof(std::uint32_t), stream);
    if (reset != hipSuccess) return reset;
    hipLaunchKernelGGL(a8g64_normalized_prepare_prefill_kernel,
                       dim3((tokens + kNormalizedPrefillTokensPerBlock - 1U) /
                            kNormalizedPrefillTokensPerBlock),
                       dim3(kNormalizedPrefillTokensPerBlock * kWaveSize), 0, stream, input,
                       norm, eps, unit_offset, tokens, workspace.low_codes, workspace.high_codes,
                       workspace.scales, workspace.status, normalized);
    return hipGetLastError();
#else
    (void)input; (void)norm; (void)eps; (void)unit_offset; (void)tokens; (void)workspace;
    (void)normalized; (void)stream; (void)clear_status;
    return hipErrorNotSupported;
#endif
}

bool a8q4g64_normalized_linear_batched_supported(std::uint32_t tokens) noexcept {
    return tokens >= 2U && scalar_base_offsets_fit(tokens, 34816U, 5120U);
}

namespace {
hipError_t normalized_linear_batched(const A8Q4G64CandidateArgs& args, const hip_bfloat16* norm,
                                     float eps, bool unit_offset, std::uint32_t* clear_status,
                                     hipStream_t stream) noexcept {
    const std::uint32_t tokens = args.tokens;
    if (!a8q4g64_normalized_linear_batched_supported(tokens) || args.rows != 34816U ||
        args.columns != 5120U || args.padded_columns != 5120U || !(eps > 0.0F) ||
        eps > std::numeric_limits<float>::max()) return hipErrorInvalidValue;
    const std::size_t input_bytes = static_cast<std::size_t>(tokens) * 5120U * sizeof(hip_bfloat16);
    constexpr std::size_t norm_bytes = 5120U * sizeof(hip_bfloat16);
    constexpr std::size_t code_bytes = 34816U * 5120U / 2U;
    constexpr std::size_t scale_bytes = 34816U * 80U * sizeof(std::uint16_t);
    const std::size_t output_bytes =
        static_cast<std::size_t>(tokens) * 34816U * sizeof(hip_bfloat16);
    const std::size_t workspace_bytes = a8q4g64_activation_workspace_capacity_bytes(tokens, 5120U);
    if (args.weight_code_bytes != code_bytes || args.weight_scale_bytes != scale_bytes ||
        args.activation_workspace_bytes != workspace_bytes)
        return hipErrorInvalidValue;
    const void* pointers[]{args.input, norm, args.weight_codes, args.weight_scales,
                           args.output, args.activation_workspace};
    const std::size_t sizes[]{input_bytes, norm_bytes, code_bytes, scale_bytes,
                              output_bytes, workspace_bytes};
    // The normalized rows and gains are read as 16-byte vectors.
    const std::size_t alignments[]{16U, 16U, 8U, 2U, 2U, 4U};
    for (std::size_t index = 0U; index < 6U; ++index) {
        const auto first = reinterpret_cast<std::uintptr_t>(pointers[index]);
        if (pointers[index] == nullptr || first % alignments[index] != 0U ||
            first > std::numeric_limits<std::uintptr_t>::max() - sizes[index])
            return hipErrorInvalidValue;
        for (std::size_t other = 0U; other < index; ++other)
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other]))
                return hipErrorInvalidValue;
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(args.activation_workspace, workspace_bytes, tokens,
                                          5120U, &workspace) != hipSuccess)
        return hipErrorInvalidValue;
    const A8Q4G64LinearArgs linear_args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes, workspace.status,
        args.weight_codes, code_bytes, args.weight_scales, scale_bytes, args.output,
        tokens, 34816U, 5120U, 5120U};
    if (!valid_a8q4_linear_args(linear_args)) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    const hipError_t prepared = launch_normalized_prepare(
        args.input, norm, eps, unit_offset, tokens, workspace, nullptr, stream, clear_status);
    if (prepared != hipSuccess) return prepared;
    return run_a8q4_projection(linear_args, stream);
#else
    (void)unit_offset; (void)clear_status;
    return hipErrorNotSupported;
#endif
}
}

hipError_t a8q4g64_normalized_linear_batched(
    const A8Q4G64CandidateArgs& args, const hip_bfloat16* norm, float eps,
    bool unit_offset, hipStream_t stream) noexcept {
    return normalized_linear_batched(args, norm, eps, unit_offset, nullptr, stream);
}

bool a8q4g64_shared_activation_supported(std::uint32_t tokens, std::uint32_t columns,
                                         std::uint32_t rows) noexcept {
    return tokens != 0U && columns != 0U && columns % kQ4KAlignment == 0U && rows != 0U &&
           rows % 16U == 0U;
}

namespace {

// Binds the shared planes and validates every projection of a shared-activation call against the
// input rows and one another. `rows_bytes` is the byte extent of each input-sized row plane that
// must stay disjoint from every output (the input, and the normalized rows when published).
bool bind_shared_projections(const A8Q4G64SharedActivationArgs& args,
                             std::initializer_list<const void*> row_planes,
                             A8G64ActivationWorkspace* workspace,
                             A8Q4G64LinearArgs (&linear)[kA8Q4G64MaximumSharedProjections]) noexcept {
    const std::uint32_t tokens = args.tokens, columns = args.columns;
    if (args.input == nullptr || args.projection_count < 2U ||
        args.projection_count > kA8Q4G64MaximumSharedProjections ||
        a8q4g64_bind_activation_workspace(args.activation_workspace,
                                          args.activation_workspace_bytes, tokens, columns,
                                          workspace) != hipSuccess)
        return false;
    std::size_t row_bytes = 0;
    if (!checked_mul(static_cast<std::size_t>(tokens) * columns, sizeof(hip_bfloat16), &row_bytes))
        return false;
    for (const void* plane : row_planes)
        if (ranges_overlap(args.activation_workspace, args.activation_workspace_bytes, plane,
                           row_bytes))
            return false;
    for (std::uint32_t index = 0; index < args.projection_count; ++index) {
        const auto& projection = args.projections[index];
        if (!a8q4g64_shared_activation_supported(tokens, columns, projection.rows))
            return false;
        const std::size_t output_bytes =
            static_cast<std::size_t>(tokens) * projection.rows * sizeof(hip_bfloat16);
        linear[index] = A8Q4G64LinearArgs{
            workspace->low_codes, workspace->low_code_bytes, workspace->high_codes,
            workspace->high_code_bytes, workspace->scales, workspace->scale_bytes,
            workspace->status, projection.weight_codes, projection.weight_code_bytes,
            projection.weight_scales, projection.weight_scale_bytes, projection.output, tokens,
            projection.rows, columns, columns};
        if (!valid_a8q4_linear_args(linear[index])) return false;
        for (const void* plane : row_planes)
            if (ranges_overlap(plane, row_bytes, projection.output, output_bytes)) return false;
        for (std::uint32_t other = 0; other < index; ++other)
            if (ranges_overlap(args.projections[other].output,
                               static_cast<std::size_t>(tokens) * args.projections[other].rows *
                                   sizeof(hip_bfloat16),
                               projection.output, output_bytes))
                return false;
    }
    return true;
}

hipError_t run_shared_projections(
    const A8Q4G64SharedActivationArgs& args,
    const A8Q4G64LinearArgs (&linear)[kA8Q4G64MaximumSharedProjections],
    hipStream_t stream) noexcept {
    if (args.projection_count == 2U &&
        detail::use_a8q4_small_batch_projection_pair(linear[0], linear[1]) &&
        detail::use_a8q4_small_batch_projection(linear[0].tokens, linear[0].rows,
                                                linear[0].columns, linear[0].padded_columns) &&
        detail::use_a8q4_small_batch_projection(linear[1].tokens, linear[1].rows,
                                                linear[1].columns, linear[1].padded_columns))
        return detail::launch_a8q4_small_batch_projection_pair(linear[0], linear[1], stream);
    for (std::uint32_t index = 0; index < args.projection_count; ++index) {
        const hipError_t result = run_a8q4_projection(linear[index], stream);
        if (result != hipSuccess) return result;
    }
    return hipSuccess;
}

} // namespace

hipError_t a8q4g64_shared_activation_linear(
    const A8Q4G64SharedActivationArgs& args, hipStream_t stream) noexcept {
    A8G64ActivationWorkspace workspace{};
    A8Q4G64LinearArgs linear[kA8Q4G64MaximumSharedProjections];
    if (!bind_shared_projections(args, {args.input}, &workspace, linear))
        return hipErrorInvalidValue;
    const hipError_t result = a8g64_quantize_activation({args.input, workspace}, stream);
    if (result != hipSuccess) return result;
    return run_shared_projections(args, linear, stream);
}

hipError_t a8q4g64_normalized_shared_activation_linear(
    const A8Q4G64SharedActivationArgs& args, const hip_bfloat16* norm, float eps,
    bool unit_offset, hip_bfloat16* normalized, hipStream_t stream,
    const A8Q4G64SideProjection* side) noexcept {
    const std::uint32_t tokens = args.tokens;
    A8G64ActivationWorkspace workspace{};
    A8Q4G64LinearArgs linear[kA8Q4G64MaximumSharedProjections];
    const std::size_t row_bytes =
        static_cast<std::size_t>(tokens) * kNormalizedPrefillColumns * sizeof(hip_bfloat16);
    if (args.columns != kNormalizedPrefillColumns || norm == nullptr || normalized == nullptr ||
        !(eps > 0.0F) || eps > std::numeric_limits<float>::max() ||
        reinterpret_cast<std::uintptr_t>(args.input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(norm) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(normalized) % 16U != 0U ||
        !bind_shared_projections(args, {args.input, normalized}, &workspace, linear) ||
        ranges_overlap(args.input, row_bytes, normalized, row_bytes) ||
        (side != nullptr && (side->stream == nullptr || side->stream == stream ||
                             side->fork == nullptr || side->join == nullptr)))
        return hipErrorInvalidValue;
#if defined(__HIPCC__)
    hipError_t result = launch_normalized_prepare(args.input, norm, eps, unit_offset, tokens,
                                                  workspace, normalized, stream);
    if (result != hipSuccess) return result;
    if (side == nullptr) return run_shared_projections(args, linear, stream);
    const std::uint32_t last = args.projection_count - 1U;
    for (std::uint32_t index = 0; index < last; ++index) {
        result = run_a8q4_projection(linear[index], stream);
        if (result != hipSuccess) return result;
    }
    if ((result = hipEventRecord(side->fork, stream)) != hipSuccess ||
        (result = hipStreamWaitEvent(side->stream, side->fork, 0)) != hipSuccess ||
        (result = run_a8q4_projection(linear[last], side->stream)) != hipSuccess)
        return result;
    return hipEventRecord(side->join, side->stream);
#else
    (void)unit_offset;
    return hipErrorNotSupported;
#endif
}

namespace {
bool gated_normalized_small(std::uint32_t tokens) noexcept {
    return tokens >= 1U && tokens <= kGatedSmallMaximumTokens &&
           detail::use_a8q4_small_batch_projection(tokens, 5120U, 6144U, 6144U);
}
}

bool a8q4g64_gated_normalized_linear_supported(std::uint32_t tokens) noexcept {
    return gated_normalized_small(tokens) ||
           (tokens != 0U && tokens % kA8Q4M128PrefillTokenTile == 0U &&
            use_a8q4_prefill_cta(tokens, 5120U, 6144U) &&
            scalar_base_offsets_fit(tokens, 5120U, 6144U));
}

hipError_t a8q4g64_gated_normalized_linear(
    const A8Q4G64CandidateArgs& args, const hip_bfloat16* norm, const hip_bfloat16* gate,
    float eps, hipStream_t stream) noexcept {
    const std::uint32_t tokens = args.tokens;
    if (!a8q4g64_gated_normalized_linear_supported(tokens) || args.rows != 5120U ||
        args.columns != 6144U || args.padded_columns != 6144U || !(eps > 0.0F) ||
        eps > std::numeric_limits<float>::max()) return hipErrorInvalidValue;
    const std::size_t input_bytes = static_cast<std::size_t>(tokens) * 6144U * sizeof(hip_bfloat16);
    constexpr std::size_t norm_bytes = 128U * sizeof(hip_bfloat16);
    constexpr std::size_t code_bytes = 5120U * 6144U / 2U;
    constexpr std::size_t scale_bytes = 5120U * 96U * sizeof(std::uint16_t);
    const std::size_t output_bytes =
        static_cast<std::size_t>(tokens) * 5120U * sizeof(hip_bfloat16);
    const std::size_t workspace_bytes = a8q4g64_activation_workspace_capacity_bytes(tokens, 6144U);
    if (args.weight_code_bytes != code_bytes || args.weight_scale_bytes != scale_bytes ||
        args.activation_workspace_bytes != workspace_bytes)
        return hipErrorInvalidValue;
    const void* pointers[]{args.input, norm, gate, args.weight_codes, args.weight_scales,
                           args.output, args.activation_workspace};
    const std::size_t sizes[]{input_bytes, norm_bytes, input_bytes, code_bytes, scale_bytes,
                              output_bytes, workspace_bytes};
    // Rows, gates and gains are read as 16-byte vectors.
    const std::size_t alignments[]{16U, 16U, 16U, 8U, 2U, 2U, 4U};
    for (std::size_t index = 0U; index < 7U; ++index) {
        const auto first = reinterpret_cast<std::uintptr_t>(pointers[index]);
        if (pointers[index] == nullptr || first % alignments[index] != 0U ||
            first > std::numeric_limits<std::uintptr_t>::max() - sizes[index])
            return hipErrorInvalidValue;
        for (std::size_t other = 0U; other < index; ++other)
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other]))
                return hipErrorInvalidValue;
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(args.activation_workspace, workspace_bytes, tokens,
                                          6144U, &workspace) != hipSuccess)
        return hipErrorInvalidValue;
    const A8Q4G64LinearArgs linear_args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes, workspace.status,
        args.weight_codes, code_bytes, args.weight_scales, scale_bytes, args.output,
        tokens, 5120U, 6144U, 6144U, true};
    if (!valid_a8q4_linear_args(linear_args, true)) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    if (gated_normalized_small(tokens)) {
        const hipError_t reset = hipMemsetAsync(workspace.status, 0, sizeof(std::uint32_t), stream);
        if (reset != hipSuccess) return reset;
        hipLaunchKernelGGL(a8g64_gated_normalized_prepare_small_kernel,
                           dim3(tokens * (kGatedPrefillColumns / 8U) / kGatedSmallThreads),
                           dim3(kGatedSmallThreads), 0, stream, args.input, norm, gate, eps,
                           workspace.low_codes, workspace.high_codes, workspace.scales,
                           workspace.status);
        const hipError_t launched = hipGetLastError();
        if (launched != hipSuccess) return launched;
        return detail::launch_a8q4_small_batch_projection(linear_args, stream);
    }
#endif
    hipError_t result = hipMemsetAsync(workspace.status, 0, sizeof(std::uint32_t), stream);
    if (result != hipSuccess) return result;
#if defined(__HIPCC__)
    hipLaunchKernelGGL(a8g64_gated_normalized_prepare_prefill_kernel,
                       dim3((tokens + kNormalizedPrefillTokensPerBlock - 1U) /
                            kNormalizedPrefillTokensPerBlock),
                       dim3(kNormalizedPrefillTokensPerBlock * kWaveSize), 0, stream,
                       args.input, norm, gate, eps, tokens, workspace.low_codes,
                       workspace.high_codes, workspace.scales, workspace.status);
    result = hipGetLastError();
    if (result != hipSuccess) return result;
    return a8q4g64_linear_prefill_cta(linear_args, stream);
#else
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_normalized_linear_t1(
    const A8Q4G64CandidateArgs& args, const hip_bfloat16* norm, float eps,
    bool unit_offset, hipStream_t stream) noexcept {
    if (args.tokens != 1U || args.rows != 34816U || args.columns != 5120U ||
        args.padded_columns != 5120U || !(eps > 0.0F) ||
        eps > std::numeric_limits<float>::max()) return hipErrorInvalidValue;
    constexpr std::size_t input_bytes = 5120U * sizeof(hip_bfloat16);
    constexpr std::size_t code_bytes = 34816U * 5120U / 2U;
    constexpr std::size_t scale_bytes = 34816U * 80U * sizeof(std::uint16_t);
    constexpr std::size_t output_bytes = 34816U * sizeof(hip_bfloat16);
    const std::size_t workspace_bytes = a8q4g64_activation_workspace_capacity_bytes(1U, 5120U);
    if (args.weight_code_bytes != code_bytes || args.weight_scale_bytes != scale_bytes ||
        args.activation_workspace_bytes != workspace_bytes)
        return hipErrorInvalidValue;
    const void* pointers[]{args.input, norm, args.weight_codes, args.weight_scales,
                           args.output, args.activation_workspace};
    const std::size_t sizes[]{input_bytes, input_bytes, code_bytes, scale_bytes,
                              output_bytes, workspace_bytes};
    const std::size_t alignments[]{2U, 2U, 8U, 2U, 2U, 4U};
    for (std::size_t index = 0U; index < 6U; ++index) {
        const auto first = reinterpret_cast<std::uintptr_t>(pointers[index]);
        if (pointers[index] == nullptr || first % alignments[index] != 0U ||
            first > std::numeric_limits<std::uintptr_t>::max() - sizes[index])
            return hipErrorInvalidValue;
        for (std::size_t other = 0U; other < index; ++other)
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other]))
                return hipErrorInvalidValue;
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(args.activation_workspace, workspace_bytes,
                                         1U, 5120U, &workspace) != hipSuccess)
        return hipErrorInvalidValue;
    const A8Q4G64LinearArgs linear_args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes, workspace.status,
        args.weight_codes, code_bytes, args.weight_scales, scale_bytes, args.output,
        1U, 34816U, 5120U, 5120U};
    if (!valid_a8q4_linear_args(linear_args)) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    hipLaunchKernelGGL(a8g64_normalized_prepare_t1_kernel, dim3(1U), dim3(kThreads), 0,
                       stream, args.input, norm, eps, unit_offset, workspace.low_codes,
                       workspace.high_codes, workspace.scales, workspace.status);
    const hipError_t prepared = hipGetLastError();
    if (prepared != hipSuccess) return prepared;
    return a8q4g64_linear_decode_dot8_t1(linear_args, stream);
#else
    (void)unit_offset;
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8q4g64_projected_residual_t1(
    const A8Q4G64ProjectedResidualT1Args& args, hipStream_t stream) noexcept {
    if (args.tokens != 1U || args.rows != 5120U ||
        (args.columns != 6144U && args.columns != 17408U) ||
        args.padded_columns != args.columns) return hipErrorInvalidValue;
    const std::size_t input_bytes = args.columns * sizeof(hip_bfloat16);
    const std::size_t code_bytes = static_cast<std::size_t>(args.rows) * args.columns / 2U;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(args.rows) * (args.columns / kQ4Group) * sizeof(std::uint16_t);
    const std::size_t residual_bytes = args.rows * sizeof(hip_bfloat16);
    const std::size_t workspace_bytes =
        a8q4g64_activation_workspace_capacity_bytes(1U, args.columns);
    if (args.input == nullptr || args.weight_codes == nullptr ||
        args.weight_scales == nullptr || args.activation_workspace == nullptr ||
        args.residual == nullptr || args.weight_code_bytes != code_bytes ||
        args.weight_scale_bytes != scale_bytes ||
        args.activation_workspace_bytes != workspace_bytes ||
        !aligned_qword(args.weight_codes)) return hipErrorInvalidValue;
    const void* pointers[]{args.input, args.weight_codes, args.weight_scales,
                           args.residual, args.activation_workspace};
    const std::size_t sizes[]{input_bytes, code_bytes, scale_bytes,
                              residual_bytes, workspace_bytes};
    for (std::size_t index = 0U; index < 5U; ++index) {
        for (std::size_t other = index + 1U; other < 5U; ++other) {
            if (ranges_overlap(pointers[index], sizes[index], pointers[other], sizes[other]))
                return hipErrorInvalidValue;
        }
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(
            args.activation_workspace, args.activation_workspace_bytes, 1U,
            args.columns, &workspace) != hipSuccess) return hipErrorInvalidValue;
    const hipError_t quantized = a8g64_quantize_activation({args.input, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
#if defined(__HIPCC__)
    if (args.columns == 17408U) {
        hipLaunchKernelGGL(a8q4g64_projected_residual_down_stage2_kernel,
                           dim3(5120U / kThreads), dim3(kThreads), 0, stream,
                           workspace.low_codes, workspace.high_codes, workspace.scales,
                           workspace.status, args.weight_codes, args.weight_scales,
                           args.residual);
        return hipGetLastError();
    }
    hipLaunchKernelGGL(a8q4g64_projected_residual_t1_kernel,
                       dim3(5120U / kThreads), dim3(kThreads), 0, stream,
                       workspace.low_codes, workspace.high_codes, workspace.scales,
                       workspace.status, args.weight_codes, args.weight_scales,
                       args.residual, args.columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

// The selected A8Q4 matrix route for already prepared activation planes: the qualified small-batch,
// T1 dot8 and DFlash small-T routes, then the prefill CTA or WMMA32 route.
hipError_t run_a8q4_projection(const A8Q4G64LinearArgs& linear_args,
                               hipStream_t stream) noexcept {
    const A8Q4G64LinearArgs& args = linear_args;
    if (detail::use_a8q4_small_batch_projection(
            args.tokens, args.rows, args.columns, args.padded_columns)) {
        return detail::launch_a8q4_small_batch_projection(linear_args, stream);
    }
    // Only the small-batch and M128 prefill routes own the in-place residual epilogue.
    if (args.accumulate_output &&
        select_a8q4_prefill_route(args.tokens, args.rows, args.columns) !=
            A8Q4PrefillRoute::M64N128PingPongProduction)
        return hipErrorInvalidValue;
    if (use_a8q4_decode_dot8_t1(
            args.tokens, args.rows, args.columns, args.padded_columns)) {
        return a8q4g64_linear_decode_dot8_t1(linear_args, stream);
    }
    if (use_a8q4_dflash_small_t(
            args.tokens, args.rows, args.columns, args.padded_columns)) {
        return a8q4g64_linear_dflash_small_t_qualification(linear_args, stream);
    }
    if (use_a8q4_dflash_mlp_down_t5(
            args.tokens, args.rows, args.columns, args.padded_columns)) {
        return a8q4g64_linear_dflash_mlp_down_t5_qualification(linear_args, stream);
    }
    switch (select_a8q4_prefill_route(args.tokens, args.rows, args.columns)) {
    case A8Q4PrefillRoute::M64N128PingPongProduction:
        return a8q4g64_linear_prefill_cta(linear_args, stream);
    case A8Q4PrefillRoute::Wmma32:
        return a8q4g64_linear_wmma32(linear_args, stream);
    }
    return hipErrorInvalidValue;
}

hipError_t a8q4g64_linear_candidate(const A8Q4G64CandidateArgs& args,
                                    hipStream_t stream) noexcept {
    std::size_t input_elements = 0, input_bytes = 0, output_elements = 0, output_bytes = 0;
    if (args.input == nullptr ||
        !checked_mul(args.tokens, args.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        !checked_mul(args.tokens, args.rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes) ||
        ranges_overlap(args.input, input_bytes, args.output, output_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_scales, args.weight_scale_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.input, input_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.output, output_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_scales, args.weight_scale_bytes)) {
        return hipErrorInvalidValue;
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(args.activation_workspace,
                                          args.activation_workspace_bytes, args.tokens,
                                          args.columns, &workspace) != hipSuccess ||
        workspace.padded_columns != args.padded_columns) return hipErrorInvalidValue;
    const A8Q4G64LinearArgs linear_args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status,
        args.weight_codes, args.weight_code_bytes, args.weight_scales,
        args.weight_scale_bytes, args.output, args.tokens, args.rows, args.columns,
        args.padded_columns};
    if (!valid_a8q4_linear_args(linear_args)) return hipErrorInvalidValue;
    const hipError_t quantized = a8g64_quantize_activation({args.input, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
    return run_a8q4_projection(linear_args, stream);
}

bool a8q4g64_projected_residual_supported(std::uint32_t tokens, std::uint32_t rows,
                                          std::uint32_t columns) noexcept {
    const bool small_batch = rows == 5U * 1024U &&
        detail::use_a8q4_small_batch_projection(tokens, rows, columns, columns);
    const bool prefill = tokens != 0U && tokens % kA8Q4M128PrefillTokenTile == 0U &&
        rows % kA8Q4M128PrefillTokenTile == 0U && columns % kQ4KAlignment == 0U &&
        use_a8q4_prefill_cta(tokens, rows, columns) &&
        scalar_base_offsets_fit(tokens, rows, columns);
    return small_batch || prefill;
}

hipError_t a8q4g64_projected_residual(const A8Q4G64CandidateArgs& args,
                                      hipStream_t stream) noexcept {
    std::size_t input_bytes = 0, residual_bytes = 0;
    if (args.input == nullptr || args.padded_columns != args.columns ||
        !a8q4g64_projected_residual_supported(args.tokens, args.rows, args.columns) ||
        !checked_mul(static_cast<std::size_t>(args.tokens) * args.columns,
                     sizeof(hip_bfloat16), &input_bytes) ||
        !checked_mul(static_cast<std::size_t>(args.tokens) * args.rows, sizeof(hip_bfloat16),
                     &residual_bytes) ||
        ranges_overlap(args.input, input_bytes, args.output, residual_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_scales, args.weight_scale_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes, args.input,
                       input_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes, args.output,
                       residual_bytes)) {
        return hipErrorInvalidValue;
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(args.activation_workspace,
                                          args.activation_workspace_bytes, args.tokens,
                                          args.columns, &workspace) != hipSuccess)
        return hipErrorInvalidValue;
    const A8Q4G64LinearArgs linear_args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes, workspace.status,
        args.weight_codes, args.weight_code_bytes, args.weight_scales, args.weight_scale_bytes,
        args.output, args.tokens, args.rows, args.columns, args.columns, true};
    if (!valid_a8q4_linear_args(linear_args, true)) return hipErrorInvalidValue;
    const hipError_t quantized = a8g64_quantize_activation({args.input, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
    return run_a8q4_projection(linear_args, stream);
}

namespace {
hipError_t launch_fused_silu_quantize(const hip_bfloat16* gate_up,
                                      const A8G64ActivationWorkspace& workspace, bool reset,
                                      hipStream_t stream) noexcept;

hipError_t fused_silu_down(const FusedSiluA8Q4G64DownArgs& args, bool status_cleared,
                           hipStream_t stream) noexcept {
    const std::uint32_t tokens = args.tokens;
    constexpr std::uint32_t rows = 5120U;
    constexpr std::uint32_t columns = 17408U;
    std::size_t gate_up_elements = 0, gate_up_bytes = 0;
    std::size_t output_elements = 0, output_bytes = 0;
    if (!a8q4g64_projected_residual_supported(tokens, rows, columns) || args.rows != rows ||
        args.columns != columns ||
        args.padded_columns != columns || args.gate_up == nullptr || args.residual == nullptr ||
        !checked_mul(tokens, 2U * columns, &gate_up_elements) ||
        !checked_mul(gate_up_elements, sizeof(hip_bfloat16), &gate_up_bytes) ||
        !checked_mul(tokens, rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes) ||
        ranges_overlap(args.gate_up, gate_up_bytes, args.residual, output_bytes) ||
        ranges_overlap(args.gate_up, gate_up_bytes, args.weight_codes,
                       args.weight_code_bytes) ||
        ranges_overlap(args.gate_up, gate_up_bytes, args.weight_scales,
                       args.weight_scale_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.gate_up, gate_up_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.residual, output_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_scales, args.weight_scale_bytes)) {
        return hipErrorInvalidValue;
    }
    A8G64ActivationWorkspace workspace{};
    if (a8q4g64_bind_activation_workspace(
            args.activation_workspace, args.activation_workspace_bytes, tokens, columns,
            &workspace) != hipSuccess) {
        return hipErrorInvalidValue;
    }
    const A8Q4G64LinearArgs linear_args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, args.weight_codes, args.weight_code_bytes, args.weight_scales,
        args.weight_scale_bytes, args.residual, tokens, rows, columns, columns, true};
    if (!valid_a8q4_linear_args(linear_args, true)) return hipErrorInvalidValue;
    hipError_t result = status_cleared
        ? launch_fused_silu_quantize(args.gate_up, workspace, false, stream)
        : fused_silu_a8g64_prepare(args.gate_up, workspace, stream);
    if (result != hipSuccess) return result;
    return run_a8q4_projection(linear_args, stream);
}
}

hipError_t fused_silu_a8q4g64_down(
    const FusedSiluA8Q4G64DownArgs& args, hipStream_t stream) noexcept {
    return fused_silu_down(args, false, stream);
}

bool a8q4g64_normalized_mlp_supported(std::uint32_t tokens) noexcept {
    return tokens <= kNormalizedSmallMaximumTokens &&
           a8q4g64_normalized_linear_batched_supported(tokens) &&
           a8q4g64_projected_residual_supported(tokens, 5120U, 17408U);
}

hipError_t a8q4g64_normalized_mlp(const A8Q4G64CandidateArgs& gate_up,
                                  const hip_bfloat16* norm, float eps, bool unit_offset,
                                  const FusedSiluA8Q4G64DownArgs& down,
                                  hipStream_t stream) noexcept {
    const std::uint32_t tokens = gate_up.tokens;
    if (!a8q4g64_normalized_mlp_supported(tokens) || down.tokens != tokens ||
        down.gate_up != gate_up.output ||
        gate_up.activation_workspace != down.activation_workspace)
        return hipErrorInvalidValue;
    A8G64ActivationWorkspace down_workspace{};
    if (a8q4g64_bind_activation_workspace(down.activation_workspace,
                                          down.activation_workspace_bytes, tokens, 17408U,
                                          &down_workspace) != hipSuccess)
        return hipErrorInvalidValue;
    // The prepare clears the down status only if the gate/up stage never touches that word.
    const auto* gate_up_end =
        static_cast<const std::uint8_t*>(gate_up.activation_workspace) +
        gate_up.activation_workspace_bytes;
    if (reinterpret_cast<const std::uint8_t*>(down_workspace.status) < gate_up_end)
        return hipErrorInvalidValue;
    const hipError_t projected =
        normalized_linear_batched(gate_up, norm, eps, unit_offset, down_workspace.status, stream);
    if (projected != hipSuccess) return projected;
    return fused_silu_down(down, true, stream);
}

hipError_t fused_silu_a8g64_prepare(
    const hip_bfloat16* gate_up, const A8G64ActivationWorkspace& workspace,
    hipStream_t stream) noexcept {
    const std::uint32_t tokens = workspace.tokens;
    constexpr std::uint32_t columns = 17408U;
    if (gate_up == nullptr || !valid_a8_workspace(workspace) || tokens == 0U ||
        workspace.columns != columns ||
        workspace.padded_columns != columns) {
        return hipErrorInvalidValue;
    }
    std::size_t gate_up_elements = 0, gate_up_bytes = 0;
    if (!checked_mul(tokens, 2U * columns, &gate_up_elements) ||
        !checked_mul(gate_up_elements, sizeof(hip_bfloat16), &gate_up_bytes) ||
        ranges_overlap(gate_up, gate_up_bytes, workspace.low_codes,
                       workspace.low_code_bytes) ||
        ranges_overlap(gate_up, gate_up_bytes, workspace.high_codes,
                       workspace.high_code_bytes) ||
        ranges_overlap(gate_up, gate_up_bytes, workspace.scales,
                       workspace.scale_bytes) ||
        ranges_overlap(gate_up, gate_up_bytes, workspace.status,
                       sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    return launch_fused_silu_quantize(gate_up, workspace, true, stream);
}

namespace {
hipError_t launch_fused_silu_quantize(const hip_bfloat16* gate_up,
                                      const A8G64ActivationWorkspace& workspace, bool reset,
                                      hipStream_t stream) noexcept {
    if (reset) {
        const hipError_t result =
            hipMemsetAsync(workspace.status, 0, sizeof(std::uint32_t), stream);
        if (result != hipSuccess) return result;
    }
    hipLaunchKernelGGL(fused_silu_a8g64_quantize_kernel,
                       dim3(17408U / kQ4Group, workspace.tokens), dim3(kWaveSize), 0, stream,
                       gate_up, workspace.low_codes, workspace.high_codes,
                       workspace.scales, workspace.status);
    return hipGetLastError();
}
}

std::size_t a8w8g32_activation_workspace_capacity_bytes(std::uint32_t tokens,
                                                         std::uint32_t columns) noexcept {
    const std::size_t padded_size = align_up(static_cast<std::size_t>(columns), kKAlignment);
    if (tokens == 0U || columns == 0U || padded_size > std::numeric_limits<std::uint32_t>::max()) {
        return 0U;
    }
    const auto padded = static_cast<std::uint32_t>(padded_size);
    std::size_t code_bytes = 0, scale_bytes = 0;
    if (!w8_layout_bytes(tokens, columns, padded, &code_bytes, &scale_bytes) ||
        code_bytes > std::numeric_limits<std::size_t>::max() - (kQ4PlaneAlignment - 1U)) {
        return 0U;
    }
    const std::size_t scale_offset = align_up(code_bytes, kQ4PlaneAlignment);
    if (scale_offset > std::numeric_limits<std::size_t>::max() - scale_bytes) return 0U;
    const std::size_t scale_end = scale_offset + scale_bytes;
    if (scale_end > std::numeric_limits<std::size_t>::max() -
                        (kQ4PlaneAlignment - 1U)) return 0U;
    const std::size_t status_offset = align_up(scale_end, kQ4PlaneAlignment);
    if (status_offset > std::numeric_limits<std::size_t>::max() - sizeof(std::uint32_t)) {
        return 0U;
    }
    return status_offset + sizeof(std::uint32_t);
}

hipError_t a8w8g32_bind_activation_workspace(
    void* storage, std::size_t storage_bytes, std::uint32_t tokens,
    std::uint32_t columns, A8G32ActivationWorkspace* out) noexcept {
    const std::size_t required = a8w8g32_activation_workspace_capacity_bytes(tokens, columns);
    const std::size_t padded_size = align_up(static_cast<std::size_t>(columns), kKAlignment);
    if (storage == nullptr || out == nullptr || required == 0U || storage_bytes != required ||
        padded_size > std::numeric_limits<std::uint32_t>::max() ||
        reinterpret_cast<std::uintptr_t>(storage) % alignof(std::uint32_t) != 0U) {
        return hipErrorInvalidValue;
    }
    const auto padded = static_cast<std::uint32_t>(padded_size);
    std::size_t code_bytes = 0, scale_bytes = 0;
    if (!w8_layout_bytes(tokens, columns, padded, &code_bytes, &scale_bytes)) {
        return hipErrorInvalidValue;
    }
    auto* bytes = static_cast<std::uint8_t*>(storage);
    const std::size_t scale_offset = align_up(code_bytes, kQ4PlaneAlignment);
    const std::size_t status_offset = align_up(scale_offset + scale_bytes, kQ4PlaneAlignment);
    *out = A8G32ActivationWorkspace{
        reinterpret_cast<std::int8_t*>(bytes), code_bytes,
        reinterpret_cast<std::uint16_t*>(bytes + scale_offset), scale_bytes,
        reinterpret_cast<std::uint32_t*>(bytes + status_offset), tokens, columns, padded};
    return hipSuccess;
}

hipError_t a8g32_quantize_activation(const A8G32ActivationQuantizeArgs& args,
                                     hipStream_t stream) noexcept {
    if (args.input == nullptr || !valid_a8w8_workspace(args.workspace)) {
        return hipErrorInvalidValue;
    }
    std::size_t input_elements = 0, input_bytes = 0;
    if (!checked_mul(args.workspace.tokens, args.workspace.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.codes,
                       args.workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.scales,
                       args.workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.status,
                       sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    hipError_t result = hipMemsetAsync(args.workspace.status, 0, sizeof(std::uint32_t), stream);
    if (result != hipSuccess) return result;
    const dim3 grid(args.workspace.padded_columns / kW8Group, args.workspace.tokens);
    hipLaunchKernelGGL(a8g32_quantize_activation_kernel, grid, dim3(kWaveSize), 0, stream,
                       args.input, args.workspace.codes, args.workspace.scales,
                       args.workspace.status, args.workspace.columns,
                       args.workspace.padded_columns);
    return hipGetLastError();
}

hipError_t a8w8g32_linear_wmma32(const A8W8G32LinearArgs& args,
                                 hipStream_t stream) noexcept {
    if (!valid_a8w8_linear_args(args)) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kWmmaTile - 1U) / kWmmaTile,
                    (args.tokens + kWmmaTile - 1U) / kWmmaTile);
    hipLaunchKernelGGL(a8w8g32_linear_wmma32_kernel, grid, dim3(kWaveSize), 0, stream,
                       args.activation_codes, args.activation_scales,
                       args.activation_status, args.weight_codes, args.weight_scales,
                       args.output, args.tokens, args.rows, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8w8g32_linear_prefill_cta(const A8W8G32LinearArgs& args,
                                      hipStream_t stream) noexcept {
    if (!valid_a8w8_linear_args(args) || args.tokens <= 128U) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kA8W8PrefillRowTile - 1U) / kA8W8PrefillRowTile,
                    (args.tokens + kA8W8PrefillTokenTile - 1U) /
                        kA8W8PrefillTokenTile);
    hipLaunchKernelGGL(a8w8g32_linear_prefill_cta_kernel, grid,
                       dim3(kA8W8PrefillThreads), 0, stream, args.activation_codes,
                       args.activation_scales, args.activation_status, args.weight_codes,
                       args.weight_scales, args.output, args.tokens, args.rows,
                       args.columns, args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8w8g32_linear_prefill_cta_global_inv_qualification(
    const A8W8G32LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_a8w8_linear_args(args) || args.tokens <= 128U) {
        return hipErrorInvalidValue;
    }
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kA8W8PrefillRowTile - 1U) / kA8W8PrefillRowTile,
                    (args.tokens + kA8W8PrefillTokenTile - 1U) /
                        kA8W8PrefillTokenTile);
    hipLaunchKernelGGL(a8w8g32_linear_prefill_cta_global_inv_qualification_kernel,
                       grid, dim3(kA8W8PrefillThreads), 0, stream,
                       args.activation_codes, args.activation_scales,
                       args.activation_status, args.weight_codes, args.weight_scales,
                       args.output, args.tokens, args.rows, args.columns,
                       args.padded_columns);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t a8w8g32_linear_candidate(const A8W8G32CandidateArgs& args,
                                    hipStream_t stream) noexcept {
    std::size_t input_elements = 0, input_bytes = 0, output_elements = 0, output_bytes = 0;
    if (args.input == nullptr ||
        !checked_mul(args.tokens, args.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        !checked_mul(args.tokens, args.rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes) ||
        ranges_overlap(args.input, input_bytes, args.output, output_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.weight_scales, args.weight_scale_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.input, input_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.output, output_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_codes, args.weight_code_bytes) ||
        ranges_overlap(args.activation_workspace, args.activation_workspace_bytes,
                       args.weight_scales, args.weight_scale_bytes)) {
        return hipErrorInvalidValue;
    }
    A8G32ActivationWorkspace workspace{};
    if (a8w8g32_bind_activation_workspace(args.activation_workspace,
                                          args.activation_workspace_bytes, args.tokens,
                                          args.columns, &workspace) != hipSuccess ||
        workspace.padded_columns != args.padded_columns) {
        return hipErrorInvalidValue;
    }
    const A8W8G32LinearArgs linear_args{
        workspace.codes, workspace.code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, args.weight_codes, args.weight_code_bytes, args.weight_scales,
        args.weight_scale_bytes, args.output, args.tokens, args.rows, args.columns,
        args.padded_columns};
    if (!valid_a8w8_linear_args(linear_args)) return hipErrorInvalidValue;
    const hipError_t quantized = a8g32_quantize_activation({args.input, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
    return use_a8w8_prefill_cta(args.tokens, args.rows, args.columns)
               ? a8w8g32_linear_prefill_cta(linear_args, stream)
               : a8w8g32_linear_wmma32(linear_args, stream);
}

hipError_t a8w8g32_kernel_resources(A8W8G32KernelResources* resources) noexcept {
    if (resources == nullptr) return hipErrorInvalidValue;
    hipFuncAttributes quantize{};
    hipFuncAttributes wmma{};
    hipFuncAttributes prefill_cta{};
    hipFuncAttributes prefill_cta_global_inv_qualification{};
    hipError_t result = hipFuncGetAttributes(
        &quantize, reinterpret_cast<const void*>(a8g32_quantize_activation_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &wmma, reinterpret_cast<const void*>(a8w8g32_linear_wmma32_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &prefill_cta, reinterpret_cast<const void*>(a8w8g32_linear_prefill_cta_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &prefill_cta_global_inv_qualification, reinterpret_cast<const void*>(
            a8w8g32_linear_prefill_cta_global_inv_qualification_kernel));
    if (result != hipSuccess) return result;
    *resources = A8W8G32KernelResources{
        quantize.numRegs, static_cast<int>(quantize.sharedSizeBytes),
        static_cast<int>(quantize.localSizeBytes), wmma.numRegs,
        static_cast<int>(wmma.sharedSizeBytes), static_cast<int>(wmma.localSizeBytes),
        prefill_cta.numRegs, static_cast<int>(prefill_cta.sharedSizeBytes),
        static_cast<int>(prefill_cta.localSizeBytes),
        prefill_cta_global_inv_qualification.numRegs,
        static_cast<int>(prefill_cta_global_inv_qualification.sharedSizeBytes),
        static_cast<int>(prefill_cta_global_inv_qualification.localSizeBytes)};
    return hipSuccess;
}

hipError_t a8q4g64_kernel_resources(A8Q4G64KernelResources* resources) noexcept {
    if (resources == nullptr) return hipErrorInvalidValue;
    hipFuncAttributes quantize{};
    hipFuncAttributes wmma{};
    hipFuncAttributes decode_dot8_t1{};
    hipFuncAttributes prefill_cta{};
    hipFuncAttributes prefill_cta_global_inv_qualification{};
    hipFuncAttributes prefill_cta_m64n64_regression{};
    hipFuncAttributes prefill_cta_m128n128_qualification{};
    hipFuncAttributes prefill_cta_m64n128_regression{};
    hipError_t result = hipFuncGetAttributes(
        &quantize, reinterpret_cast<const void*>(a8g64_quantize_activation_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &wmma, reinterpret_cast<const void*>(a8q4g64_linear_wmma32_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &decode_dot8_t1, reinterpret_cast<const void*>(
            a8q4g64_linear_decode_dot8_t1_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &prefill_cta,
        reinterpret_cast<const void*>(a8q4g64_linear_prefill_cta_kernel<false>));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &prefill_cta_global_inv_qualification, reinterpret_cast<const void*>(
            a8q4g64_linear_prefill_cta_global_inv_qualification_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &prefill_cta_m64n64_regression, reinterpret_cast<const void*>(
            a8q4g64_linear_prefill_cta_m64n64_regression_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &prefill_cta_m128n128_qualification, reinterpret_cast<const void*>(
            a8q4g64_linear_prefill_cta_m128n128_qualification_kernel));
    if (result != hipSuccess) return result;
    result = hipFuncGetAttributes(
        &prefill_cta_m64n128_regression, reinterpret_cast<const void*>(
            a8q4g64_linear_prefill_cta_m64n128_regression_kernel));
    if (result != hipSuccess) return result;
    *resources = A8Q4G64KernelResources{
        quantize.numRegs, static_cast<int>(quantize.sharedSizeBytes),
        static_cast<int>(quantize.localSizeBytes), wmma.numRegs,
        static_cast<int>(wmma.sharedSizeBytes), static_cast<int>(wmma.localSizeBytes),
        decode_dot8_t1.numRegs,
        static_cast<int>(decode_dot8_t1.sharedSizeBytes),
        static_cast<int>(decode_dot8_t1.localSizeBytes),
        decode_dot8_t1.maxThreadsPerBlock,
        prefill_cta.numRegs, static_cast<int>(prefill_cta.sharedSizeBytes),
        static_cast<int>(prefill_cta.localSizeBytes),
        prefill_cta_global_inv_qualification.numRegs,
        static_cast<int>(prefill_cta_global_inv_qualification.sharedSizeBytes),
        static_cast<int>(prefill_cta_global_inv_qualification.localSizeBytes),
        prefill_cta_m64n64_regression.numRegs,
        static_cast<int>(prefill_cta_m64n64_regression.sharedSizeBytes),
        static_cast<int>(prefill_cta_m64n64_regression.localSizeBytes),
        prefill_cta_m128n128_qualification.numRegs,
        static_cast<int>(prefill_cta_m128n128_qualification.sharedSizeBytes),
        static_cast<int>(prefill_cta_m128n128_qualification.localSizeBytes),
        prefill_cta_m128n128_qualification.maxThreadsPerBlock,
        prefill_cta_m64n128_regression.numRegs,
        static_cast<int>(prefill_cta_m64n128_regression.sharedSizeBytes),
        static_cast<int>(prefill_cta_m64n128_regression.localSizeBytes),
        prefill_cta_m64n128_regression.maxThreadsPerBlock};
    return hipSuccess;
}

hipError_t bf16_linear_baseline(const Bf16LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_qualified_width(args.input, args.output, args.tokens, args.rows, args.columns) ||
        args.weights == nullptr) return hipErrorInvalidValue;
    hipLaunchKernelGGL(bf16_linear_kernel, dim3(args.rows, args.tokens), dim3(kThreads), 0, stream,
                       args.input, args.weights, args.output, args.columns, args.rows);
    return hipGetLastError();
}

hipError_t w8g32_linear_baseline(const W8G32LinearArgs& args, hipStream_t stream) noexcept {
    std::size_t expected_codes = 0;
    std::size_t expected_scales = 0;
    if (!valid_qualified_width(args.input, args.output, args.tokens, args.rows, args.columns) ||
        args.codes == nullptr || args.scales == nullptr ||
        !w8_layout_bytes(args.rows, args.columns, args.padded_columns, &expected_codes,
                         &expected_scales) ||
        args.code_bytes != expected_codes || args.scale_bytes != expected_scales) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(w8g32_linear_kernel, dim3(args.rows, args.tokens), dim3(kThreads), 0, stream,
                       args.input, args.codes, args.scales, args.output, args.columns, args.rows,
                       args.padded_columns);
    return hipGetLastError();
}

hipError_t bf16_linear_wave8(const Bf16LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_decode(args.input, args.output, args.tokens, args.rows, args.columns) ||
        args.weights == nullptr) return hipErrorInvalidValue;
    const dim3 grid(args.rows, (args.tokens + kWavesPerDecodeBlock - 1U) / kWavesPerDecodeBlock);
    hipLaunchKernelGGL((wave8_linear_kernel<false, kWavesPerDecodeBlock>), grid, dim3(kThreads), 0, stream, args.input,
                       args.weights, nullptr, args.output, args.columns, args.rows, 0U,
                       args.tokens);
    return hipGetLastError();
}

hipError_t w8g32_linear_wave8(const W8G32LinearArgs& args, hipStream_t stream) noexcept {
    std::size_t expected_codes = 0;
    std::size_t expected_scales = 0;
    if (!valid_decode(args.input, args.output, args.tokens, args.rows, args.columns) ||
        args.codes == nullptr || args.scales == nullptr ||
        !w8_layout_bytes(args.rows, args.columns, args.padded_columns, &expected_codes,
                         &expected_scales) ||
        args.code_bytes != expected_codes || args.scale_bytes != expected_scales) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(args.rows, (args.tokens + kWavesPerDecodeBlock - 1U) / kWavesPerDecodeBlock);
    hipLaunchKernelGGL((wave8_linear_kernel<true, kWavesPerDecodeBlock>), grid, dim3(kThreads), 0, stream, args.input,
                       args.codes, args.scales, args.output, args.columns, args.rows,
                       args.padded_columns, args.tokens);
    return hipGetLastError();
}

hipError_t bf16_linear_wave16(const Bf16LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_prefill(args.input, args.output, args.tokens, args.rows, args.columns) ||
        args.weights == nullptr) return hipErrorInvalidValue;
    const dim3 grid(args.rows,
                    (args.tokens + kWavesPerPrefillBlock - 1U) / kWavesPerPrefillBlock);
    hipLaunchKernelGGL((wave8_linear_kernel<false, kWavesPerPrefillBlock>), grid,
                       dim3(kPrefillThreads), 0, stream, args.input, args.weights, nullptr,
                       args.output, args.columns, args.rows, 0U, args.tokens);
    return hipGetLastError();
}

hipError_t w8g32_linear_wave16(const W8G32LinearArgs& args, hipStream_t stream) noexcept {
    std::size_t expected_codes = 0;
    std::size_t expected_scales = 0;
    if (!valid_prefill(args.input, args.output, args.tokens, args.rows, args.columns) ||
        args.codes == nullptr || args.scales == nullptr ||
        !w8_layout_bytes(args.rows, args.columns, args.padded_columns, &expected_codes,
                         &expected_scales) ||
        args.code_bytes != expected_codes || args.scale_bytes != expected_scales) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(args.rows,
                    (args.tokens + kWavesPerPrefillBlock - 1U) / kWavesPerPrefillBlock);
    hipLaunchKernelGGL((wave8_linear_kernel<true, kWavesPerPrefillBlock>), grid,
                       dim3(kPrefillThreads), 0, stream, args.input, args.codes, args.scales,
                       args.output, args.columns, args.rows, args.padded_columns, args.tokens);
    return hipGetLastError();
}

hipError_t bf16_linear_wmma16(const Bf16LinearArgs& args, hipStream_t stream) noexcept {
    if (!valid_prefill(args.input, args.output, args.tokens, args.rows, args.columns) ||
        args.weights == nullptr || args.columns % kWmmaTile != 0U) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    const dim3 grid((args.rows + kWmmaTile - 1U) / kWmmaTile,
                    (args.tokens + kWmmaTile - 1U) / kWmmaTile);
    hipLaunchKernelGGL(bf16_linear_wmma16_kernel, grid, dim3(kWaveSize), 0, stream, args.input,
                       args.weights, args.output, args.columns, args.rows, args.tokens);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t bf16_linear(const Bf16LinearArgs& args, hipStream_t stream) noexcept {
    if (args.tokens >= 1U && args.tokens <= 24U &&
        ((args.rows == 7168U && args.columns == 5120U) ||
         (args.rows == 5120U && args.columns == 6144U))) {
        // Protected projections retain the ordinary T1 reduction at every
        // supported small-token width. One batched grid owns all tokens;
        // neither a host request loop nor width-dependent reassociation.
        return bf16_linear_baseline(args, stream);
    }
    if (args.tokens >= 9U) { return bf16_linear_prefill(args, stream); }
    // Fixed R9700 decode dispatch selected by the standalone real-shape sweep:
    // baseline wins T=1..3; coalesced wave8 reuse wins T=4..8.
    return args.tokens <= 3U ? bf16_linear_baseline(args, stream)
                              : bf16_linear_wave8(args, stream);
}

hipError_t w8g32_linear(const W8G32LinearArgs& args, hipStream_t stream) noexcept {
    if (args.tokens >= 9U) { return w8g32_linear_prefill(args, stream); }
    // W8's lighter base load shifts the crossover by one decode row.
    return args.tokens <= 4U ? w8g32_linear_baseline(args, stream)
                              : w8g32_linear_wave8(args, stream);
}

hipError_t bf16_linear_prefill(const Bf16LinearArgs& args, hipStream_t stream) noexcept {
    // Fixed prefill dispatch selected only after gfx1201 qualification. The
    // wave32 BF16xBF16->FP32 WMMA route retains represented BF16 operands.
    return bf16_linear_wmma16(args, stream);
}

hipError_t w8g32_linear_prefill(const W8G32LinearArgs& args, hipStream_t stream) noexcept {
    // The W8G32 scale is per K32 code group, so no gfx1201 WMMA operand form
    // preserves the codec exactly. Sixteen wave32 consumers reuse each decoded
    // 256-K tile while keeping the mandated FP32 dequantized FMA path.
    return w8g32_linear_wave16(args, stream);
}

} // namespace ninfer::ops::r9700::linear
