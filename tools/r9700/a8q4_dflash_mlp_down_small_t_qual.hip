#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace mlp_down_qual {

namespace linear = ninfer::ops::r9700::linear;
constexpr std::uint32_t kRows = 5120U;
constexpr std::uint32_t kColumns = 17408U;
constexpr std::uint32_t kGroup = 64U;
constexpr std::uint32_t kThreads = 256U;

bool interval(const void* pointer, std::size_t bytes, std::uintptr_t* begin,
              std::uintptr_t* end) noexcept {
    if (pointer == nullptr || bytes == 0U) return false;
    *begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (*begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    *end = *begin + bytes;
    return true;
}

bool overlap(const void* lhs, std::size_t lhs_bytes, const void* rhs,
             std::size_t rhs_bytes) noexcept {
    std::uintptr_t lb = 0, le = 0, rb = 0, re = 0;
    return !interval(lhs, lhs_bytes, &lb, &le) || !interval(rhs, rhs_bytes, &rb, &re) ||
           (lb < re && rb < le);
}

bool valid(const linear::A8Q4G64LinearArgs& args) noexcept {
    if (args.tokens != 5U || args.rows != kRows || args.columns != kColumns ||
        args.padded_columns != kColumns ||
        args.activation_low_codes == nullptr || args.activation_high_codes == nullptr ||
        args.activation_scales == nullptr || args.activation_status == nullptr ||
        args.weight_codes == nullptr || args.weight_scales == nullptr || args.output == nullptr ||
        (reinterpret_cast<std::uintptr_t>(args.activation_low_codes) & 3U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(args.activation_high_codes) & 3U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(args.weight_codes) & 7U) != 0U) return false;
    const std::size_t groups = kColumns / kGroup;
    const std::size_t activation_codes =
        static_cast<std::size_t>(args.tokens) * kColumns / 2U;
    const std::size_t activation_scales =
        static_cast<std::size_t>(args.tokens) * groups * sizeof(std::uint16_t);
    const std::size_t weight_codes = static_cast<std::size_t>(kRows) * kColumns / 2U;
    const std::size_t weight_scales =
        static_cast<std::size_t>(kRows) * groups * sizeof(std::uint16_t);
    if (args.activation_low_code_bytes != activation_codes ||
        args.activation_high_code_bytes != activation_codes ||
        args.activation_scale_bytes != activation_scales ||
        args.weight_code_bytes != weight_codes || args.weight_scale_bytes != weight_scales)
        return false;
    const std::size_t output_bytes =
        static_cast<std::size_t>(args.tokens) * kRows * sizeof(hip_bfloat16);
    const void* pointers[]{args.activation_low_codes, args.activation_high_codes,
                           args.activation_scales, args.activation_status,
                           args.weight_codes, args.weight_scales};
    const std::size_t sizes[]{activation_codes, activation_codes, activation_scales,
                              sizeof(std::uint32_t), weight_codes, weight_scales};
    for (unsigned i = 0; i < 6U; ++i) {
        if (overlap(args.output, output_bytes, pointers[i], sizes[i])) return false;
        for (unsigned j = i + 1U; j < 6U; ++j)
            if (overlap(pointers[i], sizes[i], pointers[j], sizes[j])) return false;
    }
    return true;
}

__device__ __forceinline__ int activation_x8(const std::uint8_t* codes,
                                              std::size_t element) {
    return *reinterpret_cast<const int*>(codes + element / 2U);
}

__device__ __forceinline__ int weight_x8(const std::uint8_t* codes, std::uint32_t row,
                                         std::uint32_t group, std::uint32_t word) {
    constexpr std::uint32_t groups = kColumns / kGroup;
    const std::size_t pair_index =
        (((static_cast<std::size_t>(row >> 4U) * groups + group) * 4U +
          (word >> 1U)) * 16U + (row & 15U));
    const std::uint64_t pair = reinterpret_cast<const std::uint64_t*>(codes)[pair_index];
    return static_cast<int>((word & 1U) != 0U ? pair >> 32U : pair);
}

template <std::uint32_t Tokens>
__global__ __launch_bounds__(kThreads) void candidate_kernel(
    const std::uint8_t* low_codes, const std::uint8_t* high_codes,
    const std::uint16_t* activation_scales, const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* output) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    static_assert(Tokens == 5U);
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= kRows) return;
    if (*activation_status != linear::Q4G64ActivationOk) {
        hip_bfloat16 poison;
        poison.data = 0x7fc1U;
#pragma unroll
        for (std::uint32_t token = 0; token < Tokens; ++token)
            output[static_cast<std::size_t>(token) * kRows + row] = poison;
        return;
    }
    constexpr std::uint32_t groups = kColumns / kGroup;
    float totals[Tokens]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        int low_dots[Tokens]{};
        int high_dots[Tokens]{};
#pragma unroll
        for (std::uint32_t word = 0; word < 8U; ++word) {
            const int weight = weight_x8(weight_codes, row, group, word);
#pragma unroll
            for (std::uint32_t token = 0; token < Tokens; ++token) {
                const std::size_t element = static_cast<std::size_t>(token) * kColumns +
                    static_cast<std::size_t>(group) * kGroup + word * 8U;
                low_dots[token] = __builtin_amdgcn_sudot8(
                    false, activation_x8(low_codes, element), true, weight,
                    low_dots[token], false);
                high_dots[token] = __builtin_amdgcn_sudot8(
                    true, activation_x8(high_codes, element), true, weight,
                    high_dots[token], false);
            }
        }
        const std::size_t scale_index =
            (static_cast<std::size_t>(row >> 4U) * groups + group) * 16U +
            (row & 15U);
        const float weight_scale = __half2float(__ushort_as_half(weight_scales[scale_index]));
#pragma unroll
        for (std::uint32_t token = 0; token < Tokens; ++token) {
            const float activation_scale = __half2float(__ushort_as_half(
                activation_scales[static_cast<std::size_t>(token) * groups + group]));
            totals[token] = fmaf(static_cast<float>(low_dots[token] + 16 * high_dots[token]),
                                 activation_scale * weight_scale, totals[token]);
        }
    }
#pragma unroll
    for (std::uint32_t token = 0; token < Tokens; ++token)
        output[static_cast<std::size_t>(token) * kRows + row] =
            static_cast<hip_bfloat16>(totals[token]);
#else
    (void)low_codes; (void)high_codes; (void)activation_scales;
    (void)activation_status; (void)weight_codes; (void)weight_scales; (void)output;
#endif
}

hipError_t launch(const linear::A8Q4G64LinearArgs& args) noexcept {
    if (!valid(args)) return hipErrorInvalidValue;
    const dim3 block(kThreads);
    const dim3 grid((kRows + kThreads - 1U) / kThreads);
    hipLaunchKernelGGL((candidate_kernel<5U>), grid, block, 0, nullptr,
                       args.activation_low_codes, args.activation_high_codes,
                       args.activation_scales, args.activation_status, args.weight_codes,
                       args.weight_scales, args.output);
    return hipGetLastError();
}

} // namespace mlp_down_qual

#define NINFER_DFLASH_SMALL_T_ROWS 5120U
#define NINFER_DFLASH_SMALL_T_COLUMNS 17408U
// The identical T6 expansion reports eight SGPR spills on the selected gfx1201
// compiler and is therefore excluded before physical timing.
#define NINFER_DFLASH_SMALL_T_WIDTHS 5U
#define NINFER_DFLASH_MLP_DOWN_T5_ONLY 1
#define NINFER_DFLASH_SMALL_T_CALL(args) mlp_down_qual::launch(args)
#define NINFER_DFLASH_SMALL_T_SCOPE \
    "standalone Q4 MLP-down N5120/K17408 at exact DFlash T5 only; no production routing"
#define NINFER_DFLASH_SMALL_T_SCHEMA \
    "ninfer.r9700.a8q4-dflash-mlp-down-small-t-cell.v1"
#define NINFER_DFLASH_SMALL_T_LABEL "MLP-down small-T"
#define NINFER_DFLASH_SMALL_T_QUALIFIER_SOURCE \
    "tools/r9700/a8q4_dflash_mlp_down_small_t_qual.hip"
#define NINFER_DFLASH_SMALL_T_SHARED_HARNESS_SOURCE \
    "tools/r9700/a8q4_dflash_small_t_qual.hip"
#include "a8q4_dflash_small_t_qual.hip"
