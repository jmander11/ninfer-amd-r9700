#pragma once

// Device helpers of the exact A8G64 activation codec shared by every small-T producer: eight
// adjacent lanes own one G64 group (one 16-byte BF16 vector each), codes are the rounded quotient
// by the represented FP16 scale max/127, split into signed low/high nibble planes.

#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear::codec {

__device__ __forceinline__ float bf16_word_element(const std::uint32_t (&words)[4],
                                                   std::uint32_t element) {
    const std::uint32_t word = words[element >> 1U];
    return __uint_as_float((element & 1U) ? word & 0xffff0000U : word << 16U);
}

// Maximum over the eight lanes of one G64 group (a DPP row reduction).
__device__ __forceinline__ float a8g64_group_maximum(float maximum) {
    maximum = fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(maximum),
                                                                        0x164, 0xf, 0xf, false)));
    maximum = fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(maximum),
                                                                        0x162, 0xf, 0xf, false)));
    return fmaxf(maximum, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(maximum),
                                                                     0x161, 0xf, 0xf, false)));
}

// The G64 scale overflows exactly when some finite |x| of the group does: max/127 and the FP16
// conversion are monotone, so status passes need no group reduction. FP16(|x|/127) is infinite
// exactly when |x| >= 8321040 (checked over every finite FP32 value).
__device__ __forceinline__ bool a8g64_element_overflows(float x) { return fabsf(x) >= 8321040.0F; }

// Encodes the eight finite values of one vector with its group's represented scale. Only lanes
// with (lane & 7) == 0 write the scale word.
__device__ __forceinline__ void
a8g64_encode_vector(const float (&value)[8], float maximum, std::uint32_t lane, std::size_t vector,
                    std::size_t scale_index, std::uint8_t* low_codes, std::uint8_t* high_codes,
                    std::uint16_t* scale_words) {
    std::uint16_t scale_bits = 0U;
    if (maximum != 0.0F) {
        scale_bits = __half_as_ushort(__float2half_rn(maximum / 127.0F));
        if (scale_bits == 0U) scale_bits = 1U;
        if (!isfinite(__half2float(__ushort_as_half(scale_bits)))) scale_bits = 0U;
    }
    if ((lane & 7U) == 0U) scale_words[scale_index] = scale_bits;
    const float scale      = __half2float(__ushort_as_half(scale_bits));
    std::uint32_t low_word = 0U, high_word = 0U;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const int code =
            scale != 0.0F ? max(-127, min(127, __float2int_rn(value[element] / scale))) : 0;
        const int low  = code & 0x0f;
        const int high = (code - low) / 16;
        low_word |= static_cast<std::uint32_t>(low) << (element * 4U);
        high_word |= static_cast<std::uint32_t>(high & 0x0f) << (element * 4U);
    }
    reinterpret_cast<std::uint32_t*>(low_codes)[vector]  = low_word;
    reinterpret_cast<std::uint32_t*>(high_codes)[vector] = high_word;
}

// Codec status conditions of eight represented values: a nonfinite value, or a finite value
// whose group scale overflows FP16.
__device__ __forceinline__ void a8g64_value_status(const float (&value)[8], bool& nonfinite,
                                                   bool& overflow) {
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        nonfinite |= !isfinite(value[element]);
        overflow |= isfinite(value[element]) && a8g64_element_overflows(value[element]);
    }
}

// Encodes the eight represented values of one vector (nonfinite values encode as zero, the
// status word reports them); every lane of the group must call it.
__device__ __forceinline__ void
a8g64_encode_represented(const float (&represented)[8], std::uint32_t lane, std::size_t vector,
                         std::size_t scale_index, std::uint8_t* low_codes, std::uint8_t* high_codes,
                         std::uint16_t* scale_words) {
    float value[8];
    float maximum = 0.0F;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        value[element] = isfinite(represented[element]) ? represented[element] : 0.0F;
        maximum        = fmaxf(maximum, fabsf(value[element]));
    }
    a8g64_encode_vector(value, a8g64_group_maximum(maximum), lane, vector, scale_index, low_codes,
                        high_codes, scale_words);
}

// Publishes the status word of a launch without a reset launch: each block ORs its conditions into
// completion[0] and counts itself in completion[1]; the last block to finish stores the combined
// status word and returns both completion words to zero for the next launch. The words must be zero
// before the first launch and serve one launch at a time. Every thread of every block calls it.
__device__ __forceinline__ void a8g64_complete_status(bool nonfinite, bool overflow,
                                                      std::uint32_t* status,
                                                      std::uint32_t* completion) {
    const bool any_nonfinite = __syncthreads_or(nonfinite);
    const bool any_overflow  = __syncthreads_or(overflow);
    if (threadIdx.x != 0U || threadIdx.y != 0U || threadIdx.z != 0U) return;
    const std::uint32_t flags =
        (any_nonfinite ? static_cast<std::uint32_t>(Q4G64ActivationNonfinite) : 0U) |
        (any_overflow ? static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow) : 0U);
    if (flags != 0U) atomicOr(&completion[0], flags);
    __threadfence();
    const std::uint32_t blocks = gridDim.x * gridDim.y * gridDim.z;
    if (atomicAdd(&completion[1], 1U) != blocks - 1U) return;
    __threadfence();
    *status = atomicExch(&completion[0], 0U);
    atomicExch(&completion[1], 0U);
}

// Accumulates this thread's conditions into a status word the launcher reset beforehand.
__device__ __forceinline__ void a8g64_accumulate_status(bool nonfinite, bool overflow,
                                                        std::uint32_t* status) {
    if (nonfinite) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationNonfinite));
    if (overflow) atomicOr(status, static_cast<std::uint32_t>(Q4G64ActivationScaleOverflow));
}

} // namespace ninfer::ops::r9700::linear::codec
