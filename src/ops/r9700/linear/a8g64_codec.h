#pragma once

// Device helpers of the exact A8G64 activation codec shared by every small-T producer: eight
// adjacent lanes own one G64 group (one 16-byte BF16 vector each), codes are the rounded quotient
// by the represented FP16 scale max/127, split into signed low/high nibble planes.

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

} // namespace ninfer::ops::r9700::linear::codec
