#pragma once

#include "tq_k4v3_codec.h"

#include <hip/hip_fp16.h>

#include <cstdint>

// Device decode helpers of the TQ-K4-V3 codec: byte-permute tables of the binary16 high bytes.
namespace ninfer::ops::r9700::kv::tq {

// Four K nibbles (one per byte lane, sign in bit 3) to their binary16 high bytes.
__device__ __forceinline__ std::uint32_t key_f16_high_bytes(std::uint32_t codes) {
    return __builtin_amdgcn_perm(kKeyTableHigh, kKeyTableLow, codes & 0x07070707U) |
           ((codes & 0x08080808U) << 4U);
}

// Four V magnitude indices (bits 0-1 of each byte lane) and sign bits (bit 0 of each lane).
__device__ __forceinline__ std::uint32_t
value_f16_high_bytes(std::uint32_t magnitudes, std::uint32_t signs, std::uint32_t table) {
    return __builtin_amdgcn_perm(0U, table, magnitudes & 0x03030303U) |
           ((signs & 0x01010101U) << 7U);
}

__device__ __forceinline__ float high_byte_float(std::uint32_t high_byte) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>((high_byte & 0xffU) << 8U)));
}

// Signed level of one K nibble, and of one V magnitude index and sign.
__device__ __forceinline__ float key_level(std::uint32_t code) {
    return high_byte_float(key_f16_high_bytes(code & 0x0fU));
}

__device__ __forceinline__ float value_level(std::uint32_t magnitude, std::uint32_t sign) {
    return high_byte_float(value_f16_high_bytes(magnitude, sign, kValueTable));
}

// Unnormalized Sylvester FWHT of one 256-feature vector held by a whole wave32, lane l owning
// features 8l..8l+7 in x[0..7]. Stages 1, 2 and 4 pair registers within a lane; stages 8..128 pair
// the same register across lanes l and l ^ (h / 8). Every lane of the wave must call it.
__device__ __forceinline__ void wave_fwht(float (&x)[8], std::uint32_t lane) {
#pragma unroll
    for (std::uint32_t h = 1U; h < 8U; h <<= 1U) {
#pragma unroll
        for (std::uint32_t i = 0U; i < 8U; ++i) {
            if ((i & h) != 0U) continue;
            const float a = x[i], b = x[i + h];
            x[i]     = a + b;
            x[i + h] = a - b;
        }
    }
#pragma unroll
    for (std::uint32_t m = 1U; m < 32U; m <<= 1U) {
#pragma unroll
        for (std::uint32_t i = 0U; i < 8U; ++i) {
            const float partner = __shfl_xor(x[i], static_cast<int>(m), 32);
            x[i]                = (lane & m) != 0U ? partner - x[i] : x[i] + partner;
        }
    }
}

// y = H diag(s) x / 16 of the wave's vector (see wave_fwht for the register layout).
__device__ __forceinline__ void rotate_wave(float (&x)[8], std::uint32_t lane) {
#pragma unroll
    for (std::uint32_t i = 0U; i < 8U; ++i) {
        if (negative_sign(lane * 8U + i)) x[i] = -x[i];
    }
    wave_fwht(x, lane);
#pragma unroll
    for (std::uint32_t i = 0U; i < 8U; ++i) x[i] *= 0.0625F;
}

// x = diag(s) H y / 16; a +0 or -0 vector becomes +0.
__device__ __forceinline__ void unrotate_wave(float (&x)[8], std::uint32_t lane) {
    wave_fwht(x, lane);
#pragma unroll
    for (std::uint32_t i = 0U; i < 8U; ++i) {
        const float value = x[i] * 0.0625F;
        x[i]              = (negative_sign(lane * 8U + i) ? -value : value) + 0.0F;
    }
}

} // namespace ninfer::ops::r9700::kv::tq
