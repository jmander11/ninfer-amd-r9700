#pragma once

#include <cstdint>

// TQ-K4-V3 growing-cache codec (TurboQuant MSE variant with norm correction), shared by the append,
// rotation and attention Ops, their oracles and qualifiers.
//
// Rotation. Every 256-feature K, V and Q head vector x is represented in the rotated basis
// y = R x, R = H diag(s) / 16, where H is the Sylvester-Hadamard matrix (H_ij = (-1)^popcount(i&j))
// and s_i = -1 exactly when bit i of kTqSignMask is set (s_i = +1 when splitmix64(i) is odd). R is
// orthonormal, so Q.K = (RQ).(RK); attention over rotated V yields R O, and O = R^T (R O) with
// R^T = diag(s) H / 16.
//
// Codes. With n = ||x|| (FP32), each rotated coordinate's level-domain magnitude
// r = |y_i| * 16 * alpha / n selects the first magnitude index whose midpoint threshold it does not
// exceed (r > threshold moves up; equality stays down), and the sign bit records y_i < 0. K uses
// four bits per feature (sign in bit 3, magnitude index in bits 0-2, alpha 3/4); V uses three
// (sign plus a 2-bit magnitude index, alpha 24/25). The stored binary16 scale is
// FP16_rn(n / ||c||), where c is the signed level vector; the represented rotated coordinate is
// scale * c_i. A zero vector stores scale 0 and zero codes. Every level, and every V level / 8, is
// exact in the high byte of binary16, so attention decodes codes with one byte-permute table.
//
// Storage per head vector: K 128 bytes, low nibble = even feature; V 96 bytes as eight 12-byte
// chunks of 32 features: word A0 (features 0-15), word A1 (features 16-31), word B (signs). Byte
// lane j of A_h holds feature 16h + 4k + j in bits 2k..2k+1; byte lane j of B holds feature
// 4b + j in bit b. One binary16 scale each for K and V.
namespace ninfer::ops::r9700::kv::tq {

inline constexpr std::uint32_t kHeadDim     = 256;
inline constexpr std::uint32_t kKeyBytes    = 128;
inline constexpr std::uint32_t kValueBytes  = 96;
inline constexpr std::uint32_t kValueChunk  = 12; // bytes per 32 features
inline constexpr std::uint32_t kSignMask[8] = {0xFDDC4774U, 0x4CBA29C2U, 0x8EB45918U, 0xA72B3DD0U,
                                               0x6B6924E9U, 0x1714BC1BU, 0x91236E45U, 0x974B2898U};
inline constexpr float kKeyAlpha            = 0.75F;
inline constexpr float kValueAlpha          = 0.96F;
inline constexpr float kKeyLevels[8] = {0.09375F, 0.3125F, 0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 2.0F};
inline constexpr float kKeyThresholds[7]   = {0.203125F, 0.40625F, 0.625F, 0.875F,
                                              1.125F,    1.375F,   1.75F};
inline constexpr float kValueLevels[4]     = {0.25F, 0.75F, 1.25F, 2.0F};
inline constexpr float kValueThresholds[3] = {0.5F, 1.0F, 1.625F};
// High bytes of the binary16 levels for v_perm (selectors 0-3 read the low word, 4-7 the high).
inline constexpr std::uint32_t kKeyTableLow  = 0x3A38352EU;
inline constexpr std::uint32_t kKeyTableHigh = 0x403E3D3CU;
inline constexpr std::uint32_t kValueTable   = 0x403D3A34U; // levels
inline constexpr std::uint32_t kValueOver8   = 0x34312E28U; // levels / 8

[[nodiscard]] constexpr bool negative_sign(std::uint32_t feature) noexcept {
    return ((kSignMask[feature / 32U] >> (feature % 32U)) & 1U) != 0U;
}

// Byte offset and bit position of feature f's V magnitude and sign within its 96-byte vector.
[[nodiscard]] constexpr std::uint32_t value_magnitude_byte(std::uint32_t f) noexcept {
    return (f / 32U) * kValueChunk + ((f % 32U) / 16U) * 4U + (f % 4U);
}

[[nodiscard]] constexpr std::uint32_t value_magnitude_shift(std::uint32_t f) noexcept {
    return 2U * ((f % 16U) / 4U);
}

[[nodiscard]] constexpr std::uint32_t value_sign_byte(std::uint32_t f) noexcept {
    return (f / 32U) * kValueChunk + 8U + (f % 4U);
}

[[nodiscard]] constexpr std::uint32_t value_sign_shift(std::uint32_t f) noexcept {
    return (f % 32U) / 4U;
}

} // namespace ninfer::ops::r9700::kv::tq
