#pragma once

#include <cstdint>

namespace ninfer::ops::r9700::linear {

// Compile-time A/B boundary over identical persistent W8G32 artifact bytes. The selected A8
// profile still keeps exact BF16xW8 execution below each physically measured mixed-artifact
// shape crossover. Unknown/all-W8-only shapes remain exact rather than extrapolating a synthetic
// threshold. An explicit A16 build is the matched quality/performance control.
#ifndef NINFER_R9700_W8_ACTIVATION_BITS
#define NINFER_R9700_W8_ACTIVATION_BITS 8
#endif
static_assert(NINFER_R9700_W8_ACTIVATION_BITS == 8 ||
                  NINFER_R9700_W8_ACTIVATION_BITS == 16,
              "R9700 W8 activation width must be A8 or represented BF16");
inline constexpr std::uint32_t kW8ActivationBits = NINFER_R9700_W8_ACTIVATION_BITS;
inline constexpr std::uint32_t kA8W8MinimumWinningTokens = 3;

[[nodiscard]] constexpr std::uint32_t a8w8_first_winning_tokens(
    std::uint32_t rows, std::uint32_t columns) noexcept {
    if (rows == 1152U &&
        (columns == 1152U || columns == 1536U || columns == 4304U)) return 64U;
    if ((rows == 4608U && columns == 4608U) ||
        (rows == 5120U && (columns == 4608U || columns == 6144U ||
                           columns == 10240U)) ||
        (rows == 248320U && columns == 5120U)) return 4U;
    if ((rows == 7168U || rows == 12288U || rows == 14336U) && columns == 5120U)
        return 3U;
    if (rows == 34816U && columns == 5120U) return 32U;
    if (rows == 5120U && columns == 17408U) return 3U;
    return 0xffffffffU;
}

// Workspace planning does not know N, so it reserves from the lowest admitted crossover.
[[nodiscard]] constexpr bool may_use_a8w8(std::uint32_t tokens) noexcept {
    return kW8ActivationBits == 8 && tokens >= kA8W8MinimumWinningTokens;
}

[[nodiscard]] constexpr bool use_a8w8(std::uint32_t tokens, std::uint32_t rows,
                                      std::uint32_t columns) noexcept {
    return kW8ActivationBits == 8 &&
           tokens >= a8w8_first_winning_tokens(rows, columns);
}

// Production cooperative-CTA admission is intentionally narrower than the A8
// activation crossover above: only the four physically qualified mixed-Text
// tuples at the four measured prefill extents use it.
[[nodiscard]] constexpr bool use_a8w8_prefill_cta(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns) noexcept {
    const bool qualified_tokens =
        tokens == 1024U || tokens == 2048U || tokens == 4096U || tokens == 8192U;
    const bool qualified_shape =
        ((rows == 7168U || rows == 12288U) && columns == 5120U) ||
        (rows == 5120U && (columns == 6144U || columns == 17408U));
    return kW8ActivationBits == 8U && qualified_tokens && qualified_shape;
}

static_assert(a8w8_first_winning_tokens(7168U, 5120U) == 3U);
static_assert(a8w8_first_winning_tokens(1152U, 4304U) == 64U);
static_assert(a8w8_first_winning_tokens(1U, 128U) == 0xffffffffU);

} // namespace ninfer::ops::r9700::linear
