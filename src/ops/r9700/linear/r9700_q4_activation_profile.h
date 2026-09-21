#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::ops::r9700::linear {

// Compile-time execution boundary for the two qualified activation codecs over identical
// persistent Q4G64 artifact bytes. A8 is the production-style default; A4 is retained only as an
// explicit evaluator. This is deliberately not a runtime or artifact selector.
#ifndef NINFER_R9700_Q4_ACTIVATION_BITS
#define NINFER_R9700_Q4_ACTIVATION_BITS 8
#endif
static_assert(NINFER_R9700_Q4_ACTIVATION_BITS == 4 ||
                  NINFER_R9700_Q4_ACTIVATION_BITS == 8,
              "R9700 Q4 activation width must be A4 or A8");
inline constexpr std::uint32_t kQ4ActivationBits = NINFER_R9700_Q4_ACTIVATION_BITS;

#ifndef NINFER_R9700_DFLASH_SMALL_T_CANDIDATE
#define NINFER_R9700_DFLASH_SMALL_T_CANDIDATE 0
#endif
static_assert(NINFER_R9700_DFLASH_SMALL_T_CANDIDATE == 0 ||
                  NINFER_R9700_DFLASH_SMALL_T_CANDIDATE == 1,
              "R9700 DFlash small-token candidate selector must be zero or one");
inline constexpr bool kDFlashSmallTCandidateEnabled =
    NINFER_R9700_DFLASH_SMALL_T_CANDIDATE == 1;

#ifndef NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE
#define NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE 0
#endif
static_assert(NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE == 0 ||
                  NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE == 1,
              "R9700 DFlash MLP-down T5 candidate selector must be zero or one");
inline constexpr bool kDFlashMlpDownT5CandidateEnabled =
    NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE == 1;

[[nodiscard]] constexpr bool is_a8q4_dflash_mlp_down_t5_eligible(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    return tokens == 5U && rows == 5120U && columns == 17408U &&
           padded_columns == columns;
}

[[nodiscard]] constexpr bool use_a8q4_dflash_mlp_down_t5(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    return kDFlashMlpDownT5CandidateEnabled && kQ4ActivationBits == 8U &&
           is_a8q4_dflash_mlp_down_t5_eligible(tokens, rows, columns, padded_columns);
}

inline constexpr std::string_view kQ4PrefillCtaProfile =
    "m64n128-pingpong-n16-k16-scalar-base-production";

// The one-row-per-thread native-dot8 route is selected only for the exact full-K
// Q4 tuples qualified at T=1. Two tuples are also used by DFlash2; selection follows
// the closed Linear shape contract rather than the model caller.
[[nodiscard]] constexpr bool use_a8q4_decode_dot8_t1(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    return kQ4ActivationBits == 8U && tokens == 1U && columns == padded_columns &&
        ((rows == 4096U && columns == 5120U) ||
         (rows == 5120U && columns == 6144U) ||
         (rows == 5120U && columns == 17408U) ||
         (rows == 7168U && columns == 5120U) ||
         (rows == 12288U && columns == 5120U) ||
         (rows == 34816U && columns == 5120U) ||
         (rows == 248320U && columns == 5120U));
}

// Exact direct-screen winners only. Keeping the evidence-selected domain separate from the build
// selector lets host tests prove both matched A/B profiles without adding a runtime mode.
[[nodiscard]] constexpr bool is_a8q4_dflash_small_t_eligible(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    const bool eligible_width =
        tokens == 4U || tokens == 5U || tokens == 6U || tokens == 8U || tokens == 10U ||
        tokens == 12U || tokens == 18U || tokens == 20U;
    return eligible_width && rows == 34816U && columns == 5120U &&
           padded_columns == columns;
}

[[nodiscard]] constexpr bool use_a8q4_dflash_small_t(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    return kDFlashSmallTCandidateEnabled && kQ4ActivationBits == 8U &&
           is_a8q4_dflash_small_t_eligible(tokens, rows, columns, padded_columns);
}

enum class A8Q4PrefillRoute : std::uint8_t {
    Wmma32,
    M64N128PingPongProduction,
};

// The cooperative prefill CTA is admitted only for the exact Text and bulk-MTP
// matrix tuples and token extents covered by the retained R9700 qualification.
// Shape matching keeps every decode, tail chunk, Vision, and DFlash leaf on its
// existing route without adding a runtime selector or repacking weights.
[[nodiscard]] constexpr bool use_a8q4_prefill_cta(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns) noexcept {
    const bool qualified_tokens =
        tokens == 1024U || tokens == 2048U || tokens == 4096U || tokens == 8192U;
    const bool qualified_shape =
        ((rows == 7168U || rows == 4096U || rows == 12288U || rows == 34816U ||
          rows == 1024U) &&
         columns == 5120U) ||
        (rows == 5120U &&
         (columns == 6144U || columns == 10240U || columns == 17408U));
    return kQ4ActivationBits == 8U && qualified_tokens && qualified_shape;
}

[[nodiscard]] constexpr A8Q4PrefillRoute select_a8q4_prefill_route(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns) noexcept {
    if (!use_a8q4_prefill_cta(tokens, rows, columns)) return A8Q4PrefillRoute::Wmma32;
    return A8Q4PrefillRoute::M64N128PingPongProduction;
}

} // namespace ninfer::ops::r9700::linear
