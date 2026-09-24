#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::ops::r9700::linear {

// Compile-time execution boundary for the two qualified activation codecs over identical
// persistent Q4G64 artifact bytes. The selected default is A8 with gate/up-only A4
// for T>128. Other profiles are evaluators, not runtime or artifact selectors.
#ifndef NINFER_R9700_Q4_ACTIVATION_BITS
#define NINFER_R9700_Q4_ACTIVATION_BITS 8
#endif
static_assert(NINFER_R9700_Q4_ACTIVATION_BITS == 4 ||
                  NINFER_R9700_Q4_ACTIVATION_BITS == 8,
              "R9700 Q4 activation width must be A4 or A8");
inline constexpr std::uint32_t kQ4ActivationBits = NINFER_R9700_Q4_ACTIVATION_BITS;

#ifndef NINFER_R9700_Q4_PREFILL_A4_FAMILIES
#define NINFER_R9700_Q4_PREFILL_A4_FAMILIES 1
#endif
inline constexpr unsigned kQ4PrefillA4Families = NINFER_R9700_Q4_PREFILL_A4_FAMILIES;
static_assert(kQ4PrefillA4Families <= 5U);
inline constexpr bool kQ4PrefillGateUpA4 = kQ4PrefillA4Families != 0U && kQ4PrefillA4Families != 4U;
static_assert(kQ4PrefillA4Families == 0U || kQ4ActivationBits == 8U,
              "mixed prefill policy requires global Q4 A8");

[[nodiscard]] constexpr bool is_q4_prefill_gate_up_a4_eligible(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    return tokens > 128U && rows == 34816U && columns == 5120U &&
           padded_columns == columns;
}

[[nodiscard]] constexpr bool is_q4_prefill_a4_eligible(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    if (tokens <= 128U || columns != padded_columns) return false;
    if (kQ4PrefillGateUpA4 && rows == 34816U && columns == 5120U) return true;
    if ((kQ4PrefillA4Families == 2U || kQ4PrefillA4Families == 3U) &&
        rows == 5120U && columns == 17408U) return true;
    if (kQ4PrefillA4Families >= 3U && rows == 7168U && columns == 5120U) return true;
    return kQ4PrefillA4Families == 3U &&
        (((rows == 4096U || rows == 12288U) && columns == 5120U) ||
         (rows == 5120U && columns == 6144U));
}

// Only the public Q4 Linear boundary applies this activation override. Private
// explicitly-A8 Ops, decode/verify widths and all other formats remain unchanged.
[[nodiscard]] constexpr std::uint32_t q4_linear_activation_bits(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns) noexcept {
    return is_q4_prefill_a4_eligible(tokens, rows, columns, padded_columns)
        ? 4U : kQ4ActivationBits;
}

inline constexpr std::string_view kQ4ActivationProfile = kQ4PrefillA4Families == 5U
    ? "a8-except-gate-up-attn-input-tgt128-a4"
    : kQ4PrefillA4Families == 4U ? "a8-except-n7168-k5120-tgt128-a4"
    : kQ4PrefillA4Families == 3U
    ? "a8-except-text-projections-tgt128-a4"
    : kQ4PrefillA4Families == 2U ? "a8-except-mlp-tgt128-a4"
    : kQ4PrefillGateUpA4
    ? "a8-except-n34816-k5120-tgt128-a4"
    : kQ4ActivationBits == 8U ? "uniform-a8" : "uniform-a4";

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
