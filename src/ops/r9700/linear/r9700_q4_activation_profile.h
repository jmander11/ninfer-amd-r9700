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

inline constexpr std::string_view kQ4PrefillCtaProfile =
    "m64n128-pingpong-n16-k16-production";

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
