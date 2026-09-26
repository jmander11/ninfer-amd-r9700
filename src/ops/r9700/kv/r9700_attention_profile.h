#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::kv {

// Compile-time numerical/performance profile of the long-context split-512 leaf for tree or
// device-selected T=4 schedules. Zero is the separately built PPL control profile and retains the
// fused represented-BF16-Q leaf there.
// This is not an artifact, CLI, or runtime selector.
#ifndef NINFER_R9700_FP8_QK_WMMA
#define NINFER_R9700_FP8_QK_WMMA 1
#endif
static_assert(NINFER_R9700_FP8_QK_WMMA == 0 || NINFER_R9700_FP8_QK_WMMA == 1,
              "R9700 FP8-Q/K WMMA profile must be zero or one");
inline constexpr bool kFp8QkWmmaDecode = NINFER_R9700_FP8_QK_WMMA != 0;
inline constexpr std::uint32_t kPackedDecodeMinimumContext = 64U;
inline constexpr std::size_t kPackedDecodeMaximumContext = 262144U;
inline constexpr std::uint32_t kSplit512MinimumContext = 8192U;
inline constexpr std::uint32_t kDensePrefillMinimumRows = 128U;
inline constexpr std::uint32_t kDensePrefillMaximumRows = 8192U;
inline constexpr std::size_t kDensePrefillMaximumContext = 262144U;

// Qualification-only Text P129 WMMA tail candidate. Zero retains the dense P129 route; one admits
// the exact P129 tail overwrite. The packed decode route is a production route and is not gated by
// this flag.
#ifndef NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE
#define NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE 0
#endif
static_assert(NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 0 ||
                  NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 1,
              "R9700 Text P129 WMMA tail candidate must be zero or one");
inline constexpr bool kTextP129WmmaTailCandidate =
    NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 1;

// Causal prefill covers initial and appended chunks with absolute positions and the complete
// visible cache frontier.
[[nodiscard]] constexpr bool use_dense_prefill_attention(
    std::uint32_t query_rows, std::size_t visible_context) noexcept {
    return query_rows >= kDensePrefillMinimumRows &&
           query_rows <= kDensePrefillMaximumRows && visible_context >= query_rows &&
           visible_context <= kDensePrefillMaximumContext;
}

// Production packed decode route for 1..6 causal rows per sequence (ordinary decode, MTP and
// DFlash chain verification): non-tree execution with host-fixed row counts and context
// 64..262144. The caller must additionally require G16, token-fastest FP8 keys, and
// feature-fastest INT4/FP16 values and scales; other cells retain the split/fused fallback.
[[nodiscard]] constexpr bool use_packed_decode_attention(
    std::uint32_t query_rows, std::size_t visible_context, bool tree_or_device_count) noexcept {
    return (query_rows >= 1U && query_rows <= 6U) && !tree_or_device_count &&
           visible_context >= kPackedDecodeMinimumContext &&
           visible_context <= kPackedDecodeMaximumContext;
}

[[nodiscard]] constexpr bool use_text_p129_wmma_tail(std::uint32_t query_rows,
                                                       std::size_t visible_context) noexcept {
    return kTextP129WmmaTailCandidate && query_rows == 129U && visible_context == 129U;
}

// Long-context T=4 schedules with tree or device-selected row metadata, which the packed decode
// route does not serve; host-fixed rows take the packed route.
[[nodiscard]] constexpr bool use_split512_attention(std::uint32_t query_rows,
                                                     std::size_t visible_context,
                                                     bool tree_or_device_count) noexcept {
    return kFp8QkWmmaDecode && visible_context >= kSplit512MinimumContext && query_rows == 4U &&
           tree_or_device_count;
}

} // namespace ninfer::ops::r9700::kv
