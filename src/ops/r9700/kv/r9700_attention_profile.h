#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::kv {

// Compile-time numerical/performance profile for the admitted decode-attention routes. A value
// of one permits the short-context FP8-Q/FP8-K WMMA crossover and the production split-512
// long-context leaf at their qualified row/context boundaries. Zero is the separately built PPL
// control profile and retains represented-BF16-Q score streaming.
// This is not an artifact, CLI, or runtime selector.
#ifndef NINFER_R9700_FP8_QK_WMMA
#define NINFER_R9700_FP8_QK_WMMA 1
#endif
static_assert(NINFER_R9700_FP8_QK_WMMA == 0 || NINFER_R9700_FP8_QK_WMMA == 1,
              "R9700 FP8-Q/K WMMA profile must be zero or one");
inline constexpr bool kFp8QkWmmaDecode = NINFER_R9700_FP8_QK_WMMA != 0;
inline constexpr std::uint32_t kFp8QkWmmaT1MinimumContext = 64U;
inline constexpr std::uint32_t kFp8QkWmmaT2MinimumContext = 320U;
inline constexpr std::uint32_t kSplit512MinimumContext = 8192U;
inline constexpr std::size_t kDflashVerifyWmmaMaximumContext = 262144U;
inline constexpr std::uint32_t kDensePrefillMinimumRows = 128U;
inline constexpr std::uint32_t kDensePrefillMaximumRows = 8192U;
inline constexpr std::size_t kDensePrefillMaximumContext = 262144U;
inline constexpr std::size_t kDensePrefillScoreElements = 24U * 2048U * 2048U;
inline constexpr std::size_t kDensePrefillSplitPvMinimumContext = 12288U;
inline constexpr std::uint32_t kDensePrefillPvSplits = 16U;

[[nodiscard]] constexpr std::uint32_t dense_prefill_pv_splits(std::size_t context) noexcept {
    return context >= kDensePrefillSplitPvMinimumContext ? kDensePrefillPvSplits : 1U;
}

// Qualification-only Text P129 WMMA tail candidate. Zero retains the dense P129 route; one admits
// the exact P129 tail overwrite. The DFlash W5/W6 batched-WMMA route is a production route and is
// not gated by this flag.
#ifndef NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE
#define NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE 0
#endif
static_assert(NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 0 ||
                  NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 1,
              "R9700 Text P129 WMMA tail candidate must be zero or one");
inline constexpr bool kTextP129WmmaTailCandidate =
    NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 1;

// Causal prefill covers initial and appended chunks. Query panels bound the score plane while
// preserving absolute positions and the complete visible cache frontier.
[[nodiscard]] constexpr bool use_dense_prefill_attention(
    std::uint32_t query_rows, std::size_t visible_context) noexcept {
    return query_rows >= kDensePrefillMinimumRows &&
           query_rows <= kDensePrefillMaximumRows && visible_context >= query_rows &&
           visible_context <= kDensePrefillMaximumContext;
}

[[nodiscard]] constexpr std::uint32_t dense_prefill_panel_rows(
    std::uint32_t query_rows, std::size_t context) noexcept {
    if (!use_dense_prefill_attention(query_rows, context)) return 0U;
    const auto capacity = static_cast<std::uint32_t>(
        (kDensePrefillScoreElements / (24U * context)) / 16U * 16U);
    if (query_rows <= capacity) return query_rows;
    // Keep the minimum number of panels, but distribute query tiles evenly. Filling
    // every panel to capacity can strand a few rows in a final near-empty PV grid.
    const auto panels = (query_rows + capacity - 1U) / capacity;
    return ((query_rows + panels * 16U - 1U) / (panels * 16U)) * 16U;
}

[[nodiscard]] constexpr bool use_fp8_qk_wmma(std::uint32_t query_rows,
                                              std::size_t visible_context) noexcept {
    return kFp8QkWmmaDecode &&
           ((query_rows == 1U && visible_context >= kFp8QkWmmaT1MinimumContext) ||
            (query_rows == 2U && visible_context >= kFp8QkWmmaT2MinimumContext));
}

// Production DFlash W4..6 batched-WMMA route. It is selected for the exactly qualified cell:
// DFlash target verification, rows 4..6, non-tree execution, and context 64..262144. The caller
// must additionally require G16, token-fastest FP8 keys, and feature-fastest INT4/FP16 values and
// scales; other cells retain their existing split/fused fallback.
// This route is not gated by a qualification flag.
[[nodiscard]] constexpr bool use_dflash_verify_batched_wmma(
    std::uint32_t query_rows, std::size_t visible_context,
    bool tree_or_device_count, bool dflash_target_verify) noexcept {
    return dflash_target_verify &&
           (query_rows >= 4U && query_rows <= 6U) && !tree_or_device_count &&
           visible_context >= kFp8QkWmmaT1MinimumContext &&
           visible_context <= kDflashVerifyWmmaMaximumContext;
}

[[nodiscard]] constexpr bool use_text_p129_wmma_tail(std::uint32_t query_rows,
                                                       std::size_t visible_context) noexcept {
    return kTextP129WmmaTailCandidate && query_rows == 129U && visible_context == 129U;
}

// T=1 split attention has no device-selected inactive-row or tree form; those fixed-address
// schedules retain the fused leaf. T=4 owns both metadata forms and remains split-eligible.
[[nodiscard]] constexpr bool use_split512_attention(std::uint32_t query_rows,
                                                     std::size_t visible_context,
                                                     bool tree_or_device_count = false) noexcept {
    return kFp8QkWmmaDecode && visible_context >= kSplit512MinimumContext &&
           (query_rows == 4U || (query_rows == 1U && !tree_or_device_count));
}

} // namespace ninfer::ops::r9700::kv
