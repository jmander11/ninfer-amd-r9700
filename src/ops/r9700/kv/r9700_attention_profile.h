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
inline constexpr std::uint32_t kDensePrefillMinimumRows = 128U;
inline constexpr std::uint32_t kDensePrefillMaximumRows = 4096U;

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

// Physical selection covered complete initial-prefix calls at P=128/512/1024/2048/4096. Do not
// extrapolate the route to later chunks with a longer visible frontier without matched evidence.
[[nodiscard]] constexpr bool use_dense_prefill_attention(
    std::uint32_t query_rows, std::size_t visible_context) noexcept {
    return query_rows >= kDensePrefillMinimumRows &&
           query_rows <= kDensePrefillMaximumRows && visible_context == query_rows;
}

[[nodiscard]] constexpr bool use_fp8_qk_wmma(std::uint32_t query_rows,
                                              std::size_t visible_context) noexcept {
    return kFp8QkWmmaDecode &&
           ((query_rows == 1U && visible_context >= kFp8QkWmmaT1MinimumContext) ||
            (query_rows == 2U && visible_context >= kFp8QkWmmaT2MinimumContext));
}

// Production DFlash W5/W6 batched-WMMA route. It is selected for the exactly qualified cell:
// DFlash target verification, rows 5 or 6, non-tree execution, and context 64..8191. The caller
// must additionally require G16, token-fastest FP8 keys, and feature-fastest INT4/FP16 values and
// scales; every other cell (G32, wrong layout, tree, out-of-range context) retains the fused
// fallback. This route is not gated by a qualification flag.
[[nodiscard]] constexpr bool use_dflash_w5w6_batched_wmma(
    std::uint32_t query_rows, std::size_t visible_context,
    bool tree_or_device_count, bool dflash_target_verify) noexcept {
    return dflash_target_verify &&
           (query_rows == 5U || query_rows == 6U) && !tree_or_device_count &&
           visible_context >= kFp8QkWmmaT1MinimumContext &&
           visible_context < kSplit512MinimumContext;
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
