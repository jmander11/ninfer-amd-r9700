#pragma once

#include <cstdint>

namespace ninfer::ops::r9700::eager {

// Qualification-selected DFlash RMSNorm route. The ordinary build keeps this disabled; matched
// candidate builds admit only the independently qualified K5120 rows5/6 cells.
#ifndef NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE
#define NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE 0
#endif
static_assert(NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE == 0 ||
                  NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE == 1,
              "R9700 DFlash RMSNorm rows5/6 candidate selector must be zero or one");
inline constexpr bool kDFlashRmsnormRows56CandidateEnabled =
    NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE == 1;

[[nodiscard]] constexpr bool is_rmsnorm_k5120_rows56_candidate_eligible(
    std::uint32_t features, std::uint32_t rows) noexcept {
    return features == 5120U && (rows == 5U || rows == 6U);
}

[[nodiscard]] constexpr bool use_rmsnorm_k5120_rows56_candidate(
    std::uint32_t features, std::uint32_t rows) noexcept {
    return kDFlashRmsnormRows56CandidateEnabled &&
           is_rmsnorm_k5120_rows56_candidate_eligible(features, rows);
}

} // namespace ninfer::ops::r9700::eager
