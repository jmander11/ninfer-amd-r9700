#include "ops/r9700/eager/eager_ops.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_selected;
    using ninfer::ops::r9700::eager::is_rmsnorm_k5120_rows56_candidate_eligible;
    using ninfer::ops::r9700::eager::kDFlashRmsnormRows56CandidateEnabled;
    using ninfer::ops::r9700::eager::use_rmsnorm_k5120_rows56_candidate;
    constexpr std::array<std::uint32_t, 4> selected{1U, 2U, 3U, 4U};
    for (const std::uint32_t rows : selected) {
        if (!rmsnorm_k5120_rows4_selected(5120U, rows)) {
            throw std::runtime_error("exact ordinary-decode row count was not selected");
        }
    }
    for (const std::uint32_t rows : {5U, 6U}) {
        if (!is_rmsnorm_k5120_rows56_candidate_eligible(5120U, rows) ||
            use_rmsnorm_k5120_rows56_candidate(5120U, rows) !=
                kDFlashRmsnormRows56CandidateEnabled) {
            throw std::runtime_error("exact DFlash candidate row routing differs");
        }
    }
    for (const std::uint32_t rows : {0U, 1U, 4U, 7U, 128U,
                                     std::numeric_limits<std::uint32_t>::max()}) {
        if (use_rmsnorm_k5120_rows56_candidate(5120U, rows)) {
            throw std::runtime_error("off-domain DFlash row count selected candidate");
        }
    }
    for (const std::uint32_t rows : {0U, 5U, 6U, 7U, 8U, 127U, 128U, 2048U,
                                     std::numeric_limits<std::uint32_t>::max()}) {
        if (rmsnorm_k5120_rows4_selected(5120U, rows)) {
            throw std::runtime_error("off-domain row count did not retain fallback");
        }
    }
    for (const std::uint32_t features : {0U, 1U, 128U, 256U, 5119U, 5121U, 6144U,
                                         std::numeric_limits<std::uint32_t>::max()}) {
        for (const std::uint32_t rows : selected) {
            if (rmsnorm_k5120_rows4_selected(features, rows)) {
                throw std::runtime_error("off-domain feature width did not retain fallback");
            }
        }
        for (const std::uint32_t rows : {5U, 6U}) {
            if (use_rmsnorm_k5120_rows56_candidate(features, rows)) {
                throw std::runtime_error("off-domain feature width selected DFlash candidate");
            }
        }
    }
    std::cout << "r9700 RMSNorm routing: PASS dflash_rows56_candidate="
              << (kDFlashRmsnormRows56CandidateEnabled ? "true" : "false") << '\n';
}
