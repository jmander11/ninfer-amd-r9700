#include "ops/r9700/eager/eager_ops.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_selected;
    constexpr std::array<std::uint32_t, 4> selected{1U, 2U, 3U, 4U};
    for (const std::uint32_t rows : selected) {
        if (!rmsnorm_k5120_rows4_selected(5120U, rows)) {
            throw std::runtime_error("exact ordinary-decode row count was not selected");
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
    }
    std::cout << "r9700 production RMSNorm K5120 rows1..4 routing: PASS\n";
}
