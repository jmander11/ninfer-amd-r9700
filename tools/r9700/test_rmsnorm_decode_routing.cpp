#include "ops/r9700/eager/eager_ops.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using ninfer::ops::r9700::eager::rmsnorm_k5120_row_cta_selected;
    for (const std::uint32_t rows : {1U, 24U, 127U, 128U, 2048U, 65536U}) {
        if (!rmsnorm_k5120_row_cta_selected(5120U, rows)) {
            throw std::runtime_error("K5120 row count was not selected");
        }
    }
    if (rmsnorm_k5120_row_cta_selected(5120U, 0U)) {
        throw std::runtime_error("empty row count was selected");
    }
    for (const std::uint32_t features : {0U, 1U, 128U, 256U, 5119U, 5121U, 6144U,
                                         std::numeric_limits<std::uint32_t>::max()}) {
        for (std::uint32_t rows = 1U; rows <= 24U; ++rows) {
            if (rmsnorm_k5120_row_cta_selected(features, rows)) {
                throw std::runtime_error("off-domain feature width did not retain fallback");
            }
        }
    }
    std::cout << "r9700 canonical K5120 RMSNorm routing: PASS\n";
}
