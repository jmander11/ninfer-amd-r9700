#include "core/roctx.h"

#include <rocprofiler-sdk-roctx/roctx.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        using ninfer::roctx::Category;
        using ninfer::roctx::Name;
        using ninfer::roctx::ScopedEnable;
        using ninfer::roctx::ScopedRange;
        if (std::strcmp(ninfer::roctx::registered_message(Name::DecodeDFlashRound),
                        "decode.dflash_round") != 0 ||
            std::strcmp(ninfer::roctx::category_message(Category::Attention), "attention") != 0 ||
            ninfer::roctx::color(Category::DFlash) != 0xffaf7aa1u) {
            throw std::runtime_error("stable tracing metadata changed");
        }
        {
            ScopedEnable enabled;
            ScopedRange outer(Name::Generate, Category::Runtime, 7);
            {
                ScopedRange inner(Name::DecodeDFlashRound, Category::DFlash,
                                  static_cast<std::uint64_t>(11));
            }
        }
        if (roctxRangePop() >= 0) {
            throw std::runtime_error("ScopedRange left an unbalanced ROCtx range");
        }
        std::cout << "roctx_range: PASS nested push/pop metadata\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "roctx_range: FAIL: " << error.what() << '\n';
        return 1;
    }
}
