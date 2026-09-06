#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

using ninfer::ops::r9700::linear::a8q4g64_scalar_base_u32_offsets_fit;
using ninfer::ops::r9700::linear::use_a8q4_prefill_cta;

struct LastBytes {
    std::uint64_t activation_code;
    std::uint64_t weight_code;
    std::uint64_t activation_scale;
    std::uint64_t weight_scale;
};

LastBytes independent_last_bytes(std::uint32_t tokens, std::uint32_t rows,
                                 std::uint32_t padded_columns) {
    const std::uint64_t groups = padded_columns / 64U;
    const std::uint64_t token = tokens - 1U;
    const std::uint64_t row = rows - 1U;
    const std::uint64_t group = groups - 1U;
    return {
        token * (padded_columns / 2U) + group * 32U + 31U,
        (((((row >> 4U) * groups + group) * 4U + 3U) * 16U +
          (row & 15U)) * 8U) + 7U,
        (token * groups + group) * 2U + 1U,
        (((row >> 4U) * groups + group) * 16U + (row & 15U)) * 2U + 1U,
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 8> shapes{{
        {1024U, 5120U}, {4096U, 5120U}, {5120U, 6144U}, {5120U, 10240U},
        {5120U, 17408U}, {7168U, 5120U}, {12288U, 5120U}, {34816U, 5120U},
    }};
    constexpr std::array<std::uint32_t, 4> extents{1024U, 2048U, 4096U, 8192U};
    LastBytes maxima{};
    for (const auto [rows, columns] : shapes) {
        for (const std::uint32_t tokens : extents) {
            require(use_a8q4_prefill_cta(tokens, rows, columns),
                    "active dispatch predicate omits a qualified tuple");
            require(a8q4g64_scalar_base_u32_offsets_fit(tokens, rows, columns),
                    "qualified tuple does not fit U32 offsets");
            const LastBytes value = independent_last_bytes(tokens, rows, columns);
            maxima.activation_code =
                std::max(maxima.activation_code, value.activation_code);
            maxima.weight_code = std::max(maxima.weight_code, value.weight_code);
            maxima.activation_scale =
                std::max(maxima.activation_scale, value.activation_scale);
            maxima.weight_scale = std::max(maxima.weight_scale, value.weight_scale);
        }
    }
    require(maxima.activation_code == 71303167U, "activation-code maximum changed");
    require(maxima.weight_code == 89128959U, "weight-code maximum changed");
    require(maxima.activation_scale == 4456447U, "activation-scale maximum changed");
    require(maxima.weight_scale == 5570559U, "weight-scale maximum changed");
    require(!use_a8q4_prefill_cta(1023U, 1024U, 5120U) &&
                !use_a8q4_prefill_cta(1024U, 1024U, 5056U) &&
                !use_a8q4_prefill_cta(1024U, 128U, 5120U),
            "active dispatch predicate admits an off-inventory tuple");

    require(a8q4g64_scalar_base_u32_offsets_fit(1U, 67108864U, 128U),
            "UINT32_MAX final byte must fit");
    require(!a8q4g64_scalar_base_u32_offsets_fit(1U, 67108865U, 128U),
            "first final byte above UINT32_MAX must not fit");
    require(!a8q4g64_scalar_base_u32_offsets_fit(0U, 128U, 128U) &&
                !a8q4g64_scalar_base_u32_offsets_fit(1U, 0U, 128U) &&
                !a8q4g64_scalar_base_u32_offsets_fit(1U, 128U, 127U) &&
                !a8q4g64_scalar_base_u32_offsets_fit(
                    UINT32_MAX, UINT32_MAX, UINT32_MAX - 63U),
            "malformed bound inputs accepted");
    std::cout << "PASS tuples=32 activation_code_max=71303167 "
                 "weight_code_max=89128959 activation_scale_max=4456447 "
                 "weight_scale_max=5570559 uint32_boundary=true\n";
}
