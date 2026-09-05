#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"

#include <array>
#include <cstdint>
#include <iostream>

namespace linear = ninfer::ops::r9700::linear;

namespace {

struct Shape {
    std::uint32_t rows;
    std::uint32_t columns;
};

constexpr std::array<std::uint32_t, 4> kQualifiedTokens{1024U, 2048U, 4096U, 8192U};
constexpr std::array<Shape, 8> kQ4Shapes{{
    {7168U, 5120U}, {4096U, 5120U}, {12288U, 5120U}, {5120U, 6144U},
    {34816U, 5120U}, {5120U, 17408U}, {5120U, 10240U}, {1024U, 5120U},
}};
constexpr std::array<Shape, 4> kW8Shapes{{
    {7168U, 5120U}, {12288U, 5120U}, {5120U, 6144U}, {5120U, 17408U},
}};

template <std::size_t N, typename Predicate>
constexpr bool accepts_cartesian(const std::array<Shape, N>& shapes, Predicate predicate) {
    for (const Shape shape : shapes) {
        for (const std::uint32_t tokens : kQualifiedTokens) {
            if (!predicate(tokens, shape.rows, shape.columns)) return false;
        }
    }
    return true;
}

static_assert(linear::kQ4ActivationBits != 8U ||
              accepts_cartesian(kQ4Shapes, linear::use_a8q4_prefill_cta));
static_assert(linear::kW8ActivationBits != 8U ||
              accepts_cartesian(kW8Shapes, linear::use_a8w8_prefill_cta));

static_assert(!linear::use_a8q4_prefill_cta(128U, 7168U, 5120U));
static_assert(!linear::use_a8q4_prefill_cta(129U, 7168U, 5120U));
static_assert(!linear::use_a8q4_prefill_cta(1023U, 7168U, 5120U));
static_assert(!linear::use_a8q4_prefill_cta(8193U, 7168U, 5120U));
static_assert(!linear::use_a8q4_prefill_cta(4096U, 5120U, 5120U));
static_assert(!linear::use_a8q4_prefill_cta(4096U, 5120U, 25600U));

static_assert(linear::select_a8q4_prefill_route(2048U, 7168U, 5120U) ==
              linear::A8Q4PrefillRoute::M64N128PingPongProduction);
static_assert(linear::select_a8q4_prefill_route(512U, 7168U, 5120U) ==
              linear::A8Q4PrefillRoute::Wmma32);
static_assert(linear::select_a8q4_prefill_route(2048U, 5120U, 25600U) ==
              linear::A8Q4PrefillRoute::Wmma32);

static_assert(!linear::use_a8w8_prefill_cta(128U, 7168U, 5120U));
static_assert(!linear::use_a8w8_prefill_cta(4095U, 7168U, 5120U));
static_assert(!linear::use_a8w8_prefill_cta(8193U, 7168U, 5120U));
static_assert(!linear::use_a8w8_prefill_cta(4096U, 14336U, 5120U));
static_assert(!linear::use_a8w8_prefill_cta(4096U, 34816U, 5120U));
static_assert(!linear::use_a8w8_prefill_cta(4096U, 5120U, 10240U));

} // namespace

int main() {
    std::cout << "R9700 cooperative prefill CTA dispatch predicates passed: q4_prefill_cta_profile="
              << linear::kQ4PrefillCtaProfile << '\n';
    return 0;
}
