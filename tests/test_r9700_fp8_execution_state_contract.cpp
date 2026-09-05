#include "ops/r9700/linear/linear_execution.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace detail = ninfer::targets::qwen3_8_27b::detail;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        using Variant = detail::Variant;
        static_assert(!std::is_copy_constructible_v<Variant::ExecutionState>);
        static_assert(!std::is_move_constructible_v<Variant::ExecutionState>);
        static_assert(Variant::ExecutionState::fused_mlp_down_selected(
            ninfer::QType::F8E4M3_ROW_F32S, ninfer::QType::Q4G64_F16S, 2048U, 63));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            ninfer::QType::F8E4M3_ROW_F32S, ninfer::QType::Q4G64_F16S, 2047U, 63));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            ninfer::QType::Q4G64_F16S, ninfer::QType::Q4G64_F16S, 2048U, 63));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            ninfer::QType::F8E4M3_ROW_F32S, ninfer::QType::Q4G64_F16S, 2048U, -1));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            ninfer::QType::F8E4M3_ROW_F32S, ninfer::QType::Q4G64_F16S, 2048U,
            detail::TextConfig::layers));
        require(Variant::ExecutionState::eager_widths(2048, 4, 4, 0) ==
                    std::vector<std::uint32_t>({1, 2, 3, 4, 8, 12, 16, 2048}),
                "MTP eager descriptor widths are incomplete");
        require(Variant::ExecutionState::eager_widths(2048, 4, 0, 12) ==
                    std::vector<std::uint32_t>({1, 2, 3, 4, 12, 24, 36, 48, 2048}),
                "DFlash eager descriptor widths are incomplete");
        require(Variant::ExecutionState::eager_widths(4, 4, 0, 0) ==
                    std::vector<std::uint32_t>({1, 2, 3, 4}),
                "ordinary eager descriptor widths are not unique and complete");
        const auto prefill_widths = Variant::ExecutionState::eager_widths(2048, 4, 0, 0);
        require(std::find(prefill_widths.begin(), prefill_widths.end(), 2047U) ==
                    prefill_widths.end(),
                "irregular final prefill tail was incorrectly made graph-eager");
        bool width_overflow_rejected = false;
        try {
            (void)Variant::ExecutionState::eager_widths(
                2048, 4, 0, std::numeric_limits<std::uint32_t>::max());
        } catch (const std::overflow_error&) {
            width_overflow_rejected = true;
        }
        require(width_overflow_rejected, "overflowing eager descriptor width was accepted");

        constexpr std::uint32_t kPrefillTokens = 2048;
        constexpr std::uint32_t kGraphTokens   = 48;
        const std::size_t fp8_activation =
            ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(
                kPrefillTokens, detail::TextConfig::hidden);
        const std::size_t q4_activation = Variant::linear_workspace_capacity_bytes(
            detail::WeightsProfile::R9700Q4G64Evaluation, kPrefillTokens);
        const std::size_t expected =
            (std::max(fp8_activation, q4_activation) + 255U) / 256U * 256U;
        require(Variant::execution_state_capacity_bytes(
                    detail::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation,
                    kPrefillTokens, kGraphTokens) == expected,
                "hybrid execution region differs from the shared serialized storage contract");
        const std::size_t hybrid_dflash_q4 = Variant::linear_workspace_capacity_bytes(
            detail::WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation,
            kPrefillTokens);
        const std::size_t expected_hybrid_dflash =
            (std::max(fp8_activation, hybrid_dflash_q4) + 255U) / 256U * 256U;
        require(Variant::execution_state_capacity_bytes(
                    detail::WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation,
                    kPrefillTokens, kGraphTokens) == expected_hybrid_dflash,
                "hybrid DFlash execution region omits FP8 state or the largest Q4 activation");
        require(Variant::execution_state_capacity_bytes(
                    detail::WeightsProfile::R9700Q4G64Evaluation,
                    kPrefillTokens, kGraphTokens) ==
                    (q4_activation + 255U) / 256U * 256U,
                "Q4 execution region does not replace its former arena reserve");
        const std::size_t w8_activation = Variant::linear_workspace_capacity_bytes(
            detail::WeightsProfile::R9700W8G32Candidate, kPrefillTokens);
        require(Variant::execution_state_capacity_bytes(
                    detail::WeightsProfile::R9700W8G32Candidate,
                    kPrefillTokens, kGraphTokens) ==
                    (w8_activation + 255U) / 256U * 256U,
                "W8 serialized activation region differs from its former arena reserve");
        bool rejected = false;
        try {
            (void)Variant::execution_state_capacity_bytes(
                detail::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation, 0U, kGraphTokens);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "zero-width hybrid FP8 execution capacity was accepted");
        std::cout << "R9700 FP8 Program execution-state contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
