#include "ops/r9700/linear/linear_execution.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ninfer/ops/normalized_linear.h"
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

constexpr bool normalized_selected(
    std::uint32_t bits = 8U, bool inventory = true,
    ninfer::targets::qwen3::TextPhase phase = ninfer::targets::qwen3::TextPhase::Verify,
    bool ordinary = true, std::int32_t layer = 0, std::uint32_t tokens = 1U,
    std::uint32_t rows = 34816U, std::uint32_t columns = 5120U,
    ninfer::QType weight = ninfer::QType::Q4G64_F16S) {
    return detail::Variant::ExecutionState::normalized_linear_t1_selected(
        bits, inventory, phase, ordinary, layer, tokens, rows, columns, weight);
}

void normalized_route_contract() {
    using Phase = ninfer::targets::qwen3::TextPhase;
    static_assert(normalized_selected());
    static_assert(normalized_selected(8U, true, Phase::Verify, true, 63));
    static_assert(!normalized_selected(4U));
    static_assert(!normalized_selected(8U, false));
    static_assert(!normalized_selected(8U, true, Phase::Prefill));
    // A speculative target verify can also be width one: phase/shape cannot select this route.
    static_assert(!normalized_selected(8U, true, Phase::Verify, false));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, -1));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, 64));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, 0, 2U));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, 0, 4U));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, 0, 1U, 17408U));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, 0, 1U, 34816U,
                                      6144U));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, 0, 1U, 34816U,
                                      5120U, ninfer::QType::F8E4M3_ROW_F32S));
    static_assert(!normalized_selected(8U, true, Phase::Verify, true, 0, 1U, 34816U,
                                      5120U, ninfer::QType::W8G32_F16S));

    detail::Variant::ModelView model{};
    std::vector<ninfer::Weight*> matrices;
    for (auto& layer : model.full_layers) {
        matrices.insert(matrices.end(), {&layer.projection.query_key, &layer.projection.gate_value,
            &layer.output, &layer.post_mixer.gate_up, &layer.post_mixer.down});
    }
    for (auto& layer : model.gdn_layers) {
        matrices.insert(matrices.end(), {&layer.projection.input_projection.query_key,
            &layer.projection.input_projection.value_z, &layer.output,
            &layer.post_mixer.gate_up, &layer.post_mixer.down});
    }
    for (auto* weight : matrices) weight->qtype = ninfer::QType::Q4G64_F16S;
    require(detail::Variant::ExecutionState::normalized_linear_t1_inventory_q4(model),
            "all-Q4 base Text inventory did not select normalized-linear");
    for (auto* weight : matrices) {
        weight->qtype = ninfer::QType::F8E4M3_ROW_F32S;
        require(!detail::Variant::ExecutionState::normalized_linear_t1_inventory_q4(model),
                "mixed Text inventory selected normalized-linear");
        weight->qtype = ninfer::QType::Q4G64_F16S;
    }
    if constexpr (ninfer::ops::r9700::linear::kQ4ActivationBits == 8U) {
        const std::size_t required =
            ninfer::ops::normalized_linear_workspace_capacity_bytes(1, 5120, 34816);
        require(required == 5380U, "normalized-linear T1 activation extent changed");
        require(detail::Variant::execution_state_capacity_bytes(
                    detail::WeightsProfile::R9700Q4G64Evaluation, 1, 1) >= required,
                "minimal serialized activation region cannot contain normalized-linear T1");
    }
}

} // namespace

int main() {
    try {
        using Variant = detail::Variant;
        normalized_route_contract();
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
