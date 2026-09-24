#include "ops/r9700/linear/linear_execution.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ninfer/ops/normalized_linear.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_27b/impl/variant.h"
#include <ninfer/targets/qwen3/startup_features.h>

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
    std::uint32_t bits = 8U,
    ninfer::targets::qwen3::TextPhase phase = ninfer::targets::qwen3::TextPhase::Verify,
    bool ordinary = true, std::int32_t layer = 0, std::uint32_t tokens = 1U,
    std::uint32_t rows = 34816U, std::uint32_t columns = 5120U,
    ninfer::QType weight = ninfer::QType::Q4G64_F16S) {
    return detail::Variant::ExecutionState::normalized_linear_t1_selected(
        bits, phase, ordinary, layer, tokens, rows, columns, weight);
}

void normalized_route_contract() {
    using Phase = ninfer::targets::qwen3::TextPhase;
    static_assert(normalized_selected());
    static_assert(normalized_selected(8U, Phase::Verify, true, 63));
    static_assert(!normalized_selected(4U));
    static_assert(!normalized_selected(8U, Phase::Prefill));
    // A speculative target verify can also be width one: phase/shape cannot select this route.
    static_assert(!normalized_selected(8U, Phase::Verify, false));
    static_assert(!normalized_selected(8U, Phase::Verify, true, -1));
    static_assert(!normalized_selected(8U, Phase::Verify, true, 64));
    static_assert(!normalized_selected(8U, Phase::Verify, true, 0, 2U));
    static_assert(!normalized_selected(8U, Phase::Verify, true, 0, 4U));
    static_assert(!normalized_selected(8U, Phase::Verify, true, 0, 1U, 17408U));
    static_assert(!normalized_selected(8U, Phase::Verify, true, 0, 1U, 34816U,
                                      6144U));
    static_assert(!normalized_selected(8U, Phase::Verify, true, 0, 1U, 34816U,
                                      5120U, ninfer::QType::F8E4M3_ROW_F32S));
    static_assert(!normalized_selected(8U, Phase::Verify, true, 0, 1U, 34816U,
                                      5120U, ninfer::QType::W8G32_F16S));

    if constexpr (ninfer::ops::r9700::linear::kQ4ActivationBits == 8U) {
        const std::size_t required =
            ninfer::ops::normalized_linear_workspace_capacity_bytes(1, 5120, 34816);
        require(required == 5380U, "normalized-linear T1 activation extent changed");
        require(detail::Variant::execution_state_capacity_bytes(
                    detail::WeightsProfile::R9700Q4G64Evaluation, 1, 1) >= required,
                "minimal serialized activation region cannot contain normalized-linear T1");
    }
}

void adaptive_width_contract() {
    namespace qwen3 = ninfer::targets::qwen3;
    using Backend = ninfer::SpeculativeBackend;
    for (const auto backend : {Backend::Mtp, Backend::DFlash}) {
        ninfer::EngineOptions options;
        options.speculative.backend = backend;
        options.speculative.draft_tokens = 5U;
        options.speculative.adaptive_draft = true;
        const auto startup = qwen3::startup_verify_widths<detail::DFlashConfig>(options);
        require(startup == std::vector<std::uint32_t>({4, 5, 6}),
                "adaptive startup omits a captured verification width");
        const auto ks = qwen3::adaptive_draft_ks(backend, 5U, true);
        const auto storage = backend == Backend::DFlash
            ? qwen3::dflash_storage_verify_width<detail::DFlashConfig>(ks, 5U, 0U) : 0U;
        const auto program = qwen3::captured_verify_widths<detail::DFlashConfig>(
            backend, ks, storage);
        require(startup == program, "loaded and Program FP8 preparation disagree");
        for (std::uint32_t concurrency = 1U; concurrency <= 4U; ++concurrency) {
            const auto prepared = detail::Variant::ExecutionState::eager_widths(
                2048U, concurrency, startup);
            std::vector<std::uint32_t> observed{2048U};
            for (std::uint32_t batch = 1U; batch <= concurrency; ++batch) {
                observed.push_back(batch);
                // These are the physical target Linear widths of the three
                // captured K3/K4/K5 graphs, not just the K5 storage maximum.
                for (const std::uint32_t width : {4U, 5U, 6U})
                    observed.push_back(batch * width);
            }
            std::sort(observed.begin(), observed.end());
            observed.erase(std::unique(observed.begin(), observed.end()), observed.end());
            require(prepared == observed,
                    "adaptive C1-C4 graph execution width was not prepared exactly");
            if (backend == Backend::DFlash) {
                // A tiny prefill cannot conceal insufficient serialized activation
                // storage for a K5 append after the live adaptive graph drops to K3/K4.
                const auto capacity = detail::Variant::execution_state_capacity_bytes(
                    detail::WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation,
                    1U, concurrency * storage);
                for (std::uint32_t batch = 1U; batch <= concurrency; ++batch) {
                    for (const auto width : startup) {
                        for (const auto columns : {detail::TextConfig::intermediate,
                                                   detail::DFlashConfig::feature_rows}) {
                            require(capacity >= ninfer::ops::linear_workspace_capacity_bytes(
                                        ninfer::QType::Q4G64_F16S, batch * width, columns),
                                    "adaptive graph or pending append exceeds Q4 activation storage");
                        }
                    }
                }
            }
        }
        options.speculative.adaptive_draft = false;
        require(qwen3::startup_verify_widths<detail::DFlashConfig>(options) ==
                    std::vector<std::uint32_t>({6}),
                "fixed K5 unexpectedly prepares adaptive graphs");
    }
    ninfer::EngineOptions ordinary;
    require(qwen3::startup_verify_widths<detail::DFlashConfig>(ordinary).empty(),
            "ordinary startup prepares speculative graph widths");
    ninfer::EngineOptions fixed;
    fixed.speculative.backend = Backend::DFlash;
    fixed.speculative.draft_tokens = 4U;
    fixed.speculative.adaptive_draft = true;
    require(qwen3::startup_verify_widths<detail::DFlashConfig>(fixed) ==
                std::vector<std::uint32_t>({5}),
            "K4-only adaptive policy was broadened during preparation");
    fixed.speculative.draft_tokens = 5U;
    fixed.speculative.dflash_verify_width = 6U;
    require(qwen3::startup_verify_widths<detail::DFlashConfig>(fixed) ==
                std::vector<std::uint32_t>({4, 5, 6}),
            "explicit storage width erased smaller captured graph widths");
}

} // namespace

int main() {
    try {
        using Variant = detail::Variant;
        normalized_route_contract();
        adaptive_width_contract();
        static_assert(!std::is_copy_constructible_v<Variant::ExecutionState>);
        static_assert(!std::is_move_constructible_v<Variant::ExecutionState>);
        // Prepared executions borrow a stable, explicitly owned device context.
        static_assert(!std::is_copy_constructible_v<ninfer::ops::LinearExecutionContext>);
        static_assert(!std::is_move_constructible_v<ninfer::ops::LinearExecutionContext>);
        static_assert(std::is_constructible_v<ninfer::ops::LinearExecution,
            ninfer::ops::LinearExecutionContext&, const ninfer::Weight&,
            std::size_t, std::size_t>);
        static_assert(Variant::runtime_allocation_overhead_bound(
            detail::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation) == (4U << 20U));
        static_assert(Variant::runtime_allocation_overhead_bound(
            detail::WeightsProfile::R9700Q4G64Evaluation) == 0U);
        static_assert(Variant::runtime_allocation_overhead_bound(
            detail::WeightsProfile::R9700Q4W8Evaluation) == 0U);
        static_assert(Variant::ExecutionState::fused_mlp_down_selected(
            8U, ninfer::QType::Q4G64_F16S, 2048U, 63));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            4U, ninfer::QType::Q4G64_F16S, 2048U, 63));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            8U, ninfer::QType::Q4G64_F16S, 2047U, 63));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            8U, ninfer::QType::F8E4M3_ROW_F32S, 2048U, 63));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            8U, ninfer::QType::Q4G64_F16S, 2048U, -1));
        static_assert(!Variant::ExecutionState::fused_mlp_down_selected(
            8U, ninfer::QType::Q4G64_F16S, 2048U,
            detail::TextConfig::layers));
        const std::uint32_t mtp_widths[]{4U};
        const std::uint32_t dflash_widths[]{12U};
        require(Variant::ExecutionState::eager_widths(2048, 4, mtp_widths) ==
                    std::vector<std::uint32_t>({1, 2, 3, 4, 8, 12, 16, 2048}),
                "MTP eager descriptor widths are incomplete");
        require(Variant::ExecutionState::eager_widths(2048, 4, dflash_widths) ==
                    std::vector<std::uint32_t>({1, 2, 3, 4, 12, 24, 36, 48, 2048}),
                "DFlash eager descriptor widths are incomplete");
        require(Variant::ExecutionState::eager_widths(4, 4, {}) ==
                    std::vector<std::uint32_t>({1, 2, 3, 4}),
                "ordinary eager descriptor widths are not unique and complete");
        const auto prefill_widths = Variant::ExecutionState::eager_widths(2048, 4, {});
        require(std::find(prefill_widths.begin(), prefill_widths.end(), 2047U) ==
                    prefill_widths.end(),
                "irregular final prefill tail was incorrectly made graph-eager");
        bool width_overflow_rejected = false;
        try {
            const std::uint32_t overflow_widths[]{std::numeric_limits<std::uint32_t>::max()};
            (void)Variant::ExecutionState::eager_widths(
                2048, 4, overflow_widths);
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
        const auto protected_profile = detail::WeightsProfile::R9700Q4SelectiveProtectedN16K16Evaluation;
        const auto protected_linear = std::max(q4_activation,
            ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::W8G32_F16S,
                kPrefillTokens, detail::TextConfig::hidden));
        const auto protected_fp8 = ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(
            kPrefillTokens, detail::TextConfig::intermediate);
        require(Variant::linear_workspace_capacity_bytes(protected_profile, kPrefillTokens) ==
                    protected_linear, "protected profile loses Q4 down or W8 head scratch");
        require(Variant::execution_state_capacity_bytes(protected_profile, kPrefillTokens, kGraphTokens) ==
                    (std::max(protected_linear, protected_fp8) + 255U) / 256U * 256U,
                "protected FP8 down activation K17408 is not covered");
        require(Variant::vision_linear_workspace_capacity_bytes(protected_profile, kPrefillTokens) ==
                    Variant::vision_linear_workspace_capacity_bytes(
                        detail::WeightsProfile::R9700Q4G64Evaluation, kPrefillTokens),
                "protected profile changes Vision scratch");
        require(Variant::runtime_allocation_overhead_bound(protected_profile) == (4U << 20U),
                "protected FP8 profile omits physical arena rounding bound");
        const auto protected_dflash = detail::WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation;
        require(Variant::dflash_matrix_qtype(protected_dflash) == ninfer::QType::Q4G64_F16S,
                "protected companion does not bind canonical Q4 DFlash matrices");
        for (const auto tokens : {1, 5, 6, 24, 2048}) {
            const auto base = Variant::linear_workspace_capacity_bytes(protected_profile, tokens);
            const auto companion = ninfer::ops::linear_workspace_capacity_bytes(
                ninfer::QType::Q4G64_F16S, tokens, detail::DFlashConfig::feature_rows);
            require(Variant::linear_workspace_capacity_bytes(protected_dflash, tokens) ==
                        std::max(base, companion),
                    "protected companion omits W8 endpoint or Q4 K25600 scratch");
            auto state = std::max(std::max(base, companion),
                ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(
                    tokens, detail::TextConfig::intermediate));
            require(Variant::execution_state_capacity_bytes(protected_dflash, tokens, tokens) ==
                        (state + 255U) / 256U * 256U,
                    "protected companion loses FP8 K17408 or Q4 activation workspace");
            require(Variant::vision_linear_workspace_capacity_bytes(protected_dflash, tokens) ==
                        Variant::vision_linear_workspace_capacity_bytes(protected_profile, tokens),
                    "protected companion changes base Vision workspace");
        }
        require(Variant::runtime_allocation_overhead_bound(protected_dflash) == (4U << 20U),
                "protected companion omits FP8 arena rounding bound");
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
        using Profile = detail::WeightsProfile;
#define NINFER_QWEN38_FP8_ENDPOINT(symbol, id, base, embed, head) \
        { \
            const auto endpoint = Profile::symbol; \
            require(detail::fp8_capped_base_profile(endpoint) == Profile::base && \
                    detail::fp8_capped_w8_embedding(endpoint) == (embed != 0) && \
                    detail::fp8_capped_w8_head(endpoint) == (head != 0), \
                    "capped endpoint protection identity differs"); \
            const auto base_linear = Variant::linear_workspace_capacity_bytes(Profile::base, kPrefillTokens); \
            const auto w8_linear = ninfer::ops::linear_workspace_capacity_bytes( \
                ninfer::QType::W8G32_F16S, kPrefillTokens, detail::TextConfig::hidden); \
            require(Variant::linear_workspace_capacity_bytes(endpoint, kPrefillTokens) == \
                    (head != 0 ? std::max(base_linear, w8_linear) : base_linear), \
                    "capped W8 head activation workspace differs"); \
            require(Variant::execution_state_capacity_bytes(endpoint, kPrefillTokens, kGraphTokens) >= \
                    Variant::execution_state_capacity_bytes(Profile::base, kPrefillTokens, kGraphTokens), \
                    "endpoint variant loses base execution state"); \
            require(Variant::vision_linear_workspace_capacity_bytes(endpoint, kPrefillTokens) == \
                    Variant::vision_linear_workspace_capacity_bytes(Profile::base, kPrefillTokens), \
                    "endpoint variant changes Vision workspace"); \
        }
#include "targets/qwen3_8_27b/impl/load/fp8_endpoint_selection.inc"
#undef NINFER_QWEN38_FP8_ENDPOINT
        for (const auto protected_outputs : {Profile::R9700Q4Fp8DefaultProtectedEvaluation,
                 Profile::R9700Q4Fp8OutputOnlyEvaluation,
                 Profile::R9700Q4Fp8SelectiveNoLateMlpEvaluation}) {
            const auto q4 = Variant::linear_workspace_capacity_bytes(protected_outputs, kPrefillTokens);
            const auto fp8 = ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(
                kPrefillTokens, detail::TextConfig::query_size);
            require(Variant::execution_state_capacity_bytes(protected_outputs, kPrefillTokens, kGraphTokens) ==
                        (std::max(q4, fp8) + 255U) / 256U * 256U,
                    "protected-output recipe loses K6144 FP8 activation storage");
        }
        const auto capped_companion = Profile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation;
        require(Variant::dflash_matrix_qtype(capped_companion) == ninfer::QType::Q4G64_F16S,
                "capped DFlash companion changed its canonical Q4 recipe");
        for (const int tokens : {1, 4, 5, 6, 24, 2048}) {
            const auto base = Variant::linear_workspace_capacity_bytes(
                Profile::R9700Q4Fp8SelectiveCapEvaluation, tokens);
            const auto draft = ninfer::ops::linear_workspace_capacity_bytes(
                ninfer::QType::Q4G64_F16S, tokens, detail::DFlashConfig::feature_rows);
            const auto activation = ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(
                tokens, detail::TextConfig::intermediate);
            require(Variant::linear_workspace_capacity_bytes(capped_companion, tokens) ==
                        std::max(base, draft), "capped companion loses K25600 activation");
            require(Variant::execution_state_capacity_bytes(capped_companion, tokens, tokens) ==
                        (std::max({base, draft, activation}) + 255U) / 256U * 256U,
                    "capped companion loses FP8 K17408 or Q4 activation storage");
            require(Variant::vision_linear_workspace_capacity_bytes(capped_companion, tokens) ==
                        Variant::vision_linear_workspace_capacity_bytes(
                            Profile::R9700Q4Fp8SelectiveCapEvaluation, tokens),
                    "capped companion changed base Vision workspace");
        }
        for (const auto profile : {Profile::R9700Q4G64DFlash2Q4MseEvaluation,
                                  Profile::R9700Q4W8MseDFlash2Q4MseEvaluation,
                                  Profile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation}) {
            require(Variant::dflash_matrix_qtype(profile) == ninfer::QType::Q4G64_F16S,
                    "source-MSE Q4 companion lost its represented matrix format");
        }
        for (const auto profile : {Profile::R9700Q4G64DFlash2W8MseEvaluation,
                                  Profile::R9700Q4W8MseDFlash2W8MseEvaluation,
                                  Profile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation}) {
            require(Variant::dflash_matrix_qtype(profile) == ninfer::QType::W8G32_F16S,
                    "source-MSE W8 companion lost its represented matrix format");
            const auto base = profile == Profile::R9700Q4W8MseDFlash2W8MseEvaluation
                ? Profile::R9700Q4W8Evaluation : Profile::R9700Q4G64Evaluation;
            for (const auto tokens : {1, 6, 12, 48, 128, 2048}) {
                const auto expected_linear = std::max(
                    Variant::linear_workspace_capacity_bytes(base, tokens),
                    ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::W8G32_F16S,
                        tokens, detail::DFlashConfig::feature_rows));
                require(Variant::linear_workspace_capacity_bytes(profile, tokens) == expected_linear,
                        "W8 companion workspace omits base storage or K25600 activation");
                require(Variant::vision_linear_workspace_capacity_bytes(profile, tokens) ==
                        Variant::vision_linear_workspace_capacity_bytes(base, tokens),
                        "W8 companion changed the base Vision workspace");
                auto expected_state = expected_linear;
                if (profile == Profile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation) {
                    expected_state = std::max(expected_state,
                        ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(
                            tokens, detail::TextConfig::hidden));
                }
                expected_state = (expected_state + 255U) / 256U * 256U;
                require(Variant::execution_state_capacity_bytes(profile, tokens, tokens) == expected_state,
                        "W8 companion execution storage omits target verification or FP8 state");
            }
        }
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
