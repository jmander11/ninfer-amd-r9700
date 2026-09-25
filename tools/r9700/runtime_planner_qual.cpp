#include "core/device.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/swa.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "targets/qwen3_8_27b/impl/r9700_full_attention.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#define NINFER_QWEN3_VARIANT ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN3_RUNTIME_NS qwen3_8_27b_r9700
#include "targets/qwen3/impl/runtime/layouts.h"
#include "targets/qwen3/impl/runtime/workspace_recipe.h"
#undef NINFER_QWEN3_RUNTIME_NS
#undef NINFER_QWEN3_VARIANT

#include <ninfer/targets/qwen3/runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Variant = ninfer::targets::qwen3_8_27b::detail::Variant;

void require(bool condition, const char* message);

template <class Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void qualify_host_request_lane_cap() {
    constexpr auto max_batch = static_cast<std::int32_t>(ninfer::kMaximumConcurrency);
    constexpr auto too_many = max_batch + 1;
    constexpr auto q4 = ninfer::QType::Q4G64_F16S;
    constexpr auto profile = Variant::WeightsProfile::R9700Q4G64DFlash2Q4Evaluation;

    (void)ninfer::ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(q4, 1, 16,
                                                                             max_batch);
    (void)ninfer::ops::dflash2_path_select_workspace_capacity_bytes(q4, 1, 16, max_batch);
    (void)ninfer::ops::swa_workspace_capacity_bytes({0, 4096}, 1, 16, max_batch);
    (void)ninfer::ops::bidirectional_gqa_attention_workspace_capacity_bytes(
        {0, 4096}, 1, 16, max_batch);
    (void)ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
        48, max_batch, 16);
    (void)Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
        profile, ninfer::targets::qwen3::TextPhase::Verify, max_batch, 1, 16);
    (void)Variant::gdn_input_projection_record_workspace_capacity_bytes(
        profile, ninfer::targets::qwen3::TextPhase::Verify, max_batch, 2, 16);

    require_invalid_argument(
        [&] {
            (void)ninfer::ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(
                q4, 1, 16, too_many);
        },
        "grouped dynamic convolution accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::dflash2_path_select_workspace_capacity_bytes(q4, 1, 16,
                                                                            too_many);
        },
        "DFlash path selection accepted an oversized request batch");
    require_invalid_argument(
        [&] { (void)ninfer::ops::swa_workspace_capacity_bytes({0, 4096}, 1, 16, too_many); },
        "SWA accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                {0, 4096}, 1, 16, too_many);
        },
        "bidirectional attention accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
                48, too_many, 16);
        },
        "GDN replay accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                profile, ninfer::targets::qwen3::TextPhase::Verify, too_many, 1, 16);
        },
        "target GDN snapshot accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)Variant::gdn_input_projection_record_workspace_capacity_bytes(
                profile, ninfer::targets::qwen3::TextPhase::Verify, too_many, 2, 16);
        },
        "target GDN record accepted an oversized request batch");
}

void qualify_host_split512_routing() {
    namespace kv = ninfer::ops::r9700::kv;
    namespace q27 = ninfer::targets::qwen3_8_27b::detail;
    require(!kv::use_split512_attention(1U, 8191U) &&
                kv::use_split512_attention(1U, 8192U) &&
                !kv::use_split512_attention(1U, 8192U, true) &&
                !kv::use_split512_attention(4U, 8191U) &&
                kv::use_split512_attention(4U, 8192U) &&
                kv::use_split512_attention(4U, 8192U, true) &&
                !kv::use_split512_attention(2U, 32768U) &&
                !kv::use_split512_attention(3U, 32768U) &&
                !kv::use_split512_attention(5U, 32768U),
            "split-512 production selector differs from its admitted boundary");
    require(!kv::use_dense_prefill_attention(127U, 127U) &&
                kv::use_dense_prefill_attention(128U, 128U) &&
                kv::use_dense_prefill_attention(4096U, 4096U) &&
                !kv::use_dense_prefill_attention(8193U, 8193U) &&
                kv::use_dense_prefill_attention(4096U, 8192U) &&
                kv::use_dense_prefill_attention(8192U, 32768U) &&
                kv::use_dense_prefill_attention(128U, 262144U),
            "dense-prefill production selector differs from its admitted boundary");
    require(q27::r9700_full_attention_workspace_capacity_bytes(1U, 8192U, false) ==
                kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(1U, 8192U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(1U, 8192U, true) == 0U &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8192U, false) ==
                    kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, 8192U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8192U, true) ==
                    kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, 8192U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8191U, true) == 0U,
            "split-512 caller-owned workspace selection differs");
    require(q27::r9700_full_attention_workspace_capacity_bytes(2048U, 2048U, false) ==
                kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(2048U, 2048U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4096U, 262144U, false) ==
                kv::fp8_int4_kv_attention_dense_prefill_workspace_envelope_bytes(4096U, 262144U),
            "dense full-score caller-owned planner peak differs");
    const auto has_classes = [](const std::vector<Variant::GraphExecutionProfile>& profiles,
                                std::initializer_list<std::uint32_t> wanted) {
        return std::all_of(wanted.begin(), wanted.end(), [&](std::uint32_t value) {
            return std::any_of(profiles.begin(), profiles.end(), [&](const auto& profile) {
                return profile.topology_class == value;
            });
        });
    };
    const auto class_at = [](const std::vector<Variant::GraphExecutionProfile>& profiles,
                             std::uint32_t frontier) {
        const auto found = std::find_if(profiles.begin(), profiles.end(), [&](const auto& profile) {
            return profile.min <= frontier && frontier <= profile.max;
        });
        if (found == profiles.end()) {
            throw std::runtime_error("split-512 graph profile coverage is incomplete");
        }
        return found->topology_class;
    };
    const auto ordinary = Variant::ordinary_graph_profiles(32768U);
    const auto mtp = Variant::mtp_graph_profiles(32768U, 3U);
    const auto dflash = Variant::dflash_graph_profiles(32768U, 3U, 1U, 4U);
    require(has_classes(ordinary, {0U, 1U}) && has_classes(mtp, {0U, 1U, 3U}) &&
                has_classes(dflash, {1U, 2U}) &&
                class_at(ordinary, 8190U) == 0U && class_at(ordinary, 8191U) == 1U &&
                class_at(mtp, 8185U) == 0U && class_at(mtp, 8186U) == 1U &&
                class_at(mtp, 8187U) == 1U && class_at(mtp, 8188U) == 3U &&
                class_at(dflash, 8187U) == 1U && class_at(dflash, 8188U) == 2U,
            "split-512 Device Graph topology classes are incomplete");
    for (const std::uint32_t width : {5U, 6U}) {
        for (std::uint32_t batch = 1; batch <= 4; ++batch) {
            const auto profiles = Variant::dflash_graph_profiles(16384U, width - 1U, batch, width);
            const std::uint32_t short_class =
                ninfer::targets::qwen3::detail::kR9700TextKVValueGroup == 16 ? 1U : 0U;
            require(class_at(profiles, 8191U - width) == short_class &&
                        class_at(profiles, 8192U - width) == 0U &&
                        class_at(profiles, 16383U) == 0U,
                    "DFlash W5/W6 graph key omitted the short WMMA / long fused boundary");
            const auto tiny = Variant::dflash_graph_profiles(63U, width - 1U, batch, width);
            require(class_at(tiny, 62U) == 0U,
                    "DFlash graph key must use capacity-clipped visible context");
        }
    }
    std::printf("r9700_runtime_planner: PASS host split-512 production routing\n");
}

void qualify_host_attention_parity_routing() {
    namespace kv = ninfer::ops::r9700::kv;
    namespace q27 = ninfer::targets::qwen3_8_27b::detail;
    constexpr std::size_t kW5C134Bytes = 5U * 24U * 134U * sizeof(float);
    constexpr std::size_t kW5C140Bytes = 5U * 24U * 140U * sizeof(float);
    constexpr std::size_t kW6C134Bytes = 6U * 24U * 134U * sizeof(float);
    constexpr std::size_t kW6C135Bytes = 6U * 24U * 135U * sizeof(float);
    constexpr std::size_t kTextP129Bytes = (129U * 24U * 129U + 129U * 24U) * sizeof(float);
    const bool text_enabled = kv::kTextP129WmmaTailCandidate;
    for (const std::size_t context : {64U, 133U, 4100U, 8191U}) {
        require(kv::use_dflash_verify_batched_wmma(4U, context, false, true) &&
                    q27::r9700_full_attention_workspace_capacity_bytes(4U, context, false, true) ==
                        4U * 24U * context * sizeof(float),
                "DFlash W4 route or score workspace omitted an admitted context");
    }
    require(!kv::use_dflash_verify_batched_wmma(4U, 63U, false, true) &&
                !kv::use_dflash_verify_batched_wmma(4U, 8192U, false, true) &&
                !kv::use_dflash_verify_batched_wmma(4U, 134U, true, true) &&
                !kv::use_dflash_verify_batched_wmma(4U, 134U, false, false),
            "DFlash W4 route escaped context/tree/caller admission");
    require(kv::use_dflash_verify_batched_wmma(5U, 134U, false, true) &&
                kv::use_dflash_verify_batched_wmma(6U, 135U, false, true) &&
                kv::use_text_p129_wmma_tail(129U, 129U) == text_enabled &&
                !kv::use_dflash_verify_batched_wmma(5U, 134U, false, false) &&
                !kv::use_dflash_verify_batched_wmma(6U, 135U, false, false) &&
                kv::use_dflash_verify_batched_wmma(4U, 134U, false, true) &&
                !kv::use_dflash_verify_batched_wmma(3U, 134U, false, true) &&
                !kv::use_dflash_verify_batched_wmma(7U, 135U, false, true) &&
                !kv::use_dflash_verify_batched_wmma(5U, 134U, true, true) &&
                !kv::use_dflash_verify_batched_wmma(6U, 135U, true, true) &&
                !kv::use_text_p129_wmma_tail(128U, 128U) &&
                !kv::use_text_p129_wmma_tail(129U, 130U),
            "attention parity selectors escaped their exact Text/DFlash cells");
    require(q27::r9700_full_attention_workspace_capacity_bytes(5U, 134U, false, false) == 0U &&
                q27::r9700_full_attention_workspace_capacity_bytes(5U, 134U, false, true) ==
                    kW5C134Bytes &&
                q27::r9700_full_attention_workspace_capacity_bytes(5U, 140U, false, true) ==
                    kW5C140Bytes &&
                q27::r9700_full_attention_workspace_capacity_bytes(5U, 8191U, false, true) ==
                    5U * 24U * 8191U * sizeof(float) &&
                q27::r9700_full_attention_workspace_capacity_bytes(5U, 134U, true, true) == 0U &&
                q27::r9700_full_attention_workspace_capacity_bytes(6U, 134U, false, true) ==
                    kW6C134Bytes &&
                q27::r9700_full_attention_workspace_capacity_bytes(6U, 135U, false, true) ==
                    kW6C135Bytes &&
                q27::r9700_full_attention_workspace_capacity_bytes(6U, 135U, true, true) == 0U,
            "DFlash W4..6 production workspace escaped width/tree selection");
    require(q27::r9700_full_attention_workspace_capacity_bytes(129U, 129U, false) ==
                kTextP129Bytes,
            "Text P129 tail candidate changed the dense caller-owned workspace peak");
    std::printf("r9700_runtime_planner: PASS host attention parity routing/capacity "
                "text_p129_selector=%u\n",
                text_enabled ? 1U : 0U);
}

void qualify_host_ordinary_graph_allowance() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    runtime::SequencePlanningInputs inputs{};
    inputs.weights_profile =
        Variant::WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation;
    inputs.prefill_chunk = 2048U;
    inputs.speculative_backend = ninfer::SpeculativeBackend::None;
    inputs.features = {.vision = false, .speculative = ninfer::SpeculativeBackend::None};
    for (const std::uint32_t capacity : {1024U, 4096U, 4224U}) {
        const std::size_t definitions = capacity == 1024U ? 3U : capacity == 4096U ? 4U : 5U;
        for (const std::uint32_t concurrency : {1U, 2U, 3U, 4U}) {
            inputs.capacity = capacity;
            inputs.max_concurrency = concurrency;
            inputs.use_device_graph = true;
            const auto page_size = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
            const std::uint32_t physical_pages = concurrency *
                ((capacity + page_size - 1U) / page_size);
            const auto plan = runtime::build_sequence_candidate_for_qualification(
                inputs, physical_pages);
            const std::size_t expected = (23ULL + concurrency *
                (24ULL + 4ULL * (definitions - 3U))) * 1024ULL * 1024ULL;
            require(plan->graph_definition_count == definitions * concurrency &&
                        plan->graph_executable_count == concurrency &&
                        plan->graph_allowance_bytes == expected,
                    "ordinary allowance omitted same-topology profile updates");
            if (capacity == 4224U && concurrency == 1U) {
                require(plan->graph_allowance_bytes >= 54525952ULL,
                        "ordinary allowance misses observed 52 MiB tiled preparation");
            }
            inputs.use_device_graph = false;
            const auto eager = runtime::build_sequence_candidate_for_qualification(
                inputs, physical_pages);
            require(eager->graph_allowance_bytes == 0U &&
                        eager->graph_definition_count == 0U &&
                        eager->graph_executable_count == 0U,
                    "eager ordinary plan reserves graph residency");
        }
    }
    std::printf("r9700_runtime_planner: PASS host ordinary C1..4/context1024,4096,4224 allowance\n");
}

void qualify_host_hybrid_allocation_bound() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    for (const auto profile : {Variant::WeightsProfile::R9700Q4G64Evaluation,
                              Variant::WeightsProfile::R9700Q4W8Evaluation,
                              Variant::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation}) {
        runtime::SequencePlanningInputs inputs{};
        inputs.weights_profile = profile;
        inputs.capacity = 262144U;
        inputs.max_concurrency = 4U;
        inputs.prefill_chunk = 2048U;
        inputs.speculative_backend = ninfer::SpeculativeBackend::None;
        inputs.features = {.vision = false, .speculative = ninfer::SpeculativeBackend::None};
        inputs.use_device_graph = true;
        const auto minimum = runtime::build_sequence_candidate_for_qualification(inputs, 4096U);
        const auto adjacent = runtime::build_sequence_candidate_for_qualification(inputs, 4097U);
        const auto distant = runtime::build_sequence_candidate_for_qualification(inputs, 4233U);
        const std::size_t logical = minimum->persistent.bytes + minimum->workspace.capacity +
            minimum->request_transient_capacity_bytes + minimum->graph_allowance_bytes;
        const auto expected_bound = profile ==
            Variant::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation ? 4U << 20U : 0U;
        require(minimum->device_reservation_bytes == logical + expected_bound,
                "hybrid physical arena bound missing or changed nonhybrid reservation");
        const auto increment = adjacent->device_reservation_bytes - minimum->device_reservation_bytes;
        require(distant->device_reservation_bytes == minimum->device_reservation_bytes + 137U * increment,
                "physical arena bound made the KV reservation curve non-affine");
    }
    std::printf("r9700_runtime_planner: PASS host hybrid arena bound/nonhybrid affine capacity\n");
}

void qualify_host_dflash_graph_allowance() {
    namespace runtime =
        ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    constexpr std::size_t expected_classes =
        ninfer::targets::qwen3::detail::kR9700TextKVValueGroup == 16 ? 2U : 1U;
    std::size_t expected_allowance =
        (48ULL + 26ULL * expected_classes) * 1024ULL * 1024ULL;
    const runtime::SequencePlanningInputs inputs{
        .weights_profile =
            Variant::WeightsProfile::R9700Q4G64DFlash2Q4Evaluation,
        .capacity = 8459U,
        .max_concurrency = 1U,
        .prefill_chunk = 4096U,
        .draft_window = 4U,
        .dflash_verify_width = 5U,
        .speculative_backend = ninfer::SpeculativeBackend::DFlash,
        .proposal_head = ninfer::ProposalHead::Optimized,
        .features = {.vision = false,
                     .speculative = ninfer::SpeculativeBackend::DFlash,
                     .proposal_head = ninfer::ProposalHead::Optimized},
        .use_device_graph = true,
        .device = 0,
        .kv_ram_capacity_bytes = 0U,
        .context_checkpoint_marks = {},
    };
    const auto plan = runtime::build_sequence_candidate_for_qualification(inputs, 133U);
    require(plan->capacity == 8459U && plan->kv_capacity == 8512U,
            "DFlash C1 authority geometry changed");
    const auto profiles = Variant::dflash_graph_profiles(8459U, 4U, 1U, 5U);
    std::vector<std::uint32_t> topology_classes;
    for (const auto profile : profiles) {
        if (std::find(topology_classes.begin(), topology_classes.end(),
                      profile.topology_class) == topology_classes.end()) {
            topology_classes.push_back(profile.topology_class);
        }
    }
    for (const auto topology : topology_classes) {
        const auto count = static_cast<std::size_t>(std::count_if(profiles.begin(), profiles.end(),
            [&](const auto& p) { return p.topology_class == topology; }));
        if (count > 3U) expected_allowance += (count - 3U)*4ULL*1024ULL*1024ULL;
    }
    require(plan->dflash_verify_width == 5U &&
                plan->graph_definition_count == profiles.size() &&
                plan->graph_executable_count == topology_classes.size() &&
                topology_classes.size() == expected_classes,
            "DFlash C1 K4/W5 graph topology inventory changed");
    require(plan->graph_allowance_bytes == expected_allowance,
            "DFlash C1 K4/W5 allowance must reserve each distinct executable topology");
    // Reproduce the selective-protected context 1024 startup geometry whose
    // complete preparation measured 71 MiB with selected BF16 staging and tiled W8 head.
    for(const std::uint32_t k : {4U,5U}) {
        auto measured_inputs=inputs;
        measured_inputs.weights_profile=
            Variant::WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation;
        measured_inputs.capacity=1024U;
        measured_inputs.draft_window=k;
        measured_inputs.dflash_verify_width=k+1U;
        const auto measured=runtime::build_sequence_candidate_for_qualification(measured_inputs,16U);
        require(measured->graph_executable_count==1U &&
                    measured->graph_allowance_bytes==74ULL*1024ULL*1024ULL &&
                    measured->graph_allowance_bytes>=74448896ULL,
                "DFlash selective-protected graph allowance misses measured startup residency");
    }
    // Same executable, more frontier definitions: preserve the 1K calibration and
    // account for additional installs at the reproduced compact-companion geometry.
    for (const std::uint32_t concurrency : {1U, 2U, 3U, 4U}) {
        for (const std::uint32_t capacity : {1024U, 4096U, 4240U}) {
            for (const std::uint32_t k : {4U, 5U}) {
                auto compact_inputs = inputs;
                compact_inputs.weights_profile =
                    Variant::WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation;
                compact_inputs.capacity = capacity;
                compact_inputs.prefill_chunk = 2048U;
                compact_inputs.max_concurrency = concurrency;
                compact_inputs.draft_window = k;
                compact_inputs.dflash_verify_width = k + 1U;
                const auto compact = runtime::build_sequence_candidate_for_qualification(
                    compact_inputs, ((capacity + 63U) / 64U) * concurrency);
                const std::size_t definitions_per_batch = capacity == 1024U ? 3U : 5U;
                const std::size_t expected = (48ULL + concurrency *
                    (26ULL + 4ULL*(definitions_per_batch - 3ULL))) * 1024ULL*1024ULL;
                require(compact->graph_executable_count == concurrency &&
                            compact->graph_definition_count == definitions_per_batch*concurrency &&
                            compact->graph_allowance_bytes == expected,
                        "DFlash compact companion omitted exact-B frontier updates");
                if (capacity == 4240U && concurrency == 1U)
                    require(compact->graph_allowance_bytes >= 78643200ULL,
                            "DFlash compact companion misses measured 75-MiB preparation");
                compact_inputs.use_device_graph = false;
                const auto eager = runtime::build_sequence_candidate_for_qualification(
                    compact_inputs, ((capacity + 63U) / 64U) * concurrency);
                require(eager->graph_allowance_bytes == 0U && eager->graph_definition_count == 0U &&
                            eager->graph_executable_count == 0U,
                        "eager compact companion must not reserve graph updates");
            }
        }
    }
    std::printf("r9700_runtime_planner: PASS host DFlash C1-C4 K4/K5 graph topology/update allowance\n");
}
void qualify_host_mtp_graph_allowance() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    struct Case {
        std::uint32_t context, concurrency, drafts;
        bool adaptive;
        std::size_t definitions, executables, updates;
    };
    // Includes singleton (no update), the reproduced short-context failure, full
    // context profiles/split topologies, and all three independently captured Ks.
    for (const auto c : {
             Case{64, 1, 3, false, 1, 1, 0},
             Case{1024, 1, 3, false, 3, 1, 3},
             Case{1024, 4, 5, false, 12, 4, 12},
             Case{262144, 1, 3, false, 11, 3, 11},
             Case{262144, 4, 5, false, 32, 4, 32},
             Case{262144, 4, 5, true, 108, 20, 108}}) {
        runtime::SequencePlanningInputs inputs{
            .weights_profile = Variant::WeightsProfile::R9700Q4G64Evaluation,
            .capacity = c.context,
            .max_concurrency = c.concurrency,
            .prefill_chunk = std::min(c.context, 2048U),
            .draft_window = c.drafts,
            .adaptive_draft = c.adaptive,
            .speculative_backend = ninfer::SpeculativeBackend::Mtp,
            .proposal_head = ninfer::ProposalHead::Full,
            .features = {.vision = false, .speculative = ninfer::SpeculativeBackend::Mtp,
                         .proposal_head = ninfer::ProposalHead::Full},
            .use_device_graph = true,
        };
        const auto plan = runtime::build_sequence_candidate_for_qualification(
            inputs, c.concurrency * ((c.context + 63U) / 64U));
        require(plan->graph_definition_count == c.definitions &&
                    plan->graph_executable_count == c.executables,
                "MTP fixed/adaptive graph inventory differs from its captured K/B profiles");
        const auto expected = (46ULL + 4ULL * (c.executables + c.updates)) * 1024ULL * 1024ULL;
        require(plan->graph_allowance_bytes == expected,
                "MTP allowance omitted profile update/restore residency");
        inputs.use_device_graph = false;
        const auto eager = runtime::build_sequence_candidate_for_qualification(
            inputs, c.concurrency * ((c.context + 63U) / 64U));
        require(eager->graph_allowance_bytes == 0 && eager->graph_definition_count == 0 &&
                    eager->graph_executable_count == 0,
                "eager MTP reserves graph residency");
    }
    std::printf("r9700_runtime_planner: PASS host MTP graph instantiate/update allowance\n");
}

using WeightsProfile = ninfer::targets::qwen3_8_27b::detail::WeightsProfile;

constexpr std::uint32_t kCapacityEnvelopeContext = 262144U;
constexpr std::uint32_t kCapacityEnvelopeChunk   = 4096U;
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && \
    NINFER_R9700_XATTENTION_STRIDE == 16 && NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
constexpr std::size_t kCapacityEnvelopeSparseLeafBytes = 601361408ULL;
constexpr std::size_t kCapacityEnvelopeWorkspaceBytes  = 1085967619ULL;
#endif

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::size_t qualify_plan(ninfer::DeviceContext& device, std::uint32_t concurrency,
                         ninfer::SpeculativeBackend backend, std::uint32_t drafts,
                         bool vision, std::uint32_t dflash_verify_width = 0,
                         WeightsProfile weights_profile =
                             WeightsProfile::R9700W8G32Candidate,
                         bool use_device_graph = false, std::uint32_t max_context = 2048,
                         std::uint32_t prefill_chunk = 128) {
    ninfer::EngineOptions options;
    options.device            = device.device;
    options.max_context       = max_context;
    options.kv_capacity       = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency   = concurrency;
    options.prefill_chunk     = prefill_chunk;
    options.enable_vision     = vision;
    options.use_device_graph  = use_device_graph;
    options.speculative.backend      = backend;
    options.speculative.draft_tokens = drafts;
    options.speculative.dflash_verify_width = dflash_verify_width;

    auto planner = ninfer::targets::qwen3::make_sequence_planner<Variant>(
        device, options, weights_profile);
    const auto curve = planner.capacity_curve();
    require(curve.main_page_tokens == 64U, "planner page size is not the typed-cache page size");
    const std::uint32_t logical_pages = 1U + (max_context - 1U) / curve.main_page_tokens;
    require(curve.minimum_main_page_groups == std::max(logical_pages, concurrency),
            "planner minimum physical page count is wrong");
    require(curve.maximum_main_page_groups == concurrency * logical_pages,
            "planner maximum physical page count is wrong");
    require(curve.minimum_device_reservation_bytes != 0U,
            "planner produced an empty device reservation");

    auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);
    require(plan.capacity() == options.max_context, "plan logical capacity changed");
    require(plan.kv_capacity() >= options.max_context &&
                plan.kv_capacity() - options.max_context < curve.main_page_tokens,
            "plan physical KV capacity does not match page-rounded logical capacity");
    require(plan.max_concurrency() == concurrency, "plan concurrency changed");
    require(plan.device_reservation_bytes() == curve.minimum_device_reservation_bytes,
            "plan reservation disagrees with its affine curve");
    require(plan.workspace_capacity_bytes() != 0U, "plan workspace is empty");
    require((plan.request_transient_capacity_bytes() != 0U) == vision,
            "plan Vision transient ownership is wrong");
    // Rebuild the runtime allocation prefix through ordinary Text prefill attention. Comparing
    // only the leaf scratch size with the global arena would miss live query/gate/K/V/result
    // tensors that precede Variant::full_attention in the same scope.
    {
        using TextConfig = Variant::TextConfig;
        const std::int32_t chunk = static_cast<std::int32_t>(
            std::min<std::uint32_t>(options.prefill_chunk, options.max_context));
        ninfer::WorkspaceLayoutBuilder runtime_prefix;
        (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
            text_prefill_roots<TextConfig>(runtime_prefix, chunk, vision ? 3 : 0,
                                           vision ? chunk : 0);
        {
            auto stage = runtime_prefix.scope();
            (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
                text_attention_projection<TextConfig>(runtime_prefix, chunk);
            {
                auto scratch = runtime_prefix.scope();
                const std::size_t bytes = Variant::attention_projection_workspace_capacity_bytes(
                    weights_profile, ninfer::targets::qwen3::TextPhase::Prefill, 1, chunk);
                if (bytes != 0U) (void)runtime_prefix.alloc_bytes(bytes);
            }
            (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
                text_attention_results<TextConfig>(runtime_prefix, chunk);
            {
                auto scratch = runtime_prefix.scope();
                const std::size_t bytes =
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
                    Variant::text_prefill_attention_workspace_capacity_bytes(
                        chunk, options.max_context);
#else
                    Variant::full_attention_workspace_capacity_bytes(
                        chunk, options.max_context, false);
#endif
                if (bytes != 0U) (void)runtime_prefix.alloc_bytes(bytes);
            }
        }
        require(plan.impl_->workspace.text_prefill >= runtime_prefix.peak_bytes(1),
                "Text-prefill plan does not cover its runtime attention allocation prefix");
    }
    const std::uint32_t expected_dflash_width =
        backend == ninfer::SpeculativeBackend::DFlash
            ? ninfer::targets::qwen3::dflash_verify_width<Variant::DFlashConfig>(
                  drafts, dflash_verify_width)
            : 0U;
    require(plan.impl_->dflash_verify_width == expected_dflash_width,
            "plan did not preserve its fixed DFlash verify width");
    if (use_device_graph && backend == ninfer::SpeculativeBackend::Mtp) {
        constexpr std::size_t kMiB                     = 1024ULL * 1024ULL;
        constexpr std::size_t kMtpGraphFamilyBytes     = 46ULL * kMiB;
        const auto profiles = Variant::mtp_graph_profiles(max_context, drafts);
        const std::size_t definitions =
            profiles.size() * static_cast<std::size_t>(concurrency);
        std::vector<std::uint32_t> topology_classes;
        for (std::uint32_t batch_size = 1; batch_size <= concurrency; ++batch_size) {
            for (const auto profile : profiles) {
                const std::uint32_t folded =
                    profile.topology_class * concurrency + (batch_size - 1U);
                if (std::find(topology_classes.begin(), topology_classes.end(), folded) ==
                    topology_classes.end()) {
                    topology_classes.push_back(folded);
                }
            }
        }
        require(plan.impl_->graph_definition_count == definitions,
                "MTP graph plan omitted a reachable exact-B/profile definition");
        require(plan.impl_->graph_executable_count == topology_classes.size(),
                "MTP graph plan executable inventory does not match reachable topologies");
        std::size_t operations = topology_classes.size();
        for (const auto topology : topology_classes) {
            const auto count = std::count_if(profiles.begin(), profiles.end(), [&](const auto& p) {
                return p.topology_class == topology / concurrency;
            });
            if (count > 1) operations += static_cast<std::size_t>(count);
        }
        require(plan.impl_->graph_allowance_bytes ==
                    kMtpGraphFamilyBytes + 4ULL * kMiB * operations,
                "MTP graph allowance does not match its ROCm family/executable inventory");
    }
    if (use_device_graph && backend == ninfer::SpeculativeBackend::None) {
        constexpr std::size_t kMiB                          = 1024ULL * 1024ULL;
        constexpr std::size_t kOrdinaryGraphFamilyBytes     = 23ULL * kMiB;
        constexpr std::size_t kOrdinaryGraphExecutableBytes = 24ULL * kMiB;
        const auto profiles = Variant::ordinary_graph_profiles(max_context);
        const std::size_t definitions =
            profiles.size() * static_cast<std::size_t>(concurrency);
        std::vector<std::uint32_t> topology_classes;
        for (std::uint32_t batch_size = 1; batch_size <= concurrency; ++batch_size) {
            for (const auto profile : profiles) {
                const std::uint32_t folded =
                    profile.topology_class * concurrency + (batch_size - 1U);
                if (std::find(topology_classes.begin(), topology_classes.end(), folded) ==
                    topology_classes.end()) {
                    topology_classes.push_back(folded);
                }
            }
        }
        require(plan.impl_->graph_definition_count == definitions,
                "ordinary graph plan omitted a reachable exact-B/profile definition");
        require(plan.impl_->graph_executable_count == topology_classes.size(),
                "ordinary graph plan executable inventory does not match reachable topologies");
        std::size_t update_bytes = 0U;
        for (const auto topology : topology_classes) {
            const std::size_t count = static_cast<std::size_t>(std::count_if(
                profiles.begin(), profiles.end(), [&](const auto profile) {
                    return profile.topology_class == topology / concurrency;
                }));
            if (count > 3U) update_bytes += (count - 3U) * 4ULL * kMiB;
        }
        require(plan.impl_->graph_allowance_bytes ==
                    kOrdinaryGraphFamilyBytes +
                        kOrdinaryGraphExecutableBytes * topology_classes.size() + update_bytes,
                "ordinary graph allowance does not match its ROCm family/executable inventory");
    }
    if (use_device_graph && backend == ninfer::SpeculativeBackend::DFlash) {
        constexpr std::size_t kMiB                   = 1024ULL * 1024ULL;
        constexpr std::size_t kDFlashFamilyBytes     = 48ULL * kMiB;
        constexpr std::size_t kDFlashExecutableBytes = 26ULL * kMiB;
        constexpr std::size_t kDFlashK1FusedBytes    = 10ULL * kMiB;
        constexpr std::size_t kDFlashK1WmmaBytes     = 18ULL * kMiB;
        std::size_t definitions = 0;
        std::vector<std::uint32_t> topology_classes;
        std::size_t expected_allowance = kDFlashFamilyBytes;
        for (std::uint32_t batch_size = 1; batch_size <= concurrency; ++batch_size) {
            const auto profiles = Variant::dflash_graph_profiles(
                max_context, drafts, batch_size, expected_dflash_width);
            definitions += profiles.size();
            for (const auto profile : profiles) {
                const std::uint32_t folded =
                    profile.topology_class * concurrency + (batch_size - 1U);
                if (std::find(topology_classes.begin(), topology_classes.end(), folded) !=
                    topology_classes.end()) {
                    continue;
                }
                topology_classes.push_back(folded);
                if (drafts != 1U) {
                    expected_allowance += kDFlashExecutableBytes;
                    const auto count = static_cast<std::size_t>(std::count_if(
                        profiles.begin(), profiles.end(), [&](const auto& p) {
                            return p.topology_class == profile.topology_class;
                        }));
                    if (count > 3U) expected_allowance += (count - 3U)*4ULL*kMiB;
                } else if (profile.topology_class == 0U) {
                    expected_allowance += kDFlashK1FusedBytes;
                } else if (profile.topology_class == 1U) {
                    expected_allowance += kDFlashK1WmmaBytes;
                } else {
                    throw std::runtime_error("unexpected DFlash K=1 topology class");
                }
            }
        }
        require(plan.impl_->graph_definition_count == definitions,
                "DFlash graph plan omitted a reachable exact-B/profile definition");
        require(plan.impl_->graph_executable_count == topology_classes.size(),
                "DFlash graph plan executable inventory does not match reachable topologies");
        require(plan.impl_->graph_allowance_bytes == expected_allowance,
                "DFlash graph allowance does not match its ROCm topology inventory");
    }
    return plan.workspace_capacity_bytes();
}

void qualify_host_xattention_capacity_envelope() {
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && \
    NINFER_R9700_XATTENTION_STRIDE == 16 && NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
    const std::size_t sparse_leaf = Variant::text_prefill_attention_workspace_capacity_bytes(
        static_cast<std::int32_t>(kCapacityEnvelopeChunk),
        static_cast<std::int32_t>(kCapacityEnvelopeContext));
    require(sparse_leaf == kCapacityEnvelopeSparseLeafBytes,
            "S16/tau900 Text-prefill sparse leaf workspace envelope changed");
    std::printf(
        "r9700_runtime_planner: PASS host XAttention S16/tau900 T=%u context=%u "
        "sparse_leaf_bytes=%zu expected_planner_workspace_bytes=%zu\n",
        kCapacityEnvelopeChunk, kCapacityEnvelopeContext, sparse_leaf,
        kCapacityEnvelopeWorkspaceBytes);
#else
    throw std::runtime_error(
        "host XAttention capacity envelope requires the S16/tau900 qualification build");
#endif
}

void print_host_hybrid_capacity_authority() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    constexpr std::uint32_t capacity = 262144U;
    constexpr std::uint32_t page_tokens = 64U;
    constexpr std::uint32_t minimum_groups = capacity / page_tokens;
    std::printf("target,weights_profile,capacity_tokens,page_tokens,prefill_chunk,kv_value_group,"
                "speculative_backend,draft_tokens,proposal_head,device_graph,"
                "concurrency,minimum_groups,maximum_groups,minimum_sequence_bytes,workspace_bytes,"
                "graph_allowance_bytes,request_transient_bytes,minimum_reservation_bytes,"
                "kv_payload_bytes,kv_increment_bytes\n");
    for (std::uint32_t concurrency = 1; concurrency <= ninfer::kMaximumConcurrency;
         ++concurrency) {
        const runtime::SequencePlanningInputs inputs{
            .weights_profile = WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation,
            .capacity = capacity,
            .max_concurrency = concurrency,
            .prefill_chunk = 8192U,
            .draft_window = 3U,
            .dflash_verify_width = 0U,
            .speculative_backend = ninfer::SpeculativeBackend::Mtp,
            .proposal_head = ninfer::ProposalHead::Optimized,
            .features = {.vision = false,
                         .speculative = ninfer::SpeculativeBackend::Mtp,
                         .proposal_head = ninfer::ProposalHead::Optimized},
            .use_device_graph = true,
            .device = 0,
            .kv_ram_capacity_bytes = 0U,
            .context_checkpoint_marks = {},
        };
        const auto minimum = runtime::build_sequence_candidate_for_qualification(
            inputs, minimum_groups);
        const auto adjacent = runtime::build_sequence_candidate_for_qualification(
            inputs, minimum_groups + 1U);
        const std::size_t increment =
            adjacent->device_reservation_bytes - minimum->device_reservation_bytes;
        std::printf("qwen3_8_27b_r9700,R9700Q4G64Fp8FourRoleN16K16Evaluation,"
                    "262144,64,8192,16,mtp,3,optimized,1,"
                    "%u,%u,%u,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", concurrency,
                    minimum_groups, concurrency * minimum_groups, minimum->persistent.bytes,
                    minimum->workspace.capacity, minimum->graph_allowance_bytes,
                    minimum->request_transient_capacity_bytes,
                    minimum->device_reservation_bytes, minimum->persistent.kv_payload_bytes,
                    increment);
    }
}

std::uint32_t parse_host_width_argument(const char* text, const char* label) {
    std::uint32_t value = 0;
    const std::string_view input{text};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size()) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return value;
}

void print_host_hybrid_width_authority(int argc, char** argv) {
    if (argc != 6) {
        throw std::invalid_argument(
            "--host-hybrid-widths-csv requires PREFILL MAX_CONCURRENCY MTP_WIDTH DFLASH_WIDTH");
    }
    const std::uint32_t prefill = parse_host_width_argument(argv[2], "prefill width");
    const std::uint32_t concurrency =
        parse_host_width_argument(argv[3], "maximum concurrency");
    const std::uint32_t mtp_width = parse_host_width_argument(argv[4], "MTP width");
    const std::uint32_t dflash_width = parse_host_width_argument(argv[5], "DFlash width");
    if (concurrency > ninfer::kMaximumConcurrency) {
        throw std::invalid_argument("hybrid width inventory exceeds product concurrency");
    }
    const std::uint32_t verify_widths[]{mtp_width, dflash_width};
    const auto widths = Variant::ExecutionState::eager_widths(
        prefill, concurrency, verify_widths);
    std::printf("prefill,max_concurrency,mtp_width,dflash_width,prepared_width\n");
    for (const std::uint32_t width : widths) {
        std::printf("%u,%u,%u,%u,%u\n", prefill, concurrency, mtp_width, dflash_width,
                    width);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--host-xattention-capacity-envelope") {
            qualify_host_xattention_capacity_envelope();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-request-lane-cap") {
            qualify_host_request_lane_cap();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-split512-routing") {
            qualify_host_split512_routing();
            qualify_host_hybrid_allocation_bound();
            qualify_host_ordinary_graph_allowance();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-dflash-graph-allowance") {
            qualify_host_dflash_graph_allowance();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-mtp-graph-allowance") {
            qualify_host_mtp_graph_allowance();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-attention-parity-routing") {
            qualify_host_attention_parity_routing();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-hybrid-capacity-csv") {
            print_host_hybrid_capacity_authority();
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--host-hybrid-widths-csv") {
            print_host_hybrid_width_authority(argc, argv);
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "usage: ninfer_r9700_runtime_planner_qual "
                "[--host-xattention-capacity-envelope|--host-request-lane-cap|"
                "--host-split512-routing|--host-dflash-graph-allowance|--host-mtp-graph-allowance|"
                "--host-attention-parity-routing|"
                "--host-hybrid-capacity-csv|"
                "--host-hybrid-widths-csv PREFILL MAX_CONCURRENCY MTP_WIDTH DFLASH_WIDTH]");
        }
        ninfer::DeviceContext device(0);
        for (std::uint32_t concurrency = 1; concurrency <= ninfer::kMaximumConcurrency;
             ++concurrency) {
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, true);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, true);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 4, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 5, false, 6);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, false, 0,
                         WeightsProfile::R9700Q4G64Evaluation, true, 32768);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, false, 0,
                         WeightsProfile::R9700Q4G64Evaluation, true, 32774);
            if (concurrency == 1U) {
                qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 1, false, 2,
                             WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, true, 128);
            }
            for (std::uint32_t k = 1; k <= 5; ++k) {
                qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, k, false, 0,
                             WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, true,
                             32768U + k + 1U);
            }
        }
        const std::size_t w8_workspace = qualify_plan(
            device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0, true);
        const std::size_t mixed_workspace = qualify_plan(
            device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0, true, 0,
            WeightsProfile::R9700Q4W8Evaluation);
        const std::size_t q4_workspace = qualify_plan(
            device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0, true, 0,
            WeightsProfile::R9700Q4G64Evaluation);
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && \
    NINFER_R9700_XATTENTION_STRIDE == 16 && NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
        const std::size_t capacity_envelope_workspace = qualify_plan(
            device, 1, ninfer::SpeculativeBackend::Mtp, 3, false, 0,
            WeightsProfile::R9700Q4G64Evaluation, true, kCapacityEnvelopeContext,
            kCapacityEnvelopeChunk);
        require(capacity_envelope_workspace == kCapacityEnvelopeWorkspaceBytes,
                "S16/tau900 4K/262K global planner workspace envelope changed");
#endif
        (void)qualify_plan(device, ninfer::kMaximumConcurrency,
                           ninfer::SpeculativeBackend::DFlash, 5, false, 6,
                           WeightsProfile::R9700Q4G64DFlash2Q4Evaluation);
        (void)qualify_plan(device, ninfer::kMaximumConcurrency,
                           ninfer::SpeculativeBackend::DFlash, 5, false, 6,
                           WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation);
        const std::size_t dflash_q4_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, 128);
        const std::size_t dflash_mixed_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation, 128);
        const std::size_t base_q4_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4G64Evaluation, 128);
        require(Variant::dflash_matrix_qtype(
                    WeightsProfile::R9700Q4G64DFlash2Q4Evaluation) ==
                    ninfer::QType::Q4G64_F16S &&
                    Variant::dflash_matrix_qtype(
                        WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation) ==
                        ninfer::QType::Q4G64_F16S,
                "DFlash evaluation profile did not select persistent Q4 child matrices");
        require(dflash_q4_linear == dflash_mixed_linear,
                "DFlash Q4 companions disagree on their activation workspace");
        require(dflash_q4_linear > base_q4_linear,
                "DFlash feature projection did not enlarge Q4 activation workspace to K=25600");
        if constexpr (ninfer::ops::r9700::linear::kW8ActivationBits == 8) {
            require(w8_workspace == mixed_workspace,
                    "A8 W8 and mixed plans disagree on their largest activation image");
            require(w8_workspace > q4_workspace,
                    "A8G32 W8 plan did not reserve its larger scale image");
        } else {
            require(mixed_workspace > w8_workspace,
                    "mixed Q4/W8 plan did not reserve caller-owned Q4 activation workspace");
            require(q4_workspace > mixed_workspace,
                    "all-Q4 plan did not reserve its larger MLP/Vision activation workspace");
        }
        std::printf(
            "r9700_runtime_planner: PASS q4_activation_bits=%d q4_prefill_cta_profile=%.*s "
            "w8_activation_bits=%d C=1..4 "
            "ordinary/vision/MTP/DFlash fixed widths; 32K MTP3 graph allowance; "
            "exact 32K DFlash K=1..11 C=1..4 graph inventories; profile-consistent integer "
            "workspace including DFlash Q4 companions\n",
            ninfer::ops::r9700::linear::kQ4ActivationBits,
            static_cast<int>(ninfer::ops::r9700::linear::kQ4PrefillCtaProfile.size()),
            ninfer::ops::r9700::linear::kQ4PrefillCtaProfile.data(),
            ninfer::ops::r9700::linear::kW8ActivationBits);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "r9700_runtime_planner: FAIL: %s\n", error.what());
        return 1;
    }
}
