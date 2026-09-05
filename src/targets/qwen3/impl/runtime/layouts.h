#pragma once
#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/r9700_cache_profile.h"
// Qwen3 family runtime implementation; instantiated only by exact variants.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3/decoder_state.h>
#include <ninfer/targets/qwen3/round_state.h>
#include <ninfer/targets/qwen3/startup_features.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {

using TensorLayout = TensorRegion;

struct DFlashPersistentLayout {
    CyclicKVCacheLayout local;
    CyclicKVCacheLayout rewrite_checkpoint_local;
    CyclicKVCacheLayout staging_local;
    // DFlash Full attention is a distinct BF16 state contract. It must never inherit the Text/MTP
    // FP8-K/INT4-V codec merely because both states use physical pages.
    struct FullBF16Layout {
        PagedKVPoolLayout storage;
        std::uint32_t layers      = 0;
        std::uint32_t max_context = 0;
        std::int32_t kv_heads     = 0;
        std::int32_t head_dim     = 0;

        [[nodiscard]] std::size_t payload_bytes() const noexcept {
            return storage.payload_bytes();
        }
    };
    std::optional<FullBF16Layout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return local.payload_bytes() + rewrite_checkpoint_local.payload_bytes() +
               staging_local.payload_bytes() + (full ? full->payload_bytes() : 0);
    }
};

struct PersistentLayout {
    qwen3::DecoderStateLayout decoder;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3::RoundStateLayout round;
    TensorLayout prefill_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    TensorLayout tail_hidden;
    TensorLayout rewrite_checkpoint_hidden;
    std::optional<TensorLayout> staging_hidden;
    // One Program-lifetime region shared by serialized integer and selected FP8 projections. It
    // is deliberately outside WorkspaceArena so graph-captured activation addresses never alias
    // schedule scratch.
    std::optional<LayoutRegion> linear_execution;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill   = 0;
    std::size_t ordinary_round = 0;
    std::size_t mtp_prefill    = 0;
    std::size_t mtp_round      = 0;
    std::size_t dflash_context = 0;
    std::size_t dflash_round   = 0;
    std::size_t vision_encode  = 0;
    std::size_t capacity       = 0;
};

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    std::uint32_t dflash_verify_width      = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_device_graph = true;
    int device          = 0;
    std::size_t kv_ram_capacity_bytes = 0;
    std::vector<std::uint32_t> context_checkpoint_marks;
};

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS

namespace ninfer::targets::qwen3::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN3_VARIANT> {
    typename NINFER_QWEN3_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    std::uint32_t dflash_verify_width      = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_device_graph = true;
    int device          = 0;
    std::size_t kv_ram_capacity_bytes = 0;
    std::vector<std::uint32_t> context_checkpoint_marks;
    NINFER_QWEN3_RUNTIME_NS::PersistentLayout persistent;
    NINFER_QWEN3_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t request_transient_capacity_bytes = 0;
    std::size_t graph_definition_count            = 0;
    std::size_t graph_executable_count            = 0;
    std::size_t graph_allowance_bytes            = 0;
    std::size_t device_reservation_bytes         = 0;
};

template <>
struct SequencePlannerImpl<NINFER_QWEN3_VARIANT> {
    NINFER_QWEN3_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl<NINFER_QWEN3_VARIANT>> minimum;
};

} // namespace ninfer::targets::qwen3::detail

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {

using SequencePlanImpl = qwen3::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen3::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups);
// CPU-only qualification seam over the exact production layout builder. It performs no device
// query or allocation and is used to validate deterministic capacity arithmetic.
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
build_sequence_candidate_for_qualification(const SequencePlanningInputs& inputs,
                                           std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS
