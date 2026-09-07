#pragma once
#include "targets/qwen3/impl/runtime/instance.h"
// Qwen3 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/swa.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3/prepared_prompt.h>
#include <ninfer/targets/qwen3/decoder_state.h>
#include "targets/qwen3/impl/runtime/text_context.h"
#include "targets/qwen3/impl/runtime/dflash_context.h"
#include "targets/qwen3/impl/runtime/vision_context.h"
#include "targets/qwen3/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule {

using qwen3::PreparedPromptData;
using qwen3::PromptModality;

struct ExecutionCore {
    DeviceContext& device;
    const LoadedModelData& model;
    typename Variant::ExecutionState* linear_execution;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    qwen3::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
};

struct PrefillContext {
    ExecutionCore execution;
    qwen3::PagedKVCacheView text_kv;
    qwen3::PagedKVCacheView mtp_kv;
    qwen3::PagedKVCache& text_cache;
    qwen3::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t current_state_slot                         = 0;
    std::int32_t rewrite_checkpoint_state_slot              = 0;
    std::uint32_t mtp_proposal_extent                       = 0;
    const qwen3::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    PagedKVAllocation* text_kv_allocation                   = nullptr;
    qwen3::PagedKVPublication* text_kv_publication        = nullptr;
    std::uint32_t* text_kv_status                           = nullptr;
    PagedKVAllocation* mtp_kv_allocation                    = nullptr;
    qwen3::PagedKVPublication* mtp_kv_publication         = nullptr;
    std::uint32_t* mtp_kv_status                            = nullptr;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen3::PagedKVCache& text_cache;
    qwen3::OrdinaryDecodeState& frame;
    const qwen3::OrdinaryDecodeIngress& host_ingress;
    qwen3::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    std::span<qwen3::PagedKVTransaction* const> text_kv_transactions{};
};

struct MtpBatchContext {
    ExecutionCore execution;
    const qwen3::PagedKVCache& text_cache;
    qwen3::PagedKVCache& mtp_cache;
    qwen3::MtpDecodeState& frame;
    const qwen3::MtpDecodeIngress& host_ingress;
    qwen3::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    std::span<qwen3::PagedKVTransaction* const> text_kv_transactions{};
    // One round-scoped segmented authority per compact row. The same fixed-address transaction
    // spans alignment and every AR append and is resolved by Program once after execution.
    std::span<qwen3::PagedKVTransaction* const> mtp_kv_transactions{};
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const qwen3::PagedKVCache& text_cache;
    DFlashPersistentState& dflash;
    qwen3::DFlashDecodeState& frame;
    const qwen3::DFlashDecodeIngress& host_ingress;
    qwen3::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    std::span<qwen3::PagedKVTransaction* const> text_kv_transactions{};
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct DFlashEnvelopes {
    ops::SwaContextExecutionEnvelope local;
    ops::GqaContextExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor lanes;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    bool dflash_target_verify = false;
    Tensor drafts;
    Tensor current_extents;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor selected_hidden;
    Tensor parent_index;
    Tensor ancestor_mask;
    Tensor prefix_lengths;
    Tensor accepted_column;
    Tensor fold_path;
    Tensor draft_selector_ids;
    Tensor draft_selector_q;
    bool tree_verify = false;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t rewrite_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent);
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          bool reset_workspace = true);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier, bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier,
                         bool finalize_at_end);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DecodeGraphDefinition& definition);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, std::uint32_t verify_width,
                                 DFlashEnvelopes envelopes,
                                 DecodeGraphDefinition& definition);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         std::uint32_t verify_width, DFlashEnvelopes envelopes,
                         DecodeGraphExecutable* executable);

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule
