#pragma once

#include "targets/qwen3_8_27b/impl/config.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include <ninfer/targets/qwen3/decoder_state.h>
#include <ninfer/targets/qwen3/runtime.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::targets::qwen3_8_27b::detail {

using GraphExecutionProfile = qwen3::GraphExecutionProfile;

// Compile-time data and the three closed execution leaves supplied to the Qwen3 family runtime.
// It owns no request state, execution phase, graph object, or schedule callback.
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::FullAttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::DensePostMixerPayload;
    using MtpAttentionProjectionWeights  = detail::MtpAttentionPayload;
    using MtpPostMixerWeights            = detail::DensePostMixerPayload;
    using VisionWeights                  = qwen3::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    enum class SelectedLinearRole : std::uint8_t {
        AttentionQueryKey,
        AttentionGateValue,
        GdnQueryKey,
        MlpGateUp,
    };

    // Program-owned prepared state for the decision-selected row-scaled-E4M3 Text projections.
    // The implementation owns only descriptors/handles; all device storage remains in the one
    // stable serialized region supplied by ProgramImplCore.
    class ExecutionState final {
    public:
        ExecutionState(const ModelView& model, DeviceSpan serialized_storage,
                       std::uint32_t prefill_tokens, std::uint32_t maximum_concurrency,
                       std::uint32_t mtp_width, std::uint32_t dflash_width);
        ~ExecutionState();

        ExecutionState(const ExecutionState&)            = delete;
        ExecutionState& operator=(const ExecutionState&) = delete;
        ExecutionState(ExecutionState&&)                 = delete;
        ExecutionState& operator=(ExecutionState&&)      = delete;

        [[nodiscard]] bool run(SelectedLinearRole role, std::int32_t text_layer,
                               const Tensor& input, const Weight& weight, Tensor& output,
                               hipStream_t stream);
        void linear(const Tensor& input, const Weight& weight, Tensor& output,
                    WorkspaceArena& fallback_workspace, hipStream_t stream);
        void fused_mlp_down(const Tensor& gate_up, const Weight& down,
                            Tensor& output, hipStream_t stream);
        [[nodiscard]] static constexpr bool fused_mlp_down_selected(
            QType gate_up, QType down, std::uint32_t tokens,
            std::int32_t text_layer) noexcept {
            return gate_up == QType::F8E4M3_ROW_F32S && down == QType::Q4G64_F16S &&
                   tokens == 2048U && text_layer >= 0 && text_layer < TextConfig::layers;
        }
        [[nodiscard]] std::size_t selected_count() const noexcept;
        // Complete set of token widths that graph-captured selected projections can observe,
        // plus the fixed full prefill chunk prepared before graph definition begins.
        [[nodiscard]] static std::vector<std::uint32_t>
        eager_widths(std::uint32_t prefill_tokens, std::uint32_t maximum_concurrency,
                     std::uint32_t mtp_width, std::uint32_t dflash_width);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows              = 131072;

    // The target-verify leaves receive route_tokens: the C=1 width of a packed verify round (0
    // when the round is not batched packed verify). It selects the C=1 quantization family for
    // the single aggregate launch; see the pinned policy in variant.cpp.
    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     qwen3::TextPhase phase, WorkspaceArena& workspace,
                                     hipStream_t stream, std::int32_t route_tokens = 0,
                                     ExecutionState* execution = nullptr,
                                     std::int32_t text_layer = -1);
    // The family schedule owns append/publication ordering and supplies the resulting typed read
    // capability. This target leaf owns the fixed Qwen3.8 full-attention geometry and R9700
    // implementation; it cannot manufacture a cache view or publish a frontier.
    static void full_attention(const Tensor& normalized_query,
                               const qwen3::PagedKVLayerRead& cache_read,
                               const Tensor& cache_positions, Tensor& attention_fp32,
                               WorkspaceArena& workspace, hipStream_t stream,
                               const Tensor* ancestor_masks = nullptr,
                               const Tensor* prefix_lengths = nullptr,
                               const Tensor* active_query_rows = nullptr);
    [[nodiscard]] static std::size_t full_attention_workspace_capacity_bytes(
        std::int32_t maximum_query_rows, std::uint32_t maximum_visible_context,
        bool tree_or_device_count);
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    // Compile-isolated model-gate leaf. The ordinary target library has neither this declaration
    // nor its sparse Op object, so no runtime selector or disabled-path branch reaches production.
    static void text_prefill_attention(const Tensor& normalized_query,
                                       const qwen3::PagedKVLayerRead& cache_read,
                                       const Tensor& cache_positions, Tensor& attention_fp32,
                                       WorkspaceArena& workspace, hipStream_t stream);
    [[nodiscard]] static std::size_t text_prefill_attention_workspace_capacity_bytes(
        std::int32_t maximum_query_rows, std::uint32_t maximum_visible_context);
#endif
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            Tensor& residual, qwen3::TextPhase phase,
                                            WorkspaceArena& workspace, hipStream_t stream,
                                            std::int32_t route_tokens = 0,
                                            ExecutionState* execution = nullptr);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, hipStream_t stream,
                                         ExecutionState* execution = nullptr);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, hipStream_t stream,
                                  ExecutionState* execution = nullptr);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, hipStream_t stream,
                                      ExecutionState* execution = nullptr);
    static void mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm, const Weight& weight,
                       Tensor& residual, WorkspaceArena& workspace, hipStream_t stream,
                       std::int32_t route_tokens = 0, ExecutionState* execution = nullptr);
    static void mtp_attention_output(const Tensor& attention, const Weight& weight,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     hipStream_t stream, std::int32_t route_tokens = 0,
                                     ExecutionState* execution = nullptr);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3::TextPhase phase,
                                     WorkspaceArena& workspace, hipStream_t stream,
                                     ExecutionState* execution = nullptr,
                                     std::int32_t text_layer = -1);
    [[nodiscard]] static constexpr bool gdn_input_projection_prefill_p2048_selected(
        qwen3::TextPhase phase, std::int32_t tokens) noexcept {
        return phase == qwen3::TextPhase::Prefill && tokens == 2048;
    }
    static void gdn_input_projection_prefill_p2048(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        Tensor& conv_state, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3::TextPhase phase, WorkspaceArena& workspace, hipStream_t stream,
        ExecutionState* execution = nullptr, std::int32_t text_layer = -1);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, qwen3::TextPhase phase,
                                  WorkspaceArena& workspace, hipStream_t stream,
                                  ExecutionState* execution = nullptr,
                                  std::int32_t text_layer = -1);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3::TextPhase phase, WorkspaceArena& workspace, hipStream_t stream,
        const Tensor* parent_index = nullptr, ExecutionState* execution = nullptr,
        std::int32_t text_layer = -1);
    static void gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                      qwen3::TextPhase phase, WorkspaceArena& workspace,
                                      hipStream_t stream, std::int32_t route_tokens = 0,
                                      ExecutionState* execution = nullptr);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace, hipStream_t stream,
                                            ExecutionState* execution = nullptr);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           qwen3::TextPhase phase, WorkspaceArena& workspace,
                           hipStream_t stream, std::int32_t route_tokens = 0,
                           ExecutionState* execution = nullptr,
                           std::int32_t text_layer = -1);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, hipStream_t stream,
                               std::int32_t route_tokens = 0,
                               ExecutionState* execution = nullptr);
    [[nodiscard]] static std::size_t
    mtp_attention_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                                std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_fc_workspace_capacity_bytes(std::int32_t first,
                                                                     std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_attention_output_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3::TextPhase phase,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                   qwen3::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile, qwen3::TextPhase phase,
                                        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);
    // Maximum caller-owned compile-selected activation scratch needed by any Q4 or W8 matrix in
    // the selected profile. The family schedule adds this reserve to every phase whose live
    // workspace may contain a candidate linear call.
    [[nodiscard]] static QType dflash_matrix_qtype(WeightsProfile weights_profile);
    [[nodiscard]] static std::size_t
    linear_workspace_capacity_bytes(WeightsProfile weights_profile, std::int32_t tokens);
    [[nodiscard]] static std::size_t
    vision_linear_workspace_capacity_bytes(WeightsProfile weights_profile, std::int32_t tokens);

    // One serialized Program-owned activation region replaces the former per-call arena reserve.
    // The hybrid profile also uses its prefix for selected FP8 activation quantization.
    [[nodiscard]] static std::size_t execution_state_capacity_bytes(
        WeightsProfile weights_profile, std::uint32_t prefill_tokens,
        std::uint32_t maximum_graph_tokens);

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size, std::uint32_t verify_width);
};

} // namespace ninfer::targets::qwen3_8_27b::detail
