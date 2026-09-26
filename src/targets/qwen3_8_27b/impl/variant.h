#pragma once

#include "ops/r9700/linear/a8q4_small_batch_projection.h"
#include "targets/qwen3_8_27b/impl/config.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include <ninfer/targets/qwen3/decoder_state.h>
#include <ninfer/targets/qwen3/runtime.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::ops::r9700::linear {
struct FusedSiluA8Q4G64DownArgs;
}

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
        AttentionOutput,
        MlpDown,
        GdnOutput,
    };

    // Program-owned bindings for loaded-target-prepared row-scaled-E4M3 Text projections.
    // Borrows per-weight descriptors and the loaded target's shared library context.
    // Explicit activation/matmul storage remains in ProgramImplCore's serialized region;
    // opaque library resources are already resident at the final capacity snapshot.
    class ExecutionState final {
    public:
        ExecutionState(const ModelView& model, DeviceSpan serialized_storage,
                       std::uint32_t prefill_tokens, std::uint32_t maximum_concurrency,
                       std::span<const std::uint32_t> verify_widths);
        ~ExecutionState();

        ExecutionState(const ExecutionState&)            = delete;
        ExecutionState& operator=(const ExecutionState&) = delete;
        ExecutionState(ExecutionState&&)                 = delete;
        ExecutionState& operator=(ExecutionState&&)      = delete;

        [[nodiscard]] bool run(SelectedLinearRole role, std::int32_t text_layer,
                               const Tensor& input, const Weight& weight, Tensor& output,
                               hipStream_t stream);
        // Two row-scaled E4M3 projections of one input sharing a single activation quantization;
        // false unless both roles are FP8 and bound to the same activation storage.
        [[nodiscard]] bool run_shared(SelectedLinearRole first_role, SelectedLinearRole second_role,
                                      std::int32_t text_layer, const Tensor& input,
                                      const Weight& first, Tensor& first_output,
                                      const Weight& second, Tensor& second_output,
                                      hipStream_t stream);
        void linear(const Tensor& input, const Weight& weight, Tensor& output,
                    WorkspaceArena& fallback_workspace, hipStream_t stream);
        // SiLU-gated down projection added to the residual in place.
        // Small verification widths: normalized gate/up plus fused SiLU down as one Op whose
        // prepare also clears the down status (no reset launch). gate_up is BF16 scratch.
        [[nodiscard]] bool normalized_mlp(Tensor& residual, const Tensor& norm, float eps,
                                          const Weight& gate_up_weight, const Weight& down,
                                          Tensor& gate_up, std::int32_t text_layer,
                                          hipStream_t stream);
        void fused_mlp_down(const Tensor& gate_up, const Weight& down,
                            Tensor& residual, hipStream_t stream);
        [[nodiscard]] bool gdn_q4_pair_t1(
            const Tensor& input, const Weight& weight0, const Weight& weight1,
            Tensor& output0, Tensor& output1, hipStream_t stream);
        [[nodiscard]] bool attention_q4_pair_t1(
            const Tensor& hidden, const Weight& query_key, const Weight& gate_value,
            Tensor& query, Tensor& key, Tensor& gate, Tensor& value, hipStream_t stream);
        // One A8 activation quantization shared by both GDN projections (any width the Q4 routes
        // serve).
        [[nodiscard]] bool gdn_q4_pair_shared(
            const Tensor& input, const Weight& query_key, const Weight& value_z,
            Tensor& query_key_output, Tensor& value_z_output, hipStream_t stream);
        // Ordinary P2048 GDN front: the input RMSNorm of `residual` feeds the shared codec
        // directly and is published to `normalized`; the query-key, value and output-gate rows
        // (the value-z matrix split at value_dim) are projected from the same planes.
        // The output-gate projection runs on a private side stream, overlapping the
        // convolution and recurrence; join_gdn_gate orders `stream` after it.
        // Both Q4 attention input projections from one A8G64 quantization of the hidden rows.
        [[nodiscard]] bool attention_q4_shared(const Tensor& hidden, const Weight& query_key,
                                               const Weight& gate_value, Tensor& query_key_output,
                                               Tensor& gate_value_output, hipStream_t stream);
        [[nodiscard]] bool gdn_q4_normalized_prefill(
            const Tensor& residual, const Tensor& norm, float eps, const Weight& query_key,
            const Weight& value_z, Tensor& normalized, Tensor& query_key_output,
            Tensor& value_output, Tensor& gate_output, hipStream_t stream);
        void join_gdn_gate(hipStream_t stream);
        [[nodiscard]] bool gdn_q4_pair_c2c4(
            const Tensor& input, const Weight& query_key, const Weight& value_z,
            Tensor& query_key_output, Tensor& value_z_output, hipStream_t stream);
        [[nodiscard]] bool attention_q4_pair_c2c4(
            const Tensor& hidden, const Weight& query_key, const Weight& gate_value,
            Tensor& query, Tensor& key, Tensor& gate, Tensor& value, hipStream_t stream);
        [[nodiscard]] bool projected_residual_t1(
            const Tensor& input, const Weight& weight, Tensor& residual,
            qwen3::TextPhase phase, bool ordinary_decode, hipStream_t stream);
        // A8 widths T >= 2 (verification and prefill) of a Q4 projection whose BF16 result is
        // added to the residual in the GEMM epilogue.
        [[nodiscard]] bool projected_residual_batched(
            const Tensor& input, const Weight& weight, Tensor& residual,
            qwen3::TextPhase phase, std::int32_t text_layer, hipStream_t stream);
        [[nodiscard]] bool normalized_linear_t1(
            const Tensor& input, const Tensor& norm, float eps, const Weight& weight,
            Tensor& output, qwen3::TextPhase phase, bool ordinary_decode,
            std::int32_t text_layer, hipStream_t stream);
        // These local Q4 boundaries do not depend on neighboring matrix formats. Keep explicit
        // ordinary-decode intent: speculative target verification can also have width one.
        [[nodiscard]] static constexpr bool normalized_linear_t1_selected(
            std::uint32_t activation_bits,
            qwen3::TextPhase phase, bool ordinary_decode,
            std::int32_t text_layer, std::uint32_t tokens, std::uint32_t rows,
            std::uint32_t columns, QType weight) noexcept {
            return activation_bits == 8U && ordinary_decode &&
                   phase == qwen3::TextPhase::Verify && text_layer >= 0 &&
                   text_layer < TextConfig::layers && tokens == 1U &&
                   rows == 2U * TextConfig::intermediate && columns == TextConfig::hidden &&
                   weight == QType::Q4G64_F16S;
        }
        // GDN output projection at A8 small-batch widths (T1..8) and full prefill chunks: the
        // gated RMSNorm feeds the codec directly and the GEMM epilogue adds the projection to the
        // residual.
        [[nodiscard]] bool gated_normalized_output(
            const Tensor& recurrent_output, const Tensor& norm, const Tensor& gate, float eps,
            const Weight& weight, Tensor& residual, std::int32_t text_layer, hipStream_t stream);
        // A8 widths T >= 2 (verification and prefill) of the same gate/up boundary normalize
        // straight into the codec.
        [[nodiscard]] bool normalized_linear_batched(
            const Tensor& input, const Tensor& norm, float eps, const Weight& weight,
            Tensor& output, std::int32_t text_layer, hipStream_t stream);
        [[nodiscard]] static constexpr bool projected_residual_t1_selected(
            std::uint32_t activation_bits,
            qwen3::TextPhase phase, bool ordinary_decode,
            std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
            QType weight) noexcept {
            return activation_bits == 8U &&
                   phase == qwen3::TextPhase::Verify &&
                   ordinary_decode && tokens == 1U && rows == TextConfig::hidden &&
                   (columns == TextConfig::query_size ||
                    columns == TextConfig::intermediate) &&
                   weight == QType::Q4G64_F16S;
        }
        [[nodiscard]] static constexpr bool attention_q4_pair_t1_selected(
            std::uint32_t activation_bits, std::uint32_t tokens,
            QType query_key, QType gate_value) noexcept {
            return activation_bits == 8U && tokens == 1U &&
                   query_key == QType::Q4G64_F16S && gate_value == QType::Q4G64_F16S;
        }
        [[nodiscard]] static constexpr bool gdn_q4_pair_t1_selected(
            std::uint32_t activation_bits, std::uint32_t tokens,
            QType weight0, QType weight1) noexcept {
            return activation_bits == 8U && tokens == 1U && weight0 == QType::Q4G64_F16S &&
                   weight1 == QType::Q4G64_F16S;
        }
        [[nodiscard]] static constexpr bool q4_pair_c2c4_selected(
            std::uint32_t activation_bits, std::uint32_t tokens,
            QType weight0, QType weight1) noexcept {
            return activation_bits == 8U && tokens >= 2U &&
                   tokens <= 4U && weight0 == QType::Q4G64_F16S &&
                   weight1 == QType::Q4G64_F16S;
        }
        [[nodiscard]] static constexpr bool fused_mlp_down_selected(
            std::uint32_t activation_bits, QType down, std::uint32_t tokens,
            std::int32_t text_layer) noexcept {
            // The fused Op consumes represented BF16 gate/up values, independently
            // of the producer's weight or activation precision.
            // The full prefill chunk (M128) and every small-batch N5120/K17408 width own the
            // in-place residual epilogue.
            return activation_bits == 8U && down == QType::Q4G64_F16S &&
                   (tokens == 2048U ||
                    ops::r9700::linear::detail::use_a8q4_small_batch_projection(
                        tokens, TextConfig::hidden, TextConfig::intermediate,
                        TextConfig::intermediate)) &&
                   text_layer >= 0 && text_layer < TextConfig::layers;
        }
        [[nodiscard]] std::size_t selected_count() const noexcept;
        // Complete set of token widths that graph-captured selected projections can observe,
        // plus the fixed full prefill chunk prepared before graph definition begins.
        [[nodiscard]] static std::vector<std::uint32_t>
        eager_widths(std::uint32_t prefill_tokens, std::uint32_t maximum_concurrency,
                     std::span<const std::uint32_t> verify_widths);

    private:
        struct Impl;
        [[nodiscard]] ops::r9700::linear::FusedSiluA8Q4G64DownArgs fused_down_args(
            const Tensor& gate_up, const Weight& down, Tensor& residual) const;
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
                               const Tensor* active_query_rows = nullptr,
                               bool dflash_target_verify = false);
    [[nodiscard]] static std::size_t full_attention_workspace_capacity_bytes(
        std::int32_t maximum_query_rows, std::uint32_t maximum_visible_context,
        bool tree_or_device_count, bool dflash_target_verify = false);
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
                                            std::int32_t text_layer,
                                            ExecutionState* execution = nullptr,
                                            bool ordinary_decode = false);
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
    // Ordinary P2048 prefill owns the whole GDN front: the input RMSNorm of `residual` (published
    // to `hidden`), the a/b control projection into g/beta, the query-key/value-z projections
    // and the convolution scatter. The A8 route normalizes straight into the shared codec.
    static void gdn_input_projection_prefill_p2048(
        const Tensor& residual, const Tensor& norm_weight, float eps,
        const GdnProjectionWeights& weights, const Tensor& conv_weight, Tensor& conv_state,
        Tensor& hidden, Tensor& g, Tensor& beta, Tensor& query, Tensor& key, Tensor& value,
        Tensor& output_gate, qwen3::TextPhase phase, WorkspaceArena& workspace,
        hipStream_t stream, ExecutionState* execution = nullptr, std::int32_t text_layer = -1);
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
    // The family provides the gated-RMSNorm parameters and the BF16 [128,48,T] normalized
    // scratch; the leaf owns whether the normalized output is materialized (always when
    // `materialize_normalized`) or fused into the output projection's activation codec.
    static void gdn_output_projection(const Tensor& recurrent_output, const Tensor& norm,
                                      const Tensor& gate, float eps, Tensor& normalized,
                                      bool materialize_normalized, const Weight& weight,
                                      Tensor& residual, qwen3::TextPhase phase,
                                      WorkspaceArena& workspace, hipStream_t stream,
                                      std::int32_t route_tokens = 0,
                                      ExecutionState* execution = nullptr,
                                      std::int32_t text_layer = -1,
                                      bool ordinary_decode = false);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            hipStream_t stream);
    // The family provides the normalization parameters and BF16 scratch; the leaf owns whether
    // normalization is materialized or fused into the gate/up projection. ordinary_decode is
    // supplied only by the family ordinary-decode entry point, never inferred from Verify/T1.
    static void post_mixer(const Tensor& norm, float eps, const Tensor& hidden,
                           const PostMixerWeights& weights, Tensor& residual,
                           qwen3::TextPhase phase, WorkspaceArena& workspace,
                           hipStream_t stream,
                           ExecutionState* execution = nullptr,
                           std::int32_t text_layer = -1,
                           bool ordinary_decode = false);
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

    // gfx1201/ROCm hipMalloc rounds the two Program arenas to at most 2 MiB
    // units. Keep this physical-allocation bound separate from logical arenas
    // and graph allowance; it is independent of KV pages, preserving affinity.
    [[nodiscard]] static constexpr std::size_t runtime_allocation_overhead_bound(
        WeightsProfile profile) noexcept {
        return profile == WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation ||
               is_fp8_capped_profile(profile) ||
               is_selective_protected_profile(profile) ||
               profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation ||
               profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation ||
               profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation
            ? std::size_t{4} << 20U : 0U;
    }

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size, std::uint32_t verify_width);
};

} // namespace ninfer::targets::qwen3_8_27b::detail
