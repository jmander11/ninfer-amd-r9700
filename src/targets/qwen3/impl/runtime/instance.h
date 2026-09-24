#pragma once

#ifndef NINFER_QWEN3_VARIANT
#    error "NINFER_QWEN3_VARIANT must name the complete exact Variant"
#endif
#ifndef NINFER_QWEN3_RUNTIME_NS
#    error "NINFER_QWEN3_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <ninfer/targets/qwen3/runtime.h>
#include <ninfer/targets/qwen3/startup_features.h>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {

using Variant                        = NINFER_QWEN3_VARIANT;
using WeightsProfile                 = typename Variant::WeightsProfile;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using DFlashConfig                   = typename Variant::DFlashConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using DFlashWeights                  = typename LoadedModelData::DFlash;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphExecutionProfile          = typename Variant::GraphExecutionProfile;

using SequencePlan    = qwen3::SequencePlan<Variant>;
using SequencePlanner = qwen3::SequencePlanner<Variant>;
using RequestBasePlan = qwen3::RequestBasePlan<Variant>;
using RequestPlan     = qwen3::RequestPlan<Variant>;
using Program         = qwen3::Program<Variant>;

inline constexpr float kAttentionScale                   = Variant::attention_scale;
inline constexpr float kGdnScale                         = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment    = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = Variant::maximum_mtp_draft_tokens;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = Variant::maximum_dflash_draft_tokens;

// Auto verify width from k when --dflash-verify-width is omitted. Product DFlash is chain W=k+1.
// A tree-capable package may still select a wider default for a native draft window.
[[nodiscard]] inline constexpr std::uint32_t dflash_default_verify_width(std::uint32_t draft_window) {
    return qwen3::dflash_verify_width<DFlashConfig>(draft_window);
}

[[nodiscard]] inline constexpr std::uint32_t dflash_verify_width(std::uint32_t draft_window,
                                                                std::uint32_t override_width = 0) {
    return qwen3::dflash_verify_width<DFlashConfig>(draft_window, override_width);
}

// Packed-tree verify and GDN/KV path fold for a tree-capable package. W == k+1 is chain.
[[nodiscard]] inline constexpr bool dflash_uses_tree_verify(std::uint32_t draft_window,
                                                            std::uint32_t verify_width) {
    if constexpr (!DFlashConfig::tree_verify) {
        return false;
    }
    if constexpr (DFlashConfig::two_block_first > 0) {
        if (draft_window > static_cast<std::uint32_t>(DFlashConfig::two_block_first)) {
            return false;
        }
    }
    if (verify_width == draft_window + 1U) {
        return false;
    }
    return true;
}

[[nodiscard]] inline constexpr std::uint32_t
dflash_captured_verify_width(std::uint32_t k, std::uint32_t storage_ceil) {
    return qwen3::dflash_captured_verify_width<DFlashConfig>(k, storage_ceil);
}

// Storage / ReplaySSM / pending-features width. Adaptive `{3,4,5}` is chain W<=6.
// An explicit --dflash-verify-width still wins; chain-only packages require W=k+1.
[[nodiscard]] inline std::uint32_t
dflash_storage_verify_width(std::span<const std::uint32_t> captured_ks,
                            std::uint32_t draft_window, std::uint32_t override_width) {
    return qwen3::dflash_storage_verify_width<DFlashConfig>(
        captured_ks, draft_window, override_width);
}

inline std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    return Variant::ordinary_graph_profiles(capacity);
}

inline std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                             std::uint32_t draft_window) {
    return Variant::mtp_graph_profiles(capacity, draft_window);
}

inline std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                                std::uint32_t draft_window,
                                                                std::uint32_t batch_size,
                                                                std::uint32_t verify_width) {
    return Variant::dflash_graph_profiles(capacity, draft_window, batch_size, verify_width);
}

inline std::uint32_t speculative_graph_stride(std::uint32_t capacity,
                                               std::uint32_t max_concurrency,
                                               std::span<const std::uint32_t> ks,
                                               SpeculativeBackend backend,
                                               std::uint32_t storage_width) {
    std::uint32_t maximum = 0;
    for (const auto k : ks) {
        for (std::uint32_t batch = 1; batch <= max_concurrency; ++batch) {
            const auto profiles = backend == SpeculativeBackend::Mtp
                ? mtp_graph_profiles(capacity, k)
                : dflash_graph_profiles(capacity, k, batch,
                                       dflash_captured_verify_width(k, storage_width));
            for (const auto& profile : profiles) {
                maximum = std::max(maximum, profile.topology_class);
            }
        }
    }
    return max_concurrency * (maximum + 1U);
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS
