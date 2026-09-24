#pragma once

#include "ninfer/types.h"
#include "targets/qwen3/impl/runtime/adaptive_draft.h"

#include <algorithm>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3 {

struct StartupFeatures {
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision        = options.enable_vision,
        .speculative   = options.speculative.backend,
        .proposal_head = options.speculative.proposal_head,
    };
}

// Shared by load-time Linear preparation and the sequence planner. Keep the
// native tree/chain default in one place so both prepare identical widths.
template <class DFlashConfig>
[[nodiscard]] inline constexpr std::uint32_t dflash_verify_width(
    std::uint32_t draft_window, std::uint32_t override_width = 0) {
    if (override_width != 0) return override_width;
    if constexpr (!DFlashConfig::tree_verify) {
        return draft_window + 1U;
    } else {
        if constexpr (DFlashConfig::two_block_first > 0) {
            if (draft_window > static_cast<std::uint32_t>(DFlashConfig::two_block_first))
                return draft_window + 1U;
        }
        if (draft_window <= 5U) return draft_window + 1U;
        return static_cast<std::uint32_t>(DFlashConfig::verify_width);
    }
}

template <class DFlashConfig>
[[nodiscard]] inline constexpr std::uint32_t dflash_captured_verify_width(
    std::uint32_t k, std::uint32_t storage_ceil) {
    return std::min(dflash_verify_width<DFlashConfig>(k), storage_ceil);
}

template <class DFlashConfig>
[[nodiscard]] inline std::uint32_t dflash_storage_verify_width(
    std::span<const std::uint32_t> captured_ks, std::uint32_t draft_window,
    std::uint32_t override_width) {
    if (override_width != 0U) return override_width;
    std::uint32_t width = 0U;
    for (const auto k : captured_ks)
        width = std::max(width, dflash_verify_width<DFlashConfig>(k));
    return width != 0U ? width : dflash_verify_width<DFlashConfig>(draft_window);
}

// Actual graph widths, not just the largest storage extent. Adaptive graph
// capture observes each K independently, including widths below the maximum.
template <class DFlashConfig>
[[nodiscard]] inline std::vector<std::uint32_t> captured_verify_widths(
    SpeculativeBackend backend, std::span<const std::uint32_t> captured_ks,
    std::uint32_t dflash_storage_width) {
    std::vector<std::uint32_t> widths;
    if (backend == SpeculativeBackend::None) return widths;
    for (const auto k : captured_ks) {
        const auto width = backend == SpeculativeBackend::Mtp ? k + 1U
            : dflash_captured_verify_width<DFlashConfig>(k, dflash_storage_width);
        if (std::find(widths.begin(), widths.end(), width) == widths.end())
            widths.push_back(width);
    }
    return widths;
}

template <class DFlashConfig>
[[nodiscard]] inline std::vector<std::uint32_t> startup_verify_widths(
    const EngineOptions& options) {
    const auto& speculative = options.speculative;
    const auto ks = adaptive_draft_ks(speculative.backend, speculative.draft_tokens,
                                     speculative.adaptive_draft);
    const auto storage = speculative.backend == SpeculativeBackend::DFlash
        ? dflash_storage_verify_width<DFlashConfig>(ks, speculative.draft_tokens,
                                                    speculative.dflash_verify_width)
        : 0U;
    return captured_verify_widths<DFlashConfig>(speculative.backend, ks, storage);
}

} // namespace ninfer::targets::qwen3
