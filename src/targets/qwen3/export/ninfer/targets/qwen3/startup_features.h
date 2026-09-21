#pragma once

#include "ninfer/types.h"

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

} // namespace ninfer::targets::qwen3
