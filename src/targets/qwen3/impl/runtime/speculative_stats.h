#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3::detail {

inline void trim_speculative_stats_to_commit(SpeculativeStats& stats, std::uint32_t produced,
                                             std::uint32_t committed) {
    if (committed == produced) { return; }
    // A licensed round contains accepted drafts followed by one correction/bonus.
    // A shorter committed prefix consists entirely of accepted drafts. A fallback
    // produces only its correction, so it cannot contribute discarded drafts.
    const std::uint32_t counted_accepted = produced - 1U;
    if (committed >= counted_accepted) { return; }
    const std::uint32_t drop = counted_accepted - committed;
    if (stats.accepted_tokens < drop || counted_accepted > stats.accepted_per_position.size()) {
        throw std::logic_error("partial speculative commit stats do not match pending round");
    }
    stats.accepted_tokens -= drop;
    for (std::uint32_t index = committed; index < counted_accepted; ++index) {
        if (stats.accepted_per_position[index] == 0) {
            throw std::logic_error("partial speculative position stats underflow");
        }
        --stats.accepted_per_position[index];
    }
}

} // namespace ninfer::targets::qwen3::detail
