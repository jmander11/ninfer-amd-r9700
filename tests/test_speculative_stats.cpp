#include "targets/qwen3/impl/runtime/speculative_stats.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    // The first round's accepted drafts must survive a stop inside the next
    // licensed round. Enumerate that round's output, marking only drafts as
    // accepted; its final correction/bonus is not an accepted draft.
    for (std::uint32_t drafts = 0; drafts <= 11; ++drafts) {
        std::vector<std::uint8_t> licensed(drafts, 1);
        licensed.push_back(0);
        for (std::uint32_t committed = 1; committed <= licensed.size(); ++committed) {
            ninfer::SpeculativeStats stats;
            stats.rounds = 2;
            stats.drafted_tokens = 18;
            stats.fallback_steps = 3;
            stats.accepted_tokens = 3 + drafts;
            stats.accepted_per_position.assign(11, 0);
            for (std::size_t i = 0; i < 3; ++i) { ++stats.accepted_per_position[i]; }
            for (std::size_t i = 0; i < drafts; ++i) { ++stats.accepted_per_position[i]; }

            std::uint64_t expected_total = 3;
            std::vector<std::uint64_t> expected_positions(11, 0);
            for (std::size_t i = 0; i < 3; ++i) { ++expected_positions[i]; }
            for (std::size_t i = 0; i < committed; ++i) {
                if (licensed[i]) {
                    ++expected_total;
                    ++expected_positions[i];
                }
            }
            ninfer::targets::qwen3::detail::trim_speculative_stats_to_commit(
                stats, static_cast<std::uint32_t>(licensed.size()), committed);
            if (stats.accepted_tokens != expected_total ||
                stats.accepted_per_position != expected_positions || stats.rounds != 2 ||
                stats.drafted_tokens != 18 || stats.fallback_steps != 3) {
                std::cerr << "incorrect committed speculative statistics: drafts=" << drafts
                          << " committed=" << committed << '\n';
                return 1;
            }
        }
    }
    return 0;
}
