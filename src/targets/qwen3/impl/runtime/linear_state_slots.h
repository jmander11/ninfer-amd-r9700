#pragma once

#include <ninfer/types.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {

/** Qwen3's target-local mapping from a stable request lane to its two state roles. */
struct LinearStateSlots {
    [[nodiscard]] static std::int32_t state_slot_count(std::uint32_t max_concurrency,
                                                       bool staging_slot = false) {
        if (max_concurrency == 0 || max_concurrency > kMaximumConcurrency) {
            throw std::invalid_argument("Qwen3 Linear Attention concurrency is invalid");
        }
        const std::uint32_t extra = staging_slot ? 1U : 0U;
        return static_cast<std::int32_t>(2U * max_concurrency + extra);
    }

    [[nodiscard]] static std::int32_t staging_state_slot(std::uint32_t max_concurrency) {
        return state_slot_count(max_concurrency, true) - 1;
    }

    [[nodiscard]] static std::int32_t current_state_slot(std::uint32_t lane,
                                                         std::uint32_t max_concurrency) {
        if (lane >= max_concurrency) {
            throw std::out_of_range("Qwen3 Linear Attention lane is out of range");
        }
        return static_cast<std::int32_t>(lane);
    }

    [[nodiscard]] static std::int32_t rewrite_checkpoint_state_slot(std::uint32_t lane,
                                                                    std::uint32_t max_concurrency) {
        return static_cast<std::int32_t>(max_concurrency) +
               current_state_slot(lane, max_concurrency);
    }
};

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS
