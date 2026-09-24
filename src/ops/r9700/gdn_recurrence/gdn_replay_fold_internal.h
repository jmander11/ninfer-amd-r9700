#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

struct alignas(8) GdnReplayFoldKernelRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
    std::int32_t path_length;
    std::int32_t path[16];
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[8];
};

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        hipStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
