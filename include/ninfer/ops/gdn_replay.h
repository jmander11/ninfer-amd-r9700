#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdint>
#include <span>

namespace ninfer::ops {

struct GdnReplayFoldRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
    std::array<std::int32_t, 16> path{};
    std::int32_t path_length = -1;
};

/**
 * Op: gdn_replay_fold
 *
 * Replays each row's accepted records across every registered GDN layer and updates the
 * caller-selected absolute linear-attention state slot in place. rows[b] always maps to physical
 * record row b; rows are not filtered, compressed, or reordered. The active row count is
 * rows.size() and must be in [1,records.spec.record_capacity].
 *
 * linear_state_slot is in [0,states.spec.slot_count), is distinct across rows that fold (commit or
 * path length nonzero), and is the same absolute slot used to produce that row's records.
 * commit_columns is in [0,T] and is the linear/MTP packed prefix. Zero is a strict no-op for the row: no record or state is read and
 * neither recurrent state nor convolution history is written. path_length < 0 keeps that prefix
 * behavior: the Op consumes raw key/value/{g,beta} records in packed order [0,commit_columns) and
 * sets convolution history to tail_3(old_history || conv_record[0:commit_columns]).
 *
 * path_length == 0 is also a strict no-op. path_length > 0 replays records at
 * path[0], path[1], ..., path[path_length-1] in that time order. Each path[i] is in [0,T). The
 * convolution history is tail_3(old_history || conv_record[path[0]], ...,
 * conv_record[path[path_length-1]]).
 *
 * The Op admits the sole Qwen3.8-27B all-layer geometry (L=48, Hqk=16, Hv=48, K=V=128,
 * conv_channels=10240), owns no workspace or metadata allocation, and does not read query or
 * generate token output. The four record planes are read-only, disjoint, and do not overlap
 * either state region.
 */
void gdn_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                     std::span<const GdnReplayFoldRow> rows, hipStream_t stream);

// Device-resident rows of a fold deferred to the next verification round: record row
// `record_row` of the previous round folds `commit_columns` packed columns into absolute slot
// `linear_state_slot` (0 columns: no-op). Written in stream order before the consumer runs.
struct GdnDeferredFoldRow {
    std::int32_t linear_state_slot = 0;
    std::int32_t commit_columns    = 0;
    std::int32_t record_row        = 0;
    std::int32_t reserved          = 0;
};
struct GdnDeferredFoldRows {
    std::array<GdnDeferredFoldRow, 8> row{};
};

// One GDN layer's share of a deferred fold: rows [0, batch) of the device `rows` applied to this
// layer's recurrent state pool (FP32 [128,128,48,S]) and convolution history (BF16 [10240,3,S]),
// exactly gdn_replay_fold's chain fold for that layer. Each folding row's slot is in [0,S) and
// distinct, its columns in [0,records.spec.width], and its record row in
// [0,records.spec.record_capacity); the record planes are those the previous round wrote.
struct GdnLayerFold {
    const GdnReplayRecords* records = nullptr;
    Tensor recurrent;
    Tensor conv;
    std::int32_t layer             = 0;
    const GdnDeferredFoldRows* rows = nullptr;
    std::int32_t batch             = 0;
};

/**
 * Op: gdn_replay_fold_layer
 *
 * Applies one layer of a deferred fold (GdnLayerFold) on `stream`. Consumers that fuse the same
 * fold into another launch (the FP8 GDN front) produce bitwise the same state and history.
 */
void gdn_replay_fold_layer(const GdnLayerFold& fold, hipStream_t stream);

} // namespace ninfer::ops
