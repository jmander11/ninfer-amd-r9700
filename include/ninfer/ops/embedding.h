#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Gathers one embedding row per token:
 *
 *   ideal[d,t] = dequantize(table)[ids[t],d].
 *
 * `ids` is contiguous I32 [T], `out` is contiguous BF16 [D,T], and every id is in
 * [0,vocab). `table` has logical shape [vocab,D] and is contiguous BF16_CTRL, or
 * Q4G64_F16S/Q6G64_F16S/W8G32_F16S RowSplit (row-contiguous planes) with FP16 scales. Dense BF16
 * values are copied bit-exactly. For quantized tables, the oracle independently decodes each signed
 * code and multiplies it by the exact stored FP16 scale in FP64; the BF16 output is promoted and
 * compared directly with that ideal. Final output storage rounding belongs to the quantized
 * embedding criterion, not the oracle. The registered domains are Q4/Q6 D=5120 and W8 D=2048 or
 * D=5120. `out` must not overlap `ids` or any table plane. The table planes may be device memory or
 * pinned host memory read through its unified device address. There is no workspace or persistent
 * state side effect.
 */
void embedding(const Tensor& ids, const Weight& table, Tensor& out, hipStream_t stream);

/**
 * Host staging of embedding rows whose ids are known on the host, for a table in host memory.
 *
 * `stage_embedding_rows` reads `table` through its host pointers and writes a self-contained,
 * position-independent image to `image`: first the I32 slot of every id, then a compact table
 * holding each distinct id's row once, in ascending id order, with the source qtype, layout and
 * plane geometry and the stored codes and FP16 scales unchanged. Once the first `stage.bytes`
 * bytes are copied to `device_image`, `staged_embedding(stage, table, device_image)` returns the
 * slots and the compact table, and for every window [c,c+n) of the staged ids
 *
 *   embedding(slots[c:c+n], compact, out) == embedding(ids[c:c+n], table, out)
 *
 * bit-exactly. The staging itself is an exact transform: its oracle compares the logical code and
 * scale (or BF16 value) of every staged slot's row with the source row. Supported tables are the
 * bound token-embedding encodings: BF16_CTRL Contiguous and Q4G64_F16S/W8G32_F16S RowSplit. `ids`
 * is nonempty, may repeat, and every id is in [0,vocab). `image` must hold
 * embedding_stage_capacity_bytes(table.qtype, D, ids.size()). There is no device work.
 */
struct EmbeddingStage {
    std::int32_t ids    = 0; // staged slots, one per id
    std::int32_t rows   = 0; // compact-table rows, one per distinct id
    std::uint64_t bytes = 0; // image prefix that the device copy must contain
};

struct StagedEmbedding {
    Tensor slots; // I32 [ids]
    Weight table; // compact table, [rows,D]
};

[[nodiscard]] std::uint64_t embedding_stage_capacity_bytes(QType qtype, std::int32_t features,
                                                           std::int32_t ids);
[[nodiscard]] EmbeddingStage stage_embedding_rows(std::span<const std::int32_t> ids,
                                                  const Weight& table, std::span<std::byte> image);
[[nodiscard]] StagedEmbedding staged_embedding(const EmbeddingStage& stage, const Weight& table,
                                               void* device_image);

} // namespace ninfer::ops
