#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::uint64_t kPlaneAlignment = 256;
constexpr std::uint32_t kQ4Group        = 64;
constexpr std::uint32_t kW8Group        = 32;
// Rows per host worker. Prompt windows are gathered from host DRAM, whose per-core bandwidth
// bounds a synchronous (not prefetched) window; a few workers cut that latency.
constexpr std::size_t kRowsPerWorker  = 512;
constexpr std::size_t kMaximumWorkers = 4;

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

// Geometry of one compact table image of `rows` rows, relative to the table base.
struct CompactGeometry {
    std::uint64_t code_bytes   = 0; // qdata plane (the BF16 values for BF16_CTRL)
    std::uint64_t scale_offset = 0; // 0 for BF16_CTRL
    std::uint64_t scale_bytes  = 0;
    std::uint64_t bytes        = 0;
};

std::uint32_t padded_features(std::int32_t features) {
    return static_cast<std::uint32_t>(align_up(static_cast<std::uint64_t>(features), 128U));
}

CompactGeometry compact_geometry(QType qtype, std::int32_t features, std::uint64_t rows) {
    const std::uint64_t padded = padded_features(features);
    CompactGeometry out;
    switch (qtype) {
    case QType::BF16_CTRL:
        out.code_bytes = rows * static_cast<std::uint64_t>(features) * 2U;
        out.bytes      = out.code_bytes;
        return out;
    case QType::Q4G64_F16S:
        out.code_bytes   = rows * padded / 2U;
        out.scale_offset = align_up(out.code_bytes, kPlaneAlignment);
        out.scale_bytes  = rows * (padded / kQ4Group) * 2U;
        break;
    case QType::W8G32_F16S:
        out.code_bytes   = rows * padded;
        out.scale_offset = align_up(out.code_bytes, kPlaneAlignment);
        out.scale_bytes  = rows * (padded / kW8Group) * 2U;
        break;
    default:
        throw std::invalid_argument("embedding staging: unsupported table encoding");
    }
    out.bytes = out.scale_offset + out.scale_bytes;
    return out;
}

std::uint64_t slot_bytes(std::uint64_t ids) {
    return align_up(ids * sizeof(std::int32_t), kPlaneAlignment);
}

void require_stageable(const Weight& table) {
    if (table.ndim != 2 || table.shape[0] <= 0 || table.shape[1] <= 0 || table.qdata == nullptr ||
        table.padded_shape[0] != table.shape[0]) {
        throw std::invalid_argument("embedding staging: malformed [vocabulary,features] table");
    }
    const auto padded = static_cast<std::int32_t>(padded_features(table.shape[1]));
    switch (table.qtype) {
    case QType::BF16_CTRL:
        if (table.layout != QuantLayout::Contiguous || table.qhigh != nullptr) {
            throw std::invalid_argument("embedding staging: malformed contiguous BF16 table");
        }
        return;
    case QType::Q4G64_F16S:
        if (table.layout != QuantLayout::RowSplit || table.group_size != kQ4Group ||
            table.group != static_cast<std::int32_t>(kQ4Group) ||
            table.scale_dtype != DType::FP16 || table.padded_shape[1] != padded ||
            table.qhigh != nullptr || table.scales == nullptr) {
            throw std::invalid_argument("embedding staging: malformed Q4G64_F16S RowSplit table");
        }
        return;
    case QType::W8G32_F16S:
        if (table.layout != QuantLayout::RowSplit || table.group_size != kW8Group ||
            table.group != static_cast<std::int32_t>(kW8Group) ||
            table.scale_dtype != DType::FP16 || table.padded_shape[1] != padded ||
            table.qhigh != nullptr || table.scales == nullptr) {
            throw std::invalid_argument("embedding staging: malformed W8G32_F16S RowSplit table");
        }
        return;
    default:
        throw std::invalid_argument("embedding staging: unsupported table encoding");
    }
}

// Copies source rows `rows[first,last)` into compact rows [first,last).
void copy_rows(const Weight& table, std::span<const std::int32_t> rows, std::size_t first,
               std::size_t last, std::byte* compact, const CompactGeometry& geometry) {
    const auto features        = static_cast<std::uint64_t>(table.shape[1]);
    const std::uint64_t padded = padded_features(table.shape[1]);
    switch (table.qtype) {
    case QType::BF16_CTRL: {
        const auto* source            = static_cast<const std::byte*>(table.qdata);
        const std::uint64_t row_bytes = features * 2U;
        for (std::size_t slot = first; slot < last; ++slot) {
            std::memcpy(compact + slot * row_bytes,
                        source + static_cast<std::uint64_t>(rows[slot]) * row_bytes, row_bytes);
        }
    } break;
    case QType::Q4G64_F16S:
    case QType::W8G32_F16S: {
        const std::uint64_t code_row_bytes =
            table.qtype == QType::Q4G64_F16S ? padded / 2U : padded;
        const std::uint64_t scale_row_bytes =
            (padded / (table.qtype == QType::Q4G64_F16S ? kQ4Group : kW8Group)) * 2U;
        const auto* source_codes  = static_cast<const std::byte*>(table.qdata);
        const auto* source_scales = static_cast<const std::byte*>(table.scales);
        for (std::size_t slot = first; slot < last; ++slot) {
            const auto row = static_cast<std::uint64_t>(rows[slot]);
            std::memcpy(compact + slot * code_row_bytes, source_codes + row * code_row_bytes,
                        code_row_bytes);
            std::memcpy(compact + geometry.scale_offset + slot * scale_row_bytes,
                        source_scales + row * scale_row_bytes, scale_row_bytes);
        }
    } break;
    default:
        throw std::logic_error("embedding staging: unsupported table encoding");
    }
}

} // namespace

std::uint64_t embedding_stage_capacity_bytes(QType qtype, std::int32_t features, std::int32_t ids) {
    if (features <= 0 || ids <= 0) {
        throw std::invalid_argument("embedding staging capacity needs positive features and ids");
    }
    const auto count = static_cast<std::uint64_t>(ids);
    return slot_bytes(count) + compact_geometry(qtype, features, count).bytes;
}

EmbeddingStage stage_embedding_rows(std::span<const std::int32_t> ids, const Weight& table,
                                    std::span<std::byte> image) {
    require_stageable(table);
    if (ids.empty() ||
        ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("embedding staging: ids must be a nonempty int32 extent");
    }
    const std::int32_t vocabulary = table.shape[0];
    std::vector<std::int32_t> distinct(ids.begin(), ids.end());
    for (const std::int32_t id : distinct) {
        if (id < 0 || id >= vocabulary) {
            throw std::out_of_range("embedding staging: id is outside the table");
        }
    }
    // Ascending distinct rows: tile neighbours share host cache lines during the gather.
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());

    const std::uint64_t rows         = distinct.size();
    const CompactGeometry geometry   = compact_geometry(table.qtype, table.shape[1], rows);
    const std::uint64_t table_offset = slot_bytes(ids.size());
    const std::uint64_t bytes        = table_offset + geometry.bytes;
    if (bytes > image.size() || rows > static_cast<std::uint64_t>(vocabulary)) {
        throw std::length_error("embedding staging: image is smaller than the staged rows");
    }

    auto* slots = reinterpret_cast<std::int32_t*>(image.data());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        slots[i] = static_cast<std::int32_t>(
            std::lower_bound(distinct.begin(), distinct.end(), ids[i]) - distinct.begin());
    }

    std::byte* compact = image.data() + table_offset;
    const std::size_t workers =
        std::clamp<std::size_t>(distinct.size() / kRowsPerWorker, 1, kMaximumWorkers);
    // Whole 16-row blocks per worker (16 scale rows fill whole cache lines) so no two workers
    // write the same cache lines.
    const std::size_t share =
        static_cast<std::size_t>(align_up((distinct.size() + workers - 1) / workers, 16U));
    // jthread joins on unwinding, so a throwing launch cannot terminate the process.
    std::vector<std::jthread> helpers;
    helpers.reserve(workers - 1);
    for (std::size_t worker = 1; worker < workers; ++worker) {
        const std::size_t first = std::min(distinct.size(), worker * share);
        const std::size_t last  = std::min(distinct.size(), first + share);
        helpers.emplace_back(
            [&, first, last] { copy_rows(table, distinct, first, last, compact, geometry); });
    }
    copy_rows(table, distinct, 0, std::min(distinct.size(), share), compact, geometry);
    for (std::jthread& helper : helpers) { helper.join(); }
    return EmbeddingStage{.ids   = static_cast<std::int32_t>(ids.size()),
                          .rows  = static_cast<std::int32_t>(rows),
                          .bytes = bytes};
}

StagedEmbedding staged_embedding(const EmbeddingStage& stage, const Weight& table,
                                 void* device_image) {
    require_stageable(table);
    if (device_image == nullptr || stage.ids <= 0 || stage.rows <= 0 ||
        stage.rows > table.shape[0]) {
        throw std::invalid_argument("staged embedding: invalid stage");
    }
    const auto rows                  = static_cast<std::uint64_t>(stage.rows);
    const CompactGeometry geometry   = compact_geometry(table.qtype, table.shape[1], rows);
    const std::uint64_t table_offset = slot_bytes(static_cast<std::uint64_t>(stage.ids));
    if (stage.bytes != table_offset + geometry.bytes) {
        throw std::invalid_argument("staged embedding: stage does not match the table encoding");
    }
    auto* base = static_cast<std::byte*>(device_image);
    StagedEmbedding out;
    out.slots               = Tensor(base, DType::I32, {stage.ids});
    out.table               = table;
    std::byte* compact      = base + table_offset;
    out.table.payload       = compact;
    out.table.payload_bytes = geometry.bytes;
    out.table.qdata         = compact;
    out.table.qdata_bytes   = geometry.code_bytes;
    if (table.qtype != QType::BF16_CTRL) {
        out.table.scales      = compact + geometry.scale_offset;
        out.table.scale_bytes = geometry.scale_bytes;
    }
    out.table.n               = stage.rows;
    out.table.shape[0]        = stage.rows;
    out.table.padded_shape[0] = stage.rows;
    return out;
}

} // namespace ninfer::ops
