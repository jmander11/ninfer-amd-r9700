// Exact oracle for host staging of prompt embedding rows (ops::stage_embedding_rows): every staged
// slot must carry the logical codes and scales (or BF16 values) of its source row, decoded here
// from the storage-layout definitions rather than from the staging implementation.
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Weight;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint64_t align(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

// r9700-q4g64-n16k16-v1: feature f of row r is nibble f%16 of the 8-byte word
// ((r/16*G + f/64)*4 + f%64/16)*16 + r%16; the group-g FP16 scale is at (r/16*G + g)*16 + r%16.
int q4_code(const std::uint8_t* codes, std::uint32_t groups, std::size_t row,
            std::uint32_t feature) {
    const std::size_t word =
        (((row / 16U) * groups + feature / 64U) * 4U + (feature % 64U) / 16U) * 16U + row % 16U;
    const std::uint8_t byte = codes[word * 8U + (feature % 16U) / 2U];
    const int nibble = (feature % 2U) == 0U ? (byte & 0x0f) : (byte >> 4U);
    return nibble >= 8 ? nibble - 16 : nibble;
}

std::uint16_t q4_scale(const std::uint16_t* scales, std::uint32_t groups, std::size_t row,
                       std::uint32_t group) {
    return scales[((row / 16U) * groups + group) * 16U + row % 16U];
}

struct Table {
    std::vector<std::uint8_t> payload;
    Weight weight{};
};

Table make_q4(std::int32_t vocabulary, std::int32_t features) {
    const auto padded = static_cast<std::uint32_t>(align(static_cast<std::uint64_t>(features), 128));
    const std::uint32_t groups = padded / 64U;
    const std::uint64_t code_bytes = static_cast<std::uint64_t>(vocabulary) * padded / 2U;
    const std::uint64_t scale_offset = align(code_bytes, 256);
    const std::uint64_t scale_bytes = static_cast<std::uint64_t>(vocabulary) * groups * 2U;
    Table table;
    table.payload.resize(scale_offset + scale_bytes);
    for (std::size_t i = 0; i < code_bytes; ++i) {
        table.payload[i] = static_cast<std::uint8_t>((i * 131U + 7U) & 0xffU);
    }
    auto* scales = reinterpret_cast<std::uint16_t*>(table.payload.data() + scale_offset);
    for (std::size_t i = 0; i < scale_bytes / 2U; ++i) {
        scales[i] = static_cast<std::uint16_t>(0x3000U + (i * 37U) % 0x0c00U);
    }
    Weight& w = table.weight;
    w.payload = w.qdata = table.payload.data();
    w.payload_bytes = table.payload.size();
    w.qdata_bytes = code_bytes;
    w.scales = table.payload.data() + scale_offset;
    w.scale_bytes = scale_bytes;
    w.qtype = QType::Q4G64_F16S;
    w.layout = QuantLayout::Q4N16K16;
    w.group_size = 64;
    w.group = 64;
    w.scale_dtype = DType::FP16;
    w.ndim = 2;
    w.n = w.shape[0] = w.padded_shape[0] = vocabulary;
    w.k = w.shape[1] = features;
    w.padded_shape[1] = static_cast<std::int32_t>(padded);
    return table;
}

Table make_w8(std::int32_t vocabulary, std::int32_t features) {
    const auto padded = static_cast<std::uint32_t>(align(static_cast<std::uint64_t>(features), 128));
    const std::uint32_t groups = padded / 32U;
    const std::uint64_t code_bytes = static_cast<std::uint64_t>(vocabulary) * padded;
    const std::uint64_t scale_offset = align(code_bytes, 256);
    const std::uint64_t scale_bytes = static_cast<std::uint64_t>(vocabulary) * groups * 2U;
    Table table;
    table.payload.resize(scale_offset + scale_bytes);
    for (std::size_t i = 0; i < table.payload.size(); ++i) {
        table.payload[i] = static_cast<std::uint8_t>((i * 29U + 3U) & 0xffU);
    }
    Weight& w = table.weight;
    w.payload = w.qdata = table.payload.data();
    w.payload_bytes = table.payload.size();
    w.qdata_bytes = code_bytes;
    w.scales = table.payload.data() + scale_offset;
    w.scale_bytes = scale_bytes;
    w.qtype = QType::W8G32_F16S;
    w.layout = QuantLayout::RowSplit;
    w.group_size = 32;
    w.group = 32;
    w.scale_dtype = DType::FP16;
    w.ndim = 2;
    w.n = w.shape[0] = w.padded_shape[0] = vocabulary;
    w.k = w.shape[1] = features;
    w.padded_shape[1] = static_cast<std::int32_t>(padded);
    return table;
}

Table make_bf16(std::int32_t vocabulary, std::int32_t features) {
    Table table;
    table.payload.resize(static_cast<std::size_t>(vocabulary) * features * 2U);
    for (std::size_t i = 0; i < table.payload.size(); ++i) {
        table.payload[i] = static_cast<std::uint8_t>((i * 53U + 11U) & 0xffU);
    }
    Weight& w = table.weight;
    w.payload = w.qdata = table.payload.data();
    w.payload_bytes = w.qdata_bytes = table.payload.size();
    w.qtype = QType::BF16_CTRL;
    w.layout = QuantLayout::Contiguous;
    w.ndim = 2;
    w.n = w.shape[0] = w.padded_shape[0] = vocabulary;
    w.k = w.shape[1] = w.padded_shape[1] = features;
    return table;
}

// Compares every staged slot with its source row through the logical decode of each layout.
void require_rows(const Weight& source, const ninfer::ops::StagedEmbedding& staged,
                  const std::vector<std::int32_t>& ids, const std::string& label) {
    const auto* slots = static_cast<const std::int32_t*>(staged.slots.data);
    const Weight& compact = staged.table;
    const auto features = static_cast<std::uint32_t>(source.shape[1]);
    const auto padded = static_cast<std::uint32_t>(source.padded_shape[1]);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto slot = static_cast<std::size_t>(slots[i]);
        const auto row  = static_cast<std::size_t>(ids[i]);
        require(slot < static_cast<std::size_t>(compact.shape[0]), label + ": slot out of range");
        switch (source.qtype) {
        case QType::Q4G64_F16S:
            for (std::uint32_t f = 0; f < features; ++f) {
                require(q4_code(static_cast<const std::uint8_t*>(compact.qdata), padded / 64U,
                                slot, f) ==
                            q4_code(static_cast<const std::uint8_t*>(source.qdata), padded / 64U,
                                    row, f),
                        label + ": Q4 code differs");
            }
            for (std::uint32_t g = 0; g < padded / 64U; ++g) {
                require(q4_scale(static_cast<const std::uint16_t*>(compact.scales), padded / 64U,
                                 slot, g) ==
                            q4_scale(static_cast<const std::uint16_t*>(source.scales),
                                     padded / 64U, row, g),
                        label + ": Q4 scale differs");
            }
            break;
        case QType::W8G32_F16S:
            for (std::uint32_t f = 0; f < features; ++f) {
                require(static_cast<const std::int8_t*>(compact.qdata)[slot * padded + f] ==
                            static_cast<const std::int8_t*>(source.qdata)[row * padded + f],
                        label + ": W8 code differs");
            }
            for (std::uint32_t g = 0; g < padded / 32U; ++g) {
                require(static_cast<const std::uint16_t*>(compact.scales)[slot * (padded / 32U) + g] ==
                            static_cast<const std::uint16_t*>(source.scales)[row * (padded / 32U) + g],
                        label + ": W8 scale differs");
            }
            break;
        case QType::BF16_CTRL:
            for (std::uint32_t f = 0; f < features; ++f) {
                require(static_cast<const std::uint16_t*>(compact.qdata)[slot * features + f] ==
                            static_cast<const std::uint16_t*>(source.qdata)[row * features + f],
                        label + ": BF16 value differs");
            }
            break;
        default:
            throw std::logic_error("unexpected encoding");
        }
    }
}

void check_table(const Table& table, const std::string& label) {
    const Weight& source = table.weight;
    const std::int32_t vocabulary = source.shape[0];
    std::vector<std::int32_t> ids;
    for (std::int32_t i = 0; i < 3 * vocabulary; ++i) {
        ids.push_back((i * 7 + i / 5) % vocabulary);
        if (i % 4 == 0) { ids.push_back(vocabulary - 1); }
    }
    std::vector<std::int32_t> subset(ids.begin(), ids.begin() + 5);
    for (const auto& window : {ids, subset}) {
        const std::uint64_t capacity = ninfer::ops::embedding_stage_capacity_bytes(
            source.qtype, source.shape[1], static_cast<std::int32_t>(window.size()));
        // The image is used as both the host staging buffer and its "device" copy.
        std::vector<std::byte> image(capacity, std::byte{0xa5});
        const ninfer::ops::EmbeddingStage stage =
            ninfer::ops::stage_embedding_rows(window, source, image);
        const std::size_t distinct = std::set<std::int32_t>(window.begin(), window.end()).size();
        const std::size_t rows = source.qtype == QType::Q4G64_F16S ? align(distinct, 16) : distinct;
        require(stage.ids == static_cast<std::int32_t>(window.size()) &&
                    stage.rows == static_cast<std::int32_t>(rows) && stage.bytes <= capacity,
                label + ": stage extent");
        const ninfer::ops::StagedEmbedding staged =
            ninfer::ops::staged_embedding(stage, source, image.data());
        require(staged.slots.ne[0] == stage.ids && staged.table.shape[0] == stage.rows &&
                    staged.table.shape[1] == source.shape[1] &&
                    static_cast<const std::byte*>(staged.table.payload) +
                            staged.table.payload_bytes ==
                        image.data() + stage.bytes,
                label + ": staged view geometry");
        require_rows(source, staged, window, label);
        if (source.qtype == QType::Q4G64_F16S) {
            const auto padded = static_cast<std::uint32_t>(source.padded_shape[1]);
            for (std::size_t slot = distinct; slot < rows; ++slot) {
                for (std::uint32_t f = 0; f < static_cast<std::uint32_t>(source.shape[1]); ++f) {
                    require(q4_code(static_cast<const std::uint8_t*>(staged.table.qdata),
                                    padded / 64U, slot, f) == 0,
                            label + ": Q4 tile padding is not zero");
                }
            }
        }
        std::vector<std::byte> small(stage.bytes - 1U);
        bool rejected = false;
        try {
            (void)ninfer::ops::stage_embedding_rows(window, source, small);
        } catch (const std::length_error&) { rejected = true; }
        require(rejected, label + ": undersized image accepted");
    }
    bool rejected = false;
    try {
        const std::vector<std::int32_t> outside{0, vocabulary};
        std::vector<std::byte> image(ninfer::ops::embedding_stage_capacity_bytes(
            source.qtype, source.shape[1], 2));
        (void)ninfer::ops::stage_embedding_rows(outside, source, image);
    } catch (const std::out_of_range&) { rejected = true; }
    require(rejected, label + ": out-of-range id accepted");
}

} // namespace

int main() {
    try {
        // Q4: 3 tiles, K padded 320 -> 384; enough distinct rows to split across workers is
        // covered by the large table below.
        check_table(make_q4(48, 320), "q4");
        check_table(make_q4(4096, 256), "q4-workers");
        check_table(make_w8(40, 200), "w8");
        check_table(make_bf16(30, 24), "bf16");
        std::cout << "embedding_staging: PASS q4 w8 bf16 exact rows, tile padding, bounds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "embedding_staging: FAIL: " << error.what() << '\n';
        return 1;
    }
}
