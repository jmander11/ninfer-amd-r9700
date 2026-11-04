#include "artifact/reader.h"
#include "ninfer/engine.h"
#include "artifact_fixture.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace artifact = ninfer::artifact;
namespace target   = ninfer::targets::qwen3_8_27b::detail;
using Json         = nlohmann::json;
using Backend      = ninfer::SpeculativeBackend;

void exercise(artifact::NumericFormat format, artifact::StorageLayout layout) {
    Json directory = {
        {"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "r9700-lq4"}}},
        {"objects", Json::array()},
    };
    std::uint64_t offset = 0;
    const auto add       = [&](std::string_view name, std::vector<std::uint64_t> shape,
                               artifact::NumericFormat dtype, artifact::StorageLayout storage) {
        offset           = (offset + 255) / 256 * 256;
        const auto bytes = artifact::tensor_encoded_size(storage, dtype, shape);
        directory["objects"].push_back({{"name", name},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(dtype)},
                                        {"layout", artifact::layout_name(storage)},
                                        {"offset", offset},
                                        {"bytes", bytes}});
        offset += bytes;
    };
    const auto bf16 = [&](std::string_view name) {
        add(name, {128}, artifact::NumericFormat::BF16, artifact::StorageLayout::ContiguousLeV1);
    };
    for (const auto name :
         {"text/token_embedding", "text/final_norm", "text/output_head", "text/draft_head",
          "text/draft_head_token_ids", "vision/patch_embedding", "mtp/embedding_norm",
          "mtp/hidden_norm", "mtp/layer/input_norm", "mtp/layer/attention/key_norm",
          "mtp/layer/attention/query_norm", "mtp/layer/mlp/down", "mtp/final_norm",
          "dflash/context_norm", "dflash/layers/0/attention/key_norm",
          "dflash/layers/0/attention/query_norm", "dflash/layers/0/mlp/down", "dflash/final_norm",
          "dflash/selector/predecessor_codebook"}) {
        bf16(name);
    }
    // The output-row boundaries match the real projections; K=128 keeps the fixture small
    // while exercising native tile planes and row-split groups.
    add("mtp/input_projection", {128, 128}, format, layout);
    add("mtp/layer/attention/query_key_gate_value", {14336, 128}, format, layout);
    add("dflash/feature_projection", {128, 128}, format, layout);
    add("dflash/layers/0/attention/query_key_value", {6144, 128}, format, layout);
    auto fixture = ninfer::test::artifact_fixture::write_fixture(directory, "cache-dependencies");
    const auto digest = [&](Backend backend) {
        const artifact::Reader reader(fixture.path);
        return reader.tensor_content_identity(
            target::cache_tensor_selections(reader, {.vision = false, .speculative = backend}));
    };
    constexpr std::array backends{Backend::None, Backend::DFlash, Backend::Mtp};
    const std::array baseline{digest(backends[0]), digest(backends[1]), digest(backends[2])};
    const auto mutation = [&](std::string_view name, std::uint64_t relative_offset,
                              std::array<bool, 3> affects) {
        std::uint64_t absolute = 0;
        {
            const artifact::Reader reader(fixture.path);
            const auto payload = reader.payload(name);
            if (relative_offset >= payload.data.size()) {
                throw std::runtime_error("bad fixture mutation");
            }
            absolute = payload.absolute_offset + relative_offset;
        }
        char original    = 0;
        const auto write = [&](char byte) {
            std::fstream file(fixture.path, std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(static_cast<std::streamoff>(absolute));
            file.put(byte);
            if (!file) { throw std::runtime_error("cannot mutate content fixture"); }
        };
        {
            std::ifstream file(fixture.path, std::ios::binary);
            file.seekg(static_cast<std::streamoff>(absolute));
            file.get(original);
            if (!file) { throw std::runtime_error("cannot read content fixture"); }
        }
        write(static_cast<char>(original ^ 1));
        for (std::size_t i = 0; i < backends.size(); ++i) {
            if ((digest(backends[i]) != baseline[i]) != affects[i]) {
                throw std::runtime_error("incorrect retained-state dependency: " +
                                         std::string(name));
            }
        }
        write(original);
    };
    constexpr std::array all{true, true, true};
    constexpr std::array none{false, false, false};
    constexpr std::array dflash{false, true, false};
    constexpr std::array mtp{false, false, true};
    for (const auto name : {"text/token_embedding", "text/final_norm", "vision/patch_embedding"})
        mutation(name, 0, all);
    for (const auto name : {"text/output_head", "text/draft_head", "text/draft_head_token_ids",
                            "dflash/layers/0/attention/query_norm", "dflash/layers/0/mlp/down",
                            "dflash/final_norm", "dflash/selector/predecessor_codebook"})
        mutation(name, 0, none);
    for (const auto name :
         {"mtp/embedding_norm", "mtp/hidden_norm", "mtp/layer/input_norm",
          "mtp/layer/attention/key_norm", "mtp/layer/attention/query_norm", "mtp/layer/mlp/down",
          "mtp/final_norm", "mtp/input_projection", "mtp/layer/attention/query_key_gate_value"})
        mutation(name, 0, mtp);
    for (const auto name :
         {"dflash/feature_projection", "dflash/context_norm", "dflash/layers/0/attention/key_norm"})
        mutation(name, 0, dflash);
    const auto name = "dflash/layers/0/attention/query_key_value";
    const std::array<std::uint64_t, 2> shape{6144, 128};
    std::vector<std::pair<std::uint64_t, std::uint64_t>> planes;
    if (layout == artifact::StorageLayout::ContiguousLeV1) {
        planes.push_back({0, 256});
    } else if (layout == artifact::StorageLayout::RowScaledK128V1) {
        const auto g = artifact::row_scaled_geometry(format, shape);
        planes       = {{0, g.padded_columns}, {g.scale_plane_offset, 4}};
    } else if (layout == artifact::StorageLayout::R9700Lq4N16K64V1) {
        const auto g = artifact::r9700_lq4_geometry(shape);
        planes       = {{0, g.code_plane_bytes / g.rows},
                        {g.group_plane_offset, g.group_plane_bytes / g.rows},
                        {g.scale_plane_offset, 4}};
    } else {
        const auto g = layout == artifact::StorageLayout::R9700Q4G64N16K16V1
                           ? artifact::r9700_q4g64_n16k16_geometry(shape)
                       : layout == artifact::StorageLayout::R9700W8G32N16K16V1
                           ? artifact::r9700_w8g32_n16k16_geometry(shape)
                           : artifact::row_split_geometry(format, shape);
        planes       = {{0, g.low_plane_bytes / g.rows},
                        {g.scale_plane_offset, g.scale_plane_bytes / g.rows}};
        if (g.high_plane_bytes)
            planes.push_back({g.high_plane_offset, g.high_plane_bytes / g.rows});
    }
    for (const auto& [plane_offset, stride] : planes) {
        mutation(name, plane_offset, none);
        mutation(name, plane_offset + 4096 * stride - 1, none);
        mutation(name, plane_offset + 4096 * stride, dflash);
        mutation(name, plane_offset + 5120 * stride, dflash);
        mutation(name, plane_offset + 6144 * stride - 1, dflash);
    }
    // Location/inode changes do not invalidate identical represented inputs.
    auto copy = fixture.path;
    copy += ".copy";
    std::filesystem::copy_file(fixture.path, copy,
                               std::filesystem::copy_options::overwrite_existing);
    const artifact::Reader copied(copy);
    if (copied.tensor_content_identity(target::cache_tensor_selections(
            copied, {.vision = false, .speculative = Backend::DFlash})) != baseline[1])
        throw std::runtime_error("relocation changed cache identity");
    std::filesystem::remove(copy);
}
} // namespace

int main() {
    try {
        exercise(artifact::NumericFormat::BF16, artifact::StorageLayout::ContiguousLeV1);
        exercise(artifact::NumericFormat::F8E4M3_ROW_F32S,
                 artifact::StorageLayout::RowScaledK128V1);
        exercise(artifact::NumericFormat::Q4G64_F16S, artifact::StorageLayout::RowSplitK128V1);
        exercise(artifact::NumericFormat::Q5G64_F16S, artifact::StorageLayout::RowSplitK128V1);
        exercise(artifact::NumericFormat::Q6G64_F16S, artifact::StorageLayout::RowSplitK128V1);
        exercise(artifact::NumericFormat::W8G32_F16S, artifact::StorageLayout::RowSplitK128V1);
        exercise(artifact::NumericFormat::Q4G64_F16S, artifact::StorageLayout::R9700Q4G64N16K16V1);
        exercise(artifact::NumericFormat::W8G32_F16S, artifact::StorageLayout::R9700W8G32N16K16V1);
        exercise(artifact::NumericFormat::LQ4, artifact::StorageLayout::R9700Lq4N16K64V1);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
