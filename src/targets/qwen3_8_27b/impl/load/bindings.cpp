#include "targets/qwen3_8_27b/impl/load/bindings.h"

#include "artifact/typed_binding.h"
#include "targets/qwen3_8_27b/impl/load/fp8_hybrid_selection.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

NumericFormat matrix_format(WeightsProfile profile, bool source_q4) {
    switch (profile) {
    case WeightsProfile::R9700W8G32Candidate: return NumericFormat::W8G32_F16S;
    case WeightsProfile::R9700W8Bf16EmbeddingEvaluation: return NumericFormat::W8G32_F16S;
    case WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::R9700Q4G64Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation:
    case WeightsProfile::R9700Q4G64DFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation:
        return NumericFormat::Q4G64_F16S;
    case WeightsProfile::R9700Q4W8Evaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation:
        return source_q4 ? NumericFormat::Q4G64_F16S : NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3_8_27b_r9700: invalid matrix profile");
}

NumericFormat dflash_matrix_format(WeightsProfile profile) {
    if (profile == WeightsProfile::R9700Q4G64DFlash2Q4Evaluation ||
        profile == WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation ||
        profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation) {
        return NumericFormat::Q4G64_F16S;
    }
    // No other registered identity currently carries DFlash objects. Keep this explicit so a
    // future companion cannot silently inherit its base matrix format.
    throw std::invalid_argument("qwen3_8_27b_r9700: profile has no DFlash2 matrix recipe");
}

NumericFormat token_embedding_format(WeightsProfile profile) {
    if (profile == WeightsProfile::R9700W8Bf16EmbeddingEvaluation) {
        return NumericFormat::BF16;
    }
    return matrix_format(profile, false);
}

NumericFormat full_attention_value_output_format(WeightsProfile profile) {
    if (profile == WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation) {
        return NumericFormat::BF16;
    }
    return matrix_format(profile, false);
}

NumericFormat selected_fp8_role_format(WeightsProfile profile, std::string_view name,
                                       NumericFormat fallback) {
    if (profile != WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation &&
        profile != WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation) {
        return fallback;
    }
    if (!fp8_hybrid::contains(name)) {
        throw std::logic_error("FP8/Q4 hybrid binder role is absent from its decision authority");
    }
    return NumericFormat::F8E4M3_ROW_F32S;
}

NumericFormat full_attention_query_key_format(WeightsProfile profile, std::string_view name) {
    if (profile == WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation) {
        return NumericFormat::BF16;
    }
    return selected_fp8_role_format(profile, name, matrix_format(profile, true));
}

NumericFormat full_attention_gate_value_format(WeightsProfile profile, std::string_view name) {
    if (profile == WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation) {
        return NumericFormat::BF16;
    }
    return selected_fp8_role_format(profile, name, matrix_format(profile, false));
}

NumericFormat gdn_query_key_format(WeightsProfile profile, std::string_view name) {
    if (profile == WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation) {
        return NumericFormat::BF16;
    }
    return selected_fp8_role_format(profile, name, matrix_format(profile, true));
}

NumericFormat mlp_gate_up_format(WeightsProfile profile, std::string_view name) {
    return selected_fp8_role_format(profile, name, matrix_format(profile, true));
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

WeightPlan bind_mtp_matrix(artifact::Binder& binder, std::string_view name,
                           std::initializer_list<std::uint64_t> shape,
                           artifact::TensorPlacement placement, WeightsProfile profile) {
    const NumericFormat format = matrix_format(profile, false);
    return WeightPlan{.object = artifact::bind_tensor(binder, name, format, shape, placement),
                      .format = format};
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        (block.qtype != QType::W8G32_F16S && block.qtype != QType::Q4G64_F16S) ||
        block.layout != (block.qtype == QType::Q4G64_F16S
                            ? QuantLayout::Q4N16K16 : QuantLayout::RowSplit) ||
        (block.qtype == QType::Q4G64_F16S &&
         (row_begin % 16 != 0 || row_count % 16 != 0)) ||
        block.group != (block.qtype == QType::Q4G64_F16S ? 64 : 32) ||
        block.qhigh != nullptr) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t code_row  = groups * 32;
    const std::uint64_t scale_row = groups * 2;
    Weight out                    = block;
    out.qdata = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * code_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.qdata_bytes      = static_cast<std::uint64_t>(row_count) * code_row;
    out.scale_bytes      = static_cast<std::uint64_t>(row_count) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, 34816, 5120);
    out.down    = materialized_weight(materialized, plan.down, 5120, 17408);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    return FullAttentionProjectionPayload{
        .query_key = materialized_weight(materialized, plan.projection.query_key, 7168, 5120),
        .gate_value = materialized_weight(materialized, plan.projection.gate_value, 7168, 5120),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    return GdnInputProjectionPayload{
        .query_key = materialized_weight(materialized, plan.input_projection.query_key, 4096, 5120),
        .value_z = materialized_weight(materialized, plan.input_projection.value_z, 12288, 5120),
    };
}

void bind_r9700_text_layers(artifact::Binder& binder, BindingPlan& out,
                            WeightsProfile profile) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = AttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          full_attention_query_key_format(
                                              profile, prefix + "attention/query_key"),
                                          {7168, 5120}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          full_attention_gate_value_format(
                                              profile, prefix + "attention/gate_value"),
                                          {7168, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  full_attention_value_output_format(profile),
                                                  {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.a_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/b_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.input_projection = GdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         gdn_query_key_format(profile, prefix + "gdn/query_key"),
                                         {4096, 5120}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", matrix_format(profile, false),
                                         {12288, 5120}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", matrix_format(profile, false),
                            {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up",
                        mlp_gate_up_format(profile, prefix + "mlp/gate_up"),
                        {34816, 5120});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", matrix_format(profile, false), {5120, 17408});
    }
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3::StartupFeatures features) {
    (void)matrix_format(weights_profile, false);
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3::bind_frontend_resources(binder);
    out.features     = features;

    out.token_embedding =
        bind_weight(binder, "text/token_embedding", token_embedding_format(weights_profile),
                    {248320, 5120});
    bind_r9700_text_layers(binder, out, weights_profile);
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {5120});
    out.output_head = bind_weight(binder, "text/output_head", matrix_format(weights_profile, false),
                                  {248320, 5120});
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    const NumericFormat draft_format = matrix_format(weights_profile, true);
    out.draft_head = WeightPlan{
        .object = artifact::bind_tensor(binder, "text/draft_head", draft_format,
                                        {131072, 5120}, proposal_placement),
        .format = draft_format};
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp_matrix(binder, "mtp/input_projection", {5120, 10240}, mtp_placement,
                        weights_profile);
    out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm    = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm     = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value = bind_mtp_matrix(
        binder, "mtp/layer/attention/query_key_gate_value", {14336, 5120}, mtp_placement,
        weights_profile);
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        bind_mtp_matrix(binder, "mtp/layer/attention/output", {5120, 6144}, mtp_placement,
                        weights_profile);
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up =
        bind_mtp_matrix(binder, "mtp/layer/mlp/gate_up", {34816, 5120}, mtp_placement,
                        weights_profile);
    out.mtp.mlp.down =
        bind_mtp_matrix(binder, "mtp/layer/mlp/down", {5120, 17408}, mtp_placement,
                        weights_profile);
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {5120});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    const qwen3::VisionMatrixFormats vision_formats{
        .source_q4 = matrix_format(weights_profile, true),
        .other = matrix_format(weights_profile, false)};
    out.vision_backbone = qwen3::bind_vision_backbone(binder, vision_formats, vision_placement);
    out.vision_merger_input = qwen3::bind_vision_merger_input(binder, vision_formats,
                                                              vision_placement);
    out.vision_merger_fc2 = WeightPlan{
        .object = artifact::bind_tensor(binder, "vision/merger/fc2", vision_formats.other,
                                        {5120, 4608}, vision_placement),
        .format = vision_formats.other};
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120}, vision_placement);
    out.vision_merger_norm = qwen3::bind_vision_merger_norm(binder, vision_placement);

    const bool has_dflash = binder.contains("dflash/feature_projection");
    const bool profile_requires_dflash =
        weights_profile == WeightsProfile::R9700Q4G64DFlash2Q4Evaluation ||
        weights_profile == WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation ||
        weights_profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation;
    if (profile_requires_dflash && !has_dflash) {
        throw artifact::ArtifactError(
            "DFlash2 evaluation identity is missing its DFlash2 object inventory");
    }
    if (features.dflash() && !has_dflash) {
        throw artifact::ArtifactError(
            "DFlash requires dflash/ objects in the artifact; reconvert with --dflash-model");
    }
    if (has_dflash) {
        const artifact::TensorPlacement dflash_placement =
            features.dflash() ? artifact::TensorPlacement::Device
                              : artifact::TensorPlacement::ValidateOnly;
        DFlash2Plan& dflash = out.dflash.emplace();
        const NumericFormat dflash_format = dflash_matrix_format(weights_profile);
        const auto bind_matrix = [&](std::string_view name,
                                     std::initializer_list<std::uint64_t> shape) {
            return WeightPlan{
                .object = artifact::bind_tensor(binder, name, dflash_format, shape,
                                                dflash_placement),
                .format = dflash_format};
        };
        const auto bind_bf16 = [&](std::string_view name,
                                   std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_tensor(binder, name, NumericFormat::BF16, shape, dflash_placement);
        };
        dflash.feature_projection = bind_matrix("dflash/feature_projection", {5120, 25600});
        dflash.context_norm       = bind_bf16("dflash/context_norm", {5120});
        for (std::size_t layer = 0; layer < kDFlash2Layers; ++layer) {
            DFlash2LayerPlan& target  = dflash.layers[layer];
            const std::string prefix  = "dflash/layers/" + std::to_string(layer) + "/";
            target.input_norm         = bind_bf16(prefix + "input_norm", {5120});
            target.query_key_value    = bind_matrix(prefix + "attention/query_key_value", {6144, 5120});
            target.query_norm         = bind_bf16(prefix + "attention/query_norm", {128});
            target.key_norm           = bind_bf16(prefix + "attention/key_norm", {128});
            target.attention_output   = bind_matrix(prefix + "attention/output", {5120, 4096});
            target.attention_conv.base_kernel =
                bind_bf16(prefix + "attention_conv/base_kernel", {5120, 2, 2});
            target.attention_conv.kernel_projection =
                bind_matrix(prefix + "attention_conv/kernel_projection", {1280, 5120});
            target.post_attention_norm = bind_bf16(prefix + "post_attention_norm", {5120});
            target.gate_up             = bind_matrix(prefix + "mlp/gate_up", {34816, 5120});
            target.down                = bind_matrix(prefix + "mlp/down", {5120, 17408});
            target.mlp_conv.base_kernel =
                bind_bf16(prefix + "mlp_conv/base_kernel", {5120, 2, 2});
            target.mlp_conv.kernel_projection =
                bind_matrix(prefix + "mlp_conv/kernel_projection", {1280, 5120});
        }
        dflash.final_norm        = bind_bf16("dflash/final_norm", {5120});
        dflash.hidden_projection = bind_matrix("dflash/selector/hidden_projection", {256, 5120});
        dflash.predecessor_codebook =
            bind_bf16("dflash/selector/predecessor_codebook", {256, 248320});
        dflash.successor_codebook =
            bind_bf16("dflash/selector/successor_codebook", {256, 248320});
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding        = materialized_weight(backing, plan.token_embedding, 248320, 5120);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {5120});
            target.projection            = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = materialized_weight(backing, source.attention.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {5120});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {48});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {48});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {10240, 4});
            target.projection.a_projection = artifact::materialized_weight(
                backing, source.gdn.a_projection, NumericFormat::BF16, 48, 5120);
            target.projection.b_projection = artifact::materialized_weight(
                backing, source.gdn.b_projection, NumericFormat::BF16, 48, 5120);
            target.projection.input_projection = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output = materialized_weight(backing, source.gdn.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {5120});
    output_head = materialized_weight(backing, plan.output_head, 248320, 5120);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head = materialized_weight(backing, plan.draft_head, 131072, 5120);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = materialized_weight(backing, plan.mtp.input_projection, 5120, 10240);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.attention.packed =
            materialized_weight(backing, plan.mtp.query_key_gate_value, 14336, 5120);
        if (mtp.attention.packed.layout == QuantLayout::RowSplit ||
            mtp.attention.packed.layout == QuantLayout::Q4N16K16) {
            mtp.attention.query       = row_view(mtp.attention.packed, 0, 6144);
            mtp.attention.key         = row_view(mtp.attention.packed, 6144, 1024);
            mtp.attention.output_gate = row_view(mtp.attention.packed, 7168, 6144);
            mtp.attention.value       = row_view(mtp.attention.packed, 13312, 1024);
        }
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {256});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {256});
        mtp.output = materialized_weight(backing, plan.mtp.output, 5120, 6144);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {5120});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2 = materialized_weight(backing, plan.vision_merger_fc2, 5120, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {5120});
    }

    if (plan.features.dflash()) {
        if (!plan.dflash.has_value()) {
            throw std::logic_error("DFlash startup is missing a DFlash2 bind plan");
        }
        const DFlash2Plan& source = *plan.dflash;
        auto& target              = runtime.dflash.emplace();
        target.feature_projection =
            materialized_weight(backing, source.feature_projection, 5120, 25600);
        target.context_norm = artifact::materialized_tensor(backing, source.context_norm,
                                                            NumericFormat::BF16, {5120});
        for (std::size_t layer = 0; layer < kDFlash2Layers; ++layer) {
            const DFlash2LayerPlan& layer_plan = source.layers[layer];
            DFlash2LayerWeights& weights       = target.layers[layer];
            weights.input_norm = artifact::materialized_tensor(backing, layer_plan.input_norm,
                                                               NumericFormat::BF16, {5120});
            weights.query_key_value =
                materialized_weight(backing, layer_plan.query_key_value, 6144, 5120);
            weights.query_norm = artifact::materialized_tensor(
                backing, layer_plan.query_norm, NumericFormat::BF16, {128});
            weights.key_norm = artifact::materialized_tensor(backing, layer_plan.key_norm,
                                                             NumericFormat::BF16, {128});
            weights.attention_output =
                materialized_weight(backing, layer_plan.attention_output, 5120, 4096);
            weights.attention_conv.base_kernel = artifact::materialized_tensor(
                backing, layer_plan.attention_conv.base_kernel, NumericFormat::BF16, {5120, 2, 2});
            weights.attention_conv.kernel_projection = materialized_weight(
                backing, layer_plan.attention_conv.kernel_projection, 1280, 5120);
            weights.post_attention_norm = artifact::materialized_tensor(
                backing, layer_plan.post_attention_norm, NumericFormat::BF16, {5120});
            weights.gate_up = materialized_weight(backing, layer_plan.gate_up, 34816, 5120);
            weights.down    = materialized_weight(backing, layer_plan.down, 5120, 17408);
            weights.mlp_conv.base_kernel = artifact::materialized_tensor(
                backing, layer_plan.mlp_conv.base_kernel, NumericFormat::BF16, {5120, 2, 2});
            weights.mlp_conv.kernel_projection =
                materialized_weight(backing, layer_plan.mlp_conv.kernel_projection, 1280, 5120);
        }
        target.final_norm =
            artifact::materialized_tensor(backing, source.final_norm, NumericFormat::BF16, {5120});
        target.hidden_projection =
            materialized_weight(backing, source.hidden_projection, 256, 5120);
        target.predecessor_codebook = artifact::materialized_tensor(
            backing, source.predecessor_codebook, NumericFormat::BF16, {256, 248320});
        target.successor_codebook = artifact::materialized_tensor(
            backing, source.successor_codebook, NumericFormat::BF16, {256, 248320});
    }
}

} // namespace ninfer::targets::qwen3_8_27b::detail
