#include "artifact/binder.h"
#include "artifact/reader.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include "targets/qwen3_8_27b/impl/load/fp8_hybrid_selection.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#include <ninfer/targets/qwen3_8_27b/package.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::QType;
using ninfer::targets::qwen3_8_27b::Package;
using ninfer::targets::qwen3_8_27b::detail::ArtifactLoadPlan;
using ninfer::targets::qwen3_8_27b::detail::WeightsProfile;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void require_w8_workspace_profile(WeightsProfile profile) {
    const bool linear =
        ninfer::targets::qwen3_8_27b::detail::Variant::linear_workspace_capacity_bytes(
            profile, 128) != 0;
    const bool vision = ninfer::targets::qwen3_8_27b::detail::Variant::
                            vision_linear_workspace_capacity_bytes(profile, 128) != 0;
    const bool expected = ninfer::ops::r9700::linear::kW8ActivationBits == 8;
    require(linear == expected && vision == expected,
            "W8 evaluation workspace does not match the compile-selected activation profile");
}

ArtifactLoadPlan bind_complete(const std::filesystem::path& path,
                               std::string_view expected_weights_id) {
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{
                                     "qwen3.8-27b", std::string(expected_weights_id)},
            "sparse artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) ==
                WeightsProfile::R9700W8G32Candidate,
            "package resolved the candidate to the wrong profile");
    const auto defaults = Package::sampling_defaults(reader.identity().model_id);
    require(defaults.thinking.temperature == 0.6F && defaults.thinking.top_k == 20 &&
                defaults.thinking.top_p == 0.95F,
            "Qwen3.8 sampling defaults changed");

    std::size_t bf16 = 0;
    std::size_t fp32 = 0;
    std::size_t i32  = 0;
    std::size_t w8   = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        switch (tensor->format) {
        case NumericFormat::BF16:
            ++bf16;
            break;
        case NumericFormat::FP32:
            ++fp32;
            break;
        case NumericFormat::I32:
            ++i32;
            break;
        case NumericFormat::W8G32_F16S:
            ++w8;
            break;
        default:
            throw std::runtime_error("candidate contains a non-R9700 matrix format");
        }
    }
    require(reader.objects().size() == 1124 && bf16 == 582 && fp32 == 96 && i32 == 1 &&
                w8 == 439,
            "candidate inventory/format counts differ from the exact contract");

    ninfer::artifact::Binder binder(reader);
    ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, WeightsProfile::R9700W8G32Candidate, features);
    require(plan.materialization.object_count == 1124 &&
                plan.materialization.device_objects.size() == 1118 &&
                plan.materialization.host_objects.size() == 6 &&
                plan.materialization.device_capacity_bytes > 0,
            "candidate materialization plan is incomplete");
    require(!plan.bindings.dflash.has_value(),
            "base R9700 candidate unexpectedly owns optional DFlash tensors");

    for (const auto& layer : plan.bindings.text_layers) {
        require(layer.mlp.gate_up.format == NumericFormat::W8G32_F16S &&
                    layer.mlp.down.format == NumericFormat::W8G32_F16S,
                "text MLP is not W8G32");
        if (layer.is_full_attention) {
            require(layer.attention.projection.query_key.format == NumericFormat::W8G32_F16S &&
                        layer.attention.projection.gate_value.format == NumericFormat::W8G32_F16S &&
                        layer.attention.output.format == NumericFormat::W8G32_F16S,
                    "full-attention binding does not match the candidate");
        } else {
            require(layer.gdn.input_projection.query_key.format == NumericFormat::W8G32_F16S &&
                        layer.gdn.input_projection.value_z.format == NumericFormat::W8G32_F16S &&
                        layer.gdn.output.format == NumericFormat::W8G32_F16S,
                    "GDN binding does not match the candidate");
        }
    }
    return plan;
}

ArtifactLoadPlan bind_evaluation(const std::filesystem::path& path,
                                 std::string_view weights_id, WeightsProfile profile,
                                 std::size_t expected_q4, std::size_t expected_w8) {
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{"qwen3.8-27b",
                                                                      std::string(weights_id)},
            "sparse evaluation artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) == profile,
            "package resolved an evaluation artifact to the wrong profile");

    std::size_t q4 = 0;
    std::size_t w8 = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        q4 += tensor->format == NumericFormat::Q4G64_F16S ? 1 : 0;
        w8 += tensor->format == NumericFormat::W8G32_F16S ? 1 : 0;
    }
    require(reader.objects().size() == 1124 && q4 == expected_q4 && w8 == expected_w8,
            "evaluation inventory/format counts differ from the exact contract");

    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, profile, features);
    require(plan.materialization.object_count == 1124 &&
                plan.materialization.device_objects.size() == 1118 &&
                plan.materialization.host_objects.size() == 6,
            "evaluation materialization plan is incomplete");

    const NumericFormat source = NumericFormat::Q4G64_F16S;
    const NumericFormat other = profile == WeightsProfile::R9700Q4G64Evaluation
                                    ? NumericFormat::Q4G64_F16S
                                    : NumericFormat::W8G32_F16S;
    require(plan.bindings.token_embedding.format == other &&
                plan.bindings.output_head.format == other &&
                plan.bindings.draft_head.format == source &&
                plan.bindings.mtp.input_projection.format == other &&
                plan.bindings.mtp.mlp.down.format == other &&
                plan.bindings.vision_backbone.patch_embedding.format == other &&
                plan.bindings.vision_backbone.layers[0].qkv.format == source &&
                plan.bindings.vision_backbone.layers[0].fc1.format == source &&
                plan.bindings.vision_backbone.layers[0].fc2.format == other &&
                plan.bindings.vision_merger_input.fc1.format == other &&
                plan.bindings.vision_merger_fc2.format == other,
            "evaluation global/MTP/Vision binding formats are inconsistent");
    for (const auto& layer : plan.bindings.text_layers) {
        require(layer.mlp.gate_up.format == source && layer.mlp.down.format == other,
                "evaluation MLP binding formats are inconsistent");
        if (layer.is_full_attention) {
            require(layer.attention.projection.query_key.format == source &&
                        layer.attention.projection.gate_value.format == other &&
                        layer.attention.output.format == other,
                    "evaluation full-attention binding formats are inconsistent");
        } else {
            require(layer.gdn.input_projection.query_key.format == source &&
                        layer.gdn.input_projection.value_z.format == other &&
                        layer.gdn.output.format == other,
                    "evaluation GDN binding formats are inconsistent");
        }
    }
    return plan;
}

ArtifactLoadPlan bind_dflash_q4_evaluation(const std::filesystem::path& path,
                                            std::string_view weights_id,
                                            WeightsProfile profile,
                                            std::size_t expected_q4,
                                            std::size_t expected_w8) {
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{
                                     "qwen3.8-27b", std::string(weights_id)},
            "sparse DFlash2 evaluation artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) == profile,
            "package resolved a DFlash2 evaluation artifact to the wrong profile");

    std::size_t q4 = 0;
    std::size_t w8 = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        q4 += tensor->format == NumericFormat::Q4G64_F16S ? 1 : 0;
        w8 += tensor->format == NumericFormat::W8G32_F16S ? 1 : 0;
    }
    require(reader.objects().size() == 1190 && q4 == expected_q4 && w8 == expected_w8,
            "DFlash2 evaluation inventory/format counts differ from the exact contract");

    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::DFlash,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, profile, features);
    require(plan.materialization.object_count == 1190,
            "DFlash2 evaluation materialization object count differs");
    if (plan.materialization.device_objects.size() != 1172) {
        throw std::runtime_error(
            "DFlash2 evaluation device-object count differs: got " +
            std::to_string(plan.materialization.device_objects.size()));
    }
    require(plan.materialization.host_objects.size() == 6,
            "DFlash2 evaluation host-object count differs");
    require(plan.bindings.dflash.has_value(),
            "DFlash2 evaluation bind plan is missing");

    const NumericFormat q4_format = NumericFormat::Q4G64_F16S;
    const NumericFormat base_other =
        profile == WeightsProfile::R9700Q4G64DFlash2Q4Evaluation
            ? q4_format
            : NumericFormat::W8G32_F16S;
    require(plan.bindings.token_embedding.format == base_other &&
                plan.bindings.output_head.format == base_other &&
                plan.bindings.draft_head.format == q4_format,
            "DFlash2 evaluation base formats are inconsistent");
    const auto& dflash = *plan.bindings.dflash;
    require(dflash.feature_projection.format == q4_format &&
                dflash.hidden_projection.format == q4_format,
            "DFlash2 global matrices are not directly bound Q4G64");
    for (const auto& layer : dflash.layers) {
        require(layer.query_key_value.format == q4_format &&
                    layer.attention_output.format == q4_format &&
                    layer.attention_conv.kernel_projection.format == q4_format &&
                    layer.gate_up.format == q4_format && layer.down.format == q4_format &&
                    layer.mlp_conv.kernel_projection.format == q4_format,
                "DFlash2 layer matrix binding is not uniformly Q4G64");
    }
    constexpr std::int32_t kTokens = 2;
    constexpr std::int32_t kBatch = 2;
    constexpr std::size_t kProjectionBytes =
        static_cast<std::size_t>(ninfer::ops::kGroupedDynamicConvProjRows) * kTokens * kBatch *
        sizeof(std::uint16_t);
    require(ninfer::ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(
                QType::Q4G64_F16S, kTokens, kTokens, kBatch) > kProjectionBytes,
            "DFlash2 grouped-convolution workspace omitted Q4 Linear scratch");
    require(ninfer::ops::dflash2_path_select_workspace_capacity_bytes(
                QType::Q4G64_F16S, kTokens, kTokens, kBatch) >
                ninfer::ops::dflash2_path_select_workspace_capacity_bytes(
                    QType::BF16_CTRL, kTokens, kTokens, kBatch),
            "DFlash2 selector workspace omitted Q4 Linear scratch");
    return plan;
}

void require_missing_dflash_rejection(const std::filesystem::path& path,
                                      WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = false,
        .speculative = ninfer::SpeculativeBackend::DFlash,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    try {
        (void)ninfer::targets::qwen3_8_27b::detail::bind_artifact(binder, profile, features);
    } catch (const ninfer::artifact::ArtifactError&) {
        return;
    }
    throw std::runtime_error("base evaluation identity accepted a missing DFlash2 inventory");
}

ArtifactLoadPlan bind_w8_bf16_embedding(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{
                                     "qwen3.8-27b", "r9700-w8-bf16-embed-eval"},
            "sparse W8/BF16-embedding artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) ==
                WeightsProfile::R9700W8Bf16EmbeddingEvaluation,
            "package resolved the W8/BF16-embedding artifact to the wrong profile");

    std::size_t bf16 = 0;
    std::size_t fp32 = 0;
    std::size_t i32  = 0;
    std::size_t w8   = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        switch (tensor->format) {
        case NumericFormat::BF16:
            ++bf16;
            break;
        case NumericFormat::FP32:
            ++fp32;
            break;
        case NumericFormat::I32:
            ++i32;
            break;
        case NumericFormat::W8G32_F16S:
            ++w8;
            break;
        default:
            throw std::runtime_error("W8/BF16-embedding artifact contains an unexpected format");
        }
    }
    require(reader.objects().size() == 1124 && bf16 == 583 && fp32 == 96 && i32 == 1 &&
                w8 == 438,
            "W8/BF16-embedding inventory/format counts differ from the exact contract");

    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, WeightsProfile::R9700W8Bf16EmbeddingEvaluation, features);
    require(plan.materialization.object_count == 1124 &&
                plan.materialization.device_objects.size() == 1118 &&
                plan.materialization.host_objects.size() == 6,
            "W8/BF16-embedding materialization plan is incomplete");
    require_w8_workspace_profile(WeightsProfile::R9700W8Bf16EmbeddingEvaluation);
    require(plan.bindings.token_embedding.format == NumericFormat::BF16 &&
                plan.bindings.output_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.draft_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.input_projection.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.mlp.down.format == NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.patch_embedding.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].qkv.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc2.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_input.fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_fc2.format == NumericFormat::W8G32_F16S,
            "W8/BF16-embedding global/MTP/Vision binding formats are inconsistent");
    for (const auto& layer : plan.bindings.text_layers) {
        require(layer.mlp.gate_up.format == NumericFormat::W8G32_F16S &&
                    layer.mlp.down.format == NumericFormat::W8G32_F16S,
                "W8/BF16-embedding MLP binding formats are inconsistent");
        if (layer.is_full_attention) {
            require(layer.attention.projection.query_key.format == NumericFormat::W8G32_F16S &&
                        layer.attention.projection.gate_value.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.attention.output.format == NumericFormat::W8G32_F16S,
                    "W8/BF16-embedding full-attention binding formats are inconsistent");
        } else {
            require(layer.gdn.input_projection.query_key.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.gdn.input_projection.value_z.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.gdn.output.format == NumericFormat::W8G32_F16S,
                    "W8/BF16-embedding GDN binding formats are inconsistent");
        }
    }
    return plan;
}

ArtifactLoadPlan bind_w8_bf16_attention_vo(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{
                                     "qwen3.8-27b", "r9700-w8-bf16-attn-vo-eval"},
            "sparse W8/BF16-attention artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) ==
                WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation,
            "package resolved the W8/BF16-attention artifact to the wrong profile");

    std::size_t bf16 = 0;
    std::size_t fp32 = 0;
    std::size_t i32  = 0;
    std::size_t w8   = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        switch (tensor->format) {
        case NumericFormat::BF16:
            ++bf16;
            break;
        case NumericFormat::FP32:
            ++fp32;
            break;
        case NumericFormat::I32:
            ++i32;
            break;
        case NumericFormat::W8G32_F16S:
            ++w8;
            break;
        default:
            throw std::runtime_error("W8/BF16-attention artifact contains an unexpected format");
        }
    }
    require(reader.objects().size() == 1124 && bf16 == 614 && fp32 == 96 && i32 == 1 &&
                w8 == 407,
            "W8/BF16-attention inventory/format counts differ from the exact contract");

    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation, features);
    require(plan.materialization.object_count == 1124 &&
                plan.materialization.device_objects.size() == 1118 &&
                plan.materialization.host_objects.size() == 6,
            "W8/BF16-attention materialization plan is incomplete");
    require_w8_workspace_profile(WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation);
    require(plan.bindings.token_embedding.format == NumericFormat::W8G32_F16S &&
                plan.bindings.output_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.draft_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.input_projection.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.mlp.down.format == NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.patch_embedding.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].qkv.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc2.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_input.fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_fc2.format == NumericFormat::W8G32_F16S,
            "W8/BF16-attention global/MTP/Vision bindings are inconsistent");
    for (const auto& layer : plan.bindings.text_layers) {
        require(layer.mlp.gate_up.format == NumericFormat::W8G32_F16S &&
                    layer.mlp.down.format == NumericFormat::W8G32_F16S,
                "W8/BF16-attention MLP bindings are inconsistent");
        if (layer.is_full_attention) {
            require(layer.attention.projection.query_key.format == NumericFormat::W8G32_F16S &&
                        layer.attention.projection.gate_value.format == NumericFormat::BF16 &&
                        layer.attention.output.format == NumericFormat::BF16,
                    "W8/BF16-attention full-attention bindings are inconsistent");
        } else {
            require(layer.gdn.input_projection.query_key.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.gdn.input_projection.value_z.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.gdn.output.format == NumericFormat::W8G32_F16S,
                    "W8/BF16-attention GDN bindings are inconsistent");
        }
    }
    return plan;
}

ArtifactLoadPlan bind_w8_bf16_attention_qk(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{
                                     "qwen3.8-27b", "r9700-w8-bf16-attn-qk-eval"},
            "sparse W8/BF16-attention-QK artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) ==
                WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation,
            "package resolved the W8/BF16-attention-QK artifact to the wrong profile");

    std::size_t bf16 = 0;
    std::size_t fp32 = 0;
    std::size_t i32  = 0;
    std::size_t w8   = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        switch (tensor->format) {
        case NumericFormat::BF16:
            ++bf16;
            break;
        case NumericFormat::FP32:
            ++fp32;
            break;
        case NumericFormat::I32:
            ++i32;
            break;
        case NumericFormat::W8G32_F16S:
            ++w8;
            break;
        default:
            throw std::runtime_error(
                "W8/BF16-attention-QK artifact contains an unexpected format");
        }
    }
    require(reader.objects().size() == 1124 && bf16 == 598 && fp32 == 96 && i32 == 1 &&
                w8 == 423,
            "W8/BF16-attention-QK inventory/format counts differ from the exact contract");

    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation, features);
    require(plan.materialization.object_count == 1124 &&
                plan.materialization.device_objects.size() == 1118 &&
                plan.materialization.host_objects.size() == 6,
            "W8/BF16-attention-QK materialization plan is incomplete");
    require_w8_workspace_profile(WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation);
    require(plan.bindings.token_embedding.format == NumericFormat::W8G32_F16S &&
                plan.bindings.output_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.draft_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.input_projection.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.mlp.down.format == NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.patch_embedding.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].qkv.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc2.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_input.fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_fc2.format == NumericFormat::W8G32_F16S,
            "W8/BF16-attention-QK global/MTP/Vision bindings are inconsistent");
    for (const auto& layer : plan.bindings.text_layers) {
        require(layer.mlp.gate_up.format == NumericFormat::W8G32_F16S &&
                    layer.mlp.down.format == NumericFormat::W8G32_F16S,
                "W8/BF16-attention-QK MLP bindings are inconsistent");
        if (layer.is_full_attention) {
            require(layer.attention.projection.query_key.format == NumericFormat::BF16 &&
                        layer.attention.projection.gate_value.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.attention.output.format == NumericFormat::W8G32_F16S,
                    "W8/BF16-attention-QK full-attention bindings are inconsistent");
        } else {
            require(layer.gdn.input_projection.query_key.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.gdn.input_projection.value_z.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.gdn.output.format == NumericFormat::W8G32_F16S,
                    "W8/BF16-attention-QK GDN bindings are inconsistent");
        }
    }
    return plan;
}

ArtifactLoadPlan bind_w8_bf16_gdn_qk(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{
                                     "qwen3.8-27b", "r9700-w8-bf16-gdn-qk-eval"},
            "sparse W8/BF16-GDN-QK artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) ==
                WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation,
            "package resolved the W8/BF16-GDN-QK artifact to the wrong profile");

    std::size_t bf16 = 0;
    std::size_t fp32 = 0;
    std::size_t i32  = 0;
    std::size_t w8   = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        switch (tensor->format) {
        case NumericFormat::BF16:
            ++bf16;
            break;
        case NumericFormat::FP32:
            ++fp32;
            break;
        case NumericFormat::I32:
            ++i32;
            break;
        case NumericFormat::W8G32_F16S:
            ++w8;
            break;
        default:
            throw std::runtime_error("W8/BF16-GDN-QK artifact contains an unexpected format");
        }
    }
    require(reader.objects().size() == 1124 && bf16 == 630 && fp32 == 96 && i32 == 1 &&
                w8 == 391,
            "W8/BF16-GDN-QK inventory/format counts differ from the exact contract");

    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation, features);
    require(plan.materialization.object_count == 1124 &&
                plan.materialization.device_objects.size() == 1118 &&
                plan.materialization.host_objects.size() == 6,
            "W8/BF16-GDN-QK materialization plan is incomplete");
    require_w8_workspace_profile(WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation);
    require(plan.bindings.token_embedding.format == NumericFormat::W8G32_F16S &&
                plan.bindings.output_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.draft_head.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.input_projection.format == NumericFormat::W8G32_F16S &&
                plan.bindings.mtp.mlp.down.format == NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.patch_embedding.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].qkv.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_backbone.layers[0].fc2.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_input.fc1.format ==
                    NumericFormat::W8G32_F16S &&
                plan.bindings.vision_merger_fc2.format == NumericFormat::W8G32_F16S,
            "W8/BF16-GDN-QK global/MTP/Vision bindings are inconsistent");
    for (const auto& layer : plan.bindings.text_layers) {
        require(layer.mlp.gate_up.format == NumericFormat::W8G32_F16S &&
                    layer.mlp.down.format == NumericFormat::W8G32_F16S,
                "W8/BF16-GDN-QK MLP bindings are inconsistent");
        if (layer.is_full_attention) {
            require(layer.attention.projection.query_key.format == NumericFormat::W8G32_F16S &&
                        layer.attention.projection.gate_value.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.attention.output.format == NumericFormat::W8G32_F16S,
                    "W8/BF16-GDN-QK full-attention bindings are inconsistent");
        } else {
            require(layer.gdn.input_projection.query_key.format == NumericFormat::BF16 &&
                        layer.gdn.input_projection.value_z.format ==
                            NumericFormat::W8G32_F16S &&
                        layer.gdn.output.format == NumericFormat::W8G32_F16S,
                    "W8/BF16-GDN-QK GDN bindings are inconsistent");
        }
    }
    return plan;
}

void require_wrong_format_rejection(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    try {
        (void)ninfer::targets::qwen3_8_27b::detail::bind_artifact(
            binder, WeightsProfile::R9700W8G32Candidate, {});
    } catch (const ninfer::artifact::ArtifactError&) {
        return;
    }
    throw std::runtime_error("candidate binder accepted a non-W8 vocabulary matrix");
}

void require_fp8_hybrid_format_rejection(const std::filesystem::path& all_q4_path) {
    ninfer::artifact::Reader reader(all_q4_path);
    ninfer::artifact::Binder binder(reader);
    try {
        (void)ninfer::targets::qwen3_8_27b::detail::bind_artifact(
            binder, WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation, {});
    } catch (const ninfer::artifact::ArtifactError&) {
        return;
    }
    throw std::runtime_error("FP8/Q4 hybrid binder accepted a selected matrix through Q4");
}

void require_rejected_bf16_head_identity() {
    try {
        (void)Package::resolve_weights(
            ninfer::artifact::ArtifactIdentity{"qwen3.8-27b", "r9700-w8-bf16-head-eval"});
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("package retained the rejected BF16-output-head identity");
}

ArtifactLoadPlan bind_fp8_q4_hybrid(const std::filesystem::path& path) {
    namespace hybrid = ninfer::targets::qwen3_8_27b::detail::fp8_hybrid;
    ninfer::artifact::Reader reader(path);
    require(reader.identity() == ninfer::artifact::ArtifactIdentity{
                                     "qwen3.8-27b", std::string(hybrid::kWeightsId)},
            "sparse FP8/Q4 hybrid artifact has the wrong identity");
    require(Package::resolve_weights(reader.identity()) ==
                WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation,
            "package resolved the FP8/Q4 hybrid artifact to the wrong profile");

    std::size_t q4 = 0;
    std::size_t fp8 = 0;
    std::size_t selected = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        q4 += tensor->format == NumericFormat::Q4G64_F16S ? 1 : 0;
        fp8 += tensor->format == NumericFormat::F8E4M3_ROW_F32S ? 1 : 0;
        if (hybrid::contains(tensor->name)) {
            ++selected;
            require(tensor->format == NumericFormat::F8E4M3_ROW_F32S,
                    "selected hybrid matrix retained a Q4 plane");
        } else {
            require(tensor->format != NumericFormat::F8E4M3_ROW_F32S,
                    "hybrid artifact contains FP8 outside the decision-owned set");
        }
    }
    require(reader.objects().size() == 1124 && selected == hybrid::kMatrixNames.size() &&
                fp8 == 144 && q4 == 295,
            "FP8/Q4 hybrid inventory or single-plane counts differ");

    ninfer::artifact::Binder binder(reader);
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ArtifactLoadPlan plan = ninfer::targets::qwen3_8_27b::detail::bind_artifact(
        binder, WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation, features);
    require(plan.materialization.object_count == 1124 &&
                plan.materialization.device_objects.size() == 1118 &&
                plan.materialization.host_objects.size() == 6 &&
                plan.materialization.device_capacity_bytes == 21'540'531'712ULL,
            "FP8/Q4 hybrid materialization plan or aligned arena differs");
    require(plan.bindings.token_embedding.format == NumericFormat::Q4G64_F16S &&
                plan.bindings.output_head.format == NumericFormat::Q4G64_F16S &&
                plan.bindings.draft_head.format == NumericFormat::Q4G64_F16S &&
                plan.bindings.mtp.input_projection.format == NumericFormat::Q4G64_F16S &&
                plan.bindings.mtp.mlp.gate_up.format == NumericFormat::Q4G64_F16S &&
                plan.bindings.vision_backbone.patch_embedding.format ==
                    NumericFormat::Q4G64_F16S,
            "FP8/Q4 hybrid leaked FP8 into global, MTP, or Vision binding");
    for (const auto& layer : plan.bindings.text_layers) {
        require(layer.mlp.gate_up.format == NumericFormat::F8E4M3_ROW_F32S &&
                    layer.mlp.down.format == NumericFormat::Q4G64_F16S,
                "FP8/Q4 hybrid MLP role formats differ");
        if (layer.is_full_attention) {
            require(layer.attention.projection.query_key.format ==
                            NumericFormat::F8E4M3_ROW_F32S &&
                        layer.attention.projection.gate_value.format ==
                            NumericFormat::F8E4M3_ROW_F32S &&
                        layer.attention.output.format == NumericFormat::Q4G64_F16S,
                    "FP8/Q4 hybrid full-attention role formats differ");
        } else {
            require(layer.gdn.input_projection.query_key.format ==
                            NumericFormat::F8E4M3_ROW_F32S &&
                        layer.gdn.input_projection.value_z.format ==
                            NumericFormat::Q4G64_F16S &&
                        layer.gdn.output.format == NumericFormat::Q4G64_F16S,
                    "FP8/Q4 hybrid GDN role formats differ");
        }
    }
    return plan;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 14) {
            throw std::invalid_argument(
                "usage: target_binding_qual W8 INVALID_FORMAT Q4 Q4_W8 Q4_W8_MSE "
                "W8_BF16_EMBEDDING W8_BF16_ATTN_VO W8_BF16_GDN_QK W8_BF16_ATTN_QK W8_MSE "
                "Q4_DFLASH2_Q4 Q4_W8_MSE_DFLASH2_Q4 FP8_Q4_HYBRID");
        }
        const ArtifactLoadPlan plan = bind_complete(argv[1], "r9700-int-candidate");
        require_wrong_format_rejection(argv[2]);
        require_rejected_bf16_head_identity();
        const ArtifactLoadPlan q4 = bind_evaluation(
            argv[3], "r9700-q4g64-n16k16-eval", WeightsProfile::R9700Q4G64Evaluation, 439, 0);
        require_fp8_hybrid_format_rejection(argv[3]);
        const ArtifactLoadPlan q4_w8 = bind_evaluation(
            argv[4], "r9700-q4-w8-n16k16-eval", WeightsProfile::R9700Q4W8Evaluation, 183, 256);
        const ArtifactLoadPlan q4_w8_mse = bind_evaluation(
            argv[5], "r9700-q4-w8-mse-n16k16-eval", WeightsProfile::R9700Q4W8Evaluation, 183, 256);
        const ArtifactLoadPlan w8_bf16_embedding = bind_w8_bf16_embedding(argv[6]);
        const ArtifactLoadPlan w8_bf16_attention_vo = bind_w8_bf16_attention_vo(argv[7]);
        const ArtifactLoadPlan w8_bf16_gdn_qk = bind_w8_bf16_gdn_qk(argv[8]);
        const ArtifactLoadPlan w8_bf16_attention_qk = bind_w8_bf16_attention_qk(argv[9]);
        const ArtifactLoadPlan w8_mse = bind_complete(argv[10], "r9700-w8g32-mse-eval");
        require_missing_dflash_rejection(argv[3], WeightsProfile::R9700Q4G64Evaluation);
        const ArtifactLoadPlan q4_dflash = bind_dflash_q4_evaluation(
            argv[11], "r9700-q4g64-n16k16-dflash2-q4-eval",
            WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, 471, 0);
        const ArtifactLoadPlan mixed_dflash = bind_dflash_q4_evaluation(
            argv[12], "r9700-q4-w8-mse-n16k16-dflash2-q4-eval",
            WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation, 215, 256);
        const ArtifactLoadPlan fp8_q4_hybrid = bind_fp8_q4_hybrid(argv[13]);
        std::cout << "r9700_target_binding: PASS objects=" << plan.materialization.object_count
                  << " device=" << plan.materialization.device_objects.size()
                  << " host=" << plan.materialization.host_objects.size()
                  << " w8_matrices=439 q4_matrices=439 mixed_q4=183 mixed_w8=256"
                  << " mixed_mse_q4=183 mixed_mse_w8=256"
                  << " bf16_embedding_w8=438 bf16_embedding=1"
                  << " bf16_attention_w8=407 bf16_attention=32"
                  << " bf16_gdn_qk_w8=391 bf16_gdn_qk=48"
                  << " bf16_attention_qk_w8=423 bf16_attention_qk=16"
                  << " mse_w8_matrices=439"
                  << " dflash_all_q4=471 dflash_mixed_q4=215 dflash_mixed_w8=256"
                  << " hybrid_fp8=144 hybrid_q4=295"
                  << " missing_dflash=REJECT hybrid_q4_fallback=REJECT wrong_format=REJECT\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
