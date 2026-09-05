#include <ninfer/targets/qwen3_8_27b/package.h>

#include "artifact/reader.h"
#include "targets/qwen3_8_27b/impl/load/fp8_hybrid_selection.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8_27b {
namespace {

constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 0.6F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_8Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "r9700-int-candidate") {
        return WeightsProfile::R9700W8G32Candidate;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-w8g32-mse-eval") {
        return WeightsProfile::R9700W8G32Candidate;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-w8-bf16-embed-eval") {
        return WeightsProfile::R9700W8Bf16EmbeddingEvaluation;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-w8-bf16-attn-qk-eval") {
        return WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-w8-bf16-attn-vo-eval") {
        return WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-w8-bf16-gdn-qk-eval") {
        return WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-q4g64-n16k16-eval") {
        return WeightsProfile::R9700Q4G64Evaluation;
    }
    if (identity.model_id == model_id && identity.weights_id == detail::fp8_hybrid::kWeightsId) {
        return WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-q4-w8-n16k16-eval") {
        return WeightsProfile::R9700Q4W8Evaluation;
    }
    if (identity.model_id == model_id && identity.weights_id == "r9700-q4-w8-mse-n16k16-eval") {
        return WeightsProfile::R9700Q4W8Evaluation;
    }
    if (identity.model_id == model_id &&
        identity.weights_id == "r9700-q4g64-n16k16-dflash2-q4-eval") {
        return WeightsProfile::R9700Q4G64DFlash2Q4Evaluation;
    }
    if (identity.model_id == model_id &&
        identity.weights_id == "r9700-q4-w8-mse-n16k16-dflash2-q4-eval") {
        return WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation;
    }
    if (identity.model_id == model_id &&
        identity.weights_id == "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval") {
        return WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

} // namespace ninfer::targets::qwen3_8_27b
