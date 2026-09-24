#include "targets/registry.h"

#include "artifact/reader.h"
#include "targets/qwen3_8_27b/impl/load/selective_protected.h"

#include <iostream>
#include <string>

int main() {
    using Package = ninfer::targets::qwen3_8_27b::Package;
    using ninfer::targets::registered_target_key;

    const auto active = registered_target_key("qwen3.8-27b");
    if (!active.has_value() || *active != "qwen3_8_27b_r9700" ||
        *active != Package::target_key) {
        std::cerr << "registry did not select the sole Qwen3.8-27B R9700 target\n";
        return 1;
    }
    if (registered_target_key("unsupported-model").has_value() ||
        registered_target_key("").has_value()) {
        std::cerr << "registry accepted a non-product model identity\n";
        return 1;
    }
    const auto all_q4_dflash = Package::resolve_weights(
        {std::string(Package::model_id), "r9700-q4g64-n16k16-dflash2-q4-eval"});
    const auto mixed_dflash = Package::resolve_weights(
        {std::string(Package::model_id), "r9700-q4-w8-mse-n16k16-dflash2-q4-eval"});
    const auto fp8_q4_hybrid = Package::resolve_weights(
        {std::string(Package::model_id), "r9700-q4g64-f8e4m3-four-role-n16k16-eval"});
    const auto fp8_q4_hybrid_dflash = Package::resolve_weights(
        {std::string(Package::model_id),
         "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval"});
    if (all_q4_dflash != Package::WeightsProfile::R9700Q4G64DFlash2Q4Evaluation ||
        mixed_dflash != Package::WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation ||
        fp8_q4_hybrid !=
            Package::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation ||
        fp8_q4_hybrid_dflash !=
            Package::WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation) {
        std::cerr << "registry did not preserve explicit DFlash2 evaluation identities\n";
        return 1;
    }
    using Profile = Package::WeightsProfile;
    if (Package::resolve_weights({std::string(Package::model_id),
            "r9700-q4-selective-protected-n16k16-eval"}) !=
            Profile::R9700Q4SelectiveProtectedN16K16Evaluation)
        throw std::runtime_error("selective protected identity not registered");
    using ninfer::artifact::NumericFormat;
    namespace selective = ninfer::targets::qwen3_8_27b::detail::selective_protected;
    std::size_t bf16_count = 0, fp8_count = 0;
    for (int layer = 0; layer < 64; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        const bool full = layer % 4 == 3;
        const auto qk = selective::matrix_format(prefix + (full ? "attention/query_key" : "gdn/query_key"));
        const auto gv = selective::matrix_format(prefix + (full ? "attention/gate_value" : "gdn/value_z"));
        const auto output = selective::matrix_format(prefix + (full ? "attention/output" : "gdn/output"));
        const auto gate_up = selective::matrix_format(prefix + "mlp/gate_up");
        const auto down = selective::matrix_format(prefix + "mlp/down");
        for (const auto format : {qk, gv, output, gate_up, down}) {
            bf16_count += format == NumericFormat::BF16;
            fp8_count += format == NumericFormat::F8E4M3_ROW_F32S;
        }
        const auto qk_expected = full && layer <= 23 ? NumericFormat::BF16 :
            (layer == 27 || layer == 31 || layer == 51 ? NumericFormat::F8E4M3_ROW_F32S : NumericFormat::Q4G64_F16S);
        if (qk != qk_expected || gv != qk_expected ||
            gate_up != (layer >= 62 ? NumericFormat::F8E4M3_ROW_F32S : NumericFormat::Q4G64_F16S) ||
            down != gate_up)
            throw std::runtime_error("protected per-layer recipe differs");
    }
    if (bf16_count != 15 || fp8_count != 11 ||
        selective::matrix_format("text/token_embedding") != NumericFormat::W8G32_F16S ||
        selective::matrix_format("text/output_head") != NumericFormat::W8G32_F16S ||
        selective::matrix_format("text/draft_head") != NumericFormat::Q4G64_F16S ||
        selective::matrix_format("mtp/layer/mlp/down") != NumericFormat::Q4G64_F16S ||
        selective::matrix_format("vision/merger/fc2") != NumericFormat::Q4G64_F16S)
        throw std::runtime_error("protected inventory changes unselected roles");
    if (Package::resolve_weights({std::string(Package::model_id),
            "r9700-q4-selective-protected-n16k16-dflash2-q4-eval"}) !=
            Profile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation)
        throw std::runtime_error("protected canonical Q4 companion identity not registered");
    const std::pair<const char*, Profile> mse_companions[] = {
        {"r9700-q4g64-n16k16-dflash2-q4-mse-eval", Profile::R9700Q4G64DFlash2Q4MseEvaluation},
        {"r9700-q4-w8-mse-n16k16-dflash2-q4-mse-eval", Profile::R9700Q4W8MseDFlash2Q4MseEvaluation},
        {"r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-mse-eval",
         Profile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation},
        {"r9700-q4g64-n16k16-dflash2-w8-mse-eval", Profile::R9700Q4G64DFlash2W8MseEvaluation},
        {"r9700-q4-w8-mse-n16k16-dflash2-w8-mse-eval", Profile::R9700Q4W8MseDFlash2W8MseEvaluation},
        {"r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-w8-mse-eval",
         Profile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation},
    };
    for (const auto& [identity, expected] : mse_companions) {
        if (Package::resolve_weights({std::string(Package::model_id), identity}) != expected) {
            std::cerr << "registry did not bind the source-MSE companion identity\n";
            return 1;
        }
    }
    std::cout << "target_registry: PASS model=" << Package::model_id
              << " target=" << *active << '\n';
    return 0;
}
