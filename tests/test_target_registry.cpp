#include "targets/registry.h"

#include "artifact/reader.h"

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
    std::cout << "target_registry: PASS model=" << Package::model_id
              << " target=" << *active << '\n';
    return 0;
}
