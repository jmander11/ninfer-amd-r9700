#pragma once

#include <ninfer/targets/qwen3_8_27b/package.h>
#include "artifact/reader.h"
#include <string_view>

namespace ninfer::targets::qwen3_8_27b::detail::fp8_capped {

struct Matrix { WeightsProfile profile; std::string_view name; };
inline constexpr Matrix matrices[] = {
#define NINFER_QWEN38_FP8_CAP_RECIPE(symbol, id)
#define NINFER_QWEN38_FP8_CAP_MATRIX(symbol, name) {WeightsProfile::symbol, name},
#include "targets/qwen3_8_27b/impl/load/fp8_capped_selection.inc"
#undef NINFER_QWEN38_FP8_CAP_MATRIX
#undef NINFER_QWEN38_FP8_CAP_RECIPE
};

constexpr artifact::NumericFormat matrix_format(WeightsProfile profile, std::string_view name) {
    profile = fp8_capped_base_profile(profile);
    if (profile == WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation)
        profile = WeightsProfile::R9700Q4Fp8SelectiveCapEvaluation;
    for (const auto& matrix : matrices)
        if (matrix.profile == profile && matrix.name == name)
            return artifact::NumericFormat::F8E4M3_ROW_F32S;
    return artifact::NumericFormat::Q4G64_F16S;
}

} // namespace ninfer::targets::qwen3_8_27b::detail::fp8_capped
