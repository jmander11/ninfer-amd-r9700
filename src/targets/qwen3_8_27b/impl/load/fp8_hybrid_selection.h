#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace ninfer::targets::qwen3_8_27b::detail::fp8_hybrid {

#define NINFER_QWEN38_FP8_HYBRID_IDENTITY(weights_id, recipe_id, digest)                  \
    inline constexpr std::string_view kWeightsId       = weights_id;                      \
    inline constexpr std::string_view kRecipeId        = recipe_id;                       \
    inline constexpr std::string_view kSelectionDigest = digest;
#define NINFER_QWEN38_FP8_HYBRID_MATRIX(name)
#include "targets/qwen3_8_27b/impl/load/fp8_hybrid_selection.inc"
#undef NINFER_QWEN38_FP8_HYBRID_MATRIX
#undef NINFER_QWEN38_FP8_HYBRID_IDENTITY

inline constexpr std::array<std::string_view, 144> kMatrixNames = {
#define NINFER_QWEN38_FP8_HYBRID_IDENTITY(weights_id, recipe_id, digest)
#define NINFER_QWEN38_FP8_HYBRID_MATRIX(name) name,
#include "targets/qwen3_8_27b/impl/load/fp8_hybrid_selection.inc"
#undef NINFER_QWEN38_FP8_HYBRID_MATRIX
#undef NINFER_QWEN38_FP8_HYBRID_IDENTITY
};

consteval bool unique_matrix_names() {
    for (std::size_t left = 0; left < kMatrixNames.size(); ++left) {
        if (kMatrixNames[left].empty()) { return false; }
        for (std::size_t right = left + 1; right < kMatrixNames.size(); ++right) {
            if (kMatrixNames[left] == kMatrixNames[right]) { return false; }
        }
    }
    return true;
}

static_assert(unique_matrix_names());

constexpr bool contains(std::string_view name) {
    for (const std::string_view selected : kMatrixNames) {
        if (selected == name) { return true; }
    }
    return false;
}

} // namespace ninfer::targets::qwen3_8_27b::detail::fp8_hybrid
