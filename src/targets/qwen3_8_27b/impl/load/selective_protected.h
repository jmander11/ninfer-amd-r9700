#pragma once

#include "artifact/reader.h"

#include <array>
#include <string_view>

namespace ninfer::targets::qwen3_8_27b::detail::selective_protected {

inline constexpr std::array<std::string_view, 15> bf16_names{
    "text/layers/3/attention/query_key", "text/layers/3/attention/gate_value",
    "text/layers/7/attention/query_key", "text/layers/7/attention/gate_value",
    "text/layers/11/attention/query_key", "text/layers/11/attention/gate_value",
    "text/layers/15/attention/query_key", "text/layers/15/attention/gate_value",
    "text/layers/19/attention/query_key", "text/layers/19/attention/gate_value",
    "text/layers/23/attention/query_key", "text/layers/23/attention/gate_value",
    "text/layers/3/attention/output", "text/layers/7/attention/output",
    "text/layers/4/gdn/output"};
inline constexpr std::array<std::string_view, 11> fp8_names{
    "text/layers/11/attention/output",
    "text/layers/27/attention/query_key", "text/layers/27/attention/gate_value",
    "text/layers/31/attention/query_key", "text/layers/31/attention/gate_value",
    "text/layers/51/attention/query_key", "text/layers/51/attention/gate_value",
    "text/layers/62/mlp/gate_up", "text/layers/62/mlp/down",
    "text/layers/63/mlp/gate_up", "text/layers/63/mlp/down"};

constexpr artifact::NumericFormat matrix_format(std::string_view name) {
    using artifact::NumericFormat;
    if (name == "text/token_embedding" || name == "text/output_head")
        return NumericFormat::W8G32_F16S;
    for (const auto selected : bf16_names)
        if (name == selected) return NumericFormat::BF16;
    for (const auto selected : fp8_names)
        if (name == selected) return NumericFormat::F8E4M3_ROW_F32S;
    return NumericFormat::Q4G64_F16S;
}

} // namespace ninfer::targets::qwen3_8_27b::detail::selective_protected
