#pragma once

#include "core/fp8_int4_paged_kv_cache.h"

#include <cstdint>

namespace ninfer::targets::qwen3::detail {

// The current whole-attention timing winner used by the sole R9700 schedule. The semantic
// profile lives in one header so the schedule, physical qualifier, converter gate, and eventual
// PPL selection cannot silently disagree. CMake builds the pending real-model G16 and G32 scorers
// separately; no runtime or artifact selector reaches this constant. The losing build-time branch
// is removed after the gate.
#ifndef NINFER_R9700_KV_VALUE_GROUP
#define NINFER_R9700_KV_VALUE_GROUP 16
#endif
static_assert(NINFER_R9700_KV_VALUE_GROUP == 16 || NINFER_R9700_KV_VALUE_GROUP == 32,
              "R9700 KV value group must be G16 or G32");
inline constexpr std::int32_t kR9700TextKVValueGroup = NINFER_R9700_KV_VALUE_GROUP;
inline constexpr Fp8KInt4VPlaneLayouts kR9700TextKVPlaneLayouts{
    .key         = Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
    .value       = Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
    .value_scale = Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
};
inline constexpr char kR9700TextKVKeyPlaneLayoutName[] = "token-fastest-head-major";
inline constexpr char kR9700TextKVValuePlaneLayoutName[] = "feature-fastest-page-major";
inline constexpr char kR9700TextKVValueScalePlaneLayoutName[] =
    "feature-fastest-page-major";

} // namespace ninfer::targets::qwen3::detail
