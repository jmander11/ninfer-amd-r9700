#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {

// Private fixed-P2048 implementation boundary for the normalized ordinary recurrence. The
// sidecar contains query inverse norms followed by key inverse norms, each FP32 [T,16].
[[nodiscard]] std::size_t prefill_scale_sidecar_workspace_bytes(std::int32_t width) noexcept;

[[nodiscard]] hipError_t prefill_scale_sidecar_prepare(
    const hip_bfloat16* query, const hip_bfloat16* key, float* sidecar,
    std::int32_t width, hipStream_t stream) noexcept;

[[nodiscard]] hipError_t prefill_scale_sidecar_recurrence(
    const hip_bfloat16* query, const hip_bfloat16* key, const hip_bfloat16* value,
    const float* g, const float* beta, const float* state_in, float* state_out,
    hip_bfloat16* output, const float* sidecar, std::int32_t width, float scale,
    hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn_recurrence
