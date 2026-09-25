#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {

// Private staged implementation of the normalized ordinary recurrence for widths
// kStagedMinimumWidth..kStagedMaximumWidth. The workspace holds the normalized FP32 q/k of every
// (token, Q/K head) in recurrence lane order, followed by exp(g) FP32 [T,48].
inline constexpr std::int32_t kStagedMinimumWidth = 64;
inline constexpr std::int32_t kStagedMaximumWidth = 8192;
[[nodiscard]] std::size_t prefill_staged_workspace_bytes(std::int32_t width) noexcept;
[[nodiscard]] hipError_t prefill_staged(
    const hip_bfloat16* query, const hip_bfloat16* key, const hip_bfloat16* value,
    const float* g, const float* beta, const float* state_in, float* state_out,
    hip_bfloat16* output, void* workspace, std::size_t workspace_bytes, std::int32_t width,
    float scale, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn_recurrence
