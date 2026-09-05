#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {

// Qualification-only three-stage affine chunk decomposition for the fixed Qwen3.8 ordinary
// recurrence. Production gated_delta_net does not reference this object. Workspace holds dense
// FP32 A/B transforms; stage two replaces each B slot with that chunk's input state boundary.
[[nodiscard]] std::size_t affine_chunk_workspace_capacity_bytes(std::uint32_t tokens,
                                                                 std::uint32_t chunk) noexcept;

[[nodiscard]] hipError_t affine_chunk_qualification(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    const float* g, const float* beta, const float* state_in, float* state_out,
    hip_bfloat16* output, void* workspace, std::size_t workspace_bytes,
    std::uint32_t tokens, std::uint32_t chunk, float scale,
    hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn_recurrence
