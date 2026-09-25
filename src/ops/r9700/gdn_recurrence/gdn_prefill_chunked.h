#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {

// Private chunked (WY) implementation of the normalized ordinary recurrence for widths
// >= kChunkedMinimumWidth. One workgroup per value head walks 64-token chunks: per chunk it forms
// A = strictly lower beta_i exp(b_i - b_j) k_i.k_j, the diagonal blocks of (I + A)^-1 in FP32,
// Z = beta (V - Gamma K S^T), the chunk's corrected values V' by block forward substitution,
// O = scale (Gamma q) S^T + (causal decayed q.k) V', and S <- Gamma_C S + (K exp(b_C - b))^T V'.
// Products run as FP16 WMMA with FP32 accumulation; the state stays FP32 in the accumulators.
// A partial final chunk is masked (zero q/k/v/beta/g past the width). No workspace is used, and
// state_in may alias state_out.
inline constexpr std::int32_t kChunkedMinimumWidth = 64;
[[nodiscard]] hipError_t prefill_chunked(
    const hip_bfloat16* query, const hip_bfloat16* key, const hip_bfloat16* value,
    const float* g, const float* beta, const float* state_in, float* state_out,
    hip_bfloat16* output, std::int32_t width, float scale, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn_recurrence
