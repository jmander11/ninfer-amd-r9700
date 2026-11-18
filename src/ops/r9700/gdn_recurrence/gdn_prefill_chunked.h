#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {

// Private chunked (WY) implementation of the normalized ordinary recurrence for widths
// >= kChunkedMinimumWidth (one masked chunk already beats the sequential kernel at T16). Per
// 64-token chunk and value head it forms A = strictly lower beta_i exp(b_i - b_j) k_i.k_j, the
// diagonal blocks of (I + A)^-1 in FP32, Z = beta (V - Gamma K S^T), the chunk's corrected values
// V' by block forward substitution, O = scale (Gamma q) S^T + (causal decayed q.k) V', and
// S <- Gamma_C S + (K exp(b_C - b))^T V'. Products run as FP16 WMMA with FP32 accumulation; the
// state stays FP32 in the accumulators. A partial final chunk is masked (zero q/k/v/beta/g past
// the width). Above one chunk, a chunk-parallel preparation computes everything that does not
// depend on the state (normalized keys, query scales, and per value head the FP16 A/P tiles,
// diagonal inverses and decay scalars) into explicit caller-owned workspace, so the sequential
// per-head recurrence stages one chunk with a single load round. Both routes produce identical
// bits. State may alias state_out.
inline constexpr std::int32_t kChunkedMinimumWidth  = 16;
inline constexpr std::int32_t kPreparedMinimumWidth = 65;
// The prepared route addresses value and output rows with 32-bit element offsets.
inline constexpr std::int32_t kChunkedMaximumWidth = 0x7fffffff / (48 * 128);
// Per chunk and value head: 20 FP16 16x16 tiles and four FP32 per-token scalars.
inline constexpr std::size_t kPreparedRecordBytes = 20 * 16 * 16 * 2 + 4 * 64 * 4;

[[nodiscard]] constexpr std::size_t prefill_workspace_bytes(std::int32_t width) noexcept {
    if (width < kPreparedMinimumWidth) return 0;
    const auto chunks = static_cast<std::size_t>((width + 63) / 64);
    return chunks * 16 * (64 * 128 * sizeof(_Float16) + 64 * sizeof(float)) +
           chunks * 48 * kPreparedRecordBytes;
}

[[nodiscard]] hipError_t prefill_chunked(const hip_bfloat16* query, const hip_bfloat16* key,
                                         const hip_bfloat16* value, const float* g,
                                         const float* beta, const float* state_in, float* state_out,
                                         hip_bfloat16* output, std::int32_t width, float scale,
                                         void* workspace, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn_recurrence
