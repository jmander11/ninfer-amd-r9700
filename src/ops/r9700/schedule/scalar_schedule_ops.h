#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::schedule {

// Raw gfx1201 launch layer beneath the semantic Tensor contracts. Pointer arguments are device
// addresses. The caller owns all represented-domain and non-aliasing validation.
[[nodiscard]] hipError_t add_bias_bf16(const hip_bfloat16* bias, hip_bfloat16* values,
                                       std::uint32_t features, std::size_t rows,
                                       hipStream_t stream) noexcept;

[[nodiscard]] hipError_t nll_from_logits_bf16(const hip_bfloat16* logits,
                                              const std::int32_t* targets, float* output,
                                              std::uint32_t physical_rows,
                                              std::uint32_t valid_rows, std::uint32_t columns,
                                              hipStream_t stream) noexcept;

[[nodiscard]] hipError_t prepare_masked_block_i32(const std::int32_t* anchors,
                                                  const std::int32_t* lengths,
                                                  const std::int32_t* valid_columns,
                                                  std::int32_t mask_id, std::int32_t* ids,
                                                  std::int32_t* positions, std::uint32_t width,
                                                  std::uint32_t batch,
                                                  hipStream_t stream) noexcept;

[[nodiscard]] hipError_t prepare_ragged_prefix_bf16(
    const void* source, const std::int32_t* lanes, const std::int32_t* starts,
    const std::int32_t* ends, void* destination, std::int32_t* positions,
    std::int32_t* counts, std::uint32_t vectors_per_column, std::uint32_t width,
    std::uint32_t batch, std::size_t source_column_stride,
    std::size_t source_lane_stride, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::schedule
