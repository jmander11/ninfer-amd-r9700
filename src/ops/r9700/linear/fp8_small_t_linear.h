#pragma once

#include "ops/r9700/linear/fp8_activation.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace ninfer::ops::r9700::linear {

// Row-scaled E4M3 weight rows [rows, columns] (unpadded columns) with FP32 row scales.
struct Fp8RowScaledWeight {
    const std::uint8_t* codes = nullptr;
    const float* scales       = nullptr;
    std::uint32_t rows        = 0;
};

// Small-T (T <= 16) row-scaled E4M3 projection y[t, r] = BF16(sum_k w[r, k] a[t, k] * ws[r] * as[t])
// of one prepared activation image. A nonzero activation status replaces every published element
// with the canonical BF16 quiet NaN.
inline constexpr std::uint32_t kFp8SmallTokenLimit = 16U;
[[nodiscard]] bool fp8_small_t_supported(std::uint32_t tokens, std::uint32_t rows,
                                         std::uint32_t columns,
                                         std::uint32_t padded_columns) noexcept;

// output [T, rows].
[[nodiscard]] hipError_t fp8_small_t_linear(const Fp8RowScaledWeight& weight,
                                            const Fp8ActivationWorkspace& activation,
                                            hip_bfloat16* output, hipStream_t stream) noexcept;

// Two projections of one activation in one launch. Each projection's rows [0, split) publish to
// its `leading` [T, split] tensor and rows [split, rows) to its `trailing` [T, rows - split]
// tensor, exactly the values fp8_small_t_linear followed by a feature split would store.
struct Fp8SmallTPairSplitArgs {
    Fp8RowScaledWeight first{};
    Fp8RowScaledWeight second{};
    std::uint32_t split = 0;
    hip_bfloat16* first_leading   = nullptr;
    hip_bfloat16* first_trailing  = nullptr;
    hip_bfloat16* second_leading  = nullptr;
    hip_bfloat16* second_trailing = nullptr;
};
[[nodiscard]] hipError_t fp8_small_t_pair_split(const Fp8SmallTPairSplitArgs& args,
                                                const Fp8ActivationWorkspace& activation,
                                                hipStream_t stream) noexcept;

// residual [T, rows] <- BF16(residual + BF16(y)): the rounded projection added once, exactly as a
// separate BF16 residual add of fp8_small_t_linear's output.
[[nodiscard]] hipError_t fp8_small_t_residual(const Fp8RowScaledWeight& weight,
                                              const Fp8ActivationWorkspace& activation,
                                              hip_bfloat16* residual,
                                              hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::linear
