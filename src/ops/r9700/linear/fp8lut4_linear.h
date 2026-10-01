#pragma once

#include "ops/r9700/linear/fp8_activation.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdint>

namespace ninfer::ops::r9700::linear {

// FP8LUT4 weight [rows, columns] in r9700-fp8lut4-n16k64-v1 (see artifact/reader.h): four-bit
// sign-magnitude codes (bit 3 = sign, bits 0..2 = magnitude index) and one group code byte per
// 32 columns in N16 x K64 tiles, and one FP32 row multiplier. Group code b selects E = (b >> 3) - 26, m = b & 7 and
// the E4M3 magnitudes RNE(n_j * (8 + m) * 2^(E - 7)), n = {0, 13, 27, 41, 56, 74, 94, 120}.
// Null `groups` instead names row-scaled E4M3 rows (row-major raw codes, as Fp8RowScaledWeight),
// accepted where fp8_row_scaled_projection_supported holds, without `silu_pair`.
struct Fp8Lut4Weight {
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* groups = nullptr;
    const float* scales        = nullptr;
    std::uint32_t rows         = 0;
    std::uint32_t columns      = 0;  // padded K (K128)
};

// E4M3FN magnitude words [256 group codes][8], identical to the registered codec.
[[nodiscard]] const std::array<std::array<std::uint8_t, 8>, 256>& fp8lut4_magnitude_table() noexcept;

// Output of one projection: rows [0, split) publish to `leading` [T, split] and rows
// [split, rows) to `trailing` [T, rows - split]; split 0 publishes every row to `leading` [T, rows].
// `accumulate` publishes BF16(output + BF16(y)) instead of BF16(y).
// `silu_pair` (split 0, no accumulation): the weight rows are gate/up interleaved in 16-row tiles
// (row 16 b + i is gate feature 8 b + i, row 16 b + 8 + i its up partner) and `leading` [T, rows/2]
// receives BF16(silu(g) * u) of the BF16-rounded pair, the SiLU-gated MLP activation.
struct Fp8Lut4Output {
    hip_bfloat16* leading  = nullptr;
    hip_bfloat16* trailing = nullptr;
    std::uint32_t split    = 0;
    bool accumulate        = false;
    bool silu_pair         = false;
};

// y[t, r] = BF16(sum_k decode(w[r, k]) * a[t, k] * R[r] * s[t]) over one prepared per-token E4M3
// activation image (fp8_quantize_* producers). Every output element of a token whose status word
// is nonzero is the canonical BF16 quiet NaN. A split must be a multiple of 128 rows.
[[nodiscard]] bool fp8lut4_linear_supported(std::uint32_t tokens, std::uint32_t rows,
                                        std::uint32_t columns) noexcept;
// Row-scaled E4M3 rows through fp8lut4_linear: prefill widths (the prefill CTA and its split and
// accumulate epilogues); narrower widths use the row-scaled small-T and mid-T routes.
[[nodiscard]] bool fp8_row_scaled_projection_supported(std::uint32_t tokens, std::uint32_t rows,
                                                       std::uint32_t columns) noexcept;
[[nodiscard]] hipError_t fp8lut4_linear(const Fp8Lut4Weight& weight,
                                    const Fp8ActivationWorkspace& activation,
                                    const Fp8Lut4Output& output, hipStream_t stream) noexcept;
// Two projections of one image (one launch at small T); both accumulate or neither does.
[[nodiscard]] hipError_t fp8lut4_linear_pair(const Fp8Lut4Weight& first, const Fp8Lut4Output& first_output,
                                         const Fp8Lut4Weight& second, const Fp8Lut4Output& second_output,
                                         const Fp8ActivationWorkspace& activation,
                                         hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::linear
