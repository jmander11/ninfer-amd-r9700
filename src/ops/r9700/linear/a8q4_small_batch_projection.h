#pragma once
#include "ops/r9700/linear/r9700_linear.h"

namespace ninfer::ops::r9700::linear {
// Complete regression boundary: BF16 -> A8G64 -> signed Q4N16K16/G64 Linear,
// T2..4 MLP/output: N34816/K5120, N5120/K17408, N5120/K6144.
// Tiled T10/12/15/18/20/24 MLP: N34816/K5120, N5120/K17408.
// Tiled T12/18/24 output: N5120/K6144. T5/6 projections:
// N34816/K5120, N5120/K6144, N12288/K5120, N4096/K5120.
// T5/6 draft down/feature: N5120/K17408 and N5120/K25600.
// T5/6 attention/auxiliary: N7168/K5120, N6144/K5120, N1280/K5120,
// N5120/K4096. Other widths retain their existing route.
// N12288 retains scale-gather; other admitted shapes use successor pipelining.
// No allocation or persistent-weight transformation.
[[nodiscard]] hipError_t a8q4_small_batch_projection(
    const A8Q4G64CandidateArgs& args, hipStream_t stream) noexcept;
namespace detail {
[[nodiscard]] constexpr bool use_a8q4_small_batch_projection(
    unsigned tokens, unsigned rows, unsigned columns, unsigned padded_columns) noexcept {
    return columns == padded_columns &&
        (((tokens == 5 || tokens == 6) &&
          ((rows == 5120 && (columns == 4096 || columns == 6144 || columns == 17408 || columns == 25600)) ||
           ((rows == 34816 || rows == 12288 || rows == 4096 || rows == 7168 ||
             rows == 6144 || rows == 1280) && columns == 5120))) ||
         ((tokens >= 2 && tokens <= 4) && rows == 5120 && columns == 6144) ||
         ((tokens == 12 || tokens == 18 || tokens == 24) && rows == 5120 && columns == 6144) ||
         (((tokens >= 2 && tokens <= 4) || tokens == 10 || tokens == 12 || tokens == 15 ||
           tokens == 18 || tokens == 20 || tokens == 24) &&
          ((rows == 34816 && columns == 5120) || (rows == 5120 && columns == 17408))));
}
// Internal prepared launch only: the generic owner has validated all planes,
// selected the exact predicate above and freshly quantized its A8 workspace.
// No second quantization, stream query, workspace binding or allocation.
[[nodiscard]] hipError_t launch_a8q4_small_batch_projection(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
}
}
