#pragma once
#include "ops/r9700/linear/r9700_linear.h"

namespace ninfer::ops::r9700::linear {
// Complete regression boundary: BF16 -> A8G64 -> signed Q4N16K16/G64 Linear,
// restricted to T5/T6 and N34816/K5120, N5120/K6144, N12288/K5120, N4096/K5120.
// N12288 retains scale-gather; the other exact shapes use successor pipelining.
// No allocation or persistent-weight transformation.
[[nodiscard]] hipError_t a8q4_verify_projection_candidate(
    const A8Q4G64CandidateArgs& args, hipStream_t stream) noexcept;
namespace detail {
[[nodiscard]] constexpr bool use_a8q4_verify_projection(
    unsigned tokens, unsigned rows, unsigned columns, unsigned padded_columns) noexcept {
    return (tokens == 5 || tokens == 6) && columns == padded_columns &&
        ((rows == 5120 && columns == 6144) ||
         ((rows == 34816 || rows == 12288 || rows == 4096) && columns == 5120));
}
// Internal prepared launch only: the generic owner has validated all planes,
// selected the exact predicate above and freshly quantized its A8 workspace.
// No second quantization, stream query, workspace binding or allocation.
[[nodiscard]] hipError_t launch_a8q4_verify_projection(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
}
}
