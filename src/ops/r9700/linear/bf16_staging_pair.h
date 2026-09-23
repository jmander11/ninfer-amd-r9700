#pragma once
#include "ops/r9700/linear/r9700_linear.h"

namespace ninfer::ops::r9700::linear::detail {
// Internal launch: the owning bf16_linear entry validates pointers/extents and
// selects T5/6 only, (N,K)=(7168,5120) or (5120,6144). Preserve its stream and
// BF16 alignment contract. No workspace/allocation or repeat validation.
// Same per-lane FMA/reduction as bf16_linear_wave8, paired staging in K512 tiles.
[[nodiscard]] hipError_t launch_bf16_staging_pair(
    const Bf16LinearArgs& args, hipStream_t stream) noexcept;
}
