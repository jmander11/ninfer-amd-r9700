#pragma once
#include "ops/r9700/linear/r9700_linear.h"

namespace ninfer::ops::r9700::linear {
// Resident W8N16K16 output head only: N248320/K5120/G32. Codes are signed bytes
// [N/16][K/32][2][16][16], scales are FP16 [N/16][K/32][16].
// These consumers never transform weights or allocate storage. The public
// Linear owner retains use_a8w8 policy and supplies the ordinary A8 workspace.
[[nodiscard]] hipError_t a8w8g32_tiled_head(
    const A8W8G32CandidateArgs& args, hipStream_t stream) noexcept;
// Exact BF16 x decoded W8 route, including the compile-time A16 control profile.
[[nodiscard]] hipError_t w8g32_tiled_head(
    const W8G32LinearArgs& args, hipStream_t stream) noexcept;
}
