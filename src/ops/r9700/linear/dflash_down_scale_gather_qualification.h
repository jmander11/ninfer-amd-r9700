#pragma once
#include "ops/r9700/linear/r9700_linear.h"

namespace ninfer::ops::r9700::linear::qualification {
// Temporary complete Linear boundary, not linked into production: represented
// BF16 [T,17408] -> exact A8G64 -> signed Q4N16K16/G64 [5120,17408] -> BF16
// [T,5120], T=5/6. Caller-owned exact workspace; five disjoint planes; explicit
// stream; no allocation, synchronization, graph capture or persistent state.
[[nodiscard]] hipError_t dflash_down_scale_gather(
    const A8Q4G64CandidateArgs& args, hipStream_t stream) noexcept;
}
