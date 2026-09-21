#pragma once
#include "ops/r9700/linear/r9700_linear.h"

namespace ninfer::ops::r9700::gdn::qualification {
// Temporary complete boundary: represented BF16 hidden[5120] -> exact A8G64
// codec -> signed Q4N16K16 projections QK[4096], value-Z[12288]; and independent
// BF16 [48,5120] a/b projections -> explicit BF16 seams -> FP32 g/beta.
// All thirteen input/output planes and private workspace are disjoint. The
// caller owns scratch and stream. No allocation, synchronization or retained state.
struct ProjectionControlArgs {
    linear::A8Q4G64GdnPairArgs projection;
    const hip_bfloat16* a_weight{};
    const hip_bfloat16* b_weight{};
    std::size_t control_weight_bytes{};
    const float* a_log{};
    const float* dt_bias{};
    float* g{};
    float* beta{};
};
[[nodiscard]] hipError_t projection_control_grid(
    const ProjectionControlArgs& args, hipStream_t stream) noexcept;
}
