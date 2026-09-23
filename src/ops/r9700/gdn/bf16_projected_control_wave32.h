#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace ninfer::ops::r9700::gdn {
// Internal complete projected controls. Fixed N48/K5120, T5/6.
// Represented BF16 dots, explicit BF16 seams, then existing FP32 gating.
// All seven planes are disjoint; BF16 alignment2, FP32 alignment4, nonnull stream.
// No allocation/workspace. The semantic public Op validates descriptors.
struct Bf16ProjectedControlWave32Args {
    const hip_bfloat16* hidden;
    const hip_bfloat16* a_weight;
    const hip_bfloat16* b_weight;
    const float* a_log;
    const float* dt_bias;
    float* g;
    float* beta;
    std::uint32_t tokens;
};
[[nodiscard]] hipError_t bf16_projected_control_wave32(
    const Bf16ProjectedControlWave32Args&, hipStream_t) noexcept;
}
