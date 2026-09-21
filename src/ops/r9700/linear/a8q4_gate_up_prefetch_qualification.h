#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace ninfer::ops::r9700::linear {
// Qualification-only internal launcher for exact T1 N34816/K5120, A8G64,
// persistent N16/K16 Q4G64 weights. Caller owns complete aligned nonoverlapping
// planes and the explicit stream, just as the validated production launcher.
// No preparation, allocation, synchronization, or production dispatch is added.
[[nodiscard]] hipError_t launch_a8q4_gate_up_prefetch_qualification(
    const std::uint8_t* low, const std::uint8_t* high, const std::uint16_t* as,
    const std::uint32_t* status, const std::uint8_t* codes,
    const std::uint16_t* scales, hip_bfloat16* output, hipStream_t stream);
}
