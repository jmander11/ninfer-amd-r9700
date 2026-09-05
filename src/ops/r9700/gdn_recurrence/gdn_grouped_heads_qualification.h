#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

namespace ninfer::ops::r9700::gdn_recurrence {

// Qualification-only ordinary normalized recurrence that groups the three
// value heads sharing one Q/K head. Production dispatch never calls this API.
hipError_t grouped_heads_qualification(
    const void* q, const void* k, const void* v, const float* g, const float* beta,
    const float* state_in, float* state_out, void* output, int width, float scale,
    hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn_recurrence
