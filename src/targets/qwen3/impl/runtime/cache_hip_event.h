#pragma once

#include "core/device.h"

#include <new>

namespace ninfer::targets::qwen3::detail {

// HIP documents memory exhaustion separately from asynchronous launch failures
// for event creation. Optional cache ownership can recover only the former.
inline void check_cache_hip_event_allocation(hipError_t result) {
    if (result == hipErrorMemoryAllocation) {
        const hipError_t pending = hipGetLastError();
        if (pending != hipSuccess && pending != hipErrorMemoryAllocation) {
            HIP_CHECK(pending);
        }
        throw std::bad_alloc();
    }
    HIP_CHECK(result);
}

inline void create_cache_hip_event(hipEvent_t* event, unsigned int flags) {
    // HIP specifies a newly created output on success, without promising the
    // output value on failure. Publish ownership only after creation succeeds.
    hipEvent_t candidate = nullptr;
    check_cache_hip_event_allocation(hipEventCreateWithFlags(&candidate, flags));
    *event = candidate;
}

} // namespace ninfer::targets::qwen3::detail
