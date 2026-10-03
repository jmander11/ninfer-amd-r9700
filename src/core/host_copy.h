#pragma once

#include "core/device.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace ninfer {

struct HostCopy {
    void* dst         = nullptr;
    const void* src   = nullptr;
    std::size_t bytes = 0;
};

// Copies between host buffers in `stream` order without blocking the caller. ROCm runs
// hipMemcpyAsync between two host buffers as a CPU copy inside the call (about 11 ms per
// 150 MB GDN image), so a host callback performs the memcpy on the runtime's callback thread
// instead: later work on the stream, and events recorded after this call, wait for the copies.
// Both buffers must stay valid and unaliased by other writers until such an event completes.
inline void enqueue_host_copies(std::vector<HostCopy> copies, hipStream_t stream) {
    if (copies.empty()) { return; }
    auto batch     = std::make_unique<std::vector<HostCopy>>(std::move(copies));
    const auto run = [](void* data) {
        const std::unique_ptr<std::vector<HostCopy>> owned(
            static_cast<std::vector<HostCopy>*>(data));
        for (const HostCopy& copy : *owned) {
            if (copy.bytes != 0) { std::memcpy(copy.dst, copy.src, copy.bytes); }
        }
    };
    HIP_CHECK(hipLaunchHostFunc(stream, run, batch.get()));
    // The launched callback now owns the batch and deletes it after the copies.
    [[maybe_unused]] std::vector<HostCopy>* const callback_owned = batch.release();
}

} // namespace ninfer
