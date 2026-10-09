#pragma once

#include "core/device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>

namespace ninfer {

struct HostCopy {
    void* dst         = nullptr;
    const void* src   = nullptr;
    std::size_t bytes = 0;
};

// Copies between host buffers in `stream` order without blocking the caller. A host-to-host
// hipMemcpyAsync is fully synchronous with respect to the host (about 11 ms per 150 MB GDN
// image), so host callbacks perform the memcpy on the runtime's callback thread instead: later
// work on the stream, and events recorded after this call, wait for the copies. Both buffers must
// stay valid and unaliased by other writers until such an event completes. The callbacks make no
// HIP API call.
//
// Host functions of all streams share the runtime's callback thread, so each callback copies at
// most kHostCopyChunkBytes (about 0.3 ms): a host function on another stream (the decode
// tool-mask exchange) then waits behind at most one chunk instead of a whole image.
inline constexpr std::size_t kHostCopyChunkBytes = std::size_t{4} << 20;

inline void enqueue_host_copies(std::span<const HostCopy> copies, hipStream_t stream) {
    const auto run = [](void* data) {
        const std::unique_ptr<HostCopy> owned(static_cast<HostCopy*>(data));
        std::memcpy(owned->dst, owned->src, owned->bytes);
    };
    for (const HostCopy& copy : copies) {
        for (std::size_t offset = 0; offset < copy.bytes; offset += kHostCopyChunkBytes) {
            auto chunk = std::make_unique<HostCopy>(
                HostCopy{static_cast<std::byte*>(copy.dst) + offset,
                         static_cast<const std::byte*>(copy.src) + offset,
                         std::min(kHostCopyChunkBytes, copy.bytes - offset)});
            HIP_CHECK(hipLaunchHostFunc(stream, run, chunk.get()));
            // The enqueued callback owns the chunk from here and frees it after the copy.
            HostCopy* const enqueued = chunk.release();
            static_cast<void>(enqueued);
        }
    }
}

// A startup-owned stream dedicated to host-to-host image copies. Its copies wait only for the
// fences of the images they read or write, not for the device transfers already queued on the
// stream that consumes them, so image copies overlap a cache entry's KV D2H/H2D. run() then joins
// the consumer stream: its later work and the events it records cover the copies.
class HostCopyStream {
public:
    HostCopyStream() {
        HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
        const hipError_t event = hipEventCreateWithFlags(&joined_, hipEventDisableTiming);
        if (event != hipSuccess) {
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
            HIP_CHECK(event);
        }
    }

    ~HostCopyStream() {
        if (stream_ != nullptr) { (void)hipStreamSynchronize(stream_); }
        if (joined_ != nullptr) { (void)hipEventDestroy(joined_); }
        if (stream_ != nullptr) { (void)hipStreamDestroy(stream_); }
    }

    HostCopyStream(const HostCopyStream&)            = delete;
    HostCopyStream& operator=(const HostCopyStream&) = delete;
    HostCopyStream(HostCopyStream&&)                 = delete;
    HostCopyStream& operator=(HostCopyStream&&)      = delete;

    // Copies after every fence in `fences` (null entries are skipped), re-records each fence
    // after the copies so the images' later readers and writers wait for them, and makes
    // `consumer` wait for the copies. A failure while enqueueing drains the copies already
    // enqueued before it propagates, so no callback outlives the caller's cleanup.
    void run(std::span<const HostCopy> copies, std::span<const hipEvent_t> fences,
             hipStream_t consumer) {
        if (copies.empty()) { return; }
        std::lock_guard lock(mutex_);
        try {
            enqueue_fenced(copies, fences);
            HIP_CHECK(hipEventRecord(joined_, stream_));
            HIP_CHECK(hipStreamWaitEvent(consumer, joined_, 0));
        } catch (...) {
            (void)hipStreamSynchronize(stream_);
            throw;
        }
    }

    // Like run(), but no stream joins the copies: `done` is recorded after them instead, for an
    // owner that must keep a buffer alive until they finish without ordering later work behind
    // them. The re-recorded `fences` order the images' own readers and writers.
    void run_detached(std::span<const HostCopy> copies, std::span<const hipEvent_t> fences,
                      hipEvent_t done) {
        if (copies.empty()) { return; }
        std::lock_guard lock(mutex_);
        try {
            enqueue_fenced(copies, fences);
            HIP_CHECK(hipEventRecord(done, stream_));
        } catch (...) {
            (void)hipStreamSynchronize(stream_);
            throw;
        }
    }

    [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }

private:
    void enqueue_fenced(std::span<const HostCopy> copies, std::span<const hipEvent_t> fences) {
        for (hipEvent_t fence : fences) {
            if (fence != nullptr) { HIP_CHECK(hipStreamWaitEvent(stream_, fence, 0)); }
        }
        enqueue_host_copies(copies, stream_);
        for (hipEvent_t fence : fences) {
            if (fence != nullptr) { HIP_CHECK(hipEventRecord(fence, stream_)); }
        }
    }

    hipStream_t stream_ = nullptr;
    // Recorded and immediately waited under mutex_, so one event serves every join.
    hipEvent_t joined_ = nullptr;
    std::mutex mutex_;
};

} // namespace ninfer
