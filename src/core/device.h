#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

namespace ninfer {

void hip_check(hipError_t error, const char* expr, const char* file, int line);

// Remaining driver-backed pinned-host budget on the current HIP device.
// Other users may consume the shared GTT heap before allocation succeeds.
[[nodiscard]] std::size_t available_pinned_host_memory();

#define HIP_CHECK(expr) ::ninfer::hip_check((expr), #expr, __FILE__, __LINE__)

// Owns ordered compute, load and copy streams plus blocking host events.
// Only wave32 gfx1201 devices are accepted.
struct DeviceContext {
    int device                          = 0;
    hipStream_t stream                  = nullptr;
    hipStream_t load_stream             = nullptr;
    hipStream_t copy_stream             = nullptr;
    hipEvent_t copy_order_event         = nullptr;
    hipEvent_t compute_copy_order_event = nullptr;
    hipEvent_t compute_load_order_event = nullptr;
    hipEvent_t host_wait                = nullptr;
    hipDeviceProp_t props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();
    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    [[nodiscard]] bool is_gfx1201() const noexcept;
    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    [[nodiscard]] std::size_t total_vram() const noexcept;
    // Conservative HIP/graphics-inclusive physical availability for this PCI device.
    // Required Linux amdgpu counters must be readable and valid; no HIP-only fallback.
    [[nodiscard]] std::size_t available_vram() const;
    void synchronize() const;
    void synchronize_all() const;
    void order_copy_after_compute() const;
    // Enqueue compute dependencies on the fixed prior load/copy frontiers, without a host wait.
    void order_compute_after_transfers() const;
};

class DeviceEventTimer {
public:
    explicit DeviceEventTimer(const DeviceContext& context);
    ~DeviceEventTimer();
    DeviceEventTimer(const DeviceEventTimer&)            = delete;
    DeviceEventTimer& operator=(const DeviceEventTimer&) = delete;
    DeviceEventTimer(DeviceEventTimer&& other) noexcept;
    DeviceEventTimer& operator=(DeviceEventTimer&& other) noexcept;
    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    hipStream_t stream_ = nullptr;
    hipEvent_t start_   = nullptr;
    hipEvent_t stop_    = nullptr;
};

} // namespace ninfer
