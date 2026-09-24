#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

namespace ninfer {

void hip_check(hipError_t err, const char* expr, const char* file, int line);

#define HIP_CHECK(expr) ::ninfer::hip_check((expr), #expr, __FILE__, __LINE__)

// Sole device-context contract for the R9700 product. It owns ordered compute, load, and copy
// streams plus blocking host events, and rejects every target other than wave32 gfx1201.
struct DeviceContext {
    int device = 0;
    hipStream_t stream = nullptr;
    hipStream_t load_stream = nullptr;
    hipStream_t copy_stream = nullptr;
    hipEvent_t copy_order_event = nullptr;
    hipEvent_t host_wait = nullptr;
    hipDeviceProp_t props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    [[nodiscard]] bool is_gfx1201() const noexcept;
    [[nodiscard]] std::size_t total_vram() const noexcept;
    void synchronize() const;
    void synchronize_all() const;
    void order_copy_after_compute() const;
};

class DeviceEventTimer {
public:
    explicit DeviceEventTimer(const DeviceContext& ctx);
    ~DeviceEventTimer();
    DeviceEventTimer(const DeviceEventTimer&) = delete;
    DeviceEventTimer& operator=(const DeviceEventTimer&) = delete;
    DeviceEventTimer(DeviceEventTimer&& other) noexcept;
    DeviceEventTimer& operator=(DeviceEventTimer&& other) noexcept;
    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    hipStream_t stream_ = nullptr;
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_ = nullptr;
};

} // namespace ninfer
