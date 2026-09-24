#include "device.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer {
namespace {

std::string hip_error_message(const char* prefix, hipError_t error) {
    return std::string(prefix) + ": " + hipGetErrorName(error) + ": " +
           hipGetErrorString(error);
}

void log_hip_error(const char* operation, hipError_t error) noexcept {
    if (error != hipSuccess) {
        std::fprintf(stderr, "HIP cleanup failed during %s: %s: %s\n", operation,
                     hipGetErrorName(error), hipGetErrorString(error));
    }
}

void destroy_stream(hipStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_hip_error("hipStreamDestroy", hipStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(hipEvent_t& event) noexcept {
    if (event != nullptr) {
        log_hip_error("hipEventDestroy", hipEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void hip_check(hipError_t error, const char* expr, const char* file, int line) {
    if (error == hipSuccess) { return; }
    std::fprintf(stderr, "%s:%d: HIP_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 hipGetErrorName(error), hipGetErrorString(error));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count = 0;
    hipError_t error = hipGetDeviceCount(&count);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipGetDeviceCount", error)); }
    if (count <= 0) { throw std::runtime_error("no HIP devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid HIP device id"); }

    error = hipSetDevice(device_id);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipSetDevice", error)); }
    error = hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipSetDeviceFlags", error)); }
    error = hipGetDeviceProperties(&props, device_id);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipGetDeviceProperties", error)); }
    if (!is_gfx1201() || props.warpSize != 32) {
        throw std::runtime_error("NInfer requires wave32 gfx1201");
    }

    hipStream_t compute = nullptr;
    hipStream_t load = nullptr;
    hipStream_t copy = nullptr;
    hipEvent_t order = nullptr;
    hipEvent_t wait = nullptr;
    error = hipStreamCreateWithFlags(&compute, hipStreamNonBlocking);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("create compute stream", error)); }
    error = hipStreamCreateWithFlags(&load, hipStreamNonBlocking);
    if (error != hipSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(hip_error_message("create load stream", error));
    }
    error = hipStreamCreateWithFlags(&copy, hipStreamNonBlocking);
    if (error != hipSuccess) {
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error(hip_error_message("create copy stream", error));
    }
    error = hipEventCreateWithFlags(&order, hipEventDisableTiming);
    if (error != hipSuccess) {
        destroy_stream(copy);
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error(hip_error_message("create copy-order event", error));
    }
    error = hipEventCreateWithFlags(&wait, hipEventDisableTiming | hipEventBlockingSync);
    if (error != hipSuccess) {
        destroy_event(order);
        destroy_stream(copy);
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error(hip_error_message("create host-wait event", error));
    }
    stream = compute;
    load_stream = load;
    copy_stream = copy;
    copy_order_event = order;
    host_wait = wait;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || load_stream != nullptr || copy_stream != nullptr ||
        copy_order_event != nullptr || host_wait != nullptr) {
        log_hip_error("hipSetDevice", hipSetDevice(device));
    }
    destroy_event(host_wait);
    destroy_event(copy_order_event);
    destroy_stream(copy_stream);
    destroy_stream(load_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), load_stream(other.load_stream),
      copy_stream(other.copy_stream), copy_order_event(other.copy_order_event),
      host_wait(other.host_wait), props(other.props) {
    other.stream = nullptr;
    other.load_stream = nullptr;
    other.copy_stream = nullptr;
    other.copy_order_event = nullptr;
    other.host_wait = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }
    if (stream != nullptr || load_stream != nullptr || copy_stream != nullptr ||
        copy_order_event != nullptr || host_wait != nullptr) {
        log_hip_error("hipSetDevice", hipSetDevice(device));
    }
    destroy_event(host_wait);
    destroy_event(copy_order_event);
    destroy_stream(copy_stream);
    destroy_stream(load_stream);
    destroy_stream(stream);
    device = other.device;
    props = other.props;
    stream = other.stream;
    load_stream = other.load_stream;
    copy_stream = other.copy_stream;
    copy_order_event = other.copy_order_event;
    host_wait = other.host_wait;
    other.stream = nullptr;
    other.load_stream = nullptr;
    other.copy_stream = nullptr;
    other.copy_order_event = nullptr;
    other.host_wait = nullptr;
    return *this;
}

bool DeviceContext::is_gfx1201() const noexcept {
    return std::string_view(props.gcnArchName) == "gfx1201";
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

void DeviceContext::synchronize() const {
    HIP_CHECK(hipEventRecord(host_wait, stream));
    HIP_CHECK(hipEventSynchronize(host_wait));
}

void DeviceContext::synchronize_all() const {
    synchronize();
    HIP_CHECK(hipEventRecord(host_wait, copy_stream));
    HIP_CHECK(hipEventSynchronize(host_wait));
}

void DeviceContext::order_copy_after_compute() const {
    HIP_CHECK(hipEventRecord(copy_order_event, stream));
    HIP_CHECK(hipStreamWaitEvent(copy_stream, copy_order_event, 0));
}

DeviceEventTimer::DeviceEventTimer(const DeviceContext& context) : stream_(context.stream) {
    hipError_t error = hipSetDevice(context.device);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipSetDevice", error)); }
    error = hipEventCreate(&start_);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("create timer start", error)); }
    error = hipEventCreate(&stop_);
    if (error != hipSuccess) {
        destroy_event(start_);
        throw std::runtime_error(hip_error_message("create timer stop", error));
    }
}

DeviceEventTimer::~DeviceEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

DeviceEventTimer::DeviceEventTimer(DeviceEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_ = nullptr;
    other.stop_ = nullptr;
}

DeviceEventTimer& DeviceEventTimer::operator=(DeviceEventTimer&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(stop_);
    destroy_event(start_);
    stream_ = other.stream_;
    start_ = other.start_;
    stop_ = other.stop_;
    other.stream_ = nullptr;
    other.start_ = nullptr;
    other.stop_ = nullptr;
    return *this;
}

void DeviceEventTimer::start() { HIP_CHECK(hipEventRecord(start_, stream_)); }
void DeviceEventTimer::record_stop() { HIP_CHECK(hipEventRecord(stop_, stream_)); }
float DeviceEventTimer::elapsed_ms() const {
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
}
float DeviceEventTimer::stop_ms() {
    record_stop();
    HIP_CHECK(hipEventSynchronize(stop_));
    return elapsed_ms();
}

} // namespace ninfer
