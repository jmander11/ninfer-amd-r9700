#include "device.h"

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <system_error>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer {
namespace {

std::string hip_error_message(const char* prefix, hipError_t error) {
    return std::string(prefix) + ": " + hipGetErrorName(error) + ": " + hipGetErrorString(error);
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

std::size_t read_amdgpu_memory_counter(const std::string& path) {
    std::ifstream input(path);
    std::string text;
    std::string extra;
    if (!(input >> text) || (input >> extra) || input.bad()) {
        throw std::runtime_error("cannot read required amdgpu memory counter: " + path);
    }
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid amdgpu memory counter: " + path);
    }
    return value;
}

std::string amdgpu_memory_root(int device) {
    char pci_id[64]{};
    const hipError_t pci_error = hipDeviceGetPCIBusId(pci_id, sizeof(pci_id), device);
    if (pci_error != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipDeviceGetPCIBusId", pci_error));
    }
    const std::string_view identity(pci_id);
    const char* cursor   = identity.data();
    const char* end      = cursor + identity.size();
    const auto component = [&](unsigned int maximum, char delimiter) {
        unsigned int value = 0;
        const auto parsed  = std::from_chars(cursor, end, value, 16);
        if (parsed.ec != std::errc{} || value > maximum ||
            (delimiter == '\0' ? parsed.ptr != end
                               : parsed.ptr == end || *parsed.ptr != delimiter)) {
            throw std::runtime_error("invalid HIP PCI device identity: " + std::string(pci_id));
        }
        cursor = parsed.ptr + (delimiter == '\0' ? 0 : 1);
        return value;
    };
    const unsigned int domain   = component(0xffffU, ':');
    const unsigned int bus      = component(0xffU, ':');
    const unsigned int slot     = component(0x1fU, '.');
    const unsigned int function = component(7U, '\0');
    char canonical_pci_id[32]{};
    std::snprintf(canonical_pci_id, sizeof(canonical_pci_id), "%04x:%02x:%02x.%x", domain, bus,
                  slot, function);
    return std::string("/sys/bus/pci/devices/") + canonical_pci_id + "/";
}

void configure_host_memory_policy() {
    static std::once_flag configured;
    std::call_once(configured, [] {
        // Fixed gfx1201 policy: driver-backed host memory and normal unified VA.
        // Set before the first HIP call; these options are consumed by ROCr init.
        if (::setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 1) != 0 ||
            ::setenv("HSA_USE_SVM", "1", 1) != 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "configure R9700 host-memory policy");
        }
    });
}

} // namespace

void hip_check(hipError_t error, const char* expr, const char* file, int line) {
    if (error == hipSuccess) { return; }
    std::fprintf(stderr, "%s:%d: HIP_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 hipGetErrorName(error), hipGetErrorString(error));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    configure_host_memory_policy();
    int count        = 0;
    hipError_t error = hipGetDeviceCount(&count);
    if (error != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipGetDeviceCount", error));
    }
    if (count <= 0) { throw std::runtime_error("no HIP devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid HIP device id"); }

    error = hipSetDevice(device_id);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipSetDevice", error)); }
    error = hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
    if (error != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipSetDeviceFlags", error));
    }
    error = hipGetDeviceProperties(&props, device_id);
    if (error != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipGetDeviceProperties", error));
    }
    if (!is_gfx1201() || props.warpSize != 32) {
        throw std::runtime_error("NInfer requires wave32 gfx1201");
    }

    hipStream_t compute      = nullptr;
    hipStream_t load         = nullptr;
    hipStream_t copy         = nullptr;
    hipEvent_t order         = nullptr;
    hipEvent_t compute_order = nullptr;
    hipEvent_t load_order    = nullptr;
    hipEvent_t wait          = nullptr;
    error                    = hipStreamCreateWithFlags(&compute, hipStreamNonBlocking);
    if (error != hipSuccess) {
        throw std::runtime_error(hip_error_message("create compute stream", error));
    }
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
    error = hipEventCreateWithFlags(&compute_order, hipEventDisableTiming);
    if (error != hipSuccess) {
        destroy_event(order);
        destroy_stream(copy);
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error(hip_error_message("create compute-order event", error));
    }
    error = hipEventCreateWithFlags(&load_order, hipEventDisableTiming);
    if (error != hipSuccess) {
        destroy_event(compute_order);
        destroy_event(order);
        destroy_stream(copy);
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error(hip_error_message("create load-order event", error));
    }
    error = hipEventCreateWithFlags(&wait, hipEventDisableTiming | hipEventBlockingSync);
    if (error != hipSuccess) {
        destroy_event(load_order);
        destroy_event(compute_order);
        destroy_event(order);
        destroy_stream(copy);
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error(hip_error_message("create host-wait event", error));
    }
    stream                   = compute;
    load_stream              = load;
    copy_stream              = copy;
    copy_order_event         = order;
    compute_copy_order_event = compute_order;
    compute_load_order_event = load_order;
    host_wait                = wait;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || load_stream != nullptr || copy_stream != nullptr ||
        copy_order_event != nullptr || compute_copy_order_event != nullptr ||
        compute_load_order_event != nullptr || host_wait != nullptr) {
        log_hip_error("hipSetDevice", hipSetDevice(device));
    }
    destroy_event(host_wait);
    destroy_event(compute_copy_order_event);
    destroy_event(compute_load_order_event);
    destroy_event(copy_order_event);
    destroy_stream(copy_stream);
    destroy_stream(load_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), load_stream(other.load_stream),
      copy_stream(other.copy_stream), copy_order_event(other.copy_order_event),
      compute_copy_order_event(other.compute_copy_order_event),
      compute_load_order_event(other.compute_load_order_event), host_wait(other.host_wait),
      props(other.props) {
    other.stream                   = nullptr;
    other.load_stream              = nullptr;
    other.copy_stream              = nullptr;
    other.copy_order_event         = nullptr;
    other.compute_copy_order_event = nullptr;
    other.compute_load_order_event = nullptr;
    other.host_wait                = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }
    if (stream != nullptr || load_stream != nullptr || copy_stream != nullptr ||
        copy_order_event != nullptr || compute_copy_order_event != nullptr ||
        compute_load_order_event != nullptr || host_wait != nullptr) {
        log_hip_error("hipSetDevice", hipSetDevice(device));
    }
    destroy_event(host_wait);
    destroy_event(compute_copy_order_event);
    destroy_event(compute_load_order_event);
    destroy_event(copy_order_event);
    destroy_stream(copy_stream);
    destroy_stream(load_stream);
    destroy_stream(stream);
    device                         = other.device;
    props                          = other.props;
    stream                         = other.stream;
    load_stream                    = other.load_stream;
    copy_stream                    = other.copy_stream;
    copy_order_event               = other.copy_order_event;
    compute_copy_order_event       = other.compute_copy_order_event;
    compute_load_order_event       = other.compute_load_order_event;
    host_wait                      = other.host_wait;
    other.stream                   = nullptr;
    other.load_stream              = nullptr;
    other.copy_stream              = nullptr;
    other.copy_order_event         = nullptr;
    other.compute_copy_order_event = nullptr;
    other.compute_load_order_event = nullptr;
    other.host_wait                = nullptr;
    return *this;
}

bool DeviceContext::is_gfx1201() const noexcept {
    return std::string_view(props.gcnArchName) == "gfx1201";
}

void DeviceContext::bind_to_current_thread() const {
    const hipError_t error = hipSetDevice(device);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipSetDevice", error)); }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_hip_error("hipSetDevice", hipSetDevice(device));
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

std::size_t DeviceContext::available_vram() const {
    bind_to_current_thread();
    std::size_t hip_free          = 0;
    std::size_t hip_total         = 0;
    const hipError_t memory_error = hipMemGetInfo(&hip_free, &hip_total);
    if (memory_error != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipMemGetInfo", memory_error));
    }
    const std::string root           = amdgpu_memory_root(device);
    const std::size_t physical_total = read_amdgpu_memory_counter(root + "mem_info_vram_total");
    const std::size_t physical_used  = read_amdgpu_memory_counter(root + "mem_info_vram_used");
    if (physical_total == 0 || physical_used > physical_total) {
        throw std::runtime_error("inconsistent amdgpu VRAM counters for " + root);
    }
    // KFD/HIP availability can exclude graphics allocations. External headroom must
    // remain physically available to other GPU users, not just to KFD allocations.
    return std::min(hip_free, physical_total - physical_used);
}

std::size_t available_pinned_host_memory() {
    int device             = 0;
    const hipError_t error = hipGetDevice(&device);
    if (error != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipGetDevice for GTT capacity", error));
    }
    const std::string root  = amdgpu_memory_root(device);
    const std::size_t total = read_amdgpu_memory_counter(root + "mem_info_gtt_total");
    const std::size_t used  = read_amdgpu_memory_counter(root + "mem_info_gtt_used");
    if (total == 0 || used > total) {
        throw std::runtime_error("inconsistent amdgpu GTT counters for " + root);
    }
    return total - used;
}

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

void DeviceContext::order_compute_after_transfers() const {
    HIP_CHECK(hipEventRecord(compute_load_order_event, load_stream));
    HIP_CHECK(hipStreamWaitEvent(stream, compute_load_order_event, 0));
    HIP_CHECK(hipEventRecord(compute_copy_order_event, copy_stream));
    HIP_CHECK(hipStreamWaitEvent(stream, compute_copy_order_event, 0));
}

DeviceEventTimer::DeviceEventTimer(const DeviceContext& context) : stream_(context.stream) {
    hipError_t error = hipSetDevice(context.device);
    if (error != hipSuccess) { throw std::runtime_error(hip_error_message("hipSetDevice", error)); }
    error = hipEventCreate(&start_);
    if (error != hipSuccess) {
        throw std::runtime_error(hip_error_message("create timer start", error));
    }
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
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

DeviceEventTimer& DeviceEventTimer::operator=(DeviceEventTimer&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(stop_);
    destroy_event(start_);
    stream_       = other.stream_;
    start_        = other.start_;
    stop_         = other.stop_;
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
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
