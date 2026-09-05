#include "arena.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1U)) == 0; }

void free_device(void*& data) noexcept {
    if (data != nullptr) {
        (void)hipFree(data);
        data = nullptr;
    }
}

void free_pinned(void*& data) noexcept {
    if (data != nullptr) {
        (void)hipHostFree(data);
        data = nullptr;
    }
}

[[noreturn]] void allocation_error(const char* what, hipError_t error) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorName(error) + ": " +
                             hipGetErrorString(error));
}

} // namespace

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }
    const hipError_t error = hipMalloc(&p, bytes);
    if (error != hipSuccess) { allocation_error("hipMalloc", error); }
}

DeviceBuffer::~DeviceBuffer() { free_device(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
    other.p = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        free_device(p);
        p = other.p;
        bytes = other.bytes;
        other.p = nullptr;
        other.bytes = 0;
    }
    return *this;
}

void DeviceBuffer::require_range(std::size_t offset, std::size_t count, const char* operation) const {
    if (offset > bytes || count > bytes - offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

void DeviceBuffer::fill(int value) {
    if (bytes == 0) { return; }
    const hipError_t error = hipMemset(p, value, bytes);
    if (error != hipSuccess) { allocation_error("hipMemset", error); }
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t offset) {
    require_range(offset, count, "host-to-device copy");
    if (count == 0) { return; }
    const hipError_t error = hipMemcpy(static_cast<std::uint8_t*>(p) + offset, source, count,
                                       hipMemcpyHostToDevice);
    if (error != hipSuccess) { allocation_error("hipMemcpy host-to-device", error); }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count, std::size_t offset) const {
    require_range(offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const hipError_t error = hipMemcpy(destination, static_cast<const std::uint8_t*>(p) + offset,
                                       count, hipMemcpyDeviceToHost);
    if (error != hipSuccess) { allocation_error("hipMemcpy device-to-host", error); }
}

void DeviceBuffer::fill_async(int value, hipStream_t stream) {
    if (stream == nullptr) { throw std::invalid_argument("async device fill requires a stream"); }
    if (bytes != 0) {
        const hipError_t error = hipMemsetAsync(p, value, bytes, stream);
        if (error != hipSuccess) { allocation_error("hipMemsetAsync", error); }
    }
}

void DeviceBuffer::copy_from_host_async(const void* source, std::size_t count, hipStream_t stream,
                                        std::size_t offset) {
    require_range(offset, count, "async host-to-device copy");
    if (stream == nullptr) { throw std::invalid_argument("async host-to-device copy requires a stream"); }
    if (count != 0) {
        const hipError_t error = hipMemcpyAsync(static_cast<std::uint8_t*>(p) + offset, source,
                                                count, hipMemcpyHostToDevice, stream);
        if (error != hipSuccess) { allocation_error("hipMemcpyAsync host-to-device", error); }
    }
}

void DeviceBuffer::copy_to_host_async(void* destination, std::size_t count, hipStream_t stream,
                                      std::size_t offset) const {
    require_range(offset, count, "async device-to-host copy");
    if (stream == nullptr) { throw std::invalid_argument("async device-to-host copy requires a stream"); }
    if (count != 0) {
        const hipError_t error = hipMemcpyAsync(destination, static_cast<const std::uint8_t*>(p) + offset,
                                                count, hipMemcpyDeviceToHost, stream);
        if (error != hipSuccess) { allocation_error("hipMemcpyAsync device-to-host", error); }
    }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.offset_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ != nullptr && saved_offset_ <= arena_->offset_) { arena_->offset_ = saved_offset_; }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes) : capacity_(capacity_bytes) {
    if (capacity_ == 0) { throw std::invalid_argument("DeviceArena capacity must be nonzero"); }
    const hipError_t error = hipMalloc(&base_, capacity_);
    if (error != hipSuccess) { allocation_error("hipMalloc arena", error); }
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), capacity_(storage.bytes), owns_(false) {
    if (base_ == nullptr || capacity_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() {
    if (owns_) { free_device(base_); }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), capacity_(other.capacity_), offset_(other.offset_), peak_(other.peak_),
      owns_(other.owns_) {
    other.base_ = nullptr;
    other.capacity_ = other.offset_ = other.peak_ = 0;
    other.owns_ = true;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this != &other) {
        if (owns_) { free_device(base_); }
        base_ = other.base_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        peak_ = other.peak_;
        owns_ = other.owns_;
        other.base_ = nullptr;
        other.capacity_ = other.offset_ = other.peak_ = 0;
        other.owns_ = true;
    }
    return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(base_);
    if (offset_ > std::numeric_limits<std::uintptr_t>::max() - base) {
        throw std::overflow_error("arena address overflow");
    }
    const std::uintptr_t address = base + offset_;
    if (address > std::numeric_limits<std::uintptr_t>::max() - (align - 1U)) {
        throw std::overflow_error("arena alignment overflow");
    }
    const std::uintptr_t aligned = (address + align - 1U) & ~static_cast<std::uintptr_t>(align - 1U);
    const std::size_t next = static_cast<std::size_t>(aligned - base);
    if (next > capacity_ || bytes > capacity_ - next) { throw std::bad_alloc(); }
    offset_ = next + bytes;
    peak_ = std::max(peak_, offset_);
    return {reinterpret_cast<void*>(aligned), bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }
void DeviceArena::reset() noexcept { offset_ = 0; }

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) : bytes_(size_bytes) {
    if (bytes_ == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }
    const hipError_t error = hipHostMalloc(&data_, bytes_, hipHostMallocDefault);
    if (error != hipSuccess) { allocation_error("hipHostMalloc", error); }
}

PinnedHostBuffer::~PinnedHostBuffer() { free_pinned(data_); }
PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
}
PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        free_pinned(data_);
        data_ = other.data_;
        bytes_ = other.bytes_;
        other.data_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

HostPinnedArena::HostPinnedArena(std::size_t capacity_bytes) : capacity_(capacity_bytes) {
    if (capacity_ == 0) { throw std::invalid_argument("HostPinnedArena capacity must be nonzero"); }
    const hipError_t error = hipHostMalloc(&base_, capacity_, hipHostMallocDefault);
    if (error != hipSuccess) { allocation_error("hipHostMalloc arena", error); }
    free_.push_back({0, capacity_});
}

HostPinnedArena::~HostPinnedArena() { free_pinned(base_); }
HostPinnedArena::HostPinnedArena(HostPinnedArena&& other) noexcept
    : base_(other.base_), capacity_(other.capacity_), used_(other.used_), free_(std::move(other.free_)),
      live_(std::move(other.live_)) {
    other.base_ = nullptr;
    other.capacity_ = other.used_ = 0;
}
HostPinnedArena& HostPinnedArena::operator=(HostPinnedArena&& other) noexcept {
    if (this != &other) {
        free_pinned(base_);
        base_ = other.base_;
        capacity_ = other.capacity_;
        used_ = other.used_;
        free_ = std::move(other.free_);
        live_ = std::move(other.live_);
        other.base_ = nullptr;
        other.capacity_ = other.used_ = 0;
    }
    return *this;
}

void* HostPinnedArena::try_alloc(std::size_t bytes, std::size_t align) {
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("pinned arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("pinned arena allocation must be nonzero"); }
    for (std::size_t index = 0; index < free_.size(); ++index) {
        const Span span = free_[index];
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base_) + span.offset;
        const std::uintptr_t aligned = (address + align - 1U) & ~static_cast<std::uintptr_t>(align - 1U);
        const std::size_t lead = static_cast<std::size_t>(aligned - address);
        if (lead > span.bytes || bytes > span.bytes - lead) { continue; }
        const std::size_t offset = span.offset + lead;
        const std::size_t tail = span.bytes - lead - bytes;
        free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(index));
        if (lead != 0) { insert_free(span.offset, lead); }
        if (tail != 0) { insert_free(offset + bytes, tail); }
        void* result = static_cast<std::uint8_t*>(base_) + offset;
        live_.emplace(result, Span{offset, bytes});
        used_ += bytes;
        return result;
    }
    return nullptr;
}

void HostPinnedArena::free(void* block) {
    const auto it = live_.find(block);
    if (it == live_.end()) { throw std::logic_error("pinned arena free of unknown block"); }
    const Span span = it->second;
    live_.erase(it);
    used_ -= span.bytes;
    insert_free(span.offset, span.bytes);
}

void HostPinnedArena::insert_free(std::size_t offset, std::size_t bytes) {
    auto it = free_.begin();
    while (it != free_.end() && it->offset < offset) { ++it; }
    if (it != free_.begin()) {
        auto previous = it - 1;
        if (previous->offset + previous->bytes == offset) {
            previous->bytes += bytes;
            if (it != free_.end() && previous->offset + previous->bytes == it->offset) {
                previous->bytes += it->bytes;
                free_.erase(it);
            }
            return;
        }
    }
    if (it != free_.end() && offset + bytes == it->offset) {
        it->offset = offset;
        it->bytes += bytes;
        return;
    }
    free_.insert(it, Span{offset, bytes});
}

} // namespace ninfer
