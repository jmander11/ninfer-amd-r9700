#pragma once

#include "dtype.h"
#include "tensor.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <unordered_map>
#include <vector>

namespace ninfer {

struct DeviceSpan {
    void* data = nullptr;
    std::size_t bytes = 0;
};

// Owning long-lived HIP allocation. Tensor and Weight remain non-owning views, so raw access is
// intentional. Async calls require the caller's explicit ordered HIP stream.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    void fill(int byte_value = 0);
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;
    void fill_async(int byte_value, hipStream_t stream);
    void copy_from_host_async(const void* source, std::size_t count, hipStream_t stream,
                              std::size_t byte_offset = 0);
    void copy_to_host_async(void* destination, std::size_t count, hipStream_t stream,
                            std::size_t byte_offset = 0) const;
    [[nodiscard]] void* data() const noexcept { return p; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes; }

    void* p = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;
    private:
        friend class DeviceArena;
        explicit Scope(DeviceArena& arena) noexcept;
        DeviceArena* arena_ = nullptr;
        std::size_t saved_offset_ = 0;
    };

    explicit DeviceArena(std::size_t capacity_bytes);
    explicit DeviceArena(DeviceSpan storage);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;
    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;
    [[nodiscard]] void* base() const noexcept { return base_; }
    [[nodiscard]] std::size_t used() const noexcept { return offset_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t peak_used() const noexcept { return peak_; }
    void reset_peak() noexcept { peak_ = offset_; }

private:
    void* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t offset_ = 0;
    std::size_t peak_ = 0;
    bool owns_ = true;
};

using WorkspaceArena = DeviceArena;

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes);
    ~PinnedHostBuffer();
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;
    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

class HostPinnedArena {
public:
    explicit HostPinnedArena(std::size_t capacity_bytes);
    ~HostPinnedArena();
    HostPinnedArena(const HostPinnedArena&) = delete;
    HostPinnedArena& operator=(const HostPinnedArena&) = delete;
    HostPinnedArena(HostPinnedArena&& other) noexcept;
    HostPinnedArena& operator=(HostPinnedArena&& other) noexcept;
    [[nodiscard]] void* try_alloc(std::size_t bytes, std::size_t align = 256);
    void free(void* block);
    [[nodiscard]] void* base() const noexcept { return base_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t used() const noexcept { return used_; }

private:
    struct Span { std::size_t offset = 0; std::size_t bytes = 0; };
    void insert_free(std::size_t offset, std::size_t bytes);
    void* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t used_ = 0;
    std::vector<Span> free_;
    std::unordered_map<void*, Span> live_;
};

} // namespace ninfer
