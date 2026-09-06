#pragma once

// Opt-in, eager-only diagnostic for localizing a represented-BF16 Text residual divergence.
// One selected logical column is staged after embedding and after each mixer/MLP boundary, then
// transferred to the host once. The synchronous final copy makes traced runs timing-ineligible.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/prefill_tail_trace_path.h"

#include <hip/hip_runtime.h>

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::layer_boundary_trace {

inline constexpr std::int32_t kHidden       = TextConfig::hidden;
inline constexpr std::int32_t kLayers       = TextConfig::layers;
inline constexpr std::size_t kSnapshots     = 1U + 2U * static_cast<std::size_t>(kLayers);
inline constexpr std::size_t kSnapshotBytes =
    static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
inline constexpr std::size_t kPayloadBytes  = kSnapshots * kSnapshotBytes;
inline constexpr std::size_t kMetadataBytes = 3U * sizeof(std::int32_t);

static_assert(std::endian::native == std::endian::little);
static_assert(kHidden == 5120 && kLayers == 64 && TextConfig::token_domain == 248077);

enum class Role : std::uint8_t {
    None,
    TargetOrdinary,
    TargetDFlash,
    TextFresh,
    TextAppend,
};

inline const char* role_name(Role role) {
    switch (role) {
    case Role::TargetOrdinary: return "target-ordinary-frontier130";
    case Role::TargetDFlash: return "target-dflash-frontier130-column0";
    case Role::TextFresh: return "text-fresh-frontier129-column128";
    case Role::TextAppend: return "text-append-frontier129-column0";
    case Role::None: break;
    }
    return "none";
}

inline Role requested_role() {
    static const Role role = [] {
        const char* value = std::getenv("NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE");
        if (value == nullptr) { return Role::None; }
        const std::string_view name(value);
        if (name == "target-ordinary-frontier130") { return Role::TargetOrdinary; }
        if (name == "target-dflash-frontier130-column0") { return Role::TargetDFlash; }
        if (name == "text-fresh-frontier129-column128") { return Role::TextFresh; }
        if (name == "text-append-frontier129-column0") { return Role::TextAppend; }
        throw std::invalid_argument("NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE is invalid");
    }();
    return role;
}

inline bool enabled() { return requested_role() != Role::None; }

inline const std::string& manifest_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline const std::string& sidecar_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline void require_eager(bool use_device_graph) {
    if (!enabled()) { return; }
    if (manifest_path().empty() || sidecar_path().empty()) {
        throw std::invalid_argument("layer boundary trace requires manifest and sidecar paths");
    }
    if (manifest_path() == sidecar_path()) {
        throw std::invalid_argument("layer boundary trace outputs must be distinct");
    }
    for (const std::string* path : {&manifest_path(), &sidecar_path()}) {
        if (!std::filesystem::path(*path).is_absolute() ||
            path->find_first_of("\"\\\n\r") != std::string::npos) {
            throw std::invalid_argument(
                "layer boundary trace paths must be absolute JSON-safe paths");
        }
    }
    if (use_device_graph) {
        throw std::invalid_argument("layer boundary trace requires eager execution");
    }
}

inline std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t size) {
    std::uint64_t value = UINT64_C(14695981039346656037);
    for (std::size_t i = 0; i < size; ++i) {
        value ^= data[i];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

inline void write_exclusive(const std::string& path, const void* data, std::size_t bytes,
                            const char* label) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) {
        throw std::runtime_error(std::string("layer boundary trace refuses ") + label +
                                 " output: " + std::strerror(errno));
    }
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t remaining = bytes;
    while (remaining != 0U) {
        const ssize_t written = ::write(descriptor, cursor, remaining);
        if (written <= 0) {
            const int saved = errno;
            (void)::close(descriptor);
            throw std::runtime_error(std::string("layer boundary trace ") + label +
                                     " write failed: " + std::strerror(saved));
        }
        cursor += written;
        remaining -= static_cast<std::size_t>(written);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error(std::string("layer boundary trace ") + label +
                                 " close failed");
    }
}

struct Call {
    Role role               = Role::None;
    std::int32_t width      = 0;
    std::int32_t batch      = 0;
    std::int32_t column     = 0;
    std::int32_t frontier   = 0;
    const Tensor* ids       = nullptr;
    const Tensor* positions = nullptr;
    const Tensor* rope      = nullptr;
};

inline bool& completed() {
    static bool value = false;
    return value;
}

inline bool matches(Role role) { return enabled() && !completed() && requested_role() == role; }

class Session {
public:
    Session(Call call, hipStream_t stream)
        : call_(call), stream_(stream), device_(kPayloadBytes + kMetadataBytes),
          host_(kPayloadBytes + kMetadataBytes) {
        require_call();
        ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
            manifest_path().c_str());
        ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
            sidecar_path().c_str());
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        HIP_CHECK(hipStreamIsCapturing(stream_, &capture));
        if (capture != hipStreamCaptureStatusNone) {
            throw std::invalid_argument("layer boundary trace cannot run during graph capture");
        }
        copy_scalar(*call_.ids, 0U);
        copy_scalar(*call_.positions, sizeof(std::int32_t));
        copy_scalar(*call_.rope, 2U * sizeof(std::int32_t));
    }

    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    void begin(const Tensor& value) { capture(value, 0U); }

    void capture_mixer(int layer, const Tensor& value, hipStream_t) {
        require_layer(layer);
        capture(value, 1U + 2U * static_cast<std::size_t>(layer));
    }

    void capture_layer(int layer, const Tensor& value, hipStream_t) {
        require_layer(layer);
        capture(value, 2U + 2U * static_cast<std::size_t>(layer));
    }

    void finish() {
        if (next_snapshot_ != kSnapshots) {
            throw std::logic_error("layer boundary trace did not capture every boundary");
        }
        device_.copy_to_host_async(host_.data(), host_.size(), stream_);
        HIP_CHECK(hipStreamSynchronize(stream_));
        std::array<std::int32_t, 3> metadata{};
        std::memcpy(metadata.data(), host_.data() + kPayloadBytes, kMetadataBytes);
        validate_metadata(metadata[0], metadata[1], metadata[2]);
        write_outputs(metadata[0], metadata[1], metadata[2]);
        completed() = true;
    }

private:
    void require_call() const {
        if (call_.role == Role::None || call_.role != requested_role() || call_.batch != 1 ||
            call_.column < 0 || call_.column >= call_.width || call_.ids == nullptr ||
            call_.positions == nullptr || call_.rope == nullptr) {
            throw std::logic_error("layer boundary trace call identity is invalid");
        }
        const bool target_ordinary = call_.role == Role::TargetOrdinary && call_.width == 1 &&
                                     call_.column == 0 && call_.frontier == 130;
        const bool target_dflash = call_.role == Role::TargetDFlash && call_.width == 5 &&
                                   call_.column == 0 && call_.frontier == 130;
        const bool text_fresh = call_.role == Role::TextFresh && call_.width == 129 &&
                                call_.column == 128 && call_.frontier == 129;
        const bool text_append = call_.role == Role::TextAppend && call_.width == 1 &&
                                 call_.column == 0 && call_.frontier == 129;
        if (!(target_ordinary || target_dflash || text_fresh || text_append)) {
            throw std::logic_error("layer boundary trace shape is outside its exact diagnostic");
        }
        for (const Tensor* tensor : {call_.ids, call_.positions, call_.rope}) {
            if (tensor->dtype != DType::I32 || tensor->data == nullptr ||
                !tensor->is_contiguous() || tensor->numel() < call_.width) {
                throw std::logic_error("layer boundary trace metadata tensor is invalid");
            }
        }
    }

    void copy_scalar(const Tensor& source, std::size_t metadata_offset) {
        const auto* address = static_cast<const std::uint8_t*>(source.data) +
                              static_cast<std::size_t>(call_.column) * sizeof(std::int32_t);
        HIP_CHECK(hipMemcpyAsync(static_cast<std::uint8_t*>(device_.data()) + kPayloadBytes +
                                     metadata_offset,
                                 address, sizeof(std::int32_t), hipMemcpyDeviceToDevice, stream_));
    }

    void require_layer(int layer) const {
        if (layer < 0 || layer >= kLayers) {
            throw std::logic_error("layer boundary trace layer is invalid");
        }
    }

    void capture(const Tensor& value, std::size_t snapshot) {
        if (snapshot != next_snapshot_ || value.dtype != DType::BF16 || value.data == nullptr ||
            !value.is_contiguous() || value.ne[0] != kHidden || value.ne[1] < call_.width ||
            value.ne[2] != 1 || value.ne[3] != 1) {
            throw std::logic_error("layer boundary trace residual tensor/order is invalid");
        }
        const auto* source = static_cast<const std::uint8_t*>(value.data) +
                             static_cast<std::size_t>(call_.column) * kSnapshotBytes;
        HIP_CHECK(hipMemcpyAsync(static_cast<std::uint8_t*>(device_.data()) +
                                     snapshot * kSnapshotBytes,
                                 source, kSnapshotBytes, hipMemcpyDeviceToDevice, stream_));
        ++next_snapshot_;
    }

    void validate_metadata(std::int32_t token, std::int32_t position,
                           std::int32_t rope_position) const {
        if (token < 0 || token >= TextConfig::token_domain || position + 1 != call_.frontier) {
            throw std::logic_error("layer boundary trace token/frontier differs");
        }
        if (rope_position != position) {
            throw std::logic_error("layer boundary trace requires the exact text-only RoPE position");
        }
    }

    void write_outputs(std::int32_t token, std::int32_t position,
                       std::int32_t rope_position) const {
        write_exclusive(sidecar_path(), host_.data(), kPayloadBytes, "sidecar");

        const std::uint64_t hash = fnv1a64(host_.data(), kPayloadBytes);
        std::ostringstream manifest;
        manifest << "{\n"
                 << "  \"artifact_type\": \"ninfer_qwen3_layer_boundary_trace\",\n"
                 << "  \"schema_version\": 1,\n"
                 << "  \"diagnostic_only\": true,\n"
                 << "  \"timing_evidence_eligible\": false,\n"
                 << "  \"production_routing_authorized\": false,\n"
                 << "  \"execution\": \"eager\",\n"
                 << "  \"role\": \"" << role_name(call_.role) << "\",\n"
                 << "  \"width\": " << call_.width << ",\n"
                 << "  \"selected_column\": " << call_.column << ",\n"
                 << "  \"absolute_frontier\": " << call_.frontier << ",\n"
                 << "  \"token\": " << token << ",\n"
                 << "  \"cache_position\": " << position << ",\n"
                 << "  \"rope_position\": " << rope_position << ",\n"
                 << "  \"hidden\": 5120,\n"
                 << "  \"layers\": 64,\n"
                 << "  \"snapshot_count\": 129,\n"
                 << "  \"snapshot_bytes\": 10240,\n"
                 << "  \"sidecar_path\": \"" << sidecar_path() << "\",\n"
                 << "  \"sidecar_bytes\": " << kPayloadBytes << ",\n"
                 << "  \"sidecar_fnv1a64\": \"" << std::hex << std::setfill('0')
                 << std::setw(16) << hash << std::dec << "\",\n"
                 << "  \"layout\": \"little-endian-u16: input, then layer0..63 post_mixer,post_mlp\"\n"
                 << "}\n";
        const std::string contents = manifest.str();
        write_exclusive(manifest_path(), contents.data(), contents.size(), "manifest");
    }

    Call call_;
    hipStream_t stream_ = nullptr;
    DeviceBuffer device_;
    std::vector<std::uint8_t> host_;
    std::size_t next_snapshot_ = 0;
};

template <class Primary>
struct CompositeTap {
    static constexpr bool enabled = true;
    Primary& primary;
    Session& trace;

    void begin(const Tensor& value) {
        if constexpr (requires { primary.begin(value); }) { primary.begin(value); }
        trace.begin(value);
    }
    void capture_mixer(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_mixer(layer, value, stream);
    }
    void capture_layer(int layer, const Tensor& value, hipStream_t stream) {
        if constexpr (requires { primary.capture_layer(layer, value, stream); }) {
            primary.capture_layer(layer, value, stream);
        }
        trace.capture_layer(layer, value, stream);
    }
    void capture_positions(const Tensor& value, hipStream_t stream) {
        if constexpr (requires { primary.capture_positions(value, stream); }) {
            primary.capture_positions(value, stream);
        }
    }
};

struct Tap {
    static constexpr bool enabled = true;
    Session& trace;
    void begin(const Tensor& value) { trace.begin(value); }
    void capture_mixer(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_mixer(layer, value, stream);
    }
    void capture_layer(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_layer(layer, value, stream);
    }
};

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::layer_boundary_trace
