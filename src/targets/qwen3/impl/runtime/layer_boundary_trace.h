#pragma once

// Opt-in, eager-only diagnostic for localizing a represented-BF16 Text residual divergence.
// One selected logical column is staged after embedding and after each mixer/MLP boundary, then
// transferred to the host once. The synchronous final copy makes traced runs timing-ineligible.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ops/r9700/kv/fp8_int4_kv_trace.h"
#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/prefill_tail_trace_path.h"

#include <hip/hip_runtime.h>

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <optional>
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

struct GdnField {
    const char* name;
    DType dtype;
    std::size_t elements;
    std::size_t offset;
};
inline constexpr std::array<GdnField, 10> kGdnFields{{
    {"h", DType::BF16, 5120U, 0U},
    {"g", DType::FP32, 48U, 10240U},
    {"beta", DType::FP32, 48U, 10432U},
    {"z", DType::BF16, 6144U, 10624U},
    {"q", DType::BF16, 2048U, 22912U},
    {"k", DType::BF16, 2048U, 27008U},
    {"v", DType::BF16, 6144U, 31104U},
    {"o", DType::BF16, 6144U, 43392U},
    {"on", DType::BF16, 6144U, 55680U},
    {"x", DType::BF16, 5120U, 67968U},
}};
inline constexpr std::size_t kGdnPayloadBytes = 78208U;
inline constexpr std::size_t kCombinedPayloadBytes = kPayloadBytes + kGdnPayloadBytes;
inline constexpr std::size_t kRecurrentStateElements = 128U * 128U * 48U;
inline constexpr std::size_t kRecurrentStateBytes = kRecurrentStateElements * sizeof(float);
inline constexpr std::size_t kAttentionStageBytes = 137216U;
inline constexpr std::size_t kAttentionCacheBytes = 216320U;
inline constexpr std::size_t kAttentionPayloadBytes = kAttentionStageBytes + kAttentionCacheBytes;
inline constexpr std::size_t kAttentionAuxBytes = 7U * sizeof(std::int32_t);

struct AttentionField {
    const char* name;
    DType dtype;
    std::size_t elements;
    std::size_t offset;
};
inline constexpr std::array<AttentionField, 12> kAttentionFields{{
    {"input_x", DType::BF16, 5120U, 0U},
    {"norm_h", DType::BF16, 5120U, 10240U},
    {"projection_q", DType::BF16, 6144U, 20480U},
    {"projection_gate", DType::BF16, 6144U, 32768U},
    {"projection_k", DType::BF16, 1024U, 45056U},
    {"projection_v", DType::BF16, 1024U, 47104U},
    {"normalized_qk", DType::BF16, 7168U, 49152U},
    {"rope_qk", DType::BF16, 7168U, 63488U},
    {"attention_fp32", DType::FP32, 6144U, 77824U},
    {"attention_bf16", DType::BF16, 6144U, 102400U},
    {"gated_attention", DType::BF16, 6144U, 114688U},
    {"residual_x", DType::BF16, 5120U, 126976U},
}};

consteval bool valid_gdn_layout() {
    std::size_t offset = 0;
    for (const GdnField& field : kGdnFields) {
        if (field.offset != offset) return false;
        const std::size_t element_bytes =
            field.dtype == DType::BF16 ? 2U : (field.dtype == DType::FP32 ? 4U : 0U);
        if (element_bytes == 0U) return false;
        offset += field.elements * element_bytes;
    }
    return offset == kGdnPayloadBytes;
}

consteval bool valid_attention_layout() {
    std::size_t offset = 0;
    for (const AttentionField& field : kAttentionFields) {
        if (field.offset != offset) return false;
        const std::size_t element_bytes =
            field.dtype == DType::BF16 ? 2U : (field.dtype == DType::FP32 ? 4U : 0U);
        if (element_bytes == 0U) return false;
        offset += field.elements * element_bytes;
    }
    return offset == kAttentionStageBytes &&
           kAttentionStageBytes + 130U * 4U * 256U + 130U * 4U * 128U +
                   130U * 4U * 16U * sizeof(std::uint16_t) ==
               kAttentionPayloadBytes;
}

static_assert(std::endian::native == std::endian::little);
static_assert(kHidden == 5120 && kLayers == 64 && TextConfig::token_domain == 248077);
static_assert(valid_gdn_layout());
static_assert(valid_attention_layout());

enum class Role : std::uint8_t {
    None,
    TargetOrdinary,
    TargetDFlash,
    SelectedOrdinary,
    SelectedDFlash,
    TextFresh,
    TextAppend,
};

inline const char* role_name(Role role) {
    switch (role) {
    case Role::TargetOrdinary: return "target-ordinary-frontier130";
    case Role::TargetDFlash: return "target-dflash-frontier130-column0";
    case Role::TextFresh: return "text-fresh-frontier129-column128";
    case Role::TextAppend: return "text-append-frontier129-column0";
    case Role::SelectedOrdinary: return "target-ordinary-selected";
    case Role::SelectedDFlash: return "target-dflash-selected";
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
        if (name == "target-ordinary-selected") return Role::SelectedOrdinary;
        if (name == "target-dflash-selected") return Role::SelectedDFlash;
        throw std::invalid_argument("NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE is invalid");
    }();
    return role;
}

inline bool selected_role(Role role) {
    return role == Role::SelectedOrdinary || role == Role::SelectedDFlash;
}
inline int selected_integer(const char* suffix, int minimum, int maximum) {
    const std::string name = std::string("NINFER_QWEN3_LAYER_BOUNDARY_TRACE_") + suffix;
    const char* value = std::getenv(name.c_str());
    if (!value || !*value) throw std::invalid_argument(name + " is required");
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno || *end || parsed < minimum || parsed > maximum)
        throw std::invalid_argument(name + " is out of range");
    return static_cast<int>(parsed);
}
inline std::string selected_history() {
    const char* value = std::getenv("NINFER_QWEN3_LAYER_BOUNDARY_TRACE_HISTORY_SHA256");
    if (!value || std::strlen(value) != 64 ||
        std::strspn(value, "0123456789abcdef") != 64)
        throw std::invalid_argument("selected trace requires represented prefix HISTORY_SHA256");
    return value;
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

inline const std::string& gdn_manifest_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline const std::string& gdn_sidecar_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline bool gdn_enabled() {
    return !gdn_manifest_path().empty() || !gdn_sidecar_path().empty();
}

inline int gdn_detail_layer() {
    if (!selected_role(requested_role())) return 1;
    static const int layer = [] {
        const char* value = std::getenv("NINFER_QWEN3_GDN_DETAIL_TRACE_TEXT_LAYER");
        if (value == nullptr) return 1;
        char* end = nullptr;
        errno = 0;
        const long parsed = std::strtol(value, &end, 10);
        if (errno || end == value || *end || parsed < 0 || parsed >= kLayers || parsed % 4 == 3)
            throw std::invalid_argument("GDN detail text layer must be a GDN layer in 0..63");
        return static_cast<int>(parsed);
    }();
    return layer;
}

inline const std::string& recurrent_state_manifest_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_GDN_STATE_TRACE_MANIFEST");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline const std::string& recurrent_state_sidecar_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_GDN_STATE_TRACE_SIDECAR");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline const std::string& attention_manifest_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_LAYER3_ATTENTION_TRACE_MANIFEST");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline const std::string& attention_sidecar_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_QWEN3_LAYER3_ATTENTION_TRACE_SIDECAR");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline bool attention_enabled() {
    return !attention_manifest_path().empty() || !attention_sidecar_path().empty();
}

inline bool recurrent_state_enabled() {
    return !recurrent_state_manifest_path().empty() || !recurrent_state_sidecar_path().empty();
}

inline int recurrent_state_layer() {
    if (!recurrent_state_enabled()) return -1;
    const char* value = std::getenv("NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER");
    if (value == nullptr || (std::string_view(value) != "0" && std::string_view(value) != "1")) {
        throw std::invalid_argument(
            "GDN recurrent-state trace text layer must be exactly 0 or 1");
    }
    return value[0] - '0';
}

inline void require_eager(bool use_device_graph) {
    if (!enabled()) {
        if (gdn_enabled() || recurrent_state_enabled() || attention_enabled()) {
            throw std::invalid_argument(
                "GDN diagnostic traces require the layer boundary trace");
        }
        return;
    }
    if (manifest_path().empty() || sidecar_path().empty()) {
        throw std::invalid_argument("layer boundary trace requires manifest and sidecar paths");
    }
    if (manifest_path() == sidecar_path()) {
        throw std::invalid_argument("layer boundary trace outputs must be distinct");
    }
    if (gdn_enabled() && (gdn_manifest_path().empty() || gdn_sidecar_path().empty())) {
        throw std::invalid_argument("GDN detail trace requires both manifest and sidecar paths");
    }
    if (gdn_enabled()) (void)gdn_detail_layer();
    if (recurrent_state_enabled() &&
        (recurrent_state_manifest_path().empty() || recurrent_state_sidecar_path().empty())) {
        throw std::invalid_argument(
            "GDN recurrent-state trace requires both manifest and sidecar paths");
    }
    if (recurrent_state_enabled()) (void)recurrent_state_layer();
    if (attention_enabled() &&
        (attention_manifest_path().empty() || attention_sidecar_path().empty())) {
        throw std::invalid_argument("layer3 attention trace requires manifest and sidecar paths");
    }
    if (attention_enabled() && requested_role() != Role::TargetOrdinary &&
        requested_role() != Role::TargetDFlash && !selected_role(requested_role())) {
        throw std::invalid_argument("layer3 attention trace supports only the exact target pair");
    }
    if (recurrent_state_enabled() &&
        requested_role() != Role::TextFresh && requested_role() != Role::TextAppend) {
        throw std::invalid_argument("GDN recurrent-state trace supports only the exact Text pair");
    }
    if (gdn_enabled() && (gdn_manifest_path() == gdn_sidecar_path() ||
                          gdn_manifest_path() == manifest_path() ||
                          gdn_manifest_path() == sidecar_path() ||
                          gdn_sidecar_path() == manifest_path() ||
                          gdn_sidecar_path() == sidecar_path())) {
        throw std::invalid_argument("layer and GDN trace outputs must be distinct");
    }
    const std::array<const std::string*, 8> paths{
        &manifest_path(), &sidecar_path(), &gdn_manifest_path(), &gdn_sidecar_path(),
        &recurrent_state_manifest_path(), &recurrent_state_sidecar_path(),
        &attention_manifest_path(), &attention_sidecar_path()};
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const std::string* path = paths[i];
        if (path->empty()) continue;
        if (!std::filesystem::path(*path).is_absolute() ||
            path->find_first_of("\"\\\n\r") != std::string::npos) {
            throw std::invalid_argument(
                "layer boundary trace paths must be absolute JSON-safe paths");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (!paths[j]->empty() && *path == *paths[j]) {
                throw std::invalid_argument("layer boundary trace outputs must be distinct");
            }
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
    std::int32_t row        = 0;
    std::int32_t lane       = 0;
    std::int32_t base_frontier = 0;
    const Tensor* ids       = nullptr;
    const Tensor* positions = nullptr;
    const Tensor* rope      = nullptr;
};

inline bool& completed() {
    static bool value = false;
    return value;
}

inline bool matches(Role role) { return enabled() && !completed() && requested_role() == role; }

// Selection reads metadata only: model execution stays on its original batched path.
inline std::optional<Call> select_target_call(
    Role legacy, int width, int batch, const Tensor& ids, const Tensor& positions,
    const Tensor& rope, const Tensor& state_slots, hipStream_t stream,
    const Tensor* valid_columns = nullptr) {
    const Role wanted = legacy == Role::TargetOrdinary ? Role::SelectedOrdinary : Role::SelectedDFlash;
    if (matches(legacy))
        return Call{.role=legacy,.width=width,.batch=batch,.column=0,.frontier=130,
                    .ids=&ids,.positions=&positions,.rope=&rope};
    if (!matches(wanted)) return std::nullopt;
    const int selected_batch = selected_integer("BATCH",1,4);
    const int lane = selected_integer("LANE",0,3);
    const int frontier = selected_integer("FRONTIER",1,2147483647);
    const int column = selected_integer("COLUMN",-1,5);
    const int expected_token = selected_integer("TOKEN",0,TextConfig::token_domain-1);
    (void)selected_history();
    if (batch != selected_batch) return std::nullopt;
    hipStreamCaptureStatus capture{};
    HIP_CHECK(hipStreamIsCapturing(stream,&capture));
    if (capture != hipStreamCaptureStatusNone)
        throw std::invalid_argument("selected boundary trace is eager-only");
    std::array<int,4> slots{}, valid{};
    std::array<int,24> pos{};
    HIP_CHECK(hipMemcpyAsync(slots.data(),state_slots.data,batch*sizeof(int),hipMemcpyDeviceToHost,stream));
    HIP_CHECK(hipMemcpyAsync(pos.data(),positions.data,batch*width*sizeof(int),hipMemcpyDeviceToHost,stream));
    if (valid_columns)
        HIP_CHECK(hipMemcpyAsync(valid.data(),valid_columns->data,batch*sizeof(int),hipMemcpyDeviceToHost,stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (int row=0;row<batch;++row) {
        // current_state_slot(lane,max_concurrency) is exactly the stable lane.
        if (slots[row] != lane) continue;
        const int count = valid_columns ? valid[row] : width;
        if (count<0 || count>width) throw std::logic_error("selected trace valid width is invalid");
        for(int col=0;col<count;++col) {
            if ((column>=0 && col!=column) || pos[row*width+col]!=frontier-1) continue;
            int token=-1;
            HIP_CHECK(hipMemcpyAsync(&token,static_cast<const int*>(ids.data)+row*width+col,
                                    sizeof(int),hipMemcpyDeviceToHost,stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            if(token!=expected_token)
                throw std::logic_error("selected trace consumed token differs from expected common prefix");
            return Call{.role=wanted,.width=width,.batch=batch,.column=col,.frontier=frontier,
                        .row=row,.lane=lane,.base_frontier=pos[row*width],
                        .ids=&ids,.positions=&positions,.rope=&rope};
        }
    }
    return std::nullopt;
}

struct RecurrentStateTrace {
    Tensor destination;
    std::int32_t prefix_tokens = 0;
};

class Session {
public:
    Session(Call call, hipStream_t stream)
        : call_(call), stream_(stream),
          device_(kCombinedPayloadBytes +
                  (recurrent_state_enabled() ? kRecurrentStateBytes : 0U) +
                  (attention_enabled() ? kAttentionPayloadBytes + kAttentionAuxBytes : 0U) +
                  kMetadataBytes),
          host_(kCombinedPayloadBytes +
                (recurrent_state_enabled() ? kRecurrentStateBytes : 0U) +
                (attention_enabled() ? kAttentionPayloadBytes + kAttentionAuxBytes : 0U) +
                kMetadataBytes) {
        require_call();
        ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
            manifest_path().c_str());
        ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
            sidecar_path().c_str());
        if (gdn_enabled()) {
            ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                gdn_manifest_path().c_str());
            ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                gdn_sidecar_path().c_str());
        }
        if (recurrent_state_enabled()) {
            ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                recurrent_state_manifest_path().c_str());
            ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                recurrent_state_sidecar_path().c_str());
        }
        if (attention_enabled()) {
            ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                attention_manifest_path().c_str());
            ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                attention_sidecar_path().c_str());
            HIP_CHECK(hipMemsetAsync(attention_aux(), 0, kAttentionAuxBytes, stream_));
            for (const char* name : {"NINFER_QWEN3_LAYER3_TRACE_SOURCE_COMMIT",
                                     "NINFER_QWEN3_LAYER3_TRACE_SOURCE_TREE",
                                     "NINFER_QWEN3_LAYER3_TRACE_EXECUTABLE_SHA256",
                                     "NINFER_QWEN3_LAYER3_TRACE_ARTIFACT_SHA256",
                                     "NINFER_QWEN3_LAYER3_TRACE_HISTORY_SHA256",
                                     "NINFER_QWEN3_LAYER3_TRACE_CORPUS_SHA256",
                                     "NINFER_QWEN3_LAYER3_TRACE_POWER_SOURCE"}) {
                if (!selected_role(call_.role)) (void)required_provenance(name);
            }
        }
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

    void capture_gdn_controls(int layer, const Tensor& h, const Tensor& g,
                              const Tensor& beta, hipStream_t) {
        if (!gdn_enabled() || layer != gdn_detail_layer()) return;
        capture_gdn(h, "h");
        capture_gdn(g, "g");
        capture_gdn(beta, "beta");
    }

    void capture_gdn_projection(int layer, const Tensor& z, const Tensor& q, const Tensor& k,
                                const Tensor& v, hipStream_t) {
        if (!gdn_enabled() || layer != gdn_detail_layer()) return;
        capture_gdn(z, "z");
        capture_gdn(q, "q");
        capture_gdn(k, "k");
        capture_gdn(v, "v");
    }

    void capture_gdn_recurrence(int layer, const Tensor& value, hipStream_t) {
        if (gdn_enabled() && layer == gdn_detail_layer()) capture_gdn(value, "o");
    }

    void capture_gdn_normalized(int layer, const Tensor& value, hipStream_t) {
        if (gdn_enabled() && layer == gdn_detail_layer()) capture_gdn(value, "on");
    }

    void capture_gdn_residual(int layer, const Tensor& value, hipStream_t) {
        if (gdn_enabled() && layer == gdn_detail_layer()) capture_gdn(value, "x");
    }

    void capture_attention_stage(int layer, std::string_view name, const Tensor& value,
                                 hipStream_t) {
        if (layer != 3 || !attention_enabled()) return;
        capture_attention(value, name);
    }

    void capture_attention_qk_stage(int layer, std::string_view name, const Tensor& q,
                                    const Tensor& k, hipStream_t) {
        if (layer != 3 || !attention_enabled()) return;
        if (next_attention_field_ >= kAttentionFields.size() ||
            name != kAttentionFields[next_attention_field_].name ||
            kAttentionFields[next_attention_field_].elements != 7168U) {
            throw std::logic_error("layer3 attention q/k field order differs");
        }
        capture_attention_part(q, 6144U, kAttentionFields[next_attention_field_].offset);
        capture_attention_part(k, 1024U,
                               kAttentionFields[next_attention_field_].offset + 12288U);
        ++next_attention_field_;
    }

    void capture_attention_cache(int layer, const qwen3::PagedKVLayerRead& read,
                                 const Tensor* ancestor_masks, const Tensor* prefix_lengths,
                                 hipStream_t) {
        if (layer != 3 || !attention_enabled()) return;
        // Selected traces capture actual batched stages, not the legacy fixed
        // positions0..129 cache diagnostic. This callback runs once per lane.
        if (selected_role(call_.role)) return;
        if (attention_cache_scheduled_ || next_attention_field_ != 8U ||
            !read.valid_for(stream_) || !read.pending()) {
            throw std::logic_error("layer3 attention cache capture order/read differs");
        }
        const bool dflash = call_.role == Role::TargetDFlash;
        // This diagnostic binds the delivered K4/W5 chain verifier. W == K + 1
        // deliberately uses ordinary causal attention and therefore owns no tree metadata.
        if (ancestor_masks != nullptr || prefix_lengths != nullptr) {
            throw std::logic_error("layer3 attention visibility metadata differs");
        }
        const auto& cache = read.layer();
        if (cache.head_dim != 256 || cache.num_kv_heads != 4 || cache.value_group != 16 ||
            read.visible_frontier() != (dflash ? 134U : 130U) || read.mapped_pages() < 3U ||
            read.device_table_row() == nullptr || read.pool_block_tables() == nullptr ||
            read.pool_table_row_stride() < read.mapped_pages() ||
            read.pool_table_row_count() == 0U) {
            throw std::logic_error("layer3 attention cache identity differs");
        }
        if (cache.plane_layouts.key != Fp8KInt4VPlaneLayout::TokenFastestHeadMajor ||
            cache.plane_layouts.value != Fp8KInt4VPlaneLayout::FeatureFastestPageMajor ||
            cache.plane_layouts.value_scale != Fp8KInt4VPlaneLayout::FeatureFastestPageMajor) {
            throw std::logic_error("layer3 attention cache layout differs");
        }
        const std::size_t physical_tokens =
            static_cast<std::size_t>(cache.key_fp8_pages.numel()) / (4U * 256U);
        auto* payload = attention_payload();
        auto* status = reinterpret_cast<std::uint32_t*>(attention_aux() + 24U);
        HIP_CHECK(ops::r9700::kv::fp8_int4_kv_trace_positions0_129(
            {.key_codes = static_cast<const std::uint8_t*>(cache.key_fp8_pages.data),
             .value_codes = static_cast<const std::uint8_t*>(cache.value_int4_pages.data),
             .value_scale_bits =
                 static_cast<const std::uint16_t*>(cache.value_scale_fp16_pages.data),
             .page_table = read.pool_block_tables(),
             .page_table_row = read.device_table_row(),
             .page_table_pages = read.mapped_pages(),
             .page_table_row_stride = read.pool_table_row_stride(),
             .page_table_row_count = read.pool_table_row_count(),
             .physical_tokens = physical_tokens,
             .canonical_keys = payload + kAttentionStageBytes,
             .canonical_values = payload + kAttentionStageBytes + 133120U,
             .canonical_scales = reinterpret_cast<std::uint16_t*>(
                 payload + kAttentionStageBytes + 199680U),
             .status = status},
            stream_));
        attention_visible_frontier_ = read.visible_frontier();
        attention_mapped_pages_ = read.mapped_pages();
        attention_table_stride_ = read.pool_table_row_stride();
        attention_table_rows_ = read.pool_table_row_count();
        attention_cache_scheduled_ = true;
    }

    RecurrentStateTrace capture_gdn_recurrent_state(int layer, const Tensor& state,
                                                     std::int32_t slot, hipStream_t) {
        if (!recurrent_state_enabled() || layer != recurrent_state_layer()) return {};
        if (recurrent_state_scheduled_ || slot != 0 || state.dtype != DType::FP32 ||
            state.data == nullptr || !state.is_contiguous() ||
            state.ne[0] != 128 || state.ne[1] != 128 || state.ne[2] != 48 || state.ne[3] != 1) {
            throw std::logic_error("GDN recurrent-state trace tensor/slot differs");
        }
        recurrent_state_scheduled_ = true;
        Tensor destination(static_cast<std::uint8_t*>(device_.data()) + kCombinedPayloadBytes,
                           DType::FP32, {128, 128, 48});
        if (call_.role == Role::TextAppend) {
            HIP_CHECK(hipMemcpyAsync(destination.data, state.data, kRecurrentStateBytes,
                                     hipMemcpyDeviceToDevice, stream_));
            return {};
        }
        if (call_.role != Role::TextFresh || call_.width != 129 || call_.column != 128) {
            throw std::logic_error("GDN recurrent-state trace role differs");
        }
        return {.destination = destination, .prefix_tokens = 128};
    }

    void finish() {
        if (next_snapshot_ != kSnapshots) {
            throw std::logic_error("layer boundary trace did not capture every boundary");
        }
        if (gdn_enabled() && next_gdn_field_ != kGdnFields.size()) {
            throw std::logic_error("GDN detail trace did not capture every layer-1 boundary");
        }
        if (recurrent_state_enabled() && !recurrent_state_scheduled_) {
            throw std::logic_error("GDN recurrent-state trace was not scheduled");
        }
        if (attention_enabled() &&
            (next_attention_field_ != kAttentionFields.size() ||
             (!selected_role(call_.role) && !attention_cache_scheduled_))) {
            throw std::logic_error("layer3 attention trace did not capture every boundary");
        }
        device_.copy_to_host_async(host_.data(), host_.size(), stream_);
        HIP_CHECK(hipStreamSynchronize(stream_));
        std::array<std::int32_t, 3> metadata{};
        const std::size_t metadata_offset =
            kCombinedPayloadBytes + (recurrent_state_enabled() ? kRecurrentStateBytes : 0U) +
            (attention_enabled() ? kAttentionPayloadBytes + kAttentionAuxBytes : 0U);
        std::memcpy(metadata.data(), host_.data() + metadata_offset, kMetadataBytes);
        validate_metadata(metadata[0], metadata[1], metadata[2]);
        write_outputs(metadata[0], metadata[1], metadata[2]);
        if (gdn_enabled()) write_gdn_outputs(metadata[0], metadata[1], metadata[2]);
        if (recurrent_state_enabled()) {
            write_recurrent_state_outputs(metadata[0], metadata[1], metadata[2]);
        }
        if (attention_enabled()) {
            write_attention_outputs(metadata[0], metadata[1], metadata[2]);
        }
        completed() = true;
    }

private:
    std::string selected_manifest_fields() const {
        if (!selected_role(call_.role)) return {};
        std::ostringstream out;
        out << "  \"batch\": " << call_.batch << ",\n"
            << "  \"selected_row\": " << call_.row << ",\n"
            << "  \"stable_lane\": " << call_.lane << ",\n"
            << "  \"flat_column\": " << call_.row*call_.width+call_.column << ",\n"
            << "  \"base_frontier\": " << call_.base_frontier << ",\n"
            << "  \"expected_history_sha256\": \"" << selected_history() << "\",\n";
        return out.str();
    }
    void require_call() const {
        if (call_.role == Role::None || call_.role != requested_role() || call_.batch < 1 || call_.batch > 4 ||
            call_.row < 0 || call_.row >= call_.batch ||
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
        const bool selected = selected_role(call_.role) && call_.width >= 1 && call_.width <= 6 &&
            (call_.role != Role::SelectedOrdinary || call_.width == 1);
        if ((!selected && call_.batch != 1) ||
            !(selected || target_ordinary || target_dflash || text_fresh || text_append)) {
            throw std::logic_error("layer boundary trace shape is outside its exact diagnostic");
        }
        for (const Tensor* tensor : {call_.ids, call_.positions, call_.rope}) {
            if (tensor->dtype != DType::I32 || tensor->data == nullptr ||
                !tensor->is_contiguous() || tensor->numel() < call_.width * call_.batch) {
                throw std::logic_error("layer boundary trace metadata tensor is invalid");
            }
        }
    }

    void copy_scalar(const Tensor& source, std::size_t metadata_offset) {
        const auto* address = static_cast<const std::uint8_t*>(source.data) +
                              static_cast<std::size_t>(call_.row * call_.width + call_.column) * sizeof(std::int32_t);
        HIP_CHECK(hipMemcpyAsync(static_cast<std::uint8_t*>(device_.data()) + kPayloadBytes +
                                     kGdnPayloadBytes +
                                     (recurrent_state_enabled() ? kRecurrentStateBytes : 0U) +
                                     (attention_enabled()
                                          ? kAttentionPayloadBytes + kAttentionAuxBytes
                                          : 0U) +
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
            !value.is_contiguous() || value.ne[0] != kHidden || value.ne[1] < call_.width * call_.batch ||
            value.ne[2] != 1 || value.ne[3] != 1) {
            throw std::logic_error("layer boundary trace residual tensor/order is invalid");
        }
        const auto* source = static_cast<const std::uint8_t*>(value.data) +
                             static_cast<std::size_t>(call_.row * call_.width + call_.column) * kSnapshotBytes;
        HIP_CHECK(hipMemcpyAsync(static_cast<std::uint8_t*>(device_.data()) +
                                     snapshot * kSnapshotBytes,
                                 source, kSnapshotBytes, hipMemcpyDeviceToDevice, stream_));
        ++next_snapshot_;
    }

    void capture_gdn(const Tensor& value, std::string_view expected_name) {
        if (next_gdn_field_ >= kGdnFields.size()) {
            throw std::logic_error("too many GDN detail fields");
        }
        const GdnField& field = kGdnFields[next_gdn_field_];
        if (expected_name != field.name || value.dtype != field.dtype || value.data == nullptr ||
            !value.is_contiguous() || value.numel() !=
                static_cast<std::int64_t>(field.elements * static_cast<std::size_t>(call_.width * call_.batch))) {
            throw std::logic_error("GDN detail tensor/order is invalid");
        }
        const std::size_t element_bytes = dtype_size(field.dtype);
        const auto* source = static_cast<const std::uint8_t*>(value.data) +
                             static_cast<std::size_t>(call_.row * call_.width + call_.column) * field.elements * element_bytes;
        HIP_CHECK(hipMemcpyAsync(static_cast<std::uint8_t*>(device_.data()) + kPayloadBytes +
                                     field.offset,
                                 source, field.elements * element_bytes,
                                 hipMemcpyDeviceToDevice, stream_));
        ++next_gdn_field_;
    }

    std::size_t attention_offset() const {
        return kCombinedPayloadBytes +
               (recurrent_state_enabled() ? kRecurrentStateBytes : 0U);
    }

    std::uint8_t* attention_payload() {
        return static_cast<std::uint8_t*>(device_.data()) + attention_offset();
    }

    std::uint8_t* attention_aux() {
        return attention_payload() + kAttentionPayloadBytes;
    }

    void capture_attention_part(const Tensor& value, std::size_t elements,
                                std::size_t destination_offset) {
        if (value.dtype != DType::BF16 || value.data == nullptr || !value.is_contiguous() ||
            value.numel() != static_cast<std::int64_t>(elements * call_.width * call_.batch)) {
            throw std::logic_error("layer3 attention tensor geometry differs");
        }
        const auto* source = static_cast<const std::uint8_t*>(value.data) +
                             static_cast<std::size_t>(call_.row * call_.width + call_.column) * elements * 2U;
        HIP_CHECK(hipMemcpyAsync(attention_payload() + destination_offset, source, elements * 2U,
                                 hipMemcpyDeviceToDevice, stream_));
    }

    void capture_attention(const Tensor& value, std::string_view expected_name) {
        if (next_attention_field_ >= kAttentionFields.size()) {
            throw std::logic_error("too many layer3 attention fields");
        }
        const AttentionField& field = kAttentionFields[next_attention_field_];
        if (expected_name != field.name || value.dtype != field.dtype || value.data == nullptr ||
            !value.is_contiguous() ||
            value.numel() != static_cast<std::int64_t>(field.elements * call_.width * call_.batch)) {
            throw std::logic_error("layer3 attention tensor/order differs");
        }
        const std::size_t element_bytes = dtype_size(field.dtype);
        const auto* source = static_cast<const std::uint8_t*>(value.data) +
                             static_cast<std::size_t>(call_.row * call_.width + call_.column) * field.elements * element_bytes;
        HIP_CHECK(hipMemcpyAsync(attention_payload() + field.offset, source,
                                 field.elements * element_bytes, hipMemcpyDeviceToDevice, stream_));
        ++next_attention_field_;
    }

    static const char* required_provenance(const char* name) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0' || std::strpbrk(value, "\"\\\n\r") != nullptr) {
            throw std::invalid_argument(std::string("layer3 attention trace requires ") + name);
        }
        return value;
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
                 << selected_manifest_fields()
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

    void write_gdn_outputs(std::int32_t token, std::int32_t position,
                           std::int32_t rope_position) const {
        const auto* payload = host_.data() + kPayloadBytes;
        write_exclusive(gdn_sidecar_path(), payload, kGdnPayloadBytes, "GDN sidecar");
        const std::uint64_t hash = fnv1a64(payload, kGdnPayloadBytes);
        std::ostringstream manifest;
        manifest << "{\n"
                 << "  \"artifact_type\": \""
                 << (selected_role(call_.role) ? "ninfer_qwen3_selected_gdn_detail_trace"
                                                : "ninfer_qwen3_layer1_gdn_detail_trace") << "\",\n"
                 << "  \"schema_version\": 1,\n"
                 << "  \"diagnostic_only\": true,\n"
                 << "  \"timing_evidence_eligible\": false,\n"
                 << "  \"production_routing_authorized\": false,\n"
                 << "  \"execution\": \"eager\",\n"
                 << "  \"role\": \"" << role_name(call_.role) << "\",\n"
                 << selected_manifest_fields()
                 << "  \"width\": " << call_.width << ",\n"
                 << "  \"selected_column\": " << call_.column << ",\n"
                 << "  \"absolute_frontier\": " << call_.frontier << ",\n"
                 << "  \"token\": " << token << ",\n"
                 << "  \"cache_position\": " << position << ",\n"
                 << "  \"rope_position\": " << rope_position << ",\n"
                 << "  \"text_layer\": " << gdn_detail_layer() << ",\n"
                 << "  \"gdn_index\": " << gdn_detail_layer() - gdn_detail_layer() / 4 << ",\n"
                 << "  \"sidecar_path\": \"" << gdn_sidecar_path() << "\",\n"
                 << "  \"sidecar_bytes\": " << kGdnPayloadBytes << ",\n"
                 << "  \"sidecar_fnv1a64\": \"" << std::hex << std::setfill('0')
                 << std::setw(16) << hash << std::dec << "\",\n"
                 << "  \"fields\": [\n";
        for (std::size_t index = 0; index < kGdnFields.size(); ++index) {
            const GdnField& field = kGdnFields[index];
            manifest << "    {\"name\":\"" << field.name << "\",\"dtype\":\""
                     << (field.dtype == DType::BF16 ? "bf16" : "fp32")
                     << "\",\"elements\":" << field.elements << ",\"offset\":"
                     << field.offset << ",\"bytes\":"
                     << field.elements * dtype_size(field.dtype) << "}"
                     << (index + 1U == kGdnFields.size() ? "\n" : ",\n");
        }
        manifest << "  ],\n"
                 << "  \"layout\": \""
                 << (selected_role(call_.role) ? "typed selected-column selected-layer GDN boundaries in field order"
                                                : "typed selected-column layer1 GDN boundaries in field order") << "\"\n"
                 << "}\n";
        const std::string contents = manifest.str();
        write_exclusive(gdn_manifest_path(), contents.data(), contents.size(), "GDN manifest");
    }

    void write_recurrent_state_outputs(std::int32_t token, std::int32_t position,
                                       std::int32_t rope_position) const {
        const auto* payload = host_.data() + kCombinedPayloadBytes;
        write_exclusive(recurrent_state_sidecar_path(), payload, kRecurrentStateBytes,
                        "GDN recurrent-state sidecar");
        const std::uint64_t hash = fnv1a64(payload, kRecurrentStateBytes);
        const char* point = call_.role == Role::TextFresh
            ? "wide-prefill-state-after-prefix-128-before-selected-column-128"
            : "restored-append-state-before-selected-column-0";
        std::ostringstream manifest;
        manifest << "{\n"
                 << "  \"artifact_type\": \"ninfer_qwen3_gdn_recurrent_state_trace\",\n"
                 << "  \"schema_version\": 2,\n"
                 << "  \"diagnostic_only\": true,\n"
                 << "  \"timing_evidence_eligible\": false,\n"
                 << "  \"production_routing_authorized\": false,\n"
                 << "  \"execution\": \"eager\",\n"
                 << "  \"role\": \"" << role_name(call_.role) << "\",\n"
                 << "  \"capture_point\": \"" << point << "\",\n"
                 << "  \"selected_token\": " << token << ",\n"
                 << "  \"selected_cache_position\": " << position << ",\n"
                 << "  \"selected_rope_position\": " << rope_position << ",\n"
                 << "  \"state_frontier\": 128,\n"
                 << "  \"linear_state_slot\": 0,\n"
                 << "  \"text_layer\": " << recurrent_state_layer() << ",\n"
                 << "  \"gdn_index\": " << recurrent_state_layer() << ",\n"
                 << "  \"dtype\": \"fp32\",\n"
                 << "  \"shape\": [128,128,48],\n"
                 << "  \"elements\": " << kRecurrentStateElements << ",\n"
                 << "  \"sidecar_path\": \"" << recurrent_state_sidecar_path() << "\",\n"
                 << "  \"sidecar_bytes\": " << kRecurrentStateBytes << ",\n"
                 << "  \"sidecar_fnv1a64\": \"" << std::hex << std::setfill('0')
                 << std::setw(16) << hash << std::dec << "\",\n"
                 << "  \"layout\": \"little-endian-fp32:key,value,value_head\"\n"
                 << "}\n";
        const std::string contents = manifest.str();
        write_exclusive(recurrent_state_manifest_path(), contents.data(), contents.size(),
                        "GDN recurrent-state manifest");
    }

    void write_attention_outputs(std::int32_t token, std::int32_t position,
                                 std::int32_t rope_position) const {
        const std::size_t offset = kCombinedPayloadBytes +
            (recurrent_state_enabled() ? kRecurrentStateBytes : 0U);
        const auto* payload = host_.data() + offset;
        if (selected_role(call_.role)) {
            write_exclusive(attention_sidecar_path(), payload, kAttentionStageBytes,
                            "selected layer3 attention stages");
            std::ostringstream manifest;
            manifest << "{\"artifact_type\":\"ninfer_qwen3_layer3_attention_trace\","
                     << "\"schema_version\":1,\"diagnostic_only\":true,"
                     << "\"timing_evidence_eligible\":false,\"production_routing_authorized\":false,"
                     << "\"execution\":\"eager\",\"cache_capture\":false,\"text_layer\":3,"
                     << "\"role\":\"" << role_name(call_.role) << "\",\n"
                     << selected_manifest_fields()
                     << "\"width\":" << call_.width << ",\"selected_column\":" << call_.column
                     << ",\"absolute_frontier\":" << call_.frontier << ",\"token\":" << token
                     << ",\"cache_position\":" << position << ",\"rope_position\":" << rope_position
                     << ",\"fields\":[";
            for (std::size_t i = 0; i < kAttentionFields.size(); ++i) {
                const auto& field = kAttentionFields[i];
                manifest << (i ? "," : "") << "{\"name\":\"" << field.name
                         << "\",\"dtype\":\"" << (field.dtype == DType::FP32 ? "fp32" : "bf16")
                         << "\",\"elements\":" << field.elements << ",\"offset\":" << field.offset
                         << ",\"bytes\":" << field.elements * dtype_size(field.dtype) << "}";
            }
            manifest << "],\"sidecar_path\":\"" << attention_sidecar_path()
                     << "\",\"sidecar_bytes\":" << kAttentionStageBytes
                     << ",\"sidecar_fnv1a64\":\"" << std::hex << std::setfill('0')
                     << std::setw(16) << fnv1a64(payload, kAttentionStageBytes) << "\"}\n";
            const auto contents = manifest.str();
            write_exclusive(attention_manifest_path(), contents.data(), contents.size(),
                            "selected layer3 attention manifest");
            return;
        }
        const auto* aux = reinterpret_cast<const std::int32_t*>(payload + kAttentionPayloadBytes);
        if (aux[6] != 0) {
            throw std::logic_error("layer3 attention canonical cache gather failed");
        }
        write_exclusive(attention_sidecar_path(), payload, kAttentionPayloadBytes,
                        "layer3 attention sidecar");
        const std::uint64_t hash = fnv1a64(payload, kAttentionPayloadBytes);
        const char* shapes[] = {"[5120]", "[5120]", "[256,24]", "[256,24]",
                                "[256,4]", "[256,4]", "[7168]", "[7168]",
                                "[256,24]", "[256,24]", "[256,24]", "[5120]"};
        std::ostringstream manifest;
        manifest << "{\n"
                 << "  \"artifact_type\": \"ninfer_qwen3_layer3_attention_trace\",\n"
                 << "  \"schema_version\": 1,\n"
                 << "  \"diagnostic_only\": true,\n"
                 << "  \"timing_evidence_eligible\": false,\n"
                 << "  \"production_routing_authorized\": false,\n"
                 << "  \"execution\": \"eager\",\n"
                 << "  \"role\": \"" << role_name(call_.role) << "\",\n"
                 << "  \"call\": {\"phase\":\"verify\",\"batch\":1,\"width\":"
                 << call_.width << ",\"selected_column\":0,\"absolute_frontier\":130,"
                 << "\"token\":" << token << ",\"cache_position\":" << position
                 << ",\"rope_position\":" << rope_position
                 << ",\"text_layer\":3,\"full_attention_index\":0,\"sequence_batch\":1,"
                 << "\"sequence_width\":" << call_.width
                 << ",\"transaction_position_count\":" << call_.width
                 << ",\"live_width\":" << call_.width << "},\n"
                 << "  \"cache_read\": {\"valid_for_stream\":true,\"pending\":true,"
                 << "\"visible_frontier\":" << attention_visible_frontier_
                 << ",\"mapped_pages\":" << attention_mapped_pages_
                 << ",\"head_dim\":256,\"kv_heads\":4,\"value_group\":16,"
                 << "\"key_layout\":\"token-fastest-head-major\","
                 << "\"value_layout\":\"feature-fastest-page-major\","
                 << "\"value_scale_layout\":\"feature-fastest-page-major\","
                 << "\"device_table_row_present\":true,\"pool_table_row_stride\":"
                 << attention_table_stride_ << ",\"pool_table_row_count\":"
                 << attention_table_rows_ << ",\"active_query_rows_present\":false},\n"
                 << "  \"visibility\": {\"row_position\":129,\"ancestor_masks\":";
        manifest << "null,\"prefix_lengths\":null,\"prefix_length_stride\":1},\n";
        manifest << "  \"provenance\": {"
                 << "\"source_commit\":\"" << required_provenance("NINFER_QWEN3_LAYER3_TRACE_SOURCE_COMMIT")
                 << "\",\"source_tree\":\"" << required_provenance("NINFER_QWEN3_LAYER3_TRACE_SOURCE_TREE")
                 << "\",\"executable_sha256\":\"" << required_provenance("NINFER_QWEN3_LAYER3_TRACE_EXECUTABLE_SHA256")
                 << "\",\"artifact_sha256\":\"" << required_provenance("NINFER_QWEN3_LAYER3_TRACE_ARTIFACT_SHA256")
                 << "\",\"history_sha256\":\"" << required_provenance("NINFER_QWEN3_LAYER3_TRACE_HISTORY_SHA256")
                 << "\",\"corpus_sha256\":\"" << required_provenance("NINFER_QWEN3_LAYER3_TRACE_CORPUS_SHA256")
                 << "\",\"device_index\":0,\"device_name\":\"AMD Radeon AI PRO R9700\","
                 << "\"architecture\":\"gfx1201\",\"wave_size\":32,\"pci\":\"0000:13:00.0\","
                 << "\"power_profile_source\":\"" << required_provenance("NINFER_QWEN3_LAYER3_TRACE_POWER_SOURCE")
                 << "\",\"power_profile_value\":\"auto\"},\n"
                 << "  \"fields\": [\n";
        for (std::size_t index = 0; index < kAttentionFields.size(); ++index) {
            const AttentionField& field = kAttentionFields[index];
            const bool compound = index == 6U || index == 7U;
            manifest << "    {\"name\":\"" << field.name << "\",\"dtype\":\""
                     << (field.dtype == DType::FP32 ? "fp32" : "bf16")
                     << "\",\"shape\":" << shapes[index] << ",\"layout\":\""
                     << (compound ? "q[256,24],then-k[256,4],each-feature-fastest"
                                  : "selected-token feature-fastest")
                     << "\",\"elements\":" << field.elements << ",\"offset\":"
                     << field.offset << ",\"bytes\":"
                     << field.elements * dtype_size(field.dtype) << "},\n";
        }
        manifest << "    {\"name\":\"key_fp8\",\"dtype\":\"u8\",\"shape\":[130,4,256],"
                 << "\"layout\":\"logical-position-major,head,feature-fastest\",\"elements\":133120,\"offset\":137216,\"bytes\":133120},\n"
                 << "    {\"name\":\"value_int4\",\"dtype\":\"u8\",\"shape\":[130,4,128],"
                 << "\"layout\":\"logical-position-major,head,feature-fastest\",\"elements\":66560,\"offset\":270336,\"bytes\":66560},\n"
                 << "    {\"name\":\"value_scale_fp16\",\"dtype\":\"fp16\",\"shape\":[130,4,16],"
                 << "\"layout\":\"logical-position-major,head,feature-fastest\",\"elements\":8320,\"offset\":336896,\"bytes\":16640}\n"
                 << "  ],\n"
                 << "  \"sidecar_path\": \"" << attention_sidecar_path() << "\",\n"
                 << "  \"sidecar_bytes\": " << kAttentionPayloadBytes << ",\n"
                 << "  \"sidecar_fnv1a64\": \"" << std::hex << std::setfill('0')
                 << std::setw(16) << hash << std::dec << "\",\n"
                 << "  \"layout\": \"typed selected-column layer3 attention stages then canonical logical cache positions0..129\"\n"
                 << "}\n";
        const std::string contents = manifest.str();
        write_exclusive(attention_manifest_path(), contents.data(), contents.size(),
                        "layer3 attention manifest");
    }

    Call call_;
    hipStream_t stream_ = nullptr;
    DeviceBuffer device_;
    std::vector<std::uint8_t> host_;
    std::size_t next_snapshot_ = 0;
    std::size_t next_gdn_field_ = 0;
    bool recurrent_state_scheduled_ = false;
    std::size_t next_attention_field_ = 0;
    bool attention_cache_scheduled_ = false;
    std::uint32_t attention_visible_frontier_ = 0;
    std::uint32_t attention_mapped_pages_ = 0;
    std::uint32_t attention_table_stride_ = 0;
    std::uint32_t attention_table_rows_ = 0;
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
    void capture_gdn_controls(int layer, const Tensor& h, const Tensor& g, const Tensor& beta,
                              hipStream_t stream) {
        trace.capture_gdn_controls(layer, h, g, beta, stream);
    }
    void capture_gdn_projection(int layer, const Tensor& z, const Tensor& q, const Tensor& k,
                                const Tensor& v, hipStream_t stream) {
        trace.capture_gdn_projection(layer, z, q, k, v, stream);
    }
    void capture_gdn_recurrence(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_gdn_recurrence(layer, value, stream);
    }
    void capture_gdn_normalized(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_gdn_normalized(layer, value, stream);
    }
    void capture_gdn_residual(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_gdn_residual(layer, value, stream);
    }
    RecurrentStateTrace capture_gdn_recurrent_state(int layer, const Tensor& state,
                                                     std::int32_t slot, hipStream_t stream) {
        return trace.capture_gdn_recurrent_state(layer, state, slot, stream);
    }
    void capture_attention_stage(int layer, std::string_view name, const Tensor& value,
                                 hipStream_t stream) {
        trace.capture_attention_stage(layer, name, value, stream);
    }
    void capture_attention_qk_stage(int layer, std::string_view name, const Tensor& q,
                                    const Tensor& k, hipStream_t stream) {
        trace.capture_attention_qk_stage(layer, name, q, k, stream);
    }
    void capture_attention_cache(int layer, const qwen3::PagedKVLayerRead& read,
                                 const Tensor* masks, const Tensor* prefixes,
                                 hipStream_t stream) {
        trace.capture_attention_cache(layer, read, masks, prefixes, stream);
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
    void capture_gdn_controls(int layer, const Tensor& h, const Tensor& g, const Tensor& beta,
                              hipStream_t stream) {
        trace.capture_gdn_controls(layer, h, g, beta, stream);
    }
    void capture_gdn_projection(int layer, const Tensor& z, const Tensor& q, const Tensor& k,
                                const Tensor& v, hipStream_t stream) {
        trace.capture_gdn_projection(layer, z, q, k, v, stream);
    }
    void capture_gdn_recurrence(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_gdn_recurrence(layer, value, stream);
    }
    void capture_gdn_normalized(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_gdn_normalized(layer, value, stream);
    }
    void capture_gdn_residual(int layer, const Tensor& value, hipStream_t stream) {
        trace.capture_gdn_residual(layer, value, stream);
    }
    RecurrentStateTrace capture_gdn_recurrent_state(int layer, const Tensor& state,
                                                     std::int32_t slot, hipStream_t stream) {
        return trace.capture_gdn_recurrent_state(layer, state, slot, stream);
    }
    void capture_attention_stage(int layer, std::string_view name, const Tensor& value,
                                 hipStream_t stream) {
        trace.capture_attention_stage(layer, name, value, stream);
    }
    void capture_attention_qk_stage(int layer, std::string_view name, const Tensor& q,
                                    const Tensor& k, hipStream_t stream) {
        trace.capture_attention_qk_stage(layer, name, q, k, stream);
    }
    void capture_attention_cache(int layer, const qwen3::PagedKVLayerRead& read,
                                 const Tensor* masks, const Tensor* prefixes,
                                 hipStream_t stream) {
        trace.capture_attention_cache(layer, read, masks, prefixes, stream);
    }
};

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::layer_boundary_trace
