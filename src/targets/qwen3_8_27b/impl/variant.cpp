#include "targets/qwen3_8_27b/impl/variant.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "ops/r9700/linear/linear_execution.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "targets/qwen3_8_27b/impl/r9700_full_attention.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

constexpr std::size_t kExecutionAlignment = 256U;
constexpr std::size_t kSelectedRoleCount  = 4U;
constexpr std::size_t kSelectedMatmulWorkspaceBytes = 0U;
constexpr std::size_t kSelectedProjectionCount =
    3U * static_cast<std::size_t>(TextConfig::full_attention_layers()) +
    2U * static_cast<std::size_t>(TextConfig::gdn_layers());

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* label) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        throw std::overflow_error(label);
    }
    return lhs + rhs;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    return checked_add(value, alignment - 1U, label) / alignment * alignment;
}

std::size_t execution_activation_bytes(std::uint32_t prefill_tokens,
                                       std::uint32_t maximum_graph_tokens) {
    const std::uint32_t tokens = std::max(prefill_tokens, maximum_graph_tokens);
    if (tokens == 0U) return 0U;
    const std::size_t bytes =
        ops::LinearExecution::activation_workspace_capacity_bytes(tokens, TextConfig::hidden);
    if (bytes == 0U) {
        throw std::overflow_error("R9700 FP8 execution activation capacity overflows");
    }
    return bytes;
}

std::size_t execution_storage_bytes(std::uint32_t prefill_tokens,
                                    std::uint32_t maximum_graph_tokens) {
    const std::size_t activation = align_up(
        execution_activation_bytes(prefill_tokens, maximum_graph_tokens), kExecutionAlignment,
        "R9700 FP8 execution activation alignment overflows");
    return checked_add(activation, kSelectedMatmulWorkspaceBytes,
                       "R9700 FP8 execution storage capacity overflows");
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid R9700 target-leaf token interval");
    }
}

void validate_profile(WeightsProfile profile) {
    if (profile != WeightsProfile::R9700W8G32Candidate &&
        profile != WeightsProfile::R9700W8Bf16EmbeddingEvaluation &&
        profile != WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation &&
        profile != WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation &&
        profile != WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation &&
        profile != WeightsProfile::R9700Q4G64Evaluation &&
        profile != WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation &&
        profile != WeightsProfile::R9700Q4W8Evaluation &&
        profile != WeightsProfile::R9700Q4G64DFlash2Q4Evaluation &&
        profile != WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation &&
        profile != WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation) {
        throw std::invalid_argument("invalid R9700 target weight profile");
    }
}

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> output;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        output.push_back({begin, end});
        if (end == max_frontier) { return output; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { output.push_back({begin, max_frontier}); }
    return output;
}

void copy_bf16_rows(const Tensor& source, std::int32_t source_row, Tensor& destination,
                    std::int32_t destination_row, std::int32_t rows,
                    hipStream_t stream) {
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 ||
        !source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr || source.ne[1] != destination.ne[1] ||
        source.ne[2] != 1 || source.ne[3] != 1 || destination.ne[2] != 1 ||
        destination.ne[3] != 1 || source_row < 0 || destination_row < 0 || rows <= 0 ||
        source_row > source.ne[0] - rows || destination_row > destination.ne[0] - rows) {
        throw std::invalid_argument("R9700 target row copy has invalid geometry");
    }
    const std::size_t element_bytes = sizeof(hip_bfloat16);
    const std::size_t width = static_cast<std::size_t>(rows) * element_bytes;
    HIP_CHECK(hipMemcpy2DAsync(
        static_cast<std::byte*>(destination.data) +
            static_cast<std::size_t>(destination_row) * element_bytes,
        static_cast<std::size_t>(destination.ne[0]) * element_bytes,
        static_cast<const std::byte*>(source.data) +
            static_cast<std::size_t>(source_row) * element_bytes,
        static_cast<std::size_t>(source.ne[0]) * element_bytes, width,
        static_cast<std::size_t>(source.ne[1]), hipMemcpyDeviceToDevice, stream));
}

std::size_t one_matrix_bytes(std::int32_t rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, tokens});
    return layout.peak_bytes(1);
}

std::size_t two_matrix_bytes(std::int32_t first_rows, std::int32_t second_rows,
                             std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {first_rows, tokens});
    (void)layout.alloc(DType::BF16, {second_rows, tokens});
    return layout.peak_bytes(1);
}

void selected_linear(Variant::ExecutionState* execution, Variant::SelectedLinearRole role,
                     std::int32_t text_layer, const Tensor& input, const Weight& weight,
                     Tensor& output, WorkspaceArena& workspace, hipStream_t stream) {
    if (execution != nullptr && execution->run(role, text_layer, input, weight, output, stream)) {
        return;
    }
    if (weight.qtype == QType::F8E4M3_ROW_F32S) {
        throw std::logic_error("R9700 row-scaled E4M3 projection has no Program execution state");
    }
    if (execution != nullptr) {
        execution->linear(input, weight, output, workspace, stream);
    } else {
        ops::linear(input, weight, output, workspace, stream);
    }
}

void serialized_linear(Variant::ExecutionState* execution, const Tensor& input,
                       const Weight& weight, Tensor& output, WorkspaceArena& workspace,
                       hipStream_t stream) {
    if (execution != nullptr) {
        execution->linear(input, weight, output, workspace, stream);
    } else {
        ops::linear(input, weight, output, workspace, stream);
    }
}

void project_gdn_inputs(const Tensor& hidden, const Variant::GdnProjectionWeights& weights,
                        Tensor& query_key, Tensor& value_z, WorkspaceArena& workspace,
                        hipStream_t stream, Variant::ExecutionState* execution,
                        std::int32_t text_layer) {
    if (hidden.ne[1] <= 0 || hidden.ne[2] <= 0 ||
        hidden.ne[1] > std::numeric_limits<std::int32_t>::max() / hidden.ne[2]) {
        throw std::invalid_argument("R9700 GDN projection has invalid token geometry");
    }
    const std::int32_t tokens = hidden.ne[1] * hidden.ne[2];
    Tensor hidden_flat = hidden.view({hidden.ne[0], tokens});
    selected_linear(execution, Variant::SelectedLinearRole::GdnQueryKey, text_layer, hidden_flat,
                    weights.input_projection.query_key, query_key, workspace, stream);
    serialized_linear(execution, hidden_flat, weights.input_projection.value_z, value_z,
                      workspace, stream);
}

void require_bf16_shape(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                        std::int32_t batch, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != rows || tensor.ne[1] != width || tensor.ne[2] != batch ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("R9700 GDN invalid ") + label);
    }
}

void require_i32_selector(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                          bool optional, const char* label) {
    if (optional && tensor.data == nullptr) { return; }
    if (tensor.dtype != DType::I32 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("R9700 GDN invalid ") + label);
    }
}

void require_gdn_conv_operands(const Tensor& hidden, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slots, std::int32_t minimum_width,
                               std::int32_t maximum_width) {
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    if (width < minimum_width || width > maximum_width || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("R9700 GDN invalid verification geometry");
    }
    require_bf16_shape(hidden, TextConfig::hidden, width, batch, "hidden");
    require_bf16_shape(conv_weight, TextConfig::convolution_dim, TextConfig::gdn_conv_kernel, 1,
                       "convolution weight");
    if (conv_states.dtype != DType::BF16 || conv_states.data == nullptr ||
        !conv_states.is_contiguous() || conv_states.ne[0] != TextConfig::convolution_dim ||
        conv_states.ne[1] != TextConfig::gdn_conv_state_width || conv_states.ne[2] <= 0 ||
        conv_states.ne[3] != 1) {
        throw std::invalid_argument("R9700 GDN invalid convolution state pool");
    }
    require_i32_selector(valid_columns, batch, 1, true, "valid columns");
    require_i32_selector(initial_slots, batch, 1, false, "initial state slots");
}

} // namespace

struct Variant::ExecutionState::Impl {
    struct Slot {
        const Weight* weight = nullptr;
        std::unique_ptr<ops::LinearExecution> execution;
    };

    std::array<Slot, static_cast<std::size_t>(TextConfig::layers) * kSelectedRoleCount> slots{};
    std::size_t selected = 0;
    DeviceSpan activation{};

    [[nodiscard]] static std::size_t index(SelectedLinearRole role,
                                           std::int32_t text_layer) {
        if (text_layer < 0 || text_layer >= TextConfig::layers) {
            throw std::invalid_argument("R9700 FP8 selected projection has invalid Text layer");
        }
        return static_cast<std::size_t>(text_layer) * kSelectedRoleCount +
               static_cast<std::size_t>(role);
    }
};

Variant::ExecutionState::ExecutionState(const ModelView& model, DeviceSpan serialized_storage,
                                        std::uint32_t prefill_tokens,
                                        std::uint32_t maximum_concurrency,
                                        std::uint32_t mtp_width,
                                        std::uint32_t dflash_width)
    : impl_(std::make_unique<Impl>()) {
    const std::vector<std::uint32_t> prepared_widths = eager_widths(
        prefill_tokens, maximum_concurrency, mtp_width, dflash_width);
    const std::uint64_t maximum_graph_tokens64 =
        static_cast<std::uint64_t>(maximum_concurrency) *
        std::max({1U, mtp_width, dflash_width});
    if (maximum_graph_tokens64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("R9700 FP8 execution graph width overflows");
    }
    const auto maximum_graph_tokens = static_cast<std::uint32_t>(maximum_graph_tokens64);
    const std::size_t activation_bytes =
        execution_activation_bytes(prefill_tokens, maximum_graph_tokens);
    const std::size_t activation_region = align_up(
        activation_bytes, kExecutionAlignment,
        "R9700 FP8 execution activation alignment overflows");
    const std::size_t required = execution_storage_bytes(prefill_tokens, maximum_graph_tokens);
    if (serialized_storage.data == nullptr || serialized_storage.bytes < required ||
        reinterpret_cast<std::uintptr_t>(serialized_storage.data) % kExecutionAlignment != 0U) {
        throw std::invalid_argument("R9700 FP8 execution serialized region is invalid");
    }
    auto* base = static_cast<std::byte*>(serialized_storage.data);
    impl_->activation = serialized_storage;
    void* const matmul = kSelectedMatmulWorkspaceBytes == 0U
                             ? nullptr
                             : static_cast<void*>(base + activation_region);

    const auto install = [&](SelectedLinearRole role, std::int32_t layer,
                             const Weight& weight) {
        if (weight.qtype != QType::F8E4M3_ROW_F32S) return;
        Impl::Slot& slot = impl_->slots[Impl::index(role, layer)];
        if (slot.execution != nullptr) {
            throw std::logic_error("R9700 FP8 selected projection slot is duplicated");
        }
        slot.weight = &weight;
        slot.execution = std::make_unique<ops::LinearExecution>(
            weight, base, activation_bytes, matmul, kSelectedMatmulWorkspaceBytes);
        for (const std::uint32_t width : prepared_widths) {
            (void)slot.execution->prepare(width);
        }
        ++impl_->selected;
    };

    for (std::int32_t layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            const auto& weights = model.full_layers[static_cast<std::size_t>(
                TextConfig::full_attention_index(layer))];
            install(SelectedLinearRole::AttentionQueryKey, layer,
                    weights.projection.query_key);
            install(SelectedLinearRole::AttentionGateValue, layer,
                    weights.projection.gate_value);
            install(SelectedLinearRole::MlpGateUp, layer, weights.post_mixer.gate_up);
        } else {
            const auto& weights = model.gdn_layers[static_cast<std::size_t>(
                TextConfig::gdn_index(layer))];
            install(SelectedLinearRole::GdnQueryKey, layer,
                    weights.projection.input_projection.query_key);
            install(SelectedLinearRole::MlpGateUp, layer, weights.post_mixer.gate_up);
        }
    }
    if (impl_->selected != 0U && impl_->selected != kSelectedProjectionCount) {
        throw std::invalid_argument(
            "R9700 FP8 execution state has an incomplete selected inventory");
    }
}

void Variant::ExecutionState::linear(const Tensor& input, const Weight& weight, Tensor& output,
                                     WorkspaceArena& fallback_workspace, hipStream_t stream) {
    if (impl_ == nullptr) throw std::logic_error("R9700 linear execution state is empty");
    const std::size_t required = ops::linear_workspace_capacity_bytes(
        weight.qtype, input.ne[1], input.ne[0]);
    if (required == 0U) {
        ops::linear(input, weight, output, stream);
        return;
    }
    if (required > impl_->activation.bytes) {
        throw std::invalid_argument("R9700 serialized linear activation region is too small");
    }
    (void)fallback_workspace;
    ops::linear(input, weight, output, DeviceSpan{impl_->activation.data, required}, stream);
}

void Variant::ExecutionState::fused_mlp_down(const Tensor& gate_up, const Weight& down,
                                             Tensor& output, hipStream_t stream) {
    constexpr std::uint32_t tokens = 2048U;
    constexpr std::uint32_t rows = TextConfig::hidden;
    constexpr std::uint32_t columns = TextConfig::intermediate;
    if (impl_ == nullptr || gate_up.dtype != DType::BF16 || output.dtype != DType::BF16 ||
        gate_up.data == nullptr || output.data == nullptr || !gate_up.is_contiguous() ||
        !output.is_contiguous() || gate_up.ne[0] != 2 * TextConfig::intermediate ||
        gate_up.ne[1] != static_cast<std::int32_t>(tokens) || gate_up.ne[2] != 1 ||
        gate_up.ne[3] != 1 || output.ne[0] != TextConfig::hidden ||
        output.ne[1] != static_cast<std::int32_t>(tokens) || output.ne[2] != 1 ||
        output.ne[3] != 1 || down.qtype != QType::Q4G64_F16S || down.ndim != 2 ||
        down.n != static_cast<std::int32_t>(rows) ||
        down.k != static_cast<std::int32_t>(columns) ||
        down.shape[0] != static_cast<std::int32_t>(rows) ||
        down.shape[1] != static_cast<std::int32_t>(columns) ||
        down.padded_shape[0] != static_cast<std::int32_t>(rows) ||
        down.padded_shape[1] != static_cast<std::int32_t>(columns) ||
        down.layout != QuantLayout::Q4N16K16 || down.group != 64 || down.group_size != 64 ||
        down.scale_dtype != DType::FP16 || down.qdata == nullptr || down.scales == nullptr ||
        down.qhigh != nullptr || down.high_plane_bytes != 0U) {
        throw std::invalid_argument("R9700 fused MLP-down binding differs from qualified shape");
    }
    const std::size_t required =
        ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    if (required == 0U || impl_->activation.data == nullptr ||
        impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 fused MLP-down activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::fused_silu_a8q4g64_down(
        {.gate_up = static_cast<const hip_bfloat16*>(gate_up.data),
         .weight_codes = static_cast<const std::uint8_t*>(down.qdata),
         .weight_code_bytes = static_cast<std::size_t>(down.qdata_bytes),
         .weight_scales = static_cast<const std::uint16_t*>(down.scales),
         .weight_scale_bytes = static_cast<std::size_t>(down.scale_bytes),
         .activation_workspace = impl_->activation.data,
         .activation_workspace_bytes = required,
         .output = static_cast<hip_bfloat16*>(output.data),
         .tokens = tokens,
         .rows = rows,
         .columns = columns,
         .padded_columns = columns},
        stream));
}

Variant::ExecutionState::~ExecutionState() = default;

std::vector<std::uint32_t> Variant::ExecutionState::eager_widths(
    std::uint32_t prefill_tokens, std::uint32_t maximum_concurrency,
    std::uint32_t mtp_width, std::uint32_t dflash_width) {
    if (prefill_tokens == 0U || maximum_concurrency == 0U) {
        throw std::invalid_argument("R9700 linear execution widths must be positive");
    }
    std::vector<std::uint32_t> widths;
    widths.reserve(static_cast<std::size_t>(maximum_concurrency) * 3U + 1U);
    const auto append_product = [&](std::uint32_t batch, std::uint32_t width,
                                    const char* label) {
        if (width == 0U) return;
        const std::uint64_t product = static_cast<std::uint64_t>(batch) * width;
        if (product > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(label);
        }
        widths.push_back(static_cast<std::uint32_t>(product));
    };
    for (std::uint32_t batch = 1; batch <= maximum_concurrency; ++batch) {
        widths.push_back(batch); // Ordinary decode and one-column speculative substeps.
        append_product(batch, mtp_width, "R9700 MTP linear width overflows");
        append_product(batch, dflash_width, "R9700 DFlash linear width overflows");
    }
    widths.push_back(prefill_tokens);
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    return widths;
}

bool Variant::ExecutionState::run(SelectedLinearRole role, std::int32_t text_layer,
                                  const Tensor& input, const Weight& weight, Tensor& output,
                                  hipStream_t stream) {
    if (weight.qtype != QType::F8E4M3_ROW_F32S) return false;
    if (impl_ == nullptr) {
        throw std::logic_error("R9700 FP8 execution state is empty");
    }
    Impl::Slot& slot = impl_->slots[Impl::index(role, text_layer)];
    // LinearExecution retains the exact qdata/scale pointers and geometry of the materialized
    // Weight object installed with this role.  Sharing only its enclosing payload span is not a
    // sufficient binding identity: a copied or altered view could otherwise select an execution
    // prepared for different offsets or dimensions.
    if (slot.execution == nullptr || slot.weight != &weight) {
        throw std::invalid_argument(
            "R9700 FP8 selected projection binding differs from Program state");
    }
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.data == nullptr ||
        output.data == nullptr || !input.is_contiguous() || !output.is_contiguous() ||
        input.ne[0] != weight.k || output.ne[0] != weight.n || input.ne[1] <= 0 ||
        output.ne[1] != input.ne[1] || input.ne[2] != 1 || input.ne[3] != 1 ||
        output.ne[2] != 1 || output.ne[3] != 1) {
        throw std::invalid_argument("R9700 FP8 selected projection tensor geometry differs");
    }
    const auto tokens = static_cast<std::uint32_t>(input.ne[1]);
    if (slot.execution->prepared_profile(tokens) == nullptr) {
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        HIP_CHECK(hipStreamIsCapturing(stream, &capture));
        if (capture != hipStreamCaptureStatusNone) {
            throw std::logic_error(
                "R9700 FP8 projection width was not prepared before graph capture");
        }
        (void)slot.execution->prepare(tokens);
    }
    const ops::LinearExecution::LaunchStatus status = slot.execution->run(
        tokens, static_cast<const hip_bfloat16*>(input.data),
        static_cast<hip_bfloat16*>(output.data), stream);
    if (status.hip != hipSuccess) HIP_CHECK(status.hip);
    if (status.hipblaslt != HIPBLAS_STATUS_SUCCESS) {
        std::ostringstream message;
        message << "R9700 FP8 selected projection failed with hipBLASLt status "
                << static_cast<int>(status.hipblaslt);
        throw std::runtime_error(message.str());
    }
    return true;
}

std::size_t Variant::ExecutionState::selected_count() const noexcept {
    return impl_ == nullptr ? 0U : impl_->selected;
}

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    if (capacity == 0) { return {}; }
    std::vector<GraphExecutionProfile> profiles =
        graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8190, 8191, 16383, 32767});
    for (GraphExecutionProfile& profile : profiles) {
        const std::size_t maximum_visible = static_cast<std::size_t>(profile.max) + 1U;
        profile.topology_class =
            ops::r9700::kv::use_split512_attention(1U, maximum_visible) ? 1U : 0U;
    }
    return profiles;
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8192U, 16384U, 32768U}) {
        const std::uint32_t mtp_offset = 2U * draft_window;
        if (visible_end == ops::r9700::kv::kSplit512MinimumContext &&
            draft_window == 3U) {
            // One captured MTP round contains two fixed-width T=4 attention leaves. The MTP-cache
            // leaf can see max+2K keys while target verification sees max+(K+1), so their split
            // topology transitions require distinct profile boundaries and topology bits.
            const std::uint32_t text_offset = draft_window + 1U;
            ends.push_back(visible_end - 1U - mtp_offset);
            ends.push_back(visible_end - mtp_offset);
            ends.push_back(visible_end - 1U - text_offset);
            ends.push_back(visible_end - text_offset);
            continue;
        }
        if (visible_end >= mtp_offset) { ends.push_back(visible_end - mtp_offset); }
    }
    std::vector<GraphExecutionProfile> profiles = graph_profiles_through(capacity - 1, ends);
    const std::uint32_t width = draft_window + 1U;
    for (GraphExecutionProfile& profile : profiles) {
        const std::size_t maximum_visible =
            static_cast<std::size_t>(profile.max) + 2U * draft_window;
        const std::size_t maximum_text_visible =
            static_cast<std::size_t>(profile.max) + draft_window + 1U;
        profile.topology_class =
            (ops::r9700::kv::use_split512_attention(width, maximum_visible) ? 1U : 0U) |
            (ops::r9700::kv::use_split512_attention(width, maximum_text_visible) ? 2U : 0U);
    }
    return profiles;
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                  std::uint32_t draft_window,
                                                                  std::uint32_t,
                                                                  std::uint32_t verify_width) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block = verify_width != 0 ? verify_width : draft_window + 1;
    std::vector<std::uint32_t> ends;
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8192U, 16384U, 32768U}) {
        if (visible_end == ops::r9700::kv::kSplit512MinimumContext && visible_end > block) {
            ends.push_back(visible_end - 1U - block);
        }
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    }
    std::vector<GraphExecutionProfile> profiles = graph_profiles_through(capacity - 1, ends);
    for (GraphExecutionProfile& profile : profiles) {
        const std::size_t maximum_visible =
            static_cast<std::size_t>(profile.max) + block;
        profile.topology_class =
            ops::r9700::kv::use_split512_attention(block, maximum_visible)
                ? 2U
                : (ops::r9700::kv::use_fp8_qk_wmma(block, maximum_visible) ? 1U : 0U);
    }
    return profiles;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3::TextPhase,
                                   WorkspaceArena& workspace, hipStream_t stream, std::int32_t,
                                   ExecutionState* execution, std::int32_t text_layer) {
    {
        auto query_scope = workspace.scope();
        Tensor query_key = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        selected_linear(execution, SelectedLinearRole::AttentionQueryKey, text_layer, hidden,
                        weights.query_key, query_key, workspace, stream);
        ops::extract_bf16_columns(query_key, 0, query, stream);
        ops::extract_bf16_columns(query_key, TextConfig::query_size, key, stream);
    }
    {
        auto gate_scope = workspace.scope();
        Tensor gate_value = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        selected_linear(execution, SelectedLinearRole::AttentionGateValue, text_layer, hidden,
                        weights.gate_value, gate_value, workspace, stream);
        ops::extract_bf16_columns(gate_value, 0, gate, stream);
        ops::extract_bf16_columns(gate_value, TextConfig::query_size, value, stream);
    }
}

void Variant::full_attention(const Tensor& normalized_query,
                             const qwen3::PagedKVLayerRead& cache_read,
                             const Tensor& cache_positions, Tensor& attention_fp32,
                             WorkspaceArena& workspace, hipStream_t stream,
                             const Tensor* ancestor_masks,
                             const Tensor* prefix_lengths,
                             const Tensor* active_query_rows) {
    if (cache_positions.dtype != DType::I32 || cache_positions.data == nullptr ||
        !cache_positions.is_contiguous() || cache_positions.ne[0] != normalized_query.ne[2] ||
        cache_positions.ne[1] != 1 || cache_positions.ne[2] != 1 ||
        cache_positions.ne[3] != 1) {
        throw std::invalid_argument("R9700 full attention has invalid causal positions");
    }
    if ((ancestor_masks == nullptr) != (prefix_lengths == nullptr)) {
        throw std::invalid_argument("R9700 full attention has an incomplete tree mask pair");
    }
    const auto valid_tree_vector = [](const Tensor* tensor) {
        return tensor != nullptr && tensor->dtype == DType::I32 && tensor->data != nullptr &&
               tensor->is_contiguous() && tensor->ne[1] == 1 && tensor->ne[2] == 1 &&
               tensor->ne[3] == 1;
    };
    if (ancestor_masks != nullptr &&
        (!valid_tree_vector(ancestor_masks) ||
         ancestor_masks->ne[0] != normalized_query.ne[2] ||
         !valid_tree_vector(prefix_lengths) ||
         (prefix_lengths->ne[0] != 1 &&
          prefix_lengths->ne[0] != normalized_query.ne[2]))) {
        throw std::invalid_argument("R9700 full attention has invalid tree mask geometry");
    }
    if (active_query_rows != nullptr &&
        (active_query_rows->dtype != DType::I32 || active_query_rows->data == nullptr ||
         !active_query_rows->is_contiguous() || active_query_rows->ne[0] != 1 ||
         active_query_rows->ne[1] != 1 || active_query_rows->ne[2] != 1 ||
         active_query_rows->ne[3] != 1)) {
        throw std::invalid_argument("R9700 full attention has invalid active-row scalar");
    }
    auto attention_scope = workspace.scope();
    const std::size_t attention_workspace_bytes =
        r9700_full_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(normalized_query.ne[2]),
            cache_read.visible_frontier(),
            ancestor_masks != nullptr || active_query_rows != nullptr);
    DeviceSpan attention_workspace{};
    if (attention_workspace_bytes != 0U) {
        attention_workspace = workspace.alloc_bytes(attention_workspace_bytes);
    }
    HIP_CHECK(r9700_qwen3_8_27b_full_attention(
        R9700FullAttentionArgs{
            .query = normalized_query,
            .cache_read = cache_read,
            .row_positions = static_cast<const std::int32_t*>(cache_positions.data),
            .ancestor_masks = ancestor_masks == nullptr
                                  ? nullptr
                                  : static_cast<const std::int32_t*>(ancestor_masks->data),
            .prefix_lengths = prefix_lengths == nullptr
                                  ? nullptr
                                  : static_cast<const std::int32_t*>(prefix_lengths->data),
            .prefix_length_stride = prefix_lengths != nullptr && prefix_lengths->ne[0] == 1
                                        ? 0U
                                        : 1U,
            .active_query_rows =
                active_query_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(active_query_rows->data),
            .workspace = attention_workspace.data,
            .workspace_bytes = attention_workspace.bytes,
            .output = attention_fp32,
        },
        stream));
}

std::size_t Variant::full_attention_workspace_capacity_bytes(
    std::int32_t maximum_query_rows, std::uint32_t maximum_visible_context,
    bool tree_or_device_count) {
    if (maximum_query_rows <= 0) {
        return 0U;
    }
    return r9700_full_attention_workspace_capacity_bytes(
        static_cast<std::uint32_t>(maximum_query_rows), maximum_visible_context,
        tree_or_device_count);
}

#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
void Variant::text_prefill_attention(const Tensor& normalized_query,
                                     const qwen3::PagedKVLayerRead& cache_read,
                                     const Tensor& cache_positions, Tensor& attention_fp32,
                                     WorkspaceArena& workspace, hipStream_t stream) {
    if (normalized_query.ne[2] < 2) {
        full_attention(normalized_query, cache_read, cache_positions, attention_fp32,
                       workspace, stream);
        return;
    }
    if (cache_positions.dtype != DType::I32 || cache_positions.data == nullptr ||
        !cache_positions.is_contiguous() || cache_positions.ne[0] != normalized_query.ne[2] ||
        cache_positions.ne[1] != 1 || cache_positions.ne[2] != 1 ||
        cache_positions.ne[3] != 1) {
        throw std::invalid_argument("R9700 XAttention prefill has invalid causal positions");
    }
    auto sparse_scope = workspace.scope();
    const std::size_t bytes =
        r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(normalized_query.ne[2]), cache_read.visible_frontier());
    if (bytes == 0U) {
        throw std::invalid_argument("R9700 XAttention prefill has invalid workspace geometry");
    }
    DeviceSpan sparse_workspace = workspace.alloc_bytes(bytes);
    HIP_CHECK(r9700_qwen3_8_27b_text_prefill_attention(
        R9700FullAttentionArgs{
            .query = normalized_query,
            .cache_read = cache_read,
            .row_positions = static_cast<const std::int32_t*>(cache_positions.data),
            .output = attention_fp32,
        },
        sparse_workspace.data, sparse_workspace.bytes, stream));
}

std::size_t Variant::text_prefill_attention_workspace_capacity_bytes(
    std::int32_t maximum_query_rows, std::uint32_t maximum_visible_context) {
    if (maximum_query_rows <= 0) return 0U;
    // A nominal prefill envelope can still end in a one-row remainder, which follows the
    // ordinary dense T=1 leaf.  Size that possible leaf independently of the envelope's maximum
    // width; full_attention_workspace_capacity_bytes intentionally rejects widths above two.
    const std::size_t dense =
        full_attention_workspace_capacity_bytes(1, maximum_visible_context, false);
    const std::size_t sparse =
        r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(maximum_query_rows), maximum_visible_context);
    return std::max(dense, sparse);
}
#endif

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3::TextPhase,
                                          WorkspaceArena& workspace, hipStream_t stream,
                                          std::int32_t, ExecutionState* execution) {
    auto scope = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, attention.ne[1]});
    serialized_linear(execution, attention, weight, delta, workspace, stream);
    ops::residual_add(delta, residual, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, hipStream_t stream,
                                       ExecutionState* execution) {
    serialized_linear(execution, hidden, weights.query, query, workspace, stream);
    serialized_linear(execution, hidden, weights.key, key, workspace, stream);
    serialized_linear(execution, hidden, weights.output_gate, gate, workspace, stream);
    serialized_linear(execution, hidden, weights.value, value, workspace, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden,
                                const MtpAttentionProjectionWeights& weights, Tensor& key,
                                Tensor& value, WorkspaceArena& workspace, hipStream_t stream,
                                ExecutionState* execution) {
    serialized_linear(execution, hidden, weights.key, key, workspace, stream);
    serialized_linear(execution, hidden, weights.value, value, workspace, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, hipStream_t stream,
                                    ExecutionState* execution) {
    serialized_linear(execution, hidden, weights.query, query, workspace, stream);
    serialized_linear(execution, hidden, weights.output_gate, gate, workspace, stream);
}

void Variant::mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm, const Weight& weight,
                     Tensor& residual, WorkspaceArena& workspace, hipStream_t stream,
                     std::int32_t, ExecutionState* execution) {
    auto scope = workspace.scope();
    Tensor packed = workspace.alloc(DType::BF16, {TextConfig::mtp_input_rows,
                                                  embedding_norm.ne[1]});
    ops::mtp_pack_fc_input(embedding_norm, hidden_norm, packed, stream);
    serialized_linear(execution, packed, weight, residual, workspace, stream);
}

void Variant::mtp_attention_output(const Tensor& attention, const Weight& weight, Tensor& residual,
                                   WorkspaceArena& workspace, hipStream_t stream, std::int32_t,
                                   ExecutionState* execution) {
    attention_output_projection(attention, weight, residual, qwen3::TextPhase::Verify,
                                workspace, stream, 0, execution);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3::TextPhase,
                                   WorkspaceArena& workspace, hipStream_t stream,
                                   ExecutionState* execution, std::int32_t text_layer) {
    auto scope = workspace.scope();
    Tensor query_key = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, hidden.ne[1]});
    Tensor value_z = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, hidden.ne[1]});
    project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                       text_layer);
    copy_bf16_rows(query_key, 0, qkv, 0, 2 * TextConfig::key_dim, stream);
    copy_bf16_rows(value_z, 0, qkv, 2 * TextConfig::key_dim, TextConfig::value_dim, stream);
    copy_bf16_rows(value_z, TextConfig::value_dim, output_gate, 0, TextConfig::value_dim, stream);
}

void Variant::gdn_input_projection_prefill_p2048(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_state, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
    qwen3::TextPhase phase, WorkspaceArena& workspace, hipStream_t stream,
    ExecutionState* execution, std::int32_t text_layer) {
    constexpr std::int32_t kTokens = 2048;
    if (!gdn_input_projection_prefill_p2048_selected(phase, hidden.ne[1]) || hidden.ne[2] != 1 ||
        hidden.ne[3] != 1) {
        throw std::invalid_argument("R9700 GDN direct-scatter requires ordinary P2048/B1");
    }
    require_bf16_shape(hidden, TextConfig::hidden, kTokens, 1, "direct-scatter hidden");
    require_bf16_shape(conv_weight, TextConfig::convolution_dim, TextConfig::gdn_conv_kernel, 1,
                       "direct-scatter convolution weight");
    require_bf16_shape(conv_state, TextConfig::convolution_dim,
                       TextConfig::gdn_conv_state_width, 1,
                       "direct-scatter convolution state");
    require_bf16_shape(query, TextConfig::key_dim, kTokens, 1, "direct-scatter query");
    require_bf16_shape(key, TextConfig::key_dim, kTokens, 1, "direct-scatter key");
    require_bf16_shape(value, TextConfig::value_dim, kTokens, 1, "direct-scatter value");
    require_bf16_shape(output_gate, TextConfig::value_dim, kTokens, 1,
                       "direct-scatter output gate");

    auto scope = workspace.scope();
    Tensor query_key = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, kTokens});
    Tensor value_z = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, kTokens});
    project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                       text_layer);
    HIP_CHECK(ops::r9700::gdn::projection_conv_prefill_p2048_direct_scatter_bf16(
        static_cast<const hip_bfloat16*>(query_key.data),
        static_cast<const hip_bfloat16*>(value_z.data),
        static_cast<const hip_bfloat16*>(conv_weight.data),
        static_cast<const hip_bfloat16*>(conv_state.data),
        static_cast<hip_bfloat16*>(conv_state.data), static_cast<hip_bfloat16*>(query.data),
        static_cast<hip_bfloat16*>(key.data), static_cast<hip_bfloat16*>(value.data),
        static_cast<hip_bfloat16*>(output_gate.data), kTokens, stream));
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
    const Tensor& snapshot_base_slots, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3::TextPhase, WorkspaceArena& workspace, hipStream_t stream,
    ExecutionState* execution, std::int32_t text_layer) {
    require_gdn_conv_operands(hidden, conv_weight, conv_states, valid_columns, initial_slots, 1,
                              hidden.ne[2] == 1 ? hidden.ne[1] : 16);
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    require_i32_selector(snapshot_base_slots, batch, 1, false, "snapshot base slots");
    require_bf16_shape(query, TextConfig::key_dim, width, batch, "snapshot query");
    require_bf16_shape(key, TextConfig::key_dim, width, batch, "snapshot key");
    require_bf16_shape(value, TextConfig::value_dim, width, batch, "snapshot value");
    require_bf16_shape(output_gate, TextConfig::value_dim, width, batch,
                       "snapshot output gate");

    auto scope = workspace.scope();
    const std::int32_t tokens = width * batch;
    Tensor query_key = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, tokens});
    Tensor value_z = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, tokens});
    project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                       text_layer);
    HIP_CHECK(ops::r9700::gdn::projection_conv_snapshot_bf16(
        static_cast<const hip_bfloat16*>(query_key.data),
        static_cast<const hip_bfloat16*>(value_z.data),
        static_cast<const hip_bfloat16*>(conv_weight.data),
        static_cast<hip_bfloat16*>(conv_states.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<hip_bfloat16*>(query.data), static_cast<hip_bfloat16*>(key.data),
        static_cast<hip_bfloat16*>(value.data), static_cast<hip_bfloat16*>(output_gate.data),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(batch),
        static_cast<std::uint32_t>(conv_states.ne[2]), stream));
}

void Variant::gdn_input_projection_record(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
    qwen3::TextPhase, WorkspaceArena& workspace, hipStream_t stream,
    const Tensor* parent_index, ExecutionState* execution, std::int32_t text_layer) {
    require_gdn_conv_operands(hidden, conv_weight, conv_states, valid_columns, initial_slots, 2,
                              16);
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    require_bf16_shape(conv_record, TextConfig::convolution_dim, width, batch, "conv record");
    require_bf16_shape(query, TextConfig::key_dim, width, batch, "record query");
    require_bf16_shape(key, TextConfig::key_dim, width, batch, "record key");
    require_bf16_shape(value, TextConfig::value_dim, width, batch, "record value");
    require_bf16_shape(output_gate, TextConfig::value_dim, width, batch, "record output gate");
    if (parent_index != nullptr && parent_index->data != nullptr) {
        require_i32_selector(*parent_index, width, batch, false, "parent index");
    }

    auto scope = workspace.scope();
    const std::int32_t tokens = width * batch;
    Tensor query_key = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, tokens});
    Tensor value_z = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, tokens});
    project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                       text_layer);
    HIP_CHECK(ops::r9700::gdn::projection_conv_record_bf16(
        static_cast<const hip_bfloat16*>(query_key.data),
        static_cast<const hip_bfloat16*>(value_z.data),
        static_cast<const hip_bfloat16*>(conv_weight.data),
        static_cast<const hip_bfloat16*>(conv_states.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_slots.data),
        parent_index == nullptr || parent_index->data == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(parent_index->data),
        static_cast<hip_bfloat16*>(conv_record.data), static_cast<hip_bfloat16*>(query.data),
        static_cast<hip_bfloat16*>(key.data), static_cast<hip_bfloat16*>(value.data),
        static_cast<hip_bfloat16*>(output_gate.data), static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(conv_states.ne[2]), stream));
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3::TextPhase, WorkspaceArena& workspace,
                                    hipStream_t stream, std::int32_t,
                                    ExecutionState* execution) {
    auto scope = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, hidden.ne[1]});
    serialized_linear(execution, hidden, weight, delta, workspace, stream);
    ops::residual_add(delta, residual, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, hipStream_t stream,
                                          ExecutionState* execution) {
    ops::rmsnorm(residual, norm_weight, eps, true, hidden, stream);
    auto scope = workspace.scope();
    Tensor a = workspace.alloc(DType::BF16, {TextConfig::gdn_value_heads, residual.ne[1]});
    Tensor b = workspace.alloc(DType::BF16, {TextConfig::gdn_value_heads, residual.ne[1]});
    serialized_linear(execution, hidden, weights.a_projection, a, workspace, stream);
    serialized_linear(execution, hidden, weights.b_projection, b, workspace, stream);
    HIP_CHECK(ops::r9700::gdn::control_gates_bf16(
        static_cast<const hip_bfloat16*>(a.data), static_cast<const hip_bfloat16*>(b.data),
        static_cast<const float*>(weights.a_log.data),
        static_cast<const float*>(weights.dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), TextConfig::gdn_value_heads,
        static_cast<std::uint32_t>(residual.ne[1]), stream));
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3::TextPhase, WorkspaceArena& workspace, hipStream_t stream,
                         std::int32_t, ExecutionState* execution, std::int32_t text_layer) {
    auto outer = workspace.scope();
    const bool fused_down = execution != nullptr && hidden.ne[1] > 0 &&
        ExecutionState::fused_mlp_down_selected(
            weights.gate_up.qtype, weights.down.qtype,
            static_cast<std::uint32_t>(hidden.ne[1]), text_layer);
    if (fused_down) {
        Tensor gate_up = workspace.alloc(DType::BF16, {2 * TextConfig::intermediate,
                                                       hidden.ne[1]});
        Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, hidden.ne[1]});
        selected_linear(execution, SelectedLinearRole::MlpGateUp, text_layer, hidden,
                        weights.gate_up, gate_up, workspace, stream);
        execution->fused_mlp_down(gate_up, weights.down, delta, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    {
        auto gate_scope = workspace.scope();
        Tensor gate_up = workspace.alloc(DType::BF16, {2 * TextConfig::intermediate,
                                                       hidden.ne[1]});
        selected_linear(execution, SelectedLinearRole::MlpGateUp, text_layer, hidden,
                        weights.gate_up, gate_up, workspace, stream);
        ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                      gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate),
                      activation, stream);
    }
    {
        auto delta_scope = workspace.scope();
        Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, hidden.ne[1]});
        serialized_linear(execution, activation, weights.down, delta, workspace, stream);
        ops::residual_add(delta, residual, stream);
    }
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, hipStream_t stream,
                             std::int32_t route_tokens, ExecutionState* execution) {
    post_mixer(hidden, weights, residual, qwen3::TextPhase::Verify, workspace, stream,
               route_tokens, execution, -1);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_fc_workspace_capacity_bytes(std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::mtp_input_rows, last);
}

std::size_t Variant::mtp_attention_output_workspace_capacity_bytes(std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::hidden, last);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile profile,
                                                                   qwen3::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    return one_matrix_bytes(7168, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile profile, qwen3::TextPhase, std::int32_t first, std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::hidden, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile profile,
                                                                   qwen3::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    return two_matrix_bytes(2 * TextConfig::key_dim, 2 * TextConfig::value_dim, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile profile, qwen3::TextPhase, std::int32_t batch, std::int32_t first,
    std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency) ||
        (batch > 1 && last > 16)) {
        throw std::invalid_argument("invalid R9700 GDN snapshot workspace geometry");
    }
    return two_matrix_bytes(2 * TextConfig::key_dim, 2 * TextConfig::value_dim, batch * last);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile profile, qwen3::TextPhase, std::int32_t batch, std::int32_t first,
    std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency) || first < 2 ||
        last > 16) {
        throw std::invalid_argument("invalid R9700 GDN record workspace geometry");
    }
    return two_matrix_bytes(2 * TextConfig::key_dim, 2 * TextConfig::value_dim, batch * last);
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile profile,
                                                                    qwen3::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::hidden, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    validate_token_interval(first, last);
    return two_matrix_bytes(TextConfig::gdn_value_heads, TextConfig::gdn_value_heads, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile profile,
                                                         qwen3::TextPhase,
                                                         std::int32_t first,
                                                         std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {2 * TextConfig::intermediate, last});
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    }
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    return post_mixer_workspace_capacity_bytes(WeightsProfile::R9700W8G32Candidate,
                                               qwen3::TextPhase::Verify, first, last);
}

QType Variant::dflash_matrix_qtype(WeightsProfile profile) {
    validate_profile(profile);
    switch (profile) {
    case WeightsProfile::R9700Q4G64DFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation:
        return QType::Q4G64_F16S;
    case WeightsProfile::R9700W8G32Candidate:
    case WeightsProfile::R9700W8Bf16EmbeddingEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation:
    case WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation:
    case WeightsProfile::R9700Q4G64Evaluation:
    case WeightsProfile::R9700Q4W8Evaluation:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("invalid R9700 target DFlash matrix profile");
}

std::size_t Variant::linear_workspace_capacity_bytes(WeightsProfile profile,
                                                     std::int32_t tokens) {
    validate_profile(profile);
    switch (profile) {
    case WeightsProfile::R9700W8G32Candidate:
    case WeightsProfile::R9700W8Bf16EmbeddingEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation:
    case WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation:
        return ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                     TextConfig::intermediate);
    case WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation:
        // DFlash feature projection consumes [5120,25600], which is the largest K in the
        // companion or mixed base. Reserve its A8G64 image for every DFlash schedule.
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                     DFlashConfig::feature_rows);
    case WeightsProfile::R9700Q4W8Evaluation:
        // The mixed recipe's Q4 roles top out at hidden K, while its W8 MLP-down roles consume
        // intermediate K. The compile-time W8 profile is workspace-free when exact and owns an
        // A8G32 image only in the separately built A8 evaluator.
        return std::max(
            ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                   TextConfig::hidden),
            ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                   TextConfig::intermediate));
    case WeightsProfile::R9700Q4G64DFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation:
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                     DFlashConfig::feature_rows);
    case WeightsProfile::R9700Q4G64Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation:
        // The all-Q4 recipe also quantizes MLP down, whose K is the largest Text matrix input.
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                     TextConfig::intermediate);
    }
    throw std::invalid_argument("invalid R9700 target weight profile");
}

std::size_t Variant::vision_linear_workspace_capacity_bytes(WeightsProfile profile,
                                                             std::int32_t tokens) {
    validate_profile(profile);
    switch (profile) {
    case WeightsProfile::R9700W8G32Candidate:
    case WeightsProfile::R9700W8Bf16EmbeddingEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation:
    case WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation:
        return ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                     VisionConfig::merger_hidden);
    case WeightsProfile::R9700Q4W8Evaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation:
        // Mixed Vision Q4 tops out at backbone hidden K; its W8 merger consumes the larger
        // merger-hidden K and therefore owns an A8 workspace in the A8 evaluator.
        return std::max(
            ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                   VisionConfig::hidden),
            ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                   VisionConfig::merger_hidden));
    case WeightsProfile::R9700Q4G64Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation:
    case WeightsProfile::R9700Q4G64DFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation:
        // All-Q4 includes the merger matrices, whose represented BF16 input is the largest
        // Vision K (4608, larger than backbone FC2's 4304).
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                     VisionConfig::merger_hidden);
    }
    throw std::invalid_argument("invalid R9700 target weight profile");
}

std::size_t Variant::execution_state_capacity_bytes(WeightsProfile profile,
                                                     std::uint32_t prefill_tokens,
                                                     std::uint32_t maximum_graph_tokens) {
    validate_profile(profile);
    if (prefill_tokens == 0U || maximum_graph_tokens == 0U) {
        throw std::invalid_argument("R9700 linear execution profile widths must be positive");
    }
    const std::uint32_t tokens = std::max(prefill_tokens, maximum_graph_tokens);
    std::size_t bytes = linear_workspace_capacity_bytes(profile, static_cast<std::int32_t>(tokens));
    if (profile == WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation ||
        profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation) {
        bytes = std::max(bytes, execution_storage_bytes(prefill_tokens, maximum_graph_tokens));
    }
    return bytes == 0U ? 0U
                       : align_up(bytes, kExecutionAlignment,
                                  "R9700 serialized linear activation alignment overflows");
}

} // namespace ninfer::targets::qwen3_8_27b::detail
