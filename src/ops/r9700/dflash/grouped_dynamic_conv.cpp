#include "ninfer/ops/grouped_dynamic_conv.h"

#include "ninfer/ops/linear.h"
#include "ops/r9700/dflash/grouped_dynamic_conv_launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_hidden_layout(const Tensor& hidden, const char* label) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != kGroupedDynamicConvHidden ||
        hidden.ne[1] <= 0 || hidden.ne[2] <= 0 || hidden.ne[2] > kGroupedDynamicConvMaxBatch ||
        hidden.ne[3] != 1) {
        throw std::invalid_argument(std::string("grouped_dynamic_conv: ") + label +
                                    " must be BF16 [5120,T] or [5120,T,B]");
    }
    if (hidden.ne[2] > 1 && hidden.ne[1] > kGroupedDynamicConvMaxWidthWhenBatched) {
        throw std::invalid_argument("grouped_dynamic_conv: B=2..4 admits T=1..16");
    }
    if (!hidden.is_contiguous() || hidden.data == nullptr) {
        throw std::invalid_argument(std::string("grouped_dynamic_conv: ") + label +
                                    " must be contiguous and non-null");
    }
}

void require_matching_activation(const Tensor& hidden, const Tensor& other, const char* label) {
    require_hidden_layout(other, label);
    for (int dimension = 0; dimension < 4; ++dimension) {
        if (other.ne[dimension] != hidden.ne[dimension]) {
            throw std::invalid_argument(std::string("grouped_dynamic_conv: ") + label +
                                        " shape must match hidden");
        }
    }
}

void require_base_kernel(const Tensor& base_kernel) {
    if (base_kernel.dtype != DType::BF16 || !base_kernel.is_contiguous() ||
        base_kernel.data == nullptr || base_kernel.ne[0] != kGroupedDynamicConvHidden ||
        base_kernel.ne[1] != 2 || base_kernel.ne[2] != 2 || base_kernel.ne[3] != 1) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: base_kernel must be contiguous BF16 [5120,2,2]");
    }
}

void require_finish_dynamic(const Tensor& hidden, const Tensor& finish_dynamic) {
    if (finish_dynamic.dtype != DType::BF16 || !finish_dynamic.is_contiguous() ||
        finish_dynamic.data == nullptr || finish_dynamic.ne[0] != kGroupedDynamicConvGroups ||
        finish_dynamic.ne[1] != 2 || finish_dynamic.ne[2] != hidden.ne[1] ||
        finish_dynamic.ne[3] != hidden.ne[2]) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: finish_dynamic must be BF16 [320,2,T] or [320,2,T,B]");
    }
}

void require_projection_weight(const Weight& weight) {
    if (weight.n != kGroupedDynamicConvProjRows || weight.k != kGroupedDynamicConvHidden ||
        weight.ndim != 2 || weight.shape[0] != kGroupedDynamicConvProjRows ||
        weight.shape[1] != kGroupedDynamicConvHidden) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: kernel_projection must be logical [1280,5120]");
    }
    if (weight.qtype != QType::BF16_CTRL && weight.qtype != QType::W8G32_F16S &&
        weight.qtype != QType::Q4G64_F16S) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: projection must be BF16_CTRL, W8G32_F16S, or Q4G64_F16S");
    }
    if (weight.qtype == QType::Q4G64_F16S && weight.layout != QuantLayout::Q4N16K16) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: Q4G64_F16S projection must use Q4N16K16");
    }
    if (weight.qtype == QType::W8G32_F16S && weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: W8G32_F16S projection must use RowSplit");
    }
    if (weight.qtype == QType::BF16_CTRL && weight.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: BF16 projection must be contiguous");
    }
}

void require_disjoint(const Tensor& left, const Tensor& right, const char* message) {
    if (left.data == right.data) { throw std::invalid_argument(message); }
}

std::size_t projection_bytes(std::int32_t tokens, std::int32_t batch) {
    return static_cast<std::size_t>(kGroupedDynamicConvProjRows) *
           static_cast<std::size_t>(tokens) * static_cast<std::size_t>(batch) *
           sizeof(std::uint16_t);
}

} // namespace

std::size_t grouped_dynamic_conv_prepare_workspace_capacity_bytes(
    QType qtype, std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch) {
    if (min_tokens <= 0 || max_tokens < min_tokens || batch <= 0 ||
        batch > kGroupedDynamicConvMaxBatch ||
        (batch > 1 && max_tokens > kGroupedDynamicConvMaxWidthWhenBatched)) {
        throw std::invalid_argument("grouped_dynamic_conv workspace: invalid token/batch interval");
    }
    if (qtype != QType::BF16_CTRL && qtype != QType::W8G32_F16S &&
        qtype != QType::Q4G64_F16S) {
        throw std::invalid_argument("grouped_dynamic_conv workspace: unsupported projection qtype");
    }
    const std::int32_t columns = max_tokens * batch;
    return projection_bytes(max_tokens, batch) +
           linear_workspace_capacity_bytes(qtype, columns, kGroupedDynamicConvHidden);
}

void grouped_dynamic_conv_prepare(const Tensor& hidden, const Tensor& base_kernel,
                                  const Weight& kernel_projection, Tensor& prepared,
                                  Tensor& finish_dynamic, WorkspaceArena& workspace,
                                  hipStream_t stream) {
    require_hidden_layout(hidden, "hidden");
    require_matching_activation(hidden, prepared, "prepared");
    require_base_kernel(base_kernel);
    require_finish_dynamic(hidden, finish_dynamic);
    require_projection_weight(kernel_projection);
    require_disjoint(hidden, prepared, "grouped_dynamic_conv: prepared aliases hidden");
    require_disjoint(hidden, finish_dynamic,
                     "grouped_dynamic_conv: finish_dynamic aliases hidden");
    require_disjoint(prepared, finish_dynamic,
                     "grouped_dynamic_conv: prepared aliases finish_dynamic");
    require_disjoint(prepared, base_kernel,
                     "grouped_dynamic_conv: prepared aliases base_kernel");

    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    auto scope = workspace.scope();
    const DeviceSpan projection_storage = workspace.alloc_bytes(projection_bytes(tokens, batch));
    Tensor projection(projection_storage.data, DType::BF16,
                      {kGroupedDynamicConvProjRows, tokens * batch});
    Tensor hidden_flat = hidden.view({kGroupedDynamicConvHidden, tokens * batch});
    linear(hidden_flat, kernel_projection, projection, workspace, stream);
    detail::grouped_dynamic_conv_prepare_launch(hidden, base_kernel, projection, prepared,
                                                finish_dynamic, stream);
}

void grouped_dynamic_conv_finish(const Tensor& hidden, const Tensor& base_kernel,
                                 const Tensor& finish_dynamic, Tensor& out, hipStream_t stream) {
    require_hidden_layout(hidden, "hidden");
    require_matching_activation(hidden, out, "out");
    require_base_kernel(base_kernel);
    require_finish_dynamic(hidden, finish_dynamic);
    require_disjoint(hidden, out, "grouped_dynamic_conv: out aliases hidden");
    require_disjoint(out, finish_dynamic, "grouped_dynamic_conv: out aliases finish_dynamic");
    require_disjoint(out, base_kernel, "grouped_dynamic_conv: out aliases base_kernel");
    detail::grouped_dynamic_conv_finish_launch(hidden, base_kernel, finish_dynamic, out, stream);
}

} // namespace ninfer::ops
