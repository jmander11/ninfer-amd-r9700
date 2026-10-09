#include "ninfer/ops/grouped_dynamic_conv.h"

#include "ninfer/ops/linear.h"
#include "ops/r9700/dflash/grouped_dynamic_conv_launch.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
        throw std::invalid_argument("grouped_dynamic_conv: B=2..8 admits T=1..16");
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

void require_disjoint(const Tensor& left, const Tensor& right, const char* message) {
    if (left.data == right.data) { throw std::invalid_argument(message); }
}

void require_projection(const Tensor& hidden, const Tensor& projection) {
    if (projection.dtype != DType::BF16 || !projection.is_contiguous() ||
        projection.data == nullptr || projection.ne[0] != kGroupedDynamicConvProjRows ||
        projection.ne[1] != hidden.ne[1] || projection.ne[2] != hidden.ne[2] ||
        projection.ne[3] != 1) {
        throw std::invalid_argument(
            "grouped_dynamic_conv: projection must be BF16 [1280,T] or [1280,T,B]");
    }
}

void require_norm(const Tensor& norm) {
    if (norm.dtype != DType::BF16 || !norm.is_contiguous() || norm.data == nullptr ||
        norm.ne[0] != kGroupedDynamicConvHidden || norm.ne[1] != 1 || norm.ne[2] != 1 ||
        norm.ne[3] != 1) {
        throw std::invalid_argument("grouped_dynamic_conv: norm must be contiguous BF16 [5120]");
    }
}

// Every operand is 16-byte aligned and every pair is disjoint (outputs alias nothing), including
// the image span and its completion words.
void require_pairwise_disjoint(std::initializer_list<const Tensor*> tensors,
                               const Q4ActivationImageTarget* image) {
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
    for (const Tensor* tensor : tensors) {
        if (tensor == nullptr) continue;
        const auto first = reinterpret_cast<std::uintptr_t>(tensor->data);
        // The vector kernels load and store 16-byte vectors of every operand.
        if (first % 16U != 0U) {
            throw std::invalid_argument("grouped_dynamic_conv: operands must be 16-byte aligned");
        }
        ranges.emplace_back(first, first + tensor->bytes());
    }
    if (image != nullptr) {
        const auto first = reinterpret_cast<std::uintptr_t>(image->image.data);
        ranges.emplace_back(first, first + image->image.bytes);
        const auto words = reinterpret_cast<std::uintptr_t>(image->completion);
        ranges.emplace_back(words, words + 2U * sizeof(std::uint32_t));
    }
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        for (std::size_t other = index + 1; other < ranges.size(); ++other) {
            if (ranges[index].first < ranges[other].second &&
                ranges[other].first < ranges[index].second) {
                throw std::invalid_argument("grouped_dynamic_conv: operands overlap");
            }
        }
    }
}

std::optional<r9700::linear::A8G64ActivationWorkspace>
bind_image(const Tensor& hidden, const Q4ActivationImageTarget* image) {
    if (image == nullptr) return std::nullopt;
    const std::int32_t columns = hidden.ne[1] * hidden.ne[2];
    const std::size_t bytes    = q4_activation_image_bytes(columns, kGroupedDynamicConvHidden);
    r9700::linear::A8G64ActivationWorkspace workspace{};
    if (image->image.data == nullptr || image->image.bytes < bytes ||
        image->completion == nullptr ||
        reinterpret_cast<std::uintptr_t>(image->completion) % alignof(std::uint32_t) != 0U ||
        r9700::linear::a8q4g64_bind_activation_workspace(
            image->image.data, bytes, static_cast<std::uint32_t>(columns),
            static_cast<std::uint32_t>(kGroupedDynamicConvHidden), &workspace) != hipSuccess) {
        throw std::invalid_argument("grouped_dynamic_conv: activation image target is invalid");
    }
    return workspace;
}

} // namespace

void grouped_dynamic_conv_prepare(const Tensor& hidden, const Tensor& base_kernel,
                                  const Tensor& projection, Tensor* prepared,
                                  Tensor& finish_dynamic, const Q4ActivationImageTarget* image,
                                  hipStream_t stream) {
    require_hidden_layout(hidden, "hidden");
    if (prepared == nullptr && image == nullptr) {
        throw std::invalid_argument("grouped_dynamic_conv: prepare publishes no output");
    }
    if (prepared != nullptr) require_matching_activation(hidden, *prepared, "prepared");
    require_base_kernel(base_kernel);
    require_finish_dynamic(hidden, finish_dynamic);
    require_projection(hidden, projection);
    require_pairwise_disjoint({&hidden, &base_kernel, &projection, prepared, &finish_dynamic},
                              image);
    const auto planes = bind_image(hidden, image);
    detail::grouped_dynamic_conv_prepare_launch(
        hidden, base_kernel, projection, prepared, finish_dynamic, planes ? &*planes : nullptr,
        image != nullptr ? image->completion : nullptr, stream);
}

void grouped_dynamic_conv_finish(const Tensor& hidden, const Tensor& base_kernel,
                                 const Tensor& finish_dynamic, Tensor& residual,
                                 hipStream_t stream) {
    require_hidden_layout(hidden, "hidden");
    require_matching_activation(hidden, residual, "residual");
    require_base_kernel(base_kernel);
    require_finish_dynamic(hidden, finish_dynamic);
    require_disjoint(hidden, residual, "grouped_dynamic_conv: residual aliases hidden");
    require_disjoint(residual, finish_dynamic,
                     "grouped_dynamic_conv: residual aliases finish_dynamic");
    require_disjoint(residual, base_kernel, "grouped_dynamic_conv: residual aliases base_kernel");
    detail::grouped_dynamic_conv_finish_launch(hidden, base_kernel, finish_dynamic, residual,
                                               stream);
}

void grouped_dynamic_conv_finish_normalized(const Tensor& hidden, const Tensor& base_kernel,
                                            const Tensor& finish_dynamic, const Tensor& residual,
                                            Tensor& residual_out, const Tensor& norm, float eps,
                                            Tensor& normalized,
                                            const Q4ActivationImageTarget* image,
                                            hipStream_t stream) {
    require_hidden_layout(hidden, "hidden");
    require_matching_activation(hidden, residual, "residual");
    require_matching_activation(hidden, residual_out, "residual_out");
    require_matching_activation(hidden, normalized, "normalized");
    require_base_kernel(base_kernel);
    require_finish_dynamic(hidden, finish_dynamic);
    require_norm(norm);
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("grouped_dynamic_conv: eps must be positive and finite");
    }
    require_pairwise_disjoint(
        {&hidden, &base_kernel, &finish_dynamic, &residual, &residual_out, &norm, &normalized},
        image);
    const auto planes = bind_image(hidden, image);
    detail::grouped_dynamic_conv_finish_normalized_launch(
        hidden, base_kernel, finish_dynamic, residual, residual_out, norm, eps, normalized,
        planes ? &*planes : nullptr, image != nullptr ? image->completion : nullptr, stream);
}

} // namespace ninfer::ops
