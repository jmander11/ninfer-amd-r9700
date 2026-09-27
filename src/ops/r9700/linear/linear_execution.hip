#include "ops/r9700/linear/linear_execution.h"

#include "ops/r9700/linear/fp8_small_t_linear.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::ops {
namespace {

constexpr std::uint32_t kColumnAlignment       = 128U;
constexpr std::size_t kPlaneAlignment          = 256U;

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* out) noexcept {
    if (out == nullptr || (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t* out) noexcept {
    if (out == nullptr || value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
        return false;
    }
    *out = (value + alignment - 1U) / alignment * alignment;
    return true;
}

bool weight_geometry(const Weight& weight, std::uint32_t* rows, std::uint32_t* columns,
                     std::size_t* code_bytes, std::size_t* scale_offset,
                     std::size_t* scale_bytes, std::size_t* payload_bytes) noexcept {
    if (weight.n <= 0 || weight.k <= 0 || rows == nullptr || columns == nullptr ||
        code_bytes == nullptr || scale_offset == nullptr || scale_bytes == nullptr ||
        payload_bytes == nullptr) {
        return false;
    }
    *rows    = static_cast<std::uint32_t>(weight.n);
    *columns = static_cast<std::uint32_t>(weight.k);
    if (*columns > std::numeric_limits<std::uint32_t>::max() - (kColumnAlignment - 1U)) {
        return false;
    }
    const std::uint32_t padded =
        (*columns + kColumnAlignment - 1U) / kColumnAlignment * kColumnAlignment;
    if (!checked_mul(*rows, padded, code_bytes) ||
        !align_up(*code_bytes, kPlaneAlignment, scale_offset) ||
        !checked_mul(*rows, sizeof(float), scale_bytes) ||
        *scale_bytes > std::numeric_limits<std::size_t>::max() - *scale_offset) {
        return false;
    }
    *payload_bytes = *scale_offset + *scale_bytes;
    return true;
}

bool valid_weight(const Weight& weight, std::uint32_t* rows,
                  std::uint32_t* columns, std::uint32_t* padded_columns) noexcept {
    std::size_t code_bytes = 0;
    std::size_t scale_offset = 0;
    std::size_t scale_bytes = 0;
    std::size_t payload_bytes = 0;
    if (padded_columns == nullptr ||
        !weight_geometry(weight, rows, columns, &code_bytes, &scale_offset,
                         &scale_bytes, &payload_bytes)) {
        return false;
    }
    const auto* payload = static_cast<const std::byte*>(weight.payload);
    const std::uint32_t padded = (*columns + kColumnAlignment - 1U) /
                                 kColumnAlignment * kColumnAlignment;
    *padded_columns = padded;
    return payload != nullptr && weight.qdata == payload &&
           weight.scales == payload + scale_offset && weight.qhigh == nullptr &&
           weight.payload_bytes == payload_bytes && weight.qdata_bytes == code_bytes &&
           weight.scale_bytes == scale_bytes && weight.high_plane_bytes == 0U &&
           weight.qtype == QType::F8E4M3_ROW_F32S &&
           weight.layout == QuantLayout::RowScaled && weight.scale_dtype == DType::FP32 &&
           weight.group == 0 && weight.group_size == 0U && weight.ndim == 2U &&
           weight.shape[0] == weight.n && weight.shape[1] == weight.k &&
           weight.padded_shape[0] == weight.n &&
           weight.padded_shape[1] == static_cast<std::int32_t>(padded) &&
           weight.scale_ne[0] == weight.n &&
           weight.scale_nb[0] == static_cast<std::int64_t>(sizeof(float));
}

bool range(const void* pointer, std::size_t bytes, std::uintptr_t* begin,
           std::uintptr_t* end) noexcept {
    if (pointer == nullptr || bytes == 0U || begin == nullptr || end == nullptr) return false;
    *begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (*begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    *end = *begin + bytes;
    return true;
}

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes, const void* rhs,
                    std::size_t rhs_bytes) noexcept {
    std::uintptr_t lhs_begin = 0, lhs_end = 0, rhs_begin = 0, rhs_end = 0;
    if (!range(lhs, lhs_bytes, &lhs_begin, &lhs_end) ||
        !range(rhs, rhs_bytes, &rhs_begin, &rhs_end)) {
        return true;
    }
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

} // namespace

struct LinearExecution::Impl {
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
    const void* weight_payload = nullptr;
    std::size_t weight_payload_bytes = 0;
    const void* weight_codes = nullptr;
    const void* weight_scales = nullptr;
    void* activation_storage = nullptr;
    std::size_t activation_storage_capacity_bytes = 0;

    [[nodiscard]] r9700::linear::Fp8RowScaledWeight weight() const noexcept {
        return {static_cast<const std::uint8_t*>(weight_codes),
                static_cast<const float*>(weight_scales), rows};
    }

    [[nodiscard]] bool bind(std::uint32_t tokens,
                            r9700::linear::Fp8ActivationWorkspace* out) const noexcept {
        if (activation_storage == nullptr || tokens == 0U) return false;
        const std::size_t bytes =
            r9700::linear::fp8_activation_workspace_capacity_bytes(tokens, columns);
        return bytes != 0U && bytes <= activation_storage_capacity_bytes &&
               r9700::linear::fp8_bind_activation_workspace(activation_storage, bytes, tokens,
                                                            columns, out) == hipSuccess;
    }

    // The projection of an already quantized image, including nonfinite-token poisoning.
    hipError_t project(const r9700::linear::Fp8ActivationWorkspace& activation,
                       hip_bfloat16* output, hipStream_t stream) const noexcept {
        return activation.tokens <= r9700::linear::kFp8SmallTokenLimit
            ? r9700::linear::fp8_small_t_linear(weight(), activation, output, stream)
            : r9700::linear::fp8_row_scaled_prefill_linear(weight(), activation, output, stream);
    }

    [[nodiscard]] bool output_disjoint(const hip_bfloat16* output,
                                       std::uint32_t tokens) const noexcept {
        std::size_t elements = 0, bytes = 0;
        return output != nullptr && checked_mul(tokens, rows, &elements) &&
               checked_mul(elements, sizeof(hip_bfloat16), &bytes) &&
               !ranges_overlap(output, bytes, weight_payload, weight_payload_bytes) &&
               !ranges_overlap(output, bytes, activation_storage,
                               activation_storage_capacity_bytes);
    }
};

LinearExecution::LinearExecution(const Weight& weight,
                                 std::size_t activation_storage_capacity_bytes)
    : impl_(std::make_unique<Impl>()) {
    if (!valid_weight(weight, &impl_->rows, &impl_->columns, &impl_->padded_columns) ||
        activation_storage_capacity_bytes == 0U ||
        !r9700::linear::fp8_small_t_supported(1U, impl_->rows, impl_->columns,
                                              impl_->padded_columns) ||
        !r9700::linear::fp8_row_scaled_prefill_supported(
            r9700::linear::kFp8SmallTokenLimit + 1U, impl_->rows, impl_->columns,
            impl_->padded_columns)) {
        throw std::invalid_argument("invalid row-scaled E4M3 Linear execution binding");
    }
    impl_->weight_payload = weight.payload;
    impl_->weight_payload_bytes = static_cast<std::size_t>(weight.payload_bytes);
    impl_->weight_codes = weight.qdata;
    impl_->weight_scales = weight.scales;
    impl_->activation_storage_capacity_bytes = activation_storage_capacity_bytes;
}

void LinearExecution::bind_storage(void* activation_storage,
                                    std::size_t activation_storage_capacity_bytes) {
    if (impl_ == nullptr || impl_->activation_storage != nullptr || activation_storage == nullptr ||
        activation_storage_capacity_bytes < impl_->activation_storage_capacity_bytes ||
        reinterpret_cast<std::uintptr_t>(activation_storage) % kPlaneAlignment != 0U) {
        throw std::invalid_argument("invalid row-scaled E4M3 delayed storage binding");
    }
    if (ranges_overlap(impl_->weight_payload, impl_->weight_payload_bytes, activation_storage,
                       activation_storage_capacity_bytes)) {
        throw std::invalid_argument("row-scaled E4M3 Linear retained regions overlap");
    }
    impl_->activation_storage = activation_storage;
}

LinearExecution::~LinearExecution() = default;
LinearExecution::LinearExecution(LinearExecution&&) noexcept = default;
LinearExecution& LinearExecution::operator=(LinearExecution&&) noexcept = default;

std::size_t LinearExecution::activation_workspace_capacity_bytes(
    std::uint32_t tokens, std::uint32_t columns) noexcept {
    return r9700::linear::fp8_activation_workspace_capacity_bytes(tokens, columns);
}

std::optional<r9700::linear::Fp8ActivationWorkspace> LinearExecution::activation_workspace(
    std::uint32_t tokens) const noexcept {
    r9700::linear::Fp8ActivationWorkspace workspace{};
    if (impl_ == nullptr || !impl_->bind(tokens, &workspace)) return std::nullopt;
    return workspace;
}

std::uint32_t LinearExecution::rows() const noexcept {
    return impl_ == nullptr ? 0U : impl_->rows;
}

std::uint32_t LinearExecution::columns() const noexcept {
    return impl_ == nullptr ? 0U : impl_->columns;
}

hipError_t LinearExecution::run(std::uint32_t tokens, const hip_bfloat16* input,
                                hip_bfloat16* output, hipStream_t stream) noexcept {
    r9700::linear::Fp8ActivationWorkspace activation{};
    std::size_t input_elements = 0, input_bytes = 0, output_elements = 0, output_bytes = 0;
    if (impl_ == nullptr || !impl_->bind(tokens, &activation) || input == nullptr ||
        !impl_->output_disjoint(output, tokens) ||
        !checked_mul(tokens, impl_->columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        !checked_mul(tokens, impl_->rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes) ||
        ranges_overlap(input, input_bytes, output, output_bytes) ||
        ranges_overlap(input, input_bytes, impl_->weight_payload, impl_->weight_payload_bytes) ||
        ranges_overlap(input, input_bytes, impl_->activation_storage,
                       impl_->activation_storage_capacity_bytes)) {
        return hipErrorInvalidValue;
    }
    const hipError_t quantize_status = r9700::linear::fp8_quantize_activation(
        r9700::linear::Fp8ActivationQuantizeArgs{input, activation}, stream);
    if (quantize_status != hipSuccess) return quantize_status;
    return impl_->project(activation, output, stream);
}

hipError_t LinearExecution::run_quantized(std::uint32_t tokens, hip_bfloat16* output,
                                          hipStream_t stream) noexcept {
    r9700::linear::Fp8ActivationWorkspace activation{};
    if (impl_ == nullptr || !impl_->bind(tokens, &activation) ||
        !impl_->output_disjoint(output, tokens)) {
        return hipErrorInvalidValue;
    }
    return impl_->project(activation, output, stream);
}

} // namespace ninfer::ops
