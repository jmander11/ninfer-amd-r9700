#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"

#include <hip/hip_bfloat16.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::uint32_t kW8Group = 32;
constexpr std::uint32_t kKAlignment = 128;

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* label) {
    if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string("linear: ") + label + " size overflows");
    }
    return lhs * rhs;
}

std::uint32_t checked_extent(std::int32_t value, const char* label) {
    if (value <= 0) {
        throw std::invalid_argument(std::string("linear: ") + label + " must be positive");
    }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t padded_columns(std::uint32_t columns) {
    if (columns > std::numeric_limits<std::uint32_t>::max() - (kKAlignment - 1U)) {
        throw std::overflow_error("linear: padded K overflows");
    }
    return (columns + kKAlignment - 1U) / kKAlignment * kKAlignment;
}

struct W8PayloadExtents {
    std::uint64_t code_bytes;
    std::uint64_t scale_bytes;
};

W8PayloadExtents validate_w8_weight(const Weight& weight, std::uint32_t rows,
                                    std::uint32_t columns, std::uint32_t padded) {
    const std::uint64_t code_bytes = checked_mul(rows, padded, "W8 code plane");
    const std::uint64_t scale_bytes = checked_mul(
        checked_mul(rows, padded / kW8Group, "W8 scale plane"), sizeof(std::uint16_t),
        "W8 scale plane");
    if (weight.layout != QuantLayout::RowSplit || weight.group != 32 ||
        weight.group_size != 32 || weight.scale_dtype != DType::FP16 ||
        weight.qdata == nullptr || weight.scales == nullptr || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.qdata_bytes != code_bytes ||
        weight.scale_bytes != scale_bytes || weight.padded_shape[0] != weight.shape[0] ||
        weight.padded_shape[1] != static_cast<std::int32_t>(padded) ||
        weight.n != static_cast<std::int32_t>(rows) ||
        weight.k != static_cast<std::int32_t>(columns)) {
        throw std::invalid_argument("linear: malformed W8G32_F16S RowSplit weight");
    }
    return {code_bytes, scale_bytes};
}

struct ByteRange {
    std::uintptr_t first;
    std::uintptr_t last;
};

ByteRange byte_range(const void* pointer, std::uint64_t bytes, const char* label) {
    if (pointer == nullptr || bytes == 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max()) {
        throw std::invalid_argument(std::string("linear: ") + label + " is empty");
    }
    const auto first = reinterpret_cast<std::uintptr_t>(pointer);
    const auto width = static_cast<std::uintptr_t>(bytes);
    if (first > std::numeric_limits<std::uintptr_t>::max() - width) {
        throw std::overflow_error(std::string("linear: ") + label + " range overflows");
    }
    return {first, first + width};
}

bool overlaps(ByteRange lhs, ByteRange rhs) {
    return lhs.first < rhs.last && rhs.first < lhs.last;
}

void require_disjoint(ByteRange output, ByteRange other, const char* label) {
    if (overlaps(output, other)) {
        throw std::invalid_argument(std::string("linear: output overlaps ") + label);
    }
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t tokens,
                                            std::int32_t columns) {
    const std::uint32_t checked_tokens = checked_extent(tokens, "T");
    const std::uint32_t checked_columns = checked_extent(columns, "K");
    std::size_t bytes = 0U;
    if (qtype == QType::Q4G64_F16S) {
        bytes = r9700::linear::kQ4ActivationBits == 8
                    ? r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
                          checked_tokens, checked_columns)
                    : r9700::linear::q4g64_activation_workspace_capacity_bytes(
                          checked_tokens, checked_columns);
    } else if (qtype == QType::W8G32_F16S &&
               r9700::linear::may_use_a8w8(checked_tokens)) {
        bytes = r9700::linear::a8w8g32_activation_workspace_capacity_bytes(
            checked_tokens, checked_columns);
    } else {
        return 0U;
    }
    if (bytes == 0U) throw std::overflow_error("linear: activation workspace overflows");
    return bytes;
}

namespace {

void linear_with_workspace(const Tensor& x, const Weight& weight, Tensor& output,
                           DeviceSpan activation, hipStream_t stream) {
    if (weight.qtype == QType::W8G32_F16S) {
        const std::uint32_t tokens = checked_extent(x.ne[1], "T");
        const std::uint32_t rows = checked_extent(output.ne[0], "N");
        const std::uint32_t columns = checked_extent(x.ne[0], "K");
        if (!r9700::linear::use_a8w8(tokens, rows, columns)) {
            linear(x, weight, output, stream);
            return;
        }
        if (x.dtype != DType::BF16 || output.dtype != DType::BF16 || x.ne[2] != 1 ||
            x.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1) {
            throw std::invalid_argument("linear: expected BF16 x[K,T] and output[N,T]");
        }
        if (output.ne[1] != x.ne[1] || weight.ndim != 2 ||
            weight.shape[0] != output.ne[0] || weight.shape[1] != x.ne[0] ||
            !x.is_contiguous() || !output.is_contiguous()) {
            throw std::invalid_argument("linear: shape or contiguous-layout mismatch");
        }
        const std::uint32_t padded = padded_columns(columns);
        const W8PayloadExtents extents = validate_w8_weight(weight, rows, columns, padded);
        const std::uint64_t input_bytes = checked_mul(
            checked_mul(columns, tokens, "input"), sizeof(hip_bfloat16), "input");
        const std::uint64_t output_bytes = checked_mul(
            checked_mul(rows, tokens, "output"), sizeof(hip_bfloat16), "output");
        const ByteRange input_range = byte_range(x.data, input_bytes, "input");
        const ByteRange output_range = byte_range(output.data, output_bytes, "output");
        const ByteRange code_range =
            byte_range(weight.qdata, extents.code_bytes, "W8 code plane");
        const ByteRange scale_range =
            byte_range(weight.scales, extents.scale_bytes, "W8 scale plane");
        require_disjoint(output_range, input_range, "input");
        require_disjoint(output_range, code_range, "W8 code plane");
        require_disjoint(output_range, scale_range, "W8 scale plane");
        if (overlaps(code_range, scale_range)) {
            throw std::invalid_argument("linear: W8 code and scale planes overlap");
        }
        const std::size_t workspace_bytes = linear_workspace_capacity_bytes(
            QType::W8G32_F16S, static_cast<std::int32_t>(tokens),
            static_cast<std::int32_t>(columns));
        if (activation.data == nullptr || activation.bytes < workspace_bytes) {
            throw std::invalid_argument("linear: external activation workspace is too small");
        }
        HIP_CHECK(r9700::linear::a8w8g32_linear_candidate(
            {.input = static_cast<const hip_bfloat16*>(x.data),
             .weight_codes = static_cast<const std::int8_t*>(weight.qdata),
             .weight_code_bytes = static_cast<std::size_t>(extents.code_bytes),
             .weight_scales = static_cast<const std::uint16_t*>(weight.scales),
             .weight_scale_bytes = static_cast<std::size_t>(extents.scale_bytes),
             .activation_workspace = activation.data,
             .activation_workspace_bytes = workspace_bytes,
             .output = static_cast<hip_bfloat16*>(output.data),
             .tokens = tokens,
             .rows = rows,
             .columns = columns,
             .padded_columns = padded},
            stream));
        return;
    }
    if (weight.qtype != QType::Q4G64_F16S) {
        linear(x, weight, output, stream);
        return;
    }
    if (x.dtype != DType::BF16 || output.dtype != DType::BF16 || x.ne[2] != 1 || x.ne[3] != 1 ||
        output.ne[2] != 1 || output.ne[3] != 1) {
        throw std::invalid_argument("linear: expected BF16 x[K,T] and output[N,T]");
    }
    const std::uint32_t columns = checked_extent(x.ne[0], "K");
    const std::uint32_t tokens = checked_extent(x.ne[1], "T");
    const std::uint32_t rows = checked_extent(output.ne[0], "N");
    if (output.ne[1] != x.ne[1] || weight.ndim != 2 || weight.n != output.ne[0] ||
        weight.k != x.ne[0] || weight.shape[0] != output.ne[0] || weight.shape[1] != x.ne[0] ||
        !x.is_contiguous() || !output.is_contiguous()) {
        throw std::invalid_argument("linear: shape or contiguous-layout mismatch");
    }
    const std::uint32_t padded = padded_columns(columns);
    const std::uint64_t code_bytes = checked_mul(checked_mul(rows, padded, "Q4 code plane"),
                                                  1U, "Q4 code plane") / 2U;
    const std::uint64_t scale_bytes = checked_mul(
        checked_mul(rows, padded / 64U, "Q4 scale plane"), sizeof(std::uint16_t),
        "Q4 scale plane");
    if (weight.layout != QuantLayout::Q4N16K16 || rows % 16U != 0U || weight.group != 64 ||
        weight.group_size != 64 || weight.scale_dtype != DType::FP16 ||
        weight.qdata == nullptr || weight.scales == nullptr || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.qdata_bytes != code_bytes ||
        weight.scale_bytes != scale_bytes || weight.padded_shape[0] != weight.shape[0] ||
        weight.padded_shape[1] != static_cast<std::int32_t>(padded)) {
        throw std::invalid_argument("linear: malformed Q4G64_F16S Q4N16K16 weight");
    }
    const std::uint64_t input_bytes = checked_mul(checked_mul(columns, tokens, "input"),
                                                   sizeof(hip_bfloat16), "input");
    const std::uint64_t output_bytes = checked_mul(checked_mul(rows, tokens, "output"),
                                                    sizeof(hip_bfloat16), "output");
    const ByteRange input_range = byte_range(x.data, input_bytes, "input");
    const ByteRange output_range = byte_range(output.data, output_bytes, "output");
    const ByteRange code_range = byte_range(weight.qdata, code_bytes, "Q4 code plane");
    const ByteRange scale_range = byte_range(weight.scales, scale_bytes, "Q4 scale plane");
    require_disjoint(output_range, input_range, "input");
    require_disjoint(output_range, code_range, "Q4 code plane");
    require_disjoint(output_range, scale_range, "Q4 scale plane");
    if (overlaps(code_range, scale_range)) {
        throw std::invalid_argument("linear: Q4 code and scale planes overlap");
    }
    const std::size_t workspace_bytes = linear_workspace_capacity_bytes(
        QType::Q4G64_F16S, static_cast<std::int32_t>(tokens), static_cast<std::int32_t>(columns));
    if (activation.data == nullptr || activation.bytes < workspace_bytes) {
        throw std::invalid_argument("linear: external activation workspace is too small");
    }
    if constexpr (r9700::linear::kQ4ActivationBits == 8) {
        HIP_CHECK(r9700::linear::a8q4g64_linear_candidate(
            {.input = static_cast<const hip_bfloat16*>(x.data),
             .weight_codes = static_cast<const std::uint8_t*>(weight.qdata),
             .weight_code_bytes = static_cast<std::size_t>(code_bytes),
             .weight_scales = static_cast<const std::uint16_t*>(weight.scales),
             .weight_scale_bytes = static_cast<std::size_t>(scale_bytes),
             .activation_workspace = activation.data,
             .activation_workspace_bytes = workspace_bytes,
             .output = static_cast<hip_bfloat16*>(output.data),
             .tokens = tokens,
             .rows = rows,
             .columns = columns,
             .padded_columns = padded},
            stream));
    } else {
        HIP_CHECK(r9700::linear::q4g64_linear_candidate(
            {.input = static_cast<const hip_bfloat16*>(x.data),
             .weight_codes = static_cast<const std::uint8_t*>(weight.qdata),
             .weight_code_bytes = static_cast<std::size_t>(code_bytes),
             .weight_scales = static_cast<const std::uint16_t*>(weight.scales),
             .weight_scale_bytes = static_cast<std::size_t>(scale_bytes),
             .activation_workspace = activation.data,
             .activation_workspace_bytes = workspace_bytes,
             .output = static_cast<hip_bfloat16*>(output.data),
             .tokens = tokens,
             .rows = rows,
             .columns = columns,
             .padded_columns = padded},
            stream));
    }
}

} // namespace

void linear(const Tensor& x, const Weight& weight, Tensor& output,
            WorkspaceArena& workspace, hipStream_t stream) {
    const std::size_t workspace_bytes = linear_workspace_capacity_bytes(
        weight.qtype, x.ne[1], x.ne[0]);
    auto scope = workspace.scope();
    const DeviceSpan activation = workspace_bytes == 0U ? DeviceSpan{}
                                                        : workspace.alloc_bytes(workspace_bytes);
    linear_with_workspace(x, weight, output, activation, stream);
}

void linear(const Tensor& x, const Weight& weight, Tensor& output,
            const DeviceSpan& activation_workspace, hipStream_t stream) {
    linear_with_workspace(x, weight, output, activation_workspace, stream);
}

void linear(const Tensor& x, const Weight& weight, Tensor& output, hipStream_t stream) {
    if (x.dtype != DType::BF16 || output.dtype != DType::BF16 ||
        x.ne[2] != 1 || x.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1) {
        throw std::invalid_argument("linear: expected BF16 x[K,T] and output[N,T]");
    }
    const std::uint32_t columns = checked_extent(x.ne[0], "K");
    const std::uint32_t tokens = checked_extent(x.ne[1], "T");
    const std::uint32_t rows = checked_extent(output.ne[0], "N");
    if (output.ne[1] != x.ne[1] || weight.ndim != 2 || weight.n != output.ne[0] ||
        weight.k != x.ne[0] || weight.shape[0] != output.ne[0] ||
        weight.shape[1] != x.ne[0] || !x.is_contiguous() || !output.is_contiguous()) {
        throw std::invalid_argument("linear: shape or contiguous-layout mismatch");
    }

    const std::uint64_t input_bytes = checked_mul(checked_mul(columns, tokens, "input"),
                                                  sizeof(hip_bfloat16), "input");
    const std::uint64_t output_bytes = checked_mul(checked_mul(rows, tokens, "output"),
                                                   sizeof(hip_bfloat16), "output");
    const ByteRange input_range = byte_range(x.data, input_bytes, "input");
    const ByteRange output_range = byte_range(output.data, output_bytes, "output");
    require_disjoint(output_range, input_range, "input");

    if (weight.qtype == QType::BF16_CTRL) {
        const std::uint64_t weight_bytes = checked_mul(checked_mul(rows, columns, "weight"),
                                                       sizeof(hip_bfloat16), "weight");
        if (weight.layout != QuantLayout::Contiguous || weight.qdata == nullptr ||
            weight.qhigh != nullptr || weight.scales != nullptr || weight.group != 0 ||
            weight.group_size != 0 || weight.qdata_bytes != weight_bytes ||
            weight.payload_bytes != weight_bytes || weight.padded_shape[0] != weight.shape[0] ||
            weight.padded_shape[1] != weight.shape[1]) {
            throw std::invalid_argument("linear: malformed BF16_CTRL weight");
        }
        require_disjoint(output_range, byte_range(weight.qdata, weight_bytes, "weight"),
                         "weight");
        HIP_CHECK(r9700::linear::bf16_linear(
            {.input = static_cast<const hip_bfloat16*>(x.data),
             .weights = static_cast<const hip_bfloat16*>(weight.qdata),
             .output = static_cast<hip_bfloat16*>(output.data),
             .tokens = tokens,
             .rows = rows,
             .columns = columns},
            stream));
        return;
    }

    if (weight.qtype != QType::W8G32_F16S) {
        throw std::invalid_argument(
            "linear: Q4G64_F16S requires caller-owned workspace; workspace-free route supports only BF16_CTRL and W8G32_F16S");
    }
    const std::uint32_t padded = padded_columns(columns);
    const W8PayloadExtents extents = validate_w8_weight(weight, rows, columns, padded);
    const ByteRange code_range = byte_range(weight.qdata, extents.code_bytes, "W8 code plane");
    const ByteRange scale_range =
        byte_range(weight.scales, extents.scale_bytes, "W8 scale plane");
    require_disjoint(output_range, code_range, "W8 code plane");
    require_disjoint(output_range, scale_range, "W8 scale plane");
    if (overlaps(code_range, scale_range)) {
        throw std::invalid_argument("linear: W8 code and scale planes overlap");
    }
    HIP_CHECK(r9700::linear::w8g32_linear(
        {.input = static_cast<const hip_bfloat16*>(x.data),
         .codes = static_cast<const std::int8_t*>(weight.qdata),
         .code_bytes = static_cast<std::size_t>(extents.code_bytes),
         .scales = static_cast<const std::uint16_t*>(weight.scales),
         .scale_bytes = static_cast<std::size_t>(extents.scale_bytes),
         .output = static_cast<hip_bfloat16*>(output.data),
         .tokens = tokens,
         .rows = rows,
         .columns = columns,
         .padded_columns = padded},
        stream));
}

} // namespace ninfer::ops
