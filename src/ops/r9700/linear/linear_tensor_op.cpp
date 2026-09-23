#include "ninfer/ops/linear.h"
#include "ninfer/ops/normalized_linear.h"
#include "ninfer/ops/projected_residual.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/device.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "ops/r9700/linear/dflash_verify_down.h"
#include "ops/r9700/linear/w8_tiled_head.h"

#include <hip/hip_bfloat16.h>

#include <cstddef>
#include <cmath>
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
    const bool valid_layout = weight.layout == QuantLayout::RowSplit ||
        (weight.layout == QuantLayout::W8N16K16 && rows == 248320U && columns == 5120U);
    if (!valid_layout || weight.group != 32 ||
        weight.group_size != 32 || weight.scale_dtype != DType::FP16 ||
        weight.qdata == nullptr || weight.scales == nullptr || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.qdata_bytes != code_bytes ||
        weight.scale_bytes != scale_bytes || weight.padded_shape[0] != weight.shape[0] ||
        weight.padded_shape[1] != static_cast<std::int32_t>(padded) ||
        weight.n != static_cast<std::int32_t>(rows) ||
        weight.k != static_cast<std::int32_t>(columns)) {
        throw std::invalid_argument("linear: malformed W8G32_F16S weight or unsupported layout/shape");
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

std::size_t dflash_verify_down_linear_workspace_capacity_bytes(
    QType qtype, std::int32_t tokens, std::int32_t columns, std::int32_t rows) {
    (void)checked_extent(rows, "N");
    return linear_workspace_capacity_bytes(qtype, tokens, columns);
}

namespace {

void linear_with_workspace(const Tensor& x, const Weight& weight, Tensor& output,
                            DeviceSpan activation,
                            hipStream_t stream, bool dflash_target_verify_down = false) {
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
        const auto launch = weight.layout == QuantLayout::W8N16K16
            ? r9700::linear::a8w8g32_tiled_head : r9700::linear::a8w8g32_linear_candidate;
        HIP_CHECK(launch(
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
        if (dflash_target_verify_down && (tokens == 5U || tokens == 6U) &&
            rows == 5120U && columns == 17408U && padded == 17408U) {
            // This complete launcher owns fresh A8 preparation. Dispatch before the
            // generic launcher, never after it; only the activation workspace is used.
            HIP_CHECK(r9700::linear::a8q4g64_dflash_verify_down(
                {.input = static_cast<const hip_bfloat16*>(x.data),
                 .weight_codes = static_cast<const std::uint8_t*>(weight.qdata),
                 .weight_code_bytes = static_cast<std::size_t>(code_bytes),
                 .weight_scales = static_cast<const std::uint16_t*>(weight.scales),
                 .weight_scale_bytes = static_cast<std::size_t>(scale_bytes),
                 .activation_workspace = activation.data,
                 .activation_workspace_bytes = workspace_bytes,
                 .output = static_cast<hip_bfloat16*>(output.data),
                 .tokens = tokens, .rows = rows, .columns = columns,
                 .padded_columns = padded}, stream));
            return;
        }
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

void dflash_verify_down_linear(const Tensor& x, const Weight& weight, Tensor& output,
                               WorkspaceArena& workspace, hipStream_t stream) {
    const std::size_t activation_bytes = linear_workspace_capacity_bytes(
        weight.qtype, x.ne[1], x.ne[0]);
    auto scope = workspace.scope();
    const DeviceSpan activation = activation_bytes == 0U
                                      ? DeviceSpan{}
                                      : workspace.alloc_bytes(activation_bytes);
    linear_with_workspace(x, weight, output, activation, stream, true);
}

std::size_t normalized_linear_workspace_capacity_bytes(
    std::int32_t tokens, std::int32_t columns, std::int32_t rows) {
    checked_extent(tokens, "T");
    if (columns != 5120 || rows != 34816)
        throw std::invalid_argument("normalized linear: unsupported projection shape");
    const std::size_t activation = linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens, columns);
    if (tokens == 1 && r9700::linear::kQ4ActivationBits == 8) return activation;
    // K5120 BF16 rows are already multiples of the 256-byte workspace alignment.
    const std::uint64_t hidden = checked_mul(
        checked_mul(static_cast<std::uint64_t>(tokens), 5120U, "normalized hidden"),
        sizeof(hip_bfloat16), "normalized hidden");
    if (hidden > std::numeric_limits<std::size_t>::max() - activation)
        throw std::overflow_error("normalized linear: complete workspace overflows");
    return static_cast<std::size_t>(hidden) + activation;
}

void normalized_linear(const Tensor& input, const Tensor& norm, float eps,
                       bool unit_offset, const Weight& weight, Tensor& output,
                       const DeviceSpan& workspace, hipStream_t stream) {
    if (input.dtype != DType::BF16 || norm.dtype != DType::BF16 ||
        output.dtype != DType::BF16 || input.ne[0] != 5120 ||
        input.ne[1] <= 0 || input.ne[2] != 1 || input.ne[3] != 1 ||
        norm.ne[0] != 5120 || norm.ne[1] != 1 || norm.ne[2] != 1 || norm.ne[3] != 1 ||
        output.ne[0] != 34816 || output.ne[1] != input.ne[1] ||
        output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !norm.is_contiguous() || !output.is_contiguous() ||
        !(eps > 0.0F) || !std::isfinite(eps) || weight.qtype != QType::Q4G64_F16S ||
        weight.ndim != 2 || weight.n != 34816 || weight.k != 5120 ||
        weight.shape[0] != 34816 || weight.shape[1] != 5120 ||
        weight.padded_shape[0] != 34816 || weight.padded_shape[1] != 5120 ||
        weight.layout != QuantLayout::Q4N16K16 || weight.group != 64 ||
        weight.group_size != 64 || weight.scale_dtype != DType::FP16 ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0U) {
        throw std::invalid_argument("normalized linear: malformed represented inputs");
    }
    const std::size_t required = normalized_linear_workspace_capacity_bytes(
        input.ne[1], input.ne[0], output.ne[0]);
    constexpr std::size_t code_bytes = 34816U * 5120U / 2U;
    constexpr std::size_t scale_bytes = 34816U * 80U * sizeof(std::uint16_t);
    if (weight.qdata_bytes != code_bytes || weight.scale_bytes != scale_bytes ||
        workspace.bytes != required)
        throw std::invalid_argument("normalized linear: plane or workspace extent differs");
    const auto tokens = static_cast<std::uint32_t>(input.ne[1]);
    const void* pointers[]{input.data, norm.data, weight.qdata, weight.scales,
                           output.data, workspace.data};
    const std::size_t sizes[]{static_cast<std::size_t>(tokens) * 5120U * sizeof(hip_bfloat16),
                              5120U * sizeof(hip_bfloat16), code_bytes, scale_bytes,
                              static_cast<std::size_t>(tokens) * 34816U * sizeof(hip_bfloat16),
                              required};
    const std::size_t alignments[]{alignof(hip_bfloat16), alignof(hip_bfloat16),
                                   alignof(std::uint64_t), alignof(std::uint16_t),
                                   alignof(hip_bfloat16), alignof(std::uint32_t)};
    ByteRange ranges[6]{};
    for (std::size_t index = 0U; index < 6U; ++index) {
        const auto first = reinterpret_cast<std::uintptr_t>(pointers[index]);
        if (pointers[index] == nullptr || first % alignments[index] != 0U ||
            first > std::numeric_limits<std::uintptr_t>::max() - sizes[index])
            throw std::invalid_argument("normalized linear: invalid plane address or alignment");
        ranges[index] = {first, first + sizes[index]};
        for (std::size_t other = 0U; other < index; ++other)
            if (overlaps(ranges[index], ranges[other]))
                throw std::invalid_argument("normalized linear: planes must be disjoint");
    }
    if (tokens == 1U && r9700::linear::kQ4ActivationBits == 8) {
        HIP_CHECK(r9700::linear::a8q4g64_normalized_linear_t1(
            {.input = static_cast<const hip_bfloat16*>(input.data),
             .weight_codes = static_cast<const std::uint8_t*>(weight.qdata),
             .weight_code_bytes = code_bytes,
             .weight_scales = static_cast<const std::uint16_t*>(weight.scales),
             .weight_scale_bytes = scale_bytes,
             .activation_workspace = workspace.data, .activation_workspace_bytes = required,
             .output = static_cast<hip_bfloat16*>(output.data),
             .tokens = 1U, .rows = 34816U, .columns = 5120U, .padded_columns = 5120U},
            static_cast<const hip_bfloat16*>(norm.data), eps, unit_offset, stream));
        return;
    }
    Tensor hidden(workspace.data, DType::BF16, {5120, input.ne[1]});
    const std::size_t hidden_bytes = sizes[0];
    const DeviceSpan activation{static_cast<std::uint8_t*>(workspace.data) + hidden_bytes,
                                required - hidden_bytes};
    rmsnorm(input, norm, eps, unit_offset, hidden, stream);
    linear(hidden, weight, output, activation, stream);
}

std::size_t projected_residual_t1_workspace_capacity_bytes(
    std::int32_t columns) {
    if (columns != 6144 && columns != 17408) {
        throw std::invalid_argument("projected residual: unsupported projection width");
    }
    return r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        1U, static_cast<std::uint32_t>(columns));
}

void projected_residual_t1(const Tensor& input, const Weight& weight, Tensor& residual,
                           const DeviceSpan& activation_workspace, hipStream_t stream) {
    constexpr std::uint32_t kRows = 5120U;
    if (input.dtype != DType::BF16 || residual.dtype != DType::BF16 ||
        input.data == nullptr || residual.data == nullptr || !input.is_contiguous() ||
        !residual.is_contiguous() || input.ne[1] != 1 || input.ne[2] != 1 ||
        input.ne[3] != 1 || residual.ne[0] != static_cast<std::int32_t>(kRows) ||
        residual.ne[1] != 1 || residual.ne[2] != 1 || residual.ne[3] != 1 ||
        weight.qtype != QType::Q4G64_F16S || weight.ndim != 2 ||
        weight.n != static_cast<std::int32_t>(kRows) || weight.k != input.ne[0] ||
        weight.shape[0] != static_cast<std::int32_t>(kRows) ||
        weight.shape[1] != input.ne[0] ||
        weight.padded_shape[0] != static_cast<std::int32_t>(kRows) ||
        weight.padded_shape[1] != input.ne[0] || weight.layout != QuantLayout::Q4N16K16 ||
        weight.group != 64 || weight.group_size != 64 || weight.scale_dtype != DType::FP16 ||
        weight.qdata == nullptr || weight.scales == nullptr || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0U) {
        throw std::invalid_argument("projected residual: binding differs from qualified shape");
    }
    const std::uint32_t columns = checked_extent(input.ne[0], "K");
    if (columns != 6144U && columns != 17408U) {
        throw std::invalid_argument("projected residual: unsupported projection width");
    }
    const std::size_t required = projected_residual_t1_workspace_capacity_bytes(input.ne[0]);
    const std::size_t code_bytes = static_cast<std::size_t>(kRows) * columns / 2U;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(kRows) * (columns / 64U) * sizeof(std::uint16_t);
    if (weight.qdata_bytes != code_bytes || weight.scale_bytes != scale_bytes ||
        activation_workspace.bytes != required) {
        throw std::invalid_argument("projected residual: plane or workspace extent differs");
    }
    const void* pointers[]{input.data, weight.qdata, weight.scales, residual.data,
                           activation_workspace.data};
    const std::size_t sizes[]{columns * sizeof(hip_bfloat16), code_bytes, scale_bytes,
                              kRows * sizeof(hip_bfloat16), required};
    const std::size_t alignments[]{alignof(hip_bfloat16), alignof(std::uint64_t),
                                   alignof(std::uint16_t), alignof(hip_bfloat16),
                                   alignof(std::uint32_t)};
    ByteRange ranges[5]{};
    for (std::size_t index = 0U; index < 5U; ++index) {
        const auto first = reinterpret_cast<std::uintptr_t>(pointers[index]);
        if (pointers[index] == nullptr || first % alignments[index] != 0U ||
            first > std::numeric_limits<std::uintptr_t>::max() - sizes[index]) {
            throw std::invalid_argument("projected residual: invalid plane address or alignment");
        }
        ranges[index] = {first, first + sizes[index]};
        for (std::size_t other = 0U; other < index; ++other) {
            if (overlaps(ranges[index], ranges[other])) {
                throw std::invalid_argument("projected residual: planes must be disjoint");
            }
        }
    }
    HIP_CHECK(r9700::linear::a8q4g64_projected_residual_t1(
        {.input = static_cast<const hip_bfloat16*>(input.data),
         .weight_codes = static_cast<const std::uint8_t*>(weight.qdata),
         .weight_code_bytes = static_cast<std::size_t>(weight.qdata_bytes),
         .weight_scales = static_cast<const std::uint16_t*>(weight.scales),
         .weight_scale_bytes = static_cast<std::size_t>(weight.scale_bytes),
         .activation_workspace = activation_workspace.data,
         .activation_workspace_bytes = required,
         .residual = static_cast<hip_bfloat16*>(residual.data),
         .tokens = 1U,
         .rows = kRows,
         .columns = columns,
         .padded_columns = columns},
        stream));
}

void linear(const Tensor& x, const Weight& weight, Tensor& output,
            const DeviceSpan& activation_workspace, hipStream_t stream) {
    linear_with_workspace(x, weight, output, activation_workspace, stream);
}

void dflash_verify_down_linear(const Tensor& x, const Weight& weight, Tensor& output,
                               const DeviceSpan& serialized_workspace,
                               hipStream_t stream) {
    const std::size_t activation_bytes = linear_workspace_capacity_bytes(
        weight.qtype, x.ne[1], x.ne[0]);
    linear_with_workspace(
        x, weight, output,
        DeviceSpan{serialized_workspace.data,
                   std::min(serialized_workspace.bytes, activation_bytes)},
        stream, true);
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
    const auto launch = weight.layout == QuantLayout::W8N16K16
        ? r9700::linear::w8g32_tiled_head : r9700::linear::w8g32_linear;
    HIP_CHECK(launch(
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
