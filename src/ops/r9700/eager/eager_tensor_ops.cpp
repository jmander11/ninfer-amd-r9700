#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/l2norm.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/residual_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include "core/device.h"
#include "ops/r9700/eager/eager_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using RawGeluMode = r9700::eager::GeluMode;

std::size_t checked_numel(const Tensor& tensor, const char* operation, const char* label,
                          bool allow_zero = false) {
    std::size_t total = 1;
    for (int dimension = 0; dimension < 4; ++dimension) {
        if (tensor.ne[dimension] < 0 || (!allow_zero && tensor.ne[dimension] == 0)) {
            throw std::invalid_argument(std::string(operation) + ": " + label +
                                        " dimensions must be positive");
        }
        if (tensor.ne[dimension] == 0) { return 0; }
        const auto extent = static_cast<std::size_t>(tensor.ne[dimension]);
        if (total > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error(std::string(operation) + ": tensor size overflows size_t");
        }
        total *= extent;
    }
    return total;
}

void require_dtype(const Tensor& tensor, DType dtype, const char* operation, const char* label) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string(operation) + ": " + label +
                                    " has the wrong dtype");
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* operation, const char* label) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(operation) + ": " + label +
                                    " must be contiguous and non-null");
    }
}

std::size_t require_valid_strided_nonnull(const Tensor& tensor, const char* operation,
                                          const char* label) {
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(operation) + ": " + label + " is null");
    }
    const std::size_t element_bytes = dtype_size(tensor.dtype);
    std::array<int, 4> dimensions{0, 1, 2, 3};
    std::sort(dimensions.begin(), dimensions.end(), [&](int lhs, int rhs) {
        return tensor.nb[lhs] < tensor.nb[rhs];
    });
    std::size_t occupied = element_bytes;
    for (const int dimension : dimensions) {
        if (tensor.nb[dimension] <= 0 ||
            tensor.nb[dimension] % static_cast<std::int64_t>(element_bytes) != 0) {
            throw std::invalid_argument(std::string(operation) + ": " + label +
                                        " has an invalid element stride");
        }
        if (tensor.ne[dimension] == 1) { continue; }
        const auto stride = static_cast<std::size_t>(tensor.nb[dimension]);
        if (stride < occupied) {
            throw std::invalid_argument(std::string(operation) + ": " + label +
                                        " has overlapping logical elements");
        }
        const auto tail = static_cast<std::size_t>(tensor.ne[dimension] - 1);
        if (tail != 0U && stride > (std::numeric_limits<std::size_t>::max() - occupied) / tail) {
            throw std::overflow_error(std::string(operation) + ": " + label +
                                      " storage span overflows size_t");
        }
        occupied += tail * stride;
    }
    return occupied;
}

bool storage_ranges_overlap(const void* first, std::size_t first_bytes,
                            const void* second, std::size_t second_bytes) {
    const auto first_address = reinterpret_cast<std::uintptr_t>(first);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second);
    if (first_address <= second_address) {
        return second_address - first_address < first_bytes;
    }
    return first_address - second_address < second_bytes;
}

void require_same_shape(const Tensor& first, const Tensor& second, const char* operation) {
    for (int dimension = 0; dimension < 4; ++dimension) {
        if (first.ne[dimension] != second.ne[dimension]) {
            throw std::invalid_argument(std::string(operation) + ": tensor shapes differ");
        }
    }
}

bool tensors_overlap(const Tensor& first, const Tensor& second) {
    const auto first_address = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second.data);
    const std::size_t first_bytes = first.bytes();
    const std::size_t second_bytes = second.bytes();
    if (first_address <= second_address) {
        return second_address - first_address < first_bytes;
    }
    return first_address - second_address < second_bytes;
}

void require_vector(const Tensor& tensor, DType dtype, const char* operation, const char* label) {
    require_dtype(tensor, dtype, operation, label);
    if (tensor.ne[0] <= 0 || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(operation) + ": " + label +
                                    " must be a non-empty vector");
    }
    require_contiguous_nonnull(tensor, operation, label);
}

void require_scalar(const Tensor& tensor, DType dtype, const char* operation, const char* label) {
    require_vector(tensor, dtype, operation, label);
    if (tensor.ne[0] != 1) {
        throw std::invalid_argument(std::string(operation) + ": " + label +
                                    " must be a scalar");
    }
}

std::uint32_t u32_extent(std::int32_t extent, const char* operation, const char* label) {
    if (extent <= 0) {
        throw std::invalid_argument(std::string(operation) + ": " + label +
                                    " must be positive");
    }
    return static_cast<std::uint32_t>(extent);
}

std::uint32_t rows_for(const Tensor& tensor, std::size_t elements, const char* operation) {
    const auto features = static_cast<std::size_t>(u32_extent(tensor.ne[0], operation, "ne[0]"));
    const std::size_t rows = elements / features;
    if (rows == 0 || rows > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(operation) + ": row count exceeds HIP grid range");
    }
    return static_cast<std::uint32_t>(rows);
}

void require_finite_eps(float eps, const char* operation) {
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(std::string(operation) + ": eps must be positive and finite");
    }
}

void require_norm(const Tensor& input, const Tensor& output, const Tensor* weight,
                  const Tensor* bias, const Tensor* gate, float eps,
                  const char* operation) {
    require_dtype(input, DType::BF16, operation, "input");
    require_dtype(output, DType::BF16, operation, "output");
    require_same_shape(input, output, operation);
    require_finite_eps(eps, operation);
    const std::size_t elements = checked_numel(input, operation, "input");
    (void)checked_numel(output, operation, "output");
    require_contiguous_nonnull(input, operation, "input");
    require_contiguous_nonnull(output, operation, "output");
    if (input.data == output.data) {
        throw std::invalid_argument(std::string(operation) + ": input and output must not alias");
    }
    const auto features = u32_extent(input.ne[0], operation, "feature extent");
    (void)rows_for(input, elements, operation);
    for (const auto item : {std::pair{weight, "weight"}, std::pair{bias, "bias"}}) {
        if (item.first == nullptr) { continue; }
        require_vector(*item.first, DType::BF16, operation, item.second);
        if (item.first->ne[0] != static_cast<std::int32_t>(features)) {
            throw std::invalid_argument(std::string(operation) + ": " + item.second +
                                        " feature extent differs");
        }
        if (item.first->data == output.data) {
            throw std::invalid_argument(std::string(operation) + ": output aliases an input");
        }
    }
    if (gate != nullptr) {
        require_dtype(*gate, DType::BF16, operation, "gate");
        require_same_shape(input, *gate, operation);
        require_contiguous_nonnull(*gate, operation, "gate");
        if (gate->data == output.data) {
            throw std::invalid_argument(std::string(operation) + ": output aliases gate");
        }
    }
}

std::int32_t align_up_i32(std::int32_t value, std::int32_t alignment,
                          const char* operation) {
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max() - (alignment - 1)) {
        throw std::overflow_error(std::string(operation) + ": invalid padded feature extent");
    }
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* operation) {
    if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string(operation) + ": payload size overflows uint64");
    }
    return lhs * rhs;
}

void require_embedding_common(const Tensor& ids, const Weight& table, const Tensor& output) {
    constexpr const char* operation = "embedding";
    require_vector(ids, DType::I32, operation, "ids");
    require_dtype(output, DType::BF16, operation, "output");
    if (output.ne[0] <= 0 || output.ne[1] != ids.ne[0] || output.ne[2] != 1 ||
        output.ne[3] != 1) {
        throw std::invalid_argument("embedding: output must have shape [features,T]");
    }
    require_contiguous_nonnull(output, operation, "output");
    if (table.ndim != 2 || table.shape[0] <= 0 || table.shape[1] != output.ne[0]) {
        throw std::invalid_argument("embedding: table must have shape [vocabulary,features]");
    }
    if (output.data == ids.data) {
        throw std::invalid_argument("embedding: output must not alias ids");
    }
}

} // namespace

void fill_i32_positions(Tensor& positions, std::int32_t start, hipStream_t stream) {
    constexpr const char* operation = "fill_i32_positions";
    require_vector(positions, DType::I32, operation, "positions");
    const auto count = static_cast<std::size_t>(positions.ne[0]);
    if (start < 0 || count - 1U > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - start)) {
        throw std::invalid_argument("fill_i32_positions: range is outside non-negative I32");
    }
    HIP_CHECK(r9700::eager::i32_iota(static_cast<std::int32_t*>(positions.data), count, start,
                                     stream));
}

void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          hipStream_t stream) {
    constexpr const char* operation = "offset_i32_positions";
    require_vector(source, DType::I32, operation, "source");
    require_scalar(delta, DType::I32, operation, "delta");
    require_vector(destination, DType::I32, operation, "destination");
    if (source.ne[0] != destination.ne[0]) {
        throw std::invalid_argument("offset_i32_positions: source and destination shapes differ");
    }
    if (delta.data == destination.data) {
        throw std::invalid_argument("offset_i32_positions: delta must not alias destination");
    }
    HIP_CHECK(r9700::eager::i32_offset_scalar(
        static_cast<const std::int32_t*>(source.data),
        static_cast<const std::int32_t*>(delta.data),
        static_cast<std::int32_t*>(destination.data),
        static_cast<std::size_t>(source.ne[0]), stream));
}

void set_i32_scalar(Tensor& destination, std::int32_t value, hipStream_t stream) {
    require_scalar(destination, DType::I32, "set_i32_scalar", "destination");
    HIP_CHECK(r9700::eager::i32_fill(static_cast<std::int32_t*>(destination.data), 1, value,
                                     stream));
}

void assign_i32_scalar(const Tensor& source, Tensor& destination, hipStream_t stream) {
    require_scalar(source, DType::I32, "assign_i32_scalar", "source");
    require_scalar(destination, DType::I32, "assign_i32_scalar", "destination");
    if (source.data == destination.data) {
        throw std::invalid_argument("assign_i32_scalar: source and destination must not alias");
    }
    HIP_CHECK(r9700::eager::i32_copy(static_cast<const std::int32_t*>(source.data),
                                     static_cast<std::int32_t*>(destination.data), 1, stream));
}

void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     hipStream_t stream) {
    require_scalar(lhs, DType::I32, "add_i32_scalars", "lhs");
    require_scalar(rhs, DType::I32, "add_i32_scalars", "rhs");
    require_scalar(destination, DType::I32, "add_i32_scalars", "destination");
    if (destination.data == lhs.data || destination.data == rhs.data) {
        throw std::invalid_argument("add_i32_scalars: destination must not alias inputs");
    }
    HIP_CHECK(r9700::eager::i32_add_scalars(
        static_cast<const std::int32_t*>(lhs.data), static_cast<const std::int32_t*>(rhs.data),
        static_cast<std::int32_t*>(destination.data), stream));
}

void increment_i32_scalar(Tensor& scalar, hipStream_t stream) {
    require_scalar(scalar, DType::I32, "increment_i32_scalar", "scalar");
    HIP_CHECK(r9700::eager::i32_increment(static_cast<std::int32_t*>(scalar.data), stream));
}

void increment_i64_scalar(Tensor& scalar, hipStream_t stream) {
    require_scalar(scalar, DType::I64, "increment_i64_scalar", "scalar");
    HIP_CHECK(r9700::eager::i64_increment(static_cast<std::int64_t*>(scalar.data), stream));
}

void scatter(const Tensor& source, const Tensor& indices, Tensor& destination,
             hipStream_t stream) {
    constexpr const char* operation = "scatter";
    require_dtype(source, DType::BF16, operation, "source");
    require_dtype(destination, DType::BF16, operation, "destination");
    require_vector(indices, DType::I32, operation, "indices");
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || source.ne[2] != 1 || source.ne[3] != 1 ||
        destination.ne[0] != source.ne[0] || destination.ne[1] <= 0 ||
        destination.ne[2] != 1 || destination.ne[3] != 1 || indices.ne[0] != source.ne[1]) {
        throw std::invalid_argument("scatter: expected source [D,V], indices [V], destination [D,T]");
    }
    require_contiguous_nonnull(source, operation, "source");
    require_contiguous_nonnull(destination, operation, "destination");
    if (source.data == destination.data || indices.data == destination.data) {
        throw std::invalid_argument("scatter: destination must not alias inputs");
    }
    HIP_CHECK(r9700::eager::bf16_scatter_columns(
        static_cast<const hip_bfloat16*>(source.data),
        static_cast<const std::int32_t*>(indices.data),
        static_cast<hip_bfloat16*>(destination.data),
        static_cast<std::uint32_t>(source.ne[0]), static_cast<std::uint32_t>(source.ne[1]),
        static_cast<std::uint32_t>(destination.ne[1]), stream));
}

void scatter_bf16_batch(const Tensor& source, const Tensor& lanes,
                        const Tensor& valid_columns, Tensor& destination,
                        hipStream_t stream) {
    constexpr const char* operation = "scatter_bf16_batch";
    require_dtype(source, DType::BF16, operation, "source");
    require_dtype(destination, DType::BF16, operation, "destination");
    require_vector(lanes, DType::I32, operation, "lanes");
    require_vector(valid_columns, DType::I32, operation, "valid_columns");
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || source.ne[2] <= 0 || source.ne[3] != 1 ||
        destination.ne[0] != source.ne[0] || destination.ne[1] != source.ne[1] ||
        destination.ne[2] <= 0 || destination.ne[3] != 1 || lanes.ne[0] != source.ne[2] ||
        valid_columns.ne[0] != source.ne[2]) {
        throw std::invalid_argument("scatter_bf16_batch: invalid [D,W,B] to [D,W,C] geometry");
    }
    require_contiguous_nonnull(source, operation, "source");
    if (destination.data == nullptr || destination.nb[0] != 2 || destination.nb[1] < 0 ||
        destination.nb[2] < 0 || (destination.nb[1] % 2) != 0 ||
        (destination.nb[2] % 2) != 0) {
        throw std::invalid_argument("scatter_bf16_batch: invalid destination strides");
    }
    if (source.data == destination.data || lanes.data == destination.data ||
        valid_columns.data == destination.data) {
        throw std::invalid_argument("scatter_bf16_batch: destination must not alias inputs");
    }
    HIP_CHECK(r9700::eager::bf16_scatter_batch(
        static_cast<const hip_bfloat16*>(source.data),
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<hip_bfloat16*>(destination.data),
        static_cast<std::uint32_t>(source.ne[0]), static_cast<std::uint32_t>(source.ne[1]),
        static_cast<std::uint32_t>(source.ne[2]), static_cast<std::size_t>(destination.nb[1] / 2),
        static_cast<std::size_t>(destination.nb[2] / 2), stream));
}

void gather_bf16_path(Tensor& features, const Tensor& lanes, const Tensor& path,
                      const Tensor& counts, hipStream_t stream) {
    constexpr const char* operation = "gather_bf16_path";
    require_dtype(features, DType::BF16, operation, "features");
    require_vector(lanes, DType::I32, operation, "lanes");
    require_vector(counts, DType::I32, operation, "counts");
    require_dtype(path, DType::I32, operation, "path");
    if (features.ne[0] <= 0 || features.ne[1] <= 0 || features.ne[2] <= 0 ||
        features.ne[3] != 1 || path.ne[0] != features.ne[1] || path.ne[1] != lanes.ne[0] ||
        path.ne[2] != 1 || path.ne[3] != 1 || counts.ne[0] != lanes.ne[0]) {
        throw std::invalid_argument("gather_bf16_path: invalid [D,W,C] / [W,B] geometry");
    }
    require_contiguous_nonnull(features, operation, "features");
    require_contiguous_nonnull(path, operation, "path");
    HIP_CHECK(r9700::eager::bf16_gather_path(
        static_cast<hip_bfloat16*>(features.data),
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(path.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<std::uint32_t>(features.ne[0]), static_cast<std::uint32_t>(features.ne[1]),
        static_cast<std::uint32_t>(lanes.ne[0]), static_cast<std::uint32_t>(features.ne[2]),
        stream));
}

void extract_bf16_columns(const Tensor& source, std::int32_t source_column,
                          Tensor& destination, hipStream_t stream) {
    constexpr const char* operation = "extract_bf16_columns";
    require_dtype(source, DType::BF16, operation, "source");
    require_dtype(destination, DType::BF16, operation, "destination");
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || source.ne[2] != 1 || source.ne[3] != 1 ||
        destination.ne[0] <= 0 || destination.ne[1] != source.ne[1] ||
        destination.ne[2] != 1 || destination.ne[3] != 1 || source_column < 0 ||
        source_column > source.ne[0] - destination.ne[0]) {
        throw std::invalid_argument("extract_bf16_columns: invalid rank-2 slice geometry");
    }
    require_contiguous_nonnull(source, operation, "source");
    require_contiguous_nonnull(destination, operation, "destination");
    if (source.data == destination.data) {
        throw std::invalid_argument("extract_bf16_columns: source and destination must not alias");
    }
    HIP_CHECK(r9700::eager::bf16_extract_features(
        static_cast<const hip_bfloat16*>(source.data),
        static_cast<hip_bfloat16*>(destination.data), static_cast<std::uint32_t>(source.ne[0]),
        static_cast<std::uint32_t>(source_column), static_cast<std::uint32_t>(destination.ne[0]),
        static_cast<std::uint32_t>(source.ne[1]), stream));
}

void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, hipStream_t stream) {
    constexpr const char* operation = "cast_fp32_to_bf16";
    require_dtype(source, DType::FP32, operation, "source");
    require_dtype(destination, DType::BF16, operation, "destination");
    require_same_shape(source, destination, operation);
    const std::size_t elements = checked_numel(source, operation, "source");
    (void)checked_numel(destination, operation, "destination");
    require_contiguous_nonnull(source, operation, "source");
    require_contiguous_nonnull(destination, operation, "destination");
    if (source.data == destination.data) {
        throw std::invalid_argument("cast_fp32_to_bf16: source and destination must not alias");
    }
    HIP_CHECK(r9700::eager::cast_fp32_to_bf16(static_cast<const float*>(source.data),
                                               static_cast<hip_bfloat16*>(destination.data),
                                               elements, stream));
}

void cast_bf16_to_fp32(const Tensor& source, Tensor& destination, hipStream_t stream) {
    constexpr const char* operation = "cast_bf16_to_fp32";
    require_dtype(source, DType::BF16, operation, "source");
    require_dtype(destination, DType::FP32, operation, "destination");
    require_same_shape(source, destination, operation);
    const std::size_t elements = checked_numel(source, operation, "source");
    (void)checked_numel(destination, operation, "destination");
    require_contiguous_nonnull(source, operation, "source");
    require_contiguous_nonnull(destination, operation, "destination");
    if (source.data == destination.data) {
        throw std::invalid_argument("cast_bf16_to_fp32: source and destination must not alias");
    }
    HIP_CHECK(r9700::eager::cast_bf16_to_fp32(
        static_cast<const hip_bfloat16*>(source.data), static_cast<float*>(destination.data),
        elements, stream));
}

void residual_add(const Tensor& y, Tensor& x, hipStream_t stream) {
    constexpr const char* operation = "residual_add";
    require_dtype(y, DType::BF16, operation, "y");
    require_dtype(x, DType::BF16, operation, "x");
    require_same_shape(y, x, operation);
    const std::size_t elements = checked_numel(x, operation, "x");
    require_contiguous_nonnull(y, operation, "y");
    require_contiguous_nonnull(x, operation, "x");
    if (y.data == x.data) {
        throw std::invalid_argument("residual_add: y and x must not alias");
    }
    HIP_CHECK(r9700::eager::residual_add_bf16(static_cast<const hip_bfloat16*>(y.data),
                                               static_cast<hip_bfloat16*>(x.data), elements,
                                               stream));
}

void residual_rmsnorm(const Tensor& y, Tensor& x, const Tensor& weight, float eps,
                      Tensor& output, hipStream_t stream) {
    constexpr const char* operation = "residual_rmsnorm";
    require_dtype(y, DType::BF16, operation, "y");
    require_dtype(x, DType::BF16, operation, "x");
    require_same_shape(y, x, operation);
    require_norm(x, output, &weight, nullptr, nullptr, eps, operation);
    require_contiguous_nonnull(y, operation, "y");
    if (y.data == x.data || y.data == output.data || x.data == output.data) {
        throw std::invalid_argument("residual_rmsnorm: mutable outputs must not alias inputs");
    }
    const std::size_t elements = checked_numel(x, operation, "x");
    HIP_CHECK(r9700::eager::residual_rmsnorm_bf16(
        static_cast<const hip_bfloat16*>(y.data), static_cast<hip_bfloat16*>(x.data),
        static_cast<const hip_bfloat16*>(weight.data), static_cast<hip_bfloat16*>(output.data),
        static_cast<std::uint32_t>(x.ne[0]), rows_for(x, elements, operation), eps, stream));
}

void rmsnorm(const Tensor& input, const Tensor& weight, float eps, bool unit_offset,
             Tensor& output, hipStream_t stream) {
    constexpr const char* operation = "rmsnorm";
    require_norm(input, output, &weight, nullptr, nullptr, eps, operation);
    const std::size_t elements = checked_numel(input, operation, "input");
    HIP_CHECK(r9700::eager::rmsnorm_bf16(
        static_cast<const hip_bfloat16*>(input.data),
        static_cast<const hip_bfloat16*>(weight.data), static_cast<hip_bfloat16*>(output.data),
        static_cast<std::uint32_t>(input.ne[0]), rows_for(input, elements, operation), eps,
        unit_offset, stream));
}

void gated_rmsnorm(const Tensor& input, const Tensor& weight, const Tensor& gate,
                   float eps, Tensor& output, hipStream_t stream) {
    constexpr const char* operation = "gated_rmsnorm";
    require_norm(input, output, &weight, nullptr, &gate, eps, operation);
    const std::size_t elements = checked_numel(input, operation, "input");
    HIP_CHECK(r9700::eager::gated_rmsnorm_bf16(
        static_cast<const hip_bfloat16*>(input.data),
        static_cast<const hip_bfloat16*>(weight.data),
        static_cast<const hip_bfloat16*>(gate.data), static_cast<hip_bfloat16*>(output.data),
        static_cast<std::uint32_t>(input.ne[0]), rows_for(input, elements, operation), eps,
        stream));
}

void l2norm(const Tensor& input, float eps, Tensor& output, hipStream_t stream) {
    constexpr const char* operation = "l2norm";
    require_norm(input, output, nullptr, nullptr, nullptr, eps, operation);
    const std::size_t elements = checked_numel(input, operation, "input");
    HIP_CHECK(r9700::eager::l2norm_bf16(
        static_cast<const hip_bfloat16*>(input.data), static_cast<hip_bfloat16*>(output.data),
        static_cast<std::uint32_t>(input.ne[0]), rows_for(input, elements, operation), eps,
        stream));
}

void l2norm_dump(const Tensor& input, float eps, Tensor& output, hipStream_t stream,
                 L2NormDump& dump) {
    constexpr const char* operation = "l2norm_dump";
    require_norm(input, output, nullptr, nullptr, nullptr, eps, operation);
    if (dump.sumsq == nullptr || dump.inv_r == nullptr) {
        throw std::invalid_argument("l2norm_dump: dump buffers must be non-null");
    }
    const std::size_t elements = checked_numel(input, operation, "input");
    HIP_CHECK(r9700::eager::l2norm_bf16_dump(
        static_cast<const hip_bfloat16*>(input.data), static_cast<hip_bfloat16*>(output.data),
        static_cast<std::uint32_t>(input.ne[0]), rows_for(input, elements, operation), eps,
        dump.sumsq, dump.inv_r, stream));
}

void layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps,
                Tensor& output, hipStream_t stream) {
    constexpr const char* operation = "layer_norm";
    require_norm(input, output, &weight, &bias, nullptr, eps, operation);
    const std::size_t elements = checked_numel(input, operation, "input");
    HIP_CHECK(r9700::eager::layernorm_bf16(
        static_cast<const hip_bfloat16*>(input.data),
        static_cast<const hip_bfloat16*>(weight.data),
        static_cast<const hip_bfloat16*>(bias.data), static_cast<hip_bfloat16*>(output.data),
        static_cast<std::uint32_t>(input.ne[0]), rows_for(input, elements, operation), eps,
        stream));
}

void gelu(Tensor& x, GeluMode mode, hipStream_t stream) {
    constexpr const char* operation = "gelu";
    require_dtype(x, DType::BF16, operation, "x");
    const std::size_t elements = checked_numel(x, operation, "x");
    require_contiguous_nonnull(x, operation, "x");
    const RawGeluMode raw_mode = mode == GeluMode::Exact ? RawGeluMode::Exact :
                                  mode == GeluMode::Tanh ? RawGeluMode::Tanh :
                                  throw std::invalid_argument("gelu: invalid mode");
    HIP_CHECK(r9700::eager::gelu_bf16(static_cast<hip_bfloat16*>(x.data), elements, raw_mode,
                                      stream));
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& output, hipStream_t stream) {
    constexpr const char* operation = "silu_mul";
    require_dtype(gate, DType::BF16, operation, "gate");
    require_dtype(up, DType::BF16, operation, "up");
    require_dtype(output, DType::BF16, operation, "output");
    require_same_shape(gate, up, operation);
    require_same_shape(gate, output, operation);
    const std::size_t elements = checked_numel(gate, operation, "gate");
    const std::size_t gate_span = require_valid_strided_nonnull(gate, operation, "gate");
    const std::size_t up_span = require_valid_strided_nonnull(up, operation, "up");
    require_contiguous_nonnull(output, operation, "output");
    if (storage_ranges_overlap(output.data, output.bytes(), gate.data, gate_span) ||
        storage_ranges_overlap(output.data, output.bytes(), up.data, up_span)) {
        throw std::invalid_argument("silu_mul: output must not alias inputs");
    }
    if (gate.is_contiguous() && up.is_contiguous()) {
        HIP_CHECK(r9700::eager::silu_mul_bf16(
            static_cast<const hip_bfloat16*>(gate.data),
            static_cast<const hip_bfloat16*>(up.data),
            static_cast<hip_bfloat16*>(output.data), elements, stream));
        return;
    }
    r9700::eager::SiluMulStridedShape shape{};
    for (int dimension = 0; dimension < 4; ++dimension) {
        shape.extent[dimension] = static_cast<std::uint32_t>(gate.ne[dimension]);
        shape.gate_stride[dimension] = static_cast<std::uint64_t>(
            gate.nb[dimension] / static_cast<std::int64_t>(sizeof(hip_bfloat16)));
        shape.up_stride[dimension] = static_cast<std::uint64_t>(
            up.nb[dimension] / static_cast<std::int64_t>(sizeof(hip_bfloat16)));
    }
    HIP_CHECK(r9700::eager::silu_mul_bf16_strided(
        static_cast<const hip_bfloat16*>(gate.data),
        static_cast<const hip_bfloat16*>(up.data),
        static_cast<hip_bfloat16*>(output.data), elements, shape, stream));
}

void sigmoid_mul(const Tensor& gate, Tensor& x, hipStream_t stream) {
    constexpr const char* operation = "sigmoid_mul";
    require_dtype(gate, DType::BF16, operation, "gate");
    require_dtype(x, DType::BF16, operation, "x");
    require_same_shape(gate, x, operation);
    const std::size_t elements = checked_numel(x, operation, "x");
    require_contiguous_nonnull(gate, operation, "gate");
    require_contiguous_nonnull(x, operation, "x");
    if (gate.data == x.data) {
        throw std::invalid_argument("sigmoid_mul: gate and x must not alias");
    }
    HIP_CHECK(r9700::eager::sigmoid_mul_bf16(static_cast<const hip_bfloat16*>(gate.data),
                                              static_cast<hip_bfloat16*>(x.data), elements,
                                              stream));
}

void embedding(const Tensor& ids, const Weight& table, Tensor& output, hipStream_t stream) {
    require_embedding_common(ids, table, output);
    const auto vocabulary = static_cast<std::uint32_t>(table.shape[0]);
    const auto features = static_cast<std::uint32_t>(table.shape[1]);
    const auto rows = static_cast<std::uint32_t>(ids.ne[0]);
    switch (table.qtype) {
    case QType::BF16_CTRL: {
        if (table.layout != QuantLayout::Contiguous || table.qdata == nullptr ||
            table.qhigh != nullptr || table.high_plane_bytes != 0) {
            throw std::invalid_argument("embedding: malformed contiguous BF16 table");
        }
        const std::uint64_t expected = checked_mul(checked_mul(vocabulary, features, "embedding"),
                                                   sizeof(hip_bfloat16), "embedding");
        if (table.payload_bytes != 0 && table.payload_bytes < expected) {
            throw std::invalid_argument("embedding: BF16 payload is too small");
        }
        HIP_CHECK(r9700::eager::embedding_gather_bf16(
            static_cast<const hip_bfloat16*>(table.qdata),
            static_cast<const std::int32_t*>(ids.data),
            static_cast<hip_bfloat16*>(output.data), vocabulary, features, rows, stream));
    } break;
    case QType::Q4G64_F16S: {
        const auto padded = align_up_i32(table.shape[1], 128, "embedding");
        if (table.layout != QuantLayout::Q4N16K16 || table.group_size != 64 ||
            table.group != 64 ||
            table.scale_dtype != DType::FP16 || table.padded_shape[0] != table.shape[0] ||
            table.padded_shape[1] != padded || table.shape[0] % 16 != 0 ||
            table.qdata == nullptr || table.qhigh != nullptr || table.high_plane_bytes != 0 ||
            table.scales == nullptr) {
            throw std::invalid_argument("embedding: malformed Q4G64_F16S Q4N16K16 table");
        }
        const std::uint64_t code_bytes = checked_mul(
            vocabulary, static_cast<std::uint64_t>(padded / 2), "embedding");
        const std::uint64_t scale_offset = (code_bytes + 255U) / 256U * 256U;
        const std::uint64_t scale_bytes = checked_mul(
            checked_mul(vocabulary, static_cast<std::uint64_t>(padded / 64), "embedding"), 2,
            "embedding");
        if (table.qdata_bytes != code_bytes || table.scale_bytes != scale_bytes ||
            (table.payload_bytes != 0 && table.payload_bytes < scale_offset + scale_bytes)) {
            throw std::invalid_argument("embedding: Q4G64_F16S payload is malformed");
        }
        if (storage_ranges_overlap(output.data, output.bytes(), table.qdata, code_bytes) ||
            storage_ranges_overlap(output.data, output.bytes(), table.scales, scale_bytes) ||
            storage_ranges_overlap(output.data, output.bytes(), ids.data, ids.bytes()) ||
            storage_ranges_overlap(table.qdata, code_bytes, table.scales, scale_bytes)) {
            throw std::invalid_argument("embedding: Q4G64_F16S storage planes overlap");
        }
        HIP_CHECK(r9700::eager::embedding_gather_q4g64_f16s(
            static_cast<const std::uint8_t*>(table.qdata),
            static_cast<const std::uint16_t*>(table.scales),
            static_cast<const std::int32_t*>(ids.data),
            static_cast<hip_bfloat16*>(output.data), vocabulary, features,
            static_cast<std::uint32_t>(padded), rows, stream));
    } break;
    case QType::Q6G64_F16S: {
        const auto padded = align_up_i32(table.shape[1], 128, "embedding");
        if (table.layout != QuantLayout::RowSplit || table.group_size != 64 || table.group != 64 ||
            table.scale_dtype != DType::FP16 || table.padded_shape[0] != table.shape[0] ||
            table.padded_shape[1] != padded || table.qdata == nullptr || table.qhigh == nullptr ||
            table.scales == nullptr) {
            throw std::invalid_argument("embedding: malformed Q6G64_F16S RowSplit table");
        }
        const std::uint64_t groups = static_cast<std::uint64_t>(padded / 64);
        const std::uint64_t low_bytes = checked_mul(checked_mul(vocabulary, groups, "embedding"),
                                                    32, "embedding");
        const std::uint64_t high_bytes = checked_mul(checked_mul(vocabulary, groups, "embedding"),
                                                     16, "embedding");
        const std::uint64_t scale_bytes = checked_mul(checked_mul(vocabulary, groups, "embedding"),
                                                      2, "embedding");
        const std::uint64_t high_offset = (low_bytes + 255U) / 256U * 256U;
        const std::uint64_t scale_offset = high_offset + (high_bytes + 255U) / 256U * 256U;
        if (table.high_plane_bytes < high_bytes ||
            (table.payload_bytes != 0 && table.payload_bytes < scale_offset + scale_bytes)) {
            throw std::invalid_argument("embedding: Q6G64_F16S payload is too small");
        }
        HIP_CHECK(r9700::eager::embedding_gather_q6g64_f16s(
            static_cast<const std::uint8_t*>(table.qdata),
            static_cast<const std::uint8_t*>(table.qhigh),
            static_cast<const std::uint16_t*>(table.scales),
            static_cast<const std::int32_t*>(ids.data),
            static_cast<hip_bfloat16*>(output.data), vocabulary, features,
            static_cast<std::uint32_t>(padded), rows, stream));
    } break;
    case QType::W8G32_F16S: {
        const auto padded = align_up_i32(table.shape[1], 128, "embedding");
        if (table.layout != QuantLayout::RowSplit || table.group_size != 32 || table.group != 32 ||
            table.scale_dtype != DType::FP16 || table.padded_shape[0] != table.shape[0] ||
            table.padded_shape[1] != padded || table.qdata == nullptr || table.qhigh != nullptr ||
            table.high_plane_bytes != 0 || table.scales == nullptr) {
            throw std::invalid_argument("embedding: malformed W8G32_F16S RowSplit table");
        }
        const std::uint64_t code_bytes = checked_mul(vocabulary, static_cast<std::uint64_t>(padded),
                                                     "embedding");
        const std::uint64_t scale_offset = (code_bytes + 255U) / 256U * 256U;
        const std::uint64_t scale_bytes = checked_mul(
            checked_mul(vocabulary, static_cast<std::uint64_t>(padded / 32), "embedding"), 2,
            "embedding");
        if (table.payload_bytes != 0 && table.payload_bytes < scale_offset + scale_bytes) {
            throw std::invalid_argument("embedding: W8G32_F16S payload is too small");
        }
        HIP_CHECK(r9700::eager::embedding_gather_w8g32_f16s(
            static_cast<const std::uint8_t*>(table.qdata),
            static_cast<const std::uint16_t*>(table.scales),
            static_cast<const std::int32_t*>(ids.data),
            static_cast<hip_bfloat16*>(output.data), vocabulary, features,
            static_cast<std::uint32_t>(padded), rows, stream));
    } break;
    default:
        throw std::invalid_argument("embedding: unsupported table encoding");
    }
}

void argmax(const Tensor& logits, Tensor& output, std::int32_t valid_rows,
            hipStream_t stream) {
    constexpr const char* operation = "argmax";
    require_dtype(logits, DType::BF16, operation, "logits");
    require_vector(output, DType::I32, operation, "output");
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1 ||
        output.ne[0] != logits.ne[1] || valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("argmax: invalid [physical_rows,T] geometry");
    }
    require_contiguous_nonnull(logits, operation, "logits");
    if (logits.data == output.data) {
        throw std::invalid_argument("argmax: output must not alias logits");
    }
    HIP_CHECK(r9700::eager::argmax_bf16(
        static_cast<const hip_bfloat16*>(logits.data),
        static_cast<std::int32_t*>(output.data), static_cast<std::uint32_t>(logits.ne[0]),
        static_cast<std::uint32_t>(valid_rows), static_cast<std::uint32_t>(logits.ne[1]),
        stream));
}

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& output,
                       hipStream_t stream) {
    constexpr const char* operation = "mtp_pack_fc_input";
    require_dtype(embedding_norm, DType::BF16, operation, "embedding_norm");
    require_dtype(hidden_norm, DType::BF16, operation, "hidden_norm");
    require_dtype(output, DType::BF16, operation, "output");
    if (embedding_norm.ne[0] != 5120 || embedding_norm.ne[1] <= 0 ||
        embedding_norm.ne[2] != 1 || embedding_norm.ne[3] != 1 ||
        hidden_norm.ne[0] != 5120 || hidden_norm.ne[1] != embedding_norm.ne[1] ||
        hidden_norm.ne[2] != 1 || hidden_norm.ne[3] != 1 || output.ne[0] != 10240 ||
        output.ne[1] != embedding_norm.ne[1] || output.ne[2] != 1 || output.ne[3] != 1) {
        throw std::invalid_argument(
            "mtp_pack_fc_input: expected [5120,T],[5120,T] -> [10240,T]");
    }
    require_contiguous_nonnull(embedding_norm, operation, "embedding_norm");
    require_contiguous_nonnull(hidden_norm, operation, "hidden_norm");
    require_contiguous_nonnull(output, operation, "output");
    if (tensors_overlap(embedding_norm, hidden_norm) ||
        tensors_overlap(embedding_norm, output) || tensors_overlap(hidden_norm, output)) {
        throw std::invalid_argument("mtp_pack_fc_input: tensors must not alias");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(embedding_norm.ne[0]) * sizeof(hip_bfloat16);
    const std::size_t output_pitch = row_bytes * 2U;
    HIP_CHECK(hipMemcpy2DAsync(output.data, output_pitch, embedding_norm.data, row_bytes,
                               row_bytes, static_cast<std::size_t>(embedding_norm.ne[1]),
                               hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(static_cast<std::byte*>(output.data) + row_bytes, output_pitch,
                               hidden_norm.data, row_bytes, row_bytes,
                               static_cast<std::size_t>(hidden_norm.ne[1]),
                               hipMemcpyDeviceToDevice, stream));
}

void mtp_split_attn_in(const Tensor& input, Tensor& query, Tensor& key, Tensor& gate, Tensor& value,
                       hipStream_t stream) {
    constexpr const char* operation = "mtp_split_attn_in";
    require_dtype(input, DType::BF16, operation, "input");
    require_contiguous_nonnull(input, operation, "input");
    if (input.ne[0] != 14336 || input.ne[1] <= 0 || input.ne[2] != 1 || input.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: input must have shape [14336,T]");
    }
    const std::int32_t tokens = input.ne[1];
    const auto require_output = [&](Tensor& tensor, std::int32_t heads,
                                    const char* label) {
        require_dtype(tensor, DType::BF16, operation, label);
        require_contiguous_nonnull(tensor, operation, label);
        if (tensor.ne[0] != 256 || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
            tensor.ne[3] != 1) {
            throw std::invalid_argument(std::string("mtp_split_attn_in: malformed ") + label);
        }
    };
    require_output(query, 24, "query");
    require_output(key, 4, "key");
    require_output(gate, 24, "gate");
    require_output(value, 4, "value");
    const std::array<const Tensor*, 5> tensors{&input, &query, &key, &gate, &value};
    for (std::size_t first = 0; first < tensors.size(); ++first) {
        for (std::size_t second = first + 1; second < tensors.size(); ++second) {
            if (tensors_overlap(*tensors[first], *tensors[second])) {
                throw std::invalid_argument("mtp_split_attn_in: tensors must not alias");
            }
        }
    }
    const std::size_t source_pitch = 14336U * sizeof(hip_bfloat16);
    const auto copy = [&](Tensor& destination, std::size_t source_row,
                          std::size_t rows) {
        const std::size_t width = rows * sizeof(hip_bfloat16);
        HIP_CHECK(hipMemcpy2DAsync(destination.data, width,
                                   static_cast<const std::byte*>(input.data) +
                                       source_row * sizeof(hip_bfloat16),
                                   source_pitch, width, static_cast<std::size_t>(tokens),
                                   hipMemcpyDeviceToDevice, stream));
    };
    copy(query, 0, 6144);
    copy(key, 6144, 1024);
    copy(gate, 7168, 6144);
    copy(value, 13312, 1024);
}

} // namespace ninfer::ops
