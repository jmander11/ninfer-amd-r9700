#pragma once

#include "ops/r9700/eager/r9700_rmsnorm_profile.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::eager {

// These are the small, owned eager building blocks needed before any model math is admitted.
// Every pointer is a device pointer and every nonempty launch requires one explicit HIP stream.
// They intentionally use flat, feature-fastest logical storage beneath the semantic Tensor API.
// All element counts and extents are positive; APIs return hipErrorInvalidValue before launch
// when their host-visible contract is malformed.

// Exact signed I32 movement. `iota` writes start+i; callers must keep every result representable.
[[nodiscard]] hipError_t i32_iota(std::int32_t* destination, std::size_t count,
                                  std::int32_t start, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t i32_fill(std::int32_t* destination, std::size_t count,
                                  std::int32_t value, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t i32_copy(const std::int32_t* source, std::int32_t* destination,
                                  std::size_t count, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t i32_offset(const std::int32_t* source, std::int32_t* destination,
                                    std::size_t count, std::int32_t delta,
                                    hipStream_t stream) noexcept;
[[nodiscard]] hipError_t i32_offset_scalar(const std::int32_t* source,
                                           const std::int32_t* delta,
                                           std::int32_t* destination, std::size_t count,
                                           hipStream_t stream) noexcept;
[[nodiscard]] hipError_t i32_add_scalars(const std::int32_t* lhs, const std::int32_t* rhs,
                                         std::int32_t* destination,
                                         hipStream_t stream) noexcept;
[[nodiscard]] hipError_t i32_increment(std::int32_t* scalar, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t i64_increment(std::int64_t* scalar, hipStream_t stream) noexcept;

// Exact I32 gather. indices[i] addresses source[indices[i]]; the caller supplies values in
// [0, source_count), and source/destination do not overlap.
[[nodiscard]] hipError_t i32_gather(const std::int32_t* source, std::size_t source_count,
                                    const std::int32_t* indices, std::int32_t* destination,
                                    std::size_t count, hipStream_t stream) noexcept;

// Exact BF16 bit copies over feature-fastest columns. Source is [feature_count, columns] and
// destination is [feature_count, destination_columns]. scatter writes only destination columns
// selected by `indices`; gather writes a packed [feature_count, count] output. Indices must be in
// range and scatter indices must be distinct for deterministic state publication.
[[nodiscard]] hipError_t bf16_scatter_columns(const hip_bfloat16* source,
                                               const std::int32_t* indices,
                                               hip_bfloat16* destination,
                                               std::uint32_t feature_count,
                                               std::uint32_t count,
                                               std::uint32_t destination_columns,
                                               hipStream_t stream) noexcept;
[[nodiscard]] hipError_t bf16_gather_columns(const hip_bfloat16* source,
                                              const std::int32_t* indices,
                                              hip_bfloat16* destination,
                                              std::uint32_t feature_count,
                                              std::uint32_t count,
                                              std::uint32_t source_columns,
                                              hipStream_t stream) noexcept;
[[nodiscard]] hipError_t bf16_scatter_batch(const hip_bfloat16* source,
                                            const std::int32_t* lanes,
                                            const std::int32_t* valid_columns,
                                            hip_bfloat16* destination,
                                            std::uint32_t features, std::uint32_t width,
                                            std::uint32_t batch,
                                            std::size_t destination_column_stride,
                                            std::size_t destination_lane_stride,
                                            hipStream_t stream) noexcept;
[[nodiscard]] hipError_t bf16_gather_path(hip_bfloat16* features,
                                          const std::int32_t* lanes,
                                          const std::int32_t* path,
                                          const std::int32_t* counts,
                                          std::uint32_t feature_count,
                                          std::uint32_t width,
                                          std::uint32_t batch,
                                          std::uint32_t lane_count,
                                          hipStream_t stream) noexcept;
[[nodiscard]] hipError_t bf16_extract_features(const hip_bfloat16* source,
                                               hip_bfloat16* destination,
                                               std::uint32_t source_features,
                                               std::uint32_t source_feature,
                                               std::uint32_t destination_features,
                                               std::uint32_t columns,
                                               hipStream_t stream) noexcept;

// BF16 conversion boundaries. FP32-to-BF16 is independent IEEE RNE per finite source element;
// BF16-to-FP32 is exact zero-extension of represented BF16 bits. Source and destination do not
// overlap.
[[nodiscard]] hipError_t cast_fp32_to_bf16(const float* source, hip_bfloat16* destination,
                                            std::size_t count, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t cast_bf16_to_fp32(const hip_bfloat16* source, float* destination,
                                            std::size_t count, hipStream_t stream) noexcept;

// In-place BF16 residual update: x[i] = BF16_RNE(float(x[i]) + float(y[i])). Inputs are finite,
// same-sized represented BF16 values and do not overlap.
[[nodiscard]] hipError_t residual_add_bf16(const hip_bfloat16* y, hip_bfloat16* x,
                                            std::size_t count, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t residual_rmsnorm_bf16(const hip_bfloat16* y, hip_bfloat16* x,
                                                const hip_bfloat16* weight,
                                                hip_bfloat16* output,
                                                std::uint32_t features,
                                                std::uint32_t rows, float eps,
                                                hipStream_t stream) noexcept;
// Rowwise normalization over feature-fastest BF16 [features, rows]. All sums, means, and inverse
// square roots are FP32; only the stated BF16 output is rounded. eps must be finite and positive.
// RMSNorm gain is weight or (1+weight) when unit_offset is true. LayerNorm applies affine BF16
// weight/bias after FP32 mean/variance. Inputs and output do not overlap.
[[nodiscard]] hipError_t rmsnorm_bf16(const hip_bfloat16* input, const hip_bfloat16* weight,
                                      hip_bfloat16* output, std::uint32_t features,
                                      std::uint32_t rows, float eps, bool unit_offset,
                                      hipStream_t stream) noexcept;

[[nodiscard]] constexpr bool rmsnorm_k256_token8_selected(
    std::uint32_t features, std::uint32_t rows) noexcept {
    return features == 256U && rows >= 128U;
}
static_assert(rmsnorm_k256_token8_selected(256U, 128U));
static_assert(!rmsnorm_k256_token8_selected(256U, 127U));
static_assert(!rmsnorm_k256_token8_selected(5120U, 2048U));

// Direct regression boundary for the production K5120 token8 route. It preserves the incumbent
// feature-ascending FP32 FMA chain, unit-offset behavior, and represented BF16 result while
// assigning one independent token to each wave. Ordinary callers use rmsnorm_bf16.
[[nodiscard]] hipError_t rmsnorm_k5120_token8_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t rmsnorm_incumbent_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t features, std::uint32_t rows, float eps, bool unit_offset,
    hipStream_t stream) noexcept;

// Qualification-only direct boundary for evaluating the existing one-row-per-CTA K5120
// implementation at the two C1 DFlash widths not selected by production. Only rows 5 and 6 are
// accepted; ordinary callers continue through rmsnorm_bf16 and its unchanged rows1..4 predicate.
[[nodiscard]] hipError_t rmsnorm_k5120_rows56_cta_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset, hipStream_t stream) noexcept;

// The fixed K5120 ordinary-decode route assigns one row to one 256-thread CTA and reduces the
// FP32 sum of squares across all eight wave32 waves. Other shapes retain the established routes.
[[nodiscard]] constexpr bool rmsnorm_k5120_rows4_selected(
    std::uint32_t features, std::uint32_t rows) noexcept {
    return features == 5120U && rows >= 1U && rows <= 4U;
}
static_assert(rmsnorm_k5120_rows4_selected(5120U, 1U));
static_assert(rmsnorm_k5120_rows4_selected(5120U, 4U));
static_assert(!rmsnorm_k5120_rows4_selected(5120U, 0U));
static_assert(!rmsnorm_k5120_rows4_selected(5120U, 5U));
static_assert(!rmsnorm_k5120_rows4_selected(256U, 4U));
static_assert(is_rmsnorm_k5120_rows56_candidate_eligible(5120U, 5U));
static_assert(is_rmsnorm_k5120_rows56_candidate_eligible(5120U, 6U));
static_assert(!is_rmsnorm_k5120_rows56_candidate_eligible(5120U, 4U));
static_assert(!is_rmsnorm_k5120_rows56_candidate_eligible(5120U, 7U));
// Direct regression boundary for the production fixed-K256 token8 route. It retains the incumbent
// feature-order FP32 FMA chain and BF16 result while assigning one logical row to each wave.
[[nodiscard]] hipError_t rmsnorm_k256_token8_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t layernorm_bf16(const hip_bfloat16* input, const hip_bfloat16* weight,
                                        const hip_bfloat16* bias, hip_bfloat16* output,
                                        std::uint32_t features, std::uint32_t rows, float eps,
                                        hipStream_t stream) noexcept;
[[nodiscard]] hipError_t l2norm_bf16(const hip_bfloat16* input, hip_bfloat16* output,
                                     std::uint32_t features, std::uint32_t rows, float eps,
                                     hipStream_t stream) noexcept;
[[nodiscard]] hipError_t l2norm_bf16_dump(const hip_bfloat16* input,
                                          hip_bfloat16* output,
                                          std::uint32_t features,
                                          std::uint32_t rows, float eps,
                                          float* sumsq, float* inverse,
                                          hipStream_t stream) noexcept;
[[nodiscard]] hipError_t gated_rmsnorm_bf16(const hip_bfloat16* input,
                                             const hip_bfloat16* weight,
                                             const hip_bfloat16* gate,
                                             hip_bfloat16* output,
                                             std::uint32_t features,
                                             std::uint32_t rows, float eps,
                                             hipStream_t stream) noexcept;

// The fixed Qwen3.8-27B GDN output shape uses the admitted token8 route from its lowest measured
// non-regressing prefill extent. Both routes use no caller-owned workspace.
[[nodiscard]] constexpr bool gated_rmsnorm_k6144_token8_selected(
    std::uint32_t features, std::uint32_t rows) noexcept {
    return features == 6144U && rows >= 64U;
}
static_assert(gated_rmsnorm_k6144_token8_selected(6144U, 64U));
static_assert(!gated_rmsnorm_k6144_token8_selected(6144U, 63U));
static_assert(!gated_rmsnorm_k6144_token8_selected(5120U, 2048U));

// Direct regression boundaries for the production K6144 token8 route and its incumbent. The
// token8 route preserves the feature-ordered FP32 reduction and represented BF16 output exactly.
[[nodiscard]] hipError_t gated_rmsnorm_k6144_token8_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* gate,
    hip_bfloat16* output, std::uint32_t rows, float eps, hipStream_t stream) noexcept;
// The flattened ordinary GDN K128 extent uses one wave per independently normalized value-head
// row and one 256-thread CTA per eight rows. Admission is deliberately exact to the four real
// prefill widths qualified on gfx1201; irregular eager/capture widths retain the general route.
[[nodiscard]] constexpr bool gated_rmsnorm_k128_rows8_selected(std::uint32_t features,
                                                               std::uint32_t rows) noexcept {
    if (features != 128U) { return false; }
    return rows == 48U * 1024U || rows == 48U * 2048U || rows == 48U * 4096U ||
           rows == 48U * 8192U;
}
static_assert(gated_rmsnorm_k128_rows8_selected(128U, 48U * 1024U));
static_assert(gated_rmsnorm_k128_rows8_selected(128U, 48U * 2048U));
static_assert(gated_rmsnorm_k128_rows8_selected(128U, 48U * 4096U));
static_assert(gated_rmsnorm_k128_rows8_selected(128U, 48U * 8192U));
static_assert(!gated_rmsnorm_k128_rows8_selected(128U, 48U * 2048U - 8U));
static_assert(!gated_rmsnorm_k128_rows8_selected(256U, 48U * 2048U));
[[nodiscard]] hipError_t gated_rmsnorm_incumbent_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* gate,
    hip_bfloat16* output, std::uint32_t features, std::uint32_t rows, float eps,
    hipStream_t stream) noexcept;

// Elementwise BF16 activation routes: exact/tanh GELU mutates x; SiLU(gate)*up writes output;
// sigmoid_mul mutates x. Every private elementary-function calculation is FP32 before BF16 RNE.
enum class GeluMode : std::uint8_t { Exact, Tanh };
[[nodiscard]] hipError_t gelu_bf16(hip_bfloat16* x, std::size_t count, GeluMode mode,
                                   hipStream_t stream) noexcept;
[[nodiscard]] hipError_t silu_mul_bf16(const hip_bfloat16* gate, const hip_bfloat16* up,
                                       hip_bfloat16* output, std::size_t count,
                                       hipStream_t stream) noexcept;
struct SiluMulStridedShape {
    std::uint32_t extent[4] = {1, 1, 1, 1};
    // Element strides, not byte strides. The output is dense in the extent order.
    std::uint64_t gate_stride[4] = {1, 1, 1, 1};
    std::uint64_t up_stride[4] = {1, 1, 1, 1};
};
[[nodiscard]] constexpr bool silu_mul_split17408_2d_selected(
    std::size_t count, SiluMulStridedShape shape) noexcept {
    constexpr std::uint64_t features = 17408U;
    constexpr std::uint64_t packed_stride = 34816U;
    const std::uint64_t rows = shape.extent[1];
    const std::uint64_t outer_stride = packed_stride * rows;
    return rows >= 128U && shape.extent[0] == features && shape.extent[2] == 1U &&
           shape.extent[3] == 1U && count == features * rows &&
           shape.gate_stride[0] == 1U && shape.up_stride[0] == 1U &&
           shape.gate_stride[1] == packed_stride && shape.up_stride[1] == packed_stride &&
           shape.gate_stride[2] == outer_stride && shape.up_stride[2] == outer_stride &&
           shape.gate_stride[3] == outer_stride && shape.up_stride[3] == outer_stride;
}
[[nodiscard]] hipError_t silu_mul_bf16_strided(
    const hip_bfloat16* gate, const hip_bfloat16* up, hip_bfloat16* output,
    std::size_t count, SiluMulStridedShape shape, hipStream_t stream) noexcept;
// Direct regression boundaries for the production Qwen3.8 MLP split view. The candidate maps the
// feature-fastest [17408,T] output directly onto a two-dimensional grid and preserves the generic
// strided route's FP32 SiLU/multiply and BF16 result.
[[nodiscard]] hipError_t silu_mul_split17408_2d_qualification(
    const hip_bfloat16* gate, const hip_bfloat16* up, hip_bfloat16* output,
    std::uint32_t rows, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t silu_mul_split17408_incumbent_qualification(
    const hip_bfloat16* gate, const hip_bfloat16* up, hip_bfloat16* output,
    std::uint32_t rows, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t sigmoid_mul_bf16(const hip_bfloat16* gate, hip_bfloat16* x,
                                          std::size_t count, hipStream_t stream) noexcept;

// Exact dense BF16 embedding gather. table is feature-fastest [features, vocabulary], ids is I32
// [rows], and output is [features, rows]. Every id must be in [0,vocabulary); output does not
// overlap table or ids. The dense BF16 reference route deliberately does not decode weights.
[[nodiscard]] hipError_t embedding_gather_bf16(const hip_bfloat16* table,
                                                const std::int32_t* ids,
                                                hip_bfloat16* output,
                                                std::uint32_t vocabulary,
                                                std::uint32_t features,
                                                std::uint32_t rows,
                                                hipStream_t stream) noexcept;
[[nodiscard]] hipError_t embedding_gather_q4g64_f16s(
    const std::uint8_t* codes, const std::uint16_t* scales,
    const std::int32_t* ids, hip_bfloat16* output,
    std::uint32_t vocabulary, std::uint32_t features,
    std::uint32_t padded_features, std::uint32_t rows,
    hipStream_t stream) noexcept;
[[nodiscard]] hipError_t embedding_gather_q6g64_f16s(
    const std::uint8_t* low_codes, const std::uint8_t* high_codes,
    const std::uint16_t* scales, const std::int32_t* ids, hip_bfloat16* output,
    std::uint32_t vocabulary, std::uint32_t features,
    std::uint32_t padded_features, std::uint32_t rows,
    hipStream_t stream) noexcept;
[[nodiscard]] hipError_t embedding_gather_w8g32_f16s(
    const std::uint8_t* codes, const std::uint16_t* scales,
    const std::int32_t* ids, hip_bfloat16* output,
    std::uint32_t vocabulary, std::uint32_t features,
    std::uint32_t padded_features, std::uint32_t rows,
    hipStream_t stream) noexcept;

// One deterministic greedy argmax per feature-fastest column. logits is represented finite BF16
// [physical_rows, columns]; only rows [0, valid_rows) participate. Exact ties select the lowest
// row index. `out` is I32 [columns] and cannot overlap logits.
[[nodiscard]] hipError_t argmax_bf16(const hip_bfloat16* logits, std::int32_t* out,
                                     std::uint32_t physical_rows, std::uint32_t valid_rows,
                                     std::uint32_t columns, hipStream_t stream) noexcept;

// The eager no-randomness sampling route is exactly greedy argmax and shares its finite-BF16,
// lowest-token-on-tie contract. It exists as the explicit output-publication boundary.
[[nodiscard]] hipError_t greedy_select_bf16(const hip_bfloat16* logits, std::int32_t* out,
                                            std::uint32_t physical_rows,
                                            std::uint32_t valid_rows,
                                            std::uint32_t columns,
                                            hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::eager
