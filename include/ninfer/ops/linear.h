#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Applies the sole R9700 bias-free matrix projection.
 *
 * The mathematical result is out[n,t] = sum_k dequantize(w[n,k]) * x[k,t].
 * `x` is contiguous BF16 [K,T], `w` is contiguous BF16_CTRL or canonical
 * Q4G64_F16S Q4N16K16 or W8G32_F16S RowSplit [N,K]. The fixed N248320/K5120
 * output head additionally supports resident W8N16K16 codes/scales without repacking.
 * `out` is contiguous BF16 [N,T]. Dimension
 * zero is stored fastest, so physical activation/output storage is token-major.
 * The implementation accumulates in FP32 and rounds once to BF16 output; the
 * independent qualification oracle decodes the represented weight and evaluates
 * the complete dot product in FP64.
 *
 * T is any positive value. Decode T=1..8 and prefill T>=9 use separately
 * qualified gfx1201 dispatches. The W8 code and FP16 scale planes are consumed
 * directly from artifact storage, including target-owned row views. Q4 consumes
 * canonical signed A4G64 or compile-time-evaluation A8G64 activation codes and
 * FP16 scales in caller-owned serialized activation storage. A separately compiled W8 evaluator
 * similarly uses signed A8G32 at physically measured shape-specific crossovers while retaining
 * exact BF16xW8 below them and for shapes outside the qualified mixed-artifact inventory. BF16 and
 * the default exact W8 route use no workspace. No route performs hidden allocation or runtime
 * weight repacking. Inputs, output, workspace, and weight planes must not overlap.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t tokens,
                                                          std::int32_t columns);
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
            hipStream_t stream);

// Executes an integer linear against caller-owned serialized workspace. The span may
// be larger than the exact requirement for this shape; the activation image consumes the
// required prefix; no partial-sum storage follows it.
// BF16 and exact-W8 routes ignore the span. This boundary lets a Program keep graph addresses
// stable without reserving private activation storage inside each schedule's WorkspaceArena.
void linear(const Tensor& x, const Weight& w, Tensor& out, const DeviceSpan& activation_workspace,
            hipStream_t stream);

// Workspace-free boundary retained for BF16/exact-W8 control and qualification routes. It rejects
// Q4 rather than allocating hidden activation storage; product execution uses the DeviceSpan
// overload while isolated qualifiers may use the arena overload.
void linear(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);

/**
 * Q4 activation image: the exact signed-A8G64 code planes, FP16 group scales, and status word of a
 * represented BF16 x[K,T] that the Q4G64_F16S projection evaluates, bound for (T, K) at the base
 * of a caller-owned span of at least q4_activation_image_bytes(T, K). The A8 codec is explicit:
 * it does not follow the compile-time Q4 activation-width evaluator. A producer quantizes x once
 * (quantize_q4_activation_image, or a fused producer Op documented to publish this image of its
 * BF16 output) and any number of Q4G64_F16S N16K16 projections with K columns then read it
 * through linear_q4_activation_image, which publishes BF16 out[N,T] with the route
 * linear() selects for the shape. A nonfinite input or an FP16 group-scale overflow sets the
 * status word, and the projection then publishes BF16 NaN. The span must not overlap any weight
 * plane or output; the image stays valid until the span is rewritten.
 */
[[nodiscard]] std::size_t q4_activation_image_bytes(std::int32_t tokens, std::int32_t columns);
void quantize_q4_activation_image(const Tensor& x, const DeviceSpan& image, hipStream_t stream);
void linear_q4_activation_image(const Weight& w, Tensor& out, const DeviceSpan& image,
                                hipStream_t stream);
// RMSNorm normalized = BF16(x * rsqrt(mean(x^2) + eps) * (norm + unit_offset)) of BF16 x[5120,T],
// published and encoded as the image of normalized (the BF16 seam ops::rmsnorm publishes).
void rmsnorm_q4_activation_image(const Tensor& x, const Tensor& norm, float eps, bool unit_offset,
                                 Tensor& normalized, const DeviceSpan& image, hipStream_t stream);

/**
 * Destination of a fused multi-block producer's image: the span, and two caller-owned device
 * words through which the producer's last finishing block publishes the status word, so no reset
 * launch precedes it. The words are zero before first use, every launch returns them to zero, and
 * producers using them are serialized on one stream.
 */
struct Q4ActivationImageTarget {
    DeviceSpan image;
    std::uint32_t* completion = nullptr;
};

// The image of BF16(SiLU(gate) * up) for gate_up BF16 [2 * 17408, T] (rows [0,17408) gate,
// [17408, 34816) up of each column), the value ops::silu_mul publishes, which is not materialized.
void silu_mul_q4_activation_image(const Tensor& gate_up, const Q4ActivationImageTarget& target,
                                  hipStream_t stream);
} // namespace ninfer::ops
