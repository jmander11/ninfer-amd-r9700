#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Applies the sole R9700 bias-free matrix projection.
 *
 * The mathematical result is out[n,t] = sum_k dequantize(w[n,k]) * x[k,t].
 * `x` is contiguous BF16 [K,T], `w` is contiguous BF16_CTRL or canonical
 * Q4G64_F16S Q4N16K16 or W8G32_F16S RowSplit [N,K], and `out` is contiguous BF16 [N,T]. Dimension
 * zero is stored fastest, so physical activation/output storage is token-major.
 * The implementation accumulates in FP32 and rounds once to BF16 output; the
 * independent qualification oracle decodes the represented weight and evaluates
 * the complete dot product in FP64.
 *
 * T is any positive value. Decode T=1..8 and prefill T>=9 use separately
 * qualified gfx1201 dispatches. The W8 code and FP16 scale planes are consumed
 * directly from artifact storage, including target-owned row views. Q4 consumes
 * canonical signed A4G64 or compile-time-evaluation A8G64 activation codes and
 * FP16 scales in caller-owned serialized activation storage. A separately compiled W8 evaluator similarly uses
 * signed A8G32 at physically measured shape-specific crossovers while retaining exact BF16xW8
 * below them and for shapes outside the qualified mixed-artifact inventory.
 * BF16 and the default exact W8 route use no workspace. No route performs hidden allocation or
 * runtime weight repacking. Inputs, output, workspace, and weight planes must not overlap.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype,
                                                           std::int32_t tokens,
                                                           std::int32_t columns);
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
            hipStream_t stream);

// Executes an integer linear against a caller-owned serialized activation region. The span may
// be larger than the exact image required by this shape; only the required prefix is consumed.
// BF16 and exact-W8 routes ignore the span. This boundary lets a Program keep graph addresses
// stable without reserving private activation storage inside each schedule's WorkspaceArena.
void linear(const Tensor& x, const Weight& w, Tensor& out,
            const DeviceSpan& activation_workspace, hipStream_t stream);

// Workspace-free boundary retained for BF16/exact-W8 control and qualification routes. It rejects
// Q4 rather than allocating hidden activation storage; product execution uses the DeviceSpan
// overload while isolated qualifiers may use the arena overload.
void linear(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);

} // namespace ninfer::ops
