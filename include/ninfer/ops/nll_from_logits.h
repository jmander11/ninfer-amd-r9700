#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Per-column teacher-forced negative log likelihood from BF16 logits:
 *
 *   nll[t] = logsumexp_{0 <= v < valid_rows}(float(logits[v,t])) - float(logits[targets[t], t])
 *
 * `logits` is contiguous BF16 [physical_rows,T], `targets` is contiguous I32 [T] with each
 * target in [0, valid_rows), and `out` is contiguous FP32 [T]. 1 <= valid_rows <= physical_rows.
 * Physical rows [valid_rows, physical_rows) do not participate. `out` must not overlap logits or
 * targets. The mathematical oracle evaluates the complete log-sum-exp naively in FP64 from the
 * represented BF16 inputs; production reduction and elementary-function precision are private,
 * and the observable FP32 result is compared to that oracle numerically. The Op has no workspace
 * and changes no state other than writing all of `out`.
 */
void nll_from_logits(const Tensor& logits, const Tensor& targets, Tensor& out,
                     std::int32_t valid_rows, hipStream_t stream);

} // namespace ninfer::ops
