#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Prepares Gated DeltaNet decay and update gates:
 *
 *   g[h,t]    = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `a` and `b` are contiguous BF16 [48,T], `A_log` and `dt_bias` are contiguous FP32 [48], and
 * `g` and `beta` are contiguous FP32 [48,T]. The oracle evaluates the formula naively in FP64;
 * transcendental implementation and intermediate precision are private kernel choices. Inputs and
 * the two outputs must be mutually non-overlapping. There is no workspace or persistent state side
 * effect.
 */
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, hipStream_t stream);

/**
 * Op: BF16 projected Gated DeltaNet controls.
 *
 * Math / indexing:
 *   ar[h,t] = BF16(sum_k hidden[k,t] * a_weight[h,k])
 *   br[h,t] = BF16(sum_k hidden[k,t] * b_weight[h,k])
 *   g[h,t] = -exp(A_log[h]) * softplus(FP32(ar[h,t]) + dt_bias[h])
 *   beta[h,t] = sigmoid(FP32(br[h,t])).
 *
 * Logical shapes and supported domain:
 *   hidden is contiguous BF16 [5120,T], both weights are contiguous BF16_CTRL [48,5120],
 *   A_log/dt_bias are contiguous FP32 [48], and g/beta are contiguous FP32 [48,T], T>=1.
 *
 * Numeric / effects:
 *   Each dot product accumulates in FP32 and has an observable explicit BF16 rounding boundary
 *   before the control formula. Only g and beta are published. All inputs and outputs are
 *   mutually non-overlapping; there is no workspace or persistent state effect.
 *
 * Execution:
 *   The caller supplies a non-null stream. T=1..24 use one batched grid that applies the same
 *   per-token projection arithmetic to every token (bitwise equal to T1 per column); wider
 *   extents use a batched WMMA route whose FP32 reduction association differs, so only the
 *   BF16-rounded projection boundary, not bitwise equality with T1, is shared across the two
 *   routes. Hidden and weight storage are 16-byte aligned.
 */
void bf16_gdn_projected_gating(const Tensor& hidden, const Weight& a_weight,
                                  const Weight& b_weight, const Tensor& A_log,
                                  const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                  hipStream_t stream);

} // namespace ninfer::ops
