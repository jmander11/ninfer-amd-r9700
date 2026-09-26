#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime_api.h> // hipStream_t

namespace ninfer::ops {

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], head_dim=256 and rotary_dim=64, or the DFlash full-head
 *   domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered domains are the Qwen3.8 D256/R64 Text Q/K geometry 24/4,
 * D128/R128 1-D DFlash geometry 32/8, and Vision geometry 16/16. q and k must not
 * overlap one another or positions. Disjoint token-interleaved Q/K payloads are legal even when
 * their padded bounding spans intersect. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          hipStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, hipStream_t stream);

/**
 * Qwen3.8 Text Q/K RMSNorm followed by RoPE in one pass. For each head row x[256] of q (24 heads)
 * and k (4 heads) with its gain w (q_norm or k_norm, BF16 [256], unit offset):
 *
 *   n = BF16(x * rsqrt(mean(x^2) + eps) * (w + 1))       (the public normalization seam)
 *   out = rope(n)                                        (the Text 1-D/MRoPE formula above)
 *
 * written to contiguous qn [256,24,T] and kn [256,4,T]. q/k follow rope()'s strided layout (for
 * example planes of the projection output) with 16-byte aligned rows; outputs do not overlap
 * any input. The oracle evaluates both steps in FP64 from the represented inputs with the BF16
 * seam; the RMS reduction association is implementation-defined. No workspace.
 */
void qk_norm_rope(const Tensor& positions, int rotary_dim, float theta, const Tensor& q,
                  const Tensor& k, const Tensor& q_norm, const Tensor& k_norm, float eps,
                  Tensor& qn, Tensor& kn, hipStream_t stream);

} // namespace ninfer::ops
