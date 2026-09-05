#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Depthwise causal width-4 convolution followed by SiLU. Let u[c,-3..-1] be the three values in
 * the input state and u[c,t]=x[c,t] for t>=0. Then
 *
 *   ideal[c,t] = SiLU(sum_{j=0..3} weight[c,j] * u[c,t-3+j]).
 *
 * `x` and `out` are contiguous BF16 [C,T], `weight` is contiguous BF16 [C,4], and a state is
 * contiguous BF16 [C,3] ordered oldest to newest. The supported Qwen3.8 geometry is C=10240 and
 * T may be any positive value. The oracle evaluates `ideal` naively in FP64 from
 * the represented inputs. The BF16 output is promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Kernel
 * accumulator and staging precision are implementation choices. Input, weight, output, and state
 * storage do not overlap except for the explicitly allowed exact alias between state input and
 * state output. No caller workspace is used. Execution is ordered on a non-null HIP stream.
 */

// Reads conv_state as the initial window and replaces it with the final three values after x.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        hipStream_t stream);

// Distinct-state form. conv_state_in and conv_state_out may be disjoint or exactly the same
// storage; conv_state_out receives the trailing width-3 window of concat(conv_state_in,x).
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, hipStream_t stream);

} // namespace ninfer::ops
