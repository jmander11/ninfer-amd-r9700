#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   ideal[i] = x[i] * (1 / (1 + exp(-gate[i]))).
 *
 * `gate` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. The oracle evaluates
 * `ideal` in FP64 from the represented inputs. The updated BF16 x is promoted and compared directly
 * with that result; output storage rounding belongs to the Op's numerical criterion, not the
 * oracle. Private kernel arithmetic is implementation-defined. The Op uses no workspace or other
 * persistent state.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, hipStream_t stream);

/**
 * Same gate applied to an FP32 source with an explicit BF16 cast boundary:
 *
 *   ideal[i] = BF16(x[i]) * (1 / (1 + exp(-gate[i]))),
 *
 * written to the BF16 `out`. `x` is contiguous FP32 and `gate`/`out` contiguous BF16 of the same
 * shape; the three do not overlap. It equals cast_fp32_to_bf16 followed by sigmoid_mul.
 */
void sigmoid_mul(const Tensor& gate, const Tensor& x, Tensor& out, hipStream_t stream);

} // namespace ninfer::ops
