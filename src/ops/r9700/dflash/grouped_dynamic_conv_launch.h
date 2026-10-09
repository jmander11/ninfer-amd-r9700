#pragma once

#include "core/tensor.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops::detail {

// image, when non-null, receives the A8G64 image of the published values; its status word is
// published through the two completion words (r9700::linear::codec::a8g64_complete_status).
void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& base_kernel,
                                         const Tensor& projection, Tensor* prepared,
                                         Tensor& finish_dynamic,
                                         const r9700::linear::A8G64ActivationWorkspace* image,
                                         std::uint32_t* completion, hipStream_t stream);

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& base_kernel,
                                        const Tensor& finish_dynamic, Tensor& residual,
                                        hipStream_t stream);

void grouped_dynamic_conv_finish_normalized_launch(
    const Tensor& hidden, const Tensor& base_kernel, const Tensor& finish_dynamic,
    const Tensor& residual_in, Tensor& residual_out, const Tensor& norm, float eps,
    Tensor& normalized, const r9700::linear::A8G64ActivationWorkspace* image,
    std::uint32_t* completion, hipStream_t stream);

} // namespace ninfer::ops::detail
