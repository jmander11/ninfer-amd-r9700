#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops::detail {

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& base_kernel,
                                         const Tensor& projection, Tensor& prepared,
                                         Tensor& finish_dynamic, hipStream_t stream);

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& base_kernel,
                                        const Tensor& finish_dynamic, Tensor& out,
                                        hipStream_t stream);

} // namespace ninfer::ops::detail
