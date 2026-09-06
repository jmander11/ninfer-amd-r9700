#pragma once

#include "ops/r9700/linear/r9700_linear.h"

[[nodiscard]] hipError_t a8q4_n16k16_scalar_base_qualification(
    const ninfer::ops::r9700::linear::A8Q4G64LinearArgs& args,
    hipStream_t stream) noexcept;
