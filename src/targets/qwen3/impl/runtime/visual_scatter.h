#pragma once

// Identity-free Qwen3 family runtime helper.

#include "core/arena.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3/mtp_alignment.h>

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3::detail {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const qwen3::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, hipStream_t stream);

} // namespace ninfer::targets::qwen3::detail
