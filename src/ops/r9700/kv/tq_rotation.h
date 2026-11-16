#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <span>

namespace ninfer::ops::r9700::kv {

// The TQ-K4-V3 basis change of tq_k4v3_codec.h on contiguous 256-feature head vectors, in place,
// one wave32 per vector without LDS or barriers (tq::wave_fwht). The forward map
// y = H diag(s) x / 16 takes represented BF16 Q into the rotated basis of the cached K codes (FP32
// transform, one BF16 rounding per element). The inverse x = diag(s) H y / 16 takes FP32
// attention output over rotated V back to the model basis; a +0 or -0 vector stays +0, and a
// nonfinite element poisons only its own vector. `vectors` is rows * query heads.
//
// The schedule fuses these into QK-norm/RoPE and the LQ4 gated output quantizer where it can;
// these standalone forms serve traced stages, the other output consumers, MTP panels and direct
// leaf callers. Rounding profiles of the rotated Q (implementation profiles of one oracle, not
// separate semantics): the fused route stores BF16(R RoPE_fp32) with one rounding, the standalone
// forward rotates the stored BF16 RoPE result, BF16(R BF16(RoPE)). The inverse is FP32 on both
// routes (the fused quantizer rotates back before its BF16 gate seam).
[[nodiscard]] hipError_t tq_rotate_query(hip_bfloat16* query, std::size_t vectors,
                                         hipStream_t stream) noexcept;
[[nodiscard]] hipError_t tq_unrotate_output(float* output, std::size_t vectors,
                                            hipStream_t stream) noexcept;

// One launch over up to kTqRotationMaximumBatch independent buffers (grid y = buffer).
inline constexpr std::size_t kTqRotationMaximumBatch = 16;
[[nodiscard]] hipError_t tq_rotate_query(std::span<hip_bfloat16* const> queries,
                                         std::span<const std::size_t> vectors,
                                         hipStream_t stream) noexcept;
[[nodiscard]] hipError_t tq_unrotate_output(std::span<float* const> outputs,
                                            std::span<const std::size_t> vectors,
                                            hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::kv
