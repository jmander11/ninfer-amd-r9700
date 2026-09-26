#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear {

// Caller-owned dynamic per-token E4M3 activation image. Codes are token-major [T,K128] OCP E4M3
// bytes; scales are FP32 [T] dequantization multipliers; status holds one word per token, written
// by the producer of that token row (no reset), nonzero when the represented row has a nonfinite
// value. Consumers publish the canonical BF16 quiet NaN for every output element of a flagged
// token (the rows a nonfinite input would poison in unquantized arithmetic).
struct Fp8ActivationWorkspace {
    std::uint8_t* codes           = nullptr;
    std::size_t code_bytes        = 0;
    float* scales                 = nullptr;
    std::size_t scale_bytes       = 0;
    std::uint32_t* status         = nullptr;  // [tokens]
    std::uint32_t tokens          = 0;
    std::uint32_t columns         = 0;
    std::uint32_t padded_columns = 0;
};

enum Fp8ActivationStatus : std::uint32_t {
    Fp8ActivationOk        = 0U,
    Fp8ActivationNonfinite = 1U << 0U,
};

struct Fp8ActivationQuantizeArgs {
    // This represented BF16 [T,K] tensor is the sole public input rounding boundary.
    const hip_bfloat16* input = nullptr;
    Fp8ActivationWorkspace workspace{};
};

[[nodiscard]] std::size_t fp8_activation_workspace_capacity_bytes(
    std::uint32_t tokens, std::uint32_t columns) noexcept;
[[nodiscard]] hipError_t
fp8_bind_activation_workspace(void* storage, std::size_t storage_bytes, std::uint32_t tokens,
                              std::uint32_t columns, Fp8ActivationWorkspace* out) noexcept;
[[nodiscard]] hipError_t
fp8_quantize_activation(const Fp8ActivationQuantizeArgs& args, hipStream_t stream) noexcept;

// Fused producers (any T). Each quantizes exactly the BF16 tensor its unfused producer would
// publish (which is never materialized):
// - RMSNorm of x [T,5120]: BF16(x * rsqrt(mean(x^2) + eps) * (w + unit_offset)), with the FP32
//   reduction order of the eager K5120 small-T RMSNorm;
// - output gate: BF16(BF16(attention) / (1 + exp(-gate))) of FP32 attention and BF16 gate [T,K].
struct Fp8NormalizedQuantizeArgs {
    const hip_bfloat16* input  = nullptr;
    const hip_bfloat16* weight = nullptr;
    float eps                  = 0.0F;
    bool unit_offset           = false;
    Fp8ActivationWorkspace workspace{};
};
[[nodiscard]] hipError_t fp8_quantize_normalized_activation(
    const Fp8NormalizedQuantizeArgs& args, hipStream_t stream) noexcept;

struct Fp8GatedQuantizeArgs {
    const hip_bfloat16* gate = nullptr;
    const float* attention   = nullptr;
    Fp8ActivationWorkspace workspace{};
};
[[nodiscard]] hipError_t fp8_quantize_gated_activation(const Fp8GatedQuantizeArgs& args,
                                                       hipStream_t stream) noexcept;

// GDN gated per-head RMSNorm of x [T, 48 x 128] with gate z and head weight w [128]:
// BF16(x * rsqrt(mean_head(x^2) + eps) * w * silu(z)), the head sum of squares reduced over the
// sixteen eight-feature vectors of the head.
struct Fp8GatedRmsNormQuantizeArgs {
    const hip_bfloat16* input  = nullptr;
    const hip_bfloat16* gate   = nullptr;
    const hip_bfloat16* weight = nullptr;
    float eps                  = 0.0F;
    Fp8ActivationWorkspace workspace{};
};
[[nodiscard]] hipError_t fp8_quantize_gated_rmsnorm_activation(
    const Fp8GatedRmsNormQuantizeArgs& args, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::linear
