#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear {

// Caller-owned dynamic activation image for the evaluation-only row-scaled E4M3 Linear profile.
// Codes are ordinary token-major [T,K128] OCP E4M3 bytes; scales are FP32 [T] dequantization
// multipliers. The status remains device-resident through the ordered quantize/matmul pair and
// must be consumed by a later device operation before that pair's output can be published.
struct Fp8ActivationWorkspace {
    std::uint8_t* codes           = nullptr;
    std::size_t code_bytes        = 0;
    float* scales                 = nullptr;
    std::size_t scale_bytes       = 0;
    std::uint32_t* status         = nullptr;
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

} // namespace ninfer::ops::r9700::linear
