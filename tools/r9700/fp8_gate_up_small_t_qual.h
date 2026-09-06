#pragma once

#include "ops/r9700/linear/fp8_activation.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::qualification::fp8_gate_up_small_t {

inline constexpr std::uint32_t kRows = 34816;
inline constexpr std::uint32_t kColumns = 5120;
inline constexpr std::uint32_t kThreads = 128;
inline constexpr std::uint32_t kRowsPerWave = 64;
inline constexpr std::uint32_t kRowsPerBlock = 256;

struct Arguments {
    const hip_bfloat16* input_bf16 = nullptr;       // token-major [T,5120]
    const std::uint8_t* weight_codes = nullptr;    // row-major [34816,5120]
    const float* weight_scales = nullptr;           // FP32 [34816]
    ops::r9700::linear::Fp8ActivationWorkspace activation{};
    hip_bfloat16* output_bf16 = nullptr;            // token-major [T,34816]
    std::uint32_t tokens = 0;
    std::uint32_t rows = kRows;
    std::uint32_t columns = kColumns;
};

struct Resources {
    int registers = 0;
    int static_lds_bytes = 0;
    int private_bytes = 0;
    int maximum_threads = 0;
    int active_blocks_per_cu = 0;
};

// Qualification-only complete BF16 -> row-E4M3 quantize -> FP8 WMMA -> BF16 route.
// Exactly T=5/6 and N34816/K5120 are admitted. This is deliberately disconnected from
// production Linear dispatch.
[[nodiscard]] hipError_t launch(const Arguments& arguments, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t resources(Resources* output) noexcept;

} // namespace ninfer::qualification::fp8_gate_up_small_t
