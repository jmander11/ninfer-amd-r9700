#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace ninfer::qualification::fp8_gate_up_m128n128 {

inline constexpr std::uint32_t kTokens = 2048;
inline constexpr std::uint32_t kRows = 34816;
inline constexpr std::uint32_t kColumns = 5120;
inline constexpr std::uint32_t kTileM = 128;
inline constexpr std::uint32_t kTileN = 128;
inline constexpr std::uint32_t kTileK = 32;
inline constexpr std::uint32_t kThreads = 512;
inline constexpr std::uint32_t kCtas = (kTokens / kTileM) * (kRows / kTileN);
inline constexpr std::size_t kStaticLdsBytes = 2U * (kTileM + kTileN) * kTileK;

struct Arguments {
    const std::uint8_t* activation_codes = nullptr; // row-major [2048,5120]
    const float* token_scales = nullptr;            // [2048]
    const std::uint8_t* weight_codes = nullptr;     // row-major [34816,5120]
    const float* weight_scales = nullptr;            // [34816]
    std::uint16_t* output_bf16 = nullptr;             // row-major [2048,34816]
    std::uint32_t tokens = kTokens;
    std::uint32_t rows = kRows;
    std::uint32_t columns = kColumns;
};

struct Resources {
    int registers = 0;
    int static_lds_bytes = 0;
    int private_bytes = 0;
    int maximum_threads = 0;
};

// Qualification-only exact-shape launch. It performs only the represented FP8
// matrix and outer-scale operation. Production status/poison remains a separate
// complete-LinearExecution boundary and is exercised by the qualifier harness.
hipError_t launch(const Arguments& arguments, hipStream_t stream);
hipError_t resources(Resources* output);

} // namespace ninfer::qualification::fp8_gate_up_m128n128
