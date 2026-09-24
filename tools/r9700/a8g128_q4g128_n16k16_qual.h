#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Disconnected qualification-only A8G128 x Q4G128 surface.  Nothing in this
// header is a product format or dispatch contract.
struct A8G128Workspace {
    std::uint8_t* low = nullptr;
    std::uint8_t* high = nullptr;
    std::uint16_t* scales = nullptr;
    std::uint32_t* status = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t columns = 0;
};
struct A8G128Q4G128QualificationArgs {
    const hip_bfloat16* input = nullptr;
    std::uint8_t* activation_low = nullptr;
    std::uint8_t* activation_high = nullptr;
    std::uint16_t* activation_scales = nullptr;
    std::uint32_t* status = nullptr;
    const std::uint64_t* weight_pairs = nullptr;
    const std::uint16_t* weight_scales = nullptr;
    hip_bfloat16* output = nullptr;
    std::size_t activation_plane_bytes = 0, activation_scale_bytes = 0;
    std::size_t weight_code_bytes = 0, weight_scale_bytes = 0, output_bytes = 0;
    std::uint32_t tokens = 0, rows = 0, columns = 0;
    bool fused_silu = false;
};
hipError_t a8g128_q4g128_qualification_run(
    const A8G128Q4G128QualificationArgs&, hipStream_t) noexcept;

extern "C" __global__ __launch_bounds__(32) void a8g128_q4g128_quantize(
    const hip_bfloat16*, std::uint8_t*, std::uint8_t*, std::uint16_t*,
    std::uint32_t*, std::uint32_t, std::uint32_t);
extern "C" __global__ __launch_bounds__(32) void fused_silu_a8g128_q4g128_quantize(
    const hip_bfloat16*, std::uint8_t*, std::uint8_t*, std::uint16_t*,
    std::uint32_t*, std::uint32_t, std::uint32_t);
extern "C" __global__ __launch_bounds__(512) void a8g128_q4g128_n16k16_prefill(
    const std::uint8_t*, const std::uint8_t*, const std::uint16_t*,
    const std::uint32_t*, const std::uint64_t*, const std::uint16_t*,
    hip_bfloat16*, std::uint32_t, std::uint32_t, std::uint32_t);
extern "C" __global__ __launch_bounds__(32) void a8g128_q4g128_n16k16_wave(
    const std::uint8_t*, const std::uint8_t*, const std::uint16_t*,
    const std::uint32_t*, const std::uint64_t*, const std::uint16_t*,
    hip_bfloat16*, std::uint32_t, std::uint32_t, std::uint32_t);
