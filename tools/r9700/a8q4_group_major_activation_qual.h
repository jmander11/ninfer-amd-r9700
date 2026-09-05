#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Qualification-only private A8G64 workspace experiment.  It deliberately
// does not define a product layout or alter production dispatch.
struct A8Q4GroupMajorArgs {
    const hip_bfloat16* input = nullptr;
    std::size_t input_bytes = 0;
    std::uint8_t* activation_low = nullptr;
    std::uint8_t* activation_high = nullptr;
    std::uint16_t* activation_scales = nullptr;
    std::uint32_t* activation_status = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    const std::uint16_t* weight_scales = nullptr;
    hip_bfloat16* output = nullptr;
    std::size_t activation_plane_bytes = 0;
    std::size_t activation_scale_bytes = 0;
    std::size_t weight_code_bytes = 0;
    std::size_t weight_scale_bytes = 0;
    std::size_t output_bytes = 0;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    bool fused_silu = false;
};

hipError_t a8q4_group_major_activation_run(const A8Q4GroupMajorArgs&, hipStream_t) noexcept;
