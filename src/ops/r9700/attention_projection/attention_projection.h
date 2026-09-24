#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::attention_projection {

struct Args {
    const hip_bfloat16* hidden = nullptr;
    const std::uint8_t* query_key_codes = nullptr;
    std::size_t query_key_code_bytes = 0;
    const std::uint16_t* query_key_scales = nullptr;
    std::size_t query_key_scale_bytes = 0;
    const std::uint8_t* gate_value_codes = nullptr;
    std::size_t gate_value_code_bytes = 0;
    const std::uint16_t* gate_value_scales = nullptr;
    std::size_t gate_value_scale_bytes = 0;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    hip_bfloat16* query = nullptr;
    hip_bfloat16* key = nullptr;
    hip_bfloat16* gate = nullptr;
    hip_bfloat16* value = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t columns = 0;
};

struct Resources {
    int registers = 0;
    int static_shared_bytes = 0;
    int local_bytes = 0;
    int max_threads_per_block = 0;
};

[[nodiscard]] hipError_t launch(const Args& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t resources(Resources* out) noexcept;

} // namespace ninfer::ops::r9700::attention_projection
