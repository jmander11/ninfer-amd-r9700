#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::paired_projection {

struct WeightView {
    const std::uint8_t* codes = nullptr;
    std::size_t code_bytes = 0U;
    const std::uint16_t* scales = nullptr;
    std::size_t scale_bytes = 0U;
};

struct GdnArgs {
    const hip_bfloat16* hidden = nullptr;
    WeightView query_key{};
    WeightView value_z{};
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0U;
    hip_bfloat16* query_key_output = nullptr;
    hip_bfloat16* value_z_output = nullptr;
    std::uint32_t tokens = 0U;
};

struct AttentionArgs {
    const hip_bfloat16* hidden = nullptr;
    WeightView query_key{};
    WeightView gate_value{};
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0U;
    hip_bfloat16* query = nullptr;
    hip_bfloat16* key = nullptr;
    hip_bfloat16* gate = nullptr;
    hip_bfloat16* value = nullptr;
    std::uint32_t tokens = 0U;
};

struct Resources {
    int registers = 0;
    int static_shared_bytes = 0;
    int local_bytes = 0;
    int max_threads_per_block = 0;
};

[[nodiscard]] hipError_t launch_gdn(const GdnArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t launch_attention(const AttentionArgs& args,
                                          hipStream_t stream) noexcept;
[[nodiscard]] hipError_t resources(Resources* out) noexcept;

} // namespace ninfer::ops::r9700::paired_projection
