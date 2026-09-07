#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::kv {

// Diagnostic-only canonical copy of logical positions 0..129 from the fixed R9700 G16 cache.
// Outputs are logical-position-major and never feed inference.
struct Fp8Int4KvTraceArgs {
    const std::uint8_t* key_codes = nullptr;
    const std::uint8_t* value_codes = nullptr;
    const std::uint16_t* value_scale_bits = nullptr;
    const std::uint32_t* page_table = nullptr;
    const std::int32_t* page_table_row = nullptr;
    std::uint32_t page_table_pages = 0;
    std::uint32_t page_table_row_stride = 0;
    std::uint32_t page_table_row_count = 0;
    std::size_t physical_tokens = 0;
    std::uint8_t* canonical_keys = nullptr;
    std::uint8_t* canonical_values = nullptr;
    std::uint16_t* canonical_scales = nullptr;
    std::uint32_t* status = nullptr;
};

[[nodiscard]] hipError_t fp8_int4_kv_trace_positions0_129(
    const Fp8Int4KvTraceArgs& args, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::kv
