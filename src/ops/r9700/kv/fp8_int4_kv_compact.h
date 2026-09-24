#pragma once

#include "fp8_int4_kv_append.h"

namespace ninfer::ops::r9700::kv {

struct Fp8Int4KvCompactArgs {
    std::size_t prefix = 0;
    const std::uint32_t* selected_path = nullptr;
    std::size_t selected_count = 0;
    const std::uint32_t* page_table = nullptr;
    std::uint32_t page_table_pages = 0;
    std::size_t physical_tokens = 0;
    std::uint32_t kv_heads = 0;
    std::uint8_t* key_codes = nullptr;
    std::uint8_t* value_codes = nullptr;
    std::uint16_t* value_scale_bits = nullptr;
    std::uint32_t* status = nullptr;
    std::uint32_t value_group = 0;
    Fp8Int4KvPlaneLayout key_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    Fp8Int4KvPlaneLayout value_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    Fp8Int4KvPlaneLayout value_scale_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
};

// Compacts a monotone speculative path in place across the exact three persistent cache planes.
// selected_path[i] must be >= i; the caller validates it, owns the zeroed status word, and only
// commits the new frontier after status remains zero.
[[nodiscard]] hipError_t fp8_int4_kv_compact(const Fp8Int4KvCompactArgs& args,
                                              hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::kv
