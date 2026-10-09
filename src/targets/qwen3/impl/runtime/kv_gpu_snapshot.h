#pragma once

#include <cstdint>

namespace ninfer::targets::qwen3::detail {

struct KvGpuPoolSnapshot {
    std::uint32_t page_group_count = 0;
    std::uint32_t entitled_pages   = 0;
    std::uint32_t mapped_pages     = 0;
    std::uint32_t free_pages       = 0;
};

// Host-side page counters of the device KV pools. main is the FP8-K/INT4-V Text pool; spec is the
// speculative backend's paged pool and stays zero when the backend has none (DFlash2 local state).
struct KvGpuSnapshot {
    KvGpuPoolSnapshot main;
    KvGpuPoolSnapshot spec;
};

} // namespace ninfer::targets::qwen3::detail
