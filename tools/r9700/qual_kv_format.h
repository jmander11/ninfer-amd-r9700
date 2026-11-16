#pragma once

#include "ninfer/types.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace ninfer::qual {

// Qualifiers exercise one cache format per run; unset follows the production Engine default.
// NINFER_QUAL_KV_CACHE selects an explicit codec. Production never reads this variable.
[[nodiscard]] inline KvCacheFormat kv_cache_format_from_environment() {
    const char* const selected = std::getenv("NINFER_QUAL_KV_CACHE");
    if (selected == nullptr) { return EngineOptions{}.kv_cache_format; }
    if (std::string_view(selected) == "fp8-int4") { return KvCacheFormat::Fp8KInt4V; }
    if (std::string_view(selected) == "nvfp4") { return KvCacheFormat::Nvfp4; }
    if (std::string_view(selected) == "tq-k4-v3") { return KvCacheFormat::TqK4V3; }
    throw std::invalid_argument("NINFER_QUAL_KV_CACHE must be fp8-int4, nvfp4 or tq-k4-v3");
}

} // namespace ninfer::qual
