#pragma once

#include "ninfer/types.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace ninfer::qual {

// Qualifiers exercise one cache format per run: NINFER_QUAL_KV_CACHE=nvfp4 selects the opt-in
// codec. Production options never read this variable.
[[nodiscard]] inline KvCacheFormat kv_cache_format_from_environment() {
    const char* const selected = std::getenv("NINFER_QUAL_KV_CACHE");
    if (selected == nullptr || std::string_view(selected) == "fp8-int4") {
        return KvCacheFormat::Fp8KInt4V;
    }
    if (std::string_view(selected) == "nvfp4") { return KvCacheFormat::Nvfp4; }
    throw std::invalid_argument("NINFER_QUAL_KV_CACHE must be fp8-int4 or nvfp4");
}

} // namespace ninfer::qual
