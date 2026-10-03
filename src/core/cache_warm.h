#pragma once

#include "core/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#    include <hip/hip_runtime.h>
#endif

namespace ninfer {

// Read-only device bytes that a latency-bound kernel touches with CTAs past its work grid, so
// the bandwidth-bound kernel launched after it finds the head of its stream in L2/Infinity Cache
// instead of waiting on DRAM. A pure cache hint: results never depend on it.
struct CacheWarmRange {
    const void* data  = nullptr;
    std::size_t bytes = 0;
};

struct CacheWarm {
    CacheWarmRange ranges[2]{};

    [[nodiscard]] bool empty() const noexcept {
        return ranges[0].bytes == 0U && ranges[1].bytes == 0U;
    }
};

// Bytes [offset, offset + bytes) of a weight's row-tile-major code stream and the same fraction of
// its high plane (FP8LUT4 group codes), when it has one.
[[nodiscard]] inline CacheWarm weight_warm(const Weight& weight, std::size_t offset,
                                           std::size_t bytes) noexcept {
    CacheWarm warm{};
    if (weight.qdata == nullptr || offset >= weight.qdata_bytes) return warm;
    bytes          = std::min<std::size_t>(bytes, weight.qdata_bytes - offset);
    warm.ranges[0] = {static_cast<const std::uint8_t*>(weight.qdata) + offset, bytes};
    if (weight.qhigh != nullptr && weight.high_plane_bytes != 0U) {
        // The same row fraction of the high plane, clamped to it.
        const auto scaled = [&](std::size_t value) {
            return static_cast<std::size_t>(static_cast<unsigned __int128>(value) *
                                            weight.high_plane_bytes / weight.qdata_bytes);
        };
        const std::size_t high_offset = scaled(offset) / 4U * 4U; // dword touches
        warm.ranges[1]                = {
            static_cast<const std::uint8_t*>(weight.qhigh) + high_offset,
            std::min<std::size_t>(scaled(bytes), weight.high_plane_bytes - high_offset)};
    }
    return warm;
}

// One dword per 256-byte line fills the line on gfx1201 (denser touches warm no more bytes).
inline constexpr std::size_t kCacheWarmStride = 256U;

// Warm CTAs added to `work_ctas` CTAs of `threads` threads: one touched line per thread. A grid
// of about 2016..2048 waves pays a ~30 us dispatch penalty on gfx1201 (a 256-thread grid of 253
// to 256 CTAs costs 34 us empty, 252 or 257 costs 3 us), so the total steps past it.
[[nodiscard]] inline std::uint32_t cache_warm_ctas(const CacheWarm& warm, std::uint32_t threads,
                                                   std::uint32_t work_ctas) noexcept {
    std::size_t lines = 0U;
    for (const CacheWarmRange& range : warm.ranges)
        if (range.data != nullptr) lines += range.bytes / kCacheWarmStride;
    if (lines == 0U) return 0U;
    auto ctas = static_cast<std::uint32_t>((lines + threads - 1U) / threads);
    const std::uint32_t waves_per_cta = (threads + 31U) / 32U;
    const auto waves = [&] { return static_cast<std::size_t>(work_ctas + ctas) * waves_per_cta; };
    while (waves() >= 2016U && waves() <= 2048U) ++ctas;
    return ctas;
}

#if defined(__HIPCC__)
// Called by thread `thread` of `threads` of CTA `cta` of `ctas` warm CTAs (the ones past the
// work grid); the lines are strided over every warm thread whatever the CTA width.
__device__ inline void warm_cache(const CacheWarm& warm, std::uint32_t cta, std::uint32_t ctas,
                                  std::uint32_t thread, std::uint32_t threads) {
    std::uint32_t sink = 0U;
    std::size_t line   = static_cast<std::size_t>(cta) * threads + thread;
    for (const CacheWarmRange& range : warm.ranges) {
        const std::size_t lines = range.data == nullptr ? 0U : range.bytes / kCacheWarmStride;
        for (; line < lines; line += static_cast<std::size_t>(ctas) * threads)
            sink ^= *reinterpret_cast<const std::uint32_t*>(
                static_cast<const std::uint8_t*>(range.data) + line * kCacheWarmStride);
        line -= lines;
    }
    // NOLINTNEXTLINE(portability-no-assembler): empty asm keeps the warming loads live
    asm volatile("" ::"v"(sink));
}
#endif

} // namespace ninfer
