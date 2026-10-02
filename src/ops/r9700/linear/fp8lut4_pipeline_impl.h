#pragma once

// Device-side pieces of the FP8LUT4 x per-token E4M3 Linear shared by the Linear kernels and fused
// consumers (e.g. the GDN projection-convolution epilogue).
#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear::fp8lut4 {

inline constexpr std::uint32_t kBaseSixteenths[8] = {0U, 13U, 27U, 41U, 56U, 74U, 94U, 120U};

// E4M3FN word (sign clear) nearest numerator * 2^exponent, ties to even, saturating at 448.
constexpr std::uint8_t round_e4m3_magnitude(std::uint32_t numerator, int exponent) {
    if (numerator == 0U) return 0U;
    const int e = static_cast<int>(std::bit_width(numerator)) - 1 + exponent;
    int ulp = (e > -6 ? e : -6) - 3;
    const int shift = ulp - exponent;
    std::uint64_t units = numerator;
    if (shift <= 0) {
        units <<= static_cast<unsigned>(-shift);
    } else if (shift >= 63) {
        units = 0U;
    } else {
        const std::uint64_t remainder = units & ((std::uint64_t{1} << shift) - 1U);
        const std::uint64_t half = std::uint64_t{1} << (shift - 1);
        units >>= static_cast<unsigned>(shift);
        if (remainder > half || (remainder == half && (units & 1U) != 0U)) ++units;
    }
    if (ulp == -9 && units < 8U) return static_cast<std::uint8_t>(units);
    while (units >= 16U) {
        units >>= 1U;
        ++ulp;
    }
    const int biased = ulp + 10;
    if (biased > 15 || (biased == 15 && units - 8U > 6U)) return 0x7EU;
    return static_cast<std::uint8_t>((biased << 3) | static_cast<int>(units - 8U));
}

constexpr std::array<std::array<std::uint8_t, 8>, 256> make_table() {
    std::array<std::array<std::uint8_t, 8>, 256> table{};
    for (std::uint32_t code = 0; code < 256U; ++code) {
        const std::uint32_t m = code & 7U;
        const int exponent = static_cast<int>(code >> 3U) - 26;
        for (std::uint32_t j = 0; j < 8U; ++j)
            table[code][j] = round_e4m3_magnitude(kBaseSixteenths[j] * (8U + m), exponent - 7);
    }
    return table;
}

inline constexpr auto kTable = make_table();

struct PackedTable {
    uint2 words[256];
};

constexpr PackedTable make_packed() {
    PackedTable out{};
    for (std::uint32_t code = 0; code < 256U; ++code) {
        std::uint32_t low = 0U, high = 0U;
        for (std::uint32_t j = 0; j < 4U; ++j) {
            low |= static_cast<std::uint32_t>(kTable[code][j]) << (8U * j);
            high |= static_cast<std::uint32_t>(kTable[code][j + 4U]) << (8U * j);
        }
        out.words[code] = uint2{low, high};
    }
    return out;
}

namespace {
// Internal linkage: every translation unit carries its own constant copy.
__constant__ PackedTable kDeviceTable = make_packed();
} // namespace

using I32x2 = int __attribute__((ext_vector_type(2)));
using F32x8 = float __attribute__((ext_vector_type(8)));

// Eight nibble codes (one u32, low nibble = even k) through one group codebook: two E4M3 words
// holding k0..k7 in order. The magnitude lookup is a byte permute of the eight table bytes; the
// sign moves from nibble bit 3 to byte bit 7.
__device__ __forceinline__ uint2 decode8(std::uint32_t codes, uint2 table) {
    const std::uint32_t even = codes & 0x0F0F0F0FU;
    const std::uint32_t odd = (codes >> 4U) & 0x0F0F0F0FU;
    const std::uint32_t even_bytes =
        __builtin_amdgcn_perm(table.y, table.x, even & 0x07070707U) | ((even & 0x08080808U) << 4U);
    const std::uint32_t odd_bytes =
        __builtin_amdgcn_perm(table.y, table.x, odd & 0x07070707U) | ((odd & 0x08080808U) << 4U);
    return uint2{__builtin_amdgcn_perm(odd_bytes, even_bytes, 0x05010400U),
                 __builtin_amdgcn_perm(odd_bytes, even_bytes, 0x07030602U)};
}

// Thread `thread` of `threads` copies its share of the codebook into the CTA's LDS table (the
// caller synchronizes before use).
__device__ __forceinline__ void load_table(uint2* table, std::uint32_t thread,
                                           std::uint32_t threads) {
    for (std::uint32_t i = thread; i < 256U; i += threads) table[i] = kDeviceTable.words[i];
}

// Small T (T <= 16 per token tile, kTiles tiles) on CTA context `cta` (persistent_cta.h): 16 weight rows [row_base, row_base + 16), kSplit waves each owning a
// contiguous K range. Lane (axis = lane & 15, half = lane >> 4) streams slot 16 half + axis of the
// row block's N16 x K64 tile per 64-column step (one contiguous 512-byte code tile per wave), i.e.
// row `axis`'s 32-column group `2 step + half`, and the same 32 activation bytes of token `axis`; the
// four WMMAs of a step pair those bytes identically, which only permutes the exact K sum. (Group
// codes load one byte per step: a shared 8-byte word per four steps measured 10-40% slower.) After the
// in-order combine of the K splits, wave 0 calls publish(token, row, sum * R[row] * s[token]) for
// every token < tokens (rows half * 8 + item of its lane). The caller issues load_table without
// waiting: the first batch of weight and activation loads is in flight before the CTA barrier that
// publishes the codebook. kTiles = 2 serves T 17..32: each decoded weight fragment also feeds the
// WMMA of token tile 1 (tokens 16 + axis), so weight traffic and decode stay those of one tile.
template <std::uint32_t kSplit, std::uint32_t kSteps, std::uint32_t kTiles, class Cta, class Publish>
__device__ __forceinline__ void small_t_rows(const Cta& cta, const Fp8Lut4Weight& weight,
                                             std::uint32_t row_base,
                                             const std::uint8_t* activation_codes,
                                             const float* token_scales, std::uint32_t tokens,
                                             const uint2* table,
                                             float (&partial)[kSplit][kTiles][8][32],
                                             Publish&& publish) {
    static_assert(kTiles == 1U || kTiles == 2U);
    const std::uint32_t lane = cta.thread() & 31U, wave = cta.thread() >> 5U;
    const std::uint32_t axis = lane & 15U, half = lane >> 4U;
    const std::uint32_t columns = weight.columns;
    const std::uint32_t steps = columns / 64U / kSplit;
    const std::uint32_t k_base = wave * steps * 64U + half * 32U;
    bool token[kTiles];
#pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile) token[tile] = tile * 16U + axis < tokens;
    // N16 x K64 tiles: this lane's slot of the wave's first tile; one tile per step.
    const std::size_t first_tile =
        static_cast<std::size_t>(row_base / 16U) * (columns / 64U) + wave * steps;
    const std::uint32_t slot = half * 16U + axis;
    const auto* codes = persistent::global(weight.codes) + first_tile * 512U + slot * 16U;
    const auto* groups = persistent::global(weight.groups) + first_tile * 32U + slot;
    const persistent::Global<const std::uint8_t>* activation[kTiles];
#pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile)
        activation[tile] = persistent::global(activation_codes) +
            static_cast<std::size_t>(token[tile] ? tile * 16U + axis : 0U) * columns + k_base;
    struct Batch { uint4 w[kSteps]; std::uint32_t g[kSteps]; uint4 x[kSteps][kTiles][2]; };
    const auto load = [&](std::uint32_t step) {
        Batch value;
#pragma unroll
        for (std::uint32_t i = 0; i < kSteps; ++i) {
            value.w[i] = persistent::load16(codes + static_cast<std::size_t>(step + i) * 512U);
            value.g[i] = groups[static_cast<std::size_t>(step + i) * 32U];
#pragma unroll
            for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
                if (token[tile]) {
                    const auto* x = activation[tile] + static_cast<std::size_t>(step + i) * 64U;
                    value.x[i][tile][0] = persistent::load16(x);
                    value.x[i][tile][1] = persistent::load16(x + 16U);
                } else {
                    value.x[i][tile][0] = uint4{};
                    value.x[i][tile][1] = uint4{};
                }
            }
        }
        return value;
    };
    F32x8 accumulator[kTiles]{};
    const auto consume = [&](const Batch& value) {
#pragma unroll
        for (std::uint32_t i = 0; i < kSteps; ++i) {
            const uint2 book = table[value.g[i]];
            const uint2 w0 = decode8(value.w[i].x, book), w1 = decode8(value.w[i].y, book);
            const uint2 w2 = decode8(value.w[i].z, book), w3 = decode8(value.w[i].w, book);
            const std::uint32_t w[8]{w0.x, w0.y, w1.x, w1.y, w2.x, w2.y, w3.x, w3.y};
#pragma unroll
            for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
                const uint4* xt = value.x[i][tile];
                const std::uint32_t x[8]{xt[0].x, xt[0].y, xt[0].z, xt[0].w,
                                         xt[1].x, xt[1].y, xt[1].z, xt[1].w};
#pragma unroll
                for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
                    accumulator[tile] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        I32x2{static_cast<int>(w[2U * chunk]), static_cast<int>(w[2U * chunk + 1U])},
                        I32x2{static_cast<int>(x[2U * chunk]), static_cast<int>(x[2U * chunk + 1U])},
                        accumulator[tile]);
                }
            }
        }
    };
    Batch current = load(0U);
    cta.sync();  // codebook
    for (std::uint32_t step = kSteps; step < steps; step += kSteps) {
        const Batch next = load(step);
        consume(current);
        current = next;
    }
    consume(current);
#pragma unroll
    for (std::uint32_t tile = 0; tile < kTiles; ++tile)
#pragma unroll
        for (std::uint32_t item = 0; item < 8U; ++item)
            partial[wave][tile][item][lane] = accumulator[tile][item];
    cta.sync();
    if (wave == 0U) {
#pragma unroll
        for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
            if (!token[tile]) continue;
            const std::uint32_t t = tile * 16U + axis;
            const float token_scale = persistent::global(token_scales)[t];
#pragma unroll
            for (std::uint32_t item = 0; item < 8U; ++item) {
                float sum = partial[0][tile][item][lane];
#pragma unroll
                for (std::uint32_t split = 1; split < kSplit; ++split)
                    sum += partial[split][tile][item][lane];
                const std::uint32_t row = row_base + half * 8U + item;
                publish(t, row, sum * persistent::global(weight.scales)[row] * token_scale);
            }
        }
    }
}

// Verification widths (T 2..16): four K quarters of two steps in flight, measured in whole
// inference (DFlash rounds 1.2% faster than two halves of four; eight splits of two 2% slower).
// One token uses the dedicated GEMV. small_t_rows needs (columns / 64) divisible by 8.
inline constexpr std::uint32_t kSmallTokens = 16U;
// T 17..32: two token tiles over the same weight stream.
inline constexpr std::uint32_t kSmallTwoTileTokens = 32U;
inline constexpr std::uint32_t kVerifySplit = 4U, kVerifySteps = 2U;

} // namespace ninfer::ops::r9700::linear::fp8lut4
