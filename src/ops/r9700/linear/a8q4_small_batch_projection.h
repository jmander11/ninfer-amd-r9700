#pragma once
#include "ops/r9700/linear/r9700_linear.h"

namespace ninfer::ops::r9700::linear {
// Complete regression boundary: BF16 -> A8G64 -> signed Q4N16K16/G64 Linear,
// T2..4 MLP/output: N34816/K5120, N5120/K17408, N5120/K6144.
// Tiled T10/12/14/15/16 (every C2..C4 DFlash chain width W5..8 and C2/C4 W8/W4 up to 16) MLP:
// N34816/K5120, N5120/K17408; tiled projections use all T5..8 shapes below.
// T4..8 projections (T4 is the C1 DFlash K3 chain):
// N34816/K5120, N5120/K6144, N12288/K5120, N4096/K5120.
// T5..8 draft down/feature: N5120/K17408 and N5120/K25600.
// T5..8 attention/auxiliary: N7168/K5120, N6144/K5120, N1280/K5120,
// N5120/K4096. Other widths retain their existing route.
// N12288 T5..8 retains scale-gather; other admitted cells use successor pipelining.
// Concurrent-DFlash widths T17..64 of the drafter shapes and the N131072/K5120 draft head take
// the measured wide route below (N1280/K5120: its TiledM widths and T49..64).
// No allocation or persistent-weight transformation.
[[nodiscard]] hipError_t a8q4_small_batch_projection(const A8Q4G64CandidateArgs& a,
                                                     hipStream_t stream) noexcept;

namespace detail {
// Wide cells T17..64. TokenTile* (a8q4_token_tile_cta.h): one CTA per ceil(T/16)x16 token tile
// and 64-row (N64, N64SplitK) or 128-row (N128) row tile, every staged G64 activation fragment
// shared by the CTA's row fragments; N64SplitK splits K over two in-CTA slices with two groups
// of loads in flight, N128 keeps two groups in flight. The kernel is T-generic within its
// 17..32/33..48/49..64 tile classes, so each class takes the route measured at its widths.
// TiledM is the tiled body with one CTA row per M tile. Unprofiled R9700 medians
// (profiles/bench/r9700-c8-20260930/drafter/token_tile_ab2.json; 7 interleaved trials over
// >=256 MiB of rotated weight copies), A/B baseline route -> selected, us:
//   N34816/K5120  N64 T18..32 185..322 -> 168..171; N64SplitK T35..64 342..354 -> 188..195.
//   N5120/K17408  N64SplitK T18..32 116..136 -> 86..89, T35..48 170..196 -> 102..104,
//                 T49..64 265..270 -> 113..114.
//   N5120/K25600  N64SplitK T18..32 168..217 -> 122..128, T35..48 275..339 -> 144..151,
//                 T49..64 391..393 -> 164..167.
//   N6144/K5120   N64SplitK T18..32 42..50 -> 37..39, T35..64 59..94 -> 41..45.
//   N5120/K4096   N64SplitK T25..32 37..42 -> 32..33, T35..64 46..67 -> 35..38.
//   N1280/K5120   T25..48 keep TiledM/table (15..21; token tile 16..20);
//                 N64SplitK T49..64 23..27 -> 19.
//   N131072/K5120 N64SplitK T18..30 724..1205 -> 610..627; N128 T32 970 -> 629,
//                 T35..48 (prefill CTA) 1222..1241 -> 658..685; N64SplitK T49..56 1244..1247
//                 -> 704..714.
// The in-place residual epilogue is admitted on the N5120 routes.
enum class A8Q4WideRoute : std::uint8_t {
    None,
    TiledM,
    TokenTileN64,
    TokenTileN64SplitK,
    TokenTileN128
};

[[nodiscard]] constexpr A8Q4WideRoute
select_a8q4_small_batch_wide_route(unsigned tokens, unsigned rows, unsigned columns,
                                   unsigned padded_columns) noexcept {
    using R = A8Q4WideRoute;
    if (columns != padded_columns || tokens < 17 || tokens > 64) return R::None;
    if (rows == 34816 && columns == 5120)
        return tokens <= 32 ? R::TokenTileN64 : R::TokenTileN64SplitK;
    if ((rows == 5120 && (columns == 17408 || columns == 25600 || columns == 4096)) ||
        (rows == 6144 && columns == 5120))
        return R::TokenTileN64SplitK;
    if (rows == 131072 && columns == 5120)
        return tokens >= 32 && tokens <= 48 ? R::TokenTileN128 : R::TokenTileN64SplitK;
    if (rows == 1280 && columns == 5120) {
        if (tokens >= 49) return R::TokenTileN64SplitK;
        return tokens == 25 || tokens == 30 || tokens == 35 || tokens == 36 || tokens == 40 ||
                       tokens == 42 || tokens == 48
                   ? R::TiledM
                   : R::None;
    }
    return R::None;
}

[[nodiscard]] constexpr bool use_a8q4_small_batch_wide(unsigned tokens, unsigned rows,
                                                       unsigned columns,
                                                       unsigned padded_columns) noexcept {
    return select_a8q4_small_batch_wide_route(tokens, rows, columns, padded_columns) !=
           A8Q4WideRoute::None;
}

[[nodiscard]] constexpr bool use_a8q4_small_batch_projection(unsigned tokens, unsigned rows,
                                                             unsigned columns,
                                                             unsigned padded_columns) noexcept {
    const bool t1_4 = tokens >= 1 && tokens <= 4, t4_8 = tokens >= 4 && tokens <= 8;
    const bool t10_16 =
        tokens == 10 || tokens == 12 || tokens == 14 || tokens == 15 || tokens == 16;
    const bool t18_32 = tokens == 18 || tokens == 20 || tokens == 21 || tokens == 24 ||
                        tokens == 28 || tokens == 32;
    const bool k5120_projection =
        (rows == 12288 || rows == 4096 || rows == 7168 || rows == 6144 || rows == 1280) &&
        columns == 5120;
    const bool mlp = (rows == 34816 && columns == 5120) || (rows == 5120 && columns == 17408);
    return use_a8q4_small_batch_wide(tokens, rows, columns, padded_columns) ||
           (columns == padded_columns &&
            ((t4_8 && ((rows == 5120 && (columns == 4096 || columns == 6144 || columns == 17408 ||
                                         columns == 25600)) ||
                       (rows == 34816 && columns == 5120) || k5120_projection)) ||
             (t1_4 && rows == 5120 && columns == 6144) ||
             (t10_16 &&
              ((rows == 5120 && (columns == 4096 || columns == 6144 || columns == 25600)) ||
               k5120_projection)) ||
             // T>16 table cells of shapes without a wide route at these widths.
             (t18_32 && ((rows == 5120 && columns == 6144) ||
                         ((rows == 12288 || rows == 4096 || rows == 7168 || rows == 1280) &&
                          columns == 5120))) ||
             ((t1_4 || t10_16) && mlp)));
}

// Internal prepared launch only: the generic owner has validated all planes,
// selected the exact predicate above and freshly quantized its A8 workspace.
// No second quantization, stream query, workspace binding or allocation.
[[nodiscard]] hipError_t launch_a8q4_small_batch_projection(const A8Q4G64LinearArgs& a,
                                                            hipStream_t stream) noexcept;
// Two non-accumulating small-batch projections of the same prepared K5120 activation in one
// launch (the GDN query-key/value-z and the attention query-key/gate-value pairs), with the
// per-output arithmetic of the split small-batch route. The predicate requires identical
// activation planes and widths; the launcher requires it.
[[nodiscard]] bool use_a8q4_small_batch_projection_pair(const A8Q4G64LinearArgs& a,
                                                        const A8Q4G64LinearArgs& b) noexcept;
[[nodiscard]] hipError_t launch_a8q4_small_batch_projection_pair(const A8Q4G64LinearArgs& a,
                                                                 const A8Q4G64LinearArgs& b,
                                                                 hipStream_t stream) noexcept;

// The attention pair (two N7168/K5120 matrices) at T5/T6 with each matrix's rows split at
// `split`: rows [0,split) publish to its leading [T,split] and the rest to its trailing
// [T,7168-split] output, the values the unsplit pair would store.
struct A8Q4PairSplitOutputs {
    hip_bfloat16* first_leading   = nullptr;
    hip_bfloat16* first_trailing  = nullptr;
    hip_bfloat16* second_leading  = nullptr;
    hip_bfloat16* second_trailing = nullptr;
    std::uint32_t split           = 0;
};

[[nodiscard]] bool use_a8q4_small_batch_projection_pair_split(const A8Q4G64LinearArgs& a,
                                                              const A8Q4G64LinearArgs& b,
                                                              std::uint32_t split) noexcept;
[[nodiscard]] hipError_t launch_a8q4_small_batch_projection_pair_split(
    const A8Q4G64LinearArgs& a, const A8Q4G64LinearArgs& b, const A8Q4PairSplitOutputs& outputs,
    hipStream_t stream) noexcept;
} // namespace detail
} // namespace ninfer::ops::r9700::linear
