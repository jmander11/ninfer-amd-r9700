#pragma once

// GDN verification-front kernels (gdn_ops.h): the FP8 normalized front with its fused deferred
// replay fold and cache-warm CTAs, and the FP8LUT4 pair projection with the recorded causal
// convolution. Shared by their launchers (gdn_ops.hip) and the persistent decode kernel.
#include "core/cache_warm.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn_recurrence/gdn_replay_fold_impl.h"
#include "ops/r9700/linear/a8g64_codec.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_encode_impl.h"
#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/fp8lut4_pipeline_impl.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn::front_kernels {

inline constexpr std::uint32_t kProjectionQueryRows = 2048;
inline constexpr std::uint32_t kProjectionKeyRows = 2048;
inline constexpr std::uint32_t kProjectionValueRows = 6144;
inline constexpr std::uint32_t kProjectionValueZRows = 12288;
inline constexpr std::uint32_t kProjectionChannels =
    kProjectionQueryRows + kProjectionKeyRows + kProjectionValueRows;
inline constexpr std::uint32_t kMaximumVerifyWidth = 16;
inline constexpr std::uint32_t kControlHeads = 48;
inline constexpr std::uint32_t kControlColumns = 5120;
inline constexpr std::uint32_t kControlT1Threads = kControlColumns / 8U;

__device__ __forceinline__ void publish_projection_output(
    hip_bfloat16 output, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    std::size_t column, std::uint32_t channel) {
    if (channel < kProjectionQueryRows) {
        query[column * kProjectionQueryRows + channel] = output;
    } else if (channel < kProjectionQueryRows + kProjectionKeyRows) {
        key[column * kProjectionKeyRows + channel - kProjectionQueryRows] = output;
    } else {
        value[column * kProjectionValueRows +
              channel - kProjectionQueryRows - kProjectionKeyRows] = output;
    }
}

// The inputs of one channel's recorded causal convolution for one sequence: its valid column
// count, its initial checkpoint (zero without a valid initial slot) and its four taps.
struct ConvChannel {
    std::uint32_t valid;
    bool valid_initial;
    float checkpoint0, checkpoint1, checkpoint2;
    float w0, w1, w2, w3;
};

__device__ __forceinline__ ConvChannel load_conv_channel(
    const hip_bfloat16* conv_weight, const hip_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    std::uint32_t width, std::uint32_t state_slots, std::uint32_t batch_index,
    std::uint32_t channel) {
    const std::int32_t valid_raw = valid_columns == nullptr
        ? static_cast<std::int32_t>(width)
        : valid_columns[batch_index];
    const std::int32_t initial_raw = initial_state_slots[batch_index];
    const bool valid_initial = initial_raw >= 0 &&
        static_cast<std::uint32_t>(initial_raw) < state_slots;
    const std::size_t initial_base = valid_initial
        ? static_cast<std::size_t>(initial_raw) * 3U * kProjectionChannels
        : 0U;
    return ConvChannel{
        valid_raw <= 0 ? 0U : min(static_cast<std::uint32_t>(valid_raw), width),
        valid_initial,
        valid_initial ? static_cast<float>(conv_states[initial_base + channel]) : 0.0F,
        valid_initial ? static_cast<float>(conv_states[initial_base + kProjectionChannels + channel])
                      : 0.0F,
        valid_initial
            ? static_cast<float>(conv_states[initial_base + 2U * kProjectionChannels + channel])
            : 0.0F,
        static_cast<float>(conv_weight[channel]),
        static_cast<float>(conv_weight[kProjectionChannels + channel]),
        static_cast<float>(conv_weight[2U * kProjectionChannels + channel]),
        static_cast<float>(conv_weight[3U * kProjectionChannels + channel]),
    };
}

// The recorded causal convolution of one channel of one sequence (width columns starting at
// batch_index * width) from its loaded inputs: `represented(column)` is the channel's BF16
// projection of that column.
template <class Represented>
__device__ __forceinline__ void conv_record_loaded_channel(
    Represented&& represented_at, const ConvChannel& conv, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    std::uint32_t width, std::uint32_t batch_index, std::uint32_t channel) {
    const std::uint32_t valid = conv.valid;
    const bool valid_initial = conv.valid_initial;
    const float checkpoint0 = conv.checkpoint0, checkpoint1 = conv.checkpoint1,
                checkpoint2 = conv.checkpoint2;
    const float w0 = conv.w0, w1 = conv.w1, w2 = conv.w2, w3 = conv.w3;
    float saved0[kMaximumVerifyWidth];
    float saved1[kMaximumVerifyWidth];
    float saved2[kMaximumVerifyWidth];

    for (std::uint32_t token = 0; token < width; ++token) {
        const std::size_t column = static_cast<std::size_t>(batch_index) * width + token;
        if (token >= valid || !valid_initial) {
            conv_record[column * kProjectionChannels + channel] = hip_bfloat16(0.0F);
            publish_projection_output(hip_bfloat16(0.0F), query, key, value, column, channel);
            continue;
        }

        const std::int32_t parent = parent_index == nullptr ?
            (token == 0 ? -1 : static_cast<std::int32_t>(token - 1U)) : parent_index[column];
        if (parent >= static_cast<std::int32_t>(token)) {
            conv_record[column * kProjectionChannels + channel] = hip_bfloat16(0.0F);
            publish_projection_output(hip_bfloat16(0.0F), query, key, value, column, channel);
            continue;
        }
        float h0 = parent < 0 ? checkpoint0 : saved0[parent];
        float h1 = parent < 0 ? checkpoint1 : saved1[parent];
        float h2 = parent < 0 ? checkpoint2 : saved2[parent];
        const hip_bfloat16 represented = represented_at(column);
        conv_record[column * kProjectionChannels + channel] = represented;
        const float current = static_cast<float>(represented);
        float sum = fmaf(w0, h0, 0.0F);
        sum = fmaf(w1, h1, sum);
        sum = fmaf(w2, h2, sum);
        sum = fmaf(w3, current, sum);
        publish_projection_output(hip_bfloat16(sum / (1.0F + expf(-sum))), query, key, value,
                                  column, channel);
        saved0[token] = h1;
        saved1[token] = h2;
        saved2[token] = current;
    }
}

// The recorded causal convolution of one channel of one sequence. Shared by the standalone record
// kernel and the GDN pair epilogues.
template <class Represented>
__device__ __forceinline__ void conv_record_channel(
    Represented&& represented_at, const hip_bfloat16* conv_weight,
    const hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    std::uint32_t width, std::uint32_t state_slots, std::uint32_t batch_index,
    std::uint32_t channel) {
    conv_record_loaded_channel(
        represented_at,
        load_conv_channel(conv_weight, conv_states, valid_columns, initial_state_slots, width,
                          state_slots, batch_index, channel),
        parent_index, conv_record, query, key, value, width, batch_index, channel);
}

// Token `token` of the recorded causal convolution of one channel of one sequence (width
// columns starting at batch_index * width), independently of the channel's other tokens: its
// state taps are the represented values of its first three ancestors (the checkpoint past the
// root), exactly the values conv_record_loaded_channel carries along the parent chain, so every
// output equals that sequential walk's. `represented(token)` is the channel's BF16 projection of
// a token of the sequence.
template <class Represented>
__device__ __forceinline__ void conv_record_token(
    Represented&& represented_at, const ConvChannel& conv, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    std::uint32_t width, std::uint32_t batch_index, std::uint32_t channel, std::uint32_t token) {
    const std::size_t base = static_cast<std::size_t>(batch_index) * width;
    const std::size_t column = base + token;
    const auto parent_of = [&](std::int32_t t) -> std::int32_t {
        return parent_index == nullptr ? t - 1 : parent_index[base + static_cast<std::size_t>(t)];
    };
    const std::int32_t parent = token >= conv.valid || !conv.valid_initial
        ? static_cast<std::int32_t>(token)
        : parent_of(static_cast<std::int32_t>(token));
    if (parent >= static_cast<std::int32_t>(token)) {
        conv_record[column * kProjectionChannels + channel] = hip_bfloat16(0.0F);
        publish_projection_output(hip_bfloat16(0.0F), query, key, value, column, channel);
        return;
    }
    const auto value_at = [&](std::int32_t t) {
        return static_cast<float>(represented_at(static_cast<std::uint32_t>(t)));
    };
    // (h0, h1, h2) of a token with parent p: the checkpoint at a root, else (h1, h2, x) of p.
    float h0 = conv.checkpoint0, h1 = conv.checkpoint1, h2 = conv.checkpoint2;
    if (parent >= 0) {
        // An ancestor is an earlier token (a malformed index past its child reads nothing).
        std::int32_t grandparent = parent_of(parent);
        if (grandparent >= parent) grandparent = -1;
        float parent_h1 = conv.checkpoint1, parent_h2 = conv.checkpoint2;
        if (grandparent >= 0) {
            std::int32_t great = parent_of(grandparent);
            if (great >= grandparent) great = -1;
            parent_h1 = great < 0 ? conv.checkpoint2 : value_at(great);
            parent_h2 = value_at(grandparent);
        }
        h0 = parent_h1;
        h1 = parent_h2;
        h2 = value_at(parent);
    }
    const hip_bfloat16 represented = represented_at(token);
    conv_record[column * kProjectionChannels + channel] = represented;
    const float current = static_cast<float>(represented);
    float sum = fmaf(conv.w0, h0, 0.0F);
    sum = fmaf(conv.w1, h1, sum);
    sum = fmaf(conv.w2, h2, sum);
    sum = fmaf(conv.w3, current, sum);
    publish_projection_output(hip_bfloat16(sum / (1.0F + expf(-sum))), query, key, value, column,
                              channel);
}

// ---- FP8LUT4 GDN pair projection with the recorded convolution (one sequence, T 4..8 columns):
// CTAs [0, 256) own the query-key row blocks of one per-token E4M3 image, the rest the value-z
// row blocks; each block's BF16 outputs (canonical NaN for a flagged token) are staged, then one
// thread per (row, token) runs that token of the recorded convolution of the row's channel
// (value-z rows 6144.. are the output gate z). Bitwise the pair projection followed by
// projection_conv_record_kernel.
inline constexpr std::uint32_t kFp8Lut4PairSplit = linear::fp8lut4::kVerifySplit;
inline constexpr std::uint32_t kFp8Lut4PairSteps = linear::fp8lut4::kVerifySteps;

template <unsigned T>
struct PairConvShared {
    uint2 table[256];
    float partial[kFp8Lut4PairSplit][1][8][32];
    hip_bfloat16 staged[16][T];
};

template <unsigned T, std::uint32_t kSteps = kFp8Lut4PairSteps, class Cta>
__device__ __forceinline__ void fp8lut4_pair_conv_record_body(
    const Cta& cta, PairConvShared<T>& shared, const std::uint8_t* activation,
    const float* token_scales, const std::uint32_t* status, const linear::Fp8Lut4Weight& query_key,
    const linear::Fp8Lut4Weight& value_z, const hip_bfloat16* conv_weight,
    const hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    hip_bfloat16* z, std::uint32_t state_slots) {
    constexpr unsigned kQueryKeyTiles = (kProjectionQueryRows + kProjectionKeyRows) / 16U;
    const std::uint32_t block = cta.block().x, thread = cta.thread();
    const bool second = block >= kQueryKeyTiles;
    const linear::Fp8Lut4Weight weight = second ? value_z : query_key;
    const std::uint32_t row_base = (second ? block - kQueryKeyTiles : block) * 16U;
    // Epilogue thread `thread` owns token thread / 16 of row thread % 16 and, unless the row is
    // an output-gate z row, its convolution channel, whose inputs are loaded before the
    // projection so their latency overlaps it.
    static_assert(16U * T <= 32U * kFp8Lut4PairSplit);
    const std::uint32_t token = thread / 16U;
    const std::uint32_t row = row_base + thread % 16U;
    const bool active = token < T;
    const bool convolved = active && (!second || row < kProjectionValueRows);
    const std::uint32_t channel =
        second ? kProjectionQueryRows + kProjectionKeyRows + row : row;
    ConvChannel conv{};
    if (convolved)
        conv = load_conv_channel(conv_weight, conv_states, valid_columns, initial_state_slots, T,
                                 state_slots, 0U, channel);
    linear::fp8lut4::load_table(shared.table, thread, 32U * kFp8Lut4PairSplit);
    const auto stage = [&](std::uint32_t token, std::uint32_t row, float y) {
        shared.staged[row - row_base][token] = status[token] != linear::Fp8ActivationOk
            ? hip_bfloat16(__builtin_nanf("")) : hip_bfloat16(y);
    };
    linear::fp8lut4::small_t_rows<kFp8Lut4PairSplit, kSteps, 1U>(
        cta, weight, row_base, activation, token_scales, T, shared.table, shared.partial, stage);
    cta.sync();
    if (!active) return;
    const hip_bfloat16* values = shared.staged[thread % 16U];
    if (convolved) {
        conv_record_token([&](std::uint32_t t) { return values[t]; }, conv, parent_index,
                          conv_record, query, key, value, T, 0U, channel, token);
    } else {
        z[token * kProjectionValueRows + row - kProjectionValueRows] = values[token];
    }
}

template <unsigned T>
__global__ __launch_bounds__(32U * kFp8Lut4PairSplit) void gdn_fp8lut4_pair_conv_record_kernel(
    const std::uint8_t* activation, const float* token_scales, const std::uint32_t* status,
    linear::Fp8Lut4Weight query_key, linear::Fp8Lut4Weight value_z, const hip_bfloat16* conv_weight,
    const hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    hip_bfloat16* z, std::uint32_t state_slots) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ PairConvShared<T> shared;
    fp8lut4_pair_conv_record_body<T>(persistent::LaunchCta{}, shared, activation, token_scales,
                                     status, query_key, value_z, conv_weight, conv_states,
                                     valid_columns, initial_state_slots, parent_index,
                                     conv_record, query, key, value, z, state_slots);
#else
    (void)activation; (void)token_scales; (void)status; (void)query_key; (void)value_z;
    (void)conv_weight; (void)conv_states; (void)valid_columns; (void)initial_state_slots;
    (void)parent_index; (void)conv_record; (void)query; (void)key; (void)value; (void)z;
    (void)state_slots;
#endif
}

// ---- Verification-width GDN front: one CTA per control head, eight tokens per pass. Every CTA
// normalizes each token row with the K5120 row-CTA RMSNorm arithmetic (per-vector eight-term FMA
// chain, wave32 xor tree, wave partials summed in wave order, BF16 seam) and runs the
// per-(head, token) control arithmetic of control_t1_output on the seam (thread j sums token j's
// wave partials). CTA token % kControlHeads encodes the token's seam with the exact A8G64 codec
// (and publishes it to `hidden` when non-null); CTA 0, which sees every seam value, plain-stores
// the nonfinite/overflow status word, so no reset launch is needed. The launch runs one 16-byte
// row vector per thread (kControlT1Threads); a narrower FP8 context of Threads threads holds
// vectors v * Threads + t and keeps every partial and wave-sum order of that launch.
inline constexpr std::uint32_t kFrontMaximumTokens = 32U;
inline constexpr std::uint32_t kFrontPassTokens = 8U;
inline constexpr std::uint32_t kFrontWaves = kControlT1Threads / 32U;

__device__ __forceinline__ float front_vector_sumsq(uint4 packed) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    float sumsq = 0.0F;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        const float value = linear::codec::bf16_word_element(words, element);
        sumsq = fmaf(value, value, sumsq);
    }
#pragma unroll
    for (std::uint32_t width = 16U; width != 0U; width >>= 1U)
        sumsq += __shfl_xor(sumsq, width, 32);
    return sumsq;
}

__device__ __forceinline__ float front_inverse(const float* wave_sums, float eps) {
    float block_sum = 0.0F;
#pragma unroll
    for (std::uint32_t i = 0; i < kFrontWaves; ++i) block_sum += wave_sums[i];
    return rsqrtf(block_sum / static_cast<float>(kControlColumns) + eps);
}

// BF16 seam words and their FP32 values (nonfinite kept).
__device__ __forceinline__ uint4 front_seam(uint4 packed, uint4 gains, float inverse, float offset,
                                            float (&value)[8]) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    const std::uint32_t gain_words[4] = {gains.x, gains.y, gains.z, gains.w};
    std::uint32_t seam[4];
#pragma unroll
    for (std::uint32_t w = 0; w < 4U; ++w) {
        const float low_gain = __uint_as_float(gain_words[w] << 16) + offset;
        const float high_gain = __uint_as_float(gain_words[w] & 0xffff0000U) + offset;
        const hip_bfloat16 low(__uint_as_float(words[w] << 16) * inverse * low_gain);
        const hip_bfloat16 high(__uint_as_float(words[w] & 0xffff0000U) * inverse * high_gain);
        seam[w] = static_cast<std::uint32_t>(low.data) |
                  (static_cast<std::uint32_t>(high.data) << 16);
        value[2U * w] = static_cast<float>(low);
        value[2U * w + 1U] = static_cast<float>(high);
    }
    return uint4{seam[0], seam[1], seam[2], seam[3]};
}

namespace fold = ninfer::ops::detail::gated_delta_net::fold;

// A layer's deferred replay fold run by the CTAs past the head CTAs (`ctas` of the launch width,
// zero: none). Four state tiles per fold thread keep the front and fold CTAs co-resident at the
// front's register budget (one dispatch wave for a single row).
inline constexpr int kFrontFoldTiles = 4;
static_assert(kControlT1Threads % fold::kItemThreads == 0U);
struct FrontFold {
    fold::LayerArgs args{};
    std::uint32_t ctas = 0U;
};

// Fold CTAs of a Threads-thread context folding Tiles state tiles per item (the launch's
// fold.ctas at kControlT1Threads and kFrontFoldTiles).
template <std::uint32_t Threads, int Tiles = kFrontFoldTiles>
__host__ __device__ constexpr std::uint32_t front_fold_ctas(const FrontFold& fold) {
    return fold.ctas == 0U ? 0U
                           : static_cast<std::uint32_t>(fold.args.batch) *
                                 fold::layer_ctas_per_row(Threads, Tiles);
}

template <std::uint32_t Threads>
struct FrontShared {
    union {
        struct {
            float wave_sums[kFrontPassTokens][kFrontWaves];
            float inverses[kFrontPassTokens];
            float a_waves[kFrontPassTokens][kFrontWaves];
            float b_waves[kFrontPassTokens][kFrontWaves];
            linear::fp8_encode::EncodeShared<Threads> encode;
        } head;
        struct {
            float key[Threads / fold::kItemThreads][fold::kStateDim];
            float reduction[Threads / fold::kItemThreads][fold::kStateDim];
        } fold;
    };
};

// kFp8: the owner CTA of each token encodes its seam row as a per-token E4M3 image (codes in
// low_codes, FP32 scales in fp8_scales, one status word per token); otherwise the A8G64 codec
// (launch width only).
// kPassTokens tokens per pass (at most kFrontPassTokens); tokens are independent, so the pass
// size only trades registers for barriers. kFoldTiles state tiles per fold item: any count folds
// identically (fold_item), so it only sizes the fold CTAs.
template <bool kFp8, std::uint32_t Threads, std::uint32_t kPassTokens = kFrontPassTokens,
          int kFoldTiles = kFrontFoldTiles, class Cta>
__device__ __forceinline__ void normalized_front_body(
    const Cta& cta, FrontShared<Threads>& shared, const hip_bfloat16* residual,
    const hip_bfloat16* norm, float eps, bool unit_offset, const hip_bfloat16* a_weight,
    const hip_bfloat16* b_weight, const float* a_log, const float* dt_bias, float* g, float* beta,
    std::uint8_t* low_codes, std::uint8_t* high_codes, std::uint16_t* scale_words,
    std::uint32_t* status, hip_bfloat16* hidden, std::uint32_t* clear_status,
    std::uint32_t tokens, float* fp8_scales, const CacheWarm& warm, const FrontFold& fold) {
    static_assert(Threads % 32U == 0U && Threads % fold::kItemThreads == 0U);
    static_assert(kFp8 || Threads == kControlT1Threads);
    static_assert(kPassTokens != 0U && kPassTokens <= kFrontPassTokens);
    constexpr std::uint32_t V = (kControlT1Threads + Threads - 1U) / Threads;
    constexpr std::uint32_t kWaves = Threads / 32U;
    const std::uint32_t block = cta.block().x, thread = cta.thread();
    if (block >= kControlHeads) {
        const std::uint32_t index = block - kControlHeads;
        const std::uint32_t fold_ctas = front_fold_ctas<Threads, kFoldTiles>(fold);
        if (index < fold_ctas) {
            fold::fold_layer_cta<Threads, kFoldTiles>(cta, fold.args, index, shared.fold.key,
                                                      shared.fold.reduction);
            return;
        }
        warm_cache(warm, index - fold_ctas, cta.grid().x - kControlHeads - fold_ctas, thread,
                   cta.threads());
        return;
    }
    auto& wave_sums = shared.head.wave_sums;
    auto& inverses = shared.head.inverses;
    auto& a_waves = shared.head.a_waves;
    auto& b_waves = shared.head.b_waves;
    constexpr std::uint32_t row_vectors = kControlT1Threads;
    const std::uint32_t head = block;
    const std::uint32_t lane = thread & 31U, wave = thread >> 5U;
    // Vector v of this thread exists (wave-uniform) and its launch wave.
    const auto exists = [&](std::uint32_t v) { return v * Threads + thread < row_vectors; };
    const auto vector_wave = [&](std::uint32_t v) { return v * kWaves + wave; };
    const float offset = unit_offset ? 1.0F : 0.0F;
    uint4 gains[V], av[V], bv[V];
#pragma unroll
    for (std::uint32_t v = 0; v < V; ++v) {
        const std::uint32_t vector = exists(v) ? v * Threads + thread : 0U;
        gains[v] = reinterpret_cast<const uint4*>(norm)[vector];
        av[v] = reinterpret_cast<const uint4*>(
            a_weight + static_cast<std::size_t>(head) * kControlColumns)[vector];
        bv[v] = reinterpret_cast<const uint4*>(
            b_weight + static_cast<std::size_t>(head) * kControlColumns)[vector];
    }
    const uint4* rows = reinterpret_cast<const uint4*>(residual);
    bool nonfinite = false, overflow = false;
    for (std::uint32_t first = 0; first < tokens; first += kPassTokens) {
        // Tokens are predicated, never broken out of, so the arrays stay in registers.
        uint4 loaded[kPassTokens][V];
#pragma unroll
        for (std::uint32_t row = 0; row < kPassTokens; ++row)
#pragma unroll
            for (std::uint32_t v = 0; v < V; ++v)
                loaded[row][v] = first + row < tokens && exists(v)
                    ? rows[static_cast<std::size_t>(first + row) * row_vectors + v * Threads +
                           thread]
                    : uint4{};
#pragma unroll
        for (std::uint32_t row = 0; row < kPassTokens; ++row) {
#pragma unroll
            for (std::uint32_t v = 0; v < V; ++v) {
                if (!exists(v)) continue;
                const float sumsq = front_vector_sumsq(loaded[row][v]);
                if (lane == 0U) wave_sums[row][vector_wave(v)] = sumsq;
            }
        }
        cta.sync();
        // Thread j reduces row j once; every thread then reads the row's inverse.
        if (thread < kPassTokens) inverses[thread] = front_inverse(wave_sums[thread], eps);
        cta.sync();
#pragma unroll
        for (std::uint32_t row = 0; row < kPassTokens; ++row) {
            const std::uint32_t token = first + row;
            if (token < tokens) {
                float value[V][8];
                uint4 seam[V];
#pragma unroll
                for (std::uint32_t v = 0; v < V; ++v) {
                    if (!exists(v)) continue;
                    seam[v] = front_seam(loaded[row][v], gains[v], inverses[row], offset,
                                         value[v]);
                }
                if constexpr (kFp8) {
                    // CTA-uniform owner: the whole CTA encodes the row (block reduction inside).
                    if (token % kControlHeads == head) {
                        if (hidden != nullptr) {
#pragma unroll
                            for (std::uint32_t v = 0; v < V; ++v) {
                                if (!exists(v)) continue;
                                reinterpret_cast<uint4*>(hidden)[static_cast<std::size_t>(token) *
                                                                     row_vectors +
                                                                 v * Threads + thread] = seam[v];
                            }
                        }
                        linear::fp8_encode::encode_rows<Threads, V, row_vectors>(
                            cta, shared.head.encode, value, token, low_codes, fp8_scales,
                            kControlColumns, status);
                    }
                } else {
                    if (head == 0U) {
#pragma unroll
                        for (std::uint32_t element = 0; element < 8U; ++element) {
                            nonfinite |= !isfinite(value[0][element]);
                            overflow |= isfinite(value[0][element]) &&
                                        linear::codec::a8g64_element_overflows(value[0][element]);
                        }
                    }
                    if (token % kControlHeads == head) {
                        const std::size_t vector =
                            static_cast<std::size_t>(token) * row_vectors + thread;
                        if (hidden != nullptr) reinterpret_cast<uint4*>(hidden)[vector] = seam[0];
                        float finite[8];
                        float maximum = 0.0F;
#pragma unroll
                        for (std::uint32_t element = 0; element < 8U; ++element) {
                            finite[element] =
                                isfinite(value[0][element]) ? value[0][element] : 0.0F;
                            maximum = fmaxf(maximum, fabsf(finite[element]));
                        }
                        maximum = linear::codec::a8g64_group_maximum(maximum);
                        linear::codec::a8g64_encode_vector(
                            finite, maximum, lane, vector,
                            static_cast<std::size_t>(token) * (kControlColumns / 64U) +
                                thread / 8U,
                            low_codes, high_codes, scale_words);
                    }
                }
#pragma unroll
                for (std::uint32_t v = 0; v < V; ++v) {
                    if (!exists(v)) continue;
                    const std::uint32_t xw[4] = {seam[v].x, seam[v].y, seam[v].z, seam[v].w};
                    const std::uint32_t aw[4] = {av[v].x, av[v].y, av[v].z, av[v].w};
                    const std::uint32_t bw[4] = {bv[v].x, bv[v].y, bv[v].z, bv[v].w};
                    float a_partial = 0.0F;
                    float b_partial = 0.0F;
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const auto element = [&](std::uint32_t word) {
                                return __uint_as_float(half == 0 ? word << 16
                                                                 : word & 0xffff0000U);
                            };
                            a_partial = fmaf(element(xw[w]), element(aw[w]), a_partial);
                            b_partial = fmaf(element(xw[w]), element(bw[w]), b_partial);
                        }
                    }
#pragma unroll
                    for (std::uint32_t width = 16U; width != 0U; width >>= 1U) {
                        a_partial += __shfl_xor(a_partial, width, 32);
                        b_partial += __shfl_xor(b_partial, width, 32);
                    }
                    if (lane == 0U) {
                        a_waves[row][vector_wave(v)] = a_partial;
                        b_waves[row][vector_wave(v)] = b_partial;
                    }
                }
            }
        }
        cta.sync();
        if (thread < kPassTokens && first + thread < tokens) {
            const std::uint32_t row = thread;
            float a_sum = 0.0F;
            float b_sum = 0.0F;
#pragma unroll
            for (std::uint32_t i = 0; i < kFrontWaves; ++i) {
                a_sum += a_waves[row][i];
                b_sum += b_waves[row][i];
            }
            const hip_bfloat16 ar = static_cast<hip_bfloat16>(a_sum);
            const hip_bfloat16 br = static_cast<hip_bfloat16>(b_sum);
            const float a_value = static_cast<float>(ar) + dt_bias[head];
            const float softplus = a_value > 20.0F ? a_value : log1pf(expf(a_value));
            const std::size_t output =
                static_cast<std::size_t>(first + row) * kControlHeads + head;
            g[output] = -expf(a_log[head]) * softplus;
            const float b_value = static_cast<float>(br);
            beta[output] = 1.0F / (1.0F + expf(-b_value));
        }
        cta.sync();
    }
    if constexpr (!kFp8) {
        if (head == 0U) {
            const bool any_nonfinite = __syncthreads_or(nonfinite);
            const bool any_overflow = __syncthreads_or(overflow);
            if (thread == 0U) {
                *status =
                    (any_nonfinite ? static_cast<std::uint32_t>(linear::Q4G64ActivationNonfinite)
                                   : 0U) |
                    (any_overflow
                         ? static_cast<std::uint32_t>(linear::Q4G64ActivationScaleOverflow)
                         : 0U);
                if (clear_status != nullptr) *clear_status = 0U;
            }
        }
    } else {
        (void)high_codes;
        (void)scale_words;
        (void)clear_status;
        (void)nonfinite;
        (void)overflow;
    }
}

template <bool kFp8>
__global__ __launch_bounds__(kControlT1Threads) void gdn_normalized_front_kernel(
    const hip_bfloat16* residual, const hip_bfloat16* norm, float eps, bool unit_offset,
    const hip_bfloat16* a_weight, const hip_bfloat16* b_weight, const float* a_log,
    const float* dt_bias, float* g, float* beta, std::uint8_t* low_codes,
    std::uint8_t* high_codes, std::uint16_t* scale_words, std::uint32_t* status,
    hip_bfloat16* hidden, std::uint32_t* clear_status, std::uint32_t tokens,
    float* fp8_scales, CacheWarm warm, FrontFold fold) {
    __shared__ FrontShared<kControlT1Threads> shared;
    normalized_front_body<kFp8, kControlT1Threads>(
        persistent::LaunchCta{}, shared, residual, norm, eps, unit_offset, a_weight, b_weight,
        a_log, dt_bias, g, beta, low_codes, high_codes, scale_words, status, hidden, clear_status,
        tokens, fp8_scales, warm, fold);
}

} // namespace ninfer::ops::r9700::gdn::front_kernels
