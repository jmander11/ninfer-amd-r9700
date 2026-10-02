#pragma once

// RoPE coefficient profiles and the Qwen3.8 Text Q/K norm + RoPE kernel (ops::qk_norm_rope),
// shared by the RoPE launcher (rope.hip) and the persistent decode kernel.
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::rope_kernels {

enum class Profile : std::uint8_t { Text1D, TextMrope, Dflash1D, Vision2D };

inline constexpr int kTextD = 256;
inline constexpr int kTextR = 64;
inline constexpr int kTextQ = 24;
inline constexpr int kTextK = 4;
inline constexpr int kDflashD = 128;
inline constexpr int kDflashQ = 32;
inline constexpr int kDflashK = 8;
inline constexpr int kVisionD = 72;
inline constexpr int kVisionH = 16;
inline constexpr int kCombinedBlock = 256;
inline constexpr int kSplitHeads = 4;
inline constexpr int kSplitBlock = kSplitHeads * 32;

namespace {
// Internal linkage: every translation unit carries its own constant copy.
__device__ __constant__ float kTextFrequency[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F,
    1.333521432e-01F, 8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F,
    1.778279410e-02F, 1.074607828e-02F, 6.493816316e-03F, 3.924189758e-03F,
    2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F, 5.232991147e-04F,
    3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F,
    5.623413252e-06F, 3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F,
    7.498942093e-07F, 4.531583638e-07F, 2.738419634e-07F, 1.654817100e-07F,
};

__device__ __constant__ double kDflashFrequency[64] = {
    1.00000000000000000e+00, 7.77365030238775789e-01, 6.04296390238132863e-01,
    4.69758881670649164e-01, 3.65174127254837722e-01, 2.83873596475875456e-01,
    2.20673406908458991e-01, 1.71543789634287902e-01, 1.33352143216332403e-01,
    1.03663292843769794e-01, 8.05842187761481865e-02, 6.26433536656885587e-02,
    4.86967525165863113e-02, 3.78551524925863012e-02, 2.94272717620928173e-02,
    2.28757320031839559e-02, 1.77827941003892293e-02, 1.38237222735789964e-02,
    1.07460782832131743e-02, 8.35362546957826163e-03, 6.49381631576211298e-03,
    5.04806571666747105e-03, 3.92418975848453627e-03, 3.05052789026702539e-03,
    2.37137370566165538e-03, 1.84342299240911056e-03, 1.43301257023696268e-03,
    1.11397385999480246e-03, 8.65964323360065387e-04, 6.73170382414498242e-04,
    5.23299114681494734e-04, 4.06794432108304740e-04, 3.16227766016837939e-04,
    2.45824406892019762e-04, 1.91095297497044048e-04, 1.48550801717277505e-04,
    1.15478198468945822e-04, 8.97687132447314224e-05, 6.97830584859866353e-05,
    5.42469093701132573e-05, 4.21696503428582224e-05, 3.27812115139345850e-05,
    2.54829674797934641e-05, 1.98095677855033870e-05, 1.53992652605949185e-05,
    1.19708503049572999e-05, 9.30572040929699043e-06, 7.23394162736674728e-06,
    5.62341325190349121e-06, 4.37144481261108992e-06, 3.39820832894255927e-06,
    2.64164832038609264e-06, 2.05352502645714607e-06, 1.59633854428794220e-06,
    1.24093776075171953e-06, 9.64661619911199141e-07, 7.49894209332455848e-07,
    5.82941534713607427e-07, 4.53158363760081793e-07, 3.52269465147310129e-07,
    2.73841963426436139e-07, 2.12875166179637264e-07, 1.65481709994318135e-07,
    1.28639694493697462e-07,
};

__device__ __constant__ float kVisionFrequency[18] = {
    1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F,
    1.291549665e-01F, 7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F,
    1.668100537e-02F, 1.000000000e-02F, 5.994842503e-03F, 3.593813664e-03F,
    2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F, 4.641588834e-04F,
    2.782559402e-04F, 1.668100537e-04F,
};

} // namespace

template <Profile P>
__device__ __forceinline__ void coefficient(const std::int32_t* positions, int tokens,
                                            int token, int pair, float* sine, float* cosine) {
    if constexpr (P == Profile::Dflash1D) {
        constexpr double inv_two_pi = 1.59154943091895336e-01;
        constexpr double two_pi = 6.28318530717958648e+00;
        const double angle = static_cast<double>(positions[token]) * kDflashFrequency[pair];
        const float reduced = static_cast<float>(angle - nearbyint(angle * inv_two_pi) * two_pi);
        sincosf(reduced, sine, cosine);
    } else {
        int axis = 0;
        float frequency = 0.0F;
        if constexpr (P == Profile::Vision2D) {
            axis = pair / 18;
            frequency = kVisionFrequency[pair % 18];
        } else {
            if constexpr (P == Profile::TextMrope) { axis = pair % 3; }
            frequency = kTextFrequency[pair];
        }
        const float angle = static_cast<float>(positions[axis * tokens + token]) * frequency;
        sincosf(angle, sine, cosine);
    }
}

// Qwen3.8 Text Q/K normalization and RoPE in one pass: one wave per (token, head), eight features
// per lane. out = RoPE(BF16(x * rsqrt(mean(x^2) + eps) * (w + 1))) with the per-head [256] gain w;
// only features [0,64) rotate (pair i with i + 32, lanes 0..3 with lanes 4..7).
constexpr int kQkNormRopeWaves = 4;
constexpr int kQkNormRopeGroups = (kTextQ + kTextK + kQkNormRopeWaves - 1) / kQkNormRopeWaves;
static_assert((kTextQ + kTextK) % kQkNormRopeWaves == 0);

template <Profile P, class Cta>
__device__ __forceinline__ void qk_norm_rope_body(
    const Cta& cta, const std::int32_t* positions, const hip_bfloat16* q, const hip_bfloat16* k,
    std::int64_t q_stride, std::int64_t k_stride, const hip_bfloat16* q_norm,
    const hip_bfloat16* k_norm, float eps, hip_bfloat16* qn, hip_bfloat16* kn, int tokens) {
    const int block = static_cast<int>(cta.block().x);
    const int token = block / kQkNormRopeGroups;
    const int head = (block % kQkNormRopeGroups) * kQkNormRopeWaves +
                     static_cast<int>(cta.thread()) / 32;
    const int lane = static_cast<int>(cta.thread()) & 31;
    if (token >= tokens) { return; }
    const bool is_query = head < kTextQ;
    const hip_bfloat16* source = is_query
        ? q + static_cast<std::int64_t>(token) * q_stride + static_cast<std::int64_t>(head) * kTextD
        : k + static_cast<std::int64_t>(token) * k_stride +
              static_cast<std::int64_t>(head - kTextQ) * kTextD;
    hip_bfloat16* destination = is_query
        ? qn + (static_cast<std::int64_t>(token) * kTextQ + head) * kTextD
        : kn + (static_cast<std::int64_t>(token) * kTextK + head - kTextQ) * kTextD;
    const uint4 packed = reinterpret_cast<const uint4*>(source)[lane];
    const uint4 gains = reinterpret_cast<const uint4*>(is_query ? q_norm : k_norm)[lane];
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    const std::uint32_t gain_words[4] = {gains.x, gains.y, gains.z, gains.w};
    const auto bf16 = [](const std::uint32_t (&w)[4], int i) {
        return __uint_as_float((i & 1) ? w[i >> 1] & 0xffff0000U : w[i >> 1] << 16);
    };
    float sumsq = 0.0F;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumsq = fmaf(bf16(words, i), bf16(words, i), sumsq);
#pragma unroll
    for (int width = 16; width != 0; width >>= 1) sumsq += __shfl_xor(sumsq, width, 32);
    const float inverse = rsqrtf(sumsq / static_cast<float>(kTextD) + eps);
    float value[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        value[i] = static_cast<float>(
            hip_bfloat16(bf16(words, i) * inverse * (bf16(gain_words, i) + 1.0F)));
    float partner[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) partner[i] = __shfl_xor(value[i], 4, 32);
    std::uint32_t out[4];
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        float pair_out[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int feature = lane * 8 + i + j;
            float result = value[i + j];
            if (feature < kTextR) {
                float sine = 0.0F, cosine = 0.0F;
                coefficient<P>(positions, tokens, token, feature % (kTextR / 2), &sine, &cosine);
                result = feature < kTextR / 2 ? value[i + j] * cosine - partner[i + j] * sine
                                              : value[i + j] * cosine + partner[i + j] * sine;
            }
            pair_out[j] = result;
        }
        out[i / 2] = static_cast<std::uint32_t>(hip_bfloat16(pair_out[0]).data) |
                     (static_cast<std::uint32_t>(hip_bfloat16(pair_out[1]).data) << 16);
    }
    reinterpret_cast<uint4*>(destination)[lane] = uint4{out[0], out[1], out[2], out[3]};
}

template <Profile P>
__global__ __launch_bounds__(kQkNormRopeWaves * 32)
void qk_norm_rope_kernel(const std::int32_t* positions, const hip_bfloat16* q,
                         const hip_bfloat16* k, std::int64_t q_stride, std::int64_t k_stride,
                         const hip_bfloat16* q_norm, const hip_bfloat16* k_norm, float eps,
                         hip_bfloat16* qn, hip_bfloat16* kn, int tokens) {
    qk_norm_rope_body<P>(r9700::persistent::LaunchCta{}, positions, q, k, q_stride, k_stride,
                         q_norm, k_norm, eps, qn, kn, tokens);
}

} // namespace ninfer::ops::rope_kernels
