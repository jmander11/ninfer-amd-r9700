// Native wave32 gfx1201 implementation of the complete Qwen3.8 RoPE public Op boundary.
#include "ninfer/ops/rope.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

enum class Profile : std::uint8_t { Text1D, TextMrope, Dflash1D, Vision2D };

constexpr int kTextD = 256;
constexpr int kTextR = 64;
constexpr int kTextQ = 24;
constexpr int kTextK = 4;
constexpr int kDflashD = 128;
constexpr int kDflashQ = 32;
constexpr int kDflashK = 8;
constexpr int kVisionD = 72;
constexpr int kVisionH = 16;
constexpr int kCombinedBlock = 256;
constexpr int kSplitHeads = 4;
constexpr int kSplitBlock = kSplitHeads * 32;

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

template <int HeadDim, int Half>
__device__ __forceinline__ void apply_head(hip_bfloat16* data, std::int64_t token_stride,
                                           int token, int head, int lane,
                                           const float* sine, const float* cosine) {
    const std::int64_t base = static_cast<std::int64_t>(token) * token_stride +
                              static_cast<std::int64_t>(head) * HeadDim;
    for (int pair = lane; pair < Half; pair += 32) {
        const float first = static_cast<float>(data[base + pair]);
        const float second = static_cast<float>(data[base + pair + Half]);
        data[base + pair] = hip_bfloat16(first * cosine[pair] - second * sine[pair]);
        data[base + pair + Half] =
            hip_bfloat16(second * cosine[pair] + first * sine[pair]);
    }
}

template <Profile P, int HeadDim, int Half, int QHeads, int KHeads, bool Split>
__global__ __launch_bounds__(Split ? kSplitBlock : kCombinedBlock)
void rope_kernel(const std::int32_t* positions, hip_bfloat16* q, hip_bfloat16* k,
                 int tokens, std::int64_t q_stride, std::int64_t k_stride) {
    constexpr int combined_heads = QHeads + KHeads;
    constexpr int groups = (combined_heads + kSplitHeads - 1) / kSplitHeads;
    const int token = Split ? static_cast<int>(blockIdx.x) / groups
                            : static_cast<int>(blockIdx.x);
    const int group = Split ? static_cast<int>(blockIdx.x) % groups : 0;
    if (token >= tokens) { return; }

    __shared__ float sine[Half];
    __shared__ float cosine[Half];
    for (int pair = static_cast<int>(threadIdx.x); pair < Half; pair += blockDim.x) {
        coefficient<P>(positions, tokens, token, pair, &sine[pair], &cosine[pair]);
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wave = static_cast<int>(threadIdx.x) >> 5;
    constexpr int waves = Split ? kSplitHeads : kCombinedBlock / 32;
    int combined = Split ? group * kSplitHeads + wave : wave;
    for (; combined < combined_heads && (!Split || combined < (group + 1) * kSplitHeads);
         combined += waves) {
        if (combined < QHeads) {
            apply_head<HeadDim, Half>(q, q_stride, token, combined, lane, sine, cosine);
        } else {
            apply_head<HeadDim, Half>(k, k_stride, token, combined - QHeads, lane, sine, cosine);
        }
    }
}

// Qwen3.8 Text Q/K normalization and RoPE in one pass: one wave per (token, head), eight features
// per lane. out = RoPE(BF16(x * rsqrt(mean(x^2) + eps) * (w + 1))) with the per-head [256] gain w;
// only features [0,64) rotate (pair i with i + 32, lanes 0..3 with lanes 4..7).
constexpr int kQkNormRopeWaves = 4;
constexpr int kQkNormRopeGroups = (kTextQ + kTextK + kQkNormRopeWaves - 1) / kQkNormRopeWaves;
static_assert((kTextQ + kTextK) % kQkNormRopeWaves == 0);

template <Profile P>
__global__ __launch_bounds__(kQkNormRopeWaves * 32)
void qk_norm_rope_kernel(const std::int32_t* positions, const hip_bfloat16* q,
                         const hip_bfloat16* k, std::int64_t q_stride, std::int64_t k_stride,
                         const hip_bfloat16* q_norm, const hip_bfloat16* k_norm, float eps,
                         hip_bfloat16* qn, hip_bfloat16* kn, int tokens) {
    const int token = static_cast<int>(blockIdx.x) / kQkNormRopeGroups;
    const int head = (static_cast<int>(blockIdx.x) % kQkNormRopeGroups) * kQkNormRopeWaves +
                     static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
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

template <Profile P, int HeadDim, int Half, int QHeads, int KHeads>
void launch_profile(const std::int32_t* positions, hip_bfloat16* q, hip_bfloat16* k,
                    int tokens, std::int64_t q_stride, std::int64_t k_stride,
                    hipStream_t stream) {
    if (tokens <= 8) {
        constexpr int groups = (QHeads + KHeads + kSplitHeads - 1) / kSplitHeads;
        rope_kernel<P, HeadDim, Half, QHeads, KHeads, true>
            <<<tokens * groups, kSplitBlock, 0, stream>>>(positions, q, k, tokens, q_stride,
                                                          k_stride);
    } else {
        rope_kernel<P, HeadDim, Half, QHeads, KHeads, false>
            <<<tokens, kCombinedBlock, 0, stream>>>(positions, q, k, tokens, q_stride, k_stride);
    }
}

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    bool zero = false;
    std::int64_t count = 1;
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] < 0) {
            throw std::invalid_argument(std::string("rope: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (tensor.ne[dim] == 0) { zero = true; continue; }
        if (count > std::numeric_limits<std::int64_t>::max() / tensor.ne[dim]) {
            throw std::overflow_error("rope: tensor size overflows int64");
        }
        count *= tensor.ne[dim];
    }
    return zero ? 0 : count;
}

int axes_for(const Tensor& positions, int tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_layout(const Tensor& tensor, const char* label, int head_dim, int heads, int tokens) {
    if (tensor.ne[0] != head_dim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = sizeof(hip_bfloat16);
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(head_dim) * heads ||
        tensor.nb[2] % elem != 0) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " strides");
    }
}

Profile pair_profile(int axes, int rotary_dim, float theta, const Tensor& q, const Tensor& k) {
    if ((axes == 1 || axes == 3) && rotary_dim == kTextR && theta == 1.0e7F &&
        q.ne[0] == kTextD && q.ne[1] == kTextQ && k.ne[0] == kTextD && k.ne[1] == kTextK) {
        return axes == 1 ? Profile::Text1D : Profile::TextMrope;
    }
    if (axes == 1 && rotary_dim == kDflashD && theta == 1.0e7F &&
        q.ne[0] == kDflashD && q.ne[1] == kDflashQ && k.ne[0] == kDflashD &&
        k.ne[1] == kDflashK) {
        return Profile::Dflash1D;
    }
    if (axes == 2 && rotary_dim == kVisionD && theta == 10'000.0F &&
        q.ne[0] == kVisionD && q.ne[1] == kVisionH && k.ne[0] == kVisionD &&
        k.ne[1] == kVisionH) {
        return Profile::Vision2D;
    }
    throw std::invalid_argument("rope: unsupported Qwen3.8 pair geometry");
}

Profile single_profile(int axes, int rotary_dim, float theta, const Tensor& x) {
    if ((axes == 1 || axes == 3) && rotary_dim == kTextR && theta == 1.0e7F &&
        x.ne[0] == kTextD && (x.ne[1] == kTextQ || x.ne[1] == kTextK)) {
        return axes == 1 ? Profile::Text1D : Profile::TextMrope;
    }
    if (axes == 1 && rotary_dim == kDflashD && theta == 1.0e7F && x.ne[0] == kDflashD &&
        (x.ne[1] == kDflashQ || x.ne[1] == kDflashK)) {
        return Profile::Dflash1D;
    }
    if (axes == 2 && rotary_dim == kVisionD && theta == 10'000.0F &&
        x.ne[0] == kVisionD && x.ne[1] == kVisionH) {
        return Profile::Vision2D;
    }
    throw std::invalid_argument("rope: unsupported Qwen3.8 single-tensor geometry");
}

std::size_t storage_bytes(const Tensor& tensor) {
    if (tensor.ne[2] == 0) { return 0; }
    return static_cast<std::size_t>((tensor.ne[2] - 1) * tensor.nb[2] +
                                    tensor.ne[1] * tensor.nb[1]);
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto aa = reinterpret_cast<std::uintptr_t>(a);
    const auto bb = reinterpret_cast<std::uintptr_t>(b);
    return a_bytes != 0 && b_bytes != 0 && aa < bb + b_bytes && bb < aa + a_bytes;
}

bool tensor_payloads_overlap(const Tensor& a, const Tensor& b) {
    if (!overlaps(a.data, storage_bytes(a), b.data, storage_bytes(b))) { return false; }
    // Q/K may be disjoint planes of token-interleaved QKV. Their bounding spans include
    // padding owned by the other planes, which this Op never reads or writes.
    const auto a_width = static_cast<std::uintptr_t>(a.ne[1] * a.nb[1]);
    const auto b_width = static_cast<std::uintptr_t>(b.ne[1] * b.nb[1]);
    auto aa = reinterpret_cast<std::uintptr_t>(a.data);
    auto bb = reinterpret_cast<std::uintptr_t>(b.data);
    std::int32_t ai = 0, bi = 0;
    while (ai < a.ne[2] && bi < b.ne[2]) {
        if (aa < bb + b_width && bb < aa + a_width) { return true; }
        if (aa + a_width <= bb) {
            ++ai;
            aa += a.nb[2];
        } else {
            ++bi;
            bb += b.nb[2];
        }
    }
    return false;
}

void require_common(const Tensor& positions, int rotary_dim, float theta) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || !(theta > 0.0F) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: invalid rotary_dim or theta");
    }
    (void)checked_numel(positions, "positions");
}

void launch(Profile profile, const Tensor& positions, Tensor& q, Tensor* k, hipStream_t stream) {
    auto* q_data = static_cast<hip_bfloat16*>(q.data);
    auto* k_data = k == nullptr ? nullptr : static_cast<hip_bfloat16*>(k->data);
    const auto* pos = static_cast<const std::int32_t*>(positions.data);
    const int tokens = q.ne[2];
    const auto qs = q.nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    const auto ks = k == nullptr ? 0 : k->nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    switch (profile) {
    case Profile::Text1D:
        if (k != nullptr) launch_profile<Profile::Text1D, 256, 32, 24, 4>(pos, q_data, k_data, tokens, qs, ks, stream);
        else if (q.ne[1] == 24) launch_profile<Profile::Text1D, 256, 32, 24, 0>(pos, q_data, nullptr, tokens, qs, 0, stream);
        else launch_profile<Profile::Text1D, 256, 32, 4, 0>(pos, q_data, nullptr, tokens, qs, 0, stream);
        break;
    case Profile::TextMrope:
        if (k != nullptr) launch_profile<Profile::TextMrope, 256, 32, 24, 4>(pos, q_data, k_data, tokens, qs, ks, stream);
        else if (q.ne[1] == 24) launch_profile<Profile::TextMrope, 256, 32, 24, 0>(pos, q_data, nullptr, tokens, qs, 0, stream);
        else launch_profile<Profile::TextMrope, 256, 32, 4, 0>(pos, q_data, nullptr, tokens, qs, 0, stream);
        break;
    case Profile::Dflash1D:
        if (k != nullptr) launch_profile<Profile::Dflash1D, 128, 64, 32, 8>(pos, q_data, k_data, tokens, qs, ks, stream);
        else if (q.ne[1] == 32) launch_profile<Profile::Dflash1D, 128, 64, 32, 0>(pos, q_data, nullptr, tokens, qs, 0, stream);
        else launch_profile<Profile::Dflash1D, 128, 64, 8, 0>(pos, q_data, nullptr, tokens, qs, 0, stream);
        break;
    case Profile::Vision2D:
        if (k != nullptr) launch_profile<Profile::Vision2D, 72, 36, 16, 16>(pos, q_data, k_data, tokens, qs, ks, stream);
        else launch_profile<Profile::Vision2D, 72, 36, 16, 0>(pos, q_data, nullptr, tokens, qs, 0, stream);
        break;
    }
    HIP_CHECK(hipGetLastError());
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          hipStream_t stream) {
    require_common(positions, rotary_dim, theta);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    const auto q_count = checked_numel(q, "q");
    (void)checked_numel(k, "k");
    const int tokens = q.ne[2];
    const int axes = axes_for(positions, tokens);
    const Profile profile = pair_profile(axes, rotary_dim, theta, q, k);
    require_layout(q, "q", q.ne[0], q.ne[1], tokens);
    require_layout(k, "k", k.ne[0], k.ne[1], tokens);
    if (q_count == 0) { return; }
    if (positions.data == nullptr || q.data == nullptr || k.data == nullptr ||
        !positions.is_contiguous()) {
        throw std::invalid_argument("rope: storage must be non-null and positions contiguous");
    }
    const std::size_t p_bytes = positions.bytes();
    const std::size_t q_bytes = storage_bytes(q);
    const std::size_t k_bytes = storage_bytes(k);
    if (tensor_payloads_overlap(q, k) ||
        overlaps(positions.data, p_bytes, q.data, q_bytes) ||
        overlaps(positions.data, p_bytes, k.data, k_bytes)) {
        throw std::invalid_argument("rope: positions, q, and k storage must not overlap");
    }
    launch(profile, positions, q, &k, stream);
}

void qk_norm_rope(const Tensor& positions, int rotary_dim, float theta, const Tensor& q,
                  const Tensor& k, const Tensor& q_norm, const Tensor& k_norm, float eps,
                  Tensor& qn, Tensor& kn, hipStream_t stream) {
    require_common(positions, rotary_dim, theta);
    const int tokens = q.ne[2];
    const int axes = axes_for(positions, tokens);
    const Profile profile = pair_profile(axes, rotary_dim, theta, q, k);
    if (profile != Profile::Text1D && profile != Profile::TextMrope) {
        throw std::invalid_argument("qk_norm_rope: only the Qwen3.8 Text geometry is supported");
    }
    const auto bf16_vector = [](const Tensor& t, int n) {
        return t.dtype == DType::BF16 && t.data != nullptr && t.is_contiguous() && t.ne[0] == n &&
               t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1 &&
               reinterpret_cast<std::uintptr_t>(t.data) % 16U == 0U;
    };
    const auto aligned_rows = [](const Tensor& t) {
        return t.dtype == DType::BF16 && t.data != nullptr &&
               reinterpret_cast<std::uintptr_t>(t.data) % 16U == 0U && t.nb[2] % 16 == 0;
    };
    if (!(eps > 0.0F) || !std::isfinite(eps) || !bf16_vector(q_norm, kTextD) ||
        !bf16_vector(k_norm, kTextD) || !aligned_rows(q) || !aligned_rows(k) ||
        k.ne[2] != tokens || qn.dtype != DType::BF16 || kn.dtype != DType::BF16 ||
        !qn.is_contiguous() || !kn.is_contiguous() || qn.ne[0] != kTextD ||
        qn.ne[1] != kTextQ || qn.ne[2] != tokens || kn.ne[0] != kTextD ||
        kn.ne[1] != kTextK || kn.ne[2] != tokens ||
        reinterpret_cast<std::uintptr_t>(qn.data) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(kn.data) % 16U != 0U) {
        throw std::invalid_argument("qk_norm_rope: invalid tensors");
    }
    require_layout(q, "q", kTextD, kTextQ, tokens);
    require_layout(k, "k", kTextD, kTextK, tokens);
    if (tokens == 0) { return; }
    const std::size_t p_bytes = positions.bytes();
    const std::size_t outputs[2] = {qn.bytes(), kn.bytes()};
    const void* output_data[2] = {qn.data, kn.data};
    for (int i = 0; i < 2; ++i) {
        if (overlaps(output_data[i], outputs[i], q.data, storage_bytes(q)) ||
            overlaps(output_data[i], outputs[i], k.data, storage_bytes(k)) ||
            overlaps(output_data[i], outputs[i], positions.data, p_bytes)) {
            throw std::invalid_argument("qk_norm_rope: outputs must not overlap inputs");
        }
    }
    if (overlaps(qn.data, outputs[0], kn.data, outputs[1])) {
        throw std::invalid_argument("qk_norm_rope: outputs must not overlap");
    }
    const auto qs = q.nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    const auto ks = k.nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    const dim3 grid(static_cast<unsigned>(tokens * kQkNormRopeGroups));
    const auto* pos = static_cast<const std::int32_t*>(positions.data);
    const auto* q_data = static_cast<const hip_bfloat16*>(q.data);
    const auto* k_data = static_cast<const hip_bfloat16*>(k.data);
    const auto* qw = static_cast<const hip_bfloat16*>(q_norm.data);
    const auto* kw = static_cast<const hip_bfloat16*>(k_norm.data);
    auto* qo = static_cast<hip_bfloat16*>(qn.data);
    auto* ko = static_cast<hip_bfloat16*>(kn.data);
    if (profile == Profile::Text1D) {
        qk_norm_rope_kernel<Profile::Text1D><<<grid, kQkNormRopeWaves * 32, 0, stream>>>(
            pos, q_data, k_data, qs, ks, qw, kw, eps, qo, ko, tokens);
    } else {
        qk_norm_rope_kernel<Profile::TextMrope><<<grid, kQkNormRopeWaves * 32, 0, stream>>>(
            pos, q_data, k_data, qs, ks, qw, kw, eps, qo, ko, tokens);
    }
    HIP_CHECK(hipGetLastError());
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, hipStream_t stream) {
    require_common(positions, rotary_dim, theta);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    const auto count = checked_numel(x, "tensor");
    const int tokens = x.ne[2];
    const int axes = axes_for(positions, tokens);
    const Profile profile = single_profile(axes, rotary_dim, theta, x);
    require_layout(x, "tensor", x.ne[0], x.ne[1], tokens);
    if (count == 0) { return; }
    if (positions.data == nullptr || x.data == nullptr || !positions.is_contiguous()) {
        throw std::invalid_argument("rope: storage must be non-null and positions contiguous");
    }
    if (overlaps(positions.data, positions.bytes(), x.data, storage_bytes(x))) {
        throw std::invalid_argument("rope: positions and tensor storage must not overlap");
    }
    launch(profile, positions, x, nullptr, stream);
}

} // namespace ninfer::ops
