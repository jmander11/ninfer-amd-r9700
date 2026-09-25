#include "ops/r9700/gdn_recurrence/gdn_prefill_chunked.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {
namespace {

constexpr int kDim = 128;
constexpr int kQHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadsPerQ = kValueHeads / kQHeads;
constexpr int kChunk = 64;
constexpr int kWaves = 8;
constexpr float kEpsilon = 1.0e-6F;
constexpr int kKPitch = kDim + 8;    // FP16 [token][feature] rows
constexpr int kTPitch = kChunk + 8;  // FP16 [feature][token], A and P rows
constexpr int kDiagonalPitch = 24;   // FP16 diagonal-block rows

using F16x8 = _Float16 __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));

struct ChunkShared {
    _Float16 k[kChunk * kKPitch];          // normalized k, token-major
    _Float16 k_transposed[kDim * kTPitch];  // normalized k, feature-major
    _Float16 a[kChunk * kTPitch];           // strictly lower beta_i exp(b_i - b_j) k_i.k_j
    _Float16 diagonal[kChunk * kDiagonalPitch];  // (I + A_II)^-1, row i, column i % 16
    _Float16 p[kChunk * kTPitch];           // scale |q_i|^-1 exp(b_i - b_j) q_i.k_j, i >= j
    float b[kChunk];                        // inclusive cumulative g within the chunk
    float gamma[kChunk];                    // exp(b_i)
    float beta_gamma[kChunk];               // beta_i exp(b_i)
    float beta[kChunk];
    float to_end[kChunk];                   // exp(b_last - b_j)
    float query_scale[kChunk];              // scale / |q_i|
};

// Lower 16x16 tiles (I, J), J <= I, of the 64x64 intra-chunk matrices.
__constant__ int kLowerTiles[10][2] = {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1},
                                       {2, 2}, {3, 0}, {3, 1}, {3, 2}, {3, 3}};

__device__ __forceinline__ F16x8 to_f16(const F32x8& value) {
    F16x8 result;
#pragma unroll
    for (int e = 0; e < 8; ++e) result[e] = _Float16(value[e]);
    return result;
}

__device__ __forceinline__ F16x8 bf16x8_to_f16(const hip_bfloat16* source, bool valid, float scale) {
    uint4 raw{};
    if (valid) raw = *reinterpret_cast<const uint4*>(source);
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    F16x8 result;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        result[2 * i] = _Float16(__uint_as_float(words[i] << 16U) * scale);
        result[2 * i + 1] = _Float16(__uint_as_float(words[i] & 0xffff0000U) * scale);
    }
    return result;
}

__device__ __forceinline__ F32x8 wmma(const F16x8& a, const F16x8& b, const F32x8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// One workgroup per value head. WMMA fragments use tokens/state rows as the lane index
// (lane & 15) and the reduction index as (lane >> 4) * 8 + e, so the S^T accumulator of wave w
// (value rows 16w..16w+15, eight 16-key tiles) is directly the B operand of the next chunk's
// K S^T and q S^T products.
__global__ __launch_bounds__(kWaves * 32) void chunked_kernel(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v, const float* g,
    const float* beta, const float* state_in, float* state_out, hip_bfloat16* out, int width,
    float scale) {
    __shared__ ChunkShared sh;
    const int head = blockIdx.x, query_head = head / kHeadsPerQ;
    const int tid = threadIdx.x, wave = tid / 32, lane = tid & 31;
    const int n = lane & 15, half = lane >> 4;
    const int row = wave * 16 + n;
    F32x8 state[8];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            state[t][e] = state_in[(std::size_t(head) * kDim + row) * kDim + 16 * t + 8 * half + e];
    const int chunks = (width + kChunk - 1) / kChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int t0 = chunk * kChunk;
        // Normalized k in both layouts and the query scales; wave w owns tokens 8w..8w+7.
        for (int r = 0; r < 8; ++r) {
            const int i = wave * 8 + r;
            const bool valid = t0 + i < width;
            const std::size_t base = (std::size_t(t0 + i) * kQHeads + query_head) * kDim + 4 * lane;
            uint2 kr{}, qr{};
            if (valid) {
                kr = *reinterpret_cast<const uint2*>(k + base);
                qr = *reinterpret_cast<const uint2*>(q + base);
            }
            const float kv[4] = {__uint_as_float(kr.x << 16U), __uint_as_float(kr.x & 0xffff0000U),
                                 __uint_as_float(kr.y << 16U), __uint_as_float(kr.y & 0xffff0000U)};
            const float qv[4] = {__uint_as_float(qr.x << 16U), __uint_as_float(qr.x & 0xffff0000U),
                                 __uint_as_float(qr.y << 16U), __uint_as_float(qr.y & 0xffff0000U)};
            float key_sum = 0.0F, query_sum = 0.0F;
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                key_sum = fmaf(kv[it], kv[it], key_sum);
                query_sum = fmaf(qv[it], qv[it], query_sum);
            }
#pragma unroll
            for (int w = 16; w; w >>= 1) {
                key_sum += __shfl_xor(key_sum, w, 32);
                query_sum += __shfl_xor(query_sum, w, 32);
            }
            const float key_inverse = rsqrtf(key_sum + kEpsilon);
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const _Float16 normalized = _Float16(kv[it] * key_inverse);
                sh.k[i * kKPitch + 4 * lane + it] = normalized;
                sh.k_transposed[(4 * lane + it) * kTPitch + i] = normalized;
            }
            if (lane == 0) sh.query_scale[i] = rsqrtf(query_sum + kEpsilon) * scale;
        }
        if (wave == kWaves - 1) {
            // Inclusive scan of g over the chunk, two tokens per lane; padding decays nothing.
            const int t_even = t0 + 2 * lane, t_odd = t_even + 1;
            const float g0 = t_even < width ? g[std::size_t(t_even) * kValueHeads + head] : 0.0F;
            const float g1 = t_odd < width ? g[std::size_t(t_odd) * kValueHeads + head] : 0.0F;
            float sum = g0 + g1;
#pragma unroll
            for (int w = 1; w < 32; w <<= 1) {
                const float other = __shfl_up(sum, w, 32);
                if (lane >= w) sum += other;
            }
            const float before = sum - g0 - g1;
            sh.b[2 * lane] = before + g0;
            sh.b[2 * lane + 1] = before + g0 + g1;
            sh.beta[2 * lane] = t_even < width ? beta[std::size_t(t_even) * kValueHeads + head] : 0.0F;
            sh.beta[2 * lane + 1] = t_odd < width ? beta[std::size_t(t_odd) * kValueHeads + head] : 0.0F;
        }
        __syncthreads();
        if (tid < kChunk) {
            const float gamma = expf(sh.b[tid]);
            sh.gamma[tid] = gamma;
            sh.beta_gamma[tid] = sh.beta[tid] * gamma;
            sh.to_end[tid] = expf(sh.b[kChunk - 1] - sh.b[tid]);
        }
        // A = KK^T and P = QK^T on the ten lower tiles each.
        for (int job = wave; job < 20; job += kWaves) {
            const int I = kLowerTiles[job % 10][0], J = kLowerTiles[job % 10][1];
            const int ia = 16 * I + n;
            const bool query_job = job >= 10;
            const hip_bfloat16* query_row =
                q + (std::size_t(t0 + ia) * kQHeads + query_head) * kDim + 8 * half;
            F32x8 acc = {};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const F16x8 b_operand = *reinterpret_cast<const F16x8*>(
                    &sh.k[(16 * J + n) * kKPitch + 16 * ks + 8 * half]);
                const F16x8 a_operand = query_job
                    ? bf16x8_to_f16(query_row + 16 * ks, t0 + ia < width, 1.0F)
                    : *reinterpret_cast<const F16x8*>(&sh.k[ia * kKPitch + 16 * ks + 8 * half]);
                acc = wmma(a_operand, b_operand, acc);
            }
            const int j = 16 * J + n;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = 16 * I + 8 * half + e;
                const float decay = expf(sh.b[i] - sh.b[j]);
                if (query_job) {
                    sh.p[i * kTPitch + j] =
                        _Float16(i >= j ? sh.query_scale[i] * decay * acc[e] : 0.0F);
                } else {
                    sh.a[i * kTPitch + j] = _Float16(i > j ? sh.beta[i] * decay * acc[e] : 0.0F);
                }
            }
        }
        __syncthreads();
        // Diagonal blocks (I + A_II)^-1 by FP32 forward substitution, one thread per column.
        if (tid < kChunk) {
            const int block = tid >> 4, column = tid & 15, origin = 16 * block;
            float x[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float sum = i == column ? 1.0F : 0.0F;
#pragma unroll
                for (int m = 0; m < i; ++m) {
                    sum = fmaf(-float(sh.a[(origin + i) * kTPitch + origin + m]), x[m], sum);
                }
                x[i] = sum;
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) sh.diagonal[(origin + i) * kDiagonalPitch + column] = _Float16(x[i]);
        }
        __syncthreads();
        F16x8 state_operand[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) state_operand[t] = to_f16(state[t]);
        // Z = beta (V - Gamma K S^T) for this wave's rows (lane n = row, e = token).
        F16x8 z[4];
#pragma unroll
        for (int I = 0; I < 4; ++I) {
            F32x8 acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int token = 16 * I + 8 * half + e;
                acc[e] = t0 + token < width
                    ? sh.beta[token] *
                          float(v[(std::size_t(t0 + token) * kValueHeads + head) * kDim + row])
                    : 0.0F;
            }
            const int token = 16 * I + n;
            const _Float16 factor = _Float16(-sh.beta_gamma[token]);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                F16x8 a_operand = *reinterpret_cast<const F16x8*>(
                    &sh.k[token * kKPitch + 16 * t + 8 * half]);
                a_operand = a_operand * factor;
                acc = wmma(a_operand, state_operand[t], acc);
            }
            z[I] = to_f16(acc);
        }
        // V'_I = (I + A_II)^-1 (Z_I - sum_{J<I} A_IJ V'_J), 16 tokens at a time.
        F16x8 corrected[4];
#pragma unroll
        for (int I = 0; I < 4; ++I) {
            F32x8 residual;
#pragma unroll
            for (int e = 0; e < 8; ++e) residual[e] = float(z[I][e]);
#pragma unroll
            for (int J = 0; J < I; ++J) {
                F16x8 a_operand = *reinterpret_cast<const F16x8*>(
                    &sh.a[(16 * I + n) * kTPitch + 16 * J + 8 * half]);
                a_operand = -a_operand;
                residual = wmma(a_operand, corrected[J], residual);
            }
            const F16x8 diagonal = *reinterpret_cast<const F16x8*>(
                &sh.diagonal[(16 * I + n) * kDiagonalPitch + 8 * half]);
            corrected[I] = to_f16(wmma(diagonal, to_f16(residual), F32x8{}));
        }
        // O = (scale Gamma q / |q|) S^T + P V'.
#pragma unroll
        for (int I = 0; I < 4; ++I) {
            const int token = 16 * I + n;
            const bool valid = t0 + token < width;
            const float factor = sh.query_scale[token] * sh.gamma[token];
            const hip_bfloat16* query_row =
                q + (std::size_t(t0 + token) * kQHeads + query_head) * kDim + 8 * half;
            F32x8 acc = {};
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                acc = wmma(bf16x8_to_f16(query_row + 16 * t, valid, factor), state_operand[t], acc);
            }
#pragma unroll
            for (int J = 0; J <= I; ++J) {
                const F16x8 a_operand = *reinterpret_cast<const F16x8*>(
                    &sh.p[token * kTPitch + 16 * J + 8 * half]);
                acc = wmma(a_operand, corrected[J], acc);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int output_token = t0 + 16 * I + 8 * half + e;
                if (output_token < width) {
                    out[(std::size_t(output_token) * kValueHeads + head) * kDim + row] =
                        hip_bfloat16(acc[e]);
                }
            }
        }
        // S^T <- Gamma_last S^T + (K exp(b_last - b))^T V'.
        const float gamma_last = sh.gamma[kChunk - 1];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            F32x8 acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = state[t][e] * gamma_last;
#pragma unroll
            for (int I = 0; I < 4; ++I) {
                F16x8 a_operand = *reinterpret_cast<const F16x8*>(
                    &sh.k_transposed[(16 * t + n) * kTPitch + 16 * I + 8 * half]);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    a_operand[e] = _Float16(float(a_operand[e]) * sh.to_end[16 * I + 8 * half + e]);
                }
                acc = wmma(a_operand, corrected[I], acc);
            }
            state[t] = acc;
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            state_out[(std::size_t(head) * kDim + row) * kDim + 16 * t + 8 * half + e] = state[t][e];
}

} // namespace

hipError_t prefill_chunked(const hip_bfloat16* query, const hip_bfloat16* key,
                           const hip_bfloat16* value, const float* g, const float* beta,
                           const float* state_in, float* state_out, hip_bfloat16* output,
                           std::int32_t width, float scale, hipStream_t stream) noexcept {
    if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
        beta == nullptr || state_in == nullptr || state_out == nullptr || output == nullptr ||
        width < kChunkedMinimumWidth || !std::isfinite(scale) || stream == nullptr ||
        reinterpret_cast<std::uintptr_t>(query) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(key) % 16U != 0U) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(chunked_kernel, dim3(kValueHeads), dim3(kWaves * 32), 0, stream, query,
                       key, value, g, beta, state_in, state_out, output, width, scale);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::gdn_recurrence
