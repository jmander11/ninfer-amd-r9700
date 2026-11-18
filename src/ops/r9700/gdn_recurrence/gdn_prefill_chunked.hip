#include "ops/r9700/gdn_recurrence/gdn_prefill_chunked.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {
namespace {

constexpr int kDim           = 128;
constexpr int kQHeads        = 16;
constexpr int kValueHeads    = 48;
constexpr int kHeadsPerQ     = kValueHeads / kQHeads;
constexpr int kChunk         = 64;
constexpr int kWaves         = 8;
constexpr float kEpsilon     = 1.0e-6F;
constexpr int kKPitch        = kDim + 8;   // FP16 [token][feature] rows
constexpr int kTPitch        = kChunk + 8; // FP16 [feature][token], A and P rows
constexpr int kDiagonalPitch = 24;         // FP16 diagonal-block rows
constexpr int kTile          = 16 * 16;
constexpr int kOffDiagonal   = 0;  // six strictly lower A_IJ (J < I) record tiles
constexpr int kDiagonal      = 6;  // four (I + A_II)^-1 record tiles
constexpr int kQueryKey      = 10; // ten lower P_IJ (J <= I) record tiles
constexpr int kRecordTiles   = 20;

using F16x8 = _Float16 __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));
using F16x4 = _Float16 __attribute__((ext_vector_type(4)));

struct ChunkShared {
    _Float16 k[kChunk * kKPitch];               // normalized k, token-major
    _Float16 k_transposed[kDim * kTPitch];      // normalized k, feature-major
    _Float16 a[kChunk * kTPitch];               // strictly lower beta_i exp(b_i - b_j) k_i.k_j
    _Float16 diagonal[kChunk * kDiagonalPitch]; // (I + A_II)^-1, row i, column i % 16
    _Float16 p[kChunk * kTPitch];               // scale |q_i|^-1 exp(b_i - b_j) q_i.k_j, i >= j
    float b[kChunk];                            // inclusive cumulative g within the chunk
    float gamma[kChunk];                        // exp(b_i)
    float beta_gamma[kChunk];                   // beta_i exp(b_i)
    float beta[kChunk];
    float to_end[kChunk];      // exp(b_last - b_j)
    float query_scale[kChunk]; // scale / |q_i|
};

// Everything one value head's recurrence needs from one chunk besides the shared keys and
// queries. Tiles are row-major 16x16, so a WMMA A fragment (row n, columns 8*half..+7) is one
// contiguous 16-byte vector.
struct alignas(16) ChunkRecord {
    _Float16 tiles[kRecordTiles * kTile];
    float beta[kChunk];
    float beta_gamma[kChunk]; // beta_i exp(b_i)
    float gamma[kChunk];      // exp(b_i)
    float to_end[kChunk];     // exp(b_last - b_j)
};

static_assert(sizeof(ChunkRecord) % 16 == 0);

__device__ __forceinline__ int off_diagonal_tile(int I, int J) { return I * (I - 1) / 2 + J; }

__device__ __forceinline__ int query_key_tile(int I, int J) { return I * (I + 1) / 2 + J; }

// Lower 16x16 tiles (I, J), J <= I, of the 64x64 intra-chunk matrices.
__constant__ int kLowerTiles[10][2] = {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1},
                                       {2, 2}, {3, 0}, {3, 1}, {3, 2}, {3, 3}};

__device__ __forceinline__ F16x8 to_f16(const F32x8& value) {
    F16x8 result;
#pragma unroll
    for (int e = 0; e < 8; ++e) result[e] = _Float16(value[e]);
    return result;
}

__device__ __forceinline__ F16x8 bf16x8_to_f16(const hip_bfloat16* source, bool valid,
                                               float scale) {
    uint4 raw{};
    if (valid) raw = *reinterpret_cast<const uint4*>(source);
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    F16x8 result;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        result[2 * i]     = _Float16(__uint_as_float(words[i] << 16U) * scale);
        result[2 * i + 1] = _Float16(__uint_as_float(words[i] & 0xffff0000U) * scale);
    }
    return result;
}

// Sum over all 32 lanes: DPP row_xmask within each 16-lane row, then across the two rows.
__device__ __forceinline__ float wave_sum(float value) {
    value += __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x161, 0xf, 0xf, false));
    value += __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x162, 0xf, 0xf, false));
    value += __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x164, 0xf, 0xf, false));
    value += __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x168, 0xf, 0xf, false));
    return value + __shfl_xor(value, 16, 32);
}

__device__ __forceinline__ F32x8 wmma(const F16x8& a, const F16x8& b, const F32x8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// Chunk-parallel, state-independent work for one Q/K head and its three value heads: the
// normalized FP16 keys and query scale (shared), and per value head the decayed A and P tiles,
// the diagonal inverses (I + A_II)^-1 and the per-token decay scalars.
__global__ __launch_bounds__(kWaves *
                             32) void prepare_chunks(const hip_bfloat16* q, const hip_bfloat16* k,
                                                     const float* g, const float* beta,
                                                     _Float16* keys, float* query_scales,
                                                     ChunkRecord* records, int width, float scale) {
    struct alignas(16) Shared {
        _Float16 k[kChunk * kKPitch];
        float query_scale[kChunk];
        float b[kHeadsPerQ][kChunk];
        float beta[kHeadsPerQ][kChunk];
        alignas(16) _Float16 a_diagonal[kHeadsPerQ][4 * kTile]; // A_II, row-major per tile
        alignas(16) _Float16 inverse[kHeadsPerQ][4 * kTile];    // (I + A_II)^-1, row-major per tile
    };

    __shared__ Shared sh;
    const int chunk = blockIdx.x, query_head = blockIdx.y;
    const int tid = threadIdx.x, wave = tid / 32, lane = tid & 31;
    const int n = lane & 15, half = lane >> 4, t0 = chunk * kChunk;
    // Normalized k and the query scales; wave w owns tokens 8w..8w+7 and lane l features
    // 4l..4l+3. All loads are issued first; padded tokens read a clamped valid row and are zeroed.
    {
        uint2 key_raw[8], query_raw[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int source       = min(t0 + wave * 8 + r, width - 1);
            const std::size_t base = (std::size_t(source) * kQHeads + query_head) * kDim + 4 * lane;
            key_raw[r]             = *reinterpret_cast<const uint2*>(k + base);
            query_raw[r]           = *reinterpret_cast<const uint2*>(q + base);
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int i      = wave * 8 + r;
            const float live = t0 + i < width ? 1.0F : 0.0F;
            const uint2 kr = key_raw[r], qr = query_raw[r];
            const float kv[4] = {__uint_as_float(kr.x << 16U), __uint_as_float(kr.x & 0xffff0000U),
                                 __uint_as_float(kr.y << 16U), __uint_as_float(kr.y & 0xffff0000U)};
            const float qv[4] = {__uint_as_float(qr.x << 16U), __uint_as_float(qr.x & 0xffff0000U),
                                 __uint_as_float(qr.y << 16U), __uint_as_float(qr.y & 0xffff0000U)};
            float key_sum = 0.0F, query_sum = 0.0F;
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                key_sum   = fmaf(kv[it], kv[it], key_sum);
                query_sum = fmaf(qv[it], qv[it], query_sum);
            }
            key_sum                 = wave_sum(key_sum);
            query_sum               = wave_sum(query_sum);
            const float key_inverse = rsqrtf(key_sum + kEpsilon) * live;
            _Float16 normalized[4];
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                normalized[it]                    = _Float16(kv[it] * key_inverse);
                sh.k[i * kKPitch + 4 * lane + it] = normalized[it];
            }
            *reinterpret_cast<uint2*>(
                &keys[((std::size_t(chunk) * kQHeads + query_head) * kChunk + i) * kDim +
                      4 * lane]) = *reinterpret_cast<const uint2*>(normalized);
            if (lane == 0) {
                const float query_scale = rsqrtf(query_sum + kEpsilon) * scale * live;
                sh.query_scale[i]       = query_scale;
                query_scales[(std::size_t(chunk) * kQHeads + query_head) * kChunk + i] =
                    query_scale;
            }
        }
    }
    if (wave < kHeadsPerQ) {
        // Inclusive scan of g over the chunk, two tokens per lane; padding decays nothing.
        const int head   = query_head * kHeadsPerQ + wave;
        const int t_even = t0 + 2 * lane, t_odd = t_even + 1;
        const float g0 = t_even < width ? g[std::size_t(t_even) * kValueHeads + head] : 0.0F;
        const float g1 = t_odd < width ? g[std::size_t(t_odd) * kValueHeads + head] : 0.0F;
        float sum      = g0 + g1;
#pragma unroll
        for (int w = 1; w < 32; w <<= 1) {
            const float other = __shfl_up(sum, w, 32);
            if (lane >= w) sum += other;
        }
        const float before       = sum - g0 - g1;
        sh.b[wave][2 * lane]     = before + g0;
        sh.b[wave][2 * lane + 1] = before + g0 + g1;
        sh.beta[wave][2 * lane] =
            t_even < width ? beta[std::size_t(t_even) * kValueHeads + head] : 0.0F;
        sh.beta[wave][2 * lane + 1] =
            t_odd < width ? beta[std::size_t(t_odd) * kValueHeads + head] : 0.0F;
    }
    __syncthreads();
    ChunkRecord* chunk_records =
        records + std::size_t(chunk) * kValueHeads + std::size_t(query_head) * kHeadsPerQ;
    if (tid < kHeadsPerQ * kChunk) {
        const int h = tid / kChunk, i = tid % kChunk;
        ChunkRecord& record  = chunk_records[h];
        const float gamma    = expf(sh.b[h][i]);
        record.beta[i]       = sh.beta[h][i];
        record.gamma[i]      = gamma;
        record.beta_gamma[i] = sh.beta[h][i] * gamma;
        record.to_end[i]     = expf(sh.b[h][kChunk - 1] - sh.b[h][i]);
    }
    // A = KK^T and P = QK^T on the ten lower tiles each, decayed per value head. The products
    // are formed transposed (keys of block J as the A operand), so lane n holds row i = 16I + n
    // at columns j = 16J + 8*half + e: one contiguous 16-byte row segment of the tile.
    for (int job = wave; job < 20; job += kWaves) {
        const int I = kLowerTiles[job % 10][0], J = kLowerTiles[job % 10][1];
        const int ia         = 16 * I + n;
        const bool query_job = job >= 10;
        const hip_bfloat16* query_row =
            q + (std::size_t(t0 + ia) * kQHeads + query_head) * kDim + 8 * half;
        F32x8 acc = {};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const F16x8 key_operand =
                *reinterpret_cast<const F16x8*>(&sh.k[(16 * J + n) * kKPitch + 16 * ks + 8 * half]);
            const F16x8 row_operand =
                query_job
                    ? bf16x8_to_f16(query_row + 16 * ks, t0 + ia < width, 1.0F)
                    : *reinterpret_cast<const F16x8*>(&sh.k[ia * kKPitch + 16 * ks + 8 * half]);
            acc = wmma(key_operand, row_operand, acc);
        }
        const int i = ia;
#pragma unroll
        for (int h = 0; h < kHeadsPerQ; ++h) {
            F16x8 values;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j       = 16 * J + 8 * half + e;
                const float decay = expf(sh.b[h][i] - sh.b[h][j]);
                values[e] = query_job ? _Float16(i >= j ? sh.query_scale[i] * decay * acc[e] : 0.0F)
                                      : _Float16(i > j ? sh.beta[h][i] * decay * acc[e] : 0.0F);
            }
            _Float16* tile =
                query_job ? &chunk_records[h].tiles[(kQueryKey + query_key_tile(I, J)) * kTile]
                : I == J
                    ? &sh.a_diagonal[h][I * kTile]
                    : &chunk_records[h].tiles[(kOffDiagonal + off_diagonal_tile(I, J)) * kTile];
            *reinterpret_cast<F16x8*>(&tile[n * 16 + 8 * half]) = values;
        }
    }
    __syncthreads();
    // Diagonal blocks (I + A_II)^-1 by FP32 forward substitution, one thread per column.
    if (tid < kHeadsPerQ * kChunk) {
        const int h = tid / kChunk, block = (tid % kChunk) >> 4, column = tid & 15;
        const _Float16* a = &sh.a_diagonal[h][block * kTile];
        float x[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float sum = i == column ? 1.0F : 0.0F;
#pragma unroll
            for (int m = 0; m < i; ++m) sum = fmaf(-float(a[i * 16 + m]), x[m], sum);
            x[i] = sum;
        }
        _Float16* inverse = &sh.inverse[h][block * kTile];
#pragma unroll
        for (int i = 0; i < 16; ++i) inverse[i * 16 + column] = _Float16(x[i]);
    }
    __syncthreads();
    constexpr int kInverseVectors = kHeadsPerQ * 4 * kTile * int(sizeof(_Float16)) / 16;
    for (int vector = tid; vector < kInverseVectors; vector += kWaves * 32) {
        const int h      = vector / (kInverseVectors / kHeadsPerQ);
        const int offset = vector % (kInverseVectors / kHeadsPerQ);
        reinterpret_cast<uint4*>(&chunk_records[h].tiles[kDiagonal * kTile])[offset] =
            reinterpret_cast<const uint4*>(sh.inverse[h])[offset];
    }
}

// One workgroup per value head walks the chunks. WMMA fragments use tokens/state rows as the
// lane index (lane & 15) and the reduction index as (lane >> 4) * 8 + e, so the S^T accumulator
// of wave w (value rows 16w..16w+15, eight 16-key tiles) is directly the B operand of the next
// chunk's K S^T and q S^T products. Each chunk stages its prepared operands with one load round,
// and the values of the next chunk are prefetched during the current one.
__global__ __launch_bounds__(kWaves * 32) void prepared_kernel(
    const hip_bfloat16* q, const hip_bfloat16* v, const float* state_in, float* state_out,
    hip_bfloat16* out, int width, const _Float16* keys, const float* query_scales,
    const ChunkRecord* records) {
    struct alignas(16) Shared {
        _Float16 k[kChunk * kKPitch];          // -beta_i Gamma_i k_i, token-major
        _Float16 k_transposed[kDim * kTPitch]; // exp(b_last - b_j) k_j, feature-major
        _Float16 q[kChunk * kKPitch];          // scale Gamma_i q_i / |q_i|, token-major
        ChunkRecord record;
    };

    __shared__ Shared sh;
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
    // Values of this lane's row for tokens 16I + 8*half + e; padded tokens read a valid row
    // (beta is 0 there).
    hip_bfloat16 values[4][8];
    const hip_bfloat16* value_base = v + head * kDim + row;
    constexpr int kTokenStride     = kValueHeads * kDim;
#define NINFER_GDN_LOAD_VALUES(CHUNK)                                                              \
    if ((CHUNK) * kChunk + kChunk <= width) {                                                      \
        const hip_bfloat16* chunk_values = value_base + (CHUNK) * kChunk * kTokenStride;           \
        _Pragma("unroll") for (int I = 0; I < 4; ++I)                                              \
            _Pragma("unroll") for (int e = 0; e < 8; ++e) values[I][e] =                           \
                chunk_values[(16 * I + 8 * half + e) * kTokenStride];                              \
    } else {                                                                                       \
        _Pragma("unroll") for (int I = 0; I < 4; ++I) _Pragma("unroll") for (int e = 0; e < 8;     \
                                                                             ++e) values[I][e] =   \
            value_base[min((CHUNK) * kChunk + 16 * I + 8 * half + e, width - 1) * kTokenStride];   \
    }
    NINFER_GDN_LOAD_VALUES(0)
    const int chunks = (width + kChunk - 1) / kChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int t0 = chunk * kChunk;
        {
            // One load round stages the chunk's operands with every per-token factor applied
            // once for all eight waves. Keys: wave w tokens 8w..8w+7, lane l features 4l..4l+3.
            // Queries: thread token tid/4, features 32*(tid%4)..+31.
            const ChunkRecord& record = records[std::size_t(chunk) * kValueHeads + head];
            uint2 key_raw[8];
#pragma unroll
            for (int r = 0; r < 8; ++r)
                key_raw[r] = *reinterpret_cast<const uint2*>(
                    &keys[((std::size_t(chunk) * kQHeads + query_head) * kChunk + wave * 8 + r) *
                              kDim +
                          4 * lane]);
            const float4 beta_gamma0 =
                *reinterpret_cast<const float4*>(&record.beta_gamma[wave * 8]);
            const float4 beta_gamma1 =
                *reinterpret_cast<const float4*>(&record.beta_gamma[wave * 8 + 4]);
            const float4 to_end0  = *reinterpret_cast<const float4*>(&record.to_end[wave * 8]);
            const float4 to_end1  = *reinterpret_cast<const float4*>(&record.to_end[wave * 8 + 4]);
            const int query_token = tid >> 2, query_part = tid & 3;
            const bool query_live     = t0 + query_token < width;
            const uint4* query_source = reinterpret_cast<const uint4*>(
                q + (std::size_t(min(t0 + query_token, width - 1)) * kQHeads + query_head) * kDim +
                32 * query_part);
            const uint4 query_raw0 = query_source[0], query_raw1 = query_source[1];
            const uint4 query_raw2 = query_source[2], query_raw3 = query_source[3];
            const float query_factor =
                query_scales[(std::size_t(chunk) * kQHeads + query_head) * kChunk + query_token] *
                record.gamma[query_token];
            constexpr int kRecordVectors = int(sizeof(ChunkRecord)) / 16;
            const uint4* record_source   = reinterpret_cast<const uint4*>(&record);
            static_assert(kRecordVectors > 2 * kWaves * 32 && kRecordVectors <= 3 * kWaves * 32);
            const bool record_tail  = 2 * kWaves * 32 + tid < kRecordVectors;
            const uint4 record_raw0 = record_source[tid];
            const uint4 record_raw1 = record_source[kWaves * 32 + tid];
            const uint4 record_raw2 = record_source[record_tail ? 2 * kWaves * 32 + tid : tid];

            const float beta_gamma[8] = {beta_gamma0.x, beta_gamma0.y, beta_gamma0.z,
                                         beta_gamma0.w, beta_gamma1.x, beta_gamma1.y,
                                         beta_gamma1.z, beta_gamma1.w};
            const float to_end[8]     = {to_end0.x, to_end0.y, to_end0.z, to_end0.w,
                                         to_end1.x, to_end1.y, to_end1.z, to_end1.w};
            _Float16 transposed[4][8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const F16x4 normalized = __builtin_bit_cast(F16x4, key_raw[r]);
                *reinterpret_cast<F16x4*>(&sh.k[(wave * 8 + r) * kKPitch + 4 * lane]) =
                    normalized * _Float16(-beta_gamma[r]);
#pragma unroll
                for (int it = 0; it < 4; ++it)
                    transposed[it][r] = _Float16(float(normalized[it]) * to_end[r]);
            }
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                F16x8 features;
#pragma unroll
                for (int r = 0; r < 8; ++r) features[r] = transposed[it][r];
                *reinterpret_cast<F16x8*>(&sh.k_transposed[(4 * lane + it) * kTPitch + wave * 8]) =
                    features;
            }
            // Past the width the raw query is zero, exactly as the unstaged masked read.
            const unsigned query_mask = query_live ? ~0U : 0U;
            const uint4 query_raw[4]  = {query_raw0, query_raw1, query_raw2, query_raw3};
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint4 raw         = query_raw[p];
                const unsigned words[4] = {raw.x & query_mask, raw.y & query_mask,
                                           raw.z & query_mask, raw.w & query_mask};
                F16x8 converted;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    converted[2 * w] = _Float16(__uint_as_float(words[w] << 16U) * query_factor);
                    converted[2 * w + 1] =
                        _Float16(__uint_as_float(words[w] & 0xffff0000U) * query_factor);
                }
                *reinterpret_cast<F16x8*>(&sh.q[query_token * kKPitch + 32 * query_part + 8 * p]) =
                    converted;
            }
            uint4* record_target             = reinterpret_cast<uint4*>(&sh.record);
            record_target[tid]               = record_raw0;
            record_target[kWaves * 32 + tid] = record_raw1;
            if (record_tail) record_target[2 * kWaves * 32 + tid] = record_raw2;
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
            for (int e = 0; e < 8; ++e)
                acc[e] = sh.record.beta[16 * I + 8 * half + e] * float(values[I][e]);
            const int token = 16 * I + n;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const F16x8 a_operand =
                    *reinterpret_cast<const F16x8*>(&sh.k[token * kKPitch + 16 * t + 8 * half]);
                acc = wmma(a_operand, state_operand[t], acc);
            }
            z[I] = to_f16(acc);
        }
        if (chunk + 1 < chunks) { NINFER_GDN_LOAD_VALUES(chunk + 1) }
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
                    &sh.record.tiles[(kOffDiagonal + off_diagonal_tile(I, J)) * kTile + n * 16 +
                                     8 * half]);
                a_operand = -a_operand;
                residual  = wmma(a_operand, corrected[J], residual);
            }
            const F16x8 diagonal = *reinterpret_cast<const F16x8*>(
                &sh.record.tiles[(kDiagonal + I) * kTile + n * 16 + 8 * half]);
            corrected[I] = to_f16(wmma(diagonal, to_f16(residual), F32x8{}));
        }
        // O = (scale Gamma q / |q|) S^T + P V'.
        const bool full_chunk = t0 + kChunk <= width;
#pragma unroll
        for (int I = 0; I < 4; ++I) {
            const _Float16* query_row = &sh.q[(16 * I + n) * kKPitch + 8 * half];
            F32x8 acc                 = {};
#pragma unroll
            for (int t = 0; t < 8; ++t)
                acc = wmma(*reinterpret_cast<const F16x8*>(query_row + 16 * t), state_operand[t],
                           acc);
#pragma unroll
            for (int J = 0; J <= I; ++J) {
                const F16x8 a_operand = *reinterpret_cast<const F16x8*>(
                    &sh.record
                         .tiles[(kQueryKey + query_key_tile(I, J)) * kTile + n * 16 + 8 * half]);
                acc = wmma(a_operand, corrected[J], acc);
            }
            hip_bfloat16* output =
                out + (t0 + 16 * I + 8 * half) * kTokenStride + head * kDim + row;
            if (full_chunk) {
#pragma unroll
                for (int e = 0; e < 8; ++e) output[e * kTokenStride] = hip_bfloat16(acc[e]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    if (t0 + 16 * I + 8 * half + e < width)
                        output[e * kTokenStride] = hip_bfloat16(acc[e]);
            }
        }
        // S^T <- Gamma_last S^T + (K exp(b_last - b))^T V'.
        const float gamma_last = sh.record.gamma[kChunk - 1];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            F32x8 acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = state[t][e] * gamma_last;
#pragma unroll
            for (int I = 0; I < 4; ++I) {
                const F16x8 a_operand = *reinterpret_cast<const F16x8*>(
                    &sh.k_transposed[(16 * t + n) * kTPitch + 16 * I + 8 * half]);
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
            state_out[(std::size_t(head) * kDim + row) * kDim + 16 * t + 8 * half + e] =
                state[t][e];
#undef NINFER_GDN_LOAD_VALUES
}

// Single-chunk route (one masked chunk, no workspace): the prepared route's arithmetic, with the
// chunk's state-independent work done in place by one workgroup per value head.
__global__ __launch_bounds__(kWaves *
                             32) void direct_kernel(const hip_bfloat16* q, const hip_bfloat16* k,
                                                    const hip_bfloat16* v, const float* g,
                                                    const float* beta, const float* state_in,
                                                    float* state_out, hip_bfloat16* out, int width,
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
        // Normalized k in both layouts and the query scales; wave w owns tokens 8w..8w+7 and
        // lane l features 4l..4l+3. All loads are issued first; padded tokens read a clamped
        // valid row and are zeroed.
        {
            uint2 key_raw[8], query_raw[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int source = min(t0 + wave * 8 + r, width - 1);
                const std::size_t base =
                    (std::size_t(source) * kQHeads + query_head) * kDim + 4 * lane;
                key_raw[r]   = *reinterpret_cast<const uint2*>(k + base);
                query_raw[r] = *reinterpret_cast<const uint2*>(q + base);
            }
            _Float16 transposed[4][8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i      = wave * 8 + r;
                const float live = t0 + i < width ? 1.0F : 0.0F;
                const uint2 kr = key_raw[r], qr = query_raw[r];
                const float kv[4] = {
                    __uint_as_float(kr.x << 16U), __uint_as_float(kr.x & 0xffff0000U),
                    __uint_as_float(kr.y << 16U), __uint_as_float(kr.y & 0xffff0000U)};
                const float qv[4] = {
                    __uint_as_float(qr.x << 16U), __uint_as_float(qr.x & 0xffff0000U),
                    __uint_as_float(qr.y << 16U), __uint_as_float(qr.y & 0xffff0000U)};
                float key_sum = 0.0F, query_sum = 0.0F;
#pragma unroll
                for (int it = 0; it < 4; ++it) {
                    key_sum   = fmaf(kv[it], kv[it], key_sum);
                    query_sum = fmaf(qv[it], qv[it], query_sum);
                }
                key_sum                 = wave_sum(key_sum);
                query_sum               = wave_sum(query_sum);
                const float key_inverse = rsqrtf(key_sum + kEpsilon) * live;
#pragma unroll
                for (int it = 0; it < 4; ++it) {
                    const _Float16 normalized         = _Float16(kv[it] * key_inverse);
                    sh.k[i * kKPitch + 4 * lane + it] = normalized;
                    transposed[it][r]                 = normalized;
                }
                if (lane == 0) sh.query_scale[i] = rsqrtf(query_sum + kEpsilon) * scale * live;
            }
            // Feature-major rows: eight consecutive tokens of each of this lane's features.
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                F16x8 features;
#pragma unroll
                for (int r = 0; r < 8; ++r) features[r] = transposed[it][r];
                *reinterpret_cast<F16x8*>(&sh.k_transposed[(4 * lane + it) * kTPitch + wave * 8]) =
                    features;
            }
        }
        if (wave == kWaves - 1) {
            // Inclusive scan of g over the chunk, two tokens per lane; padding decays nothing.
            const int t_even = t0 + 2 * lane, t_odd = t_even + 1;
            const float g0 = t_even < width ? g[std::size_t(t_even) * kValueHeads + head] : 0.0F;
            const float g1 = t_odd < width ? g[std::size_t(t_odd) * kValueHeads + head] : 0.0F;
            float sum      = g0 + g1;
#pragma unroll
            for (int w = 1; w < 32; w <<= 1) {
                const float other = __shfl_up(sum, w, 32);
                if (lane >= w) sum += other;
            }
            const float before = sum - g0 - g1;
            sh.b[2 * lane]     = before + g0;
            sh.b[2 * lane + 1] = before + g0 + g1;
            sh.beta[2 * lane] =
                t_even < width ? beta[std::size_t(t_even) * kValueHeads + head] : 0.0F;
            sh.beta[2 * lane + 1] =
                t_odd < width ? beta[std::size_t(t_odd) * kValueHeads + head] : 0.0F;
        }
        __syncthreads();
        if (tid < kChunk) {
            const float gamma  = expf(sh.b[tid]);
            sh.gamma[tid]      = gamma;
            sh.beta_gamma[tid] = sh.beta[tid] * gamma;
            sh.to_end[tid]     = expf(sh.b[kChunk - 1] - sh.b[tid]);
        }
        // A = KK^T and P = QK^T on the ten lower tiles each.
        for (int job = wave; job < 20; job += kWaves) {
            const int I = kLowerTiles[job % 10][0], J = kLowerTiles[job % 10][1];
            const bool query_job = job >= 10;
            F32x8 acc;
            const int ia = 16 * I + n;
            const hip_bfloat16* query_row =
                q + (std::size_t(t0 + ia) * kQHeads + query_head) * kDim + 8 * half;
            acc = {};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const F16x8 b_operand = *reinterpret_cast<const F16x8*>(
                    &sh.k[(16 * J + n) * kKPitch + 16 * ks + 8 * half]);
                const F16x8 a_operand =
                    query_job
                        ? bf16x8_to_f16(query_row + 16 * ks, t0 + ia < width, 1.0F)
                        : *reinterpret_cast<const F16x8*>(&sh.k[ia * kKPitch + 16 * ks + 8 * half]);
                acc = wmma(a_operand, b_operand, acc);
            }
            const int j = 16 * J + n;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i       = 16 * I + 8 * half + e;
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
            for (int i = 0; i < 16; ++i)
                sh.diagonal[(origin + i) * kDiagonalPitch + column] = _Float16(x[i]);
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
                // Padded tokens have beta 0; read a valid clamped token instead of branching.
                const int token  = 16 * I + 8 * half + e;
                const int source = min(t0 + token, width - 1);
                acc[e] = sh.beta[token] *
                         float(v[(std::size_t(source) * kValueHeads + head) * kDim + row]);
            }
            const int token       = 16 * I + n;
            const _Float16 factor = _Float16(-sh.beta_gamma[token]);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                F16x8 a_operand =
                    *reinterpret_cast<const F16x8*>(&sh.k[token * kKPitch + 16 * t + 8 * half]);
                a_operand = a_operand * factor;
                acc       = wmma(a_operand, state_operand[t], acc);
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
                residual  = wmma(a_operand, corrected[J], residual);
            }
            const F16x8 diagonal = *reinterpret_cast<const F16x8*>(
                &sh.diagonal[(16 * I + n) * kDiagonalPitch + 8 * half]);
            corrected[I] = to_f16(wmma(diagonal, to_f16(residual), F32x8{}));
        }
        // O = (scale Gamma q / |q|) S^T + P V'.
#pragma unroll
        for (int I = 0; I < 4; ++I) {
            const int token    = 16 * I + n;
            const bool valid   = t0 + token < width;
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
                const F16x8 a_operand =
                    *reinterpret_cast<const F16x8*>(&sh.p[token * kTPitch + 16 * J + 8 * half]);
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
            state_out[(std::size_t(head) * kDim + row) * kDim + 16 * t + 8 * half + e] =
                state[t][e];
}

} // namespace

static_assert(sizeof(ChunkRecord) == kPreparedRecordBytes);
static_assert(kChunkedMaximumWidth == 0x7fffffff / (kValueHeads * kDim));

hipError_t prefill_chunked(const hip_bfloat16* query, const hip_bfloat16* key,
                           const hip_bfloat16* value, const float* g, const float* beta,
                           const float* state_in, float* state_out, hip_bfloat16* output,
                           std::int32_t width, float scale, void* workspace,
                           hipStream_t stream) noexcept {
    if (query == nullptr || key == nullptr || value == nullptr || g == nullptr || beta == nullptr ||
        state_in == nullptr || state_out == nullptr || output == nullptr ||
        width < kChunkedMinimumWidth || width > kChunkedMaximumWidth || !std::isfinite(scale) ||
        stream == nullptr || reinterpret_cast<std::uintptr_t>(query) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(key) % 16U != 0U) {
        return hipErrorInvalidValue;
    }
    if (width < kPreparedMinimumWidth) {
        hipLaunchKernelGGL(direct_kernel, dim3(kValueHeads), dim3(kWaves * 32), 0, stream, query,
                           key, value, g, beta, state_in, state_out, output, width, scale);
        return hipGetLastError();
    }
    if (workspace == nullptr || reinterpret_cast<std::uintptr_t>(workspace) % 16U != 0U) {
        return hipErrorInvalidValue;
    }
    const int chunks = (width + kChunk - 1) / kChunk;
    auto* keys       = static_cast<_Float16*>(workspace);
    auto* query_scales =
        reinterpret_cast<float*>(keys + std::size_t(chunks) * kQHeads * kChunk * kDim);
    auto* records =
        reinterpret_cast<ChunkRecord*>(query_scales + std::size_t(chunks) * kQHeads * kChunk);
    hipLaunchKernelGGL(prepare_chunks, dim3(chunks, kQHeads), dim3(kWaves * 32), 0, stream, query,
                       key, g, beta, keys, query_scales, records, width, scale);
    const hipError_t prepare_status = hipGetLastError();
    if (prepare_status != hipSuccess) return prepare_status;
    hipLaunchKernelGGL(prepared_kernel, dim3(kValueHeads), dim3(kWaves * 32), 0, stream, query,
                       value, state_in, state_out, output, width, keys, query_scales, records);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::gdn_recurrence
