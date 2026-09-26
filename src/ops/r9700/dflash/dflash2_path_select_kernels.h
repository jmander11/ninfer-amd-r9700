#pragma once

// Implements: include/ninfer/ops/dflash2_path_select.h

#include "ops/common/math.h"
#include "ninfer/ops/sampling.h"

#include <hip/hip_bfloat16.h>
#include <climits>
#include <cmath>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kDflash2PathSelectBlock     = 256;
inline constexpr int kDflash2PathSelectK         = 16;
inline constexpr int kDflash2PathSelectRank      = 256;
inline constexpr int kDflash2PathSelectSuccStride  = kDflash2PathSelectRank + 2;
inline constexpr int kDflash2PathSelectRngPurposeDevice = 16;
inline constexpr int kDflash2PathSelectTopkSplits = 32;

struct Dflash2CodebookDevice {
    const hip_bfloat16* bf16 = nullptr;
};

__device__ __forceinline__ hip_bfloat16 dflash2_codebook_load(Dflash2CodebookDevice book, int token,
                                                              int rank) {
    if (token < 0) { token = 0; }
    return book.bf16[static_cast<std::int64_t>(token) * kDflash2PathSelectRank + rank];
}

__device__ __forceinline__ unsigned long long dflash2_path_select_splitmix64(
    unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float dflash2_path_select_uniform(unsigned long long seed, int position,
                                                              int purpose) {
    unsigned long long key = seed;
    key                    = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21));
    const unsigned int bits = static_cast<unsigned int>(key >> 40);
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ bool dflash2_logit_better(float value, int index, float best_value,
                                                     int best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ std::int64_t dflash2_column_index(int batch, int tokens, int t, int b) {
    return static_cast<std::int64_t>(b) * tokens + t;
}

// Register-resident sorted top-K list: constant indices only, so it never spills to scratch.
struct Dflash2TopkList {
    float val[kDflash2PathSelectK];
    int idx[kDflash2PathSelectK];
};

__device__ __forceinline__ void dflash2_topk_clear(Dflash2TopkList& list) {
#pragma unroll
    for (int j = 0; j < kDflash2PathSelectK; ++j) {
        list.val[j] = -INFINITY;
        list.idx[j] = INT_MAX;
    }
}

// Admits finite values only, in strict (value desc, lower index) order.
__device__ __forceinline__ void dflash2_topk_push(Dflash2TopkList& list, float value, int index) {
    constexpr int last = kDflash2PathSelectK - 1;
    if (index < 0 || isnan(value) || isinf(value) ||
        !dflash2_logit_better(value, index, list.val[last], list.idx[last])) {
        return;
    }
    list.val[last] = value;
    list.idx[last] = index;
#pragma unroll
    for (int j = last; j > 0; --j) {
        if (dflash2_logit_better(list.val[j], list.idx[j], list.val[j - 1], list.idx[j - 1])) {
            const float v = list.val[j];
            const int i   = list.idx[j];
            list.val[j]     = list.val[j - 1];
            list.idx[j]     = list.idx[j - 1];
            list.val[j - 1] = v;
            list.idx[j - 1] = i;
        }
    }
}

// Merges the 32 per-lane sorted lists of one wave into the wave's top K, returned to every lane.
// Each round selects the best remaining head across the wave; its (unique) owner pops it.
__device__ __forceinline__ void dflash2_topk_wave_merge(Dflash2TopkList& list, float* out_val,
                                                        int* out_idx) {
#pragma unroll 1
    for (int j = 0; j < kDflash2PathSelectK; ++j) {
        float best_val = list.val[0];
        int best_idx   = list.idx[0];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_val = __shfl_xor(best_val, offset, 32);
            const int other_idx   = __shfl_xor(best_idx, offset, 32);
            if (dflash2_logit_better(other_val, other_idx, best_val, best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        out_val[j] = best_val;
        out_idx[j] = best_idx;
        if (best_idx != INT_MAX && list.idx[0] == best_idx) {
#pragma unroll
            for (int k = 0; k < kDflash2PathSelectK - 1; ++k) {
                list.val[k] = list.val[k + 1];
                list.idx[k] = list.idx[k + 1];
            }
            list.val[kDflash2PathSelectK - 1] = -INFINITY;
            list.idx[kDflash2PathSelectK - 1] = INT_MAX;
        }
    }
}

// Block top K of per-thread lists: each wave merges its lanes, then wave zero merges the
// kDflash2PathSelectBlock / 32 wave results. Every lane of wave zero returns the block result.
__device__ __forceinline__ void dflash2_topk_block_merge(Dflash2TopkList& list, float* out_val,
                                                         int* out_idx) {
    constexpr int waves = kDflash2PathSelectBlock / 32;
    static_assert(waves <= 32);
    __shared__ float wave_val[waves][kDflash2PathSelectK];
    __shared__ int wave_idx[waves][kDflash2PathSelectK];
    const int wave = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    float merged_val[kDflash2PathSelectK];
    int merged_idx[kDflash2PathSelectK];
    dflash2_topk_wave_merge(list, merged_val, merged_idx);
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            wave_val[wave][j] = merged_val[j];
            wave_idx[wave][j] = merged_idx[j];
        }
    }
    __syncthreads();
    if (wave != 0) { return; }
    dflash2_topk_clear(list);
    if (lane < waves) {
#pragma unroll
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            list.val[j] = wave_val[lane][j];
            list.idx[j] = wave_idx[lane][j];
        }
    }
    dflash2_topk_wave_merge(list, out_val, out_idx);
}

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_column_topk_split_kernel(const hip_bfloat16* logits, float* split_val,
                                          int* split_idx, std::int32_t vocab, std::int32_t tokens,
                                          std::int32_t batch) {
    const int split = static_cast<int>(blockIdx.x);
    const int t     = static_cast<int>(blockIdx.y);
    const int b     = static_cast<int>(blockIdx.z);
    const int tid   = static_cast<int>(threadIdx.x);
    if (split >= kDflash2PathSelectTopkSplits || t >= tokens || b >= batch) { return; }

    const int chunk = (vocab + kDflash2PathSelectTopkSplits - 1) / kDflash2PathSelectTopkSplits;
    const int v0    = split * chunk;
    const int v1    = v0 + chunk < vocab ? v0 + chunk : vocab;

    Dflash2TopkList list;
    dflash2_topk_clear(list);
    if (v0 < v1) {
        const std::int64_t logit_col =
            dflash2_column_index(batch, tokens, t, b) * static_cast<std::int64_t>(vocab);
        for (int v = v0 + tid; v < v1; v += kDflash2PathSelectBlock) {
            dflash2_topk_push(list, static_cast<float>(logits[logit_col + v]), v);
        }
    }
    float merged_val[kDflash2PathSelectK];
    int merged_idx[kDflash2PathSelectK];
    dflash2_topk_block_merge(list, merged_val, merged_idx);
    if (tid == 0) {
        const std::int64_t out =
            (dflash2_column_index(batch, tokens, t, b) * kDflash2PathSelectTopkSplits + split) *
            kDflash2PathSelectK;
#pragma unroll
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            split_val[out + j] = merged_val[j];
            split_idx[out + j] = merged_idx[j];
        }
    }
}

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_column_topk_merge_kernel(const float* split_val, const int* split_idx,
                                          float* cand_val, int* cand_idx,
                                          const std::int32_t* logit_token_ids, std::int32_t vocab,
                                          std::int32_t tokens, std::int32_t batch) {
    const int t   = static_cast<int>(blockIdx.x);
    const int b   = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (t >= tokens || b >= batch) { return; }

    const std::int64_t column = dflash2_column_index(batch, tokens, t, b);
    const std::int64_t src    = column * kDflash2PathSelectTopkSplits * kDflash2PathSelectK;
    Dflash2TopkList list;
    dflash2_topk_clear(list);
    for (int p = tid; p < kDflash2PathSelectTopkSplits * kDflash2PathSelectK;
         p += kDflash2PathSelectBlock) {
        if (split_idx[src + p] == INT_MAX) { continue; }
        dflash2_topk_push(list, split_val[src + p], split_idx[src + p]);
    }
    float out_val[kDflash2PathSelectK];
    int out_idx[kDflash2PathSelectK];
    dflash2_topk_block_merge(list, out_val, out_idx);
    if (tid != 0) { return; }
    for (int c = 0; c < kDflash2PathSelectK; ++c) {
        if (out_idx[c] != INT_MAX) { continue; }
        out_idx[c] = c < vocab ? c : 0;
        out_val[c] = -INFINITY;
    }
    if (logit_token_ids != nullptr) {
        for (int c = 0; c < kDflash2PathSelectK; ++c) {
            const int row = out_idx[c];
            out_idx[c]    = (row >= 0 && row < vocab) ? logit_token_ids[row] : 0;
        }
    }
    const std::int64_t dst = column * kDflash2PathSelectK;
    for (int c = 0; c < kDflash2PathSelectK; ++c) {
        cand_val[dst + c] = out_val[c];
        cand_idx[dst + c] = out_idx[c];
    }
}

__device__ __forceinline__ float dflash2_markov_score_serial(const hip_bfloat16* hidden_proj,
                                                             const hip_bfloat16* pred_code,
                                                             const hip_bfloat16* succ_code,
                                                             std::int64_t h_col, int prev, int cand,
                                                             float unary) {
    float acc               = unary;
    const std::int64_t pred = static_cast<std::int64_t>(prev) * kDflash2PathSelectRank;
    const std::int64_t succ = static_cast<std::int64_t>(cand) * kDflash2PathSelectRank;
    for (int r = 0; r < kDflash2PathSelectRank; ++r) {
        const float hr = static_cast<float>(hidden_proj[h_col + r]);
        const float pr = static_cast<float>(pred_code[pred + r]);
        const float sr = static_cast<float>(succ_code[succ + r]);
        acc += (pr * hr) * sr;
    }
    return acc;
}

__device__ __forceinline__ float dflash2_markov_score_staged(const float* hidden,
                                                            const hip_bfloat16* pred,
                                                            const hip_bfloat16* succ,
                                                            float unary) {
    float acc = unary;
    for (int r = 0; r < kDflash2PathSelectRank; ++r) {
        const float pr = static_cast<float>(pred[r]);
        const float sr = static_cast<float>(succ[r]);
        acc += (pr * hidden[r]) * sr;
    }
    return acc;
}

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_path_select_kernel(const float* cand_val, const int* cand_idx,
                                    const hip_bfloat16* hidden_proj, Dflash2CodebookDevice pred_code,
                                    Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
                                     const std::int32_t* logical_positions, std::int32_t* path,
                                     std::int32_t* selector_ids, float* selector_q,
                                    std::int32_t tokens, std::int32_t batch,
                                    const SamplingConfig* configs, unsigned long long seed_xor,
                                     std::int32_t position_offset, bool force_greedy) {
    const int b   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (b >= batch) { return; }
    const SamplingConfig cfg       = configs[b];
    // P-less temperature belongs to the target distribution; its draft proposal is a point mass.
    const float temperature        = (force_greedy || cfg.p_less != 0) ? 0.0f : cfg.temperature;
    const unsigned long long seed  = cfg.seed ^ seed_xor;

    __shared__ float scores[kDflash2PathSelectK];
    __shared__ float sm_val[kDflash2PathSelectK];
    __shared__ int sm_idx[kDflash2PathSelectK];
    __shared__ float sm_h[kDflash2PathSelectRank];
    __shared__ hip_bfloat16 sm_pred[kDflash2PathSelectRank];
    __shared__ hip_bfloat16 sm_succ[kDflash2PathSelectK * kDflash2PathSelectSuccStride];
    __shared__ int prev_id;

    if (tid == 0) { prev_id = anchors[b]; }
    __syncthreads();

    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int64_t col   = dflash2_column_index(batch, tokens, t, b);
        const std::int64_t h_col = col * kDflash2PathSelectRank;
        if (tid < kDflash2PathSelectK) {
            sm_val[tid] = cand_val[col * kDflash2PathSelectK + tid];
            sm_idx[tid] = cand_idx[col * kDflash2PathSelectK + tid];
        }
        __syncthreads();

        if (tid < kDflash2PathSelectRank) {
            sm_h[tid]    = static_cast<float>(hidden_proj[h_col + tid]);
            sm_pred[tid] = dflash2_codebook_load(pred_code, prev_id, tid);
        }
        for (int i = tid; i < kDflash2PathSelectK * kDflash2PathSelectRank;
             i += kDflash2PathSelectBlock) {
            const int c = i / kDflash2PathSelectRank;
            const int r = i - c * kDflash2PathSelectRank;
            sm_succ[c * kDflash2PathSelectSuccStride + r]  = dflash2_codebook_load(succ_code, sm_idx[c], r);
        }
        __syncthreads();

        if (tid < kDflash2PathSelectK) {
            scores[tid] = dflash2_markov_score_staged(
                sm_h, sm_pred, sm_succ + tid * kDflash2PathSelectSuccStride, sm_val[tid]);
        }
        __syncthreads();

        if (tid == 0) {
            int pick = 0;
            if (temperature > 0.0f) {
                float m = scores[0];
                for (int c = 1; c < kDflash2PathSelectK; ++c) { m = fmaxf(m, scores[c]); }
                const float inv_temp = 1.0f / temperature;
                float sum            = 0.0f;
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    scores[c] = expf((scores[c] - m) * inv_temp);
                    sum += scores[c];
                }
                const int position =
                    logical_positions[b] + position_offset + static_cast<int>(t) + 1;
                const float u = dflash2_path_select_uniform(
                    seed, position, kDflash2PathSelectRngPurposeDevice);
                const float goal = u * sum;
                float run        = 0.0f;
                pick             = kDflash2PathSelectK - 1;
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    run += scores[c];
                    if (goal < run) {
                        pick = c;
                        break;
                    }
                }
            } else {
                float best  = scores[0];
                int best_id = sm_idx[0];
                for (int c = 1; c < kDflash2PathSelectK; ++c) {
                    const float s = scores[c];
                    const int id  = sm_idx[c];
                    if (s > best || (s == best && id < best_id)) {
                        best    = s;
                        best_id = id;
                        pick    = c;
                    }
                }
            }

            const int chosen = sm_idx[pick];
            path[static_cast<std::int64_t>(t) + static_cast<std::int64_t>(b) * tokens] = chosen;
            if (selector_ids != nullptr) {
                const std::int64_t sel_base =
                    (static_cast<std::int64_t>(t) + static_cast<std::int64_t>(b) * tokens) *
                    kDflash2PathSelectK;
                float qsum = 0.0f;
                if (temperature > 0.0f) {
                    for (int c = 0; c < kDflash2PathSelectK; ++c) { qsum += scores[c]; }
                }
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    selector_ids[sel_base + c] = sm_idx[c];
                    if (selector_q != nullptr) {
                        if (temperature > 0.0f) {
                            selector_q[sel_base + c] = qsum > 0.0f ? scores[c] / qsum : 0.0f;
                        } else {
                            selector_q[sel_base + c] = c == pick ? 1.0f : 0.0f;
                        }
                    }
                }
            }
            prev_id = chosen;
        }
        __syncthreads();
    }
}

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_tree_select_kernel(const float* cand_val, const int* cand_idx,
                                    const hip_bfloat16* hidden_proj, Dflash2CodebookDevice pred_code,
                                    Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
                                    const std::int32_t* frontiers, std::int32_t* verify_ids,
                                    std::int32_t* parent_index, std::int32_t* cache_positions,
                                    std::int32_t* rope_positions, std::int32_t* ancestor_mask,
                                    std::int32_t* valid_columns, std::int32_t tokens,
                                    std::int32_t batch, std::int32_t out_width) {
    constexpr int kExpand   = 16;
    constexpr int kFrontier = 2;
    const int kOut          = out_width;
    const int b             = static_cast<int>(blockIdx.x);
    const int tid           = static_cast<int>(threadIdx.x);
    if (b >= batch || kOut < 2 || kOut > kExpand) { return; }

    __shared__ float sm_val[kDflash2PathSelectK];
    __shared__ int sm_idx[kDflash2PathSelectK];
    __shared__ int node_id[kExpand];
    __shared__ int node_parent[kExpand];
    __shared__ int node_depth[kExpand];
    __shared__ float node_score[kExpand];
    __shared__ int frontier[kFrontier];
    __shared__ int frontier_n;
    __shared__ int live;
    __shared__ float sm_h[kDflash2PathSelectRank];
    __shared__ hip_bfloat16 sm_pred[kFrontier * kDflash2PathSelectRank];
    __shared__ hip_bfloat16 sm_succ[kDflash2PathSelectK * kDflash2PathSelectSuccStride];
    __shared__ float pair_score[kFrontier * kDflash2PathSelectK];

    if (tid == 0) {
        node_id[0]     = anchors[b];
        node_parent[0] = -1;
        node_depth[0]  = 0;
        node_score[0]  = 0.0f;
        frontier[0]    = 0;
        frontier_n     = 1;
        live           = 1;
    }
    __syncthreads();

    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int64_t col   = dflash2_column_index(batch, tokens, t, b);
        const std::int64_t h_col = col * kDflash2PathSelectRank;
        if (tid < kDflash2PathSelectK) {
            sm_val[tid] = cand_val[col * kDflash2PathSelectK + tid];
            sm_idx[tid] = cand_idx[col * kDflash2PathSelectK + tid];
        }
        __syncthreads();

        if (tid < kDflash2PathSelectRank) {
            sm_h[tid] = static_cast<float>(hidden_proj[h_col + tid]);
        }
        for (int i = tid; i < kFrontier * kDflash2PathSelectRank; i += kDflash2PathSelectBlock) {
            const int f    = i / kDflash2PathSelectRank;
            const int r    = i - f * kDflash2PathSelectRank;
            const int prev = f < frontier_n ? node_id[frontier[f]] : 0;
            sm_pred[i] = dflash2_codebook_load(pred_code, prev, r);
        }
        for (int i = tid; i < kDflash2PathSelectK * kDflash2PathSelectRank;
             i += kDflash2PathSelectBlock) {
            const int c = i / kDflash2PathSelectRank;
            const int r = i - c * kDflash2PathSelectRank;
            sm_succ[c * kDflash2PathSelectSuccStride + r] = dflash2_codebook_load(succ_code, sm_idx[c], r);
        }
        __syncthreads();

        if (tid < kFrontier * kDflash2PathSelectK) {
            const int f = tid / kDflash2PathSelectK;
            const int c = tid - f * kDflash2PathSelectK;
            if (f < frontier_n) {
                pair_score[tid] = dflash2_markov_score_staged(
                    sm_h, sm_pred + f * kDflash2PathSelectRank,
                    sm_succ + c * kDflash2PathSelectSuccStride, sm_val[c]);
            } else {
                pair_score[tid] = -INFINITY;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best_score[kFrontier];
            int best_parent[kFrontier];
            int best_cand[kFrontier];
            for (int s = 0; s < kFrontier; ++s) {
                best_score[s]  = -INFINITY;
                best_parent[s] = -1;
                best_cand[s]   = INT_MAX;
            }

            const int parents = frontier_n;
            for (int f = 0; f < parents; ++f) {
                const int pcol   = frontier[f];
                const float base = node_score[pcol];
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    const int cand    = sm_idx[c];
                    const float joint = base + pair_score[f * kDflash2PathSelectK + c];
                    int slot = kFrontier;
                    for (int s = 0; s < kFrontier; ++s) {
                        if (joint > best_score[s] ||
                            (joint == best_score[s] &&
                             (cand < best_cand[s] ||
                              (cand == best_cand[s] && pcol < best_parent[s])))) {
                            slot = s;
                            break;
                        }
                    }
                    if (slot == kFrontier) { continue; }
                    for (int s = kFrontier - 1; s > slot; --s) {
                        best_score[s]  = best_score[s - 1];
                        best_parent[s] = best_parent[s - 1];
                        best_cand[s]   = best_cand[s - 1];
                    }
                    best_score[slot]  = joint;
                    best_parent[slot] = pcol;
                    best_cand[slot]   = cand;
                }
            }

            const int depth = static_cast<int>(t) + 1;
            int next_n      = 0;
            for (int s = 0; s < kFrontier; ++s) {
                if (best_parent[s] < 0) { continue; }
                const int col_i = live;
                if (col_i >= kExpand) { break; }
                node_id[col_i]     = best_cand[s];
                node_parent[col_i] = best_parent[s];
                node_depth[col_i]  = depth;
                node_score[col_i]  = best_score[s];
                frontier[next_n]   = col_i;
                ++next_n;
                ++live;
            }
            frontier_n = next_n > 0 ? next_n : frontier_n;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int e     = frontiers[b];
        const int out_n = live < kOut ? live : kOut;
        for (int i = 0; i < out_n; ++i) {
            verify_ids[b * kOut + i]      = node_id[i];
            parent_index[b * kOut + i]    = i == 0 ? -1 : node_parent[i];
            cache_positions[b * kOut + i] = e + i;
            rope_positions[b * kOut + i]  = e + node_depth[i];
            int mask                      = 0;
            int cur                       = i;
            while (cur >= 0) {
                mask |= 1 << cur;
                cur = node_parent[cur];
            }
            ancestor_mask[b * kOut + i] = mask;
        }
        const int last = out_n > 0 ? out_n - 1 : 0;
        for (int col = out_n; col < kOut; ++col) {
            verify_ids[b * kOut + col]      = verify_ids[b * kOut + last];
            parent_index[b * kOut + col]    = parent_index[b * kOut + last];
            cache_positions[b * kOut + col] = e + col;
            rope_positions[b * kOut + col]  = rope_positions[b * kOut + last];
            ancestor_mask[b * kOut + col]   = ancestor_mask[b * kOut + last];
        }
        valid_columns[b] = out_n;
    }
}

} // namespace ninfer::ops
