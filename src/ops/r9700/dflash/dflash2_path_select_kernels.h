#pragma once

// Implements: include/ninfer/ops/dflash2_path_select.h

#include "ops/common/math.h"
#include "ninfer/ops/sampling.h"

#include <hip/hip_bfloat16.h>
#include <climits>
#include <cmath>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kDflash2PathSelectBlock            = 256;
inline constexpr int kDflash2PathSelectK                = 16;
inline constexpr int kDflash2PathSelectRank             = 256;
inline constexpr int kDflash2PathSelectSuccStride       = kDflash2PathSelectRank + 2;
inline constexpr int kDflash2PathSelectRngPurposeDevice = 16;

struct Dflash2CodebookDevice {
    const hip_bfloat16* bf16 = nullptr;
};

__device__ __forceinline__ hip_bfloat16 dflash2_codebook_load(Dflash2CodebookDevice book, int token,
                                                              int rank) {
    if (token < 0) { token = 0; }
    return book.bf16[static_cast<std::int64_t>(token) * kDflash2PathSelectRank + rank];
}

__device__ __forceinline__ unsigned long long dflash2_path_select_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float dflash2_path_select_uniform(unsigned long long seed, int position,
                                                             int purpose, unsigned int hop) {
    unsigned long long key = seed;
    key                    = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (static_cast<unsigned long long>(hop) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40);
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ bool dflash2_logit_better(float value, int index, float best_value,
                                                     int best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ std::int64_t dflash2_column_index(int tokens, int t, int b) {
    return static_cast<std::int64_t>(b) * tokens + t;
}

// Column top-K in one launch: one CTA of kDflash2ColumnTopkThreads per (t,b) column. The 16th best
// of the waves' best logits is a threshold that at least K logits meet, so only logits at least as
// good as it are kept as candidates (typically a few dozen) and ranked exactly in (value desc,
// lower index) order. A candidate overflow falls back to K exact exclusion rounds. Finite values
// only; a column with fewer than K finite logits pads slot c with (c, -inf) like the selector.
inline constexpr int kDflash2ColumnTopkThreads    = 1024;
inline constexpr int kDflash2ColumnTopkWaves      = kDflash2ColumnTopkThreads / 32;
inline constexpr int kDflash2ColumnTopkCandidates = 2048;
static_assert(kDflash2ColumnTopkWaves == 32 && kDflash2ColumnTopkWaves >= kDflash2PathSelectK);

// Trivial so it can live in LDS; dflash2_logit_none() is the empty slot.
struct Dflash2Logit {
    float value;
    int index; // INT_MAX: none
};

__device__ __forceinline__ Dflash2Logit dflash2_logit_none() { return {-INFINITY, INT_MAX}; }

__device__ __forceinline__ bool dflash2_logit_better(Dflash2Logit lhs, Dflash2Logit rhs) {
    return dflash2_logit_better(lhs.value, lhs.index, rhs.value, rhs.index);
}

// Visits every finite logit (value, index) of one contiguous column.
template <class Visit>
__device__ __forceinline__ void dflash2_column_logits(const hip_bfloat16* column,
                                                      std::int32_t vocab, Visit visit) {
    const int tid = static_cast<int>(threadIdx.x);
    if (vocab % 8 == 0 && reinterpret_cast<std::uintptr_t>(column) % 16U == 0U) {
        const uint4* vectors = reinterpret_cast<const uint4*>(column);
        for (int vector = tid; vector < vocab / 8; vector += kDflash2ColumnTopkThreads) {
            const uint4 packed = vectors[vector];
            const std::uint32_t words[4]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const std::uint32_t word = words[element >> 1];
                const float value =
                    __uint_as_float((element & 1) ? word & 0xffff0000U : word << 16U);
                if (isfinite(value)) visit(Dflash2Logit{value, vector * 8 + element});
            }
        }
    } else {
        for (int index = tid; index < vocab; index += kDflash2ColumnTopkThreads) {
            const float value = static_cast<float>(column[index]);
            if (isfinite(value)) visit(Dflash2Logit{value, index});
        }
    }
}

__device__ __forceinline__ Dflash2Logit dflash2_wave_best(Dflash2Logit best) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const Dflash2Logit other{__shfl_xor(best.value, offset, 32),
                                 __shfl_xor(best.index, offset, 32)};
        if (dflash2_logit_better(other, best)) best = other;
    }
    return best;
}

// The block's best logit; every thread returns it.
__device__ __forceinline__ Dflash2Logit dflash2_block_best(Dflash2Logit best,
                                                           Dflash2Logit* wave_best) {
    const int wave = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    best = dflash2_wave_best(best);
    __syncthreads(); // wave_best may still be read from the previous use
    if (lane == 0) wave_best[wave] = best;
    __syncthreads();
    return dflash2_wave_best(wave_best[lane]);
}

__launch_bounds__(kDflash2ColumnTopkThreads) __global__
    void dflash2_column_topk_kernel(const hip_bfloat16* logits, float* cand_val, int* cand_idx,
                                    const std::int32_t* logit_token_ids, std::int32_t vocab,
                                    std::int32_t columns) {
    __shared__ Dflash2Logit wave_best[kDflash2ColumnTopkWaves];
    __shared__ Dflash2Logit candidates[kDflash2ColumnTopkCandidates];
    __shared__ Dflash2Logit selected[kDflash2PathSelectK];
    __shared__ Dflash2Logit threshold;
    __shared__ int candidate_count;
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int wave = tid / 32, lane = tid % 32;
    if (column >= columns) return;
    const hip_bfloat16* source = logits + static_cast<std::int64_t>(column) * vocab;

    Dflash2Logit best = dflash2_logit_none();
    dflash2_column_logits(source, vocab, [&](Dflash2Logit logit) {
        if (dflash2_logit_better(logit, best)) best = logit;
    });
    best = dflash2_wave_best(best);
    if (lane == 0) wave_best[wave] = best;
    if (tid < kDflash2PathSelectK) selected[tid] = dflash2_logit_none();
    if (tid == 0) {
        candidate_count = 0;
        // Kept when waves without a finite logit tie below rank K-1: then every finite logit is a
        // candidate.
        threshold = dflash2_logit_none();
    }
    __syncthreads();
    if (wave == 0) {
        // Rank of this wave's best among the 32 (unique indices make the order strict).
        const Dflash2Logit mine = wave_best[lane];
        int rank                = 0;
        for (int other = 0; other < kDflash2ColumnTopkWaves; ++other)
            rank += dflash2_logit_better(wave_best[other], mine) ? 1 : 0;
        if (rank == kDflash2PathSelectK - 1) threshold = mine;
    }
    __syncthreads();
    // At least K logits (the K best wave maxima) are at least as good as the threshold; a logit
    // worse than it has K better ones. A threshold without a logit admits every finite one.
    const Dflash2Logit bound = threshold;
    dflash2_column_logits(source, vocab, [&](Dflash2Logit logit) {
        if (dflash2_logit_better(bound, logit)) return;
        const int slot = atomicAdd(&candidate_count, 1);
        if (slot < kDflash2ColumnTopkCandidates) candidates[slot] = logit;
    });
    __syncthreads();
    const int count = candidate_count;
    if (count <= kDflash2ColumnTopkCandidates) {
        for (int slot = tid; slot < count; slot += kDflash2ColumnTopkThreads) {
            const Dflash2Logit mine = candidates[slot];
            int rank                = 0;
            for (int other = 0; other < count && rank < kDflash2PathSelectK; ++other)
                rank += dflash2_logit_better(candidates[other], mine) ? 1 : 0;
            if (rank < kDflash2PathSelectK) selected[rank] = mine;
        }
    } else {
        // Exact exclusion rounds: the best logit strictly worse than the previous pick.
        Dflash2Logit previous{INFINITY, -1};
        for (int round = 0; round < kDflash2PathSelectK; ++round) {
            Dflash2Logit next = dflash2_logit_none();
            dflash2_column_logits(source, vocab, [&](Dflash2Logit logit) {
                if (dflash2_logit_better(previous, logit) && dflash2_logit_better(logit, next))
                    next = logit;
            });
            next = dflash2_block_best(next, wave_best);
            if (tid == 0) selected[round] = next;
            previous = next;
        }
    }
    __syncthreads();
    if (tid >= kDflash2PathSelectK) return;
    Dflash2Logit out = selected[tid];
    if (out.index == INT_MAX) out = Dflash2Logit{-INFINITY, tid < vocab ? tid : 0};
    if (logit_token_ids != nullptr) {
        out.index = (out.index >= 0 && out.index < vocab) ? logit_token_ids[out.index] : 0;
    }
    const std::int64_t dst = static_cast<std::int64_t>(column) * kDflash2PathSelectK + tid;
    cand_val[dst]          = out.value;
    cand_idx[dst]          = out.index;
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

__launch_bounds__(kDflash2PathSelectBlock) __global__ void dflash2_path_select_kernel(
    const float* cand_val, const int* cand_idx, const hip_bfloat16* hidden_proj,
    Dflash2CodebookDevice pred_code, Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
    const std::int32_t* logical_positions, std::int32_t* path, std::int32_t* selector_ids,
    float* selector_q, std::int32_t tokens, std::int32_t batch, const SamplingConfig* configs,
    unsigned long long seed_xor, std::int32_t position_offset, bool force_greedy) {
    const int b   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (b >= batch) { return; }
    const SamplingConfig cfg = configs[b];
    // q written below is the law each draft is drawn from (one-hot when greedy); verification
    // accepts with min(1, p/q) under every target sampler, p-less included. P-less rows draw at
    // their own draft temperature: the shortlist softmax at the p-less target temperature itself
    // accepts less than argmax, while a lower one accepts more.
    const float temperature =
        force_greedy ? 0.0f : (cfg.p_less != 0 ? cfg.draft_temperature : cfg.temperature);
    const unsigned long long seed = cfg.seed ^ seed_xor;

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
        const std::int64_t col   = dflash2_column_index(tokens, t, b);
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
            sm_succ[c * kDflash2PathSelectSuccStride + r] =
                dflash2_codebook_load(succ_code, sm_idx[c], r);
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
                // Keyed by the round's first position and the hop: block verification's accepted
                // length depends on drafts past it, so the next round must not reuse a draft
                // uniform at the same absolute position.
                const int round_start = logical_positions[b] + position_offset + 1;
                const float u    = dflash2_path_select_uniform(seed, round_start,
                                                               kDflash2PathSelectRngPurposeDevice,
                                                               static_cast<unsigned int>(t));
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

__launch_bounds__(kDflash2PathSelectBlock) __global__ void dflash2_tree_select_kernel(
    const float* cand_val, const int* cand_idx, const hip_bfloat16* hidden_proj,
    Dflash2CodebookDevice pred_code, Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
    const std::int32_t* frontiers, std::int32_t* verify_ids, std::int32_t* parent_index,
    std::int32_t* cache_positions, std::int32_t* rope_positions, std::int32_t* ancestor_mask,
    std::int32_t* valid_columns, std::int32_t tokens, std::int32_t batch, std::int32_t out_width) {
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
        const std::int64_t col   = dflash2_column_index(tokens, t, b);
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
            sm_pred[i]     = dflash2_codebook_load(pred_code, prev, r);
        }
        for (int i = tid; i < kDflash2PathSelectK * kDflash2PathSelectRank;
             i += kDflash2PathSelectBlock) {
            const int c = i / kDflash2PathSelectRank;
            const int r = i - c * kDflash2PathSelectRank;
            sm_succ[c * kDflash2PathSelectSuccStride + r] =
                dflash2_codebook_load(succ_code, sm_idx[c], r);
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
                    int slot          = kFrontier;
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
