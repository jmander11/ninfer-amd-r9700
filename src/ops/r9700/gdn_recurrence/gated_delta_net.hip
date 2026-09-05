// Native wave32 implementation of the Qwen3.8 Gated DeltaNet recurrence Tensor boundary.
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/types.h"

#include "core/device.h"
#include "ops/r9700/gdn_recurrence/gdn_prefill_scale_sidecar.h"

#include <hip/hip_bfloat16.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr int kDim = 128;
constexpr int kQHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kGroup = kValueHeads / kQHeads;
constexpr int kThreadsPerRow = 8;
constexpr int kBlock = kDim * kThreadsPerRow;
constexpr int kOrdinaryRowTiles = 4;
constexpr int kOrdinaryRowsPerBlock = kDim / kOrdinaryRowTiles;
constexpr int kOrdinaryBlock = kOrdinaryRowsPerBlock * kThreadsPerRow;
constexpr int kKeysPerThread = kDim / kThreadsPerRow;
constexpr int kMaximumVerifyWidth = 16;
constexpr int kMaximumPrefillWidth = 262144;
constexpr int kMaximumBatch = static_cast<int>(kMaximumConcurrency);
constexpr float kEpsilon = 1.0e-6F;

struct Geometry {
    int width;
    int batch;
};

__device__ __forceinline__ float group_sum(float value) {
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

__device__ __forceinline__ float wave_sum(float value) {
    value += __shfl_xor(value, 16, 32);
    value += __shfl_xor(value, 8, 32);
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

template <bool Normalize>
__device__ __forceinline__ void stage_qk(const hip_bfloat16* q, const hip_bfloat16* k,
                                         std::size_t base, float* staged_q, float* staged_k,
                                         float* reduction) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < kDim) {
        const float qv = static_cast<float>(q[base + tid]);
        const float kv = static_cast<float>(k[base + tid]);
        staged_q[tid] = qv;
        staged_k[tid] = kv;
        reduction[tid] = qv * qv;
    }
    __syncthreads();
    if constexpr (Normalize) {
        for (int stride = kDim / 2; stride != 0; stride >>= 1) {
            if (tid < stride) { reduction[tid] += reduction[tid + stride]; }
            __syncthreads();
        }
        const float q_scale = rsqrtf(reduction[0] + kEpsilon);
        __syncthreads();
        if (tid < kDim) { reduction[tid] = staged_k[tid] * staged_k[tid]; }
        __syncthreads();
        for (int stride = kDim / 2; stride != 0; stride >>= 1) {
            if (tid < stride) { reduction[tid] += reduction[tid + stride]; }
            __syncthreads();
        }
        const float k_scale = rsqrtf(reduction[0] + kEpsilon);
        if (tid < kDim) {
            staged_q[tid] *= q_scale;
            staged_k[tid] *= k_scale;
        }
        __syncthreads();
    }
}

template <bool Normalize>
__device__ __forceinline__ void stage_qk_ordinary(const hip_bfloat16* q,
                                                  const hip_bfloat16* k,
                                                  std::size_t base,
                                                  float* staged_q,
                                                  float* staged_k,
                                                  float* wave_reductions) {
    const int tid = static_cast<int>(threadIdx.x);
    if constexpr (Normalize) {
        if (tid < 32) {
            float q_sum = 0.0F;
            float k_sum = 0.0F;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = tid + item * 32;
                const float q_value = static_cast<float>(q[base + index]);
                const float k_value = static_cast<float>(k[base + index]);
                staged_q[index] = q_value;
                staged_k[index] = k_value;
                q_sum = fmaf(q_value, q_value, q_sum);
                k_sum = fmaf(k_value, k_value, k_sum);
            }
            q_sum = wave_sum(q_sum);
            k_sum = wave_sum(k_sum);
            if (tid == 0) {
                wave_reductions[0] = rsqrtf(q_sum + kEpsilon);
                wave_reductions[1] = rsqrtf(k_sum + kEpsilon);
            }
        }
        __syncthreads();
        if (tid < kDim) {
            staged_q[tid] *= wave_reductions[0];
            staged_k[tid] *= wave_reductions[1];
        }
    } else if (tid < kDim) {
        staged_q[tid] = static_cast<float>(q[base + tid]);
        staged_k[tid] = static_cast<float>(k[base + tid]);
    }
    __syncthreads();
}

__device__ __forceinline__ void load_state_row(const float* state, std::size_t base,
                                                int value_row, int sublane,
                                                float (&local)[kKeysPerThread]) {
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        local[item] = state[base + static_cast<std::size_t>(value_row) * kDim +
                            sublane + item * kThreadsPerRow];
    }
}

__device__ __forceinline__ void store_state_row(float* state, std::size_t base,
                                                 int value_row, int sublane,
                                                 const float (&local)[kKeysPerThread]) {
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        state[base + static_cast<std::size_t>(value_row) * kDim +
              sublane + item * kThreadsPerRow] = local[item];
    }
}

__device__ __forceinline__ void transition(float (&local)[kKeysPerThread],
                                            const float* staged_q, const float* staged_k,
                                            float value, float alpha, float beta,
                                            float scale, int sublane,
                                            hip_bfloat16* output) {
    float state_key = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        const int key = sublane + item * kThreadsPerRow;
        state_key = fmaf(local[item], staged_k[key], state_key);
    }
    state_key = group_sum(state_key);
    const float delta = beta * (value - alpha * state_key);
    float state_query = 0.0F;
#pragma unroll
    for (int item = 0; item < kKeysPerThread; ++item) {
        const int key = sublane + item * kThreadsPerRow;
        local[item] = fmaf(delta, staged_k[key], alpha * local[item]);
        state_query = fmaf(local[item], staged_q[key], state_query);
    }
    state_query = group_sum(state_query);
    if (sublane == 0) { *output = hip_bfloat16(state_query * scale); }
}

template <bool Normalize>
__global__ __launch_bounds__(kOrdinaryBlock, 1)
void ordinary_kernel(const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
                     const float* g, const float* beta, const float* state_in, float* state_out,
                     hip_bfloat16* output, int width, float scale) {
    const int value_head = static_cast<int>(blockIdx.x) / kOrdinaryRowTiles;
    const int row_tile = static_cast<int>(blockIdx.x) % kOrdinaryRowTiles;
    const int tid = static_cast<int>(threadIdx.x);
    const int value_row = row_tile * kOrdinaryRowsPerBlock + tid / kThreadsPerRow;
    const int sublane = tid % kThreadsPerRow;
    const int q_head = value_head / kGroup;
    __shared__ float staged_q[kDim];
    __shared__ float staged_k[kDim];
    __shared__ float wave_reductions[8];
    __shared__ float controls[2];
    float local[kKeysPerThread];
    const std::size_t state_base = static_cast<std::size_t>(value_head) * kDim * kDim;
    load_state_row(state_in, state_base, value_row, sublane, local);
    __syncthreads();
    for (int token = 0; token < width; ++token) {
        const std::size_t qk_base =
            (static_cast<std::size_t>(token) * kQHeads + q_head) * kDim;
        const std::size_t vh = static_cast<std::size_t>(token) * kValueHeads + value_head;
        if (tid == 0) {
            controls[0] = expf(g[vh]);
            controls[1] = beta[vh];
        }
        stage_qk_ordinary<Normalize>(q, k, qk_base, staged_q, staged_k,
                                     wave_reductions);
        float value = 0.0F;
        if (sublane == 0) { value = static_cast<float>(v[vh * kDim + value_row]); }
        value = __shfl(value, (tid & 31) & -kThreadsPerRow, 32);
        transition(local, staged_q, staged_k, value, controls[0], controls[1], scale, sublane,
                   output + vh * kDim + value_row);
        __syncthreads();
    }
    store_state_row(state_out, state_base, value_row, sublane, local);
}

template <bool Normalize>
__global__ __launch_bounds__(kBlock, 1)
void snapshot_kernel(const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
                     const float* g, const float* beta, float* states,
                     const std::int32_t* valid_columns,
                     const std::int32_t* initial_slots,
                     const std::int32_t* snapshot_bases, hip_bfloat16* output,
                     int width, int slots, float scale) {
    const int combined = static_cast<int>(blockIdx.x);
    const int batch = combined / kValueHeads;
    const int value_head = combined % kValueHeads;
    const int tid = static_cast<int>(threadIdx.x);
    const int value_row = tid / kThreadsPerRow;
    const int sublane = tid % kThreadsPerRow;
    const int q_head = value_head / kGroup;
    const int valid = valid_columns == nullptr ? width : valid_columns[batch];
    const int initial = initial_slots[batch];
    const int destination = snapshot_bases[batch];
    const std::size_t slot_stride = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    __shared__ float staged_q[kDim];
    __shared__ float staged_k[kDim];
    __shared__ float reduction[kDim];
    float local[kKeysPerThread];
    if (initial < 0 || initial >= slots || destination < 0 || destination + width > slots ||
        valid < 1 || valid > width) {
        for (int token = 0; token < width; ++token) {
            if (sublane == 0) {
                const std::size_t out_index =
                    ((static_cast<std::size_t>(batch) * width + token) * kValueHeads + value_head) *
                        kDim + value_row;
                output[out_index] = hip_bfloat16(NAN);
            }
        }
        return;
    }
    const std::size_t initial_base = static_cast<std::size_t>(initial) * slot_stride +
                                     static_cast<std::size_t>(value_head) * kDim * kDim;
    load_state_row(states, initial_base, value_row, sublane, local);
    __syncthreads();
    for (int token = 0; token < valid; ++token) {
        const std::size_t column = static_cast<std::size_t>(batch) * width + token;
        const std::size_t qk_base = (column * kQHeads + q_head) * kDim;
        stage_qk<Normalize>(q, k, qk_base, staged_q, staged_k, reduction);
        const std::size_t vh = column * kValueHeads + value_head;
        transition(local, staged_q, staged_k, static_cast<float>(v[vh * kDim + value_row]),
                   expf(g[vh]), beta[vh], scale, sublane,
                   output + vh * kDim + value_row);
        const std::size_t snapshot_base =
            static_cast<std::size_t>(destination + token) * slot_stride +
            static_cast<std::size_t>(value_head) * kDim * kDim;
        store_state_row(states, snapshot_base, value_row, sublane, local);
        __syncthreads();
    }
    for (int token = valid; token < width; ++token) {
        if (sublane == 0) {
            const std::size_t vh =
                (static_cast<std::size_t>(batch) * width + token) * kValueHeads + value_head;
            output[vh * kDim + value_row] = hip_bfloat16(0.0F);
        }
    }
}

template <bool Tree>
__global__ __launch_bounds__(kBlock, 1)
void record_kernel(const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
                   const float* g, const float* beta, const float* states,
                   const std::int32_t* valid_columns, const std::int32_t* initial_slots,
                   const std::int32_t* parent_index, float* tree_states,
                   hip_bfloat16* key_record, hip_bfloat16* value_record, float* gate_record,
                   hip_bfloat16* output, int width, int slots, float scale) {
    const int combined = static_cast<int>(blockIdx.x);
    const int batch = combined / kValueHeads;
    const int value_head = combined % kValueHeads;
    const int tid = static_cast<int>(threadIdx.x);
    const int value_row = tid / kThreadsPerRow;
    const int sublane = tid % kThreadsPerRow;
    const int q_head = value_head / kGroup;
    const int valid = valid_columns == nullptr ? width : valid_columns[batch];
    const int initial = initial_slots[batch];
    const std::size_t state_head = static_cast<std::size_t>(value_head) * kDim * kDim;
    const std::size_t slot_stride = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::size_t tree_token_stride = slot_stride;
    const std::size_t tree_batch_stride = static_cast<std::size_t>(width) * slot_stride;
    __shared__ float staged_q[kDim];
    __shared__ float staged_k[kDim];
    __shared__ float reduction[kDim];
    float local[kKeysPerThread];
    if (initial < 0 || initial >= slots || valid < 1 || valid > width) { return; }
    load_state_row(states, static_cast<std::size_t>(initial) * slot_stride + state_head,
                   value_row, sublane, local);
    __syncthreads();
    for (int token = 0; token < valid; ++token) {
        const std::size_t column = static_cast<std::size_t>(batch) * width + token;
        if constexpr (Tree) {
            const int parent = parent_index[column];
            if ((token == 0 && parent != -1) || (token > 0 && (parent < 0 || parent >= token))) {
                if (sublane == 0) {
                    const std::size_t vh = column * kValueHeads + value_head;
                    output[vh * kDim + value_row] = hip_bfloat16(NAN);
                }
                return;
            }
            if (parent < 0) {
                load_state_row(states, static_cast<std::size_t>(initial) * slot_stride + state_head,
                               value_row, sublane, local);
            } else {
                const std::size_t parent_base = static_cast<std::size_t>(batch) * tree_batch_stride +
                    static_cast<std::size_t>(parent) * tree_token_stride + state_head;
                load_state_row(tree_states, parent_base, value_row, sublane, local);
            }
        }
        const std::size_t qk_base = (column * kQHeads + q_head) * kDim;
        stage_qk<true>(q, k, qk_base, staged_q, staged_k, reduction);
        const std::size_t vh = column * kValueHeads + value_head;
        if (tid < kDim) {
            value_record[vh * kDim + tid] = v[vh * kDim + tid];
            if (value_head % kGroup == 0) {
                key_record[qk_base + tid] = k[qk_base + tid];
            }
        }
        if (tid == 0) {
            gate_record[vh * 2] = g[vh];
            gate_record[vh * 2 + 1] = beta[vh];
        }
        transition(local, staged_q, staged_k, static_cast<float>(v[vh * kDim + value_row]),
                   expf(g[vh]), beta[vh], scale, sublane,
                   output + vh * kDim + value_row);
        if constexpr (Tree) {
            const std::size_t child_base = static_cast<std::size_t>(batch) * tree_batch_stride +
                static_cast<std::size_t>(token) * tree_token_stride + state_head;
            store_state_row(tree_states, child_base, value_row, sublane, local);
        }
        __syncthreads();
    }
    for (int token = valid; token < width; ++token) {
        if (sublane == 0) {
            const std::size_t vh =
                (static_cast<std::size_t>(batch) * width + token) * kValueHeads + value_head;
            output[vh * kDim + value_row] = hip_bfloat16(0.0F);
        }
    }
}

void require_tensor(const Tensor& tensor, DType dtype, std::initializer_list<std::int32_t> shape,
                    const char* label) {
    Tensor expected(nullptr, dtype, shape);
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.bytes() != expected.bytes()) {
        throw std::invalid_argument(std::string("gated_delta_net: invalid ") + label);
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected.ne[dim]) {
            throw std::invalid_argument(std::string("gated_delta_net: invalid ") + label);
        }
    }
}

bool empty(const Tensor& tensor) { return tensor.data == nullptr; }

struct Range { std::uintptr_t begin; std::uintptr_t end; };

Range range(const Tensor& tensor) {
    const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto bytes = tensor.bytes();
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("gated_delta_net: tensor range overflow");
    }
    return {begin, begin + bytes};
}

bool overlaps(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }

void disjoint(const std::vector<const Tensor*>& tensors) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i] == nullptr || empty(*tensors[i])) { continue; }
        for (std::size_t j = i + 1; j < tensors.size(); ++j) {
            if (tensors[j] != nullptr && !empty(*tensors[j]) &&
                overlaps(range(*tensors[i]), range(*tensors[j]))) {
                throw std::invalid_argument("gated_delta_net: tensors must not overlap");
            }
        }
    }
}

void require_scale(float scale) {
    const float expected = 1.0F / std::sqrt(128.0F);
    if (!std::isfinite(scale) || std::abs(scale - expected) > 1.0e-6F) {
        throw std::invalid_argument("gated_delta_net: scale must be 1/sqrt(128)");
    }
}

Geometry validate_inputs(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, float scale, const Tensor& out, int batch,
                         int maximum_width) {
    const int width = q.ne[2];
    if (width < 1 || width > maximum_width || batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument("gated_delta_net: unsupported B/W domain");
    }
    require_tensor(q, DType::BF16, {kDim, kQHeads, width, batch}, "q");
    require_tensor(k, DType::BF16, {kDim, kQHeads, width, batch}, "k");
    require_tensor(v, DType::BF16, {kDim, kValueHeads, width, batch}, "v");
    require_tensor(g, DType::FP32, {kValueHeads, width, batch}, "g");
    require_tensor(beta, DType::FP32, {kValueHeads, width, batch}, "beta");
    require_tensor(out, DType::BF16, {kDim, kValueHeads, width, batch}, "out");
    require_scale(scale);
    return {width, batch};
}

void require_selector(const Tensor& tensor, int batch, const char* label, bool allow_empty) {
    if (allow_empty && empty(tensor)) { return; }
    require_tensor(tensor, DType::I32, {batch}, label);
}

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                      std::int32_t value_heads,
                                                      bool normalize_qk, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (qk_heads != kQHeads || value_heads != kValueHeads || min_tokens < 1 ||
        max_tokens < min_tokens || max_tokens > kMaximumPrefillWidth) {
        throw std::invalid_argument("gated_delta_net workspace: unsupported Qwen3.8 profile");
    }
    constexpr int kSidecarWidth = 2048;
    if (normalize_qk && min_tokens <= kSidecarWidth && max_tokens >= kSidecarWidth) {
        return r9700::gdn_recurrence::prefill_scale_sidecar_workspace_bytes(kSidecarWidth);
    }
    return 0;
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& state, Tensor& out, hipStream_t stream) {
    gated_delta_net(q, k, v, g, beta, scale, normalize_qk, ws, state, state, out, stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& state_in, Tensor& state_out, Tensor& out, hipStream_t stream) {
    const Geometry geometry =
        validate_inputs(q, k, v, g, beta, scale, out, 1, kMaximumPrefillWidth);
    require_tensor(state_in, DType::FP32, {kDim, kDim, kValueHeads}, "state_in");
    require_tensor(state_out, DType::FP32, {kDim, kDim, kValueHeads}, "state_out");
    const bool same_state = state_in.data == state_out.data;
    disjoint({&q, &k, &v, &g, &beta, &out, &state_in});
    if (!same_state) {
        disjoint({&q, &k, &v, &g, &beta, &out, &state_out});
        if (overlaps(range(state_in), range(state_out))) {
            throw std::invalid_argument("gated_delta_net: states partially overlap");
        }
    }
    if (stream == nullptr) { throw std::invalid_argument("gated_delta_net: null stream"); }
    if (normalize_qk && geometry.width == 2048) {
        auto scope = ws.scope();
        Tensor sidecar = ws.alloc(DType::FP32, {2 * kQHeads, geometry.width});
        HIP_CHECK(r9700::gdn_recurrence::prefill_scale_sidecar_prepare(
            static_cast<const hip_bfloat16*>(q.data),
            static_cast<const hip_bfloat16*>(k.data), static_cast<float*>(sidecar.data),
            geometry.width, stream));
        HIP_CHECK(r9700::gdn_recurrence::prefill_scale_sidecar_recurrence(
            static_cast<const hip_bfloat16*>(q.data),
            static_cast<const hip_bfloat16*>(k.data),
            static_cast<const hip_bfloat16*>(v.data), static_cast<const float*>(g.data),
            static_cast<const float*>(beta.data), static_cast<const float*>(state_in.data),
            static_cast<float*>(state_out.data), static_cast<hip_bfloat16*>(out.data),
            static_cast<const float*>(sidecar.data), geometry.width, scale, stream));
        return;
    }
#define NINFER_LAUNCH_ORDINARY(NORM) \
    hipLaunchKernelGGL((ordinary_kernel<NORM>), dim3(kValueHeads * kOrdinaryRowTiles), \
                       dim3(kOrdinaryBlock), 0, stream, \
                       static_cast<const hip_bfloat16*>(q.data), \
                       static_cast<const hip_bfloat16*>(k.data), \
                       static_cast<const hip_bfloat16*>(v.data), \
                       static_cast<const float*>(g.data), static_cast<const float*>(beta.data), \
                       static_cast<const float*>(state_in.data), \
                       static_cast<float*>(state_out.data), \
                       static_cast<hip_bfloat16*>(out.data), geometry.width, scale)
    if (normalize_qk) { NINFER_LAUNCH_ORDINARY(true); }
    else { NINFER_LAUNCH_ORDINARY(false); }
#undef NINFER_LAUNCH_ORDINARY
    HIP_CHECK(hipGetLastError());
}

void gated_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, bool normalize_qk,
                              Tensor& states, const Tensor& valid_columns,
                              const Tensor& initial_slots, const Tensor& snapshot_bases,
                              Tensor& out, hipStream_t stream) {
    const Geometry geometry =
        validate_inputs(q, k, v, g, beta, scale, out, q.ne[3], kMaximumVerifyWidth);
    if (states.ne[3] < 1) { throw std::invalid_argument("gated_delta_net: empty state pool"); }
    require_tensor(states, DType::FP32, {kDim, kDim, kValueHeads, states.ne[3]}, "states");
    require_selector(valid_columns, geometry.batch, "valid_columns", true);
    require_selector(initial_slots, geometry.batch, "initial_slots", false);
    require_selector(snapshot_bases, geometry.batch, "snapshot_bases", false);
    // Both selector vectors are read-only. The ordinary in-place verify path deliberately uses
    // one device slot vector for both the initial and destination base when width is one, so they
    // may alias each other while remaining disjoint from every tensor read or written by the Op.
    disjoint({&q, &k, &v, &g, &beta, &out, &states, &valid_columns, &initial_slots});
    disjoint({&q, &k, &v, &g, &beta, &out, &states, &valid_columns, &snapshot_bases});
    if (stream == nullptr) { throw std::invalid_argument("gated_delta_net: null stream"); }
#define NINFER_LAUNCH_SNAPSHOT(NORM) \
    hipLaunchKernelGGL((snapshot_kernel<NORM>), dim3(geometry.batch * kValueHeads), dim3(kBlock), \
                       0, stream, static_cast<const hip_bfloat16*>(q.data), \
                       static_cast<const hip_bfloat16*>(k.data), static_cast<const hip_bfloat16*>(v.data), \
                       static_cast<const float*>(g.data), static_cast<const float*>(beta.data), \
                       static_cast<float*>(states.data), \
                       empty(valid_columns) ? nullptr : static_cast<const std::int32_t*>(valid_columns.data), \
                       static_cast<const std::int32_t*>(initial_slots.data), \
                       static_cast<const std::int32_t*>(snapshot_bases.data), \
                       static_cast<hip_bfloat16*>(out.data), geometry.width, states.ne[3], scale)
    if (normalize_qk) { NINFER_LAUNCH_SNAPSHOT(true); }
    else { NINFER_LAUNCH_SNAPSHOT(false); }
#undef NINFER_LAUNCH_SNAPSHOT
    HIP_CHECK(hipGetLastError());
}

std::size_t gated_delta_net_replay_record_workspace_capacity_bytes(std::int32_t value_heads,
                                                                   std::int32_t batch,
                                                                   std::int32_t width) {
    if (value_heads != kValueHeads || batch < 1 || batch > kMaximumBatch || width < 2 ||
        width > kMaximumVerifyWidth) {
        throw std::invalid_argument("gated_delta_net replay workspace: unsupported profile");
    }
    constexpr std::size_t state_bytes = kValueHeads * kDim * kDim * sizeof(float);
    if (static_cast<std::size_t>(batch) >
        std::numeric_limits<std::size_t>::max() / (static_cast<std::size_t>(width) * state_bytes)) {
        throw std::overflow_error("gated_delta_net replay workspace overflow");
    }
    return static_cast<std::size_t>(batch) * width * state_bytes;
}

void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& states, const Tensor& valid_columns,
                                   const Tensor& initial_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   hipStream_t stream, const Tensor* parent_index,
                                   WorkspaceArena* workspace) {
    const Geometry geometry =
        validate_inputs(q, k, v, g, beta, scale, out, q.ne[3], kMaximumVerifyWidth);
    if (geometry.width < 2 || states.ne[3] < 1) {
        throw std::invalid_argument("gated_delta_net replay: unsupported width/state pool");
    }
    require_tensor(states, DType::FP32, {kDim, kDim, kValueHeads, states.ne[3]}, "states");
    require_selector(valid_columns, geometry.batch, "valid_columns", true);
    require_selector(initial_slots, geometry.batch, "initial_slots", false);
    require_tensor(key_record, DType::BF16, {kDim, kQHeads, geometry.width, geometry.batch},
                   "key_record");
    require_tensor(value_record, DType::BF16,
                   {kDim, kValueHeads, geometry.width, geometry.batch}, "value_record");
    require_tensor(gate_record, DType::FP32,
                   {2, kValueHeads, geometry.width, geometry.batch}, "gate_record");
    const bool tree = parent_index != nullptr && !empty(*parent_index);
    if (tree) {
        require_tensor(*parent_index, DType::I32, {geometry.width, geometry.batch}, "parent_index");
        if (workspace == nullptr) {
            throw std::invalid_argument("gated_delta_net replay: tree mode requires workspace");
        }
    }
    std::vector<const Tensor*> tensors{&q, &k, &v, &g, &beta, &states, &valid_columns,
                                       &initial_slots, &key_record, &value_record, &gate_record, &out};
    if (tree) { tensors.push_back(parent_index); }
    disjoint(tensors);
    if (stream == nullptr) { throw std::invalid_argument("gated_delta_net: null stream"); }
    if (tree) {
        auto scope = workspace->scope();
        const std::size_t bytes = gated_delta_net_replay_record_workspace_capacity_bytes(
            kValueHeads, geometry.batch, geometry.width);
        float* scratch = static_cast<float*>(workspace->alloc_bytes(bytes).data);
        hipLaunchKernelGGL((record_kernel<true>), dim3(geometry.batch * kValueHeads), dim3(kBlock),
                           0, stream, static_cast<const hip_bfloat16*>(q.data),
                           static_cast<const hip_bfloat16*>(k.data),
                           static_cast<const hip_bfloat16*>(v.data), static_cast<const float*>(g.data),
                           static_cast<const float*>(beta.data), static_cast<const float*>(states.data),
                           empty(valid_columns) ? nullptr : static_cast<const std::int32_t*>(valid_columns.data),
                           static_cast<const std::int32_t*>(initial_slots.data),
                           static_cast<const std::int32_t*>(parent_index->data), scratch,
                           static_cast<hip_bfloat16*>(key_record.data),
                           static_cast<hip_bfloat16*>(value_record.data),
                           static_cast<float*>(gate_record.data), static_cast<hip_bfloat16*>(out.data),
                           geometry.width, states.ne[3], scale);
    } else {
        hipLaunchKernelGGL((record_kernel<false>), dim3(geometry.batch * kValueHeads), dim3(kBlock),
                           0, stream, static_cast<const hip_bfloat16*>(q.data),
                           static_cast<const hip_bfloat16*>(k.data),
                           static_cast<const hip_bfloat16*>(v.data), static_cast<const float*>(g.data),
                           static_cast<const float*>(beta.data), static_cast<const float*>(states.data),
                           empty(valid_columns) ? nullptr : static_cast<const std::int32_t*>(valid_columns.data),
                           static_cast<const std::int32_t*>(initial_slots.data), nullptr, nullptr,
                           static_cast<hip_bfloat16*>(key_record.data),
                           static_cast<hip_bfloat16*>(value_record.data),
                           static_cast<float*>(gate_record.data), static_cast<hip_bfloat16*>(out.data),
                           geometry.width, states.ne[3], scale);
    }
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops
