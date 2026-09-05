// Implements the sole gfx1201 production form of include/ninfer/ops/sampling.h.
// Large vocabularies use an exact two-stage top-20 selection: 512-token partials followed by a
// bounded final merge. The caller owns every byte of transient storage, and the counter RNG has
// no mutable device state, so graph replay is address- and result-stable.
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kBlock = 256;
constexpr int kTile = 512;
constexpr int kCandidateCap = 20;
constexpr int kMaximumRows = 16;
constexpr int kMaximumRequestRows = static_cast<int>(kMaximumConcurrency);

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1U) / alignment * alignment;
}

[[nodiscard]] constexpr int partial_count(int token_domain) noexcept {
    return (token_domain + kTile - 1) / kTile;
}

[[nodiscard]] std::size_t exact_workspace_bytes(int token_domain, int rows) noexcept {
    if (token_domain <= kTile || rows <= 0 || rows > kMaximumRows) { return 0; }
    return align_up(static_cast<std::size_t>(partial_count(token_domain)) * rows *
                        kCandidateCap * sizeof(unsigned long long),
                    256U);
}

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ float uniform(unsigned long long seed, int position, int purpose) {
    unsigned long long key = splitmix64(
        seed ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
                0xD1B54A32D192ED03ULL));
    key = splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (0ULL * 0x2545F4914F6CDD1DULL));
    return static_cast<float>(static_cast<unsigned int>(key >> 40)) *
           (1.0F / 16777216.0F);
}

__device__ __forceinline__ unsigned int ordered_float(float value) {
    if (value == 0.0F) { value = 0.0F; }
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0U ? ~bits : (bits ^ 0x80000000U);
}

__device__ __forceinline__ unsigned long long sort_key(float value, int token) {
    return (static_cast<unsigned long long>(ordered_float(value)) << 32) |
           static_cast<unsigned int>(0xffffffffU - static_cast<unsigned int>(token));
}

__device__ __forceinline__ float key_float(unsigned long long key) {
    const unsigned int ordered = static_cast<unsigned int>(key >> 32);
    const unsigned int bits = (ordered & 0x80000000U) != 0U
                                  ? (ordered ^ 0x80000000U)
                                  : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ int key_token(unsigned long long key) {
    return static_cast<int>(0xffffffffU - static_cast<unsigned int>(key));
}

__device__ __forceinline__ int candidate_count(const SamplingConfig& config, int token_domain) {
    int count = kCandidateCap;
    if (config.top_k > 0 && config.top_k < kCandidateCap) { count = config.top_k; }
    return token_domain < count ? token_domain : count;
}

__device__ __forceinline__ float adjusted_logit(const hip_bfloat16* logits, int token,
                                                 const SamplingConfig& config,
                                                 bool greedy) {
    float value = static_cast<float>(logits[token]);
    if (greedy || (config.presence_penalty == 0.0F && config.frequency_penalty == 0.0F)) {
        return value;
    }
    const int count = config.token_counts == nullptr ? 0 : config.token_counts[token];
    if (count > 0) { value -= config.presence_penalty; }
    if (config.frequency_penalty != 0.0F) {
        value -= config.frequency_penalty * static_cast<float>(count);
    }
    return value;
}

__device__ unsigned long long block_max(unsigned long long value,
                                        unsigned long long* wave_values) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down(value, offset, 32);
        if (other > value) { value = other; }
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wave = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { wave_values[wave] = value; }
    __syncthreads();
    value = (wave == 0 && lane < (kBlock / 32)) ? wave_values[lane] : 0ULL;
    if (wave == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down(value, offset, 32);
            if (other > value) { value = other; }
        }
    }
    return value;
}

__device__ __forceinline__ unsigned long long wave_max(unsigned long long value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down(value, offset, 32);
        if (other > value) { value = other; }
    }
    return __shfl(value, 0, 32);
}

// Exact top-N over two keys per lane. Each wave first emits its top-N; wave zero then merges the
// eight ordered shortlists. It needs one workgroup barrier rather than a 512-key sorting network.
__device__ void select_block_top(unsigned long long first, unsigned long long second, int count,
                                 unsigned long long* wave_candidates,
                                 unsigned long long* selected) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wave = static_cast<int>(threadIdx.x) >> 5;
    for (int rank = 0; rank < count; ++rank) {
        const unsigned long long best = wave_max(first > second ? first : second);
        if (lane == 0) { wave_candidates[wave * kCandidateCap + rank] = best; }
        if (first == best) { first = 0ULL; }
        if (second == best) { second = 0ULL; }
    }
    __syncthreads();
    if (wave != 0) { return; }

    unsigned long long local[5];
#pragma unroll
    for (int item = 0; item < 5; ++item) {
        const int packed = lane + item * 32;
        if (packed < (kBlock / 32) * count) {
            const int source_wave = packed / count;
            const int source_rank = packed - source_wave * count;
            local[item] = wave_candidates[source_wave * kCandidateCap + source_rank];
        } else {
            local[item] = 0ULL;
        }
    }
    for (int rank = 0; rank < count; ++rank) {
        unsigned long long lane_best = 0ULL;
#pragma unroll
        for (int item = 0; item < 5; ++item) {
            if (local[item] > lane_best) { lane_best = local[item]; }
        }
        const unsigned long long best = wave_max(lane_best);
        if (lane == 0) { selected[rank] = best; }
#pragma unroll
        for (int item = 0; item < 5; ++item) {
            if (local[item] == best) { local[item] = 0ULL; }
        }
    }
}

__device__ void publish_sample(const SamplingConfig& config, int logical_position, int purpose,
                               const unsigned long long* selected, int selected_count,
                               float* probabilities, int* output) {
    if (!(config.temperature > 0.0F)) {
        *output = key_token(selected[0]);
        return;
    }

    const float inverse_temperature = 1.0F / config.temperature;
    const float maximum = key_float(selected[0]) * inverse_temperature;
    float total = 0.0F;
    for (int index = 0; index < selected_count; ++index) {
        probabilities[index] = expf(key_float(selected[index]) * inverse_temperature - maximum);
        total += probabilities[index];
    }

    const float min_threshold = config.min_p > 0.0F
                                    ? config.min_p * probabilities[0]
                                    : -1.0F;
    const bool use_top_p = config.top_p < 1.0F;
    const float top_p_target = config.top_p * total;
    float cumulative = 0.0F;
    int support = 0;
    for (int index = 0; index < selected_count; ++index) {
        if (min_threshold >= 0.0F && probabilities[index] < min_threshold) { break; }
        cumulative += probabilities[index];
        support = index + 1;
        if (use_top_p && cumulative >= top_p_target) { break; }
    }
    if (support < 1) { support = 1; }
    float support_total = 0.0F;
    for (int index = 0; index < support; ++index) { support_total += probabilities[index]; }

    const float goal = uniform(config.seed, logical_position, purpose) * support_total;
    cumulative = 0.0F;
    int picked = key_token(selected[support - 1]);
    for (int index = 0; index < support; ++index) {
        cumulative += probabilities[index];
        if (goal < cumulative) {
            picked = key_token(selected[index]);
            break;
        }
    }
    *output = picked;
    if (config.token_counts != nullptr) { atomicAdd(&config.token_counts[picked], 1); }
}

__global__ __launch_bounds__(kBlock) void sampling_partial_kernel(
    const hip_bfloat16* logits, const SamplingConfig* configs, int token_domain,
    int physical_rows, unsigned long long* partial_keys, int partials) {
    const int partial = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const SamplingConfig config = configs[row];
    const bool greedy = !(config.temperature > 0.0F);
    const int base_token = partial * kTile;
    const hip_bfloat16* row_logits = logits + static_cast<std::size_t>(row) * physical_rows;

    const int count = greedy ? 1 : candidate_count(config, token_domain);
    const int first_token = base_token + static_cast<int>(threadIdx.x) * 2;
    const int second_token = first_token + 1;
    const unsigned long long first =
        first_token < token_domain
            ? sort_key(adjusted_logit(row_logits, first_token, config, greedy), first_token)
            : 0ULL;
    const unsigned long long second =
        second_token < token_domain
            ? sort_key(adjusted_logit(row_logits, second_token, config, greedy), second_token)
            : 0ULL;
    __shared__ unsigned long long wave_candidates[(kBlock / 32) * kCandidateCap];
    __shared__ unsigned long long selected[kCandidateCap];
    select_block_top(first, second, count, wave_candidates, selected);
    if (threadIdx.x == 0) {
        const std::size_t base =
            (static_cast<std::size_t>(row) * partials + partial) * kCandidateCap;
        for (int rank = 0; rank < count; ++rank) {
            partial_keys[base + rank] = selected[rank];
        }
    }
}

__global__ __launch_bounds__(kBlock) void sampling_finalize_kernel(
    const unsigned long long* partial_keys, int partials, const SamplingConfig* configs,
    const int* logical_positions, int purpose, int token_domain, int* output) {
    const int row = static_cast<int>(blockIdx.x);
    const SamplingConfig config = configs[row];
    const int count = !(config.temperature > 0.0F) ? 1 : candidate_count(config, token_domain);
    const int available = partials * count;
    const unsigned long long* row_keys =
        partial_keys + static_cast<std::size_t>(row) * partials * kCandidateCap;

    __shared__ unsigned long long chosen[kCandidateCap];
    __shared__ unsigned long long wave_values[kBlock / 32];
    __shared__ float probabilities[kCandidateCap];
    for (int rank = 0; rank < count; ++rank) {
        unsigned long long best = 0ULL;
        for (int item = static_cast<int>(threadIdx.x); item < available; item += kBlock) {
            const int partial = item / count;
            const int local_rank = item - partial * count;
            const unsigned long long candidate =
                row_keys[static_cast<std::size_t>(partial) * kCandidateCap + local_rank];
            bool used = false;
            for (int previous = 0; previous < rank; ++previous) {
                if (chosen[previous] == candidate) { used = true; break; }
            }
            if (!used && candidate > best) { best = candidate; }
        }
        best = block_max(best, wave_values);
        if (threadIdx.x == 0) { chosen[rank] = best; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        publish_sample(config, logical_positions[row], purpose, chosen, count, probabilities,
                       &output[row]);
    }
}

__global__ __launch_bounds__(kBlock) void sampling_small_kernel(
    const hip_bfloat16* logits, int* output, const SamplingConfig* configs,
    const int* logical_positions, int purpose, int token_domain, int physical_rows) {
    const int row = static_cast<int>(blockIdx.x);
    const SamplingConfig config = configs[row];
    const bool greedy = !(config.temperature > 0.0F);
    const hip_bfloat16* row_logits = logits + static_cast<std::size_t>(row) * physical_rows;
    __shared__ unsigned long long chosen[kCandidateCap];
    __shared__ float probabilities[kCandidateCap];
    const int count = greedy ? 1 : candidate_count(config, token_domain);
    const int first_token = static_cast<int>(threadIdx.x) * 2;
    const int second_token = first_token + 1;
    const unsigned long long first =
        first_token < token_domain
            ? sort_key(adjusted_logit(row_logits, first_token, config, greedy), first_token)
            : 0ULL;
    const unsigned long long second =
        second_token < token_domain
            ? sort_key(adjusted_logit(row_logits, second_token, config, greedy), second_token)
            : 0ULL;
    __shared__ unsigned long long wave_candidates[(kBlock / 32) * kCandidateCap];
    select_block_top(first, second, count, wave_candidates, chosen);
    if (threadIdx.x == 0) {
        publish_sample(config, logical_positions[row], purpose, chosen, count, probabilities,
                       &output[row]);
    }
}

} // namespace

std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain,
                                              std::int32_t min_lanes,
                                              std::int32_t max_lanes) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes ||
        max_lanes > kMaximumRequestRows) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    return exact_workspace_bytes(token_domain, max_lanes);
}

void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions,
            std::int32_t purpose, WorkspaceArena& workspace, hipStream_t stream) {
    if (stream == nullptr) { throw std::invalid_argument("sample: stream must be non-null"); }
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("sample: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1 || logits.ne[0] <= 0 || logits.ne[1] <= 0 ||
        logits.ne[1] > kMaximumRequestRows) {
        throw std::invalid_argument("sample: logits must be rank-2 [physical_rows,B], B in [1,4]");
    }
    if (out.ne[0] != logits.ne[1] || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be I32 [B]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1,physical_rows]");
    }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != logits.ne[1] ||
        logical_positions.ne[1] != 1 || logical_positions.ne[2] != 1 ||
        logical_positions.ne[3] != 1) {
        throw std::invalid_argument("sample: logical_positions must be I32 [B]");
    }
    if (!logits.is_contiguous() || !out.is_contiguous() || !logical_positions.is_contiguous()) {
        throw std::invalid_argument("sample: tensors must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr || logical_positions.data == nullptr ||
        configs == nullptr) {
        throw std::invalid_argument("sample: device pointers must be non-null");
    }

    const int rows = logits.ne[1];
    auto scope = workspace.scope();
    const std::size_t bytes = exact_workspace_bytes(token_domain, rows);
    if (bytes == 0) {
        hipLaunchKernelGGL(sampling_small_kernel, dim3(rows), dim3(kBlock), 0, stream,
                           static_cast<const hip_bfloat16*>(logits.data),
                           static_cast<int*>(out.data), configs,
                           static_cast<const int*>(logical_positions.data), purpose,
                           token_domain, logits.ne[0]);
        HIP_CHECK(hipGetLastError());
        return;
    }

    DeviceSpan scratch = workspace.alloc_bytes(bytes);
    auto* partial_keys = static_cast<unsigned long long*>(scratch.data);
    const int partials = partial_count(token_domain);
    hipLaunchKernelGGL(sampling_partial_kernel, dim3(partials, rows), dim3(kBlock), 0, stream,
                       static_cast<const hip_bfloat16*>(logits.data), configs, token_domain,
                       logits.ne[0], partial_keys, partials);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(sampling_finalize_kernel, dim3(rows), dim3(kBlock), 0, stream,
                       partial_keys, partials, configs,
                       static_cast<const int*>(logical_positions.data), purpose,
                       token_domain, static_cast<int*>(out.data));
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops
