// Sole gfx1201 implementation of include/ninfer/ops/speculative_round.h.
//
// The stochastic route uses two explicit stages at the real vocabulary: one workgroup per
// 512-token partial produces its exact top-20, then one workgroup merges the partials and stores
// the filtered distribution for the row/column.  A final row-local kernel performs the ordered
// Leviathan or SpecInfer state transition.  Every transient byte belongs to the caller's
// WorkspaceArena; there is no allocation, host readback, or mutable RNG state in the launch path.
#include "ninfer/ops/speculative_round.h"
#include "ninfer/types.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kBlock = 256;
constexpr int kTile = 512;
constexpr int kCandidateCap = 20;
constexpr int kMaximumColumns = 16;
constexpr int kMaximumBatch = static_cast<int>(kMaximumConcurrency);
constexpr std::size_t kWorkspaceAlignment = 256;

[[nodiscard]] constexpr std::size_t align_up(std::size_t value,
                                             std::size_t alignment) noexcept {
    return (value + alignment - 1U) / alignment * alignment;
}

[[nodiscard]] constexpr int partial_count(int token_domain) noexcept {
    return (token_domain + kTile - 1) / kTile;
}

[[nodiscard]] std::size_t checked_product(std::size_t lhs, std::size_t rhs,
                                          const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return lhs * rhs;
}

struct SpeculativeWorkspaceLayout {
    std::size_t partial_keys_offset = 0;
    std::size_t dist_ids_offset = 0;
    std::size_t dist_prob_offset = 0;
    std::size_t dist_support_offset = 0;
    std::size_t bytes = 0;
    int partials = 0;
    int columns = 0;
    int batch = 0;

    struct View {
        unsigned long long* partial_keys = nullptr;
        std::int32_t* dist_ids = nullptr;
        float* dist_prob = nullptr;
        std::int32_t* dist_support = nullptr;
    };

    [[nodiscard]] View bind(DeviceSpan span) const {
        if (bytes == 0) { return {}; }
        if (span.data == nullptr || span.bytes < bytes) {
            throw std::invalid_argument("speculative acceptance workspace is undersized");
        }
        auto* base = static_cast<std::byte*>(span.data);
        return {
            reinterpret_cast<unsigned long long*>(base + partial_keys_offset),
            reinterpret_cast<std::int32_t*>(base + dist_ids_offset),
            reinterpret_cast<float*>(base + dist_prob_offset),
            reinterpret_cast<std::int32_t*>(base + dist_support_offset),
        };
    }
};

[[nodiscard]] SpeculativeWorkspaceLayout make_workspace_layout(int token_domain, int columns,
                                                               int batch) {
    SpeculativeWorkspaceLayout out{};
    if (token_domain <= kTile) { return out; }
    out.partials = partial_count(token_domain);
    out.columns = columns;
    out.batch = batch;
    const std::size_t lanes = checked_product(static_cast<std::size_t>(columns),
                                              static_cast<std::size_t>(batch),
                                              "speculative workspace lanes");
    const std::size_t partial_items = checked_product(
        checked_product(lanes, static_cast<std::size_t>(out.partials),
                        "speculative workspace partials"),
        static_cast<std::size_t>(kCandidateCap), "speculative workspace candidates");
    const std::size_t dist_items = checked_product(lanes,
                                                   static_cast<std::size_t>(kCandidateCap),
                                                   "speculative workspace distributions");
    std::size_t cursor = 0;
    out.partial_keys_offset = cursor;
    cursor = align_up(checked_product(partial_items, sizeof(unsigned long long),
                                      "speculative partial-key bytes"),
                      kWorkspaceAlignment);
    out.dist_ids_offset = cursor;
    cursor += align_up(checked_product(dist_items, sizeof(std::int32_t),
                                       "speculative distribution-id bytes"),
                       kWorkspaceAlignment);
    out.dist_prob_offset = cursor;
    cursor += align_up(checked_product(dist_items, sizeof(float),
                                       "speculative distribution-probability bytes"),
                       kWorkspaceAlignment);
    out.dist_support_offset = cursor;
    cursor += align_up(checked_product(lanes, sizeof(std::int32_t),
                                       "speculative support-size bytes"),
                       kWorkspaceAlignment);
    out.bytes = align_up(cursor, kWorkspaceAlignment);
    return out;
}

struct ByteRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

[[nodiscard]] ByteRange tensor_range(const Tensor& tensor) {
    const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const std::size_t bytes = tensor.bytes();
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
        throw std::invalid_argument("speculative_round: tensor storage range overflows");
    }
    return {begin, begin + bytes};
}

[[nodiscard]] bool overlaps(ByteRange lhs, ByteRange rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

[[nodiscard]] ByteRange pointer_range(const void* pointer, std::size_t bytes,
                                      const char* label) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
        throw std::invalid_argument(std::string(label) + " storage range overflows");
    }
    return {begin, begin + bytes};
}

void require_disjoint(const Tensor* const* tensors, std::size_t count, const char* op) {
    for (std::size_t lhs = 0; lhs < count; ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < count; ++rhs) {
            if (overlaps(tensor_range(*tensors[lhs]), tensor_range(*tensors[rhs]))) {
                throw std::invalid_argument(std::string(op) + ": tensor operands overlap");
            }
        }
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

void require_matrix(const Tensor& tensor, DType dtype, int rows, int columns, const char* op,
                    const char* name) {
    if (tensor.dtype != dtype || rows <= 0 || columns <= 0 || tensor.ne[0] != rows ||
        tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape or dtype for " + name);
    }
    require_contiguous_nonnull(tensor, op, name);
}

void require_vector(const Tensor& tensor, DType dtype, int count, const char* op,
                    const char* name) {
    require_matrix(tensor, dtype, count, 1, op, name);
}

void require_stream(hipStream_t stream, const char* op) {
    if (stream == nullptr) {
        throw std::invalid_argument(std::string(op) + ": stream must be non-null");
    }
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
    key = splitmix64(key ^
                     (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21));
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
    const unsigned int bits = (ordered & 0x80000000U) != 0U ? (ordered ^ 0x80000000U) : ~ordered;
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
                                                 const int* overlay, int overlay_count) {
    float value = static_cast<float>(logits[token]);
    if (config.presence_penalty == 0.0F && config.frequency_penalty == 0.0F) { return value; }
    int count = config.token_counts == nullptr ? 0 : config.token_counts[token];
    for (int index = 0; index < overlay_count; ++index) {
        if (overlay[index] == token) { ++count; }
    }
    if (count > 0) { value -= config.presence_penalty; }
    if (config.frequency_penalty != 0.0F) {
        value -= config.frequency_penalty * static_cast<float>(count);
    }
    return value;
}

__device__ __forceinline__ float adjusted_tree_logit(
    const hip_bfloat16* logits, int token, const SamplingConfig& config,
    const int* verify_ids, const int* parent_index, int node) {
    float value = static_cast<float>(logits[token]);
    if (config.presence_penalty == 0.0F && config.frequency_penalty == 0.0F) { return value; }
    int count = config.token_counts == nullptr ? 0 : config.token_counts[token];
    int current = node;
    for (int depth = 0; current > 0 && depth < kMaximumColumns; ++depth) {
        if (verify_ids[current] == token) { ++count; }
        current = parent_index[current];
    }
    if (count > 0) { value -= config.presence_penalty; }
    if (config.frequency_penalty != 0.0F) {
        value -= config.frequency_penalty * static_cast<float>(count);
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
    return __shfl(value, 0, 32);
}

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

__device__ int normalize_distribution(const SamplingConfig& config,
                                      const unsigned long long* selected, int selected_count,
                                      int* ids, float* probabilities) {
    const float inverse_temperature = 1.0F / config.temperature;
    const float maximum = key_float(selected[0]) * inverse_temperature;
    float total = 0.0F;
    for (int index = 0; index < selected_count; ++index) {
        ids[index] = key_token(selected[index]);
        probabilities[index] = expf(key_float(selected[index]) * inverse_temperature - maximum);
        total += probabilities[index];
    }
    const float min_threshold = config.min_p > 0.0F ? config.min_p * probabilities[0] : -1.0F;
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
    const float inverse_total = 1.0F / support_total;
    for (int index = 0; index < support; ++index) { probabilities[index] *= inverse_total; }
    return support;
}

__device__ __forceinline__ float selector_probability(const int* ids, const float* probabilities,
                                                       int count, int token) {
    if (ids == nullptr || probabilities == nullptr || count <= 0) { return 1.0F; }
    for (int index = 0; index < count; ++index) {
        if (ids[index] == token) { return probabilities[index]; }
    }
    return 0.0F;
}

__device__ int pick_distribution(const int* ids, const float* probabilities, int count,
                                 int excluded, float draw) {
    float mass = 0.0F;
    for (int index = 0; index < count; ++index) {
        if (ids[index] != excluded) { mass += probabilities[index]; }
    }
    if (!(mass > 0.0F)) { return ids[0]; }
    const float target = draw * mass;
    float cumulative = 0.0F;
    int picked = ids[0];
    for (int index = 0; index < count; ++index) {
        if (ids[index] == excluded) { continue; }
        cumulative += probabilities[index];
        picked = ids[index];
        if (target < cumulative) { break; }
    }
    return picked;
}

__device__ int pick_residual(const int* ids, const float* probabilities, int count,
                             const int* selector_ids, const float* selector_q,
                             int selector_count, int draft, float draw) {
    if (selector_ids == nullptr || selector_q == nullptr || selector_count <= 0) {
        return pick_distribution(ids, probabilities, count, draft, draw);
    }
    float mass = 0.0F;
    for (int index = 0; index < count; ++index) {
        mass += fmaxf(0.0F, probabilities[index] -
                                selector_probability(selector_ids, selector_q, selector_count,
                                                     ids[index]));
    }
    if (!(mass > 0.0F)) { return draft; }
    const float target = draw * mass;
    float cumulative = 0.0F;
    int picked = ids[0];
    for (int index = 0; index < count; ++index) {
        const float residual =
            fmaxf(0.0F, probabilities[index] -
                            selector_probability(selector_ids, selector_q, selector_count,
                                                 ids[index]));
        if (!(residual > 0.0F)) { continue; }
        cumulative += residual;
        picked = ids[index];
        if (target < cumulative) { break; }
    }
    return picked;
}

__device__ int tree_child(const int* parents, const int* ids, int node, int token, int valid) {
    for (int child = node + 1; child < valid; ++child) {
        if (parents[child] == node && ids[child] == token) { return child; }
    }
    return -1;
}

__device__ void chain_commit_greedy(const int* target_tokens, const int* drafts, int extent,
                                    int columns, int* lengths, int* anchors,
                                    int* licensed_tokens, int* licensed_counts, int* accepted,
                                    int row) {
    int count = 0;
    while (count < extent && target_tokens[count] == drafts[count]) { ++count; }
    const int terminal = target_tokens[count];
    for (int index = 0; index < columns; ++index) { licensed_tokens[index] = 0; }
    for (int index = 0; index < count; ++index) { licensed_tokens[index] = drafts[index]; }
    licensed_tokens[count] = terminal;
    licensed_counts[row] = count + 1;
    accepted[row] = count;
    anchors[row] = terminal;
    lengths[row] += count + 1;
}

__device__ void tree_commit_greedy(const int* target_tokens, const int* ids, const int* parents,
                                   int valid, int extent, int width, int* lengths, int* anchors,
                                   int* licensed_tokens, int* licensed_counts, int* accepted,
                                   int* accepted_column, int* fold_path, int row) {
    for (int index = 0; index < width; ++index) {
        licensed_tokens[index] = 0;
        fold_path[index] = 0;
    }
    int node = 0;
    int count = 0;
    fold_path[0] = 0;
    while (count < extent) {
        const int predicted = target_tokens[node];
        const int child = tree_child(parents, ids, node, predicted, valid);
        if (child < 0) {
            licensed_tokens[count] = predicted;
            licensed_counts[row] = count + 1;
            accepted[row] = count;
            accepted_column[row] = node;
            anchors[row] = predicted;
            lengths[row] += count + 1;
            return;
        }
        licensed_tokens[count] = predicted;
        node = child;
        fold_path[count + 1] = child;
        ++count;
    }
    licensed_tokens[count] = target_tokens[node];
    licensed_counts[row] = count + 1;
    accepted[row] = count;
    accepted_column[row] = node;
    anchors[row] = target_tokens[node];
    lengths[row] += count + 1;
}

__global__ void prepare_verify_kernel(const int* anchors, const int* drafts,
                                      const int* base_positions, const int* current_extents,
                                      int* verify_ids, int* positions, int drafts_per_row) {
    const int row = static_cast<int>(blockIdx.y);
    const int columns = drafts_per_row + 1;
    int extent = current_extents[row];
    extent = extent < 0 ? 0 : (extent > drafts_per_row ? drafts_per_row : extent);
    for (int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
         column < columns;
         column += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x)) {
        const int offset = row * columns + column;
        verify_ids[offset] = column == 0
                                 ? anchors[row]
                                 : (column <= extent
                                        ? drafts[row * drafts_per_row + column - 1]
                                        : anchors[row]);
        if (positions != nullptr) {
            positions[offset] = base_positions[row] + (column <= extent ? column : extent);
        }
    }
}

__global__ __launch_bounds__(kBlock) void distribution_partial_kernel(
    const hip_bfloat16* logits, int physical_rows, int token_domain, int columns, int partials,
    const SamplingConfig* configs, const int* drafts, const int* current_extents,
    const int* verify_ids, const int* parent_index, const int* valid_columns,
    unsigned long long* partial_keys) {
    const int partial = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int row = static_cast<int>(blockIdx.z);
    const SamplingConfig config = configs[row];
    if (!(config.temperature > 0.0F)) { return; }
    const bool tree = parent_index != nullptr;
    if (tree) {
        int valid = valid_columns[row];
        valid = valid < 1 ? 1 : (valid > columns ? columns : valid);
        if (column >= valid) { return; }
    } else {
        int extent = current_extents[row];
        extent = extent < 0 ? 0 : (extent > columns - 1 ? columns - 1 : extent);
        if (column > extent) { return; }
    }

    const int* row_ids = tree ? verify_ids + row * columns : nullptr;
    const int* row_parents = tree ? parent_index + row * columns : nullptr;
    const int* row_drafts = tree ? nullptr : drafts + row * (columns - 1);

    const int count = candidate_count(config, token_domain);
    const int base_token = partial * kTile;
    const int first_token = base_token + static_cast<int>(threadIdx.x) * 2;
    const int second_token = first_token + 1;
    const hip_bfloat16* column_logits =
        logits + (static_cast<std::size_t>(row) * columns + column) * physical_rows;
    const unsigned long long first = first_token < token_domain
                                         ? sort_key(tree ? adjusted_tree_logit(
                                                               column_logits, first_token, config,
                                                               row_ids, row_parents, column)
                                                         : adjusted_logit(
                                                               column_logits, first_token, config,
                                                               row_drafts, column),
                                                    first_token)
                                         : 0ULL;
    const unsigned long long second = second_token < token_domain
                                          ? sort_key(tree ? adjusted_tree_logit(
                                                                column_logits, second_token,
                                                                config, row_ids, row_parents,
                                                                column)
                                                          : adjusted_logit(
                                                                column_logits, second_token,
                                                                config, row_drafts, column),
                                                     second_token)
                                          : 0ULL;
    __shared__ unsigned long long wave_candidates[(kBlock / 32) * kCandidateCap];
    __shared__ unsigned long long selected[kCandidateCap];
    select_block_top(first, second, count, wave_candidates, selected);
    if (threadIdx.x == 0) {
        const std::size_t base =
            (((static_cast<std::size_t>(row) * columns + column) * partials + partial) *
             kCandidateCap);
        for (int rank = 0; rank < count; ++rank) { partial_keys[base + rank] = selected[rank]; }
    }
}

__global__ __launch_bounds__(kBlock) void distribution_finalize_kernel(
    const unsigned long long* partial_keys, int token_domain, int columns, int partials,
    const SamplingConfig* configs, const int* current_extents, const int* valid_columns,
    bool tree, int* dist_ids, float* dist_prob, int* dist_support) {
    const int column = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const SamplingConfig config = configs[row];
    if (!(config.temperature > 0.0F)) { return; }
    if (tree) {
        int valid = valid_columns[row];
        valid = valid < 1 ? 1 : (valid > columns ? columns : valid);
        if (column >= valid) { return; }
    } else {
        int extent = current_extents[row];
        extent = extent < 0 ? 0 : (extent > columns - 1 ? columns - 1 : extent);
        if (column > extent) { return; }
    }
    const int count = candidate_count(config, token_domain);
    const int available = partials * count;
    const std::size_t partial_base =
        (static_cast<std::size_t>(row) * columns + column) * partials * kCandidateCap;
    const std::size_t dist_base =
        (static_cast<std::size_t>(row) * columns + column) * kCandidateCap;
    __shared__ unsigned long long chosen[kCandidateCap];
    __shared__ unsigned long long wave_values[kBlock / 32];
    for (int rank = 0; rank < count; ++rank) {
        unsigned long long best = 0ULL;
        for (int item = static_cast<int>(threadIdx.x); item < available; item += kBlock) {
            const int partial = item / count;
            const int local_rank = item - partial * count;
            const unsigned long long candidate =
                partial_keys[partial_base + static_cast<std::size_t>(partial) * kCandidateCap +
                             local_rank];
            bool used = false;
            for (int previous = 0; previous < rank; ++previous) {
                if (chosen[previous] == candidate) {
                    used = true;
                    break;
                }
            }
            if (!used && candidate > best) { best = candidate; }
        }
        best = block_max(best, wave_values);
        if (threadIdx.x == 0) { chosen[rank] = best; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int support = normalize_distribution(config, chosen, count, dist_ids + dist_base,
                                                   dist_prob + dist_base);
        dist_support[static_cast<std::size_t>(row) * columns + column] = support;
    }
}

__device__ void selector_hop(const int* selector_ids, const float* selector_q, int selector_count,
                             int hop, int row, int drafts_per_row, const int*& ids,
                             const float*& probabilities) {
    ids = nullptr;
    probabilities = nullptr;
    if (selector_ids == nullptr || selector_q == nullptr || selector_count <= 0) { return; }
    const int base = selector_count * (hop + row * drafts_per_row);
    ids = selector_ids + base;
    probabilities = selector_q + base;
}

__global__ void chain_commit_kernel(
    const int* target_tokens, const int* drafts, const int* current_extents, int* lengths,
    int* anchors, int* licensed_tokens, int* licensed_counts, int* accepted,
    const SamplingConfig* configs, int columns, const int* selector_ids, const float* selector_q,
    int selector_count, const int* dist_ids, const float* dist_prob, const int* dist_support) {
    const int row = static_cast<int>(blockIdx.x);
    const int drafts_per_row = columns - 1;
    int extent = current_extents[row];
    extent = extent < 0 ? 0 : (extent > drafts_per_row ? drafts_per_row : extent);
    const int* row_targets = target_tokens + row * columns;
    const int* row_drafts = drafts + row * drafts_per_row;
    int* row_licensed = licensed_tokens + row * columns;
    const SamplingConfig config = configs[row];
    if (!(config.temperature > 0.0F)) {
        chain_commit_greedy(row_targets, row_drafts, extent, columns, lengths, anchors,
                            row_licensed, licensed_counts, accepted, row);
        return;
    }
    const int old_length = lengths[row];
    int accepted_count = 0;
    int terminal = 0;
    for (int column = 0; column <= extent; ++column) {
        const std::size_t dist_base =
            (static_cast<std::size_t>(row) * columns + column) * kCandidateCap;
        const int support = dist_support[static_cast<std::size_t>(row) * columns + column];
        const int* ids = dist_ids + dist_base;
        const float* probabilities = dist_prob + dist_base;
        if (column < extent) {
            const int draft = row_drafts[column];
            float target_probability = 0.0F;
            for (int candidate = 0; candidate < support; ++candidate) {
                if (ids[candidate] == draft) {
                    target_probability = probabilities[candidate];
                    break;
                }
            }
            const int* hop_ids = nullptr;
            const float* hop_q = nullptr;
            selector_hop(selector_ids, selector_q, selector_count, column, row, drafts_per_row,
                         hop_ids, hop_q);
            const float proposal_probability =
                selector_probability(hop_ids, hop_q, selector_count, draft);
            const float draw = uniform(config.seed, old_length + column + 1,
                                       kSamplePurposeSpeculativeAccept);
            if (proposal_probability > 0.0F &&
                draw < fminf(1.0F, target_probability / proposal_probability)) {
                accepted_count = column + 1;
                continue;
            }
            terminal = pick_residual(
                ids, probabilities, support, hop_ids, hop_q, selector_count, draft,
                uniform(config.seed, old_length + column + 1,
                        kSamplePurposeSpeculativeCorrection));
            break;
        }
        terminal = pick_distribution(
            ids, probabilities, support, -1,
            uniform(config.seed, old_length + extent + 1, kSamplePurposeSpeculativeBonus));
    }
    for (int index = 0; index < columns; ++index) { row_licensed[index] = 0; }
    for (int index = 0; index < accepted_count; ++index) { row_licensed[index] = row_drafts[index]; }
    row_licensed[accepted_count] = terminal;
    const int produced = accepted_count + 1;
    licensed_counts[row] = produced;
    accepted[row] = accepted_count;
    anchors[row] = terminal;
    lengths[row] = old_length + produced;
    if (config.token_counts != nullptr) {
        for (int index = 0; index < produced; ++index) {
            atomicAdd(&config.token_counts[row_licensed[index]], 1);
        }
    }
}

__global__ void tree_commit_kernel(
    const int* target_tokens, const int* verify_ids, const int* parent_index,
    const int* valid_columns, const int* current_extents, int* lengths, int* anchors,
    int* licensed_tokens, int* licensed_counts, int* accepted, int* accepted_column,
    int* fold_path, const SamplingConfig* configs, int width, const int* dist_ids,
    const float* dist_prob, const int* dist_support) {
    const int row = static_cast<int>(blockIdx.x);
    int valid = valid_columns[row];
    valid = valid < 1 ? 1 : (valid > width ? width : valid);
    int extent = current_extents[row];
    extent = extent < 0 ? 0 : (extent > width - 1 ? width - 1 : extent);
    const int* row_targets = target_tokens + row * width;
    const int* row_ids = verify_ids + row * width;
    const int* row_parents = parent_index + row * width;
    int* row_licensed = licensed_tokens + row * width;
    int* row_path = fold_path + row * width;
    const SamplingConfig config = configs[row];
    if (!(config.temperature > 0.0F)) {
        tree_commit_greedy(row_targets, row_ids, row_parents, valid, extent, width, lengths,
                           anchors, row_licensed, licensed_counts, accepted, accepted_column,
                           row_path, row);
        return;
    }
    for (int index = 0; index < width; ++index) {
        row_licensed[index] = 0;
        row_path[index] = 0;
    }
    const int old_length = lengths[row];
    int node = 0;
    int accepted_count = 0;
    int terminal = 0;
    row_path[0] = 0;
    for (int step = 0; step <= extent; ++step) {
        const std::size_t dist_base =
            (static_cast<std::size_t>(row) * width + node) * kCandidateCap;
        const int support = dist_support[static_cast<std::size_t>(row) * width + node];
        const int sampled = pick_distribution(
            dist_ids + dist_base, dist_prob + dist_base, support, -1,
            uniform(config.seed, old_length + accepted_count + 1,
                    kSamplePurposeSpeculativeAccept));
        const int child = step < extent ? tree_child(row_parents, row_ids, node, sampled, valid) : -1;
        if (child >= 0) {
            row_licensed[accepted_count] = sampled;
            row_path[accepted_count + 1] = child;
            node = child;
            ++accepted_count;
        } else {
            terminal = sampled;
            break;
        }
    }
    row_licensed[accepted_count] = terminal;
    const int produced = accepted_count + 1;
    licensed_counts[row] = produced;
    accepted[row] = accepted_count;
    accepted_column[row] = node;
    anchors[row] = terminal;
    lengths[row] = old_length + produced;
    if (config.token_counts != nullptr) {
        for (int index = 0; index < produced; ++index) {
            atomicAdd(&config.token_counts[row_licensed[index]], 1);
        }
    }
}

__device__ int build_small_distribution(
    const hip_bfloat16* logits, int token_domain, const SamplingConfig& config,
    const int* overlay, int overlay_count, unsigned long long* wave_candidates,
    unsigned long long* selected, int* ids, float* probabilities) {
    const int count = candidate_count(config, token_domain);
    const int first_token = static_cast<int>(threadIdx.x) * 2;
    const int second_token = first_token + 1;
    const unsigned long long first =
        first_token < token_domain
            ? sort_key(adjusted_logit(logits, first_token, config, overlay, overlay_count),
                       first_token)
            : 0ULL;
    const unsigned long long second =
        second_token < token_domain
            ? sort_key(adjusted_logit(logits, second_token, config, overlay, overlay_count),
                       second_token)
            : 0ULL;
    select_block_top(first, second, count, wave_candidates, selected);
    __syncthreads();
    int support = 0;
    if (threadIdx.x == 0) {
        support = normalize_distribution(config, selected, count, ids, probabilities);
    }
    return support;
}

__global__ __launch_bounds__(kBlock) void chain_small_kernel(
    const int* target_tokens, const hip_bfloat16* logits, int physical_rows,
    const int* drafts, const int* current_extents, int* lengths, int* anchors,
    int* licensed_tokens, int* licensed_counts, int* accepted, int token_domain,
    const SamplingConfig* configs, int columns, const int* selector_ids, const float* selector_q,
    int selector_count) {
    const int row = static_cast<int>(blockIdx.x);
    const int drafts_per_row = columns - 1;
    int extent = current_extents[row];
    extent = extent < 0 ? 0 : (extent > drafts_per_row ? drafts_per_row : extent);
    const int* row_targets = target_tokens + row * columns;
    const int* row_drafts = drafts + row * drafts_per_row;
    int* row_licensed = licensed_tokens + row * columns;
    const SamplingConfig config = configs[row];
    if (!(config.temperature > 0.0F)) {
        if (threadIdx.x == 0) {
            chain_commit_greedy(row_targets, row_drafts, extent, columns, lengths, anchors,
                                row_licensed, licensed_counts, accepted, row);
        }
        return;
    }
    __shared__ unsigned long long wave_candidates[(kBlock / 32) * kCandidateCap];
    __shared__ unsigned long long selected[kCandidateCap];
    __shared__ int ids[kCandidateCap];
    __shared__ float probabilities[kCandidateCap];
    __shared__ int support;
    __shared__ int accepted_count;
    __shared__ int terminal;
    __shared__ int done;
    __shared__ int old_length;
    if (threadIdx.x == 0) {
        accepted_count = 0;
        terminal = 0;
        done = 0;
        old_length = lengths[row];
    }
    __syncthreads();
    for (int column = 0; column <= extent; ++column) {
        const hip_bfloat16* column_logits =
            logits + (static_cast<std::size_t>(row) * columns + column) * physical_rows;
        const int local_support = build_small_distribution(
            column_logits, token_domain, config, row_drafts, column, wave_candidates, selected,
            ids, probabilities);
        if (threadIdx.x == 0) {
            support = local_support;
            if (column < extent) {
                const int draft = row_drafts[column];
                float target_probability = 0.0F;
                for (int candidate = 0; candidate < support; ++candidate) {
                    if (ids[candidate] == draft) {
                        target_probability = probabilities[candidate];
                        break;
                    }
                }
                const int* hop_ids = nullptr;
                const float* hop_q = nullptr;
                selector_hop(selector_ids, selector_q, selector_count, column, row,
                             drafts_per_row, hop_ids, hop_q);
                const float proposal_probability =
                    selector_probability(hop_ids, hop_q, selector_count, draft);
                const float draw = uniform(config.seed, old_length + column + 1,
                                           kSamplePurposeSpeculativeAccept);
                if (proposal_probability > 0.0F &&
                    draw < fminf(1.0F, target_probability / proposal_probability)) {
                    accepted_count = column + 1;
                } else {
                    terminal = pick_residual(
                        ids, probabilities, support, hop_ids, hop_q, selector_count, draft,
                        uniform(config.seed, old_length + column + 1,
                                kSamplePurposeSpeculativeCorrection));
                    done = 1;
                }
            } else {
                terminal = pick_distribution(
                    ids, probabilities, support, -1,
                    uniform(config.seed, old_length + extent + 1,
                            kSamplePurposeSpeculativeBonus));
                done = 1;
            }
        }
        __syncthreads();
        if (done != 0) { break; }
    }
    if (threadIdx.x == 0) {
        for (int index = 0; index < columns; ++index) { row_licensed[index] = 0; }
        for (int index = 0; index < accepted_count; ++index) {
            row_licensed[index] = row_drafts[index];
        }
        row_licensed[accepted_count] = terminal;
        const int produced = accepted_count + 1;
        licensed_counts[row] = produced;
        accepted[row] = accepted_count;
        anchors[row] = terminal;
        lengths[row] = old_length + produced;
        if (config.token_counts != nullptr) {
            for (int index = 0; index < produced; ++index) {
                atomicAdd(&config.token_counts[row_licensed[index]], 1);
            }
        }
    }
}

__global__ __launch_bounds__(kBlock) void tree_small_kernel(
    const int* target_tokens, const hip_bfloat16* logits, int physical_rows,
    const int* verify_ids, const int* parent_index, const int* valid_columns,
    const int* current_extents, int* lengths, int* anchors, int* licensed_tokens,
    int* licensed_counts, int* accepted, int* accepted_column, int* fold_path,
    int token_domain, const SamplingConfig* configs, int width) {
    const int row = static_cast<int>(blockIdx.x);
    int valid = valid_columns[row];
    valid = valid < 1 ? 1 : (valid > width ? width : valid);
    int extent = current_extents[row];
    extent = extent < 0 ? 0 : (extent > width - 1 ? width - 1 : extent);
    const int* row_targets = target_tokens + row * width;
    const int* row_ids = verify_ids + row * width;
    const int* row_parents = parent_index + row * width;
    int* row_licensed = licensed_tokens + row * width;
    int* row_path = fold_path + row * width;
    const SamplingConfig config = configs[row];
    if (!(config.temperature > 0.0F)) {
        if (threadIdx.x == 0) {
            tree_commit_greedy(row_targets, row_ids, row_parents, valid, extent, width, lengths,
                               anchors, row_licensed, licensed_counts, accepted, accepted_column,
                               row_path, row);
        }
        return;
    }
    __shared__ unsigned long long wave_candidates[(kBlock / 32) * kCandidateCap];
    __shared__ unsigned long long selected[kCandidateCap];
    __shared__ int ids[kCandidateCap];
    __shared__ float probabilities[kCandidateCap];
    __shared__ int path_tokens[kMaximumColumns];
    __shared__ int node;
    __shared__ int accepted_count;
    __shared__ int terminal;
    __shared__ int done;
    __shared__ int old_length;
    if (threadIdx.x == 0) {
        for (int index = 0; index < width; ++index) {
            row_licensed[index] = 0;
            row_path[index] = 0;
            path_tokens[index] = 0;
        }
        row_path[0] = 0;
        node = 0;
        accepted_count = 0;
        terminal = 0;
        done = 0;
        old_length = lengths[row];
    }
    __syncthreads();
    for (int step = 0; step <= extent; ++step) {
        const hip_bfloat16* node_logits =
            logits + (static_cast<std::size_t>(row) * width + node) * physical_rows;
        const int support = build_small_distribution(
            node_logits, token_domain, config, path_tokens, accepted_count, wave_candidates,
            selected, ids, probabilities);
        if (threadIdx.x == 0) {
            const int sampled = pick_distribution(
                ids, probabilities, support, -1,
                uniform(config.seed, old_length + accepted_count + 1,
                        kSamplePurposeSpeculativeAccept));
            const int child =
                step < extent ? tree_child(row_parents, row_ids, node, sampled, valid) : -1;
            if (child >= 0) {
                row_licensed[accepted_count] = sampled;
                path_tokens[accepted_count] = sampled;
                row_path[accepted_count + 1] = child;
                node = child;
                ++accepted_count;
            } else {
                terminal = sampled;
                done = 1;
            }
        }
        __syncthreads();
        if (done != 0) { break; }
    }
    if (threadIdx.x == 0) {
        row_licensed[accepted_count] = terminal;
        const int produced = accepted_count + 1;
        licensed_counts[row] = produced;
        accepted[row] = accepted_count;
        accepted_column[row] = node;
        anchors[row] = terminal;
        lengths[row] = old_length + produced;
        if (config.token_counts != nullptr) {
            for (int index = 0; index < produced; ++index) {
                atomicAdd(&config.token_counts[row_licensed[index]], 1);
            }
        }
    }
}

__global__ void select_hidden_kernel(const hip_bfloat16* hidden, const int* selectors,
                                     hip_bfloat16* output, int rows, int columns) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (row >= rows) { return; }
    const int column = selectors[batch];
    if (column < 0 || column >= columns) { return; }
    output[static_cast<std::size_t>(batch) * rows + row] =
        hidden[(static_cast<std::size_t>(batch) * columns + column) * rows + row];
}

__global__ void remap_kernel(int* tokens, int token_count, const int* id_map, int map_count) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    if (index >= token_count) { return; }
    const int token = tokens[index];
    if (token >= 0 && token < map_count) { tokens[index] = id_map[token]; }
}

void launch_large_distributions(const Tensor& logits, int token_domain, int columns, int batch,
                                const SamplingConfig* configs, const Tensor* drafts,
                                const Tensor& current_extents, const Tensor* verify_ids,
                                const Tensor* parent_index, const Tensor* valid_columns,
                                SpeculativeWorkspaceLayout::View workspace,
                                int partials, hipStream_t stream) {
    const dim3 partial_grid(static_cast<unsigned int>(partials),
                            static_cast<unsigned int>(columns),
                            static_cast<unsigned int>(batch));
    hipLaunchKernelGGL(
        distribution_partial_kernel, partial_grid, dim3(kBlock), 0, stream,
        static_cast<const hip_bfloat16*>(logits.data), logits.ne[0], token_domain, columns,
        partials, configs,
        drafts == nullptr ? nullptr : static_cast<const int*>(drafts->data),
        static_cast<const int*>(current_extents.data),
        verify_ids == nullptr ? nullptr : static_cast<const int*>(verify_ids->data),
        parent_index == nullptr ? nullptr : static_cast<const int*>(parent_index->data),
        valid_columns == nullptr ? nullptr : static_cast<const int*>(valid_columns->data),
        workspace.partial_keys);
    HIP_CHECK(hipGetLastError());
    const dim3 finalize_grid(static_cast<unsigned int>(columns),
                             static_cast<unsigned int>(batch));
    hipLaunchKernelGGL(
        distribution_finalize_kernel, finalize_grid, dim3(kBlock), 0, stream,
        workspace.partial_keys, token_domain, columns, partials, configs,
        static_cast<const int*>(current_extents.data),
        valid_columns == nullptr ? nullptr : static_cast<const int*>(valid_columns->data),
        parent_index != nullptr, workspace.dist_ids, workspace.dist_prob, workspace.dist_support);
    HIP_CHECK(hipGetLastError());
}

} // namespace

std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch) {
    if (token_domain <= 0 || min_drafts < 1 || max_drafts < min_drafts || max_drafts >=
            kMaximumColumns || min_batch < 1 || max_batch < min_batch ||
        max_batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative accept workspace: require K in [1,15] and B in [1,4]");
    }
    return make_workspace_layout(token_domain, max_drafts + 1, max_batch).bytes;
}

std::size_t speculative_accept_tree_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch, std::int32_t max_batch) {
    if (token_domain <= 0 || min_width < 2 || max_width < min_width ||
        max_width > kMaximumColumns || min_batch < 1 || max_batch < min_batch ||
        max_batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative tree workspace: require W in [2,16] and B in [1,4]");
    }
    return make_workspace_layout(token_domain, max_width, max_batch).bytes;
}

void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions,
                                       const Tensor& current_extents, Tensor& verify_ids,
                                       Tensor& positions, hipStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_inputs";
    require_stream(stream, op);
    const int drafts_per_row = drafts.ne[0];
    const int batch = drafts.ne[1];
    if (drafts_per_row < 1 || drafts_per_row >= kMaximumColumns || batch < 1 ||
        batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_prepare_verify_inputs: require K in [1,15] and B in [1,4]");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, drafts_per_row, batch, op, "drafts");
    require_vector(base_positions, DType::I32, batch, op, "base_positions");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, drafts_per_row + 1, batch, op, "verify_ids");
    require_matrix(positions, DType::I32, drafts_per_row + 1, batch, op, "positions");
    const Tensor* tensors[] = {&anchors, &drafts, &base_positions, &current_extents, &verify_ids,
                               &positions};
    require_disjoint(tensors, std::size(tensors), op);
    const dim3 grid(static_cast<unsigned int>((drafts_per_row + 32) / 32),
                    static_cast<unsigned int>(batch));
    hipLaunchKernelGGL(prepare_verify_kernel, grid, dim3(32), 0, stream,
                       static_cast<const int*>(anchors.data),
                       static_cast<const int*>(drafts.data),
                       static_cast<const int*>(base_positions.data),
                       static_cast<const int*>(current_extents.data),
                       static_cast<int*>(verify_ids.data), static_cast<int*>(positions.data),
                       drafts_per_row);
    HIP_CHECK(hipGetLastError());
}

void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    hipStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_ids";
    require_stream(stream, op);
    const int drafts_per_row = drafts.ne[0];
    const int batch = drafts.ne[1];
    if (drafts_per_row < 1 || drafts_per_row >= kMaximumColumns || batch < 1 ||
        batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_prepare_verify_ids: require K in [1,15] and B in [1,4]");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, drafts_per_row, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, drafts_per_row + 1, batch, op, "verify_ids");
    const Tensor* tensors[] = {&anchors, &drafts, &current_extents, &verify_ids};
    require_disjoint(tensors, std::size(tensors), op);
    const dim3 grid(static_cast<unsigned int>((drafts_per_row + 32) / 32),
                    static_cast<unsigned int>(batch));
    hipLaunchKernelGGL(prepare_verify_kernel, grid, dim3(32), 0, stream,
                       static_cast<const int*>(anchors.data),
                       static_cast<const int*>(drafts.data), nullptr,
                       static_cast<const int*>(current_extents.data),
                       static_cast<int*>(verify_ids.data), nullptr, drafts_per_row);
    HIP_CHECK(hipGetLastError());
}

void speculative_accept_greedy_drafts(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& current_extents, Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
    Tensor& licensed_counts, Tensor& accepted, std::int32_t token_domain,
    const SamplingConfig* configs, WorkspaceArena& workspace, hipStream_t stream,
    const Tensor* selector_ids, const Tensor* selector_q) {
    constexpr const char* op = "speculative_accept_greedy_drafts";
    require_stream(stream, op);
    const int drafts_per_row = drafts.ne[0];
    const int columns = drafts_per_row + 1;
    const int batch = drafts.ne[1];
    if (drafts_per_row < 1 || columns > kMaximumColumns || batch < 1 ||
        batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: require K in [1,15] and B in [1,4]");
    }
    require_matrix(target_tokens, DType::I32, columns, batch, op, "target_tokens");
    require_matrix(drafts, DType::I32, drafts_per_row, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(lengths, DType::I32, batch, op, "lengths");
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(licensed_tokens, DType::I32, columns, batch, op, "licensed_tokens");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    if (logits.dtype != DType::BF16 || logits.ne[0] <= 0 || logits.ne[1] != columns ||
        logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: logits must be BF16 [physical_rows,K+1,B]");
    }
    require_contiguous_nonnull(logits, op, "logits");
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: token_domain is outside physical logits");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: configs must be non-null");
    }
    if ((selector_ids == nullptr) != (selector_q == nullptr)) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: selector ids/q must be paired");
    }
    int selector_count = 0;
    if (selector_ids != nullptr) {
        selector_count = selector_ids->ne[0];
        if (selector_ids->dtype != DType::I32 || selector_q->dtype != DType::FP32 ||
            selector_count < 1 || selector_q->ne[0] != selector_count ||
            selector_ids->ne[1] != drafts_per_row || selector_q->ne[1] != drafts_per_row ||
            selector_ids->ne[2] != batch || selector_q->ne[2] != batch ||
            selector_ids->ne[3] != 1 || selector_q->ne[3] != 1) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: selector ids/q must be matching [C,K,B]");
        }
        require_contiguous_nonnull(*selector_ids, op, "selector_ids");
        require_contiguous_nonnull(*selector_q, op, "selector_q");
    }
    const Tensor* base_tensors[] = {&target_tokens, &logits, &drafts, &current_extents,
                                    &lengths, &anchors, &licensed_tokens, &licensed_counts,
                                    &accepted};
    require_disjoint(base_tensors, std::size(base_tensors), op);
    const ByteRange config_range = pointer_range(
        configs, checked_product(static_cast<std::size_t>(batch), sizeof(SamplingConfig),
                                 "speculative chain config bytes"),
        op);
    for (const Tensor* tensor : base_tensors) {
        if (overlaps(tensor_range(*tensor), config_range)) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: configs overlap tensor storage");
        }
    }
    if (selector_ids != nullptr) {
        for (const Tensor* tensor : base_tensors) {
            if (overlaps(tensor_range(*tensor), tensor_range(*selector_ids)) ||
                overlaps(tensor_range(*tensor), tensor_range(*selector_q)) ||
                overlaps(tensor_range(*selector_ids), tensor_range(*selector_q))) {
                throw std::invalid_argument(
                    "speculative_accept_greedy_drafts: selector operands overlap");
            }
        }
        if (overlaps(tensor_range(*selector_ids), config_range) ||
            overlaps(tensor_range(*selector_q), config_range)) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: configs overlap selector storage");
        }
    }

    auto scope = workspace.scope();
    const SpeculativeWorkspaceLayout layout =
        make_workspace_layout(token_domain, columns, batch);
    if (layout.bytes == 0) {
        hipLaunchKernelGGL(
            chain_small_kernel, dim3(batch), dim3(kBlock), 0, stream,
            static_cast<const int*>(target_tokens.data),
            static_cast<const hip_bfloat16*>(logits.data), logits.ne[0],
            static_cast<const int*>(drafts.data),
            static_cast<const int*>(current_extents.data), static_cast<int*>(lengths.data),
            static_cast<int*>(anchors.data), static_cast<int*>(licensed_tokens.data),
            static_cast<int*>(licensed_counts.data), static_cast<int*>(accepted.data),
            token_domain, configs, columns,
            selector_ids == nullptr ? nullptr : static_cast<const int*>(selector_ids->data),
            selector_q == nullptr ? nullptr : static_cast<const float*>(selector_q->data),
            selector_count);
        HIP_CHECK(hipGetLastError());
        return;
    }
    const auto scratch = layout.bind(workspace.alloc_bytes(layout.bytes));
    launch_large_distributions(logits, token_domain, columns, batch, configs, &drafts,
                               current_extents, nullptr, nullptr, nullptr, scratch,
                               layout.partials, stream);
    hipLaunchKernelGGL(
        chain_commit_kernel, dim3(batch), dim3(1), 0, stream,
        static_cast<const int*>(target_tokens.data), static_cast<const int*>(drafts.data),
        static_cast<const int*>(current_extents.data), static_cast<int*>(lengths.data),
        static_cast<int*>(anchors.data), static_cast<int*>(licensed_tokens.data),
        static_cast<int*>(licensed_counts.data), static_cast<int*>(accepted.data), configs,
        columns,
        selector_ids == nullptr ? nullptr : static_cast<const int*>(selector_ids->data),
        selector_q == nullptr ? nullptr : static_cast<const float*>(selector_q->data),
        selector_count, scratch.dist_ids, scratch.dist_prob, scratch.dist_support);
    HIP_CHECK(hipGetLastError());
}

void speculative_accept_tree_drafts(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& verify_ids,
    const Tensor& parent_index, const Tensor& valid_columns, const Tensor& current_extents,
    Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens, Tensor& licensed_counts,
    Tensor& accepted, Tensor& accepted_column, Tensor& fold_path, std::int32_t token_domain,
    const SamplingConfig* configs, WorkspaceArena& workspace, hipStream_t stream) {
    constexpr const char* op = "speculative_accept_tree_drafts";
    require_stream(stream, op);
    const int width = verify_ids.ne[0];
    const int batch = verify_ids.ne[1];
    if (width < 2 || width > kMaximumColumns || batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_accept_tree_drafts: require W in [2,16] and B in [1,4]");
    }
    require_matrix(target_tokens, DType::I32, width, batch, op, "target_tokens");
    require_matrix(verify_ids, DType::I32, width, batch, op, "verify_ids");
    require_matrix(parent_index, DType::I32, width, batch, op, "parent_index");
    require_vector(valid_columns, DType::I32, batch, op, "valid_columns");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(lengths, DType::I32, batch, op, "lengths");
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(licensed_tokens, DType::I32, width, batch, op, "licensed_tokens");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    require_vector(accepted_column, DType::I32, batch, op, "accepted_column");
    require_matrix(fold_path, DType::I32, width, batch, op, "fold_path");
    if (logits.dtype != DType::BF16 || logits.ne[0] <= 0 || logits.ne[1] != width ||
        logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_tree_drafts: logits must be BF16 [physical_rows,W,B]");
    }
    require_contiguous_nonnull(logits, op, "logits");
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_tree_drafts: token_domain is outside physical logits");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_tree_drafts: configs must be non-null");
    }
    const Tensor* tensors[] = {&target_tokens, &logits, &verify_ids, &parent_index,
                               &valid_columns, &current_extents, &lengths, &anchors,
                               &licensed_tokens, &licensed_counts, &accepted, &accepted_column,
                               &fold_path};
    require_disjoint(tensors, std::size(tensors), op);
    const ByteRange config_range = pointer_range(
        configs, checked_product(static_cast<std::size_t>(batch), sizeof(SamplingConfig),
                                 "speculative tree config bytes"),
        op);
    for (const Tensor* tensor : tensors) {
        if (overlaps(tensor_range(*tensor), config_range)) {
            throw std::invalid_argument(
                "speculative_accept_tree_drafts: configs overlap tensor storage");
        }
    }

    auto scope = workspace.scope();
    const SpeculativeWorkspaceLayout layout = make_workspace_layout(token_domain, width, batch);
    if (layout.bytes == 0) {
        hipLaunchKernelGGL(
            tree_small_kernel, dim3(batch), dim3(kBlock), 0, stream,
            static_cast<const int*>(target_tokens.data),
            static_cast<const hip_bfloat16*>(logits.data), logits.ne[0],
            static_cast<const int*>(verify_ids.data), static_cast<const int*>(parent_index.data),
            static_cast<const int*>(valid_columns.data),
            static_cast<const int*>(current_extents.data), static_cast<int*>(lengths.data),
            static_cast<int*>(anchors.data), static_cast<int*>(licensed_tokens.data),
            static_cast<int*>(licensed_counts.data), static_cast<int*>(accepted.data),
            static_cast<int*>(accepted_column.data), static_cast<int*>(fold_path.data),
            token_domain, configs, width);
        HIP_CHECK(hipGetLastError());
        return;
    }
    const auto scratch = layout.bind(workspace.alloc_bytes(layout.bytes));
    launch_large_distributions(logits, token_domain, width, batch, configs, nullptr,
                               current_extents, &verify_ids, &parent_index, &valid_columns,
                               scratch, layout.partials, stream);
    hipLaunchKernelGGL(
        tree_commit_kernel, dim3(batch), dim3(1), 0, stream,
        static_cast<const int*>(target_tokens.data), static_cast<const int*>(verify_ids.data),
        static_cast<const int*>(parent_index.data),
        static_cast<const int*>(valid_columns.data),
        static_cast<const int*>(current_extents.data), static_cast<int*>(lengths.data),
        static_cast<int*>(anchors.data), static_cast<int*>(licensed_tokens.data),
        static_cast<int*>(licensed_counts.data), static_cast<int*>(accepted.data),
        static_cast<int*>(accepted_column.data), static_cast<int*>(fold_path.data), configs,
        width, scratch.dist_ids, scratch.dist_prob, scratch.dist_support);
    HIP_CHECK(hipGetLastError());
}

void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        hipStream_t stream) {
    constexpr const char* op = "speculative_select_accepted_hidden";
    require_stream(stream, op);
    if (hidden.dtype != DType::BF16 || hidden.ne[0] <= 0 || hidden.ne[1] <= 0 ||
        hidden.ne[1] > kMaximumColumns || hidden.ne[2] <= 0 || hidden.ne[2] > kMaximumBatch ||
        hidden.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_select_accepted_hidden: hidden must be BF16 [D,T,B]");
    }
    require_contiguous_nonnull(hidden, op, "hidden");
    require_vector(selectors, DType::I32, hidden.ne[2], op, "selectors");
    require_matrix(out, DType::BF16, hidden.ne[0], hidden.ne[2], op, "out");
    const Tensor* tensors[] = {&hidden, &selectors, &out};
    require_disjoint(tensors, std::size(tensors), op);
    const dim3 grid(static_cast<unsigned int>((hidden.ne[0] + kBlock - 1) / kBlock),
                    static_cast<unsigned int>(hidden.ne[2]));
    hipLaunchKernelGGL(select_hidden_kernel, grid, dim3(kBlock), 0, stream,
                       static_cast<const hip_bfloat16*>(hidden.data),
                       static_cast<const int*>(selectors.data),
                       static_cast<hip_bfloat16*>(out.data), hidden.ne[0], hidden.ne[1]);
    HIP_CHECK(hipGetLastError());
}

void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, hipStream_t stream) {
    constexpr const char* op = "proposal_remap_token_ids";
    require_stream(stream, op);
    if (proposal_tokens.dtype != DType::I32 || proposal_tokens.ne[0] <= 0 ||
        proposal_tokens.ne[1] != 1 || proposal_tokens.ne[2] != 1 || proposal_tokens.ne[3] != 1) {
        throw std::invalid_argument(
            "proposal_remap_token_ids: proposal_tokens must be non-empty I32 [T]");
    }
    require_contiguous_nonnull(proposal_tokens, op, "proposal_tokens");
    if (id_map == nullptr || count <= 0) {
        throw std::invalid_argument("proposal_remap_token_ids: id_map must be non-null and count>0");
    }
    const ByteRange tokens = tensor_range(proposal_tokens);
    const std::size_t map_bytes = checked_product(static_cast<std::size_t>(count), sizeof(int),
                                                  "proposal id-map bytes");
    if (overlaps(tokens, pointer_range(id_map, map_bytes, op))) {
        throw std::invalid_argument("proposal_remap_token_ids: token and id-map storage overlap");
    }
    const int blocks = (proposal_tokens.ne[0] + kBlock - 1) / kBlock;
    hipLaunchKernelGGL(remap_kernel, dim3(blocks), dim3(kBlock), 0, stream,
                       static_cast<int*>(proposal_tokens.data), proposal_tokens.ne[0], id_map,
                       count);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops
