#include "ninfer/ops/vision_attention.h"

#include "core/device.h"
#include "core/layout.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim          = 72;
constexpr int kHeads            = 16;
constexpr int kQueryTile        = 64;
constexpr int kMatrixTile       = 16;
constexpr int kFeatureTiles     = (kHeadDim + kMatrixTile - 1) / kMatrixTile;
constexpr int kOutputTiles      = kFeatureTiles;
constexpr int kWave             = 32;
constexpr int kWavesPerBlock    = kQueryTile / kMatrixTile;
constexpr int kThreads          = kWave * kWavesPerBlock;
constexpr float kScale          = 0.11785113019775792073F;

using Bf16Fragment = short __attribute__((ext_vector_type(8)));
using F32Fragment  = float __attribute__((ext_vector_type(8)));

struct alignas(16) VisionAttentionTile {
    std::int32_t q0;
    std::int32_t begin;
    std::int32_t end;
    std::int32_t reserved;
};
static_assert(sizeof(VisionAttentionTile) == 16);

__global__ void vision_attention_prepare_tiles(const std::int32_t* cu_seqlens, int segments,
                                               VisionAttentionTile* tiles, int max_tiles,
                                               int patches) {
    for (int tile = static_cast<int>(threadIdx.x); tile < max_tiles; tile += blockDim.x) {
        tiles[tile] = {-1, 0, 0, 0};
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }
    int next = 0;
    for (int segment = 0; segment < segments; ++segment) {
        const int begin = cu_seqlens[segment];
        const int end   = cu_seqlens[segment + 1];
        if (begin < 0 || end <= begin || end > patches) { continue; }
        for (int q0 = begin; q0 < end && next < max_tiles; q0 += kQueryTile) {
            tiles[next++] = {q0, begin, end, 0};
        }
    }
}

__device__ __forceinline__ float subgroup_max(float value) {
#pragma unroll
    for (int offset = kMatrixTile / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor(value, offset, kMatrixTile));
    }
    return value;
}

__device__ __forceinline__ float subgroup_sum(float value) {
#pragma unroll
    for (int offset = kMatrixTile / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, kMatrixTile);
    }
    return value;
}

__global__ __launch_bounds__(kThreads) void vision_attention_wave32_kernel(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v, const VisionAttentionTile* __restrict__ tiles,
    int patches, int uniform_segment_length, hip_bfloat16* __restrict__ output,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    VisionAttentionTile tile;
    if (tiles != nullptr) {
        tile = tiles[blockIdx.x];
    } else if (uniform_segment_length > 0) {
        const int tiles_per_segment =
            (uniform_segment_length + kQueryTile - 1) / kQueryTile;
        const int segment = static_cast<int>(blockIdx.x) / tiles_per_segment;
        const int within  = static_cast<int>(blockIdx.x) - segment * tiles_per_segment;
        const int begin   = segment * uniform_segment_length;
        tile = {begin + within * kQueryTile, begin, begin + uniform_segment_length, 0};
    } else {
        tile = {static_cast<int>(blockIdx.x) * kQueryTile, 0, patches, 0};
    }
    if (tile.q0 < 0) { return; }

    const int tid  = static_cast<int>(threadIdx.x);
    const int wave = tid / kWave;
    const int lane = tid % kWave;
    const int half = lane / kMatrixTile;
    const int axis = lane % kMatrixTile;
    const int head = static_cast<int>(blockIdx.y);
    const int query_base = tile.q0 + wave * kMatrixTile;
    __shared__ float probability[kWavesPerBlock][kMatrixTile][kMatrixTile];

    float maximum[8];
    float denominator[8];
    F32Fragment attended[kOutputTiles];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        maximum[item] = -INFINITY;
        denominator[item] = 0.0F;
    }
#pragma unroll
    for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
        attended[output_tile] = F32Fragment{};
    }

    for (int key_base = tile.begin; key_base < tile.end; key_base += kMatrixTile) {
        F32Fragment score{};
#pragma unroll
        for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
            Bf16Fragment q_fragment{};
            Bf16Fragment k_fragment{};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const int feature = feature_tile * kMatrixTile + half * 8 + item;
                const int query = query_base + axis;
                const int key = key_base + axis;
                if (query < tile.end && feature < kHeadDim) {
                    const std::int64_t offset = static_cast<std::int64_t>(query) * q_stride_t +
                                                static_cast<std::int64_t>(head) * q_stride_h +
                                                static_cast<std::int64_t>(feature) * q_stride_d;
                    q_fragment[item] = static_cast<short>(q[offset].data);
                }
                if (key < tile.end && feature < kHeadDim) {
                    const std::int64_t offset = static_cast<std::int64_t>(key) * k_stride_t +
                                                static_cast<std::int64_t>(head) * k_stride_h +
                                                static_cast<std::int64_t>(feature) * k_stride_d;
                    k_fragment[item] = static_cast<short>(k[offset].data);
                }
            }
            score = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                q_fragment, k_fragment, score);
        }

        float alpha[8];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int query_row = half * 8 + item;
            const bool valid_query = query_base + query_row < tile.end;
            const bool valid_key = key_base + axis < tile.end;
            const float scaled_score =
                valid_query && valid_key ? score[item] * kScale : -INFINITY;
            const float block_maximum = subgroup_max(scaled_score);
            const float next_maximum = fmaxf(maximum[item], block_maximum);
            alpha[item] = isfinite(maximum[item]) ? expf(maximum[item] - next_maximum) : 0.0F;
            const float unnormalized =
                valid_query && valid_key ? expf(scaled_score - next_maximum) : 0.0F;
            const float block_sum = subgroup_sum(unnormalized);
            denominator[item] = denominator[item] * alpha[item] + block_sum;
            maximum[item] = next_maximum;
            probability[wave][query_row][axis] = unnormalized;
        }
        __syncthreads();

        for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                attended[output_tile][item] *= alpha[item];
            }
            Bf16Fragment probability_fragment{};
            Bf16Fragment probability_residual{};
            Bf16Fragment value_fragment{};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const int local_key = half * 8 + item;
                const int dimension = output_tile * kMatrixTile + axis;
                const float probability_value = probability[wave][axis][local_key];
                const hip_bfloat16 probability_high(probability_value);
                const hip_bfloat16 probability_low(
                    probability_value - static_cast<float>(probability_high));
                probability_fragment[item] = static_cast<short>(probability_high.data);
                probability_residual[item] = static_cast<short>(probability_low.data);
                if (key_base + local_key < tile.end && dimension < kHeadDim) {
                    const std::int64_t offset =
                        static_cast<std::int64_t>(key_base + local_key) * v_stride_t +
                        static_cast<std::int64_t>(head) * v_stride_h +
                        static_cast<std::int64_t>(dimension) * v_stride_d;
                    value_fragment[item] = static_cast<short>(v[offset].data);
                }
            }
            attended[output_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                probability_fragment, value_fragment, attended[output_tile]);
            attended[output_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                probability_residual, value_fragment, attended[output_tile]);
        }
        __syncthreads();
    }

    for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
        const int dimension = output_tile * kMatrixTile + axis;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int query_row = half * 8 + item;
            const int query = query_base + query_row;
            if (query < tile.end && dimension < kHeadDim) {
                const std::int64_t offset =
                    (static_cast<std::int64_t>(query) * kHeads + head) * kHeadDim + dimension;
                output[offset] = hip_bfloat16(attended[output_tile][item] / denominator[item]);
            }
        }
    }
}

std::int32_t scratch_tiles(std::int32_t patches, std::int32_t segments) {
    if (segments == 1) { return 0; }
    const std::int64_t tiles =
        (static_cast<std::int64_t>(patches) + kQueryTile - 1) / kQueryTile + segments - 1LL;
    if (tiles > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("vision_attention: descriptor tile count exceeds int32");
    }
    return static_cast<std::int32_t>(tiles);
}

template <class Allocator>
Tensor allocate_workspace(Allocator& allocator, std::int32_t patches, std::int32_t segments) {
    const std::int32_t tiles = scratch_tiles(patches, segments);
    return tiles == 0 ? Tensor{} : allocator.alloc(DType::I32, {4, tiles});
}

void require_qkv(const Tensor& tensor, std::int32_t patches, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHeadDim || tensor.ne[1] != kHeads ||
        tensor.ne[2] != patches || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("vision_attention: invalid ") + label + " shape");
    }
    constexpr std::int64_t element = sizeof(hip_bfloat16);
    if (tensor.nb[0] != element || tensor.nb[1] != element * kHeadDim ||
        tensor.nb[2] < element * kHeadDim * kHeads || tensor.nb[2] % element != 0) {
        throw std::invalid_argument(std::string("vision_attention: invalid ") + label +
                                    " strides");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("vision_attention: ") + label +
                                    " data must be non-null");
    }
}

std::int64_t stride_elements(const Tensor& tensor, int dimension) {
    return tensor.nb[dimension] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
}

void launch_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      const VisionAttentionTile* tiles, int uniform_segment_length,
                      int query_tiles, Tensor& out, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>(query_tiles), static_cast<unsigned>(kHeads), 1U);
    vision_attention_wave32_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const hip_bfloat16*>(q.data), static_cast<const hip_bfloat16*>(k.data),
        static_cast<const hip_bfloat16*>(v.data), tiles, q.ne[2], uniform_segment_length,
        static_cast<hip_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
        stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1),
        stride_elements(k, 2), stride_elements(v, 0), stride_elements(v, 1),
        stride_elements(v, 2));
    HIP_CHECK(hipGetLastError());
}

} // namespace

std::size_t vision_attention_workspace_capacity_bytes(std::int32_t min_patches,
                                                      std::int32_t max_patches,
                                                      std::int32_t min_segments,
                                                      std::int32_t max_segments) {
    if (min_patches <= 0 || max_patches < min_patches || min_segments <= 0 ||
        max_segments < min_segments || min_segments > max_patches) {
        throw std::invalid_argument("vision_attention workspace: invalid execution envelope");
    }
    const std::int32_t segments = std::min(max_segments, max_patches);
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_patches, segments);
    return layout.peak_bytes(1);
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      const Tensor& cu_seqlens, WorkspaceArena& workspace, Tensor& out,
                      hipStream_t stream) {
    const int patches  = q.ne[2];
    const int segments = cu_seqlens.ne[0] - 1;
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: out must be contiguous");
    }
    if (cu_seqlens.dtype != DType::I32 || segments <= 0 || segments > patches ||
        cu_seqlens.ne[1] != 1 || cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 ||
        !cu_seqlens.is_contiguous() || cu_seqlens.data == nullptr) {
        throw std::invalid_argument("vision_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    auto scope = workspace.scope();
    Tensor descriptor_tensor = allocate_workspace(workspace, patches, segments);
    VisionAttentionTile* descriptors =
        descriptor_tensor.data == nullptr
            ? nullptr
            : static_cast<VisionAttentionTile*>(descriptor_tensor.data);
    const int query_tiles = descriptors == nullptr ? (patches + kQueryTile - 1) / kQueryTile
                                                   : descriptor_tensor.ne[1];
    if (descriptors != nullptr) {
        vision_attention_prepare_tiles<<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), segments, descriptors,
            query_tiles, patches);
        HIP_CHECK(hipGetLastError());
    }
    launch_attention(q, k, v, descriptors, 0, query_tiles, out, stream);
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      std::int32_t segment_length, Tensor& out, hipStream_t stream) {
    const int patches = q.ne[2];
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: out must be contiguous");
    }
    if (segment_length <= 0 || patches % segment_length != 0) {
        throw std::invalid_argument("vision_attention: invalid uniform segment length");
    }
    const int segments = patches / segment_length;
    const int query_tiles = segments * ((segment_length + kQueryTile - 1) / kQueryTile);
    launch_attention(q, k, v, nullptr, segment_length, query_tiles, out, stream);
}

} // namespace ninfer::ops
