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

constexpr int kHeadDim      = 72;
constexpr int kHeads        = 16;
constexpr int kMatrixTile   = 16;
constexpr int kWave         = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kQueryTile    = kWavesPerBlock * kMatrixTile;
constexpr int kThreads      = kWave * kWavesPerBlock;
// Keys staged per iteration; the score and value products run over these two 16-key subtiles.
constexpr int kKeyTile      = 32;
constexpr int kKeySubtiles  = kKeyTile / kMatrixTile;
constexpr int kFeatureTiles = (kHeadDim + kMatrixTile - 1) / kMatrixTile;
constexpr int kPaddedDim    = kFeatureTiles * kMatrixTile;
constexpr int kChunk        = 8; // BF16 elements per 16-byte vector
constexpr int kChunks       = kHeadDim / kChunk;
// LDS row pitches (elements): 16-byte aligned rows, offset to spread banks.
constexpr int kKeyPitch     = kPaddedDim + kChunk;
constexpr int kValuePitch   = kKeyTile + kChunk;
constexpr float kScale      = 0.11785113019775792073F;
constexpr float kLog2Scale  = kScale * 1.44269504088896340736F;
static_assert(kHeadDim % kChunk == 0);

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

__device__ __forceinline__ short bf16_bits(float value) {
    return static_cast<short>(hip_bfloat16(value).data);
}

// Flash attention over one segment-bounded 128-query tile and one head. Scores are formed
// transposed, S^T = K Q^T, so each lane owns one query column of the accumulator: the
// softmax statistics are per-lane values combined with the partner half-wave, and the
// accumulator already has the B-operand layout of O^T = V^T P^T, with no LDS transpose. K and a
// transposed V are staged per 32-key tile with 16-byte loads; probabilities enter the value
// product as a BF16 high part plus its BF16 residual.
__global__ __launch_bounds__(kThreads) void vision_attention_wave32_kernel(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v, const VisionAttentionTile* __restrict__ tiles,
    int patches, int uniform_segment_length, hip_bfloat16* __restrict__ output,
    std::int64_t q_stride_t, std::int64_t k_stride_t, std::int64_t v_stride_t) {
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

    __shared__ __attribute__((aligned(16))) short keys[kKeyTile * kKeyPitch];
    __shared__ __attribute__((aligned(16))) short values_t[kPaddedDim * kValuePitch];

    const int tid   = static_cast<int>(threadIdx.x);
    const int wave  = tid / kWave;
    const int lane  = tid % kWave;
    const int half  = lane / kMatrixTile;
    const int axis  = lane % kMatrixTile;
    const int head  = static_cast<int>(blockIdx.y);
    const int query = tile.q0 + wave * kMatrixTile + axis;
    const bool query_valid = query < tile.end;

    // Padding features 72..79 of K and of V^T stay zero for every key tile.
    for (int index = tid; index < kKeyTile; index += kThreads) {
        *reinterpret_cast<uint4*>(&keys[index * kKeyPitch + kHeadDim]) = uint4{};
    }
    for (int index = tid; index < (kPaddedDim - kHeadDim) * kValuePitch; index += kThreads) {
        values_t[kHeadDim * kValuePitch + index] = 0;
    }

    // Q^T as the B operand: this lane's query column, features ft*16 + half*8 + 0..7.
    Bf16Fragment query_fragment[kFeatureTiles];
#pragma unroll
    for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
        const int chunk = feature_tile * 2 + half;
        query_fragment[feature_tile] = Bf16Fragment{};
        if (query_valid && chunk < kChunks) {
            query_fragment[feature_tile] = *reinterpret_cast<const Bf16Fragment*>(
                q + static_cast<std::int64_t>(query) * q_stride_t + head * kHeadDim +
                chunk * kChunk);
        }
    }

    float maximum     = -INFINITY;
    float denominator = 0.0F;
    F32Fragment attended[kFeatureTiles];
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
        attended[dimension_tile] = F32Fragment{};
    }

    for (int key_base = tile.begin; key_base < tile.end; key_base += kKeyTile) {
        __syncthreads();
        for (int index = tid; index < kKeyTile * kChunks; index += kThreads) {
            const int key_local = index / kChunks;
            const int chunk     = index - key_local * kChunks;
            const int key       = key_base + key_local;
            uint4 key_bits{};
            uint4 value_bits{};
            if (key < tile.end) {
                key_bits = *reinterpret_cast<const uint4*>(
                    k + static_cast<std::int64_t>(key) * k_stride_t + head * kHeadDim +
                    chunk * kChunk);
                value_bits = *reinterpret_cast<const uint4*>(
                    v + static_cast<std::int64_t>(key) * v_stride_t + head * kHeadDim +
                    chunk * kChunk);
            }
            *reinterpret_cast<uint4*>(&keys[key_local * kKeyPitch + chunk * kChunk]) = key_bits;
            const short* value_elements = reinterpret_cast<const short*>(&value_bits);
#pragma unroll
            for (int element = 0; element < kChunk; ++element) {
                values_t[(chunk * kChunk + element) * kValuePitch + key_local] =
                    value_elements[element];
            }
        }
        __syncthreads();

        // S^T subtiles: rows are keys kt*16 + half*8 + item, the column is this lane's query.
        F32Fragment score[kKeySubtiles];
#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
            score[key_tile] = F32Fragment{};
#pragma unroll
            for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
                const Bf16Fragment key_fragment = *reinterpret_cast<const Bf16Fragment*>(
                    &keys[(key_tile * kMatrixTile + axis) * kKeyPitch +
                          feature_tile * kMatrixTile + half * kChunk]);
                score[key_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    key_fragment, query_fragment[feature_tile], score[key_tile]);
            }
        }

        float tile_maximum = -INFINITY;
#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const int key = key_base + key_tile * kMatrixTile + half * kChunk + item;
                const float scaled = key < tile.end ? score[key_tile][item] * kLog2Scale : -INFINITY;
                score[key_tile][item] = scaled;
                tile_maximum = fmaxf(tile_maximum, scaled);
            }
        }
        tile_maximum = fmaxf(tile_maximum, __shfl_xor(tile_maximum, kMatrixTile, kWave));
        const float next_maximum = fmaxf(maximum, tile_maximum);
        const float alpha = exp2f(maximum - next_maximum);
        float tile_sum = 0.0F;
#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const float probability = exp2f(score[key_tile][item] - next_maximum);
                score[key_tile][item] = probability;
                tile_sum += probability;
            }
        }
        tile_sum += __shfl_xor(tile_sum, kMatrixTile, kWave);
        denominator = denominator * alpha + tile_sum;
        maximum = next_maximum;
#pragma unroll
        for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
            attended[dimension_tile] *= alpha;
        }

        // O^T += V^T P^T: P^T's B operand is exactly this lane's score column.
#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
            Bf16Fragment probability_high;
            Bf16Fragment probability_low;
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const float probability = score[key_tile][item];
                const hip_bfloat16 high(probability);
                probability_high[item] = static_cast<short>(high.data);
                probability_low[item]  = bf16_bits(probability - static_cast<float>(high));
            }
#pragma unroll
            for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
                const Bf16Fragment value_fragment = *reinterpret_cast<const Bf16Fragment*>(
                    &values_t[(dimension_tile * kMatrixTile + axis) * kValuePitch +
                              key_tile * kMatrixTile + half * kChunk]);
                attended[dimension_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    value_fragment, probability_high, attended[dimension_tile]);
                attended[dimension_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    value_fragment, probability_low, attended[dimension_tile]);
            }
        }
    }

    if (!query_valid) { return; }
    const float inverse = 1.0F / denominator;
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
        const int chunk = dimension_tile * 2 + half;
        if (chunk >= kChunks) { continue; }
        Bf16Fragment packed;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            packed[item] = bf16_bits(attended[dimension_tile][item] * inverse);
        }
        *reinterpret_cast<Bf16Fragment*>(
            output + (static_cast<std::int64_t>(query) * kHeads + head) * kHeadDim +
            chunk * kChunk) = packed;
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
    if (reinterpret_cast<std::uintptr_t>(tensor.data) % 16 != 0 || tensor.nb[2] % 16 != 0) {
        throw std::invalid_argument(std::string("vision_attention: ") + label +
                                    " rows must be 16-byte aligned");
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
        static_cast<hip_bfloat16*>(out.data), stride_elements(q, 2), stride_elements(k, 2),
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
