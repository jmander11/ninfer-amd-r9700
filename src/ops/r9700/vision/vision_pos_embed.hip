#include "ninfer/ops/vision_pos_embed.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kProductDimensions = 1152;
constexpr int kWave              = 32;
constexpr int kWaveTile          = 256;
constexpr int kCtaThreads        = 256;
constexpr int kSmallTargetBlocks = 4096;

__global__ __launch_bounds__(kWave) void vision_pos_embed_wave_kernel(
    const hip_bfloat16* __restrict__ table, const std::int32_t* __restrict__ indices,
    const float* __restrict__ weights, hip_bfloat16* __restrict__ x, int dimensions, int patches,
    int tiles_per_patch) {
    const int lane  = static_cast<int>(threadIdx.x);
    const int patch = static_cast<int>(blockIdx.x) / tiles_per_patch;
    const int tile  = static_cast<int>(blockIdx.x) - patch * tiles_per_patch;
    if (patch >= patches) { return; }

    const std::int64_t control = static_cast<std::int64_t>(patch) * 4 + lane;
    const int lane_index       = lane < 4 ? indices[control] : 0;
    const float lane_weight    = lane < 4 ? weights[control] : 0.0F;
    int corner_indices[4];
    float corner_weights[4];
#pragma unroll
    for (int corner = 0; corner < 4; ++corner) {
        corner_indices[corner] = __shfl(lane_index, corner, kWave);
        corner_weights[corner] = __shfl(lane_weight, corner, kWave);
    }

    const int first = tile * kWaveTile + lane;
    const std::int64_t x_base = static_cast<std::int64_t>(patch) * dimensions;
#pragma unroll
    for (int item = 0; item < kWaveTile / kWave; ++item) {
        const int dimension = first + item * kWave;
        if (dimension >= dimensions) { continue; }
        float value = static_cast<float>(x[x_base + dimension]);
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            value += static_cast<float>(
                         table[static_cast<std::int64_t>(corner_indices[corner]) * dimensions +
                               dimension]) *
                     corner_weights[corner];
        }
        x[x_base + dimension] = hip_bfloat16(value);
    }
}

__global__ __launch_bounds__(kCtaThreads) void vision_pos_embed_cta_kernel(
    const hip_bfloat16* __restrict__ table, const std::int32_t* __restrict__ indices,
    const float* __restrict__ weights, hip_bfloat16* __restrict__ x, int dimensions, int patches) {
    const int patch = static_cast<int>(blockIdx.x);
    if (patch >= patches) { return; }
    __shared__ std::int32_t corner_indices[4];
    __shared__ float corner_weights[4];
    if (threadIdx.x < 4) {
        const std::int64_t control = static_cast<std::int64_t>(patch) * 4 + threadIdx.x;
        corner_indices[threadIdx.x] = indices[control];
        corner_weights[threadIdx.x] = weights[control];
    }
    __syncthreads();

    const std::int64_t x_base = static_cast<std::int64_t>(patch) * dimensions;
    for (int dimension = static_cast<int>(threadIdx.x); dimension < dimensions;
         dimension += kCtaThreads) {
        float value = static_cast<float>(x[x_base + dimension]);
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            value += static_cast<float>(
                         table[static_cast<std::int64_t>(corner_indices[corner]) * dimensions +
                               dimension]) *
                     corner_weights[corner];
        }
        x[x_base + dimension] = hip_bfloat16(value);
    }
}

void require_contiguous(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.dtype != dtype || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("vision_pos_embed_add: invalid ") + label);
    }
}

} // namespace

void vision_pos_embed_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                          Tensor& x, hipStream_t stream) {
    require_contiguous(table, DType::BF16, "table");
    require_contiguous(indices, DType::I32, "indices");
    require_contiguous(weights, DType::FP32, "weights");
    require_contiguous(x, DType::BF16, "x");
    if (table.ne[2] != 1 || table.ne[3] != 1 || x.ne[2] != 1 || x.ne[3] != 1 ||
        indices.ne[0] != 4 || indices.ne[1] != x.ne[1] || indices.ne[2] != 1 ||
        indices.ne[3] != 1 || weights.ne[0] != 4 || weights.ne[1] != x.ne[1] ||
        weights.ne[2] != 1 || weights.ne[3] != 1 || table.ne[0] != x.ne[0]) {
        throw std::invalid_argument(
            "vision_pos_embed_add: expected table [D,R], controls [4,P], x [D,P]");
    }
    if (x.ne[0] < 0 || x.ne[1] < 0 || table.ne[1] <= 0) {
        throw std::invalid_argument("vision_pos_embed_add: invalid dimensions");
    }
    if (x.ne[0] == 0 || x.ne[1] == 0) { return; }
    if (table.data == nullptr || indices.data == nullptr || weights.data == nullptr ||
        x.data == nullptr) {
        throw std::invalid_argument("vision_pos_embed_add: tensor data must be non-null");
    }
    const int dimensions = x.ne[0];
    const int patches    = x.ne[1];
    if (dimensions == kProductDimensions && patches < 1024) {
        const int dimension_tiles = (dimensions + kWaveTile - 1) / kWaveTile;
        const int occupancy_tiles = std::max(1, (kSmallTargetBlocks + patches - 1) / patches);
        const int tiles = std::min(dimension_tiles, occupancy_tiles);
        vision_pos_embed_wave_kernel<<<static_cast<unsigned>(patches * tiles), kWave, 0, stream>>>(
            static_cast<const hip_bfloat16*>(table.data),
            static_cast<const std::int32_t*>(indices.data),
            static_cast<const float*>(weights.data), static_cast<hip_bfloat16*>(x.data),
            dimensions, patches, tiles);
    } else {
        vision_pos_embed_cta_kernel<<<static_cast<unsigned>(patches), kCtaThreads, 0, stream>>>(
            static_cast<const hip_bfloat16*>(table.data),
            static_cast<const std::int32_t*>(indices.data),
            static_cast<const float*>(weights.data), static_cast<hip_bfloat16*>(x.data),
            dimensions, patches);
    }
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops
