#include "ops/r9700/eager/eager_ops.h"

#include <hip/hip_fp16.h>

#include <climits>
#include <cmath>
#include <limits>

namespace ninfer::ops::r9700::eager {
namespace {

constexpr std::uint32_t kBlock = 256;

[[nodiscard]] bool valid_stream(hipStream_t stream) noexcept { return stream != nullptr; }

[[nodiscard]] bool valid_linear_count(std::size_t count) noexcept {
    return count != 0 && count <= static_cast<std::size_t>(UINT32_MAX) * kBlock;
}

[[nodiscard]] dim3 linear_grid(std::size_t count) noexcept {
    return dim3(static_cast<unsigned int>((count + kBlock - 1U) / kBlock));
}

template <typename T>
__global__ void fill_kernel(T* destination, std::size_t count, T value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = value; }
}

__global__ void iota_kernel(std::int32_t* destination, std::size_t count, std::int32_t start) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = start + static_cast<std::int32_t>(index); }
}

template <typename T>
__global__ void copy_kernel(const T* source, T* destination, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = source[index]; }
}

__global__ void offset_kernel(const std::int32_t* source, std::int32_t* destination,
                              std::size_t count, std::int32_t delta) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = source[index] + delta; }
}

__global__ void offset_scalar_kernel(const std::int32_t* source, const std::int32_t* delta,
                                     std::int32_t* destination, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = source[index] + delta[0]; }
}

__global__ void add_i32_scalars_kernel(const std::int32_t* lhs, const std::int32_t* rhs,
                                       std::int32_t* destination) {
    destination[0] = lhs[0] + rhs[0];
}

template <typename T>
__global__ void increment_kernel(T* scalar) {
    ++scalar[0];
}

__global__ void i32_gather_kernel(const std::int32_t* source, const std::int32_t* indices,
                                  std::int32_t* destination, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = source[indices[index]]; }
}

__global__ void bf16_scatter_kernel(const hip_bfloat16* source, const std::int32_t* indices,
                                    hip_bfloat16* destination, std::uint32_t features,
                                    std::uint32_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = static_cast<std::size_t>(features) * count;
    if (index >= elements) { return; }
    const std::uint32_t column = static_cast<std::uint32_t>(index / features);
    const std::uint32_t feature = static_cast<std::uint32_t>(index % features);
    destination[static_cast<std::size_t>(indices[column]) * features + feature] = source[index];
}

__global__ void bf16_gather_kernel(const hip_bfloat16* source, const std::int32_t* indices,
                                   hip_bfloat16* destination, std::uint32_t features,
                                   std::uint32_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = static_cast<std::size_t>(features) * count;
    if (index >= elements) { return; }
    const std::uint32_t column = static_cast<std::uint32_t>(index / features);
    const std::uint32_t feature = static_cast<std::uint32_t>(index % features);
    destination[index] = source[static_cast<std::size_t>(indices[column]) * features + feature];
}

__global__ void bf16_scatter_batch_kernel(const hip_bfloat16* source,
                                          const std::int32_t* lanes,
                                          const std::int32_t* valid_columns,
                                          hip_bfloat16* destination,
                                          std::uint32_t features,
                                          std::uint32_t width,
                                          std::size_t destination_column_stride,
                                          std::size_t destination_lane_stride) {
    const std::uint32_t column = blockIdx.x;
    const std::uint32_t batch = blockIdx.y;
    if (column >= static_cast<std::uint32_t>(valid_columns[batch])) { return; }
    const std::uint32_t lane = static_cast<std::uint32_t>(lanes[batch]);
    const std::size_t source_base =
        (static_cast<std::size_t>(batch) * width + column) * features;
    const std::size_t destination_base =
        static_cast<std::size_t>(lane) * destination_lane_stride +
        static_cast<std::size_t>(column) * destination_column_stride;
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        destination[destination_base + feature] = source[source_base + feature];
    }
}

__global__ void bf16_gather_path_kernel(hip_bfloat16* features,
                                        const std::int32_t* lanes,
                                        const std::int32_t* path,
                                        const std::int32_t* counts,
                                        std::uint32_t feature_count,
                                        std::uint32_t width) {
    const std::uint32_t batch = blockIdx.x;
    const std::uint32_t lane = static_cast<std::uint32_t>(lanes[batch]);
    const std::uint32_t count = static_cast<std::uint32_t>(counts[batch]);
    const std::int32_t* row = path + static_cast<std::size_t>(batch) * width;
    for (std::uint32_t column = 0; column < count; ++column) {
        const std::uint32_t source_column = static_cast<std::uint32_t>(row[column]);
        if (source_column != column) {
            const std::size_t source_base =
                (static_cast<std::size_t>(lane) * width + source_column) * feature_count;
            const std::size_t destination_base =
                (static_cast<std::size_t>(lane) * width + column) * feature_count;
            for (std::uint32_t feature = threadIdx.x; feature < feature_count;
                 feature += blockDim.x) {
                features[destination_base + feature] = features[source_base + feature];
            }
        }
        __syncthreads();
    }
}

__global__ void bf16_extract_features_kernel(const hip_bfloat16* source,
                                             hip_bfloat16* destination,
                                             std::uint32_t source_features,
                                             std::uint32_t source_feature,
                                             std::uint32_t destination_features,
                                             std::size_t elements) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    const std::size_t column = index / destination_features;
    const std::size_t feature = index - column * destination_features;
    destination[index] = source[column * source_features + source_feature + feature];
}

__global__ void cast_fp32_to_bf16_kernel(const float* source, hip_bfloat16* destination,
                                         std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = hip_bfloat16(source[index]); }
}

__global__ void cast_bf16_to_fp32_kernel(const hip_bfloat16* source, float* destination,
                                         std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { destination[index] = static_cast<float>(source[index]); }
}

__global__ void residual_add_kernel(const hip_bfloat16* y, hip_bfloat16* x,
                                    std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { x[index] = hip_bfloat16(static_cast<float>(x[index]) +
                                                  static_cast<float>(y[index])); }
}

__global__ void residual_rmsnorm_kernel(const hip_bfloat16* y, hip_bfloat16* x,
                                        const hip_bfloat16* weight,
                                        hip_bfloat16* output,
                                        std::uint32_t features, float eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        x[base + feature] = hip_bfloat16(static_cast<float>(x[base + feature]) +
                                        static_cast<float>(y[base + feature]));
    }
    __syncthreads();
    __shared__ float inverse;
    if (threadIdx.x == 0) {
        float sumsq = 0.0F;
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            const float value = static_cast<float>(x[base + feature]);
            sumsq = fmaf(value, value, sumsq);
        }
        inverse = rsqrtf(sumsq / static_cast<float>(features) + eps);
    }
    __syncthreads();
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        output[base + feature] = hip_bfloat16(
            static_cast<float>(x[base + feature]) * inverse *
            (1.0F + static_cast<float>(weight[feature])));
    }
}

__global__ void rmsnorm_kernel(const hip_bfloat16* input, const hip_bfloat16* weight,
                               hip_bfloat16* output, std::uint32_t features, float eps,
                               bool unit_offset) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    __shared__ float inverse;
    if (threadIdx.x == 0) {
        float sumsq = 0.0F;
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            const float value = static_cast<float>(input[base + feature]);
            sumsq = fmaf(value, value, sumsq);
        }
        inverse = rsqrtf(sumsq / static_cast<float>(features) + eps);
    }
    __syncthreads();
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        const float gain = static_cast<float>(weight[feature]) + (unit_offset ? 1.0F : 0.0F);
        output[base + feature] = hip_bfloat16(static_cast<float>(input[base + feature]) * inverse * gain);
    }
}

__device__ __forceinline__ hip_bfloat16 bf16_from_raw(std::uint32_t word, bool high) {
    hip_bfloat16 value;
    value.data = static_cast<std::uint16_t>(high ? word >> 16U : word);
    return value;
}

__global__ __launch_bounds__(kBlock) void rmsnorm_k5120_token8_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset) {
    constexpr std::uint32_t features = 5120;
    constexpr std::uint32_t tokens_per_block = kBlock / 32U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t wave = threadIdx.x >> 5U;
    const std::uint32_t row = blockIdx.x * tokens_per_block + wave;
    if (row >= rows) { return; }
    const std::size_t base = static_cast<std::size_t>(row) * features;
    float inverse = 0.0F;
    if (lane == 0U) {
        float sumsq = 0.0F;
        for (std::uint32_t vector = 0; vector < features / 8U; ++vector) {
            uint4 packed{};
            __builtin_memcpy(&packed, input + base + static_cast<std::size_t>(vector) * 8U,
                             sizeof(packed));
            const float x0 = static_cast<float>(bf16_from_raw(packed.x, false));
            const float x1 = static_cast<float>(bf16_from_raw(packed.x, true));
            const float x2 = static_cast<float>(bf16_from_raw(packed.y, false));
            const float x3 = static_cast<float>(bf16_from_raw(packed.y, true));
            const float x4 = static_cast<float>(bf16_from_raw(packed.z, false));
            const float x5 = static_cast<float>(bf16_from_raw(packed.z, true));
            const float x6 = static_cast<float>(bf16_from_raw(packed.w, false));
            const float x7 = static_cast<float>(bf16_from_raw(packed.w, true));
            sumsq = fmaf(x0, x0, sumsq);
            sumsq = fmaf(x1, x1, sumsq);
            sumsq = fmaf(x2, x2, sumsq);
            sumsq = fmaf(x3, x3, sumsq);
            sumsq = fmaf(x4, x4, sumsq);
            sumsq = fmaf(x5, x5, sumsq);
            sumsq = fmaf(x6, x6, sumsq);
            sumsq = fmaf(x7, x7, sumsq);
        }
        inverse = rsqrtf(sumsq / static_cast<float>(features) + eps);
    }
    inverse = __shfl(inverse, 0, 32);
    for (std::uint32_t feature = lane; feature < features; feature += 32U) {
        const float gain = static_cast<float>(weight[feature]) +
            (unit_offset ? 1.0F : 0.0F);
        output[base + feature] = hip_bfloat16(
            static_cast<float>(input[base + feature]) * inverse * gain);
    }
}

__global__ __launch_bounds__(kBlock) void rmsnorm_k5120_rows4_cta_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    float eps, bool unit_offset) {
    constexpr std::uint32_t features = 5120U;
    constexpr std::uint32_t wave_size = 32U;
    constexpr std::uint32_t waves = kBlock / wave_size;
    const std::uint32_t lane = threadIdx.x & (wave_size - 1U);
    const std::uint32_t wave = threadIdx.x / wave_size;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * features;

    float sumsq = 0.0F;
#pragma unroll
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += kBlock) {
        const float value = static_cast<float>(input[base + feature]);
        sumsq = fmaf(value, value, sumsq);
    }
#pragma unroll
    for (std::uint32_t width = wave_size / 2U; width != 0U; width >>= 1U) {
        sumsq += __shfl_down(sumsq, width, wave_size);
    }

    __shared__ float wave_sums[waves];
    if (lane == 0U) wave_sums[wave] = sumsq;
    __syncthreads();
    if (wave == 0U) {
        float block_sum = lane < waves ? wave_sums[lane] : 0.0F;
#pragma unroll
        for (std::uint32_t width = wave_size / 2U; width != 0U; width >>= 1U) {
            block_sum += __shfl_down(block_sum, width, wave_size);
        }
        if (lane == 0U) {
            wave_sums[0] = rsqrtf(block_sum / static_cast<float>(features) + eps);
        }
    }
    __syncthreads();
    const float inverse = wave_sums[0];
#pragma unroll
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += kBlock) {
        const float gain = static_cast<float>(weight[feature]) +
            (unit_offset ? 1.0F : 0.0F);
        output[base + feature] = hip_bfloat16(
            static_cast<float>(input[base + feature]) * inverse * gain);
    }
}

__global__ __launch_bounds__(kBlock) void rmsnorm_k256_token8_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset) {
    constexpr std::uint32_t features = 256U;
    constexpr std::uint32_t tokens_per_block = kBlock / 32U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t wave = threadIdx.x >> 5U;
    const std::uint32_t row = blockIdx.x * tokens_per_block + wave;
    if (row >= rows) return;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    float inverse = 0.0F;
    if (lane == 0U) {
        float sumsq = 0.0F;
        for (std::uint32_t vector = 0; vector < features / 8U; ++vector) {
            uint4 packed{};
            __builtin_memcpy(&packed, input + base + static_cast<std::size_t>(vector) * 8U,
                             sizeof(packed));
            const float x0 = static_cast<float>(bf16_from_raw(packed.x, false));
            const float x1 = static_cast<float>(bf16_from_raw(packed.x, true));
            const float x2 = static_cast<float>(bf16_from_raw(packed.y, false));
            const float x3 = static_cast<float>(bf16_from_raw(packed.y, true));
            const float x4 = static_cast<float>(bf16_from_raw(packed.z, false));
            const float x5 = static_cast<float>(bf16_from_raw(packed.z, true));
            const float x6 = static_cast<float>(bf16_from_raw(packed.w, false));
            const float x7 = static_cast<float>(bf16_from_raw(packed.w, true));
            sumsq = fmaf(x0, x0, sumsq);
            sumsq = fmaf(x1, x1, sumsq);
            sumsq = fmaf(x2, x2, sumsq);
            sumsq = fmaf(x3, x3, sumsq);
            sumsq = fmaf(x4, x4, sumsq);
            sumsq = fmaf(x5, x5, sumsq);
            sumsq = fmaf(x6, x6, sumsq);
            sumsq = fmaf(x7, x7, sumsq);
        }
        inverse = rsqrtf(sumsq / static_cast<float>(features) + eps);
    }
    inverse = __shfl(inverse, 0, 32);
    for (std::uint32_t feature = lane; feature < features; feature += 32U) {
        const float gain = static_cast<float>(weight[feature]) + (unit_offset ? 1.0F : 0.0F);
        output[base + feature] =
            hip_bfloat16(static_cast<float>(input[base + feature]) * inverse * gain);
    }
}

__global__ void layernorm_kernel(const hip_bfloat16* input, const hip_bfloat16* weight,
                                 const hip_bfloat16* bias, hip_bfloat16* output,
                                 std::uint32_t features, float eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    __shared__ float mean;
    __shared__ float inverse;
    if (threadIdx.x == 0) {
        float sum = 0.0F;
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            sum += static_cast<float>(input[base + feature]);
        }
        mean = sum / static_cast<float>(features);
        float variance = 0.0F;
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            const float centered = static_cast<float>(input[base + feature]) - mean;
            variance = fmaf(centered, centered, variance);
        }
        inverse = rsqrtf(variance / static_cast<float>(features) + eps);
    }
    __syncthreads();
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        const float normalized = (static_cast<float>(input[base + feature]) - mean) * inverse;
        output[base + feature] = hip_bfloat16(normalized * static_cast<float>(weight[feature]) +
                                               static_cast<float>(bias[feature]));
    }
}

__global__ void l2norm_kernel(const hip_bfloat16* input, hip_bfloat16* output,
                              std::uint32_t features, float eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    __shared__ float inverse;
    if (threadIdx.x == 0) {
        float sumsq = 0.0F;
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            const float value = static_cast<float>(input[base + feature]);
            sumsq = fmaf(value, value, sumsq);
        }
        inverse = rsqrtf(sumsq + eps);
    }
    __syncthreads();
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        output[base + feature] = hip_bfloat16(static_cast<float>(input[base + feature]) * inverse);
    }
}

__global__ void l2norm_dump_kernel(const hip_bfloat16* input, hip_bfloat16* output,
                                   std::uint32_t features, float eps,
                                   float* sumsq_output, float* inverse_output) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    __shared__ float inverse;
    if (threadIdx.x == 0) {
        float sumsq = 0.0F;
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            const float value = static_cast<float>(input[base + feature]);
            sumsq = fmaf(value, value, sumsq);
        }
        inverse = rsqrtf(sumsq + eps);
        sumsq_output[row] = sumsq;
        inverse_output[row] = inverse;
    }
    __syncthreads();
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        output[base + feature] = hip_bfloat16(static_cast<float>(input[base + feature]) * inverse);
    }
}

__global__ void gated_rmsnorm_kernel(const hip_bfloat16* input,
                                     const hip_bfloat16* weight,
                                     const hip_bfloat16* gate,
                                     hip_bfloat16* output,
                                     std::uint32_t features, float eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    __shared__ float inverse;
    if (threadIdx.x == 0) {
        float sumsq = 0.0F;
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            const float value = static_cast<float>(input[base + feature]);
            sumsq = fmaf(value, value, sumsq);
        }
        inverse = rsqrtf(sumsq / static_cast<float>(features) + eps);
    }
    __syncthreads();
    for (std::uint32_t feature = threadIdx.x; feature < features; feature += blockDim.x) {
        const float z = static_cast<float>(gate[base + feature]);
        const float silu = z / (1.0F + expf(-z));
        output[base + feature] = hip_bfloat16(
            static_cast<float>(input[base + feature]) * inverse *
            static_cast<float>(weight[feature]) * silu);
    }
}

__global__ __launch_bounds__(kBlock) void gated_rmsnorm_k6144_token8_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* gate,
    hip_bfloat16* output, std::uint32_t rows, float eps) {
    constexpr std::uint32_t features = 6144;
    constexpr std::uint32_t tokens_per_block = kBlock / 32U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t wave = threadIdx.x >> 5U;
    const std::uint32_t row = blockIdx.x * tokens_per_block + wave;
    if (row >= rows) { return; }
    const std::size_t base = static_cast<std::size_t>(row) * features;
    float inverse = 0.0F;
    if (lane == 0U) {
        float sumsq = 0.0F;
        for (std::uint32_t vector = 0; vector < features / 8U; ++vector) {
            uint4 packed{};
            __builtin_memcpy(&packed, input + base + static_cast<std::size_t>(vector) * 8U,
                             sizeof(packed));
            const float x0 = static_cast<float>(bf16_from_raw(packed.x, false));
            const float x1 = static_cast<float>(bf16_from_raw(packed.x, true));
            const float x2 = static_cast<float>(bf16_from_raw(packed.y, false));
            const float x3 = static_cast<float>(bf16_from_raw(packed.y, true));
            const float x4 = static_cast<float>(bf16_from_raw(packed.z, false));
            const float x5 = static_cast<float>(bf16_from_raw(packed.z, true));
            const float x6 = static_cast<float>(bf16_from_raw(packed.w, false));
            const float x7 = static_cast<float>(bf16_from_raw(packed.w, true));
            sumsq = fmaf(x0, x0, sumsq);
            sumsq = fmaf(x1, x1, sumsq);
            sumsq = fmaf(x2, x2, sumsq);
            sumsq = fmaf(x3, x3, sumsq);
            sumsq = fmaf(x4, x4, sumsq);
            sumsq = fmaf(x5, x5, sumsq);
            sumsq = fmaf(x6, x6, sumsq);
            sumsq = fmaf(x7, x7, sumsq);
        }
        inverse = rsqrtf(sumsq / static_cast<float>(features) + eps);
    }
    inverse = __shfl(inverse, 0, 32);
    for (std::uint32_t feature = lane; feature < features; feature += 32U) {
        const float z = static_cast<float>(gate[base + feature]);
        const float silu = z / (1.0F + expf(-z));
        output[base + feature] = hip_bfloat16(
            static_cast<float>(input[base + feature]) * inverse *
            static_cast<float>(weight[feature]) * silu);
    }
}

__global__ __launch_bounds__(kBlock) void gated_rmsnorm_k128_rows8_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* gate,
    hip_bfloat16* output, float eps) {
    constexpr std::uint32_t features = 128U;
    constexpr std::uint32_t rows_per_block = kBlock / 32U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t wave = threadIdx.x >> 5U;
    const std::uint32_t row = blockIdx.x * rows_per_block + wave;
    const std::size_t base = static_cast<std::size_t>(row) * features;
    float inverse = 0.0F;
    if (lane == 0U) {
        float sumsq = 0.0F;
        for (std::uint32_t vector = 0U; vector < features / 8U; ++vector) {
            uint4 packed{};
            __builtin_memcpy(&packed, input + base + static_cast<std::size_t>(vector) * 8U,
                             sizeof(packed));
            const float x0 = static_cast<float>(bf16_from_raw(packed.x, false));
            const float x1 = static_cast<float>(bf16_from_raw(packed.x, true));
            const float x2 = static_cast<float>(bf16_from_raw(packed.y, false));
            const float x3 = static_cast<float>(bf16_from_raw(packed.y, true));
            const float x4 = static_cast<float>(bf16_from_raw(packed.z, false));
            const float x5 = static_cast<float>(bf16_from_raw(packed.z, true));
            const float x6 = static_cast<float>(bf16_from_raw(packed.w, false));
            const float x7 = static_cast<float>(bf16_from_raw(packed.w, true));
            sumsq = fmaf(x0, x0, sumsq);
            sumsq = fmaf(x1, x1, sumsq);
            sumsq = fmaf(x2, x2, sumsq);
            sumsq = fmaf(x3, x3, sumsq);
            sumsq = fmaf(x4, x4, sumsq);
            sumsq = fmaf(x5, x5, sumsq);
            sumsq = fmaf(x6, x6, sumsq);
            sumsq = fmaf(x7, x7, sumsq);
        }
        inverse = rsqrtf(sumsq / static_cast<float>(features) + eps);
    }
    inverse = __shfl(inverse, 0, 32);
#pragma unroll
    for (std::uint32_t item = 0U; item < features / 32U; ++item) {
        const std::uint32_t feature = lane + item * 32U;
        const float z = static_cast<float>(gate[base + feature]);
        const float silu = z / (1.0F + expf(-z));
        output[base + feature] = hip_bfloat16(
            static_cast<float>(input[base + feature]) * inverse *
            static_cast<float>(weight[feature]) * silu);
    }
}

__global__ void gelu_kernel(hip_bfloat16* x, std::size_t count, GeluMode mode) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const float value = static_cast<float>(x[index]);
    const float result = mode == GeluMode::Exact
        ? 0.5F * value * (1.0F + erff(value * 0.7071067811865475244F))
        : 0.5F * value * (1.0F + tanhf(0.7978845608028653559F *
                                        (value + 0.044715F * value * value * value)));
    x[index] = hip_bfloat16(result);
}

__global__ void silu_mul_kernel(const hip_bfloat16* gate, const hip_bfloat16* up,
                                hip_bfloat16* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const float value = static_cast<float>(gate[index]);
    output[index] = hip_bfloat16((value / (1.0F + expf(-value))) * static_cast<float>(up[index]));
}

__global__ void silu_mul_strided_kernel(const hip_bfloat16* gate, const hip_bfloat16* up,
                                        hip_bfloat16* output, std::size_t count,
                                        SiluMulStridedShape shape) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    std::size_t coordinate = index;
    std::uint64_t gate_offset = 0;
    std::uint64_t up_offset = 0;
#pragma unroll
    for (unsigned dimension = 0; dimension < 4U; ++dimension) {
        const std::uint32_t axis = static_cast<std::uint32_t>(
            coordinate % shape.extent[dimension]);
        coordinate /= shape.extent[dimension];
        gate_offset += static_cast<std::uint64_t>(axis) * shape.gate_stride[dimension];
        up_offset += static_cast<std::uint64_t>(axis) * shape.up_stride[dimension];
    }
    const float value = static_cast<float>(gate[gate_offset]);
    output[index] = hip_bfloat16((value / (1.0F + expf(-value))) *
                                 static_cast<float>(up[up_offset]));
}

__global__ __launch_bounds__(kBlock) void silu_mul_split17408_2d_kernel(
    const hip_bfloat16* gate, const hip_bfloat16* up, hip_bfloat16* output) {
    constexpr std::uint32_t features = 17408U;
    constexpr std::uint32_t packed_stride = 34816U;
    const std::uint32_t feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= features) return;
    const std::uint32_t row = blockIdx.y;
    const std::size_t input_offset = static_cast<std::size_t>(row) * packed_stride + feature;
    const std::size_t output_offset = static_cast<std::size_t>(row) * features + feature;
    const float value = static_cast<float>(gate[input_offset]);
    output[output_offset] = hip_bfloat16(
        (value / (1.0F + expf(-value))) * static_cast<float>(up[input_offset]));
}

__global__ void sigmoid_mul_kernel(const hip_bfloat16* gate, hip_bfloat16* x,
                                   std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const float value = static_cast<float>(gate[index]);
    x[index] = hip_bfloat16(static_cast<float>(x[index]) / (1.0F + expf(-value)));
}

__global__ void embedding_gather_kernel(const hip_bfloat16* table, const std::int32_t* ids,
                                        hip_bfloat16* output, std::uint32_t features,
                                        std::uint32_t rows) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = static_cast<std::size_t>(features) * rows;
    if (index >= elements) { return; }
    const std::uint32_t row = static_cast<std::uint32_t>(index / features);
    const std::uint32_t feature = static_cast<std::uint32_t>(index % features);
    output[index] = table[static_cast<std::size_t>(ids[row]) * features + feature];
}

__device__ int unpack_q6(const std::uint8_t* low, const std::uint8_t* high,
                         std::uint32_t lane) {
    const std::uint8_t low_byte = low[lane >> 1U];
    const std::uint32_t low_bits = (lane & 1U) != 0U ? low_byte >> 4U : low_byte & 0x0fU;
    const std::uint32_t high_position = lane * 2U;
    const std::uint32_t high_bits =
        (high[high_position >> 3U] >> (high_position & 7U)) & 0x03U;
    const std::uint32_t code = low_bits | (high_bits << 4U);
    return (code & 0x20U) != 0U ? static_cast<int>(code) - 64 : static_cast<int>(code);
}

__global__ void embedding_gather_q6_kernel(const std::uint8_t* low_codes,
                                           const std::uint8_t* high_codes,
                                           const std::uint16_t* scales,
                                           const std::int32_t* ids,
                                           hip_bfloat16* output,
                                           std::uint32_t features,
                                           std::uint32_t padded_features,
                                           std::size_t elements) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    constexpr std::uint32_t group_size = 64;
    constexpr std::uint32_t low_bytes = 32;
    constexpr std::uint32_t high_bytes = 16;
    const std::uint32_t token = static_cast<std::uint32_t>(index / features);
    const std::uint32_t feature = static_cast<std::uint32_t>(index % features);
    const std::uint32_t groups_per_row = padded_features / group_size;
    const std::uint32_t group = feature / group_size;
    const std::uint32_t lane = feature % group_size;
    const std::size_t group_index =
        static_cast<std::size_t>(ids[token]) * groups_per_row + group;
    const int code = unpack_q6(low_codes + group_index * low_bytes,
                               high_codes + group_index * high_bytes, lane);
    const float scale = __half2float(__ushort_as_half(scales[group_index]));
    output[index] = hip_bfloat16(static_cast<float>(code) * scale);
}

__global__ void embedding_gather_q4_kernel(const std::uint8_t* codes,
                                           const std::uint16_t* scales,
                                           const std::int32_t* ids,
                                           hip_bfloat16* output,
                                           std::uint32_t features,
                                           std::uint32_t padded_features,
                                           std::size_t elements) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    constexpr std::uint32_t group_size = 64;
    const std::uint32_t token = static_cast<std::uint32_t>(index / features);
    const std::uint32_t feature = static_cast<std::uint32_t>(index % features);
    const std::uint32_t groups_per_row = padded_features / group_size;
    const std::size_t row = static_cast<std::size_t>(ids[token]);
    const std::uint32_t group = feature / group_size;
    const std::uint32_t pair = (feature % group_size) / 16U;
    const std::uint32_t lane = feature % 16U;
    const std::size_t code_word =
        (((row >> 4U) * groups_per_row + group) * 4U + pair) * 16U + (row & 15U);
    const std::uint8_t packed = codes[code_word * 8U + lane / 2U];
    const std::uint32_t nibble = (feature & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
    const int code = (nibble & 0x08U) != 0U ? static_cast<int>(nibble) - 16
                                            : static_cast<int>(nibble);
    const std::size_t group_index =
        ((row >> 4U) * groups_per_row + group) * 16U + (row & 15U);
    const float scale = __half2float(__ushort_as_half(scales[group_index]));
    output[index] = hip_bfloat16(static_cast<float>(code) * scale);
}

__global__ void embedding_gather_w8_kernel(const std::uint8_t* codes,
                                           const std::uint16_t* scales,
                                           const std::int32_t* ids,
                                           hip_bfloat16* output,
                                           std::uint32_t features,
                                           std::uint32_t padded_features,
                                           std::size_t elements) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    constexpr std::uint32_t group_size = 32;
    const std::uint32_t token = static_cast<std::uint32_t>(index / features);
    const std::uint32_t feature = static_cast<std::uint32_t>(index % features);
    const std::uint32_t groups_per_row = padded_features / group_size;
    const std::size_t row = static_cast<std::size_t>(ids[token]);
    const std::size_t group_index = row * groups_per_row + feature / group_size;
    const auto code = static_cast<std::int8_t>(codes[row * padded_features + feature]);
    const float scale = __half2float(__ushort_as_half(scales[group_index]));
    output[index] = hip_bfloat16(static_cast<float>(code) * scale);
}

__device__ bool argmax_better(float value, std::int32_t index, float current,
                               std::int32_t current_index) {
    return value > current || (value == current && index < current_index);
}

__global__ void argmax_bf16_kernel(const hip_bfloat16* logits, std::int32_t* output,
                                   std::uint32_t physical_rows, std::uint32_t valid_rows) {
    const std::uint32_t column = blockIdx.x;
    float best = -INFINITY;
    std::int32_t best_index = INT_MAX;
    const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
    for (std::uint32_t row = threadIdx.x; row < valid_rows; row += blockDim.x) {
        const float value = static_cast<float>(logits[base + row]);
        if (argmax_better(value, static_cast<std::int32_t>(row), best, best_index)) {
            best = value;
            best_index = static_cast<std::int32_t>(row);
        }
    }
    __shared__ float values[kBlock];
    __shared__ std::int32_t indices[kBlock];
    values[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (std::uint32_t stride = kBlock / 2U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride && argmax_better(values[threadIdx.x + stride],
                                                   indices[threadIdx.x + stride],
                                                   values[threadIdx.x], indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { output[column] = indices[0]; }
}

template <typename Launch>
[[nodiscard]] hipError_t checked_launch(bool valid, Launch&& launch) noexcept {
    if (!valid) { return hipErrorInvalidValue; }
    launch();
    return hipGetLastError();
}

} // namespace

hipError_t i32_iota(std::int32_t* destination, std::size_t count, std::int32_t start,
                    hipStream_t stream) noexcept {
    const bool no_overflow = start >= 0 &&
        count <= static_cast<std::size_t>(INT_MAX - start) + 1U;
    return checked_launch(destination != nullptr && valid_linear_count(count) && valid_stream(stream) &&
                              no_overflow,
                          [&] { hipLaunchKernelGGL(iota_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, destination, count, start); });
}

hipError_t i32_fill(std::int32_t* destination, std::size_t count, std::int32_t value,
                    hipStream_t stream) noexcept {
    return checked_launch(destination != nullptr && valid_linear_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL((fill_kernel<std::int32_t>), linear_grid(count),
                                                   dim3(kBlock), 0, stream, destination, count, value); });
}

hipError_t i32_copy(const std::int32_t* source, std::int32_t* destination, std::size_t count,
                    hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && destination != nullptr &&
                              valid_linear_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL((copy_kernel<std::int32_t>), linear_grid(count),
                                                   dim3(kBlock), 0, stream, source, destination, count); });
}

hipError_t i32_offset(const std::int32_t* source, std::int32_t* destination, std::size_t count,
                      std::int32_t delta, hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && destination != nullptr &&
                              valid_linear_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(offset_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, source, destination, count, delta); });
}

hipError_t i32_offset_scalar(const std::int32_t* source, const std::int32_t* delta,
                             std::int32_t* destination, std::size_t count,
                             hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && delta != nullptr && destination != nullptr &&
                              valid_linear_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(offset_scalar_kernel, linear_grid(count),
                                                   dim3(kBlock), 0, stream, source, delta,
                                                   destination, count); });
}

hipError_t i32_add_scalars(const std::int32_t* lhs, const std::int32_t* rhs,
                           std::int32_t* destination, hipStream_t stream) noexcept {
    return checked_launch(lhs != nullptr && rhs != nullptr && destination != nullptr &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(add_i32_scalars_kernel, dim3(1), dim3(1), 0,
                                                   stream, lhs, rhs, destination); });
}

hipError_t i32_increment(std::int32_t* scalar, hipStream_t stream) noexcept {
    return checked_launch(scalar != nullptr && valid_stream(stream),
                          [&] { hipLaunchKernelGGL((increment_kernel<std::int32_t>), dim3(1),
                                                   dim3(1), 0, stream, scalar); });
}

hipError_t i64_increment(std::int64_t* scalar, hipStream_t stream) noexcept {
    return checked_launch(scalar != nullptr && valid_stream(stream),
                          [&] { hipLaunchKernelGGL((increment_kernel<std::int64_t>), dim3(1),
                                                   dim3(1), 0, stream, scalar); });
}

hipError_t i32_gather(const std::int32_t* source, std::size_t source_count,
                      const std::int32_t* indices, std::int32_t* destination,
                      std::size_t count, hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && indices != nullptr && destination != nullptr &&
                              source_count != 0 && valid_linear_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(i32_gather_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, source, indices, destination, count); });
}

hipError_t bf16_scatter_columns(const hip_bfloat16* source, const std::int32_t* indices,
                                hip_bfloat16* destination, std::uint32_t feature_count,
                                std::uint32_t count, std::uint32_t destination_columns,
                                hipStream_t stream) noexcept {
    const std::size_t elements = static_cast<std::size_t>(feature_count) * count;
    return checked_launch(source != nullptr && indices != nullptr && destination != nullptr &&
                              feature_count != 0 && count != 0 && destination_columns != 0 &&
                              valid_linear_count(elements) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(bf16_scatter_kernel, linear_grid(elements), dim3(kBlock), 0,
                                                   stream, source, indices, destination, feature_count,
                                                   count); });
}

hipError_t bf16_gather_columns(const hip_bfloat16* source, const std::int32_t* indices,
                               hip_bfloat16* destination, std::uint32_t feature_count,
                               std::uint32_t count, std::uint32_t source_columns,
                               hipStream_t stream) noexcept {
    const std::size_t elements = static_cast<std::size_t>(feature_count) * count;
    return checked_launch(source != nullptr && indices != nullptr && destination != nullptr &&
                              feature_count != 0 && count != 0 && source_columns != 0 &&
                              valid_linear_count(elements) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(bf16_gather_kernel, linear_grid(elements), dim3(kBlock), 0,
                                                   stream, source, indices, destination, feature_count,
                                                   count); });
}

hipError_t bf16_scatter_batch(const hip_bfloat16* source, const std::int32_t* lanes,
                              const std::int32_t* valid_columns,
                              hip_bfloat16* destination, std::uint32_t features,
                              std::uint32_t width, std::uint32_t batch,
                              std::size_t destination_column_stride,
                              std::size_t destination_lane_stride,
                              hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && lanes != nullptr && valid_columns != nullptr &&
                              destination != nullptr && features != 0 && width != 0 && batch != 0 &&
                              destination_column_stride >= features &&
                              destination_lane_stride >=
                                  destination_column_stride * static_cast<std::size_t>(width) &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(bf16_scatter_batch_kernel,
                                                   dim3(width, batch), dim3(kBlock), 0, stream,
                                                   source, lanes, valid_columns, destination,
                                                   features, width, destination_column_stride,
                                                   destination_lane_stride); });
}

hipError_t bf16_gather_path(hip_bfloat16* features, const std::int32_t* lanes,
                            const std::int32_t* path, const std::int32_t* counts,
                            std::uint32_t feature_count, std::uint32_t width,
                            std::uint32_t batch, std::uint32_t lane_count,
                            hipStream_t stream) noexcept {
    return checked_launch(features != nullptr && lanes != nullptr && path != nullptr &&
                              counts != nullptr && feature_count != 0 && width != 0 && batch != 0 &&
                              lane_count != 0 && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(bf16_gather_path_kernel, dim3(batch),
                                                   dim3(kBlock), 0, stream, features, lanes, path,
                                                   counts, feature_count, width); });
}

hipError_t bf16_extract_features(const hip_bfloat16* source, hip_bfloat16* destination,
                                 std::uint32_t source_features,
                                 std::uint32_t source_feature,
                                 std::uint32_t destination_features,
                                 std::uint32_t columns,
                                 hipStream_t stream) noexcept {
    const std::size_t elements = static_cast<std::size_t>(destination_features) * columns;
    return checked_launch(source != nullptr && destination != nullptr && source_features != 0 &&
                              destination_features != 0 && columns != 0 &&
                              source_feature <= source_features &&
                              destination_features <= source_features - source_feature &&
                              valid_linear_count(elements) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(bf16_extract_features_kernel,
                                                   linear_grid(elements), dim3(kBlock), 0, stream,
                                                   source, destination, source_features,
                                                   source_feature, destination_features,
                                                   elements); });
}

hipError_t cast_fp32_to_bf16(const float* source, hip_bfloat16* destination, std::size_t count,
                             hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && destination != nullptr && valid_linear_count(count) &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(cast_fp32_to_bf16_kernel, linear_grid(count),
                                                   dim3(kBlock), 0, stream, source, destination, count); });
}

hipError_t cast_bf16_to_fp32(const hip_bfloat16* source, float* destination, std::size_t count,
                             hipStream_t stream) noexcept {
    return checked_launch(source != nullptr && destination != nullptr && valid_linear_count(count) &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(cast_bf16_to_fp32_kernel, linear_grid(count),
                                                   dim3(kBlock), 0, stream, source, destination, count); });
}

hipError_t residual_add_bf16(const hip_bfloat16* y, hip_bfloat16* x, std::size_t count,
                             hipStream_t stream) noexcept {
    return checked_launch(y != nullptr && x != nullptr && valid_linear_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(residual_add_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, y, x, count); });
}

hipError_t residual_rmsnorm_bf16(const hip_bfloat16* y, hip_bfloat16* x,
                                  const hip_bfloat16* weight, hip_bfloat16* output,
                                  std::uint32_t features, std::uint32_t rows, float eps,
                                  hipStream_t stream) noexcept {
    return checked_launch(y != nullptr && x != nullptr && weight != nullptr && output != nullptr &&
                              features != 0 && rows != 0 && std::isfinite(eps) && eps > 0.0F &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(residual_rmsnorm_kernel, dim3(rows),
                                                   dim3(kBlock), 0, stream, y, x, weight, output,
                                                   features, eps); });
}

hipError_t rmsnorm_bf16(const hip_bfloat16* input, const hip_bfloat16* weight,
                        hip_bfloat16* output, std::uint32_t features, std::uint32_t rows,
                        float eps, bool unit_offset, hipStream_t stream) noexcept {
    return checked_launch(
        input != nullptr && weight != nullptr && output != nullptr && features != 0 && rows != 0 &&
            std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
        [&] {
            if (rmsnorm_k5120_rows4_selected(features, rows) ||
                use_rmsnorm_k5120_rows56_candidate(features, rows)) {
                hipLaunchKernelGGL(rmsnorm_k5120_rows4_cta_kernel, dim3(rows), dim3(kBlock), 0,
                                   stream, input, weight, output, eps, unit_offset);
                return;
            }
            if (features == 5120U && rows >= 128U) {
                constexpr std::uint32_t tokens_per_block = kBlock / 32U;
                hipLaunchKernelGGL(rmsnorm_k5120_token8_kernel,
                                   dim3((rows + tokens_per_block - 1U) / tokens_per_block),
                                   dim3(kBlock), 0, stream, input, weight, output, rows, eps,
                                   unit_offset);
            } else if (rmsnorm_k256_token8_selected(features, rows)) {
                constexpr std::uint32_t tokens_per_block = kBlock / 32U;
                hipLaunchKernelGGL(rmsnorm_k256_token8_kernel,
                                   dim3((rows + tokens_per_block - 1U) / tokens_per_block),
                                   dim3(kBlock), 0, stream, input, weight, output, rows, eps,
                                   unit_offset);
            } else {
                hipLaunchKernelGGL(rmsnorm_kernel, dim3(rows), dim3(kBlock), 0, stream,
                                   input, weight, output, features, eps, unit_offset);
            }
        });
}

hipError_t rmsnorm_k5120_token8_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset, hipStream_t stream) noexcept {
    constexpr std::uint32_t tokens_per_block = kBlock / 32U;
    return checked_launch(
        input != nullptr && weight != nullptr && output != nullptr && rows != 0 &&
            std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(rmsnorm_k5120_token8_kernel,
                               dim3((rows + tokens_per_block - 1U) / tokens_per_block),
                               dim3(kBlock), 0, stream, input, weight, output, rows, eps,
                               unit_offset);
        });
}

hipError_t rmsnorm_incumbent_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t features, std::uint32_t rows, float eps, bool unit_offset,
    hipStream_t stream) noexcept {
    return checked_launch(
        input != nullptr && weight != nullptr && output != nullptr && features != 0 && rows != 0 &&
            std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(rmsnorm_kernel, dim3(rows), dim3(kBlock), 0, stream,
                               input, weight, output, features, eps, unit_offset);
        });
}

hipError_t rmsnorm_k5120_rows56_cta_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset, hipStream_t stream) noexcept {
    const auto overlaps = [](const hip_bfloat16* lhs, std::size_t lhs_count,
                             const hip_bfloat16* rhs, std::size_t rhs_count) {
        const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
        const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
        const auto lhs_bytes = lhs_count * sizeof(hip_bfloat16);
        const auto rhs_bytes = rhs_count * sizeof(hip_bfloat16);
        return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
    };
    const std::size_t elements = static_cast<std::size_t>(rows) * 5120U;
    return checked_launch(
        input != nullptr && weight != nullptr && output != nullptr &&
            !overlaps(input, elements, output, elements) &&
            !overlaps(weight, 5120U, output, elements) && (rows == 5U || rows == 6U) &&
            std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(rmsnorm_k5120_rows4_cta_kernel, dim3(rows), dim3(kBlock), 0,
                               stream, input, weight, output, eps, unit_offset);
        });
}

hipError_t rmsnorm_k256_token8_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, hip_bfloat16* output,
    std::uint32_t rows, float eps, bool unit_offset, hipStream_t stream) noexcept {
    constexpr std::uint32_t tokens_per_block = kBlock / 32U;
    return checked_launch(
        input != nullptr && weight != nullptr && output != nullptr && rows != 0U &&
            std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(rmsnorm_k256_token8_kernel,
                               dim3((rows + tokens_per_block - 1U) / tokens_per_block),
                               dim3(kBlock), 0, stream, input, weight, output, rows, eps,
                               unit_offset);
        });
}

hipError_t layernorm_bf16(const hip_bfloat16* input, const hip_bfloat16* weight,
                          const hip_bfloat16* bias, hip_bfloat16* output,
                          std::uint32_t features, std::uint32_t rows, float eps,
                          hipStream_t stream) noexcept {
    return checked_launch(input != nullptr && weight != nullptr && bias != nullptr && output != nullptr &&
                              features != 0 && rows != 0 && std::isfinite(eps) && eps > 0.0F &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(layernorm_kernel, dim3(rows), dim3(kBlock), 0, stream,
                                                   input, weight, bias, output, features, eps); });
}

hipError_t l2norm_bf16(const hip_bfloat16* input, hip_bfloat16* output,
                       std::uint32_t features, std::uint32_t rows, float eps,
                       hipStream_t stream) noexcept {
    return checked_launch(input != nullptr && output != nullptr && features != 0 && rows != 0 &&
                              std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(l2norm_kernel, dim3(rows), dim3(kBlock), 0, stream,
                                                   input, output, features, eps); });
}

hipError_t l2norm_bf16_dump(const hip_bfloat16* input, hip_bfloat16* output,
                            std::uint32_t features, std::uint32_t rows, float eps,
                            float* sumsq, float* inverse, hipStream_t stream) noexcept {
    return checked_launch(input != nullptr && output != nullptr && sumsq != nullptr &&
                              inverse != nullptr && features != 0 && rows != 0 &&
                              std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(l2norm_dump_kernel, dim3(rows), dim3(kBlock),
                                                   0, stream, input, output, features, eps,
                                                   sumsq, inverse); });
}

hipError_t gated_rmsnorm_bf16(const hip_bfloat16* input, const hip_bfloat16* weight,
                              const hip_bfloat16* gate, hip_bfloat16* output,
                              std::uint32_t features, std::uint32_t rows, float eps,
                              hipStream_t stream) noexcept {
    return checked_launch(input != nullptr && weight != nullptr && gate != nullptr &&
                              output != nullptr && features != 0 && rows != 0 &&
                              std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
                          [&] {
                              if (gated_rmsnorm_k128_rows8_selected(features, rows)) {
                                  constexpr std::uint32_t rows_per_block = kBlock / 32U;
                                  hipLaunchKernelGGL(
                                      gated_rmsnorm_k128_rows8_kernel,
                                      dim3(rows / rows_per_block), dim3(kBlock), 0, stream,
                                      input, weight, gate, output, eps);
                              } else if (gated_rmsnorm_k6144_token8_selected(features, rows)) {
                                  constexpr std::uint32_t tokens_per_block = kBlock / 32U;
                                  hipLaunchKernelGGL(
                                      gated_rmsnorm_k6144_token8_kernel,
                                      dim3((rows + tokens_per_block - 1U) / tokens_per_block),
                                      dim3(kBlock), 0, stream, input, weight, gate, output,
                                      rows, eps);
                              } else {
                                  hipLaunchKernelGGL(gated_rmsnorm_kernel, dim3(rows),
                                                     dim3(kBlock), 0, stream, input, weight,
                                                     gate, output, features, eps);
                              }
                          });
}

hipError_t gated_rmsnorm_k6144_token8_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* gate,
    hip_bfloat16* output, std::uint32_t rows, float eps, hipStream_t stream) noexcept {
    constexpr std::uint32_t tokens_per_block = kBlock / 32U;
    return checked_launch(
        input != nullptr && weight != nullptr && gate != nullptr && output != nullptr &&
            rows != 0U && std::isfinite(eps) && eps > 0.0F && valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(gated_rmsnorm_k6144_token8_kernel,
                               dim3((rows + tokens_per_block - 1U) / tokens_per_block),
                               dim3(kBlock), 0, stream, input, weight, gate, output, rows, eps);
        });
}

hipError_t gated_rmsnorm_incumbent_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* gate,
    hip_bfloat16* output, std::uint32_t features, std::uint32_t rows, float eps,
    hipStream_t stream) noexcept {
    return checked_launch(
        input != nullptr && weight != nullptr && gate != nullptr && output != nullptr &&
            features != 0U && rows != 0U && std::isfinite(eps) && eps > 0.0F &&
            valid_stream(stream),
        [&] {
            hipLaunchKernelGGL(gated_rmsnorm_kernel, dim3(rows), dim3(kBlock), 0, stream,
                               input, weight, gate, output, features, eps);
        });
}

hipError_t gelu_bf16(hip_bfloat16* x, std::size_t count, GeluMode mode,
                     hipStream_t stream) noexcept {
    return checked_launch(x != nullptr && valid_linear_count(count) && valid_stream(stream) &&
                              (mode == GeluMode::Exact || mode == GeluMode::Tanh),
                          [&] { hipLaunchKernelGGL(gelu_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, x, count, mode); });
}

hipError_t silu_mul_bf16(const hip_bfloat16* gate, const hip_bfloat16* up,
                         hip_bfloat16* output, std::size_t count,
                         hipStream_t stream) noexcept {
    return checked_launch(gate != nullptr && up != nullptr && output != nullptr &&
                              valid_linear_count(count) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(silu_mul_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, gate, up, output, count); });
}

hipError_t silu_mul_bf16_strided(const hip_bfloat16* gate, const hip_bfloat16* up,
                                 hip_bfloat16* output, std::size_t count,
                                 SiluMulStridedShape shape,
                                 hipStream_t stream) noexcept {
    std::size_t represented_count = 1;
    std::uint64_t gate_max_offset = 0;
    std::uint64_t up_max_offset = 0;
    bool valid = gate != nullptr && up != nullptr && output != nullptr &&
                 valid_linear_count(count) && valid_stream(stream);
    for (unsigned dimension = 0; dimension < 4U; ++dimension) {
        const std::size_t extent = shape.extent[dimension];
        const std::uint64_t tail = extent == 0U ? 0U : extent - 1U;
        valid = valid && extent != 0U && shape.gate_stride[dimension] != 0U &&
                shape.up_stride[dimension] != 0U &&
                represented_count <= std::numeric_limits<std::size_t>::max() / extent &&
                (tail == 0U ||
                 shape.gate_stride[dimension] <=
                     (std::numeric_limits<std::uint64_t>::max() - gate_max_offset) / tail) &&
                (tail == 0U ||
                 shape.up_stride[dimension] <=
                     (std::numeric_limits<std::uint64_t>::max() - up_max_offset) / tail);
        if (valid) {
            represented_count *= extent;
            gate_max_offset += tail * shape.gate_stride[dimension];
            up_max_offset += tail * shape.up_stride[dimension];
        }
    }
    constexpr std::uint64_t kMaxElementOffset =
        std::numeric_limits<std::size_t>::max() / sizeof(hip_bfloat16);
    valid = valid && represented_count == count && gate_max_offset <= kMaxElementOffset &&
            up_max_offset <= kMaxElementOffset;
    return checked_launch(valid, [&] {
        if (silu_mul_split17408_2d_selected(count, shape)) {
            constexpr std::uint32_t features = 17408U;
            hipLaunchKernelGGL(silu_mul_split17408_2d_kernel,
                               dim3((features + kBlock - 1U) / kBlock, shape.extent[1]),
                               dim3(kBlock), 0, stream, gate, up, output);
        } else {
            hipLaunchKernelGGL(silu_mul_strided_kernel, linear_grid(count), dim3(kBlock), 0,
                               stream, gate, up, output, count, shape);
        }
    });
}

hipError_t silu_mul_split17408_2d_qualification(
    const hip_bfloat16* gate, const hip_bfloat16* up, hip_bfloat16* output,
    std::uint32_t rows, hipStream_t stream) noexcept {
    constexpr std::uint32_t features = 17408U;
    return checked_launch(gate != nullptr && up != nullptr && output != nullptr && rows != 0U &&
                              valid_stream(stream),
                          [&] {
                              hipLaunchKernelGGL(
                                  silu_mul_split17408_2d_kernel,
                                  dim3((features + kBlock - 1U) / kBlock, rows), dim3(kBlock),
                                  0, stream, gate, up, output);
                          });
}

hipError_t silu_mul_split17408_incumbent_qualification(
    const hip_bfloat16* gate, const hip_bfloat16* up, hip_bfloat16* output,
    std::uint32_t rows, hipStream_t stream) noexcept {
    constexpr std::uint32_t features = 17408U;
    constexpr std::uint32_t packed_stride = 34816U;
    SiluMulStridedShape shape{};
    shape.extent[0] = features;
    shape.extent[1] = rows;
    shape.gate_stride[0] = 1U;
    shape.gate_stride[1] = packed_stride;
    shape.up_stride[0] = 1U;
    shape.up_stride[1] = packed_stride;
    return silu_mul_bf16_strided(gate, up, output,
                                 static_cast<std::size_t>(features) * rows, shape, stream);
}

hipError_t sigmoid_mul_bf16(const hip_bfloat16* gate, hip_bfloat16* x, std::size_t count,
                            hipStream_t stream) noexcept {
    return checked_launch(gate != nullptr && x != nullptr && valid_linear_count(count) &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(sigmoid_mul_kernel, linear_grid(count), dim3(kBlock), 0,
                                                   stream, gate, x, count); });
}

hipError_t embedding_gather_bf16(const hip_bfloat16* table, const std::int32_t* ids,
                                 hip_bfloat16* output, std::uint32_t vocabulary,
                                 std::uint32_t features, std::uint32_t rows,
                                 hipStream_t stream) noexcept {
    const std::size_t elements = static_cast<std::size_t>(features) * rows;
    return checked_launch(table != nullptr && ids != nullptr && output != nullptr && vocabulary != 0 &&
                              features != 0 && rows != 0 && valid_linear_count(elements) &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(embedding_gather_kernel, linear_grid(elements),
                                                   dim3(kBlock), 0, stream, table, ids, output,
                                                   features, rows); });
}

hipError_t embedding_gather_q6g64_f16s(
    const std::uint8_t* low_codes, const std::uint8_t* high_codes,
    const std::uint16_t* scales, const std::int32_t* ids, hip_bfloat16* output,
    std::uint32_t vocabulary, std::uint32_t features, std::uint32_t padded_features,
    std::uint32_t rows, hipStream_t stream) noexcept {
    const std::size_t elements = static_cast<std::size_t>(features) * rows;
    return checked_launch(low_codes != nullptr && high_codes != nullptr && scales != nullptr &&
                              ids != nullptr && output != nullptr && vocabulary != 0 &&
                              features != 0 && padded_features >= features &&
                              padded_features % 64U == 0U && rows != 0 &&
                              valid_linear_count(elements) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(embedding_gather_q6_kernel,
                                                   linear_grid(elements), dim3(kBlock), 0, stream,
                                                   low_codes, high_codes, scales, ids, output,
                                                   features, padded_features, elements); });
}

hipError_t embedding_gather_q4g64_f16s(
    const std::uint8_t* codes, const std::uint16_t* scales,
    const std::int32_t* ids, hip_bfloat16* output, std::uint32_t vocabulary,
    std::uint32_t features, std::uint32_t padded_features, std::uint32_t rows,
    hipStream_t stream) noexcept {
    const std::size_t elements = static_cast<std::size_t>(features) * rows;
    return checked_launch(codes != nullptr && scales != nullptr && ids != nullptr &&
                              output != nullptr && vocabulary != 0 && features != 0 &&
                              padded_features >= features && padded_features % 128U == 0U &&
                              vocabulary % 16U == 0U && rows != 0 &&
                              valid_linear_count(elements) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(embedding_gather_q4_kernel,
                                                   linear_grid(elements), dim3(kBlock), 0, stream,
                                                   codes, scales, ids, output, features,
                                                   padded_features, elements); });
}

hipError_t embedding_gather_w8g32_f16s(
    const std::uint8_t* codes, const std::uint16_t* scales,
    const std::int32_t* ids, hip_bfloat16* output, std::uint32_t vocabulary,
    std::uint32_t features, std::uint32_t padded_features, std::uint32_t rows,
    hipStream_t stream) noexcept {
    const std::size_t elements = static_cast<std::size_t>(features) * rows;
    return checked_launch(codes != nullptr && scales != nullptr && ids != nullptr &&
                              output != nullptr && vocabulary != 0 && features != 0 &&
                              padded_features >= features && padded_features % 32U == 0U &&
                              rows != 0 && valid_linear_count(elements) && valid_stream(stream),
                          [&] { hipLaunchKernelGGL(embedding_gather_w8_kernel,
                                                   linear_grid(elements), dim3(kBlock), 0, stream,
                                                   codes, scales, ids, output, features,
                                                   padded_features, elements); });
}

hipError_t argmax_bf16(const hip_bfloat16* logits, std::int32_t* out,
                       std::uint32_t physical_rows, std::uint32_t valid_rows,
                       std::uint32_t columns, hipStream_t stream) noexcept {
    return checked_launch(logits != nullptr && out != nullptr && physical_rows != 0 &&
                              valid_rows != 0 && valid_rows <= physical_rows && columns != 0 &&
                              valid_stream(stream),
                          [&] { hipLaunchKernelGGL(argmax_bf16_kernel, dim3(columns), dim3(kBlock), 0,
                                                   stream, logits, out, physical_rows, valid_rows); });
}

hipError_t greedy_select_bf16(const hip_bfloat16* logits, std::int32_t* out,
                              std::uint32_t physical_rows, std::uint32_t valid_rows,
                              std::uint32_t columns, hipStream_t stream) noexcept {
    return argmax_bf16(logits, out, physical_rows, valid_rows, columns, stream);
}

} // namespace ninfer::ops::r9700::eager
