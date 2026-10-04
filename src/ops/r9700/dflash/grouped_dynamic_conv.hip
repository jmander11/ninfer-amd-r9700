#include "ops/r9700/dflash/grouped_dynamic_conv_launch.h"

#include "core/device.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ops/r9700/linear/a8g64_codec.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using r9700::linear::codec::a8g64_complete_status;
using r9700::linear::codec::a8g64_encode_represented;
using r9700::linear::codec::a8g64_value_status;
using r9700::linear::codec::bf16_word_element;

constexpr int kBlock = 256;
// One 16-byte vector of eight features per thread, one CTA per column.
constexpr std::uint32_t kRowVectors = kGroupedDynamicConvHidden / 8;
constexpr std::uint32_t kWave       = 32U;
constexpr std::uint32_t kWaves      = kRowVectors / kWave;
static_assert(kGroupedDynamicConvGroupSize % 8 == 0);

struct ImagePlanes {
    std::uint8_t* low     = nullptr;
    std::uint8_t* high    = nullptr;
    std::uint16_t* scales = nullptr;
    std::uint32_t* status = nullptr; // null: no image
    // Two words through which the last block publishes the status word (a8g64_complete_status).
    std::uint32_t* completion = nullptr;
};

// Encodes this thread's vector of an image row and publishes the launch's status word. Called
// by every thread of every block when an image is requested.
__device__ __forceinline__ void encode_image(const ImagePlanes& image, const float (&value)[8],
                                             std::int32_t column, std::uint32_t slot) {
    const std::size_t vector = static_cast<std::size_t>(column) * kRowVectors + slot;
    a8g64_encode_represented(value, slot, vector,
                             static_cast<std::size_t>(column) * (kRowVectors / 8U) + slot / 8U,
                             image.low, image.high, image.scales);
    bool nonfinite = false, overflow = false;
    a8g64_value_status(value, nonfinite, overflow);
    a8g64_complete_status(nonfinite, overflow, image.status, image.completion);
}

__device__ __forceinline__ uint4 load_vector(const hip_bfloat16* rows, std::int32_t column,
                                             std::uint32_t slot) {
    return reinterpret_cast<const uint4*>(
        rows)[static_cast<std::size_t>(column) * kRowVectors + slot];
}

// The represented conv values BF16((base[d,0,p] + dynamic_current) * x[d,t] +
// (base[d,1,p] + dynamic_previous) * x[d,t-1]) of one vector, with x[d,-1] = 0 inside each block.
template <int Phase>
__device__ __forceinline__ void
conv_vector(const hip_bfloat16* hidden, const hip_bfloat16* base_kernel, float dynamic_current,
            float dynamic_previous, std::int32_t column, std::int32_t token, std::uint32_t slot,
            float (&value)[8]) {
    const uint4 current_words       = load_vector(hidden, column, slot);
    const uint4 previous_words      = token == 0 ? uint4{} : load_vector(hidden, column - 1, slot);
    const uint4* base               = reinterpret_cast<const uint4*>(base_kernel) +
                                      static_cast<std::size_t>(Phase) * 2U * kRowVectors;
    const uint4 base_current_words  = base[slot];
    const uint4 base_previous_words = base[kRowVectors + slot];
    const std::uint32_t current[4]{current_words.x, current_words.y, current_words.z,
                                   current_words.w};
    const std::uint32_t previous[4]{previous_words.x, previous_words.y, previous_words.z,
                                    previous_words.w};
    const std::uint32_t base_current[4]{base_current_words.x, base_current_words.y,
                                        base_current_words.z, base_current_words.w};
    const std::uint32_t base_previous[4]{base_previous_words.x, base_previous_words.y,
                                         base_previous_words.z, base_previous_words.w};
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        value[element] = static_cast<float>(
            hip_bfloat16((bf16_word_element(base_current, element) + dynamic_current) *
                             bf16_word_element(current, element) +
                         (bf16_word_element(base_previous, element) + dynamic_previous) *
                             bf16_word_element(previous, element)));
    }
}

__device__ __forceinline__ uint4 pack_vector(const float (&value)[8]) {
    std::uint32_t words[4]{};
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        words[element >> 1U] |= static_cast<std::uint32_t>(hip_bfloat16(value[element]).data)
                                << ((element & 1U) * 16U);
    }
    return uint4{words[0], words[1], words[2], words[3]};
}

// Prepare: phase 0 of every column into `prepared` and/or the activation image, and the stashed
// phase-1 dynamic kernels into finish_dynamic.
__global__ __launch_bounds__(kRowVectors) void grouped_dynamic_conv_prepare_kernel(
    const hip_bfloat16* hidden, const hip_bfloat16* base_kernel, const hip_bfloat16* projection,
    hip_bfloat16* prepared, hip_bfloat16* finish_dynamic, ImagePlanes image, std::int32_t tokens) {
    const std::uint32_t slot  = threadIdx.x;
    const std::uint32_t group = slot * 8U / kGroupedDynamicConvGroupSize;
    const auto dynamic        = [&](std::int32_t column, std::int32_t row) {
        return projection[static_cast<std::size_t>(column) * kGroupedDynamicConvProjRows + row];
    };
    const auto column = static_cast<std::int32_t>(blockIdx.x);
    const auto g      = static_cast<std::int32_t>(group);
    float value[8];
    conv_vector<0>(hidden, base_kernel, static_cast<float>(dynamic(column, g)),
                   static_cast<float>(dynamic(column, kGroupedDynamicConvGroups + g)), column,
                   column % tokens, slot, value);
    if (slot * 8U % kGroupedDynamicConvGroupSize == 0U) {
        const std::size_t stash = static_cast<std::size_t>(column) * 2U * kGroupedDynamicConvGroups;
        finish_dynamic[stash + g] = dynamic(column, 2 * kGroupedDynamicConvGroups + g);
        finish_dynamic[stash + kGroupedDynamicConvGroups + g] =
            dynamic(column, 3 * kGroupedDynamicConvGroups + g);
    }
    const std::size_t vector = static_cast<std::size_t>(column) * kRowVectors + slot;
    if (prepared != nullptr) reinterpret_cast<uint4*>(prepared)[vector] = pack_vector(value);
    if (image.status != nullptr) encode_image(image, value, column, slot);
}

// The finished residual of one vector: BF16(residual_in + BF16(phase-1 conv)).
__device__ __forceinline__ void
finished_vector(const hip_bfloat16* hidden, const hip_bfloat16* base_kernel,
                const hip_bfloat16* finish_dynamic, const hip_bfloat16* residual_in,
                std::int32_t column, std::int32_t token, std::uint32_t slot, float (&value)[8]) {
    const std::int32_t group = static_cast<std::int32_t>(slot * 8U / kGroupedDynamicConvGroupSize);
    const std::size_t stash  = static_cast<std::size_t>(column) * 2U * kGroupedDynamicConvGroups;
    float conv[8];
    conv_vector<1>(hidden, base_kernel, static_cast<float>(finish_dynamic[stash + group]),
                   static_cast<float>(finish_dynamic[stash + kGroupedDynamicConvGroups + group]),
                   column, token, slot, conv);
    const uint4 residual_words = load_vector(residual_in, column, slot);
    const std::uint32_t residual[4]{residual_words.x, residual_words.y, residual_words.z,
                                    residual_words.w};
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        value[element] =
            static_cast<float>(hip_bfloat16(bf16_word_element(residual, element) + conv[element]));
    }
}

// This thread's wave-reduced sum of squares (the eager K5120 row-CTA RMSNorm reduction: per-thread
// eight-term FMA chain, wave32 xor tree; wave partials are summed in wave order).
__device__ __forceinline__ float vector_wave_sumsq(const float (&value)[8]) {
    float sumsq = 0.0F;
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element)
        sumsq = fmaf(value[element], value[element], sumsq);
#pragma unroll
    for (std::uint32_t width = kWave / 2U; width != 0U; width >>= 1U)
        sumsq += __shfl_xor(sumsq, width, kWave);
    return sumsq;
}

__device__ __forceinline__ float row_inverse(const float* wave_sums, float eps) {
    float sum = 0.0F;
#pragma unroll
    for (std::uint32_t wave = 0; wave < kWaves; ++wave) sum += wave_sums[wave];
    return rsqrtf(sum / static_cast<float>(kGroupedDynamicConvHidden) + eps);
}

__device__ __forceinline__ void normalized_vector(const float (&residual)[8], const uint4 gains,
                                                  float inverse, float (&value)[8]) {
    const std::uint32_t gain[4]{gains.x, gains.y, gains.z, gains.w};
#pragma unroll
    for (std::uint32_t element = 0; element < 8U; ++element) {
        value[element] = static_cast<float>(
            hip_bfloat16(residual[element] * inverse * bf16_word_element(gain, element)));
    }
}

// Finish fused with the next sublayer's RMSNorm: residual_out = BF16(residual_in + BF16(phase-1
// conv)), normalized = BF16(residual_out * rsqrt(mean(residual_out^2) + eps) * norm), and the
// image of normalized.
__global__ __launch_bounds__(kRowVectors) void grouped_dynamic_conv_finish_normalized_kernel(
    const hip_bfloat16* hidden, const hip_bfloat16* base_kernel, const hip_bfloat16* finish_dynamic,
    const hip_bfloat16* residual_in, hip_bfloat16* residual_out, const hip_bfloat16* norm,
    float eps, hip_bfloat16* normalized, ImagePlanes image, std::int32_t tokens) {
    __shared__ float wave_sums[kWaves];
    const std::uint32_t slot = threadIdx.x;
    const std::uint32_t wave = slot / kWave, lane = slot % kWave;
    const auto column = static_cast<std::int32_t>(blockIdx.x);
    float residual[8];
    finished_vector(hidden, base_kernel, finish_dynamic, residual_in, column, column % tokens, slot,
                    residual);
    const std::size_t vector = static_cast<std::size_t>(column) * kRowVectors + slot;
    reinterpret_cast<uint4*>(residual_out)[vector] = pack_vector(residual);
    const float sumsq                              = vector_wave_sumsq(residual);
    if (lane == 0U) wave_sums[wave] = sumsq;
    __syncthreads();
    float value[8];
    normalized_vector(residual, reinterpret_cast<const uint4*>(norm)[slot],
                      row_inverse(wave_sums, eps), value);
    reinterpret_cast<uint4*>(normalized)[vector] = pack_vector(value);
    if (image.status != nullptr) encode_image(image, value, column, slot);
}

__global__ void grouped_dynamic_conv_finish_kernel(const hip_bfloat16* hidden,
                                                   const hip_bfloat16* base_kernel,
                                                   const hip_bfloat16* finish_dynamic,
                                                   hip_bfloat16* residual, std::int32_t tokens,
                                                   std::int32_t columns) {
    const std::int32_t d      = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.y);
    if (d >= kGroupedDynamicConvHidden || column >= columns) return;

    const std::int32_t token = column % tokens;
    const std::int32_t group = d / kGroupedDynamicConvGroupSize;
    const std::int64_t hidden_index =
        static_cast<std::int64_t>(column) * kGroupedDynamicConvHidden + d;
    const float current = static_cast<float>(hidden[hidden_index]);
    const float previous =
        token == 0 ? 0.0F : static_cast<float>(hidden[hidden_index - kGroupedDynamicConvHidden]);
    const std::int64_t phase_base = 2 * kGroupedDynamicConvHidden;
    const float base_current      = static_cast<float>(base_kernel[phase_base + d]);
    const float base_previous =
        static_cast<float>(base_kernel[phase_base + kGroupedDynamicConvHidden + d]);
    const std::int64_t dynamic_base =
        static_cast<std::int64_t>(column) * 2 * kGroupedDynamicConvGroups;
    const float dynamic_current = static_cast<float>(finish_dynamic[dynamic_base + group]);
    const float dynamic_previous =
        static_cast<float>(finish_dynamic[dynamic_base + kGroupedDynamicConvGroups + group]);
    const hip_bfloat16 result((base_current + dynamic_current) * current +
                              (base_previous + dynamic_previous) * previous);
    // Finish adds the rounded result to the residual stream in place.
    residual[hidden_index] =
        hip_bfloat16(static_cast<float>(residual[hidden_index]) + static_cast<float>(result));
}

ImagePlanes image_planes(const r9700::linear::A8G64ActivationWorkspace* image,
                         std::uint32_t* completion) {
    if (image == nullptr) return {};
    return {image->low_codes, image->high_codes, image->scales, image->status, completion};
}

} // namespace

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& base_kernel,
                                         const Tensor& projection, Tensor* prepared,
                                         Tensor& finish_dynamic,
                                         const r9700::linear::A8G64ActivationWorkspace* image,
                                         std::uint32_t* completion, hipStream_t stream) {
    const int columns = hidden.ne[1] * hidden.ne[2];
    grouped_dynamic_conv_prepare_kernel<<<static_cast<unsigned int>(columns), kRowVectors, 0,
                                          stream>>>(
        static_cast<const hip_bfloat16*>(hidden.data),
        static_cast<const hip_bfloat16*>(base_kernel.data),
        static_cast<const hip_bfloat16*>(projection.data),
        prepared != nullptr ? static_cast<hip_bfloat16*>(prepared->data) : nullptr,
        static_cast<hip_bfloat16*>(finish_dynamic.data), image_planes(image, completion),
        hidden.ne[1]);
    HIP_CHECK(hipGetLastError());
}

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& base_kernel,
                                        const Tensor& finish_dynamic, Tensor& residual,
                                        hipStream_t stream) {
    const int columns = hidden.ne[1] * hidden.ne[2];
    const dim3 grid((kGroupedDynamicConvHidden + kBlock - 1) / kBlock,
                    static_cast<unsigned int>(columns));
    grouped_dynamic_conv_finish_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(hidden.data),
        static_cast<const hip_bfloat16*>(base_kernel.data),
        static_cast<const hip_bfloat16*>(finish_dynamic.data),
        static_cast<hip_bfloat16*>(residual.data), hidden.ne[1], columns);
    HIP_CHECK(hipGetLastError());
}

void grouped_dynamic_conv_finish_normalized_launch(
    const Tensor& hidden, const Tensor& base_kernel, const Tensor& finish_dynamic,
    const Tensor& residual_in, Tensor& residual_out, const Tensor& norm, float eps,
    Tensor& normalized, const r9700::linear::A8G64ActivationWorkspace* image,
    std::uint32_t* completion, hipStream_t stream) {
    const int columns = hidden.ne[1] * hidden.ne[2];
    grouped_dynamic_conv_finish_normalized_kernel<<<static_cast<unsigned int>(columns), kRowVectors,
                                                    0, stream>>>(
        static_cast<const hip_bfloat16*>(hidden.data),
        static_cast<const hip_bfloat16*>(base_kernel.data),
        static_cast<const hip_bfloat16*>(finish_dynamic.data),
        static_cast<const hip_bfloat16*>(residual_in.data),
        static_cast<hip_bfloat16*>(residual_out.data), static_cast<const hip_bfloat16*>(norm.data),
        eps, static_cast<hip_bfloat16*>(normalized.data), image_planes(image, completion),
        hidden.ne[1]);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
