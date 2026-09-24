#include "fp8_gate_up_small_t_qual.h"

#include <limits>

namespace ninfer::qualification::fp8_gate_up_small_t {
namespace {

using I32x2 = int __attribute__((ext_vector_type(2)));
using F32x8 = float __attribute__((ext_vector_type(8)));

constexpr std::uint16_t kPoison = 0x7fc0U;

__device__ __forceinline__ __attribute__((unused)) I32x2 load_fp8x8(
    const std::uint8_t* pointer) {
    I32x2 value{};
    __builtin_memcpy(&value, pointer, sizeof(value));
    return value;
}

// Four waves each own four N16 fragments. The single active M16 fragment is padded in
// registers, so T5/T6 do not pay the incumbent's general small-M tile machinery. Four
// independent accumulators hide the dependent FP8-WMMA chain without LDS or scratch.
__global__ __launch_bounds__(kThreads)
void fp8_gate_up_n34816_k5120_t5t6_wmma_kernel(
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ token_scales,
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    hip_bfloat16* __restrict__ output,
    std::uint32_t tokens) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t wave = threadIdx.x >> 5U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t row_base = blockIdx.x * kRowsPerBlock + wave * kRowsPerWave;
    F32x8 totals[4]{};

#pragma unroll 1
    for (std::uint32_t k_base = 0; k_base < kColumns; k_base += 16U) {
        I32x2 activation{};
        if (axis < tokens) {
            activation = load_fp8x8(
                activation_codes + static_cast<std::size_t>(axis) * kColumns + k_base +
                lane_group * 8U);
        }
#pragma unroll
        for (std::uint32_t fragment = 0; fragment < 4U; ++fragment) {
            const std::uint32_t row = row_base + fragment * 16U + axis;
            const I32x2 weight = load_fp8x8(
                weight_codes + static_cast<std::size_t>(row) * kColumns + k_base +
                lane_group * 8U);
            totals[fragment] =
                __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    activation, weight, totals[fragment]);
        }
    }

#pragma unroll
    for (std::uint32_t fragment = 0; fragment < 4U; ++fragment) {
        const std::uint32_t row = row_base + fragment * 16U + axis;
        const float row_scale = weight_scales[row];
#pragma unroll
        for (std::uint32_t element = 0; element < 8U; ++element) {
            const std::uint32_t token = lane_group * 8U + element;
            if (token < tokens) {
                output[static_cast<std::size_t>(token) * kRows + row] =
                    hip_bfloat16((totals[fragment][element] * token_scales[token]) * row_scale);
            }
        }
    }
#else
    (void)activation_codes;
    (void)token_scales;
    (void)weight_codes;
    (void)weight_scales;
    (void)output;
    (void)tokens;
#endif
}

__global__ void poison_nonfinite(const std::uint32_t* status, hip_bfloat16* output,
                                 std::size_t elements) {
    if (*status == ops::r9700::linear::Fp8ActivationOk) return;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        reinterpret_cast<std::uint16_t*>(output)[index] = kPoison;
    }
}

bool checked_bytes(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (result == nullptr || (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool disjoint(const void* lhs, std::size_t lhs_bytes, const void* rhs,
              std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0U || rhs_bytes == 0U) return false;
    const auto a = reinterpret_cast<std::uintptr_t>(lhs);
    const auto b = reinterpret_cast<std::uintptr_t>(rhs);
    if (a > std::numeric_limits<std::uintptr_t>::max() - lhs_bytes ||
        b > std::numeric_limits<std::uintptr_t>::max() - rhs_bytes) return false;
    return a + lhs_bytes <= b || b + rhs_bytes <= a;
}

bool valid(const Arguments& a) {
    std::size_t input_elements = 0, input_bytes = 0, weight_bytes = 0;
    std::size_t output_elements = 0, output_bytes = 0;
    const bool shape_and_sizes = (a.tokens == 5U || a.tokens == 6U) && a.rows == kRows &&
           a.columns == kColumns && a.input_bf16 != nullptr &&
           a.weight_codes != nullptr && a.weight_scales != nullptr &&
           a.output_bf16 != nullptr && a.activation.tokens == a.tokens &&
           a.activation.columns == kColumns && a.activation.padded_columns == kColumns &&
           a.activation.codes != nullptr &&
           a.activation.code_bytes == static_cast<std::size_t>(a.tokens) * kColumns &&
           a.activation.scales != nullptr &&
           a.activation.scale_bytes == static_cast<std::size_t>(a.tokens) * sizeof(float) &&
           a.activation.status != nullptr &&
           checked_bytes(a.tokens, kColumns, &input_elements) &&
           checked_bytes(input_elements, sizeof(hip_bfloat16), &input_bytes) &&
           checked_bytes(kRows, kColumns, &weight_bytes) &&
           checked_bytes(a.tokens, kRows, &output_elements) &&
           checked_bytes(output_elements, sizeof(hip_bfloat16), &output_bytes);
    if (!shape_and_sizes) return false;
    struct Region { const void* pointer; std::size_t bytes; };
    const Region regions[] = {
        {a.input_bf16, input_bytes},
        {a.weight_codes, weight_bytes},
        {a.weight_scales, static_cast<std::size_t>(kRows) * sizeof(float)},
        {a.activation.codes, a.activation.code_bytes},
        {a.activation.scales, a.activation.scale_bytes},
        {a.activation.status, sizeof(std::uint32_t)},
        {a.output_bf16, output_bytes},
    };
    constexpr std::size_t kRegionCount = sizeof(regions) / sizeof(regions[0]);
    for (std::size_t lhs = 0; lhs < kRegionCount; ++lhs) {
        for (std::size_t rhs = lhs + 1U; rhs < kRegionCount; ++rhs) {
            if (!disjoint(regions[lhs].pointer, regions[lhs].bytes,
                          regions[rhs].pointer, regions[rhs].bytes)) return false;
        }
    }
    return true;
}

} // namespace

hipError_t launch(const Arguments& a, hipStream_t stream) noexcept {
    if (!valid(a)) return hipErrorInvalidValue;
    hipError_t status = ops::r9700::linear::fp8_quantize_activation(
        ops::r9700::linear::Fp8ActivationQuantizeArgs{a.input_bf16, a.activation}, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(fp8_gate_up_n34816_k5120_t5t6_wmma_kernel,
                       dim3(kRows / kRowsPerBlock), dim3(kThreads), 0, stream,
                       a.activation.codes, a.activation.scales, a.weight_codes,
                       a.weight_scales, a.output_bf16, a.tokens);
    status = hipGetLastError();
    if (status != hipSuccess) return status;
    const std::size_t elements = static_cast<std::size_t>(a.tokens) * kRows;
    hipLaunchKernelGGL(poison_nonfinite, dim3((elements + 255U) / 256U), dim3(256), 0,
                       stream, a.activation.status, a.output_bf16, elements);
    return hipGetLastError();
}

hipError_t resources(Resources* output) noexcept {
    if (output == nullptr) return hipErrorInvalidValue;
    hipFuncAttributes attributes{};
    hipError_t status = hipFuncGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(fp8_gate_up_n34816_k5120_t5t6_wmma_kernel));
    if (status != hipSuccess) return status;
    int blocks = 0;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, fp8_gate_up_n34816_k5120_t5t6_wmma_kernel, kThreads, 0);
    if (status != hipSuccess) return status;
    *output = Resources{attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
                        static_cast<int>(attributes.localSizeBytes),
                        attributes.maxThreadsPerBlock, blocks};
    return hipSuccess;
}

} // namespace ninfer::qualification::fp8_gate_up_small_t
