#include "ops/r9700/linear/fp8_activation.h"

#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::r9700::linear {
namespace {

constexpr std::uint32_t kWaveSize        = 32U;
constexpr std::uint32_t kBlockSize       = 256U;
constexpr std::uint32_t kWavesPerBlock   = kBlockSize / kWaveSize;
constexpr std::uint32_t kColumnAlignment = 128U;
constexpr std::size_t kPlaneAlignment    = 256U;
constexpr float kE4M3FiniteMaximum       = 448.0F;

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* out) noexcept {
    if (out == nullptr || (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* out) noexcept {
    if (out == nullptr || rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t* out) noexcept {
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) return false;
    *out = (value + alignment - 1U) / alignment * alignment;
    return true;
}

bool padded_columns(std::uint32_t columns, std::uint32_t* out) noexcept {
    if (out == nullptr || columns == 0U ||
        columns > std::numeric_limits<std::uint32_t>::max() - (kColumnAlignment - 1U)) {
        return false;
    }
    *out = (columns + kColumnAlignment - 1U) / kColumnAlignment * kColumnAlignment;
    return true;
}

struct WorkspaceGeometry {
    std::uint32_t padded      = 0;
    std::size_t code_bytes    = 0;
    std::size_t scale_offset = 0;
    std::size_t scale_bytes   = 0;
    std::size_t status_offset = 0;
    std::size_t total_bytes   = 0;
};

bool workspace_geometry(std::uint32_t tokens, std::uint32_t columns,
                        WorkspaceGeometry* out) noexcept {
    if (out == nullptr || tokens == 0U) return false;
    WorkspaceGeometry result;
    if (!padded_columns(columns, &result.padded) ||
        !checked_mul(tokens, result.padded, &result.code_bytes) ||
        !align_up(result.code_bytes, kPlaneAlignment, &result.scale_offset) ||
        !checked_mul(tokens, sizeof(float), &result.scale_bytes)) {
        return false;
    }
    std::size_t scale_end = 0;
    if (!checked_add(result.scale_offset, result.scale_bytes, &scale_end) ||
        !align_up(scale_end, kPlaneAlignment, &result.status_offset) ||
        !checked_add(result.status_offset, sizeof(std::uint32_t), &result.total_bytes)) {
        return false;
    }
    *out = result;
    return true;
}

bool range(const void* pointer, std::size_t bytes, std::uintptr_t* begin,
           std::uintptr_t* end) noexcept {
    if (pointer == nullptr || bytes == 0U || begin == nullptr || end == nullptr) return false;
    *begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (*begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    *end = *begin + bytes;
    return true;
}

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes, const void* rhs,
                    std::size_t rhs_bytes) noexcept {
    std::uintptr_t lhs_begin = 0, lhs_end = 0, rhs_begin = 0, rhs_end = 0;
    if (!range(lhs, lhs_bytes, &lhs_begin, &lhs_end) ||
        !range(rhs, rhs_bytes, &rhs_begin, &rhs_end)) {
        return true;
    }
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool valid_workspace(const Fp8ActivationWorkspace& workspace) noexcept {
    WorkspaceGeometry geometry;
    return workspace_geometry(workspace.tokens, workspace.columns, &geometry) &&
           workspace.padded_columns == geometry.padded &&
           workspace.code_bytes == geometry.code_bytes &&
           workspace.scale_bytes == geometry.scale_bytes && workspace.codes != nullptr &&
           workspace.scales != nullptr && workspace.status != nullptr &&
           reinterpret_cast<std::uintptr_t>(workspace.codes) % kPlaneAlignment == 0U &&
           reinterpret_cast<std::uintptr_t>(workspace.scales) % alignof(float) == 0U &&
           reinterpret_cast<std::uintptr_t>(workspace.status) % alignof(std::uint32_t) == 0U &&
           !ranges_overlap(workspace.codes, workspace.code_bytes, workspace.scales,
                           workspace.scale_bytes) &&
           !ranges_overlap(workspace.codes, workspace.code_bytes, workspace.status,
                           sizeof(std::uint32_t)) &&
           !ranges_overlap(workspace.scales, workspace.scale_bytes, workspace.status,
                           sizeof(std::uint32_t));
}

__global__ __launch_bounds__(kBlockSize) void fp8_quantize_activation_kernel(
    const hip_bfloat16* input, std::uint8_t* codes, float* scales, std::uint32_t* status,
    std::uint32_t columns, std::uint32_t padded) {
    const std::uint32_t token = blockIdx.x;
    const std::uint32_t lane = threadIdx.x % kWaveSize;
    const std::uint32_t wave = threadIdx.x / kWaveSize;
    float local_maximum = 0.0F;
    unsigned local_bad  = 0U;
    for (std::uint32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = static_cast<float>(
            input[static_cast<std::size_t>(token) * columns + column]);
        local_bad |= static_cast<unsigned>(!isfinite(value));
        if (isfinite(value)) local_maximum = fmaxf(local_maximum, fabsf(value));
    }
#pragma unroll
    for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
        local_maximum = fmaxf(local_maximum, __shfl_down(local_maximum, delta, kWaveSize));
        local_bad |= __shfl_down(local_bad, delta, kWaveSize);
    }

    __shared__ float wave_maxima[kWavesPerBlock];
    __shared__ unsigned wave_bad[kWavesPerBlock];
    __shared__ float token_scale;
    __shared__ unsigned token_bad;
    if (lane == 0U) {
        wave_maxima[wave] = local_maximum;
        wave_bad[wave] = local_bad;
    }
    __syncthreads();

    if (wave == 0U) {
        float maximum = lane < kWavesPerBlock ? wave_maxima[lane] : 0.0F;
        unsigned bad  = lane < kWavesPerBlock ? wave_bad[lane] : 0U;
#pragma unroll
        for (std::uint32_t delta = kWaveSize / 2U; delta != 0U; delta >>= 1U) {
            maximum = fmaxf(maximum, __shfl_down(maximum, delta, kWaveSize));
            bad |= __shfl_down(bad, delta, kWaveSize);
        }
        if (lane == 0U) {
            token_bad = bad != 0U;
            token_scale = bad != 0U ? 0.0F : maximum / kE4M3FiniteMaximum;
            scales[token] = token_scale;
            if (bad != 0U) {
                atomicOr(status, static_cast<std::uint32_t>(Fp8ActivationNonfinite));
            }
        }
    }
    __syncthreads();

    for (std::uint32_t column = threadIdx.x; column < padded; column += blockDim.x) {
        std::uint8_t code = 0U;
        if (token_bad == 0U && token_scale != 0.0F && column < columns) {
            const float value = static_cast<float>(
                input[static_cast<std::size_t>(token) * columns + column]);
            code = static_cast<std::uint8_t>(
                __hip_cvt_float_to_fp8(value / token_scale, __HIP_SATFINITE, __HIP_E4M3));
        }
        codes[static_cast<std::size_t>(token) * padded + column] = code;
    }
}

} // namespace

std::size_t fp8_activation_workspace_capacity_bytes(std::uint32_t tokens,
                                                    std::uint32_t columns) noexcept {
    WorkspaceGeometry geometry;
    return workspace_geometry(tokens, columns, &geometry) ? geometry.total_bytes : 0U;
}

hipError_t fp8_bind_activation_workspace(void* storage, std::size_t storage_bytes,
                                         std::uint32_t tokens, std::uint32_t columns,
                                         Fp8ActivationWorkspace* out) noexcept {
    WorkspaceGeometry geometry;
    if (storage == nullptr || out == nullptr ||
        reinterpret_cast<std::uintptr_t>(storage) % kPlaneAlignment != 0U ||
        !workspace_geometry(tokens, columns, &geometry) ||
        storage_bytes != geometry.total_bytes) {
        return hipErrorInvalidValue;
    }
    auto* bytes = static_cast<std::uint8_t*>(storage);
    *out = Fp8ActivationWorkspace{
        bytes,
        geometry.code_bytes,
        reinterpret_cast<float*>(bytes + geometry.scale_offset),
        geometry.scale_bytes,
        reinterpret_cast<std::uint32_t*>(bytes + geometry.status_offset),
        tokens,
        columns,
        geometry.padded,
    };
    return hipSuccess;
}

hipError_t fp8_quantize_activation(const Fp8ActivationQuantizeArgs& args,
                                   hipStream_t stream) noexcept {
    if (args.input == nullptr || !valid_workspace(args.workspace)) return hipErrorInvalidValue;
    std::size_t input_elements = 0, input_bytes = 0;
    if (!checked_mul(args.workspace.tokens, args.workspace.columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.codes,
                       args.workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.scales,
                       args.workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.status, sizeof(std::uint32_t))) {
        return hipErrorInvalidValue;
    }
    hipError_t result = hipMemsetAsync(args.workspace.status, 0, sizeof(std::uint32_t), stream);
    if (result != hipSuccess) return result;
    hipLaunchKernelGGL(fp8_quantize_activation_kernel, dim3(args.workspace.tokens),
                       dim3(kBlockSize), 0, stream, args.input, args.workspace.codes,
                       args.workspace.scales, args.workspace.status, args.workspace.columns,
                       args.workspace.padded_columns);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::linear
