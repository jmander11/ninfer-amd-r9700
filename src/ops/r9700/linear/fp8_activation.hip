#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_activation_kernels.h"

#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::r9700::linear {
namespace fp8_activation_kernels {

// Retain each prefill row across the reduction: BF16 magnitude bits have unsigned
// numeric order, including infinities and NaNs above every finite value.
template <std::uint32_t Columns>
__global__ __launch_bounds__(256) void fp8_quantize_prefill_kernel(
    const hip_bfloat16* input, std::uint8_t* codes, float* scales, std::uint32_t* status,
    unsigned tokens, unsigned padded, CacheWarm warm, unsigned work_ctas) {
    static_assert(Columns == 5120U || Columns == 17408U);
    constexpr unsigned Vectors = Columns / 8U, V = (Vectors + 255U) / 256U;
    __shared__ unsigned maxima[8];
    __shared__ float row_scale;
    __shared__ unsigned row_bad;
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    if (blockIdx.x >= work_ctas) {
        warm_cache(warm, blockIdx.x - work_ctas, gridDim.x - work_ctas, tid, 256U);
        return;
    }
    for (unsigned token = blockIdx.x; token < tokens; token += work_ctas) {
        const auto* row = reinterpret_cast<const uint4*>(input + std::size_t(token) * Columns);
        uint4 retained[V];
        unsigned maximum = 0;
#pragma unroll
        for (unsigned v = 0; v < V; ++v) {
            const unsigned index = v * 256U + tid;
            if (index < Vectors) {
                const uint4 value       = row[index];
                retained[v]             = value;
                const unsigned words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
                for (unsigned w = 0; w < 4; ++w) {
                    maximum = max(maximum, words[w] & 0x7fffU);
                    maximum = max(maximum, (words[w] >> 16U) & 0x7fffU);
                }
            }
        }
#pragma unroll
        for (unsigned d = 16; d; d >>= 1U) maximum = max(maximum, __shfl_down(maximum, d, 32));
        if (lane == 0) maxima[wave] = maximum;
        __syncthreads();
        if (wave == 0) {
            maximum = lane < 8U ? maxima[lane] : 0U;
#pragma unroll
            for (unsigned d = 16; d; d >>= 1U) maximum = max(maximum, __shfl_down(maximum, d, 32));
            if (lane == 0) {
                row_bad       = maximum >= 0x7f80U;
                row_scale     = row_bad ? 0.0F : __uint_as_float(maximum << 16U) / 448.0F;
                scales[token] = row_scale;
                status[token] = row_bad ? Fp8ActivationNonfinite : Fp8ActivationOk;
            }
        }
        __syncthreads();
        const float scale  = row_scale;
        const bool encoded = row_bad == 0U && scale != 0.0F;
#pragma unroll
        for (unsigned v = 0; v < V; ++v) {
            const unsigned index = v * 256U + tid;
            if (index < Vectors) {
                unsigned packed[2] = {0, 0};
                if (encoded) {
                    const uint4 value       = retained[v];
                    const unsigned words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
                    for (unsigned w = 0; w < 4; ++w) {
                        const float low     = __uint_as_float(words[w] << 16U),
                                    high    = __uint_as_float(words[w] & 0xffff0000U);
                        const unsigned pair = __hip_cvt_float2_to_fp8x2(
                            float2{low / scale, high / scale}, __HIP_SATFINITE, __HIP_E4M3);
                        packed[w / 2U] |= pair << (16U * (w % 2U));
                    }
                }
                *reinterpret_cast<uint2*>(codes + std::size_t(token) * padded + index * 8U) =
                    uint2{packed[0], packed[1]};
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kBlockSize) void fp8_quantize_activation_kernel(
    const hip_bfloat16* input, std::uint8_t* codes, float* scales, std::uint32_t* status,
    std::uint32_t tokens, std::uint32_t columns, std::uint32_t padded, CacheWarm warm,
    std::uint32_t work_ctas) {
    __shared__ QuantizeShared shared;
    quantize_activation_body(persistent::LaunchCta{}, shared, input, codes, scales, status, tokens,
                             columns, padded, warm, work_ctas);
}

__global__ __launch_bounds__(kNormThreads) void fp8_quantize_normalized_kernel(
    const hip_bfloat16* input, const hip_bfloat16* weight, float eps, bool unit_offset,
    std::uint8_t* codes, float* scales, std::uint32_t* status, hip_bfloat16* normalized,
    CacheWarm warm, std::uint32_t tokens) {
    __shared__ NormalizedShared<kNormThreads> shared;
    quantize_normalized_body<kNormThreads>(persistent::LaunchCta{}, shared, input, weight, eps,
                                           unit_offset, codes, scales, status, normalized, warm,
                                           tokens);
}

__global__ __launch_bounds__(kGatedNormThreads) void fp8_quantize_gated_rmsnorm_kernel(
    const hip_bfloat16* input, const hip_bfloat16* gate, const hip_bfloat16* weight, float eps,
    std::uint8_t* codes, float* scales, std::uint32_t* status, CacheWarm warm,
    std::uint32_t tokens) {
    __shared__ EncodeShared<kGatedNormThreads> shared;
    quantize_gated_rmsnorm_body<kGatedNormThreads>(persistent::LaunchCta{}, shared, input, gate,
                                                   weight, eps, codes, scales, status, warm,
                                                   tokens);
}

} // namespace fp8_activation_kernels

namespace {

using fp8_activation_kernels::fp8_quantize_activation_kernel;
using fp8_activation_kernels::fp8_quantize_gated_kernel;
using fp8_activation_kernels::fp8_quantize_gated_rmsnorm_kernel;
using fp8_activation_kernels::fp8_quantize_normalized_kernel;
using fp8_activation_kernels::kBlockSize;
using fp8_activation_kernels::kGatedNormColumns;
using fp8_activation_kernels::kGatedNormThreads;
using fp8_activation_kernels::kNormFeatures;
using fp8_activation_kernels::kNormThreads;

constexpr std::uint32_t kQuantizeResidentBlocks = 128U;
constexpr std::uint32_t kColumnAlignment        = 128U;
constexpr std::size_t kPlaneAlignment           = 256U;

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
    std::size_t scale_offset  = 0;
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
        !checked_add(result.status_offset, static_cast<std::size_t>(tokens) * sizeof(std::uint32_t),
                     &result.total_bytes)) {
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

std::size_t status_bytes(const Fp8ActivationWorkspace& workspace) noexcept {
    return static_cast<std::size_t>(workspace.tokens) * sizeof(std::uint32_t);
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
                           status_bytes(workspace)) &&
           !ranges_overlap(workspace.scales, workspace.scale_bytes, workspace.status,
                           status_bytes(workspace));
}

constexpr std::uint32_t kGatedColumns = 6144U;

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
        !workspace_geometry(tokens, columns, &geometry) || storage_bytes != geometry.total_bytes) {
        return hipErrorInvalidValue;
    }
    auto* bytes = static_cast<std::uint8_t*>(storage);
    *out        = Fp8ActivationWorkspace{
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
        ranges_overlap(args.input, input_bytes, args.workspace.codes, args.workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.scales,
                       args.workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, args.workspace.status,
                       status_bytes(args.workspace))) {
        return hipErrorInvalidValue;
    }
    if (args.workspace.tokens >= 256U &&
        (args.workspace.columns == 5120U || args.workspace.columns == 17408U) &&
        reinterpret_cast<std::uintptr_t>(args.input) % 16U == 0U) {
        const unsigned blocks = std::min(args.workspace.tokens, kQuantizeResidentBlocks);
        const dim3 grid(blocks + cache_warm_ctas(args.warm, kBlockSize, blocks));
        if (args.workspace.columns == 5120U) {
            hipLaunchKernelGGL((fp8_activation_kernels::fp8_quantize_prefill_kernel<5120U>), grid,
                               dim3(kBlockSize), 0, stream, args.input, args.workspace.codes,
                               args.workspace.scales, args.workspace.status, args.workspace.tokens,
                               args.workspace.padded_columns, args.warm, blocks);
        } else {
            hipLaunchKernelGGL((fp8_activation_kernels::fp8_quantize_prefill_kernel<17408U>), grid,
                               dim3(kBlockSize), 0, stream, args.input, args.workspace.codes,
                               args.workspace.scales, args.workspace.status, args.workspace.tokens,
                               args.workspace.padded_columns, args.warm, blocks);
        }
        return hipGetLastError();
    }
    const std::uint32_t work_ctas = std::min(args.workspace.tokens, kQuantizeResidentBlocks);
    hipLaunchKernelGGL(fp8_quantize_activation_kernel,
                       dim3(work_ctas + cache_warm_ctas(args.warm, kBlockSize, work_ctas)),
                       dim3(kBlockSize), 0, stream, args.input, args.workspace.codes,
                       args.workspace.scales, args.workspace.status, args.workspace.tokens,
                       args.workspace.columns, args.workspace.padded_columns, args.warm, work_ctas);
    return hipGetLastError();
}

hipError_t fp8_quantize_normalized_activation(const Fp8NormalizedQuantizeArgs& args,
                                              hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t input_bytes                 = 0;
    if (args.input == nullptr || args.weight == nullptr || !valid_workspace(workspace) ||
        workspace.columns != kNormFeatures || workspace.padded_columns != kNormFeatures ||
        !std::isfinite(args.eps) || !(args.eps > 0.0F) ||
        reinterpret_cast<std::uintptr_t>(args.input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.weight) % 16U != 0U ||
        !checked_mul(static_cast<std::size_t>(workspace.tokens) * kNormFeatures,
                     sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.status, status_bytes(workspace))) {
        return hipErrorInvalidValue;
    }
    if (args.normalized != nullptr &&
        (reinterpret_cast<std::uintptr_t>(args.normalized) % 16U != 0U ||
         ranges_overlap(args.normalized, input_bytes, args.input, input_bytes) ||
         ranges_overlap(args.normalized, input_bytes, args.weight,
                        kNormFeatures * sizeof(hip_bfloat16)) ||
         ranges_overlap(args.normalized, input_bytes, workspace.codes, workspace.code_bytes) ||
         ranges_overlap(args.normalized, input_bytes, workspace.scales, workspace.scale_bytes) ||
         ranges_overlap(args.normalized, input_bytes, workspace.status, status_bytes(workspace)))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        fp8_quantize_normalized_kernel,
        dim3(workspace.tokens + cache_warm_ctas(args.warm, kNormThreads, workspace.tokens)),
        dim3(kNormThreads), 0, stream, args.input, args.weight, args.eps, args.unit_offset,
        workspace.codes, workspace.scales, workspace.status, args.normalized, args.warm,
        workspace.tokens);
    return hipGetLastError();
}

hipError_t fp8_quantize_gated_activation(const Fp8GatedQuantizeArgs& args,
                                         hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t elements = 0, gate_bytes = 0, attention_bytes = 0;
    if (args.gate == nullptr || args.attention == nullptr || !valid_workspace(workspace) ||
        workspace.columns != kGatedColumns || workspace.padded_columns != kGatedColumns ||
        reinterpret_cast<std::uintptr_t>(args.gate) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.attention) % 16U != 0U ||
        !checked_mul(workspace.tokens, kGatedColumns, &elements) ||
        !checked_mul(elements, sizeof(hip_bfloat16), &gate_bytes) ||
        !checked_mul(elements, sizeof(float), &attention_bytes) ||
        ranges_overlap(args.gate, gate_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.attention, attention_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.gate, gate_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.attention, attention_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.gate, gate_bytes, workspace.status, status_bytes(workspace)) ||
        ranges_overlap(args.attention, attention_bytes, workspace.status,
                       status_bytes(workspace))) {
        return hipErrorInvalidValue;
    }
    constexpr std::uint32_t threads = kGatedColumns / 8U;
    const dim3 grid(workspace.tokens + cache_warm_ctas(args.warm, threads, workspace.tokens));
    const auto kernel = args.tq_unrotate ? fp8_quantize_gated_kernel<threads, true>
                                         : fp8_quantize_gated_kernel<threads, false>;
    hipLaunchKernelGGL(kernel, grid, dim3(threads), 0, stream, args.gate, args.attention,
                       workspace.codes, workspace.scales, workspace.status, args.warm,
                       workspace.tokens);
    return hipGetLastError();
}

hipError_t fp8_quantize_gated_rmsnorm_activation(const Fp8GatedRmsNormQuantizeArgs& args,
                                                 hipStream_t stream) noexcept {
    const Fp8ActivationWorkspace& workspace = args.workspace;
    std::size_t elements = 0, input_bytes = 0;
    if (args.input == nullptr || args.gate == nullptr || args.weight == nullptr ||
        !valid_workspace(workspace) || workspace.columns != kGatedNormColumns ||
        workspace.padded_columns != kGatedNormColumns || !std::isfinite(args.eps) ||
        !(args.eps > 0.0F) || reinterpret_cast<std::uintptr_t>(args.input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.gate) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(args.weight) % 16U != 0U ||
        !checked_mul(workspace.tokens, kGatedNormColumns, &elements) ||
        !checked_mul(elements, sizeof(hip_bfloat16), &input_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.gate, input_bytes, workspace.codes, workspace.code_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.gate, input_bytes, workspace.scales, workspace.scale_bytes) ||
        ranges_overlap(args.input, input_bytes, workspace.status, status_bytes(workspace)) ||
        ranges_overlap(args.gate, input_bytes, workspace.status, status_bytes(workspace))) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        fp8_quantize_gated_rmsnorm_kernel,
        dim3(workspace.tokens + cache_warm_ctas(args.warm, kGatedNormThreads, workspace.tokens)),
        dim3(kGatedNormThreads), 0, stream, args.input, args.gate, args.weight, args.eps,
        workspace.codes, workspace.scales, workspace.status, args.warm, workspace.tokens);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::linear
