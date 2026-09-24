#include "w8_tiled_head.h"
#include <hip/hip_fp16.h>
#include <array>
#include <limits>

namespace ninfer::ops::r9700::linear {
namespace {
constexpr unsigned N = 248320, K = 5120, G = K / 32;
using I2 = int __attribute__((ext_vector_type(2)));
using I8 = int __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ std::size_t code_index(unsigned row, unsigned k) {
    return (((static_cast<std::size_t>(row / 16) * G + k / 32) * 2 +
             (k % 32) / 16) * 16 + row % 16) * 16 + k % 16;
}
__device__ __forceinline__ std::size_t scale_index(unsigned row, unsigned group) {
    return (static_cast<std::size_t>(row / 16) * G + group) * 16 + row % 16;
}
__device__ __forceinline__ I2 load8(const std::int8_t* p, bool valid) {
    I2 result{};
    if (valid) {
        const auto* words = reinterpret_cast<const std::int32_t*>(p);
        result[0] = words[0]; result[1] = words[1];
    }
    return result;
}

__global__ __launch_bounds__(32) void a8w8_tiled_head_kernel(
    const std::int8_t* a, const std::uint16_t* as, const std::uint32_t* status,
    const std::int8_t* b, const std::uint16_t* bs, hip_bfloat16* output, unsigned tokens) {
    const unsigned lane = threadIdx.x, axis = lane & 15U, half = lane >> 4U;
    const unsigned row = blockIdx.x * 16 + axis, tile_token = blockIdx.y * 16;
    const unsigned token_base = tile_token + half * 8;
    if (*status != A8G32ActivationOk) {
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) if (token_base + e < tokens) {
            hip_bfloat16 value; value.data = 0x7fc1;
            output[static_cast<std::size_t>(token_base + e) * N + row] = value;
        }
        return;
    }
    F8 total{};
    for (unsigned g = 0; g < G; ++g) {
        I8 dot{};
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile) {
            const unsigned token = tile_token + axis;
            const auto av = load8(a + (token < tokens ?
                static_cast<std::size_t>(token) * K + g * 32 + tile * 16 + half * 8 : 0), token < tokens);
            const auto bv = load8(b + code_index(row, g * 32 + tile * 16 + half * 8), true);
            dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av, true, bv, dot, false);
        }
        const float ws = __half2float(__ushort_as_half(bs[scale_index(row, g)]));
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned token = token_base + e;
            const float scale = token < tokens ? __half2float(__ushort_as_half(
                as[static_cast<std::size_t>(token) * G + g])) : 0.0F;
            total[e] = fmaf(static_cast<float>(dot[e]), scale * ws, total[e]);
        }
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) if (token_base + e < tokens)
        output[static_cast<std::size_t>(token_base + e) * N + row] = hip_bfloat16(total[e]);
}

// Same per-thread ascending K chain and 256-thread reduction as the incumbent
// ordinary exact-W8 baseline; only the resident code/scale addressing differs.
__global__ __launch_bounds__(256) void bf16_w8_tiled_head_baseline_kernel(
    const hip_bfloat16* input, const std::int8_t* codes, const std::uint16_t* scales,
    hip_bfloat16* output) {
    const unsigned row = blockIdx.x, token = blockIdx.y;
    float partial = 0.0F;
    for (unsigned k = threadIdx.x; k < K; k += 256) {
        const float weight = static_cast<float>(codes[code_index(row, k)]) *
            __half2float(__ushort_as_half(scales[scale_index(row, k / 32)]));
        partial = fmaf(static_cast<float>(input[static_cast<std::size_t>(token) * K + k]), weight, partial);
    }
    __shared__ float reductions[256];
    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned width = 128; width; width >>= 1) {
        if (threadIdx.x < width) reductions[threadIdx.x] += reductions[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[static_cast<std::size_t>(token) * N + row] = hip_bfloat16(reductions[0]);
}

// A bijection of [logical partial][row] that makes adjacent physical half-waves
// address distinct LDS banks. No padding or change to reduction association.
__device__ __forceinline__ unsigned partial_index(unsigned logical, unsigned row) {
    return (logical ^ ((logical >> 4U) & 1U)) * 16U + row;
}

__global__ __launch_bounds__(256) void bf16_w8_tiled_head_coalesced_kernel(
    const hip_bfloat16* input, const std::int8_t* codes, const std::uint16_t* scales,
    hip_bfloat16* output) {
    const unsigned axis = threadIdx.x & 15U, subgroup = threadIdx.x >> 4U;
    const unsigned row = blockIdx.x * 16U + axis, token = blockIdx.y;
    float partial[16]{};
    // Each register j is exactly incumbent logical thread subgroup*16+j.
    // Every chain retains ascending k, the FP32 weight multiply, then FP32 FMA.
    for (unsigned base = 0; base < K; base += 256U) {
        const unsigned k = base + subgroup * 16U;
        const auto* words = reinterpret_cast<const std::uint32_t*>(codes + code_index(row, k));
        const float scale = __half2float(__ushort_as_half(scales[scale_index(row, k / 32U)]));
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            const std::uint32_t packed = words[word]; // only four-byte alignment required
#pragma unroll
            for (unsigned byte = 0; byte < 4; ++byte) {
                const unsigned j = word * 4U + byte;
                const auto code = static_cast<std::int8_t>(packed >> (byte * 8U));
                const float weight = static_cast<float>(code) * scale;
                partial[j] = fmaf(static_cast<float>(input[static_cast<std::size_t>(token) * K + k + j]),
                                  weight, partial[j]);
            }
        }
    }
    __shared__ float reductions[16 * 256];
#pragma unroll
    for (unsigned j = 0; j < 16; ++j)
        reductions[partial_index(subgroup * 16U + j, axis)] = partial[j];
    __syncthreads();
#pragma unroll
    for (unsigned width = 128; width >= 16; width >>= 1) {
        if (subgroup * 16U < width) {
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                partial[j] += reductions[partial_index(subgroup * 16U + j + width, axis)];
                reductions[partial_index(subgroup * 16U + j, axis)] = partial[j];
            }
        }
        __syncthreads();
    }
    if (subgroup == 0) {
#pragma unroll
        for (unsigned width = 8; width; width >>= 1) {
#pragma unroll
            for (unsigned j = 0; j < width; ++j) partial[j] += partial[j + width];
        }
        output[static_cast<std::size_t>(token) * N + row] = hip_bfloat16(partial[0]);
    }
}

// A16 control profile at larger T: keep the existing wave-per-token ascending-K
// chain, FP32 dequantization and shuffle reduction. Only 256 threads stage each
// 256-element tile even when the prefill CTA contains sixteen consumer waves.
template <unsigned Waves>
__global__ __launch_bounds__(Waves * 32) void bf16_w8_tiled_head_wave_kernel(
    const hip_bfloat16* input, const std::int8_t* codes, const std::uint16_t* scales,
    hip_bfloat16* output, unsigned tokens) {
    const unsigned row = blockIdx.x, lane = threadIdx.x & 31U;
    const unsigned token = blockIdx.y * Waves + threadIdx.x / 32;
    __shared__ float weights[256];
    float partial = 0.0F;
    for (unsigned base = 0; base < K; base += 256) {
        if (threadIdx.x < 256) {
            const unsigned k = base + threadIdx.x;
            weights[threadIdx.x] = static_cast<float>(codes[code_index(row, k)]) *
                __half2float(__ushort_as_half(scales[scale_index(row, k / 32)]));
        }
        __syncthreads();
        if (token < tokens) {
            const auto offset = static_cast<std::size_t>(token) * K + base;
            for (unsigned local = lane; local < 256; local += 32)
                partial = fmaf(static_cast<float>(input[offset + local]), weights[local], partial);
        }
        __syncthreads();
    }
    for (unsigned delta = 16; delta; delta >>= 1) partial += __shfl_down(partial, delta, 32);
    if (token < tokens && lane == 0)
        output[static_cast<std::size_t>(token) * N + row] = hip_bfloat16(partial);
}

struct Plane { const void* pointer; std::size_t bytes, alignment; };
template <std::size_t Count>
bool valid_planes(const std::array<Plane, Count>& planes, hipStream_t stream) {
    for (std::size_t i = 0; i < Count; ++i) {
        const auto p = reinterpret_cast<std::uintptr_t>(planes[i].pointer);
        if (!p || p % planes[i].alignment ||
            p > std::numeric_limits<std::uintptr_t>::max() - planes[i].bytes) return false;
        for (std::size_t j = 0; j < i; ++j) {
            const auto q = reinterpret_cast<std::uintptr_t>(planes[j].pointer);
            if (p < q + planes[j].bytes && q < p + planes[i].bytes) return false;
        }
    }
    unsigned flags = 0;
    return !stream || hipStreamGetFlags(stream, &flags) == hipSuccess;
}
bool shape(unsigned tokens, unsigned rows, unsigned columns, unsigned padded,
           std::size_t codes, std::size_t scales) {
    return tokens > 0 && rows == N && columns == K && padded == K &&
        codes == static_cast<std::size_t>(N) * K && scales == static_cast<std::size_t>(N) * G * 2;
}
}

hipError_t a8w8g32_tiled_head(const A8W8G32CandidateArgs& a, hipStream_t stream) noexcept {
    const auto bytes = a8w8g32_activation_workspace_capacity_bytes(a.tokens, K);
    if (!shape(a.tokens, a.rows, a.columns, a.padded_columns, a.weight_code_bytes, a.weight_scale_bytes) ||
        !bytes || a.activation_workspace_bytes != bytes ||
        !valid_planes(std::array<Plane, 5>{{
            {a.input, static_cast<std::size_t>(a.tokens) * K * 2, 2},
            {a.weight_codes, a.weight_code_bytes, 4}, {a.weight_scales, a.weight_scale_bytes, 2},
            {a.activation_workspace, bytes, 256},
            {a.output, static_cast<std::size_t>(a.tokens) * N * 2, 2}}}, stream)) return hipErrorInvalidValue;
    A8G32ActivationWorkspace scratch{};
    auto result = a8w8g32_bind_activation_workspace(a.activation_workspace, bytes, a.tokens, K, &scratch);
    if (result != hipSuccess) return result;
    result = a8g32_quantize_activation({a.input, scratch}, stream);
    if (result != hipSuccess) return result;
    hipLaunchKernelGGL(a8w8_tiled_head_kernel, dim3(N / 16, (a.tokens - 1) / 16 + 1), dim3(32), 0, stream,
        scratch.codes, scratch.scales, scratch.status, a.weight_codes, a.weight_scales, a.output, a.tokens);
    return hipGetLastError();
}

hipError_t w8g32_tiled_head(const W8G32LinearArgs& a, hipStream_t stream) noexcept {
    if (!shape(a.tokens, a.rows, a.columns, a.padded_columns, a.code_bytes, a.scale_bytes) ||
        !valid_planes(std::array<Plane, 4>{{
            {a.input, static_cast<std::size_t>(a.tokens) * K * 2, 2},
            {a.codes, a.code_bytes, 4}, {a.scales, a.scale_bytes, 2},
            {a.output, static_cast<std::size_t>(a.tokens) * N * 2, 2}}}, stream)) return hipErrorInvalidValue;
    if (a.tokens <= 3) {
        hipLaunchKernelGGL(bf16_w8_tiled_head_coalesced_kernel, dim3(N / 16, a.tokens), dim3(256), 0,
                          stream, a.input, a.codes, a.scales, a.output);
    } else if (a.tokens == 4) {
        hipLaunchKernelGGL(bf16_w8_tiled_head_baseline_kernel, dim3(N, a.tokens), dim3(256), 0,
                          stream, a.input, a.codes, a.scales, a.output);
    } else if (a.tokens <= 8) {
        hipLaunchKernelGGL(bf16_w8_tiled_head_wave_kernel<8>, dim3(N, (a.tokens + 7) / 8), dim3(256), 0,
                          stream, a.input, a.codes, a.scales, a.output, a.tokens);
    } else {
        hipLaunchKernelGGL(bf16_w8_tiled_head_wave_kernel<16>, dim3(N, (a.tokens - 1) / 16 + 1), dim3(512), 0,
                          stream, a.input, a.codes, a.scales, a.output, a.tokens);
    }
    return hipGetLastError();
}

}
