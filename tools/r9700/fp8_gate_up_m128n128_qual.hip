#include "fp8_gate_up_m128n128_qual.h"

#include <hip/hip_bfloat16.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::qualification::fp8_gate_up_m128n128 {
namespace {

using U32x2 = std::uint32_t __attribute__((ext_vector_type(2)));
using U32x4 = std::uint32_t __attribute__((ext_vector_type(4)));
using I32x2 = int __attribute__((ext_vector_type(2)));
using F32x8 = float __attribute__((ext_vector_type(8)));

struct alignas(16) Bank {
    alignas(16) std::uint8_t activation[kTileM][kTileK];
    alignas(16) std::uint8_t weight[kTileN][kTileK];
};

struct alignas(16) Tile {
    Bank bank[2];
};

static_assert(sizeof(Bank) == 8192U);
static_assert(sizeof(Tile) == kStaticLdsBytes);

__device__ __forceinline__ __attribute__((unused)) void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ __attribute__((unused)) U32x4 load_stage(
    const std::uint8_t* activation, const std::uint8_t* weight,
    std::uint32_t thread, std::uint32_t token_base, std::uint32_t row_base,
    std::uint32_t k_base) {
    const bool load_activation = thread < 256U;
    const std::uint32_t local = load_activation ? thread : thread - 256U;
    const std::uint32_t matrix_row = local >> 1U;
    const std::uint32_t k_offset = (local & 1U) * 16U;
    const std::uint8_t* source = load_activation
        ? activation + static_cast<std::size_t>(token_base + matrix_row) * kColumns
        : weight + static_cast<std::size_t>(row_base + matrix_row) * kColumns;
    return *reinterpret_cast<const U32x4*>(source + k_base + k_offset);
}

__device__ __forceinline__ __attribute__((unused)) void publish_stage(
    Tile& tile, U32x4 value, std::uint32_t bank, std::uint32_t thread) {
    const bool activation = thread < 256U;
    const std::uint32_t local = activation ? thread : thread - 256U;
    const std::uint32_t matrix_row = local >> 1U;
    const std::uint32_t k_offset = (local & 1U) * 16U;
    std::uint8_t* destination = activation
        ? &tile.bank[bank].activation[matrix_row][k_offset]
        : &tile.bank[bank].weight[matrix_row][k_offset];
    *reinterpret_cast<U32x4*>(destination) = value;
}

__device__ __forceinline__ __attribute__((unused)) I32x2 load_fragment(
    const std::uint8_t* source) {
    const U32x2 packed = *reinterpret_cast<const U32x2*>(source);
    I32x2 result{};
    result[0] = static_cast<int>(packed[0]);
    result[1] = static_cast<int>(packed[1]);
    return result;
}

__global__ __launch_bounds__(kThreads)
void fp8_gate_up_m128n128_k32_kernel(
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ token_scales,
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    std::uint16_t* __restrict__ output_bf16) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ Tile tile;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_token_tile = wave >> 2U;
    const std::uint32_t wave_row_tile = wave & 3U;
    const std::uint32_t token_base = blockIdx.y * kTileM;
    const std::uint32_t row_base = blockIdx.x * kTileN;

    F32x8 totals[2][2]{};
    U32x4 staged = load_stage(
        activation_codes, weight_codes, thread, token_base, row_base, 0U);
    publish_stage(tile, staged, 0U, thread);
    lds_barrier();

    std::uint32_t bank = 0U;
#pragma unroll 1
    for (std::uint32_t k_base = 0U; k_base < kColumns; k_base += kTileK) {
        U32x4 next{};
        const bool has_next = k_base + kTileK < kColumns;
        if (has_next) {
            // This load must remain live across the current bank's eight WMMAs.
            next = load_stage(activation_codes, weight_codes, thread,
                              token_base, row_base, k_base + kTileK);
        }

#pragma unroll
        for (std::uint32_t k_half = 0U; k_half < 2U; ++k_half) {
            I32x2 a[2]{};
            I32x2 b[2]{};
#pragma unroll
            for (std::uint32_t token_half = 0U; token_half < 2U; ++token_half) {
                const std::uint32_t token_fragment =
                    token_half * 64U + wave_token_tile * 16U + axis;
                a[token_half] = load_fragment(
                    &tile.bank[bank].activation[token_fragment]
                        [k_half * 16U + lane_group * 8U]);
            }
#pragma unroll
            for (std::uint32_t row_half = 0U; row_half < 2U; ++row_half) {
                const std::uint32_t row_fragment =
                    row_half * 64U + wave_row_tile * 16U + axis;
                b[row_half] = load_fragment(
                    &tile.bank[bank].weight[row_fragment]
                        [k_half * 16U + lane_group * 8U]);
            }

            // Four independent chains per K16. Keeping the first and second K16
            // in separate batches maximizes distance between updates to a chain.
            totals[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                a[0], b[0], totals[0][0]);
            totals[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                a[0], b[1], totals[0][1]);
            totals[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                a[1], b[0], totals[1][0]);
            totals[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                a[1], b[1], totals[1][1]);
        }

        if (has_next) {
            publish_stage(tile, next, bank ^ 1U, thread);
            // All consumers have finished the old bank and all stores to the
            // successor bank are visible. One generation serves both purposes.
            lds_barrier();
            bank ^= 1U;
        }
    }

    // Both operand banks are dead. Reuse their first 1024 bytes to load each
    // outer scale once per CTA rather than once per output fragment.
    float* const scales = reinterpret_cast<float*>(&tile);
    if (thread < kTileM) {
        scales[thread] = token_scales[token_base + thread];
    } else if (thread < kTileM + kTileN) {
        scales[thread] = weight_scales[row_base + thread - kTileM];
    }
    lds_barrier();

#pragma unroll
    for (std::uint32_t token_half = 0U; token_half < 2U; ++token_half) {
#pragma unroll
        for (std::uint32_t row_half = 0U; row_half < 2U; ++row_half) {
            const std::uint32_t row = row_base + row_half * 64U +
                                      wave_row_tile * 16U + axis;
            const float weight_scale = scales[kTileM + row - row_base];
#pragma unroll
            for (std::uint32_t element = 0U; element < 8U; ++element) {
                const std::uint32_t token = token_base + token_half * 64U +
                                            wave_token_tile * 16U +
                                            lane_group * 8U + element;
                const float value = (totals[token_half][row_half][element] *
                                     scales[token - token_base]) * weight_scale;
                reinterpret_cast<hip_bfloat16*>(output_bf16)
                    [static_cast<std::size_t>(token) * kRows + row] =
                        static_cast<hip_bfloat16>(value);
            }
        }
    }
#else
    (void)activation_codes;
    (void)token_scales;
    (void)weight_codes;
    (void)weight_scales;
    (void)output_bf16;
#endif
}

bool disjoint(const void* first, std::size_t first_bytes,
              const void* second, std::size_t second_bytes) {
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    if (a > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
        b > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
        return false;
    }
    return a + first_bytes <= b || b + second_bytes <= a;
}

} // namespace

hipError_t launch(const Arguments& a, hipStream_t stream) {
    constexpr std::size_t activation_bytes =
        static_cast<std::size_t>(kTokens) * kColumns;
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kRows) * kColumns;
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kTokens) * kRows * sizeof(std::uint16_t);
    constexpr std::size_t token_scale_bytes =
        static_cast<std::size_t>(kTokens) * sizeof(float);
    constexpr std::size_t weight_scale_bytes =
        static_cast<std::size_t>(kRows) * sizeof(float);
    if (a.tokens != kTokens || a.rows != kRows || a.columns != kColumns ||
        a.activation_codes == nullptr || a.token_scales == nullptr ||
        a.weight_codes == nullptr || a.weight_scales == nullptr ||
        a.output_bf16 == nullptr ||
        (reinterpret_cast<std::uintptr_t>(a.activation_codes) & 15U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(a.weight_codes) & 15U) != 0U ||
        !disjoint(a.output_bf16, output_bytes, a.activation_codes, activation_bytes) ||
        !disjoint(a.output_bf16, output_bytes, a.weight_codes, weight_bytes) ||
        !disjoint(a.output_bf16, output_bytes, a.token_scales, token_scale_bytes) ||
        !disjoint(a.output_bf16, output_bytes, a.weight_scales, weight_scale_bytes)) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(kRows / kTileN, kTokens / kTileM);
    hipLaunchKernelGGL(fp8_gate_up_m128n128_k32_kernel, grid, dim3(kThreads), 0, stream,
                       a.activation_codes, a.token_scales, a.weight_codes,
                       a.weight_scales, a.output_bf16);
    return hipGetLastError();
}

hipError_t resources(Resources* output) {
    if (output == nullptr) return hipErrorInvalidValue;
    hipFuncAttributes attributes{};
    const hipError_t status = hipFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(fp8_gate_up_m128n128_k32_kernel));
    if (status != hipSuccess) return status;
    output->registers = attributes.numRegs;
    output->static_lds_bytes = static_cast<int>(attributes.sharedSizeBytes);
    output->private_bytes = static_cast<int>(attributes.localSizeBytes);
    output->maximum_threads = attributes.maxThreadsPerBlock;
    return hipSuccess;
}

} // namespace ninfer::qualification::fp8_gate_up_m128n128
