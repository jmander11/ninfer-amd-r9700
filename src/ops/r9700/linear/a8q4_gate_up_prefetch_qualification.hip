// Qualification-only exact T1 N34816/K5120 consumer. Not linked by production.
#include "a8q4_gate_up_prefetch_qualification.h"
#include <hip/hip_fp16.h>
#include <cstdint>

namespace ninfer::ops::r9700::linear {
namespace {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
struct PackedGroup { std::uint64_t word[4]; };

__device__ __forceinline__ PackedGroup load_group(const std::uint8_t* codes,
                                                 unsigned row, unsigned group) {
    const unsigned offset = ((row / 16U * 80U + group) * 64U + row % 16U);
    const auto* words = reinterpret_cast<const std::uint64_t*>(codes);
    return {{words[offset], words[offset + 16U], words[offset + 32U], words[offset + 48U]}};
}

__device__ __forceinline__ float consume_group(
    PackedGroup weights, const std::uint32_t* low, const std::uint32_t* high,
    unsigned group, float scale, float total) {
    int ld = 0, hd = 0;
#pragma unroll
    for (unsigned w = 0; w < 8; ++w) {
        const int weight = static_cast<int>(weights.word[w / 2] >> ((w % 2) * 32));
        ld = __builtin_amdgcn_sudot8(false, low[group * 8U + w], true, weight, ld, false);
        hd = __builtin_amdgcn_sudot8(true, high[group * 8U + w], true, weight, hd, false);
    }
    return fmaf(static_cast<float>(ld + 16 * hd), scale, total);
}
__device__ __forceinline__ float group_scale(const std::uint16_t* as,
                                            const std::uint16_t* ws,
                                            unsigned row, unsigned group) {
    const float a = __half2float(__ushort_as_half(as[group]));
    const float b = __half2float(__ushort_as_half(ws[(row / 16U * 80U + group) * 16U + row % 16U]));
    return a * b;
}
#endif
}

extern "C" __global__ __launch_bounds__(256)
void a8q4_gate_up_prefetch_qualification_kernel(
    const std::uint32_t* low, const std::uint32_t* high, const std::uint16_t* as,
    const std::uint32_t* status, const std::uint8_t* codes,
    const std::uint16_t* scales, hip_bfloat16* output) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned row = blockIdx.x * 256U + threadIdx.x;
    if (row >= 34816U) return;
    if (*status != 0U) {
        hip_bfloat16 poisoned;
        poisoned.data = 0x7fc1U;
        output[row] = poisoned;
        return;
    }
    PackedGroup current = load_group(codes, row, 0);
    float total = 0.0F;
#pragma unroll 1
    for (unsigned group = 0; group < 79; ++group) {
        // Finish the current scale loads before issuing the successor payload.
        // Otherwise their VMEM wait drains that payload before the current FMA.
        const float scale = group_scale(as, scales, row, group);
        __builtin_amdgcn_sched_barrier(0);
        PackedGroup next = load_group(codes, row, group + 1U);
        // These scheduler barriers preserve load/compute phases without forcing
        // a memory wait. The static checker validates the emitted dependencies.
        __builtin_amdgcn_sched_barrier(0);
        total = consume_group(current, low, high, group, scale, total);
        __builtin_amdgcn_sched_barrier(0);
        current = next;
    }
    total = consume_group(current, low, high, 79U, group_scale(as, scales, row, 79U), total);
    output[row] = hip_bfloat16(total);
#else
    (void)low; (void)high; (void)as; (void)status; (void)codes;
    (void)scales; (void)output;
#endif
}

hipError_t launch_a8q4_gate_up_prefetch_qualification(
    const std::uint8_t* low, const std::uint8_t* high, const std::uint16_t* as,
    const std::uint32_t* status, const std::uint8_t* codes,
    const std::uint16_t* scales, hip_bfloat16* output, hipStream_t stream) {
    if (!low || !high || !as || !status || !codes || !scales || !output || !stream)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(a8q4_gate_up_prefetch_qualification_kernel, dim3(136), dim3(256),
                      0, stream, reinterpret_cast<const std::uint32_t*>(low),
                      reinterpret_cast<const std::uint32_t*>(high), as, status,
                      codes, scales, output);
    return hipGetLastError();
}
}
