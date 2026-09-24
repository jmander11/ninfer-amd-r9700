#include "ops/r9700/dflash/grouped_dynamic_conv_launch.h"

#include "core/device.h"
#include "ninfer/ops/grouped_dynamic_conv.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <bool Prepare>
__global__ void grouped_dynamic_conv_kernel(const hip_bfloat16* hidden,
                                            const hip_bfloat16* base_kernel,
                                            const hip_bfloat16* dynamic_source,
                                            hip_bfloat16* output,
                                            hip_bfloat16* finish_dynamic,
                                            std::int32_t tokens, std::int32_t columns) {
    const std::int32_t d = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.y);
    if (d >= kGroupedDynamicConvHidden || column >= columns) return;

    const std::int32_t token = column % tokens;
    const std::int32_t group = d / kGroupedDynamicConvGroupSize;
    const std::int64_t hidden_index =
        static_cast<std::int64_t>(column) * kGroupedDynamicConvHidden + d;
    const float current = static_cast<float>(hidden[hidden_index]);
    const float previous = token == 0
                               ? 0.0F
                               : static_cast<float>(hidden[hidden_index - kGroupedDynamicConvHidden]);
    constexpr std::int32_t phase = Prepare ? 0 : 1;
    const std::int64_t phase_base =
        static_cast<std::int64_t>(phase) * 2 * kGroupedDynamicConvHidden;
    const float base_current = static_cast<float>(base_kernel[phase_base + d]);
    const float base_previous =
        static_cast<float>(base_kernel[phase_base + kGroupedDynamicConvHidden + d]);

    float dynamic_current;
    float dynamic_previous;
    if constexpr (Prepare) {
        const std::int64_t projection_base =
            static_cast<std::int64_t>(column) * kGroupedDynamicConvProjRows;
        dynamic_current = static_cast<float>(dynamic_source[projection_base + group]);
        dynamic_previous = static_cast<float>(
            dynamic_source[projection_base + kGroupedDynamicConvGroups + group]);
        if ((d % kGroupedDynamicConvGroupSize) == 0) {
            const std::int64_t stash_base =
                static_cast<std::int64_t>(column) * 2 * kGroupedDynamicConvGroups;
            finish_dynamic[stash_base + group] =
                dynamic_source[projection_base + 2 * kGroupedDynamicConvGroups + group];
            finish_dynamic[stash_base + kGroupedDynamicConvGroups + group] =
                dynamic_source[projection_base + 3 * kGroupedDynamicConvGroups + group];
        }
    } else {
        const std::int64_t dynamic_base =
            static_cast<std::int64_t>(column) * 2 * kGroupedDynamicConvGroups;
        dynamic_current = static_cast<float>(dynamic_source[dynamic_base + group]);
        dynamic_previous =
            static_cast<float>(dynamic_source[dynamic_base + kGroupedDynamicConvGroups + group]);
    }
    output[hidden_index] = hip_bfloat16((base_current + dynamic_current) * current +
                                        (base_previous + dynamic_previous) * previous);
}

} // namespace

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& base_kernel,
                                         const Tensor& projection, Tensor& prepared,
                                         Tensor& finish_dynamic, hipStream_t stream) {
    const int columns = hidden.ne[1] * hidden.ne[2];
    const dim3 grid((kGroupedDynamicConvHidden + kBlock - 1) / kBlock,
                    static_cast<unsigned int>(columns));
    grouped_dynamic_conv_kernel<true><<<grid, kBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(hidden.data),
        static_cast<const hip_bfloat16*>(base_kernel.data),
        static_cast<const hip_bfloat16*>(projection.data),
        static_cast<hip_bfloat16*>(prepared.data),
        static_cast<hip_bfloat16*>(finish_dynamic.data), hidden.ne[1], columns);
    HIP_CHECK(hipGetLastError());
}

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& base_kernel,
                                        const Tensor& finish_dynamic, Tensor& out,
                                        hipStream_t stream) {
    const int columns = hidden.ne[1] * hidden.ne[2];
    const dim3 grid((kGroupedDynamicConvHidden + kBlock - 1) / kBlock,
                    static_cast<unsigned int>(columns));
    grouped_dynamic_conv_kernel<false><<<grid, kBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(hidden.data),
        static_cast<const hip_bfloat16*>(base_kernel.data),
        static_cast<const hip_bfloat16*>(finish_dynamic.data),
        static_cast<hip_bfloat16*>(out.data), nullptr, hidden.ne[1], columns);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
