#include "ops/r9700/linear/fp8_small_t_linear.h"
#include "ops/r9700/linear/fp8_small_t_kernel.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear {
namespace {

using fp8_small_t_kernels::Epilogue;
using fp8_small_t_kernels::fp8_small_t_kernel;
using fp8_small_t_kernels::KernelArgs;
using fp8_small_t_kernels::kKAlignment;
using fp8_small_t_kernels::kSplit;

template <Epilogue Kind>
void launch(const KernelArgs& args, std::uint32_t ctas, hipStream_t stream) {
    if (args.tokens <= 16U)
        hipLaunchKernelGGL((fp8_small_t_kernel<Kind, 1U>), dim3(ctas), dim3(32U * kSplit), 0,
                           stream, args);
    else
        hipLaunchKernelGGL((fp8_small_t_kernel<Kind, 2U>), dim3(ctas), dim3(32U * kSplit), 0,
                           stream, args);
}

bool valid_weight(const Fp8RowScaledWeight& weight) noexcept {
    return weight.codes != nullptr && weight.scales != nullptr && weight.rows != 0U &&
           weight.rows % 16U == 0U;
}

bool valid_activation(const Fp8ActivationWorkspace& activation) noexcept {
    return activation.codes != nullptr && activation.scales != nullptr &&
           activation.status != nullptr &&
           fp8_small_t_supported(activation.tokens, 16U, activation.columns,
                                 activation.padded_columns);
}

KernelArgs kernel_args(const Fp8RowScaledWeight& weight,
                       const Fp8ActivationWorkspace& activation) noexcept {
    KernelArgs args;
    args.first = weight;
    args.activation = activation.codes;
    args.token_scales = activation.scales;
    args.status = activation.status;
    args.tokens = activation.tokens;
    args.columns = activation.columns;
    return args;
}

} // namespace

bool fp8_small_t_supported(std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns,
                           std::uint32_t padded_columns) noexcept {
    return tokens != 0U && tokens <= kFp8SmallTokenLimit && rows != 0U && rows % 16U == 0U &&
           columns == padded_columns && columns != 0U && columns % kKAlignment == 0U;
}

hipError_t fp8_small_t_linear(const Fp8RowScaledWeight& weight,
                              const Fp8ActivationWorkspace& activation, hip_bfloat16* output,
                              hipStream_t stream) noexcept {
    if (!valid_weight(weight) || !valid_activation(activation) || output == nullptr)
        return hipErrorInvalidValue;
    KernelArgs args = kernel_args(weight, activation);
    args.outputs[0] = output;
    launch<Epilogue::Store>(args, weight.rows / 16U, stream);
    return hipGetLastError();
}

hipError_t fp8_small_t_pair_split(const Fp8SmallTPairSplitArgs& pair,
                                  const Fp8ActivationWorkspace& activation,
                                  hipStream_t stream) noexcept {
    if (!valid_weight(pair.first) || !valid_weight(pair.second) ||
        pair.first.rows != pair.second.rows || pair.split == 0U ||
        pair.split >= pair.first.rows || !valid_activation(activation) ||
        pair.first_leading == nullptr || pair.first_trailing == nullptr ||
        pair.second_leading == nullptr || pair.second_trailing == nullptr) {
        return hipErrorInvalidValue;
    }
    KernelArgs args = kernel_args(pair.first, activation);
    args.second = pair.second;
    args.split = pair.split;
    args.outputs[0] = pair.first_leading;
    args.outputs[1] = pair.first_trailing;
    args.outputs[2] = pair.second_leading;
    args.outputs[3] = pair.second_trailing;
    launch<Epilogue::PairSplit>(args, (pair.first.rows + pair.second.rows) / 16U, stream);
    return hipGetLastError();
}

hipError_t fp8_small_t_residual(const Fp8RowScaledWeight& weight,
                                const Fp8ActivationWorkspace& activation, hip_bfloat16* residual,
                                hipStream_t stream) noexcept {
    if (!valid_weight(weight) || !valid_activation(activation) || residual == nullptr)
        return hipErrorInvalidValue;
    KernelArgs args = kernel_args(weight, activation);
    args.outputs[0] = residual;
    launch<Epilogue::Residual>(args, weight.rows / 16U, stream);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::linear
