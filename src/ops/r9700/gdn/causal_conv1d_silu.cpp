#include "ninfer/ops/causal_conv1d_silu.h"

#include "core/device.h"
#include "ops/r9700/gdn/gdn_ops.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kChannels = 10240;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1,
                   const char* label) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(label);
    }
}

void require_bf16_contiguous(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.data == nullptr || !tensor.is_contiguous()) {
        throw std::invalid_argument(label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto first_address = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second.data);
    const std::size_t first_bytes = first.bytes();
    const std::size_t second_bytes = second.bytes();
    if (first_address <= second_address) {
        return second_address - first_address < first_bytes;
    }
    return first_address - second_address < second_bytes;
}

} // namespace

void causal_conv1d_silu(const Tensor& input, const Tensor& weight, const Tensor& history_in,
                        Tensor& history_out, Tensor& output, hipStream_t stream) {
    require_bf16_contiguous(input, "causal_conv1d_silu: input must be contiguous BF16");
    require_bf16_contiguous(weight, "causal_conv1d_silu: weight must be contiguous BF16");
    require_bf16_contiguous(history_in,
                            "causal_conv1d_silu: history input must be contiguous BF16");
    require_bf16_contiguous(history_out,
                            "causal_conv1d_silu: history output must be contiguous BF16");
    require_bf16_contiguous(output, "causal_conv1d_silu: output must be contiguous BF16");
    if (input.ne[1] <= 0) {
        throw std::invalid_argument("causal_conv1d_silu: T must be positive");
    }
    require_shape(input, kChannels, input.ne[1],
                  "causal_conv1d_silu: input must have shape [10240,T]");
    require_shape(weight, kChannels, 4,
                  "causal_conv1d_silu: weight must have shape [10240,4]");
    require_shape(history_in, kChannels, 3,
                  "causal_conv1d_silu: history input must have shape [10240,3]");
    require_shape(history_out, kChannels, 3,
                  "causal_conv1d_silu: history output must have shape [10240,3]");
    require_shape(output, kChannels, input.ne[1],
                  "causal_conv1d_silu: output must match input shape");
    if (stream == nullptr) {
        throw std::invalid_argument("causal_conv1d_silu: stream must be non-null");
    }

    const std::array<const Tensor*, 5> tensors{
        &input, &weight, &history_in, &history_out, &output};
    for (std::size_t first = 0; first < tensors.size(); ++first) {
        for (std::size_t second = first + 1; second < tensors.size(); ++second) {
            const bool exact_history_alias = first == 2 && second == 3 &&
                                             history_in.data == history_out.data;
            if (!exact_history_alias && overlaps(*tensors[first], *tensors[second])) {
                throw std::invalid_argument(
                    "causal_conv1d_silu: tensors overlap outside the exact history alias");
            }
        }
    }

    HIP_CHECK(r9700::gdn::causal_conv1d_silu_bf16(
        static_cast<const hip_bfloat16*>(input.data),
        static_cast<const hip_bfloat16*>(weight.data),
        static_cast<const hip_bfloat16*>(history_in.data),
        static_cast<hip_bfloat16*>(history_out.data),
        static_cast<hip_bfloat16*>(output.data), static_cast<std::uint32_t>(kChannels),
        static_cast<std::uint32_t>(input.ne[1]), stream));
}

void causal_conv1d_silu(const Tensor& input, const Tensor& weight, Tensor& history,
                        Tensor& output, hipStream_t stream) {
    causal_conv1d_silu(input, weight, static_cast<const Tensor&>(history), history, output,
                       stream);
}

} // namespace ninfer::ops
