#include "ninfer/ops/gdn_gating.h"

#include "core/device.h"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/gdn/bf16_projected_control_wave32.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require_matrix48(const Tensor& tensor, DType dtype, std::int32_t tokens,
                      const char* label) {
    if (tensor.dtype != dtype || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != 48 || tensor.ne[1] != tokens || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(label);
    }
}

void require_vector48(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::FP32 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != 48 || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto first_address  = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second.data);
    if (first_address <= second_address) {
        return second_address - first_address < first.bytes();
    }
    return first_address - second_address < second.bytes();
}

bool overlaps_bytes(const void* first, std::size_t first_bytes,
                    const void* second, std::size_t second_bytes) {
    const auto first_address  = reinterpret_cast<std::uintptr_t>(first);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second);
    return first_address < second_address + second_bytes &&
           second_address < first_address + first_bytes;
}

} // namespace

void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, hipStream_t stream) {
    if (a.ne[1] <= 0) { throw std::invalid_argument("gdn_gating: T must be positive"); }
    const std::int32_t tokens = a.ne[1];
    require_matrix48(a, DType::BF16, tokens,
                     "gdn_gating: a must be contiguous BF16 [48,T]");
    require_matrix48(b, DType::BF16, tokens,
                     "gdn_gating: b must be contiguous BF16 [48,T]");
    require_matrix48(g, DType::FP32, tokens,
                     "gdn_gating: g must be contiguous FP32 [48,T]");
    require_matrix48(beta, DType::FP32, tokens,
                     "gdn_gating: beta must be contiguous FP32 [48,T]");
    require_vector48(A_log, "gdn_gating: A_log must be contiguous FP32 [48]");
    require_vector48(dt_bias, "gdn_gating: dt_bias must be contiguous FP32 [48]");
    if (stream == nullptr) { throw std::invalid_argument("gdn_gating: stream must be non-null"); }

    const std::array<const Tensor*, 6> tensors{&a, &b, &A_log, &dt_bias, &g, &beta};
    for (std::size_t first = 0; first < tensors.size(); ++first) {
        for (std::size_t second = first + 1; second < tensors.size(); ++second) {
            if (overlaps(*tensors[first], *tensors[second])) {
                throw std::invalid_argument("gdn_gating: tensors must not overlap");
            }
        }
    }
    HIP_CHECK(r9700::gdn::control_gates_bf16(
        static_cast<const hip_bfloat16*>(a.data), static_cast<const hip_bfloat16*>(b.data),
        static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
        static_cast<float*>(g.data), static_cast<float*>(beta.data), 48U,
        static_cast<std::uint32_t>(tokens), stream));
}

void bf16_gdn_projected_gating(const Tensor& hidden, const Weight& a_weight,
                                  const Weight& b_weight, const Tensor& A_log,
                                  const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                  hipStream_t stream) {
    constexpr std::int32_t kColumns = 5120;
    constexpr std::int32_t kHeads = 48;
    if (hidden.dtype != DType::BF16 || hidden.data == nullptr || !hidden.is_contiguous() ||
        hidden.ne[0] != kColumns || (hidden.ne[1] != 1 && hidden.ne[1] != 5 && hidden.ne[1] != 6) || hidden.ne[2] != 1 || hidden.ne[3] != 1) {
        throw std::invalid_argument(
            "bf16_gdn_projected_gating: hidden must be contiguous BF16 [5120,T=1/5/6]");
    }
    const auto require_weight = [](const Weight& weight, const char* label) {
        constexpr std::uint64_t kBytes =
            static_cast<std::uint64_t>(kHeads) * kColumns * sizeof(hip_bfloat16);
        if (weight.qtype != QType::BF16_CTRL || weight.ndim != 2U ||
            weight.n != kHeads || weight.k != kColumns || weight.shape[0] != kHeads ||
            weight.shape[1] != kColumns || weight.padded_shape[0] != kHeads ||
            weight.padded_shape[1] != kColumns || weight.layout != QuantLayout::Contiguous ||
            weight.qdata == nullptr || weight.qhigh != nullptr || weight.scales != nullptr ||
            weight.group != 0 || weight.group_size != 0 || weight.qdata_bytes != kBytes ||
            weight.payload_bytes != kBytes) {
            throw std::invalid_argument(label);
        }
    };
    require_weight(a_weight,
                   "bf16_gdn_projected_gating: malformed a BF16_CTRL weight");
    require_weight(b_weight,
                   "bf16_gdn_projected_gating: malformed b BF16_CTRL weight");
    require_matrix48(g, DType::FP32, hidden.ne[1],
                     "bf16_gdn_projected_gating: g must be contiguous FP32 [48,T]");
    require_matrix48(beta, DType::FP32, hidden.ne[1],
                     "bf16_gdn_projected_gating: beta must be contiguous FP32 [48,T]");
    require_vector48(A_log,
                     "bf16_gdn_projected_gating: A_log must be contiguous FP32 [48]");
    require_vector48(dt_bias,
                     "bf16_gdn_projected_gating: dt_bias must be contiguous FP32 [48]");
    if (stream == nullptr) {
        throw std::invalid_argument("bf16_gdn_projected_gating: stream must be non-null");
    }
    const std::array<const Tensor*, 5> tensors{&hidden, &A_log, &dt_bias, &g, &beta};
    for (std::size_t first = 0; first < tensors.size(); ++first) {
        for (std::size_t second = first + 1; second < tensors.size(); ++second) {
            if (overlaps(*tensors[first], *tensors[second])) {
                throw std::invalid_argument(
                    "bf16_gdn_projected_gating: tensors must not overlap");
            }
        }
    }
    constexpr std::size_t kWeightBytes =
        static_cast<std::size_t>(kHeads) * kColumns * sizeof(hip_bfloat16);
    const std::array<const void*, 2> weight_data{a_weight.qdata, b_weight.qdata};
    if (overlaps_bytes(weight_data[0], kWeightBytes, weight_data[1], kWeightBytes)) {
        throw std::invalid_argument(
            "bf16_gdn_projected_gating: weights must not overlap");
    }
    for (const void* weight : weight_data) {
        for (const Tensor* tensor : tensors) {
            if (overlaps_bytes(weight, kWeightBytes, tensor->data, tensor->bytes())) {
                throw std::invalid_argument(
                    "bf16_gdn_projected_gating: inputs and outputs must not overlap");
            }
        }
    }
    if (hidden.ne[1] != 1) {
        HIP_CHECK(r9700::gdn::bf16_projected_control_wave32({
            static_cast<const hip_bfloat16*>(hidden.data),
            static_cast<const hip_bfloat16*>(a_weight.qdata),
            static_cast<const hip_bfloat16*>(b_weight.qdata),
            static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
            static_cast<float*>(g.data), static_cast<float*>(beta.data),
            static_cast<std::uint32_t>(hidden.ne[1])}, stream));
        return;
    }
    HIP_CHECK(r9700::gdn::bf16_projected_control_t1(
        static_cast<const hip_bfloat16*>(hidden.data),
        static_cast<const hip_bfloat16*>(a_weight.qdata),
        static_cast<const hip_bfloat16*>(b_weight.qdata),
        static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
        static_cast<float*>(g.data), static_cast<float*>(beta.data), stream));
}

} // namespace ninfer::ops
