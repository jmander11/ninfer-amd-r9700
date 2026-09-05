#include "ninfer/ops/gdn_gating.h"

#include "core/device.h"
#include "ops/r9700/gdn/gdn_ops.h"

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

} // namespace ninfer::ops
