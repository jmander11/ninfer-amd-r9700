#include "ninfer/ops/attention_projection.h"
#include "ninfer/ops/gdn_projection.h"

#include "core/device.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/paired_projection/paired_projection.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::r9700::paired_projection {
namespace {

constexpr std::uint32_t kColumns = 5120U;
constexpr std::uint32_t kGroup = 64U;
constexpr std::uint32_t kGroups = kColumns / kGroup;
constexpr std::uint32_t kQueryRows = 6144U;
constexpr std::uint32_t kKeyRows = 1024U;
constexpr std::uint32_t kAttentionRows = kQueryRows + kKeyRows;
constexpr std::uint32_t kGdnRows0 = 4096U;
constexpr std::uint32_t kGdnRows1 = 12288U;

using WmmaF32 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using WmmaI32x2 = int __attribute__((ext_vector_type(2)));
using WmmaI32x8 = int __attribute__((ext_vector_type(8)));

constexpr std::size_t pair_index(std::uint32_t row, std::uint32_t group,
                                 std::uint32_t pair) {
    return (((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 4U + pair) *
            16U + (row & 15U));
}

constexpr std::size_t scale_index(std::uint32_t row, std::uint32_t group) {
    return ((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 16U +
            (row & 15U));
}

__device__ __forceinline__ int load_q4x8(const std::uint8_t* codes,
                                         std::size_t element) {
    return *reinterpret_cast<const int*>(codes + element / 2U);
}

__device__ __forceinline__ int load_weight_x8(const std::uint8_t* codes,
                                              std::uint32_t row,
                                              std::uint32_t group,
                                              std::uint32_t word) {
    const std::uint64_t packed =
        reinterpret_cast<const std::uint64_t*>(codes)[pair_index(row, group, word >> 1U)];
    return static_cast<int>((word & 1U) != 0U ? packed >> 32U : packed);
}

__device__ hip_bfloat16 poison() {
    hip_bfloat16 value;
    value.data = 0x7fc1U;
    return value;
}

// One physical kernel serves both semantic Ops. `attention` controls only the direct output
// scatter; matrix geometry and every pointer are validated by the owning host entry point.
__global__ __launch_bounds__(32) void paired_projection_q4_c2c4_kernel(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* activation_status,
    const std::uint8_t* weight0_codes, const std::uint16_t* weight0_scales,
    const std::uint8_t* weight1_codes, const std::uint16_t* weight1_scales,
    hip_bfloat16* output0, hip_bfloat16* output1,
    hip_bfloat16* output2, hip_bfloat16* output3,
    std::uint32_t tokens, std::uint32_t rows0, std::uint32_t rows1,
    std::uint32_t attention) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t combined_base = blockIdx.x * 16U;
    const bool second = combined_base >= rows0;
    const std::uint32_t row_base = second ? combined_base - rows0 : combined_base;
    const std::uint32_t row = row_base + axis;
    const std::uint32_t matrix_rows = second ? rows1 : rows0;
    if (row >= matrix_rows) return;
    const std::uint8_t* weight_codes = second ? weight1_codes : weight0_codes;
    const std::uint16_t* weight_scales = second ? weight1_scales : weight0_scales;
    WmmaF32 total{};
    if (*activation_status == linear::Q4G64ActivationOk) {
        for (std::uint32_t group = 0U; group < kGroups; ++group) {
            WmmaI32x8 low_dot{};
            WmmaI32x8 high_dot{};
#pragma unroll
            for (std::uint32_t half = 0U; half < 2U; ++half) {
                const std::uint32_t k_base = group * kGroup + half * 32U + lane_group * 16U;
                WmmaI32x2 a_low{};
                WmmaI32x2 a_high{};
                WmmaI32x2 weight{};
                if (axis < tokens) {
                    const std::size_t element =
                        static_cast<std::size_t>(axis) * kColumns + k_base;
                    a_low[0] = load_q4x8(activation_low, element);
                    a_low[1] = load_q4x8(activation_low, element + 8U);
                    a_high[0] = load_q4x8(activation_high, element);
                    a_high[1] = load_q4x8(activation_high, element + 8U);
                }
                const std::uint32_t word = (k_base & (kGroup - 1U)) / 8U;
                weight[0] = load_weight_x8(weight_codes, row, group, word);
                weight[1] = load_weight_x8(weight_codes, row, group, word + 1U);
                low_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    false, a_low, true, weight, low_dot, false);
                high_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    true, a_high, true, weight, high_dot, false);
            }
            const float weight_scale = __half2float(__ushort_as_half(
                weight_scales[scale_index(row, group)]));
#pragma unroll
            for (std::uint32_t element = 0U; element < 8U; ++element) {
                const std::uint32_t token = lane_group * 8U + element;
                const float activation_scale = token < tokens
                    ? __half2float(__ushort_as_half(activation_scales[
                          static_cast<std::size_t>(token) * kGroups + group]))
                    : 0.0F;
                total[element] = fmaf(
                    static_cast<float>(low_dot[element] + 16 * high_dot[element]),
                    activation_scale * weight_scale, total[element]);
            }
        }
    }
#pragma unroll
    for (std::uint32_t element = 0U; element < 8U; ++element) {
        const std::uint32_t token = lane_group * 8U + element;
        if (token >= tokens) continue;
        hip_bfloat16* destination = second ? output1 : output0;
        std::uint32_t destination_rows = matrix_rows;
        std::uint32_t destination_row = row;
        if (attention != 0U) {
            if (!second && row >= kQueryRows) {
                destination = output1;
                destination_rows = kKeyRows;
                destination_row = row - kQueryRows;
            } else if (second && row < kQueryRows) {
                destination = output2;
                destination_rows = kQueryRows;
            } else if (second) {
                destination = output3;
                destination_rows = kKeyRows;
                destination_row = row - kQueryRows;
            } else {
                destination_rows = kQueryRows;
            }
        }
        destination[static_cast<std::size_t>(token) * destination_rows + destination_row] =
            *activation_status == linear::Q4G64ActivationOk
                ? static_cast<hip_bfloat16>(total[element]) : poison();
    }
#else
    (void)activation_low; (void)activation_high; (void)activation_scales;
    (void)activation_status; (void)weight0_codes; (void)weight0_scales;
    (void)weight1_codes; (void)weight1_scales; (void)output0; (void)output1;
    (void)output2; (void)output3; (void)tokens; (void)rows0; (void)rows1;
    (void)attention;
#endif
}

bool overlap(const void* lhs, std::size_t lhs_bytes,
             const void* rhs, std::size_t rhs_bytes) noexcept {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0U || rhs_bytes == 0U) return false;
    const auto left = reinterpret_cast<std::uintptr_t>(lhs);
    const auto right = reinterpret_cast<std::uintptr_t>(rhs);
    if (left > std::numeric_limits<std::uintptr_t>::max() - lhs_bytes ||
        right > std::numeric_limits<std::uintptr_t>::max() - rhs_bytes) return true;
    return left < right + rhs_bytes && right < left + lhs_bytes;
}

struct CommonArgs {
    const hip_bfloat16* hidden;
    WeightView weight0;
    WeightView weight1;
    void* workspace;
    std::size_t workspace_bytes;
    std::array<hip_bfloat16*, 4> outputs;
    std::array<std::uint32_t, 4> output_rows;
    std::uint32_t tokens;
    std::uint32_t rows0;
    std::uint32_t rows1;
    bool attention;
};

hipError_t launch(const CommonArgs& args, hipStream_t stream) noexcept {
    if (args.tokens < 2U || args.tokens > 4U || args.hidden == nullptr ||
        args.weight0.codes == nullptr || args.weight0.scales == nullptr ||
        args.weight1.codes == nullptr || args.weight1.scales == nullptr ||
        args.workspace == nullptr || args.outputs[0] == nullptr || args.outputs[1] == nullptr ||
        (args.attention && (args.outputs[2] == nullptr || args.outputs[3] == nullptr)) ||
        (!args.attention && (args.outputs[2] != nullptr || args.outputs[3] != nullptr)) ||
        (reinterpret_cast<std::uintptr_t>(args.weight0.codes) & 7U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(args.weight1.codes) & 7U) != 0U) {
        return hipErrorInvalidValue;
    }
    const std::size_t code0 = static_cast<std::size_t>(args.rows0) * kColumns / 2U;
    const std::size_t code1 = static_cast<std::size_t>(args.rows1) * kColumns / 2U;
    const std::size_t scale0 = static_cast<std::size_t>(args.rows0) * kGroups * 2U;
    const std::size_t scale1 = static_cast<std::size_t>(args.rows1) * kGroups * 2U;
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(args.tokens, kColumns);
    if (args.weight0.code_bytes != code0 || args.weight1.code_bytes != code1 ||
        args.weight0.scale_bytes != scale0 || args.weight1.scale_bytes != scale1 ||
        args.workspace_bytes != workspace_bytes) return hipErrorInvalidValue;
    const std::array<std::pair<const void*, std::size_t>, 9> planes{{
        {args.hidden, static_cast<std::size_t>(args.tokens) * kColumns * 2U},
        {args.weight0.codes, code0}, {args.weight0.scales, scale0},
        {args.weight1.codes, code1}, {args.weight1.scales, scale1},
        {args.outputs[0], static_cast<std::size_t>(args.tokens) * args.output_rows[0] * 2U},
        {args.outputs[1], static_cast<std::size_t>(args.tokens) * args.output_rows[1] * 2U},
        {args.outputs[2], static_cast<std::size_t>(args.tokens) * args.output_rows[2] * 2U},
        {args.outputs[3], static_cast<std::size_t>(args.tokens) * args.output_rows[3] * 2U}}};
    for (std::size_t i = 0U; i < planes.size(); ++i) {
        if (overlap(args.workspace, workspace_bytes, planes[i].first, planes[i].second))
            return hipErrorInvalidValue;
        for (std::size_t j = i + 1U; j < planes.size(); ++j) {
            if (overlap(planes[i].first, planes[i].second,
                        planes[j].first, planes[j].second)) return hipErrorInvalidValue;
        }
    }
    linear::A8G64ActivationWorkspace workspace{};
    if (linear::a8q4g64_bind_activation_workspace(
            args.workspace, workspace_bytes, args.tokens, kColumns, &workspace) != hipSuccess) {
        return hipErrorInvalidValue;
    }
    const hipError_t quantized = linear::a8g64_quantize_activation(
        {args.hidden, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
#if defined(__HIPCC__)
    hipLaunchKernelGGL(paired_projection_q4_c2c4_kernel,
                       dim3((args.rows0 + args.rows1) / 16U), dim3(32U), 0, stream,
                       workspace.low_codes, workspace.high_codes, workspace.scales,
                       workspace.status, args.weight0.codes, args.weight0.scales,
                       args.weight1.codes, args.weight1.scales, args.outputs[0],
                       args.outputs[1], args.outputs[2], args.outputs[3], args.tokens,
                       args.rows0, args.rows1, args.attention ? 1U : 0U);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

} // namespace

hipError_t launch_gdn(const GdnArgs& args, hipStream_t stream) noexcept {
    return launch({args.hidden, args.query_key, args.value_z, args.activation_workspace,
                   args.activation_workspace_bytes,
                   {args.query_key_output, args.value_z_output, nullptr, nullptr},
                   {kGdnRows0, kGdnRows1, 0U, 0U}, args.tokens,
                   kGdnRows0, kGdnRows1, false}, stream);
}

hipError_t launch_attention(const AttentionArgs& args, hipStream_t stream) noexcept {
    return launch({args.hidden, args.query_key, args.gate_value, args.activation_workspace,
                   args.activation_workspace_bytes, {args.query, args.key, args.gate, args.value},
                   {kQueryRows, kKeyRows, kQueryRows, kKeyRows}, args.tokens,
                   kAttentionRows, kAttentionRows, true}, stream);
}

hipError_t resources(Resources* out) noexcept {
    if (out == nullptr) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    hipFuncAttributes attributes{};
    const hipError_t status = hipFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(paired_projection_q4_c2c4_kernel));
    if (status != hipSuccess) return status;
    *out = {attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
            static_cast<int>(attributes.localSizeBytes), attributes.maxThreadsPerBlock};
    return hipSuccess;
#else
    return hipErrorNotSupported;
#endif
}

} // namespace ninfer::ops::r9700::paired_projection

namespace ninfer::ops {
namespace {

constexpr std::int32_t kColumns = 5120;

std::uint32_t require_decode_tensor(const Tensor& tensor, std::int32_t rows,
                                    std::uint32_t expected_tokens, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != rows || tensor.ne[1] <= 0 || tensor.ne[2] <= 0 || tensor.ne[3] != 1 ||
        tensor.ne[1] > std::numeric_limits<std::int32_t>::max() / tensor.ne[2]) {
        throw std::invalid_argument(label);
    }
    const auto tokens = static_cast<std::uint32_t>(tensor.ne[1] * tensor.ne[2]);
    if (tokens < 2U || tokens > 4U || (expected_tokens != 0U && tokens != expected_tokens)) {
        throw std::invalid_argument(label);
    }
    return tokens;
}

void require_decode_weight(const Weight& weight, std::int32_t rows, const char* label) {
    const std::uint64_t code_bytes = static_cast<std::uint64_t>(rows) * kColumns / 2U;
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(rows) * (kColumns / 64U) * sizeof(std::uint16_t);
    if (weight.qtype != QType::Q4G64_F16S || weight.ndim != 2U ||
        weight.n != rows || weight.k != kColumns || weight.shape[0] != rows ||
        weight.shape[1] != kColumns || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != kColumns || weight.layout != QuantLayout::Q4N16K16 ||
        weight.group != 64 || weight.group_size != 64U ||
        weight.scale_dtype != DType::FP16 || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.qhigh != nullptr || weight.high_plane_bytes != 0U ||
        weight.qdata_bytes != code_bytes || weight.scale_bytes != scale_bytes) {
        throw std::invalid_argument(label);
    }
}

DeviceSpan require_workspace(const DeviceSpan& workspace, std::uint32_t tokens,
                             const char* label) {
    const std::size_t required =
        r9700::linear::a8q4g64_activation_workspace_capacity_bytes(tokens, kColumns);
    if (required == 0U || workspace.data == nullptr || workspace.bytes < required) {
        throw std::invalid_argument(label);
    }
    return {workspace.data, required};
}

r9700::paired_projection::WeightView weight_view(const Weight& weight) {
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<std::size_t>(weight.qdata_bytes),
            static_cast<const std::uint16_t*>(weight.scales),
            static_cast<std::size_t>(weight.scale_bytes)};
}

} // namespace

void full_attention_projection_decode(const Tensor& hidden, const Weight& query_key,
                                      const Weight& gate_value, Tensor& query, Tensor& key,
                                      Tensor& gate, Tensor& value,
                                      const DeviceSpan& activation_workspace,
                                      hipStream_t stream) {
    const std::uint32_t tokens = require_decode_tensor(
        hidden, kColumns, 0U, "attention decode projection: invalid hidden tensor");
    require_decode_tensor(query, 6144, tokens, "attention decode projection: invalid query");
    require_decode_tensor(key, 1024, tokens, "attention decode projection: invalid key");
    require_decode_tensor(gate, 6144, tokens, "attention decode projection: invalid gate");
    require_decode_tensor(value, 1024, tokens, "attention decode projection: invalid value");
    require_decode_weight(query_key, 7168,
                          "attention decode projection: malformed query-key weight");
    require_decode_weight(gate_value, 7168,
                          "attention decode projection: malformed gate-value weight");
    const DeviceSpan workspace = require_workspace(
        activation_workspace, tokens, "attention decode projection: activation workspace");
    HIP_CHECK(r9700::paired_projection::launch_attention(
        {static_cast<const hip_bfloat16*>(hidden.data), weight_view(query_key),
         weight_view(gate_value), workspace.data, workspace.bytes,
         static_cast<hip_bfloat16*>(query.data), static_cast<hip_bfloat16*>(key.data),
         static_cast<hip_bfloat16*>(gate.data), static_cast<hip_bfloat16*>(value.data), tokens},
        stream));
}

void gdn_input_projection_decode(const Tensor& hidden, const Weight& query_key,
                                 const Weight& value_z, Tensor& query_key_output,
                                 Tensor& value_z_output,
                                 const DeviceSpan& activation_workspace,
                                 hipStream_t stream) {
    const std::uint32_t tokens = require_decode_tensor(
        hidden, kColumns, 0U, "GDN decode projection: invalid hidden tensor");
    require_decode_tensor(query_key_output, 4096, tokens,
                          "GDN decode projection: invalid query-key output");
    require_decode_tensor(value_z_output, 12288, tokens,
                          "GDN decode projection: invalid value-z output");
    require_decode_weight(query_key, 4096,
                          "GDN decode projection: malformed query-key weight");
    require_decode_weight(value_z, 12288,
                          "GDN decode projection: malformed value-z weight");
    const DeviceSpan workspace = require_workspace(
        activation_workspace, tokens, "GDN decode projection: activation workspace");
    HIP_CHECK(r9700::paired_projection::launch_gdn(
        {static_cast<const hip_bfloat16*>(hidden.data), weight_view(query_key),
         weight_view(value_z), workspace.data, workspace.bytes,
         static_cast<hip_bfloat16*>(query_key_output.data),
         static_cast<hip_bfloat16*>(value_z_output.data), tokens}, stream));
}

} // namespace ninfer::ops
