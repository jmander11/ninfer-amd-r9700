#include "ninfer/ops/attention_projection.h"

#include "core/device.h"
#include "ops/r9700/attention_projection/attention_projection.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::r9700::attention_projection {
namespace {

constexpr std::uint32_t kColumns = 5120U;
constexpr std::uint32_t kRows = 7168U;
constexpr std::uint32_t kQueryRows = 6144U;
constexpr std::uint32_t kKeyRows = 1024U;
constexpr std::uint32_t kGroup = 64U;
constexpr std::uint32_t kGroups = kColumns / kGroup;
constexpr std::uint32_t kThreads = 256U;
constexpr std::size_t kCodeBytes = static_cast<std::size_t>(kRows) * kColumns / 2U;
constexpr std::size_t kScaleBytes = static_cast<std::size_t>(kRows) * kGroups * 2U;

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
    const std::uint64_t pair =
        reinterpret_cast<const std::uint64_t*>(codes)[pair_index(row, group, word >> 1U)];
    return static_cast<int>((word & 1U) != 0U ? pair >> 32U : pair);
}

__device__ hip_bfloat16 poison() {
    hip_bfloat16 value;
    value.data = 0x7fc1U;
    return value;
}

__global__ __launch_bounds__(kThreads) void attention_projection_q4_t1_kernel(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* activation_status,
    const std::uint8_t* query_key_codes, const std::uint16_t* query_key_scales,
    const std::uint8_t* gate_value_codes, const std::uint16_t* gate_value_scales,
    hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* gate, hip_bfloat16* value) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t combined_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (combined_row >= 2U * kRows) return;
    const bool second = combined_row >= kRows;
    const std::uint32_t row = second ? combined_row - kRows : combined_row;
    hip_bfloat16* output = nullptr;
    std::uint32_t output_row = row;
    if (!second) {
        if (row < kQueryRows) output = query;
        else { output = key; output_row -= kQueryRows; }
    } else {
        if (row < kQueryRows) output = gate;
        else { output = value; output_row -= kQueryRows; }
    }
    if (*activation_status != linear::Q4G64ActivationOk) {
        output[output_row] = poison();
        return;
    }
    const std::uint8_t* codes = second ? gate_value_codes : query_key_codes;
    const std::uint16_t* scales = second ? gate_value_scales : query_key_scales;
    float total = 0.0F;
    for (std::uint32_t group = 0U; group < kGroups; ++group) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (std::uint32_t word = 0U; word < 8U; ++word) {
            const std::size_t element = static_cast<std::size_t>(group) * kGroup + word * 8U;
            const int low = load_q4x8(activation_low, element);
            const int high = load_q4x8(activation_high, element);
            const int weight = load_weight_x8(codes, row, group, word);
            low_dot = __builtin_amdgcn_sudot8(false, low, true, weight, low_dot, false);
            high_dot = __builtin_amdgcn_sudot8(true, high, true, weight, high_dot, false);
        }
        const int combined = low_dot + 16 * high_dot;
        const float activation_scale =
            __half2float(__ushort_as_half(activation_scales[group]));
        const float weight_scale =
            __half2float(__ushort_as_half(scales[scale_index(row, group)]));
        total = fmaf(static_cast<float>(combined), activation_scale * weight_scale, total);
    }
    output[output_row] = static_cast<hip_bfloat16>(total);
#else
    (void)activation_low; (void)activation_high; (void)activation_scales;
    (void)activation_status; (void)query_key_codes; (void)query_key_scales;
    (void)gate_value_codes; (void)gate_value_scales; (void)query; (void)key;
    (void)gate; (void)value;
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

} // namespace

hipError_t launch(const Args& args, hipStream_t stream) noexcept {
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(1U, kColumns);
    if (args.hidden == nullptr || args.query_key_codes == nullptr ||
        args.query_key_scales == nullptr || args.gate_value_codes == nullptr ||
        args.gate_value_scales == nullptr || args.activation_workspace == nullptr ||
        args.query == nullptr || args.key == nullptr || args.gate == nullptr ||
        args.value == nullptr || args.tokens != 1U || args.columns != kColumns ||
        args.query_key_code_bytes != kCodeBytes ||
        args.gate_value_code_bytes != kCodeBytes ||
        args.query_key_scale_bytes != kScaleBytes ||
        args.gate_value_scale_bytes != kScaleBytes ||
        args.activation_workspace_bytes != workspace_bytes ||
        (reinterpret_cast<std::uintptr_t>(args.query_key_codes) & 7U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(args.gate_value_codes) & 7U) != 0U) {
        return hipErrorInvalidValue;
    }
    const std::array<std::pair<const void*, std::size_t>, 9> planes{{
        {args.hidden, kColumns * sizeof(hip_bfloat16)},
        {args.query_key_codes, kCodeBytes}, {args.query_key_scales, kScaleBytes},
        {args.gate_value_codes, kCodeBytes}, {args.gate_value_scales, kScaleBytes},
        {args.query, kQueryRows * sizeof(hip_bfloat16)},
        {args.key, kKeyRows * sizeof(hip_bfloat16)},
        {args.gate, kQueryRows * sizeof(hip_bfloat16)},
        {args.value, kKeyRows * sizeof(hip_bfloat16)}}};
    for (std::size_t i = 0U; i < planes.size(); ++i) {
        if (overlap(args.activation_workspace, workspace_bytes,
                    planes[i].first, planes[i].second)) return hipErrorInvalidValue;
        for (std::size_t j = i + 1U; j < planes.size(); ++j) {
            if (overlap(planes[i].first, planes[i].second,
                        planes[j].first, planes[j].second)) return hipErrorInvalidValue;
        }
    }
    linear::A8G64ActivationWorkspace workspace{};
    if (linear::a8q4g64_bind_activation_workspace(
            args.activation_workspace, workspace_bytes, 1U, kColumns, &workspace) != hipSuccess) {
        return hipErrorInvalidValue;
    }
    const hipError_t quantized =
        linear::a8g64_quantize_activation({args.hidden, workspace}, stream);
    if (quantized != hipSuccess) return quantized;
#if defined(__HIPCC__)
    hipLaunchKernelGGL(attention_projection_q4_t1_kernel,
                       dim3((2U * kRows + kThreads - 1U) / kThreads), dim3(kThreads), 0,
                       stream, workspace.low_codes, workspace.high_codes,
                       workspace.scales, workspace.status, args.query_key_codes,
                       args.query_key_scales, args.gate_value_codes,
                       args.gate_value_scales, args.query, args.key, args.gate, args.value);
    return hipGetLastError();
#else
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t resources(Resources* out) noexcept {
    if (out == nullptr) return hipErrorInvalidValue;
#if defined(__HIPCC__)
    hipFuncAttributes attributes{};
    const hipError_t status = hipFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(attention_projection_q4_t1_kernel));
    if (status != hipSuccess) return status;
    *out = {attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
            static_cast<int>(attributes.localSizeBytes), attributes.maxThreadsPerBlock};
    return hipSuccess;
#else
    return hipErrorNotSupported;
#endif
}

} // namespace ninfer::ops::r9700::attention_projection

namespace ninfer::ops {
namespace {

constexpr std::int32_t kColumns = 5120;
constexpr std::int32_t kRows = 7168;
constexpr std::int32_t kQueryRows = 6144;
constexpr std::int32_t kKeyRows = 1024;

void require_tensor(const Tensor& tensor, std::int32_t rows, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != rows || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(label);
    }
}

void require_weight(const Weight& weight, const char* label) {
    constexpr std::uint64_t code_bytes =
        static_cast<std::uint64_t>(kRows) * kColumns / 2U;
    constexpr std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(kRows) * (kColumns / 64U) * sizeof(std::uint16_t);
    if (weight.qtype != QType::Q4G64_F16S || weight.ndim != 2U ||
        weight.n != kRows || weight.k != kColumns || weight.shape[0] != kRows ||
        weight.shape[1] != kColumns || weight.padded_shape[0] != kRows ||
        weight.padded_shape[1] != kColumns || weight.layout != QuantLayout::Q4N16K16 ||
        weight.group != 64 || weight.group_size != 64U ||
        weight.scale_dtype != DType::FP16 || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0U || weight.qdata_bytes != code_bytes ||
        weight.scale_bytes != scale_bytes) {
        throw std::invalid_argument(label);
    }
}

} // namespace

void full_attention_projection_t1(const Tensor& hidden, const Weight& query_key,
                                  const Weight& gate_value, Tensor& query, Tensor& key,
                                  Tensor& gate, Tensor& value,
                                  const DeviceSpan& activation_workspace,
                                  hipStream_t stream) {
    require_tensor(hidden, kColumns, "attention projection: invalid hidden tensor");
    require_tensor(query, kQueryRows, "attention projection: invalid query tensor");
    require_tensor(key, kKeyRows, "attention projection: invalid key tensor");
    require_tensor(gate, kQueryRows, "attention projection: invalid gate tensor");
    require_tensor(value, kKeyRows, "attention projection: invalid value tensor");
    require_weight(query_key, "attention projection: malformed query-key weight");
    require_weight(gate_value, "attention projection: malformed gate-value weight");
    const std::size_t required = r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        1U, static_cast<std::uint32_t>(kColumns));
    if (required == 0U || activation_workspace.data == nullptr ||
        activation_workspace.bytes < required) {
        throw std::invalid_argument("attention projection: activation workspace is too small");
    }
    HIP_CHECK(r9700::attention_projection::launch(
        {.hidden = static_cast<const hip_bfloat16*>(hidden.data),
         .query_key_codes = static_cast<const std::uint8_t*>(query_key.qdata),
         .query_key_code_bytes = static_cast<std::size_t>(query_key.qdata_bytes),
         .query_key_scales = static_cast<const std::uint16_t*>(query_key.scales),
         .query_key_scale_bytes = static_cast<std::size_t>(query_key.scale_bytes),
         .gate_value_codes = static_cast<const std::uint8_t*>(gate_value.qdata),
         .gate_value_code_bytes = static_cast<std::size_t>(gate_value.qdata_bytes),
         .gate_value_scales = static_cast<const std::uint16_t*>(gate_value.scales),
         .gate_value_scale_bytes = static_cast<std::size_t>(gate_value.scale_bytes),
         .activation_workspace = activation_workspace.data,
         .activation_workspace_bytes = required,
         .query = static_cast<hip_bfloat16*>(query.data),
         .key = static_cast<hip_bfloat16*>(key.data),
         .gate = static_cast<hip_bfloat16*>(gate.data),
         .value = static_cast<hip_bfloat16*>(value.data),
         .tokens = 1U,
         .columns = static_cast<std::uint32_t>(kColumns)}, stream));
}

} // namespace ninfer::ops
