#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

#include <functional>
#include <map>
#include <regex>

namespace {

constexpr std::uint32_t kColumns = 5120U;
constexpr std::uint32_t kGroups = kColumns / kGroup;
constexpr std::uint32_t kQueryRows = 6144U;
constexpr std::uint32_t kKeyRows = 1024U;
constexpr std::size_t kCopies = 3U;
constexpr std::size_t kPairTrials = 12U;
constexpr std::size_t kSamplesPerCopy = kPairTrials / kCopies;
constexpr std::array<std::uint32_t, 3> kTokenWidths{2U, 3U, 4U};

enum class PairKind : std::uint32_t { Gdn = 0U, Attention = 1U };

struct PairShape {
    PairKind kind;
    std::uint32_t rows0;
    std::uint32_t rows1;
    std::uint32_t calls;
    const char* name;
};

constexpr std::array<PairShape, 2> kPairShapes{{
    {PairKind::Gdn, 4096U, 12288U, 48U, "gdn"},
    {PairKind::Attention, 7168U, 7168U, 16U, "attention"},
}};

using WmmaF32 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using WmmaI32x2 = int __attribute__((ext_vector_type(2)));
using WmmaI32x8 = int __attribute__((ext_vector_type(8)));

struct DeviceWeights {
    std::unique_ptr<DeviceBuffer<std::uint8_t>> codes0;
    std::unique_ptr<DeviceBuffer<std::uint16_t>> scales0;
    std::unique_ptr<DeviceBuffer<std::uint8_t>> codes1;
    std::unique_ptr<DeviceBuffer<std::uint16_t>> scales1;
};

struct Outputs {
    std::array<std::unique_ptr<DeviceBuffer<hip_bfloat16>>, 4> buffers;
    std::array<std::uint32_t, 4> rows{};
};

struct PairArgs {
    const hip_bfloat16* input = nullptr;
    const std::uint8_t* weight0_codes = nullptr;
    std::size_t weight0_code_bytes = 0U;
    const std::uint16_t* weight0_scales = nullptr;
    std::size_t weight0_scale_bytes = 0U;
    const std::uint8_t* weight1_codes = nullptr;
    std::size_t weight1_code_bytes = 0U;
    const std::uint16_t* weight1_scales = nullptr;
    std::size_t weight1_scale_bytes = 0U;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0U;
    hip_bfloat16* output0 = nullptr;
    hip_bfloat16* output1 = nullptr;
    hip_bfloat16* output2 = nullptr;
    hip_bfloat16* output3 = nullptr;
    std::uint32_t tokens = 0U;
    std::uint32_t rows0 = 0U;
    std::uint32_t rows1 = 0U;
    PairKind kind = PairKind::Gdn;
};

constexpr std::size_t pair_index(std::uint32_t row, std::uint32_t group,
                                 std::uint32_t pair) {
    return (((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 4U + pair) *
            16U + (row & 15U));
}

constexpr std::size_t scale_index(std::uint32_t row, std::uint32_t group) {
    return ((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 16U +
            (row & 15U));
}

__device__ __forceinline__ int load_q4x8_pair(const std::uint8_t* codes,
                                               std::size_t element) {
    return *reinterpret_cast<const int*>(codes + element / 2U);
}

__device__ __forceinline__ int load_weight_x8_pair(
    const std::uint8_t* codes, std::uint32_t row, std::uint32_t group,
    std::uint32_t word) {
    const std::uint64_t packed =
        reinterpret_cast<const std::uint64_t*>(codes)[pair_index(row, group, word >> 1U)];
    return static_cast<int>((word & 1U) != 0U ? packed >> 32U : packed);
}

__device__ hip_bfloat16 pair_poison() {
    hip_bfloat16 value;
    value.data = 0x7fc1U;
    return value;
}

__global__ __launch_bounds__(32) void a8q4g64_pair_wmma_c2c4_kernel(
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
    if (*activation_status == 0U) {
        for (std::uint32_t group = 0U; group < kGroups; ++group) {
            WmmaI32x8 low_dot{};
            WmmaI32x8 high_dot{};
#pragma unroll
            for (std::uint32_t half = 0U; half < 2U; ++half) {
                const std::uint32_t k_base = group * kGroup + half * 32U + lane_group * 16U;
                WmmaI32x2 a_low{};
                WmmaI32x2 a_high{};
                WmmaI32x2 b{};
                const std::uint32_t token = axis;
                if (token < tokens) {
                    const std::size_t element = static_cast<std::size_t>(token) * kColumns + k_base;
                    a_low[0] = load_q4x8_pair(activation_low, element);
                    a_low[1] = load_q4x8_pair(activation_low, element + 8U);
                    a_high[0] = load_q4x8_pair(activation_high, element);
                    a_high[1] = load_q4x8_pair(activation_high, element + 8U);
                }
                const std::uint32_t word = (k_base & (kGroup - 1U)) / 8U;
                b[0] = load_weight_x8_pair(weight_codes, row, group, word);
                b[1] = load_weight_x8_pair(weight_codes, row, group, word + 1U);
                low_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    false, a_low, true, b, low_dot, false);
                high_dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    true, a_high, true, b, high_dot, false);
            }
            const float ws = __half2float(__ushort_as_half(
                weight_scales[scale_index(row, group)]));
#pragma unroll
            for (std::uint32_t element = 0U; element < 8U; ++element) {
                const std::uint32_t token = lane_group * 8U + element;
                const float as = token < tokens ? __half2float(__ushort_as_half(
                    activation_scales[static_cast<std::size_t>(token) * kGroups + group])) : 0.0F;
                total[element] = fmaf(static_cast<float>(low_dot[element] + 16 * high_dot[element]),
                                      as * ws, total[element]);
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
            *activation_status == 0U ? static_cast<hip_bfloat16>(total[element]) : pair_poison();
    }
#else
    (void)activation_low; (void)activation_high; (void)activation_scales;
    (void)activation_status; (void)weight0_codes; (void)weight0_scales;
    (void)weight1_codes; (void)weight1_scales; (void)output0; (void)output1;
    (void)output2; (void)output3; (void)tokens; (void)rows0; (void)rows1;
    (void)attention;
#endif
}

__global__ void pair_extract_kernel(const hip_bfloat16* source,
                                    hip_bfloat16* destination,
                                    std::uint32_t source_rows,
                                    std::uint32_t source_row,
                                    std::uint32_t destination_rows,
                                    std::uint32_t tokens) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(destination_rows) * tokens;
    if (index >= count) return;
    const std::uint32_t token = index / destination_rows;
    const std::uint32_t row = index - static_cast<std::size_t>(token) * destination_rows;
    destination[index] = source[static_cast<std::size_t>(token) * source_rows + source_row + row];
}

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes,
                    const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0U || rhs_bytes == 0U) return false;
    const auto l = reinterpret_cast<std::uintptr_t>(lhs);
    const auto r = reinterpret_cast<std::uintptr_t>(rhs);
    return l < r + rhs_bytes && r < l + lhs_bytes;
}

std::array<std::uint32_t, 4> output_rows(const PairShape& shape) {
    return shape.kind == PairKind::Attention
        ? std::array<std::uint32_t, 4>{kQueryRows, kKeyRows, kQueryRows, kKeyRows}
        : std::array<std::uint32_t, 4>{shape.rows0, shape.rows1, 0U, 0U};
}

Outputs make_outputs(const PairShape& shape, std::uint32_t tokens) {
    Outputs result;
    result.rows = output_rows(shape);
    for (std::size_t i = 0U; i < 4U; ++i) {
        if (result.rows[i] != 0U) {
            result.buffers[i] = std::make_unique<DeviceBuffer<hip_bfloat16>>(
                static_cast<std::size_t>(tokens) * result.rows[i] + 2U);
        }
    }
    return result;
}

hip_bfloat16* payload(Outputs& outputs, std::size_t index) {
    return outputs.buffers[index] == nullptr ? nullptr : outputs.buffers[index]->get() + 1U;
}

PairArgs make_args(const PairShape& shape, std::uint32_t tokens,
                   const hip_bfloat16* input, const DeviceWeights& weights,
                   void* workspace, std::size_t workspace_bytes, Outputs& outputs) {
    return {input, weights.codes0->get(), weights.codes0->bytes(), weights.scales0->get(),
            weights.scales0->bytes(), weights.codes1->get(), weights.codes1->bytes(),
            weights.scales1->get(), weights.scales1->bytes(), workspace, workspace_bytes,
            payload(outputs, 0U), payload(outputs, 1U), payload(outputs, 2U), payload(outputs, 3U),
            tokens, shape.rows0, shape.rows1, shape.kind};
}

bool valid_args(const PairArgs& args) {
    const bool token_ok = args.tokens >= 2U && args.tokens <= 4U;
    const bool kind_ok = args.kind == PairKind::Gdn || args.kind == PairKind::Attention;
    const bool attention = args.kind == PairKind::Attention;
    const bool shape_ok = attention ? args.rows0 == 7168U && args.rows1 == 7168U
                                    : args.rows0 == 4096U && args.rows1 == 12288U;
    const std::size_t code0 = static_cast<std::size_t>(args.rows0) * kColumns / 2U;
    const std::size_t code1 = static_cast<std::size_t>(args.rows1) * kColumns / 2U;
    const std::size_t scale0 = static_cast<std::size_t>(args.rows0) * kGroups * 2U;
    const std::size_t scale1 = static_cast<std::size_t>(args.rows1) * kGroups * 2U;
    const std::size_t workspace =
        linear::a8q4g64_activation_workspace_capacity_bytes(args.tokens, kColumns);
    if (!token_ok || !kind_ok || !shape_ok || args.input == nullptr || args.weight0_codes == nullptr ||
        args.weight0_scales == nullptr || args.weight1_codes == nullptr ||
        args.weight1_scales == nullptr || args.workspace == nullptr ||
        args.output0 == nullptr || args.output1 == nullptr ||
        (attention && (args.output2 == nullptr || args.output3 == nullptr)) ||
        (!attention && (args.output2 != nullptr || args.output3 != nullptr)) ||
        args.weight0_code_bytes != code0 || args.weight1_code_bytes != code1 ||
        args.weight0_scale_bytes != scale0 || args.weight1_scale_bytes != scale1 ||
        args.workspace_bytes != workspace ||
        (reinterpret_cast<std::uintptr_t>(args.weight0_codes) & 7U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(args.weight1_codes) & 7U) != 0U) return false;
    const auto rows = attention
        ? std::array<std::uint32_t, 4>{kQueryRows, kKeyRows, kQueryRows, kKeyRows}
        : std::array<std::uint32_t, 4>{args.rows0, args.rows1, 0U, 0U};
    std::array<std::pair<const void*, std::size_t>, 9> regions{{
        {args.input, static_cast<std::size_t>(args.tokens) * kColumns * 2U},
        {args.weight0_codes, code0}, {args.weight0_scales, scale0},
        {args.weight1_codes, code1}, {args.weight1_scales, scale1},
        {args.output0, static_cast<std::size_t>(args.tokens) * rows[0] * 2U},
        {args.output1, static_cast<std::size_t>(args.tokens) * rows[1] * 2U},
        {args.output2, static_cast<std::size_t>(args.tokens) * rows[2] * 2U},
        {args.output3, static_cast<std::size_t>(args.tokens) * rows[3] * 2U}}};
    for (std::size_t i = 0U; i < regions.size(); ++i) {
        if (ranges_overlap(args.workspace, workspace, regions[i].first, regions[i].second))
            return false;
        for (std::size_t j = i + 1U; j < regions.size(); ++j)
            if (ranges_overlap(regions[i].first, regions[i].second,
                               regions[j].first, regions[j].second)) return false;
    }
    return true;
}

hipError_t bind_and_quantize(const PairArgs& args,
                             linear::A8G64ActivationWorkspace* workspace,
                             hipStream_t stream) {
    if (!valid_args(args) || linear::a8q4g64_bind_activation_workspace(
            args.workspace, args.workspace_bytes, args.tokens, kColumns,
            workspace) != hipSuccess) return hipErrorInvalidValue;
    return linear::a8g64_quantize_activation({args.input, *workspace}, stream);
}

linear::A8Q4G64LinearArgs wmma_args(
    const PairArgs& args, const linear::A8G64ActivationWorkspace& workspace,
    bool second, hip_bfloat16* output) {
    return {workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
            workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
            workspace.status, second ? args.weight1_codes : args.weight0_codes,
            second ? args.weight1_code_bytes : args.weight0_code_bytes,
            second ? args.weight1_scales : args.weight0_scales,
            second ? args.weight1_scale_bytes : args.weight0_scale_bytes,
            output, args.tokens, second ? args.rows1 : args.rows0, kColumns, kColumns};
}

hipError_t launch_combined(const PairArgs& args, hipStream_t stream = nullptr) {
    linear::A8G64ActivationWorkspace workspace{};
    hipError_t status = bind_and_quantize(args, &workspace, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(a8q4g64_pair_wmma_c2c4_kernel,
                       dim3((args.rows0 + args.rows1) / 16U), dim3(32U), 0, stream,
                       workspace.low_codes, workspace.high_codes, workspace.scales,
                       workspace.status, args.weight0_codes, args.weight0_scales,
                       args.weight1_codes, args.weight1_scales, args.output0, args.output1,
                       args.output2, args.output3, args.tokens, args.rows0, args.rows1,
                       args.kind == PairKind::Attention ? 1U : 0U);
    return hipGetLastError();
}

hipError_t extract(const hip_bfloat16* source, hip_bfloat16* destination,
                   std::uint32_t source_rows, std::uint32_t source_row,
                   std::uint32_t rows, std::uint32_t tokens,
                   hipStream_t stream = nullptr) {
    const std::size_t count = static_cast<std::size_t>(rows) * tokens;
    hipLaunchKernelGGL(pair_extract_kernel, dim3((count + 255U) / 256U), dim3(256U), 0,
                       stream, source, destination, source_rows, source_row, rows, tokens);
    return hipGetLastError();
}

linear::A8Q4G64CandidateArgs candidate_args(
    const hip_bfloat16* input, const DeviceBuffer<std::uint8_t>& codes,
    const DeviceBuffer<std::uint16_t>& scales, void* workspace,
    std::size_t workspace_bytes, hip_bfloat16* output,
    std::uint32_t tokens, std::uint32_t rows) {
    return {input, codes.get(), codes.bytes(), scales.get(), scales.bytes(), workspace,
            workspace_bytes, output, tokens, rows, kColumns, kColumns};
}

hipError_t scatter_if_attention(const PairShape& shape, std::uint32_t tokens,
                                const hip_bfloat16* temp0, const hip_bfloat16* temp1,
                                Outputs& outputs, hipStream_t stream = nullptr) {
    if (shape.kind == PairKind::Gdn) return hipSuccess;
    hipError_t status = extract(temp0, payload(outputs, 0U), shape.rows0, 0U,
                                kQueryRows, tokens, stream);
    if (status != hipSuccess) return status;
    status = extract(temp0, payload(outputs, 1U), shape.rows0, kQueryRows,
                     kKeyRows, tokens, stream);
    if (status != hipSuccess) return status;
    status = extract(temp1, payload(outputs, 2U), shape.rows1, 0U,
                     kQueryRows, tokens, stream);
    if (status != hipSuccess) return status;
    return extract(temp1, payload(outputs, 3U), shape.rows1, kQueryRows,
                   kKeyRows, tokens, stream);
}

hipError_t launch_serial(const PairShape& shape, std::uint32_t tokens,
                         const hip_bfloat16* input, const DeviceWeights& weights,
                         void* workspace, std::size_t workspace_bytes,
                         hip_bfloat16* temp0, hip_bfloat16* temp1, Outputs& outputs) {
    hip_bfloat16* out0 = shape.kind == PairKind::Gdn ? payload(outputs, 0U) : temp0;
    hip_bfloat16* out1 = shape.kind == PairKind::Gdn ? payload(outputs, 1U) : temp1;
    hipError_t status = linear::a8q4g64_linear_candidate(
        candidate_args(input, *weights.codes0, *weights.scales0, workspace,
                       workspace_bytes, out0, tokens, shape.rows0), nullptr);
    if (status != hipSuccess) return status;
    if (shape.kind == PairKind::Attention) {
        status = extract(temp0, payload(outputs, 0U), shape.rows0, 0U,
                         kQueryRows, tokens);
        if (status != hipSuccess) return status;
        status = extract(temp0, payload(outputs, 1U), shape.rows0, kQueryRows,
                         kKeyRows, tokens);
        if (status != hipSuccess) return status;
    }
    status = linear::a8q4g64_linear_candidate(
        candidate_args(input, *weights.codes1, *weights.scales1, workspace,
                       workspace_bytes, out1, tokens, shape.rows1), nullptr);
    if (status != hipSuccess) return status;
    if (shape.kind == PairKind::Gdn) return hipSuccess;
    status = extract(temp1, payload(outputs, 2U), shape.rows1, 0U,
                     kQueryRows, tokens);
    if (status != hipSuccess) return status;
    return extract(temp1, payload(outputs, 3U), shape.rows1, kQueryRows,
                   kKeyRows, tokens);
}

hipError_t launch_shared_two(const PairShape& shape, const PairArgs& args,
                             hip_bfloat16* temp0, hip_bfloat16* temp1,
                             Outputs& outputs) {
    linear::A8G64ActivationWorkspace workspace{};
    hipError_t status = bind_and_quantize(args, &workspace, nullptr);
    if (status != hipSuccess) return status;
    hip_bfloat16* out0 = shape.kind == PairKind::Gdn ? payload(outputs, 0U) : temp0;
    hip_bfloat16* out1 = shape.kind == PairKind::Gdn ? payload(outputs, 1U) : temp1;
    status = linear::a8q4g64_linear_wmma32(wmma_args(args, workspace, false, out0), nullptr);
    if (status != hipSuccess) return status;
    status = linear::a8q4g64_linear_wmma32(wmma_args(args, workspace, true, out1), nullptr);
    if (status != hipSuccess) return status;
    return scatter_if_attention(shape, args.tokens, temp0, temp1, outputs);
}

void guard_outputs(Outputs& outputs, std::uint32_t tokens, std::uint16_t bits) {
    for (std::size_t i = 0U; i < 4U; ++i) {
        if (outputs.buffers[i] == nullptr) continue;
        std::vector<hip_bfloat16> host(static_cast<std::size_t>(tokens) * outputs.rows[i] + 2U);
        for (auto& value : host) value.data = bits;
        HIP_CHECK(hipMemcpy(outputs.buffers[i]->get(), host.data(),
                            host.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    }
}

std::array<std::vector<hip_bfloat16>, 4> download_outputs(
    const Outputs& outputs, std::uint32_t tokens, std::uint16_t bits) {
    std::array<std::vector<hip_bfloat16>, 4> result;
    for (std::size_t i = 0U; i < 4U; ++i) {
        if (outputs.buffers[i] == nullptr) continue;
        std::vector<hip_bfloat16> host(static_cast<std::size_t>(tokens) * outputs.rows[i] + 2U);
        HIP_CHECK(hipMemcpy(host.data(), outputs.buffers[i]->get(),
                            host.size() * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
        if (host.front().data != bits || host.back().data != bits)
            fail("paired WMMA changed an output canary");
        result[i].assign(host.begin() + 1, host.end() - 1);
    }
    return result;
}

int represented_code(const DecodeDot8Weights& weights, std::uint32_t row,
                     std::uint32_t group, std::uint32_t lane) {
    const std::size_t word = pair_index(row, group, lane >> 4U);
    const std::size_t byte = word * 8U + ((lane & 15U) >> 1U);
    const unsigned code = (weights.codes[byte] >> ((lane & 1U) * 4U)) & 15U;
    return code >= 8U ? static_cast<int>(code) - 16 : static_cast<int>(code);
}

void salt_second_matrix(DecodeDot8Weights& weights) {
    for (std::size_t byte = 0U; byte < weights.codes.size(); ++byte) {
        std::uint8_t transformed = 0U;
        for (unsigned half = 0U; half < 2U; ++half) {
            const unsigned nibble = (weights.codes[byte] >> (half * 4U)) & 15U;
            const int code = nibble >= 8U ? static_cast<int>(nibble) - 16
                                          : static_cast<int>(nibble);
            const int offset = static_cast<int>((byte * 5U + half * 3U + 1U) % 15U) + 1;
            const int salted = ((code + offset + 8) & 15) - 8;
            transformed |= static_cast<std::uint8_t>((salted & 15) << (half * 4U));
        }
        weights.codes[byte] = transformed;
    }
    for (std::size_t index = 0U; index < weights.scales.size(); ++index) {
        const float factor = 1.03125F + static_cast<float>(index % 5U) * 0.015625F;
        weights.scales[index] = half_bits(half_value(weights.scales[index]) * factor);
    }
}

std::uint64_t check_oracle(const PairShape& shape, std::uint32_t tokens,
                           const HostActivation& activation,
                           const DecodeDot8Weights& weights0,
                           const DecodeDot8Weights& weights1,
                           const std::array<std::vector<hip_bfloat16>, 4>& actual) {
    std::uint64_t maximum = 0U;
    for (std::uint32_t matrix = 0U; matrix < 2U; ++matrix) {
        const auto& weights = matrix == 0U ? weights0 : weights1;
        const std::uint32_t rows = matrix == 0U ? shape.rows0 : shape.rows1;
        for (std::uint32_t token = 0U; token < tokens; ++token) {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                double sum = 0.0;
                for (std::uint32_t group = 0U; group < kGroups; ++group) {
                    std::int32_t dot = 0;
                    const std::size_t base = static_cast<std::size_t>(token) * kColumns +
                                             group * kGroup;
                    for (std::uint32_t lane = 0U; lane < kGroup; ++lane) {
                        const int a = unsigned_nibble(activation.low, base + lane) +
                                      16 * signed_nibble(activation.high, base + lane);
                        dot += a * represented_code(weights, row, group, lane);
                    }
                    sum = std::fma(static_cast<double>(dot),
                        static_cast<double>(half_value(activation.scales[
                            static_cast<std::size_t>(token) * kGroups + group])) *
                        half_value(weights.scales[scale_index(row, group)]), sum);
                }
                std::size_t part = matrix;
                std::uint32_t part_row = row;
                if (shape.kind == PairKind::Attention) {
                    part = matrix * 2U + (row >= kQueryRows ? 1U : 0U);
                    part_row = row < kQueryRows ? row : row - kQueryRows;
                }
                const std::size_t index = static_cast<std::size_t>(token) *
                                          output_rows(shape)[part] + part_row;
                const std::uint64_t steps = bf16_steps(
                    actual[part][index], hip_bfloat16(static_cast<float>(sum)));
                maximum = std::max(maximum, steps);
                if (steps > 2U) fail("paired WMMA differs from independent FP64 oracle");
            }
        }
    }
    return maximum;
}

double timed(const std::function<hipError_t()>& launch) {
    Event begin;
    Event end;
    HIP_CHECK(hipEventRecord(begin.get()));
    HIP_CHECK(launch());
    HIP_CHECK(hipEventRecord(end.get()));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float elapsed = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed, begin.get(), end.get()));
    return elapsed;
}

template <std::size_t N> double even_median(std::array<double, N> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (values[N / 2U - 1U] + values[N / 2U]);
}

struct CaseResult {
    PairShape shape{};
    std::uint32_t tokens = 0U;
    std::uint64_t max_steps = 0U;
    double serial_ms = 0.0;
    double shared_two_ms = 0.0;
    double combined_ms = 0.0;
    std::array<double, kPairTrials> serial_samples{};
    std::array<double, kPairTrials> shared_samples{};
    std::array<double, kPairTrials> combined_samples{};
    std::array<double, kCopies> serial_copy{};
    std::array<double, kCopies> shared_copy{};
    std::array<double, kCopies> combined_copy{};
};

CaseResult run_case(const PairShape& shape, std::uint32_t tokens) {
    const auto input = make_input(kColumns, tokens);
    const auto activation = quantize_host(input, tokens, kColumns);
    const auto host0 = make_decode_dot8_weights(shape.rows0, kColumns);
    auto host1 = make_decode_dot8_weights(shape.rows1, kColumns);
    salt_second_matrix(host1);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    std::array<DeviceWeights, kCopies> copies;
    for (auto& copy : copies) {
        copy.codes0 = std::make_unique<DeviceBuffer<std::uint8_t>>(host0.codes.size());
        copy.scales0 = std::make_unique<DeviceBuffer<std::uint16_t>>(host0.scales.size());
        copy.codes1 = std::make_unique<DeviceBuffer<std::uint8_t>>(host1.codes.size());
        copy.scales1 = std::make_unique<DeviceBuffer<std::uint16_t>>(host1.scales.size());
        HIP_CHECK(hipMemcpy(copy.codes0->get(), host0.codes.data(), copy.codes0->bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(copy.scales0->get(), host0.scales.data(), copy.scales0->bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(copy.codes1->get(), host1.codes.data(), copy.codes1->bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(copy.scales1->get(), host1.scales.data(), copy.scales1->bytes(), hipMemcpyHostToDevice));
    }
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, kColumns);
    DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
    DeviceBuffer<hip_bfloat16> temp0(static_cast<std::size_t>(tokens) * shape.rows0);
    DeviceBuffer<hip_bfloat16> temp1(static_cast<std::size_t>(tokens) * shape.rows1);
    Outputs serial = make_outputs(shape, tokens);
    Outputs shared = make_outputs(shape, tokens);
    Outputs combined = make_outputs(shape, tokens);
    auto args = make_args(shape, tokens, d_input.get(), copies[0], workspace.get(),
                          workspace_bytes, combined);
    auto malformed = args;
    malformed.tokens = 1U;
    if (launch_combined(malformed) != hipErrorInvalidValue) fail("paired WMMA accepted T1");
    malformed = args; --malformed.weight1_code_bytes;
    if (launch_combined(malformed) != hipErrorInvalidValue) fail("paired WMMA accepted short weights");
    malformed = args; ++malformed.weight0_codes;
    if (launch_combined(malformed) != hipErrorInvalidValue) fail("paired WMMA accepted misalignment");
    malformed = args; malformed.output1 = malformed.output0;
    if (launch_combined(malformed) != hipErrorInvalidValue) fail("paired WMMA accepted output alias");
    malformed = args; malformed.workspace = malformed.output0;
    if (launch_combined(malformed) != hipErrorInvalidValue) fail("paired WMMA accepted workspace alias");
    malformed = args;
    malformed.input = reinterpret_cast<const hip_bfloat16*>(malformed.weight0_codes);
    if (launch_combined(malformed) != hipErrorInvalidValue) fail("paired WMMA accepted input alias");
    malformed = args; malformed.rows0 -= 16U;
    if (launch_combined(malformed) != hipErrorInvalidValue) fail("paired WMMA accepted wrong rows");

    guard_outputs(serial, tokens, 0x7fc2U);
    guard_outputs(shared, tokens, 0x7fc3U);
    guard_outputs(combined, tokens, 0x7fc4U);
    HIP_CHECK(launch_serial(shape, tokens, d_input.get(), copies[0], workspace.get(),
                            workspace_bytes, temp0.get(), temp1.get(), serial));
    HIP_CHECK(launch_shared_two(shape, make_args(shape, tokens, d_input.get(), copies[0],
                                                 workspace.get(), workspace_bytes, shared),
                                temp0.get(), temp1.get(), shared));
    HIP_CHECK(launch_combined(args));
    HIP_CHECK(hipDeviceSynchronize());
    const auto serial_host = download_outputs(serial, tokens, 0x7fc2U);
    const auto shared_host = download_outputs(shared, tokens, 0x7fc3U);
    const auto combined_host = download_outputs(combined, tokens, 0x7fc4U);
    for (std::size_t part = 0U; part < 4U; ++part) {
        if (serial_host[part].size() != shared_host[part].size() ||
            serial_host[part].size() != combined_host[part].size() ||
            !std::equal(serial_host[part].begin(), serial_host[part].end(), shared_host[part].begin(),
                        [](hip_bfloat16 a, hip_bfloat16 b) { return a.data == b.data; }) ||
            !std::equal(serial_host[part].begin(), serial_host[part].end(), combined_host[part].begin(),
                        [](hip_bfloat16 a, hip_bfloat16 b) { return a.data == b.data; }))
            fail("paired WMMA three-arm BF16 parity failed");
    }
    const std::uint64_t max_steps = check_oracle(shape, tokens, activation, host0, host1,
                                                  combined_host);

    hip_bfloat16 nan{}; nan.data = 0x7fc1U;
    HIP_CHECK(hipMemcpy(d_input.get(), &nan, sizeof(nan), hipMemcpyHostToDevice));
    guard_outputs(combined, tokens, 0x7fc5U);
    HIP_CHECK(launch_combined(make_args(shape, tokens, d_input.get(), copies[0], workspace.get(),
                                        workspace_bytes, combined)));
    HIP_CHECK(hipDeviceSynchronize());
    const auto poisoned = download_outputs(combined, tokens, 0x7fc5U);
    for (const auto& part : poisoned)
        if (!std::all_of(part.begin(), part.end(), [](hip_bfloat16 v) { return v.data == 0x7fc1U; }))
            fail("paired WMMA did not poison every output");
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));

    CaseResult result{shape, tokens, max_steps};
    std::array<std::array<double, kSamplesPerCopy>, kCopies> serial_by_copy{},
        shared_by_copy{}, combined_by_copy{};
    std::array<std::size_t, kCopies> serial_count{}, shared_count{}, combined_count{};
    for (std::size_t warm = 0U; warm < 9U; ++warm) {
        const auto& weights = copies[warm % kCopies];
        const std::size_t arm = warm / kCopies;
        if (arm == 0U) HIP_CHECK(launch_serial(shape, tokens, d_input.get(), weights,
            workspace.get(), workspace_bytes, temp0.get(), temp1.get(), serial));
        else if (arm == 1U) HIP_CHECK(launch_shared_two(shape, make_args(shape, tokens,
            d_input.get(), weights, workspace.get(), workspace_bytes, shared),
            temp0.get(), temp1.get(), shared));
        else HIP_CHECK(launch_combined(make_args(shape, tokens, d_input.get(), weights,
            workspace.get(), workspace_bytes, combined)));
    }
    HIP_CHECK(hipDeviceSynchronize());
    constexpr std::array<std::array<std::uint32_t, 3>, 6> orders{{
        {0U, 1U, 2U}, {0U, 2U, 1U}, {1U, 0U, 2U},
        {1U, 2U, 0U}, {2U, 0U, 1U}, {2U, 1U, 0U}}};
    for (std::size_t trial = 0U; trial < kPairTrials; ++trial) {
        for (std::size_t position = 0U; position < 3U; ++position) {
            const std::uint32_t arm = orders[trial % orders.size()][position];
            const std::size_t copy = position;
            if (arm == 0U) {
                result.serial_samples[trial] = timed([&] { return launch_serial(
                    shape, tokens, d_input.get(), copies[copy], workspace.get(), workspace_bytes,
                    temp0.get(), temp1.get(), serial); });
                serial_by_copy[copy][serial_count[copy]++] = result.serial_samples[trial];
            } else if (arm == 1U) {
                result.shared_samples[trial] = timed([&] { return launch_shared_two(shape,
                    make_args(shape, tokens, d_input.get(), copies[copy], workspace.get(),
                              workspace_bytes, shared), temp0.get(), temp1.get(), shared); });
                shared_by_copy[copy][shared_count[copy]++] = result.shared_samples[trial];
            } else {
                result.combined_samples[trial] = timed([&] { return launch_combined(make_args(
                    shape, tokens, d_input.get(), copies[copy], workspace.get(),
                    workspace_bytes, combined)); });
                combined_by_copy[copy][combined_count[copy]++] = result.combined_samples[trial];
            }
        }
    }
    result.serial_ms = even_median(result.serial_samples);
    result.shared_two_ms = even_median(result.shared_samples);
    result.combined_ms = even_median(result.combined_samples);
    for (std::size_t copy = 0U; copy < kCopies; ++copy) {
        if (serial_count[copy] != kSamplesPerCopy || shared_count[copy] != kSamplesPerCopy ||
            combined_count[copy] != kSamplesPerCopy)
            fail("paired WMMA cold schedule is not allocation balanced");
        result.serial_copy[copy] = even_median(serial_by_copy[copy]);
        result.shared_copy[copy] = even_median(shared_by_copy[copy]);
        result.combined_copy[copy] = even_median(combined_by_copy[copy]);
        if (result.combined_copy[copy] > result.serial_copy[copy])
            fail("paired WMMA regressed on a balanced cold allocation");
    }
    return result;
}

std::map<std::uint32_t, double> read_round_baselines(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open whole baseline receipt");
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    if (input.bad()) fail("cannot read whole baseline receipt");
    std::map<std::uint32_t, double> result;
    for (std::uint32_t c : kTokenWidths) {
        const std::regex expression("\\\"c" + std::to_string(c) +
                                    "_round_median_ms\\\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)");
        std::smatch match;
        if (!std::regex_search(text, match, expression)) fail("whole baseline receipt lacks C width");
        result[c] = std::stod(match[1].str());
        if (!(result[c] > 0.0)) fail("whole baseline receipt has invalid round time");
    }
    return result;
}

void write_array(std::ostream& out, const auto& values) {
    out << '[';
    for (std::size_t i = 0U; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 9 || std::string_view(argv[1]) != "--out-json" ||
            std::string_view(argv[3]) != "--assembly" ||
            std::string_view(argv[5]) != "--static-receipt" ||
            std::string_view(argv[7]) != "--whole-baselines") {
            std::cerr << "usage: a8q4_pair_wmma_c2c4_qual --out-json NEW.json "
                         "--assembly FILE.s --static-receipt RECEIPT.txt "
                         "--whole-baselines BOUND.json\n";
            return 2;
        }
        const std::filesystem::path output = argv[2];
        const std::filesystem::path assembly = std::filesystem::canonical(argv[4]);
        const std::filesystem::path static_receipt = std::filesystem::canonical(argv[6]);
        const std::filesystem::path baselines = std::filesystem::canonical(argv[8]);
        require_fresh_output(output);
        const auto whole = read_round_baselines(baselines);
        const std::filesystem::path root = std::filesystem::canonical(NINFER_SOURCE_DIR);
        const std::filesystem::path checker = std::filesystem::canonical(
            root / "tools/r9700/check_a8q4_pair_wmma_c2c4_static.py");
        const std::string static_command = "python3 " + shell_quote(checker.string()) + " " +
                                           shell_quote(assembly.string()) +
                                           " --expect-receipt " +
                                           shell_quote(static_receipt.string());
        if (std::system(static_command.c_str()) != 0) fail("paired WMMA static check failed");
        const std::string static_text = read_text(static_receipt);
        std::smatch static_match;
        const std::regex static_expression(
            "a8q4_pair_wmma_c2c4_static: PASS .*vgpr=([0-9]+) sgpr=([0-9]+) "
            "wave32=true lds=0 scratch=0 spills=0");
        if (!std::regex_match(static_text, static_match, static_expression))
            fail("bound paired WMMA static receipt differs");
        const std::uint32_t static_vgpr = static_cast<std::uint32_t>(
            std::stoul(static_match[1].str()));
        const std::uint32_t static_sgpr = static_cast<std::uint32_t>(
            std::stoul(static_match[2].str()));
        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        if (std::string_view(properties.name) != "AMD Radeon AI PRO R9700" ||
            std::string_view(properties.gcnArchName) != "gfx1201" || properties.warpSize != 32)
            fail("paired WMMA qualification requires R9700 gfx1201 wave32");
        char pci_bus_id[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
        std::string pci = pci_bus_id;
        if (pci.size() == 7U) pci = "0000:" + pci;
        std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const std::filesystem::path power = std::filesystem::path("/sys/bus/pci/devices") /
                                             pci / "power_dpm_force_performance_level";
        const std::string power_before = read_text(power);
        if (power_before != "auto") fail("paired WMMA qualification requires auto power");
        int runtime_version = 0;
        int driver_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK(hipDriverGetVersion(&driver_version));

        std::array<CaseResult, 6> results;
        std::size_t index = 0U;
        for (std::uint32_t tokens : kTokenWidths)
            for (const auto& shape : kPairShapes) results[index++] = run_case(shape, tokens);
        const std::string power_after = read_text(power);
        if (power_after != "auto") fail("power changed during paired WMMA qualification");

        std::map<std::uint32_t, double> saved;
        for (const auto& result : results)
            saved[result.tokens] += result.shape.calls * (result.serial_ms - result.combined_ms);
        for (std::uint32_t c : kTokenWidths)
            if (saved[c] < 0.01 * whole.at(c)) fail("paired WMMA misses whole admission");

        std::ostringstream report;
        const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
        const std::array<std::pair<const char*, const char*>, 5> source_specs{{
            {"qualifier", "tools/r9700/a8q4_pair_wmma_c2c4_qual.hip"},
            {"shared_qualifier", "tools/r9700/a8q4_shape_sweep_qual.hip"},
            {"static_checker", "tools/r9700/check_a8q4_pair_wmma_c2c4_static.py"},
            {"linear_contract", "src/ops/r9700/linear/r9700_linear.h"},
            {"linear_implementation", "src/ops/r9700/linear/r9700_linear.hip"},
        }};
        report << std::setprecision(17)
               << "{\n  \"schema\": \"ninfer.r9700.a8q4-pair-wmma-c2c4-qualification.v1\",\n"
               << "  \"status\": \"qualified_for_whole_ab_only\",\n"
               << "  \"production_routing_authorized\": false,\n"
               << "  \"invocation\": {\"argv\":[\"" << json_escape(executable.string())
               << "\",\"--out-json\",\"" << json_escape(output.string())
               << "\",\"--assembly\",\"" << json_escape(assembly.string())
               << "\",\"--static-receipt\",\"" << json_escape(static_receipt.string())
               << "\",\"--whole-baselines\",\"" << json_escape(baselines.string())
               << "\"]},\n"
               << "  \"hardware\": {\"device\":0,\"name\":\""
               << json_escape(properties.name) << "\",\"architecture\":\""
               << json_escape(properties.gcnArchName) << "\",\"pci\":\""
               << json_escape(pci) << "\",\"wave_size\":" << properties.warpSize
               << ",\"hip_runtime_version\":" << runtime_version
               << ",\"hip_driver_version\":" << driver_version << "},\n"
               << "  \"power_profile\": {\"path\":\"" << json_escape(power.string())
               << "\",\"required\":\"auto\",\"before\":\""
               << json_escape(power_before) << "\",\"after\":\""
               << json_escape(power_after) << "\"},\n"
               << "  \"executable\": {\"path\": \"" << json_escape(executable.string())
               << "\", \"bytes\": " << std::filesystem::file_size(executable)
               << ", \"sha256\": \"" << sha256_file(executable) << "\"},\n"
               << "  \"sources\": [\n";
        for (std::size_t i = 0U; i < source_specs.size(); ++i) {
            const auto path = std::filesystem::canonical(root / source_specs[i].second);
            report << "    {\"role\":\"" << source_specs[i].first << "\",\"path\":\""
                   << json_escape(path.string()) << "\",\"bytes\":"
                   << std::filesystem::file_size(path) << ",\"sha256\":\""
                   << sha256_file(path) << "\"}" << (i + 1U == source_specs.size() ? "\n" : ",\n");
        }
        report << "  ],\n"
               << "  \"whole_baselines\": {\"path\": \"" << json_escape(baselines.string())
               << "\", \"sha256\": \"" << sha256_file(baselines) << "\"},\n"
               << "  \"static\": {\"status\": \"passed\", \"native_iu4_wmma\": true, "
                  "\"wave_size\": 32, \"vgpr\": " << static_vgpr
               << ", \"sgpr\": " << static_sgpr
               << ", \"lds_bytes\": 0, \"scratch_bytes\": 0, \"spill_count\": 0, "
                  "\"assembly\":{\"path\":\"" << json_escape(assembly.string())
               << "\",\"sha256\":\"" << sha256_file(assembly)
               << "\"},\"receipt\":{\"path\":\"" << json_escape(static_receipt.string())
               << "\",\"sha256\":\"" << sha256_file(static_receipt)
               << "\",\"text\":\"" << json_escape(static_text)
               << "\"},\"checker_sha256\": \"" << sha256_file(checker) << "\"},\n"
               << "  \"correctness\": {\"three_arm_complete_bf16_bit_exact\": true, "
                  "\"independent_fp64_represented_oracle\": true, \"status_poisoning\": true, "
                  "\"canaries\": true, \"malformed_alignment_alias_rejection\": true},\n"
               << "  \"cases\": [\n";
        for (std::size_t i = 0U; i < results.size(); ++i) {
            const auto& r = results[i];
            const std::size_t weight_bytes = static_cast<std::size_t>(r.shape.rows0 + r.shape.rows1) *
                                             kColumns * 17U / 32U;
            report << "    {\"role\":\"" << r.shape.name << "\",\"tokens\":" << r.tokens
                   << ",\"rows\":[" << r.shape.rows0 << ',' << r.shape.rows1 << "],"
                   << "\"calls_per_round\":" << r.shape.calls << ",\"weight_pair_bytes\":"
                   << weight_bytes << ",\"disjoint_weight_copies\":3,"
                   << "\"intervening_weight_bytes\":" << 2U * weight_bytes
                   << ",\"allocation_balanced\":true,\"three_arm_order_balanced\":true,"
                      "\"every_combined_copy_nonregressing\":true,"
                   << "\"maximum_bf16_steps\":" << r.max_steps << ",\"serial_samples_ms\":";
            write_array(report, r.serial_samples);
            report << ",\"shared_two_samples_ms\":"; write_array(report, r.shared_samples);
            report << ",\"combined_samples_ms\":"; write_array(report, r.combined_samples);
            report << ",\"serial_copy_medians_ms\":"; write_array(report, r.serial_copy);
            report << ",\"shared_two_copy_medians_ms\":"; write_array(report, r.shared_copy);
            report << ",\"combined_copy_medians_ms\":"; write_array(report, r.combined_copy);
            report << ",\"serial_median_ms\":" << r.serial_ms
                   << ",\"shared_two_median_ms\":" << r.shared_two_ms
                   << ",\"combined_median_ms\":" << r.combined_ms << "}"
                   << (i + 1U == results.size() ? "\n" : ",\n");
        }
        report << "  ],\n  \"whole_bounds\": [";
        for (std::size_t i = 0U; i < kTokenWidths.size(); ++i) {
            const auto c = kTokenWidths[i];
            report << (i ? "," : "") << "{\"concurrency\":" << c
                   << ",\"round_median_ms\":" << whole.at(c)
                   << ",\"saved_ms_per_round\":" << saved[c]
                   << ",\"credible_gain_percent\":" << 100.0 * saved[c] / whole.at(c) << '}';
        }
        report << "],\n  \"decision\": \"prepare_selector_off_on_whole_ab_c2c4\"\n}\n";
        const std::string payload = report.str();
        const int descriptor = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                      0644);
        if (descriptor < 0) fail("cannot create paired WMMA report exclusively");
        FILE* stream = fdopen(descriptor, "wb");
        if (stream == nullptr) {
            (void)::close(descriptor);
            (void)::unlink(output.c_str());
            fail("cannot bind paired WMMA report stream");
        }
        const bool wrote = fwrite(payload.data(), 1U, payload.size(), stream) == payload.size();
        const bool closed = fclose(stream) == 0;
        if (!wrote || !closed) {
            (void)::unlink(output.c_str());
            fail("cannot finish paired WMMA report");
        }
        std::cout << "a8q4_pair_wmma_c2c4: PASS";
        for (std::uint32_t c : kTokenWidths)
            std::cout << " c" << c << "_gain_percent=" << 100.0 * saved[c] / whole.at(c);
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "A8Q4 paired WMMA C2-C4 qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
