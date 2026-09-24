// Qualification-only challenger. Not linked by any product target or public Op.
// The complete qualification entry accepts represented BF16 hidden, prepares
// its private A8G64 scratch on every invocation, then launch this 112 x 256 grid.
// Its semantic outputs are QK[4096], value-Z[12288], g[48], beta[48]. Convolution
// and recurrent state are outside this computation. No prepared scratch may escape
// that complete boundary. The static check never launches this file's kernels.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <limits>
#include "gdn_projection_control_grid_qualification.h"

namespace ninfer::ops::r9700::gdn::qualification {
namespace {
constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kQ4Blocks = 64U;
constexpr std::uint32_t kControlBlocks = 48U;
constexpr std::uint32_t kColumns = 5120U;
constexpr std::uint32_t kGroups = 80U;
static_assert(kQ4Blocks * kThreads == 4096U + 12288U);
static_assert(kQ4Blocks + kControlBlocks == 112U);

// Arithmetic copied without changing association from a8q4g64_gdn_pair_t1_kernel.
__device__ __forceinline__ void projection(
    std::uint32_t block, const std::uint8_t* low_codes,
    const std::uint8_t* high_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* status, const std::uint8_t* qk_codes,
    const std::uint16_t* qk_scales, hip_bfloat16* qk,
    const std::uint8_t* vz_codes, const std::uint16_t* vz_scales,
    hip_bfloat16* vz) {
    const std::uint32_t combined_row = block * blockDim.x + threadIdx.x;
    if (combined_row >= 16384U) return;
    const bool second = combined_row >= 4096U;
    const std::uint32_t row = second ? combined_row - 4096U : combined_row;
    hip_bfloat16* const output = second ? vz : qk;
    if (*status != 0U) {
        hip_bfloat16 poison;
        poison.data = 0x7fc1U;
        output[row] = poison;
        return;
    }
    const std::uint8_t* const codes = second ? vz_codes : qk_codes;
    const std::uint16_t* const scales = second ? vz_scales : qk_scales;
    float total = 0.0F;
    for (std::uint32_t group = 0; group < kGroups; ++group) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (std::uint32_t word = 0; word < 8U; ++word) {
            const std::size_t element = static_cast<std::size_t>(group) * 64U + word * 8U;
            const int low = *reinterpret_cast<const int*>(low_codes + element / 2U);
            const int high = *reinterpret_cast<const int*>(high_codes + element / 2U);
            const std::size_t pair =
                (((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 4U +
                   (word >> 1U)) * 16U + (row & 15U));
            const std::uint64_t packed = reinterpret_cast<const std::uint64_t*>(codes)[pair];
            const int weight = static_cast<int>((word & 1U) != 0U ? packed >> 32U : packed);
            low_dot = __builtin_amdgcn_sudot8(false, low, true, weight, low_dot, false);
            high_dot = __builtin_amdgcn_sudot8(true, high, true, weight, high_dot, false);
        }
        const std::int32_t combined = low_dot + 16 * high_dot;
        const float activation_scale = __half2float(__ushort_as_half(activation_scales[group]));
        const std::size_t scale =
            ((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 16U + (row & 15U));
        const float weight_scale = __half2float(__ushort_as_half(scales[scale]));
        total = fmaf(static_cast<float>(combined), activation_scale * weight_scale, total);
    }
    output[row] = static_cast<hip_bfloat16>(total);
}

// Arithmetic and BF16 seam copied from bf16_projected_control_t1_kernel.
// Every caller is uniform over the whole CTA, including all nine barriers.
__device__ __forceinline__ void control(
    std::uint32_t head, const hip_bfloat16* hidden,
    const hip_bfloat16* a_weight, const hip_bfloat16* b_weight,
    const float* a_log, const float* dt_bias, float* g, float* beta) {
    const std::size_t base = static_cast<std::size_t>(head) * kColumns;
    float a_partial = 0.0F;
    float b_partial = 0.0F;
    for (std::uint32_t k = threadIdx.x; k < kColumns; k += blockDim.x) {
        const float x = static_cast<float>(hidden[k]);
        a_partial = fmaf(x, static_cast<float>(a_weight[base + k]), a_partial);
        b_partial = fmaf(x, static_cast<float>(b_weight[base + k]), b_partial);
    }
    __shared__ float a_reduction[kThreads];
    __shared__ float b_reduction[kThreads];
    a_reduction[threadIdx.x] = a_partial;
    b_reduction[threadIdx.x] = b_partial;
    __syncthreads();
    for (std::uint32_t width = kThreads / 2U; width != 0U; width >>= 1U) {
        if (threadIdx.x < width) {
            a_reduction[threadIdx.x] += a_reduction[threadIdx.x + width];
            b_reduction[threadIdx.x] += b_reduction[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const hip_bfloat16 ar = static_cast<hip_bfloat16>(a_reduction[0]);
        const hip_bfloat16 br = static_cast<hip_bfloat16>(b_reduction[0]);
        const float av = static_cast<float>(ar) + dt_bias[head];
        const float softplus = av > 20.0F ? av : log1pf(expf(av));
        g[head] = -expf(a_log[head]) * softplus;
        const float bv = static_cast<float>(br);
        beta[head] = 1.0F / (1.0F + expf(-bv));
    }
}
} // namespace

// The scalar block discriminator is uniform for all eight waves. Q4 CTAs return
// before any LDS/barrier instruction; control CTA head indices are exactly 0..47.
extern "C" __global__ __launch_bounds__(kThreads)
void gdn_projection_control_grid_qualification_kernel(
    const hip_bfloat16* hidden, const std::uint8_t* low_codes,
    const std::uint8_t* high_codes, const std::uint16_t* activation_scales,
    const std::uint32_t* status, const std::uint8_t* qk_codes,
    const std::uint16_t* qk_scales, hip_bfloat16* qk,
    const std::uint8_t* vz_codes, const std::uint16_t* vz_scales,
    hip_bfloat16* vz, const hip_bfloat16* a_weight,
    const hip_bfloat16* b_weight, const float* a_log, const float* dt_bias,
    float* g, float* beta) {
    if (blockIdx.x < kQ4Blocks) {
        projection(blockIdx.x, low_codes, high_codes, activation_scales, status,
                   qk_codes, qk_scales, qk, vz_codes, vz_scales, vz);
        return;
    }
    if (blockIdx.x < kQ4Blocks + kControlBlocks) {
        control(blockIdx.x - kQ4Blocks, hidden, a_weight, b_weight, a_log, dt_bias, g, beta);
    }
}
hipError_t projection_control_grid(const ProjectionControlArgs& args,
                                  hipStream_t stream) noexcept {
    const auto& p = args.projection;
    const auto bytes = linear::a8q4g64_activation_workspace_capacity_bytes(1, 5120);
    if (p.tokens != 1 || p.columns != 5120 ||
        p.activation_workspace_bytes != bytes ||
        p.weight0_code_bytes != 4096ULL*5120/2 ||
        p.weight1_code_bytes != 12288ULL*5120/2 ||
        p.weight0_scale_bytes != 4096ULL*80*2 ||
        p.weight1_scale_bytes != 12288ULL*80*2 ||
        args.control_weight_bytes != 48ULL*5120*2) return hipErrorInvalidValue;
    struct Plane { const void* data; std::size_t size; std::size_t alignment; };
    const std::array<Plane,14> planes{{
        {p.input,5120*2,2}, {p.activation_workspace,bytes,256},
        {p.weight0_codes,p.weight0_code_bytes,8},
        {p.weight0_scales,p.weight0_scale_bytes,2}, {p.output0,4096*2,2},
        {p.weight1_codes,p.weight1_code_bytes,8},
        {p.weight1_scales,p.weight1_scale_bytes,2}, {p.output1,12288*2,2},
        {args.a_weight,args.control_weight_bytes,2},
        {args.b_weight,args.control_weight_bytes,2},
        {args.a_log,48*4,4}, {args.dt_bias,48*4,4},
        {args.g,48*4,4}, {args.beta,48*4,4}}};
    for (std::size_t i=0;i<planes.size();++i) {
        const auto begin=reinterpret_cast<std::uintptr_t>(planes[i].data);
        if (!begin || begin%planes[i].alignment ||
            begin>std::numeric_limits<std::uintptr_t>::max()-planes[i].size)
            return hipErrorInvalidValue;
        for (std::size_t j=0;j<i;++j) {
            const auto other=reinterpret_cast<std::uintptr_t>(planes[j].data);
            if (begin<other+planes[j].size && other<begin+planes[i].size)
                return hipErrorInvalidValue;
        }
    }
    unsigned flags=0;
    if (stream==nullptr || hipStreamGetFlags(stream,&flags)!=hipSuccess)
        return hipErrorInvalidValue;
    linear::A8G64ActivationWorkspace workspace{};
    auto status=linear::a8q4g64_bind_activation_workspace(
        p.activation_workspace,bytes,1,5120,&workspace);
    if (status!=hipSuccess) return status;
    status=linear::a8g64_quantize_activation({p.input,workspace},stream);
    if (status!=hipSuccess) return status;
    hipLaunchKernelGGL(gdn_projection_control_grid_qualification_kernel,
        dim3(112),dim3(256),0,stream,p.input,workspace.low_codes,workspace.high_codes,
        workspace.scales,workspace.status,p.weight0_codes,p.weight0_scales,p.output0,
        p.weight1_codes,p.weight1_scales,p.output1,args.a_weight,args.b_weight,
        args.a_log,args.dt_bias,args.g,args.beta);
    return hipGetLastError();
}
} // namespace ninfer::ops::r9700::gdn::qualification
