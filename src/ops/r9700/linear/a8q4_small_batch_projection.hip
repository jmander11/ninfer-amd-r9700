#include "a8q4_small_batch_projection.h"
#include "a8q4_pipeline_impl.h"

namespace ninfer::ops::r9700::linear {
namespace {
template <unsigned N, unsigned K, unsigned T>
__global__ __launch_bounds__(32) void small_batch_projection_kernel(
    const std::uint8_t* low, const std::uint8_t* high, const std::uint16_t* scales,
    const std::uint32_t* status, const std::uint8_t* codes,
    const std::uint16_t* weight_scales, hip_bfloat16* output) {
    if constexpr(N==12288)
        detail::a8q4_scale_gather_body<N,K,T>(low,high,scales,status,codes,weight_scales,output);
    else
        detail::a8q4_pipeline_body<N,K,T>(low,high,scales,status,codes,weight_scales,output);
}
template <unsigned N, unsigned K, unsigned T>
hipError_t launch_width(const A8Q4G64LinearArgs& a, hipStream_t stream) {
    hipLaunchKernelGGL((small_batch_projection_kernel<N,K,T>),dim3(N/16,(T+15)/16),dim3(32),0,stream,
        a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,
        a.weight_codes,a.weight_scales,a.output);
    return hipGetLastError();
}
template <unsigned N, unsigned K>
hipError_t launch_prepared(const A8Q4G64LinearArgs& a, hipStream_t stream) {
    if constexpr ((N==34816 && K==5120) || (N==5120 && (K==17408 || K==6144))) {
        switch(a.tokens) {
        case 2: return launch_width<N,K,2>(a,stream);
        case 3: return launch_width<N,K,3>(a,stream);
        case 4: return launch_width<N,K,4>(a,stream);
        }
    }
    if constexpr ((N==34816 && K==5120) || (N==5120 && K==17408)) {
        switch(a.tokens) {
        case 12: return launch_width<N,K,12>(a,stream);
        case 18: return launch_width<N,K,18>(a,stream);
        case 24: return launch_width<N,K,24>(a,stream);
        }
    }
    if constexpr (K!=17408) {
        if(a.tokens==5)return launch_width<N,K,5>(a,stream);
        if(a.tokens==6)return launch_width<N,K,6>(a,stream);
    }
    return hipErrorInvalidValue;
}
template <unsigned N, unsigned K>
hipError_t launch(const A8Q4G64CandidateArgs& a, hipStream_t stream) {
    const auto validation = detail::validate_a8q4_scale_gather<N,K>(a,stream);
    if (validation != hipSuccess) return validation;
    A8G64ActivationWorkspace scratch{};
    auto result = a8q4g64_bind_activation_workspace(a.activation_workspace,
        a.activation_workspace_bytes,a.tokens,K,&scratch);
    if (result != hipSuccess) return result;
    result = a8g64_quantize_activation({a.input,scratch},stream);
    if (result != hipSuccess) return result;
    return launch_prepared<N,K>(
        {scratch.low_codes,scratch.low_code_bytes,scratch.high_codes,scratch.high_code_bytes,
         scratch.scales,scratch.scale_bytes,scratch.status,a.weight_codes,a.weight_code_bytes,
         a.weight_scales,a.weight_scale_bytes,a.output,a.tokens,a.rows,a.columns,a.padded_columns},stream);
}
}
hipError_t detail::launch_a8q4_small_batch_projection(
    const A8Q4G64LinearArgs& a, hipStream_t stream) noexcept {
    if (a.rows == 34816) return launch_prepared<34816,5120>(a,stream);
    if (a.rows == 5120 && a.columns == 17408) return launch_prepared<5120,17408>(a,stream);
    if (a.rows == 5120) return launch_prepared<5120,6144>(a,stream);
    if (a.rows == 12288) return launch_prepared<12288,5120>(a,stream);
    return launch_prepared<4096,5120>(a,stream);
}
hipError_t a8q4_small_batch_projection(
    const A8Q4G64CandidateArgs& a, hipStream_t stream) noexcept {
    if (!detail::use_a8q4_small_batch_projection(a.tokens,a.rows,a.columns,a.padded_columns))
        return hipErrorInvalidValue;
    if (a.rows == 34816 && a.columns == 5120) return launch<34816,5120>(a,stream);
    if (a.rows == 5120 && a.columns == 17408) return launch<5120,17408>(a,stream);
    if (a.rows == 5120 && a.columns == 6144) return launch<5120,6144>(a,stream);
    if (a.rows == 12288 && a.columns == 5120) return launch<12288,5120>(a,stream);
    if (a.rows == 4096 && a.columns == 5120) return launch<4096,5120>(a,stream);
    return hipErrorInvalidValue;
}
}
