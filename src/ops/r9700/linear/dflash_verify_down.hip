#include "dflash_verify_down.h"
#include "a8q4_pipeline_impl.h"

namespace ninfer::ops::r9700::linear {
namespace {
constexpr unsigned N=5120,K=17408;
template<unsigned T>
__global__ __launch_bounds__(32) void dflash_down_pipeline_kernel(
    const std::uint8_t* low,const std::uint8_t* high,const std::uint16_t* scales,
    const std::uint32_t* status,const std::uint8_t* codes,
    const std::uint16_t* weight_scales,hip_bfloat16* output) {
    detail::a8q4_pipeline_body<N,K,T>(low,high,scales,status,codes,weight_scales,output);
}
}

hipError_t a8q4g64_dflash_verify_down(const A8Q4G64CandidateArgs& a,hipStream_t stream) noexcept {
    const auto validated=detail::validate_a8q4_scale_gather<N,K>(a,stream);
    if(validated!=hipSuccess)return validated;
    A8G64ActivationWorkspace scratch{};
    auto status=a8q4g64_bind_activation_workspace(a.activation_workspace,a.activation_workspace_bytes,a.tokens,K,&scratch);
    if(status!=hipSuccess)return status;
    status=a8g64_quantize_activation({a.input,scratch},stream);
    if(status!=hipSuccess)return status;
    if(a.tokens==5)hipLaunchKernelGGL(dflash_down_pipeline_kernel<5>,dim3(N/16),dim3(32),0,stream,
        scratch.low_codes,scratch.high_codes,scratch.scales,scratch.status,a.weight_codes,a.weight_scales,a.output);
    else hipLaunchKernelGGL(dflash_down_pipeline_kernel<6>,dim3(N/16),dim3(32),0,stream,
        scratch.low_codes,scratch.high_codes,scratch.scales,scratch.status,a.weight_codes,a.weight_scales,a.output);
    return hipGetLastError();
}
}
