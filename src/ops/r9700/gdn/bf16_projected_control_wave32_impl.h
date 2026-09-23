#pragma once
#include "bf16_projected_control_wave32.h"
#include <array>
#include <cstddef>
#include <limits>

namespace ninfer::ops::r9700::gdn {
namespace projected_control_wave32_detail {
constexpr unsigned N=48,K=5120;
__global__ __launch_bounds__(32) void projected_control_paired_wave32_kernel(
    const hip_bfloat16* hidden,const hip_bfloat16* aw,const hip_bfloat16* bw,
    const float* a_log,const float* dt_bias,float* g,float* beta) {
    const unsigned lane=threadIdx.x, head=blockIdx.x, token=blockIdx.y;
    float a=0.0F,b=0.0F;
    // Exactly the admitted wave8 chain: lane, lane+32, ... lane+5088.
    // That kernel's K256 staging boundaries introduce no arithmetic operation.
    // Pairing the two independent chains shares x, not their reduction.
    // Four consecutive logical steps are loaded before consuming them. This
    // exposes independent loads without changing either ascending FMA chain.
    // K/32=160 is divisible by four; no tail or speculative out-of-range read.
    for(unsigned base=lane;base<K;base+=128) {
        float x[4],wa[4],wb[4];
#pragma unroll
        for(unsigned step=0;step<4;++step) {
            const unsigned k=base+step*32;
            x[step]=static_cast<float>(hidden[token*K+k]);
            wa[step]=static_cast<float>(aw[head*K+k]);
            wb[step]=static_cast<float>(bw[head*K+k]);
        }
#pragma unroll
        for(unsigned step=0;step<4;++step) {
            a=fmaf(x[step],wa[step],a);
            b=fmaf(x[step],wb[step],b);
        }
    }
    for(unsigned delta=16;delta;delta>>=1) {
        a+=__shfl_down(a,delta,32);
        b+=__shfl_down(b,delta,32);
    }
    if(lane==0) {
        const hip_bfloat16 ar=static_cast<hip_bfloat16>(a);
        const hip_bfloat16 br=static_cast<hip_bfloat16>(b);
        const float av=static_cast<float>(ar)+dt_bias[head];
        const float softplus=av>20.0F?av:log1pf(expf(av));
        g[token*N+head]=-expf(a_log[head])*softplus;
        const float bv=static_cast<float>(br);
        beta[token*N+head]=1.0F/(1.0F+expf(-bv));
    }
}
}
hipError_t bf16_projected_control_wave32(
    const Bf16ProjectedControlWave32Args& a,hipStream_t stream) noexcept {
    using namespace projected_control_wave32_detail;
    if(!stream || (a.tokens!=5 && a.tokens!=6))return hipErrorInvalidValue;
    const std::array<const void*,7> p{a.hidden,a.a_weight,a.b_weight,a.a_log,a.dt_bias,a.g,a.beta};
    const std::array<std::size_t,7> bytes{a.tokens*K*2ULL,N*K*2ULL,N*K*2ULL,N*4ULL,N*4ULL,a.tokens*N*4ULL,a.tokens*N*4ULL};
    for(unsigned i=0;i<p.size();++i) {
        const auto begin=reinterpret_cast<std::uintptr_t>(p[i]);
        if(!begin || begin%(i<3?2:4) || begin>std::numeric_limits<std::uintptr_t>::max()-bytes[i])
            return hipErrorInvalidValue;
        for(unsigned j=0;j<i;++j) {
            const auto other=reinterpret_cast<std::uintptr_t>(p[j]);
            if(begin<other+bytes[j] && other<begin+bytes[i])return hipErrorInvalidValue;
        }
    }
    hipLaunchKernelGGL(projected_control_paired_wave32_kernel,dim3(N,a.tokens),dim3(32),0,stream,
        a.hidden,a.a_weight,a.b_weight,a.a_log,a.dt_bias,a.g,a.beta);
    return hipGetLastError();
}
}
