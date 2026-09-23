#pragma once
#include "ops/r9700/linear/r9700_linear.h"
#include <hip/hip_fp16.h>
#include <array>
#include <limits>

namespace ninfer::ops::r9700::linear::detail {
using ScaleGatherI2 = int __attribute__((ext_vector_type(2)));
using ScaleGatherI8 = int __attribute__((ext_vector_type(8)));

// Shared exact-order A8/Q4 G64 Linear mechanism. Shape policy remains with
// each explicit entry point; no generic shape-based production dispatch.
template<unsigned N, unsigned K, unsigned T>
__device__ __forceinline__ void a8q4_scale_gather_body(
    const std::uint8_t* low,const std::uint8_t* high,const std::uint16_t* scales,
    const std::uint32_t* status,const std::uint8_t* codes,
    const std::uint16_t* weight_scales,hip_bfloat16* output) {
    constexpr unsigned G=K/64;
    const unsigned lane=threadIdx.x,axis=lane&15U,half_lane=lane>>4U;
    const unsigned row=blockIdx.x*16+axis;
    if(*status!=0) {
        if(lane<16) for(unsigned t=0;t<T;++t) {
            hip_bfloat16 poison;poison.data=0x7fc1;output[t*N+row]=poison;
        }
        return;
    }
    float total[T]{};
    for(unsigned group=0;group<G;++group) {
        ScaleGatherI8 low_dot{},high_dot{};
#pragma unroll
        for(unsigned half=0;half<2;++half) {
            const unsigned k=group*64+half*32+half_lane*16;
            ScaleGatherI2 al{},ah{},b{};
            if(axis<T) {
                const std::size_t offset=(static_cast<std::size_t>(axis)*K+k)/2;
                al[0]=*reinterpret_cast<const int*>(low+offset);
                al[1]=*reinterpret_cast<const int*>(low+offset+4);
                ah[0]=*reinterpret_cast<const int*>(high+offset);
                ah[1]=*reinterpret_cast<const int*>(high+offset+4);
            }
            const std::size_t pair=(((static_cast<std::size_t>(row/16)*G+group)*4+
                                    (k%64)/16)*16+row%16);
            const auto packed=reinterpret_cast<const std::uint64_t*>(codes)[pair];
            b[0]=static_cast<int>(packed);b[1]=static_cast<int>(packed>>32);
            low_dot=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,al,true,b,low_dot,false);
            high_dot=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,ah,true,b,high_dot,false);
        }
        const float ws=__half2float(__ushort_as_half(weight_scales[(row/16)*G*16+group*16+row%16]));
        // One lane per token gathers the strided token-major FP16 scale. The
        // shuffle executes with the complete wave, after reconverging the load.
        // This removes separate token-scale load/wait phases without reordering
        // any per-output integer dot, FP32 scale product or group FMA.
        const float gathered=lane<T?__half2float(__ushort_as_half(scales[lane*G+group])):0.0F;
#pragma unroll
        for(unsigned t=0;t<T;++t) {
            const float as=__shfl(gathered,t,32);
            const int combined=low_dot[t]+16*high_dot[t];
            total[t]=fmaf(static_cast<float>(combined),as*ws,total[t]);
        }
    }
    if(lane<16) {
#pragma unroll
        for(unsigned t=0;t<T;++t)output[t*N+row]=hip_bfloat16(total[t]);
    }
}

template<unsigned N, unsigned K>
inline hipError_t validate_a8q4_scale_gather(
    const A8Q4G64CandidateArgs& a,hipStream_t stream) noexcept {
    constexpr unsigned G=K/64;
    const auto bytes=a8q4g64_activation_workspace_capacity_bytes(a.tokens,K);
    if((a.tokens!=5 && a.tokens!=6) || a.rows!=N || a.columns!=K ||
       a.padded_columns!=K || a.weight_code_bytes!=static_cast<std::size_t>(N)*K/2 ||
       a.weight_scale_bytes!=static_cast<std::size_t>(N)*G*2 ||
       a.activation_workspace_bytes!=bytes || stream==nullptr)return hipErrorInvalidValue;
    struct Plane {const void* p;std::size_t n,align;};
    const std::array<Plane,5> planes{{
        {a.input,static_cast<std::size_t>(a.tokens)*K*2,2},
        {a.weight_codes,a.weight_code_bytes,8},{a.weight_scales,a.weight_scale_bytes,2},
        {a.activation_workspace,bytes,256},
        {a.output,static_cast<std::size_t>(a.tokens)*N*2,2}}};
    for(unsigned i=0;i<5;++i) {
        const auto p=reinterpret_cast<std::uintptr_t>(planes[i].p);
        if(!p || p%planes[i].align || p>std::numeric_limits<std::uintptr_t>::max()-planes[i].n)
            return hipErrorInvalidValue;
        for(unsigned j=0;j<i;++j) {
            const auto q=reinterpret_cast<std::uintptr_t>(planes[j].p);
            if(p<q+planes[j].n && q<p+planes[i].n)return hipErrorInvalidValue;
        }
    }
    unsigned flags=0;if(hipStreamGetFlags(stream,&flags)!=hipSuccess)return hipErrorInvalidValue;

    return hipSuccess;
}
}
