#pragma once
#include "a8q4_scale_gather_impl.h"

namespace ninfer::ops::r9700::linear::detail {
// Exact ascending G64 arithmetic with one successor payload in flight. Shape
// selection and complete/prepared boundaries remain with their owning launchers.
using PipelineI2=ScaleGatherI2;
using PipelineI8=ScaleGatherI8;
struct PipelineGroup {
    PipelineI2 low[2],high[2],weight[2];
    std::uint16_t weight_scale,activation_scale;
};

template<unsigned K,unsigned T>
__device__ __forceinline__ PipelineGroup load_pipeline_group(
    const std::uint8_t* low,const std::uint8_t* high,const std::uint16_t* scales,
    const std::uint8_t* codes,const std::uint16_t* weight_scales,
    unsigned row,unsigned group) {
    constexpr unsigned G=K/64;
    const unsigned lane=threadIdx.x,axis=lane&15U,half_lane=lane>>4U;
    PipelineGroup value{};
#pragma unroll
    for(unsigned half=0;half<2;++half) {
        const unsigned k=group*64+half*32+half_lane*16;
        if(axis<T) {
            const std::size_t offset=(static_cast<std::size_t>(axis)*K+k)/2;
            value.low[half][0]=*reinterpret_cast<const int*>(low+offset);
            value.low[half][1]=*reinterpret_cast<const int*>(low+offset+4);
            value.high[half][0]=*reinterpret_cast<const int*>(high+offset);
            value.high[half][1]=*reinterpret_cast<const int*>(high+offset+4);
        }
        const std::size_t pair=(((static_cast<std::size_t>(row/16)*G+group)*4+
                                (k%64)/16)*16+row%16);
        const auto packed=reinterpret_cast<const std::uint64_t*>(codes)[pair];
        value.weight[half][0]=static_cast<int>(packed);
        value.weight[half][1]=static_cast<int>(packed>>32);
    }
    value.weight_scale=weight_scales[(row/16)*G*16+group*16+row%16];
    value.activation_scale=lane<T?scales[lane*G+group]:0;
    return value;
}

template<unsigned T>
__device__ __forceinline__ void consume_pipeline_group(
    const PipelineGroup& value,float ws,float gathered,float (&total)[T]) {
    PipelineI8 low_dot{},high_dot{};
#pragma unroll
    for(unsigned half=0;half<2;++half) {
        low_dot=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            false,value.low[half],true,value.weight[half],low_dot,false);
        high_dot=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true,value.high[half],true,value.weight[half],high_dot,false);
    }
#pragma unroll
    for(unsigned t=0;t<T;++t) {
        const float as=__shfl(gathered,t,32);
        const int combined=low_dot[t]+16*high_dot[t];
        total[t]=fmaf(static_cast<float>(combined),as*ws,total[t]);
    }
}

template<unsigned N,unsigned K,unsigned T>
__device__ __forceinline__ void a8q4_pipeline_body(
    const std::uint8_t* low,const std::uint8_t* high,const std::uint16_t* scales,
    const std::uint32_t* status,const std::uint8_t* codes,
    const std::uint16_t* weight_scales,hip_bfloat16* output) {
    constexpr unsigned G=K/64;
    const unsigned lane=threadIdx.x,row=blockIdx.x*16+(lane&15U);
    if(*status!=0) {
        if(lane<16)for(unsigned t=0;t<T;++t) {
            hip_bfloat16 poison;poison.data=0x7fc1;output[t*N+row]=poison;
        }
        return;
    }
    float total[T]{};
    PipelineGroup current=load_pipeline_group<K,T>(low,high,scales,codes,weight_scales,row,0);
#pragma unroll 1
    for(unsigned group=0;group<G-1;++group) {
        float ws=__half2float(__ushort_as_half(current.weight_scale));
        float gathered=__half2float(__ushort_as_half(current.activation_scale));
        // Materialize full FP32 values before issuing successor loads. Keeping
        // raw current FP16 scales live allowed the compiler to pack current and
        // successor into opposite halves of one VGPR, forcing an early wait.
        // This empty register-only boundary prevents that fold; it changes no
        // bits, arithmetic or memory ordering.
        asm volatile("" : "+v"(ws), "+v"(gathered));
        // Scheduling barriers constrain instruction motion, not memory
        // completion. ISA admission must demonstrate successor VMEM before
        // current compute without a wait that drains it prematurely.
        __builtin_amdgcn_sched_barrier(0);
        PipelineGroup next=load_pipeline_group<K,T>(low,high,scales,codes,weight_scales,row,group+1);
        __builtin_amdgcn_sched_barrier(0);
        consume_pipeline_group<T>(current,ws,gathered,total);
        __builtin_amdgcn_sched_barrier(0);
        current=next;
    }
    consume_pipeline_group<T>(current,__half2float(__ushort_as_half(current.weight_scale)),
                     __half2float(__ushort_as_half(current.activation_scale)),total);
    if(lane<16) {
#pragma unroll
        for(unsigned t=0;t<T;++t)output[t*N+row]=hip_bfloat16(total[t]);
    }
}

}
