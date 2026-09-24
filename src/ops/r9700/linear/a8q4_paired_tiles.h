#pragma once
#include "a8q4_pipeline_impl.h"
// Two M tiles share each weight payload; their ordered G64 sums remain independent.
namespace ninfer::ops::r9700::linear::detail {
struct PairedTileGroup { PipelineGroup first,tail; };
template<unsigned T>
__device__ __forceinline__ PairedTileGroup load_paired_tiles(const std::uint8_t* low,const std::uint8_t* high,
 const std::uint16_t* scales,const std::uint8_t* codes,const std::uint16_t* ws,
 unsigned row,unsigned group) {
 PairedTileGroup p; p.first=load_pipeline_group<5120,T>(low,high,scales,codes,ws,row,group);
 p.tail=p.first;
 const unsigned lane=threadIdx.x,axis=lane&15U;
 for(unsigned half=0;half<2;++half) {
   p.tail.low[half]={};p.tail.high[half]={};
   if(axis<T-16) {
     const unsigned k=group*64+half*32+(lane>>4U)*16;
     const auto offset=((16U+axis)*5120+k)/2;
     p.tail.low[half][0]=*reinterpret_cast<const int*>(low+offset);
     p.tail.low[half][1]=*reinterpret_cast<const int*>(low+offset+4);
     p.tail.high[half][0]=*reinterpret_cast<const int*>(high+offset);
     p.tail.high[half][1]=*reinterpret_cast<const int*>(high+offset+4);
   }
 }
 p.tail.activation_scale=lane<T-16?scales[(16+lane)*80+group]:0;
 return p;
}
template<unsigned T>
__device__ __forceinline__ void consume_paired_tiles(const PairedTileGroup& p,float ws,float a,float b,
 float (&first)[8],float (&tail)[T-16]) {
 consume_pipeline_group<16>(p.first,ws,a,first);
 PipelineI8 lo{},hi{};
 #pragma unroll
 for(unsigned h=0;h<2;++h) {
   lo=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,p.tail.low[h],true,p.first.weight[h],lo,false);
   hi=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,p.tail.high[h],true,p.first.weight[h],hi,false);
 }
 #pragma unroll
 for(unsigned t=0;t<T-16;++t) {
   const float scale=__shfl(b,t,32);
   tail[t]=fmaf(static_cast<float>(lo[t]+16*hi[t]),scale*ws,tail[t]);
 }
}
template<unsigned T>
__global__ __launch_bounds__(32) void gate_up_paired_tiles_kernel(
 const std::uint8_t* low,const std::uint8_t* high,const std::uint16_t* scales,
 const std::uint32_t* status,const std::uint8_t* codes,const std::uint16_t* ws,
 hip_bfloat16* output) {
 static_assert(T == 18 || T == 20 || T == 24);
 constexpr unsigned N=34816;const unsigned lane=threadIdx.x,row=blockIdx.x*16+(lane&15U);
 if(*status) {if(lane<16)for(unsigned t=0;t<T;++t){hip_bfloat16 x;x.data=0x7fc1;output[t*N+row]=x;}return;}
 float first[8]{},tail[T-16]{};PairedTileGroup current=load_paired_tiles<T>(low,high,scales,codes,ws,row,0);
 #pragma unroll 1
 for(unsigned g=0;g<79;++g) {
   float w=__half2float(__ushort_as_half(current.first.weight_scale));
   float a=__half2float(__ushort_as_half(current.first.activation_scale));
   float b=__half2float(__ushort_as_half(current.tail.activation_scale));
   asm volatile("" : "+v"(w),"+v"(a),"+v"(b));
   __builtin_amdgcn_sched_barrier(0);
   PairedTileGroup next=load_paired_tiles<T>(low,high,scales,codes,ws,row,g+1);
   __builtin_amdgcn_sched_barrier(0);
   consume_paired_tiles<T>(current,w,a,b,first,tail);
   __builtin_amdgcn_sched_barrier(0);current=next;
 }
 consume_paired_tiles<T>(current,__half2float(__ushort_as_half(current.first.weight_scale)),
   __half2float(__ushort_as_half(current.first.activation_scale)),
   __half2float(__ushort_as_half(current.tail.activation_scale)),first,tail);
 for(unsigned t=0;t<8;++t)output[((lane>>4U)*8+t)*N+row]=hip_bfloat16(first[t]);
 if(lane<16)for(unsigned t=0;t<T-16;++t)output[(16+t)*N+row]=hip_bfloat16(tail[t]);
}
}
