#include "a8q4_group_major_activation_qual.h"

#include <hip/hip_fp16.h>

namespace {
constexpr unsigned kWave=32,kGroup=64,kM=64,kN=128,kThreads=512;
using I32x2=int __attribute__((ext_vector_type(2)));
using I32x8=int __attribute__((ext_vector_type(8)));
using F32x8=float __attribute__((ext_vector_type(8)));
struct Tile { std::uint32_t al[8*kM],ah[8*kM],w[8*kN]; std::uint16_t as[kM],ws[kN]; };
struct Banks { Tile bank[2]; };
static_assert(sizeof(Tile)==8576);
static_assert(sizeof(Banks)==17152);

__host__ __device__ constexpr std::size_t gm_byte(unsigned g,unsigned token,
                                                   unsigned byte,unsigned tokens) {
    return (static_cast<std::size_t>(g)*tokens+token)*32U+byte;
}
__host__ __device__ constexpr std::size_t gm_scale(unsigned g,unsigned token,
                                                    unsigned tokens) {
    return static_cast<std::size_t>(g)*tokens+token;
}
__host__ __device__ constexpr std::size_t wpair(unsigned row,unsigned g,unsigned pair,unsigned groups){
    return (((static_cast<std::size_t>(row>>4)*groups+g)*4U+pair)*16U+(row&15U));
}
__host__ __device__ constexpr std::size_t wscale(unsigned row,unsigned g,unsigned groups){
    return ((static_cast<std::size_t>(row>>4)*groups+g)*16U+(row&15U));
}
__device__ __forceinline__ unsigned li(unsigned row,unsigned word,unsigned rows){
    return word*rows+(row^(((word>>1U)&1U)<<4U));
}
__device__ __forceinline__ std::uint32_t ld32(const std::uint8_t*p,std::size_t byte){
    return *reinterpret_cast<const std::uint32_t*>(p+byte);
}
__device__ __forceinline__ std::uint64_t ldw(const std::uint8_t*p,unsigned row,unsigned g,unsigned pair,unsigned groups){
    return reinterpret_cast<const std::uint64_t*>(p)[wpair(row,g,pair,groups)];
}
__device__ __forceinline__ float hf(std::uint16_t b){return __half2float(__ushort_as_half(b));}
__device__ __forceinline__ hip_bfloat16 nanbf(){hip_bfloat16 x;x.data=0x7fc1U;return x;}
__device__ __forceinline__ void barrier(){
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");
    __builtin_amdgcn_s_barrier_signal(-1);__builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");
}

template<bool Fused> __device__ void quantize(const hip_bfloat16*in,std::uint8_t*lo,
 std::uint8_t*hi,std::uint16_t*sc,std::uint32_t*status,unsigned tokens,unsigned columns){
    const unsigned g=blockIdx.x,t=blockIdx.y,lane=threadIdx.x,c0=g*kGroup+2U*lane,c1=c0+1U;
    const std::size_t base=static_cast<std::size_t>(t)*(Fused?2U*columns:columns);
    auto value=[&](unsigned c){float v;if constexpr(Fused){float a=float(in[base+c]),b=float(in[base+columns+c]);v=float(hip_bfloat16((a/(1.0F+expf(-a)))*b));}else v=c<columns?float(in[base+c]):0.0F;if(!isfinite(v)){atomicOr(status,1U);v=0;}return v;};
    float a=value(c0),b=value(c1),mx=fmaxf(fabsf(a),fabsf(b));
#pragma unroll
    for(unsigned w=16;w;w>>=1)mx=fmaxf(mx,__shfl_down(mx,w,kWave));
    std::uint16_t sb=0;if(lane==0){if(mx!=0){sb=__half_as_ushort(__float2half_rn(mx/127.0F));if(sb==0)sb=1;if(!isfinite(hf(sb))){atomicOr(status,2U);sb=0;}}sc[gm_scale(g,t,tokens)]=sb;}
    sb=static_cast<std::uint16_t>(__shfl(static_cast<unsigned>(sb),0,kWave));float s=hf(sb);
    int q0=s==0?0:max(-127,min(127,__float2int_rn(a/s))),q1=s==0?0:max(-127,min(127,__float2int_rn(b/s)));
    int l0=q0&15,l1=q1&15,h0=(q0-l0)/16,h1=(q1-l1)/16;auto o=gm_byte(g,t,lane,tokens);
    lo[o]=static_cast<std::uint8_t>(l0|(l1<<4));hi[o]=static_cast<std::uint8_t>((h0&15)|((h1&15)<<4));
}

extern "C" __global__ __launch_bounds__(kWave) void a8g64_group_major_quantize(const hip_bfloat16*in,std::uint8_t*lo,std::uint8_t*hi,std::uint16_t*sc,std::uint32_t*st,unsigned tokens,unsigned columns){quantize<false>(in,lo,hi,sc,st,tokens,columns);}
extern "C" __global__ __launch_bounds__(kWave) void fused_silu_a8g64_group_major_quantize(const hip_bfloat16*in,std::uint8_t*lo,std::uint8_t*hi,std::uint16_t*sc,std::uint32_t*st,unsigned tokens,unsigned columns){quantize<true>(in,lo,hi,sc,st,tokens,columns);}

extern "C" __global__ __launch_bounds__(kThreads) void a8q4_group_major_prefill(
 const std::uint8_t*al,const std::uint8_t*ah,const std::uint16_t*as,const std::uint32_t*st,
 const std::uint8_t*wc,const std::uint16_t*ws,hip_bfloat16*out,unsigned tokens,unsigned rows,unsigned columns){
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    (void)tokens;__shared__ Banks tiles;unsigned th=threadIdx.x,lane=th&31,wave=th>>5,axis=lane&15,lg=lane>>4;
    unsigned wm=wave>>2,wn=wave&3,m0=blockIdx.y*kM,n0=blockIdx.x*kN,groups=columns/kGroup;
    unsigned ar=th>>3,word=th&7,token=m0+ar,ai=li(ar,word,kM);
    if(*st){for(unsigned nh=0;nh<2;++nh)for(unsigned e=0;e<8;++e)out[std::size_t(m0+wm*16+lg*8+e)*rows+n0+nh*64+wn*16+axis]=nanbf();return;}
    unsigned lt=th>>6,lp=(th>>4)&3,ll=th&15,lr=lt*16+ll,row=n0+lr,lword=lp*2;
    auto load_a=[&](unsigned g,std::uint32_t&x,std::uint32_t&y){auto b=gm_byte(g,token,word*4U,tokens);x=ld32(al,b);y=ld32(ah,b);};
    auto publish=[&](Tile&d,unsigned g,std::uint32_t x,std::uint32_t y,std::uint64_t w){d.al[ai]=x;d.ah[ai]=y;d.w[li(lr,lword,kN)]=w;d.w[li(lr,lword+1,kN)]=w>>32;if(th<kM)d.as[th]=as[gm_scale(g,m0+th,tokens)];if(th<kN)d.ws[th]=ws[wscale(n0+th,g,groups)];};
    std::uint32_t x,y;load_a(0,x,y);publish(tiles.bank[0],0,x,y,ldw(wc,row,0,lp,groups));barrier();F32x8 total[2]{};
    for(unsigned g=0;g<groups;++g){bool nextp=g+1<groups;std::uint32_t nx=0,ny=0;std::uint64_t nw=0;if(nextp){load_a(g+1,nx,ny);nw=ldw(wc,row,g+1,lp,groups);}Tile&cur=tiles.bank[g&1];I32x8 low[2]{},high[2]{};
#pragma unroll
      for(unsigned half=0;half<2;++half){I32x2 a0{},a1{},b[2]{};unsigned fw=half*4+lg*2,rr=wm*16+axis,ix=li(rr,fw,kM);a0[0]=cur.al[ix];a0[1]=cur.al[li(rr,fw+1,kM)];a1[0]=cur.ah[ix];a1[1]=cur.ah[li(rr,fw+1,kM)];
#pragma unroll
        for(unsigned nh=0;nh<2;++nh){unsigned wr=nh*64+wn*16+axis;b[nh][0]=cur.w[li(wr,fw,kN)];b[nh][1]=cur.w[li(wr,fw+1,kN)];}
        low[0]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,a0,true,b[0],low[0],false);high[0]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a1,true,b[0],high[0],false);low[1]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,a0,true,b[1],low[1],false);high[1]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a1,true,b[1],high[1],false);}
      for(unsigned nh=0;nh<2;++nh){unsigned rr=nh*64+wn*16+axis;float sw=hf(cur.ws[rr]);for(unsigned e=0;e<8;++e){unsigned mt=wm*16+lg*8+e;total[nh][e]=fmaf(float(low[nh][e]+16*high[nh][e]),hf(cur.as[mt])*sw,total[nh][e]);}}
      if(nextp){publish(tiles.bank[(g+1)&1],g+1,nx,ny,nw);barrier();}}
    for(unsigned nh=0;nh<2;++nh)for(unsigned e=0;e<8;++e)out[std::size_t(m0+wm*16+lg*8+e)*rows+n0+nh*64+wn*16+axis]=hip_bfloat16(total[nh][e]);
#else
    (void)al;(void)ah;(void)as;(void)st;(void)wc;(void)ws;(void)out;(void)tokens;(void)rows;(void)columns;
#endif
}

extern "C" __global__ __launch_bounds__(kWave) void a8q4_group_major_wave(
 const std::uint8_t*al,const std::uint8_t*ah,const std::uint16_t*as,const std::uint32_t*st,
 const std::uint8_t*wc,const std::uint16_t*ws,hip_bfloat16*out,unsigned tokens,unsigned rows,unsigned columns){
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
 unsigned lane=threadIdx.x,axis=lane&15,lg=lane>>4,m0=blockIdx.y*16,n0=blockIdx.x*16,groups=columns/64;if(*st){for(unsigned e=0;e<8;++e){unsigned m=m0+lg*8+e;if(m<tokens)out[std::size_t(m)*rows+n0+axis]=nanbf();}return;}F32x8 total{};
 for(unsigned g=0;g<groups;++g){I32x8 low{},high{};for(unsigned half=0;half<2;++half){I32x2 a0{},a1{},b{};unsigned k=half*32+lg*16;for(unsigned j=0;j<2;++j){unsigned m=m0+axis;if(m<tokens){auto o=gm_byte(g,m,k/2+j*4,tokens);a0[j]=ld32(al,o);a1[j]=ld32(ah,o);}}auto w=ldw(wc,n0+axis,g,half*2+lg,groups);b[0]=w;b[1]=w>>32;low=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,a0,true,b,low,false);high=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a1,true,b,high,false);}float sw=hf(ws[wscale(n0+axis,g,groups)]);for(unsigned e=0;e<8;++e){unsigned m=m0+lg*8+e;if(m<tokens)total[e]=fmaf(float(low[e]+16*high[e]),hf(as[gm_scale(g,m,tokens)])*sw,total[e]);}}
 for(unsigned e=0;e<8;++e){unsigned m=m0+lg*8+e;if(m<tokens)out[std::size_t(m)*rows+n0+axis]=hip_bfloat16(total[e]);}
#else
 (void)al;(void)ah;(void)as;(void)st;(void)wc;(void)ws;(void)out;(void)tokens;(void)rows;(void)columns;
#endif
}

bool aligned(const void*p,std::size_t a){return p&&(reinterpret_cast<std::uintptr_t>(p)&(a-1))==0;}
bool overlap(const void*a,std::size_t an,const void*b,std::size_t bn){auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<y+bn&&y<x+an;}
}

hipError_t a8q4_group_major_activation_run(const A8Q4GroupMajorArgs&a,hipStream_t stream) noexcept {
 bool shape=(a.rows==12288&&a.columns==5120&&!a.fused_silu)||(a.rows==5120&&a.columns==6144&&!a.fused_silu)||(a.rows==5120&&a.columns==17408&&a.fused_silu);
 bool tok=a.tokens==1||a.tokens==2||a.tokens==3||a.tokens==4||a.tokens==8||a.tokens==12||a.tokens==16||a.tokens==24||a.tokens==36||a.tokens==48||a.tokens==2048;
 std::size_t ap=std::size_t(a.tokens)*a.columns/2,as=std::size_t(a.tokens)*(a.columns/64)*2,wc=std::size_t(a.rows)*a.columns/2,ws=std::size_t(a.rows)*(a.columns/64)*2,ob=std::size_t(a.tokens)*a.rows*2,ib=std::size_t(a.tokens)*a.columns*2*(a.fused_silu?2:1);
 if(!shape||!tok||!aligned(a.input,2)||!aligned(a.activation_low,4)||!aligned(a.activation_high,4)||!aligned(a.activation_scales,2)||!aligned(a.activation_status,4)||!aligned(a.weight_codes,8)||!aligned(a.weight_scales,2)||!aligned(a.output,2)||a.input_bytes!=ib||a.activation_plane_bytes!=ap||a.activation_scale_bytes!=as||a.weight_code_bytes!=wc||a.weight_scale_bytes!=ws||a.output_bytes!=ob)return hipErrorInvalidValue;
 const void*p[]={a.input,a.activation_low,a.activation_high,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output};std::size_t z[]={ib,ap,ap,as,4,wc,ws,ob};for(unsigned i=0;i<8;++i)for(unsigned j=i+1;j<8;++j)if(overlap(p[i],z[i],p[j],z[j]))return hipErrorInvalidValue;
 auto e=hipMemsetAsync(a.activation_status,0,4,stream);if(e!=hipSuccess)return e;dim3 q(a.columns/64,a.tokens);if(a.fused_silu)hipLaunchKernelGGL(fused_silu_a8g64_group_major_quantize,q,dim3(32),0,stream,a.input,a.activation_low,a.activation_high,a.activation_scales,a.activation_status,a.tokens,a.columns);else hipLaunchKernelGGL(a8g64_group_major_quantize,q,dim3(32),0,stream,a.input,a.activation_low,a.activation_high,a.activation_scales,a.activation_status,a.tokens,a.columns);e=hipGetLastError();if(e!=hipSuccess)return e;
 if(a.tokens==2048)hipLaunchKernelGGL(a8q4_group_major_prefill,dim3(a.rows/128,a.tokens/64),dim3(512),0,stream,a.activation_low,a.activation_high,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output,a.tokens,a.rows,a.columns);else hipLaunchKernelGGL(a8q4_group_major_wave,dim3(a.rows/16,(a.tokens+15)/16),dim3(32),0,stream,a.activation_low,a.activation_high,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output,a.tokens,a.rows,a.columns);return hipGetLastError();
}
