#include "a8q4_n16k16_scalar_base_qual.h"

#include <hip/hip_fp16.h>

#include <cstdint>
#include <limits>

namespace {
using ninfer::ops::r9700::linear::A8Q4G64LinearArgs;
[[maybe_unused]] constexpr std::uint32_t Group=64;
constexpr std::uint32_t MTile=64,NTile=128,Threads=512;
using I32x2=int __attribute__((ext_vector_type(2)));
using I32x8=int __attribute__((ext_vector_type(8)));
using F32x8=float __attribute__((ext_vector_type(8)));
struct Tile { std::uint32_t al[8*MTile],ah[8*MTile],w[8*NTile];std::uint16_t as[MTile],ws[NTile];};
struct Banks { Tile bank[2];};
static_assert(sizeof(Tile)==8576&&sizeof(Banks)==17152);
[[maybe_unused]] __device__ __forceinline__ std::uint32_t li(std::uint32_t r,std::uint32_t w,std::uint32_t rows){return w*rows+(r^(((w>>1)&1)<<4));}
[[maybe_unused]] __device__ __forceinline__ void barrier(){__builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");__builtin_amdgcn_s_barrier_signal(-1);__builtin_amdgcn_s_barrier_wait(-1);__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");}
[[maybe_unused]] __device__ __forceinline__ std::uint32_t ld32(const std::uint8_t*base,std::uint32_t off){return *reinterpret_cast<const std::uint32_t*>(base+off);}
[[maybe_unused]] __device__ __forceinline__ std::uint64_t ld64(const std::uint8_t*base,std::uint32_t off){return *reinterpret_cast<const std::uint64_t*>(base+off);}
[[maybe_unused]] __device__ __forceinline__ std::uint16_t ld16(const std::uint16_t*base,std::uint32_t off){return *reinterpret_cast<const std::uint16_t*>(reinterpret_cast<const std::uint8_t*>(base)+off);}
[[maybe_unused]] __device__ __forceinline__ float hf(std::uint16_t x){return __half2float(__ushort_as_half(x));}
[[maybe_unused]] __device__ __forceinline__ hip_bfloat16 poison(){hip_bfloat16 x;x.data=0x7fc1;return x;}
[[maybe_unused]] __host__ __device__ constexpr std::uint32_t woff(std::uint32_t row,std::uint32_t g,std::uint32_t pair,std::uint32_t groups){return ((((row>>4)*groups+g)*4+pair)*16+(row&15))*8;}
[[maybe_unused]] __host__ __device__ constexpr std::uint32_t wsoff(std::uint32_t row,std::uint32_t g,std::uint32_t groups){return (((row>>4)*groups+g)*16+(row&15))*2;}

extern "C" __global__ __launch_bounds__(Threads) void a8q4_n16k16_scalar_base_p2048_m64n128(
 const std::uint8_t*al,const std::uint8_t*ah,const std::uint16_t*as,const std::uint32_t*status,
 const std::uint8_t*w,const std::uint16_t*ws,hip_bfloat16*out,std::uint32_t tokens,std::uint32_t rows,
 std::uint32_t columns,std::uint32_t padded){
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
 (void)tokens;(void)columns;__shared__ Banks tiles;
 const std::uint32_t th=threadIdx.x,lane=th&31,wave=th>>5,axis=lane&15,lg=lane>>4,wm=wave>>2,wn=wave&3;
 const std::uint32_t m0=blockIdx.y*MTile,n0=blockIdx.x*NTile,groups=padded/Group;
 const std::uint32_t ar=th>>3,aw=th&7,token=m0+ar,alds=li(ar,aw,MTile);
 if(*status!=0){
#pragma unroll
  for(std::uint32_t nh=0;nh<2;++nh){
#pragma unroll
   for(std::uint32_t e=0;e<8;++e)out[(std::size_t(m0+wm*16+lg*8+e)*rows)+(n0+nh*64+wn*16+axis)]=poison();
  }return;
 }
 const std::uint32_t lt=th>>6,lp=(th>>4)&3,ll=th&15,lrlocal=lt*16+ll,lr=n0+lrlocal,lword=lp*2;
 std::uint32_t aoff=token*(padded>>1)+aw*4;
 std::uint32_t woff_cur=woff(lr,0,lp,groups);
 std::uint32_t asoff=(m0+th)*groups*2;
 std::uint32_t wsoff_cur=wsoff(n0+th,0,groups);
 tiles.bank[0].al[alds]=ld32(al,aoff);tiles.bank[0].ah[alds]=ld32(ah,aoff);
 const std::uint64_t w0=ld64(w,woff_cur);
 tiles.bank[0].w[li(lrlocal,lword,NTile)]=std::uint32_t(w0);tiles.bank[0].w[li(lrlocal,lword+1,NTile)]=std::uint32_t(w0>>32);
 if(th<MTile)tiles.bank[0].as[th]=ld16(as,asoff);
 if(th<NTile)tiles.bank[0].ws[th]=ld16(ws,wsoff_cur);
 barrier();F32x8 totals[2]{};
 for(std::uint32_t g=0;g<groups;++g){
  const bool nextp=g+1<groups;std::uint32_t nal=0,nah=0;std::uint64_t nw=0;
  if(nextp){aoff+=32;woff_cur+=512;nal=ld32(al,aoff);nah=ld32(ah,aoff);nw=ld64(w,woff_cur);}
  Tile&cur=tiles.bank[g&1];I32x8 low[2]{},high[2]{};
#pragma unroll
  for(std::uint32_t half=0;half<2;++half){I32x2 avl{},avh{},bv[2]{};const std::uint32_t word=half*4+lg*2,rr=wm*16+axis,ai=li(rr,word,MTile);
   avl[0]=int(cur.al[ai]);avl[1]=int(cur.al[li(rr,word+1,MTile)]);avh[0]=int(cur.ah[ai]);avh[1]=int(cur.ah[li(rr,word+1,MTile)]);
#pragma unroll
   for(std::uint32_t nh=0;nh<2;++nh){const std::uint32_t wr=nh*64+wn*16+axis;bv[nh][0]=int(cur.w[li(wr,word,NTile)]);bv[nh][1]=int(cur.w[li(wr,word+1,NTile)]);}
   low[0]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,avl,true,bv[0],low[0],false);
   high[0]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,avh,true,bv[0],high[0],false);
   low[1]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,avl,true,bv[1],low[1],false);
   high[1]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,avh,true,bv[1],high[1],false);
  }
#pragma unroll
  for(std::uint32_t nh=0;nh<2;++nh){const std::uint32_t wr=nh*64+wn*16+axis;const float sw=hf(cur.ws[wr]);
#pragma unroll
   for(std::uint32_t e=0;e<8;++e){const std::uint32_t mr=wm*16+lg*8+e;const int dot=low[nh][e]+16*high[nh][e];totals[nh][e]=fmaf(float(dot),hf(cur.as[mr])*sw,totals[nh][e]);}
  }
  if(nextp){Tile&next=tiles.bank[(g+1)&1];next.al[alds]=nal;next.ah[alds]=nah;next.w[li(lrlocal,lword,NTile)]=std::uint32_t(nw);next.w[li(lrlocal,lword+1,NTile)]=std::uint32_t(nw>>32);
   if(th<MTile){asoff+=2;next.as[th]=ld16(as,asoff);}if(th<NTile){wsoff_cur+=32;next.ws[th]=ld16(ws,wsoff_cur);}barrier();}
 }
#pragma unroll
 for(std::uint32_t nh=0;nh<2;++nh){
#pragma unroll
  for(std::uint32_t e=0;e<8;++e)out[(std::size_t(m0+wm*16+lg*8+e)*rows)+(n0+nh*64+wn*16+axis)]=hip_bfloat16(totals[nh][e]);
 }
#else
 (void)al;(void)ah;(void)as;(void)status;(void)w;(void)ws;(void)out;(void)tokens;(void)rows;(void)columns;(void)padded;
#endif
}

bool mul(std::size_t a,std::size_t b,std::size_t&z){if(a&&b>std::numeric_limits<std::size_t>::max()/a)return false;z=a*b;return true;}
bool ov(const void*a,std::size_t an,const void*b,std::size_t bn){auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return !a||!b||x>UINTPTR_MAX-an||y>UINTPTR_MAX-bn||(x<y+bn&&y<x+an);}
bool shape(std::uint32_t n,std::uint32_t k){return (n==12288&&k==5120)||(n==5120&&k==6144)||(n==5120&&k==17408);}
}

hipError_t a8q4_n16k16_scalar_base_qualification(const A8Q4G64LinearArgs&a,hipStream_t stream) noexcept{
 if(!shape(a.rows,a.columns)||a.tokens!=2048||a.padded_columns!=a.columns||!a.activation_low_codes||!a.activation_high_codes||!a.activation_scales||!a.activation_status||!a.weight_codes||!a.weight_scales||!a.output||
    (reinterpret_cast<std::uintptr_t>(a.activation_low_codes)&3)||(reinterpret_cast<std::uintptr_t>(a.activation_high_codes)&3)||(reinterpret_cast<std::uintptr_t>(a.activation_scales)&1)||(reinterpret_cast<std::uintptr_t>(a.activation_status)&3)||(reinterpret_cast<std::uintptr_t>(a.weight_codes)&7)||(reinterpret_cast<std::uintptr_t>(a.weight_scales)&1)||(reinterpret_cast<std::uintptr_t>(a.output)&1))return hipErrorInvalidValue;
 std::size_t ap=0,asc=0,wc=0,wsc=0,ob=0;if(!mul(a.tokens,a.columns,ap)||!mul(a.tokens,a.columns/64,asc)||!mul(a.rows,a.columns,wc)||!mul(a.rows,a.columns/64,wsc)||!mul(a.tokens,a.rows,ob))return hipErrorInvalidValue;ap/=2;asc*=2;wc/=2;wsc*=2;ob*=2;
 if(a.activation_low_code_bytes!=ap||a.activation_high_code_bytes!=ap||a.activation_scale_bytes!=asc||a.weight_code_bytes!=wc||a.weight_scale_bytes!=wsc)return hipErrorInvalidValue;
 const void*p[]={a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output};const std::size_t z[]={ap,ap,asc,4,wc,wsc,ob};for(int i=0;i<7;++i)for(int j=i+1;j<7;++j)if(ov(p[i],z[i],p[j],z[j]))return hipErrorInvalidValue;
 hipLaunchKernelGGL(a8q4_n16k16_scalar_base_p2048_m64n128,dim3(a.rows/128,a.tokens/64),dim3(512),0,stream,a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output,a.tokens,a.rows,a.columns,a.padded_columns);return hipGetLastError();
}
