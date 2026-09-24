#include "a8g128_q4g128_n16k16_qual.h"

#include <hip/hip_fp16.h>

namespace {
[[maybe_unused]] constexpr std::uint32_t kWave = 32, kK64 = 64, kG128 = 128;
constexpr std::uint32_t kM = 64, kN = 128, kThreads = 512;
using I32x2 = int __attribute__((ext_vector_type(2)));
using I32x8 = int __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));

struct Tile {
    std::uint32_t activation_low[8 * kM];
    std::uint32_t activation_high[8 * kM];
    std::uint32_t weights[8 * kN];
    std::uint16_t activation_scales[kM];
    std::uint16_t weight_scales[kN];
};
struct PingPong { Tile bank[2]; };
static_assert(sizeof(Tile) == 8576);
static_assert(sizeof(PingPong) == 17152);

[[maybe_unused]] __host__ __device__ constexpr std::size_t pair_index(
    std::uint32_t row, std::uint32_t group, std::uint32_t pair,
    std::uint32_t groups) {
    return (((static_cast<std::size_t>(row >> 4U) * groups + group) * 8U + pair) *
            16U + (row & 15U));
}
[[maybe_unused]] __host__ __device__ constexpr std::size_t scale_index(
    std::uint32_t row, std::uint32_t group, std::uint32_t groups) {
    return ((static_cast<std::size_t>(row >> 4U) * groups + group) * 16U +
            (row & 15U));
}
[[maybe_unused]] __device__ __forceinline__ std::uint32_t lds_index(
    std::uint32_t row, std::uint32_t word, std::uint32_t rows) {
    return word * rows + (row ^ (((word >> 1U) & 1U) << 4U));
}
[[maybe_unused]] __device__ __forceinline__ void barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
[[maybe_unused]] __device__ __forceinline__ std::uint32_t load32(const std::uint8_t* p,
                                                 std::size_t element) {
    return *reinterpret_cast<const std::uint32_t*>(p + element / 2U);
}
__device__ __forceinline__ float hfloat(std::uint16_t b) {
    return __half2float(__ushort_as_half(b));
}
[[maybe_unused]] __device__ __forceinline__ hip_bfloat16 poison() {
    hip_bfloat16 v; v.data = 0x7fc1U; return v;
}

template <bool Fused>
__device__ __forceinline__ float represented_value(const hip_bfloat16* input,
                                                    std::size_t token_base,
                                                    std::uint32_t column,
                                                    std::uint32_t columns,
                                                    std::uint32_t* status) {
    float value;
    if constexpr (Fused) {
        const float gate = static_cast<float>(input[token_base + column]);
        const float up = static_cast<float>(input[token_base + columns + column]);
        value = static_cast<float>(hip_bfloat16((gate / (1.0F + expf(-gate))) * up));
    } else {
        value = static_cast<float>(input[token_base + column]);
    }
    if (!isfinite(value)) { atomicOr(status, 1U); value = 0.0F; }
    return value;
}

template <bool Fused>
__device__ void quantize_body(const hip_bfloat16* input, std::uint8_t* low,
                              std::uint8_t* high, std::uint16_t* scales,
                              std::uint32_t* status, std::uint32_t columns) {
    const std::uint32_t group = blockIdx.x, token = blockIdx.y, lane = threadIdx.x;
    const std::size_t input_stride = Fused ? 2U * columns : columns;
    const std::size_t token_base = static_cast<std::size_t>(token) * input_stride;
    float values[4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
        values[i] = represented_value<Fused>(input, token_base,
                                             group * kG128 + lane * 4U + i,
                                             columns, status);
    float maximum = 0.0F;
#pragma unroll
    for (float v : values) maximum = fmaxf(maximum, fabsf(v));
#pragma unroll
    for (unsigned width = 16; width; width >>= 1U)
        maximum = fmaxf(maximum, __shfl_down(maximum, width, kWave));
    std::uint16_t sb = 0;
    if (lane == 0) {
        if (maximum != 0.0F) {
            sb = __half_as_ushort(__float2half_rn(maximum / 127.0F));
            if (sb == 0) sb = 1;
            if (!isfinite(hfloat(sb))) { atomicOr(status, 2U); sb = 0; }
        }
        scales[static_cast<std::size_t>(token) * (columns / kG128) + group] = sb;
    }
    sb = static_cast<std::uint16_t>(__shfl(static_cast<unsigned>(sb), 0, kWave));
    const float scale = hfloat(sb);
    int q[4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
        q[i] = scale == 0.0F ? 0 : max(-127, min(127, __float2int_rn(values[i] / scale)));
    const std::size_t packed = (static_cast<std::size_t>(token) * columns +
                                group * kG128 + lane * 4U) / 2U;
#pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        const int q0 = q[2*i], q1 = q[2*i+1];
        const int l0 = q0 & 15, l1 = q1 & 15;
        const int h0 = (q0 - l0) / 16, h1 = (q1 - l1) / 16;
        low[packed+i] = static_cast<std::uint8_t>(l0 | (l1 << 4));
        high[packed+i] = static_cast<std::uint8_t>((h0 & 15) | ((h1 & 15) << 4));
    }
}
} // namespace

extern "C" __global__ __launch_bounds__(kWave) void a8g128_q4g128_quantize(
    const hip_bfloat16* input, std::uint8_t* low, std::uint8_t* high,
    std::uint16_t* scales, std::uint32_t* status, std::uint32_t tokens,
    std::uint32_t columns) {
    (void)tokens; quantize_body<false>(input, low, high, scales, status, columns);
}
extern "C" __global__ __launch_bounds__(kWave) void fused_silu_a8g128_q4g128_quantize(
    const hip_bfloat16* input, std::uint8_t* low, std::uint8_t* high,
    std::uint16_t* scales, std::uint32_t* status, std::uint32_t tokens,
    std::uint32_t columns) {
    (void)tokens; quantize_body<true>(input, low, high, scales, status, columns);
}

extern "C" __global__ __launch_bounds__(kThreads)
void a8g128_q4g128_n16k16_prefill(
    const std::uint8_t* alow, const std::uint8_t* ahigh,
    const std::uint16_t* ascales, const std::uint32_t* status,
    const std::uint64_t* wpairs, const std::uint16_t* wscales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    (void)tokens;
    __shared__ PingPong tiles;
    const unsigned th=threadIdx.x,lane=th&31U,wave=th>>5U,axis=lane&15U,lg=lane>>4U;
    const unsigned wm=wave>>2U,wn=wave&3U,m0=blockIdx.y*kM,n0=blockIdx.x*kN;
    const unsigned groups=columns/kG128, arow=th>>3U, aword=th&7U;
    const unsigned token=m0+arow, ai=lds_index(arow,aword,kM);
    if (*status) {
#pragma unroll
        for(unsigned nh=0;nh<2;++nh) for(unsigned e=0;e<8;++e)
            output[static_cast<std::size_t>(m0+wm*16+lg*8+e)*rows+
                   n0+nh*64+wn*16+axis]=poison();
        return;
    }
    const unsigned lt=th>>6U,lp=(th>>4U)&3U,ll=th&15U,lr=lt*16U+ll;
    const unsigned row=n0+lr, lword=lp*2U;
    auto publish_payload = [&](Tile& dst, unsigned slab, std::uint32_t la,
                               std::uint32_t ha, std::uint64_t w) {
        dst.activation_low[ai]=la; dst.activation_high[ai]=ha;
        dst.weights[lds_index(lr,lword,kN)]=static_cast<std::uint32_t>(w);
        dst.weights[lds_index(lr,lword+1,kN)]=static_cast<std::uint32_t>(w>>32U);
        Tile& scale_dst=tiles.bank[(slab/2U)&1U];
        if ((slab&1U)==0U && th<kM)
            scale_dst.activation_scales[th]=ascales[static_cast<std::size_t>(m0+th)*groups+slab/2U];
        if ((slab&1U)==0U && th<kN)
            scale_dst.weight_scales[th]=wscales[scale_index(n0+th,slab/2U,groups)];
    };
    auto load_payload = [&](unsigned slab, std::uint32_t& la, std::uint32_t& ha,
                            std::uint64_t& w) {
        const std::size_t ae=static_cast<std::size_t>(token)*columns+slab*kK64+aword*8U;
        la=load32(alow,ae); ha=load32(ahigh,ae);
        w=wpairs[pair_index(row,slab/2U,(slab&1U)*4U+lp,groups)];
    };
    std::uint32_t la,ha; std::uint64_t w;
    load_payload(0,la,ha,w); publish_payload(tiles.bank[0],0,la,ha,w); barrier();
    F32x8 total[2]{}; I32x8 low[2]{}, high[2]{};
    const unsigned slabs=groups*2U;
    for(unsigned slab=0;slab<slabs;++slab) {
        const bool has_next=slab+1U<slabs;
        std::uint32_t nla=0,nha=0; std::uint64_t nw=0;
        if(has_next) load_payload(slab+1U,nla,nha,nw);
        Tile& cur=tiles.bank[slab&1U];
        if((slab&1U)==0U) { low[0]={};low[1]={};high[0]={};high[1]={}; }
#pragma unroll
        for(unsigned half=0;half<2;++half) {
            I32x2 a0{},a1{},b[2]{}; const unsigned word=half*4U+lg*2U;
            const unsigned ar=wm*16U+axis, ix=lds_index(ar,word,kM);
            a0[0]=cur.activation_low[ix]; a0[1]=cur.activation_low[lds_index(ar,word+1,kM)];
            a1[0]=cur.activation_high[ix];a1[1]=cur.activation_high[lds_index(ar,word+1,kM)];
#pragma unroll
            for(unsigned nh=0;nh<2;++nh){const unsigned wr=nh*64U+wn*16U+axis;
                b[nh][0]=cur.weights[lds_index(wr,word,kN)];
                b[nh][1]=cur.weights[lds_index(wr,word+1,kN)];}
            low[0]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,a0,true,b[0],low[0],false);
            high[0]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a1,true,b[0],high[0],false);
            low[1]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,a0,true,b[1],low[1],false);
            high[1]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a1,true,b[1],high[1],false);
        }
        if((slab&1U)!=0U) {
            Tile& scale_bank=tiles.bank[(slab/2U)&1U];
#pragma unroll
            for(unsigned nh=0;nh<2;++nh){const unsigned wr=nh*64U+wn*16U+axis;
                const float ws=hfloat(scale_bank.weight_scales[wr]);
#pragma unroll
                for(unsigned e=0;e<8;++e){const unsigned mr=wm*16U+lg*8U+e;
                    total[nh][e]=fmaf(float(low[nh][e]+16*high[nh][e]),
                        hfloat(scale_bank.activation_scales[mr])*ws,total[nh][e]);}}
        }
        // Scale arrays use logical-G128 parity, independently from the K64
        // payload parity.  Successor group g+1 therefore writes the opposite
        // scale array and cannot race group g's reads.
        if(has_next){publish_payload(tiles.bank[(slab+1U)&1U],slab+1U,nla,nha,nw);barrier();}
    }
#pragma unroll
    for(unsigned nh=0;nh<2;++nh) for(unsigned e=0;e<8;++e)
        output[static_cast<std::size_t>(m0+wm*16+lg*8+e)*rows+
               n0+nh*64+wn*16+axis]=hip_bfloat16(total[nh][e]);
#else
    (void)alow;(void)ahigh;(void)ascales;(void)status;(void)wpairs;(void)wscales;
    (void)output;(void)tokens;(void)rows;(void)columns;
#endif
}

extern "C" __global__ __launch_bounds__(kWave)
void a8g128_q4g128_n16k16_wave(
    const std::uint8_t* alow, const std::uint8_t* ahigh,
    const std::uint16_t* ascales, const std::uint32_t* status,
    const std::uint64_t* wpairs, const std::uint16_t* wscales,
    hip_bfloat16* output, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned lane=threadIdx.x,axis=lane&15U,lg=lane>>4U;
    const unsigned m0=blockIdx.y*16U,n0=blockIdx.x*16U,groups=columns/kG128;
    if(*status){for(unsigned e=0;e<8;++e){unsigned m=m0+lg*8+e;if(m<tokens)output[std::size_t(m)*rows+n0+axis]=poison();}return;}
    F32x8 total{};
    for(unsigned g=0;g<groups;++g){I32x8 low{},high{};
#pragma unroll
        for(unsigned pair=0;pair<4;++pair){
            I32x2 a0{},a1{},b{}; const unsigned p=pair*2U+lg,k=g*kG128+p*16U;
#pragma unroll
            for(unsigned j=0;j<2;++j){unsigned m=m0+axis;if(m<tokens){a0[j]=load32(alow,std::size_t(m)*columns+k+j*8U);a1[j]=load32(ahigh,std::size_t(m)*columns+k+j*8U);}}
            const auto ww=wpairs[pair_index(n0+axis,g,p,groups)];b[0]=ww;b[1]=ww>>32U;
            low=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,a0,true,b,low,false);
            high=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a1,true,b,high,false);
        }
        const float ws=hfloat(wscales[scale_index(n0+axis,g,groups)]);
        for(unsigned e=0;e<8;++e){unsigned m=m0+lg*8+e;if(m<tokens)total[e]=fmaf(float(low[e]+16*high[e]),hfloat(ascales[std::size_t(m)*groups+g])*ws,total[e]);}
    }
    for(unsigned e=0;e<8;++e){unsigned m=m0+lg*8+e;if(m<tokens)output[std::size_t(m)*rows+n0+axis]=hip_bfloat16(total[e]);}
#else
    (void)alow;(void)ahigh;(void)ascales;(void)status;(void)wpairs;(void)wscales;
    (void)output;(void)tokens;(void)rows;(void)columns;
#endif
}

namespace {
bool aligned(const void* p,std::size_t n){return p&&(reinterpret_cast<std::uintptr_t>(p)&(n-1U))==0;}
bool overlap(const void*a,std::size_t an,const void*b,std::size_t bn){auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<y+bn&&y<x+an;}
}
hipError_t a8g128_q4g128_qualification_run(
    const A8G128Q4G128QualificationArgs& a, hipStream_t stream) noexcept {
    const bool token_ok=a.tokens==1U||a.tokens==2U||a.tokens==3U||a.tokens==4U||
        a.tokens==8U||a.tokens==12U||a.tokens==16U||a.tokens==24U||
        a.tokens==36U||a.tokens==48U||a.tokens==2048U;
    const bool shape_ok=(a.rows==12288U&&a.columns==5120U&&!a.fused_silu)||
        (a.rows==5120U&&a.columns==6144U&&!a.fused_silu)||
        (a.rows==5120U&&a.columns==17408U&&a.fused_silu);
    if(!token_ok||!shape_ok||
       !aligned(a.input,2)||!aligned(a.activation_low,4)||!aligned(a.activation_high,4)||
       !aligned(a.activation_scales,2)||!aligned(a.status,4)||!aligned(a.weight_pairs,8)||
       !aligned(a.weight_scales,2)||!aligned(a.output,2)) return hipErrorInvalidValue;
    const std::size_t ap=std::size_t(a.tokens)*a.columns/2U,as=std::size_t(a.tokens)*(a.columns/128U)*2U;
    const std::size_t wc=std::size_t(a.rows)*a.columns/2U,ws=std::size_t(a.rows)*(a.columns/128U)*2U;
    const std::size_t ob=std::size_t(a.tokens)*a.rows*2U,ib=std::size_t(a.tokens)*a.columns*2U*(a.fused_silu?2U:1U);
    if(a.activation_plane_bytes!=ap||a.activation_scale_bytes!=as||a.weight_code_bytes!=wc||a.weight_scale_bytes!=ws||a.output_bytes!=ob)return hipErrorInvalidValue;
    const void*p[]={a.input,a.activation_low,a.activation_high,a.activation_scales,a.status,a.weight_pairs,a.weight_scales,a.output};
    const std::size_t z[]={ib,ap,ap,as,4,wc,ws,ob};for(unsigned i=0;i<8;++i)for(unsigned j=i+1;j<8;++j)if(overlap(p[i],z[i],p[j],z[j]))return hipErrorInvalidValue;
    hipError_t e=hipMemsetAsync(a.status,0,4,stream);if(e!=hipSuccess)return e;
    dim3 qgrid(a.columns/128U,a.tokens);
    if(a.fused_silu)hipLaunchKernelGGL(fused_silu_a8g128_q4g128_quantize,qgrid,dim3(32),0,stream,a.input,a.activation_low,a.activation_high,a.activation_scales,a.status,a.tokens,a.columns);
    else hipLaunchKernelGGL(a8g128_q4g128_quantize,qgrid,dim3(32),0,stream,a.input,a.activation_low,a.activation_high,a.activation_scales,a.status,a.tokens,a.columns);
    e=hipGetLastError();if(e!=hipSuccess)return e;
    if(a.tokens==2048U)hipLaunchKernelGGL(a8g128_q4g128_n16k16_prefill,dim3(a.rows/128U,a.tokens/64U),dim3(512),0,stream,a.activation_low,a.activation_high,a.activation_scales,a.status,a.weight_pairs,a.weight_scales,a.output,a.tokens,a.rows,a.columns);
    else hipLaunchKernelGGL(a8g128_q4g128_n16k16_wave,dim3(a.rows/16U,(a.tokens+15U)/16U),dim3(32),0,stream,a.activation_low,a.activation_high,a.activation_scales,a.status,a.weight_pairs,a.weight_scales,a.output,a.tokens,a.rows,a.columns);
    return hipGetLastError();
}
