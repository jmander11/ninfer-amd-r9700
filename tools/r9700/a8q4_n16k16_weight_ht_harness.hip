#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ninfer::ops::r9700::linear::A8Q4G64LinearArgs;
constexpr std::uint32_t T=2048, N=5120, K=17408, G=64, Groups=K/G;
constexpr unsigned Trials=7, Iterations=128;

void ok(hipError_t e,const char* w){if(e!=hipSuccess)throw std::runtime_error(std::string(w)+": "+hipGetErrorString(e));}
std::uint16_t hb(float f){__half h=__float2half_rn(f);std::uint16_t b;std::memcpy(&b,&h,2);return b;}
float hf(std::uint16_t b){__half h;std::memcpy(&h,&b,2);return __half2float(h);}
std::int8_t av(unsigned t,unsigned k){return static_cast<std::int8_t>(int((t*13U+k*17U)%255U)-127);}
std::int8_t wv(unsigned n,unsigned k){unsigned x=n*1664525U+k*1013904223U+(n^(k>>3U))*2246822519U;x^=x>>13U;x^=x>>22U;unsigned q=x&15U;return static_cast<std::int8_t>(q<8?q:int(q)-16);}
std::uint8_t nib(std::int8_t q){return static_cast<std::uint8_t>(q)&15U;}
std::size_t wi(unsigned n,unsigned g,unsigned p){return (((std::size_t(n>>4U)*Groups+g)*4U+p)*16U+(n&15U));}
std::size_t si(unsigned n,unsigned g){return (std::size_t(n>>4U)*Groups+g)*16U+(n&15U);}
template<class X>struct D{X*p{};std::size_t n;explicit D(std::size_t z):n(z){ok(hipMalloc(&p,z*sizeof(X)),"malloc");}~D(){if(p)(void)hipFree(p);}D(const D&)=delete;};
template<class X>void put(D<X>&d,const std::vector<X>&h){ok(hipMemcpy(d.p,h.data(),h.size()*sizeof(X),hipMemcpyHostToDevice),"upload");}
template<class X>std::vector<X>get(const X*p,std::size_t n){std::vector<X>h(n);ok(hipMemcpy(h.data(),p,n*sizeof(X),hipMemcpyDeviceToHost),"download");return h;}
struct Stream{hipStream_t s{};Stream(){ok(hipStreamCreate(&s),"stream");}~Stream(){if(s)(void)hipStreamDestroy(s);}};
void require_device(){int ordinal=-1;ok(hipGetDevice(&ordinal),"get device");if(ordinal!=0)throw std::runtime_error("device ordinal is not 0");hipDeviceProp_t p{};ok(hipGetDeviceProperties(&p,ordinal),"device properties");if(std::string(p.gcnArchName)!="gfx1201"||std::string(p.name).find("R9700")==std::string::npos)throw std::runtime_error("device is not R9700/gfx1201");}
struct Kernarg{const std::uint8_t*lo;const std::uint8_t*hi;const std::uint16_t*as;const std::uint32_t*status;const std::uint8_t*w;const std::uint16_t*ws;hip_bfloat16*out;std::uint32_t tokens,rows,columns,padded;};
static_assert(sizeof(Kernarg)==72);
constexpr const char* Symbol="_ZN6ninfer3ops5r97006linear12_GLOBAL__N_133a8q4g64_linear_prefill_cta_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjjj";
struct Module{hipModule_t m{};hipFunction_t f{};explicit Module(const char*path){ok(hipModuleLoad(&m,path),"module load");ok(hipModuleGetFunction(&f,m,Symbol),"module function");}~Module(){if(m)(void)hipModuleUnload(m);}};
Module* baseline_module{};Module* candidate_module{};
struct Images{std::vector<std::uint8_t> lo,hi,w;std::vector<std::uint16_t>as,ws;};
Images images(){Images x;x.lo.resize(std::size_t(T)*K/2);x.hi.resize(x.lo.size());
 for(unsigned t=0;t<T;++t)for(unsigned k=0;k<K;k+=2){auto a0=av(t,k),a1=av(t,k+1);auto l0=nib(a0),l1=nib(a1);auto h0=nib(static_cast<std::int8_t>((int(a0)-l0)/16)),h1=nib(static_cast<std::int8_t>((int(a1)-l1)/16));auto o=(std::size_t(t)*K+k)/2;x.lo[o]=l0|(l1<<4);x.hi[o]=h0|(h1<<4);}
 x.w.resize(std::size_t(N)*K/2);for(unsigned n=0;n<N;++n)for(unsigned g=0;g<Groups;++g)for(unsigned p=0;p<4;++p){std::uint64_t q=0;for(unsigned b=0;b<8;++b){unsigned k=g*64+p*16+b*2;std::uint8_t v=nib(wv(n,k))|(nib(wv(n,k+1))<<4);q|=std::uint64_t(v)<<(b*8);}std::memcpy(x.w.data()+wi(n,g,p)*8,&q,8);}
 x.as.resize(std::size_t(T)*Groups);for(unsigned t=0;t<T;++t)for(unsigned g=0;g<Groups;++g)x.as[std::size_t(t)*Groups+g]=hb(float(1+((t*3+g*5)&7))/8);
 x.ws.resize(std::size_t(N)*Groups);for(unsigned n=0;n<N;++n)for(unsigned g=0;g<Groups;++g)x.ws[si(n,g)]=hb(float(1+(((n*29)^(g*11)^(n>>(g&7)))&7))/8);return x;}
struct Fixture{Images h;D<std::uint8_t>lo,hi,w;D<std::uint16_t>as,ws;D<std::uint32_t>st;D<hip_bfloat16>inc,cand;
 Fixture():h(images()),lo(h.lo.size()),hi(h.hi.size()),w(h.w.size()),as(h.as.size()),ws(h.ws.size()),st(1),inc(std::size_t(T)*N+64),cand(std::size_t(T)*N+64){put(lo,h.lo);put(hi,h.hi);put(w,h.w);put(as,h.as);put(ws,h.ws);std::vector<std::uint32_t>z(1);put(st,z);}
 A8Q4G64LinearArgs args(hip_bfloat16*out){return{lo.p,lo.n,hi.p,hi.n,as.p,as.n*2,st.p,w.p,w.n,ws.p,ws.n*2,out,T,N,K,K};}};
bool aligned(const void*p,std::size_t n){return p&&(reinterpret_cast<std::uintptr_t>(p)&(n-1))==0;}
bool overlap(const void*a,std::size_t an,const void*b,std::size_t bn){auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<y+bn&&y<x+an;}
hipError_t module_call(Module&m,const A8Q4G64LinearArgs&a,hipStream_t s){constexpr std::size_t ab=std::size_t(T)*K/2,asb=std::size_t(T)*Groups*2,wb=std::size_t(N)*K/2,wsb=std::size_t(N)*Groups*2,ob=std::size_t(T)*N*2;
 if(a.tokens!=T||a.rows!=N||a.columns!=K||a.padded_columns!=K||a.activation_low_code_bytes!=ab||a.activation_high_code_bytes!=ab||a.activation_scale_bytes!=asb||a.weight_code_bytes!=wb||a.weight_scale_bytes!=wsb||!aligned(a.activation_low_codes,4)||!aligned(a.activation_high_codes,4)||!aligned(a.activation_scales,2)||!aligned(a.activation_status,4)||!aligned(a.weight_codes,8)||!aligned(a.weight_scales,2)||!aligned(a.output,2))return hipErrorInvalidValue;
 const void*p[]={a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output};const std::size_t z[]={ab,ab,asb,4,wb,wsb,ob};for(unsigned i=0;i<7;++i)for(unsigned j=i+1;j<7;++j)if(overlap(p[i],z[i],p[j],z[j]))return hipErrorInvalidValue;
 Kernarg k{a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output,T,N,K,K};std::size_t size=sizeof(k);void*config[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,&k,HIP_LAUNCH_PARAM_BUFFER_SIZE,&size,HIP_LAUNCH_PARAM_END};return hipModuleLaunchKernel(m.f,N/128,T/64,1,512,1,1,0,s,nullptr,config);}
void launch_prod(const A8Q4G64LinearArgs&a,hipStream_t s){ok(module_call(*baseline_module,a,s),"baseline");}
void launch_ht(const A8Q4G64LinearArgs&a,hipStream_t s){ok(module_call(*candidate_module,a,s),"candidate");}
void parity(const hip_bfloat16*a,const hip_bfloat16*b){auto x=get(a,std::size_t(T)*N),y=get(b,std::size_t(T)*N);for(std::size_t i=0;i<x.size();++i)if(x[i].data!=y[i].data)throw std::runtime_error("full bit parity differs at "+std::to_string(i));}
void oracle(const Fixture&f,const hip_bfloat16*out){auto z=get(out,std::size_t(T)*N);const unsigned ts[]={0,1,1023,2047},ns[]={0,15,127,255,N-1};for(unsigned t:ts)for(unsigned n:ns){double e=0;for(unsigned g=0;g<Groups;++g){std::int32_t dot=0;for(unsigned q=0;q<64;++q)dot+=int(av(t,g*64+q))*int(wv(n,g*64+q));e+=double(dot)*hf(f.h.as[std::size_t(t)*Groups+g])*hf(f.h.ws[si(n,g)]);}float a=float(z[std::size_t(t)*N+n]);float bound=std::max(8.0f,std::abs(float(e))*.008f);if(!std::isfinite(a)||std::abs(a-float(e))>bound)throw std::runtime_error("complete represented FP64 oracle");}}
void canaries(Fixture&f){std::vector<hip_bfloat16>c(64);for(auto&x:c)x.data=0x3f81;ok(hipMemcpy(f.inc.p+std::size_t(T)*N,c.data(),128,hipMemcpyHostToDevice),"inc guard");ok(hipMemcpy(f.cand.p+std::size_t(T)*N,c.data(),128,hipMemcpyHostToDevice),"cand guard");}
void check_canaries(Fixture&f){for(auto*p:{f.inc.p,f.cand.p}){auto c=get(p+std::size_t(T)*N,64);for(auto x:c)if(x.data!=0x3f81)throw std::runtime_error("output guard");}}
void rejected_guards(Fixture&f){auto base=f.args(f.cand.p);std::vector<hip_bfloat16>sent(32);for(auto&x:sent)x.data=0x3f81;D<hip_bfloat16>guard(32);put(guard,sent);base.output=guard.p;
 auto reject=[&](A8Q4G64LinearArgs a,const char*n){if(module_call(*candidate_module,a,nullptr)!=hipErrorInvalidValue)throw std::runtime_error(std::string(n)+" accepted");auto q=get(guard.p,32);for(auto x:q)if(x.data!=0x3f81)throw std::runtime_error(std::string(n)+" clobber");};
 auto a=base;a.tokens--;reject(a,"tokens");a=base;a.rows--;reject(a,"rows");a=base;a.columns-=64;a.padded_columns-=64;reject(a,"columns");a=base;a.activation_low_code_bytes--;reject(a,"A-low span");a=base;a.activation_high_code_bytes--;reject(a,"A-high span");a=base;a.activation_scale_bytes--;reject(a,"A-scale span");a=base;a.weight_code_bytes--;reject(a,"W span");a=base;a.weight_scale_bytes--;reject(a,"W-scale span");
 auto bump=[](const void*p){return reinterpret_cast<std::uintptr_t>(p)+1;};
 a=base;a.activation_low_codes=nullptr;reject(a,"A-low null");a=base;a.activation_high_codes=nullptr;reject(a,"A-high null");a=base;a.activation_scales=nullptr;reject(a,"A-scale null");a=base;a.activation_status=nullptr;reject(a,"status null");a=base;a.weight_codes=nullptr;reject(a,"W null");a=base;a.weight_scales=nullptr;reject(a,"W-scale null");a=base;a.output=nullptr;reject(a,"output null");a=base;a.activation_low_codes=reinterpret_cast<const std::uint8_t*>(bump(a.activation_low_codes));reject(a,"A-low alignment");a=base;a.activation_high_codes=reinterpret_cast<const std::uint8_t*>(bump(a.activation_high_codes));reject(a,"A-high alignment");a=base;a.activation_scales=reinterpret_cast<const std::uint16_t*>(bump(a.activation_scales));reject(a,"A-scale alignment");a=base;a.activation_status=reinterpret_cast<const std::uint32_t*>(bump(a.activation_status));reject(a,"status alignment");a=base;a.weight_codes=reinterpret_cast<const std::uint8_t*>(bump(a.weight_codes));reject(a,"W alignment");a=base;a.weight_scales=reinterpret_cast<const std::uint16_t*>(bump(a.weight_scales));reject(a,"W-scale alignment");a=base;a.output=reinterpret_cast<hip_bfloat16*>(bump(a.output));reject(a,"output alignment");a=base;a.activation_high_codes=a.activation_low_codes;reject(a,"input overlap");a=base;a.weight_codes=reinterpret_cast<const std::uint8_t*>(a.output);reject(a,"output overlap");}
void graph_check(Fixture&f,Stream&s){auto a=f.args(f.cand.p);hipGraph_t g{};hipGraphExec_t e{};ok(hipStreamBeginCapture(s.s,hipStreamCaptureModeGlobal),"capture begin");launch_ht(a,s.s);ok(hipStreamEndCapture(s.s,&g),"capture end");ok(hipGraphInstantiate(&e,g,nullptr,nullptr,0),"graph instantiate");ok(hipMemsetAsync(f.cand.p,0xa5,std::size_t(T)*N*2,s.s),"graph poison");ok(hipGraphLaunch(e,s.s),"graph launch");ok(hipStreamSynchronize(s.s),"graph sync");(void)hipGraphExecDestroy(e);(void)hipGraphDestroy(g);}
void regression(Fixture&f){Stream s;canaries(f);auto ia=f.args(f.inc.p),ca=f.args(f.cand.p);launch_prod(ia,s.s);launch_ht(ca,s.s);ok(hipStreamSynchronize(s.s),"correctness sync");parity(f.inc.p,f.cand.p);oracle(f,f.cand.p);graph_check(f,s);parity(f.inc.p,f.cand.p);check_canaries(f);rejected_guards(f);
 std::vector<std::uint32_t>bad(1,1),zero(1,0);put(f.st,bad);launch_ht(ca,s.s);ok(hipStreamSynchronize(s.s),"status sync");auto z=get(f.cand.p,std::size_t(T)*N);for(auto x:z)if(x.data!=0x7fc1)throw std::runtime_error("status poison");put(f.st,zero);launch_ht(ca,s.s);ok(hipStreamSynchronize(s.s),"status reset");parity(f.inc.p,f.cand.p);std::cout<<"PASS exact_shape=2048x5120x17408 full_bit_parity=true complete_k_fp64_samples=20 graph_replay=true status_poison_reset=true guards=true\n";}
double interval(hipStream_t s,const std::function<void()>&fn){hipEvent_t a{},b{};ok(hipEventCreate(&a),"event");ok(hipEventCreate(&b),"event");ok(hipEventRecord(a,s),"record start");for(unsigned i=0;i<Iterations;++i)fn();ok(hipEventRecord(b,s),"record end");ok(hipEventSynchronize(b),"event sync");float ms=0;ok(hipEventElapsedTime(&ms,a,b),"elapsed");(void)hipEventDestroy(a);(void)hipEventDestroy(b);return ms;}
void benchmark(Fixture&f){Stream s;auto ia=f.args(f.inc.p),ca=f.args(f.cand.p);auto inc=[&]{launch_prod(ia,s.s);};auto cand=[&]{launch_ht(ca,s.s);};for(int i=0;i<2;++i){inc();cand();}ok(hipStreamSynchronize(s.s),"warmup");std::array<double,Trials>iv{},cv{};for(unsigned q=0;q<Trials;++q){if(q&1){cv[q]=interval(s.s,cand);iv[q]=interval(s.s,inc);}else{iv[q]=interval(s.s,inc);cv[q]=interval(s.s,cand);}}
 hipDeviceProp_t p{};ok(hipGetDeviceProperties(&p,0),"device");std::cout<<std::fixed<<std::setprecision(9)<<"{\"schema\":\"ninfer.r9700.a8q4-n16k16-weight-ht-measurement.v1\",\"device\":{\"ordinal\":0,\"name\":\""<<p.name<<"\",\"architecture\":\""<<p.gcnArchName<<"\"},\"tokens\":2048,\"rows\":5120,\"columns\":17408,\"calls\":64,\"iterations_per_sample\":128,\"control_interval_samples_ms\":[";for(unsigned i=0;i<Trials;++i){if(i)std::cout<<',';std::cout<<iv[i];}std::cout<<"],\"candidate_interval_samples_ms\":[";for(unsigned i=0;i<Trials;++i){if(i)std::cout<<',';std::cout<<cv[i];}std::cout<<"]}\n";}
}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("use --regression or --benchmark");require_device();Module b("tools/r9700/build/a8q4_n16k16_weight_ht_baseline.hsaco"),c("tools/r9700/build/a8q4_n16k16_weight_ht_candidate.hsaco");baseline_module=&b;candidate_module=&c;Fixture f;if(std::string(argv[1])=="--regression")regression(f);else if(std::string(argv[1])=="--benchmark")benchmark(f);else throw std::runtime_error("mode");baseline_module=candidate_module=nullptr;return 0;}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
