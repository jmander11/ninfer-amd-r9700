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
constexpr unsigned T=2048,G=64,Trials=8,Iterations=128;
struct Shape{unsigned n,k,calls;const char*name;};
constexpr Shape Shapes[]={{12288,5120,48,"value_z"},{5120,6144,64,"gdn_output"},{5120,17408,64,"mlp_down"}};
void ok(hipError_t e,const char*w){if(e!=hipSuccess)throw std::runtime_error(std::string(w)+": "+hipGetErrorString(e));}
std::uint16_t hb(float f){__half h=__float2half_rn(f);std::uint16_t b;std::memcpy(&b,&h,2);return b;}
float hf(std::uint16_t b){__half h;std::memcpy(&h,&b,2);return __half2float(h);}
std::int8_t av(unsigned t,unsigned k){return static_cast<std::int8_t>(int((t*13U+k*17U+(t^(k>>2))*7U)%255U)-127);}
std::int8_t wv(unsigned n,unsigned k){unsigned x=n*1664525U+k*1013904223U+(n^(k>>3U))*2246822519U;x^=x>>13U;x^=x>>22U;unsigned q=x&15U;return static_cast<std::int8_t>(q<8?q:int(q)-16);}
std::uint8_t nib(std::int8_t q){return static_cast<std::uint8_t>(q)&15U;}
std::size_t wi(unsigned n,unsigned g,unsigned p,unsigned groups){return (((std::size_t(n>>4U)*groups+g)*4U+p)*16U+(n&15U));}
std::size_t si(unsigned n,unsigned g,unsigned groups){return (std::size_t(n>>4U)*groups+g)*16U+(n&15U);}
template<class X>struct D{X*raw{},*p{};std::size_t n;explicit D(std::size_t z):n(z){ok(hipMalloc(&raw,(z+64)*sizeof(X)),"malloc");p=raw+32;ok(hipMemset(raw,0xa5,(z+64)*sizeof(X)),"guard init");}~D(){if(raw)(void)hipFree(raw);}D(const D&)=delete;};
template<class X>void put(D<X>&d,const std::vector<X>&h){ok(hipMemcpy(d.p,h.data(),h.size()*sizeof(X),hipMemcpyHostToDevice),"upload");}
template<class X>std::vector<X>get(const X*p,std::size_t n){std::vector<X>h(n);ok(hipMemcpy(h.data(),p,n*sizeof(X),hipMemcpyDeviceToHost),"download");return h;}
struct Stream{hipStream_t s{};Stream(){ok(hipStreamCreate(&s),"stream");}~Stream(){if(s)(void)hipStreamDestroy(s);}};
void require_device(){int d=-1;ok(hipGetDevice(&d),"device");hipDeviceProp_t p{};ok(hipGetDeviceProperties(&p,d),"properties");if(d!=0||std::string(p.gcnArchName)!="gfx1201"||std::string(p.name).find("R9700")==std::string::npos)throw std::runtime_error("device is not device0 R9700/gfx1201");}
struct Kernarg{const std::uint8_t*lo;const std::uint8_t*hi;const std::uint16_t*as;const std::uint32_t*status;const std::uint8_t*w;const std::uint16_t*ws;hip_bfloat16*out;std::uint32_t tokens,rows,columns,padded;};static_assert(sizeof(Kernarg)==72);
constexpr const char* ProdSymbol="_ZN6ninfer3ops5r97006linear12_GLOBAL__N_133a8q4g64_linear_prefill_cta_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjjj";
constexpr const char* CandSymbol="a8q4_n16k16_scalar_base_p2048_m64n128";
struct Module{hipModule_t m{};hipFunction_t f{};Module(const char*p,const char*s){ok(hipModuleLoad(&m,p),"module load");ok(hipModuleGetFunction(&f,m,s),"module function");}~Module(){if(m)(void)hipModuleUnload(m);}};
Module*prod{},*cand{};
struct Images{std::vector<std::uint8_t>lo,hi,w;std::vector<std::uint16_t>as,ws;};
Images images(const Shape&s){unsigned groups=s.k/G;Images x;x.lo.resize(std::size_t(T)*s.k/2);x.hi.resize(x.lo.size());
 for(unsigned t=0;t<T;++t)for(unsigned k=0;k<s.k;k+=2){auto a0=av(t,k),a1=av(t,k+1);auto l0=nib(a0),l1=nib(a1);auto h0=nib(static_cast<std::int8_t>((int(a0)-l0)/16)),h1=nib(static_cast<std::int8_t>((int(a1)-l1)/16));auto o=(std::size_t(t)*s.k+k)/2;x.lo[o]=l0|(l1<<4);x.hi[o]=h0|(h1<<4);}
 x.w.resize(std::size_t(s.n)*s.k/2);for(unsigned n=0;n<s.n;++n)for(unsigned g=0;g<groups;++g)for(unsigned p=0;p<4;++p){std::uint64_t q=0;for(unsigned b=0;b<8;++b){unsigned k=g*64+p*16+b*2;q|=std::uint64_t(nib(wv(n,k))|(nib(wv(n,k+1))<<4))<<(b*8);}std::memcpy(x.w.data()+wi(n,g,p,groups)*8,&q,8);}
 x.as.resize(std::size_t(T)*groups);for(unsigned t=0;t<T;++t)for(unsigned g=0;g<groups;++g)x.as[std::size_t(t)*groups+g]=hb(float(1+((t*3+g*5)&7))/8);
 x.ws.resize(std::size_t(s.n)*groups);for(unsigned n=0;n<s.n;++n)for(unsigned g=0;g<groups;++g)x.ws[si(n,g,groups)]=hb(float(1+(((n*29)^(g*11)^(n>>(g&7)))&7))/8);return x;}
struct Fixture{Shape s;unsigned groups;Images h;D<std::uint8_t>lo,hi,w;D<std::uint16_t>as,ws;D<std::uint32_t>st;D<hip_bfloat16>inc,out;
 explicit Fixture(Shape q):s(q),groups(q.k/G),h(images(q)),lo(h.lo.size()),hi(h.hi.size()),w(h.w.size()),as(h.as.size()),ws(h.ws.size()),st(1),inc(std::size_t(T)*q.n+64),out(std::size_t(T)*q.n+64){put(lo,h.lo);put(hi,h.hi);put(w,h.w);put(as,h.as);put(ws,h.ws);std::vector<std::uint32_t>z(1);put(st,z);}
 A8Q4G64LinearArgs args(hip_bfloat16*o){return{lo.p,lo.n,hi.p,hi.n,as.p,as.n*2,st.p,w.p,w.n,ws.p,ws.n*2,o,T,s.n,s.k,s.k};}};
template<class X>void allocation_guard(const D<X>&d){for(auto x:get(reinterpret_cast<const std::uint8_t*>(d.raw),32*sizeof(X)))if(x!=0xa5)throw std::runtime_error("prefix guard");for(auto x:get(reinterpret_cast<const std::uint8_t*>(d.p+d.n),32*sizeof(X)))if(x!=0xa5)throw std::runtime_error("suffix guard");}
void allocation_guards(const Fixture&f){allocation_guard(f.lo);allocation_guard(f.hi);allocation_guard(f.as);allocation_guard(f.st);allocation_guard(f.w);allocation_guard(f.ws);allocation_guard(f.inc);allocation_guard(f.out);}
bool aligned(const void*p,std::size_t n){return p&&(reinterpret_cast<std::uintptr_t>(p)&(n-1))==0;}
bool overlap(const void*a,std::size_t an,const void*b,std::size_t bn){auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<y+bn&&y<x+an;}
bool exact_shape(unsigned n,unsigned k){for(auto s:Shapes)if(s.n==n&&s.k==k)return true;return false;}
hipError_t call(Module&m,const A8Q4G64LinearArgs&a,hipStream_t stream){if(a.tokens!=T||!exact_shape(a.rows,a.columns)||a.padded_columns!=a.columns)return hipErrorInvalidValue;std::size_t groups=a.columns/G,ab=std::size_t(T)*a.columns/2,asb=std::size_t(T)*groups*2,wb=std::size_t(a.rows)*a.columns/2,wsb=std::size_t(a.rows)*groups*2,ob=std::size_t(T)*a.rows*2;
 if(a.activation_low_code_bytes!=ab||a.activation_high_code_bytes!=ab||a.activation_scale_bytes!=asb||a.weight_code_bytes!=wb||a.weight_scale_bytes!=wsb||!aligned(a.activation_low_codes,4)||!aligned(a.activation_high_codes,4)||!aligned(a.activation_scales,2)||!aligned(a.activation_status,4)||!aligned(a.weight_codes,8)||!aligned(a.weight_scales,2)||!aligned(a.output,2))return hipErrorInvalidValue;
 const void*p[]={a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output};const std::size_t z[]={ab,ab,asb,4,wb,wsb,ob};for(unsigned i=0;i<7;++i)for(unsigned j=i+1;j<7;++j)if(overlap(p[i],z[i],p[j],z[j]))return hipErrorInvalidValue;
 Kernarg k{a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output,T,a.rows,a.columns,a.columns};std::size_t size=sizeof(k);void*config[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,&k,HIP_LAUNCH_PARAM_BUFFER_SIZE,&size,HIP_LAUNCH_PARAM_END};return hipModuleLaunchKernel(m.f,a.rows/128,T/64,1,512,1,1,0,stream,nullptr,config);}
void launch(Module&m,const A8Q4G64LinearArgs&a,hipStream_t s,const char*n){ok(call(m,a,s),n);}
void parity(Fixture&f){auto x=get(f.inc.p,std::size_t(T)*f.s.n),y=get(f.out.p,std::size_t(T)*f.s.n);for(std::size_t i=0;i<x.size();++i)if(x[i].data!=y[i].data)throw std::runtime_error(std::string(f.s.name)+" full bit parity differs at "+std::to_string(i));}
void oracle(Fixture&f){auto z=get(f.out.p,std::size_t(T)*f.s.n);unsigned ts[]={0,1,1023,2047},ns[]={0,15,127,255,f.s.n-1};for(unsigned t:ts)for(unsigned n:ns){double e=0;for(unsigned g=0;g<f.groups;++g){std::int32_t dot=0;for(unsigned q=0;q<64;++q)dot+=int(av(t,g*64+q))*int(wv(n,g*64+q));e+=double(dot)*hf(f.h.as[std::size_t(t)*f.groups+g])*hf(f.h.ws[si(n,g,f.groups)]);}float a=float(z[std::size_t(t)*f.s.n+n]),bound=std::max(8.0f,std::abs(float(e))*.008f);if(!std::isfinite(a)||std::abs(a-float(e))>bound)throw std::runtime_error(std::string(f.s.name)+" complete represented FP64 oracle");}}
void canary(Fixture&f){std::vector<hip_bfloat16>x(64);for(auto&v:x)v.data=0x3f81;ok(hipMemcpy(f.inc.p+std::size_t(T)*f.s.n,x.data(),128,hipMemcpyHostToDevice),"inc guard");ok(hipMemcpy(f.out.p+std::size_t(T)*f.s.n,x.data(),128,hipMemcpyHostToDevice),"out guard");}
void check_canary(Fixture&f){for(auto*p:{f.inc.p,f.out.p})for(auto x:get(p+std::size_t(T)*f.s.n,64))if(x.data!=0x3f81)throw std::runtime_error("output suffix guard");}
void graph(Fixture&f,Stream&s){auto a=f.args(f.out.p);hipGraph_t g{};hipGraphExec_t e{};ok(hipStreamBeginCapture(s.s,hipStreamCaptureModeGlobal),"capture");launch(*cand,a,s.s,"candidate graph");ok(hipStreamEndCapture(s.s,&g),"capture end");ok(hipGraphInstantiate(&e,g,nullptr,nullptr,0),"instantiate");ok(hipMemsetAsync(f.out.p,0xa5,std::size_t(T)*f.s.n*2,s.s),"poison");ok(hipGraphLaunch(e,s.s),"replay");ok(hipStreamSynchronize(s.s),"replay sync");(void)hipGraphExecDestroy(e);(void)hipGraphDestroy(g);parity(f);}
void rejects(Fixture&f){auto base=f.args(f.out.p);std::vector<hip_bfloat16>before(64);for(auto&x:before)x.data=0x3f81;D<hip_bfloat16>guard(64);put(guard,before);base.output=guard.p;auto reject=[&](A8Q4G64LinearArgs a,const char*n){if(call(*cand,a,nullptr)!=hipErrorInvalidValue)throw std::runtime_error(std::string(n)+" accepted");if(get(guard.p,64)!=before)throw std::runtime_error(std::string(n)+" clobber");};
 auto a=base;a.tokens--;reject(a,"tokens");a=base;a.rows--;reject(a,"row");a=base;a.columns-=64;a.padded_columns-=64;reject(a,"K");
 a=base;a.activation_low_code_bytes--;reject(a,"A-low span");a=base;a.activation_high_code_bytes--;reject(a,"A-high span");a=base;a.activation_scale_bytes--;reject(a,"A-scale span");a=base;a.weight_code_bytes--;reject(a,"W span");a=base;a.weight_scale_bytes--;reject(a,"W-scale span");
 auto bump=[](const void*p){return reinterpret_cast<std::uintptr_t>(p)+1;};
 a=base;a.activation_low_codes=nullptr;reject(a,"A-low null");a=base;a.activation_high_codes=nullptr;reject(a,"A-high null");a=base;a.activation_scales=nullptr;reject(a,"A-scale null");a=base;a.activation_status=nullptr;reject(a,"status null");a=base;a.weight_codes=nullptr;reject(a,"W null");a=base;a.weight_scales=nullptr;reject(a,"W-scale null");a=base;a.output=nullptr;reject(a,"output null");
 a=base;a.activation_low_codes=reinterpret_cast<const std::uint8_t*>(bump(a.activation_low_codes));reject(a,"A-low alignment");a=base;a.activation_high_codes=reinterpret_cast<const std::uint8_t*>(bump(a.activation_high_codes));reject(a,"A-high alignment");a=base;a.activation_scales=reinterpret_cast<const std::uint16_t*>(bump(a.activation_scales));reject(a,"A-scale alignment");a=base;a.activation_status=reinterpret_cast<const std::uint32_t*>(bump(a.activation_status));reject(a,"status alignment");a=base;a.weight_codes=reinterpret_cast<const std::uint8_t*>(bump(a.weight_codes));reject(a,"W alignment");a=base;a.weight_scales=reinterpret_cast<const std::uint16_t*>(bump(a.weight_scales));reject(a,"W-scale alignment");a=base;a.output=reinterpret_cast<hip_bfloat16*>(bump(a.output));reject(a,"output alignment");
 a=base;a.activation_high_codes=a.activation_low_codes;reject(a,"input overlap");a=base;a.weight_codes=reinterpret_cast<const std::uint8_t*>(a.output);reject(a,"output overlap");}
void regress_one(Shape s){Fixture f(s);Stream stream;canary(f);auto ia=f.args(f.inc.p),ca=f.args(f.out.p);launch(*prod,ia,stream.s,"control");launch(*cand,ca,stream.s,"candidate");ok(hipStreamSynchronize(stream.s),"sync");parity(f);oracle(f);graph(f,stream);check_canary(f);rejects(f);std::vector<std::uint32_t>bad(1,1),zero(1,0);put(f.st,bad);launch(*cand,ca,stream.s,"status");ok(hipStreamSynchronize(stream.s),"status sync");for(auto x:get(f.out.p,std::size_t(T)*s.n))if(x.data!=0x7fc1)throw std::runtime_error("status poison");put(f.st,zero);launch(*cand,ca,stream.s,"reset");ok(hipStreamSynchronize(stream.s),"reset sync");parity(f);allocation_guards(f);}
double interval(hipStream_t s,const std::function<void()>&fn){hipEvent_t a{},b{};ok(hipEventCreate(&a),"event");ok(hipEventCreate(&b),"event");ok(hipEventRecord(a,s),"start");for(unsigned i=0;i<Iterations;++i)fn();ok(hipEventRecord(b,s),"end");ok(hipEventSynchronize(b),"event sync");float ms=0;ok(hipEventElapsedTime(&ms,a,b),"elapsed");(void)hipEventDestroy(a);(void)hipEventDestroy(b);return ms;}
void benchmark(){std::cout<<std::fixed<<std::setprecision(9)<<"{\"schema\":\"ninfer.r9700.a8q4-n16k16-scalar-base-measurement.v2\",\"device\":{\"ordinal\":0,\"name\":\"AMD Radeon AI PRO R9700\",\"architecture\":\"gfx1201\"},\"warmup_order\":[\"A\",\"B\",\"B\",\"A\"],\"pair_order\":[\"AB\",\"BA\",\"AB\",\"BA\",\"AB\",\"BA\",\"AB\",\"BA\"],\"cells\":[";bool first=true;for(auto s:Shapes){Fixture f(s);Stream stream;auto ia=f.args(f.inc.p),ca=f.args(f.out.p);auto a=[&]{launch(*prod,ia,stream.s,"control");};auto b=[&]{launch(*cand,ca,stream.s,"candidate");};a();b();b();a();ok(hipStreamSynchronize(stream.s),"warmup");std::array<double,Trials>x{},y{};for(unsigned q=0;q<Trials;++q){if(q&1){y[q]=interval(stream.s,b);x[q]=interval(stream.s,a);}else{x[q]=interval(stream.s,a);y[q]=interval(stream.s,b);}}if(!first)std::cout<<',';first=false;std::cout<<"{\"name\":\""<<s.name<<"\",\"tokens\":2048,\"rows\":"<<s.n<<",\"columns\":"<<s.k<<",\"calls\":"<<s.calls<<",\"iterations_per_sample\":128,\"control_interval_samples_ms\":[";for(unsigned i=0;i<Trials;++i){if(i)std::cout<<',';std::cout<<x[i];}std::cout<<"],\"candidate_interval_samples_ms\":[";for(unsigned i=0;i<Trials;++i){if(i)std::cout<<',';std::cout<<y[i];}std::cout<<"]}";}std::cout<<"]}\n";}
}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("use --regression or --benchmark");require_device();Module p("tools/r9700/build/a8q4_n16k16_scalar_base_baseline.hsaco",ProdSymbol),c("tools/r9700/build/a8q4_n16k16_scalar_base_candidate.hsaco",CandSymbol);prod=&p;cand=&c;if(std::string(argv[1])=="--regression"){for(auto s:Shapes)regress_one(s);std::cout<<"PASS exact_shapes=3 full_bit_parity=true complete_k_fp64_samples=60 graph_replay=true status_poison_reset=true guards=true\n";}else if(std::string(argv[1])=="--benchmark")benchmark();else throw std::runtime_error("mode");prod=cand=nullptr;return 0;}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
