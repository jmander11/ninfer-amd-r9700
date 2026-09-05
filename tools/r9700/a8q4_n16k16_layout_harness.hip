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

extern "C" __global__ void a8q4_n16k16_prefill_p2048_m64n128(
    const std::uint8_t*, const std::uint8_t*, const std::uint16_t*, const std::uint32_t*,
    const std::uint64_t*, const std::uint16_t*, hip_bfloat16*, std::uint32_t, std::uint32_t);
extern "C" __global__ void a8q4_n16k16_decode_t1_wmma32(
    const std::uint8_t*, const std::uint8_t*, const std::uint16_t*, const std::uint32_t*,
    const std::uint64_t*, const std::uint16_t*, hip_bfloat16*, std::uint32_t, std::uint32_t);

namespace {
using ninfer::ops::r9700::linear::A8Q4G64LinearArgs;
constexpr std::uint32_t kTokens = 2048;
constexpr int kTrials = 9, kDecodeRepeats = 101;
struct Shape { const char* name; std::uint32_t n, k, calls; };
// Exact selected P2048 Q4 call inventory from tools/bench/analyze_q4_pmc.py.
constexpr Shape kShapes[] = {{"gdn_value_z", 12288, 5120, 48},
                             {"attention_gdn_output", 5120, 6144, 64},
                             {"post_mixer_down", 5120, 17408, 64}};

void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint16_t half_bits(float f) { __half h = __float2half(f); std::uint16_t b; std::memcpy(&b,&h,2); return b; }
float half_value(std::uint16_t b) { __half h; std::memcpy(&h,&b,2); return __half2float(h); }
std::int8_t activation(std::uint32_t t, std::uint32_t k) {
    return static_cast<std::int8_t>(static_cast<int>((t*13U+k*17U)%255U)-127);
}
std::int8_t weight(std::uint32_t n, std::uint32_t k) {
    if (k < 4U) {
        const std::uint8_t code = static_cast<std::uint8_t>((n >> (k * 4U)) & 15U);
        return static_cast<std::int8_t>(code < 8U ? code : static_cast<int>(code) - 16);
    }
    std::uint32_t mixed = n * 1664525U + k * 1013904223U + (n ^ (k >> 3U)) * 2246822519U;
    mixed ^= mixed >> 13U; mixed ^= mixed >> 22U;
    const std::uint8_t code = static_cast<std::uint8_t>(mixed & 15U);
    return static_cast<std::int8_t>(code < 8U ? code : static_cast<int>(code) - 16);
}
std::uint8_t nibble(std::int8_t q) { return static_cast<std::uint8_t>(q)&15U; }
std::size_t code_index(std::uint32_t n,std::uint32_t g,std::uint32_t p,std::uint32_t gs) {
    return (((static_cast<std::size_t>(n>>4U)*gs+g)*4U+p)*16U+(n&15U));
}
std::size_t scale_index(std::uint32_t n,std::uint32_t g,std::uint32_t gs) {
    return (static_cast<std::size_t>(n>>4U)*gs+g)*16U+(n&15U);
}
template<class T> struct Device {
    T* ptr{}; std::size_t count{};
    explicit Device(std::size_t n):count(n){hip_ok(hipMalloc(&ptr,n*sizeof(T)),"hipMalloc");}
    ~Device(){if(ptr)(void)hipFree(ptr);} Device(const Device&)=delete;
};
template<class T> void upload(Device<T>& d,const std::vector<T>& h){
    hip_ok(hipMemcpy(d.ptr,h.data(),h.size()*sizeof(T),hipMemcpyHostToDevice),"upload");
}
template<class T> std::vector<T> download(const Device<T>& d,std::size_t n){
    std::vector<T> h(n); hip_ok(hipMemcpy(h.data(),d.ptr,n*sizeof(T),hipMemcpyDeviceToHost),"download"); return h;
}
struct Images { std::vector<std::uint8_t> low,high,wrow; std::vector<std::uint64_t> wn16;
    std::vector<std::uint16_t> as,ws,wsn16; };

Images make_images(const Shape& s) {
    const std::uint32_t gs=s.k/64U; Images x;
    x.low.resize(static_cast<std::size_t>(kTokens)*s.k/2U); x.high.resize(x.low.size());
    for(std::uint32_t t=0;t<kTokens;++t) for(std::uint32_t c=0;c<s.k;c+=2U){
        const auto a0=activation(t,c),a1=activation(t,c+1U); const auto l0=nibble(a0),l1=nibble(a1);
        const auto h0=nibble(static_cast<std::int8_t>((static_cast<int>(a0)-l0)/16));
        const auto h1=nibble(static_cast<std::int8_t>((static_cast<int>(a1)-l1)/16));
        const auto o=(static_cast<std::size_t>(t)*s.k+c)/2U;
        x.low[o]=l0|static_cast<std::uint8_t>(l1<<4U); x.high[o]=h0|static_cast<std::uint8_t>(h1<<4U);
    }
    x.wrow.resize(static_cast<std::size_t>(s.n)*s.k/2U);
    x.wn16.resize(static_cast<std::size_t>(s.n)*s.k/16U);
    for(std::uint32_t r=0;r<s.n;++r) for(std::uint32_t g=0;g<gs;++g)
      for(std::uint32_t p=0;p<4U;++p){ std::uint64_t word=0;
        for(std::uint32_t b=0;b<8U;++b){const auto c=g*64U+p*16U+b*2U;
          const auto v=nibble(weight(r,c))|static_cast<std::uint8_t>(nibble(weight(r,c+1U))<<4U);
          x.wrow[(static_cast<std::size_t>(r)*s.k+c)/2U]=v; word|=static_cast<std::uint64_t>(v)<<(b*8U);}
        x.wn16[code_index(r,g,p,gs)]=word; }
    x.as.resize(static_cast<std::size_t>(kTokens)*gs);
    for(std::uint32_t t=0;t<kTokens;++t) for(std::uint32_t g=0;g<gs;++g)
      x.as[static_cast<std::size_t>(t)*gs+g]=half_bits(float(1U+((t*3U+g*5U)&7U))/8.0F);
    x.ws.resize(static_cast<std::size_t>(s.n)*gs); x.wsn16.resize(x.ws.size());
    for(std::uint32_t r=0;r<s.n;++r) for(std::uint32_t g=0;g<gs;++g){
      const auto b=half_bits(float(1U+(((r*29U)^(g*11U)^(r>>(g&7U)))&7U))/8.0F);
      x.ws[static_cast<std::size_t>(r)*gs+g]=b; x.wsn16[scale_index(r,g,gs)]=b; }
    return x;
}
void layout_oracle(const Shape&s,const Images&x){const auto gs=s.k/64U;
  for(std::uint32_t r=0;r<s.n;++r)for(std::uint32_t g=0;g<gs;++g){
    const std::size_t independent_scale=(static_cast<std::size_t>(r)/16U)*gs*16U+g*16U+r%16U;
    if(x.wsn16[independent_scale]!=x.ws[static_cast<std::size_t>(r)*gs+g])throw std::runtime_error("scale roundtrip");
    for(std::uint32_t p=0;p<4U;++p){std::uint64_t word=0;const auto o=static_cast<std::size_t>(r)*s.k/2U+g*32U+p*8U;
      const std::size_t independent=(static_cast<std::size_t>(r)/16U)*gs*64U+g*64U+p*16U+r%16U;
      const auto tile=independent/(gs*64U),within=independent%(gs*64U),decoded_g=within/64U;
      const auto rem=within%64U,decoded_p=rem/16U,decoded_r=tile*16U+rem%16U;
      if(decoded_r!=r||decoded_g!=g||decoded_p!=p)throw std::runtime_error("independent offset inverse");
      std::memcpy(&word,x.wrow.data()+o,8);if(word!=x.wn16[independent])throw std::runtime_error("code roundtrip");}}
  for(std::uint32_t base=0;base<s.n;base+=128U)for(std::uint32_t a=0;a<128U;++a)for(std::uint32_t b=a+1;b<128U;++b){
    const auto ao=static_cast<std::size_t>(base+a)*s.k/2U,bo=static_cast<std::size_t>(base+b)*s.k/2U;
    if(std::memcmp(x.wrow.data()+ao,x.wrow.data()+bo,s.k/2U)==0)throw std::runtime_error("fixture row alias");}
}

struct CArgs {const std::uint8_t*lo;std::size_t lob;const std::uint8_t*hi;std::size_t hib;
 const std::uint16_t*as;std::size_t asb;const std::uint32_t*status;const std::uint64_t*w;std::size_t wb;
 const std::uint16_t*ws;std::size_t wsb;hip_bfloat16*out;std::uint32_t t,n,k;};
bool aligned(const void*p,std::size_t a){return p&&(reinterpret_cast<std::uintptr_t>(p)&(a-1U))==0;}
bool overlap(const void* a,std::size_t an,const void* b,std::size_t bn){
 const auto ab=reinterpret_cast<std::uintptr_t>(a),bb=reinterpret_cast<std::uintptr_t>(b);
 return ab<bb+bn&&bb<ab+an;
}
hipError_t launch(CArgs a,bool prefill,hipStream_t stream){
 const auto ab=static_cast<std::size_t>(a.t)*a.k/2U,asb=static_cast<std::size_t>(a.t)*(a.k/64U)*2U;
 const auto wb=static_cast<std::size_t>(a.n)*a.k/2U,wsb=static_cast<std::size_t>(a.n)*(a.k/64U)*2U;
 const auto outb=static_cast<std::size_t>(a.t)*a.n*2U;
 if(!aligned(a.lo,4)||!aligned(a.hi,4)||!aligned(a.as,2)||!aligned(a.status,4)||!aligned(a.w,8)||
    !aligned(a.ws,2)||!aligned(a.out,2)||!a.n||!a.k||a.k%128U||a.lob!=ab||a.hib!=ab||a.asb!=asb||a.wb!=wb||a.wsb!=wsb)
   return hipErrorInvalidValue;
 const void* p[]={a.lo,a.hi,a.as,a.status,a.w,a.ws,a.out};
 const std::size_t z[]={ab,ab,asb,sizeof(std::uint32_t),wb,wsb,outb};
 for(unsigned i=0;i<7;++i)for(unsigned j=i+1;j<7;++j)if(overlap(p[i],z[i],p[j],z[j]))return hipErrorInvalidValue;
 if(prefill){if(a.t!=kTokens||a.n%128U)return hipErrorInvalidValue;
  hipLaunchKernelGGL(a8q4_n16k16_prefill_p2048_m64n128,dim3(a.n/128U,kTokens/64U),dim3(512),0,stream,
   a.lo,a.hi,a.as,a.status,a.w,a.ws,a.out,a.n,a.k);
 }else{if(a.t!=1U||a.n%16U)return hipErrorInvalidValue;
  hipLaunchKernelGGL(a8q4_n16k16_decode_t1_wmma32,dim3(a.n/16U),dim3(32),0,stream,
   a.lo,a.hi,a.as,a.status,a.w,a.ws,a.out,a.n,a.k);}
 return hipGetLastError();
}
float elapsed(const std::function<void()>&f,int reps){hipEvent_t a{},b{};hip_ok(hipEventCreate(&a),"event");hip_ok(hipEventCreate(&b),"event");
 hip_ok(hipEventRecord(a),"record");for(int i=0;i<reps;++i)f();hip_ok(hipEventRecord(b),"record");hip_ok(hipEventSynchronize(b),"sync");
 float ms=0;hip_ok(hipEventElapsedTime(&ms,a,b),"elapsed");(void)hipEventDestroy(a);(void)hipEventDestroy(b);return ms/reps;}
double median(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2U];}
struct M {std::vector<double>i,c;};
M trials(const char*phase,const Shape&s,const std::function<void()>&i,const std::function<void()>&c,int reps){
 for(int j=0;j<2;++j){i();c();}hip_ok(hipDeviceSynchronize(),"warmup");M m;std::cout<<std::fixed<<std::setprecision(6);
 for(int q=0;q<kTrials;++q){double im,cm;const char*order=(q&1)?"candidate_incumbent":"incumbent_candidate";
  if(q&1){cm=elapsed(c,reps);im=elapsed(i,reps);}else{im=elapsed(i,reps);cm=elapsed(c,reps);}
  m.i.push_back(im);m.c.push_back(cm);std::cout<<"TRIAL\t"<<phase<<'\t'<<s.name<<'\t'<<s.n<<'\t'<<s.k<<'\t'<<s.calls<<'\t'<<q<<'\t'<<order<<'\t'<<im<<'\t'<<cm<<'\n';}return m;}
std::vector<hip_bfloat16> download_output(const hip_bfloat16*p,std::size_t n){std::vector<hip_bfloat16>h(n);
 hip_ok(hipMemcpy(h.data(),p,n*2U,hipMemcpyDeviceToHost),"output download");return h;}
void parity(const hip_bfloat16*a,const hip_bfloat16*b,std::size_t n,const std::string&label){
 auto x=download_output(a,n),y=download_output(b,n);for(std::size_t j=0;j<n;++j)if(x[j].data!=y[j].data)throw std::runtime_error(label+" parity "+std::to_string(j));}
void fp64_samples(const Shape&s,const Images&x,const hip_bfloat16*d,std::uint32_t ts){const auto gs=s.k/64U;
 const std::array<std::uint32_t,4> tt=ts==1?std::array<std::uint32_t,4>{0,0,0,0}:std::array<std::uint32_t,4>{0,1,1023,2047};
 const std::array<std::uint32_t,5> rr={0,15,127,s.n/2U,s.n-1U};auto got=download_output(d,static_cast<std::size_t>(ts)*s.n);
 for(auto t:tt)for(auto r:rr){double expected=0;for(std::uint32_t g=0;g<gs;++g){std::int32_t dot=0;
   for(std::uint32_t c=0;c<64U;++c)dot+=int(activation(t,g*64U+c))*int(weight(r,g*64U+c));
   expected+=double(dot)*half_value(x.as[static_cast<std::size_t>(t)*gs+g])*half_value(x.ws[static_cast<std::size_t>(r)*gs+g]);}
  const float actual=float(got[static_cast<std::size_t>(t)*s.n+r]);const float bound=std::max(8.0F,std::abs(float(expected))*.008F);
  if(!std::isfinite(actual)||std::abs(actual-float(expected))>bound)throw std::runtime_error(std::string(s.name)+" complete-K FP64");}}
void guard_checks(CArgs v){Device<hip_bfloat16>d(32);std::vector<hip_bfloat16>p(32);for(auto&x:p)x.data=0x3f81U;upload(d,p);v.out=d.ptr;
 auto reject=[&](CArgs b,bool pf,const char*n){if(launch(b,pf,nullptr)!=hipErrorInvalidValue)throw std::runtime_error(std::string(n)+" accepted");
  auto a=download(d,p.size());for(std::size_t j=0;j<p.size();++j)if(a[j].data!=p[j].data)throw std::runtime_error(std::string(n)+" clobber");};
 auto b=v;b.n--;reject(b,true,"N tail");b=v;b.k-=64;b.lob=static_cast<std::size_t>(b.t)*b.k/2;b.hib=b.lob;b.asb=static_cast<std::size_t>(b.t)*(b.k/64)*2;
 b.wb=static_cast<std::size_t>(b.n)*b.k/2;b.wsb=static_cast<std::size_t>(b.n)*(b.k/64)*2;reject(b,true,"K tail");
 b=v;b.t--;b.lob=static_cast<std::size_t>(b.t)*b.k/2;b.hib=b.lob;b.asb=static_cast<std::size_t>(b.t)*(b.k/64)*2;reject(b,true,"P2048 shape");
 const auto bump=[](const void*p){return reinterpret_cast<std::uintptr_t>(p)+1U;};
 b=v;b.lo=reinterpret_cast<const std::uint8_t*>(bump(v.lo));reject(b,true,"low alignment");b=v;b.hi=reinterpret_cast<const std::uint8_t*>(bump(v.hi));reject(b,true,"high alignment");
 b=v;b.as=reinterpret_cast<const std::uint16_t*>(bump(v.as));reject(b,true,"A-scale alignment");b=v;b.status=reinterpret_cast<const std::uint32_t*>(bump(v.status));reject(b,true,"status alignment");
 b=v;b.w=reinterpret_cast<const std::uint64_t*>(bump(v.w));reject(b,true,"W alignment");b=v;b.ws=reinterpret_cast<const std::uint16_t*>(bump(v.ws));reject(b,true,"W-scale alignment");
 b=v;b.out=reinterpret_cast<hip_bfloat16*>(bump(v.out));reject(b,true,"output alignment");
 b=v;b.lo=nullptr;reject(b,true,"low null");b=v;b.hi=nullptr;reject(b,true,"high null");
 b=v;b.as=nullptr;reject(b,true,"A-scale null");b=v;b.status=nullptr;reject(b,true,"status null");
 b=v;b.w=nullptr;reject(b,true,"W null");b=v;b.ws=nullptr;reject(b,true,"W-scale null");b=v;b.out=nullptr;reject(b,true,"output null");
 b=v;b.hi=b.lo;reject(b,true,"input overlap");
 b=v;b.lo=reinterpret_cast<const std::uint8_t*>(v.out);reject(b,true,"low-output overlap");
 b=v;b.hi=reinterpret_cast<const std::uint8_t*>(v.out);reject(b,true,"high-output overlap");
 b=v;b.as=reinterpret_cast<const std::uint16_t*>(v.out);reject(b,true,"A-scale-output overlap");
 b=v;b.status=reinterpret_cast<const std::uint32_t*>(v.out);reject(b,true,"status-output overlap");
 b=v;b.w=reinterpret_cast<const std::uint64_t*>(v.out);reject(b,true,"W-output overlap");
 b=v;b.ws=reinterpret_cast<const std::uint16_t*>(v.out);reject(b,true,"W-scale-output overlap");
 b=v;b.t=2;b.lob=b.k;b.hib=b.k;b.asb=(b.k/64)*4;reject(b,false,"T1 shape");}

void run(){hipDeviceProp_t props{};hip_ok(hipGetDeviceProperties(&props,0),"device properties");
 std::cout<<"DEVICE\t0\t"<<props.name<<'\t'<<props.gcnArchName<<'\n';
 std::array<double,3>pi{},pc{},di_m{},dc_m{};bool guarded=false,prefill_status=false,decode_status=false;
 for(std::size_t si=0;si<std::size(kShapes);++si){const auto&s=kShapes[si];auto x=make_images(s);layout_oracle(s,x);std::vector<std::uint32_t>ok(1,0);
  Device<std::uint8_t>dl(x.low.size()),dh(x.high.size()),wr(x.wrow.size());Device<std::uint64_t>wn(x.wn16.size());
  Device<std::uint16_t>as(x.as.size()),ws(x.ws.size()),wsn(x.wsn16.size());Device<std::uint32_t>st(1);
  const auto outputs=static_cast<std::size_t>(kTokens)*s.n;Device<hip_bfloat16>inc(outputs+64U),cand(outputs+64U);
  std::vector<hip_bfloat16>canary(64);for(auto&v:canary)v.data=0x3f81U;
  hip_ok(hipMemcpy(inc.ptr,canary.data(),32U*2U,hipMemcpyHostToDevice),"inc prefix");hip_ok(hipMemcpy(inc.ptr+32U+outputs,canary.data()+32U,32U*2U,hipMemcpyHostToDevice),"inc suffix");
  hip_ok(hipMemcpy(cand.ptr,canary.data(),32U*2U,hipMemcpyHostToDevice),"cand prefix");hip_ok(hipMemcpy(cand.ptr+32U+outputs,canary.data()+32U,32U*2U,hipMemcpyHostToDevice),"cand suffix");
  auto*incout=inc.ptr+32U;auto*candout=cand.ptr+32U;
  upload(dl,x.low);upload(dh,x.high);upload(wr,x.wrow);upload(wn,x.wn16);upload(as,x.as);upload(ws,x.ws);upload(wsn,x.wsn16);upload(st,ok);
  A8Q4G64LinearArgs ia{dl.ptr,x.low.size(),dh.ptr,x.high.size(),as.ptr,x.as.size()*2U,st.ptr,wr.ptr,x.wrow.size(),ws.ptr,x.ws.size()*2U,incout,kTokens,s.n,s.k,s.k};
  CArgs ca{dl.ptr,x.low.size(),dh.ptr,x.high.size(),as.ptr,x.as.size()*2U,st.ptr,wn.ptr,x.wn16.size()*8U,wsn.ptr,x.wsn16.size()*2U,candout,kTokens,s.n,s.k};
  auto ip=[&]{hip_ok(ninfer::ops::r9700::linear::a8q4g64_linear_prefill_cta(ia,nullptr),"inc prefill");};auto cp=[&]{hip_ok(launch(ca,true,nullptr),"cand prefill");};
  ip();cp();hip_ok(hipDeviceSynchronize(),"prefill correctness");parity(incout,candout,outputs,std::string(s.name)+" P2048");fp64_samples(s,x,candout,kTokens);
  if(!prefill_status){std::vector<std::uint32_t>bad(1,1);upload(st,bad);cp();hip_ok(hipDeviceSynchronize(),"prefill status");auto z=download_output(candout,outputs);
   if(!std::all_of(z.begin(),z.end(),[](hip_bfloat16 v){return v.data==0x7fc1U;}))throw std::runtime_error("prefill status poison");upload(st,ok);prefill_status=true;}
  if(!guarded){guard_checks(ca);guarded=true;}auto pm=trials("prefill",s,ip,cp,1);pi[si]=median(pm.i);pc[si]=median(pm.c);
  ia.tokens=1;ia.activation_low_code_bytes=s.k/2U;ia.activation_high_code_bytes=s.k/2U;ia.activation_scale_bytes=(s.k/64U)*2U;
  ca.t=1;ca.lob=s.k/2U;ca.hib=s.k/2U;ca.asb=(s.k/64U)*2U;
  hip_ok(hipMemcpy(incout+s.n,canary.data(),32U*2U,hipMemcpyHostToDevice),"decode inc suffix");
  hip_ok(hipMemcpy(candout+s.n,canary.data()+32U,32U*2U,hipMemcpyHostToDevice),"decode cand suffix");
  auto id=[&]{hip_ok(ninfer::ops::r9700::linear::a8q4g64_linear_wmma32(ia,nullptr),"inc decode");};auto cd=[&]{hip_ok(launch(ca,false,nullptr),"cand decode");};
  id();cd();hip_ok(hipDeviceSynchronize(),"decode correctness");parity(incout,candout,s.n,std::string(s.name)+" T1");fp64_samples(s,x,candout,1);
  if(!decode_status){std::vector<std::uint32_t>bad(1,1);upload(st,bad);cd();hip_ok(hipDeviceSynchronize(),"status");auto z=download_output(candout,s.n);
   if(!std::all_of(z.begin(),z.end(),[](hip_bfloat16 v){return v.data==0x7fc1U;}))throw std::runtime_error("decode status poison");upload(st,ok);decode_status=true;}
  auto dm=trials("decode",s,id,cd,kDecodeRepeats);di_m[si]=median(dm.i);dc_m[si]=median(dm.c);
  auto decode_inc_guard=download_output(incout+s.n,32),decode_cand_guard=download_output(candout+s.n,32);
  for(unsigned j=0;j<32;++j)if(decode_inc_guard[j].data!=canary[j].data||decode_cand_guard[j].data!=canary[j+32].data)throw std::runtime_error("adjacent T1 output canary");
  auto check_canary=[&](Device<hip_bfloat16>&d){auto pre=download_output(d.ptr,32),post=download_output(d.ptr+32U+outputs,32);for(unsigned j=0;j<32;++j)if(pre[j].data!=canary[j].data||post[j].data!=canary[j+32].data)throw std::runtime_error("output canary");};
  check_canary(inc);check_canary(cand);
  std::cout<<"CHECK\t"<<s.name<<"\tlayout_exact\tfull_prefill_bit_parity\tfull_decode_bit_parity\t20_sampled_complete_k_fp64_coordinates\tnonuniform_fp16_scales\toutput_canaries\n";}
 double saving=0,dii=0,dcc=0;bool each=true;for(std::size_t i=0;i<std::size(kShapes);++i){saving+=kShapes[i].calls*(pi[i]-pc[i]);dii+=kShapes[i].calls*di_m[i];dcc+=kShapes[i].calls*dc_m[i];
  each=each&&pc[i]/pi[i]<=1.01&&dc_m[i]/di_m[i]<=1.01;std::cout<<"MEDIAN\t"<<kShapes[i].name<<'\t'<<pi[i]<<'\t'<<pc[i]<<'\t'<<di_m[i]<<'\t'<<dc_m[i]<<'\n';}
 const double dr=dcc/dii;const bool pass=saving>=30.0&&dr<=.99&&each;
 std::cout<<"SUMMARY\tprefill_weighted_saving_ms\t"<<saving<<"\tdecode_weighted_ratio\t"<<dr<<"\tprefill_gate_ge_30ms\t"<<(saving>=30)<<"\tdecode_gate_le_0.99\t"<<(dr<=.99)<<"\teach_shape_le_1.01\t"<<each<<"\tguard_no_clobber\t"<<guarded<<"\tprefill_status\t"<<prefill_status<<"\tdecode_status\t"<<decode_status<<'\n';
 std::cout<<"COMPLETE\tadmission_pass\t"<<pass<<'\n';}
}
int main(){try{run();return 0;}catch(const std::exception&e){std::cerr<<"FAIL\t"<<e.what()<<'\n';return 1;}}
