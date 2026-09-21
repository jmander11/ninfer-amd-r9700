#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"
#include "ops/r9700/linear/dflash_down_scale_gather_qualification.h"
#include <cstring>
#include <functional>

namespace {
constexpr unsigned N=5120,K=17408,G=272,Copies=3,Trials=24;
constexpr std::size_t Guard=256;
const std::filesystem::path Pci="/sys/bus/pci/devices/0000:13:00.0";
void power() {
    if(read_text(Pci/"vendor")!="0x1002" || read_text(Pci/"device")!="0x7551" ||
       read_text(Pci/"power_dpm_force_performance_level")!="auto")fail("R9700 auto required");
}
void transfer(void* dst,const void* src,std::size_t bytes,hipMemcpyKind kind,hipStream_t s) {
    HIP_CHECK(hipMemcpyAsync(dst,src,bytes,kind,s));HIP_CHECK(hipStreamSynchronize(s));
}
template<class T> struct Guarded {
    DeviceBuffer<std::uint8_t> storage;std::size_t count;
    Guarded(std::size_t n,hipStream_t s):storage(n*sizeof(T)+2*Guard),count(n) {
        HIP_CHECK(hipMemsetAsync(storage.get(),0xa5,storage.bytes(),s));
    }
    T* data() const{return reinterpret_cast<T*>(storage.get()+Guard);}
    std::size_t bytes() const{return count*sizeof(T);}
    std::vector<T> read(hipStream_t s) const{
        std::vector<T> v(count);transfer(v.data(),data(),bytes(),hipMemcpyDeviceToHost,s);return v;
    }
    void put(const std::vector<T>& v,hipStream_t s){transfer(data(),v.data(),bytes(),hipMemcpyHostToDevice,s);}
    void guards(hipStream_t s) const{
        std::array<std::uint8_t,Guard> b;
        for(auto offset:{std::size_t{0},storage.bytes()-Guard}) {
            transfer(b.data(),storage.get()+offset,Guard,hipMemcpyDeviceToHost,s);
            if(!std::all_of(b.begin(),b.end(),[](auto x){return x==0xa5;}))fail("canary changed");
        }
    }
};
template<class T> void exact(const std::vector<T>& a,const std::vector<T>& b,const char* why) {
    if(a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))fail(why);
}
struct Weights {
    Guarded<std::uint8_t> codes;Guarded<std::uint16_t> scales;
    Weights(const DecodeDot8Weights& w,hipStream_t s):codes(w.codes.size(),s),scales(w.scales.size(),s) {
        codes.put(w.codes,s);scales.put(w.scales,s);
    }
};
struct Output {
    Guarded<hip_bfloat16> value;Guarded<std::uint8_t> scratch;
    Output(unsigned t,hipStream_t s):value(t*N,s),scratch(linear::a8q4g64_activation_workspace_capacity_bytes(t,K),s){}
};
linear::A8Q4G64CandidateArgs arguments(unsigned t,Guarded<hip_bfloat16>& x,Weights& w,Output& y) {
    return {x.data(),w.codes.data(),w.codes.bytes(),w.scales.data(),w.scales.bytes(),
        y.scratch.data(),y.scratch.bytes(),y.value.data(),t,N,K,K};
}
void launch(bool candidate,const linear::A8Q4G64CandidateArgs& a,hipStream_t s) {
    if(candidate)HIP_CHECK(linear::qualification::dflash_down_scale_gather(a,s));
    else {
        linear::A8G64ActivationWorkspace w{};
        HIP_CHECK(linear::a8q4g64_bind_activation_workspace(a.activation_workspace,a.activation_workspace_bytes,a.tokens,K,&w));
        HIP_CHECK(linear::a8g64_quantize_activation({a.input,w},s));
        HIP_CHECK(linear::a8q4g64_linear_wmma32({w.low_codes,w.low_code_bytes,w.high_codes,w.high_code_bytes,
            w.scales,w.scale_bytes,w.status,a.weight_codes,a.weight_code_bytes,a.weight_scales,
            a.weight_scale_bytes,a.output,a.tokens,N,K,K},s));
    }
}
struct Graph {
    hipGraph_t graph{};hipGraphExec_t exec{};
    Graph(hipStream_t s,const std::function<void()>& f) {
        HIP_CHECK(hipStreamBeginCapture(s,hipStreamCaptureModeGlobal));f();
        HIP_CHECK(hipStreamEndCapture(s,&graph));HIP_CHECK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Graph(){(void)hipGraphExecDestroy(exec);(void)hipGraphDestroy(graph);}
    void run(hipStream_t s){HIP_CHECK(hipGraphLaunch(exec,s));}
};
std::vector<double> oracle(unsigned t,const HostActivation& a,const DecodeDot8Weights& w) {
    std::vector<double> result(t*N);
    for(unsigned token=0;token<t;++token)for(unsigned row=0;row<N;++row) {
        double sum=0;
        for(unsigned group=0;group<G;++group) {
            int dot=0;
            for(unsigned lane=0;lane<64;++lane) {
                const std::size_t word=(row/16)*G*64+group*64+(lane/16)*16+row%16;
                const int nibble=(w.codes[word*8+(lane%16)/2]>>((lane%2)*4))&15;
                const int wc=nibble>=8?nibble-16:nibble;
                const auto ai=static_cast<std::size_t>(token)*K+group*64+lane;
                dot+=(unsigned_nibble(a.low,ai)+16*signed_nibble(a.high,ai))*wc;
            }
            sum+=static_cast<double>(dot)*half_value(a.scales[token*G+group])*
                 static_cast<double>(half_value(w.scales[(row/16)*G*16+group*16+row%16]));
        }
        result[token*N+row]=sum;
    }
    return result;
}
struct Error {double relative_l2=0,maximum_absolute=0,reference_maximum=0;};
Error compare(const std::vector<hip_bfloat16>& actual,const std::vector<double>& expected) {
    Error e;double e2=0,r2=0;
    for(std::size_t i=0;i<actual.size();++i) {
        const double v=static_cast<float>(actual[i]);if(!std::isfinite(v))fail("nonfinite finite output");
        const double delta=v-expected[i];e2+=delta*delta;r2+=expected[i]*expected[i];
        e.maximum_absolute=std::max(e.maximum_absolute,std::abs(delta));
        e.reference_maximum=std::max(e.reference_maximum,std::abs(expected[i]));
    }
    e.relative_l2=std::sqrt(e2/std::max(r2,1e-30));
    // BF16 represented-output criterion: normwise plus finite gross pointwise
    // cap, permitting cancellation near zero without allowing isolated corruption.
    if(e.relative_l2>0.01 || e.maximum_absolute>0.01*e.reference_maximum+1e-5)
        fail("independent decoded-Q4 FP64 oracle failed");
    return e;
}
linear::A8G64ActivationWorkspace workspace(unsigned t,Output& o) {
    linear::A8G64ActivationWorkspace w{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(o.scratch.data(),o.scratch.bytes(),t,K,&w));return w;
}
void codec(unsigned t,Output& o,const HostActivation& expected,hipStream_t s) {
    const auto w=workspace(t,o);std::vector<std::uint8_t> low(expected.low.size()),high(expected.high.size());
    std::vector<std::uint16_t> scales(expected.scales.size());std::uint32_t status=~0U;
    transfer(low.data(),w.low_codes,low.size(),hipMemcpyDeviceToHost,s);
    transfer(high.data(),w.high_codes,high.size(),hipMemcpyDeviceToHost,s);
    transfer(scales.data(),w.scales,scales.size()*2,hipMemcpyDeviceToHost,s);
    transfer(&status,w.status,4,hipMemcpyDeviceToHost,s);
    if(low!=expected.low || high!=expected.high || scales!=expected.scales || status)fail("exact A8 codec/status failed");
}
__global__ void scrub(std::uint32_t* p,std::size_t n) {
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=i;
}
double timing(Graph& graph,DeviceBuffer<std::uint32_t>& flush,hipStream_t s) {
    hipLaunchKernelGGL(scrub,dim3(4096),dim3(256),0,s,flush.get(),flush.bytes()/4);
    Event start,end;HIP_CHECK(hipEventRecord(start.get(),s));graph.run(s);HIP_CHECK(hipEventRecord(end.get(),s));
    HIP_CHECK(hipEventSynchronize(end.get()));float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,start.get(),end.get()));return ms;
}
void cell(unsigned t,hipStream_t s,std::ostream& out) {
    const auto host=make_decode_dot8_weights(N,K);const auto base=make_decode_dot8_input(K);
    std::vector<hip_bfloat16> x(t*K);
    for(unsigned token=0;token<t;++token)for(unsigned k=0;k<K;++k)
        x[token*K+k]=hip_bfloat16(static_cast<float>(base[(k/64)*64+(k+token*13)%64])*(token+4)/8.0F);
    Guarded<hip_bfloat16> input(x.size(),s);input.put(x,s);Output control(t,s),candidate(t,s);
    std::array<std::unique_ptr<Weights>,Copies> weights;
    for(auto& w:weights)w=std::make_unique<Weights>(host,s);
    const auto valid=arguments(t,input,*weights[0],candidate);unsigned malformed=0;
    auto reject=[&](auto a,hipStream_t stream){if(linear::qualification::dflash_down_scale_gather(a,stream)!=hipErrorInvalidValue)fail("malformed accepted");++malformed;};
    reject(valid,nullptr);
    auto bad=valid;bad.tokens=4;reject(bad,s);bad=valid;bad.tokens=7;reject(bad,s);
    bad=valid;bad.rows=4096;reject(bad,s);bad=valid;bad.columns-=64;reject(bad,s);
    bad=valid;bad.padded_columns-=64;reject(bad,s);
    bad=valid;--bad.weight_code_bytes;reject(bad,s);bad=valid;--bad.weight_scale_bytes;reject(bad,s);
    bad=valid;--bad.activation_workspace_bytes;reject(bad,s);
    bad=valid;bad.input=nullptr;reject(bad,s);bad=valid;bad.weight_codes=nullptr;reject(bad,s);
    bad=valid;bad.weight_scales=nullptr;reject(bad,s);bad=valid;bad.output=nullptr;reject(bad,s);
    bad=valid;bad.activation_workspace=nullptr;reject(bad,s);
    bad=valid;++bad.weight_codes;reject(bad,s);
    bad=valid;bad.output=const_cast<hip_bfloat16*>(bad.input);reject(bad,s);
    bad=valid;bad.output=reinterpret_cast<hip_bfloat16*>(bad.activation_workspace);reject(bad,s);
    bad=valid;bad.weight_scales=reinterpret_cast<const std::uint16_t*>(bad.weight_codes);reject(bad,s);
    bad=valid;bad.partial_sums=reinterpret_cast<float*>(bad.activation_workspace);reject(bad,s);
    std::array<std::unique_ptr<Graph>,Copies> cg,rg;
    for(unsigned i=0;i<Copies;++i) {
        cg[i]=std::make_unique<Graph>(s,[&]{launch(false,arguments(t,input,*weights[i],control),s);});
        rg[i]=std::make_unique<Graph>(s,[&]{launch(true,arguments(t,input,*weights[i],candidate),s);});
    }
    double max_l2=0,max_abs=0,max_gross_fraction=0;unsigned cases=0;
    auto verify_fixture=[&](bool eager){
        const auto represented=quantize_host(x,t,K);const auto expected=oracle(t,represented,host);
        for(unsigned i=0;i<Copies;++i) {
            if(eager){launch(false,arguments(t,input,*weights[i],control),s);launch(true,arguments(t,input,*weights[i],candidate),s);}
            else {cg[i]->run(s);rg[i]->run(s);}
            for(auto* o:{&control,&candidate}) {
                codec(t,*o,represented,s);auto e=compare(o->value.read(s),expected);
                max_l2=std::max(max_l2,e.relative_l2);max_abs=std::max(max_abs,e.maximum_absolute);
                max_gross_fraction=std::max(max_gross_fraction,e.maximum_absolute/(0.01*e.reference_maximum+1e-5));
                o->value.guards(s);o->scratch.guards(s);
            }
            exact(control.value.read(s),candidate.value.read(s),"candidate/incumbent BF16 mismatch");
            exact(input.read(s),x,"hidden mutation");input.guards(s);
            exact(weights[i]->codes.read(s),host.codes,"code mutation");
            exact(weights[i]->scales.read(s),host.scales,"scale mutation");
            weights[i]->codes.guards(s);weights[i]->scales.guards(s);
        }
        ++cases;
    };
    verify_fixture(true);verify_fixture(false);
    const auto original=x;x[K+17].data=0x7fc1;input.put(x,s);cg[0]->run(s);rg[0]->run(s);
    for(auto* o:{&control,&candidate}) {
        const auto values=o->value.read(s);
        if(!std::all_of(values.begin(),values.end(),[](auto v){return v.data==0x7fc1;}))fail("poison output/status propagation");
        std::uint32_t status=0;transfer(&status,workspace(t,*o).status,4,hipMemcpyDeviceToHost,s);
        if(!status)fail("poison status absent");
    }
    x=original;for(auto& v:x)v=hip_bfloat16(-0.75F*static_cast<float>(v));input.put(x,s);
    HIP_CHECK(hipMemsetAsync(control.scratch.data(),0xff,control.scratch.bytes(),s));
    HIP_CHECK(hipMemsetAsync(candidate.scratch.data(),0xff,candidate.scratch.bytes(),s));verify_fixture(false);
    std::fill(x.begin(),x.end(),hip_bfloat16(0.0F));input.put(x,s);verify_fixture(false);
    x=original;input.put(x,s);verify_fixture(false);power();
    DeviceBuffer<std::uint32_t> flush(80ULL*1024*1024/4);
    for(unsigned i=0;i<Copies;++i){(void)timing(*cg[i],flush,s);(void)timing(*rg[i],flush,s);}
    out<<"{\"tokens\":"<<t<<",\"rows\":5120,\"columns\":17408,\"correctness\":{"
       <<"\"maximum_relative_l2\":"<<max_l2<<",\"maximum_absolute\":"<<max_abs
       <<",\"maximum_gross_cap_fraction\":"<<max_gross_fraction
       <<",\"criterion\":\"fp64_rel_l2_1e-2_gross_1e-2_refmax_plus_1e-5\","
       <<"\"exact_codec\":true,\"exact_incumbent\":true,\"graph_poison_stale_finite\":true,"
       <<"\"guards_immutability\":true,\"finite_cases\":"<<cases<<",\"malformed_cases\":"<<malformed
       <<"},\"samples\":[";
    for(unsigned trial=0;trial<Trials;++trial) {
        const auto i=trial%Copies;const bool first=(trial/Copies)%2==0;double c,r;
        if(first){c=timing(*cg[i],flush,s);r=timing(*rg[i],flush,s);}else{r=timing(*rg[i],flush,s);c=timing(*cg[i],flush,s);}
        if(!(c>0 && r>0))fail("invalid event");if(trial)out<<',';
        out<<"{\"allocation\":"<<i<<",\"control_first\":"<<(first?"true":"false")
           <<",\"control_ms\":"<<c<<",\"candidate_ms\":"<<r<<'}';
    }
    power();out<<"]}";
}
}

int main(int argc,char** argv) {
 try {
    if(argc!=3 || std::string_view(argv[1])!="--out-json")fail("usage: dflash_down_scale_gather_qual --out-json FRESH.json");
    const std::filesystem::path output=argv[2];require_fresh_output(output);power();
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(props.name)!="AMD Radeon AI PRO R9700" || std::string_view(props.gcnArchName)!="gfx1201" ||
       props.warpSize!=32 || (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong device");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\"ninfer.r9700.dflash-down-scale-gather.v1\","
        <<"\"status\":\"qualified\",\"pci\":\"0000:13:00.0\",\"power\":\"auto\","
        <<"\"copies\":3,\"scrub_bytes\":83886080,\"complete_boundary_graph\":true,\"cells\":[";
    cell(5,stream,out);out<<',';cell(6,stream,out);out<<"]}\n";
    HIP_CHECK(hipStreamDestroy(stream));
    const int fd=::open(output.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(fd<0)fail("cannot create immutable result");const auto result=out.str();
    const auto count=::write(fd,result.data(),result.size());::close(fd);
    if(count!=static_cast<ssize_t>(result.size()))fail("short result write");
    std::cout<<"complete boundary qualified; independent timing admission remains\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
