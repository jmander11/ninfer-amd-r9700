#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/gdn/gdn_projection_control_grid_qualification.h"
#include <functional>
#include <numeric>
#include <cstring>

namespace {
namespace candidate = ninfer::ops::r9700::gdn::qualification;
constexpr int K=5120, H=48, Copies=3, Trials=24;
constexpr std::size_t Guard=256;
const std::filesystem::path Pci="/sys/bus/pci/devices/0000:13:00.0";
void power() {
    if(read_text(Pci/"vendor")!="0x1002" || read_text(Pci/"device")!="0x7551" ||
       read_text(Pci/"power_dpm_force_performance_level")!="auto")
        fail("physical R9700 auto power required");
}
void copy(void* dst,const void* src,std::size_t n,hipMemcpyKind kind,hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(dst,src,n,kind,stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}
template<class T> struct Guarded {
    DeviceBuffer<std::uint8_t> storage;
    std::size_t count;
    explicit Guarded(std::size_t n):storage(n*sizeof(T)+2*Guard),count(n) {}
    T* data() const { return reinterpret_cast<T*>(storage.get()+Guard); }
    std::size_t bytes() const { return count*sizeof(T); }
    void init(hipStream_t s) { HIP_CHECK(hipMemsetAsync(storage.get(),0xa5,storage.bytes(),s)); }
    std::vector<T> read(hipStream_t s) const {
        std::vector<T> v(count); copy(v.data(),data(),bytes(),hipMemcpyDeviceToHost,s);return v;
    }
    void guards(hipStream_t s) const {
        std::array<unsigned char,Guard> v{};
        for(auto offset:{std::size_t{0},storage.bytes()-Guard}) {
            copy(v.data(),storage.get()+offset,Guard,hipMemcpyDeviceToHost,s);
            if(!std::all_of(v.begin(),v.end(),[](auto x){return x==0xa5;})) fail("changed canary");
        }
    }
};
struct Weights {
    Guarded<std::uint8_t> q{4096ULL*K/2},v{12288ULL*K/2};
    Guarded<std::uint16_t> qs{4096*80},vs{12288*80};
    Guarded<hip_bfloat16> a{H*K},b{H*K};
    Guarded<float> log{H},bias{H};
};
struct Outputs {
    Guarded<hip_bfloat16> q{4096},v{12288};
    Guarded<float> g{H},beta{H};
    Guarded<std::uint8_t> scratch{linear::a8q4g64_activation_workspace_capacity_bytes(1,K)};
    void init(hipStream_t s) {q.init(s);v.init(s);g.init(s);beta.init(s);scratch.init(s);}
    void guards(hipStream_t s) {q.guards(s);v.guards(s);g.guards(s);beta.guards(s);scratch.guards(s);}
};
candidate::ProjectionControlArgs args(Guarded<hip_bfloat16>& x,Weights& w,Outputs& o) {
    return {{x.data(),w.q.data(),w.q.bytes(),w.qs.data(),w.qs.bytes(),o.q.data(),
        w.v.data(),w.v.bytes(),w.vs.data(),w.vs.bytes(),o.v.data(),
        o.scratch.data(),o.scratch.bytes(),1,K},w.a.data(),w.b.data(),w.a.bytes(),
        w.log.data(),w.bias.data(),o.g.data(),o.beta.data()};
}
void incumbent(const candidate::ProjectionControlArgs& a,hipStream_t s) {
    HIP_CHECK(ninfer::ops::r9700::gdn::bf16_projected_control_t1(a.projection.input,
        a.a_weight,a.b_weight,a.a_log,a.dt_bias,a.g,a.beta,s));
    HIP_CHECK(linear::a8q4g64_gdn_pair_t1(a.projection,s));
}
struct Graph {
    hipGraph_t graph{};hipGraphExec_t exec{};
    Graph(hipStream_t s,const std::function<void()>& f) {
        HIP_CHECK(hipStreamBeginCapture(s,hipStreamCaptureModeGlobal));f();
        HIP_CHECK(hipStreamEndCapture(s,&graph));
        HIP_CHECK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Graph(){(void)hipGraphExecDestroy(exec);(void)hipGraphDestroy(graph);}
    void run(hipStream_t s){HIP_CHECK(hipGraphLaunch(exec,s));}
};
hip_bfloat16 rounded(double value) {
    const auto sign=std::signbit(value)?0x8000U:0U;
    hip_bfloat16 result{};
    if(value==0){result.data=sign;return result;}
    int exponent;std::frexp(std::abs(value),&exponent);--exponent;
    auto mantissa=static_cast<unsigned>(std::nearbyint(std::ldexp(std::abs(value),7-exponent)));
    if(mantissa==256){mantissa=128;++exponent;}
    result.data=static_cast<std::uint16_t>(sign | ((exponent+127)*128+mantissa-128));
    return result;
}
struct Error {double l2=0,absolute=0;std::uint64_t steps=0;};
Error projection_oracle(const std::vector<hip_bfloat16>& actual,const HostActivation& a,
                        const DecodeDot8Weights& w) {
    double e2=0,r2=0;Error e;
    for(std::size_t n=0;n<actual.size();++n) {
        double expected=0;
        for(int group=0;group<80;++group) {
            int dot=0;
            for(int lane=0;lane<64;++lane) {
                const std::size_t word=(n/16)*80*64+group*64+(lane/16)*16+n%16;
                const int nibble=(w.codes[word*8+(lane%16)/2]>>((lane%2)*4))&15;
                const int weight=nibble>=8?nibble-16:nibble;
                const auto i=group*64+lane;
                dot+=(unsigned_nibble(a.low,i)+16*signed_nibble(a.high,i))*weight;
            }
            expected+=static_cast<double>(dot)*half_value(a.scales[group])*
                static_cast<double>(half_value(w.scales[(n/16)*80*16+group*16+n%16]));
        }
        const double got=static_cast<float>(actual[n]);
        if(!std::isfinite(got))fail("nonfinite finite projection");
        e.steps=std::max(e.steps,bf16_steps(actual[n],rounded(expected)));
        const double delta=got-expected;e2+=delta*delta;r2+=expected*expected;
        e.absolute=std::max(e.absolute,std::abs(delta));
    }
    e.l2=std::sqrt(e2/std::max(r2,1e-30));
    if(e.l2>1e-2 || e.steps>2)fail("FP64 projection criterion failed");return e;
}
std::pair<double,double> control_oracle(const std::vector<hip_bfloat16>& x,const std::vector<hip_bfloat16>& a,
    const std::vector<hip_bfloat16>& b,const std::vector<float>& logs,
    const std::vector<float>& bias,const std::vector<float>& g,const std::vector<float>& beta) {
    double max_g=0,max_beta=0;
    for(int h=0;h<H;++h) {
        double da=0,db=0;
        for(int k=0;k<K;++k) {
            da+=static_cast<double>(static_cast<float>(x[k]))*static_cast<float>(a[h*K+k]);
            db+=static_cast<double>(static_cast<float>(x[k]))*static_cast<float>(b[h*K+k]);
        }
        const double av=static_cast<float>(hip_bfloat16(static_cast<float>(da)))+static_cast<double>(bias[h]);
        const double bv=static_cast<float>(hip_bfloat16(static_cast<float>(db)));
        const double eg=-std::exp(static_cast<double>(logs[h]))*(av>20?av:std::log1p(std::exp(av)));
        const double eb=1/(1+std::exp(-bv));
        const double ge=std::abs(g[h]-eg),be=std::abs(beta[h]-eb);
        if(!std::isfinite(g[h]) || !std::isfinite(beta[h]) || ge>0.0625 || be>0.0003)
            fail("independent FP64 BF16-seam control oracle failed");
        max_g=std::max(max_g,ge);max_beta=std::max(max_beta,be);
    }
    return {max_g,max_beta};
}
template<class T> void exact(const std::vector<T>& a,const std::vector<T>& b,const char* why) {
    if(a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))fail(why);
}
void codec(Outputs& o,const HostActivation& expected,hipStream_t s) {
    linear::A8G64ActivationWorkspace w{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(o.scratch.data(),o.scratch.bytes(),1,K,&w));
    std::vector<std::uint8_t> lo(2560),hi(2560);std::vector<std::uint16_t> scales(80);
    std::uint32_t status=123;
    copy(lo.data(),w.low_codes,lo.size(),hipMemcpyDeviceToHost,s);
    copy(hi.data(),w.high_codes,hi.size(),hipMemcpyDeviceToHost,s);
    copy(scales.data(),w.scales,scales.size()*2,hipMemcpyDeviceToHost,s);
    copy(&status,w.status,4,hipMemcpyDeviceToHost,s);
    if(lo!=expected.low || hi!=expected.high || scales!=expected.scales || status)fail("exact codec/status failed");
}
__global__ void scrub(std::uint32_t* p,std::size_t n) {
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=i;
}
double timed(Graph& graph,DeviceBuffer<std::uint32_t>& buffer,hipStream_t s) {
    hipLaunchKernelGGL(scrub,dim3(4096),dim3(256),0,s,buffer.get(),buffer.bytes()/4);
    Event start,end;HIP_CHECK(hipEventRecord(start.get(),s));graph.run(s);
    HIP_CHECK(hipEventRecord(end.get(),s));HIP_CHECK(hipEventSynchronize(end.get()));
    float ms;HIP_CHECK(hipEventElapsedTime(&ms,start.get(),end.get()));return ms;
}
}

int main(int argc,char** argv) {
 try {
    if(argc!=3 || std::string_view(argv[1])!="--out-json")fail("usage: gdn_projection_control_grid_qual --out-json FRESH.json");
    const std::filesystem::path output=argv[2];require_fresh_output(output);power();
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(props.name)!="AMD Radeon AI PRO R9700" ||
       std::string_view(props.gcnArchName)!="gfx1201" || props.warpSize!=32 ||
       (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong GPU");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    auto x=make_decode_dot8_input(K);const auto hostq=make_decode_dot8_weights(4096,K);
    const auto hostv=make_decode_dot8_weights(12288,K);
    std::vector<hip_bfloat16> a(H*K),b(H*K);std::vector<float> logs(H),bias(H);
    for(int h=0;h<H;++h) {
        logs[h]=(h%9-4)/16.0F;bias[h]=(h%13-6)/8.0F;
        if(h==0)bias[h]=25.0F;
        if(h==1)bias[h]=-25.0F;
        for(int k=0;k<K;++k) {
            a[h*K+k]=hip_bfloat16(((h*11+k*7)%31-15)/2048.0F);
            b[h*K+k]=hip_bfloat16(((h*3+k*13)%29-14)/2048.0F);
        }
    }
    Guarded<hip_bfloat16> input(K);input.init(stream);
    std::array<Weights,Copies> weights;Outputs control,result;control.init(stream);result.init(stream);
    for(auto& w:weights) {
#define UPLOAD(plane,host) w.plane.init(stream);copy(w.plane.data(),host.data(),w.plane.bytes(),hipMemcpyHostToDevice,stream)
        UPLOAD(q,hostq.codes);UPLOAD(qs,hostq.scales);UPLOAD(v,hostv.codes);UPLOAD(vs,hostv.scales);
        UPLOAD(a,a);UPLOAD(b,b);UPLOAD(log,logs);UPLOAD(bias,bias);
#undef UPLOAD
    }
    auto ca=args(input,weights[0],result);
    int malformed=0;
    auto reject=[&](candidate::ProjectionControlArgs bad,hipStream_t s){
        if(candidate::projection_control_grid(bad,s)!=hipErrorInvalidValue)fail("accepted malformed boundary");++malformed;};
    reject(ca,nullptr);
    auto bad=ca;bad.projection.tokens=2;reject(bad,stream);
    bad=ca;bad.projection.columns=4992;reject(bad,stream);
    bad=ca;--bad.projection.activation_workspace_bytes;reject(bad,stream);
    bad=ca;--bad.control_weight_bytes;reject(bad,stream);
    bad=ca;--bad.projection.weight0_code_bytes;reject(bad,stream);
    bad=ca;--bad.projection.weight1_scale_bytes;reject(bad,stream);
    bad=ca;++bad.projection.weight0_codes;reject(bad,stream);
    bad=ca;bad.g=bad.beta;reject(bad,stream);
    bad=ca;bad.a_weight=bad.projection.input;reject(bad,stream);
    bad=ca;bad.projection.output0=reinterpret_cast<hip_bfloat16*>(bad.projection.activation_workspace);reject(bad,stream);
    bad=ca;bad.projection.input=nullptr;reject(bad,stream);
    bad=ca;bad.b_weight=nullptr;reject(bad,stream);
    bad=ca;bad.dt_bias=nullptr;reject(bad,stream);
    bad=ca;--bad.projection.weight1_code_bytes;reject(bad,stream);
    bad=ca;--bad.projection.weight0_scale_bytes;reject(bad,stream);
    double max_l2=0,max_g=0,max_beta=0;std::uint64_t max_steps=0;
    auto verify=[&](int allocation){
        const auto activation=quantize_host(x,1,K);
        for(auto* o:{&control,&result}) {
            codec(*o,activation,stream);const auto qo=o->q.read(stream),vo=o->v.read(stream);
            for(const auto e:{projection_oracle(qo,activation,hostq),projection_oracle(vo,activation,hostv)}) {
                max_l2=std::max(max_l2,e.l2);max_steps=std::max(max_steps,e.steps);
            }
            const auto errors=control_oracle(x,a,b,logs,bias,o->g.read(stream),o->beta.read(stream));
            max_g=std::max(max_g,errors.first);max_beta=std::max(max_beta,errors.second);
            o->guards(stream);
        }
        exact(control.q.read(stream),result.q.read(stream),"QK incumbent parity");
        exact(control.v.read(stream),result.v.read(stream),"value-Z incumbent parity");
        exact(control.g.read(stream),result.g.read(stream),"g incumbent parity");
        exact(control.beta.read(stream),result.beta.read(stream),"beta incumbent parity");
        exact(input.read(stream),x,"hidden mutation");input.guards(stream);
        auto& w=weights[allocation];
#define IMMUTABLE(plane,host) exact(w.plane.read(stream),host,"weight mutation");w.plane.guards(stream)
        IMMUTABLE(q,hostq.codes);IMMUTABLE(qs,hostq.scales);IMMUTABLE(v,hostv.codes);IMMUTABLE(vs,hostv.scales);
        IMMUTABLE(a,a);IMMUTABLE(b,b);IMMUTABLE(log,logs);IMMUTABLE(bias,bias);
#undef IMMUTABLE
    };
    copy(input.data(),x.data(),input.bytes(),hipMemcpyHostToDevice,stream);
    for(int i=0;i<Copies;++i) {
        incumbent(args(input,weights[i],control),stream);
        HIP_CHECK(candidate::projection_control_grid(args(input,weights[i],result),stream));verify(i);
    }
    std::array<std::unique_ptr<Graph>,Copies> cg,rg;
    for(int i=0;i<Copies;++i) {
        cg[i]=std::make_unique<Graph>(stream,[&]{incumbent(args(input,weights[i],control),stream);});
        rg[i]=std::make_unique<Graph>(stream,[&]{HIP_CHECK(candidate::projection_control_grid(args(input,weights[i],result),stream));});
    }
    // Replay a captured complete boundary after poison, stale workspace, and changed finite input.
    const auto original=x;x[17].data=0x7fc1;
    copy(input.data(),x.data(),input.bytes(),hipMemcpyHostToDevice,stream);
    cg[0]->run(stream);rg[0]->run(stream);
    for(auto* o:{&control,&result}) {
        linear::A8G64ActivationWorkspace poisoned{};
        HIP_CHECK(linear::a8q4g64_bind_activation_workspace(o->scratch.data(),o->scratch.bytes(),1,K,&poisoned));
        std::uint32_t status=0;copy(&status,poisoned.status,4,hipMemcpyDeviceToHost,stream);
        if(status==0)fail("poison status not published");
        for(const auto& values:{o->q.read(stream),o->v.read(stream)})
            if(!std::all_of(values.begin(),values.end(),[](auto y){return y.data==0x7fc1;}))fail("poison projection");
        for(const auto& values:{o->g.read(stream),o->beta.read(stream)})
            if(!std::all_of(values.begin(),values.end(),[](float y){return std::isnan(y);}))fail("poison controls");
    }
    x=original;for(auto& value:x)value=hip_bfloat16(-0.5F*static_cast<float>(value));
    copy(input.data(),x.data(),input.bytes(),hipMemcpyHostToDevice,stream);
    HIP_CHECK(hipMemsetAsync(control.scratch.data(),0xff,control.scratch.bytes(),stream));
    HIP_CHECK(hipMemsetAsync(result.scratch.data(),0xff,result.scratch.bytes(),stream));
    cg[0]->run(stream);rg[0]->run(stream);verify(0);
    std::fill(x.begin(),x.end(),hip_bfloat16(0.0F));
    copy(input.data(),x.data(),input.bytes(),hipMemcpyHostToDevice,stream);
    cg[0]->run(stream);rg[0]->run(stream);verify(0);
    x=original;copy(input.data(),x.data(),input.bytes(),hipMemcpyHostToDevice,stream);
    for(int i=0;i<Copies;++i){cg[i]->run(stream);rg[i]->run(stream);verify(i);}
    power();DeviceBuffer<std::uint32_t> flush(80ULL*1024*1024/4);
    for(int i=0;i<Copies;++i){(void)timed(*cg[i],flush,stream);(void)timed(*rg[i],flush,stream);}
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"schema\":\"ninfer.r9700.gdn-projection-control-grid.v1\",\"status\":\"qualified\","
        <<"\"projection_max_relative_l2\":"<<max_l2<<",\"projection_max_bf16_steps\":"<<max_steps
        <<",\"control_g_max_absolute_error\":"<<max_g<<",\"control_beta_max_absolute_error\":"<<max_beta
        <<",\"malformed_cases\":"<<malformed
        <<",\"codec_exact\":true,\"incumbent_exact\":true,\"graph_poison_stale_finite\":true,"
        <<"\"guards_and_immutability\":true,\"pci\":\"0000:13:00.0\",\"power\":\"auto\","
        <<"\"copies\":3,\"scrub_bytes\":83886080,\"calls_per_token\":48,\"samples\":[";
    for(int trial=0;trial<Trials;++trial) {
        const int i=trial%Copies;const bool control_first=(trial/Copies)%2==0;
        double c,r;
        if(control_first){c=timed(*cg[i],flush,stream);r=timed(*rg[i],flush,stream);}
        else{r=timed(*rg[i],flush,stream);c=timed(*cg[i],flush,stream);}
        if(!(c>0 && r>0))fail("invalid event timing");
        if(trial)out<<',';
        out<<"{\"allocation\":"<<i<<",\"control_first\":"<<(control_first?"true":"false")
           <<",\"control_ms\":"<<c<<",\"candidate_ms\":"<<r<<'}';
    }
    power();out<<"]}\n";
    const int fd=::open(output.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(fd<0)fail("cannot create immutable qualification result");
    const auto text=out.str();const auto wrote=::write(fd,text.data(),text.size());::close(fd);
    if(wrote!=static_cast<ssize_t>(text.size()))fail("short result write");
    // Graphs must die before their stream.
    for(auto& g:cg)g.reset();for(auto& g:rg)g.reset();
    HIP_CHECK(hipStreamDestroy(stream));
    std::cout<<"GDN complete projection/control boundary qualified; inspect timing admission separately\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
