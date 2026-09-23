#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"
#include "ops/r9700/linear/dflash_verify_down.h"
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
#include "ops/r9700/linear/a8q4_projection_scale_gather.h"
#endif
#include "ninfer/ops/linear.h"
#include <cstring>
#include <functional>

namespace {
constexpr unsigned Copies=3,Trials=24;
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
// This standalone screen executes exactly three shapes serially. These explicit
// parameters let it reuse the established oracle/codec/graph harness unchanged.
unsigned N=0,K=0,G=0;
#else
constexpr unsigned N=5120,K=17408,G=K/64;
#endif
hipError_t candidate_launch(const linear::A8Q4G64CandidateArgs& a,hipStream_t s) {
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
    return linear::a8q4_projection_scale_gather_candidate(a,s);
#else
    return linear::a8q4g64_dflash_verify_down(a,s);
#endif
}
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
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
    if(candidate) { HIP_CHECK(linear::a8q4g64_linear_candidate(a,s)); return; }
#endif
    if(candidate) {
        ninfer::Tensor input(const_cast<hip_bfloat16*>(a.input), ninfer::DType::BF16,
                             {static_cast<int>(K), static_cast<int>(a.tokens)});
        ninfer::Tensor output(a.output, ninfer::DType::BF16,
                              {static_cast<int>(N), static_cast<int>(a.tokens)});
        ninfer::Weight weight{};
        weight.qtype=ninfer::QType::Q4G64_F16S;weight.layout=ninfer::QuantLayout::Q4N16K16;
        weight.ndim=2;weight.n=weight.shape[0]=weight.padded_shape[0]=N;
        weight.k=weight.shape[1]=weight.padded_shape[1]=K;
        weight.group=weight.group_size=64;weight.scale_dtype=ninfer::DType::FP16;
        weight.qdata=const_cast<std::uint8_t*>(a.weight_codes);
        weight.scales=const_cast<std::uint16_t*>(a.weight_scales);
        weight.qdata_bytes=a.weight_code_bytes;weight.scale_bytes=a.weight_scale_bytes;
        ninfer::ops::dflash_verify_down_linear(input,weight,output,
            ninfer::DeviceSpan{a.activation_workspace,a.activation_workspace_bytes},s);
    }
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
struct PublicReference {
    std::vector<unsigned> rows;
    std::vector<double> values;
};
PublicReference public_oracle(unsigned tokens,const std::vector<hip_bfloat16>& input,
                              const DecodeDot8Weights& weights) {
    PublicReference reference;
    reference.rows={0,1,15,16,N-17,N-16,N-2,N-1};
    for(unsigned i=0;i<24;++i)reference.rows.push_back(i*(N-1)/23);
    std::sort(reference.rows.begin(),reference.rows.end());
    reference.rows.erase(std::unique(reference.rows.begin(),reference.rows.end()),reference.rows.end());
    // Original represented BF16 public input, not the private A8 image.
    // Decode each signed stored Q4 value with its exact FP16 scale and
    // evaluate the complete K reduction in FP64, without staging casts.
    for(unsigned token=0;token<tokens;++token)for(const unsigned row:reference.rows) {
        double sum=0;
        for(unsigned k=0;k<K;++k) {
            const unsigned group=k/64,lane=k%64;
            const std::size_t word=(row/16)*G*64+group*64+(lane/16)*16+row%16;
            const int nibble=(weights.codes[word*8+(lane%16)/2]>>((lane%2)*4))&15;
            const int code=nibble>=8?nibble-16:nibble;
            const double scale=half_value(weights.scales[(row/16)*G*16+group*16+row%16]);
            sum+=static_cast<double>(static_cast<float>(input[token*K+k]))*code*scale;
        }
        reference.values.push_back(sum);
    }
    return reference;
}
struct PublicError {double relative_rms=0,gross_rms=0;};
PublicError compare_public(unsigned tokens,const std::vector<hip_bfloat16>& actual,
                           const PublicReference& reference) {
    PublicError result;
    for(unsigned token=0;token<tokens;++token) {
        double e2=0,r2=0,maximum=0;
        for(std::size_t i=0;i<reference.rows.size();++i) {
            const double expected=reference.values[token*reference.rows.size()+i];
            const double observed=static_cast<float>(actual[token*N+reference.rows[i]]);
            if(!std::isfinite(observed))fail("nonfinite public Linear output");
            const double error=observed-expected;
            e2+=error*error;r2+=expected*expected;maximum=std::max(maximum,std::abs(error));
        }
        // Declared before physical execution: each token must meet 2% RMS
        // and 10%-of-reference-RMS gross error. A zero reference requires
        // exact zero, rather than an arbitrary tolerance floor.
        if(r2==0) {
            if(e2!=0)fail("public BF16/Q4 zero-reference mismatch");
            continue;
        }
        const double relative=std::sqrt(e2/r2);
        const double gross=maximum/std::sqrt(r2/reference.rows.size());
        if(relative>0.02 || gross>0.10)
            fail("public BF16/Q4 FP64 oracle criterion failed: token="+std::to_string(token)+
                 " relative_rms="+std::to_string(relative)+" gross_rms="+std::to_string(gross));
        result.relative_rms=std::max(result.relative_rms,relative);
        result.gross_rms=std::max(result.gross_rms,gross);
    }
    return result;
}
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
struct CellTiming { double control_ms, candidate_ms; };
CellTiming cell(unsigned t,hipStream_t s,std::ostream& out) {
    const auto host=make_decode_dot8_weights(N,K);const auto base=make_decode_dot8_input(K);
    std::vector<hip_bfloat16> x(t*K);
    for(unsigned token=0;token<t;++token)for(unsigned k=0;k<K;++k)
        x[token*K+k]=hip_bfloat16(static_cast<float>(base[(k/64)*64+(k+token*13)%64])*(token+4)/8.0F);
    Guarded<hip_bfloat16> input(x.size(),s);input.put(x,s);Output control(t,s),candidate(t,s);
    std::array<std::unique_ptr<Weights>,Copies> weights;
    for(auto& w:weights)w=std::make_unique<Weights>(host,s);
    const auto valid=arguments(t,input,*weights[0],candidate);unsigned malformed=0;
    auto reject=[&](auto a,hipStream_t stream){if(candidate_launch(a,stream)!=hipErrorInvalidValue)fail("malformed accepted");++malformed;};
    reject(valid,nullptr);
    auto bad=valid;bad.tokens=4;reject(bad,s);bad=valid;bad.tokens=7;reject(bad,s);
    bad=valid;bad.rows=N-16;reject(bad,s);bad=valid;bad.columns-=64;reject(bad,s);
    bad=valid;bad.padded_columns-=64;reject(bad,s);
    bad=valid;--bad.weight_code_bytes;reject(bad,s);bad=valid;--bad.weight_scale_bytes;reject(bad,s);
    bad=valid;--bad.activation_workspace_bytes;reject(bad,s);
    bad=valid;++bad.activation_workspace_bytes;reject(bad,s);
    bad=valid;bad.input=nullptr;reject(bad,s);bad=valid;bad.weight_codes=nullptr;reject(bad,s);
    bad=valid;bad.weight_scales=nullptr;reject(bad,s);bad=valid;bad.output=nullptr;reject(bad,s);
    bad=valid;bad.activation_workspace=nullptr;reject(bad,s);
    bad=valid;++bad.weight_codes;reject(bad,s);
    bad=valid;bad.output=const_cast<hip_bfloat16*>(bad.input);reject(bad,s);
    bad=valid;bad.output=reinterpret_cast<hip_bfloat16*>(bad.activation_workspace);reject(bad,s);
    bad=valid;bad.weight_scales=reinterpret_cast<const std::uint16_t*>(bad.weight_codes);reject(bad,s);
    std::array<std::unique_ptr<Graph>,Copies> cg,rg;
    for(unsigned i=0;i<Copies;++i) {
        cg[i]=std::make_unique<Graph>(s,[&]{launch(false,arguments(t,input,*weights[i],control),s);});
        rg[i]=std::make_unique<Graph>(s,[&]{launch(true,arguments(t,input,*weights[i],candidate),s);});
    }
    double max_l2=0,max_abs=0,max_gross_fraction=0;unsigned cases=0;
    PublicError public_error{};std::vector<unsigned> public_rows;
    auto verify_fixture=[&](bool eager){
        const auto represented=quantize_host(x,t,K);const auto expected=oracle(t,represented,host);
        const auto public_expected=public_oracle(t,x,host);public_rows=public_expected.rows;
        for(unsigned i=0;i<Copies;++i) {
            if(eager){launch(false,arguments(t,input,*weights[i],control),s);launch(true,arguments(t,input,*weights[i],candidate),s);}
            else {cg[i]->run(s);rg[i]->run(s);}
            for(auto* o:{&control,&candidate}) {
                codec(t,*o,represented,s);const auto actual=o->value.read(s);
                auto e=compare(actual,expected);
                const auto pe=compare_public(t,actual,public_expected);
                public_error.relative_rms=std::max(public_error.relative_rms,pe.relative_rms);
                public_error.gross_rms=std::max(public_error.gross_rms,pe.gross_rms);
                max_l2=std::max(max_l2,e.relative_l2);max_abs=std::max(max_abs,e.maximum_absolute);
                max_gross_fraction=std::max(max_gross_fraction,e.maximum_absolute/(0.01*e.reference_maximum+1e-5));
                o->value.guards(s);o->scratch.guards(s);
            }
            exact(control.value.read(s),candidate.value.read(s),"candidate/incumbent BF16 mismatch");
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
            // The selected generic dispatcher above is the timed/graph route.
            // Retain direct owning-entry checks against both independent oracles.
            HIP_CHECK(candidate_launch(arguments(t,input,*weights[i],candidate),s));
            codec(t,candidate,represented,s);
            const auto direct=candidate.value.read(s);
            (void)compare(direct,expected);(void)compare_public(t,direct,public_expected);
            exact(control.value.read(s),direct,"direct/generic production BF16 mismatch");
            candidate.value.guards(s);candidate.scratch.guards(s);
#endif
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
    out<<"{\"tokens\":"<<t<<",\"rows\":"<<N<<",\"columns\":"<<K<<",\"correctness\":{"
       <<"\"maximum_relative_l2\":"<<max_l2<<",\"maximum_absolute\":"<<max_abs
       <<",\"maximum_gross_cap_fraction\":"<<max_gross_fraction
       <<",\"criterion\":\"fp64_rel_l2_1e-2_gross_1e-2_refmax_plus_1e-5\","
       <<"\"exact_codec\":true,\"exact_incumbent\":true,\"graph_poison_stale_finite\":true,"
       <<"\"guards_immutability\":true,\"finite_cases\":"<<cases<<",\"malformed_cases\":"<<malformed
       <<"},\"public_bf16_oracle\":{\"criterion\":\"per_token_relative_rms_le_0.02_and_max_error_le_0.10_reference_rms\","
       <<"\"full_k\":"<<K<<",\"all_tokens\":true,\"maximum_relative_rms\":"<<public_error.relative_rms
       <<",\"maximum_error_over_reference_rms\":"<<public_error.gross_rms<<",\"sampled_rows\":[";
    for(std::size_t i=0;i<public_rows.size();++i)out<<(i?",":"")<<public_rows[i];
    out<<"]},\"samples\":[";
    std::array<double,Trials> control_times{},candidate_times{};
    for(unsigned trial=0;trial<Trials;++trial) {
        const auto i=trial%Copies;const bool first=(trial/Copies)%2==0;double c,r;
        if(first){c=timing(*cg[i],flush,s);r=timing(*rg[i],flush,s);}else{r=timing(*rg[i],flush,s);c=timing(*cg[i],flush,s);}
        if(!(c>0 && r>0))fail("invalid event");if(trial)out<<',';
        control_times[trial]=c;candidate_times[trial]=r;
        out<<"{\"allocation\":"<<i<<",\"control_first\":"<<(first?"true":"false")
           <<",\"control_ms\":"<<c<<",\"candidate_ms\":"<<r<<'}';
    }
    std::sort(control_times.begin(),control_times.end());
    std::sort(candidate_times.begin(),candidate_times.end());
    const CellTiming result{(control_times[Trials/2-1]+control_times[Trials/2])/2,
                            (candidate_times[Trials/2-1]+candidate_times[Trials/2])/2};
    power();out<<"],\"control_median_ms\":"<<result.control_ms
               <<",\"candidate_median_ms\":"<<result.candidate_ms<<'}';
    return result;
}
}

int main(int argc,char** argv) {
 try {
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
    if(argc!=7 || std::string_view(argv[1])!="--out-json" ||
       std::string_view(argv[3])!="--round-ms-t5" || std::string_view(argv[5])!="--round-ms-t6")
        fail("usage: qualifier --out-json FRESH.json --round-ms-t5 MS --round-ms-t6 MS");
    const auto duration=[](const char* value) {
        std::size_t consumed=0;const double result=std::stod(value,&consumed);
        if(consumed!=std::strlen(value) || !std::isfinite(result) || result<=0)fail("invalid reference round duration");
        return result;
    };
    const std::array<double,2> round_ms{duration(argv[4]),duration(argv[6])};
#else
    if(argc!=3 || std::string_view(argv[1])!="--out-json")fail("usage: scale_gather_qual --out-json FRESH.json");
#endif
    const std::filesystem::path output=argv[2];require_fresh_output(output);power();
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(props.name)!="AMD Radeon AI PRO R9700" || std::string_view(props.gcnArchName)!="gfx1201" ||
       props.warpSize!=32 || (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong device");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\""
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
        <<"ninfer.r9700.a8q4-projection-scale-gather.v2"
#else
        <<"ninfer.r9700.dflash-down-scale-gather.v1\","
#endif
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
        <<"\",\"production_dispatch_tested\":true,\"owning_direct_oracle_tested\":true,"
#endif
        <<"\"status\":\"qualified\",\"pci\":\"0000:13:00.0\",\"power\":\"auto\","
        <<"\"copies\":3,\"scrub_bytes\":83886080,\"complete_boundary_graph\":true,\"cells\":[";
#if defined(NINFER_QUAL_PROJECTION_SCREEN)
    struct Projection {unsigned rows,columns,calls;const char* role;};
    constexpr std::array<Projection,3> projections{{{5120,6144,60,"output"},
        {12288,5120,48,"value_z"},{4096,5120,48,"query_key"}}};
    std::array<std::array<CellTiming,2>,3> measured{};
    bool first=true;
    for(std::size_t shape=0;shape<projections.size();++shape) {
        N=projections[shape].rows;K=projections[shape].columns;G=K/64;
        for(unsigned width=0;width<2;++width) {
            if(!first)out<<',';first=false;
            measured[shape][width]=cell(width+5,stream,out);
        }
    }
    out<<"],\"weighted_round_screen\":[";
    for(unsigned width=0;width<2;++width) {
        double control=0,candidate=0;
        for(std::size_t shape=0;shape<projections.size();++shape) {
            control+=projections[shape].calls*measured[shape][width].control_ms;
            candidate+=projections[shape].calls*measured[shape][width].candidate_ms;
        }
        const double saving=control-candidate;
        out<<(width?",":"")<<"{\"tokens\":"<<width+5
           <<",\"formula\":\"60*output + 48*value_z + 48*query_key\",\"control_weighted_ms\":"<<control
           <<",\"candidate_weighted_ms\":"<<candidate<<",\"saving_ms\":"<<saving
           <<",\"reference_round_ms\":"<<round_ms[width]<<",\"minimum_fraction\":0.02"
           <<",\"clears_screen\":"<<(saving>0.02*round_ms[width]?"true":"false")<<'}';
    }
    out<<"],\"timing_scope\":\"complete fresh quantize+Linear graph; 24 alternating trials, three allocations,80MiB scrub before each call\","
          "\"screen_is_whole_admission\":false}\n";
#else
    cell(5,stream,out);out<<',';cell(6,stream,out);out<<"]}\n";
#endif
    HIP_CHECK(hipStreamDestroy(stream));
    const int fd=::open(output.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(fd<0)fail("cannot create immutable result");const auto result=out.str();
    const auto count=::write(fd,result.data(),result.size());::close(fd);
    if(count!=static_cast<ssize_t>(result.size()))fail("short result write");
    std::cout<<"complete boundary qualified; independent timing admission remains\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
