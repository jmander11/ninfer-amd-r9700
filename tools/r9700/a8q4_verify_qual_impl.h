#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"
#include "ops/r9700/linear/dflash_verify_down.h"
#include "ops/r9700/linear/a8q4_small_batch_projection.h"
#include "ninfer/ops/linear.h"
#include <cstring>
#include <functional>

namespace {
constexpr unsigned Copies=3;
#if defined(NINFER_QUAL_SMALL_BATCH_PROJECTIONS)
unsigned N=0,K=0,G=0;
#else
constexpr unsigned N=5120,K=17408,G=K/64;
#endif
hipError_t owning_launch(const linear::A8Q4G64CandidateArgs& a,hipStream_t s) {
#if defined(NINFER_QUAL_SMALL_BATCH_PROJECTIONS)
    return linear::a8q4_small_batch_projection(a,s);
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
void launch(const linear::A8Q4G64CandidateArgs& a,hipStream_t s) {
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

#if defined(NINFER_QUAL_SMALL_BATCH_PROJECTIONS)
    ninfer::ops::linear(input,weight,output,
#else
    ninfer::ops::dflash_verify_down_linear(input,weight,output,
#endif
        ninfer::DeviceSpan{a.activation_workspace,a.activation_workspace_bytes},s);
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
                              const DecodeDot8Weights& weights,bool full_output=true) {
    PublicReference reference;
    if(full_output) {
        reference.rows.resize(N);
        for(unsigned row=0;row<N;++row)reference.rows[row]=row;
    } else {
        reference.rows={0,1,15,16,N-17,N-16,N-2,N-1};
        for(unsigned i=0;i<24;++i)reference.rows.push_back(i*(N-1)/23);
        std::sort(reference.rows.begin(),reference.rows.end());
        reference.rows.erase(std::unique(reference.rows.begin(),reference.rows.end()),reference.rows.end());
    }
    // Original represented BF16 public input, not the private A8 image.
    // Decode each signed stored Q4 value with its exact FP16 scale and
    // evaluate the complete K reduction in FP64, without staging casts.
    // Decode invariant stored scales once, not once for every scalar product.
    std::vector<double> decoded_scales(weights.scales.size());
    for(std::size_t i=0;i<decoded_scales.size();++i)
        decoded_scales[i]=half_value(weights.scales[i]);
    for(unsigned token=0;token<tokens;++token)for(const unsigned row:reference.rows) {
        double sum=0;
        for(unsigned k=0;k<K;++k) {
            const unsigned group=k/64,lane=k%64;
            const std::size_t word=(row/16)*G*64+group*64+(lane/16)*16+row%16;
            const int nibble=(weights.codes[word*8+(lane%16)/2]>>((lane%2)*4))&15;
            const int code=nibble>=8?nibble-16:nibble;
            const double scale=decoded_scales[(row/16)*G*16+group*16+row%16];
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
        // Suite criterion: each complete output token must meet 2% RMS
        // and 10%-of-reference-RMS gross error. A zero reference requires
        // exact zero, rather than an arbitrary tolerance floor.
        if(r2==0) {
            if(e2!=0)fail("public BF16/Q4 zero-reference mismatch");
            continue;
        }
        const double relative=std::sqrt(e2/r2);
        const double gross=maximum/std::sqrt(r2/reference.rows.size());
        if(relative>0.02 || gross>0.10)
            fail("public BF16/Q4 FP64 oracle criterion failed: N="+std::to_string(N)+
                 " K="+std::to_string(K)+" T="+std::to_string(tokens)+" token="+std::to_string(token)+
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
void cell(unsigned t,hipStream_t s,std::ostream& out) {
    const auto host=make_decode_dot8_weights(N,K);const auto base=make_decode_dot8_input(K);
    std::vector<hip_bfloat16> x(t*K);
    for(unsigned token=0;token<t;++token)for(unsigned k=0;k<K;++k)
        x[token*K+k]=hip_bfloat16(static_cast<float>(base[(k/64)*64+(k+token*13)%64])*(token+4)/8.0F);
    Guarded<hip_bfloat16> input(x.size(),s);input.put(x,s);Output candidate(t,s),control(t,s);
    std::array<std::unique_ptr<Weights>,Copies> weights;
    for(auto& w:weights)w=std::make_unique<Weights>(host,s);
    const auto valid=arguments(t,input,*weights[0],candidate);unsigned malformed=0;
    auto reject=[&](auto a,hipStream_t stream){if(owning_launch(a,stream)!=hipErrorInvalidValue)fail("malformed accepted");++malformed;};
    reject(valid,nullptr);
    auto bad=valid;bad.tokens=1;reject(bad,s);bad=valid;bad.tokens=7;reject(bad,s);
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
    std::array<std::unique_ptr<Graph>,Copies> graphs;
    for(unsigned i=0;i<Copies;++i)
        graphs[i]=std::make_unique<Graph>(s,[&]{launch(arguments(t,input,*weights[i],candidate),s);});
    double max_l2=0,max_abs=0,max_gross_fraction=0;unsigned cases=0;
    PublicError public_error{};std::vector<unsigned> public_rows;
    auto verify_fixture=[&](bool eager){
        const auto represented=quantize_host(x,t,K);const auto expected=oracle(t,represented,host);
        const auto public_expected=public_oracle(t,x,host);public_rows=public_expected.rows;
        // An unchanged generic route is supplementary regression evidence, not
        // the oracle. Check it against the same complete public-input formula.
        const auto control_args=arguments(t,input,*weights[0],control);
        const auto cw=workspace(t,control);
        HIP_CHECK(linear::a8g64_quantize_activation({control_args.input,cw},s));
        HIP_CHECK(linear::a8q4g64_linear_wmma32({cw.low_codes,cw.low_code_bytes,
            cw.high_codes,cw.high_code_bytes,cw.scales,cw.scale_bytes,cw.status,
            control_args.weight_codes,control_args.weight_code_bytes,
            control_args.weight_scales,control_args.weight_scale_bytes,
            control_args.output,t,N,K,K},s));
        const auto control_values=control.value.read(s);
        compare_public(t,control_values,public_expected);
        launch(arguments(t,input,*weights[0],candidate),s);
        const auto eager_reference=candidate.value.read(s);
        for(unsigned i=0;i<Copies;++i) {
            // Poison workspace before every launch, including restored/zero
            // fixtures, so graph replay cannot reuse a stale activation image.
            HIP_CHECK(hipMemsetAsync(candidate.scratch.data(),0xff,candidate.scratch.bytes(),s));
            if(eager)launch(arguments(t,input,*weights[i],candidate),s);
            else graphs[i]->run(s);
            codec(t,candidate,represented,s);const auto actual=candidate.value.read(s);
            const auto e=compare(actual,expected);
            exact(actual,control_values,"public candidate/generic BF16 mismatch");
            const auto pe=compare_public(t,actual,public_expected);
            public_error.relative_rms=std::max(public_error.relative_rms,pe.relative_rms);
            public_error.gross_rms=std::max(public_error.gross_rms,pe.gross_rms);
            max_l2=std::max(max_l2,e.relative_l2);max_abs=std::max(max_abs,e.maximum_absolute);
            max_gross_fraction=std::max(max_gross_fraction,e.maximum_absolute/(0.01*e.reference_maximum+1e-5));
            candidate.value.guards(s);candidate.scratch.guards(s);
            exact(actual,eager_reference,"public eager/graph/allocation BF16 mismatch");
            exact(input.read(s),x,"hidden mutation");input.guards(s);
            exact(weights[i]->codes.read(s),host.codes,"code mutation");
            exact(weights[i]->scales.read(s),host.scales,"scale mutation");
            weights[i]->codes.guards(s);weights[i]->scales.guards(s);
        }
        ++cases;
    };
    verify_fixture(true);verify_fixture(false);
    const auto original=x;x[K+17].data=0x7fc1;input.put(x,s);graphs[0]->run(s);
    for(auto* o:{&candidate}) {
        const auto values=o->value.read(s);
        if(!std::all_of(values.begin(),values.end(),[](auto v){return v.data==0x7fc1;}))fail("poison output/status propagation");
        std::uint32_t status=0;transfer(&status,workspace(t,*o).status,4,hipMemcpyDeviceToHost,s);
        if(!status)fail("poison status absent");
    }
    x=original;for(auto& v:x)v=hip_bfloat16(-0.75F*static_cast<float>(v));input.put(x,s);
    HIP_CHECK(hipMemsetAsync(candidate.scratch.data(),0xff,candidate.scratch.bytes(),s));verify_fixture(false);
    std::fill(x.begin(),x.end(),hip_bfloat16(0.0F));input.put(x,s);verify_fixture(false);
    x=original;input.put(x,s);verify_fixture(false);power();
    out<<"{\"tokens\":"<<t<<",\"rows\":"<<N<<",\"columns\":"<<K<<",\"correctness\":{"
       <<"\"maximum_relative_l2\":"<<max_l2<<",\"maximum_absolute\":"<<max_abs
       <<",\"maximum_gross_cap_fraction\":"<<max_gross_fraction
       <<",\"criterion\":\"fp64_rel_l2_1e-2_gross_1e-2_refmax_plus_1e-5\","
       <<"\"exact_codec\":true,\"exact_eager_graph\":true,\"graph_poison_stale_finite\":true,"
       <<"\"guards_immutability\":true,\"finite_cases\":"<<cases<<",\"malformed_cases\":"<<malformed
       <<"},\"public_bf16_oracle\":{\"criterion\":\"per_token_all_rows_relative_rms_le_0.02_and_max_error_le_0.10_reference_rms\","
       <<"\"full_k\":"<<K<<",\"all_tokens\":true,\"maximum_relative_rms\":"<<public_error.relative_rms
       <<",\"maximum_error_over_reference_rms\":"<<public_error.gross_rms
       <<",\"rows_checked\":"<<public_rows.size()<<",\"generic_control_checked\":true}}";
}
}
#ifndef NINFER_A8Q4_VERIFY_QUAL_NO_MAIN
int main(int argc,char** argv) {
 try {
    bool mlp_only=false,output_only=false;
#if defined(NINFER_QUAL_SMALL_BATCH_PROJECTIONS)
    mlp_only=argc==4 && std::string_view(argv[3])=="--mlp-only";
    output_only=argc==4 && std::string_view(argv[3])=="--output-only";
#endif
    if((argc!=3 && !mlp_only && !output_only) || std::string_view(argv[1])!="--out-json")
        fail("usage: selected_q4_qual --out-json FRESH.json [--mlp-only|--output-only]");
    const std::filesystem::path output=argv[2];require_fresh_output(output);power();
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(props.name)!="AMD Radeon AI PRO R9700" || std::string_view(props.gcnArchName)!="gfx1201" ||
       props.warpSize!=32 || (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong device");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\""
#if defined(NINFER_QUAL_SMALL_BATCH_PROJECTIONS)
        <<"ninfer.r9700.a8q4-small-batch-projections.v2"
#else
        <<"ninfer.r9700.dflash-verify-down.v2"
#endif
        <<"\",\"status\":\"qualified\",\"public_dispatch_tested\":true,"
          "\"pci\":\"0000:13:00.0\",\"power\":\"auto\",\"copies\":3,\"scope\":\""
        <<(mlp_only?"mlp_only":output_only?"output_only":"complete_owner")<<"\",\"cells\":[";
#if defined(NINFER_QUAL_SMALL_BATCH_PROJECTIONS)
    constexpr std::array<std::array<unsigned,2>,5> shapes{{{34816,5120},{5120,6144},{12288,5120},{4096,5120},{5120,17408}}};
    bool first=true;
    for(const auto& shape:shapes) {
        if(mlp_only && shape!=std::array<unsigned,2>{34816,5120} &&
           shape!=std::array<unsigned,2>{5120,17408})continue;
        if(output_only && shape!=std::array<unsigned,2>{5120,6144})continue;
        N=shape[0];K=shape[1];G=K/64;
        for(unsigned t:{2U,3U,4U,5U,6U,12U,18U,24U}) {
            if(!linear::detail::use_a8q4_small_batch_projection(t,N,K,K))continue;
            if(!first)out<<',';first=false;cell(t,stream,out);
        }
    }
#else
    cell(5,stream,out);out<<',';cell(6,stream,out);
#endif
    out<<"]}\n";
    HIP_CHECK(hipStreamDestroy(stream));power();
    const int fd=::open(output.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(fd<0)fail("cannot create immutable result");const auto result=out.str();
    const auto count=::write(fd,result.data(),result.size());::close(fd);
    if(count!=static_cast<ssize_t>(result.size()))fail("short result write");
    std::cout<<"selected public Q4 routes qualified\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
