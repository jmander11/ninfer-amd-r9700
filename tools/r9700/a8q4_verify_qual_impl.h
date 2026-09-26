#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"
#include "ops/r9700/linear/a8q4_small_batch_projection.h"
#include "ninfer/ops/linear.h"
#include <cstring>
#include <functional>

namespace {
constexpr unsigned Copies=3;
unsigned N=0,K=0,G=0;
hipError_t owning_launch(const linear::A8Q4G64CandidateArgs& a,hipStream_t s) {
    return linear::a8q4_small_batch_projection(a,s);
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

    ninfer::ops::linear(input,weight,output,
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
std::vector<double> oracle(unsigned t,const HostActivation& a,const DecodeDot8Weights& w,
                           std::vector<double>* absolute_group_sums=nullptr) {
    std::vector<double> result(t*N);
    if(absolute_group_sums)absolute_group_sums->assign(t*N,0.0);
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
            const double term=static_cast<double>(dot)*half_value(a.scales[token*G+group])*
                 static_cast<double>(half_value(w.scales[(row/16)*G*16+group*16+row%16]));
            sum+=term;
            if(absolute_group_sums)(*absolute_group_sums)[token*N+row]+=std::abs(term);
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
// Implementation-profile error budget, not the public mathematical oracle.
// Exact INT32 G64 dots and exact FP16*FP16 scale products feed G FP32 FMAs,
// followed by one BF16 RNE output cast. See tools/r9700/README.md.
struct A8ErrorBudget {
    std::vector<double> represented;
    std::vector<double> arithmetic;
};
double gamma(unsigned count,double unit) {
    return std::nextafter((count*unit)/(1.0-count*unit),
                          std::numeric_limits<double>::infinity());
}
double bound_add(double a,double b) {
    return a==0.0 && b==0.0 ? 0.0 :
        std::nextafter(a+b,std::numeric_limits<double>::infinity());
}
double bound_multiply(double a,double b) {
    return a==0.0 || b==0.0 ? 0.0 :
        std::nextafter(a*b,std::numeric_limits<double>::infinity());
}
A8ErrorBudget a8_error_budget(unsigned tokens,const HostActivation& activation,
                              const DecodeDot8Weights& weights) {
    std::vector<double> absolute_sums;
    A8ErrorBudget budget;
    budget.represented=oracle(tokens,activation,weights,&absolute_sums);
    budget.arithmetic.resize(budget.represented.size());
    constexpr double ub=0x1p-8,uf=0x1p-24,ud=0x1p-53;
    for(std::size_t i=0;i<budget.represented.size();++i) {
        // Inflate the FP64 absolute sum to bound its own accumulation error.
        const double denominator=std::nextafter(1.0-gamma(G,ud),0.0);
        const double sum=absolute_sums[i]==0.0 ? 0.0 :
            std::nextafter(absolute_sums[i]/denominator,std::numeric_limits<double>::infinity());
        const double oracle_round=bound_multiply(gamma(G,ud),sum);
        const double accumulation=bound_multiply(gamma(G,uf),sum);
        const double before_cast=bound_add(oracle_round,accumulation);
        budget.arithmetic[i]=bound_add(before_cast,bound_multiply(ub,
            bound_add(std::abs(budget.represented[i]),before_cast)));
    }
    return budget;
}
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
struct PublicError {
    double relative_rms=0,gross_rms=0,public_bound_fraction=0,arithmetic_bound_fraction=0;
    unsigned zero_reference_nonzero_tokens=0;
};
PublicError compare_public(unsigned tokens,const std::vector<hip_bfloat16>& actual,
                           const PublicReference& reference,const A8ErrorBudget& budget) {
    PublicError result;
    for(unsigned token=0;token<tokens;++token) {
        double e2=0,r2=0,maximum=0,b2=0,a2=0,ab2=0;
        for(std::size_t i=0;i<reference.rows.size();++i) {
            const auto index=token*N+reference.rows[i];
            const double expected=reference.values[token*reference.rows.size()+i];
            const double observed=static_cast<float>(actual[index]);
            const double represented=budget.represented[index];
            const double arithmetic=budget.arithmetic[index];
            if(!std::isfinite(observed) || !std::isfinite(expected) ||
               !std::isfinite(represented) || !std::isfinite(arithmetic))
                fail("nonfinite public Linear output/reference/budget");
            const double error=observed-expected;
            const double arithmetic_error=std::abs(observed-represented);
            const double quantization=represented==expected ? 0.0 : std::nextafter(
                std::abs(represented-expected),std::numeric_limits<double>::infinity());
            const double bound=bound_add(quantization,arithmetic);
            // Both checks are necessary: a large quantization allowance must
            // never hide an arithmetic defect in the represented A8 operation.
            if(std::abs(error)>bound || arithmetic_error>arithmetic)
                fail("public BF16/Q4 A8 forward bound failed: N="+std::to_string(N)+
                     " K="+std::to_string(K)+" T="+std::to_string(tokens)+
                     " token="+std::to_string(token)+" row="+std::to_string(reference.rows[i]));
            e2+=error*error;r2+=expected*expected;maximum=std::max(maximum,std::abs(error));
            b2+=bound*bound;a2+=arithmetic_error*arithmetic_error;ab2+=arithmetic*arithmetic;
        }
        // Normwise budget plus finite per-output caps above. There is no
        // empirical threshold or zero-reference exception to quantization.
        if(e2>b2 || a2>ab2)fail("A8 forward norm bound failed");
        if(b2>0)result.public_bound_fraction=std::max(result.public_bound_fraction,std::sqrt(e2/b2));
        if(ab2>0)result.arithmetic_bound_fraction=std::max(result.arithmetic_bound_fraction,std::sqrt(a2/ab2));
        if(r2==0) {
            if(e2!=0)++result.zero_reference_nonzero_tokens;
            continue;
        }
        const double relative=std::sqrt(e2/r2);
        const double gross=maximum/std::sqrt(r2/reference.rows.size());
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
    double max_l2=0,max_abs=0,max_gross_fraction=0;unsigned cases=0,mutations_rejected=0;
    PublicError public_error{};std::vector<unsigned> public_rows;
    auto verify_fixture=[&](bool eager){
        const auto represented=quantize_host(x,t,K);const auto budget=a8_error_budget(t,represented,host);
        const auto& expected=budget.represented;
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
        compare_public(t,control_values,public_expected,budget);
        if(eager) {
            // A profile allowance must not make this a plausibility test.
            // Deliberately corrupt a real output at its largest-magnitude row.
            const auto largest=static_cast<std::size_t>(std::max_element(
                control_values.begin(),control_values.end(),[](auto a,auto b){
                    return std::abs(static_cast<float>(a))<std::abs(static_cast<float>(b));
                })-control_values.begin());
            for(float multiplier:{-1.0F,0.0F,1.125F}) {
                auto damaged=control_values;
                damaged[largest]=hip_bfloat16(multiplier*static_cast<float>(damaged[largest]));
                bool rejected=false;
                try {(void)compare_public(t,damaged,public_expected,budget);}
                catch(const std::runtime_error&){rejected=true;}
                if(!rejected)fail("A8 criterion accepted output corruption");
                ++mutations_rejected;
            }
        }
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
            const auto pe=compare_public(t,actual,public_expected,budget);
            public_error.relative_rms=std::max(public_error.relative_rms,pe.relative_rms);
            public_error.gross_rms=std::max(public_error.gross_rms,pe.gross_rms);
            public_error.public_bound_fraction=std::max(public_error.public_bound_fraction,pe.public_bound_fraction);
            public_error.arithmetic_bound_fraction=std::max(public_error.arithmetic_bound_fraction,pe.arithmetic_bound_fraction);
            public_error.zero_reference_nonzero_tokens=std::max(public_error.zero_reference_nonzero_tokens,
                                                               pe.zero_reference_nonzero_tokens);
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
    const auto original=x;x[(t>1U?K:0U)+17].data=0x7fc1;input.put(x,s);graphs[0]->run(s);
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
       <<"},\"public_bf16_oracle\":{\"criterion\":\"a8g64_exact_int_fp32_fma_bf16_forward_bound_v1\","
       <<"\"full_k\":"<<K<<",\"all_tokens\":true,\"maximum_relative_rms\":"<<public_error.relative_rms
       <<",\"maximum_error_over_reference_rms\":"<<public_error.gross_rms
       <<",\"maximum_public_norm_bound_fraction\":"<<public_error.public_bound_fraction
       <<",\"maximum_arithmetic_norm_bound_fraction\":"<<public_error.arithmetic_bound_fraction
       <<",\"zero_reference_nonzero_tokens\":"<<public_error.zero_reference_nonzero_tokens
       <<",\"output_corruptions_rejected\":"<<mutations_rejected
       <<",\"historical_2pct_rms_10pct_gross_pass\":"
       <<(public_error.relative_rms<=0.02 && public_error.gross_rms<=0.10 &&
          public_error.zero_reference_nonzero_tokens==0?"true":"false")
       <<",\"rows_checked\":"<<public_rows.size()<<",\"generic_control_oracle_checked\":true}}";
}
}
// The paired small-batch launch (two projections of one prepared activation) through the public
// shared-activation Op, each output checked against the complete FP64 public oracle bound.
void pair_cell(unsigned t,unsigned n0,unsigned n1,hipStream_t s,std::ostream& out) {
    K=5120;G=K/64;
    const auto base=make_decode_dot8_input(K);
    std::vector<hip_bfloat16> x(t*K);
    for(unsigned token=0;token<t;++token)for(unsigned k=0;k<K;++k)
        x[token*K+k]=hip_bfloat16(static_cast<float>(base[(k/64)*64+(k+token*13)%64])*(token+4)/8.0F);
    Guarded<hip_bfloat16> input(x.size(),s);input.put(x,s);
    Guarded<std::uint8_t> scratch(linear::a8q4g64_activation_workspace_capacity_bytes(t,K),s);
    const std::array<unsigned,2> rows{n0,n1};
    std::array<DecodeDot8Weights,2> host{make_decode_dot8_weights(n0,K),make_decode_dot8_weights(n1,K)};
    std::array<std::unique_ptr<Weights>,2> weights{std::make_unique<Weights>(host[0],s),
                                                   std::make_unique<Weights>(host[1],s)};
    std::array<std::unique_ptr<Guarded<hip_bfloat16>>,2> outputs{
        std::make_unique<Guarded<hip_bfloat16>>(t*n0,s),std::make_unique<Guarded<hip_bfloat16>>(t*n1,s)};
    linear::A8Q4G64SharedActivationArgs args{};
    args.input=input.data();args.activation_workspace=scratch.data();
    args.activation_workspace_bytes=scratch.bytes();args.projection_count=2;args.tokens=t;args.columns=K;
    for(unsigned i=0;i<2;++i)
        args.projections[i]={weights[i]->codes.data(),weights[i]->codes.bytes(),
                              weights[i]->scales.data(),weights[i]->scales.bytes(),
                              outputs[i]->data(),rows[i]};
    HIP_CHECK(hipMemsetAsync(scratch.data(),0xff,scratch.bytes(),s));  // stale status/planes
    HIP_CHECK(linear::a8q4g64_shared_activation_linear(args,s));
    HIP_CHECK(hipStreamSynchronize(s));
    const auto represented=quantize_host(x,t,K);
    double worst_public=0,worst_arithmetic=0;
    for(unsigned i=0;i<2;++i) {
        N=rows[i];
        const auto budget=a8_error_budget(t,represented,host[i]);
        const auto reference=public_oracle(t,x,host[i]);
        const auto error=compare_public(t,outputs[i]->read(s),reference,budget);
        worst_public=std::max(worst_public,error.public_bound_fraction);
        worst_arithmetic=std::max(worst_arithmetic,error.arithmetic_bound_fraction);
        outputs[i]->guards(s);
    }
    scratch.guards(s);input.guards(s);
    out<<"{\"tokens\":"<<t<<",\"rows\":["<<n0<<','<<n1<<"],\"columns\":"<<K
       <<",\"maximum_public_norm_bound_fraction\":"<<worst_public
       <<",\"maximum_arithmetic_norm_bound_fraction\":"<<worst_arithmetic<<'}';
}
#ifndef NINFER_A8Q4_VERIFY_QUAL_NO_MAIN
int main(int argc,char** argv) {
 try {
    bool mlp_only=false,output_only=false,draft_only=false,projection_only=false,concurrent_only=false;
    mlp_only=argc==4 && std::string_view(argv[3])=="--mlp-only";
    output_only=argc==4 && std::string_view(argv[3])=="--output-only";
    draft_only=argc==4 && std::string_view(argv[3])=="--draft-only";
    projection_only=argc==4 && std::string_view(argv[3])=="--projection-only";
    concurrent_only=argc==4 && std::string_view(argv[3])=="--concurrent-only";
    if((argc!=3 && !mlp_only && !output_only && !draft_only && !projection_only && !concurrent_only) || std::string_view(argv[1])!="--out-json")
        fail("usage: selected_q4_qual --out-json FRESH.json [--mlp-only|--output-only|--draft-only|--projection-only|--concurrent-only]");
    const std::filesystem::path output=argv[2];require_fresh_output(output);power();
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(props.name)!="AMD Radeon AI PRO R9700" || std::string_view(props.gcnArchName)!="gfx1201" ||
       props.warpSize!=32 || (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong device");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\""
        <<"ninfer.r9700.a8q4-small-batch-projections.v3"
        <<"\",\"status\":\"qualified\",\"public_dispatch_tested\":true,"
          "\"pci\":\"0000:13:00.0\",\"power\":\"auto\",\"copies\":3,\"scope\":\""
        <<(mlp_only?"mlp_only":output_only?"output_only":draft_only?"draft_only":projection_only?"projection_only":concurrent_only?"concurrent_only":"complete_owner")<<"\",\"cells\":[";
    constexpr std::array<std::array<unsigned,2>,10> shapes{{{34816,5120},{5120,6144},{12288,5120},{4096,5120},{5120,17408},{5120,25600},{7168,5120},{6144,5120},{1280,5120},{5120,4096}}};
    bool first=true;
    for(const auto& shape:shapes) {
        if(concurrent_only && (shape==std::array<unsigned,2>{34816,5120} ||
                               shape==std::array<unsigned,2>{5120,17408}))continue;
        if(mlp_only && shape!=std::array<unsigned,2>{34816,5120} &&
           shape!=std::array<unsigned,2>{5120,17408})continue;
        if(output_only && shape!=std::array<unsigned,2>{5120,6144})continue;
        if(draft_only && shape!=std::array<unsigned,2>{5120,17408} &&
           shape!=std::array<unsigned,2>{5120,25600})continue;
        if(projection_only && shape!=std::array<unsigned,2>{7168,5120} &&
           shape!=std::array<unsigned,2>{6144,5120} &&
           shape!=std::array<unsigned,2>{1280,5120} &&
           shape!=std::array<unsigned,2>{5120,4096})continue;
        N=shape[0];K=shape[1];G=K/64;
        for(unsigned t:{1U,2U,3U,4U,5U,6U,10U,12U,15U,18U,20U,24U}) {
            if(concurrent_only && t<10)continue;
            if(draft_only && t!=5 && t!=6)continue;
            if(!linear::detail::use_a8q4_small_batch_projection(t,N,K,K))continue;
            if(!first)out<<',';first=false;cell(t,stream,out);
        }
    }
    out<<"],\"pairs\":[";
    if(!mlp_only && !output_only && !draft_only) {
        bool first_pair=true;
        for(unsigned t:{5U,6U,12U}) {
            for(const auto& pair:{std::array<unsigned,2>{4096,12288},std::array<unsigned,2>{7168,7168}}) {
                if(!first_pair)out<<',';first_pair=false;pair_cell(t,pair[0],pair[1],stream,out);
            }
        }
    }
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
