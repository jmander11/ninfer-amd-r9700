#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"
#include "ninfer/ops/normalized_linear.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"

#include <functional>
#include <numeric>

namespace {
constexpr int kNormK = 5120;
constexpr int kNormN = 34816;
constexpr int kNormCopies = 3;
constexpr int kNormTrials = 24;
constexpr float kNormEps = 1.0e-6F;
constexpr std::size_t kNormGuard = 256;
const std::filesystem::path kNormPower =
    "/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level";

void auto_power() {
    if (read_text(kNormPower) != "auto") fail("normalized Linear requires auto power");
}
void transfer(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind,
              hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

struct NormGraph {
    hipGraph_t graph{};
    hipGraphExec_t executable{};
    NormGraph(hipStream_t stream, const std::function<void()>& launch) {
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        launch();
        HIP_CHECK(hipStreamEndCapture(stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    ~NormGraph() { (void)hipGraphExecDestroy(executable); (void)hipGraphDestroy(graph); }
    void run(hipStream_t stream) const { HIP_CHECK(hipGraphLaunch(executable, stream)); }
};

// Direct double -> BF16 RNE, without a private FP32 staging cast.
hip_bfloat16 oracle_bf16(double value) {
    hip_bfloat16 result{};
    const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
    const double magnitude = std::abs(value);
    if (!std::isfinite(value)) { result.data = sign | (std::isnan(value) ? 0x7fc1U : 0x7f80U); return result; }
    if (magnitude == 0) { result.data = sign; return result; }
    int exponent = 0;
    (void)std::frexp(magnitude, &exponent);
    --exponent;
    if (exponent < -126) {
        const auto fraction = static_cast<unsigned>(std::nearbyint(std::ldexp(magnitude, 133)));
        result.data = sign | static_cast<std::uint16_t>(fraction);
        return result;
    }
    unsigned significand = static_cast<unsigned>(std::nearbyint(std::ldexp(magnitude, 7-exponent)));
    if (significand == 256U) { significand = 128U; ++exponent; }
    result.data = exponent > 127 ? sign | 0x7f80U :
        sign | static_cast<std::uint16_t>((exponent+127)*128 + significand-128U);
    return result;
}

std::vector<hip_bfloat16> normalized_oracle(const std::vector<hip_bfloat16>& x,
                                           const std::vector<hip_bfloat16>& norm,
                                           bool unit_offset) {
    double sum = 0;
    for (auto value : x) { const double v = static_cast<float>(value); sum += v*v; }
    const double inv = 1.0/std::sqrt(sum/kNormK + static_cast<double>(kNormEps));
    std::vector<hip_bfloat16> result(kNormK);
    for (int k=0; k<kNormK; ++k)
        result[k] = oracle_bf16(static_cast<double>(static_cast<float>(x[k])) * inv *
            (static_cast<double>(static_cast<float>(norm[k])) + (unit_offset ? 1.0 : 0.0)));
    return result;
}

std::vector<double> linear_oracle(const HostActivation& activation,
                                  const DecodeDot8Weights& weight) {
    constexpr std::size_t groups = kNormK/64;
    std::vector<double> result(kNormN);
    for (int n=0; n<kNormN; ++n) {
        double sum=0;
        for (std::size_t group=0; group<groups; ++group) {
            int dot=0;
            for (std::size_t lane=0; lane<64; ++lane) {
                // Independent Q4N16K16 address calculation from the physical byte contract.
                const std::size_t word = (n/16)*groups*64 + group*64 + (lane/16)*16 + n%16;
                const unsigned nibble = (weight.codes[word*8+(lane%16)/2] >> ((lane%2)*4)) & 15U;
                const int w = nibble>=8 ? static_cast<int>(nibble)-16 : static_cast<int>(nibble);
                const auto a_index=group*64+lane;
                dot += (unsigned_nibble(activation.low,a_index)+16*signed_nibble(activation.high,a_index))*w;
            }
            const auto scale_index=(n/16)*groups*16+group*16+n%16;
            sum += static_cast<double>(dot)*half_value(activation.scales[group])*
                static_cast<double>(half_value(weight.scales[scale_index]));
        }
        result[n]=sum;
    }
    return result;
}

unsigned ordered_bf16(hip_bfloat16 value) {
    return (value.data & 0x8000U) ? 0x8000U-(value.data & 0x7fffU) : 0x8000U+value.data;
}
struct NormwiseError {
    unsigned maximum_bf16_steps = 0;
    double relative_l2 = 0;
    double maximum_absolute = 0;
};

// Named normalized-linear A8Q4 criterion: relative L2 <= 1e-2 with a gross
// pointwise cap of two BF16 steps from the independently rounded FP64 result.
NormwiseError compare_oracle(const std::vector<hip_bfloat16>& actual,
                             const std::vector<double>& expected) {
    NormwiseError error;
    double squared_error=0,squared_reference=0;
    for (std::size_t i=0;i<actual.size();++i) {
        if (!std::isfinite(static_cast<float>(actual[i]))) fail("nonfinite finite-case output");
        const auto rounded=oracle_bf16(expected[i]);
        const unsigned a=ordered_bf16(actual[i]), b=ordered_bf16(rounded);
        error.maximum_bf16_steps=std::max(error.maximum_bf16_steps,a>b?a-b:b-a);
        const double delta=static_cast<double>(static_cast<float>(actual[i]))-expected[i];
        error.maximum_absolute=std::max(error.maximum_absolute,std::abs(delta));
        squared_error+=delta*delta;squared_reference+=expected[i]*expected[i];
    }
    error.relative_l2=std::sqrt(squared_error/std::max(1.0e-30,squared_reference));
    if(error.relative_l2>1.0e-2 || error.maximum_bf16_steps>2)
        fail("normalized Linear exceeds named FP64 normwise criterion");
    return error;
}

struct NormFixture {
    hipStream_t stream;
    DeviceBuffer<hip_bfloat16> input{kNormK}, norm{kNormK}, hidden{kNormK};
    DeviceBuffer<hip_bfloat16> candidate{kNormN+2}, control{kNormN+2};
    std::size_t candidate_bytes = ninfer::ops::normalized_linear_workspace_capacity_bytes(1,kNormK,kNormN);
    std::size_t control_bytes = ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::Q4G64_F16S,1,kNormK);
    DeviceBuffer<std::uint8_t> candidate_storage{candidate_bytes+2*kNormGuard};
    DeviceBuffer<std::uint8_t> control_storage{control_bytes+2*kNormGuard};
    std::array<std::unique_ptr<DeviceBuffer<std::uint8_t>>,kNormCopies> codes;
    std::array<std::unique_ptr<DeviceBuffer<std::uint16_t>>,kNormCopies> scales;
    std::array<ninfer::Weight,kNormCopies> weights{};
    bool unit_offset=true;
    explicit NormFixture(hipStream_t execution, const DecodeDot8Weights& host):stream(execution) {
        for (int i=0;i<kNormCopies;++i) {
            codes[i]=std::make_unique<DeviceBuffer<std::uint8_t>>(host.codes.size());
            scales[i]=std::make_unique<DeviceBuffer<std::uint16_t>>(host.scales.size());
            transfer(codes[i]->get(),host.codes.data(),codes[i]->bytes(),hipMemcpyHostToDevice,stream);
            transfer(scales[i]->get(),host.scales.data(),scales[i]->bytes(),hipMemcpyHostToDevice,stream);
            auto& w=weights[i];
            w.qtype=ninfer::QType::Q4G64_F16S; w.ndim=2;
            w.n=w.shape[0]=w.padded_shape[0]=kNormN;
            w.k=w.shape[1]=w.padded_shape[1]=kNormK;
            w.group=w.group_size=64; w.layout=ninfer::QuantLayout::Q4N16K16;
            w.scale_dtype=ninfer::DType::FP16;
            w.qdata=codes[i]->get(); w.qdata_bytes=codes[i]->bytes();
            w.scales=scales[i]->get(); w.scale_bytes=scales[i]->bytes();
        }
        HIP_CHECK(hipMemsetAsync(candidate_storage.get(),0xa5,candidate_storage.bytes(),stream));
        HIP_CHECK(hipMemsetAsync(control_storage.get(),0xa5,control_storage.bytes(),stream));
        HIP_CHECK(hipMemsetAsync(candidate.get(),0x55,candidate.bytes(),stream));
        HIP_CHECK(hipMemsetAsync(control.get(),0x55,control.bytes(),stream));
    }
    ninfer::Tensor x() { return {input.get(),ninfer::DType::BF16,{kNormK,1}}; }
    ninfer::Tensor gain() { return {norm.get(),ninfer::DType::BF16,{kNormK}}; }
    ninfer::Tensor output(bool fused) { return {(fused?candidate.get():control.get())+1,ninfer::DType::BF16,{kNormN,1}}; }
    ninfer::DeviceSpan workspace(bool fused) {
        return {(fused?candidate_storage.get():control_storage.get())+kNormGuard,
                fused?candidate_bytes:control_bytes};
    }
    void launch(bool fused,int copy=0) {
        auto a=x(),g=gain(),out=output(fused);
        if (fused) ninfer::ops::normalized_linear(a,g,kNormEps,unit_offset,weights[copy],out,workspace(true),stream);
        else {
            ninfer::Tensor normalized(hidden.get(),ninfer::DType::BF16,{kNormK,1});
            ninfer::ops::rmsnorm(a,g,kNormEps,unit_offset,normalized,stream);
            ninfer::ops::linear(normalized,weights[copy],out,workspace(false),stream);
        }
    }
    std::vector<hip_bfloat16> read(bool fused) {
        std::vector<hip_bfloat16> values(kNormN+2);
        transfer(values.data(),fused?candidate.get():control.get(),values.size()*2,hipMemcpyDeviceToHost,stream);
        if (values.front().data!=0x5555U || values.back().data!=0x5555U) fail("output canary changed");
        return {values.begin()+1,values.end()-1};
    }
    linear::A8G64ActivationWorkspace activation() {
        linear::A8G64ActivationWorkspace result{};
        HIP_CHECK(linear::a8q4g64_bind_activation_workspace(workspace(true).data,candidate_bytes,1,kNormK,&result));
        return result;
    }
    void codec(const HostActivation& expected) {
        const auto a=activation();
        std::vector<std::uint8_t> low(expected.low.size()),high(expected.high.size());
        std::vector<std::uint16_t> scale(expected.scales.size());
        std::uint32_t status=~0U;
        transfer(low.data(),a.low_codes,low.size(),hipMemcpyDeviceToHost,stream);
        transfer(high.data(),a.high_codes,high.size(),hipMemcpyDeviceToHost,stream);
        transfer(scale.data(),a.scales,scale.size()*2,hipMemcpyDeviceToHost,stream);
        transfer(&status,a.status,sizeof(status),hipMemcpyDeviceToHost,stream);
        if (low!=expected.low || high!=expected.high || scale!=expected.scales || status!=0)
            fail("fused normalization's A8 codec differs from independent BF16-seam oracle");
    }
    void guards() {
        for (bool fused:{false,true}) {
            const auto& storage=fused?candidate_storage:control_storage;
            std::array<std::uint8_t,kNormGuard> bytes{};
            for (const auto offset:{std::size_t{0},storage.bytes()-kNormGuard}) {
                transfer(bytes.data(),storage.get()+offset,bytes.size(),hipMemcpyDeviceToHost,stream);
                if (!std::all_of(bytes.begin(),bytes.end(),[](auto v){return v==0xa5;})) fail("workspace canary changed");
            }
        }
    }
};

void malformed(NormFixture& f) {
    const auto expect=[&](const std::function<void()>& call) {
        bool rejected=false;
        try { call(); } catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) fail("normalized Linear accepted malformed public call");
    };
    auto valid_x=f.x(),valid_norm=f.gain(),valid_output=f.output(true);
    auto valid_weight=f.weights[0]; auto valid_workspace=f.workspace(true);
    expect([]{(void)ninfer::ops::normalized_linear_workspace_capacity_bytes(0,kNormK,kNormN);});
    expect([]{(void)ninfer::ops::normalized_linear_workspace_capacity_bytes(1,kNormK-1,kNormN);});
    expect([]{(void)ninfer::ops::normalized_linear_workspace_capacity_bytes(1,kNormK,kNormN-1);});
    const auto call=[&](ninfer::Tensor x,ninfer::Tensor g,ninfer::Weight w,ninfer::Tensor out,ninfer::DeviceSpan ws,float eps=kNormEps) {
        ninfer::ops::normalized_linear(x,g,eps,true,w,out,ws,f.stream);
    };
    auto x=valid_x; auto ws=valid_workspace; x.dtype=ninfer::DType::FP16;
    expect([&]{call(x,valid_norm,valid_weight,valid_output,valid_workspace);});
    auto g=valid_norm; g.ne[0]-=1;
    expect([&]{call(valid_x,g,valid_weight,valid_output,valid_workspace);});
    auto out=valid_output; out.ne[0]-=1;
    expect([&]{call(valid_x,valid_norm,valid_weight,out,valid_workspace);});
    for (float eps:{0.0F,-1.0F,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        expect([&]{call(valid_x,valid_norm,valid_weight,valid_output,valid_workspace,eps);});
    auto w=valid_weight; --w.qdata_bytes;
    expect([&]{call(valid_x,valid_norm,w,valid_output,valid_workspace);});
    w=valid_weight; --w.scale_bytes;
    expect([&]{call(valid_x,valid_norm,w,valid_output,valid_workspace);});
    // Every public plane must reject the one-byte misalignment forbidden by its contract.
    for(int plane=0;plane<6;++plane) {
        x=valid_x;g=valid_norm;out=valid_output;w=valid_weight;ws=valid_workspace;
        if(plane==0)x.data=static_cast<std::uint8_t*>(x.data)+1;
        if(plane==1)g.data=static_cast<std::uint8_t*>(g.data)+1;
        if(plane==2)out.data=static_cast<std::uint8_t*>(out.data)+1;
        if(plane==3)w.qdata=static_cast<const std::uint8_t*>(w.qdata)+1;
        if(plane==4)w.scales=static_cast<const std::uint8_t*>(w.scales)+1;
        if(plane==5)ws.data=static_cast<std::uint8_t*>(ws.data)+1;
        expect([&]{call(x,g,w,out,ws);});
    }
    ws=valid_workspace; --ws.bytes;
    expect([&]{call(valid_x,valid_norm,valid_weight,valid_output,ws);});
    // Every pair of the six public planes must reject overlap.
    for (int a=0;a<6;++a) for (int b=a+1;b<6;++b) {
        x=valid_x;g=valid_norm;out=valid_output;w=valid_weight;ws=valid_workspace;
        std::array<const void*,6> pointers{x.data,g.data,out.data,w.qdata,w.scales,ws.data};
        pointers[b]=pointers[a];
        x.data=const_cast<void*>(pointers[0]);g.data=const_cast<void*>(pointers[1]);out.data=const_cast<void*>(pointers[2]);
        w.qdata=pointers[3];w.scales=pointers[4];ws.data=const_cast<void*>(pointers[5]);
        expect([&]{call(x,g,w,out,ws);});
    }
}

__global__ void normalized_linear_scrub(std::uint32_t* data,std::size_t size) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index<size) data[index]=data[index]*1664525U+1013904223U;
}
double graph_ms(const NormGraph& graph,hipStream_t stream) {
    Event begin,end;
    HIP_CHECK(hipEventRecord(begin.get(),stream));graph.run(stream);
    HIP_CHECK(hipEventRecord(end.get(),stream));HIP_CHECK(hipEventSynchronize(end.get()));
    float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,begin.get(),end.get()));return ms;
}
double sorted_median(std::vector<double> values) {
    std::sort(values.begin(),values.end());return (values[(values.size()-1)/2]+values[values.size()/2])*0.5;
}
} // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=3 || std::string_view(argv[1])!="--out-json") {
            std::cerr<<"usage: a8q4_normalized_linear_t1_qual --out-json NEW.json\n";return 2;
        }
        require_fresh_output(argv[2]);
        if (linear::kQ4ActivationBits!=8) fail("qualifier must link the A8 profile");
        if (read_text(kNormPower.parent_path()/"vendor")!="0x1002" ||
            read_text(kNormPower.parent_path()/"device")!="0x7551") fail("wrong physical R9700 PCI function");
        auto_power();HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t properties{};HIP_CHECK(hipGetDeviceProperties(&properties,0));
        if (std::string_view(properties.gcnArchName)!="gfx1201" || properties.warpSize!=32) fail("requires gfx1201 wave32");
        char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
        std::string pci_text=pci;if(pci_text.size()==7)pci_text="0000:"+pci_text;
        if(std::filesystem::canonical(std::filesystem::path("/sys/bus/pci/devices")/pci_text)!=std::filesystem::canonical(kNormPower.parent_path()))
            fail("HIP device differs from pre-initialization power device");
        hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        const auto host_weights=make_decode_dot8_weights(kNormN,kNormK);
        NormFixture f(stream,host_weights);malformed(f);
        const auto normal_input=make_decode_dot8_input(kNormK);
        std::vector<hip_bfloat16> host_norm(kNormK);
        for(int k=0;k<kNormK;++k)host_norm[k]=hip_bfloat16(static_cast<float>((k*7)%97-48)/128.0F);
        NormwiseError maximum_error;
        std::vector<hip_bfloat16> finite_result;
        std::vector<double> finite_oracle;
        HostActivation finite_codec;
        for (int test=0;test<4;++test) {
            auto input=normal_input;f.unit_offset=test!=1;
            if(test==2)std::fill(input.begin(),input.end(),hip_bfloat16(0.0F));
            if(test==3)for(auto& v:input)v=hip_bfloat16(static_cast<float>(v)*0x1p-30F);
            transfer(f.input.get(),input.data(),f.input.bytes(),hipMemcpyHostToDevice,stream);
            transfer(f.norm.get(),host_norm.data(),f.norm.bytes(),hipMemcpyHostToDevice,stream);
            const auto normalized=normalized_oracle(input,host_norm,f.unit_offset);
            const auto codec=quantize_host(normalized,1,kNormK);
            const auto expected=linear_oracle(codec,host_weights);
            f.launch(false);f.launch(true);HIP_CHECK(hipStreamSynchronize(stream));
            auto candidate=f.read(true),control=f.read(false);
            for(const auto& error:{compare_oracle(candidate,expected),compare_oracle(control,expected)}) {
                maximum_error.maximum_bf16_steps=std::max(maximum_error.maximum_bf16_steps,error.maximum_bf16_steps);
                maximum_error.relative_l2=std::max(maximum_error.relative_l2,error.relative_l2);
                maximum_error.maximum_absolute=std::max(maximum_error.maximum_absolute,error.maximum_absolute);
            }
            f.codec(codec);f.guards();
            std::vector<hip_bfloat16> input_after(kNormK),norm_after(kNormK);
            transfer(input_after.data(),f.input.get(),f.input.bytes(),hipMemcpyDeviceToHost,stream);
            transfer(norm_after.data(),f.norm.get(),f.norm.bytes(),hipMemcpyDeviceToHost,stream);
            const auto equal_bits=[](auto a,auto b){return a.data==b.data;};
            if(!std::equal(input_after.begin(),input_after.end(),input.begin(),equal_bits) ||
               !std::equal(norm_after.begin(),norm_after.end(),host_norm.begin(),equal_bits))
                fail("normalized Linear mutated input or norm weights");
            if(!std::equal(candidate.begin(),candidate.end(),control.begin(),[](auto a,auto b){return a.data==b.data;}))
                fail("normalized Linear differs from public RMSNorm plus Linear");
            if(test==0){finite_result=candidate;finite_oracle=expected;finite_codec=codec;}
        }
        f.unit_offset=true;
        transfer(f.input.get(),normal_input.data(),f.input.bytes(),hipMemcpyHostToDevice,stream);
        // One supported positive-T fallback regression with distinct represented
        // inputs and an independently evaluated normalization/codec/projection per token.
        {
            std::vector<hip_bfloat16> second_input=normal_input;
            for(int k=0;k<kNormK;++k)
                second_input[k]=hip_bfloat16(-static_cast<float>(normal_input[k])+
                    static_cast<float>((k*11)%13-6)/256.0F);
            std::vector<hip_bfloat16> input2=normal_input;
            input2.insert(input2.end(),second_input.begin(),second_input.end());
            DeviceBuffer<hip_bfloat16> d_input2(input2.size()),d_output2(2*kNormN);
            const auto bytes=ninfer::ops::normalized_linear_workspace_capacity_bytes(2,kNormK,kNormN);
            DeviceBuffer<std::uint8_t> storage2(bytes);
            transfer(d_input2.get(),input2.data(),d_input2.bytes(),hipMemcpyHostToDevice,stream);
            ninfer::Tensor x2(d_input2.get(),ninfer::DType::BF16,{kNormK,2});
            ninfer::Tensor out2(d_output2.get(),ninfer::DType::BF16,{kNormN,2});
            auto norm=f.gain();
            ninfer::ops::normalized_linear(x2,norm,kNormEps,true,f.weights[0],out2,{storage2.get(),bytes},stream);
            std::vector<hip_bfloat16> actual(2*kNormN);
            transfer(actual.data(),d_output2.get(),d_output2.bytes(),hipMemcpyDeviceToHost,stream);
            const auto second_normalized=normalized_oracle(second_input,host_norm,true);
            const auto second_oracle=linear_oracle(quantize_host(second_normalized,1,kNormK),host_weights);
            for(int token=0;token<2;++token) {
                const auto error=compare_oracle(
                    {actual.begin()+token*kNormN,actual.begin()+(token+1)*kNormN},
                    token==0?finite_oracle:second_oracle);
                maximum_error.maximum_bf16_steps=std::max(maximum_error.maximum_bf16_steps,error.maximum_bf16_steps);
                maximum_error.relative_l2=std::max(maximum_error.relative_l2,error.relative_l2);
                maximum_error.maximum_absolute=std::max(maximum_error.maximum_absolute,error.maximum_absolute);
            }
        }
        std::array<std::unique_ptr<NormGraph>,kNormCopies> controls,candidates;
        for(int copy=0;copy<kNormCopies;++copy) {
            controls[copy]=std::make_unique<NormGraph>(stream,[&]{f.launch(false,copy);});
            candidates[copy]=std::make_unique<NormGraph>(stream,[&]{f.launch(true,copy);});
        }
        // Captured poisoned/stale/finite replay must reset per-call status.
        for(int replay=0;replay<3;++replay) {
            hip_bfloat16 bad{};bad.data=replay==1?0x7f80U:0x7fc1U;
            auto* destination=replay==2?f.norm.get():f.input.get();
            transfer(destination,&bad,sizeof(bad),hipMemcpyHostToDevice,stream);
            candidates[0]->run(stream);HIP_CHECK(hipStreamSynchronize(stream));
            const auto poisoned=f.read(true);
            if(!std::all_of(poisoned.begin(),poisoned.end(),[](auto v){return std::isnan(static_cast<float>(v));}))
                fail("nonfinite normalization failed to poison every output");
            transfer(f.input.get(),normal_input.data(),f.input.bytes(),hipMemcpyHostToDevice,stream);
            transfer(f.norm.get(),host_norm.data(),f.norm.bytes(),hipMemcpyHostToDevice,stream);
            const std::uint32_t stale=~0U;
            transfer(f.activation().status,&stale,sizeof(stale),hipMemcpyHostToDevice,stream);
            candidates[0]->run(stream);HIP_CHECK(hipStreamSynchronize(stream));
            const auto restored=f.read(true);
            if(!std::equal(restored.begin(),restored.end(),finite_result.begin(),[](auto a,auto b){return a.data==b.data;}))
                fail("finite graph replay retained stale codec failure");
            f.codec(finite_codec);f.guards();
        }
        auto_power();
        DeviceBuffer<std::uint32_t> scrub(80U*1024U*1024U/4U);
        HIP_CHECK(hipMemsetAsync(scrub.get(),0x5a,scrub.bytes(),stream));
        for(int warm=0;warm<6;++warm){controls[warm%3]->run(stream);candidates[warm%3]->run(stream);}
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<double> control_ms,candidate_ms,ratios;
        std::array<std::vector<double>,3> control_by_copy,candidate_by_copy;
        for(int trial=0;trial<kNormTrials;++trial) {
            const int copy=trial%3;double c=0,q=0;
            const auto timed=[&](const NormGraph& graph) {
                hipLaunchKernelGGL(normalized_linear_scrub,dim3((scrub.bytes()/4+255)/256),dim3(256),0,stream,scrub.get(),scrub.bytes()/4);
                HIP_CHECK(hipGetLastError());HIP_CHECK(hipStreamSynchronize(stream));
                return graph_ms(graph,stream);
            };
            if((trial/3)%2==0){c=timed(*controls[copy]);q=timed(*candidates[copy]);}
            else{q=timed(*candidates[copy]);c=timed(*controls[copy]);}
            if(!(c>0&&q>0&&std::isfinite(c)&&std::isfinite(q)))fail("invalid event sample");
            control_ms.push_back(c);candidate_ms.push_back(q);ratios.push_back(q/c);
            control_by_copy[copy].push_back(c);candidate_by_copy[copy].push_back(q);
        }
        auto_power();f.guards();
        const double mean=std::accumulate(ratios.begin(),ratios.end(),0.0)/ratios.size();
        double variance=0;for(double ratio:ratios)variance+=(ratio-mean)*(ratio-mean);
        const double upper=mean+2*std::sqrt(variance/(ratios.size()-1)/ratios.size());
        const double saving=64*(sorted_median(control_ms)-sorted_median(candidate_ms));
        bool every_copy=true;for(int copy=0;copy<3;++copy)every_copy&=sorted_median(candidate_by_copy[copy])<sorted_median(control_by_copy[copy]);
        const bool admitted=every_copy&&upper<1.0&&saving>=0.2;
        std::ofstream report(argv[2]);report<<std::setprecision(17);
        report<<"{\"schema\":\"ninfer.r9700.normalized-linear-t1-qualification.v1\",\"status\":\""
              <<(admitted?"qualified_for_whole_ab_only":"direct_timing_rejected")
              <<"\",\"shape\":{\"tokens\":1,\"rows\":34816,\"columns\":5120},"
              <<"\"correctness\":{\"criterion\":\"normalized_linear_a8q4_fp64_rel_l2_1e-2_bf16_steps_2_v1\","
              <<"\"maximum_bf16_steps\":"<<maximum_error.maximum_bf16_steps
              <<",\"maximum_relative_l2\":"<<maximum_error.relative_l2
              <<",\"maximum_absolute\":"<<maximum_error.maximum_absolute
              <<",\"exact_codec\":true,\"exact_public_control\":true,\"poison_stale_finite_replays\":3,"
              <<"\"canaries\":true,\"malformed_alias_rejections\":true,\"t2_fallback\":true,\"immutable_inputs\":true},"
              <<"\"power_profile\":{\"pci\":\"0000:13:00.0\",\"before_hip\":\"auto\",\"after_correctness\":\"auto\",\"after_timing\":\"auto\"},"
              <<"\"timing\":{\"complete_boundary_graph\":true,\"copies\":3,\"scrub_bytes\":83886080,"
              <<"\"control_median_ms\":"<<sorted_median(control_ms)<<",\"candidate_median_ms\":"<<sorted_median(candidate_ms)
              <<",\"saved_ms_per_token\":"<<saving<<",\"ratio_mean_plus_2se\":"<<upper<<",\"every_copy_wins\":"<<(every_copy?"true":"false")
              <<",\"samples\":[";
        for(int i=0;i<kNormTrials;++i)report<<(i?",":"")<<"{\"copy\":"<<i%3
            <<",\"control_first\":"<<((i/3)%2==0?"true":"false")<<",\"control_ms\":"<<control_ms[i]<<",\"candidate_ms\":"<<candidate_ms[i]<<"}";
        report<<"]}}\n";report.close();if(!report)fail("report write failed");
        std::cout<<"normalized_linear_t1: "<<(admitted?"PASS":"REJECT")<<" saved_ms_per_token="<<saving<<" ratio_upper="<<upper<<'\n';
        // Graphs/fixtures remain alive until after their last synchronized use.
        return admitted?0:1;
    } catch(const std::exception& error) {std::cerr<<"normalized_linear_t1: FAIL "<<error.what()<<'\n';return 1;}
}
