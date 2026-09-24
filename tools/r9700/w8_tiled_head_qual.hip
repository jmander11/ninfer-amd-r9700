#include "ops/r9700/linear/w8_tiled_head.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "ninfer/ops/linear.h"
#include <hip/hip_fp16.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace linear=ninfer::ops::r9700::linear;
constexpr unsigned N=248320,K=5120,G=K/32;
constexpr std::size_t Guard=256;
void check(hipError_t result) { if(result!=hipSuccess)throw std::runtime_error(hipGetErrorString(result)); }
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
std::string read(const std::filesystem::path& path) {
    std::ifstream file(path);std::string value;require(bool(file>>value),"cannot read PCI attribute");return value;
}
const std::filesystem::path Pci="/sys/bus/pci/devices/0000:13:00.0";
void power() { require(read(Pci/"power_dpm_force_performance_level")=="auto","auto power required"); }
template<class T> struct Buffer {
    T* allocation{};std::size_t count,offset;
    Buffer(std::size_t n,hipStream_t stream,std::size_t prefix=Guard):count(n),offset(prefix) {
        check(hipMalloc(&allocation,n*sizeof(T)+prefix+Guard));
        check(hipMemsetAsync(allocation,0xa5,n*sizeof(T)+prefix+Guard,stream));
    }
    ~Buffer(){(void)hipFree(allocation);}
    Buffer(const Buffer&)=delete;
    T* data()const{return reinterpret_cast<T*>(reinterpret_cast<char*>(allocation)+offset);}
    std::size_t bytes()const{return count*sizeof(T);}
    void put(const std::vector<T>& v,hipStream_t s){require(v.size()==count,"upload extent");check(hipMemcpyAsync(data(),v.data(),bytes(),hipMemcpyHostToDevice,s));check(hipStreamSynchronize(s));}
    std::vector<T> get(hipStream_t s)const{std::vector<T> v(count);check(hipMemcpyAsync(v.data(),data(),bytes(),hipMemcpyDeviceToHost,s));check(hipStreamSynchronize(s));return v;}
    void poison(hipStream_t s){check(hipMemsetAsync(data(),0xa5,bytes(),s));}
    void guards(hipStream_t s)const{
        std::vector<unsigned char> before(offset),after(Guard);
        check(hipMemcpyAsync(before.data(),allocation,offset,hipMemcpyDeviceToHost,s));
        check(hipMemcpyAsync(after.data(),reinterpret_cast<char*>(data())+bytes(),Guard,hipMemcpyDeviceToHost,s));
        check(hipStreamSynchronize(s));
        require(std::all_of(before.begin(),before.end(),[](auto x){return x==0xa5;}) &&
                std::all_of(after.begin(),after.end(),[](auto x){return x==0xa5;}),"guard changed");
    }
};
std::size_t tiled_code(unsigned row,unsigned k){return (((static_cast<std::size_t>(row/16)*G+k/32)*2+(k%32)/16)*16+row%16)*16+k%16;}
std::size_t tiled_scale(unsigned row,unsigned group){return (static_cast<std::size_t>(row/16)*G+group)*16+row%16;}
int code(unsigned row,unsigned k){return int((row*17U+(k/32)*13U+(k%32)*19U)%255U)-127;}
std::uint16_t scale_bits(unsigned row,unsigned g){return __half_as_ushort(__float2half(float(1U+(row*11U+g*7U)%31U)/4096.0F));}
double scale(unsigned row,unsigned g){return __half2float(__ushort_as_half(scale_bits(row,g)));}
struct Weights {
    Buffer<std::int8_t> row_codes,tile_codes;
    Buffer<std::uint16_t> row_scales,tile_scales;
    explicit Weights(hipStream_t stream):row_codes(std::size_t(N)*K,stream,260),tile_codes(std::size_t(N)*K,stream,260),
        row_scales(std::size_t(N)*G,stream),tile_scales(std::size_t(N)*G,stream) {
        // Host-only fixture generation and lossless permutation, never a product load path.
        std::vector<std::int8_t> row(std::size_t(N)*K),tile(row.size());
        std::array<std::array<std::int8_t,K>,255> patterns{};
        for(unsigned r=0;r<255;++r)for(unsigned k=0;k<K;++k)patterns[r][k]=code(r,k);
        for(unsigned r=0;r<N;++r)std::memcpy(row.data()+std::size_t(r)*K,patterns[r%255].data(),K);
        for(unsigned r=0;r<N;++r)for(unsigned k=0;k<K;k+=16)
            std::memcpy(tile.data()+tiled_code(r,k),row.data()+std::size_t(r)*K+k,16);
        for(unsigned r=0;r<N;++r)for(unsigned k=0;k<K;k+=16)
            require(std::memcmp(tile.data()+tiled_code(r,k),row.data()+std::size_t(r)*K+k,16)==0,"lossless code permutation failed");
        row_codes.put(row,stream);tile_codes.put(tile,stream);
        std::vector<std::uint16_t> rs(std::size_t(N)*G),ts(rs.size());
        for(unsigned r=0;r<N;++r)for(unsigned g=0;g<G;++g){rs[std::size_t(r)*G+g]=scale_bits(r,g);ts[tiled_scale(r,g)]=rs[std::size_t(r)*G+g];}
        for(unsigned r=0;r<N;++r)for(unsigned g=0;g<G;++g)
            require(ts[tiled_scale(r,g)]==rs[std::size_t(r)*G+g],"lossless scale permutation failed");
        row_scales.put(rs,stream);tile_scales.put(ts,stream);
    }
    void guards(hipStream_t s){row_codes.guards(s);tile_codes.guards(s);row_scales.guards(s);tile_scales.guards(s);}
};
std::vector<hip_bfloat16> inputs(unsigned tokens){
    std::vector<hip_bfloat16> x(tokens*K);
    for(unsigned t=0;t<tokens;++t)for(unsigned k=0;k<K;++k)
        x[t*K+k]=hip_bfloat16(float(int((t*31U+k*17U)%255U)-127)/64.0F);
    return x;
}
struct Error {double relative=0,gross=0;};
Error oracle(const std::vector<hip_bfloat16>& actual,const std::vector<hip_bfloat16>& x,unsigned tokens,bool a8){
    const std::array<unsigned,12> rows{0,1,7,15,16,31,32,N/4,N/2,N-17,N-16,N-1};
    for(auto y:actual)require(std::isfinite(float(y)),"nonfinite output");
    Error error;
    for(unsigned t=0;t<tokens;++t){
        double e2=0,r2=0,maximum=0;
        for(auto row:rows){
            double expected=0;
            for(unsigned k=0;k<K;++k)expected+=double(float(x[t*K+k]))*code(row,k)*scale(row,k/32);
            const double delta=double(float(actual[t*N+row]))-expected;
            if (!a8) require(std::abs(delta) <= 1e-5 + .0079 * std::abs(expected),
                             "exact BF16/W8 pointwise FP64 oracle failed");
            e2+=delta*delta;r2+=expected*expected;maximum=std::max(maximum,std::abs(delta));
        }
        require(r2>0,"fixture reference is zero");
        const double relative=std::sqrt(e2/r2),gross=maximum/std::sqrt(r2/rows.size());
        // Same predeclared public BF16×W8 criteria as the packed-load qualifier.
        if(relative>0.02 || gross>0.10)throw std::runtime_error(
            "original BF16 full-K FP64 oracle failed: token="+std::to_string(t)+
            " relative="+std::to_string(relative)+" gross="+std::to_string(gross));
        error.relative=std::max(error.relative,relative);error.gross=std::max(error.gross,gross);
    }
    return error;
}
void equal(const std::vector<hip_bfloat16>& a,const std::vector<hip_bfloat16>& b){require(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*2)==0,"full output parity failed");}
linear::W8G32LinearArgs exact_args(const linear::A8W8G32CandidateArgs& args) {
    return {args.input,args.weight_codes,args.weight_code_bytes,args.weight_scales,
            args.weight_scale_bytes,args.output,args.tokens,args.rows,args.columns,args.padded_columns};
}
void invoke(const linear::A8W8G32CandidateArgs& args, bool tiled, bool exact, hipStream_t stream) {
    ninfer::Tensor input(const_cast<hip_bfloat16*>(args.input), ninfer::DType::BF16,
                         {static_cast<int>(K), static_cast<int>(args.tokens)});
    ninfer::Tensor output(args.output, ninfer::DType::BF16,
                          {static_cast<int>(N), static_cast<int>(args.tokens)});
    ninfer::Weight weight{};
    weight.qtype = ninfer::QType::W8G32_F16S;
    weight.layout = tiled ? ninfer::QuantLayout::W8N16K16 : ninfer::QuantLayout::RowSplit;
    weight.ndim = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = N;
    weight.k = weight.shape[1] = weight.padded_shape[1] = K;
    weight.group = weight.group_size = 32;
    weight.scale_dtype = ninfer::DType::FP16;
    weight.qdata = const_cast<std::int8_t*>(args.weight_codes);
    weight.scales = const_cast<std::uint16_t*>(args.weight_scales);
    weight.qdata_bytes = args.weight_code_bytes;
    weight.scale_bytes = args.weight_scale_bytes;
    if (exact) ninfer::ops::linear(input, weight, output, stream);
    else ninfer::ops::linear(input, weight, output,
        ninfer::DeviceSpan{args.activation_workspace, args.activation_workspace_bytes}, stream);
}
struct Graph {
    hipGraph_t definition{};hipGraphExec_t execution{};
    Graph(const linear::A8W8G32CandidateArgs& args,bool tiled,bool exact,hipStream_t stream){
        check(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
        invoke(args,tiled,exact,stream);
        check(hipStreamEndCapture(stream,&definition));check(hipGraphInstantiate(&execution,definition,nullptr,nullptr,0));
    }
    ~Graph(){(void)hipGraphExecDestroy(execution);(void)hipGraphDestroy(definition);}
    void run(hipStream_t s){check(hipGraphLaunch(execution,s));}
};
__global__ void scrub(std::uint32_t* p,std::size_t count){for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)p[i]=static_cast<std::uint32_t>(i);}
double timing(Graph& graph,bool cold,Buffer<std::uint32_t>& flush,hipStream_t stream,unsigned warm_iterations=16){
    if(cold){hipLaunchKernelGGL(scrub,dim3(4096),dim3(256),0,stream,flush.data(),flush.count);check(hipGetLastError());}
    hipEvent_t begin{},end{};check(hipEventCreate(&begin));check(hipEventCreate(&end));
    check(hipEventRecord(begin,stream));
    const unsigned iterations=cold?1:warm_iterations;
    for(unsigned i=0;i<iterations;++i)graph.run(stream);
    check(hipEventRecord(end,stream));check(hipEventSynchronize(end));
    float ms=0;check(hipEventElapsedTime(&ms,begin,end));check(hipEventDestroy(begin));check(hipEventDestroy(end));
    require(std::isfinite(ms) && ms>0,"invalid timing");return ms/iterations;
}
void cell(unsigned tokens,bool exact,Weights& weights,Buffer<std::uint32_t>& flush,hipStream_t stream,std::ostream& report,bool ordinary_only=false){
    const bool a8 = !exact && linear::use_a8w8(tokens,N,K);
    const auto x=inputs(tokens);Buffer<hip_bfloat16> input(x.size(),stream);input.put(x,stream);
    Buffer<hip_bfloat16> control(tokens*N,stream),candidate(tokens*N,stream);
    const auto workspace_bytes=linear::a8w8g32_activation_workspace_capacity_bytes(tokens,K);
    Buffer<std::uint8_t> wc(workspace_bytes,stream),wt(workspace_bytes,stream);
    auto args=[&](bool tiled){return linear::A8W8G32CandidateArgs{input.data(),tiled?weights.tile_codes.data():weights.row_codes.data(),weights.row_codes.bytes(),
        tiled?weights.tile_scales.data():weights.row_scales.data(),weights.row_scales.bytes(),tiled?wt.data():wc.data(),workspace_bytes,
        tiled?candidate.data():control.data(),tokens,N,K,K};};
    auto invalid=args(true);invalid.rows=N-1;require(linear::a8w8g32_tiled_head(invalid,stream)==hipErrorInvalidValue,"invalid shape accepted");
    invalid=args(true);++invalid.weight_codes;require(linear::a8w8g32_tiled_head(invalid,stream)==hipErrorInvalidValue,"invalid alignment accepted");
    invalid=args(true);invalid.output=input.data();require(linear::a8w8g32_tiled_head(invalid,stream)==hipErrorInvalidValue,"overlap accepted");
    if (!a8) {
        auto rejected=exact_args(args(true));rejected.rows=N-1;
        require(linear::w8g32_tiled_head(rejected,stream)==hipErrorInvalidValue,"exact invalid shape accepted");
        rejected=exact_args(args(true));++rejected.codes;
        require(linear::w8g32_tiled_head(rejected,stream)==hipErrorInvalidValue,"exact invalid alignment accepted");
        rejected=exact_args(args(true));rejected.output=input.data();
        require(linear::w8g32_tiled_head(rejected,stream)==hipErrorInvalidValue,"exact overlap accepted");
    }
    invoke(args(false),false,exact,stream);invoke(args(true),true,exact,stream);
    const auto baseline=control.get(stream),tiled=candidate.get(stream);equal(baseline,tiled);
    const auto ce=oracle(baseline,x,tokens,a8),te=oracle(tiled,x,tokens,a8);
    Graph cg(args(false),false,exact,stream),tg(args(true),true,exact,stream);
    for(unsigned replay=0;replay<2;++replay){control.poison(stream);candidate.poison(stream);cg.run(stream);tg.run(stream);equal(control.get(stream),baseline);equal(candidate.get(stream),baseline);}
    if (a8) {
        auto bad_input=x;bad_input[0].data=0x7fc1;input.put(bad_input,stream);cg.run(stream);tg.run(stream);
        for(const auto& y:{control.get(stream),candidate.get(stream)})require(std::all_of(y.begin(),y.end(),[](auto v){return v.data==0x7fc1;}),"nonfinite status propagation failed");
    }
    input.put(x,stream);control.poison(stream);candidate.poison(stream);cg.run(stream);tg.run(stream);
    equal(control.get(stream),baseline);equal(candidate.get(stream),baseline);equal(input.get(stream),x);
    input.guards(stream);control.guards(stream);candidate.guards(stream);wc.guards(stream);wt.guards(stream);weights.guards(stream);
    const bool timed = !exact;
    const unsigned warmup_calls = tokens <= 6 ? 32U : 2U;
    const unsigned pairs = ordinary_only ? 8U : (tokens <= 6 ? 7U : 3U);
    const unsigned warm_iterations = tokens <= 6 ? 16U : 2U;
    power();
    if (timed) for(unsigned i=0;i<warmup_calls;++i){cg.run(stream);tg.run(stream);}
    check(hipStreamSynchronize(stream));
    report<<"{\"tokens\":"<<tokens<<",\"rows\":"<<N<<",\"columns\":"<<K
          <<",\"ordinary_only\":"<<(ordinary_only?"true":"false")
          <<",\"force_exact_bf16\":"<<(exact?"true":"false")<<",\"a8_profile\":"<<(a8?"true":"false")
          <<",\"warmup_calls_per_route\":"<<(timed?warmup_calls:0U)
          <<",\"exact_all_outputs\":true,\"poisoned_graph_replays\":2,\"status_recovery\":"<<(a8?"true":"null")<<",\"guards\":true,"
          <<"\"control_public_relative_rms\":"<<ce.relative<<",\"candidate_public_relative_rms\":"<<te.relative
          <<",\"control_public_gross_rms\":"<<ce.gross<<",\"candidate_public_gross_rms\":"<<te.gross<<",\"samples\":[";
    bool first=true;
    if (timed) for(bool cold:{false,true})for(unsigned pair=0;pair<pairs;++pair){
        double c,t;
        if(pair%2==0){c=timing(cg,cold,flush,stream,warm_iterations);t=timing(tg,cold,flush,stream,warm_iterations);}
        else{t=timing(tg,cold,flush,stream,warm_iterations);c=timing(cg,cold,flush,stream,warm_iterations);}
        if(!first)report<<',';first=false;
        report<<"{\"cache_scrubbed\":"<<(cold?"true":"false")<<",\"control_first\":"<<(pair%2==0?"true":"false")
              <<",\"calls\":"<<(cold?1:warm_iterations)<<",\"control_ms\":"<<c<<",\"candidate_ms\":"<<t<<'}';
    }
    power();report<<"]}";
}
}
int main(int argc,char** argv){
    try{
        require(argc==2 || (argc==3 && std::string(argv[2])=="--ordinary-only"),
                "usage: w8_tiled_head_qual FRESH_DIRECTORY [--ordinary-only]");
        const bool ordinary_only=argc==3;
        const std::filesystem::path directory=argv[1];
        require(std::filesystem::create_directory(directory),"output directory already exists");
        require(read(Pci/"vendor")=="0x1002" && read(Pci/"device")=="0x7551","R9700 PCI identity required");power();
        check(hipSetDevice(0));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,0));char pci[32]{};check(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
        require(std::string(pci)=="0000:13:00.0" && std::string(props.gcnArchName)=="gfx1201" && props.warpSize==32,"HIP device mismatch");
        hipStream_t stream{};check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        {
            Weights weights(stream);Buffer<std::uint32_t> flush(80ULL*1024*1024/4,stream);
            std::ofstream report(directory/"report.json");require(bool(report),"cannot create report");report<<std::setprecision(17);
            report<<"{\"schema\":\"ninfer.r9700.w8-tiled-head-consumer.v1\",\"public_linear_tested\":true"
                  <<",\"ordinary_only\":"<<(ordinary_only?"true":"false")<<','
                  <<"\"lossless_host_permutation\":true,\"public_oracle\":\"original BF16 times exact signed W8/FP16, full-K FP64 sampled rows\","
                  <<"\"relative_rms_limit\":0.02,\"gross_rms_limit\":0.10,"
                  <<"\"exact_bf16_pointwise_criterion\":\"abs_error <= 1e-5 + 0.0079 * abs(reference)\","
                  <<"\"pci\":\"0000:13:00.0\",\"target\":\"gfx1201\",\"power\":\"auto\","
                  <<"\"scrub_bytes\":83886080,\"timing\":\"complete public Linear graph, same-stream events, milliseconds per call; ordinary-only eight balanced pairs, default T1..6 seven alternating pairs, T17/129 three short pairs; forced exact cells correctness only\",\"cells\":[";
            bool first = true;
            const std::vector<unsigned> token_cases=ordinary_only?std::vector<unsigned>{1U,2U,3U}:
                std::vector<unsigned>{1U,3U,4U,5U,6U,17U,129U};
            for (unsigned tokens : token_cases) {
                if (!first) report << ','; first = false;
                cell(tokens,false,weights,flush,stream,report,ordinary_only);
            }
            if (!ordinary_only) for (unsigned tokens : {4U,6U,17U}) {
                report << ','; cell(tokens,true,weights,flush,stream,report);
            }
            report<<"],\"passed\":true}\n";
            require(bool(report),"report write failed");
        }
        check(hipStreamDestroy(stream));std::cout<<"W8 tiled-head qualification PASS; performance admission is separate\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
