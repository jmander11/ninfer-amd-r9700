#include "ops/r9700/linear/r9700_linear.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace linear=ninfer::ops::r9700::linear;
constexpr std::size_t Guard=256;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void check(hipError_t status){if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));}
std::string read(const std::filesystem::path& file){std::ifstream in(file);std::string value;require(bool(in>>value),"PCI attribute read failed");return value;}
const std::filesystem::path Pci="/sys/bus/pci/devices/0000:13:00.0";
void power(){require(read(Pci/"power_dpm_force_performance_level")=="auto","automatic power required");}
struct Buffer {
    unsigned char* allocation{};std::size_t count,prefix;
    Buffer(std::size_t n,unsigned offset,hipStream_t s):count(n),prefix(Guard+offset){
        check(hipMalloc(&allocation,prefix+count*2+Guard));check(hipMemsetAsync(allocation,0xa5,prefix+count*2+Guard,s));
        require(reinterpret_cast<std::uintptr_t>(data())%16==offset,"fixture alignment mismatch");
    }
    ~Buffer(){(void)hipFree(allocation);}Buffer(const Buffer&)=delete;
    hip_bfloat16* data()const{return reinterpret_cast<hip_bfloat16*>(allocation+prefix);}
    void put(const std::vector<hip_bfloat16>& x,hipStream_t s){require(x.size()==count,"upload size mismatch");check(hipMemcpyAsync(data(),x.data(),count*2,hipMemcpyHostToDevice,s));check(hipStreamSynchronize(s));}
    std::vector<hip_bfloat16> get(hipStream_t s)const{std::vector<hip_bfloat16>x(count);check(hipMemcpyAsync(x.data(),data(),count*2,hipMemcpyDeviceToHost,s));check(hipStreamSynchronize(s));return x;}
    void poison(hipStream_t s){check(hipMemsetAsync(data(),0xff,count*2,s));}
    void guards(hipStream_t s)const{
        std::vector<unsigned char> a(prefix),b(Guard);
        check(hipMemcpyAsync(a.data(),allocation,prefix,hipMemcpyDeviceToHost,s));
        check(hipMemcpyAsync(b.data(),allocation+prefix+count*2,Guard,hipMemcpyDeviceToHost,s));check(hipStreamSynchronize(s));
        require(std::all_of(a.begin(),a.end(),[](auto x){return x==0xa5;}) && std::all_of(b.begin(),b.end(),[](auto x){return x==0xa5;}),"canary changed");
    }
};
void equal(const std::vector<hip_bfloat16>& a,const std::vector<hip_bfloat16>& b){require(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*2)==0,"exact BF16 comparison failed");}
std::vector<hip_bfloat16> weights(unsigned n,unsigned k){
    std::vector<hip_bfloat16> w(static_cast<std::size_t>(n)*k);
    for(unsigned r=0;r<n;++r)for(unsigned c=0;c<k;++c)
        w[static_cast<std::size_t>(r)*k+c]=hip_bfloat16(float(int((r*17U+c*11U)%511U)-255)/128.0F);
    return w;
}
std::vector<hip_bfloat16> input(unsigned t,unsigned k,unsigned fixture){
    std::vector<hip_bfloat16> x(t*k);
    for(unsigned row=0;row<t;++row)for(unsigned c=0;c<k;++c)
        x[row*k+c]=hip_bfloat16(fixture==0?float(int((row*31U+c*37U)%509U)-254)/127.0F:
            (c%2?-1.0F:1.0F)*(1.0F+float(row+1)/16.0F));
    return x;
}
double oracle(const std::vector<hip_bfloat16>& y,const std::vector<hip_bfloat16>& x,
              const std::vector<hip_bfloat16>& w,unsigned t,unsigned n,unsigned k){
    for(auto value:y)require(std::isfinite(float(value)),"nonfinite or unwritten output");
    const std::array<unsigned,11> rows{0,1,7,15,16,31,32,n/3,n/2,n-2,n-1};
    double maximum=0;
    for(unsigned token=0;token<t;++token)for(auto row:rows){
        double sum=0;
        for(unsigned c=0;c<k;++c)sum=std::fma(double(float(x[token*k+c])),double(float(w[static_cast<std::size_t>(row)*k+c])),sum);
        const double error=std::abs(double(float(y[token*n+row]))-sum);
        // Unchanged selective-protected BF16 qualification criterion.
        const double normalized=error/(1e-5+0.0079*std::abs(sum));
        if(!std::isfinite(normalized)||normalized>1)throw std::runtime_error(
            "BF16 FP64 oracle failed N="+std::to_string(n)+" K="+std::to_string(k)+
            " T="+std::to_string(t)+" token="+std::to_string(token)+" row="+std::to_string(row));
        maximum=std::max(maximum,normalized);
    }
    return maximum;
}
using Launch=hipError_t(*)(const linear::Bf16LinearArgs&,hipStream_t) noexcept;
struct Graph {
    hipGraph_t graph{};hipGraphExec_t executable{};
    Graph(Launch launch,const linear::Bf16LinearArgs& a,hipStream_t s){
        check(hipStreamBeginCapture(s,hipStreamCaptureModeGlobal));check(launch(a,s));
        check(hipStreamEndCapture(s,&graph));check(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    }
    ~Graph(){(void)hipGraphExecDestroy(executable);(void)hipGraphDestroy(graph);}
    void run(hipStream_t s){check(hipGraphLaunch(executable,s));}
};
double measure(Graph& graph,hipStream_t stream){
    hipEvent_t start{},end{};check(hipEventCreate(&start));check(hipEventCreate(&end));
    check(hipEventRecord(start,stream));for(unsigned i=0;i<16;++i)graph.run(stream);
    check(hipEventRecord(end,stream));check(hipEventSynchronize(end));float ms=0;check(hipEventElapsedTime(&ms,start,end));
    check(hipEventDestroy(start));check(hipEventDestroy(end));require(std::isfinite(ms)&&ms>0,"invalid timing");return ms/16;
}
void cell(unsigned n,unsigned k,unsigned t,unsigned offset,const std::vector<hip_bfloat16>& w,
          Buffer& dw,hipStream_t s,std::ostream& out){
    Buffer dx(t*k,offset,s),control(t*n,offset,s),candidate(t*n,offset,s);
    const linear::Bf16LinearArgs ca{dx.data(),dw.data(),control.data(),t,n,k},ta{dx.data(),dw.data(),candidate.data(),t,n,k};
    auto bad=ta;bad.weights=nullptr;require(linear::bf16_linear(bad,s)==hipErrorInvalidValue,"null weight accepted");
    bad=ta;bad.input=nullptr;require(linear::bf16_linear(bad,s)==hipErrorInvalidValue,"null input accepted");
    bad=ta;bad.output=nullptr;require(linear::bf16_linear(bad,s)==hipErrorInvalidValue,"null output accepted");
    Graph cg(linear::bf16_linear_wave8,ca,s),tg(linear::bf16_linear,ta,s);
    double error=0;
    for(unsigned fixture=0;fixture<2;++fixture){
        const auto x=input(t,k,fixture);dx.put(x,s);control.poison(s);candidate.poison(s);
        check(linear::bf16_linear_wave8(ca,s));check(linear::bf16_linear(ta,s));
        const auto baseline=control.get(s);const auto actual=candidate.get(s);equal(actual,baseline);
        error=std::max({error,oracle(baseline,x,w,t,n,k),oracle(actual,x,w,t,n,k)});
        for(unsigned replay=0;replay<2;++replay){control.poison(s);candidate.poison(s);cg.run(s);tg.run(s);equal(control.get(s),baseline);equal(candidate.get(s),baseline);}
        equal(dx.get(s),x);dx.guards(s);control.guards(s);candidate.guards(s);dw.guards(s);
    }
    equal(dw.get(s),w);dx.put(input(t,k,0),s);power();
    for(unsigned i=0;i<32;++i){cg.run(s);tg.run(s);}check(hipStreamSynchronize(s));
    out<<"{\"rows\":"<<n<<",\"columns\":"<<k<<",\"tokens\":"<<t<<",\"address_mod16\":"<<offset
       <<",\"all_outputs_exact\":true,\"fixtures\":2,\"poisoned_graph_replays_per_fixture\":2,"
       <<"\"maximum_normalized_fp64_error\":"<<error<<",\"guards_and_immutability\":true,\"samples\":[";
    for(unsigned pair=0;pair<7;++pair){double c,v;if(pair%2==0){c=measure(cg,s);v=measure(tg,s);}else{v=measure(tg,s);c=measure(cg,s);}
        if(pair)out<<',';out<<"{\"control_first\":"<<(pair%2==0?"true":"false")<<",\"control_ms\":"<<c<<",\"candidate_ms\":"<<v<<'}';}
    power();out<<"]}";
}
}
int main(int argc,char** argv){try{
    require(argc==2,"usage: bf16_staging_qual FRESH_DIRECTORY");
    const std::filesystem::path directory=argv[1];require(std::filesystem::create_directory(directory),"output already exists");
    require(read(Pci/"vendor")=="0x1002" && read(Pci/"device")=="0x7551","AMD R9700 required");power();
    check(hipSetDevice(0));hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));char pci[32]{};check(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    require(std::string(pci)=="0000:13:00.0" && std::string(properties.gcnArchName)=="gfx1201" && properties.warpSize==32,"HIP/PCI device mismatch");
    hipStream_t stream{};check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::ofstream report(directory/"report.json");require(bool(report),"cannot open report");report<<std::setprecision(17);
    report<<"{\"schema\":\"ninfer.r9700.bf16-staging-qualification.v1\",\"production_selected\":true,"
          <<"\"pci\":\"0000:13:00.0\",\"target\":\"gfx1201\",\"power\":\"auto\","
          <<"\"oracle\":\"original BF16 input and weights, full-K FP64 at every token and sampled rows\","
          <<"\"absolute_tolerance\":1e-5,\"relative_tolerance\":0.0079,\"warmup_calls_per_route\":32,"
          <<"\"calls_per_sample\":16,\"timing\":\"same-stream graph events, milliseconds per complete Linear call\",\"cells\":[";
    bool first=true;
    for(const auto shape:{std::array<unsigned,2>{7168,5120},std::array<unsigned,2>{5120,6144}}){
        const auto w=weights(shape[0],shape[1]);
        for(unsigned offset:{0U,2U}){Buffer dw(w.size(),offset,stream);dw.put(w,stream);
            for(unsigned t:{5U,6U}){if(!first)report<<',';first=false;cell(shape[0],shape[1],t,offset,w,dw,stream,report);}}
    }
    report<<"],\"passed\":true}\n";require(bool(report),"report write failed");check(hipStreamDestroy(stream));
    std::cout<<"BF16 staging qualification PASS; performance admission remains separate\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
