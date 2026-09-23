#include "ninfer/ops/gdn_gating.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr unsigned N=48,K=5120,Guard=256;
void require(bool p,const char* message){if(!p)throw std::runtime_error(message);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<class T> struct Buffer {
    unsigned char* allocation{};std::size_t count,prefix;
    Buffer(std::size_t n,unsigned offset,hipStream_t s):count(n),prefix(Guard+offset){
        check(hipMalloc(&allocation,prefix+count*sizeof(T)+Guard));
        check(hipMemsetAsync(allocation,0xa5,prefix+count*sizeof(T)+Guard,s));
        require(reinterpret_cast<std::uintptr_t>(data())%16==offset,"fixture alignment mismatch");
    }
    ~Buffer(){(void)hipFree(allocation);}Buffer(const Buffer&)=delete;
    T* data()const{return reinterpret_cast<T*>(allocation+prefix);}
    void put(const std::vector<T>& x,hipStream_t s){require(x.size()==count,"upload size mismatch");
        check(hipMemcpyAsync(data(),x.data(),count*sizeof(T),hipMemcpyHostToDevice,s));check(hipStreamSynchronize(s));}
    std::vector<T> get(hipStream_t s)const{std::vector<T>x(count);
        check(hipMemcpyAsync(x.data(),data(),count*sizeof(T),hipMemcpyDeviceToHost,s));check(hipStreamSynchronize(s));return x;}
    void poison(hipStream_t s){check(hipMemsetAsync(data(),0xff,count*sizeof(T),s));}
    void guards(hipStream_t s)const{
        std::vector<unsigned char>a(prefix),b(Guard);
        check(hipMemcpyAsync(a.data(),allocation,prefix,hipMemcpyDeviceToHost,s));
        check(hipMemcpyAsync(b.data(),allocation+prefix+count*sizeof(T),Guard,hipMemcpyDeviceToHost,s));
        check(hipStreamSynchronize(s));
        require(std::all_of(a.begin(),a.end(),[](auto v){return v==0xa5;}) &&
                std::all_of(b.begin(),b.end(),[](auto v){return v==0xa5;}),"canary changed");
    }
};
template<class T>void exact(const std::vector<T>& a,const std::vector<T>& b){
    require(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0,"exact output/input comparison failed");}
struct Graph {
    hipGraph_t graph{};hipGraphExec_t executable{};
    Graph(hipStream_t s,const std::function<void()>& f){check(hipStreamBeginCapture(s,hipStreamCaptureModeGlobal));f();
        check(hipStreamEndCapture(s,&graph));check(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));}
    ~Graph(){(void)hipGraphExecDestroy(executable);(void)hipGraphDestroy(graph);}
    void run(hipStream_t s){check(hipGraphLaunch(executable,s));}
};
// Independent mathematical BF16 RNE seam, not the production FP32 bit-cast.
double bf16_rne(double x){
    if(x==0)return x;
    int exponent=0;std::frexp(std::abs(x),&exponent);
    const double quantum=std::ldexp(1.0,std::max(exponent-8,-133));
    return std::nearbyint(x/quantum)*quantum;
}
struct Reference{std::vector<double>g,beta;};
Reference oracle(unsigned t,const std::vector<hip_bfloat16>& x,
                 const std::vector<hip_bfloat16>& aw,const std::vector<hip_bfloat16>& bw,
                 const std::vector<float>& log,const std::vector<float>& bias){
    Reference result;
    for(unsigned token=0;token<t;++token)for(unsigned head=0;head<N;++head){
        double a=0,b=0;
        for(unsigned k=0;k<K;++k){const double v=float(x[token*K+k]);
            a+=v*double(float(aw[head*K+k]));b+=v*double(float(bw[head*K+k]));}
        const double av=bf16_rne(a)+double(bias[head]),bv=bf16_rne(b);
        // Stable independent softplus/sigmoid; matches the public formula,
        // not the production branch, intrinsic approximations or reduction.
        const double softplus=std::max(av,0.0)+std::log1p(std::exp(-std::abs(av)));
        result.g.push_back(-std::exp(double(log[head]))*softplus);
        result.beta.push_back(bv>=0?1/(1+std::exp(-bv)):std::exp(bv)/(1+std::exp(bv)));
    }
    return result;
}
double compare(const std::vector<float>& actual,const std::vector<double>& reference,bool beta){
    double worst=0;require(actual.size()==reference.size(),"oracle size mismatch");
    for(std::size_t i=0;i<actual.size();++i){
        // Predeclared existing GDN public criteria, not tuned after execution.
        const double limit=std::max(beta?0.0003:0.0625,(beta?0.00003:0.001)*std::abs(reference[i]));
        const double e=std::abs(double(actual[i])-reference[i])/limit;
        require(std::isfinite(e)&&e<=1,"independent public FP64 gating oracle failed");worst=std::max(worst,e);
    }
    return worst;
}
void cell(unsigned t,unsigned offset,hipStream_t s,std::ostream& out){
    std::vector<hip_bfloat16>aw(N*K),bw(N*K);
    for(unsigned h=0;h<N;++h)for(unsigned k=0;k<K;++k){
        aw[h*K+k]=hip_bfloat16(float(int((h*31+k*17)%509)-254)/1024.0F);
        bw[h*K+k]=hip_bfloat16(float(int((h*43+k*23+7)%503)-251)/1024.0F);}
    std::vector<float>log(N),bias(N);
    for(unsigned h=0;h<N;++h){log[h]=float(int(h%13)-6)/8;bias[h]=float(int(h%17)-8)/4;}
    bias[0]=21;bias[1]=-21;bias[2]=100;bias[3]=-100;
    Buffer<hip_bfloat16>x(t*K,offset,s),a(N*K,offset,s),b(N*K,offset,s);
    Buffer<float>al(N,4,s),dt(N,4,s),tg(t*N,4,s),tb(t*N,4,s);
    a.put(aw,s);b.put(bw,s);al.put(log,s);dt.put(bias,s);
    const auto weight=[](hip_bfloat16* data) {
        ninfer::Weight w{}; w.payload=data; w.payload_bytes=N*K*sizeof(hip_bfloat16);
        w.qtype=ninfer::QType::BF16_CTRL; w.layout=ninfer::QuantLayout::Contiguous;
        w.ndim=2; w.n=w.shape[0]=w.padded_shape[0]=N;
        w.k=w.shape[1]=w.padded_shape[1]=K; w.qdata=data; w.qdata_bytes=w.payload_bytes;
        return w;
    };
    const auto wa=weight(a.data()),wb=weight(b.data());
    const ninfer::Tensor input(x.data(),ninfer::DType::BF16,{int(K),int(t)});
    const ninfer::Tensor log_tensor(al.data(),ninfer::DType::FP32,{int(N)});
    const ninfer::Tensor bias_tensor(dt.data(),ninfer::DType::FP32,{int(N)});
    ninfer::Tensor g_tensor(tg.data(),ninfer::DType::FP32,{int(N),int(t)});
    ninfer::Tensor beta_tensor(tb.data(),ninfer::DType::FP32,{int(N),int(t)});
    auto target=[&]{ninfer::ops::bf16_gdn_projected_gating(
        input,wa,wb,log_tensor,bias_tensor,g_tensor,beta_tensor,s);};
    const auto reject=[&](const std::function<void()>& call) {
        bool rejected=false;try{call();}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"malformed public Op input accepted");
    };
    reject([&]{ninfer::ops::bf16_gdn_projected_gating(
        input,wa,wb,log_tensor,bias_tensor,g_tensor,beta_tensor,nullptr);});
    for(int bad_t:{2,4,7}) {
        const ninfer::Tensor bad_input(x.data(),ninfer::DType::BF16,{int(K),bad_t});
        reject([&]{ninfer::ops::bf16_gdn_projected_gating(
            bad_input,wa,wb,log_tensor,bias_tensor,g_tensor,beta_tensor,s);});
    }
    reject([&]{ninfer::ops::bf16_gdn_projected_gating(
        input,wa,wb,log_tensor,bias_tensor,g_tensor,g_tensor,s);});
    Graph graph(s,target);
    double new_g=0,new_b=0;
    for(unsigned fixture=0;fixture<3;++fixture){
        std::vector<hip_bfloat16>host(t*K);
        for(unsigned token=0;token<t;++token)for(unsigned k=0;k<K;++k){
            const float value=fixture==0?float(int((k*37+token*31)%509)-254)/127.0F:
                fixture==1?(k%2?-1.0F:1.0F)*(1.0F+float(token+1)/16):
                float(int((k*71+token*53)%521)-260)/float(33+(k%7));
            host[token*K+k]=hip_bfloat16(value);}
        x.put(host,s);const auto ref=oracle(t,host,aw,bw,log,bias);
        target();const auto expected_g=tg.get(s),expected_b=tb.get(s);
        for(unsigned replay=0;replay<3;++replay){
            if(replay){tg.poison(s);tb.poison(s);graph.run(s);}
            const auto actual_g=tg.get(s),actual_b=tb.get(s);
            exact(expected_g,actual_g);exact(expected_b,actual_b);
            new_g=std::max(new_g,compare(actual_g,ref.g,false));new_b=std::max(new_b,compare(actual_b,ref.beta,true));
        }
        exact(x.get(s),host);exact(a.get(s),aw);exact(b.get(s),bw);exact(al.get(s),log);exact(dt.get(s),bias);
        x.guards(s);a.guards(s);b.guards(s);al.guards(s);dt.guards(s);
        tg.guards(s);tb.guards(s);
    }
    out<<"{\"tokens\":"<<t<<",\"bf16_address_mod16\":"<<offset<<",\"fp32_address_mod16\":4,"
       <<"\"own_eager_graph_exact\":true,\"guards_and_immutability\":true,\"fixtures\":3,"
       <<"\"maximum_normalized_oracle_g\":"<<new_g<<",\"maximum_normalized_oracle_beta\":"<<new_b<<'}';
}
}
int main(int argc,char**argv){try{
    require(argc==3 && std::string(argv[1])=="--out-json",
            "usage: gdn_projected_controls_qual --out-json FRESH.json");
    const std::filesystem::path path=argv[2];require(!std::filesystem::exists(path),"output exists");
    check(hipSetDevice(0));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,0));char pci[32]{};check(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    require(std::string(props.name)=="AMD Radeon AI PRO R9700" && std::string(props.gcnArchName)=="gfx1201" && props.warpSize==32 &&
            (std::string(pci)=="0000:13:00.0" || std::string(pci)=="13:00.0"),"wrong device");
    hipStream_t s{};check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"schema\":\"ninfer.r9700.gdn-projected-controls.v1\",\"public_op_tested\":true,\"target\":\"gfx1201\","
          "\"pci\":\"0000:13:00.0\",\"oracle\":\"original BF16 dots FP64, independent BF16 RNE seams, FP64 gates every output\","
          "\"criterion\":\"g abs<=0.0625 OR rel<=0.001; beta abs<=0.0003 OR rel<=0.00003\",\"cells\":[";
    bool first=true;for(unsigned t:{1U,5U,6U})for(unsigned offset:{0U,2U}){
        if(!first)out<<',';first=false;cell(t,offset,s,out);}
    out<<"],\"numerical_passed\":true}\n";
    check(hipStreamDestroy(s));const auto data=out.str();
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);require(fd>=0,"cannot create output");
    const auto written=::write(fd,data.data(),data.size());::close(fd);require(written==static_cast<ssize_t>(data.size()),"short output write");
    std::cout<<"selected public projected controls: PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
