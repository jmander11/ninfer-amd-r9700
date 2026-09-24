// Captured-input diagnostic only: no independent numerical qualification or timing claim.
#include "artifact/reader.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include <hip/hip_bfloat16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr unsigned H=5120,N=248320,DOMAIN=248077;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Device {
    void* p{};
    explicit Device(std::size_t bytes){HIP_CHECK(hipMalloc(&p,bytes));}
    ~Device(){(void)hipFree(p);} Device(const Device&)=delete;
};
const ninfer::artifact::TensorDescriptor& descriptor(const ninfer::artifact::Reader& reader,const char* name){
    const auto* object=reader.find(name);require(object,"missing artifact object");
    const auto* d=std::get_if<ninfer::artifact::TensorDescriptor>(object);
    require(d,"artifact object is not tensor");return *d;
}
struct Result {std::vector<hip_bfloat16> norm,logits;};
Result replay(unsigned tokens,unsigned column,const std::vector<hip_bfloat16>& input,
              const ninfer::Tensor& norm,const ninfer::Weight& head,hipStream_t stream){
    Device x(H*tokens*2),h(H*tokens*2),y(std::size_t(N)*tokens*2);
    const auto capacity=ninfer::ops::linear_workspace_capacity_bytes(head.qtype,tokens,H);
    Device scratch(std::max<std::size_t>(capacity,1));
    std::vector<hip_bfloat16> batch(H*tokens);
    for(unsigned t=0;t<tokens;++t)for(unsigned k=0;k<H;++k)
        batch[t*H+k]=hip_bfloat16(float(int((k*13+t*19)%127)-63)/64.F);
    std::copy(input.begin(),input.end(),batch.begin()+column*H);
    HIP_CHECK(hipMemcpyAsync(x.p,batch.data(),batch.size()*2,hipMemcpyHostToDevice,stream));
    ninfer::Tensor tx(x.p,ninfer::DType::BF16,{int(H),int(tokens)}),th(h.p,ninfer::DType::BF16,{int(H),int(tokens)});
    ninfer::Tensor ty(y.p,ninfer::DType::BF16,{int(N),int(tokens)});
    ninfer::ops::rmsnorm(tx,norm,1.e-6F,true,th,stream);
    ninfer::ops::linear(th,head,ty,ninfer::DeviceSpan{scratch.p,capacity},stream);
    Result result;result.norm.resize(H);result.logits.resize(N);
    HIP_CHECK(hipMemcpyAsync(result.norm.data(),static_cast<hip_bfloat16*>(h.p)+column*H,H*2,hipMemcpyDeviceToHost,stream));
    HIP_CHECK(hipMemcpyAsync(result.logits.data(),static_cast<hip_bfloat16*>(y.p)+std::size_t(column)*N,N*2,hipMemcpyDeviceToHost,stream));
    HIP_CHECK(hipStreamSynchronize(stream));return result;
}
unsigned argmax(const std::vector<hip_bfloat16>& values){
    unsigned best=0;
    for(unsigned i=1;i<DOMAIN;++i)if(float(values[i])>float(values[best]))best=i;
    return best;
}
void write(const std::filesystem::path& path,const std::vector<hip_bfloat16>& data){
    std::ofstream out(path,std::ios::binary);require(bool(out.write(reinterpret_cast<const char*>(data.data()),data.size()*2)),"write failed");
}
}
int main(int argc,char** argv){try{
    require(argc==7 && std::string(argv[1])=="--weights" && std::string(argv[3])=="--trace" && std::string(argv[5])=="--out-dir",
        "usage: head_width_discriminator --weights FILE --trace ORDINARY_BOUNDARY.bin --out-dir FRESH");
    const std::filesystem::path out=argv[6];require(!std::filesystem::exists(out),"output exists");
    require(std::filesystem::file_size(argv[4])==129ULL*H*2,"wrong boundary sidecar size");
    std::ifstream trace(argv[4],std::ios::binary);trace.seekg(128ULL*H*2);
    std::vector<hip_bfloat16> input(H);require(bool(trace.read(reinterpret_cast<char*>(input.data()),H*2)),"trace read failed");
    ninfer::artifact::Reader reader(argv[2]);
    require(reader.identity().model_id=="qwen3.8-27b" && reader.identity().weights_id=="r9700-q4-selective-protected-n16k16-dflash2-q4-eval","wrong selected artifact");
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    require(std::string(props.name)=="AMD Radeon AI PRO R9700" && std::string(props.gcnArchName)=="gfx1201" && props.warpSize==32 &&
        (std::string(pci)=="0000:13:00.0" || std::string(pci)=="13:00.0"),"wrong GPU target");
    const auto& d=descriptor(reader,"text/output_head");
    require(d.format==ninfer::artifact::NumericFormat::W8G32_F16S && d.layout==ninfer::artifact::StorageLayout::R9700W8G32N16K16V1 && d.shape==std::vector<std::uint64_t>{N,H},"wrong tiled head descriptor");
    const auto geometry=ninfer::artifact::r9700_w8g32_n16k16_geometry(d.shape);
    require(d.bytes==geometry.encoded_bytes,"wrong head extent");Device weights(d.bytes);
    const auto payload=reader.payload(d);HIP_CHECK(hipMemcpy(weights.p,payload.data.data(),d.bytes,hipMemcpyHostToDevice));
    ninfer::Weight head{};head.qtype=ninfer::QType::W8G32_F16S;head.layout=ninfer::QuantLayout::W8N16K16;head.ndim=2;
    head.n=head.shape[0]=head.padded_shape[0]=N;head.k=head.shape[1]=head.padded_shape[1]=H;
    head.group=head.group_size=32;head.scale_dtype=ninfer::DType::FP16;
    head.qdata=head.payload=weights.p;head.payload_bytes=d.bytes;head.qdata_bytes=geometry.low_plane_bytes;
    head.scales=static_cast<std::byte*>(weights.p)+geometry.scale_plane_offset;head.scale_bytes=geometry.scale_plane_bytes;
    const auto& nd=descriptor(reader,"text/final_norm");
    require(nd.format==ninfer::artifact::NumericFormat::BF16 && nd.layout==ninfer::artifact::StorageLayout::ContiguousLeV1 && nd.shape==std::vector<std::uint64_t>{H} && nd.bytes==H*2,"wrong final norm");
    Device normdata(nd.bytes);const auto np=reader.payload(nd);HIP_CHECK(hipMemcpy(normdata.p,np.data.data(),nd.bytes,hipMemcpyHostToDevice));
    ninfer::Tensor norm(normdata.p,ninfer::DType::BF16,{int(H)});
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    const auto a=replay(2,1,input,norm,head,stream),b=replay(10,5,input,norm,head,stream);
    const bool norm_exact=std::memcmp(a.norm.data(),b.norm.data(),H*2)==0;
    std::size_t differences=0;for(unsigned i=0;i<N;++i){
        require(std::isfinite(float(a.logits[i])) && std::isfinite(float(b.logits[i])),"nonfinite logits");
        differences+=a.logits[i].data!=b.logits[i].data;
    }
    require(std::filesystem::create_directory(out),"cannot create output");
    write(out/"ordinary-norm.bin",a.norm);write(out/"verify-norm.bin",b.norm);
    write(out/"ordinary-logits.bin",a.logits);write(out/"verify-logits.bin",b.logits);
    std::ofstream report(out/"report.json");
    report<<"{\"diagnostic_only\":true,\"numerical_qualification\":false,\"norm_exact\":"<<(norm_exact?"true":"false")
        <<",\"logit_mismatches\":"<<differences<<",\"token_domain\":"<<DOMAIN
        <<",\"ordinary_argmax\":"<<argmax(a.logits)<<",\"verify_argmax\":"<<argmax(b.logits)
        <<",\"ordinary_4031\":"<<float(a.logits[4031])<<",\"ordinary_854\":"<<float(a.logits[854])
        <<",\"verify_4031\":"<<float(b.logits[4031])<<",\"verify_854\":"<<float(b.logits[854])<<"}\n";
    report.close();require(bool(report),"report write failed");HIP_CHECK(hipStreamDestroy(stream));
    require(norm_exact,"head profile isolation invalid: normalized inputs differ");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
