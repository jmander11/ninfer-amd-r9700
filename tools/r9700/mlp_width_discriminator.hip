// Diagnostic replay, not an independent numerical qualification or speed test.
#include "artifact/reader.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/residual_add.h"
#include <hip/hip_bfloat16.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace artifact=ninfer::artifact;
constexpr unsigned H=5120,I=17408;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Device {
    void* p{};std::size_t bytes;
    explicit Device(std::size_t n):bytes(n){HIP_CHECK(hipMalloc(&p,n));}
    ~Device(){(void)hipFree(p);} Device(const Device&)=delete;
};
std::vector<std::uint16_t> read_trace(const char* path) {
    require(std::filesystem::file_size(path)==129ULL*H*2,"trace sidecar size differs");
    std::vector<std::uint16_t> data(129ULL*H);std::ifstream file(path,std::ios::binary);
    require(bool(file.read(reinterpret_cast<char*>(data.data()),data.size()*2)),"trace read failed");return data;
}
const artifact::TensorDescriptor& descriptor(const artifact::Reader& reader,const std::string& name) {
    const auto* object=reader.find(name);require(object!=nullptr,"artifact object absent");
    const auto* tensor=std::get_if<artifact::TensorDescriptor>(object);require(tensor!=nullptr,"object not tensor");return *tensor;
}
struct Matrix {
    Device storage;ninfer::Weight weight{};
    Matrix(const artifact::Reader& reader,const char* name,unsigned n,unsigned k)
        :storage(descriptor(reader,name).bytes) {
        const auto& d=descriptor(reader,name);
        require(d.format==artifact::NumericFormat::Q4G64_F16S &&
                d.layout==artifact::StorageLayout::R9700Q4G64N16K16V1 &&
                d.shape==std::vector<std::uint64_t>{n,k},"wrong layer0 Q4 descriptor");
        const auto geometry=artifact::r9700_q4g64_n16k16_geometry(d.shape);
        require(d.bytes==geometry.encoded_bytes,"wrong Q4 payload extent");
        const auto payload=reader.payload(d);
        HIP_CHECK(hipMemcpy(storage.p,payload.data.data(),d.bytes,hipMemcpyHostToDevice));
        weight.qtype=ninfer::QType::Q4G64_F16S;weight.layout=ninfer::QuantLayout::Q4N16K16;
        weight.ndim=2;weight.n=weight.shape[0]=weight.padded_shape[0]=n;
        weight.k=weight.shape[1]=weight.padded_shape[1]=k;
        weight.group=weight.group_size=64;weight.scale_dtype=ninfer::DType::FP16;
        weight.payload=weight.qdata=storage.p;weight.payload_bytes=d.bytes;
        weight.qdata_bytes=geometry.low_plane_bytes;
        weight.scales=static_cast<std::byte*>(storage.p)+geometry.scale_plane_offset;
        weight.scale_bytes=geometry.scale_plane_bytes;
    }
};
using Stages=std::array<std::vector<std::uint16_t>,5>;
constexpr std::array<const char*,5> names{"norm","gate_up","silu","down","residual"};
Stages replay(unsigned tokens,unsigned column,const std::vector<std::uint16_t>& trace,
              const ninfer::Tensor& norm,const Matrix& gate,const Matrix& down,hipStream_t stream) {
    Device x(H*tokens*2),h(H*tokens*2),gu(2*I*tokens*2),activation(I*tokens*2),delta(H*tokens*2);
    const auto capacity=std::max(ninfer::ops::linear_workspace_capacity_bytes(gate.weight.qtype,tokens,H),
                                  ninfer::ops::linear_workspace_capacity_bytes(down.weight.qtype,tokens,I));
    Device scratch(capacity);
    std::vector<hip_bfloat16> input(H*tokens);
    for(unsigned t=0;t<tokens;++t)for(unsigned k=0;k<H;++k)
        input[t*H+k]=hip_bfloat16(float(int((k*13+t*19)%127)-63)/64.0F);
    std::memcpy(input.data()+column*H,trace.data()+H,H*2); // layer0 post-mixer
    HIP_CHECK(hipMemcpyAsync(x.p,input.data(),input.size()*2,hipMemcpyHostToDevice,stream));
    ninfer::Tensor tx(x.p,ninfer::DType::BF16,{int(H),int(tokens)}),th(h.p,ninfer::DType::BF16,{int(H),int(tokens)});
    ninfer::Tensor tg(gu.p,ninfer::DType::BF16,{int(2*I),int(tokens)}),ta(activation.p,ninfer::DType::BF16,{int(I),int(tokens)});
    ninfer::Tensor td(delta.p,ninfer::DType::BF16,{int(H),int(tokens)});
    Stages result;
    auto capture=[&](unsigned stage,const Device& buffer,unsigned rows){
        result[stage].resize(rows);
        HIP_CHECK(hipMemcpyAsync(result[stage].data(),static_cast<std::byte*>(buffer.p)+column*rows*2,
                                rows*2,hipMemcpyDeviceToHost,stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    };
    ninfer::ops::rmsnorm(tx,norm,1.e-6F,true,th,stream);capture(0,h,H);
    ninfer::ops::linear(th,gate.weight,tg,ninfer::DeviceSpan{scratch.p,capacity},stream);capture(1,gu,2*I);
    ninfer::ops::silu_mul(tg.slice(0,0,I),tg.slice(0,I,I),ta,stream);capture(2,activation,I);
    ninfer::ops::linear(ta,down.weight,td,ninfer::DeviceSpan{scratch.p,capacity},stream);capture(3,delta,H);
    ninfer::ops::residual_add(td,tx,stream);capture(4,x,H);
    return result;
}
void write(const std::filesystem::path& path,const std::vector<std::uint16_t>& data) {
    std::ofstream out(path,std::ios::binary);require(bool(out.write(reinterpret_cast<const char*>(data.data()),data.size()*2)),"stage write failed");
}
}
int main(int argc,char** argv){try{
    const bool invariant=argc==10 && std::string(argv[9])=="--expect-width-invariant";
    require((argc==9 || invariant) && std::string(argv[1])=="--weights" && std::string(argv[3])=="--ordinary-trace" &&
            std::string(argv[5])=="--verify-trace" && std::string(argv[7])=="--out-dir",
            "usage: mlp_width_discriminator --weights FILE --ordinary-trace BIN --verify-trace BIN --out-dir FRESH [--expect-width-invariant]");
    const auto ordinary=read_trace(argv[4]),verify=read_trace(argv[6]);
    require(std::equal(ordinary.begin()+H,ordinary.begin()+2*H,verify.begin()+H),"post-mixer inputs differ");
    const std::filesystem::path output=argv[8];require(!std::filesystem::exists(output),"output already exists");
    artifact::Reader reader(argv[2]);
    require(reader.identity().model_id=="qwen3.8-27b" &&
            reader.identity().weights_id=="r9700-q4-selective-protected-n16k16-dflash2-q4-eval","wrong selected artifact identity");
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    require(std::string(props.name)=="AMD Radeon AI PRO R9700" &&
            std::string(props.gcnArchName)=="gfx1201" && props.warpSize==32 &&
            (std::string(pci)=="0000:13:00.0" || std::string(pci)=="13:00.0"),"wrong GPU target");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    Matrix gate(reader,"text/layers/0/mlp/gate_up",2*I,H),down(reader,"text/layers/0/mlp/down",H,I);
    const auto& n=descriptor(reader,"text/layers/0/post_attention_norm");
    require(n.format==artifact::NumericFormat::BF16 && n.layout==artifact::StorageLayout::ContiguousLeV1 &&
            n.shape==std::vector<std::uint64_t>{H} && n.bytes==H*2,"wrong norm descriptor");
    Device norm_storage(n.bytes);const auto payload=reader.payload(n);
    HIP_CHECK(hipMemcpy(norm_storage.p,payload.data.data(),n.bytes,hipMemcpyHostToDevice));
    ninfer::Tensor norm(norm_storage.p,ninfer::DType::BF16,{int(H)});
    const auto a=replay(2,1,ordinary,norm,gate,down,stream),b=replay(10,5,verify,norm,gate,down,stream);
    const bool a_match=std::equal(a[4].begin(),a[4].end(),ordinary.begin()+2*H);
    const bool b_match=std::equal(b[4].begin(),b[4].end(),verify.begin()+2*H);
    require(std::filesystem::create_directory(output),"cannot create fresh output directory");
    std::ofstream report(output/"report.json");
    report<<"{\"mode\":"<<std::quoted(invariant?"width-invariant-regression":"captured-endpoint-diagnostic")
          <<",\"diagnostic_only\":"<<(invariant?"false":"true")
          <<",\"numerical_qualification\":false,\"ordinary_T2_column1_matches_trace\":"<<(a_match?"true":"false")
          <<",\"verify_T10_column5_matches_trace\":"<<(b_match?"true":"false")<<",\"stages\":[";
    bool all_stages_equal=true;
    for(unsigned i=0;i<names.size();++i){
        write(output/(std::string("ordinary-")+names[i]+".bin"),a[i]);write(output/(std::string("verify-")+names[i]+".bin"),b[i]);
        std::size_t count=0,first=a[i].size();for(std::size_t j=0;j<a[i].size();++j)if(a[i][j]!=b[i][j]){++count;first=std::min(first,j);}
        all_stages_equal=all_stages_equal && count==0;
        report<<(i?",":"")<<"{\"name\":"<<std::quoted(names[i])<<",\"elements\":"<<a[i].size()<<",\"mismatches\":"<<count
              <<",\"first_index\":"<<(count?std::to_string(first):"null")<<'}';
    }
    const bool passed=a_match && (invariant?all_stages_equal:b_match);
    report<<"],\"all_stages_width_invariant\":"<<(all_stages_equal?"true":"false")
          <<",\"passed\":"<<(passed?"true":"false")<<"}\n";
    report.close();require(bool(report),"report write failed");HIP_CHECK(hipStreamDestroy(stream));
    require(a_match,"ordinary replay does not reproduce captured post-MLP boundary");
    if(invariant)require(all_stages_equal,"width-invariance regression: T2/T10 replay stages differ");
    else require(b_match,"verify replay does not reproduce captured post-MLP boundary; localization is invalid");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
