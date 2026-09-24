// Exact affected-domain qualification through the canonical public Op suite.
#define NINFER_QUAL_SMALL_BATCH_PROJECTIONS 1
#define NINFER_A8Q4_VERIFY_QUAL_NO_MAIN 1
#include "a8q4_verify_qual_impl.h"
int main(int argc,char** argv) {
 try {
    if(argc!=2)fail("usage: k4_qual FRESH.json");
    const std::filesystem::path path=argv[1];require_fresh_output(path);power();
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props,0));char pci[32]{};
    HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(props.gcnArchName)!="gfx1201" || props.warpSize!=32 ||
       (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong GPU");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::ostringstream out;out<<std::setprecision(17)
        <<"{\"scope\":\"mlp_k4_t10_t15_t20\",\"cells\":[";
    bool first=true;
    for(const auto shape:std::array<std::array<unsigned,2>,2>{{{34816,5120},{5120,17408}}}) {
        N=shape[0];K=shape[1];G=K/64;
        for(unsigned t:{10U,15U,20U}) {
            if(!first)out<<',';first=false;cell(t,stream,out);
            std::cout<<N<<'/'<<K<<" T"<<t<<" passed\n"<<std::flush;
        }
    }
    out<<"],\"status\":\"qualified\"}\n";power();HIP_CHECK(hipStreamDestroy(stream));
    std::ofstream file(path);file<<out.str();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
