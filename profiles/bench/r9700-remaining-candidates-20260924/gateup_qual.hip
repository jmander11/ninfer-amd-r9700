// Focused invocation of the canonical complete public Linear qualification cells.
#define NINFER_A8Q4_VERIFY_QUAL_NO_MAIN 1
#include "a8q4_verify_qual_impl.h"
int main(int argc,char** argv) {
 try {
  if(argc!=2)fail("usage: gateup_qual FRESH.json");require_fresh_output(argv[1]);power();
  HIP_CHECK(hipSetDevice(0));hipDeviceProp_t p{};HIP_CHECK(hipGetDeviceProperties(&p,0));
  char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
  if(std::string_view(p.gcnArchName)!="gfx1201" || p.warpSize!=32 || std::string_view(pci)!="0000:13:00.0")fail("wrong GPU");
  hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
  N=34816;K=5120;G=80;std::ostringstream out;out<<std::setprecision(17)<<"{\"cells\":[";
  bool first=true;for(unsigned t:{15U,18U,20U,24U}){if(!first)out<<',';first=false;cell(t,stream,out);std::cout<<"T"<<t<<" qualified\n"<<std::flush;}
  out<<"],\"power\":\"auto\"}\n";power();std::ofstream file(argv[1]);file<<out.str();
  HIP_CHECK(hipStreamDestroy(stream));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
