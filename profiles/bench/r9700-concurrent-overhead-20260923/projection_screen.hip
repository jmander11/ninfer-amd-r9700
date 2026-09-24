// Temporary complete-Op bound/A-B screen; canonical numerical oracle owns math.
#define NINFER_A8Q4_VERIFY_QUAL_NO_MAIN 1
#include "a8q4_verify_qual_impl.h"

int main(int argc,char** argv) {
 try {
    const bool timing_only=argc==4 && std::string_view(argv[3])=="--timing-only";
    if((argc!=3 && !timing_only) || std::string_view(argv[2])!="--concurrent-projections")
        fail("usage: screen FRESH.json --concurrent-projections [--timing-only]");
    const std::filesystem::path path=argv[1];require_fresh_output(path);power();
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t props{};HIP_CHECK(hipGetDeviceProperties(&props,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(props.gcnArchName)!="gfx1201" || props.warpSize!=32 ||
       (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong GPU");
    hipStream_t stream{};HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    DeviceBuffer<std::uint8_t> scrub(80U<<20U);
    std::ostringstream out;out<<std::setprecision(17)<<"{\"cells\":[";
    bool first=true;
    const std::array<std::array<unsigned,2>,8> shapes{{{4096,5120},{12288,5120},{5120,6144},{7168,5120},{6144,5120},{1280,5120},{5120,4096},{5120,25600}}};
    for(const auto shape:shapes) {
      N=shape[0];K=shape[1];G=K/64;
      const auto w=make_decode_dot8_weights(N,K);const auto base=make_decode_dot8_input(K);
      const std::array<unsigned,6> widths{10,12,15,18,20,24};
      for(unsigned t:widths) {
        std::vector<hip_bfloat16> x(t*K);
        for(unsigned token=0;token<t;++token)for(unsigned k=0;k<K;++k)
          x[token*K+k]=hip_bfloat16(static_cast<float>(base[(k/64)*64+(k+token*13)%64])*(token+4)/8.0F);
        Guarded<hip_bfloat16> input(x.size(),stream);input.put(x,stream);
        Output selected(t,stream),control(t,stream);
        std::array<std::unique_ptr<Weights>,Copies> weights;
        for(auto& weight:weights)weight=std::make_unique<Weights>(w,stream);
        auto raw=[&](unsigned i) {
          auto a=arguments(t,input,*weights[i],control);auto ws=workspace(t,control);
          HIP_CHECK(linear::a8g64_quantize_activation({a.input,ws},stream));
          HIP_CHECK(linear::a8q4g64_linear_wmma32({ws.low_codes,ws.low_code_bytes,
            ws.high_codes,ws.high_code_bytes,ws.scales,ws.scale_bytes,ws.status,
            a.weight_codes,a.weight_code_bytes,a.weight_scales,a.weight_scale_bytes,
            a.output,t,N,K,K},stream));
        };
        const auto represented=quantize_host(x,t,K);
        // Timing-only is run only after the same linked core/fixtures passed the
        // canonical --concurrent-only suite. Retain exact generic/codec checks.
        std::array<std::unique_ptr<Graph>,Copies> controls,candidates;
        for(unsigned i=0;i<Copies;++i) {
          raw(i);launch(arguments(t,input,*weights[i],selected),stream);
          auto actual=selected.value.read(stream);
          if(!timing_only) {
            const auto budget=a8_error_budget(t,represented,w);
            const auto reference=public_oracle(t,x,w);
            compare(actual,budget.represented);compare_public(t,actual,reference,budget);
          }
          exact(actual,control.value.read(stream),"candidate/control differ");
          codec(t,selected,represented,stream);selected.value.guards(stream);selected.scratch.guards(stream);
          controls[i]=std::make_unique<Graph>(stream,[&]{raw(i);});
          candidates[i]=std::make_unique<Graph>(stream,[&]{launch(arguments(t,input,*weights[i],selected),stream);});
        }
        auto measure=[&](Graph& graph) {
          HIP_CHECK(hipMemsetAsync(scrub.get(),0x5a,scrub.bytes(),stream));
          Event start,end;HIP_CHECK(hipEventRecord(start.get(),stream));graph.run(stream);
          HIP_CHECK(hipEventRecord(end.get(),stream));HIP_CHECK(hipEventSynchronize(end.get()));
          float ms;HIP_CHECK(hipEventElapsedTime(&ms,start.get(),end.get()));return ms;
        };
        for(unsigned i=0;i<Copies;++i){measure(*controls[i]);measure(*candidates[i]);}
        if(!first)out<<',';first=false;
        out<<"{\"tokens\":"<<t<<",\"rows\":"<<N<<",\"columns\":"<<K
           <<",\"oracle_prequalified\":"<<(timing_only?"true":"false")
           <<",\"exact_control_and_codec\":true,\"samples\":[";
        for(unsigned j=0;j<24;++j) {
          float a,b;unsigned i=j%Copies;
          if(j%2){b=measure(*candidates[i]);a=measure(*controls[i]);}
          else {a=measure(*controls[i]);b=measure(*candidates[i]);}
          if(j)out<<',';out<<"{\"control_ms\":"<<a<<",\"selected_ms\":"<<b<<'}';
        }
        out<<"]}";std::cout<<N<<'/'<<K<<" T"<<t<<" qualified and timed\n"<<std::flush;
      }
    }
    out<<"],\"power\":\"auto\",\"scrub_bytes\":83886080,\"copies\":3}\n";power();
    std::ofstream file(path);file<<out.str();
    HIP_CHECK(hipStreamDestroy(stream));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
