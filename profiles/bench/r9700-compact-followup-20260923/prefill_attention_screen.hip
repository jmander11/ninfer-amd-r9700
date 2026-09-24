// Reuse the independent complete-attention oracle; time the public Op afterwards.
#define main retained_dense_qualification_entry
#include "dense_prefill_attention_qual.hip"
#undef main
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

void require_auto() {
    const std::filesystem::path pci="/sys/bus/pci/devices/0000:13:00.0";
    for(const auto& [name,expected]:std::array<std::pair<const char*,const char*>,3>{{
        {"vendor","0x1002"},{"device","0x7551"},{"power_dpm_force_performance_level","auto"}}}) {
        std::ifstream f(pci/name);std::string value;f>>value;if(value!=expected)fail("R9700 auto required");
    }
}
void measure(unsigned context,std::ostream& out) {
    constexpr unsigned rows=2048,group=16;
    run_case(rows,group,context);
    const Fixture fixture=make_fixture(group,context);const auto queries=make_queries(rows);
    auto positions=make_positions(rows);
    for(unsigned row=0;row<rows;++row)positions[row]=context-rows+row;
    DeviceBuffer<hip_bfloat16> q(queries.size());DeviceBuffer<std::uint8_t> k(fixture.keys.size()),v(fixture.values.size());
    DeviceBuffer<std::uint16_t> scales(fixture.scales.size());DeviceBuffer<std::uint32_t> pages(fixture.pages.size());
    DeviceBuffer<std::int32_t> pos(positions.size());DeviceBuffer<float> y(rows*kHq*kD);
    DeviceBuffer<std::uint8_t> workspace(kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(rows,context));
    DeviceBuffer<std::uint8_t> scrub(80U<<20U);
    HIP_CHECK(hipMemcpy(q.get(),queries.data(),q.bytes(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k.get(),fixture.keys.data(),k.bytes(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v.get(),fixture.values.data(),v.bytes(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(scales.get(),fixture.scales.data(),scales.bytes(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(pages.get(),fixture.pages.data(),pages.bytes(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(pos.get(),positions.data(),pos.bytes(),hipMemcpyHostToDevice));
    const kv::Fp8Int4KvAttentionArgs args{.query=q.get(),.key_codes=k.get(),.value_codes=v.get(),
        .value_scale_bits=scales.get(),.page_table=pages.get(),.page_table_pages=static_cast<unsigned>(fixture.pages.size()),
        .context=context,.physical_tokens=fixture.physical_tokens,.query_heads=kHq,.kv_heads=kHkv,
        .q_to_kv=6,.value_group=group,.query_rows=rows,.row_positions=pos.get(),.attention_scale=kScale,
        .key_layout=kv::Fp8Int4KvPlaneLayout::TokenFastest,.value_layout=kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .value_scale_layout=kv::Fp8Int4KvPlaneLayout::FeatureFastest,.output=y.get()};
    Event begin,end;out<<"{\"query_rows\":2048,\"context\":"<<context<<",\"milliseconds\":[";
    for(unsigned iteration=0;iteration<27;++iteration) {
        HIP_CHECK(hipMemset(scrub.get(),0x5a,scrub.bytes()));
        HIP_CHECK(hipEventRecord(begin.get()));
        HIP_CHECK(launch_selected_dense(args,workspace.get(),workspace.bytes(),nullptr));
        HIP_CHECK(hipEventRecord(end.get()));HIP_CHECK(hipEventSynchronize(end.get()));
        float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,begin.get(),end.get()));
        if(iteration>=3){if(iteration>3)out<<',';out<<ms;}
    }
    out<<"]}";
}
int main(int argc,char** argv) {
 try {
    if(argc!=2)fail("usage: prefill_attention_screen FRESH.json");
    if(std::filesystem::exists(argv[1]))fail("output exists");require_auto();
    HIP_CHECK(hipSetDevice(0));hipDeviceProp_t prop{};HIP_CHECK(hipGetDeviceProperties(&prop,0));
    char pci[32]{};HIP_CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::string_view(prop.gcnArchName)!="gfx1201" || prop.warpSize!=32 ||
       (std::string_view(pci)!="0000:13:00.0" && std::string_view(pci)!="13:00.0"))fail("wrong GPU");
    std::ostringstream out;out<<std::setprecision(17)<<"{\"power\":\"auto\",\"cold_scrub_bytes\":83886080,\"cells\":[";
    measure(2048,out);out<<',';measure(4096,out);out<<"]}\n";require_auto();
    std::ofstream file(argv[1]);file<<out.str();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
