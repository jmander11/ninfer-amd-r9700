// Diagnostic decomposition, not a replacement oracle or admission criterion.
#define NINFER_QUAL_SMALL_BATCH_PROJECTIONS 1
#define NINFER_A8Q4_VERIFY_QUAL_NO_MAIN 1
#include "a8q4_verify_qual_impl.h"

double relative(const std::vector<double>& a,const std::vector<double>& b) {
    double e2=0,r2=0;
    for(std::size_t i=0;i<a.size();++i){const double e=a[i]-b[i];e2+=e*e;r2+=b[i]*b[i];}
    return std::sqrt(e2/r2);
}
int main(int argc,char** argv) {
 try {
    if(argc!=2)fail("usage: accuracy_diagnostic FRESH.json");
    const std::filesystem::path path=argv[1];require_fresh_output(path);power();
    HIP_CHECK(hipSetDevice(0));hipStream_t s{};
    HIP_CHECK(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
    N=4096;K=5120;G=K/64;constexpr unsigned T=6;
    const auto host=make_decode_dot8_weights(N,K);
    const auto base=make_decode_dot8_input(K);Weights weights(host,s);
    std::ostringstream out;out<<std::setprecision(17)<<"{\"shape\":[4096,5120,6],\"fixtures\":[";
    bool first=true;
    for(float multiplier:{1.0F,-0.75F}) {
        std::vector<hip_bfloat16> x(T*K);
        for(unsigned t=0;t<T;++t)for(unsigned k=0;k<K;++k) {
            const hip_bfloat16 initial(static_cast<float>(base[(k/64)*64+(k+t*13)%64])*(t+4)/8.0F);
            x[t*K+k]=hip_bfloat16(multiplier*static_cast<float>(initial));
        }
        const auto quantized=quantize_host(x,T,K);
        const auto qref=oracle(T,quantized,host);
        const auto reference=public_oracle(T,x,host);
        Guarded<hip_bfloat16> input(x.size(),s);input.put(x,s);Output selected(T,s),control(T,s);
        launch(arguments(T,input,weights,selected),s);
        const auto a=arguments(T,input,weights,control);const auto w=workspace(T,control);
        HIP_CHECK(linear::a8g64_quantize_activation({a.input,w},s));
        HIP_CHECK(linear::a8q4g64_linear_wmma32({w.low_codes,w.low_code_bytes,w.high_codes,
            w.high_code_bytes,w.scales,w.scale_bytes,w.status,a.weight_codes,a.weight_code_bytes,
            a.weight_scales,a.weight_scale_bytes,a.output,T,N,K,K},s));
        const auto gpu=selected.value.read(s);
        exact(gpu,control.value.read(s),"selected differs from generic");
        codec(T,selected,quantized,s);compare(gpu,qref);
        if(!first)out<<',';first=false;
        out<<"{\"multiplier\":"<<multiplier<<",\"exact_generic_codec\":true,\"tokens\":[";
        for(unsigned t=0;t<T;++t) {
            std::vector<double> actual(N),q(N),ref(N),qx(K),original(K),absolute_dot(N);
            for(unsigned k=0;k<K;++k) {
                const auto i=t*K+k;
                qx[k]=(unsigned_nibble(quantized.low,i)+16*signed_nibble(quantized.high,i))*
                       static_cast<double>(half_value(quantized.scales[t*G+k/64]));
                original[k]=static_cast<float>(x[i]);
            }
            for(unsigned r=0;r<N;++r) {
                actual[r]=static_cast<float>(gpu[t*N+r]);q[r]=qref[t*N+r];ref[r]=reference.values[t*N+r];
                for(unsigned k=0;k<K;++k) {
                    const auto g=k/64,l=k%64;
                    const std::size_t word=(r/16)*G*64+g*64+(l/16)*16+r%16;
                    const int nibble=(host.codes[word*8+(l%16)/2]>>((l%2)*4))&15;
                    absolute_dot[r]+=std::abs(original[k]*(nibble>=8?nibble-16:nibble)*
                        static_cast<double>(half_value(host.scales[(r/16)*G*16+g*16+r%16])));
                }
            }
            double abs2=0,ref2=0;for(unsigned r=0;r<N;++r){abs2+=absolute_dot[r]*absolute_dot[r];ref2+=ref[r]*ref[r];}
            if(t)out<<',';
            out<<"{\"token\":"<<t<<",\"gpu_public_relative_l2\":"<<relative(actual,ref)
               <<",\"ideal_a8_public_relative_l2\":"<<relative(q,ref)
               <<",\"gpu_ideal_a8_relative_l2\":"<<relative(actual,q)
               <<",\"activation_relative_l2\":"<<relative(qx,original)
               <<",\"absolute_dot_over_output_l2\":"<<std::sqrt(abs2/ref2)<<'}';
        }
        out<<"]}";
    }
    out<<"]}\n";power();HIP_CHECK(hipStreamDestroy(s));
    std::ofstream file(path);file<<out.str();std::cout<<out.str();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
