// Host-only feasibility: fit an A8 scale to represented BF16 values, not outputs.
#define NINFER_QUAL_SMALL_BATCH_PROJECTIONS 1
#define NINFER_A8Q4_VERIFY_QUAL_NO_MAIN 1
#include "a8q4_verify_qual_impl.h"

int main() {
 try {
    N=4096;K=5120;G=K/64;constexpr unsigned T=6;
    const auto host=make_decode_dot8_weights(N,K);const auto base=make_decode_dot8_input(K);
    for(float multiplier:{1.0F,-0.75F}) {
        std::vector<hip_bfloat16> x(T*K);
        for(unsigned t=0;t<T;++t)for(unsigned k=0;k<K;++k) {
            const hip_bfloat16 initial(static_cast<float>(base[(k/64)*64+(k+t*13)%64])*(t+4)/8.0F);
            x[t*K+k]=hip_bfloat16(multiplier*static_cast<float>(initial));
        }
        auto q=quantize_host(x,T,K);
        for(unsigned t=0;t<T;++t)for(unsigned g=0;g<G;++g) {
            const auto si=t*G+g;
            double numerator=0,denominator=0;
            for(unsigned k=0;k<64;++k) {
                const auto i=t*K+g*64+k;
                const int code=unsigned_nibble(q.low,i)+16*signed_nibble(q.high,i);
                numerator+=static_cast<double>(static_cast<float>(x[i]))*code;
                denominator+=code*code;
            }
            const auto fitted=denominator==0?0:half_bits(numerator/denominator);
            if(!fitted)continue;
            const double old_scale=half_value(q.scales[si]),new_scale=half_value(fitted);
            double old_error=0,new_error=0;
            for(unsigned k=0;k<64;++k) {
                const auto i=t*K+g*64+k;const double value=static_cast<float>(x[i]);
                const int code=std::clamp(static_cast<int>(std::nearbyint(value/new_scale)),-127,127);
                const double old_delta=value-(unsigned_nibble(q.low,i)+16*signed_nibble(q.high,i))*old_scale;
                const double new_delta=value-code*new_scale;
                old_error+=old_delta*old_delta;new_error+=new_delta*new_delta;
            }
            if(new_error>=old_error)continue;
            q.scales[si]=fitted;
            for(unsigned k=0;k<64;++k) {
                const auto i=t*K+g*64+k;
                const int code=std::clamp(static_cast<int>(std::nearbyint(static_cast<float>(x[i])/new_scale)),-127,127);
                const int low=code&15;set_nibble(q.low,i,low);set_nibble(q.high,i,(code-low)/16);
            }
        }
        const auto result=oracle(T,q,host),ref=public_oracle(T,x,host).values;
        for(unsigned t=0;t<T;++t) {
            double e2=0,r2=0;
            for(unsigned r=0;r<N;++r){const double e=result[t*N+r]-ref[t*N+r];e2+=e*e;r2+=ref[t*N+r]*ref[t*N+r];}
            std::cout<<"multiplier="<<multiplier<<" token="<<t<<" fitted_a8_public_relative_l2="<<std::sqrt(e2/r2)<<'\n';
        }
    }
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
