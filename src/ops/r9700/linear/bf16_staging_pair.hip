#include "bf16_staging_pair.h"

namespace ninfer::ops::r9700::linear {
namespace {
__global__ __launch_bounds__(256) void bf16_staging_pair_kernel(
    const hip_bfloat16* input,const hip_bfloat16* weights,hip_bfloat16* output,
    unsigned columns,unsigned rows,unsigned tokens) {
    const unsigned row=blockIdx.x,wave=threadIdx.x/32,lane=threadIdx.x&31U;
    __shared__ float tile[512];
    float partial=0.0F;
    for(unsigned base=0;base<columns;base+=512) {
        std::uint32_t pair=0;
        // memcpy preserves the public two-byte BF16 alignment contract.
        __builtin_memcpy(&pair,weights+static_cast<std::size_t>(row)*columns+base+2*threadIdx.x,4);
        hip_bfloat16 low,high;low.data=static_cast<std::uint16_t>(pair);high.data=static_cast<std::uint16_t>(pair>>16);
        tile[2*threadIdx.x]=static_cast<float>(low);
        tile[2*threadIdx.x+1]=static_cast<float>(high);
        __syncthreads();
        if(wave<tokens) {
            for(unsigned local=lane;local<512;local+=32)
                partial=fmaf(static_cast<float>(input[static_cast<std::size_t>(wave)*columns+base+local]),tile[local],partial);
        }
        // Inactive token waves still participate in both staging barriers.
        __syncthreads();
    }
    for(unsigned delta=16;delta!=0;delta>>=1)partial+=__shfl_down(partial,delta,32);
    if(wave<tokens && lane==0)output[static_cast<std::size_t>(wave)*rows+row]=hip_bfloat16(partial);
}
}
hipError_t detail::launch_bf16_staging_pair(const Bf16LinearArgs& a,hipStream_t stream) noexcept {
    hipLaunchKernelGGL(bf16_staging_pair_kernel,dim3(a.rows),dim3(256),0,stream,a.input,a.weights,a.output,a.columns,a.rows,a.tokens);
    return hipGetLastError();
}
}
