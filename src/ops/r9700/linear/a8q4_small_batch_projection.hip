#include "a8q4_small_batch_projection.h"
#include "a8q4_pipeline_impl.h"
#include "a8q4_paired_tiles.h"

namespace ninfer::ops::r9700::linear {
namespace {
// Waves per CTA sharing one 16-row tile over contiguous K ranges. One wave per tile leaves too
// few G64 payloads in flight to saturate DRAM on the narrower projections.
template <unsigned N, unsigned K>
constexpr unsigned k_split() {
    if constexpr (N == 1280) return 16;
    else if constexpr (N == 5120 && (K == 4096 || K == 6144)) return 8;
    else if constexpr (N == 12288) return 1;
    else return 4;
}
template <unsigned N, unsigned K, unsigned T, bool Accumulate = false>
__global__ __launch_bounds__((32 * k_split<N, K>())) void small_batch_projection_kernel(
    const std::uint8_t* low, const std::uint8_t* high, const std::uint16_t* scales,
    const std::uint32_t* status, const std::uint8_t* codes,
    const std::uint16_t* weight_scales, hip_bfloat16* output) {
    if constexpr(N==12288 && T<=8)
        detail::a8q4_scale_gather_body<N,K,T>(low,high,scales,status,codes,weight_scales,output);
    else
        detail::a8q4_pipeline_body<N,K,T,Accumulate,k_split<N,K>()>(
            low,high,scales,status,codes,weight_scales,output,blockIdx.x);
}
template <unsigned N, unsigned K, unsigned T>
hipError_t launch_width(const A8Q4G64LinearArgs& a, hipStream_t stream) {
    if constexpr (N == 34816 && K == 5120 && (T == 18 || T == 20 || T == 24)) {
        if (a.accumulate_output) return hipErrorInvalidValue;
        hipLaunchKernelGGL((detail::gate_up_paired_tiles_kernel<T>),dim3(N/16),dim3(32),0,stream,
            a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,
            a.weight_codes,a.weight_scales,a.output);
        return hipGetLastError();
    } else {
    // Only the N5120 residual-stream projections admit the in-place residual epilogue.
    if constexpr (N == 5120) {
        if (a.accumulate_output) {
            hipLaunchKernelGGL((small_batch_projection_kernel<N,K,T,true>),dim3(N/16,(T+15)/16),
                dim3(32*k_split<N,K>()),0,stream,a.activation_low_codes,a.activation_high_codes,
                a.activation_scales,a.activation_status,a.weight_codes,a.weight_scales,a.output);
            return hipGetLastError();
        }
    }
    if (a.accumulate_output) return hipErrorInvalidValue;
    hipLaunchKernelGGL((small_batch_projection_kernel<N,K,T>),dim3(N/16,(T+15)/16),
        dim3(32*k_split<N,K>()),0,stream,
        a.activation_low_codes,a.activation_high_codes,a.activation_scales,a.activation_status,
        a.weight_codes,a.weight_scales,a.output);
    return hipGetLastError();
    }
}
// Two projections of one prepared activation in one launch: CTAs [0, N0/16) own the first
// matrix's row tiles and the rest the second's, so the pair pays one launch ramp and tail.
constexpr unsigned kPairSplit = 4;
template <unsigned N0, unsigned N1, unsigned K, unsigned T>
__global__ __launch_bounds__(32 * kPairSplit) void small_batch_projection_pair_kernel(
    const std::uint8_t* low, const std::uint8_t* high, const std::uint16_t* scales,
    const std::uint32_t* status, const std::uint8_t* codes0, const std::uint16_t* scales0,
    hip_bfloat16* output0, const std::uint8_t* codes1, const std::uint16_t* scales1,
    hip_bfloat16* output1) {
    if (blockIdx.x < N0 / 16)
        detail::a8q4_pipeline_body<N0,K,T,false,kPairSplit>(
            low,high,scales,status,codes0,scales0,output0,blockIdx.x);
    else
        detail::a8q4_pipeline_body<N1,K,T,false,kPairSplit>(
            low,high,scales,status,codes1,scales1,output1,blockIdx.x - N0 / 16);
}
template <unsigned N0, unsigned N1, unsigned K, unsigned T>
hipError_t launch_pair_width(const A8Q4G64LinearArgs& a, const A8Q4G64LinearArgs& b,
                             hipStream_t stream) {
    hipLaunchKernelGGL((small_batch_projection_pair_kernel<N0,N1,K,T>),
                       dim3((N0 + N1) / 16, (T + 15) / 16), dim3(32 * kPairSplit), 0, stream,
                       a.activation_low_codes, a.activation_high_codes, a.activation_scales,
                       a.activation_status, a.weight_codes, a.weight_scales, a.output,
                       b.weight_codes, b.weight_scales, b.output);
    return hipGetLastError();
}
template <unsigned N0, unsigned N1, unsigned K>
hipError_t launch_pair(const A8Q4G64LinearArgs& a, const A8Q4G64LinearArgs& b,
                       hipStream_t stream) {
    switch (a.tokens) {
    case 5: return launch_pair_width<N0,N1,K,5>(a,b,stream);
    case 6: return launch_pair_width<N0,N1,K,6>(a,b,stream);
    case 10: return launch_pair_width<N0,N1,K,10>(a,b,stream);
    case 12: return launch_pair_width<N0,N1,K,12>(a,b,stream);
    case 15: return launch_pair_width<N0,N1,K,15>(a,b,stream);
    case 18: return launch_pair_width<N0,N1,K,18>(a,b,stream);
    case 20: return launch_pair_width<N0,N1,K,20>(a,b,stream);
    case 24: return launch_pair_width<N0,N1,K,24>(a,b,stream);
    }
    return hipErrorNotSupported;
}
template <unsigned N, unsigned K>
hipError_t launch_prepared(const A8Q4G64LinearArgs& a, hipStream_t stream) {
    if constexpr ((N==34816 && K==5120) || (N==5120 && (K==17408 || K==6144))) {
        switch(a.tokens) {
        case 2: return launch_width<N,K,2>(a,stream);
        case 3: return launch_width<N,K,3>(a,stream);
        case 4: return launch_width<N,K,4>(a,stream);
        }
    }
    {
        switch(a.tokens) {
        case 10: return launch_width<N,K,10>(a,stream);
        case 12: return launch_width<N,K,12>(a,stream);
        case 15: return launch_width<N,K,15>(a,stream);
        case 18: return launch_width<N,K,18>(a,stream);
        case 20: return launch_width<N,K,20>(a,stream);
        case 24: return launch_width<N,K,24>(a,stream);
        }
    }
    if(a.tokens==5)return launch_width<N,K,5>(a,stream);
    if(a.tokens==6)return launch_width<N,K,6>(a,stream);
    return hipErrorInvalidValue;
}
template <unsigned N, unsigned K>
hipError_t launch(const A8Q4G64CandidateArgs& a, hipStream_t stream) {
    const auto validation = detail::validate_a8q4_scale_gather<N,K>(a,stream);
    if (validation != hipSuccess) return validation;
    A8G64ActivationWorkspace scratch{};
    auto result = a8q4g64_bind_activation_workspace(a.activation_workspace,
        a.activation_workspace_bytes,a.tokens,K,&scratch);
    if (result != hipSuccess) return result;
    result = a8g64_quantize_activation({a.input,scratch},stream);
    if (result != hipSuccess) return result;
    return launch_prepared<N,K>(
        {scratch.low_codes,scratch.low_code_bytes,scratch.high_codes,scratch.high_code_bytes,
         scratch.scales,scratch.scale_bytes,scratch.status,a.weight_codes,a.weight_code_bytes,
         a.weight_scales,a.weight_scale_bytes,a.output,a.tokens,a.rows,a.columns,a.padded_columns},stream);
}
}
bool detail::use_a8q4_small_batch_projection_pair(const A8Q4G64LinearArgs& a,
                                                  const A8Q4G64LinearArgs& b) noexcept {
    const bool same_activation = a.tokens == b.tokens && a.columns == 5120U &&
        b.columns == 5120U && a.padded_columns == 5120U && b.padded_columns == 5120U &&
        a.activation_low_codes == b.activation_low_codes &&
        a.activation_high_codes == b.activation_high_codes &&
        a.activation_scales == b.activation_scales && a.activation_status == b.activation_status &&
        !a.accumulate_output && !b.accumulate_output;
    const bool widths = a.tokens == 5U || a.tokens == 6U || a.tokens == 10U || a.tokens == 12U ||
        a.tokens == 15U || a.tokens == 18U || a.tokens == 20U || a.tokens == 24U;
    return same_activation && widths &&
        ((a.rows == 4096U && b.rows == 12288U) || (a.rows == 7168U && b.rows == 7168U));
}

hipError_t detail::launch_a8q4_small_batch_projection_pair(
    const A8Q4G64LinearArgs& a, const A8Q4G64LinearArgs& b, hipStream_t stream) noexcept {
    if (!use_a8q4_small_batch_projection_pair(a, b)) return hipErrorInvalidValue;
    if (a.rows == 4096U) return launch_pair<4096,12288,5120>(a,b,stream);
    return launch_pair<7168,7168,5120>(a,b,stream);
}

hipError_t detail::launch_a8q4_small_batch_projection(
    const A8Q4G64LinearArgs& a, hipStream_t stream) noexcept {
    if (a.rows == 34816) return launch_prepared<34816,5120>(a,stream);
    if (a.rows == 5120 && a.columns == 17408) return launch_prepared<5120,17408>(a,stream);
    if (a.rows == 5120 && a.columns == 25600) return launch_prepared<5120,25600>(a,stream);
    if (a.rows == 5120 && a.columns == 4096) return launch_prepared<5120,4096>(a,stream);
    if (a.rows == 5120) return launch_prepared<5120,6144>(a,stream);
    if (a.rows == 7168) return launch_prepared<7168,5120>(a,stream);
    if (a.rows == 6144) return launch_prepared<6144,5120>(a,stream);
    if (a.rows == 1280) return launch_prepared<1280,5120>(a,stream);
    if (a.rows == 12288) return launch_prepared<12288,5120>(a,stream);
    return launch_prepared<4096,5120>(a,stream);
}
hipError_t a8q4_small_batch_projection(
    const A8Q4G64CandidateArgs& a, hipStream_t stream) noexcept {
    if (!detail::use_a8q4_small_batch_projection(a.tokens,a.rows,a.columns,a.padded_columns))
        return hipErrorInvalidValue;
    if (a.rows == 34816 && a.columns == 5120) return launch<34816,5120>(a,stream);
    if (a.rows == 5120 && a.columns == 17408) return launch<5120,17408>(a,stream);
    if (a.rows == 5120 && a.columns == 25600) return launch<5120,25600>(a,stream);
    if (a.rows == 5120 && a.columns == 4096) return launch<5120,4096>(a,stream);
    if (a.rows == 5120 && a.columns == 6144) return launch<5120,6144>(a,stream);
    if (a.rows == 7168 && a.columns == 5120) return launch<7168,5120>(a,stream);
    if (a.rows == 6144 && a.columns == 5120) return launch<6144,5120>(a,stream);
    if (a.rows == 1280 && a.columns == 5120) return launch<1280,5120>(a,stream);
    if (a.rows == 12288 && a.columns == 5120) return launch<12288,5120>(a,stream);
    if (a.rows == 4096 && a.columns == 5120) return launch<4096,5120>(a,stream);
    return hipErrorInvalidValue;
}
}
