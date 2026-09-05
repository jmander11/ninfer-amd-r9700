#include "fp8_int4_kv_compact.h"

namespace ninfer::ops::r9700::kv {
namespace {
inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kPageSize = 64;
[[nodiscard]] constexpr bool valid_plane_layout(Fp8Int4KvPlaneLayout layout) noexcept {
    return layout == Fp8Int4KvPlaneLayout::FeatureFastest ||
           layout == Fp8Int4KvPlaneLayout::TokenFastest;
}

template <std::uint32_t Group, bool KToken, bool VToken, bool SToken>
struct Offset {
    static constexpr std::uint32_t kGroups = kHeadDim / Group;
    static constexpr std::uint32_t kBytes = Group / 2;
    __device__ static std::size_t at(std::size_t token, std::uint32_t head,
                                     std::uint32_t feature, std::uint32_t leading,
                                     std::uint32_t heads, std::size_t physical_tokens,
                                     bool token_fastest) {
        const std::size_t page = token / kPageSize, in_page = token % kPageSize;
        const std::size_t pages = physical_tokens / kPageSize;
        return token_fastest
                   ? in_page + kPageSize * (static_cast<std::size_t>(feature) + leading * (page + pages * head))
                   : feature + leading * (in_page + kPageSize * (static_cast<std::size_t>(head) + heads * page));
    }
    __device__ static std::size_t key(std::size_t t, std::uint32_t h, std::uint32_t f,
                                      std::uint32_t heads, std::size_t tokens) {
        return at(t, h, f, kHeadDim, heads, tokens, KToken);
    }
    __device__ static std::size_t value(std::size_t t, std::uint32_t h, std::uint32_t group,
                                        std::uint32_t byte, std::uint32_t heads, std::size_t tokens) {
        return at(t, h, group * kBytes + byte, kGroups * kBytes, heads, tokens, VToken);
    }
    __device__ static std::size_t scale(std::size_t t, std::uint32_t h, std::uint32_t group,
                                        std::uint32_t heads, std::size_t tokens) {
        return at(t, h, group, kGroups, heads, tokens, SToken);
    }
};

template <std::uint32_t Group, bool KToken, bool VToken, bool SToken>
__global__ void compact_kernel(std::size_t prefix, const std::uint32_t* selected_path,
                               std::size_t selected_count, const std::uint32_t* page_table,
                               std::uint32_t table_pages, std::size_t physical_tokens,
                               std::uint8_t* keys, std::uint8_t* values, std::uint16_t* scales,
                               std::uint32_t* status) {
    const std::uint32_t head = blockIdx.x, index = threadIdx.x;
    for (std::size_t selected = 0; selected < selected_count; ++selected) {
        const std::size_t source_logical = prefix + selected_path[selected];
        const std::size_t destination_logical = prefix + selected;
        if (source_logical / kPageSize >= table_pages || destination_logical / kPageSize >= table_pages) {
            if (index == 0) atomicOr(status, Fp8Int4KvAppendInvalidAddress);
            return;
        }
        const std::size_t source = static_cast<std::size_t>(page_table[source_logical / kPageSize]) * kPageSize + source_logical % kPageSize;
        const std::size_t destination = static_cast<std::size_t>(page_table[destination_logical / kPageSize]) * kPageSize + destination_logical % kPageSize;
        if (source >= physical_tokens || destination >= physical_tokens) {
            if (index == 0) atomicOr(status, Fp8Int4KvAppendInvalidAddress);
            return;
        }
        keys[Offset<Group,KToken,VToken,SToken>::key(destination,head,index,gridDim.x,physical_tokens)] =
            keys[Offset<Group,KToken,VToken,SToken>::key(source,head,index,gridDim.x,physical_tokens)];
        if (index < kHeadDim / 2) {
            const std::uint32_t group = index / Offset<Group,KToken,VToken,SToken>::kBytes;
            const std::uint32_t byte = index % Offset<Group,KToken,VToken,SToken>::kBytes;
            values[Offset<Group,KToken,VToken,SToken>::value(destination,head,group,byte,gridDim.x,physical_tokens)] =
                values[Offset<Group,KToken,VToken,SToken>::value(source,head,group,byte,gridDim.x,physical_tokens)];
        }
        if (index < Offset<Group,KToken,VToken,SToken>::kGroups) {
            scales[Offset<Group,KToken,VToken,SToken>::scale(destination,head,index,gridDim.x,physical_tokens)] =
                scales[Offset<Group,KToken,VToken,SToken>::scale(source,head,index,gridDim.x,physical_tokens)];
        }
        __syncthreads();
    }
}

template <std::uint32_t G, bool K, bool V, bool S>
hipError_t launch(const Fp8Int4KvCompactArgs& a, hipStream_t stream) {
    hipLaunchKernelGGL((compact_kernel<G,K,V,S>), dim3(a.kv_heads), dim3(kHeadDim), 0, stream,
                       a.prefix,a.selected_path,a.selected_count,a.page_table,a.page_table_pages,
                       a.physical_tokens,a.key_codes,a.value_codes,a.value_scale_bits,a.status);
    return hipGetLastError();
}
template <std::uint32_t G> hipError_t dispatch(const Fp8Int4KvCompactArgs& a, hipStream_t s) {
    const unsigned m=(a.key_layout==Fp8Int4KvPlaneLayout::TokenFastest?1U:0U)|(a.value_layout==Fp8Int4KvPlaneLayout::TokenFastest?2U:0U)|(a.value_scale_layout==Fp8Int4KvPlaneLayout::TokenFastest?4U:0U);
    switch(m) { case 0:return launch<G,false,false,false>(a,s); case 1:return launch<G,true,false,false>(a,s); case 2:return launch<G,false,true,false>(a,s); case 3:return launch<G,true,true,false>(a,s); case 4:return launch<G,false,false,true>(a,s); case 5:return launch<G,true,false,true>(a,s); case 6:return launch<G,false,true,true>(a,s); default:return launch<G,true,true,true>(a,s); }
}
} // namespace
hipError_t fp8_int4_kv_compact(const Fp8Int4KvCompactArgs& a, hipStream_t stream) noexcept {
    if (!a.selected_path || !a.selected_count || !a.page_table || !a.page_table_pages || !a.physical_tokens || a.physical_tokens%kPageSize || !a.kv_heads || !a.key_codes || !a.value_codes || !a.value_scale_bits || !a.status || !valid_plane_layout(a.key_layout) || !valid_plane_layout(a.value_layout) || !valid_plane_layout(a.value_scale_layout)) return hipErrorInvalidValue;
    return a.value_group==16 ? dispatch<16>(a,stream) : a.value_group==32 ? dispatch<32>(a,stream) : hipErrorInvalidValue;
}
} // namespace ninfer::ops::r9700::kv
