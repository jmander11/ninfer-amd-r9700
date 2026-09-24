#include "fp8_int4_kv_trace.h"

#include <hip/hip_runtime.h>

#include <limits>

namespace ninfer::ops::r9700::kv {
namespace {

constexpr std::size_t kPositions = 130;
constexpr std::size_t kPage = 64;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kDim = 256;
constexpr std::size_t kGroups = 16;
constexpr std::size_t kKeyBytes = kPositions * kHeads * kDim;
constexpr std::size_t kValueBytes = kPositions * kHeads * (kDim / 2);
constexpr std::size_t kScaleElements = kPositions * kHeads * kGroups;

__device__ std::size_t physical_position(std::size_t logical,
                                         const std::uint32_t* page_table) {
    return static_cast<std::size_t>(page_table[logical / kPage]) * kPage + logical % kPage;
}

__global__ void gather_kernel(Fp8Int4KvTraceArgs args) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint32_t* table = args.page_table;
    if (args.page_table_row != nullptr) {
        const std::int32_t row = *args.page_table_row;
        if (row < 0 || static_cast<std::uint32_t>(row) >= args.page_table_row_count) {
            if (index == 0) atomicOr(args.status, 1U);
            return;
        }
        table += static_cast<std::size_t>(row) * args.page_table_row_stride;
    }
    if (index < kKeyBytes) {
        const std::size_t feature = index % kDim;
        const std::size_t head = (index / kDim) % kHeads;
        const std::size_t logical = index / (kHeads * kDim);
        const std::size_t physical = physical_position(logical, table);
        if (physical >= args.physical_tokens) {
            atomicOr(args.status, 2U);
            return;
        }
        const std::size_t page = physical / kPage;
        const std::size_t in_page = physical % kPage;
        const std::size_t pages = args.physical_tokens / kPage;
        const std::size_t source = in_page + kPage * (feature + kDim * (page + pages * head));
        args.canonical_keys[index] = args.key_codes[source];
    }
    if (index < kValueBytes) {
        constexpr std::size_t leading = kDim / 2;
        const std::size_t feature_byte = index % leading;
        const std::size_t head = (index / leading) % kHeads;
        const std::size_t logical = index / (kHeads * leading);
        const std::size_t physical = physical_position(logical, table);
        if (physical >= args.physical_tokens) {
            atomicOr(args.status, 2U);
            return;
        }
        const std::size_t page = physical / kPage;
        const std::size_t in_page = physical % kPage;
        const std::size_t source = feature_byte + leading *
            (in_page + kPage * (head + kHeads * page));
        args.canonical_values[index] = args.value_codes[source];
    }
    if (index < kScaleElements) {
        const std::size_t group = index % kGroups;
        const std::size_t head = (index / kGroups) % kHeads;
        const std::size_t logical = index / (kHeads * kGroups);
        const std::size_t physical = physical_position(logical, table);
        if (physical >= args.physical_tokens) {
            atomicOr(args.status, 2U);
            return;
        }
        const std::size_t page = physical / kPage;
        const std::size_t in_page = physical % kPage;
        const std::size_t source = group + kGroups *
            (in_page + kPage * (head + kHeads * page));
        args.canonical_scales[index] = args.value_scale_bits[source];
    }
}

} // namespace

hipError_t fp8_int4_kv_trace_positions0_129(const Fp8Int4KvTraceArgs& args,
                                             hipStream_t stream) noexcept {
    if (args.key_codes == nullptr || args.value_codes == nullptr ||
        args.value_scale_bits == nullptr || args.page_table == nullptr ||
        args.canonical_keys == nullptr || args.canonical_values == nullptr ||
        args.canonical_scales == nullptr || args.status == nullptr || stream == nullptr ||
        args.page_table_pages < 3 || args.physical_tokens == 0 ||
        args.physical_tokens % kPage != 0 ||
        args.page_table_pages > args.physical_tokens / kPage ||
        (args.page_table_row != nullptr &&
         (args.page_table_row_stride < args.page_table_pages ||
          args.page_table_row_count == 0))) {
        return hipErrorInvalidValue;
    }
    constexpr unsigned kThreads = 256;
    constexpr unsigned kBlocks = static_cast<unsigned>((kKeyBytes + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(gather_kernel, dim3(kBlocks), dim3(kThreads), 0, stream, args);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::kv
