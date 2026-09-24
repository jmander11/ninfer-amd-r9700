#include "ops/r9700/dflash/kv_cache_append_prefix_launch.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::dflash {
namespace {

constexpr int kHeadDim = 128;
constexpr int kKvHeads = 8;
constexpr int kPage = 64;
constexpr int kBlock = 256;
constexpr int kBf16PerUnit = 16;
constexpr int kUnitsPerHead = kHeadDim / kBf16PerUnit;
constexpr int kUnitsPerToken = kKvHeads * kUnitsPerHead;
constexpr int kTokensPerBlock = kBlock / kUnitsPerToken;
static_assert(kTokensPerBlock == 4);

__device__ __forceinline__ void copy_32_bytes(const std::uint16_t* source,
                                               std::uint16_t* destination) {
    const uint4 first = *reinterpret_cast<const uint4*>(source);
    const uint4 second = *reinterpret_cast<const uint4*>(source + 8);
    *reinterpret_cast<uint4*>(destination) = first;
    *reinterpret_cast<uint4*>(destination + 8) = second;
}

__global__ __launch_bounds__(kBlock) void append_cyclic_kernel(
    const std::uint16_t* __restrict__ key, const std::uint16_t* __restrict__ value,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, std::uint16_t* __restrict__ cache_key,
    std::uint16_t* __restrict__ cache_value, int min_count, int max_count, int width,
    int lane_capacity) {
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) { return; }

    const int local = static_cast<int>(threadIdx.x);
    const int local_token = local / kUnitsPerToken;
    const int unit = local - local_token * kUnitsPerToken;
    const int token = static_cast<int>(blockIdx.x) * kTokensPerBlock + local_token;
    if (token >= count) { return; }

    constexpr std::int64_t elements_per_token = kHeadDim * kKvHeads;
    const std::int64_t batch_input = elements_per_token * width * batch;
    const std::int64_t batch_position = static_cast<std::int64_t>(width) * batch;
    const int wave_lane = local & 31;
    int lane = 0;
    int position = 0;
    if (wave_lane == 0) {
        lane = lanes[batch];
        position = positions[batch_position + token];
    }
    lane = __shfl(lane, 0, 32);
    position = __shfl(position, 0, 32);
    if (lane < 0 || lane >= lane_capacity) { return; }
    if (position < 0) { return; }
    const int slot = position & 2047;
    const int head = unit / kUnitsPerHead;
    const int feature = (unit - head * kUnitsPerHead) * kBf16PerUnit;
    const std::int64_t source = batch_input + feature +
                                static_cast<std::int64_t>(kHeadDim) *
                                    (head + kKvHeads * token);
    const std::int64_t destination = feature + static_cast<std::int64_t>(kHeadDim) *
        (slot + 2048LL * (head + kKvHeads * lane));
    copy_32_bytes(key + source, cache_key + destination);
    copy_32_bytes(value + source, cache_value + destination);
}

__global__ __launch_bounds__(kBlock) void append_paged_kernel(
    const std::uint16_t* __restrict__ key, const std::uint16_t* __restrict__ value,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ table_rows, std::uint16_t* __restrict__ cache_key,
    std::uint16_t* __restrict__ cache_value, const std::int32_t* __restrict__ block_tables,
    int physical_pages, int logical_pages, int table_row_count, int min_count, int max_count,
    int width) {
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) { return; }

    const int local = static_cast<int>(threadIdx.x);
    const int local_token = local / kUnitsPerToken;
    const int unit = local - local_token * kUnitsPerToken;
    const int token = static_cast<int>(blockIdx.x) * kTokensPerBlock + local_token;
    if (token >= count) { return; }

    constexpr std::int64_t elements_per_token = kHeadDim * kKvHeads;
    const std::int64_t batch_input = elements_per_token * width * batch;
    const std::int64_t batch_position = static_cast<std::int64_t>(width) * batch;
    const int wave_lane = local & 31;
    int position = 0;
    int physical_page = -1;
    if (wave_lane == 0) {
        const int row = table_rows[batch];
        position = positions[batch_position + token];
        if (row >= 0 && row < table_row_count && position >= 0 &&
            position / kPage < logical_pages) {
            physical_page = block_tables[static_cast<std::int64_t>(logical_pages) * row +
                                         position / kPage];
        }
    }
    position = __shfl(position, 0, 32);
    physical_page = __shfl(physical_page, 0, 32);
    if (physical_page < 0 || physical_page >= physical_pages) { return; }
    const int head = unit / kUnitsPerHead;
    const int feature = (unit - head * kUnitsPerHead) * kBf16PerUnit;
    const std::int64_t source = batch_input + feature +
                                static_cast<std::int64_t>(kHeadDim) *
                                    (head + kKvHeads * token);
    const std::int64_t destination = feature + static_cast<std::int64_t>(kHeadDim) *
        ((position & (kPage - 1)) + kPage *
             (physical_page + static_cast<std::int64_t>(physical_pages) * head));
    copy_32_bytes(key + source, cache_key + destination);
    copy_32_bytes(value + source, cache_value + destination);
}

} // namespace

void kv_cache_append_prefix_paged_launch(
    const Tensor& key, const Tensor& value, const Tensor& positions, const Tensor& counts,
    const Tensor& table_rows, const KVCacheAppendPrefixPagedView& cache,
    KVCacheAppendPrefixExecutionEnvelope envelope, hipStream_t stream) {
    if (envelope.max_count == 0U) { return; }
    const dim3 grid((envelope.max_count + kTokensPerBlock - 1U) / kTokensPerBlock,
                    static_cast<unsigned int>(key.ne[3]));
    append_paged_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::uint16_t*>(key.data),
        static_cast<const std::uint16_t*>(value.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<std::uint16_t*>(cache.key_pages.data),
        static_cast<std::uint16_t*>(cache.value_pages.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.key_pages.ne[2],
        cache.block_tables.ne[0], cache.block_tables.ne[1],
        static_cast<int>(envelope.min_count), static_cast<int>(envelope.max_count), key.ne[2]);
    HIP_CHECK(hipGetLastError());
}

void kv_cache_append_prefix_cyclic_launch(
    const Tensor& key, const Tensor& value, const Tensor& positions, const Tensor& counts,
    const Tensor& lanes, const CyclicKVCacheLayerView& cache,
    KVCacheAppendPrefixExecutionEnvelope envelope, hipStream_t stream) {
    if (envelope.max_count == 0U) { return; }
    const dim3 grid((envelope.max_count + kTokensPerBlock - 1U) / kTokensPerBlock,
                    static_cast<unsigned int>(key.ne[3]));
    append_cyclic_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::uint16_t*>(key.data),
        static_cast<const std::uint16_t*>(value.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<std::uint16_t*>(cache.k.data), static_cast<std::uint16_t*>(cache.v.data),
        static_cast<int>(envelope.min_count), static_cast<int>(envelope.max_count), key.ne[2],
        cache.lane_capacity);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::r9700::dflash
