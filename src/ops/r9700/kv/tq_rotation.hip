#include "tq_k4v3_device.h"
#include "tq_rotation.h"

#include <algorithm>
#include <limits>

namespace ninfer::ops::r9700::kv {
namespace {

inline constexpr std::uint32_t kWaves   = 4U; // vectors per block
inline constexpr std::uint32_t kThreads = kWaves * 32U;

template <class T>
struct RotationBatch {
    T* data[kTqRotationMaximumBatch];
    std::uint32_t vectors[kTqRotationMaximumBatch];
};

static_assert(sizeof(RotationBatch<float>) <= 4096U, "rotation batch must fit the kernarg segment");

__global__ __launch_bounds__(kThreads) void rotate_query_kernel(RotationBatch<hip_bfloat16> batch) {
    const std::uint32_t vector = blockIdx.x * kWaves + threadIdx.x / 32U;
    if (vector >= batch.vectors[blockIdx.y]) return; // wave-uniform
    const std::uint32_t lane = threadIdx.x & 31U;
    auto* element      = reinterpret_cast<uint4*>(batch.data[blockIdx.y] +
                                                  static_cast<std::size_t>(vector) * tq::kHeadDim) +
                         lane;
    const uint4 packed = *element;
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    float x[8];
#pragma unroll
    for (std::uint32_t i = 0U; i < 8U; ++i) {
        x[i] = __uint_as_float((i & 1U) != 0U ? words[i / 2U] & 0xffff0000U : words[i / 2U] << 16U);
    }
    tq::rotate_wave(x, lane);
    std::uint32_t out[4];
#pragma unroll
    for (std::uint32_t i = 0U; i < 8U; i += 2U) {
        out[i / 2U] = static_cast<std::uint32_t>(hip_bfloat16(x[i]).data) |
                      (static_cast<std::uint32_t>(hip_bfloat16(x[i + 1U]).data) << 16U);
    }
    *element = uint4{out[0], out[1], out[2], out[3]};
}

__global__ __launch_bounds__(kThreads) void unrotate_output_kernel(RotationBatch<float> batch) {
    const std::uint32_t vector = blockIdx.x * kWaves + threadIdx.x / 32U;
    if (vector >= batch.vectors[blockIdx.y]) return; // wave-uniform
    const std::uint32_t lane = threadIdx.x & 31U;
    auto* element    = reinterpret_cast<float4*>(batch.data[blockIdx.y] +
                                                 static_cast<std::size_t>(vector) * tq::kHeadDim) +
                       2U * lane;
    const float4 low = element[0], high = element[1];
    float x[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
    tq::unrotate_wave(x, lane);
    element[0] = float4{x[0], x[1], x[2], x[3]};
    element[1] = float4{x[4], x[5], x[6], x[7]};
}

template <class T, class Kernel>
hipError_t launch(Kernel kernel, std::span<T* const> data, std::span<const std::size_t> vectors,
                  std::size_t alignment, hipStream_t stream) noexcept {
    if (data.empty() || data.size() > kTqRotationMaximumBatch || data.size() != vectors.size()) {
        return hipErrorInvalidValue;
    }
    RotationBatch<T> batch{};
    std::size_t widest = 0U;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] == nullptr || vectors[i] == 0U ||
            vectors[i] > std::numeric_limits<std::uint32_t>::max() - kWaves ||
            reinterpret_cast<std::uintptr_t>(data[i]) % alignment != 0U) {
            return hipErrorInvalidValue;
        }
        batch.data[i]    = data[i];
        batch.vectors[i] = static_cast<std::uint32_t>(vectors[i]);
        widest           = std::max(widest, vectors[i]);
    }
    hipLaunchKernelGGL(kernel,
                       dim3(static_cast<unsigned int>((widest + kWaves - 1U) / kWaves),
                            static_cast<unsigned int>(data.size())),
                       dim3(kThreads), 0, stream, batch);
    return hipGetLastError();
}

} // namespace

hipError_t tq_rotate_query(std::span<hip_bfloat16* const> queries,
                           std::span<const std::size_t> vectors, hipStream_t stream) noexcept {
    return launch(rotate_query_kernel, queries, vectors, 16U, stream);
}

hipError_t tq_unrotate_output(std::span<float* const> outputs, std::span<const std::size_t> vectors,
                              hipStream_t stream) noexcept {
    return launch(unrotate_output_kernel, outputs, vectors, 16U, stream);
}

hipError_t tq_rotate_query(hip_bfloat16* query, std::size_t vectors, hipStream_t stream) noexcept {
    return tq_rotate_query(std::span<hip_bfloat16* const>(&query, 1U),
                           std::span<const std::size_t>(&vectors, 1U), stream);
}

hipError_t tq_unrotate_output(float* output, std::size_t vectors, hipStream_t stream) noexcept {
    return tq_unrotate_output(std::span<float* const>(&output, 1U),
                              std::span<const std::size_t>(&vectors, 1U), stream);
}

} // namespace ninfer::ops::r9700::kv
