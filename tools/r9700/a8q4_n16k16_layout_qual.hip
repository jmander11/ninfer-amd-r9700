#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr std::uint32_t kWave = 32;
constexpr std::uint32_t kTile = 16;
constexpr std::uint32_t kGroup = 64;
constexpr std::uint32_t kM = 64;
constexpr std::uint32_t kN = 128;
constexpr std::uint32_t kThreads = 512;

using I32x2 = int __attribute__((ext_vector_type(2)));
using I32x8 = int __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));

struct Tile {
    std::uint32_t activation_low[8 * kM];
    std::uint32_t activation_high[8 * kM];
    std::uint32_t weights[8 * kN];
    std::uint16_t activation_scales[kM];
    std::uint16_t weight_scales[kN];
};
struct PingPong { Tile bank[2]; };
static_assert(sizeof(Tile) == 8576);
static_assert(sizeof(PingPong) == 17152);

__device__ __forceinline__ std::uint32_t lds_index(
    std::uint32_t row, std::uint32_t word, std::uint32_t rows) {
    return word * rows + (row ^ (((word >> 1U) & 1U) << 4U));
}

// Persistent code plane: [N/16][G][four K16 pairs][16 rows][packed-u64].
__host__ __device__ constexpr std::size_t code_pair_index(
    std::uint32_t row, std::uint32_t group, std::uint32_t pair,
    std::uint32_t groups) {
    return (((static_cast<std::size_t>(row >> 4U) * groups + group) * 4U + pair) *
            16U + (row & 15U));
}

// Persistent scale plane: [N/16][G][16 rows].
__host__ __device__ constexpr std::size_t scale_index(
    std::uint32_t row, std::uint32_t group, std::uint32_t groups) {
    return ((static_cast<std::size_t>(row >> 4U) * groups + group) * 16U +
            (row & 15U));
}

__device__ __forceinline__ void barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ std::uint32_t load32(const std::uint8_t* p,
                                                 std::size_t element) {
    return *reinterpret_cast<const std::uint32_t*>(p + element / 2U);
}

__device__ __forceinline__ std::uint64_t load64(const std::uint64_t* p,
                                                 std::size_t index) {
    return p[index];
}

__device__ __forceinline__ float half_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ hip_bfloat16 poison() {
    hip_bfloat16 value;
    value.data = 0x7fc1U;
    return value;
}

} // namespace

extern "C" __global__ __launch_bounds__(kThreads)
void a8q4_n16k16_prefill_p2048_m64n128(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* status,
    const std::uint64_t* weight_pairs, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t rows, std::uint32_t columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ PingPong tiles;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t wave_m = wave >> 2U;
    const std::uint32_t wave_n = wave & 3U;
    const std::uint32_t m0 = blockIdx.y * kM;
    const std::uint32_t n0 = blockIdx.x * kN;
    const std::uint32_t groups = columns / kGroup;
    const std::uint32_t arow = thread >> 3U;
    const std::uint32_t aword = thread & 7U;
    const std::uint32_t token = m0 + arow;
    const std::uint32_t a_lds = lds_index(arow, aword, kM);

    if (*status != 0U) {
#pragma unroll
        for (std::uint32_t nh = 0; nh < 2U; ++nh) {
#pragma unroll
            for (std::uint32_t e = 0; e < 8U; ++e) {
                const std::uint32_t m = m0 + wave_m * 16U + lane_group * 8U + e;
                const std::uint32_t n = n0 + nh * 64U + wave_n * 16U + axis;
                output[static_cast<std::size_t>(m) * rows + n] = poison();
            }
        }
        return;
    }

    const std::uint32_t load_tile = thread >> 6U;
    const std::uint32_t load_pair = (thread >> 4U) & 3U;
    const std::uint32_t load_lane = thread & 15U;
    const std::uint32_t load_row_local = load_tile * 16U + load_lane;
    const std::uint32_t load_row = n0 + load_row_local;
    const std::uint32_t load_word = load_pair * 2U;

    std::size_t ae = static_cast<std::size_t>(token) * columns + aword * 8U;
    tiles.bank[0].activation_low[a_lds] = load32(activation_low, ae);
    tiles.bank[0].activation_high[a_lds] = load32(activation_high, ae);
    const std::uint64_t w0 = load64(weight_pairs,
                                    code_pair_index(load_row, 0U, load_pair, groups));
    tiles.bank[0].weights[lds_index(load_row_local, load_word, kN)] =
        static_cast<std::uint32_t>(w0);
    tiles.bank[0].weights[lds_index(load_row_local, load_word + 1U, kN)] =
        static_cast<std::uint32_t>(w0 >> 32U);
    if (thread < kM) {
        tiles.bank[0].activation_scales[thread] =
            activation_scales[static_cast<std::size_t>(m0 + thread) * groups];
    }
    if (thread < kN) {
        const std::uint32_t row = n0 + thread;
        tiles.bank[0].weight_scales[thread] = weight_scales[scale_index(row, 0U, groups)];
    }
    barrier();

    F32x8 totals[2]{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        const bool has_next = group + 1U < groups;
        std::uint32_t next_a_low = 0, next_a_high = 0;
        std::uint64_t next_w = 0;
        if (has_next) {
            ae = static_cast<std::size_t>(token) * columns +
                 (group + 1U) * kGroup + aword * 8U;
            next_a_low = load32(activation_low, ae);
            next_a_high = load32(activation_high, ae);
            next_w = load64(weight_pairs,
                            code_pair_index(load_row, group + 1U, load_pair, groups));
        }

        Tile& current = tiles.bank[group & 1U];
        I32x8 low[2]{}, high[2]{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            I32x2 al{}, ah{}, b[2]{};
            const std::uint32_t word = half * 4U + lane_group * 2U;
            const std::uint32_t ar = wave_m * 16U + axis;
            const std::uint32_t ai = lds_index(ar, word, kM);
            al[0] = static_cast<int>(current.activation_low[ai]);
            al[1] = static_cast<int>(current.activation_low[lds_index(ar, word + 1U, kM)]);
            ah[0] = static_cast<int>(current.activation_high[ai]);
            ah[1] = static_cast<int>(current.activation_high[lds_index(ar, word + 1U, kM)]);
#pragma unroll
            for (std::uint32_t nh = 0; nh < 2U; ++nh) {
                const std::uint32_t wr = nh * 64U + wave_n * 16U + axis;
                b[nh][0] = static_cast<int>(current.weights[lds_index(wr, word, kN)]);
                b[nh][1] = static_cast<int>(current.weights[lds_index(wr, word + 1U, kN)]);
            }
            low[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, al, true, b[0], low[0], false);
            high[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, ah, true, b[0], high[0], false);
            low[1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, al, true, b[1], low[1], false);
            high[1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, ah, true, b[1], high[1], false);
        }
#pragma unroll
        for (std::uint32_t nh = 0; nh < 2U; ++nh) {
            const std::uint32_t wr = nh * 64U + wave_n * 16U + axis;
            const float ws = half_float(current.weight_scales[wr]);
#pragma unroll
            for (std::uint32_t e = 0; e < 8U; ++e) {
                const std::uint32_t mr = wave_m * 16U + lane_group * 8U + e;
                const float as = half_float(current.activation_scales[mr]);
                const int dot = low[nh][e] + 16 * high[nh][e];
                totals[nh][e] = fmaf(static_cast<float>(dot), as * ws, totals[nh][e]);
            }
        }

        if (has_next) {
            Tile& next = tiles.bank[(group + 1U) & 1U];
            next.activation_low[a_lds] = next_a_low;
            next.activation_high[a_lds] = next_a_high;
            next.weights[lds_index(load_row_local, load_word, kN)] =
                static_cast<std::uint32_t>(next_w);
            next.weights[lds_index(load_row_local, load_word + 1U, kN)] =
                static_cast<std::uint32_t>(next_w >> 32U);
            if (thread < kM) {
                next.activation_scales[thread] = activation_scales[
                    static_cast<std::size_t>(m0 + thread) * groups + group + 1U];
            }
            if (thread < kN) {
                const std::uint32_t row = n0 + thread;
                next.weight_scales[thread] =
                    weight_scales[scale_index(row, group + 1U, groups)];
            }
            barrier();
        }
    }

#pragma unroll
    for (std::uint32_t nh = 0; nh < 2U; ++nh) {
#pragma unroll
        for (std::uint32_t e = 0; e < 8U; ++e) {
            const std::uint32_t m = m0 + wave_m * 16U + lane_group * 8U + e;
            const std::uint32_t n = n0 + nh * 64U + wave_n * 16U + axis;
            output[static_cast<std::size_t>(m) * rows + n] =
                static_cast<hip_bfloat16>(totals[nh][e]);
        }
    }
#endif
}

extern "C" __global__ __launch_bounds__(kWave)
void a8q4_n16k16_decode_t1_wmma32(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* status,
    const std::uint64_t* weight_pairs, const std::uint16_t* weight_scales,
    hip_bfloat16* output, std::uint32_t rows, std::uint32_t columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t lane_group = lane >> 4U;
    const std::uint32_t n0 = blockIdx.x * 16U;
    const std::uint32_t groups = columns / 64U;
    if (*status != 0U) {
        if (lane_group == 0U) output[n0 + axis] = poison();
        return;
    }
    F32x8 total{};
    for (std::uint32_t group = 0; group < groups; ++group) {
        I32x8 low{}, high{};
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
            const std::uint32_t pair = half * 2U + lane_group;
            I32x2 al{}, ah{}, b{};
            const std::uint32_t k = group * 64U + pair * 16U;
            al[0] = static_cast<int>(load32(activation_low, k));
            al[1] = static_cast<int>(load32(activation_low, k + 8U));
            ah[0] = static_cast<int>(load32(activation_high, k));
            ah[1] = static_cast<int>(load32(activation_high, k + 8U));
            const std::uint64_t packed = load64(
                weight_pairs, code_pair_index(n0 + axis, group, pair, groups));
            b[0] = static_cast<int>(packed);
            b[1] = static_cast<int>(packed >> 32U);
            low = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, al, true, b, low, false);
            high = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, ah, true, b, high, false);
        }
        const float ws = half_float(weight_scales[scale_index(n0 + axis, group, groups)]);
        const float as = half_float(activation_scales[group]);
#pragma unroll
        for (std::uint32_t e = 0; e < 8U; ++e) {
            total[e] = fmaf(static_cast<float>(low[e] + 16 * high[e]), as * ws, total[e]);
        }
    }
    if (lane_group == 0U) output[n0 + axis] = static_cast<hip_bfloat16>(total[0]);
#endif
}
