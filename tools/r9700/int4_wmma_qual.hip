#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned kM = 16;
constexpr unsigned kN = 16;
constexpr unsigned kWave = 32;
[[maybe_unused]] constexpr unsigned kOutputsPerLane = 8;

#define HIP_CHECK(expression)                                                                    \
    do {                                                                                         \
        const hipError_t error = (expression);                                                   \
        if (error != hipSuccess) {                                                               \
            std::ostringstream os;                                                               \
            os << #expression << " failed: " << hipGetErrorString(error);                        \
            throw std::runtime_error(os.str());                                                  \
        }                                                                                        \
    } while (false)

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) { (void)hipFree(data_); }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }
    std::size_t size_bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

using I32x2 = int __attribute__((ext_vector_type(2)));
using I32x8 = int __attribute__((ext_vector_type(8)));

__host__ __device__ int decode_int4(const std::uint8_t* packed, std::size_t index) {
    const unsigned nibble = (packed[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

__device__ __attribute__((unused)) int pack_a8(const std::uint8_t* matrix_a,
                                               unsigned k,
                                               unsigned row,
                                               unsigned k_begin) {
    std::uint32_t packed = 0;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned code = static_cast<unsigned>(decode_int4(matrix_a, row * k + k_begin + i)) &
                              0x0fU;
        packed |= code << (4U * i);
    }
    return static_cast<int>(packed);
}

__device__ __attribute__((unused)) int pack_b8(const std::uint8_t* matrix_b,
                                               unsigned column,
                                               unsigned k_begin) {
    std::uint32_t packed = 0;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned code =
            static_cast<unsigned>(decode_int4(matrix_b, (k_begin + i) * kN + column)) & 0x0fU;
        packed |= code << (4U * i);
    }
    return static_cast<int>(packed);
}

// Literal gfx12 wave32 map under test. Lanes 0..15 own K's low half and lanes 16..31 own
// its high half; lane%16 selects A's row and B's column. Accumulator element j maps to
// C[(lane/16)*8+j, lane%16]. These loads deliberately do not use rocWMMA/CK layout helpers.
__global__ __launch_bounds__(kWave) void int4_wmma16_kernel(const std::uint8_t* matrix_a,
                                                            const std::uint8_t* matrix_b,
                                                            const std::int32_t* seed,
                                                            std::int32_t* output,
                                                            unsigned k,
                                                            unsigned batch) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned tile = blockIdx.x;
    if (tile >= batch) { return; }
    const unsigned lane = threadIdx.x & 31U;
    const unsigned axis = lane & 15U;
    const unsigned k_half = lane >> 4U;
    I32x8 accumulator;
#pragma unroll
    for (unsigned j = 0; j < kOutputsPerLane; ++j) {
        const unsigned row = k_half * 8U + j;
        accumulator[j] = seed[row * kN + axis];
    }
    for (unsigned k_tile = 0; k_tile < k; k_tile += 16U) {
        const unsigned k_begin = k_tile + k_half * 8U;
        const int a = pack_a8(matrix_a, k, axis, k_begin);
        const int b = pack_b8(matrix_b, axis, k_begin);
        accumulator = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
            true, a, true, b, accumulator, false);
    }
#pragma unroll
    for (unsigned j = 0; j < kOutputsPerLane; ++j) {
        const unsigned row = k_half * 8U + j;
        output[(tile * kM + row) * kN + axis] = accumulator[j];
    }
#else
    (void)matrix_a;
    (void)matrix_b;
    (void)seed;
    (void)output;
    (void)k;
    (void)batch;
#endif
}

__global__ __launch_bounds__(kWave) void int4_wmma32_kernel(const std::uint8_t* matrix_a,
                                                            const std::uint8_t* matrix_b,
                                                            const std::int32_t* seed,
                                                            std::int32_t* output,
                                                            unsigned k,
                                                            unsigned batch) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned tile = blockIdx.x;
    if (tile >= batch) { return; }
    const unsigned lane = threadIdx.x & 31U;
    const unsigned axis = lane & 15U;
    const unsigned k_half = lane >> 4U;
    I32x8 accumulator;
#pragma unroll
    for (unsigned j = 0; j < kOutputsPerLane; ++j) {
        const unsigned row = k_half * 8U + j;
        accumulator[j] = seed[row * kN + axis];
    }
    for (unsigned k_tile = 0; k_tile < k; k_tile += 32U) {
        const unsigned k_begin = k_tile + k_half * 16U;
        const I32x2 a = {pack_a8(matrix_a, k, axis, k_begin),
                         pack_a8(matrix_a, k, axis, k_begin + 8U)};
        const I32x2 b = {pack_b8(matrix_b, axis, k_begin),
                         pack_b8(matrix_b, axis, k_begin + 8U)};
        accumulator = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true, a, true, b, accumulator, false);
    }
#pragma unroll
    for (unsigned j = 0; j < kOutputsPerLane; ++j) {
        const unsigned row = k_half * 8U + j;
        output[(tile * kM + row) * kN + axis] = accumulator[j];
    }
#else
    (void)matrix_a;
    (void)matrix_b;
    (void)seed;
    (void)output;
    (void)k;
    (void)batch;
#endif
}

__global__ __launch_bounds__(256) void int4_vector_dot_kernel(const std::uint8_t* matrix_a,
                                                              const std::uint8_t* matrix_b,
                                                              const std::int32_t* seed,
                                                              std::int32_t* output,
                                                              unsigned k,
                                                              unsigned batch) {
    const unsigned tile = blockIdx.x;
    const unsigned element = threadIdx.x;
    if (tile >= batch || element >= kM * kN) { return; }
    const unsigned row = element / kN;
    const unsigned column = element % kN;
    std::int32_t sum = seed[element];
    for (unsigned inner = 0; inner < k; ++inner) {
        sum += decode_int4(matrix_a, row * k + inner) *
               decode_int4(matrix_b, inner * kN + column);
    }
    output[tile * kM * kN + element] = sum;
}

void set_int4(std::vector<std::uint8_t>& packed, std::size_t index, int value) {
    if (value < -8 || value > 7) { fail("INT4 fixture value is out of range"); }
    const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
    const std::uint8_t mask = static_cast<std::uint8_t>(0x0fU << shift);
    packed[index >> 1U] = static_cast<std::uint8_t>(
        (packed[index >> 1U] & static_cast<std::uint8_t>(~mask)) |
        ((static_cast<unsigned>(value) & 0x0fU) << shift));
}

std::vector<std::int32_t> host_oracle(const std::vector<std::uint8_t>& matrix_a,
                                      const std::vector<std::uint8_t>& matrix_b,
                                      const std::vector<std::int32_t>& seed,
                                      unsigned k) {
    std::vector<std::int32_t> expected(kM * kN);
    for (unsigned row = 0; row < kM; ++row) {
        for (unsigned column = 0; column < kN; ++column) {
            std::int32_t sum = seed[row * kN + column];
            for (unsigned inner = 0; inner < k; ++inner) {
                sum += decode_int4(matrix_a.data(), row * k + inner) *
                       decode_int4(matrix_b.data(), inner * kN + column);
            }
            expected[row * kN + column] = sum;
        }
    }
    return expected;
}

struct Fixture {
    unsigned k;
    std::vector<std::uint8_t> matrix_a;
    std::vector<std::uint8_t> matrix_b;
    std::vector<std::int32_t> seed;
};

Fixture sparse_fixture(unsigned k) {
    Fixture fixture{k,
                    std::vector<std::uint8_t>(kM * k / 2U, 0),
                    std::vector<std::uint8_t>(k * kN / 2U, 0),
                    std::vector<std::int32_t>(kM * kN)};
    for (unsigned row = 0; row < kM; ++row) {
        fixture.seed[row * kN + ((row * 11U + 3U) & 15U)] =
            static_cast<std::int32_t>(1009U + row * 37U);
        const unsigned inner0 = (row * 13U + 1U) % k;
        const unsigned inner1 = (row * 7U + k / 2U + 3U) % k;
        set_int4(fixture.matrix_a, row * k + inner0, static_cast<int>((row * 5U) & 15U) - 8);
        set_int4(fixture.matrix_a, row * k + inner1, static_cast<int>((row * 3U + 6U) & 15U) - 8);
    }
    for (unsigned column = 0; column < kN; ++column) {
        const unsigned inner0 = (column * 9U + 1U) % k;
        const unsigned inner1 = (column * 5U + k / 2U + 3U) % k;
        set_int4(fixture.matrix_b,
                 inner0 * kN + column,
                 static_cast<int>((column * 7U + 2U) & 15U) - 8);
        set_int4(fixture.matrix_b,
                 inner1 * kN + column,
                 static_cast<int>((column * 11U + 5U) & 15U) - 8);
    }
    return fixture;
}

Fixture dense_fixture(unsigned k) {
    Fixture fixture{k,
                    std::vector<std::uint8_t>(kM * k / 2U, 0),
                    std::vector<std::uint8_t>(k * kN / 2U, 0),
                    std::vector<std::int32_t>(kM * kN)};
    for (unsigned row = 0; row < kM; ++row) {
        for (unsigned inner = 0; inner < k; ++inner) {
            set_int4(fixture.matrix_a,
                     row * k + inner,
                     static_cast<int>((row * 13U + inner * 7U + 5U) & 15U) - 8);
        }
    }
    for (unsigned inner = 0; inner < k; ++inner) {
        for (unsigned column = 0; column < kN; ++column) {
            set_int4(fixture.matrix_b,
                     inner * kN + column,
                     static_cast<int>((inner * 11U + column * 3U + 1U) & 15U) - 8);
        }
    }
    for (unsigned row = 0; row < kM; ++row) {
        for (unsigned column = 0; column < kN; ++column) {
            fixture.seed[row * kN + column] =
                static_cast<std::int32_t>(row * 101U - column * 17U + 29U);
        }
    }
    return fixture;
}

enum class Route { Wmma16, Wmma32, Vector };

void launch(Route route,
            const Fixture& fixture,
            const DeviceBuffer<std::uint8_t>& matrix_a,
            const DeviceBuffer<std::uint8_t>& matrix_b,
            const DeviceBuffer<std::int32_t>& seed,
            DeviceBuffer<std::int32_t>& output,
            unsigned batch) {
    if (route == Route::Wmma16) {
        hipLaunchKernelGGL(int4_wmma16_kernel,
                           dim3(batch),
                           dim3(kWave),
                           0,
                           0,
                           matrix_a.get(),
                           matrix_b.get(),
                           seed.get(),
                           output.get(),
                           fixture.k,
                           batch);
    } else if (route == Route::Wmma32) {
        hipLaunchKernelGGL(int4_wmma32_kernel,
                           dim3(batch),
                           dim3(kWave),
                           0,
                           0,
                           matrix_a.get(),
                           matrix_b.get(),
                           seed.get(),
                           output.get(),
                           fixture.k,
                           batch);
    } else {
        hipLaunchKernelGGL(int4_vector_dot_kernel,
                           dim3(batch),
                           dim3(kM * kN),
                           0,
                           0,
                           matrix_a.get(),
                           matrix_b.get(),
                           seed.get(),
                           output.get(),
                           fixture.k,
                           batch);
    }
    HIP_CHECK(hipGetLastError());
}

void check_output(const std::vector<std::int32_t>& actual,
                  const std::vector<std::int32_t>& expected,
                  unsigned batch,
                  const char* route) {
    for (unsigned tile = 0; tile < batch; ++tile) {
        for (unsigned row = 0; row < kM; ++row) {
            for (unsigned column = 0; column < kN; ++column) {
                const std::size_t element = row * kN + column;
                const std::int32_t got = actual[tile * kM * kN + element];
                if (got != expected[element]) {
                    std::ostringstream os;
                    os << route << " mismatch at tile " << tile << " C[" << row << ',' << column
                       << "]: got " << got << " expected " << expected[element];
                    fail(os.str());
                }
            }
        }
    }
}

void copy_fixture(const Fixture& fixture,
                  DeviceBuffer<std::uint8_t>& matrix_a,
                  DeviceBuffer<std::uint8_t>& matrix_b,
                  DeviceBuffer<std::int32_t>& seed) {
    HIP_CHECK(hipMemcpy(matrix_a.get(),
                        fixture.matrix_a.data(),
                        fixture.matrix_a.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(matrix_b.get(),
                        fixture.matrix_b.data(),
                        fixture.matrix_b.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        seed.get(), fixture.seed.data(), fixture.seed.size() * sizeof(std::int32_t), hipMemcpyHostToDevice));
}

void test_fragment_map(Route route, unsigned k, const char* name) {
    const Fixture fixture = sparse_fixture(k);
    const std::vector<std::int32_t> expected =
        host_oracle(fixture.matrix_a, fixture.matrix_b, fixture.seed, fixture.k);
    DeviceBuffer<std::uint8_t> matrix_a(fixture.matrix_a.size());
    DeviceBuffer<std::uint8_t> matrix_b(fixture.matrix_b.size());
    DeviceBuffer<std::int32_t> seed(fixture.seed.size());
    DeviceBuffer<std::int32_t> output(kM * kN);
    copy_fixture(fixture, matrix_a, matrix_b, seed);
    launch(route, fixture, matrix_a, matrix_b, seed, output, 1);
    std::vector<std::int32_t> actual(kM * kN);
    HIP_CHECK(hipMemcpy(actual.data(), output.get(), output.size_bytes(), hipMemcpyDeviceToHost));
    check_output(actual, expected, 1, name);
    std::cout << "check " << name
              << ": PASS (all 256 int32 outputs exact; sparse asymmetric A/B/C seed)\n";
}

struct Timing {
    float milliseconds;
    double nanoseconds_per_tile;
    double tops;
};

Timing time_route(Route route,
                  const Fixture& fixture,
                  const DeviceBuffer<std::uint8_t>& matrix_a,
                  const DeviceBuffer<std::uint8_t>& matrix_b,
                  const DeviceBuffer<std::int32_t>& seed,
                  DeviceBuffer<std::int32_t>& output,
                  unsigned batch,
                  unsigned repeats) {
    for (unsigned warmup = 0; warmup < 5; ++warmup) {
        launch(route, fixture, matrix_a, matrix_b, seed, output, batch);
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin));
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        launch(route, fixture, matrix_a, matrix_b, seed, output, batch);
    }
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float total_ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&total_ms, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    const float milliseconds = total_ms / static_cast<float>(repeats);
    const double operations = 2.0 * static_cast<double>(kM) * static_cast<double>(kN) *
                              static_cast<double>(fixture.k) * static_cast<double>(batch);
    return Timing{milliseconds,
                  static_cast<double>(milliseconds) * 1.0e6 / static_cast<double>(batch),
                  operations / (static_cast<double>(milliseconds) * 1.0e9)};
}

const char* route_name(Route route) {
    switch (route) {
    case Route::Wmma16: return "wmma16";
    case Route::Wmma32: return "wmma32";
    case Route::Vector: return "vector";
    }
    return "unknown";
}

void benchmark(unsigned k, unsigned batch, unsigned repeats) {
    const Fixture fixture = dense_fixture(k);
    const std::vector<std::int32_t> expected =
        host_oracle(fixture.matrix_a, fixture.matrix_b, fixture.seed, fixture.k);
    DeviceBuffer<std::uint8_t> matrix_a(fixture.matrix_a.size());
    DeviceBuffer<std::uint8_t> matrix_b(fixture.matrix_b.size());
    DeviceBuffer<std::int32_t> seed(fixture.seed.size());
    DeviceBuffer<std::int32_t> output(static_cast<std::size_t>(batch) * kM * kN);
    copy_fixture(fixture, matrix_a, matrix_b, seed);

    std::array<Route, 3> routes = {Route::Wmma16, Route::Wmma32, Route::Vector};
    std::array<Timing, 3> timings{};
    for (std::size_t index = 0; index < routes.size(); ++index) {
        launch(routes[index], fixture, matrix_a, matrix_b, seed, output, batch);
        std::vector<std::int32_t> actual(static_cast<std::size_t>(batch) * kM * kN);
        HIP_CHECK(hipMemcpy(actual.data(), output.get(), output.size_bytes(), hipMemcpyDeviceToHost));
        check_output(actual, expected, batch, route_name(routes[index]));
        timings[index] =
            time_route(routes[index], fixture, matrix_a, matrix_b, seed, output, batch, repeats);
    }

    std::cout << "bench K=" << k << " tiles=" << batch << " repeats=" << repeats;
    for (std::size_t index = 0; index < routes.size(); ++index) {
        std::cout << ' ' << route_name(routes[index]) << '=' << std::fixed << std::setprecision(2)
                  << timings[index].nanoseconds_per_tile << "ns/tile," << std::setprecision(3)
                  << timings[index].tops << "TOPS";
    }
    std::cout << " speedup16=" << std::setprecision(2)
              << timings[2].nanoseconds_per_tile / timings[0].nanoseconds_per_tile
              << "x speedup32="
              << timings[2].nanoseconds_per_tile / timings[1].nanoseconds_per_tile << "x\n";
}

} // namespace

int main() {
    try {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        if (std::string(properties.gcnArchName).find("gfx1201") == std::string::npos) {
            fail(std::string("qualifier requires gfx1201, found ") + properties.gcnArchName);
        }
        if (properties.warpSize != static_cast<int>(kWave)) {
            fail("qualifier requires wave32");
        }
        std::cout << "device " << properties.name << " arch=" << properties.gcnArchName
                  << " wave=" << properties.warpSize << '\n';
        std::cout << "map A(row=lane%16,k=tile+(lane/16)*K/2+nibble) "
                     "B(k=tile+(lane/16)*K/2+nibble,column=lane%16) "
                     "C(row=(lane/16)*8+element,column=lane%16)\n";
        test_fragment_map(Route::Wmma16, 16, "gfx12_i32_16x16x16_iu4_w32_map");
        test_fragment_map(Route::Wmma32, 32, "gfx12_i32_16x16x32_iu4_w32_map");
        benchmark(64, 256, 100);
        benchmark(256, 256, 75);
        benchmark(1024, 256, 40);
        benchmark(5120, 256, 20);
        std::cout << "int4 WMMA feasibility: PASS (primitive evidence only; no Q4G64 selection)\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "int4 WMMA feasibility: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
