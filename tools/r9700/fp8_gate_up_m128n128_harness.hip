#include "fp8_gate_up_m128n128_qual.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace candidate = ninfer::qualification::fp8_gate_up_m128n128;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void check_lt(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": hipBLASLt status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        check_hip(hipMalloc(reinterpret_cast<void**>(&pointer_), bytes()), "hipMalloc");
    }
    ~DeviceBuffer() { if (pointer_ != nullptr) (void)hipFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const { return pointer_; }
    [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }
private:
    T* pointer_ = nullptr;
    std::size_t elements_ = 0;
};

class Stream {
public:
    Stream() { check_hip(hipStreamCreate(&value_), "hipStreamCreate"); }
    ~Stream() { if (value_ != nullptr) (void)hipStreamDestroy(value_); }
    [[nodiscard]] hipStream_t get() const { return value_; }
private:
    hipStream_t value_ = nullptr;
};

class Event {
public:
    Event() { check_hip(hipEventCreate(&value_), "hipEventCreate"); }
    ~Event() { if (value_ != nullptr) (void)hipEventDestroy(value_); }
    [[nodiscard]] hipEvent_t get() const { return value_; }
private:
    hipEvent_t value_ = nullptr;
};

class LtContext {
public:
    LtContext(float* row_scales, float* token_scales) {
        check_lt(hipblasLtCreate(&handle), "hipblasLtCreate");
        check_lt(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                 "hipblasLtMatmulDescCreate");
        constexpr hipblasOperation_t transpose = HIPBLAS_OP_T;
        check_lt(hipblasLtMatmulDescSetAttribute(
                     operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
                 "set TRANSA");
        constexpr hipblasLtMatmulMatrixScale_t outer =
            HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        check_lt(hipblasLtMatmulDescSetAttribute(
                     operation, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &outer, sizeof(outer)),
                 "set A scale mode");
        check_lt(hipblasLtMatmulDescSetAttribute(
                     operation, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &outer, sizeof(outer)),
                 "set B scale mode");
        const void* row_scale_pointer = row_scales;
        const void* token_scale_pointer = token_scales;
        check_lt(hipblasLtMatmulDescSetAttribute(
                     operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                     &row_scale_pointer, sizeof(row_scale_pointer)),
                 "set A scale pointer");
        check_lt(hipblasLtMatmulDescSetAttribute(
                     operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                     &token_scale_pointer, sizeof(token_scale_pointer)),
                 "set B scale pointer");
        check_lt(hipblasLtMatrixLayoutCreate(
                     &weight, HIP_R_8F_E4M3, candidate::kColumns,
                     candidate::kRows, candidate::kColumns),
                 "create weight layout");
        check_lt(hipblasLtMatrixLayoutCreate(
                     &activation, HIP_R_8F_E4M3, candidate::kColumns,
                     candidate::kTokens, candidate::kColumns),
                 "create activation layout");
        check_lt(hipblasLtMatrixLayoutCreate(
                     &output, HIP_R_16BF, candidate::kRows,
                     candidate::kTokens, candidate::kRows),
                 "create output layout");

        std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
        check_lt(hipblaslt_ext::getAllAlgos(
                     handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                     HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_8F_E4M3, HIP_R_8F_E4M3,
                     HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F, algorithms),
                 "getAllAlgos");
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        bool found = false;
        for (auto result : algorithms) {
            if (hipblaslt_ext::getIndexFromAlgo(result.algo) != 123104) continue;
            std::size_t workspace = 0;
            if (hipblaslt_ext::matmulIsAlgoSupported(
                    handle, operation, &alpha, weight, activation, &beta,
                    output, output, result.algo, workspace) != HIPBLAS_STATUS_SUCCESS) {
                continue;
            }
            require(workspace == 0, "incumbent solution 123104 unexpectedly needs workspace");
            algorithm = result.algo;
            found = true;
            break;
        }
        require(found, "exact incumbent hipBLASLt solution 123104 is unavailable");
    }

    ~LtContext() {
        if (output != nullptr) (void)hipblasLtMatrixLayoutDestroy(output);
        if (activation != nullptr) (void)hipblasLtMatrixLayoutDestroy(activation);
        if (weight != nullptr) (void)hipblasLtMatrixLayoutDestroy(weight);
        if (operation != nullptr) (void)hipblasLtMatmulDescDestroy(operation);
        if (handle != nullptr) (void)hipblasLtDestroy(handle);
    }

    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t weight = nullptr;
    hipblasLtMatrixLayout_t activation = nullptr;
    hipblasLtMatrixLayout_t output = nullptr;
    hipblasLtMatmulAlgo_t algorithm{};
};

float decode_e4m3(std::uint8_t bits) {
    const unsigned magnitude = bits & 0x7fU;
    const unsigned exponent = magnitude >> 3U;
    const unsigned mantissa = magnitude & 7U;
    require(!(exponent == 15U && mantissa == 7U), "oracle encountered E4M3 NaN");
    float value = exponent == 0U
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                     static_cast<int>(exponent) - 7);
    return (bits & 0x80U) != 0U ? -value : value;
}

std::uint16_t round_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::uint32_t ordered_bf16(std::uint16_t bits) {
    return (bits & 0x8000U) != 0U ? 0x8000U - (bits & 0x7fffU) : 0x8000U + bits;
}

std::uint32_t bf16_steps(std::uint16_t lhs, std::uint16_t rhs) {
    const std::uint32_t a = ordered_bf16(lhs);
    const std::uint32_t b = ordered_bf16(rhs);
    return a > b ? a - b : b - a;
}

__global__ void poison_if_invalid(std::uint16_t* output, const std::uint32_t* status,
                                  std::size_t elements) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements && *status != 0U) output[index] = 0x7fc1U;
}

struct Fixture {
    Fixture()
        : activation_host(static_cast<std::size_t>(candidate::kTokens) * candidate::kColumns),
          weight_host(static_cast<std::size_t>(candidate::kRows) * candidate::kColumns),
          token_scale_host(candidate::kTokens), weight_scale_host(candidate::kRows),
          activation(activation_host.size()), weight(weight_host.size()),
          token_scales(token_scale_host.size()), weight_scales(weight_scale_host.size()),
          output(static_cast<std::size_t>(candidate::kTokens) * candidate::kRows), status(1) {
        constexpr std::array<std::uint8_t, 8> values{
            0x28U, 0xa8U, 0x30U, 0xb0U, 0x38U, 0xb8U, 0x40U, 0xc0U};
        for (std::uint32_t token = 0; token < candidate::kTokens; ++token) {
            for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
                activation_host[static_cast<std::size_t>(token) * candidate::kColumns + k] =
                    values[(token * 3U + k * 5U + (k >> 5U)) & 7U];
            }
            token_scale_host[token] = std::ldexp(1.0F, -7 + static_cast<int>(token % 4U));
        }
        for (std::uint32_t row = 0; row < candidate::kRows; ++row) {
            for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
                weight_host[static_cast<std::size_t>(row) * candidate::kColumns + k] =
                    values[(row * 7U + k * 3U + (k >> 4U)) & 7U];
            }
            weight_scale_host[row] = std::ldexp(1.0F, -6 + static_cast<int>(row % 4U));
        }
        check_hip(hipMemcpy(activation.get(), activation_host.data(), activation.bytes(),
                            hipMemcpyHostToDevice), "copy activation");
        check_hip(hipMemcpy(weight.get(), weight_host.data(), weight.bytes(),
                            hipMemcpyHostToDevice), "copy weight");
        check_hip(hipMemcpy(token_scales.get(), token_scale_host.data(), token_scales.bytes(),
                            hipMemcpyHostToDevice), "copy token scales");
        check_hip(hipMemcpy(weight_scales.get(), weight_scale_host.data(), weight_scales.bytes(),
                            hipMemcpyHostToDevice), "copy weight scales");
        check_hip(hipMemset(status.get(), 0, status.bytes()), "clear status");
    }

    candidate::Arguments arguments() const {
        return {activation.get(), token_scales.get(), weight.get(), weight_scales.get(),
                output.get(), candidate::kTokens, candidate::kRows, candidate::kColumns};
    }

    std::vector<std::uint8_t> activation_host;
    std::vector<std::uint8_t> weight_host;
    std::vector<float> token_scale_host;
    std::vector<float> weight_scale_host;
    DeviceBuffer<std::uint8_t> activation;
    DeviceBuffer<std::uint8_t> weight;
    DeviceBuffer<float> token_scales;
    DeviceBuffer<float> weight_scales;
    DeviceBuffer<std::uint16_t> output;
    DeviceBuffer<std::uint32_t> status;
};

void launch_complete(Fixture& fixture, hipStream_t stream) {
    check_hip(candidate::launch(fixture.arguments(), stream), "candidate launch");
    constexpr std::size_t elements =
        static_cast<std::size_t>(candidate::kTokens) * candidate::kRows;
    hipLaunchKernelGGL(poison_if_invalid, dim3((elements + 255U) / 256U), dim3(256),
                       0, stream, fixture.output.get(), fixture.status.get(), elements);
    check_hip(hipGetLastError(), "poison launch");
}

void launch_incumbent(Fixture& fixture, LtContext& context, hipStream_t stream) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    check_lt(hipblasLtMatmul(
                 context.handle, context.operation, &alpha,
                 fixture.weight.get(), context.weight,
                 fixture.activation.get(), context.activation, &beta,
                 fixture.output.get(), context.output,
                 fixture.output.get(), context.output,
                 &context.algorithm, nullptr, 0, stream),
             "incumbent hipblasLtMatmul");
}

constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 16> kProbes{{
    {0, 0}, {0, 127}, {0, 128}, {17, 12345}, {127, 127}, {127, 128},
    {128, 127}, {128, 128}, {511, 8191}, {1023, 17407}, {1024, 17408},
    {1535, 24575}, {2046, 34687}, {2047, 34688}, {2047, 34814}, {2047, 34815}}};

std::uint32_t validate_oracle(const Fixture& fixture) {
    std::uint32_t maximum_steps = 0;
    for (const auto [token, row] : kProbes) {
        std::uint16_t actual = 0;
        const std::size_t output_index = static_cast<std::size_t>(token) * candidate::kRows + row;
        check_hip(hipMemcpy(&actual, fixture.output.get() + output_index, sizeof(actual),
                            hipMemcpyDeviceToHost), "copy oracle probe");
        double dot = 0.0;
        const std::size_t activation_base = static_cast<std::size_t>(token) * candidate::kColumns;
        const std::size_t weight_base = static_cast<std::size_t>(row) * candidate::kColumns;
        for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
            dot += static_cast<double>(decode_e4m3(fixture.activation_host[activation_base + k])) *
                   static_cast<double>(decode_e4m3(fixture.weight_host[weight_base + k]));
        }
        const float represented = static_cast<float>(dot) * fixture.token_scale_host[token] *
                                  fixture.weight_scale_host[row];
        maximum_steps = std::max(maximum_steps, bf16_steps(actual, round_bf16(represented)));
    }
    require(maximum_steps <= 1U, "candidate differs from independent FP64 oracle by >1 BF16 step");
    return maximum_steps;
}

void regression(Fixture& fixture, LtContext& context, Stream& stream) {
    candidate::Resources resources{};
    check_hip(candidate::resources(&resources), "candidate resources");
    require(resources.registers <= 88 && resources.static_lds_bytes == 16384 &&
                resources.private_bytes == 0 && resources.maximum_threads >= 512,
            "runtime resource gate failed");

    check_hip(hipMemset(fixture.output.get(), 0xa5, fixture.output.bytes()), "sentinel output");
    auto malformed = fixture.arguments();
    --malformed.columns;
    require(candidate::launch(malformed, stream.get()) == hipErrorInvalidValue,
            "malformed exact shape was accepted");
    auto aliased = fixture.arguments();
    aliased.output_bf16 = reinterpret_cast<std::uint16_t*>(
        const_cast<std::uint8_t*>(aliased.activation_codes));
    require(candidate::launch(aliased, stream.get()) == hipErrorInvalidValue,
            "output alias was accepted");
    std::uint16_t endpoint = 0;
    check_hip(hipMemcpy(&endpoint, fixture.output.get(), sizeof(endpoint), hipMemcpyDeviceToHost),
              "copy no-clobber endpoint");
    require(endpoint == 0xa5a5U, "rejected launch clobbered output");

    launch_complete(fixture, stream.get());
    check_hip(hipStreamSynchronize(stream.get()), "candidate synchronize");
    (void)validate_oracle(fixture);

    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    check_hip(hipStreamBeginCapture(stream.get(), hipStreamCaptureModeGlobal),
              "begin graph capture");
    launch_complete(fixture, stream.get());
    check_hip(hipStreamEndCapture(stream.get(), &graph), "end graph capture");
    check_hip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
              "instantiate graph");
    check_hip(hipGraphLaunch(executable, stream.get()), "launch graph");
    check_hip(hipStreamSynchronize(stream.get()), "graph synchronize");
    (void)validate_oracle(fixture);
    (void)hipGraphExecDestroy(executable);
    (void)hipGraphDestroy(graph);

    launch_incumbent(fixture, context, stream.get());
    check_hip(hipStreamSynchronize(stream.get()), "incumbent synchronize");
    (void)validate_oracle(fixture);

    const std::uint32_t invalid = 1U;
    check_hip(hipMemcpy(fixture.status.get(), &invalid, sizeof(invalid), hipMemcpyHostToDevice),
              "set invalid status");
    launch_complete(fixture, stream.get());
    check_hip(hipStreamSynchronize(stream.get()), "poison synchronize");
    for (const auto [token, row] : kProbes) {
        std::uint16_t value = 0;
        check_hip(hipMemcpy(&value,
                            fixture.output.get() + static_cast<std::size_t>(token) *
                                                       candidate::kRows + row,
                            sizeof(value), hipMemcpyDeviceToHost), "copy poison probe");
        require(value == 0x7fc1U, "invalid status did not poison complete output");
    }
    check_hip(hipMemset(fixture.status.get(), 0, fixture.status.bytes()), "restore status");
}

template <typename Launch>
float time_one(Launch&& launch, hipStream_t stream, Event& begin, Event& end) {
    check_hip(hipEventRecord(begin.get(), stream), "record begin");
    launch();
    check_hip(hipEventRecord(end.get(), stream), "record end");
    check_hip(hipEventSynchronize(end.get()), "synchronize event");
    float milliseconds = 0.0F;
    check_hip(hipEventElapsedTime(&milliseconds, begin.get(), end.get()), "elapsed time");
    return milliseconds;
}

double median(std::array<float, 14> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (static_cast<double>(values[6]) + static_cast<double>(values[7]));
}

void benchmark(Fixture& fixture, LtContext& context, Stream& stream) {
    for (int index = 0; index < 2; ++index) {
        launch_incumbent(fixture, context, stream.get());
        check_hip(candidate::launch(fixture.arguments(), stream.get()),
                  "candidate matrix warmup");
    }
    check_hip(hipStreamSynchronize(stream.get()), "warmup synchronize");
    Event begin;
    Event end;
    std::array<float, 14> incumbent{};
    std::array<float, 14> challenger{};
    for (std::size_t pair = 0; pair < 14; ++pair) {
        const auto incumbent_launch = [&] {
            launch_incumbent(fixture, context, stream.get());
        };
        const auto challenger_launch = [&] {
            check_hip(candidate::launch(fixture.arguments(), stream.get()),
                      "timed candidate matrix launch");
        };
        if ((pair & 1U) == 0U) {
            incumbent[pair] = time_one(incumbent_launch, stream.get(), begin, end);
            challenger[pair] = time_one(challenger_launch, stream.get(), begin, end);
        } else {
            challenger[pair] = time_one(challenger_launch, stream.get(), begin, end);
            incumbent[pair] = time_one(incumbent_launch, stream.get(), begin, end);
        }
    }
    const double incumbent_ms = median(incumbent);
    const double challenger_ms = median(challenger);
    const double saving_64 = 64.0 * (incumbent_ms - challenger_ms);
    const bool pass = challenger_ms <= 2.499050617 && saving_64 >= 100.0;
    std::cout << std::fixed << std::setprecision(9)
              << "{\n  \"schema\": \"ninfer.r9700.fp8_gate_up_m128n128_ab.v1\",\n"
              << "  \"shape_N_K_T\": [34816,5120,2048],\n"
              << "  \"incumbent_solution\": 123104,\n"
              << "  \"timing_scope\": \"matrix_only_same_represented_fp8_inputs\",\n"
              << "  \"incumbent_matrix_median_ms\": " << incumbent_ms << ",\n"
              << "  \"challenger_matrix_median_ms\": " << challenger_ms << ",\n"
              << "  \"projected_64_call_saving_ms\": " << saving_64 << ",\n"
              << "  \"maximum_challenger_ms\": 2.499050617,\n"
              << "  \"minimum_64_call_saving_ms\": 100.000000000,\n"
              << "  \"pass\": " << (pass ? "true" : "false") << "\n}\n";
    require(pass, "challenger failed the exact direct performance gate");
}

void require_device_and_auto() {
    check_hip(hipSetDevice(0), "hipSetDevice(0)");
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    require(std::string_view(properties.name) == "AMD Radeon AI PRO R9700" &&
                std::string_view(properties.gcnArchName) == "gfx1201" &&
                properties.warpSize == 32,
            "qualifier requires device-0 R9700 gfx1201 wave32");
    std::array<char, 32> pci{};
    check_hip(hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), 0),
              "hipDeviceGetPCIBusId");
    std::string bus = pci.data();
    if (bus.size() == 7U) bus = "0000:" + bus;
    std::ifstream power("/sys/bus/pci/devices/" + bus +
                        "/power_dpm_force_performance_level");
    std::string profile;
    power >> profile;
    require(power.good() || power.eof(), "cannot read R9700 power profile");
    require(profile == "auto", "qualifier requires R9700 power profile auto");
}

} // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "usage: fp8_gate_up_m128n128_harness --regression|--benchmark");
    const std::string_view mode = argv[1];
    require(mode == "--regression" || mode == "--benchmark", "unknown qualifier mode");
    require_device_and_auto();
    Stream stream;
    Fixture fixture;
    LtContext context(fixture.weight_scales.get(), fixture.token_scales.get());
    regression(fixture, context, stream);
    if (mode == "--benchmark") benchmark(fixture, context, stream);
    else std::cout << "FP8 gate/up M128xN128 regression PASS\n";
    require_device_and_auto();
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "fp8_gate_up_m128n128_harness: " << error.what() << '\n';
    return EXIT_FAILURE;
}
