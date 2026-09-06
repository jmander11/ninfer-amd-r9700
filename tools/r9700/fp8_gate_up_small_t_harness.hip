#include "fp8_gate_up_small_t_qual.h"

#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/linear_execution.h"
#include "tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace candidate = ninfer::qualification::fp8_gate_up_small_t;
namespace fp8 = ninfer::ops::r9700::linear;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}
#define HIP_CHECK(expression) hip_check((expression), #expression)

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), bytes()));
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    std::size_t bytes() const { return elements_ * sizeof(T); }
private:
    T* data_ = nullptr;
    std::size_t elements_ = 0;
};

class Stream {
public:
    Stream() { HIP_CHECK(hipStreamCreate(&stream_)); }
    ~Stream() { if (stream_ != nullptr) (void)hipStreamDestroy(stream_); }
    hipStream_t get() const { return stream_; }
private:
    hipStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    ~Event() { if (event_ != nullptr) (void)hipEventDestroy(event_); }
    hipEvent_t get() const { return event_; }
private:
    hipEvent_t event_ = nullptr;
};

std::uint16_t round_bf16(double value) {
    const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
    const double magnitude = std::fabs(value);
    if (std::isnan(magnitude)) return static_cast<std::uint16_t>(sign | 0x7fc0U);
    if (std::isinf(magnitude)) return static_cast<std::uint16_t>(sign | 0x7f80U);
    if (magnitude == 0.0) return sign;

    // Round the complete FP64 formula directly to BF16, independently of the
    // production FP32 accumulator/store conversion. BF16 subnormals have a fixed
    // 2^-133 quantum; normals have a 2^(frexp_exponent-8) quantum.
    int exponent = 0;
    (void)std::frexp(magnitude, &exponent);
    const bool subnormal = exponent <= -126;
    const double quantum = std::ldexp(1.0, subnormal ? -133 : exponent - 8);
    const double quotient = magnitude / quantum;
    const double floor_value = std::floor(quotient);
    const double remainder = quotient - floor_value;
    std::uint64_t rounded = static_cast<std::uint64_t>(floor_value);
    if (remainder > 0.5 || (remainder == 0.5 && (rounded & 1U) != 0U)) ++rounded;
    if (subnormal) {
        if (rounded >= 128U) return static_cast<std::uint16_t>(sign | 0x0080U);
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (rounded == 256U) {
        rounded = 128U;
        ++exponent;
    }
    if (exponent > 128) return static_cast<std::uint16_t>(sign | 0x7f80U);
    const std::uint16_t biased = static_cast<std::uint16_t>(exponent + 126);
    return static_cast<std::uint16_t>(sign | (biased << 7U) |
                                      static_cast<std::uint16_t>(rounded - 128U));
}

std::uint32_t ordered_bf16(std::uint16_t value) {
    return (value & 0x8000U) != 0U ? 0x8000U - (value & 0x7fffU) : 0x8000U + value;
}

std::uint32_t steps(std::uint16_t lhs, std::uint16_t rhs) {
    const auto a = ordered_bf16(lhs), b = ordered_bf16(rhs);
    return a > b ? a - b : b - a;
}

struct Cell {
    static constexpr std::size_t kOutputGuardElements = 128U;

    explicit Cell(std::uint32_t width)
        : tokens(width), input_host(static_cast<std::size_t>(tokens) * candidate::kColumns),
          weight_host(static_cast<std::size_t>(candidate::kRows) * candidate::kColumns),
          weight_scale_host(candidate::kRows), input(input_host.size()),
          weight(weight_host.size()), weight_scales(weight_scale_host.size()),
          incumbent_output(static_cast<std::size_t>(tokens) * candidate::kRows +
                           2U * kOutputGuardElements),
          candidate_output(static_cast<std::size_t>(tokens) * candidate::kRows +
                           2U * kOutputGuardElements),
          incumbent_storage(fp8::fp8_activation_workspace_capacity_bytes(
              tokens, candidate::kColumns)),
          candidate_storage(fp8::fp8_activation_workspace_capacity_bytes(
              tokens, candidate::kColumns)) {
        constexpr std::array<float, 8> values{1.0F, -0.5F, 0.25F, -0.125F,
                                               0.75F, -0.375F, 0.0625F, -1.0F};
        for (std::uint32_t token = 0; token < tokens; ++token) {
            for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
                input_host[static_cast<std::size_t>(token) * candidate::kColumns + k] =
                    hip_bfloat16(values[(token * 3U + k * 5U + (k >> 5U)) & 7U]);
            }
        }
        constexpr std::array<float, 8> represented_weights{1.0F, -1.0F, 0.5F, -0.5F,
                                                            2.0F, -2.0F, 0.25F, -0.25F};
        for (std::uint32_t row = 0; row < candidate::kRows; ++row) {
            for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
                weight_host[static_cast<std::size_t>(row) * candidate::kColumns + k] =
                    oracle::encode_e4m3fn(represented_weights[
                        (row * 7U + k * 3U + (k >> 4U)) & 7U]);
            }
            weight_scale_host[row] = std::ldexp(1.0F, -7 + static_cast<int>(row & 3U));
        }
        HIP_CHECK(hipMemcpy(input.get(), input_host.data(), input.bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(weight.get(), weight_host.data(), weight.bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(weight_scales.get(), weight_scale_host.data(), weight_scales.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(fp8::fp8_bind_activation_workspace(
            incumbent_storage.get(), incumbent_storage.bytes(), tokens, candidate::kColumns,
            &incumbent_activation));
        HIP_CHECK(fp8::fp8_bind_activation_workspace(
            candidate_storage.get(), candidate_storage.bytes(), tokens, candidate::kColumns,
            &candidate_activation));

        ninfer::Weight bound{};
        bound.payload = weight.get();
        bound.payload_bytes = weight.bytes() + weight_scales.bytes();
        // The qualifier uses separate allocations, while LinearExecution requires its persisted
        // row scales to follow the codes. Rebind one contiguous incumbent image below.
        incumbent_weight = std::make_unique<DeviceBuffer<std::uint8_t>>(bound.payload_bytes);
        HIP_CHECK(hipMemcpy(incumbent_weight->get(), weight_host.data(), weight.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(incumbent_weight->get() + weight.bytes(), weight_scale_host.data(),
                            weight_scales.bytes(), hipMemcpyHostToDevice));
        bound.payload = incumbent_weight->get();
        bound.qdata = incumbent_weight->get();
        bound.scales = incumbent_weight->get() + weight.bytes();
        bound.qtype = ninfer::QType::F8E4M3_ROW_F32S;
        bound.layout = ninfer::QuantLayout::RowScaled;
        bound.scale_dtype = ninfer::DType::FP32;
        bound.n = candidate::kRows;
        bound.k = candidate::kColumns;
        bound.qdata_bytes = weight.bytes();
        bound.scale_bytes = weight_scales.bytes();
        bound.ndim = 2;
        bound.shape[0] = candidate::kRows;
        bound.shape[1] = candidate::kColumns;
        bound.padded_shape[0] = candidate::kRows;
        bound.padded_shape[1] = candidate::kColumns;
        bound.scale_ne[0] = candidate::kRows;
        bound.scale_nb[0] = sizeof(float);
        incumbent = std::make_unique<ninfer::ops::LinearExecution>(
            bound, incumbent_storage.get(), incumbent_storage.bytes(), nullptr, 0U);
        (void)incumbent->prepare(tokens);
    }

    candidate::Arguments candidate_args() const {
        return {input.get(), weight.get(), weight_scales.get(), candidate_activation,
                candidate_output.get() + kOutputGuardElements, tokens,
                candidate::kRows, candidate::kColumns};
    }

    std::uint32_t tokens;
    std::vector<hip_bfloat16> input_host;
    std::vector<std::uint8_t> weight_host;
    std::vector<float> weight_scale_host;
    DeviceBuffer<hip_bfloat16> input;
    DeviceBuffer<std::uint8_t> weight;
    DeviceBuffer<float> weight_scales;
    DeviceBuffer<hip_bfloat16> incumbent_output;
    DeviceBuffer<hip_bfloat16> candidate_output;
    DeviceBuffer<std::uint8_t> incumbent_storage;
    DeviceBuffer<std::uint8_t> candidate_storage;
    fp8::Fp8ActivationWorkspace incumbent_activation{};
    fp8::Fp8ActivationWorkspace candidate_activation{};
    std::unique_ptr<DeviceBuffer<std::uint8_t>> incumbent_weight;
    std::unique_ptr<ninfer::ops::LinearExecution> incumbent;
};

void launch_incumbent(Cell& cell, hipStream_t stream) {
    require(cell.incumbent->run(cell.tokens, cell.input.get(),
                                cell.incumbent_output.get() + Cell::kOutputGuardElements,
                                stream).ok(),
            "incumbent complete route failed");
}

void launch_candidate(Cell& cell, hipStream_t stream) {
    HIP_CHECK(candidate::launch(cell.candidate_args(), stream));
}

void validate_activation(const Cell& cell, const fp8::Fp8ActivationWorkspace& activation) {
    std::vector<std::uint8_t> actual_codes(activation.code_bytes);
    std::vector<float> actual_scales(cell.tokens);
    HIP_CHECK(hipMemcpy(actual_codes.data(), activation.codes, actual_codes.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_scales.data(), activation.scales,
                        actual_scales.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (std::uint32_t token = 0; token < cell.tokens; ++token) {
        float maximum = 0.0F;
        for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
            maximum = std::max(maximum, std::fabs(static_cast<float>(
                cell.input_host[static_cast<std::size_t>(token) * candidate::kColumns + k])));
        }
        const float scale = maximum / 448.0F;
        require(std::bit_cast<std::uint32_t>(actual_scales[token]) ==
                    std::bit_cast<std::uint32_t>(scale),
                "activation scale differs from independent BF16-source oracle");
        for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
            const float value = static_cast<float>(
                cell.input_host[static_cast<std::size_t>(token) * candidate::kColumns + k]);
            require(actual_codes[static_cast<std::size_t>(token) * candidate::kColumns + k] ==
                        oracle::encode_e4m3fn(value / scale),
                    "activation code differs from independent BF16-source oracle");
        }
    }
}

std::uint32_t validate_outputs(const Cell& cell, const hip_bfloat16* device_output,
                               const fp8::Fp8ActivationWorkspace& activation) {
    std::vector<std::uint8_t> codes(activation.code_bytes);
    std::vector<float> scales(cell.tokens);
    HIP_CHECK(hipMemcpy(codes.data(), activation.codes, codes.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(scales.data(), activation.scales, scales.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    std::vector<std::uint16_t> complete(static_cast<std::size_t>(cell.tokens) * candidate::kRows);
    HIP_CHECK(hipMemcpy(complete.data(), device_output,
                        complete.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for (const std::uint16_t value : complete) {
        require((value & 0x7f80U) != 0x7f80U,
                "candidate output contains unwritten poison or a nonfinite value");
    }

    std::vector<std::uint32_t> rows;
    rows.reserve(candidate::kRows / candidate::kRowsPerBlock + 6U);
    for (std::uint32_t tile = 0; tile < candidate::kRows / candidate::kRowsPerBlock; ++tile) {
        rows.push_back(tile * candidate::kRowsPerBlock + (tile * 17U % 256U));
    }
    for (const std::uint32_t boundary : {0U, 63U, 64U, 255U, 256U,
                                         candidate::kRows - 1U}) {
        rows.push_back(boundary);
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

    std::uint32_t maximum_steps = 0;
    for (std::uint32_t token = 0; token < cell.tokens; ++token) {
        for (const std::uint32_t row : rows) {
            const std::uint16_t actual =
                complete[static_cast<std::size_t>(token) * candidate::kRows + row];
            double dot = 0.0;
            for (std::uint32_t k = 0; k < candidate::kColumns; ++k) {
                dot += static_cast<double>(oracle::decode_e4m3fn(
                           codes[static_cast<std::size_t>(token) * candidate::kColumns + k])) *
                       static_cast<double>(oracle::decode_e4m3fn(
                           cell.weight_host[static_cast<std::size_t>(row) *
                                            candidate::kColumns + k]));
            }
            const std::uint16_t expected = round_bf16(
                dot * static_cast<double>(scales[token]) *
                static_cast<double>(cell.weight_scale_host[row]));
            maximum_steps = std::max(maximum_steps, steps(actual, expected));
        }
    }
    require(maximum_steps <= 1U, "complete route differs from independent FP64 oracle");
    return maximum_steps;
}

void regression(Cell& cell, Stream& stream) {
    candidate::Resources resources{};
    HIP_CHECK(candidate::resources(&resources));
    require(resources.registers <= 64 && resources.static_lds_bytes == 0 &&
                resources.private_bytes == 0 && resources.maximum_threads >=
                    static_cast<int>(candidate::kThreads) &&
                resources.active_blocks_per_cu >= 1,
            "candidate runtime resource gate failed");
    HIP_CHECK(hipMemset(cell.candidate_output.get(), 0xa5, cell.candidate_output.bytes()));
    auto malformed = cell.candidate_args();
    malformed.tokens = 4U;
    require(candidate::launch(malformed, stream.get()) == hipErrorInvalidValue,
            "unsupported token width was accepted");
    malformed = cell.candidate_args();
    --malformed.columns;
    require(candidate::launch(malformed, stream.get()) == hipErrorInvalidValue,
            "malformed K was accepted");
    malformed = cell.candidate_args();
    malformed.output_bf16 = const_cast<hip_bfloat16*>(malformed.input_bf16);
    require(candidate::launch(malformed, stream.get()) == hipErrorInvalidValue,
            "aliased output was accepted");
    malformed = cell.candidate_args();
    malformed.weight_codes = malformed.activation.codes;
    require(candidate::launch(malformed, stream.get()) == hipErrorInvalidValue,
            "weight/activation-code overlap was accepted");
    malformed = cell.candidate_args();
    malformed.weight_scales = malformed.activation.scales;
    require(candidate::launch(malformed, stream.get()) == hipErrorInvalidValue,
            "weight/activation-scale overlap was accepted");
    malformed = cell.candidate_args();
    malformed.activation.status = reinterpret_cast<std::uint32_t*>(malformed.activation.codes);
    require(candidate::launch(malformed, stream.get()) == hipErrorInvalidValue,
            "overlapping activation workspace fields were accepted");
    std::uint16_t sentinel = 0;
    HIP_CHECK(hipMemcpy(&sentinel,
                        cell.candidate_output.get() + Cell::kOutputGuardElements,
                        sizeof(sentinel),
                        hipMemcpyDeviceToHost));
    require(sentinel == 0xa5a5U, "rejected candidate clobbered output");

    launch_candidate(cell, stream.get());
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    validate_activation(cell, cell.candidate_activation);
    (void)validate_outputs(cell,
                           cell.candidate_output.get() + Cell::kOutputGuardElements,
                           cell.candidate_activation);

    std::array<std::uint16_t, 2> candidate_canaries{};
    HIP_CHECK(hipMemcpy(&candidate_canaries[0],
                        cell.candidate_output.get() + Cell::kOutputGuardElements - 1U,
                        sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&candidate_canaries[1],
                        cell.candidate_output.get() + Cell::kOutputGuardElements +
                            static_cast<std::size_t>(cell.tokens) * candidate::kRows,
                        sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    require(candidate_canaries[0] == 0xa5a5U && candidate_canaries[1] == 0xa5a5U,
            "candidate wrote outside its output allocation");

    HIP_CHECK(hipMemset(cell.incumbent_output.get(), 0xa5, cell.incumbent_output.bytes()));
    launch_incumbent(cell, stream.get());
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    validate_activation(cell, cell.incumbent_activation);
    (void)validate_outputs(cell,
                           cell.incumbent_output.get() + Cell::kOutputGuardElements,
                           cell.incumbent_activation);
    std::array<std::uint16_t, 2> incumbent_canaries{};
    HIP_CHECK(hipMemcpy(&incumbent_canaries[0],
                        cell.incumbent_output.get() + Cell::kOutputGuardElements - 1U,
                        sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&incumbent_canaries[1],
                        cell.incumbent_output.get() + Cell::kOutputGuardElements +
                            static_cast<std::size_t>(cell.tokens) * candidate::kRows,
                        sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    require(incumbent_canaries[0] == 0xa5a5U && incumbent_canaries[1] == 0xa5a5U,
            "incumbent wrote outside its output allocation");

    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream.get(), hipStreamCaptureModeGlobal));
    launch_candidate(cell, stream.get());
    HIP_CHECK(hipStreamEndCapture(stream.get(), &graph));
    HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipMemsetAsync(cell.candidate_output.get() + Cell::kOutputGuardElements,
                             0xa5,
                             static_cast<std::size_t>(cell.tokens) * candidate::kRows *
                                 sizeof(hip_bfloat16),
                             stream.get()));
    HIP_CHECK(hipGraphLaunch(executable, stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    (void)validate_outputs(cell,
                           cell.candidate_output.get() + Cell::kOutputGuardElements,
                           cell.candidate_activation);
    (void)hipGraphExecDestroy(executable);
    (void)hipGraphDestroy(graph);

    const hip_bfloat16 infinity(std::numeric_limits<float>::infinity());
    HIP_CHECK(hipMemcpy(cell.input.get(), &infinity, sizeof(infinity), hipMemcpyHostToDevice));
    launch_candidate(cell, stream.get());
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    std::uint32_t status = 0;
    HIP_CHECK(hipMemcpy(&status, cell.candidate_activation.status, sizeof(status),
                        hipMemcpyDeviceToHost));
    std::uint16_t first = 0, last = 0;
    HIP_CHECK(hipMemcpy(&first,
                        cell.candidate_output.get() + Cell::kOutputGuardElements,
                        sizeof(first),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&last, cell.candidate_output.get() + Cell::kOutputGuardElements +
                        static_cast<std::size_t>(cell.tokens) * candidate::kRows - 1U,
                        sizeof(last), hipMemcpyDeviceToHost));
    require(status == fp8::Fp8ActivationNonfinite && first == 0x7fc0U && last == 0x7fc0U,
            "nonfinite status did not poison the complete candidate output");
    HIP_CHECK(hipMemcpy(cell.input.get(), cell.input_host.data(), cell.input.bytes(),
                        hipMemcpyHostToDevice));
}

template <typename Launch>
float timed(Launch&& launch, hipStream_t stream, Event& begin, Event& end) {
    HIP_CHECK(hipEventRecord(begin.get(), stream));
    launch();
    HIP_CHECK(hipEventRecord(end.get(), stream));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float result = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&result, begin.get(), end.get()));
    return result;
}

double median(std::array<float, 14> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (values[6] + values[7]);
}

void benchmark(Cell& cell, Stream& stream) {
    Event begin, end;
    std::array<float, 14> incumbent{}, challenger{};
    for (int warmup = 0; warmup < 2; ++warmup) {
        launch_incumbent(cell, stream.get());
        launch_candidate(cell, stream.get());
    }
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    for (std::size_t trial = 0; trial < 14; ++trial) {
        const auto incumbent_call = [&] { launch_incumbent(cell, stream.get()); };
        const auto candidate_call = [&] { launch_candidate(cell, stream.get()); };
        if ((trial & 1U) == 0U) {
            incumbent[trial] = timed(incumbent_call, stream.get(), begin, end);
            challenger[trial] = timed(candidate_call, stream.get(), begin, end);
        } else {
            challenger[trial] = timed(candidate_call, stream.get(), begin, end);
            incumbent[trial] = timed(incumbent_call, stream.get(), begin, end);
        }
    }
    const double incumbent_ms = median(incumbent), challenger_ms = median(challenger);
    const auto array = [](const std::array<float, 14>& values) {
        std::cout << '[';
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << values[index];
        }
        std::cout << ']';
    };
    std::cout << "{\"schema\":\"ninfer.r9700.fp8_target_gate_up_small_t_cell.v1\","
              << "\"production_routing_changed\":false,\"tokens\":" << cell.tokens
              << ",\"rows\":34816,\"columns\":5120,\"timing_scope\":"
              << "\"complete_bf16_quantize_matmul_status_poison\","
              << "\"incumbent_samples_ms\":";
    array(incumbent);
    std::cout << ",\"challenger_samples_ms\":";
    array(challenger);
    std::cout << ",\"incumbent_median_ms\":" << incumbent_ms
              << ",\"challenger_median_ms\":" << challenger_ms
              << ",\"challenger_over_incumbent\":" << challenger_ms / incumbent_ms
              << ",\"projected_64_call_saving_ms\":"
              << 64.0 * (incumbent_ms - challenger_ms)
              << ",\"decision\":\"unmade_until_gpu_measurement_is_validated\"}\n";
}

void require_target() {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    require(std::string_view(properties.name) == "AMD Radeon AI PRO R9700" &&
                std::string_view(properties.gcnArchName) == "gfx1201" &&
                properties.warpSize == 32,
            "qualifier requires device-0 R9700 gfx1201 wave32");
}

} // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "usage: fp8_gate_up_small_t_harness --regression|--benchmark");
    const std::string_view mode(argv[1]);
    require(mode == "--regression" || mode == "--benchmark", "unknown mode");
    require_target();
    Stream stream;
    for (const std::uint32_t tokens : {5U, 6U}) {
        Cell cell(tokens);
        regression(cell, stream);
        if (mode == "--benchmark") benchmark(cell, stream);
    }
    if (mode == "--regression") std::cout << "FP8 target gate-up T5/T6 regression PASS\n";
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "fp8_gate_up_small_t_harness: " << error.what() << '\n';
    return EXIT_FAILURE;
}
