// Qualification of the small-T (T <= 16) row-scaled E4M3 Linear route used by verification and
// ordinary decode: the public LinearExecution run on the real Qwen3.8 attention projection shapes
// (query-key/gate-value N7168/K5120 and output N5120/K6144). Evidence: the activation planes the
// route consumed are read back and an independent FP64 product of the decoded E4M3 weight and
// activation codes with their FP32 row/token scales checks every output element; output guards;
// and a nonfinite input poisons every element with the canonical BF16 NaN.
#include "ninfer/types.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/linear_execution.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fp8 = ninfer::ops::r9700::linear;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;

void check(hipError_t error, const char* what) {
    if (error != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
}
void require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) { check(hipMalloc(&data, n * sizeof(T)), "malloc"); }
    ~Device() { (void)hipFree(data); }
    Device(const Device&) = delete;
    std::size_t bytes() const { return count * sizeof(T); }
};

struct Rng {
    std::uint64_t state;
    std::uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 32U);
    }
    double uniform() { return (next() + 0.5) / 4294967296.0; }
};

constexpr std::size_t kGuard = 64;
constexpr std::uint16_t kGuardBits = 0x7bcdU;
constexpr std::uint16_t kCanonicalNan = 0x7fc0U;

double run_shape(std::uint32_t rows, std::uint32_t columns, std::uint32_t tokens, Rng& rng) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows) * columns);
    std::vector<float> row_scales(rows);
    for (auto& code : codes) code = oracle::encode_e4m3fn(static_cast<float>((rng.uniform() - 0.5) * 6.0));
    for (auto& scale : row_scales) scale = static_cast<float>(std::ldexp(0.5 + rng.uniform(), -8));
    // One contiguous persistent image: codes followed by row scales.
    Device<std::uint8_t> image(codes.size() + rows * sizeof(float));
    check(hipMemcpy(image.data, codes.data(), codes.size(), hipMemcpyHostToDevice), "codes");
    check(hipMemcpy(image.data + codes.size(), row_scales.data(), rows * sizeof(float),
                    hipMemcpyHostToDevice),
          "scales");
    ninfer::Weight weight{};
    weight.payload = weight.qdata = image.data;
    weight.scales = image.data + codes.size();
    weight.payload_bytes = image.bytes();
    weight.qdata_bytes = codes.size();
    weight.scale_bytes = rows * sizeof(float);
    weight.qtype = ninfer::QType::F8E4M3_ROW_F32S;
    weight.layout = ninfer::QuantLayout::RowScaled;
    weight.scale_dtype = ninfer::DType::FP32;
    weight.n = weight.shape[0] = weight.padded_shape[0] = static_cast<std::int32_t>(rows);
    weight.k = weight.shape[1] = weight.padded_shape[1] = static_cast<std::int32_t>(columns);
    weight.ndim = 2;
    weight.scale_ne[0] = static_cast<std::int32_t>(rows);
    weight.scale_nb[0] = sizeof(float);

    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const double amplitude = 0.05 + 6.0 * rng.uniform();
        for (std::uint32_t k = 0; k < columns; ++k)
            input[static_cast<std::size_t>(token) * columns + k] =
                hip_bfloat16(static_cast<float>((rng.uniform() - 0.5) * amplitude));
    }
    Device<hip_bfloat16> d_input(input.size());
    check(hipMemcpy(d_input.data, input.data(), d_input.bytes(), hipMemcpyHostToDevice), "input");
    const std::size_t storage_bytes =
        ninfer::ops::LinearExecution::activation_workspace_capacity_bytes(tokens, columns);
    Device<std::uint8_t> storage(storage_bytes);
    ninfer::ops::LinearExecutionContext context;
    ninfer::ops::LinearExecution execution(context, weight, storage_bytes, 0U);
    execution.bind_storage(storage.data, storage_bytes, nullptr, 0U);
    (void)execution.prepare(tokens);
    const std::size_t outputs = static_cast<std::size_t>(tokens) * rows;
    Device<hip_bfloat16> d_output(outputs + 2U * kGuard);
    std::vector<std::uint16_t> fill(outputs + 2U * kGuard, kGuardBits);
    const auto run = [&] {
        check(hipMemcpy(d_output.data, fill.data(), d_output.bytes(), hipMemcpyHostToDevice), "fill");
        const auto status = execution.run(tokens, d_input.data, d_output.data + kGuard, nullptr);
        require(status.ok(), "row-scaled E4M3 run failed");
        check(hipDeviceSynchronize(), "sync");
        std::vector<std::uint16_t> host(fill.size());
        check(hipMemcpy(host.data(), d_output.data, d_output.bytes(), hipMemcpyDeviceToHost), "read");
        for (std::size_t index = 0; index < kGuard; ++index)
            require(host[index] == kGuardBits && host[host.size() - 1U - index] == kGuardBits,
                    "output guard overwritten");
        return host;
    };
    const std::vector<std::uint16_t> output = run();
    fp8::Fp8ActivationWorkspace planes{};
    check(fp8::fp8_bind_activation_workspace(storage.data, storage_bytes, tokens, columns, &planes),
          "bind planes");
    std::vector<std::uint8_t> activation(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> token_scales(tokens);
    check(hipMemcpy(activation.data(), planes.codes, activation.size(), hipMemcpyDeviceToHost), "codes");
    check(hipMemcpy(token_scales.data(), planes.scales, tokens * sizeof(float), hipMemcpyDeviceToHost),
          "token scales");
    double worst = 0.0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            double sum = 0.0, magnitude = 0.0;
            for (std::uint32_t k = 0; k < columns; ++k) {
                const double term =
                    oracle::decode_e4m3fn(codes[static_cast<std::size_t>(row) * columns + k]) *
                    oracle::decode_e4m3fn(activation[static_cast<std::size_t>(token) * columns + k]);
                sum += term;
                magnitude += std::abs(term);
            }
            const double scale = static_cast<double>(row_scales[row]) * token_scales[token];
            const double reference = sum * scale;
            std::uint32_t bits = static_cast<std::uint32_t>(
                output[kGuard + static_cast<std::size_t>(token) * rows + row]) << 16U;
            float actual_float;
            std::memcpy(&actual_float, &bits, sizeof(actual_float));
            // BF16 round-to-nearest (half an ulp, 2^-8 relative at a binade bottom) plus the FP32
            // accumulation bound of exact E4M3 products:
            // K/16 sequential WMMA chunk additions and the in-order quarter sum, each at most one
            // FP32 rounding of the running magnitude.
            const double accumulation = (columns / 16.0 + 8.0) * std::ldexp(1.0, -24);
            const double limit = std::abs(reference) * std::ldexp(1.0, -8) +
                                 magnitude * scale * accumulation + 1e-30;
            const double error = std::abs(static_cast<double>(actual_float) - reference) / limit;
            require(std::isfinite(error) && error <= 1.0,
                    "row-scaled E4M3 output differs from the FP64 oracle at rows=" +
                        std::to_string(rows) + " T=" + std::to_string(tokens));
            worst = std::max(worst, error);
        }
    }
    // A nonfinite represented input poisons every output element.
    input[static_cast<std::size_t>(tokens - 1U) * columns + 3U] =
        hip_bfloat16(std::numeric_limits<float>::infinity());
    check(hipMemcpy(d_input.data, input.data(), d_input.bytes(), hipMemcpyHostToDevice), "poison input");
    const std::vector<std::uint16_t> poisoned = run();
    for (std::size_t index = 0; index < outputs; ++index)
        require(poisoned[kGuard + index] == kCanonicalNan, "nonfinite input did not poison output");
    return worst;
}

} // namespace

int main() {
    try {
        Rng rng{0xf8f8ULL};
        double worst = 0.0;
        for (const auto [rows, columns] : {std::pair<std::uint32_t, std::uint32_t>{7168U, 5120U},
                                           {5120U, 6144U}}) {
            for (const std::uint32_t tokens : {1U, 2U, 5U, 6U, 8U, 16U})
                worst = std::max(worst, run_shape(rows, columns, tokens, rng));
        }
        std::cout << "fp8_row_scaled_small_t: PASS N7168/K5120 and N5120/K6144 at T=1,2,5,6,8,16 "
                     "maximum_normalized_error="
                  << worst << " nonfinite_poison=PASS guards=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fp8_row_scaled_small_t: FAIL: " << error.what() << '\n';
        return 1;
    }
}
