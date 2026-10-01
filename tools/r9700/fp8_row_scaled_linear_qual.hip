// Qualification of the row-scaled E4M3 Linear (LinearExecution): the small-T route (T <= 16,
// verification and ordinary decode) and the prefill GEMM route (T > 16) on the real Qwen3.8
// protected projection shapes. Evidence: the activation planes the route consumed are read back and
// an independent FP64 product of the decoded E4M3 weight and activation codes with their FP32
// row/token scales checks every output element (small T) or sampled tokens and rows (prefill);
// output guards; and a nonfinite input token publishes the canonical BF16 NaN in exactly its row
// while every other token's output is unchanged. At prefill widths the same rows also run through
// fp8lut4_linear (raw staging) with a split output and with residual accumulation, which must
// equal the plain route bitwise (BF16(residual + BF16(y)) for accumulation).
#include "ninfer/types.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_small_t_linear.h"
#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/linear_execution.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

// Every E4M3 code's exact value, decoded once by the independent oracle.
const std::vector<double>& e4m3_values() {
    static const std::vector<double> values = [] {
        std::vector<double> table(256);
        for (std::uint32_t code = 0; code < 256U; ++code)  // 0x7f/0xff are NaN, never produced
            table[code] = (code & 0x7fU) == 0x7fU
                ? std::numeric_limits<double>::quiet_NaN()
                : oracle::decode_e4m3fn(static_cast<std::uint8_t>(code));
        return table;
    }();
    return values;
}

constexpr std::size_t kGuard = 64;
constexpr std::uint16_t kGuardBits = 0x7bcdU;
constexpr std::uint16_t kCanonicalNan = 0x7fc0U;

double run_shape(std::uint32_t rows, std::uint32_t columns, std::uint32_t tokens, Rng& rng) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows) * columns);
    std::vector<float> row_scales(rows);
    // Codes of uniform(-3, 3) values, drawn from a pre-encoded pool (the real shapes need up to
    // 178M codes per call).
    std::vector<std::uint8_t> pool(4096);
    for (auto& code : pool) code = oracle::encode_e4m3fn(static_cast<float>((rng.uniform() - 0.5) * 6.0));
    for (auto& code : codes) code = pool[rng.next() & 4095U];
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
    ninfer::ops::LinearExecution execution(weight, storage_bytes);
    execution.bind_storage(storage.data, storage_bytes);
    const std::size_t outputs = static_cast<std::size_t>(tokens) * rows;
    Device<hip_bfloat16> d_output(outputs + 2U * kGuard);
    std::vector<std::uint16_t> fill(outputs + 2U * kGuard, kGuardBits);
    const auto run = [&] {
        check(hipMemcpy(d_output.data, fill.data(), d_output.bytes(), hipMemcpyHostToDevice), "fill");
        check(execution.run(tokens, d_input.data, d_output.data + kGuard, nullptr),
              "row-scaled E4M3 run");
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
    // Every element at small T; at prefill widths the first, a middle, the last and the tile-edge
    // tokens over every seventh row.
    std::vector<std::uint32_t> checked_tokens;
    if (tokens <= fp8::kFp8SmallTokenLimit) {
        for (std::uint32_t token = 0; token < tokens; ++token) checked_tokens.push_back(token);
    } else {
        for (const std::uint32_t token : {0U, tokens / 2U, 255U, 256U, tokens - 1U})
            if (token < tokens) checked_tokens.push_back(token);
    }
    const std::uint32_t row_step = tokens <= fp8::kFp8SmallTokenLimit ? 1U : 7U;
    const std::vector<double>& decode = e4m3_values();
    double worst = 0.0;
    for (const std::uint32_t token : checked_tokens) {
        for (std::uint32_t row = 0; row < rows; row += row_step) {
            double sum = 0.0, magnitude = 0.0;
            for (std::uint32_t k = 0; k < columns; ++k) {
                const double term =
                    decode[codes[static_cast<std::size_t>(row) * columns + k]] *
                    decode[activation[static_cast<std::size_t>(token) * columns + k]];
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
    if (fp8::fp8_row_scaled_projection_supported(tokens, rows, columns)) {
        const fp8::Fp8Lut4Weight raw{image.data, nullptr,
                                     reinterpret_cast<const float*>(image.data + codes.size()), rows,
                                     columns};
        // Split at a 128-row boundary inside the rows.
        const std::uint32_t split = rows / 2U / 128U * 128U;
        Device<hip_bfloat16> leading(static_cast<std::size_t>(tokens) * split);
        Device<hip_bfloat16> trailing(static_cast<std::size_t>(tokens) * (rows - split));
        check(fp8::fp8lut4_linear(raw, planes,
                                  {.leading = leading.data, .trailing = trailing.data, .split = split},
                                  nullptr),
              "raw split run");
        check(hipDeviceSynchronize(), "sync split");
        std::vector<std::uint16_t> lead(static_cast<std::size_t>(tokens) * split),
            trail(static_cast<std::size_t>(tokens) * (rows - split));
        check(hipMemcpy(lead.data(), leading.data, leading.bytes(), hipMemcpyDeviceToHost), "lead");
        check(hipMemcpy(trail.data(), trailing.data, trailing.bytes(), hipMemcpyDeviceToHost), "trail");
        for (std::uint32_t token = 0; token < tokens; ++token)
            for (std::uint32_t row = 0; row < rows; ++row) {
                const std::uint16_t got = row < split
                    ? lead[static_cast<std::size_t>(token) * split + row]
                    : trail[static_cast<std::size_t>(token) * (rows - split) + row - split];
                require(got == output[kGuard + static_cast<std::size_t>(token) * rows + row],
                        "raw split output differs from the plain route");
            }
        // Accumulation into a random BF16 residual.
        std::vector<std::uint16_t> base(outputs);
        for (auto& bits : base) {
            const hip_bfloat16 value(static_cast<float>((rng.uniform() - 0.5) * 8.0));
            std::memcpy(&bits, &value, sizeof(bits));
        }
        Device<hip_bfloat16> residual(outputs);
        check(hipMemcpy(residual.data, base.data(), residual.bytes(), hipMemcpyHostToDevice), "base");
        check(fp8::fp8lut4_linear(raw, planes, {.leading = residual.data, .accumulate = true}, nullptr),
              "raw accumulate run");
        check(hipDeviceSynchronize(), "sync accumulate");
        std::vector<std::uint16_t> summed(outputs);
        check(hipMemcpy(summed.data(), residual.data, residual.bytes(), hipMemcpyDeviceToHost), "sum");
        const auto as_float = [](std::uint16_t bits) {
            const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
            float value;
            std::memcpy(&value, &word, sizeof(value));
            return value;
        };
        for (std::size_t index = 0; index < outputs; ++index) {
            const hip_bfloat16 expected(as_float(base[index]) + as_float(output[kGuard + index]));
            std::uint16_t expected_bits;
            std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
            require(summed[index] == expected_bits,
                    "raw accumulate output differs from BF16(residual + plain output)");
        }
    }
    // A nonfinite represented input poisons exactly its token's output row.
    const std::uint32_t bad_token = tokens / 2U;
    input[static_cast<std::size_t>(bad_token) * columns + 3U] =
        hip_bfloat16(std::numeric_limits<float>::infinity());
    check(hipMemcpy(d_input.data, input.data(), d_input.bytes(), hipMemcpyHostToDevice), "poison input");
    const std::vector<std::uint16_t> poisoned = run();
    for (std::size_t index = 0; index < outputs; ++index) {
        const bool bad = index / rows == bad_token;
        require(bad ? poisoned[kGuard + index] == kCanonicalNan
                    : poisoned[kGuard + index] == output[kGuard + index],
                "nonfinite input did not poison exactly its token");
    }
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
        // Two-tile small route (T 17..32) on every protected projection shape.
        for (const auto [rows, columns] : {std::pair<std::uint32_t, std::uint32_t>{7168U, 5120U},
                                           {5120U, 6144U}, {34816U, 5120U}, {5120U, 17408U},
                                           {4096U, 5120U}}) {
            for (const std::uint32_t tokens : {17U, 24U, 32U})
                worst = std::max(worst, run_shape(rows, columns, tokens, rng));
        }
        // Prefill route: partial and full token tiles on every protected projection shape.
        for (const auto [rows, columns] : {std::pair<std::uint32_t, std::uint32_t>{7168U, 5120U},
                                           {5120U, 6144U}, {34816U, 5120U}, {5120U, 17408U},
                                           {4096U, 5120U}}) {
            for (const std::uint32_t tokens : {33U, 64U, 100U, 128U, 129U, 200U, 300U, 640U, 2048U, 2600U})
                worst = std::max(worst, run_shape(rows, columns, tokens, rng));
        }
        std::cout << "fp8_row_scaled_linear: PASS small T=1..32 and prefill T=33/64/100/128/129/200/300/640/2048/2600 on "
                     "N7168/K5120, N5120/K6144, N34816/K5120, N5120/K17408, N4096/K5120 "
                     "maximum_normalized_error="
                  << worst << " nonfinite_poison=PASS guards=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fp8_row_scaled_linear: FAIL: " << error.what() << '\n';
        return 1;
    }
}
