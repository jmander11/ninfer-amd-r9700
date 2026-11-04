#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "qualification_device_guard.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

// Every logical dot is evaluated in FP64 from original BF16 input words and the
// exact represented BF16 weight, without production staging casts.
void qualify(std::uint32_t n, std::uint32_t k, std::uint32_t t, bool offset = false) {
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(t) * k);
    std::vector<hip_bfloat16> weights(static_cast<std::size_t>(n) * k);
    for (std::uint32_t token = 0; token < t; ++token)
        for (std::uint32_t col = 0; col < k; ++col)
            input[static_cast<std::size_t>(token) * k + col] = hip_bfloat16(
                static_cast<float>(static_cast<int>((col * 13U + (token % 7U) * 7U) % 71U) - 35) /
                16.0F);
    for (std::uint32_t row = 0; row < n; ++row)
        for (std::uint32_t col = 0; col < k; ++col)
            weights[static_cast<std::size_t>(row) * k + col] = hip_bfloat16(
                static_cast<float>(static_cast<int>((row * 19U + col * 29U) % 255U) - 127) *
                0.001123F);
    const std::size_t displacement = offset ? 2U : 0U;
    ninfer::DeviceBuffer dx(input.size() * 2U + displacement),
        dw(weights.size() * 2U + displacement), dy(static_cast<std::size_t>(n) * t * 2U);
    auto* input_data  = static_cast<char*>(dx.data()) + displacement;
    auto* weight_data = static_cast<char*>(dw.data()) + displacement;
    check(hipMemcpy(input_data, input.data(), input.size() * 2U, hipMemcpyHostToDevice));
    check(hipMemcpy(weight_data, weights.data(), weights.size() * 2U, hipMemcpyHostToDevice));
    ninfer::Weight weight{};
    weight.payload = weight.qdata = weight_data;
    weight.payload_bytes = weight.qdata_bytes = weights.size() * 2U;
    weight.qtype                              = ninfer::QType::BF16_CTRL;
    weight.layout                             = ninfer::QuantLayout::Contiguous;
    weight.ndim                               = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = static_cast<std::int32_t>(n);
    weight.k = weight.shape[1] = weight.padded_shape[1] = static_cast<std::int32_t>(k);
    ninfer::Tensor x(input_data, ninfer::DType::BF16, {static_cast<int>(k), static_cast<int>(t)});
    ninfer::Tensor y(dy.data(), ninfer::DType::BF16, {static_cast<int>(n), static_cast<int>(t)});
    hipStream_t stream{};
    check(hipStreamCreate(&stream));
    ninfer::ops::linear(x, weight, y, stream);
    check(hipStreamSynchronize(stream));
    std::vector<hip_bfloat16> actual(static_cast<std::size_t>(n) * t);
    dy.copy_to_host(actual.data(), actual.size() * 2U);
    double error2 = 0.0, reference2 = 0.0, max_error = 0.0, max_reference = 0.0;
    // The represented fixture repeats every seven tokens. Evaluate its complete FP64 dot
    // once per distinct input, then check every output at the larger execution extents.
    std::vector<double> expected_rows(static_cast<std::size_t>(std::min(t, 7U)) * n);
    for (std::uint32_t token = 0; token < t; ++token) {
        for (std::uint32_t row = 0; row < n; ++row) {
            double expected = 0.0;
            if (token < 7U) {
                for (std::uint32_t col = 0; col < k; ++col) {
                    expected += static_cast<double>(
                                    float(input[static_cast<std::size_t>(token) * k + col])) *
                                static_cast<double>(
                                    float(weights[static_cast<std::size_t>(row) * k + col]));
                }
                expected_rows[static_cast<std::size_t>(token) * n + row] = expected;
            } else {
                expected = expected_rows[static_cast<std::size_t>(token % 7U) * n + row];
            }
            const double got = float(actual[static_cast<std::size_t>(token) * n + row]);
            if (!std::isfinite(got)) throw std::runtime_error("nonfinite output");
            const double error = got - expected;
            error2 += error * error;
            reference2 += expected * expected;
            max_error     = std::max(max_error, std::abs(error));
            max_reference = std::max(max_reference, std::abs(expected));
        }
    }
    const double relative = std::sqrt(error2 / reference2);
    if (relative > 0.006 || max_error > 0.01 * max_reference + 1.0e-5)
        throw std::runtime_error("public-input FP64 criterion failed");
    if (((n == 1152U && k == 4304U) || (n == 5120U && k == 4608U)) && t == 33U) {
        hipGraph_t graph{};
        hipGraphExec_t executable{};
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        ninfer::ops::linear(x, weight, y, stream);
        check(hipStreamEndCapture(stream, &graph));
        check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        check(hipGraphLaunch(executable, stream));
        check(hipGraphLaunch(executable, stream));
        check(hipStreamSynchronize(stream));
        std::vector<hip_bfloat16> replay(actual.size());
        dy.copy_to_host(replay.data(), replay.size() * 2U);
        if (std::memcmp(actual.data(), replay.data(), actual.size() * 2U) != 0)
            throw std::runtime_error("graph replay differs");
        check(hipGraphExecDestroy(executable));
        check(hipGraphDestroy(graph));
        input.back() = hip_bfloat16(std::numeric_limits<float>::quiet_NaN());
        check(hipMemcpy(input_data, input.data(), input.size() * 2U, hipMemcpyHostToDevice));
        ninfer::ops::linear(x, weight, y, stream);
        check(hipStreamSynchronize(stream));
        dy.copy_to_host(replay.data(), replay.size() * 2U);
        for (std::uint32_t row = 0; row < n; ++row)
            if (!std::isnan(float(replay[static_cast<std::size_t>(t - 1U) * n + row])))
                throw std::runtime_error("NaN input was masked");
    }
    check(hipStreamDestroy(stream));
    std::cout << "vision-bf16 PASS N=" << n << " K=" << k << " T=" << t
              << " offset=" << displacement << " fp64_relative_l2=" << relative << '\n';
}
} // namespace

int main() {
    try {
        setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 1);
        setenv("HSA_USE_SVM", "1", 1);
        ninfer::r9700_qualification::DeviceGuard guard;
        guard.select();
        constexpr std::array<std::array<std::uint32_t, 2>, 7> shapes{{{1152, 1536},
                                                                      {3456, 1152},
                                                                      {1152, 1152},
                                                                      {4304, 1152},
                                                                      {1152, 4304},
                                                                      {4608, 4608},
                                                                      {5120, 4608}}};
        for (const auto& shape : shapes)
            for (const auto t : {9U, 17U, 33U, 129U, 256U, 512U, 1024U, 1025U})
                qualify(shape[0], shape[1], t);
        for (const auto& shape : shapes) qualify(shape[0], shape[1], 33U, true);
        guard.check("after-Vision-BF16-qualification");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
