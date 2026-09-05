#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDimensions = 1152;
constexpr int kRows = 2304;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

float represented(hip_bfloat16 value) { return static_cast<float>(value); }

std::vector<hip_bfloat16> make_table() {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(kDimensions) * kRows);
    for (std::size_t index = 0; index < result.size(); ++index) {
        const int code = static_cast<int>((index * 37U + index / kDimensions * 11U) % 251U) - 125;
        result[index] = hip_bfloat16(static_cast<float>(code) / 96.0F);
    }
    return result;
}

void run_case(const std::vector<hip_bfloat16>& table, int patches, hipStream_t stream) {
    std::vector<hip_bfloat16> x(static_cast<std::size_t>(kDimensions) * patches);
    std::vector<std::int32_t> indices(static_cast<std::size_t>(4) * patches);
    std::vector<float> weights(static_cast<std::size_t>(4) * patches);
    for (std::size_t index = 0; index < x.size(); ++index) {
        const int code = static_cast<int>((index * 19U + 7U) % 191U) - 95;
        x[index] = hip_bfloat16(static_cast<float>(code) / 32.0F);
    }
    for (int patch = 0; patch < patches; ++patch) {
        float normalizer = 0.0F;
        for (int corner = 0; corner < 4; ++corner) {
            const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
            indices[control] = (patch * 37 + corner * 101 + 11) % kRows;
            weights[control] = static_cast<float>(((patch + 3) * (corner + 5)) % 17 + 2);
            normalizer += weights[control];
        }
        for (int corner = 0; corner < 4; ++corner) {
            weights[static_cast<std::size_t>(patch) * 4 + corner] /= normalizer;
        }
    }

    const std::vector<hip_bfloat16> original_x = x;
    std::vector<double> oracle(x.size());
    for (int patch = 0; patch < patches; ++patch) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            double value = static_cast<double>(represented(x[static_cast<std::size_t>(patch) *
                                                              kDimensions + dimension]));
            for (int corner = 0; corner < 4; ++corner) {
                const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
                value += static_cast<double>(represented(
                             table[static_cast<std::size_t>(indices[control]) * kDimensions +
                                   dimension])) *
                         static_cast<double>(weights[control]);
            }
            oracle[static_cast<std::size_t>(patch) * kDimensions + dimension] = value;
        }
    }

    ninfer::DeviceBuffer d_table(table.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_indices(indices.size() * sizeof(std::int32_t));
    ninfer::DeviceBuffer d_weights(weights.size() * sizeof(float));
    ninfer::DeviceBuffer d_x(x.size() * sizeof(hip_bfloat16));
    d_table.copy_from_host(table.data(), d_table.size());
    d_indices.copy_from_host(indices.data(), d_indices.size());
    d_weights.copy_from_host(weights.data(), d_weights.size());
    d_x.copy_from_host(x.data(), d_x.size());
    ninfer::Tensor table_tensor(d_table.data(), ninfer::DType::BF16, {kDimensions, kRows});
    ninfer::Tensor index_tensor(d_indices.data(), ninfer::DType::I32, {4, patches});
    ninfer::Tensor weight_tensor(d_weights.data(), ninfer::DType::FP32, {4, patches});
    ninfer::Tensor x_tensor(d_x.data(), ninfer::DType::BF16, {kDimensions, patches});

    ninfer::ops::vision_pos_embed_add(table_tensor, index_tensor, weight_tensor, x_tensor, stream);
    d_x.copy_from_host_async(original_x.data(), d_x.size(), stream);
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin, stream));
    ninfer::ops::vision_pos_embed_add(table_tensor, index_tensor, weight_tensor, x_tensor, stream);
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed_ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));

    d_x.copy_to_host(x.data(), d_x.size());
    double squared_error = 0.0;
    double squared_oracle = 0.0;
    double max_abs = 0.0;
    for (std::size_t index = 0; index < x.size(); ++index) {
        const double actual = represented(x[index]);
        const double error = actual - oracle[index];
        squared_error += error * error;
        squared_oracle += oracle[index] * oracle[index];
        max_abs = std::max(max_abs, std::abs(error));
        if (std::abs(error) > 1.0e-5 + 3.95e-3 * std::abs(oracle[index])) {
            fail("vision_pos_embed pointwise FP64-oracle criterion failed at P=" +
                 std::to_string(patches) + " index=" + std::to_string(index));
        }
    }
    std::vector<hip_bfloat16> preserved_table(table.size());
    std::vector<std::int32_t> preserved_indices(indices.size());
    std::vector<float> preserved_weights(weights.size());
    d_table.copy_to_host(preserved_table.data(), d_table.size());
    d_indices.copy_to_host(preserved_indices.data(), d_indices.size());
    d_weights.copy_to_host(preserved_weights.data(), d_weights.size());
    if (preserved_table != table || preserved_indices != indices || preserved_weights != weights) {
        fail("vision_pos_embed modified a read-only input");
    }
    const double relative_l2 = std::sqrt(squared_error / std::max(squared_oracle, 1.0e-30));
    std::cout << "vision_pos_embed P=" << patches << " max_abs=" << max_abs
              << " rel_l2=" << relative_l2 << " time_ms=" << elapsed_ms << '\n';
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            fail("vision_pos_embed qualifier requires wave32 gfx1201");
        }
        const std::vector<hip_bfloat16> table = make_table();
        run_case(table, 17, device.stream);
        run_case(table, 1024, device.stream);
        std::cout << "vision_pos_embed: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "vision_pos_embed: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
