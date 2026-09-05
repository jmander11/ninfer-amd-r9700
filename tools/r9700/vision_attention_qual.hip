#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/vision_attention.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDim = 72;
constexpr int kHeads = 16;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

float represented(hip_bfloat16 value) { return static_cast<float>(value); }

std::size_t packed_index(int token, int head, int dimension) {
    return (static_cast<std::size_t>(token) * kHeads + head) * kDim + dimension;
}

struct Fixture {
    std::vector<hip_bfloat16> q;
    std::vector<hip_bfloat16> k;
    std::vector<hip_bfloat16> v;
};

Fixture make_fixture(int patches, int salt, bool isolate_segments,
                     const std::vector<int>& cumulative) {
    const std::size_t elements = static_cast<std::size_t>(patches) * kHeads * kDim;
    Fixture result{std::vector<hip_bfloat16>(elements), std::vector<hip_bfloat16>(elements),
                   std::vector<hip_bfloat16>(elements)};
    for (std::size_t index = 0; index < elements; ++index) {
        result.q[index] = hip_bfloat16(static_cast<float>(static_cast<int>((index * 17U + salt) % 97U) - 48) /
                                       64.0F);
        result.k[index] = hip_bfloat16(static_cast<float>(static_cast<int>((index * 29U + salt + 3) % 101U) - 50) /
                                       64.0F);
        result.v[index] = hip_bfloat16(static_cast<float>(static_cast<int>((index * 43U + salt + 7) % 109U) - 54) /
                                       32.0F);
    }
    if (isolate_segments) {
        std::fill(result.q.begin(), result.q.end(), hip_bfloat16(0.0F));
        std::fill(result.k.begin(), result.k.end(), hip_bfloat16(0.0F));
        for (std::size_t segment = 0; segment + 1 < cumulative.size(); ++segment) {
            const float value = (segment & 1U) == 0 ? 4.0F : -3.0F;
            for (int token = cumulative[segment]; token < cumulative[segment + 1]; ++token) {
                const std::size_t begin = static_cast<std::size_t>(token) * kHeads * kDim;
                std::fill_n(result.v.begin() + begin, kHeads * kDim, hip_bfloat16(value));
            }
        }
    }
    return result;
}

std::vector<double> oracle(const Fixture& fixture, const std::vector<int>& cumulative) {
    constexpr double scale = 0.117851130197757920733474060350808;
    std::vector<double> output(fixture.q.size(), 0.0);
    for (std::size_t segment = 0; segment + 1 < cumulative.size(); ++segment) {
        const int begin = cumulative[segment];
        const int end = cumulative[segment + 1];
        std::vector<double> scores(static_cast<std::size_t>(end - begin));
        for (int query = begin; query < end; ++query) {
            for (int head = 0; head < kHeads; ++head) {
                double maximum = -std::numeric_limits<double>::infinity();
                for (int key = begin; key < end; ++key) {
                    double dot = 0.0;
                    for (int dimension = 0; dimension < kDim; ++dimension) {
                        dot += static_cast<double>(represented(
                                   fixture.q[packed_index(query, head, dimension)])) *
                               static_cast<double>(represented(
                                   fixture.k[packed_index(key, head, dimension)]));
                    }
                    const double score = dot * scale;
                    scores[static_cast<std::size_t>(key - begin)] = score;
                    maximum = std::max(maximum, score);
                }
                double denominator = 0.0;
                for (double& score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (int dimension = 0; dimension < kDim; ++dimension) {
                    double numerator = 0.0;
                    for (int key = begin; key < end; ++key) {
                        numerator += scores[static_cast<std::size_t>(key - begin)] *
                                     static_cast<double>(represented(
                                         fixture.v[packed_index(key, head, dimension)]));
                    }
                    output[packed_index(query, head, dimension)] = numerator / denominator;
                }
            }
        }
    }
    return output;
}

void run_case(const std::vector<int>& lengths, bool uniform_entry, bool padded_qkv,
              bool isolate_segments, int salt, hipStream_t stream) {
    std::vector<int> cumulative(lengths.size() + 1, 0);
    std::partial_sum(lengths.begin(), lengths.end(), cumulative.begin() + 1);
    const int patches = cumulative.back();
    const std::size_t token_plane = static_cast<std::size_t>(kHeads) * kDim;
    const Fixture fixture = make_fixture(patches, salt, isolate_segments, cumulative);
    const std::vector<double> expected = oracle(fixture, cumulative);

    ninfer::DeviceBuffer q_storage;
    ninfer::DeviceBuffer k_storage;
    ninfer::DeviceBuffer v_storage;
    ninfer::DeviceBuffer interleaved;
    ninfer::Tensor q;
    ninfer::Tensor k;
    ninfer::Tensor v;
    std::vector<hip_bfloat16> expected_storage;
    if (padded_qkv) {
        expected_storage.assign(static_cast<std::size_t>(patches) * token_plane * 3,
                                hip_bfloat16(-9.0F));
        for (int token = 0; token < patches; ++token) {
            const std::size_t source = static_cast<std::size_t>(token) * token_plane;
            const std::size_t target = source * 3;
            std::copy_n(fixture.q.begin() + source, token_plane, expected_storage.begin() + target);
            std::copy_n(fixture.k.begin() + source, token_plane,
                        expected_storage.begin() + target + token_plane);
            std::copy_n(fixture.v.begin() + source, token_plane,
                        expected_storage.begin() + target + 2 * token_plane);
        }
        interleaved = ninfer::DeviceBuffer(expected_storage.size() * sizeof(hip_bfloat16));
        interleaved.copy_from_host(expected_storage.data(), interleaved.size());
        q = ninfer::Tensor(interleaved.data(), ninfer::DType::BF16, {kDim, kHeads, patches});
        q.nb[2] = static_cast<std::int64_t>(3 * token_plane * sizeof(hip_bfloat16));
        k = q;
        v = q;
        k.data = static_cast<hip_bfloat16*>(interleaved.data()) + token_plane;
        v.data = static_cast<hip_bfloat16*>(interleaved.data()) + 2 * token_plane;
    } else {
        q_storage = ninfer::DeviceBuffer(fixture.q.size() * sizeof(hip_bfloat16));
        k_storage = ninfer::DeviceBuffer(fixture.k.size() * sizeof(hip_bfloat16));
        v_storage = ninfer::DeviceBuffer(fixture.v.size() * sizeof(hip_bfloat16));
        q_storage.copy_from_host(fixture.q.data(), q_storage.size());
        k_storage.copy_from_host(fixture.k.data(), k_storage.size());
        v_storage.copy_from_host(fixture.v.data(), v_storage.size());
        q = ninfer::Tensor(q_storage.data(), ninfer::DType::BF16, {kDim, kHeads, patches});
        k = ninfer::Tensor(k_storage.data(), ninfer::DType::BF16, {kDim, kHeads, patches});
        v = ninfer::Tensor(v_storage.data(), ninfer::DType::BF16, {kDim, kHeads, patches});
    }
    ninfer::DeviceBuffer d_cumulative(cumulative.size() * sizeof(int));
    d_cumulative.copy_from_host(cumulative.data(), d_cumulative.size());
    ninfer::Tensor cu(d_cumulative.data(), ninfer::DType::I32,
                      {static_cast<int>(cumulative.size())});
    ninfer::DeviceBuffer d_output(fixture.q.size() * sizeof(hip_bfloat16));
    d_output.fill(0x7f);
    ninfer::Tensor output(d_output.data(), ninfer::DType::BF16, {kDim, kHeads, patches});
    const std::size_t workspace_bytes = ninfer::ops::vision_attention_workspace_capacity_bytes(
        patches, patches, static_cast<int>(lengths.size()), static_cast<int>(lengths.size()));
    ninfer::DeviceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    const auto launch = [&] {
        if (uniform_entry) {
            ninfer::ops::vision_attention(q, k, v, lengths.front(), output, stream);
        } else {
            ninfer::ops::vision_attention(q, k, v, cu, workspace, output, stream);
        }
    };
    launch();

    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin, stream));
    launch();
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed_ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));

    std::vector<hip_bfloat16> actual(fixture.q.size());
    d_output.copy_to_host(actual.data(), d_output.size());
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double max_abs = 0.0;
    double max_reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double got = represented(actual[index]);
        const double error = got - expected[index];
        squared_error += error * error;
        squared_reference += expected[index] * expected[index];
        max_abs = std::max(max_abs, std::abs(error));
        max_reference = std::max(max_reference, std::abs(expected[index]));
    }
    const double relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
    if (relative_l2 > 2.5e-3 || max_abs > 1.0e-3 + 2.8e-3 * max_reference) {
        fail("vision_attention FP64-oracle criterion failed at P=" + std::to_string(patches) +
             " rel_l2=" + std::to_string(relative_l2) + " max_abs=" +
             std::to_string(max_abs));
    }
    if (!uniform_entry && (workspace.used() != 0 || workspace.peak_used() != workspace_bytes)) {
        fail("vision_attention workspace sizing/execution high-water mismatch");
    }
    std::vector<int> preserved_cumulative(cumulative.size());
    d_cumulative.copy_to_host(preserved_cumulative.data(), d_cumulative.size());
    if (preserved_cumulative != cumulative) { fail("vision_attention modified cu_seqlens"); }
    if (padded_qkv) {
        std::vector<hip_bfloat16> preserved(expected_storage.size());
        interleaved.copy_to_host(preserved.data(), interleaved.size());
        if (preserved != expected_storage) { fail("vision_attention modified interleaved QKV"); }
    } else {
        std::vector<hip_bfloat16> preserved_q(fixture.q.size());
        std::vector<hip_bfloat16> preserved_k(fixture.k.size());
        std::vector<hip_bfloat16> preserved_v(fixture.v.size());
        q_storage.copy_to_host(preserved_q.data(), q_storage.size());
        k_storage.copy_to_host(preserved_k.data(), k_storage.size());
        v_storage.copy_to_host(preserved_v.data(), v_storage.size());
        if (preserved_q != fixture.q || preserved_k != fixture.k || preserved_v != fixture.v) {
            fail("vision_attention modified a QKV input");
        }
    }
    std::cout << "vision_attention P=" << patches << " S=" << lengths.size()
              << " entry=" << (uniform_entry ? "uniform" : "packed")
              << " storage=" << (padded_qkv ? "interleaved" : "contiguous")
              << " max_abs=" << max_abs << " rel_l2=" << relative_l2
              << " time_ms=" << elapsed_ms << '\n';
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            fail("vision_attention qualifier requires wave32 gfx1201");
        }
        if (ninfer::ops::vision_attention_workspace_capacity_bytes(4, 194, 1, 1) != 0 ||
            ninfer::ops::vision_attention_workspace_capacity_bytes(4, 194, 1, 3) !=
                ninfer::ops::vision_attention_workspace_capacity_bytes(194, 194, 3, 3)) {
            fail("vision_attention rectangular workspace envelope is not maximal");
        }
        try {
            (void)ninfer::ops::vision_attention_workspace_capacity_bytes(1, 2, 3, 4);
            fail("vision_attention accepted an envelope without a legal segment pair");
        } catch (const std::invalid_argument&) {}
        run_case({4}, false, false, false, 1, device.stream);
        run_case({4, 7}, false, true, true, 7, device.stream);
        run_case({64, 65, 65}, false, true, false, 31, device.stream);
        run_case({68, 68}, true, true, false, 101, device.stream);
        std::cout << "vision_attention: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "vision_attention: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
