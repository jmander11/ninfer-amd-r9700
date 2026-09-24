#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 128;
constexpr int kQHeads = 16;
constexpr int kValueHeads = 48;
constexpr float kScale = 0.08838834764831845F;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;
double g_max_output_abs = 0.0;
double g_max_output_rel_l2 = 0.0;
double g_max_state_abs = 0.0;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device));
        if (std::string(properties.gcnArchName).starts_with("gfx1201") &&
            properties.warpSize == 32) {
            if (selected >= 0) { fail("multiple gfx1201 devices found"); }
            selected = device;
        }
    }
    if (selected < 0) { fail("no wave32 gfx1201 device found"); }
    return selected;
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t absolute = word & 0x7fffffffU;
    if ((absolute & 0x7f800000U) != 0x7f800000U) {
        word += 0x7fffU + ((word >> 16U) & 1U);
    } else if ((word & 0xffffU) != 0U) {
        word |= 0x10000U;
    }
    return static_cast<std::uint16_t>(word >> 16U);
}

hip_bfloat16 bf16(float value) {
    hip_bfloat16 result;
    result.data = bf16_bits(value);
    return result;
}

float fp32(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

template <typename T>
std::vector<T> copied(const ninfer::DeviceBuffer& buffer, std::size_t count) {
    std::vector<T> result(count);
    buffer.copy_to_host(result.data(), result.size() * sizeof(T));
    return result;
}

template <typename T>
ninfer::DeviceBuffer uploaded(const std::vector<T>& values) {
    ninfer::DeviceBuffer result(values.size() * sizeof(T));
    result.copy_from_host(values.data(), result.size());
    return result;
}

std::vector<hip_bfloat16> bf16_fixture(std::size_t count, int salt) {
    std::vector<hip_bfloat16> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int code = static_cast<int>((index * 37U + static_cast<unsigned>(salt)) % 127U) - 63;
        result[index] = bf16(static_cast<float>(code) / 1024.0F);
    }
    return result;
}

std::vector<float> fp32_fixture(std::size_t count, int salt, float divisor) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int code = static_cast<int>((index * 29U + static_cast<unsigned>(salt)) % 113U) - 56;
        result[index] = static_cast<float>(code) / divisor;
    }
    return result;
}

struct Inputs {
    int width;
    int batch;
    std::vector<hip_bfloat16> q;
    std::vector<hip_bfloat16> k;
    std::vector<hip_bfloat16> v;
    std::vector<float> g;
    std::vector<float> beta;
};

Inputs inputs(int width, int batch, int salt) {
    const int columns = width * batch;
    Inputs result{width,
                  batch,
                  bf16_fixture(static_cast<std::size_t>(columns) * kQHeads * kDim, salt),
                  bf16_fixture(static_cast<std::size_t>(columns) * kQHeads * kDim, salt + 7),
                  bf16_fixture(static_cast<std::size_t>(columns) * kValueHeads * kDim, salt + 13),
                  fp32_fixture(static_cast<std::size_t>(columns) * kValueHeads, salt + 19, 2048.0F),
                  fp32_fixture(static_cast<std::size_t>(columns) * kValueHeads, salt + 23, 1024.0F)};
    for (std::size_t index = 0; index < result.g.size(); ++index) {
        result.g[index] = -std::abs(result.g[index]) - 0.03F;
        result.beta[index] = 0.45F + 0.2F * result.beta[index];
    }
    return result;
}

struct OracleStep {
    std::vector<double> state;
    std::vector<double> output;
};

OracleStep step(const Inputs& in, int column, const std::vector<double>& initial,
                bool normalize = true) {
    OracleStep result{initial, std::vector<double>(static_cast<std::size_t>(kValueHeads) * kDim)};
    std::vector<double> qn(static_cast<std::size_t>(kQHeads) * kDim);
    std::vector<double> kn(static_cast<std::size_t>(kQHeads) * kDim);
    for (int q_head = 0; q_head < kQHeads; ++q_head) {
        double qsum = 0.0;
        double ksum = 0.0;
        const std::size_t base =
            (static_cast<std::size_t>(column) * kQHeads + q_head) * kDim;
        for (int key = 0; key < kDim; ++key) {
            const double qvalue = fp32(in.q[base + key]);
            const double kvalue = fp32(in.k[base + key]);
            qn[static_cast<std::size_t>(q_head) * kDim + key] = qvalue;
            kn[static_cast<std::size_t>(q_head) * kDim + key] = kvalue;
            qsum += qvalue * qvalue;
            ksum += kvalue * kvalue;
        }
        if (normalize) {
            const double qi = 1.0 / std::sqrt(qsum + 1.0e-6);
            const double ki = 1.0 / std::sqrt(ksum + 1.0e-6);
            for (int key = 0; key < kDim; ++key) {
                qn[static_cast<std::size_t>(q_head) * kDim + key] *= qi;
                kn[static_cast<std::size_t>(q_head) * kDim + key] *= ki;
            }
        }
    }
    for (int head = 0; head < kValueHeads; ++head) {
        const int q_head = head / 3;
        const std::size_t control = static_cast<std::size_t>(column) * kValueHeads + head;
        const double alpha = std::exp(static_cast<double>(in.g[control]));
        const double beta = in.beta[control];
        for (int row = 0; row < kDim; ++row) {
            const std::size_t state_base =
                (static_cast<std::size_t>(head) * kDim + row) * kDim;
            double state_key = 0.0;
            for (int key = 0; key < kDim; ++key) {
                state_key += result.state[state_base + key] *
                             kn[static_cast<std::size_t>(q_head) * kDim + key];
            }
            const std::size_t value_index = control * kDim + row;
            const double delta = beta * (fp32(in.v[value_index]) - alpha * state_key);
            double state_query = 0.0;
            for (int key = 0; key < kDim; ++key) {
                double& element = result.state[state_base + key];
                element = alpha * element +
                          delta * kn[static_cast<std::size_t>(q_head) * kDim + key];
                state_query += element * qn[static_cast<std::size_t>(q_head) * kDim + key];
            }
            result.output[static_cast<std::size_t>(head) * kDim + row] =
                static_cast<double>(kScale) * state_query;
        }
    }
    return result;
}

void require_output(const std::vector<hip_bfloat16>& actual,
                    const std::vector<double>& expected, const std::string& label) {
    if (actual.size() != expected.size()) { fail(label + ": size mismatch"); }
    double max_abs = 0.0;
    std::size_t max_index = 0;
    double error2 = 0.0;
    double oracle2 = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = static_cast<double>(fp32(actual[index])) - expected[index];
        if (std::abs(error) > max_abs) {
            max_abs = std::abs(error);
            max_index = index;
        }
        error2 += error * error;
        oracle2 += expected[index] * expected[index];
    }
    const double relative = std::sqrt(error2 / std::max(oracle2, 1.0e-30));
    g_max_output_abs = std::max(g_max_output_abs, max_abs);
    g_max_output_rel_l2 = std::max(g_max_output_rel_l2, relative);
    if (max_abs > 0.025 || relative > 0.02) {
        fail(label + ": max_abs=" + std::to_string(max_abs) +
             " rel_l2=" + std::to_string(relative) + " index=" +
             std::to_string(max_index) + " got=" + std::to_string(fp32(actual[max_index])) +
             " expected=" + std::to_string(expected[max_index]));
    }
}

void require_state(const std::vector<float>& actual, const std::vector<double>& expected,
                   const std::string& label) {
    if (actual.size() != expected.size()) { fail(label + ": size mismatch"); }
    double max_abs = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(actual[index]) - expected[index]));
    }
    g_max_state_abs = std::max(g_max_state_abs, max_abs);
    if (max_abs > 0.003) { fail(label + ": max_abs=" + std::to_string(max_abs)); }
}

struct DeviceInputs {
    ninfer::DeviceBuffer q;
    ninfer::DeviceBuffer k;
    ninfer::DeviceBuffer v;
    ninfer::DeviceBuffer g;
    ninfer::DeviceBuffer beta;
    ninfer::Tensor tq;
    ninfer::Tensor tk;
    ninfer::Tensor tv;
    ninfer::Tensor tg;
    ninfer::Tensor tbeta;

    explicit DeviceInputs(const Inputs& input)
        : q(uploaded(input.q)), k(uploaded(input.k)), v(uploaded(input.v)),
          g(uploaded(input.g)), beta(uploaded(input.beta)),
          tq(q.data(), ninfer::DType::BF16, {kDim, kQHeads, input.width, input.batch}),
          tk(k.data(), ninfer::DType::BF16, {kDim, kQHeads, input.width, input.batch}),
          tv(v.data(), ninfer::DType::BF16, {kDim, kValueHeads, input.width, input.batch}),
          tg(g.data(), ninfer::DType::FP32, {kValueHeads, input.width, input.batch}),
          tbeta(beta.data(), ninfer::DType::FP32, {kValueHeads, input.width, input.batch}) {}
};

void qualify_ordinary(int width, bool in_place, bool normalize, int salt, hipStream_t stream) {
    const Inputs in = inputs(width, 1, salt);
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::vector<float> initial = fp32_fixture(state_count, 31, 8192.0F);
    ninfer::DeviceBuffer state_in = uploaded(initial);
    ninfer::DeviceBuffer state_out(in_place ? 1 : state_in.size());
    ninfer::DeviceBuffer output(in.v.size() * sizeof(hip_bfloat16));
    ninfer::Tensor tin(state_in.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor tout(in_place ? state_in.data() : state_out.data(), ninfer::DType::FP32,
                        {kDim, kDim, kValueHeads});
    ninfer::Tensor o(output.data(), ninfer::DType::BF16, {kDim, kValueHeads, in.width});
    ninfer::WorkspaceArena arena(1);
    if (in_place) {
        ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg, device.tbeta,
                                     kScale, normalize, arena, tin, o, stream);
    } else {
        ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg, device.tbeta,
                                     kScale, normalize, arena, tin, tout, o, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<double> state(initial.begin(), initial.end());
    std::vector<double> expected_output;
    for (int token = 0; token < in.width; ++token) {
        OracleStep next = step(in, token, state, normalize);
        state = std::move(next.state);
        expected_output.insert(expected_output.end(), next.output.begin(), next.output.end());
    }
    require_output(copied<hip_bfloat16>(output, in.v.size()), expected_output,
                   in_place ? "ordinary in-place output" : "ordinary distinct output");
    require_state(copied<float>(in_place ? state_in : state_out, state_count), state,
                  in_place ? "ordinary in-place state" : "ordinary distinct state");
    if (!in_place && copied<float>(state_in, state_count) != initial) {
        fail("ordinary distinct input state mutated");
    }
}

void qualify_selected_p2048(hipStream_t stream) {
    constexpr int kWidth = 2048;
    constexpr std::array<int, 6> kHeads{0, 2, 3, 23, 24, 47};
    constexpr std::array<int, 4> kRows{0, 31, 64, 127};
    constexpr std::array<int, 6> kOutputTokens{0, 1, 2, 1023, 1024, 2047};
    constexpr std::array<int, 8> kStateKeys{0, 1, 31, 32, 63, 64, 95, 127};
    const Inputs in = inputs(kWidth, 1, 211);
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::vector<float> initial = fp32_fixture(state_count, 223, 8192.0F);
    ninfer::DeviceBuffer state_in = uploaded(initial);
    ninfer::DeviceBuffer state_out(state_in.size());
    ninfer::DeviceBuffer state_in_place = uploaded(initial);
    ninfer::DeviceBuffer output(in.v.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer output_in_place(output.size());
    state_out.fill_async(0xff, stream);
    output.fill_async(0xff, stream);
    output_in_place.fill_async(0xff, stream);
    ninfer::Tensor tin(state_in.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor tout(state_out.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor tin_place(state_in_place.data(), ninfer::DType::FP32,
                             {kDim, kDim, kValueHeads});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16, {kDim, kValueHeads, kWidth});
    ninfer::Tensor out_in_place(output_in_place.data(), ninfer::DType::BF16,
                                {kDim, kValueHeads, kWidth});
    const std::size_t workspace_bytes =
        ninfer::ops::gated_delta_net_workspace_capacity_bytes(kQHeads, kValueHeads, true,
                                                               kWidth, kWidth);
    if (workspace_bytes != static_cast<std::size_t>(kWidth) * 2U * kQHeads * sizeof(float)) {
        fail("selected P2048 sidecar workspace mismatch");
    }
    ninfer::WorkspaceArena arena(workspace_bytes);
    ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg, device.tbeta,
                                 kScale, true, arena, tin, tout, out, stream);
    ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg, device.tbeta,
                                 kScale, true, arena, tin_place, out_in_place, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    const auto actual_output = copied<hip_bfloat16>(output, in.v.size());
    const auto actual_output_in_place = copied<hip_bfloat16>(output_in_place, in.v.size());
    const auto actual_state = copied<float>(state_out, state_count);
    const auto actual_state_in_place = copied<float>(state_in_place, state_count);
    if (actual_output != actual_output_in_place) {
        fail("selected P2048 distinct/in-place outputs differ");
    }
    if (actual_state != actual_state_in_place) {
        fail("selected P2048 distinct/in-place states differ");
    }
    if (copied<float>(state_in, state_count) != initial) {
        fail("selected P2048 distinct input state mutated");
    }
    if (copied<hip_bfloat16>(device.q, in.q.size()) != in.q ||
        copied<hip_bfloat16>(device.k, in.k.size()) != in.k ||
        copied<hip_bfloat16>(device.v, in.v.size()) != in.v ||
        copied<float>(device.g, in.g.size()) != in.g ||
        copied<float>(device.beta, in.beta.size()) != in.beta) {
        fail("selected P2048 represented input mutated");
    }
    if (std::any_of(actual_output.begin(), actual_output.end(), [](hip_bfloat16 value) {
            return value.data == kBf16Poison;
        })) {
        fail("selected P2048 output retained poison");
    }
    if (std::any_of(actual_state.begin(), actual_state.end(), [](float value) {
            return std::bit_cast<std::uint32_t>(value) == kFp32Poison;
        })) {
        fail("selected P2048 state retained poison");
    }

    struct SampleRow {
        int head;
        int row;
        std::array<double, kDim> state;
    };
    std::vector<SampleRow> samples;
    samples.reserve(kHeads.size() * kRows.size());
    for (const int head : kHeads) {
        for (const int row : kRows) {
            SampleRow sample{head, row, {}};
            const std::size_t base =
                (static_cast<std::size_t>(head) * kDim + row) * kDim;
            for (int key = 0; key < kDim; ++key) {
                sample.state[key] = initial[base + key];
            }
            samples.push_back(sample);
        }
    }

    double selected_max_output_abs = 0.0;
    for (int token = 0; token < kWidth; ++token) {
        const bool compare_output =
            std::find(kOutputTokens.begin(), kOutputTokens.end(), token) != kOutputTokens.end();
        for (SampleRow& sample : samples) {
            const int query_head = sample.head / 3;
            const std::size_t query_base =
                (static_cast<std::size_t>(token) * kQHeads + query_head) * kDim;
            std::array<double, kDim> normalized_query{};
            std::array<double, kDim> normalized_key{};
            double query_sum = 0.0;
            double key_sum = 0.0;
            for (int key = 0; key < kDim; ++key) {
                const double query_value = fp32(in.q[query_base + key]);
                const double key_value = fp32(in.k[query_base + key]);
                normalized_query[key] = query_value;
                normalized_key[key] = key_value;
                query_sum += query_value * query_value;
                key_sum += key_value * key_value;
            }
            const double query_inverse = 1.0 / std::sqrt(query_sum + 1.0e-6);
            const double key_inverse = 1.0 / std::sqrt(key_sum + 1.0e-6);
            double state_key = 0.0;
            for (int key = 0; key < kDim; ++key) {
                normalized_query[key] *= query_inverse;
                normalized_key[key] *= key_inverse;
                state_key += sample.state[key] * normalized_key[key];
            }
            const std::size_t control =
                static_cast<std::size_t>(token) * kValueHeads + sample.head;
            const double alpha = std::exp(static_cast<double>(in.g[control]));
            const double beta = static_cast<double>(in.beta[control]);
            const double represented_value =
                fp32(in.v[control * kDim + sample.row]);
            const double delta = beta * (represented_value - alpha * state_key);
            double state_query = 0.0;
            for (int key = 0; key < kDim; ++key) {
                sample.state[key] =
                    alpha * sample.state[key] + delta * normalized_key[key];
                state_query += sample.state[key] * normalized_query[key];
            }
            if (compare_output) {
                const double expected = static_cast<double>(kScale) * state_query;
                const std::size_t output_index = control * kDim + sample.row;
                const double error =
                    std::abs(static_cast<double>(fp32(actual_output[output_index])) - expected);
                selected_max_output_abs = std::max(selected_max_output_abs, error);
                if (error > 0.025) {
                    fail("selected P2048 sampled output exceeds FP64 criterion at token=" +
                         std::to_string(token) + " head=" + std::to_string(sample.head) +
                         " row=" + std::to_string(sample.row) + " error=" +
                         std::to_string(error));
                }
            }
        }
    }
    g_max_output_abs = std::max(g_max_output_abs, selected_max_output_abs);
    double selected_max_state_abs = 0.0;
    for (const SampleRow& sample : samples) {
        const std::size_t base =
            (static_cast<std::size_t>(sample.head) * kDim + sample.row) * kDim;
        for (const int key : kStateKeys) {
            const double error =
                std::abs(static_cast<double>(actual_state[base + key]) - sample.state[key]);
            selected_max_state_abs = std::max(selected_max_state_abs, error);
            if (error > 0.003) {
                fail("selected P2048 sampled state exceeds FP64 criterion at head=" +
                     std::to_string(sample.head) + " row=" + std::to_string(sample.row) +
                     " key=" + std::to_string(key) + " error=" + std::to_string(error));
            }
        }
    }
    g_max_state_abs = std::max(g_max_state_abs, selected_max_state_abs);

    bool alias_rejected = false;
    try {
        ninfer::Tensor aliased_output(device.v.data(), ninfer::DType::BF16,
                                      {kDim, kValueHeads, kWidth});
        ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg,
                                     device.tbeta, kScale, true, arena, tin, tout,
                                     aliased_output, stream);
    } catch (const std::invalid_argument&) {
        alias_rejected = true;
    }
    if (!alias_rejected) { fail("selected P2048 overlapping output accepted"); }
}

double time_ordinary(int width, hipStream_t stream) {
    const Inputs in = inputs(width, 1, 97 + width);
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::vector<float> initial = fp32_fixture(state_count, 101, 8192.0F);
    ninfer::DeviceBuffer state_in = uploaded(initial);
    ninfer::DeviceBuffer state_out(state_in.size());
    ninfer::DeviceBuffer output(in.v.size() * sizeof(hip_bfloat16));
    ninfer::Tensor tin(state_in.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor tout(state_out.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16, {kDim, kValueHeads, in.width});
    const std::size_t workspace_bytes = ninfer::ops::gated_delta_net_workspace_capacity_bytes(
        kQHeads, kValueHeads, true, width, width);
    ninfer::WorkspaceArena arena(std::max<std::size_t>(1U, workspace_bytes));
    auto launch = [&] {
        ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg, device.tbeta,
                                     kScale, true, arena, tin, tout, out, stream);
    };
    for (int warmup = 0; warmup < 5; ++warmup) { launch(); }
    HIP_CHECK(hipStreamSynchronize(stream));
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin, stream));
    constexpr int kIterations = 31;
    for (int iteration = 0; iteration < kIterations; ++iteration) { launch(); }
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    return static_cast<double>(milliseconds) / kIterations;
}

struct PoolCase {
    Inputs in;
    int slots;
    std::vector<float> pool;
    std::vector<std::int32_t> valid;
    std::vector<std::int32_t> initial;
    std::vector<std::int32_t> bases;
};

PoolCase pool_case(int width, int batch, int salt) {
    const int slots = batch * (width + 1);
    const std::size_t slot_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    PoolCase result{inputs(width, batch, salt), slots,
                    fp32_fixture(static_cast<std::size_t>(slots) * slot_count, salt + 41, 8192.0F),
                    std::vector<std::int32_t>(batch), std::vector<std::int32_t>(batch),
                    std::vector<std::int32_t>(batch)};
    for (int row = 0; row < batch; ++row) {
        result.valid[row] = std::max(1, width - (row & 1));
        result.initial[row] = row * (width + 1);
        result.bases[row] = result.initial[row] + 1;
    }
    return result;
}

void qualify_snapshot(int width, int batch, int salt, hipStream_t stream) {
    PoolCase test = pool_case(width, batch, salt);
    if (width == 1) { test.bases = test.initial; }
    DeviceInputs device(test.in);
    ninfer::DeviceBuffer pool = uploaded(test.pool);
    ninfer::DeviceBuffer valid = uploaded(test.valid);
    ninfer::DeviceBuffer initial = uploaded(test.initial);
    ninfer::DeviceBuffer bases = uploaded(test.bases);
    ninfer::DeviceBuffer output(test.in.v.size() * sizeof(hip_bfloat16));
    ninfer::Tensor states(pool.data(), ninfer::DType::FP32,
                          {kDim, kDim, kValueHeads, test.slots});
    ninfer::Tensor tvalid(valid.data(), ninfer::DType::I32, {test.in.batch});
    ninfer::Tensor tinitial(initial.data(), ninfer::DType::I32, {test.in.batch});
    ninfer::Tensor tbases(width == 1 ? initial.data() : bases.data(), ninfer::DType::I32,
                           {test.in.batch});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16,
                       {kDim, kValueHeads, test.in.width, test.in.batch});
    ninfer::ops::gated_delta_net_snapshot(device.tq, device.tk, device.tv, device.tg,
                                          device.tbeta, kScale, true, states, tvalid, tinitial,
                                          tbases, out, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const auto actual_pool = copied<float>(pool, test.pool.size());
    const auto actual_output = copied<hip_bfloat16>(output, test.in.v.size());
    for (int row = 0; row < test.in.batch; ++row) {
        const std::size_t source = static_cast<std::size_t>(test.initial[row]) * state_count;
        std::vector<double> state(test.pool.begin() + source, test.pool.begin() + source + state_count);
        for (int token = 0; token < test.valid[row]; ++token) {
            OracleStep next = step(test.in, row * test.in.width + token, state);
            state = std::move(next.state);
            const std::size_t out_base =
                static_cast<std::size_t>(row * test.in.width + token) * kValueHeads * kDim;
            require_output({actual_output.begin() + out_base,
                            actual_output.begin() + out_base + kValueHeads * kDim},
                           next.output, "snapshot output");
            const std::size_t snapshot =
                static_cast<std::size_t>(test.bases[row] + token) * state_count;
            require_state({actual_pool.begin() + snapshot, actual_pool.begin() + snapshot + state_count},
                          state, "snapshot state");
        }
        for (int token = test.valid[row]; token < test.in.width; ++token) {
            const std::size_t base =
                static_cast<std::size_t>(row * test.in.width + token) * kValueHeads * kDim;
            for (std::size_t index = base; index < base + kValueHeads * kDim; ++index) {
                if (actual_output[index].data != 0) { fail("snapshot invalid tail not zero"); }
            }
            const std::size_t snapshot =
                static_cast<std::size_t>(test.bases[row] + token) * state_count;
            if (!std::equal(actual_pool.begin() + snapshot,
                            actual_pool.begin() + snapshot + state_count,
                            test.pool.begin() + snapshot)) {
                fail("snapshot invalid-tail state mutated");
            }
        }
    }
}

void qualify_record(bool tree, int width, int batch, int salt, hipStream_t stream) {
    PoolCase test = pool_case(width, batch, salt);
    DeviceInputs device(test.in);
    ninfer::DeviceBuffer pool = uploaded(test.pool);
    ninfer::DeviceBuffer valid = uploaded(test.valid);
    ninfer::DeviceBuffer initial = uploaded(test.initial);
    std::vector<std::int32_t> parents(static_cast<std::size_t>(width) * batch);
    for (int row = 0; row < batch; ++row) {
        for (int token = 0; token < width; ++token) {
            parents[static_cast<std::size_t>(row) * width + token] =
                token == 0 ? -1 : (token > 1 && (token & 1) == 0 ? token - 2 : token - 1);
        }
    }
    ninfer::DeviceBuffer parent = uploaded(parents);
    ninfer::DeviceBuffer key_record(test.in.k.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer value_record(test.in.v.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer gate_record(test.in.g.size() * 2 * sizeof(float));
    ninfer::DeviceBuffer output(test.in.v.size() * sizeof(hip_bfloat16));
    key_record.fill_async(0xff, stream);
    value_record.fill_async(0xff, stream);
    gate_record.fill_async(0xff, stream);
    output.fill_async(0xff, stream);
    ninfer::Tensor states(pool.data(), ninfer::DType::FP32,
                          {kDim, kDim, kValueHeads, test.slots});
    ninfer::Tensor tvalid(valid.data(), ninfer::DType::I32, {test.in.batch});
    ninfer::Tensor tinitial(initial.data(), ninfer::DType::I32, {test.in.batch});
    ninfer::Tensor tparent(parent.data(), ninfer::DType::I32, {test.in.width, test.in.batch});
    ninfer::Tensor keys(key_record.data(), ninfer::DType::BF16,
                        {kDim, kQHeads, test.in.width, test.in.batch});
    ninfer::Tensor values(value_record.data(), ninfer::DType::BF16,
                          {kDim, kValueHeads, test.in.width, test.in.batch});
    ninfer::Tensor gates(gate_record.data(), ninfer::DType::FP32,
                         {2, kValueHeads, test.in.width, test.in.batch});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16,
                       {kDim, kValueHeads, test.in.width, test.in.batch});
    const std::size_t workspace_bytes =
        ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
            kValueHeads, test.in.batch, test.in.width);
    ninfer::WorkspaceArena arena(tree ? workspace_bytes : 1);
    ninfer::ops::gated_delta_net_replay_record(
        device.tq, device.tk, device.tv, device.tg, device.tbeta, kScale, states, tvalid,
        tinitial, keys, values, gates, out, stream, tree ? &tparent : nullptr,
        tree ? &arena : nullptr);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (copied<float>(pool, test.pool.size()) != test.pool) { fail("record mutated state pool"); }
    const auto actual_keys = copied<hip_bfloat16>(key_record, test.in.k.size());
    const auto actual_values = copied<hip_bfloat16>(value_record, test.in.v.size());
    const auto actual_gates = copied<std::uint32_t>(gate_record, test.in.g.size() * 2);
    const auto actual_output = copied<hip_bfloat16>(output, test.in.v.size());
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    for (int row = 0; row < test.in.batch; ++row) {
        const std::size_t source = static_cast<std::size_t>(test.initial[row]) * state_count;
        const std::vector<double> checkpoint(test.pool.begin() + source,
                                             test.pool.begin() + source + state_count);
        std::vector<std::vector<double>> token_states(static_cast<std::size_t>(test.valid[row]));
        std::vector<double> sequential = checkpoint;
        for (int token = 0; token < test.valid[row]; ++token) {
            const int column = row * test.in.width + token;
            const std::vector<double>& start = tree
                ? (parents[column] < 0 ? checkpoint : token_states[parents[column]])
                : sequential;
            OracleStep next = step(test.in, column, start);
            token_states[token] = next.state;
            sequential = next.state;
            const std::size_t out_base = static_cast<std::size_t>(column) * kValueHeads * kDim;
            require_output({actual_output.begin() + out_base,
                            actual_output.begin() + out_base + kValueHeads * kDim},
                           next.output, tree ? "tree record output" : "sequential record output");
        }
        for (int token = 0; token < test.in.width; ++token) {
            const bool active = token < test.valid[row];
            const int column = row * test.in.width + token;
            const std::size_t key_base = static_cast<std::size_t>(column) * kQHeads * kDim;
            for (std::size_t index = 0; index < static_cast<std::size_t>(kQHeads) * kDim; ++index) {
                const std::uint16_t expected = active ? test.in.k[key_base + index].data : kBf16Poison;
                if (actual_keys[key_base + index].data != expected) { fail("key record mismatch"); }
            }
            const std::size_t value_base = static_cast<std::size_t>(column) * kValueHeads * kDim;
            for (std::size_t index = 0; index < static_cast<std::size_t>(kValueHeads) * kDim; ++index) {
                const std::uint16_t expected = active ? test.in.v[value_base + index].data : kBf16Poison;
                if (actual_values[value_base + index].data != expected) { fail("value record mismatch"); }
            }
            for (int head = 0; head < kValueHeads; ++head) {
                const std::size_t control = static_cast<std::size_t>(column) * kValueHeads + head;
                const std::size_t record = control * 2;
                const std::uint32_t expected_g =
                    active ? std::bit_cast<std::uint32_t>(test.in.g[control]) : kFp32Poison;
                const std::uint32_t expected_beta =
                    active ? std::bit_cast<std::uint32_t>(test.in.beta[control]) : kFp32Poison;
                if (actual_gates[record] != expected_g || actual_gates[record + 1] != expected_beta) {
                    fail("gate record mismatch");
                }
            }
            if (!active) {
                for (std::size_t index = value_base;
                     index < value_base + static_cast<std::size_t>(kValueHeads) * kDim; ++index) {
                    if (actual_output[index].data != 0) { fail("record invalid tail not zero"); }
                }
            }
        }
    }
}

void qualify_contract_rejection() {
    constexpr std::size_t kP2048SidecarBytes = 2048U * 2U * 16U * sizeof(float);
    if (ninfer::ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 2048, 2048) !=
            kP2048SidecarBytes ||
        ninfer::ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 1, 4096) !=
            kP2048SidecarBytes ||
        ninfer::ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 1, 2047) != 0U ||
        ninfer::ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 2049, 4096) != 0U ||
        ninfer::ops::gated_delta_net_workspace_capacity_bytes(16, 48, false, 1, 4096) != 0U) {
        fail("ordinary P2048 scale-sidecar workspace selection mismatch");
    }
    constexpr int maximum_batch = static_cast<int>(ninfer::kMaximumConcurrency);
    const std::size_t maximum =
        ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
            48, maximum_batch, 16);
    if (maximum != static_cast<std::size_t>(maximum_batch) * 16 * 48 * 128 * 128 *
                       sizeof(float)) {
        fail("maximum replay workspace mismatch");
    }
    bool oversized_batch_rejected = false;
    try {
        (void)ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
            48, maximum_batch + 1, 16);
    } catch (const std::invalid_argument&) {
        oversized_batch_rejected = true;
    }
    if (!oversized_batch_rejected) { fail("oversized replay batch accepted"); }
    try {
        (void)ninfer::ops::gated_delta_net_workspace_capacity_bytes(16, 32, true, 1, 8);
    } catch (const std::invalid_argument&) {
        return;
    }
    fail("retired Hv32 geometry accepted");
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(select_r9700());
        qualify_ordinary(3, false, true, 11, device.stream);
        qualify_ordinary(3, true, false, 17, device.stream);
        qualify_ordinary(64, true, false, 19, device.stream);
        qualify_ordinary(128, false, true, 23, device.stream);
        qualify_snapshot(3, 2, 53, device.stream);
        qualify_snapshot(1, static_cast<int>(ninfer::kMaximumConcurrency), 59, device.stream);
        qualify_snapshot(16, 1, 61, device.stream);
        qualify_record(false, 4, 2, 71, device.stream);
        qualify_record(true, 4, 2, 79, device.stream);
        qualify_record(true, 2, static_cast<int>(ninfer::kMaximumConcurrency), 83,
                       device.stream);
        qualify_record(true, 16, 1, 89, device.stream);
        qualify_contract_rejection();
        qualify_selected_p2048(device.stream);
        const double ordinary_t1_ms = time_ordinary(1, device.stream);
        const double ordinary_t2_ms = time_ordinary(2, device.stream);
        const double ordinary_t4_ms = time_ordinary(4, device.stream);
        const double ordinary_t8_ms = time_ordinary(8, device.stream);
        const double ordinary_t16_ms = time_ordinary(16, device.stream);
        const double ordinary_t32_ms = time_ordinary(32, device.stream);
        const double ordinary_t33_ms = time_ordinary(33, device.stream);
        const double ordinary_t40_ms = time_ordinary(40, device.stream);
        const double ordinary_t48_ms = time_ordinary(48, device.stream);
        const double ordinary_t64_ms = time_ordinary(64, device.stream);
        const double ordinary_t128_ms = time_ordinary(128, device.stream);
        const double ordinary_t256_ms = time_ordinary(256, device.stream);
        const double ordinary_t512_ms = time_ordinary(512, device.stream);
        const double ordinary_t1024_ms = time_ordinary(1024, device.stream);
        const double ordinary_t2048_ms = time_ordinary(2048, device.stream);
        const double ordinary_t4096_ms = time_ordinary(4096, device.stream);
        std::cout << "PASS gfx1201 public GDN recurrence: Hq16/Hv48 K=V=128 "
                     "ordinary/snapshot/sequential-record/tree-record max_output_abs="
                  << g_max_output_abs << " max_output_rel_l2=" << g_max_output_rel_l2
                  << " max_state_abs=" << g_max_state_abs
                  << " ordinary_normalized_ms[T1/T2/T4/T8/T16/T32/T33/T40/T48/T64/T128/T256/T512/T1024/T2048/T4096]="
                  << ordinary_t1_ms << '/' << ordinary_t2_ms << '/'
                  << ordinary_t4_ms << '/' << ordinary_t8_ms << '/'
                  << ordinary_t16_ms << '/' << ordinary_t32_ms << '/'
                  << ordinary_t33_ms << '/' << ordinary_t40_ms << '/'
                  << ordinary_t48_ms << '/'
                  << ordinary_t64_ms << '/' << ordinary_t128_ms << '/'
                  << ordinary_t256_ms << '/' << ordinary_t512_ms << '/'
                  << ordinary_t1024_ms << '/' << ordinary_t2048_ms << '/'
                  << ordinary_t4096_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
