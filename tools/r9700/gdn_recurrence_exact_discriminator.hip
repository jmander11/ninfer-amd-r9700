#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ops/r9700/gdn_recurrence/gdn_verify_profile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDim = 128;
constexpr int kQHeads = 16;
constexpr int kValueHeads = 48;
constexpr float kScale = 0.08838834764831845F;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

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
std::vector<T> bf_fixture(std::size_t count, int salt);

template <>
std::vector<hip_bfloat16> bf_fixture(std::size_t count, int salt) {
    std::vector<hip_bfloat16> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int code = static_cast<int>((index * 37U + static_cast<unsigned>(salt)) % 127U) - 63;
        result[index] = bf16(static_cast<float>(code) / 1024.0F);
    }
    return result;
}

std::vector<float> fp_fixture(std::size_t count, int salt, float divisor) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int code = static_cast<int>((index * 29U + static_cast<unsigned>(salt)) % 113U) - 56;
        result[index] = static_cast<float>(code) / divisor;
    }
    return result;
}

struct Inputs {
    int width;
    std::vector<hip_bfloat16> q;
    std::vector<hip_bfloat16> k;
    std::vector<hip_bfloat16> v;
    std::vector<float> g;
    std::vector<float> beta;
};

Inputs make_inputs(int width, int salt) {
    Inputs result{width,
                  bf_fixture<hip_bfloat16>(static_cast<std::size_t>(width) * kQHeads * kDim, salt),
                  bf_fixture<hip_bfloat16>(static_cast<std::size_t>(width) * kQHeads * kDim, salt + 7),
                  bf_fixture<hip_bfloat16>(static_cast<std::size_t>(width) * kValueHeads * kDim, salt + 13),
                  fp_fixture(static_cast<std::size_t>(width) * kValueHeads, salt + 19, 2048.0F),
                  fp_fixture(static_cast<std::size_t>(width) * kValueHeads, salt + 23, 1024.0F)};
    for (std::size_t index = 0; index < result.g.size(); ++index) {
        result.g[index] = -std::abs(result.g[index]) - 0.03F;
        result.beta[index] = 0.45F + 0.2F * result.beta[index];
    }
    return result;
}

std::vector<hip_bfloat16> adversarial_normalization_row() {
    // A fixed xorshift sequence selects signed powers of sqrt(2) across a wide, finite BF16
    // dynamic range. Seed 92 is deliberately retained: the ordinary wave32/FMA association and
    // the snapshot 128-element tree association produce adjacent, unequal FP32 sum bits.
    std::uint32_t state = 92;
    std::vector<hip_bfloat16> result(kDim);
    for (int index = 0; index < kDim; ++index) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        const int exponent = static_cast<int>(state % 41U) - 20;
        float value = std::pow(2.0F, static_cast<float>(exponent) / 2.0F);
        if ((state & 0x100U) != 0U) { value = -value; }
        result[index] = bf16(value);
    }
    return result;
}

Inputs make_adversarial_inputs(int width = 1) {
    Inputs result = make_inputs(width, 857);
    const std::vector<hip_bfloat16> row = adversarial_normalization_row();
    for (int token = 0; token < width; ++token) {
        for (int head = 0; head < kQHeads; ++head) {
            const std::size_t base =
                (static_cast<std::size_t>(token) * kQHeads + head) * kDim;
            std::copy(row.begin(), row.end(), result.q.begin() + base);
            std::copy(row.begin(), row.end(), result.k.begin() + base);
        }
    }
    for (int token = 0; token < width; ++token) {
        for (int head = 0; head < kValueHeads; ++head) {
            const std::size_t control =
                static_cast<std::size_t>(token) * kValueHeads + head;
            result.g[control] = -0.125F - 0.00390625F * static_cast<float>(token);
            result.beta[control] = 0.75F - 0.015625F * static_cast<float>(token % 3);
        }
    }
    return result;
}

struct ReductionSums {
    float tree;
    float wave;
};

float fp32_add(float left, float right) {
    volatile float rounded = left + right;
    return rounded;
}

ReductionSums emulated_normalization_sums(const std::vector<hip_bfloat16>& row) {
    if (row.size() != kDim) { fail("adversarial normalization row extent mismatch"); }
    std::vector<float> tree(kDim);
    for (int index = 0; index < kDim; ++index) {
        const float value = fp32(row[index]);
        tree[index] = value * value;
    }
    for (int stride = kDim / 2; stride != 0; stride >>= 1) {
        for (int index = 0; index < stride; ++index) {
            tree[index] = fp32_add(tree[index], tree[index + stride]);
        }
    }
    std::vector<float> lanes(32);
    for (int lane = 0; lane < 32; ++lane) {
        float sum = 0.0F;
        for (int item = 0; item < 4; ++item) {
            const float value = fp32(row[lane + item * 32]);
            sum = std::fma(value, value, sum);
        }
        lanes[lane] = sum;
    }
    for (const int mask : {16, 8, 4, 2, 1}) {
        const std::vector<float> previous = lanes;
        for (int lane = 0; lane < 32; ++lane) {
            lanes[lane] = fp32_add(previous[lane], previous[lane ^ mask]);
        }
    }
    const ReductionSums result{tree[0], lanes[0]};
    if (!std::isfinite(result.tree) || !std::isfinite(result.wave) ||
        std::bit_cast<std::uint32_t>(result.tree) == std::bit_cast<std::uint32_t>(result.wave)) {
        fail("adversarial fixture does not distinguish normalization reduction associations");
    }
    return result;
}

Inputs slice(const Inputs& source, int begin, int count) {
    const auto take = [begin, count](const auto& values, std::size_t stride) {
        const std::size_t first = static_cast<std::size_t>(begin) * stride;
        const std::size_t last = static_cast<std::size_t>(begin + count) * stride;
        return std::vector(values.begin() + static_cast<std::ptrdiff_t>(first),
                           values.begin() + static_cast<std::ptrdiff_t>(last));
    };
    return Inputs{count, take(source.q, kQHeads * kDim), take(source.k, kQHeads * kDim),
                  take(source.v, kValueHeads * kDim), take(source.g, kValueHeads),
                  take(source.beta, kValueHeads)};
}

class GuardedBuffer {
public:
    explicit GuardedBuffer(std::size_t payload_bytes)
        : storage_(payload_bytes + 2 * kGuardBytes), payload_bytes_(payload_bytes) {
        storage_.fill(0xa5);
    }

    template <typename T>
    explicit GuardedBuffer(const std::vector<T>& values) : GuardedBuffer(values.size() * sizeof(T)) {
        storage_.copy_from_host(values.data(), payload_bytes_, kGuardBytes);
    }

    [[nodiscard]] void* data() const {
        return static_cast<void*>(static_cast<std::byte*>(storage_.data()) + kGuardBytes);
    }

    void fill_payload(int byte) {
        HIP_CHECK(hipMemset(data(), byte, payload_bytes_));
    }

    template <typename T>
    [[nodiscard]] std::vector<T> copy(std::size_t count) const {
        if (count * sizeof(T) != payload_bytes_) { fail("guarded copy size mismatch"); }
        std::vector<T> result(count);
        storage_.copy_to_host(result.data(), payload_bytes_, kGuardBytes);
        return result;
    }

    void require_guards(const std::string& label) const {
        std::vector<std::uint8_t> guards(2 * kGuardBytes);
        storage_.copy_to_host(guards.data(), kGuardBytes);
        storage_.copy_to_host(guards.data() + kGuardBytes, kGuardBytes,
                              kGuardBytes + payload_bytes_);
        if (!std::all_of(guards.begin(), guards.end(), [](std::uint8_t byte) {
                return byte == 0xa5U;
            })) {
            fail(label + ": boundary canary changed");
        }
    }

private:
    ninfer::DeviceBuffer storage_;
    std::size_t payload_bytes_;
};

struct DeviceInputs {
    GuardedBuffer q;
    GuardedBuffer k;
    GuardedBuffer v;
    GuardedBuffer g;
    GuardedBuffer beta;
    ninfer::Tensor tq;
    ninfer::Tensor tk;
    ninfer::Tensor tv;
    ninfer::Tensor tg;
    ninfer::Tensor tbeta;

    explicit DeviceInputs(const Inputs& in)
        : q(in.q), k(in.k), v(in.v), g(in.g), beta(in.beta),
          tq(q.data(), ninfer::DType::BF16, {kDim, kQHeads, in.width, 1}),
          tk(k.data(), ninfer::DType::BF16, {kDim, kQHeads, in.width, 1}),
          tv(v.data(), ninfer::DType::BF16, {kDim, kValueHeads, in.width, 1}),
          tg(g.data(), ninfer::DType::FP32, {kValueHeads, in.width, 1}),
          tbeta(beta.data(), ninfer::DType::FP32, {kValueHeads, in.width, 1}) {}

    void require_unchanged(const Inputs& in, const std::string& label) const {
        if (q.copy<hip_bfloat16>(in.q.size()) != in.q ||
            k.copy<hip_bfloat16>(in.k.size()) != in.k ||
            v.copy<hip_bfloat16>(in.v.size()) != in.v || g.copy<float>(in.g.size()) != in.g ||
            beta.copy<float>(in.beta.size()) != in.beta) {
            fail(label + ": represented input mutated");
        }
        q.require_guards(label + " q");
        k.require_guards(label + " k");
        v.require_guards(label + " v");
        g.require_guards(label + " g");
        beta.require_guards(label + " beta");
    }
};

struct Oracle {
    std::vector<double> outputs;
    std::vector<double> state;
};

Oracle oracle(const Inputs& in, const std::vector<float>& initial) {
    Oracle result{{}, std::vector<double>(initial.begin(), initial.end())};
    result.outputs.reserve(in.v.size());
    std::vector<double> qn(static_cast<std::size_t>(kQHeads) * kDim);
    std::vector<double> kn(static_cast<std::size_t>(kQHeads) * kDim);
    for (int token = 0; token < in.width; ++token) {
        for (int qhead = 0; qhead < kQHeads; ++qhead) {
            const std::size_t base =
                (static_cast<std::size_t>(token) * kQHeads + qhead) * kDim;
            double qsum = 0.0;
            double ksum = 0.0;
            for (int key = 0; key < kDim; ++key) {
                const double qvalue = fp32(in.q[base + key]);
                const double kvalue = fp32(in.k[base + key]);
                qn[static_cast<std::size_t>(qhead) * kDim + key] = qvalue;
                kn[static_cast<std::size_t>(qhead) * kDim + key] = kvalue;
                qsum += qvalue * qvalue;
                ksum += kvalue * kvalue;
            }
            const double qinverse = 1.0 / std::sqrt(qsum + 1.0e-6);
            const double kinverse = 1.0 / std::sqrt(ksum + 1.0e-6);
            for (int key = 0; key < kDim; ++key) {
                qn[static_cast<std::size_t>(qhead) * kDim + key] *= qinverse;
                kn[static_cast<std::size_t>(qhead) * kDim + key] *= kinverse;
            }
        }
        for (int head = 0; head < kValueHeads; ++head) {
            const int qhead = head / 3;
            const std::size_t control = static_cast<std::size_t>(token) * kValueHeads + head;
            const double alpha = std::exp(static_cast<double>(in.g[control]));
            const double beta = in.beta[control];
            for (int row = 0; row < kDim; ++row) {
                const std::size_t state_base =
                    (static_cast<std::size_t>(head) * kDim + row) * kDim;
                double state_key = 0.0;
                for (int key = 0; key < kDim; ++key) {
                    state_key += result.state[state_base + key] *
                                 kn[static_cast<std::size_t>(qhead) * kDim + key];
                }
                const double delta = beta *
                    (fp32(in.v[control * kDim + row]) - alpha * state_key);
                double state_query = 0.0;
                for (int key = 0; key < kDim; ++key) {
                    double& element = result.state[state_base + key];
                    element = alpha * element +
                              delta * kn[static_cast<std::size_t>(qhead) * kDim + key];
                    state_query += element * qn[static_cast<std::size_t>(qhead) * kDim + key];
                }
                result.outputs.push_back(static_cast<double>(kScale) * state_query);
            }
        }
    }
    return result;
}

struct Metrics {
    double output_max_abs = 0.0;
    double output_rel_l2 = 0.0;
    double state_max_abs = 0.0;
};

Metrics require_oracle(const std::vector<hip_bfloat16>& output,
                       const std::vector<double>& expected_output,
                       const std::vector<float>& state,
                       const std::vector<double>& expected_state, const std::string& label) {
    if (output.size() != expected_output.size() || state.size() != expected_state.size()) {
        fail(label + ": oracle extent mismatch");
    }
    Metrics metrics;
    double error2 = 0.0;
    double oracle2 = 0.0;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const double error = static_cast<double>(fp32(output[index])) - expected_output[index];
        metrics.output_max_abs = std::max(metrics.output_max_abs, std::abs(error));
        error2 += error * error;
        oracle2 += expected_output[index] * expected_output[index];
    }
    metrics.output_rel_l2 = std::sqrt(error2 / std::max(oracle2, 1.0e-30));
    for (std::size_t index = 0; index < state.size(); ++index) {
        metrics.state_max_abs = std::max(
            metrics.state_max_abs,
            std::abs(static_cast<double>(state[index]) - expected_state[index]));
    }
    if (!std::isfinite(metrics.output_max_abs) || !std::isfinite(metrics.output_rel_l2) ||
        !std::isfinite(metrics.state_max_abs) || metrics.output_max_abs > 0.025 ||
        metrics.output_rel_l2 > 0.02 || metrics.state_max_abs > 0.003) {
        fail(label + ": independent FP64 oracle criterion failed");
    }
    return metrics;
}

void require_output_oracle(const std::vector<hip_bfloat16>& output,
                           const std::vector<double>& expected, const std::string& label) {
    if (output.size() != expected.size()) { fail(label + ": output oracle extent mismatch"); }
    double maximum = 0.0;
    double error2 = 0.0;
    double oracle2 = 0.0;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const double error = static_cast<double>(fp32(output[index])) - expected[index];
        maximum = std::max(maximum, std::abs(error));
        error2 += error * error;
        oracle2 += expected[index] * expected[index];
    }
    const double relative = std::sqrt(error2 / std::max(oracle2, 1.0e-30));
    if (!std::isfinite(maximum) || !std::isfinite(relative) || maximum > 0.025 ||
        relative > 0.02) {
        fail(label + ": independent FP64 output oracle criterion failed");
    }
}

struct ExactComparison {
    bool exact;
    std::size_t mismatch_count;
    std::size_t first_index;
    std::uint32_t left_bits;
    std::uint32_t right_bits;
};

template <typename T>
std::uint32_t bits(T value);

template <>
std::uint32_t bits(hip_bfloat16 value) { return value.data; }

template <>
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

template <typename T>
ExactComparison compare_exact(const std::vector<T>& left, const std::vector<T>& right) {
    if (left.size() != right.size()) { fail("exact comparison extent mismatch"); }
    ExactComparison result{true, 0, 0, 0, 0};
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (bits(left[index]) != bits(right[index])) {
            if (result.exact) {
                result.first_index = index;
                result.left_bits = bits(left[index]);
                result.right_bits = bits(right[index]);
            }
            result.exact = false;
            ++result.mismatch_count;
        }
    }
    return result;
}

struct RunResult {
    std::vector<hip_bfloat16> output;
    std::vector<float> state;
    Metrics metrics;
};

struct RecordResult {
    std::vector<hip_bfloat16> output;
    std::vector<hip_bfloat16> key_record;
    std::vector<hip_bfloat16> value_record;
    std::vector<float> gate_record;
    std::vector<float> tree_states;
};

RunResult run_ordinary(const Inputs& in, const std::vector<float>& initial,
                       const Oracle& expected, hipStream_t stream, const std::string& label) {
    DeviceInputs device(in);
    GuardedBuffer state_in(initial);
    GuardedBuffer state_out(initial.size() * sizeof(float));
    GuardedBuffer output(in.v.size() * sizeof(hip_bfloat16));
    state_out.fill_payload(0xff);
    output.fill_payload(0xff);
    ninfer::Tensor tin(state_in.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor tout(state_out.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16,
                       {kDim, kValueHeads, in.width});
    ninfer::WorkspaceArena workspace(1);
    ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg, device.tbeta,
                                 kScale, true, workspace, tin, tout, out, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    RunResult result{output.copy<hip_bfloat16>(in.v.size()),
                     state_out.copy<float>(initial.size()), {}};
    result.metrics = require_oracle(result.output, expected.outputs, result.state,
                                    expected.state, label);
    if (state_in.copy<float>(initial.size()) != initial) { fail(label + ": input state mutated"); }
    if (std::any_of(result.output.begin(), result.output.end(), [](hip_bfloat16 value) {
            return value.data == kBf16Poison;
        }) || std::any_of(result.state.begin(), result.state.end(), [](float value) {
            return std::bit_cast<std::uint32_t>(value) == kFp32Poison;
        })) {
        fail(label + ": output retained poison");
    }
    device.require_unchanged(in, label);
    state_in.require_guards(label + " state_in");
    state_out.require_guards(label + " state_out");
    output.require_guards(label + " output");
    return result;
}

RunResult run_snapshot(const Inputs& in, const std::vector<float>& initial,
                       const Oracle& expected, hipStream_t stream, const std::string& label) {
    if (in.width != 1) { fail(label + ": snapshot discriminator width must be one"); }
    DeviceInputs device(in);
    const std::size_t state_count = initial.size();
    std::vector<float> pool(2 * state_count,
                            std::bit_cast<float>(kFp32Poison));
    std::copy(initial.begin(), initial.end(), pool.begin());
    GuardedBuffer states(pool);
    const std::vector<std::int32_t> valid{1};
    const std::vector<std::int32_t> initial_slot{0};
    const std::vector<std::int32_t> base_slot{1};
    GuardedBuffer dvalid(valid);
    GuardedBuffer dinitial(initial_slot);
    GuardedBuffer dbase(base_slot);
    GuardedBuffer output(in.v.size() * sizeof(hip_bfloat16));
    output.fill_payload(0xff);
    ninfer::Tensor tstates(states.data(), ninfer::DType::FP32,
                           {kDim, kDim, kValueHeads, 2});
    ninfer::Tensor tvalid(dvalid.data(), ninfer::DType::I32, {1});
    ninfer::Tensor tinitial(dinitial.data(), ninfer::DType::I32, {1});
    ninfer::Tensor tbase(dbase.data(), ninfer::DType::I32, {1});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16, {kDim, kValueHeads, 1, 1});
    ninfer::ops::gated_delta_net_snapshot(device.tq, device.tk, device.tv, device.tg,
                                          device.tbeta, kScale, true, tstates, tvalid,
                                          tinitial, tbase, out, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto actual_pool = states.copy<float>(pool.size());
    RunResult result{output.copy<hip_bfloat16>(in.v.size()),
                     {actual_pool.begin() + static_cast<std::ptrdiff_t>(state_count),
                      actual_pool.end()}, {}};
    result.metrics = require_oracle(result.output, expected.outputs, result.state,
                                    expected.state, label);
    if (!std::equal(actual_pool.begin(),
                    actual_pool.begin() + static_cast<std::ptrdiff_t>(state_count),
                    initial.begin())) {
        fail(label + ": source state slot mutated");
    }
    if (std::any_of(result.output.begin(), result.output.end(), [](hip_bfloat16 value) {
            return value.data == kBf16Poison;
        }) || std::any_of(result.state.begin(), result.state.end(), [](float value) {
            return std::bit_cast<std::uint32_t>(value) == kFp32Poison;
        })) {
        fail(label + ": destination retained poison");
    }
    device.require_unchanged(in, label);
    states.require_guards(label + " states");
    dvalid.require_guards(label + " valid");
    dinitial.require_guards(label + " initial_slot");
    dbase.require_guards(label + " base_slot");
    output.require_guards(label + " output");
    return result;
}

RecordResult run_record(const Inputs& in, const std::vector<float>& initial,
                        const std::vector<std::int32_t>* parents, hipStream_t stream,
                        const std::string& label) {
    if (in.width < 2 || in.width > 16) { fail(label + ": record width is outside [2,16]"); }
    if (parents != nullptr && parents->size() != static_cast<std::size_t>(in.width)) {
        fail(label + ": parent extent mismatch");
    }
    DeviceInputs device(in);
    GuardedBuffer states(initial);
    const std::vector<std::int32_t> valid{in.width};
    const std::vector<std::int32_t> initial_slot{0};
    GuardedBuffer dvalid(valid);
    GuardedBuffer dinitial(initial_slot);
    GuardedBuffer key_record(in.k.size() * sizeof(hip_bfloat16));
    GuardedBuffer value_record(in.v.size() * sizeof(hip_bfloat16));
    GuardedBuffer gate_record(in.g.size() * 2U * sizeof(float));
    GuardedBuffer output(in.v.size() * sizeof(hip_bfloat16));
    key_record.fill_payload(0xff);
    value_record.fill_payload(0xff);
    gate_record.fill_payload(0xff);
    output.fill_payload(0xff);
    ninfer::Tensor tstates(states.data(), ninfer::DType::FP32,
                           {kDim, kDim, kValueHeads, 1});
    ninfer::Tensor tvalid(dvalid.data(), ninfer::DType::I32, {1});
    ninfer::Tensor tinitial(dinitial.data(), ninfer::DType::I32, {1});
    ninfer::Tensor tkey(key_record.data(), ninfer::DType::BF16,
                        {kDim, kQHeads, in.width, 1});
    ninfer::Tensor tvalue(value_record.data(), ninfer::DType::BF16,
                          {kDim, kValueHeads, in.width, 1});
    ninfer::Tensor tgate(gate_record.data(), ninfer::DType::FP32,
                         {2, kValueHeads, in.width, 1});
    ninfer::Tensor tout(output.data(), ninfer::DType::BF16,
                        {kDim, kValueHeads, in.width, 1});

    GuardedBuffer dparents(parents == nullptr ? std::vector<std::int32_t>{0} : *parents);
    ninfer::Tensor tparents(dparents.data(), ninfer::DType::I32,
                            {parents == nullptr ? 1 : in.width, 1});
    const std::size_t tree_state_count =
        static_cast<std::size_t>(in.width) * initial.size();
    GuardedBuffer tree_storage(parents == nullptr ? sizeof(float)
                                                  : tree_state_count * sizeof(float));
    tree_storage.fill_payload(0xff);
    ninfer::WorkspaceArena tree_workspace(
        ninfer::DeviceSpan{tree_storage.data(),
                           parents == nullptr ? sizeof(float)
                                              : tree_state_count * sizeof(float)});
    ninfer::ops::gated_delta_net_replay_record(
        device.tq, device.tk, device.tv, device.tg, device.tbeta, kScale, tstates, tvalid,
        tinitial, tkey, tvalue, tgate, tout, stream,
        parents == nullptr ? nullptr : &tparents,
        parents == nullptr ? nullptr : &tree_workspace);
    HIP_CHECK(hipStreamSynchronize(stream));

    RecordResult result{
        output.copy<hip_bfloat16>(in.v.size()),
        key_record.copy<hip_bfloat16>(in.k.size()),
        value_record.copy<hip_bfloat16>(in.v.size()),
        gate_record.copy<float>(in.g.size() * 2U),
        parents == nullptr ? std::vector<float>{}
                           : tree_storage.copy<float>(tree_state_count)};
    if (states.copy<float>(initial.size()) != initial) {
        fail(label + ": source state mutated");
    }
    std::vector<float> expected_gate(in.g.size() * 2U);
    for (std::size_t index = 0; index < in.g.size(); ++index) {
        expected_gate[2U * index] = in.g[index];
        expected_gate[2U * index + 1U] = in.beta[index];
    }
    if (!compare_exact(result.key_record, in.k).exact ||
        !compare_exact(result.value_record, in.v).exact ||
        !compare_exact(result.gate_record, expected_gate).exact) {
        fail(label + ": replay record payload is not a bit-preserving semantic copy");
    }
    if (std::any_of(result.output.begin(), result.output.end(), [](hip_bfloat16 value) {
            return value.data == kBf16Poison;
        })) {
        fail(label + ": output retained poison");
    }
    if (parents != nullptr && std::any_of(result.tree_states.begin(), result.tree_states.end(),
                                          [](float value) {
                                              return std::bit_cast<std::uint32_t>(value) ==
                                                     kFp32Poison;
                                          })) {
        fail(label + ": tree state retained poison");
    }
    device.require_unchanged(in, label);
    states.require_guards(label + " states");
    dvalid.require_guards(label + " valid");
    dinitial.require_guards(label + " initial_slot");
    dparents.require_guards(label + " parents");
    key_record.require_guards(label + " key_record");
    value_record.require_guards(label + " value_record");
    gate_record.require_guards(label + " gate_record");
    output.require_guards(label + " output");
    tree_storage.require_guards(label + " tree_states");
    return result;
}

struct TreeReference {
    std::vector<hip_bfloat16> output;
    std::vector<float> states;
    std::vector<double> oracle_output;
    std::vector<double> oracle_states;
};

TreeReference tree_reference(const Inputs& in, const std::vector<float>& initial,
                             const std::vector<std::int32_t>& parents,
                             hipStream_t stream, const std::string& label);

struct RecordEvidence {
    int width = 0;
    ExactComparison sequential_output{};
    ExactComparison sequential_key_record{};
    ExactComparison sequential_value_record{};
    ExactComparison sequential_gate_record{};
    ExactComparison tree_output{};
    ExactComparison tree_states{};
    ExactComparison tree_key_record{};
    ExactComparison tree_value_record{};
    ExactComparison tree_gate_record{};
};

std::vector<float> expected_gate_record(const Inputs& in) {
    std::vector<float> result(in.g.size() * 2U);
    for (std::size_t index = 0; index < in.g.size(); ++index) {
        result[2U * index] = in.g[index];
        result[2U * index + 1U] = in.beta[index];
    }
    return result;
}

RecordEvidence exercise_record_width(int width, const std::vector<float>& initial,
                                     hipStream_t stream) {
    const Inputs in = make_adversarial_inputs(width);
    const Oracle sequential_oracle = oracle(in, initial);
    const RunResult ordinary = run_ordinary(
        in, initial, sequential_oracle, stream,
        "record_reference_ordinary_w" + std::to_string(width));
    const RecordResult sequential = run_record(
        in, initial, nullptr, stream, "record_sequential_w" + std::to_string(width));
    require_output_oracle(sequential.output, sequential_oracle.outputs,
                          "record_sequential_w" + std::to_string(width));

    std::vector<std::int32_t> parents = {-1, 0, 0, 2, 1};
    if (width == 6) parents.push_back(4);
    if (width != 5 && width != 6) { fail("record discriminator width must be 5 or 6"); }
    const TreeReference reference = tree_reference(
        in, initial, parents, stream, "record_tree_reference_w" + std::to_string(width));
    const RecordResult tree = run_record(
        in, initial, &parents, stream, "record_tree_w" + std::to_string(width));
    (void)require_oracle(tree.output, reference.oracle_output, tree.tree_states,
                         reference.oracle_states,
                         "record_tree_w" + std::to_string(width));
    const std::vector<float> expected_gate = expected_gate_record(in);

    RecordEvidence evidence{
        .width = width,
        .sequential_output = compare_exact(ordinary.output, sequential.output),
        .sequential_key_record = compare_exact(in.k, sequential.key_record),
        .sequential_value_record = compare_exact(in.v, sequential.value_record),
        .sequential_gate_record = compare_exact(expected_gate, sequential.gate_record),
        .tree_output = compare_exact(reference.output, tree.output),
        .tree_states = compare_exact(reference.states, tree.tree_states),
        .tree_key_record = compare_exact(in.k, tree.key_record),
        .tree_value_record = compare_exact(in.v, tree.value_record),
        .tree_gate_record = compare_exact(expected_gate, tree.gate_record),
    };
    const bool selector = ninfer::ops::r9700::gdn_recurrence::kVerifyWaveQkCandidate;
    if (evidence.sequential_output.exact != selector ||
        (evidence.tree_output.exact && evidence.tree_states.exact) != selector) {
        fail("record adversarial exactness does not match the compiled verify-wave selector");
    }
    if (!evidence.sequential_key_record.exact || !evidence.sequential_value_record.exact ||
        !evidence.sequential_gate_record.exact || !evidence.tree_key_record.exact ||
        !evidence.tree_value_record.exact || !evidence.tree_gate_record.exact) {
        fail("record semantic payload comparison failed");
    }
    return evidence;
}

TreeReference tree_reference(const Inputs& in, const std::vector<float>& initial,
                             const std::vector<std::int32_t>& parents,
                             hipStream_t stream, const std::string& label) {
    if (parents.size() != static_cast<std::size_t>(in.width)) {
        fail(label + ": reference parent extent mismatch");
    }
    TreeReference result;
    result.output.reserve(in.v.size());
    result.states.reserve(static_cast<std::size_t>(in.width) * initial.size());
    result.oracle_output.reserve(in.v.size());
    result.oracle_states.reserve(static_cast<std::size_t>(in.width) * initial.size());
    for (int token = 0; token < in.width; ++token) {
        const int parent = parents[static_cast<std::size_t>(token)];
        if ((token == 0 && parent != -1) ||
            (token > 0 && (parent < 0 || parent >= token))) {
            fail(label + ": invalid parent topology");
        }
        std::vector<float> parent_state = initial;
        if (parent >= 0) {
            const auto begin = result.states.begin() +
                static_cast<std::ptrdiff_t>(parent) * static_cast<std::ptrdiff_t>(initial.size());
            parent_state.assign(begin, begin + static_cast<std::ptrdiff_t>(initial.size()));
        }
        const Inputs one = slice(in, token, 1);
        const Oracle expected = oracle(one, parent_state);
        const RunResult node = run_ordinary(one, parent_state, expected, stream,
                                            label + "_node" + std::to_string(token));
        result.output.insert(result.output.end(), node.output.begin(), node.output.end());
        result.states.insert(result.states.end(), node.state.begin(), node.state.end());
        result.oracle_output.insert(result.oracle_output.end(), expected.outputs.begin(),
                                    expected.outputs.end());
        result.oracle_states.insert(result.oracle_states.end(), expected.state.begin(),
                                    expected.state.end());
    }
    return result;
}

void require_alias_rejection(hipStream_t stream) {
    const Inputs in = make_inputs(1, 901);
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    GuardedBuffer state(fp_fixture(state_count, 907, 8192.0F));
    ninfer::Tensor tstate(state.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor aliased_out(device.v.data(), ninfer::DType::BF16,
                               {kDim, kValueHeads, 1});
    ninfer::WorkspaceArena workspace(1);
    bool rejected = false;
    try {
        ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg,
                                     device.tbeta, kScale, true, workspace, tstate,
                                     aliased_out, stream);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) { fail("ordinary output/input alias was accepted"); }
}

void print_exact(const char* name, const ExactComparison& value, bool comma) {
    std::cout << "    \"" << name << "\": {\"exact\": "
              << (value.exact ? "true" : "false")
              << ", \"mismatch_count\": " << value.mismatch_count
              << ", \"first_index\": " << value.first_index
              << ", \"left_bits\": " << value.left_bits
              << ", \"right_bits\": " << value.right_bits << "}";
    if (comma) { std::cout << ','; }
    std::cout << '\n';
}

void print_metrics(const char* name, const Metrics& metrics, bool comma) {
    std::cout << "    \"" << name << "\": {\"output_max_abs\": "
              << metrics.output_max_abs << ", \"output_rel_l2\": "
              << metrics.output_rel_l2 << ", \"state_max_abs\": "
              << metrics.state_max_abs << "}";
    if (comma) { std::cout << ','; }
    std::cout << '\n';
}

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device));
        if (std::string(properties.gcnArchName).starts_with("gfx1201") &&
            properties.warpSize == 32) {
            if (selected >= 0) { fail("multiple wave32 gfx1201 devices found"); }
            selected = device;
        }
    }
    if (selected < 0) { fail("no wave32 gfx1201 device found"); }
    return selected;
}

} // namespace

int main(int argc, char** argv) try {
    if (argc == 2 && std::string(argv[1]) == "--describe") {
        std::cout << "{\"artifact_type\":\"gdn_recurrence_exact_discriminator\","
                     "\"schema_version\":2,\"device\":\"gfx1201-wave32\","
                     "\"verify_wave_qk_candidate\":"
                  << (ninfer::ops::r9700::gdn_recurrence::kVerifyWaveQkCandidate ? 1 : 0)
                  << ","
                     "\"normalize_qk\":true,\"cases\":[\"ordinary-t1-vs-snapshot-b1w1\","
                     "\"adversarial-ordinary-t1-vs-snapshot-b1w1\","
                     "\"ordinary-t129-vs-ordinary-t128-plus-snapshot-b1w1\","
                     "\"adversarial-sequential-record-b1w5-w6\","
                     "\"adversarial-tree-record-b1w5-w6\"],"
                     "\"claim\":\"diagnostic-only-no-performance-or-production-claim\"}\n";
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--check-fixture") {
        const ReductionSums sums =
            emulated_normalization_sums(adversarial_normalization_row());
        std::cout << "{\"tree_sum_bits\":" << std::bit_cast<std::uint32_t>(sums.tree)
                  << ",\"wave_sum_bits\":" << std::bit_cast<std::uint32_t>(sums.wave)
                  << ",\"different\":true}\n";
        return 0;
    }
    if (argc != 1) {
        fail("usage: gdn_recurrence_exact_discriminator [--describe|--check-fixture]");
    }
    const int device = select_r9700();
    HIP_CHECK(hipSetDevice(device));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::vector<float> initial = fp_fixture(state_count, 811, 8192.0F);

    const Inputs one = make_inputs(1, 821);
    const Oracle one_oracle = oracle(one, initial);
    const RunResult ordinary_one = run_ordinary(one, initial, one_oracle, stream, "ordinary_t1");
    const RunResult snapshot_one = run_snapshot(one, initial, one_oracle, stream, "snapshot_b1w1");
    const ExactComparison one_output = compare_exact(ordinary_one.output, snapshot_one.output);
    const ExactComparison one_state = compare_exact(ordinary_one.state, snapshot_one.state);

    const Inputs adversarial = make_adversarial_inputs();
    const ReductionSums adversarial_sums =
        emulated_normalization_sums(adversarial_normalization_row());
    const std::vector<float> adversarial_initial = fp_fixture(state_count, 863, 256.0F);
    const Oracle adversarial_oracle = oracle(adversarial, adversarial_initial);
    const RunResult adversarial_ordinary = run_ordinary(
        adversarial, adversarial_initial, adversarial_oracle, stream, "adversarial_ordinary_t1");
    const RunResult adversarial_snapshot = run_snapshot(
        adversarial, adversarial_initial, adversarial_oracle, stream, "adversarial_snapshot_b1w1");
    const ExactComparison adversarial_output =
        compare_exact(adversarial_ordinary.output, adversarial_snapshot.output);
    const ExactComparison adversarial_state =
        compare_exact(adversarial_ordinary.state, adversarial_snapshot.state);
    const bool adversarial_exact = adversarial_output.exact && adversarial_state.exact;
    if (adversarial_exact !=
        ninfer::ops::r9700::gdn_recurrence::kVerifyWaveQkCandidate) {
        fail("adversarial exactness does not match the compiled verify-wave selector");
    }

    const Inputs full = make_inputs(129, 829);
    const Inputs prefix = slice(full, 0, 128);
    const Inputs tail = slice(full, 128, 1);
    const Oracle full_oracle = oracle(full, initial);
    const Oracle prefix_oracle = oracle(prefix, initial);
    const RunResult ordinary_full =
        run_ordinary(full, initial, full_oracle, stream, "ordinary_t129");
    const RunResult ordinary_prefix =
        run_ordinary(prefix, initial, prefix_oracle, stream, "ordinary_t128");
    // The snapshot's public initial state is the represented FP32 result of the ordinary T128
    // call. Its independent oracle must begin from those represented bytes, rather than from the
    // unrounded FP64 state carried internally by the separate T129 oracle.
    const Oracle tail_oracle = oracle(tail, ordinary_prefix.state);
    const RunResult snapshot_tail = run_snapshot(tail, ordinary_prefix.state, tail_oracle,
                                                  stream, "snapshot_tail_b1w1");
    const std::vector<hip_bfloat16> full_last{
        ordinary_full.output.end() - static_cast<std::ptrdiff_t>(kValueHeads * kDim),
        ordinary_full.output.end()};
    const std::vector<hip_bfloat16> full_prefix{
        ordinary_full.output.begin(),
        ordinary_full.output.end() - static_cast<std::ptrdiff_t>(kValueHeads * kDim)};
    const ExactComparison prefix_output = compare_exact(full_prefix, ordinary_prefix.output);
    const ExactComparison composed_output = compare_exact(full_last, snapshot_tail.output);
    const ExactComparison composed_state = compare_exact(ordinary_full.state, snapshot_tail.state);
    const std::array<RecordEvidence, 2> record_evidence{
        exercise_record_width(5, adversarial_initial, stream),
        exercise_record_width(6, adversarial_initial, stream),
    };
    require_alias_rejection(stream);
    HIP_CHECK(hipStreamDestroy(stream));

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "{\n  \"artifact_type\": \"gdn_recurrence_exact_discriminator_result\",\n"
                 "  \"schema_version\": 2,\n  \"device\": \"gfx1201-wave32\",\n"
              << "  \"verify_wave_qk_candidate\": "
              << (ninfer::ops::r9700::gdn_recurrence::kVerifyWaveQkCandidate ? 1 : 0)
              << ",\n"
                 "  \"w1\": {\n";
    print_exact("output", one_output, true);
    print_exact("state", one_state, true);
    print_metrics("ordinary_oracle", ordinary_one.metrics, true);
    print_metrics("snapshot_oracle", snapshot_one.metrics, false);
    std::cout << "  },\n  \"adversarial_w1\": {\n"
              << "    \"fixture\": \"xorshift92-signed-pow-sqrt2-bf16\",\n"
              << "    \"tree_sum_bits\": " << std::bit_cast<std::uint32_t>(adversarial_sums.tree)
              << ",\n    \"wave_sum_bits\": "
              << std::bit_cast<std::uint32_t>(adversarial_sums.wave) << ",\n";
    print_exact("output", adversarial_output, true);
    print_exact("state", adversarial_state, true);
    print_metrics("ordinary_oracle", adversarial_ordinary.metrics, true);
    print_metrics("snapshot_oracle", adversarial_snapshot.metrics, false);
    std::cout << "  },\n  \"t129_composition\": {\n";
    print_exact("ordinary_prefix_output", prefix_output, true);
    print_exact("final_output", composed_output, true);
    print_exact("final_state", composed_state, true);
    print_metrics("ordinary_t129_oracle", ordinary_full.metrics, true);
    print_metrics("ordinary_t128_oracle", ordinary_prefix.metrics, true);
    print_metrics("snapshot_tail_oracle", snapshot_tail.metrics, false);
    std::cout << "  },\n  \"record_w5_w6\": {\n"
                 "    \"topology\": \"B1; W5 [-1,0,0,2,1]; W6 [-1,0,0,2,1,4]\",\n"
                 "    \"widths\": [\n";
    for (std::size_t index = 0; index < record_evidence.size(); ++index) {
        const RecordEvidence& evidence = record_evidence[index];
        std::cout << "      {\n        \"width\": " << evidence.width << ",\n";
        print_exact("sequential_output_vs_ordinary", evidence.sequential_output, true);
        print_exact("sequential_key_record", evidence.sequential_key_record, true);
        print_exact("sequential_value_record", evidence.sequential_value_record, true);
        print_exact("sequential_gate_record", evidence.sequential_gate_record, true);
        print_exact("tree_output_vs_wave_reference", evidence.tree_output, true);
        print_exact("tree_states_vs_wave_reference", evidence.tree_states, true);
        print_exact("tree_key_record", evidence.tree_key_record, true);
        print_exact("tree_value_record", evidence.tree_value_record, true);
        print_exact("tree_gate_record", evidence.tree_gate_record, false);
        std::cout << "      }" << (index + 1U == record_evidence.size() ? "\n" : ",\n");
    }
    std::cout << "    ],\n"
                 "    \"independent_oracle\": \"each ordinary sequential/tree reference transition passed the represented-input FP64 criterion\"\n"
                 "  },\n  \"guards_and_alias_checks\": \"pass\",\n"
                 "  \"claim\": \"diagnostic-only-no-performance-or-production-claim\"\n}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "gdn_recurrence_exact_discriminator: " << error.what() << '\n';
    return 1;
}
