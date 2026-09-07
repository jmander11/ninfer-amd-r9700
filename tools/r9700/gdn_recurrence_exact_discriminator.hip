#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ops/r9700/gdn_recurrence/gdn_verify_profile.h"

#include <algorithm>
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

Inputs make_adversarial_inputs() {
    Inputs result = make_inputs(1, 857);
    const std::vector<hip_bfloat16> row = adversarial_normalization_row();
    for (int head = 0; head < kQHeads; ++head) {
        std::copy(row.begin(), row.end(), result.q.begin() + head * kDim);
        std::copy(row.begin(), row.end(), result.k.begin() + head * kDim);
    }
    for (int head = 0; head < kValueHeads; ++head) {
        result.g[head] = -0.125F;
        result.beta[head] = 0.75F;
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
                     "\"schema_version\":1,\"device\":\"gfx1201-wave32\","
                     "\"verify_wave_qk_candidate\":"
                  << (ninfer::ops::r9700::gdn_recurrence::kVerifyWaveQkCandidate ? 1 : 0)
                  << ","
                     "\"normalize_qk\":true,\"cases\":[\"ordinary-t1-vs-snapshot-b1w1\","
                     "\"adversarial-ordinary-t1-vs-snapshot-b1w1\","
                     "\"ordinary-t129-vs-ordinary-t128-plus-snapshot-b1w1\"],"
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
    require_alias_rejection(stream);
    HIP_CHECK(hipStreamDestroy(stream));

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "{\n  \"artifact_type\": \"gdn_recurrence_exact_discriminator_result\",\n"
                 "  \"schema_version\": 1,\n  \"device\": \"gfx1201-wave32\",\n"
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
    std::cout << "  },\n  \"guards_and_alias_checks\": \"pass\",\n"
                 "  \"claim\": \"diagnostic-only-no-performance-or-production-claim\"\n}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "gdn_recurrence_exact_discriminator: " << error.what() << '\n';
    return 1;
}
