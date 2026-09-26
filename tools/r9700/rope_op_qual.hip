#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/rope.h"

#include <algorithm>
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

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, device));
        if (std::string(props.gcnArchName).starts_with("gfx1201") && props.warpSize == 32) {
            if (selected >= 0) { fail("multiple gfx1201 devices require an explicit qualifier selector"); }
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
    hip_bfloat16 out;
    out.data = bf16_bits(value);
    return out;
}

float fp32(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_ != nullptr) { (void)hipFree(data_); } }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct Result {
    double max_abs = 0.0;
    double squared_error = 0.0;
    double squared_oracle = 0.0;
    std::size_t checked = 0;
};

void merge(Result& dst, const Result& src) {
    dst.max_abs = std::max(dst.max_abs, src.max_abs);
    dst.squared_error += src.squared_error;
    dst.squared_oracle += src.squared_oracle;
    dst.checked += src.checked;
}

struct Case {
    std::string name;
    int d;
    int r;
    int q_heads;
    int k_heads;
    int axes;
    int tokens;
    int padding;
    float theta;
};

double frequency(const Case& c, int pair) {
    if (c.axes == 2) {
        return std::pow(static_cast<double>(c.theta),
                        -2.0 * static_cast<double>(pair % 18) / 36.0);
    }
    return std::pow(static_cast<double>(c.theta),
                    -2.0 * static_cast<double>(pair) / static_cast<double>(c.r));
}

int axis_for(const Case& c, int pair) {
    if (c.axes == 2) { return pair / 18; }
    return c.axes == 3 ? pair % 3 : 0;
}

std::vector<hip_bfloat16> fixture(int d, int heads, int tokens, int stride, int salt) {
    std::vector<hip_bfloat16> data(static_cast<std::size_t>(tokens) * stride, bf16(-9.0F));
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int feature = 0; feature < d; ++feature) {
                const int code = ((feature * 17 + head * 29 + token * 13 + salt) % 127) - 63;
                data[static_cast<std::size_t>(token) * stride + head * d + feature] =
                    bf16(static_cast<float>(code) / 96.0F);
            }
        }
    }
    return data;
}

Result check_tensor(const Case& c, const std::vector<std::int32_t>& positions,
                    const std::vector<hip_bfloat16>& input,
                    const std::vector<hip_bfloat16>& output, int heads, int stride) {
    Result result;
    const int half = c.r / 2;
    for (int token = 0; token < c.tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            const std::size_t base = static_cast<std::size_t>(token) * stride + head * c.d;
            for (int pair = 0; pair < half; ++pair) {
                const double angle = static_cast<double>(positions[axis_for(c, pair) * c.tokens + token]) *
                                     frequency(c, pair);
                const double first = fp32(input[base + pair]);
                const double second = fp32(input[base + pair + half]);
                const double expected_first = first * std::cos(angle) - second * std::sin(angle);
                const double expected_second = second * std::cos(angle) + first * std::sin(angle);
                for (const auto [got, expected] : {
                         std::pair{static_cast<double>(fp32(output[base + pair])), expected_first},
                         std::pair{static_cast<double>(fp32(output[base + pair + half])), expected_second}}) {
                    const double error = got - expected;
                    result.max_abs = std::max(result.max_abs, std::abs(error));
                    result.squared_error += error * error;
                    result.squared_oracle += expected * expected;
                    ++result.checked;
                }
            }
            for (int feature = c.r; feature < c.d; ++feature) {
                if (output[base + feature].data != input[base + feature].data) {
                    fail(c.name + ": unrotated feature changed");
                }
            }
        }
        for (int index = heads * c.d; index < stride; ++index) {
            const std::size_t offset = static_cast<std::size_t>(token) * stride + index;
            if (output[offset].data != input[offset].data) { fail(c.name + ": padding changed"); }
        }
    }
    return result;
}

Result run_pair(const Case& c, hipStream_t stream) {
    std::vector<std::int32_t> positions(static_cast<std::size_t>(c.axes) * c.tokens);
    for (int axis = 0; axis < c.axes; ++axis) {
        for (int token = 0; token < c.tokens; ++token) {
            positions[static_cast<std::size_t>(axis) * c.tokens + token] =
                token == c.tokens - 1 ? 262143 - axis * 997 : token * 37 + axis * 113;
        }
    }
    const int q_stride = c.d * c.q_heads + c.padding;
    const int k_stride = c.d * c.k_heads + c.padding + 2;
    const auto q_input = fixture(c.d, c.q_heads, c.tokens, q_stride, 5);
    const auto k_input = fixture(c.d, c.k_heads, c.tokens, k_stride, 41);
    DeviceBuffer<std::int32_t> d_positions(positions.size());
    DeviceBuffer<hip_bfloat16> d_q(q_input.size());
    DeviceBuffer<hip_bfloat16> d_k(k_input.size());
    HIP_CHECK(hipMemcpy(d_positions.get(), positions.data(), d_positions.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q.get(), q_input.data(), d_q.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k.get(), k_input.data(), d_k.bytes(), hipMemcpyHostToDevice));
    ninfer::Tensor p(d_positions.get(), ninfer::DType::I32, {c.tokens, c.axes});
    ninfer::Tensor q(d_q.get(), ninfer::DType::BF16, {c.d, c.q_heads, c.tokens});
    ninfer::Tensor k(d_k.get(), ninfer::DType::BF16, {c.d, c.k_heads, c.tokens});
    q.nb[2] = static_cast<std::int64_t>(q_stride * sizeof(hip_bfloat16));
    k.nb[2] = static_cast<std::int64_t>(k_stride * sizeof(hip_bfloat16));
    ninfer::ops::rope(p, c.r, c.theta, q, k, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<hip_bfloat16> q_output(q_input.size());
    std::vector<hip_bfloat16> k_output(k_input.size());
    HIP_CHECK(hipMemcpy(q_output.data(), d_q.get(), d_q.bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(k_output.data(), d_k.get(), d_k.bytes(), hipMemcpyDeviceToHost));
    Result result = check_tensor(c, positions, q_input, q_output, c.q_heads, q_stride);
    merge(result, check_tensor(c, positions, k_input, k_output, c.k_heads, k_stride));
    return result;
}

Result run_single(const Case& c, int heads, hipStream_t stream) {
    std::vector<std::int32_t> positions(static_cast<std::size_t>(c.axes) * c.tokens);
    for (int axis = 0; axis < c.axes; ++axis) {
        for (int token = 0; token < c.tokens; ++token) {
            positions[static_cast<std::size_t>(axis) * c.tokens + token] = 4093 + axis * 71 + token;
        }
    }
    const int stride = c.d * heads + c.padding;
    const auto input = fixture(c.d, heads, c.tokens, stride, 73);
    DeviceBuffer<std::int32_t> d_positions(positions.size());
    DeviceBuffer<hip_bfloat16> d_x(input.size());
    HIP_CHECK(hipMemcpy(d_positions.get(), positions.data(), d_positions.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x.get(), input.data(), d_x.bytes(), hipMemcpyHostToDevice));
    ninfer::Tensor p(d_positions.get(), ninfer::DType::I32, {c.tokens, c.axes});
    ninfer::Tensor x(d_x.get(), ninfer::DType::BF16, {c.d, heads, c.tokens});
    x.nb[2] = static_cast<std::int64_t>(stride * sizeof(hip_bfloat16));
    ninfer::ops::rope(p, c.r, c.theta, x, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<hip_bfloat16> output(input.size());
    HIP_CHECK(hipMemcpy(output.data(), d_x.get(), d_x.bytes(), hipMemcpyDeviceToHost));
    return check_tensor(c, positions, input, output, heads, stride);
}

template <typename Function>
void expect_invalid(Function&& function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    fail(std::string(label) + ": expected invalid_argument");
}

void check_rejection(hipStream_t stream) {
    DeviceBuffer<std::int32_t> positions(1);
    DeviceBuffer<hip_bfloat16> q(256 * 24);
    DeviceBuffer<hip_bfloat16> k(256 * 4);
    ninfer::Tensor p(positions.get(), ninfer::DType::I32, {1});
    ninfer::Tensor qt(q.get(), ninfer::DType::BF16, {256, 16, 1});
    ninfer::Tensor kt(k.get(), ninfer::DType::BF16, {256, 2, 1});
    expect_invalid([&] { ninfer::ops::rope(p, 64, 1.0e7F, qt, kt, stream); },
                   "retired 16/2 geometry");
    ninfer::Tensor q_live(q.get(), ninfer::DType::BF16, {256, 24, 1});
    ninfer::Tensor k_live(k.get(), ninfer::DType::BF16, {256, 4, 1});
    expect_invalid([&] { ninfer::ops::rope(p, 64, 10'000.0F, q_live, k_live, stream); },
                   "wrong theta");
    ninfer::Tensor alias(k.get(), ninfer::DType::BF16, {256, 24, 1});
    expect_invalid([&] { ninfer::ops::rope(p, 64, 1.0e7F, alias, k_live, stream); },
                   "overlapping q/k");
}

Result check_interleaved_vision(hipStream_t stream) {
    const Case c{"vision-interleaved-qkv", 72, 72, 16, 16, 2, 5, 0, 10'000.0F};
    const int plane = c.d * c.q_heads;
    const auto q_input = fixture(c.d, c.q_heads, c.tokens, plane, 5);
    const auto k_input = fixture(c.d, c.k_heads, c.tokens, plane, 41);
    std::vector<hip_bfloat16> packed(c.tokens * 3 * plane, bf16(-7.0F));
    std::vector<std::int32_t> positions(c.tokens * 2);
    for (int t = 0; t < c.tokens; ++t) {
        positions[t] = t * 3;
        positions[c.tokens + t] = t * 7;
        std::copy_n(q_input.data() + t * plane, plane, packed.data() + t * 3 * plane);
        std::copy_n(k_input.data() + t * plane, plane, packed.data() + (t * 3 + 1) * plane);
    }
    DeviceBuffer<hip_bfloat16> storage(packed.size());
    DeviceBuffer<std::int32_t> position_storage(positions.size());
    HIP_CHECK(hipMemcpy(storage.get(), packed.data(), storage.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(position_storage.get(), positions.data(), position_storage.bytes(), hipMemcpyHostToDevice));
    ninfer::Tensor p(position_storage.get(), ninfer::DType::I32, {c.tokens, 2});
    ninfer::Tensor q(storage.get(), ninfer::DType::BF16, {c.d, c.q_heads, c.tokens});
    ninfer::Tensor k(storage.get() + plane, ninfer::DType::BF16, {c.d, c.k_heads, c.tokens});
    q.nb[2] = k.nb[2] = 3 * plane * sizeof(hip_bfloat16);
    auto alias = k;
    alias.data = storage.get() + 3 * plane; // aliases Q's next token, not its first
    expect_invalid([&] { ninfer::ops::rope(p, c.r, c.theta, q, alias, stream); },
                   "cross-token overlapping interleaved q/k");
    ninfer::ops::rope(p, c.r, c.theta, q, k, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<hip_bfloat16> output(packed.size()), q_output(q_input.size()), k_output(k_input.size());
    HIP_CHECK(hipMemcpy(output.data(), storage.get(), storage.bytes(), hipMemcpyDeviceToHost));
    for (int t = 0; t < c.tokens; ++t) {
        std::copy_n(output.data() + t * 3 * plane, plane, q_output.data() + t * plane);
        std::copy_n(output.data() + (t * 3 + 1) * plane, plane, k_output.data() + t * plane);
        for (int i = 0; i < plane; ++i) {
            if (output[(t * 3 + 2) * plane + i].data != packed[(t * 3 + 2) * plane + i].data)
                fail("interleaved RoPE changed the value plane");
        }
    }
    Result result = check_tensor(c, positions, q_input, q_output, c.q_heads, plane);
    merge(result, check_tensor(c, positions, k_input, k_output, c.k_heads, plane));
    return result;
}

float time_decode(hipStream_t stream, int tokens) {
    constexpr int iterations = 1000;
    const Case c{"text-1d-decode", 256, 64, 24, 4, 1, 1, 0, 1.0e7F};
    DeviceBuffer<std::int32_t> positions(tokens);
    DeviceBuffer<hip_bfloat16> q(static_cast<std::size_t>(tokens) * 256 * 24);
    DeviceBuffer<hip_bfloat16> k(static_cast<std::size_t>(tokens) * 256 * 4);
    ninfer::Tensor p(positions.get(), ninfer::DType::I32, {tokens});
    ninfer::Tensor qt(q.get(), ninfer::DType::BF16, {256, 24, tokens});
    ninfer::Tensor kt(k.get(), ninfer::DType::BF16, {256, 4, tokens});
    for (int i = 0; i < 20; ++i) { ninfer::ops::rope(p, c.r, c.theta, qt, kt, stream); }
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) { ninfer::ops::rope(p, c.r, c.theta, qt, kt, stream); }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    return elapsed * 1000.0F / iterations;
}

} // namespace

// Fused Q/K RMSNorm + RoPE against an FP64 oracle with the BF16 normalization seam. Q and K are
// planes of a token-interleaved [7168,T] projection image, as in production. Criterion per element:
// the seam and output roundings allow one BF16 step (2^-7 relative) of each contributing operand.
void check_qk_norm_rope(hipStream_t stream, int tokens, int axes, Result& total) {
    constexpr int d = 256, qh = 24, kh = 4, row = 7168, r = 64, half = 32;
    constexpr float eps = 1.0e-6F;
    const Case c{"qk-norm-rope", d, r, qh, kh, axes, tokens, 0, 1.0e7F};
    std::vector<std::int32_t> positions(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis)
        for (int token = 0; token < tokens; ++token)
            positions[static_cast<std::size_t>(axis) * tokens + token] =
                token == tokens - 1 ? 262143 - axis * 997 : token * 53 + axis * 211;
    std::vector<hip_bfloat16> image(static_cast<std::size_t>(tokens) * row);
    for (std::size_t i = 0; i < image.size(); ++i) {
        const int code = static_cast<int>((i * 2654435761ULL >> 7) % 2001) - 1000;
        const double scale = 1.0 + static_cast<double>((i / d) % 5) * 7.0;  // per-head magnitude
        image[i] = bf16(static_cast<float>(code / 1000.0 * scale));
    }
    for (int f = 0; f < d; ++f) image[static_cast<std::size_t>(tokens - 1) * row + 3 * d + f] = bf16(0.0F);
    std::vector<hip_bfloat16> q_gain(d), k_gain(d);
    for (int f = 0; f < d; ++f) {
        q_gain[f] = bf16(static_cast<float>(((f * 37) % 101) - 50) / 80.0F);
        k_gain[f] = bf16(static_cast<float>(((f * 53) % 97) - 48) / 70.0F);
    }
    DeviceBuffer<std::int32_t> d_positions(positions.size());
    DeviceBuffer<hip_bfloat16> d_image(image.size()), d_qw(d), d_kw(d);
    DeviceBuffer<hip_bfloat16> d_qn(static_cast<std::size_t>(tokens) * qh * d);
    DeviceBuffer<hip_bfloat16> d_kn(static_cast<std::size_t>(tokens) * kh * d);
    HIP_CHECK(hipMemcpy(d_positions.get(), positions.data(), d_positions.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_image.get(), image.data(), d_image.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qw.get(), q_gain.data(), d_qw.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kw.get(), k_gain.data(), d_kw.bytes(), hipMemcpyHostToDevice));
    ninfer::Tensor p(d_positions.get(), ninfer::DType::I32, {tokens, axes});
    ninfer::Tensor q(d_image.get(), ninfer::DType::BF16, {d, qh, tokens});
    ninfer::Tensor k(d_image.get() + qh * d, ninfer::DType::BF16, {d, kh, tokens});
    q.nb[2] = k.nb[2] = static_cast<std::int64_t>(row * sizeof(hip_bfloat16));
    ninfer::Tensor qw(d_qw.get(), ninfer::DType::BF16, {d});
    ninfer::Tensor kw(d_kw.get(), ninfer::DType::BF16, {d});
    ninfer::Tensor qn(d_qn.get(), ninfer::DType::BF16, {d, qh, tokens});
    ninfer::Tensor kn(d_kn.get(), ninfer::DType::BF16, {d, kh, tokens});
    ninfer::ops::qk_norm_rope(p, r, 1.0e7F, q, k, qw, kw, eps, qn, kn, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<hip_bfloat16> qo(static_cast<std::size_t>(tokens) * qh * d), ko(static_cast<std::size_t>(tokens) * kh * d);
    HIP_CHECK(hipMemcpy(qo.data(), d_qn.get(), d_qn.bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ko.data(), d_kn.get(), d_kn.bytes(), hipMemcpyDeviceToHost));
    const double step = std::ldexp(1.0, -7);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < qh + kh; ++head) {
            const bool query = head < qh;
            const std::size_t in = static_cast<std::size_t>(token) * row + static_cast<std::size_t>(head) * d;
            const auto& gain = query ? q_gain : k_gain;
            const hip_bfloat16* out = query
                ? qo.data() + (static_cast<std::size_t>(token) * qh + head) * d
                : ko.data() + (static_cast<std::size_t>(token) * kh + head - qh) * d;
            double sumsq = 0.0;
            for (int f = 0; f < d; ++f) { const double x = fp32(image[in + f]); sumsq += x * x; }
            const double inverse = 1.0 / std::sqrt(sumsq / d + eps);
            std::vector<double> n(d);
            for (int f = 0; f < d; ++f)
                n[f] = fp32(bf16(static_cast<float>(fp32(image[in + f]) * inverse *
                                                    (fp32(gain[f]) + 1.0))));
            for (int f = 0; f < d; ++f) {
                double expected = n[f], operands = std::abs(n[f]);
                if (f < r) {
                    const int pair = f % half;
                    const double angle =
                        static_cast<double>(positions[axis_for(c, pair) * tokens + token]) *
                        frequency(c, pair);
                    const double first = n[pair], second = n[pair + half];
                    expected = f < half ? first * std::cos(angle) - second * std::sin(angle)
                                        : second * std::cos(angle) + first * std::sin(angle);
                    operands = std::abs(first) + std::abs(second);
                }
                const double got = fp32(out[f]);
                const double error = std::abs(got - expected);
                if (!(error <= step * (operands + std::abs(expected)) + 1.0e-30))
                    fail("qk_norm_rope oracle mismatch at token " + std::to_string(token) +
                         " head " + std::to_string(head) + " feature " + std::to_string(f));
                total.max_abs = std::max(total.max_abs, error / std::max(1.0, std::abs(expected)));
                total.squared_error += error * error;
                total.squared_oracle += expected * expected;
                ++total.checked;
            }
        }
    }
    // Rejections: unsupported geometry and overlapping output.
    expect_invalid([&] { ninfer::ops::qk_norm_rope(p, 128, 1.0e7F, q, k, qw, kw, eps, qn, kn, stream); },
                   "qk_norm_rope rotary_dim");
    expect_invalid([&] { ninfer::ops::qk_norm_rope(p, r, 1.0e7F, q, k, qw, kw, eps, q, kn, stream); },
                   "qk_norm_rope aliasing output");
}

int main() {
    try {
        ninfer::DeviceContext device(select_r9700());
        const std::vector<Case> cases = {
            {"text-1d-split", 256, 64, 24, 4, 1, 8, 14, 1.0e7F},
            {"text-mrope-combined", 256, 64, 24, 4, 3, 9, 10, 1.0e7F},
            {"dflash-1d", 128, 128, 32, 8, 1, 3, 6, 1.0e7F},
            {"vision-2d", 72, 72, 16, 16, 2, 3, 4, 10'000.0F},
        };
        Result total;
        for (const Case& test : cases) {
            const Result result = run_pair(test, device.stream);
            merge(total, result);
            std::cout << test.name << " max_abs=" << result.max_abs << '\n';
        }
        const Case single{"text-mrope-single-k", 256, 64, 24, 4, 3, 5, 8, 1.0e7F};
        merge(total, run_single(single, 4, device.stream));
        check_rejection(device.stream);
        Result fused;
        for (const int tokens : {1, 6, 130})
            for (const int axes : {1, 3}) check_qk_norm_rope(device.stream, tokens, axes, fused);
        std::cout << "qk-norm-rope FP64 seam criterion PASS checked=" << fused.checked
                  << " rel_l2=" << std::sqrt(fused.squared_error / fused.squared_oracle) << '\n';
        merge(total, check_interleaved_vision(device.stream));
        const double rel_l2 = std::sqrt(total.squared_error / total.squared_oracle);
        if (total.max_abs > 0.008 || rel_l2 > 0.006) {
            fail("RoPE numerical criterion failed: max_abs=" + std::to_string(total.max_abs) +
                 " rel_l2=" + std::to_string(rel_l2));
        }
        std::cout << "PASS gfx1201 RoPE: checked=" << total.checked
                  << " max_abs=" << total.max_abs << " rel_l2=" << rel_l2
                  << " text_us_T1..8=";
        for (int tokens = 1; tokens <= 8; ++tokens) {
            if (tokens != 1) { std::cout << ','; }
            std::cout << time_decode(device.stream, tokens);
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
