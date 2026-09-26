// Qualification of the fused small-width FP8 attention boundaries on the real Qwen3.8 shapes
// (input RMSNorm K5120 -> query-key/gate-value N7168 split at 6144; output gate K6144 -> N5120
// residual). The fused producers and epilogues are exact transforms of the unfused chain (eager
// RMSNorm or output gate, E4M3 quantization, small-T projection, feature split or BF16 residual
// add), so every activation code, scale, status word and published BF16 element must be
// bit-identical to it. An independent FP64 product of the decoded consumed activation and weight
// codes with their scales checks the projection itself. Nonfinite inputs must poison every
// published element; every status word starts poisoned, so a finite image must store OK without a
// reset launch.
#include "ops/r9700/eager/eager_ops.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_small_t_linear.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fp8 = ninfer::ops::r9700::linear;
namespace eager = ninfer::ops::r9700::eager;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;

constexpr std::uint32_t kHidden = 5120U, kQuery = 6144U, kKv = 1024U, kPair = kQuery + kKv;
constexpr std::uint16_t kNan = 0x7fc0U;
hipStream_t g_stream = nullptr;

void check(hipError_t error, const char* what) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
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
    void upload(const std::vector<T>& host) {
        check(hipMemcpy(data, host.data(), count * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        check(hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return host;
    }
};

struct Rng {
    std::uint64_t state;
    std::uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 32U);
    }
    double uniform() { return (next() + 0.5) / 4294967296.0; }
    double normal() { return std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307 * uniform()); }
};

std::uint16_t bf16_bits(float value) {
    return hip_bfloat16(value).data;
}
float bf16_value(std::uint16_t bits) {
    std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// Finite E4M3FN weight codes (never 0x7f/0xff) with positive FP32 row scales.
struct Fp8Weight {
    std::vector<std::uint8_t> codes;
    std::vector<float> scales;
    Device<std::uint8_t> device_codes;
    Device<float> device_scales;
    Fp8Weight(std::uint32_t rows, std::uint32_t columns, Rng& rng)
        : codes(static_cast<std::size_t>(rows) * columns), scales(rows),
          device_codes(codes.size()), device_scales(rows) {
        for (auto& code : codes) {
            code = static_cast<std::uint8_t>(rng.next());
            if ((code & 0x7fU) == 0x7fU) code ^= 0x01U;
        }
        for (auto& scale : scales) scale = static_cast<float>(1e-3 * (0.5 + rng.uniform()));
        device_codes.upload(codes);
        device_scales.upload(scales);
    }
    fp8::Fp8RowScaledWeight view(std::uint32_t rows) const {
        return {device_codes.data, device_scales.data, rows};
    }
};

struct Activation {
    Device<std::uint8_t> storage;
    fp8::Fp8ActivationWorkspace workspace{};
    Activation(std::uint32_t tokens, std::uint32_t columns)
        : storage(fp8::fp8_activation_workspace_capacity_bytes(tokens, columns)) {
        check(fp8::fp8_bind_activation_workspace(storage.data, storage.count, tokens, columns,
                                                 &workspace),
              "bind activation");
    }
    // Codes and scales (the planes' alignment padding is unspecified).
    std::vector<std::uint8_t> image() const {
        std::vector<std::uint8_t> result(workspace.code_bytes + workspace.scale_bytes);
        check(hipMemcpy(result.data(), workspace.codes, workspace.code_bytes,
                        hipMemcpyDeviceToHost), "codes");
        check(hipMemcpy(result.data() + workspace.code_bytes, workspace.scales,
                        workspace.scale_bytes, hipMemcpyDeviceToHost), "scales");
        return result;
    }
    std::uint32_t status() const {
        std::uint32_t value = 0;
        check(hipMemcpy(&value, workspace.status, sizeof(value), hipMemcpyDeviceToHost), "status");
        return value;
    }
};

// Checks published BF16 y[t, r] against the FP64 product of the consumed activation planes.
void check_oracle(const std::vector<std::uint16_t>& output, std::uint32_t output_width,
                  std::uint32_t output_offset, const Fp8Weight& weight, std::uint32_t row_begin,
                  std::uint32_t row_end, const Activation& activation, std::uint32_t tokens,
                  std::uint32_t columns, const std::vector<std::uint16_t>* base,
                  const std::string& label) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> scales(tokens);
    check(hipMemcpy(codes.data(), activation.workspace.codes, codes.size(), hipMemcpyDeviceToHost),
          "codes");
    check(hipMemcpy(scales.data(), activation.workspace.scales, tokens * sizeof(float),
                    hipMemcpyDeviceToHost), "scales");
    double worst = 0.0;
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t r = row_begin; r < row_end; r += 7U) {
            double sum = 0.0;
            const std::uint8_t* w = weight.codes.data() + static_cast<std::size_t>(r) * columns;
            const std::uint8_t* a = codes.data() + static_cast<std::size_t>(t) * columns;
            for (std::uint32_t k = 0; k < columns; ++k)
                sum += oracle::decode_e4m3fn(w[k]) * oracle::decode_e4m3fn(a[k]);
            double expected = sum * weight.scales[r] * scales[t];
            const std::size_t index =
                static_cast<std::size_t>(t) * output_width + (r - row_begin) + output_offset;
            double got = bf16_value(output[index]);
            if (base != nullptr) {
                // Residual: got = BF16(base + BF16(y)); compare y = got - base within both roundings.
                const double before = bf16_value((*base)[index]);
                expected += before;
                got = bf16_value(output[index]);
            }
            const double error = std::abs(got - expected);
            const double bound = std::abs(expected) * (1.0 / 128.0) + 1e-6 +
                                 (base != nullptr ? std::abs(bf16_value((*base)[index])) / 128.0
                                                  : 0.0);
            worst = std::max(worst, error / bound);
        }
    }
    require(worst <= 1.0, label + " FP64 oracle bound exceeded: " + std::to_string(worst));
}

void run_front(std::uint32_t tokens, Rng& rng, bool poison) {
    const std::string label = "front T" + std::to_string(tokens) + (poison ? " poison" : "");
    const std::size_t hidden_elements = static_cast<std::size_t>(tokens) * kHidden;
    std::vector<std::uint16_t> x(hidden_elements), norm(kHidden);
    for (auto& v : x) v = bf16_bits(static_cast<float>(rng.normal() * 3.0));
    for (auto& v : norm) v = bf16_bits(static_cast<float>(rng.normal() * 0.2));
    if (poison) x[static_cast<std::size_t>(tokens - 1U) * kHidden + 17U] = 0x7f80U;
    Device<std::uint16_t> device_x(hidden_elements), device_norm(kHidden), h(hidden_elements);
    device_x.upload(x);
    device_norm.upload(norm);
    Fp8Weight query_key(kPair, kHidden, rng), gate_value(kPair, kHidden, rng);
    Activation reference(tokens, kHidden), fused(tokens, kHidden);
    const std::size_t pair_elements = static_cast<std::size_t>(tokens) * kPair;
    Device<std::uint16_t> qk(pair_elements), gv(pair_elements);
    Device<std::uint16_t> q(static_cast<std::size_t>(tokens) * kQuery),
        g(static_cast<std::size_t>(tokens) * kQuery), k(static_cast<std::size_t>(tokens) * kKv),
        v(static_cast<std::size_t>(tokens) * kKv);
    const float eps = 1e-6F;
    auto* x_bf16 = reinterpret_cast<const hip_bfloat16*>(device_x.data);
    auto* norm_bf16 = reinterpret_cast<const hip_bfloat16*>(device_norm.data);
    // Unfused chain; both status words start poisoned to prove the plain stores.
    const std::uint32_t stale = fp8::Fp8ActivationNonfinite;
    check(hipMemcpy(reference.workspace.status, &stale, sizeof(stale), hipMemcpyHostToDevice),
          "stale");
    check(eager::rmsnorm_bf16(x_bf16, norm_bf16, reinterpret_cast<hip_bfloat16*>(h.data), kHidden,
                              tokens, eps, true, g_stream), "rmsnorm");
    check(fp8::fp8_quantize_activation({reinterpret_cast<const hip_bfloat16*>(h.data),
                                        reference.workspace}, g_stream), "quantize");
    check(fp8::fp8_small_t_linear(query_key.view(kPair), reference.workspace,
                                  reinterpret_cast<hip_bfloat16*>(qk.data), g_stream), "qk");
    check(fp8::fp8_small_t_linear(gate_value.view(kPair), reference.workspace,
                                  reinterpret_cast<hip_bfloat16*>(gv.data), g_stream), "gv");
    check(hipMemcpy(fused.workspace.status, &stale, sizeof(stale), hipMemcpyHostToDevice), "stale");
    check(fp8::fp8_quantize_normalized_activation(
              {x_bf16, norm_bf16, eps, true, fused.workspace}, g_stream), "fused quantize");
    check(fp8::fp8_small_t_pair_split(
              {query_key.view(kPair), gate_value.view(kPair), kQuery,
               reinterpret_cast<hip_bfloat16*>(q.data), reinterpret_cast<hip_bfloat16*>(k.data),
               reinterpret_cast<hip_bfloat16*>(g.data), reinterpret_cast<hip_bfloat16*>(v.data)},
              fused.workspace, g_stream), "pair split");
    check(hipDeviceSynchronize(), "sync");
    require(reference.status() == (poison ? fp8::Fp8ActivationNonfinite : fp8::Fp8ActivationOk),
            label + " reference status");
    require(fused.status() == reference.status(), label + " fused status");
    if (!poison) {
        const auto reference_bytes = reference.image();
        const auto fused_bytes = fused.image();
        require(reference_bytes == fused_bytes, label + " activation image differs");
    }
    const auto qk_host = qk.download(), gv_host = gv.download();
    const auto q_host = q.download(), k_host = k.download(), g_host = g.download(),
               v_host = v.download();
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t r = 0; r < kPair; ++r) {
            const std::size_t source = static_cast<std::size_t>(t) * kPair + r;
            const bool leading = r < kQuery;
            const std::size_t target = leading ? static_cast<std::size_t>(t) * kQuery + r
                                               : static_cast<std::size_t>(t) * kKv + r - kQuery;
            const std::uint16_t first = leading ? q_host[target] : k_host[target];
            const std::uint16_t second = leading ? g_host[target] : v_host[target];
            require(first == qk_host[source] && second == gv_host[source],
                    label + " split output differs at token " + std::to_string(t) + " row " +
                        std::to_string(r));
            if (poison) require(first == kNan && second == kNan, label + " not poisoned");
        }
    }
    if (!poison) {
        check_oracle(q_host, kQuery, 0U, query_key, 0U, kQuery, fused, tokens, kHidden, nullptr,
                     label + " query");
        check_oracle(v_host, kKv, 0U, gate_value, kQuery, kPair, fused, tokens, kHidden, nullptr,
                     label + " value");
    }
}

void run_output(std::uint32_t tokens, Rng& rng, bool poison) {
    const std::string label = "output T" + std::to_string(tokens) + (poison ? " poison" : "");
    const std::size_t elements = static_cast<std::size_t>(tokens) * kQuery;
    const std::size_t residual_elements = static_cast<std::size_t>(tokens) * kHidden;
    std::vector<std::uint16_t> gate(elements), residual(residual_elements);
    std::vector<float> attention(elements);
    for (auto& v : gate) v = bf16_bits(static_cast<float>(rng.normal() * 2.0));
    for (auto& v : attention) v = static_cast<float>(rng.normal() * 0.5);
    for (auto& v : residual) v = bf16_bits(static_cast<float>(rng.normal() * 4.0));
    if (poison) attention[static_cast<std::size_t>(tokens / 2U) * kQuery + 5U] = NAN;
    Device<std::uint16_t> device_gate(elements), gated(elements), delta(residual_elements),
        reference_residual(residual_elements), fused_residual(residual_elements);
    Device<float> device_attention(elements);
    device_gate.upload(gate);
    device_attention.upload(attention);
    reference_residual.upload(residual);
    fused_residual.upload(residual);
    Fp8Weight weight(kHidden, kQuery, rng);
    Activation reference(tokens, kQuery), fused(tokens, kQuery);
    auto* gate_bf16 = reinterpret_cast<const hip_bfloat16*>(device_gate.data);
    const std::uint32_t stale = fp8::Fp8ActivationNonfinite;
    check(hipMemcpy(reference.workspace.status, &stale, sizeof(stale), hipMemcpyHostToDevice),
          "stale");
    check(eager::sigmoid_mul_fp32_bf16(gate_bf16, device_attention.data,
                                       reinterpret_cast<hip_bfloat16*>(gated.data), elements,
                                       g_stream), "sigmoid mul");
    check(fp8::fp8_quantize_activation({reinterpret_cast<const hip_bfloat16*>(gated.data),
                                        reference.workspace}, g_stream), "quantize");
    check(fp8::fp8_small_t_linear(weight.view(kHidden), reference.workspace,
                                  reinterpret_cast<hip_bfloat16*>(delta.data), g_stream), "out");
    check(eager::residual_add_bf16(reinterpret_cast<const hip_bfloat16*>(delta.data),
                                   reinterpret_cast<hip_bfloat16*>(reference_residual.data),
                                   residual_elements, g_stream), "residual add");
    check(hipMemcpy(fused.workspace.status, &stale, sizeof(stale), hipMemcpyHostToDevice), "stale");
    check(fp8::fp8_quantize_gated_activation({gate_bf16, device_attention.data, fused.workspace},
                                             g_stream), "fused quantize");
    check(fp8::fp8_small_t_residual(weight.view(kHidden), fused.workspace,
                                    reinterpret_cast<hip_bfloat16*>(fused_residual.data), g_stream),
          "fused residual");
    check(hipDeviceSynchronize(), "sync");
    require(reference.status() == (poison ? fp8::Fp8ActivationNonfinite : fp8::Fp8ActivationOk),
            label + " reference status");
    require(fused.status() == reference.status(), label + " fused status");
    if (!poison) require(reference.image() == fused.image(), label + " activation image differs");
    const auto expected = reference_residual.download(), got = fused_residual.download();
    for (std::size_t i = 0; i < residual_elements; ++i) {
        require(expected[i] == got[i], label + " residual differs at " + std::to_string(i));
        if (poison) require((got[i] & 0x7fffU) > 0x7f80U, label + " residual not poisoned");
    }
    if (!poison)
        check_oracle(got, kHidden, 0U, weight, 0U, kHidden, fused, tokens, kQuery, &residual,
                     label + " projection");
}

} // namespace

int main() {
    try {
        check(hipStreamCreate(&g_stream), "stream");
        Rng rng{0x5eed1234ULL};
        for (const std::uint32_t tokens : {1U, 2U, 5U, 6U, 8U, 16U}) {
            run_front(tokens, rng, false);
            run_output(tokens, rng, false);
        }
        for (const std::uint32_t tokens : {1U, 6U, 16U}) {
            run_front(tokens, rng, true);
            run_output(tokens, rng, true);
        }
        std::cout << "fp8 attention fused qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
