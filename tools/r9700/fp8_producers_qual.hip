// Qualification of the per-token E4M3 activation producers (plain rows, RMSNorm, attention output
// gate, GDN gated per-head RMSNorm) against an FP64 oracle of their
// represented BF16 values: per-token scale max|v| / 448, every decoded code within one E4M3 rounding
// of the value (plus the BF16 seam rounding), and the per-token status words, including a token
// with a nonfinite input, at small and prefill widths.
#include "ops/r9700/linear/fp8_activation.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace linear = ninfer::ops::r9700::linear;
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

double bf16(double value) { return static_cast<float>(hip_bfloat16(static_cast<float>(value))); }
double silu(double z) { return z / (1.0 + std::exp(-z)); }

std::vector<hip_bfloat16> random_rows(std::size_t count, double sigma, Rng& rng) {
    std::vector<hip_bfloat16> out(count);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = hip_bfloat16(static_cast<float>(rng.normal() * sigma * (i % 211U == 0U ? 12.0 : 1.0)));
    return out;
}

// Checks one produced image against the oracle's represented values v[t][k] (NaN marks a token
// whose status must be set).
void check_image(const linear::Fp8ActivationWorkspace& image, const std::vector<double>& v,
                 std::uint32_t tokens, std::uint32_t columns, const std::string& label) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * image.padded_columns);
    std::vector<float> scales(tokens);
    std::vector<std::uint32_t> status(tokens);
    check(hipMemcpy(codes.data(), image.codes, codes.size(), hipMemcpyDeviceToHost), "codes");
    check(hipMemcpy(scales.data(), image.scales, tokens * sizeof(float), hipMemcpyDeviceToHost), "scales");
    check(hipMemcpy(status.data(), image.status, tokens * sizeof(std::uint32_t), hipMemcpyDeviceToHost),
          "status");
    double worst = 0.0;
    for (std::uint32_t t = 0; t < tokens; ++t) {
        const double* row = v.data() + static_cast<std::size_t>(t) * columns;
        bool bad = false;
        double maximum = 0.0;
        for (std::uint32_t k = 0; k < columns; ++k) {
            bad |= !std::isfinite(row[k]);
            if (std::isfinite(row[k])) maximum = std::max(maximum, std::abs(row[k]));
        }
        require(status[t] == (bad ? linear::Fp8ActivationNonfinite : linear::Fp8ActivationOk),
                label + " token " + std::to_string(t) + " status");
        if (bad) continue;
        const double scale = scales[t];
        require(std::abs(scale - maximum / 448.0) <= maximum / 448.0 / 128.0 + 1e-38,
                label + " token " + std::to_string(t) + " scale");
        for (std::uint32_t k = 0; k < columns; ++k) {
            const double decoded =
                oracle::decode_e4m3fn(codes[static_cast<std::size_t>(t) * image.padded_columns + k]) * scale;
            // One E4M3 rounding (half ulp <= 1/16 of the value, 2^-10 of the row maximum in the
            // subnormal range) plus one BF16 seam step.
            const double bound = std::abs(row[k]) * (1.0 / 16.0 + 1.0 / 128.0) + scale * 448.0 * std::ldexp(1.0, -18) + 1e-30;
            worst = std::max(worst, std::abs(decoded - row[k]) / bound);
        }
    }
    require(worst <= 1.0, label + " oracle bound exceeded: " + std::to_string(worst));
    std::cout << label << ": worst/bound " << worst << "\n";
}

struct Image {
    Device<std::uint8_t> storage;
    linear::Fp8ActivationWorkspace workspace{};
    Image(std::uint32_t tokens, std::uint32_t columns)
        : storage(linear::fp8_activation_workspace_capacity_bytes(tokens, columns)) {
        check(hipMemset(storage.data, 0, storage.count), "clear");  // deterministic plane padding
        check(linear::fp8_bind_activation_workspace(storage.data, storage.count, tokens, columns, &workspace),
              "bind");
        const std::vector<std::uint32_t> stale(tokens, 7U);  // producers must overwrite every word
        check(hipMemcpy(workspace.status, stale.data(), tokens * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice), "stale");
    }
};

void run(std::uint32_t tokens, bool poison, Rng& rng, hipStream_t stream) {
    const std::string suffix = " T" + std::to_string(tokens) + (poison ? " poisoned" : "");
    const std::uint32_t bad_token = tokens / 2U;
    const auto inject = [&](std::vector<hip_bfloat16>& rows, std::uint32_t columns) {
        if (poison) rows[static_cast<std::size_t>(bad_token) * columns + columns / 3U] = hip_bfloat16(NAN);
    };
    {  // Plain rows (K5120).
        constexpr std::uint32_t K = 5120;
        auto x = random_rows(static_cast<std::size_t>(tokens) * K, 1.0, rng);
        inject(x, K);
        Device<hip_bfloat16> dx(x.size());
        dx.upload(x);
        Image image(tokens, K);
        check(linear::fp8_quantize_activation({dx.data, image.workspace}, stream), "plain");
        check(hipStreamSynchronize(stream), "sync");
        std::vector<double> v(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) v[i] = static_cast<float>(x[i]);
        check_image(image.workspace, v, tokens, K, "plain" + suffix);
    }
    {  // RMSNorm with unit offset (K5120).
        constexpr std::uint32_t K = 5120;
        auto x = random_rows(static_cast<std::size_t>(tokens) * K, 3.0, rng);
        inject(x, K);
        const auto w = random_rows(K, 0.2, rng);
        Device<hip_bfloat16> dx(x.size()), dw(K);
        dx.upload(x);
        dw.upload(w);
        Image image(tokens, K);
        check(linear::fp8_quantize_normalized_activation({dx.data, dw.data, 1e-6F, true, image.workspace},
                                                         stream), "normalized");
        check(hipStreamSynchronize(stream), "sync");
        std::vector<double> v(x.size());
        for (std::uint32_t t = 0; t < tokens; ++t) {
            double sumsq = 0.0;
            for (std::uint32_t k = 0; k < K; ++k) {
                const double value = static_cast<float>(x[static_cast<std::size_t>(t) * K + k]);
                sumsq += value * value;
            }
            const double inverse = 1.0 / std::sqrt(sumsq / K + 1e-6);
            for (std::uint32_t k = 0; k < K; ++k)
                v[static_cast<std::size_t>(t) * K + k] =
                    bf16(static_cast<float>(x[static_cast<std::size_t>(t) * K + k]) * inverse *
                         (1.0 + static_cast<float>(w[k])));
        }
        check_image(image.workspace, v, tokens, K, "normalized" + suffix);
        // The BF16 side output: the same image, and exactly the rows it encodes (a plain
        // quantization of the side output reproduces it), each within one BF16 step of the oracle.
        Device<hip_bfloat16> dn(x.size());
        Image side(tokens, K), requantized(tokens, K);
        linear::Fp8NormalizedQuantizeArgs args{dx.data, dw.data, 1e-6F, true, side.workspace};
        args.normalized = dn.data;
        check(linear::fp8_quantize_normalized_activation(args, stream), "normalized side");
        check(linear::fp8_quantize_activation({dn.data, requantized.workspace}, stream), "requantize");
        check(hipStreamSynchronize(stream), "sync");
        const auto bytes = [&](const Image& i) {
            std::vector<std::uint8_t> host(i.storage.count);
            check(hipMemcpy(host.data(), i.storage.data, host.size(), hipMemcpyDeviceToHost), "image");
            return host;
        };
        const auto reference = bytes(image);
        require(bytes(side) == reference, "normalized side image differs" + suffix);
        require(bytes(requantized) == reference, "normalized side rows differ from the image" + suffix);
        const auto n = dn.download();
        for (std::size_t i = 0; i < n.size(); ++i) {
            if (!std::isfinite(v[i])) continue;
            const double step = std::ldexp(std::max(std::abs(v[i]), 1e-30), -7);
            require(std::abs(static_cast<float>(n[i]) - v[i]) <= step,
                    "normalized side row outside one BF16 step" + suffix);
        }
    }
    {  // Attention output gate (K6144).
        constexpr std::uint32_t K = 6144;
        auto gate = random_rows(static_cast<std::size_t>(tokens) * K, 2.0, rng);
        inject(gate, K);
        std::vector<float> attention(gate.size());
        for (auto& a : attention) a = static_cast<float>(rng.normal());
        Device<hip_bfloat16> dg(gate.size());
        Device<float> da(attention.size());
        dg.upload(gate);
        da.upload(attention);
        Image image(tokens, K);
        check(linear::fp8_quantize_gated_activation({dg.data, da.data, image.workspace}, stream), "gated");
        check(hipStreamSynchronize(stream), "sync");
        std::vector<double> v(gate.size());
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = bf16(bf16(attention[i]) / (1.0 + std::exp(-static_cast<double>(static_cast<float>(gate[i])))));
        check_image(image.workspace, v, tokens, K, "attention gate" + suffix);
    }
    {  // GDN gated per-head RMSNorm (48 x 128).
        constexpr std::uint32_t K = 6144, head = 128;
        auto x = random_rows(static_cast<std::size_t>(tokens) * K, 0.5, rng);
        inject(x, K);
        const auto z = random_rows(static_cast<std::size_t>(tokens) * K, 2.0, rng);
        const auto w = random_rows(head, 1.0, rng);
        Device<hip_bfloat16> dx(x.size()), dz(z.size()), dw(head);
        dx.upload(x);
        dz.upload(z);
        dw.upload(w);
        Image image(tokens, K);
        check(linear::fp8_quantize_gated_rmsnorm_activation({dx.data, dz.data, dw.data, 1e-6F, image.workspace},
                                                            stream), "gated rmsnorm");
        check(hipStreamSynchronize(stream), "sync");
        std::vector<double> v(x.size());
        for (std::size_t h = 0; h < x.size() / head; ++h) {
            double sumsq = 0.0;
            for (std::uint32_t k = 0; k < head; ++k) {
                const double value = static_cast<float>(x[h * head + k]);
                sumsq += value * value;
            }
            const double inverse = 1.0 / std::sqrt(sumsq / head + 1e-6);
            for (std::uint32_t k = 0; k < head; ++k)
                v[h * head + k] = bf16(static_cast<float>(x[h * head + k]) * inverse *
                                       static_cast<float>(w[k]) * silu(static_cast<float>(z[h * head + k])));
        }
        // A nonfinite input poisons its whole token row in the oracle.
        for (std::uint32_t t = 0; t < tokens; ++t) {
            bool bad = false;
            for (std::uint32_t k = 0; k < K; ++k) bad |= !std::isfinite(v[static_cast<std::size_t>(t) * K + k]);
            if (bad) v[static_cast<std::size_t>(t) * K] = NAN;
        }
        check_image(image.workspace, v, tokens, K, "gdn gated rmsnorm" + suffix);
    }
}

} // namespace

int main() {
    try {
        hipStream_t stream = nullptr;
        check(hipStreamCreate(&stream), "stream");
        Rng rng{0xf8f8ULL};
        for (const std::uint32_t tokens : {1U, 6U, 16U, 17U, 130U, 2048U}) run(tokens, false, rng, stream);
        for (const std::uint32_t tokens : {6U, 300U}) run(tokens, true, rng, stream);
        std::cout << "fp8 producers qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
