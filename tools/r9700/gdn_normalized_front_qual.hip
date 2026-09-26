// Qualifies the verification-width GDN front (ops::r9700::gdn::bf16_gdn_normalized_front) at its
// semantic boundaries on physical gfx1201, for T in {2, 5, 6, 12, 18, 24}:
//   * the published BF16 RMSNorm seam against FP64 x * rsqrt(mean(x^2) + eps) * (1 + w);
//   * g/beta against the independent FP64 control oracle of that seam (BF16 dot seams with the
//     FP32-accumulation neighbour, FP64 gates), under the existing public control criteria;
//   * the A8G64 planes against an exact host codec of that seam, byte for byte;
//   * the production call (no seam output) bitwise equal to the seam-publishing call;
//   * a stale status word replaced without a reset launch, and the nonfinite/overflow flags;
//   * supplementary parity with the eager RMSNorm + T1..24 control + small-T codec composition,
//     and the A/B time of the two routes.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
namespace gdn = ninfer::ops::r9700::gdn;
namespace lin = ninfer::ops::r9700::linear;
constexpr unsigned N = 48, K = 5120;
constexpr float kEps = 1e-6F;

void require(bool p, const std::string& message) {
    if (!p) throw std::runtime_error(message);
}
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
template <class T> struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) { check(hipMalloc(&data, n * sizeof(T)), "malloc"); }
    ~Device() { (void)hipFree(data); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& host) {
        require(host.size() == count, "upload size");
        check(hipMemcpy(data, host.data(), count * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> get() const {
        std::vector<T> host(count);
        check(hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return host;
    }
    void poison() {
        check(hipMemset(data, 0xff, count * sizeof(T)), "poison");
        check(hipDeviceSynchronize(), "poison sync");
    }
};
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

std::uint16_t half_bits(float value) {
    const _Float16 h = static_cast<_Float16>(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}
float half_value(std::uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(bits));
    return static_cast<float>(h);
}

// Independent mathematical BF16 RNE seam.
double bf16_rne(double x) {
    if (x == 0) return x;
    int exponent = 0;
    std::frexp(std::abs(x), &exponent);
    const double quantum = std::ldexp(1.0, std::max(exponent - 8, -133));
    return std::nearbyint(x / quantum) * quantum;
}

// The control oracle of gdn_projected_controls_qual: FP64 dots of the BF16 seam, the BF16 RNE
// projection seam plus the adjacent value when an FP32 accumulation bound straddles its midpoint.
struct Reference { std::vector<double> g, g_alt, beta, beta_alt; };
Reference control_oracle(unsigned t, const std::vector<hip_bfloat16>& x,
                         const std::vector<hip_bfloat16>& aw, const std::vector<hip_bfloat16>& bw,
                         const std::vector<float>& log, const std::vector<float>& bias) {
    Reference result;
    const double gamma = double(K) * std::ldexp(1.0, -24);
    for (unsigned token = 0; token < t; ++token)
        for (unsigned head = 0; head < N; ++head) {
            double a = 0, b = 0, a_abs = 0, b_abs = 0;
            for (unsigned k = 0; k < K; ++k) {
                const double v = float(x[token * K + k]);
                const double pa = v * double(float(aw[head * K + k]));
                const double pb = v * double(float(bw[head * K + k]));
                a += pa; b += pb; a_abs += std::abs(pa); b_abs += std::abs(pb);
            }
            const auto gate = [&](double ar) {
                const double av = ar + double(bias[head]);
                return -std::exp(double(log[head])) *
                       (std::max(av, 0.0) + std::log1p(std::exp(-std::abs(av))));
            };
            const auto sigmoid = [](double bv) {
                return bv >= 0 ? 1 / (1 + std::exp(-bv)) : std::exp(bv) / (1 + std::exp(bv));
            };
            const auto alternate = [&](double dot, double bound) {
                const double r = bf16_rne(dot);
                if (dot == r) return r;
                int exponent = 0;
                std::frexp(std::abs(r != 0 ? r : dot), &exponent);
                const double step = std::ldexp(1.0, std::max(exponent - 8, -133));
                double neighbour = dot > r ? r + step : r - step;
                if (std::abs(neighbour) < std::abs(r) && std::abs(r) == std::ldexp(1.0, exponent - 1))
                    neighbour = dot > r ? r + step / 2 : r - step / 2;
                return std::abs(dot - (r + neighbour) / 2) <= gamma * bound ? neighbour : r;
            };
            const double ar = bf16_rne(a), br = bf16_rne(b);
            result.g.push_back(gate(ar));
            result.g_alt.push_back(gate(alternate(a, a_abs)));
            result.beta.push_back(sigmoid(br));
            result.beta_alt.push_back(sigmoid(alternate(b, b_abs)));
        }
    return result;
}
double compare(const std::vector<float>& actual, const std::vector<double>& reference,
               const std::vector<double>& alternate, bool beta) {
    double worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto error = [&](double r) {
            const double limit = std::max(beta ? 0.0003 : 0.0625, (beta ? 0.00003 : 0.001) * std::abs(r));
            return std::abs(double(actual[i]) - r) / limit;
        };
        const double e = std::min(error(reference[i]), error(alternate[i]));
        require(std::isfinite(e) && e <= 1, std::string("control oracle failed: ") + (beta ? "beta" : "g"));
        worst = std::max(worst, e);
    }
    return worst;
}

struct Planes { std::vector<std::uint8_t> low, high; std::vector<std::uint16_t> scales; };
Planes host_codec(const std::vector<hip_bfloat16>& seam, unsigned t) {
    Planes p{std::vector<std::uint8_t>(t * K / 2), std::vector<std::uint8_t>(t * K / 2),
             std::vector<std::uint16_t>(t * K / 64)};
    for (unsigned token = 0; token < t; ++token)
        for (unsigned group = 0; group < K / 64; ++group) {
            float maximum = 0.0F;
            for (unsigned lane = 0; lane < 64; ++lane)
                maximum = std::max(maximum, std::abs(float(seam[token * K + group * 64 + lane])));
            std::uint16_t bits = maximum == 0.0F ? 0U : half_bits(maximum / 127.0F);
            if (maximum != 0.0F && bits == 0U) bits = 1U;
            require(std::isfinite(half_value(bits)), "host codec fixture overflows");
            p.scales[token * (K / 64) + group] = bits;
            const float scale = half_value(bits);
            for (unsigned lane = 0; lane < 64; ++lane) {
                const std::size_t index = std::size_t(token) * K + group * 64 + lane;
                const int code = bits == 0U ? 0
                    : std::clamp(static_cast<int>(std::nearbyint(float(seam[index]) / scale)), -127, 127);
                const int low = code & 0x0f, high = (code - low) / 16;
                const unsigned shift = (index & 1U) * 4U;
                p.low[index / 2] |= static_cast<std::uint8_t>(low << shift);
                p.high[index / 2] |= static_cast<std::uint8_t>((high & 0x0f) << shift);
            }
        }
    return p;
}

struct Result { double seam = 0, g = 0, beta = 0, front_us = 0, composition_us = 0; };

Result cell(unsigned t, hipStream_t s) {
    std::vector<hip_bfloat16> aw(N * K), bw(N * K), gains(K), x(t * K);
    for (unsigned h = 0; h < N; ++h)
        for (unsigned k = 0; k < K; ++k) {
            aw[h * K + k] = hip_bfloat16(float(int((h * 31 + k * 17) % 509) - 254) / 1024.0F);
            bw[h * K + k] = hip_bfloat16(float(int((h * 43 + k * 23 + 7) % 503) - 251) / 1024.0F);
        }
    for (unsigned k = 0; k < K; ++k) gains[k] = hip_bfloat16(float(int((k * 29) % 97) - 48) / 64.0F);
    for (unsigned token = 0; token < t; ++token)
        for (unsigned k = 0; k < K; ++k) {
            float v = float(int((k * 37 + token * 31) % 509) - 254) / float(17 + token);
            if ((k * 13 + token) % 997 == 0) v *= 40.0F;  // outliers
            x[token * K + k] = hip_bfloat16(v);
        }
    std::vector<float> log(N), bias(N);
    for (unsigned h = 0; h < N; ++h) { log[h] = float(int(h % 13) - 6) / 8; bias[h] = float(int(h % 17) - 8) / 4; }
    bias[0] = 21; bias[1] = -21;

    Device<hip_bfloat16> d_x(t * K), d_gains(K), d_a(N * K), d_b(N * K), d_seam(t * K), d_hidden(t * K);
    Device<float> d_log(N), d_bias(N), d_g(t * N), d_beta(t * N), d_g2(t * N), d_beta2(t * N);
    d_x.put(x); d_gains.put(gains); d_a.put(aw); d_b.put(bw); d_log.put(log); d_bias.put(bias);
    const std::size_t ws = lin::a8q4g64_activation_workspace_capacity_bytes(t, K);
    Device<std::uint8_t> d_ws(ws), d_ws2(ws);
    lin::A8G64ActivationWorkspace planes{}, planes2{};
    check(lin::a8q4g64_bind_activation_workspace(d_ws.data, ws, t, K, &planes), "bind");
    check(lin::a8q4g64_bind_activation_workspace(d_ws2.data, ws, t, K, &planes2), "bind2");
    const auto front = [&](const hip_bfloat16* input, const hip_bfloat16* norm,
                           const lin::A8G64ActivationWorkspace& p, float* g, float* beta,
                           hip_bfloat16* seam) {
        check(gdn::bf16_gdn_normalized_front(input, norm, kEps, true, d_a.data, d_b.data,
                                             d_log.data, d_bias.data, g, beta, p, seam, nullptr,
                                             s),
              "front");
    };
    const auto plane_bytes = [&](const Device<std::uint8_t>& region, const lin::A8G64ActivationWorkspace& p) {
        const std::vector<std::uint8_t> host = region.get();
        const auto at = [&](const void* ptr) {
            return host.data() + (static_cast<const std::uint8_t*>(ptr) - region.data);
        };
        Planes out{std::vector<std::uint8_t>(at(p.low_codes), at(p.low_codes) + t * K / 2),
                   std::vector<std::uint8_t>(at(p.high_codes), at(p.high_codes) + t * K / 2),
                   std::vector<std::uint16_t>(t * K / 64)};
        std::memcpy(out.scales.data(), at(p.scales), out.scales.size() * sizeof(std::uint16_t));
        std::uint32_t status = 0;
        std::memcpy(&status, at(p.status), sizeof(status));
        return std::make_pair(out, status);
    };

    // Seam-publishing call over a stale workspace.
    d_ws.poison(); d_g.poison(); d_beta.poison(); d_seam.poison();
    front(d_x.data, d_gains.data, planes, d_g.data, d_beta.data, d_seam.data);
    check(hipStreamSynchronize(s), "sync");
    const std::vector<hip_bfloat16> seam = d_seam.get();
    const std::vector<float> g = d_g.get(), beta = d_beta.get();
    const auto [actual, status] = plane_bytes(d_ws, planes);
    require(status == lin::Q4G64ActivationOk, "stale status was not replaced");

    Result result;
    // (1) Seam against the FP64 RMSNorm.
    for (unsigned token = 0; token < t; ++token) {
        double sumsq = 0;
        for (unsigned k = 0; k < K; ++k) { const double v = float(x[token * K + k]); sumsq += v * v; }
        const double inverse = 1.0 / std::sqrt(sumsq / K + double(kEps));
        for (unsigned k = 0; k < K; ++k) {
            const double reference = double(float(x[token * K + k])) * inverse * (1.0 + double(float(gains[k])));
            // One BF16 half step (2^-8 relative at worst) plus FP32 evaluation slack.
            const double limit = std::abs(reference) * (std::ldexp(1.0, -8) + std::ldexp(1.0, -16)) + 1e-30;
            const double error = std::abs(double(float(seam[token * K + k])) - reference) / limit;
            require(std::isfinite(error) && error <= 1.0, "RMSNorm seam differs from FP64");
            result.seam = std::max(result.seam, error);
        }
    }
    // (2) Controls of that seam against the FP64 control oracle.
    const Reference reference = control_oracle(t, seam, aw, bw, log, bias);
    result.g = compare(g, reference.g, reference.g_alt, false);
    result.beta = compare(beta, reference.beta, reference.beta_alt, true);
    // (3) Planes against the exact host codec of the seam.
    const Planes expected = host_codec(seam, t);
    require(same(actual.low, expected.low) && same(actual.high, expected.high) &&
                same(actual.scales, expected.scales),
            "A8G64 planes differ from the exact codec of the seam");
    // (4) The production call (no seam) is bitwise identical and zeroes the handed-off status
    // word of a later Op.
    d_ws2.poison(); d_g2.poison(); d_beta2.poison();
    Device<std::uint32_t> d_later_status(1);
    d_later_status.poison();
    check(gdn::bf16_gdn_normalized_front(d_x.data, d_gains.data, kEps, true, d_a.data, d_b.data,
                                         d_log.data, d_bias.data, d_g2.data, d_beta2.data,
                                         planes2, nullptr, d_later_status.data, s),
          "front with handed-off status");
    check(hipStreamSynchronize(s), "sync");
    require(d_later_status.get()[0] == 0U, "front did not zero the handed-off status word");
    d_ws2.poison(); d_g2.poison(); d_beta2.poison();
    front(d_x.data, d_gains.data, planes2, d_g2.data, d_beta2.data, nullptr);
    check(hipStreamSynchronize(s), "sync");
    const auto [production, production_status] = plane_bytes(d_ws2, planes2);
    require(production_status == lin::Q4G64ActivationOk, "production status");
    require(same(production.low, actual.low) && same(production.high, actual.high) &&
                same(production.scales, actual.scales),
            "production planes differ from the seam-publishing call");
    require(same(d_g2.get(), g) && same(d_beta2.get(), beta),
            "production controls differ from the seam-publishing call");
    // (5) Status flags over a stale word.
    {
        std::vector<hip_bfloat16> bad = x;
        bad[(t - 1) * K + 77] = hip_bfloat16(std::numeric_limits<float>::infinity());
        Device<hip_bfloat16> d_bad(t * K);
        d_bad.put(bad);
        d_ws2.poison();
        front(d_bad.data, d_gains.data, planes2, d_g2.data, d_beta2.data, nullptr);
        check(hipStreamSynchronize(s), "sync");
        require(plane_bytes(d_ws2, planes2).second == lin::Q4G64ActivationNonfinite,
                "nonfinite input not flagged");
        Device<hip_bfloat16> d_big(K);
        d_big.put(std::vector<hip_bfloat16>(K, hip_bfloat16(1.0e7F)));
        d_ws2.poison();
        front(d_x.data, d_big.data, planes2, d_g2.data, d_beta2.data, nullptr);
        check(hipStreamSynchronize(s), "sync");
        require(plane_bytes(d_ws2, planes2).second == lin::Q4G64ActivationScaleOverflow,
                "scale overflow not flagged");
    }
    // (6) Supplementary parity with the unfused composition.
    const ninfer::Tensor x_tensor(d_x.data, ninfer::DType::BF16, {int(K), int(t)});
    const ninfer::Tensor gain_tensor(d_gains.data, ninfer::DType::BF16, {int(K)});
    ninfer::Tensor hidden_tensor(d_hidden.data, ninfer::DType::BF16, {int(K), int(t)});
    const auto composition = [&] {
        ninfer::ops::rmsnorm(x_tensor, gain_tensor, kEps, true, hidden_tensor, s);
        check(gdn::bf16_projected_control(d_hidden.data, d_a.data, d_b.data, d_log.data, d_bias.data,
                                          d_g2.data, d_beta2.data, t, s), "control");
        check(lin::a8g64_quantize_activation({d_hidden.data, planes2}, s), "quantize");
    };
    d_ws2.poison();
    composition();
    check(hipStreamSynchronize(s), "sync");
    const auto [composed, composed_status] = plane_bytes(d_ws2, planes2);
    require(composed_status == 0U && same(d_hidden.get(), seam) && same(composed.low, actual.low) &&
                same(composed.high, actual.high) && same(composed.scales, actual.scales) &&
                same(d_g2.get(), g) && same(d_beta2.get(), beta),
            "front differs from the eager composition");
    // (7) A/B time.
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
    const auto time = [&](const auto& launch) {
        for (int i = 0; i < 20; ++i) launch();
        check(hipEventRecord(e0, s), "record");
        constexpr int kIterations = 200;
        for (int i = 0; i < kIterations; ++i) launch();
        check(hipEventRecord(e1, s), "record");
        check(hipEventSynchronize(e1), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return 1e3 * ms / kIterations;
    };
    result.front_us = time([&] { front(d_x.data, d_gains.data, planes, d_g.data, d_beta.data, nullptr); });
    result.composition_us = time(composition);
    (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--out-json",
                "usage: gdn_normalized_front_qual --out-json FRESH.json");
        const std::filesystem::path path = argv[2];
        require(!std::filesystem::exists(path), "output exists");
        check(hipSetDevice(0), "device");
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props, 0), "props");
        require(std::string(props.gcnArchName).rfind("gfx1201", 0) == 0, "wrong device");
        hipStream_t s{};
        check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
        require(!gdn::bf16_gdn_normalized_front_supported(1) &&
                    !gdn::bf16_gdn_normalized_front_supported(25),
                "front admits a width outside T2..24");
        std::ostringstream out;
        out << std::setprecision(6);
        out << "{\"schema\":\"ninfer.r9700.gdn-normalized-front.v1\",\"target\":\"gfx1201\","
               "\"oracle\":\"FP64 RMSNorm seam (|err|<=(2^-8+2^-16)|ref|); FP64 control oracle of the seam; exact A8G64 codec of the seam\","
               "\"production_equals_seam_call\":true,\"status_flags\":true,\"composition_parity\":true,\"cells\":[";
        bool first = true;
        for (unsigned t : {2U, 5U, 6U, 12U, 18U, 24U}) {
            const Result r = cell(t, s);
            out << (first ? "" : ",") << "{\"tokens\":" << t << ",\"max_seam_error\":" << r.seam
                << ",\"max_g_error\":" << r.g << ",\"max_beta_error\":" << r.beta
                << ",\"front_us\":" << r.front_us << ",\"composition_us\":" << r.composition_us << "}";
            first = false;
            std::cout << "T" << t << ": front " << r.front_us << " us, composition "
                      << r.composition_us << " us\n";
        }
        out << "],\"numerical_passed\":true}\n";
        check(hipStreamDestroy(s), "stream");
        const auto data = out.str();
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        require(fd >= 0, "cannot create output");
        const auto written = ::write(fd, data.data(), data.size());
        ::close(fd);
        require(written == static_cast<ssize_t>(data.size()), "short output write");
        std::cout << "GDN normalized front: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
