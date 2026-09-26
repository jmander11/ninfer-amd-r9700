// Qualifies the fused GDN pair projection + recorded convolution
// (ops::r9700::gdn::gdn_pair_conv_record_bf16) on physical gfx1201 at widths 5 and 6, with and
// without a parent-index tree and a partially valid sequence:
//   * z and the convolution record (the BF16 projections) against an FP64 projection of the
//     decoded A8G64 activation and Q4 weights (relative L2 <= 1e-2 per token, every element
//     within 1e-2 of the token's reference maximum);
//   * query/key/value against the FP64 causal convolution + SiLU of those recorded BF16 values
//     (one BF16 step: |err| <= 2^-8|ref| + 2^-20);
//   * bitwise equality with the prepared pair projection followed by projection_conv_record_bf16;
//   * a nonzero activation status yields the composition's canonical-NaN outputs.
#include "ninfer/types.h"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
namespace gdn = ninfer::ops::r9700::gdn;
namespace lin = ninfer::ops::r9700::linear;
constexpr unsigned K = 5120, G = K / 64, QK = 4096, VZ = 12288, V = 6144, C = QK + V, Slots = 3;

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
        check(hipMemcpy(data, host.data(), count * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> get() const {
        std::vector<T> host(count);
        check(hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return host;
    }
    void poison() {
        check(hipMemset(data, 0xa5, count * sizeof(T)), "poison");
        check(hipDeviceSynchronize(), "poison sync");
    }
};
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
struct Rng {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    std::uint32_t next() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return std::uint32_t(state >> 16); }
    double uniform() { return (next() & 0xffffff) / double(0x1000000); }
};
double half_value(std::uint16_t bits) { return static_cast<double>(std::bit_cast<_Float16>(bits)); }
std::uint16_t half_bits(float value) { return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value)); }

double weight_at(const std::vector<std::uint8_t>& codes, const std::vector<std::uint16_t>& scales,
                 std::uint32_t row, std::uint32_t column) {
    const std::uint32_t group = column / 64, within = column % 64;
    const std::size_t pair = (((std::size_t(row >> 4U) * G + group) * 4U + within / 16U) * 16U + (row & 15U));
    std::uint64_t word = 0;
    std::memcpy(&word, codes.data() + pair * 8U, 8U);
    const unsigned nibble = unsigned(word >> ((within & 15U) * 4U)) & 15U;
    const int code = nibble >= 8U ? int(nibble) - 16 : int(nibble);
    return code * half_value(scales[(std::size_t(row >> 4U) * G + group) * 16U + (row & 15U)]);
}

struct Outputs { std::vector<hip_bfloat16> record, query, key, value, z; };

struct Case { unsigned width; bool tree; bool partial; };

double run_case(const Case& c, hipStream_t s, std::ostream& out) {
    const unsigned T = c.width;
    Rng rng;
    std::vector<hip_bfloat16> x(T * K), conv_weight(4 * C), states(Slots * 3 * C);
    for (auto& v : x) v = hip_bfloat16(float(2.0 * rng.uniform() - 1.0) * (rng.next() % 61 == 0 ? 8.0F : 1.0F));
    for (auto& v : conv_weight) v = hip_bfloat16(float(rng.uniform() - 0.5));
    for (auto& v : states) v = hip_bfloat16(float(2.0 * rng.uniform() - 1.0));
    std::vector<std::uint8_t> qk_codes(std::size_t(QK) * K / 2), vz_codes(std::size_t(VZ) * K / 2);
    std::vector<std::uint16_t> qk_scales(std::size_t(QK) * G), vz_scales(std::size_t(VZ) * G);
    for (auto* codes : {&qk_codes, &vz_codes}) for (auto& b : *codes) b = std::uint8_t(rng.next());
    for (auto* scales : {&qk_scales, &vz_scales})
        for (auto& w : *scales) w = half_bits(float(0.004 * (0.5 + rng.uniform())));
    const std::int32_t initial_slot = 1, valid = c.partial ? std::int32_t(T - 2) : std::int32_t(T);
    std::vector<std::int32_t> parents(T);
    for (unsigned t = 0; t < T; ++t) parents[t] = c.tree ? (t == 0 ? -1 : std::int32_t(t / 2)) : std::int32_t(t) - 1;

    Device<hip_bfloat16> d_x(T * K), d_conv_weight(4 * C), d_states(Slots * 3 * C);
    Device<std::uint8_t> d_qk_codes(qk_codes.size()), d_vz_codes(vz_codes.size());
    Device<std::uint16_t> d_qk_scales(qk_scales.size()), d_vz_scales(vz_scales.size());
    Device<std::int32_t> d_valid(1), d_initial(1), d_parents(T);
    d_x.put(x); d_conv_weight.put(conv_weight); d_states.put(states);
    d_qk_codes.put(qk_codes); d_vz_codes.put(vz_codes); d_qk_scales.put(qk_scales); d_vz_scales.put(vz_scales);
    d_valid.put({valid}); d_initial.put({initial_slot}); d_parents.put(parents);
    const std::size_t ws = lin::a8q4g64_activation_workspace_capacity_bytes(T, K);
    Device<std::uint8_t> d_ws(ws);
    lin::A8G64ActivationWorkspace planes{};
    check(lin::a8q4g64_bind_activation_workspace(d_ws.data, ws, T, K, &planes), "bind");
    check(lin::a8g64_quantize_activation({d_x.data, planes}, s), "quantize");
    check(hipStreamSynchronize(s), "sync");
    const std::int32_t* parent_arg = c.tree ? d_parents.data : nullptr;

    Device<hip_bfloat16> record(T * C), query(T * 2048), key(T * 2048), value(T * V), z(T * V);
    Device<hip_bfloat16> qk_out(T * QK), vz_out(T * VZ);
    const auto fused = [&] {
        check(gdn::gdn_pair_conv_record_bf16(planes, d_qk_codes.data, d_qk_scales.data, d_vz_codes.data,
                                             d_vz_scales.data, d_conv_weight.data, d_states.data,
                                             d_valid.data, d_initial.data, parent_arg, record.data,
                                             query.data, key.data, value.data, z.data, T, Slots, s),
              "fused");
    };
    const auto composition = [&] {
        lin::A8Q4G64SharedActivationArgs args{};
        args.input = nullptr;
        args.activation_workspace = d_ws.data;
        args.activation_workspace_bytes = ws;
        args.projections[0] = {d_qk_codes.data, qk_codes.size(), d_qk_scales.data,
                               qk_scales.size() * 2, qk_out.data, QK};
        args.projections[1] = {d_vz_codes.data, vz_codes.size(), d_vz_scales.data,
                               vz_scales.size() * 2, vz_out.data, VZ};
        args.projection_count = 2;
        args.tokens = T;
        args.columns = K;
        check(lin::a8q4g64_prepared_shared_activation_linear(args, s), "pair");
        check(gdn::projection_conv_record_bf16(qk_out.data, vz_out.data, d_conv_weight.data, d_states.data,
                                               d_valid.data, d_initial.data, parent_arg, record.data,
                                               query.data, key.data, value.data, z.data, T, 1, Slots, s),
              "record");
    };
    const auto collect = [&] {
        check(hipStreamSynchronize(s), "sync");
        return Outputs{record.get(), query.get(), key.get(), value.get(), z.get()};
    };
    const auto poison_outputs = [&] { record.poison(); query.poison(); key.poison(); value.poison(); z.poison(); };
    const auto equal = [](const Outputs& a, const Outputs& b) {
        return same(a.record, b.record) && same(a.query, b.query) && same(a.key, b.key) &&
               same(a.value, b.value) && same(a.z, b.z);
    };

    poison_outputs(); fused(); const Outputs f = collect();
    poison_outputs(); composition(); const Outputs e = collect();
    require(equal(f, e), "fused pair/convolution differs from the composition");

    // FP64 projection of the decoded planes.
    const std::vector<std::uint8_t> host_ws = d_ws.get();
    const auto at = [&](const void* p) { return host_ws.data() + (static_cast<const std::uint8_t*>(p) - d_ws.data); };
    std::vector<double> xa(T * K);
    for (unsigned t = 0; t < T; ++t)
        for (unsigned k = 0; k < K; ++k) {
            const std::size_t i = std::size_t(t) * K + k;
            const unsigned low = (at(planes.low_codes)[i / 2] >> ((i & 1U) * 4U)) & 15U;
            const unsigned hn = (at(planes.high_codes)[i / 2] >> ((i & 1U) * 4U)) & 15U;
            const int high = hn >= 8U ? int(hn) - 16 : int(hn);
            std::uint16_t scale = 0;
            std::memcpy(&scale, at(planes.scales) + (std::size_t(t) * G + k / 64) * 2, 2);
            xa[i] = (int(low) + 16 * high) * half_value(scale);
        }
    double worst = 0;
    for (unsigned t = 0; t < T; ++t) {
        std::vector<double> reference(QK + VZ);
        double ss = 0, maximum = 0;
        for (unsigned r = 0; r < QK + VZ; ++r) {
            double sum = 0;
            for (unsigned k = 0; k < K; ++k)
                sum += xa[std::size_t(t) * K + k] *
                       (r < QK ? weight_at(qk_codes, qk_scales, r, k) : weight_at(vz_codes, vz_scales, r - QK, k));
            reference[r] = sum; ss += sum * sum; maximum = std::max(maximum, std::abs(sum));
        }
        double err2 = 0;
        for (unsigned r = 0; r < QK + VZ; ++r) {
            // z rows publish every token; record rows only the valid tree columns (zero otherwise).
            double actual;
            if (r >= QK + V) actual = float(f.z[std::size_t(t) * V + r - QK - V]);
            else if (std::int32_t(t) < valid) actual = float(f.record[std::size_t(t) * C + r]);
            else { require(float(f.record[std::size_t(t) * C + r]) == 0.0F, "invalid column record not zero"); actual = reference[r]; }
            const double d = actual - reference[r];
            err2 += d * d;
            require(std::abs(d) <= 1e-2 * maximum + 1e-5, "projection element outside the FP64 bound");
        }
        const double rel = std::sqrt(err2 / ss);
        require(rel <= 1e-2, "projection relative L2 outside the FP64 bound");
        worst = std::max(worst, rel);
    }
    // FP64 convolution + SiLU of the recorded values.
    for (unsigned ch = 0; ch < C; ++ch) {
        std::vector<double> h0(T), h1(T), h2(T);
        const double c0 = float(states[initial_slot * 3 * C + ch]), c1 = float(states[initial_slot * 3 * C + C + ch]),
                     c2 = float(states[initial_slot * 3 * C + 2 * C + ch]);
        for (unsigned t = 0; t < unsigned(valid); ++t) {
            const std::int32_t p = parents[t];
            const double a = p < 0 ? c0 : h0[p], b = p < 0 ? c1 : h1[p], cc = p < 0 ? c2 : h2[p];
            const double current = float(f.record[std::size_t(t) * C + ch]);
            const double sum = double(float(conv_weight[ch])) * a + double(float(conv_weight[C + ch])) * b +
                               double(float(conv_weight[2 * C + ch])) * cc + double(float(conv_weight[3 * C + ch])) * current;
            const double reference = sum / (1.0 + std::exp(-sum));
            h0[t] = b; h1[t] = cc; h2[t] = current;
            const double actual = ch < 2048 ? float(f.query[std::size_t(t) * 2048 + ch])
                                : ch < 4096 ? float(f.key[std::size_t(t) * 2048 + ch - 2048])
                                            : float(f.value[std::size_t(t) * V + ch - 4096]);
            require(std::abs(actual - reference) <= std::ldexp(std::abs(reference), -8) + std::ldexp(1.0, -20),
                    "convolution output outside one BF16 step of FP64");
        }
    }
    // Poisoned status: both routes publish the same canonical-NaN outputs.
    const std::uint32_t bad = lin::Q4G64ActivationNonfinite;
    check(hipMemcpy(planes.status, &bad, 4, hipMemcpyHostToDevice), "status");
    poison_outputs(); fused(); const Outputs fp = collect();
    poison_outputs(); composition(); const Outputs ep = collect();
    require(equal(fp, ep) && fp.z[0].data == 0x7fc1U, "poisoned fused outputs differ from the composition");
    out << "{\"width\":" << T << ",\"tree\":" << (c.tree ? "true" : "false") << ",\"partial\":"
        << (c.partial ? "true" : "false") << ",\"max_projection_rel_l2\":" << worst << "}";
    return worst;
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--out-json", "usage: gdn_pair_conv_record_qual --out-json FRESH.json");
        const std::filesystem::path path = argv[2];
        require(!std::filesystem::exists(path), "output exists");
        check(hipSetDevice(0), "device");
        hipStream_t s{};
        check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
        require(gdn::gdn_pair_conv_record_supported(6, 1) && !gdn::gdn_pair_conv_record_supported(6, 2) &&
                    !gdn::gdn_pair_conv_record_supported(7, 1),
                "unexpected fused support set");
        std::ostringstream out;
        out << std::setprecision(6) << "{\"schema\":\"ninfer.r9700.gdn-pair-conv-record.v1\",\"target\":\"gfx1201\","
               "\"oracle\":\"FP64 projection of decoded planes (rel L2<=1e-2, elements<=1e-2 max); FP64 conv+SiLU of the record (one BF16 step)\","
               "\"composition_bitwise\":true,\"poison_bitwise\":true,\"cases\":[";
        bool first = true;
        for (const Case c : {Case{6, false, false}, Case{6, true, false}, Case{6, false, true}, Case{5, true, true}}) {
            if (!first) out << ',';
            first = false;
            run_case(c, s, out);
        }
        out << "],\"numerical_passed\":true}\n";
        check(hipStreamDestroy(s), "stream");
        const auto data = out.str();
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        require(fd >= 0, "cannot create output");
        const auto written = ::write(fd, data.data(), data.size());
        ::close(fd);
        require(written == static_cast<ssize_t>(data.size()), "short output write");
        std::cout << "GDN pair convolution record: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
