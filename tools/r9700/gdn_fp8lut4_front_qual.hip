// Qualifies the FP8LUT4 GDN verification front on physical gfx1201:
//   * fp8_gdn_normalized_front: g/beta bitwise equal to the A8G64 front (the same seam and control
//     arithmetic), and the per-token E4M3 image bitwise equal to fp8_quantize_activation of that
//     front's published seam, for T in {1, 2, 5, 6, 7, 8, 12, 16, 24, 28, 32}, with a nonfinite
//     token flagged;
//   * gdn_fp8lut4_pair_conv_record_bf16 bitwise equal to fp8lut4_linear_pair followed by
//     projection_conv_record_bf16 (chain, tree and partial columns; a poisoned token). The FP8LUT4
//     projection itself is qualified against its FP64 oracle by ninfer_r9700_fp8lut4_linear_qual.
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/r9700_linear.h"

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
namespace gdn             = ninfer::ops::r9700::gdn;
namespace lin             = ninfer::ops::r9700::linear;
constexpr unsigned kHeads = 48, K = 5120, QK = 4096, VZ = 12288, V = 6144, C = QK + V, Slots = 4;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template <class T>
struct Device {
    T* data           = nullptr;
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

    void poison() { check(hipMemset(data, 0xa5, count * sizeof(T)), "poison"); }
};

template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

struct Rng {
    std::uint64_t state = 0x6d2b79f5ULL;

    std::uint32_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return std::uint32_t(state >> 16);
    }

    double uniform() { return (next() & 0xffffff) / double(0x1000000); }
};

struct Image {
    Device<std::uint8_t> storage;
    lin::Fp8ActivationWorkspace workspace{};

    Image(unsigned tokens, unsigned columns)
        : storage(lin::fp8_activation_workspace_capacity_bytes(tokens, columns)) {
        check(lin::fp8_bind_activation_workspace(storage.data, storage.count, tokens, columns,
                                                 &workspace),
              "bind");
    }

    std::vector<std::uint8_t> bytes() const {
        std::vector<std::uint8_t> out(workspace.code_bytes + workspace.scale_bytes +
                                      workspace.tokens * 4U);
        check(hipMemcpy(out.data(), workspace.codes, workspace.code_bytes, hipMemcpyDeviceToHost),
              "codes");
        check(hipMemcpy(out.data() + workspace.code_bytes, workspace.scales, workspace.scale_bytes,
                        hipMemcpyDeviceToHost),
              "scales");
        check(hipMemcpy(out.data() + workspace.code_bytes + workspace.scale_bytes, workspace.status,
                        workspace.tokens * 4U, hipMemcpyDeviceToHost),
              "status");
        return out;
    }
};

// A CacheWarm over a scratch allocation (lengths not multiples of the warm stride): the Op runs
// with its warm CTAs, which must not change any result.
ninfer::CacheWarm scratch_warm() {
    static std::uint8_t* scratch = [] {
        void* bytes = nullptr;
        if (hipMalloc(&bytes, std::size_t{6} << 20U) != hipSuccess)
            throw std::runtime_error("warm scratch");
        return static_cast<std::uint8_t*>(bytes);
    }();
    return {{{scratch, (std::size_t{4} << 20U) + 100U},
             {scratch + (std::size_t{4} << 20U), (std::size_t{1} << 20U) + 7U}}};
}

void run_front(unsigned T, bool poison, hipStream_t s) {
    const std::string label = "front T" + std::to_string(T) + (poison ? " poisoned" : "");
    Rng rng;
    std::vector<hip_bfloat16> residual(T * K), norm(K), a(kHeads * K), b(kHeads * K);
    for (auto& v : residual) v = hip_bfloat16(float(4.0 * rng.uniform() - 2.0));
    for (auto& v : norm) v = hip_bfloat16(float(0.4 * rng.uniform() - 0.2));
    for (auto* w : {&a, &b})
        for (auto& v : *w) v = hip_bfloat16(float(0.04 * rng.uniform() - 0.02));
    if (poison) residual[K / 2] = hip_bfloat16(NAN); // token 0
    std::vector<float> a_log(kHeads), dt_bias(kHeads);
    for (auto& v : a_log) v = float(rng.uniform() - 0.5);
    for (auto& v : dt_bias) v = float(rng.uniform() - 0.5);
    Device<hip_bfloat16> d_residual(T * K), d_norm(K), d_a(kHeads * K), d_b(kHeads * K),
        seam(T * K);
    Device<float> d_a_log(kHeads), d_dt(kHeads), g_q4(T * kHeads), beta_q4(T * kHeads),
        g8(T * kHeads), beta8(T * kHeads);
    d_residual.put(residual);
    d_norm.put(norm);
    d_a.put(a);
    d_b.put(b);
    d_a_log.put(a_log);
    d_dt.put(dt_bias);
    const std::size_t ws = lin::a8q4g64_activation_workspace_capacity_bytes(T, K);
    Device<std::uint8_t> d_ws(ws);
    lin::A8G64ActivationWorkspace planes{};
    check(lin::a8q4g64_bind_activation_workspace(d_ws.data, ws, T, K, &planes), "bind a8");
    check(gdn::bf16_gdn_normalized_front(d_residual.data, d_norm.data, 1e-6F, true, d_a.data,
                                         d_b.data, d_a_log.data, d_dt.data, g_q4.data, beta_q4.data,
                                         planes, seam.data, nullptr, s),
          "a8 front");
    Image fused(T, K), reference(T, K);
    std::vector<std::uint32_t> stale(T, 7U);
    check(hipMemcpy(fused.workspace.status, stale.data(), T * 4U, hipMemcpyHostToDevice), "stale");
    check(gdn::fp8_gdn_normalized_front(d_residual.data, d_norm.data, 1e-6F, true, d_a.data,
                                        d_b.data, d_a_log.data, d_dt.data, g8.data, beta8.data,
                                        fused.workspace, s, scratch_warm()),
          "fp8 front");
    check(lin::fp8_quantize_activation({seam.data, reference.workspace}, s), "reference quantize");
    check(hipStreamSynchronize(s), "sync");
    require(same(g8.get(), g_q4.get()) && same(beta8.get(), beta_q4.get()),
            label + " controls differ");
    require(same(fused.bytes(), reference.bytes()), label + " E4M3 image differs");
    std::vector<std::uint32_t> status(T);
    check(hipMemcpy(status.data(), fused.workspace.status, T * 4U, hipMemcpyDeviceToHost),
          "status");
    for (unsigned t = 0; t < T; ++t)
        require(status[t] ==
                    ((poison && t == 0U) ? lin::Fp8ActivationNonfinite : lin::Fp8ActivationOk),
                label + " status");
    std::cout << label << ": bitwise\n";
}

struct Fp8Lut4Matrix {
    Device<std::uint8_t> codes, groups;
    Device<float> scales;
    unsigned rows;

    Fp8Lut4Matrix(unsigned n, Rng& rng)
        : codes(std::size_t(n) * K / 2), groups(std::size_t(n) * K / 32), scales(n), rows(n) {
        std::vector<std::uint8_t> c(codes.count), g(groups.count);
        for (auto& v : c) v = std::uint8_t(rng.next());
        for (auto& v : g) v = std::uint8_t((20U + rng.next() % 10U) << 3U | (rng.next() & 7U));
        std::vector<float> r(n);
        for (auto& v : r) v = float(std::ldexp(1.0 + rng.uniform(), -10));
        codes.put(c);
        groups.put(g);
        scales.put(r);
    }

    lin::Fp8Lut4Weight view() const { return {codes.data, groups.data, scales.data, rows, K}; }
};

struct Case {
    unsigned width;
    bool tree;
    bool partial;
    bool poison;
};

void run_pair(const Case& c, hipStream_t s) {
    const unsigned T        = c.width;
    const std::string label = "pair conv W" + std::to_string(T) + (c.tree ? " tree" : "") +
                              (c.partial ? " partial" : "") + (c.poison ? " poisoned" : "");
    Rng rng;
    std::vector<hip_bfloat16> x(T * K), conv_weight(4 * C), states(Slots * 3 * C);
    for (auto& v : x)
        v = hip_bfloat16(float(2.0 * rng.uniform() - 1.0) * (rng.next() % 61 == 0 ? 8.0F : 1.0F));
    if (c.poison) x[std::size_t(T / 2U) * K + 11U] = hip_bfloat16(NAN);
    for (auto& v : conv_weight) v = hip_bfloat16(float(rng.uniform() - 0.5));
    for (auto& v : states) v = hip_bfloat16(float(2.0 * rng.uniform() - 1.0));
    Fp8Lut4Matrix qk(QK, rng), vz(VZ, rng);
    const std::int32_t initial_slot = 1, valid = c.partial ? std::int32_t(T - 2) : std::int32_t(T);
    std::vector<std::int32_t> parents(T);
    for (unsigned t = 0; t < T; ++t)
        parents[t] = c.tree ? (t == 0 ? -1 : std::int32_t(t / 2)) : std::int32_t(t) - 1;
    Device<hip_bfloat16> d_x(T * K), d_conv_weight(4 * C), d_states(Slots * 3 * C);
    Device<std::int32_t> d_valid(1), d_initial(1), d_parents(T);
    d_x.put(x);
    d_conv_weight.put(conv_weight);
    d_states.put(states);
    d_valid.put({valid});
    d_initial.put({initial_slot});
    d_parents.put(parents);
    Image image(T, K);
    check(lin::fp8_quantize_activation({d_x.data, image.workspace}, s), "quantize");
    const std::int32_t* parent_arg = c.tree ? d_parents.data : nullptr;
    Device<hip_bfloat16> record(T * C), query(T * 2048), key(T * 2048), value(T * V), z(T * V);
    Device<hip_bfloat16> qk_out(T * QK), vz_out(T * VZ);

    struct Outputs {
        std::vector<hip_bfloat16> record, query, key, value, z;
    };

    const auto collect = [&] {
        check(hipStreamSynchronize(s), "sync");
        return Outputs{record.get(), query.get(), key.get(), value.get(), z.get()};
    };
    const auto poison_outputs = [&] {
        record.poison();
        query.poison();
        key.poison();
        value.poison();
        z.poison();
    };
    poison_outputs();
    check(gdn::gdn_fp8lut4_pair_conv_record_bf16(
              image.workspace, qk.view(), vz.view(), d_conv_weight.data, d_states.data,
              d_valid.data, d_initial.data, parent_arg, record.data, query.data, key.data,
              value.data, z.data, T, Slots, s),
          "fused");
    const Outputs fused = collect();
    poison_outputs();
    check(lin::fp8lut4_linear_pair(qk.view(), {.leading = qk_out.data}, vz.view(),
                                   {.leading = vz_out.data}, image.workspace, s),
          "pair");
    check(gdn::projection_conv_record_bf16(qk_out.data, vz_out.data, d_conv_weight.data,
                                           d_states.data, d_valid.data, d_initial.data, parent_arg,
                                           record.data, query.data, key.data, value.data, z.data, T,
                                           1, Slots, s),
          "record");
    const Outputs composed = collect();
    require(same(fused.record, composed.record) && same(fused.query, composed.query) &&
                same(fused.key, composed.key) && same(fused.value, composed.value) &&
                same(fused.z, composed.z),
            label + " differs from the composition");
    std::cout << label << ": bitwise\n";
}

} // namespace

int main() {
    try {
        hipStream_t s = nullptr;
        check(hipStreamCreate(&s), "stream");
        for (const unsigned T : {1U, 2U, 5U, 6U, 7U, 8U, 12U, 16U, 24U, 28U, 32U})
            run_front(T, false, s);
        run_front(6U, true, s);
        for (const Case c :
             {Case{4, false, false, false}, Case{4, true, false, true},
              Case{5, false, false, false}, Case{6, false, false, false},
              Case{6, true, false, false}, Case{6, false, true, false}, Case{5, true, true, false},
              Case{6, false, false, true}, Case{7, false, false, false}, Case{7, false, true, true},
              Case{8, false, false, false}, Case{8, true, false, true}})
            run_pair(c, s);
        std::cout << "gdn fp8lut4 front qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
