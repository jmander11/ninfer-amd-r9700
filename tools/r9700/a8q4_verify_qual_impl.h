#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"
#include "ops/r9700/linear/a8q4_small_batch_projection.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ninfer/ops/linear.h"
#include <cstring>
#include <functional>
#include <sys/file.h>
#include <thread>

namespace {
constexpr unsigned Copies = 3;
unsigned N = 0, K = 0, G = 0;

hipError_t owning_launch(const linear::A8Q4G64CandidateArgs& a, hipStream_t s) {
    return linear::a8q4_small_batch_projection(a, s);
}

constexpr std::size_t Guard     = 256;
const std::filesystem::path Pci = "/sys/bus/pci/devices/0000:13:00.0";

void power() {
    if (read_text(Pci / "vendor") != "0x1002" || read_text(Pci / "device") != "0x7551" ||
        read_text(Pci / "power_dpm_force_performance_level") != "auto")
        fail("R9700 auto required");
}

void transfer(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind, hipStream_t s) {
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, s));
    HIP_CHECK(hipStreamSynchronize(s));
}

template <class T>
struct Guarded {
    DeviceBuffer<std::uint8_t> storage;
    std::size_t count;

    Guarded(std::size_t n, hipStream_t s) : storage(n * sizeof(T) + 2 * Guard), count(n) {
        HIP_CHECK(hipMemsetAsync(storage.get(), 0xa5, storage.bytes(), s));
    }

    T* data() const { return reinterpret_cast<T*>(storage.get() + Guard); }

    std::size_t bytes() const { return count * sizeof(T); }

    std::vector<T> read(hipStream_t s) const {
        std::vector<T> v(count);
        transfer(v.data(), data(), bytes(), hipMemcpyDeviceToHost, s);
        return v;
    }

    void put(const std::vector<T>& v, hipStream_t s) {
        transfer(data(), v.data(), bytes(), hipMemcpyHostToDevice, s);
    }

    void guards(hipStream_t s) const {
        std::array<std::uint8_t, Guard> b;
        for (auto offset : {std::size_t{0}, storage.bytes() - Guard}) {
            transfer(b.data(), storage.get() + offset, Guard, hipMemcpyDeviceToHost, s);
            if (!std::all_of(b.begin(), b.end(), [](auto x) { return x == 0xa5; }))
                fail("canary changed");
        }
    }
};

// Optional in-process GPU lease: the FP64 CPU references run unleased and the
// lease covers only device allocation, launches and readbacks of each cell.
// Do not also wrap the process in flock(1) on the same path (self-deadlock).
std::filesystem::path gpu_lock_path;

struct GpuLease {
    int fd = -1;

    GpuLease() {
        if (gpu_lock_path.empty()) return;
        fd = ::open(gpu_lock_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) fail("cannot open GPU lock " + gpu_lock_path.string());
        while (::flock(fd, LOCK_EX) != 0)
            if (errno != EINTR) {
                ::close(fd);
                fail("cannot take GPU lock");
            }
    }

    ~GpuLease() {
        if (fd >= 0) ::close(fd);
    }

    GpuLease(const GpuLease&)            = delete;
    GpuLease& operator=(const GpuLease&) = delete;
};

// Each index is evaluated by one thread in its own fixed order, so results are
// identical to the sequential loop.
void parallel_for(unsigned count, const std::function<void(unsigned)>& body) {
    const unsigned workers =
        std::max(1U, std::min({count, 8U, std::max(1U, std::thread::hardware_concurrency() / 2)}));
    std::vector<std::thread> threads;
    for (unsigned w = 0; w < workers; ++w)
        threads.emplace_back([&, w] {
            for (unsigned i = w; i < count; i += workers) body(i);
        });
    for (auto& thread : threads) thread.join();
}

template <class T>
void exact(const std::vector<T>& a, const std::vector<T>& b, const char* why) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T))) fail(why);
}

struct Weights {
    Guarded<std::uint8_t> codes;
    Guarded<std::uint16_t> scales;

    Weights(const DecodeDot8Weights& w, hipStream_t s)
        : codes(w.codes.size(), s), scales(w.scales.size(), s) {
        codes.put(w.codes, s);
        scales.put(w.scales, s);
    }
};

struct Output {
    Guarded<hip_bfloat16> value;
    Guarded<std::uint8_t> scratch;

    Output(unsigned t, hipStream_t s)
        : value(t * N, s), scratch(linear::a8q4g64_activation_workspace_capacity_bytes(t, K), s) {}
};

linear::A8Q4G64CandidateArgs arguments(unsigned t, Guarded<hip_bfloat16>& x, Weights& w,
                                       Output& y) {
    return {x.data(),
            w.codes.data(),
            w.codes.bytes(),
            w.scales.data(),
            w.scales.bytes(),
            y.scratch.data(),
            y.scratch.bytes(),
            y.value.data(),
            t,
            N,
            K,
            K};
}

void launch(const linear::A8Q4G64CandidateArgs& a, hipStream_t s) {
    ninfer::Tensor input(const_cast<hip_bfloat16*>(a.input), ninfer::DType::BF16,
                         {static_cast<int>(K), static_cast<int>(a.tokens)});
    ninfer::Tensor output(a.output, ninfer::DType::BF16,
                          {static_cast<int>(N), static_cast<int>(a.tokens)});
    ninfer::Weight weight{};
    weight.qtype  = ninfer::QType::Q4G64_F16S;
    weight.layout = ninfer::QuantLayout::Q4N16K16;
    weight.ndim   = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = N;
    weight.k = weight.shape[1] = weight.padded_shape[1] = K;
    weight.group = weight.group_size = 64;
    weight.scale_dtype               = ninfer::DType::FP16;
    weight.qdata                     = const_cast<std::uint8_t*>(a.weight_codes);
    weight.scales                    = const_cast<std::uint16_t*>(a.weight_scales);
    weight.qdata_bytes               = a.weight_code_bytes;
    weight.scale_bytes               = a.weight_scale_bytes;

    ninfer::ops::linear(input, weight, output,
                        ninfer::DeviceSpan{a.activation_workspace, a.activation_workspace_bytes},
                        s);
}

struct Graph {
    hipGraph_t graph{};
    hipGraphExec_t exec{};

    Graph(hipStream_t s, const std::function<void()>& f) {
        HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        f();
        HIP_CHECK(hipStreamEndCapture(s, &graph));
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }

    ~Graph() {
        (void)hipGraphExecDestroy(exec);
        (void)hipGraphDestroy(graph);
    }

    void run(hipStream_t s) { HIP_CHECK(hipGraphLaunch(exec, s)); }
};

std::vector<double> oracle(unsigned t, const HostActivation& a, const DecodeDot8Weights& w,
                           std::vector<double>* absolute_group_sums = nullptr) {
    std::vector<double> result(t * N);
    if (absolute_group_sums) absolute_group_sums->assign(t * N, 0.0);
    parallel_for(N, [&](unsigned row) {
        for (unsigned token = 0; token < t; ++token) {
            double sum = 0;
            for (unsigned group = 0; group < G; ++group) {
                int dot = 0;
                for (unsigned lane = 0; lane < 64; ++lane) {
                    const std::size_t word =
                        (row / 16) * G * 64 + group * 64 + (lane / 16) * 16 + row % 16;
                    const int nibble =
                        (w.codes[word * 8 + (lane % 16) / 2] >> ((lane % 2) * 4)) & 15;
                    const int wc  = nibble >= 8 ? nibble - 16 : nibble;
                    const auto ai = static_cast<std::size_t>(token) * K + group * 64 + lane;
                    dot += (unsigned_nibble(a.low, ai) + 16 * signed_nibble(a.high, ai)) * wc;
                }
                const double term =
                    static_cast<double>(dot) * half_value(a.scales[token * G + group]) *
                    static_cast<double>(
                        half_value(w.scales[(row / 16) * G * 16 + group * 16 + row % 16]));
                sum += term;
                if (absolute_group_sums) (*absolute_group_sums)[token * N + row] += std::abs(term);
            }
            result[token * N + row] = sum;
        }
    });
    return result;
}

struct Error {
    double relative_l2 = 0, maximum_absolute = 0, reference_maximum = 0;
};

struct PublicReference {
    std::vector<unsigned> rows;
    std::vector<double> values;
};

// Implementation-profile error budget, not the public mathematical oracle.
// Exact INT32 G64 dots and exact FP16*FP16 scale products feed G FP32 FMAs,
// followed by one BF16 RNE output cast. See tools/r9700/README.md.
struct A8ErrorBudget {
    std::vector<double> represented;
    std::vector<double> arithmetic;
};

double gamma(unsigned count, double unit) {
    return std::nextafter((count * unit) / (1.0 - count * unit),
                          std::numeric_limits<double>::infinity());
}

double bound_add(double a, double b) {
    return a == 0.0 && b == 0.0 ? 0.0
                                : std::nextafter(a + b, std::numeric_limits<double>::infinity());
}

double bound_multiply(double a, double b) {
    return a == 0.0 || b == 0.0 ? 0.0
                                : std::nextafter(a * b, std::numeric_limits<double>::infinity());
}

A8ErrorBudget a8_error_budget(unsigned tokens, const HostActivation& activation,
                              const DecodeDot8Weights& weights) {
    std::vector<double> absolute_sums;
    A8ErrorBudget budget;
    budget.represented = oracle(tokens, activation, weights, &absolute_sums);
    budget.arithmetic.resize(budget.represented.size());
    constexpr double ub = 0x1p-8, uf = 0x1p-24, ud = 0x1p-53;
    for (std::size_t i = 0; i < budget.represented.size(); ++i) {
        // Inflate the FP64 absolute sum to bound its own accumulation error.
        const double denominator  = std::nextafter(1.0 - gamma(G, ud), 0.0);
        const double sum          = absolute_sums[i] == 0.0
                                        ? 0.0
                                        : std::nextafter(absolute_sums[i] / denominator,
                                                         std::numeric_limits<double>::infinity());
        const double oracle_round = bound_multiply(gamma(G, ud), sum);
        const double accumulation = bound_multiply(gamma(G, uf), sum);
        const double before_cast  = bound_add(oracle_round, accumulation);
        budget.arithmetic[i] =
            bound_add(before_cast,
                      bound_multiply(ub, bound_add(std::abs(budget.represented[i]), before_cast)));
    }
    return budget;
}

PublicReference public_oracle(unsigned tokens, const std::vector<hip_bfloat16>& input,
                              const DecodeDot8Weights& weights, bool full_output = true) {
    PublicReference reference;
    if (full_output) {
        reference.rows.resize(N);
        for (unsigned row = 0; row < N; ++row) reference.rows[row] = row;
    } else {
        reference.rows = {0, 1, 15, 16, N - 17, N - 16, N - 2, N - 1};
        for (unsigned i = 0; i < 24; ++i) reference.rows.push_back(i * (N - 1) / 23);
        std::sort(reference.rows.begin(), reference.rows.end());
        reference.rows.erase(std::unique(reference.rows.begin(), reference.rows.end()),
                             reference.rows.end());
    }
    // Original represented BF16 public input, not the private A8 image.
    // Decode each signed stored Q4 value with its exact FP16 scale and
    // evaluate the complete K reduction in FP64, without staging casts.
    // Decode invariant stored scales once, not once for every scalar product.
    std::vector<double> decoded_scales(weights.scales.size());
    for (std::size_t i = 0; i < decoded_scales.size(); ++i)
        decoded_scales[i] = half_value(weights.scales[i]);
    const auto count = static_cast<unsigned>(reference.rows.size());
    reference.values.resize(static_cast<std::size_t>(tokens) * count);
    parallel_for(count, [&](unsigned i) {
        const unsigned row = reference.rows[i];
        for (unsigned token = 0; token < tokens; ++token) {
            double sum = 0;
            for (unsigned k = 0; k < K; ++k) {
                const unsigned group = k / 64, lane = k % 64;
                const std::size_t word =
                    (row / 16) * G * 64 + group * 64 + (lane / 16) * 16 + row % 16;
                const int nibble =
                    (weights.codes[word * 8 + (lane % 16) / 2] >> ((lane % 2) * 4)) & 15;
                const int code     = nibble >= 8 ? nibble - 16 : nibble;
                const double scale = decoded_scales[(row / 16) * G * 16 + group * 16 + row % 16];
                sum += static_cast<double>(static_cast<float>(input[token * K + k])) * code * scale;
            }
            reference.values[static_cast<std::size_t>(token) * count + i] = sum;
        }
    });
    return reference;
}

struct PublicError {
    double relative_rms = 0, gross_rms = 0, public_bound_fraction = 0,
           arithmetic_bound_fraction       = 0;
    unsigned zero_reference_nonzero_tokens = 0;
};

PublicError compare_public(unsigned tokens, const std::vector<hip_bfloat16>& actual,
                           const PublicReference& reference, const A8ErrorBudget& budget) {
    PublicError result;
    for (unsigned token = 0; token < tokens; ++token) {
        double e2 = 0, r2 = 0, maximum = 0, b2 = 0, a2 = 0, ab2 = 0;
        for (std::size_t i = 0; i < reference.rows.size(); ++i) {
            const auto index         = token * N + reference.rows[i];
            const double expected    = reference.values[token * reference.rows.size() + i];
            const double observed    = static_cast<float>(actual[index]);
            const double represented = budget.represented[index];
            const double arithmetic  = budget.arithmetic[index];
            if (!std::isfinite(observed) || !std::isfinite(expected) ||
                !std::isfinite(represented) || !std::isfinite(arithmetic))
                fail("nonfinite public Linear output/reference/budget");
            const double error            = observed - expected;
            const double arithmetic_error = std::abs(observed - represented);
            const double quantization =
                represented == expected ? 0.0
                                        : std::nextafter(std::abs(represented - expected),
                                                         std::numeric_limits<double>::infinity());
            const double bound = bound_add(quantization, arithmetic);
            // Both checks are necessary: a large quantization allowance must
            // never hide an arithmetic defect in the represented A8 operation.
            if (std::abs(error) > bound || arithmetic_error > arithmetic)
                fail("public BF16/Q4 A8 forward bound failed: N=" + std::to_string(N) +
                     " K=" + std::to_string(K) + " T=" + std::to_string(tokens) + " token=" +
                     std::to_string(token) + " row=" + std::to_string(reference.rows[i]));
            e2 += error * error;
            r2 += expected * expected;
            maximum = std::max(maximum, std::abs(error));
            b2 += bound * bound;
            a2 += arithmetic_error * arithmetic_error;
            ab2 += arithmetic * arithmetic;
        }
        // Normwise budget plus finite per-output caps above. There is no
        // empirical threshold or zero-reference exception to quantization.
        if (e2 > b2 || a2 > ab2) fail("A8 forward norm bound failed");
        if (b2 > 0)
            result.public_bound_fraction =
                std::max(result.public_bound_fraction, std::sqrt(e2 / b2));
        if (ab2 > 0)
            result.arithmetic_bound_fraction =
                std::max(result.arithmetic_bound_fraction, std::sqrt(a2 / ab2));
        if (r2 == 0) {
            if (e2 != 0) ++result.zero_reference_nonzero_tokens;
            continue;
        }
        const double relative = std::sqrt(e2 / r2);
        const double gross    = maximum / std::sqrt(r2 / reference.rows.size());
        result.relative_rms   = std::max(result.relative_rms, relative);
        result.gross_rms      = std::max(result.gross_rms, gross);
    }
    return result;
}

Error compare(const std::vector<hip_bfloat16>& actual, const std::vector<double>& expected) {
    Error e;
    double e2 = 0, r2 = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double v = static_cast<float>(actual[i]);
        if (!std::isfinite(v)) fail("nonfinite finite output");
        const double delta = v - expected[i];
        e2 += delta * delta;
        r2 += expected[i] * expected[i];
        e.maximum_absolute  = std::max(e.maximum_absolute, std::abs(delta));
        e.reference_maximum = std::max(e.reference_maximum, std::abs(expected[i]));
    }
    e.relative_l2 = std::sqrt(e2 / std::max(r2, 1e-30));
    // BF16 represented-output criterion: normwise plus finite gross pointwise
    // cap, permitting cancellation near zero without allowing isolated corruption.
    if (e.relative_l2 > 0.01 || e.maximum_absolute > 0.01 * e.reference_maximum + 1e-5)
        fail("independent decoded-Q4 FP64 oracle failed");
    return e;
}

linear::A8G64ActivationWorkspace workspace(unsigned t, Output& o) {
    linear::A8G64ActivationWorkspace w{};
    HIP_CHECK(
        linear::a8q4g64_bind_activation_workspace(o.scratch.data(), o.scratch.bytes(), t, K, &w));
    return w;
}

void codec(unsigned t, Output& o, const HostActivation& expected, hipStream_t s) {
    const auto w = workspace(t, o);
    std::vector<std::uint8_t> low(expected.low.size()), high(expected.high.size());
    std::vector<std::uint16_t> scales(expected.scales.size());
    std::uint32_t status = ~0U;
    transfer(low.data(), w.low_codes, low.size(), hipMemcpyDeviceToHost, s);
    transfer(high.data(), w.high_codes, high.size(), hipMemcpyDeviceToHost, s);
    transfer(scales.data(), w.scales, scales.size() * 2, hipMemcpyDeviceToHost, s);
    transfer(&status, w.status, 4, hipMemcpyDeviceToHost, s);
    if (low != expected.low || high != expected.high || scales != expected.scales || status)
        fail("exact A8 codec/status failed");
}

// One represented public input with its complete CPU references.
struct Fixture {
    std::vector<hip_bfloat16> x;
    HostActivation represented;
    A8ErrorBudget budget;
    PublicReference reference;
};

Fixture fixture(unsigned t, std::vector<hip_bfloat16> x, const DecodeDot8Weights& host) {
    Fixture f{std::move(x)};
    f.represented = quantize_host(f.x, t, K);
    f.budget      = a8_error_budget(t, f.represented, host);
    f.reference   = public_oracle(t, f.x, host);
    return f;
}

// Device outputs of one verification pass, compared after the GPU lease ends.
struct Run {
    const Fixture* fixture;
    bool eager;
    std::vector<hip_bfloat16> control;
    std::array<std::vector<hip_bfloat16>, Copies> actual;
};

const char* wide_route_name(linear::detail::A8Q4WideRoute route) {
    using R = linear::detail::A8Q4WideRoute;
    return route == R::TiledM               ? "tiled_m"
           : route == R::TokenTileN64       ? "token_tile_n64"
           : route == R::TokenTileN64SplitK ? "token_tile_n64_split_k"
           : route == R::TokenTileN128      ? "token_tile_n128"
                                            : "none";
}

linear::A8Q4G64LinearArgs prepared(unsigned t, const linear::A8Q4G64CandidateArgs& a,
                                   const linear::A8G64ActivationWorkspace& w) {
    return {w.low_codes,
            w.low_code_bytes,
            w.high_codes,
            w.high_code_bytes,
            w.scales,
            w.scale_bytes,
            w.status,
            a.weight_codes,
            a.weight_code_bytes,
            a.weight_scales,
            a.weight_scale_bytes,
            a.output,
            t,
            N,
            K,
            K};
}

std::vector<hip_bfloat16> activation(unsigned t) {
    const auto base = make_decode_dot8_input(K);
    std::vector<hip_bfloat16> x(t * K);
    for (unsigned token = 0; token < t; ++token)
        for (unsigned k = 0; k < K; ++k)
            x[token * K + k] =
                hip_bfloat16(static_cast<float>(base[(k / 64) * 64 + (k + token * 13) % 64]) *
                             (token + 4) / 8.0F);
    return x;
}

void cell(unsigned t, hipStream_t s, std::ostream& out) {
    const auto host                       = make_decode_dot8_weights(N, K);
    const auto x                          = activation(t);
    auto poisoned                         = x;
    poisoned[(t > 1U ? K : 0U) + 17].data = 0x7fc1;
    auto negated                          = x;
    for (auto& v : negated) v = hip_bfloat16(-0.75F * static_cast<float>(v));
    const Fixture original = fixture(t, x, host), negative = fixture(t, std::move(negated), host),
                  zero = fixture(t, std::vector<hip_bfloat16>(t * K, hip_bfloat16(0.0F)), host);
    std::vector<Run> runs;
    unsigned malformed = 0;
    {
        const GpuLease lease;
        Guarded<hip_bfloat16> input(x.size(), s);
        input.put(x, s);
        Output candidate(t, s), control(t, s);
        std::array<std::unique_ptr<Weights>, Copies> weights;
        for (auto& w : weights) w = std::make_unique<Weights>(host, s);
        const auto valid = arguments(t, input, *weights[0], candidate);
        auto reject      = [&](auto a, hipStream_t stream) {
            if (owning_launch(a, stream) != hipErrorInvalidValue) fail("malformed accepted");
            ++malformed;
        };
        reject(valid, nullptr);
        auto bad   = valid;
        bad.tokens = 1;
        reject(bad, s);
        bad        = valid;
        bad.tokens = 9;
        reject(bad, s);
        bad      = valid;
        bad.rows = N - 16;
        reject(bad, s);
        bad = valid;
        bad.columns -= 64;
        reject(bad, s);
        bad = valid;
        bad.padded_columns -= 64;
        reject(bad, s);
        bad = valid;
        --bad.weight_code_bytes;
        reject(bad, s);
        bad = valid;
        --bad.weight_scale_bytes;
        reject(bad, s);
        bad = valid;
        --bad.activation_workspace_bytes;
        reject(bad, s);
        bad = valid;
        ++bad.activation_workspace_bytes;
        reject(bad, s);
        bad       = valid;
        bad.input = nullptr;
        reject(bad, s);
        bad              = valid;
        bad.weight_codes = nullptr;
        reject(bad, s);
        bad               = valid;
        bad.weight_scales = nullptr;
        reject(bad, s);
        bad        = valid;
        bad.output = nullptr;
        reject(bad, s);
        bad                      = valid;
        bad.activation_workspace = nullptr;
        reject(bad, s);
        bad = valid;
        ++bad.weight_codes;
        reject(bad, s);
        bad        = valid;
        bad.output = const_cast<hip_bfloat16*>(bad.input);
        reject(bad, s);
        bad        = valid;
        bad.output = reinterpret_cast<hip_bfloat16*>(bad.activation_workspace);
        reject(bad, s);
        bad               = valid;
        bad.weight_scales = reinterpret_cast<const std::uint16_t*>(bad.weight_codes);
        reject(bad, s);
        std::array<std::unique_ptr<Graph>, Copies> graphs;
        for (unsigned i = 0; i < Copies; ++i)
            graphs[i] = std::make_unique<Graph>(
                s, [&] { launch(arguments(t, input, *weights[i], candidate), s); });
        auto run_fixture = [&](const Fixture& f, bool eager) {
            Run run{&f, eager};
            // An unchanged generic route is supplementary regression evidence, not
            // the oracle. Check it against the same complete public-input formula.
            const auto control_args = arguments(t, input, *weights[0], control);
            const auto cw           = workspace(t, control);
            HIP_CHECK(linear::a8g64_quantize_activation({control_args.input, cw}, s));
            HIP_CHECK(linear::a8q4g64_linear_wmma32(prepared(t, control_args, cw), s));
            run.control = control.value.read(s);
            launch(arguments(t, input, *weights[0], candidate), s);
            const auto eager_reference = candidate.value.read(s);
            for (unsigned i = 0; i < Copies; ++i) {
                // Poison workspace before every launch, including restored/zero
                // fixtures, so graph replay cannot reuse a stale activation image.
                HIP_CHECK(
                    hipMemsetAsync(candidate.scratch.data(), 0xff, candidate.scratch.bytes(), s));
                if (eager)
                    launch(arguments(t, input, *weights[i], candidate), s);
                else
                    graphs[i]->run(s);
                codec(t, candidate, f.represented, s);
                run.actual[i] = candidate.value.read(s);
                candidate.value.guards(s);
                candidate.scratch.guards(s);
                exact(run.actual[i], eager_reference,
                      "public eager/graph/allocation BF16 mismatch");
                exact(input.read(s), f.x, "hidden mutation");
                input.guards(s);
                exact(weights[i]->codes.read(s), host.codes, "code mutation");
                exact(weights[i]->scales.read(s), host.scales, "scale mutation");
                weights[i]->codes.guards(s);
                weights[i]->scales.guards(s);
            }
            runs.push_back(std::move(run));
        };
        run_fixture(original, true);
        run_fixture(original, false);
        input.put(poisoned, s);
        graphs[0]->run(s);
        for (auto* o : {&candidate}) {
            const auto values = o->value.read(s);
            if (!std::all_of(values.begin(), values.end(), [](auto v) { return v.data == 0x7fc1; }))
                fail("poison output/status propagation");
            std::uint32_t status = 0;
            transfer(&status, workspace(t, *o).status, 4, hipMemcpyDeviceToHost, s);
            if (!status) fail("poison status absent");
        }
        input.put(negative.x, s);
        HIP_CHECK(hipMemsetAsync(candidate.scratch.data(), 0xff, candidate.scratch.bytes(), s));
        run_fixture(negative, false);
        input.put(zero.x, s);
        run_fixture(zero, false);
        input.put(original.x, s);
        run_fixture(original, false);
        power();
    }
    double max_l2 = 0, max_abs = 0, max_gross_fraction = 0;
    unsigned mutations_rejected = 0;
    PublicError public_error{};
    for (const auto& run : runs) {
        const auto& f = *run.fixture;
        compare_public(t, run.control, f.reference, f.budget);
        if (run.eager) {
            // A profile allowance must not make this a plausibility test.
            // Deliberately corrupt a real output at its largest-magnitude row.
            const auto largest = static_cast<std::size_t>(
                std::max_element(run.control.begin(), run.control.end(),
                                 [](auto a, auto b) {
                                     return std::abs(static_cast<float>(a)) <
                                            std::abs(static_cast<float>(b));
                                 }) -
                run.control.begin());
            for (float multiplier : {-1.0F, 0.0F, 1.125F}) {
                auto damaged     = run.control;
                damaged[largest] = hip_bfloat16(multiplier * static_cast<float>(damaged[largest]));
                bool rejected    = false;
                try {
                    (void)compare_public(t, damaged, f.reference, f.budget);
                } catch (const std::runtime_error&) { rejected = true; }
                if (!rejected) fail("A8 criterion accepted output corruption");
                ++mutations_rejected;
            }
        }
        for (const auto& actual : run.actual) {
            const auto e              = compare(actual, f.budget.represented);
            const auto pe             = compare_public(t, actual, f.reference, f.budget);
            public_error.relative_rms = std::max(public_error.relative_rms, pe.relative_rms);
            public_error.gross_rms    = std::max(public_error.gross_rms, pe.gross_rms);
            public_error.public_bound_fraction =
                std::max(public_error.public_bound_fraction, pe.public_bound_fraction);
            public_error.arithmetic_bound_fraction =
                std::max(public_error.arithmetic_bound_fraction, pe.arithmetic_bound_fraction);
            public_error.zero_reference_nonzero_tokens = std::max(
                public_error.zero_reference_nonzero_tokens, pe.zero_reference_nonzero_tokens);
            max_l2             = std::max(max_l2, e.relative_l2);
            max_abs            = std::max(max_abs, e.maximum_absolute);
            max_gross_fraction = std::max(max_gross_fraction,
                                          e.maximum_absolute / (0.01 * e.reference_maximum + 1e-5));
        }
    }
    const auto cases = static_cast<unsigned>(runs.size());
    out << "{\"tokens\":" << t << ",\"rows\":" << N << ",\"columns\":" << K << ",\"correctness\":{"
        << "\"maximum_relative_l2\":" << max_l2 << ",\"maximum_absolute\":" << max_abs
        << ",\"maximum_gross_cap_fraction\":" << max_gross_fraction
        << ",\"criterion\":\"fp64_rel_l2_1e-2_gross_1e-2_refmax_plus_1e-5\","
        << "\"exact_codec\":true,\"exact_eager_graph\":true,\"graph_poison_stale_finite\":true,"
        << "\"guards_immutability\":true,\"finite_cases\":" << cases
        << ",\"malformed_cases\":" << malformed
        << "},\"public_bf16_oracle\":{\"criterion\":\"a8g64_exact_int_fp32_fma_bf16_forward_bound_"
           "v1\","
        << "\"full_k\":" << K
        << ",\"all_tokens\":true,\"maximum_relative_rms\":" << public_error.relative_rms
        << ",\"maximum_error_over_reference_rms\":" << public_error.gross_rms
        << ",\"maximum_public_norm_bound_fraction\":" << public_error.public_bound_fraction
        << ",\"maximum_arithmetic_norm_bound_fraction\":" << public_error.arithmetic_bound_fraction
        << ",\"zero_reference_nonzero_tokens\":" << public_error.zero_reference_nonzero_tokens
        << ",\"output_corruptions_rejected\":" << mutations_rejected
        << ",\"historical_2pct_rms_10pct_gross_pass\":"
        << (public_error.relative_rms <= 0.02 && public_error.gross_rms <= 0.10 &&
                    public_error.zero_reference_nonzero_tokens == 0
                ? "true"
                : "false")
        << ",\"rows_checked\":" << original.reference.rows.size()
        << ",\"generic_control_oracle_checked\":true}";
    const auto route = linear::detail::select_a8q4_small_batch_wide_route(t, N, K, K);
    if (route != linear::detail::A8Q4WideRoute::None)
        out << ",\"wide_route\":\"" << wide_route_name(route) << '"';
    out << '}';
}

// In-place residual epilogue of a shared-weight N5120 cell through the public projected-residual
// Op. The product is checked against the FP64 oracle bounds; the accumulate contract
// BF16(residual + BF16(product)) is then checked exactly against that qualified product.
void accumulate_cell(unsigned t, hipStream_t s, std::ostream& out) {
    const auto host = make_decode_dot8_weights(N, K);
    const auto x    = activation(t);
    const Fixture f = fixture(t, x, host);
    std::vector<hip_bfloat16> residual(static_cast<std::size_t>(t) * N);
    for (std::size_t i = 0; i < residual.size(); ++i)
        residual[i] =
            hip_bfloat16(static_cast<float>(static_cast<int>((i * 37U) % 201U) - 100) * 0.0137F);
    std::vector<hip_bfloat16> product, accumulated;
    {
        const GpuLease lease;
        Guarded<hip_bfloat16> input(x.size(), s);
        input.put(x, s);
        Output plain(t, s), inplace(t, s);
        Weights weights(host, s);
        HIP_CHECK(hipMemsetAsync(plain.scratch.data(), 0xff, plain.scratch.bytes(), s));
        launch(arguments(t, input, weights, plain), s);
        product = plain.value.read(s);
        inplace.value.put(residual, s);
        HIP_CHECK(hipMemsetAsync(inplace.scratch.data(), 0xff, inplace.scratch.bytes(), s));
        if (!linear::a8q4g64_projected_residual_supported(t, N, K))
            fail("accumulate cell unsupported");
        HIP_CHECK(linear::a8q4g64_projected_residual(arguments(t, input, weights, inplace), s));
        codec(t, inplace, f.represented, s);
        accumulated = inplace.value.read(s);
        inplace.value.guards(s);
        inplace.scratch.guards(s);
        plain.value.guards(s);
        exact(input.read(s), f.x, "hidden mutation");
        exact(weights.codes.read(s), host.codes, "code mutation");
        exact(weights.scales.read(s), host.scales, "scale mutation");
    }
    const auto pe = compare_public(t, product, f.reference, f.budget);
    for (std::size_t i = 0; i < accumulated.size(); ++i) {
        const hip_bfloat16 expected(static_cast<float>(residual[i]) +
                                    static_cast<float>(product[i]));
        if (accumulated[i].data != expected.data)
            fail("accumulate epilogue mismatch: N=" + std::to_string(N) + " K=" +
                 std::to_string(K) + " T=" + std::to_string(t) + " index=" + std::to_string(i));
    }
    out << "{\"tokens\":" << t << ",\"rows\":" << N << ",\"columns\":" << K
        << ",\"accumulate\":true,\"exact_bf16_residual_plus_bf16_product\":true"
        << ",\"maximum_public_norm_bound_fraction\":" << pe.public_bound_fraction
        << ",\"maximum_arithmetic_norm_bound_fraction\":" << pe.arithmetic_bound_fraction << '}';
}

// Unprofiled event A/B of one oracle-qualified wide cell on one prepared activation: the route
// these widths took before the wide table (prefill CTA above T32 on its qualified tuples, else
// WMMA32) and the selected wide route. Launches rotate over disjoint weight copies totalling at
// least 256 MiB, so each reads its weights from DRAM rather than the 64 MiB last-level cache, as
// one decode round's distinct layers do.
void timing_cell(unsigned t, hipStream_t s, std::ostream& out) {
    constexpr unsigned Trials      = 7;
    const auto host                = make_decode_dot8_weights(N, K);
    const auto x                   = activation(t);
    const std::size_t weight_bytes = host.codes.size() + host.scales.size() * sizeof(std::uint16_t);
    const unsigned copies          = static_cast<unsigned>(
        std::max<std::size_t>(2, ((std::size_t{256} << 20) + weight_bytes - 1) / weight_bytes));
    const unsigned iterations = std::max(32U, copies);
    const bool prefill_cta    = linear::use_a8q4_prefill_cta(t, N, K);
    constexpr unsigned routes = 2;
    std::array<std::array<double, Trials>, routes> samples{};
    {
        const GpuLease lease;
        Guarded<hip_bfloat16> input(x.size(), s);
        input.put(x, s);
        Output output(t, s);
        std::vector<std::unique_ptr<Weights>> weights;
        for (unsigned i = 0; i < copies; ++i) weights.push_back(std::make_unique<Weights>(host, s));
        const auto w = workspace(t, output);
        HIP_CHECK(linear::a8g64_quantize_activation({input.data(), w}, s));
        const auto launch_route = [&](unsigned route, unsigned copy) {
            const auto a = prepared(t, arguments(t, input, *weights[copy], output), w);
            if (route == 0)
                return prefill_cta ? linear::a8q4g64_linear_prefill_cta(a, s)
                                   : linear::a8q4g64_linear_wmma32(a, s);
            return linear::detail::launch_a8q4_small_batch_projection(a, s);
        };
        Event begin, end;
        const auto measure = [&](unsigned route) {
            for (unsigned i = 0; i < 2; ++i) HIP_CHECK(launch_route(route, i % copies));
            HIP_CHECK(hipEventRecord(begin.get(), s));
            for (unsigned i = 0; i < iterations; ++i) HIP_CHECK(launch_route(route, i % copies));
            HIP_CHECK(hipEventRecord(end.get(), s));
            HIP_CHECK(hipEventSynchronize(end.get()));
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, begin.get(), end.get()));
            return 1000.0 * ms / iterations;
        };
        power();
        for (unsigned trial = 0; trial < Trials; ++trial)
            for (unsigned i = 0; i < routes; ++i) {
                const unsigned route  = trial % 2 == 0 ? i : routes - 1 - i;
                samples[route][trial] = measure(route);
            }
        power();
    }
    const auto median = [](std::array<double, Trials> v) {
        std::sort(v.begin(), v.end());
        return v[Trials / 2];
    };
    const double incumbent = median(samples[0]), selected = median(samples[1]);
    const auto route = linear::detail::select_a8q4_small_batch_wide_route(t, N, K, K);
    const auto emit  = [&](const char* name, const std::array<double, Trials>& v) {
        out << ",\"" << name << "\":[";
        for (unsigned i = 0; i < Trials; ++i) out << (i ? "," : "") << v[i];
        out << ']';
    };
    out << "{\"tokens\":" << t << ",\"rows\":" << N << ",\"columns\":" << K
        << ",\"incumbent_route\":\"" << (prefill_cta ? "prefill_cta" : "wmma32") << '"'
        << ",\"selected_route\":\"" << wide_route_name(route) << '"'
        << ",\"weight_copies\":" << copies << ",\"launches_per_trial\":" << iterations
        << ",\"incumbent_median_us\":" << incumbent << ",\"selected_median_us\":" << selected
        << ",\"selected_over_incumbent\":" << selected / incumbent
        << ",\"selected_dram_gbps\":" << static_cast<double>(weight_bytes) / (selected * 1e3);
    emit("incumbent_us", samples[0]);
    emit("selected_us", samples[1]);
    out << '}';
}
} // namespace

// The paired small-batch launch (two projections of one prepared activation) through the public
// shared-activation Op, each output checked against the complete FP64 public oracle bound.
void pair_cell(unsigned t, unsigned n0, unsigned n1, hipStream_t s, std::ostream& out) {
    K                      = 5120;
    G                      = K / 64;
    const auto x           = activation(t);
    const auto represented = quantize_host(x, t, K);
    const std::array<unsigned, 2> rows{n0, n1};
    std::array<DecodeDot8Weights, 2> host{make_decode_dot8_weights(n0, K),
                                          make_decode_dot8_weights(n1, K)};
    std::array<A8ErrorBudget, 2> budgets;
    std::array<PublicReference, 2> references;
    for (unsigned i = 0; i < 2; ++i) {
        N             = rows[i];
        budgets[i]    = a8_error_budget(t, represented, host[i]);
        references[i] = public_oracle(t, x, host[i]);
    }
    std::array<std::vector<hip_bfloat16>, 2> actual;
    {
        const GpuLease lease;
        Guarded<hip_bfloat16> input(x.size(), s);
        input.put(x, s);
        Guarded<std::uint8_t> scratch(linear::a8q4g64_activation_workspace_capacity_bytes(t, K), s);
        std::array<std::unique_ptr<Weights>, 2> weights{std::make_unique<Weights>(host[0], s),
                                                        std::make_unique<Weights>(host[1], s)};
        std::array<std::unique_ptr<Guarded<hip_bfloat16>>, 2> outputs{
            std::make_unique<Guarded<hip_bfloat16>>(t * n0, s),
            std::make_unique<Guarded<hip_bfloat16>>(t * n1, s)};
        linear::A8Q4G64SharedActivationArgs args{};
        args.input                      = input.data();
        args.activation_workspace       = scratch.data();
        args.activation_workspace_bytes = scratch.bytes();
        args.projection_count           = 2;
        args.tokens                     = t;
        args.columns                    = K;
        for (unsigned i = 0; i < 2; ++i)
            args.projections[i] = {weights[i]->codes.data(),  weights[i]->codes.bytes(),
                                   weights[i]->scales.data(), weights[i]->scales.bytes(),
                                   outputs[i]->data(),        rows[i]};
        HIP_CHECK(hipMemsetAsync(scratch.data(), 0xff, scratch.bytes(), s)); // stale status/planes
        HIP_CHECK(linear::a8q4g64_shared_activation_linear(args, s));
        HIP_CHECK(hipStreamSynchronize(s));
        for (unsigned i = 0; i < 2; ++i) {
            actual[i] = outputs[i]->read(s);
            outputs[i]->guards(s);
        }
        scratch.guards(s);
        input.guards(s);
    }
    double worst_public = 0, worst_arithmetic = 0;
    for (unsigned i = 0; i < 2; ++i) {
        N                = rows[i];
        const auto error = compare_public(t, actual[i], references[i], budgets[i]);
        worst_public     = std::max(worst_public, error.public_bound_fraction);
        worst_arithmetic = std::max(worst_arithmetic, error.arithmetic_bound_fraction);
    }
    out << "{\"tokens\":" << t << ",\"rows\":[" << n0 << ',' << n1 << "],\"columns\":" << K
        << ",\"maximum_public_norm_bound_fraction\":" << worst_public
        << ",\"maximum_arithmetic_norm_bound_fraction\":" << worst_arithmetic << '}';
}
#ifndef NINFER_A8Q4_VERIFY_QUAL_NO_MAIN
int main(int argc, char** argv) {
    try {
        bool mlp_only = false, output_only = false, draft_only = false, projection_only = false,
             concurrent_only = false, wide_only = false, wide_ab = false;
        std::vector<std::string_view> args(argv + 1, argv + argc);
        if (args.size() >= 2 && args[args.size() - 2] == "--gpu-lock") {
            gpu_lock_path = args.back();
            args.resize(args.size() - 2);
        }
        const bool scoped = args.size() == 3;
        mlp_only          = scoped && args[2] == "--mlp-only";
        output_only       = scoped && args[2] == "--output-only";
        draft_only        = scoped && args[2] == "--draft-only";
        projection_only   = scoped && args[2] == "--projection-only";
        concurrent_only   = scoped && args[2] == "--concurrent-only";
        wide_only         = scoped && args[2] == "--wide-only";
        wide_ab           = scoped && args[2] == "--wide-ab";
        if ((args.size() != 2 && !mlp_only && !output_only && !draft_only && !projection_only &&
             !concurrent_only && !wide_only && !wide_ab) ||
            args[0] != "--out-json")
            fail("usage: selected_q4_qual --out-json FRESH.json "
                 "[--mlp-only|--output-only|--draft-only|--projection-only|--concurrent-only|--"
                 "wide-only|--wide-ab] [--gpu-lock PATH]");
        // --wide-only/--wide-ab: the wide-route cells only; --wide-ab times each after its oracle
        // passes.
        wide_only                          = wide_only || wide_ab;
        const std::filesystem::path output = std::string(args[1]);
        require_fresh_output(output);
        power();
        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, 0));
        char pci[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
        if (std::string_view(props.name) != "AMD Radeon AI PRO R9700" ||
            std::string_view(props.gcnArchName) != "gfx1201" || props.warpSize != 32 ||
            (std::string_view(pci) != "0000:13:00.0" && std::string_view(pci) != "13:00.0"))
            fail("wrong device");
        hipStream_t stream{};
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        std::ostringstream out;
        out << std::setprecision(17) << "{\"schema\":\""
            << "ninfer.r9700.a8q4-small-batch-projections.v3"
            << "\",\"status\":\"qualified\",\"public_dispatch_tested\":true,"
               "\"pci\":\"0000:13:00.0\",\"power\":\"auto\",\"copies\":3,\"scope\":\""
            << (mlp_only          ? "mlp_only"
                : output_only     ? "output_only"
                : draft_only      ? "draft_only"
                : projection_only ? "projection_only"
                : concurrent_only ? "concurrent_only"
                : wide_ab         ? "wide_ab"
                : wide_only       ? "wide_only"
                                  : "complete_owner")
            << "\",\"cells\":[";
        constexpr std::array<std::array<unsigned, 2>, 11> shapes{{{34816, 5120},
                                                                  {5120, 6144},
                                                                  {12288, 5120},
                                                                  {4096, 5120},
                                                                  {5120, 17408},
                                                                  {5120, 25600},
                                                                  {7168, 5120},
                                                                  {6144, 5120},
                                                                  {1280, 5120},
                                                                  {5120, 4096},
                                                                  {131072, 5120}}};
        std::vector<std::array<unsigned, 3>> timed;
        bool first = true;
        for (const auto& shape : shapes) {
            if (concurrent_only && (shape == std::array<unsigned, 2>{34816, 5120} ||
                                    shape == std::array<unsigned, 2>{5120, 17408}))
                continue;
            if (mlp_only && shape != std::array<unsigned, 2>{34816, 5120} &&
                shape != std::array<unsigned, 2>{5120, 17408})
                continue;
            if (output_only && shape != std::array<unsigned, 2>{5120, 6144}) continue;
            if (draft_only && shape != std::array<unsigned, 2>{5120, 17408} &&
                shape != std::array<unsigned, 2>{5120, 25600})
                continue;
            if (projection_only && shape != std::array<unsigned, 2>{7168, 5120} &&
                shape != std::array<unsigned, 2>{6144, 5120} &&
                shape != std::array<unsigned, 2>{1280, 5120} &&
                shape != std::array<unsigned, 2>{5120, 4096})
                continue;
            N = shape[0];
            K = shape[1];
            G = K / 64;
            // T17/33/63 are off-table widths at the token-tile class bounds (generic in T).
            for (unsigned t :
                 {1U,  2U,  3U,  4U,  5U,  6U,  7U,  8U,  10U, 12U, 14U, 15U, 16U, 17U, 18U, 20U,
                  21U, 24U, 25U, 28U, 30U, 32U, 33U, 35U, 36U, 40U, 42U, 48U, 49U, 56U, 63U, 64U}) {
                const bool wide = linear::detail::use_a8q4_small_batch_wide(t, N, K, K);
                if (concurrent_only && (t < 10 || wide)) continue;
                if (draft_only && (t < 5 || t > 8)) continue;
                if (wide_only && !wide) continue;
                if (!linear::detail::use_a8q4_small_batch_projection(t, N, K, K)) continue;
                if (!first) out << ',';
                first = false;
                cell(t, stream, out);
                if (wide_ab) timed.push_back({N, K, t});
            }
        }
        out << "],\"accumulate\":[";
        if (!mlp_only && !output_only && !draft_only && !projection_only && !concurrent_only) {
            bool first_accumulate = true;
            for (const unsigned columns : {4096U, 17408U, 25600U})
                for (const unsigned t : {17U, 25U, 30U, 33U, 35U, 48U, 49U, 56U, 63U, 64U}) {
                    N = 5120;
                    K = columns;
                    G = K / 64;
                    // The wide routes' in-place epilogues (token tile on every N5120 shape).
                    if (!linear::detail::use_a8q4_small_batch_wide(t, N, K, K)) continue;
                    if (!first_accumulate) out << ',';
                    first_accumulate = false;
                    accumulate_cell(t, stream, out);
                }
        }
        out << "],\"timing\":[";
        for (std::size_t i = 0; i < timed.size(); ++i) {
            N = timed[i][0];
            K = timed[i][1];
            G = K / 64;
            if (i) out << ',';
            timing_cell(timed[i][2], stream, out);
        }
        out << "],\"pairs\":[";
        if (!mlp_only && !output_only && !draft_only && !wide_only) {
            bool first_pair = true;
            for (unsigned t : {5U, 6U, 7U, 8U, 12U, 28U, 32U}) {
                for (const auto& pair :
                     {std::array<unsigned, 2>{4096, 12288}, std::array<unsigned, 2>{7168, 7168}}) {
                    if (!first_pair) out << ',';
                    first_pair = false;
                    pair_cell(t, pair[0], pair[1], stream, out);
                }
            }
        }
        out << "]}\n";
        HIP_CHECK(hipStreamDestroy(stream));
        power();
        const int fd = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) fail("cannot create immutable result");
        const auto result = out.str();
        const auto count  = ::write(fd, result.data(), result.size());
        ::close(fd);
        if (count != static_cast<ssize_t>(result.size())) fail("short result write");
        std::cout << "selected public Q4 routes qualified\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
#endif
