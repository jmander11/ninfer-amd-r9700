// Qualifies the persistent decode lowering (ops::persistent_decode_lower) on physical gfx1201:
// each scenario captures a single-stream chain of production launches (the public Op launchers at
// Qwen3.8-27B decode shapes, verification widths T 4..8, with their production cache-warm ranges
// and, for the GDN front, an active deferred replay fold), lowers a clone of that graph, and runs
// both graphs from identical device contents. The lowered graph must write exactly the bytes of
// the captured launches in every buffer (bitwise, inputs included) and be run-to-run
// deterministic; constant operands must stay untouched. Operands are small finite pseudo-random
// values (valid FP8LUT4 codes, group codes and scales), not production weights.
#include "core/arena.h"
#include "core/cache_warm.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/persistent_decode.h"
#include "ninfer/ops/rope.h"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/fp8_small_t_linear.h"
#include "ops/r9700/linear/fp8lut4_linear.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace lin = ninfer::ops::r9700::linear;
namespace gdn = ninfer::ops::r9700::gdn;
using ninfer::CacheWarm;
using ninfer::DType;
using ninfer::Tensor;

// Qwen3.8-27B Text decode geometry.
constexpr unsigned kHidden        = 5120;
constexpr unsigned kQuery         = 6144; // 24 heads x 256
constexpr unsigned kKv            = 1024; // 4 heads x 256
constexpr unsigned kAttentionRows = kQuery + kKv;
constexpr unsigned kIntermediate  = 17408;
constexpr unsigned kGdnQueryKey   = 4096;  // q 2048 | k 2048
constexpr unsigned kGdnValueZ     = 12288; // v 6144 | z 6144
constexpr unsigned kGdnValue      = 6144;
constexpr unsigned kGdnChannels   = 10240;
constexpr unsigned kGdnHeads      = 48;
constexpr unsigned kStateDim      = 128;
constexpr std::size_t kMiB        = std::size_t{1} << 20U;

void check(hipError_t error, const char* what) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Rng {
    std::uint64_t state;

    explicit Rng(std::uint64_t seed) : state(seed * 0x9e3779b97f4a7c15ULL + 0x6d2b79f5ULL) {}

    std::uint64_t next() {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        return state;
    }

    double uniform() { return static_cast<double>(next() >> 11U) / 9007199254740992.0; }

    float range(double lo, double hi) { return static_cast<float>(lo + (hi - lo) * uniform()); }
};

std::vector<hip_bfloat16> bf16_values(std::size_t count, double lo, double hi, Rng& rng) {
    std::vector<hip_bfloat16> out(count);
    for (auto& v : out) v = hip_bfloat16(rng.range(lo, hi));
    return out;
}

std::vector<float> f32_values(std::size_t count, double lo, double hi, Rng& rng) {
    std::vector<float> out(count);
    for (auto& v : out) v = rng.range(lo, hi);
    return out;
}

std::vector<std::uint8_t> random_bytes(std::size_t count, Rng& rng) {
    std::vector<std::uint8_t> out(count);
    std::size_t i = 0;
    for (; i + 8U <= count; i += 8U) {
        const std::uint64_t word = rng.next();
        std::memcpy(out.data() + i, &word, 8U);
    }
    for (; i < count; ++i) out[i] = static_cast<std::uint8_t>(rng.next());
    return out;
}

// ---- Scenario buffers. Const: uploaded once, must stay untouched. Data/Output: restored from a
// device-resident pristine copy (stream-ordered) before every run and compared after it; an
// Output must also differ from its initial contents after the reference run.
enum class Role { Const, Data, Output };

struct Buffer {
    std::string name;
    Role role = Role::Const;
    std::vector<std::uint8_t> host;
    void* device   = nullptr;
    void* pristine = nullptr;
};

using Snapshot = std::vector<std::vector<std::uint8_t>>; // per buffer (empty for Const)

class Scenario {
public:
    explicit Scenario(std::string name) : name_(std::move(name)) {}

    ~Scenario() {
        for (auto& buffer : buffers_) {
            if (buffer->device != nullptr) (void)hipFree(buffer->device);
            if (buffer->pristine != nullptr) (void)hipFree(buffer->pristine);
        }
    }

    Scenario(const Scenario&)            = delete;
    Scenario& operator=(const Scenario&) = delete;

    const std::string& name() const { return name_; }

    Buffer& add_buffer(const std::string& name, Role role, std::vector<std::uint8_t> host) {
        require(!host.empty(), name + ": empty buffer");
        auto buffer  = std::make_unique<Buffer>();
        buffer->name = name;
        buffer->role = role;
        buffer->host = std::move(host);
        check(hipMalloc(&buffer->device, buffer->host.size()), "hipMalloc");
        buffers_.push_back(std::move(buffer));
        return *buffers_.back();
    }

    template <class T>
    T* add(const std::string& name, Role role, const std::vector<T>& values) {
        std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return static_cast<T*>(add_buffer(name, role, std::move(bytes)).device);
    }

    // An output initialized with a fixed finite byte pattern (BF16/FP32 ~0.0115).
    template <class T>
    T* output(const std::string& name, std::size_t count) {
        return static_cast<T*>(
            add_buffer(name, Role::Output, std::vector<std::uint8_t>(count * sizeof(T), 0x3CU))
                .device);
    }

    void upload() {
        for (auto& buffer : buffers_) {
            check(hipMemcpy(buffer->device, buffer->host.data(), buffer->host.size(),
                            hipMemcpyHostToDevice),
                  "upload");
            if (buffer->role != Role::Const) {
                check(hipMalloc(&buffer->pristine, buffer->host.size()), "hipMalloc pristine");
                check(hipMemcpy(buffer->pristine, buffer->host.data(), buffer->host.size(),
                                hipMemcpyHostToDevice),
                      "upload pristine");
            }
        }
        check(hipDeviceSynchronize(), "upload sync");
    }

    void restore(hipStream_t stream) const {
        for (const auto& buffer : buffers_)
            if (buffer->role != Role::Const)
                check(hipMemcpyAsync(buffer->device, buffer->pristine, buffer->host.size(),
                                     hipMemcpyDeviceToDevice, stream),
                      "restore");
    }

    Snapshot snapshot(hipStream_t stream) const {
        Snapshot out(buffers_.size());
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            if (buffers_[i]->role == Role::Const) continue;
            out[i].resize(buffers_[i]->host.size());
            check(hipMemcpyAsync(out[i].data(), buffers_[i]->device, out[i].size(),
                                 hipMemcpyDeviceToHost, stream),
                  "snapshot");
        }
        check(hipStreamSynchronize(stream), "snapshot sync");
        return out;
    }

    // Every Const buffer still holds its uploaded bytes.
    std::string constants_unchanged() const {
        std::string failures;
        for (const auto& buffer : buffers_) {
            if (buffer->role != Role::Const) continue;
            std::vector<std::uint8_t> now(buffer->host.size());
            check(hipMemcpy(now.data(), buffer->device, now.size(), hipMemcpyDeviceToHost),
                  "const");
            if (now != buffer->host) failures += " const '" + buffer->name + "' modified;";
        }
        return failures;
    }

    // Outputs the reference run left at their initial bytes (a launch that did not run).
    std::string unwritten_outputs(const Snapshot& snapshot) const {
        std::string failures;
        for (std::size_t i = 0; i < buffers_.size(); ++i)
            if (buffers_[i]->role == Role::Output && snapshot[i] == buffers_[i]->host)
                failures += " output '" + buffers_[i]->name + "' never written;";
        return failures;
    }

    // First difference of every differing buffer ("" when bitwise equal).
    std::string differences(const Snapshot& a, const Snapshot& b, const char* label) const {
        std::ostringstream out;
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            if (buffers_[i]->role == Role::Const || a[i] == b[i]) continue;
            std::size_t first = a[i].size(), count = 0;
            for (std::size_t j = 0; j < a[i].size(); ++j) {
                if (a[i][j] == b[i][j]) continue;
                if (first == a[i].size()) first = j;
                ++count;
            }
            out << ' ' << label << ": '" << buffers_[i]->name << "' first differing byte " << first
                << " (0x" << std::hex << static_cast<unsigned>(a[i][first]) << " vs 0x"
                << static_cast<unsigned>(b[i][first]) << std::dec << "), " << count << " of "
                << a[i].size() << " bytes differ;";
        }
        return out.str();
    }

    std::size_t compared_bytes() const {
        std::size_t total = 0;
        for (const auto& buffer : buffers_)
            if (buffer->role != Role::Const) total += buffer->host.size();
        return total;
    }

    std::size_t compared_buffers() const {
        return static_cast<std::size_t>(
            std::count_if(buffers_.begin(), buffers_.end(),
                          [](const auto& b) { return b->role != Role::Const; }));
    }

private:
    std::string name_;
    std::vector<std::unique_ptr<Buffer>> buffers_;
};

// ---- Operands.
struct Lut4 {
    lin::Fp8Lut4Weight view{};
    std::size_t code_bytes  = 0;
    std::size_t group_bytes = 0;
};

// FP8LUT4 [rows, columns] in N16 x K64 tiles: random sign-magnitude nibbles, group codes with
// E = (b >> 3) - 26 in [-6, 3], row multipliers 2^-10 (1 + u).
Lut4 add_lut4(Scenario& s, const std::string& name, unsigned rows, unsigned columns, Rng& rng) {
    Lut4 w;
    w.code_bytes  = static_cast<std::size_t>(rows) * columns / 2U;
    w.group_bytes = static_cast<std::size_t>(rows) * columns / 32U;
    std::vector<std::uint8_t> groups(w.group_bytes);
    for (auto& g : groups) {
        const std::uint64_t r = rng.next();
        g                     = static_cast<std::uint8_t>((20U + (r >> 8U) % 10U) << 3U | (r & 7U));
    }
    std::vector<float> scales(rows);
    for (auto& v : scales) v = static_cast<float>(std::ldexp(1.0 + rng.uniform(), -10));
    const auto* codes =
        s.add<std::uint8_t>(name + ".codes", Role::Const, random_bytes(w.code_bytes, rng));
    const auto* group_codes = s.add<std::uint8_t>(name + ".groups", Role::Const, groups);
    const auto* row_scales  = s.add<float>(name + ".scales", Role::Const, scales);
    w.view                  = {codes, group_codes, row_scales, rows, columns};
    return w;
}

// The head of a FP8LUT4 weight's stream (core/cache_warm.h weight_warm): codes
// [offset, offset + bytes) and the same fraction of the group-code plane.
CacheWarm lut4_warm(const Lut4& w, std::size_t offset, std::size_t bytes) {
    CacheWarm warm{};
    require(offset < w.code_bytes, "warm offset");
    bytes                         = std::min(bytes, w.code_bytes - offset);
    warm.ranges[0]                = {w.view.codes + offset, bytes};
    const std::size_t high_offset = offset * w.group_bytes / w.code_bytes / 4U * 4U;
    warm.ranges[1] = {w.view.groups + high_offset,
                      std::min(bytes * w.group_bytes / w.code_bytes, w.group_bytes - high_offset)};
    return warm;
}

struct Fp8Rows {
    lin::Fp8RowScaledWeight view{};
    std::size_t code_bytes = 0;
};

// Row-scaled E4M3 [rows, columns]: finite codes (exponent field 1..9, never NaN), row scales
// 2^-8 (1 + u).
Fp8Rows add_fp8_rows(Scenario& s, const std::string& name, unsigned rows, unsigned columns,
                     Rng& rng) {
    Fp8Rows w;
    w.code_bytes = static_cast<std::size_t>(rows) * columns;
    std::vector<std::uint8_t> codes(w.code_bytes);
    for (auto& c : codes) {
        const std::uint64_t r = rng.next();
        c = static_cast<std::uint8_t>((r & 0x80U) | ((1U + (r >> 8U) % 9U) << 3U) |
                                      ((r >> 4U) & 7U));
    }
    std::vector<float> scales(rows);
    for (auto& v : scales) v = static_cast<float>(std::ldexp(1.0 + rng.uniform(), -8));
    w.view = {s.add<std::uint8_t>(name + ".codes", Role::Const, codes),
              s.add<float>(name + ".scales", Role::Const, scales), rows};
    return w;
}

CacheWarm fp8_warm(const Fp8Rows& w) {
    CacheWarm warm{};
    warm.ranges[0] = {w.view.codes, std::min(2U * kMiB, w.code_bytes)};
    return warm;
}

lin::Fp8ActivationWorkspace add_image(Scenario& s, const std::string& name, unsigned tokens,
                                      unsigned columns) {
    const std::size_t bytes = lin::fp8_activation_workspace_capacity_bytes(tokens, columns);
    require(bytes != 0U, name + ": image geometry");
    Buffer& buffer = s.add_buffer(name, Role::Output, std::vector<std::uint8_t>(bytes, 0x3CU));
    lin::Fp8ActivationWorkspace image{};
    check(lin::fp8_bind_activation_workspace(buffer.device, bytes, tokens, columns, &image),
          "bind image");
    return image;
}

// ---- Graph execution.
struct Outcome {
    std::string name;
    std::size_t captured = 0;
    std::size_t lowered  = 0;
    std::size_t buffers  = 0;
    std::size_t bytes    = 0;
    double seconds       = 0.0;
    bool pass            = false;
    std::string detail;
};

std::size_t node_count(hipGraph_t graph) {
    std::size_t count = 0;
    check(hipGraphGetNodes(graph, nullptr, &count), "hipGraphGetNodes");
    return count;
}

struct GraphSet {
    hipGraph_t captured = nullptr, lowered = nullptr;
    hipGraphExec_t captured_exec = nullptr, lowered_exec = nullptr;
    ninfer::DeviceBuffer storage; // must outlive the lowered graph and executable

    ~GraphSet() {
        if (lowered_exec != nullptr) (void)hipGraphExecDestroy(lowered_exec);
        if (captured_exec != nullptr) (void)hipGraphExecDestroy(captured_exec);
        if (lowered != nullptr) (void)hipGraphDestroy(lowered);
        if (captured != nullptr) (void)hipGraphDestroy(captured);
    }
};

Outcome qualify(Scenario& s, std::size_t launches, std::size_t expected_lowered,
                const std::function<void(hipStream_t)>& body, hipStream_t stream) {
    const auto start = std::chrono::steady_clock::now();
    Outcome outcome;
    outcome.name = s.name();
    try {
        s.upload();
        GraphSet graphs;
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin capture");
        try {
            body(stream);
        } catch (...) {
            hipGraph_t discarded = nullptr;
            (void)hipStreamEndCapture(stream, &discarded);
            if (discarded != nullptr) (void)hipGraphDestroy(discarded);
            throw;
        }
        check(hipStreamEndCapture(stream, &graphs.captured), "end capture");
        outcome.captured = node_count(graphs.captured);
        require(outcome.captured == launches, "captured " + std::to_string(outcome.captured) +
                                                  " nodes, expected " + std::to_string(launches) +
                                                  " launches");
        check(hipGraphClone(&graphs.lowered, graphs.captured), "hipGraphClone");
        graphs.storage  = ninfer::ops::persistent_decode_lower(graphs.lowered);
        outcome.lowered = node_count(graphs.lowered);
        require(outcome.lowered < outcome.captured && outcome.lowered == expected_lowered,
                "lowered to " + std::to_string(outcome.lowered) + " nodes, expected " +
                    std::to_string(expected_lowered));
        check(hipGraphInstantiate(&graphs.captured_exec, graphs.captured, nullptr, nullptr, 0),
              "instantiate captured");
        check(hipGraphInstantiate(&graphs.lowered_exec, graphs.lowered, nullptr, nullptr, 0),
              "instantiate lowered");
        const auto run = [&](hipGraphExec_t exec) {
            s.restore(stream);
            check(hipGraphLaunch(exec, stream), "hipGraphLaunch");
            check(hipStreamSynchronize(stream), "graph sync");
            return s.snapshot(stream);
        };
        const Snapshot reference       = run(graphs.captured_exec);
        const Snapshot reference_again = run(graphs.captured_exec);
        const Snapshot lowered         = run(graphs.lowered_exec);
        const Snapshot lowered_again   = run(graphs.lowered_exec);
        outcome.detail  = s.unwritten_outputs(reference) +
                          s.differences(reference, reference_again, "captured run-to-run") +
                          s.differences(reference, lowered, "captured vs lowered") +
                          s.differences(lowered, lowered_again, "lowered run-to-run") +
                          s.constants_unchanged();
        outcome.pass    = outcome.detail.empty();
        outcome.buffers = s.compared_buffers();
        outcome.bytes   = s.compared_bytes();
    } catch (const std::exception& error) {
        outcome.pass   = false;
        outcome.detail = std::string(" exception: ") + error.what();
    }
    outcome.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << (outcome.pass ? "PASS " : "FAIL ") << outcome.name << ": " << outcome.captured
              << " launches -> " << outcome.lowered << " node(s), " << outcome.buffers
              << " buffers (" << outcome.bytes / 1024U << " KiB) compared bitwise x3, "
              << outcome.seconds << " s" << (outcome.pass ? "" : "\n  ") << outcome.detail << '\n';
    return outcome;
}

// ---- Scenarios.

// Text attention front: fused RMSNorm quantization -> row-scaled FP8 Q/K and gate/V projections
// with the feature split -> Q/K norm + 1-D RoPE (QuantizeNormalized, Fp8SmallT<PairSplit>,
// QkNormRope<Text1D>).
Outcome attention_front(unsigned T, bool poison, hipStream_t stream) {
    Scenario s("attention front T" + std::to_string(T) + (poison ? " poisoned" : ""));
    Rng rng(100U + T + (poison ? 50U : 0U));
    auto residual_host = bf16_values(std::size_t(T) * kHidden, -2.0, 2.0, rng);
    if (poison) residual_host[std::size_t(T / 2U) * kHidden + 77U] = hip_bfloat16(NAN);
    auto* residual = s.add("residual", Role::Data, residual_host);
    auto* norm     = s.add("input_norm", Role::Const, bf16_values(kHidden, -0.2, 0.2, rng));
    const Fp8Rows query_key  = add_fp8_rows(s, "query_key", kAttentionRows, kHidden, rng);
    const Fp8Rows gate_value = add_fp8_rows(s, "gate_value", kAttentionRows, kHidden, rng);
    const auto image         = add_image(s, "image", T, kHidden);
    auto* query              = s.output<hip_bfloat16>("query", std::size_t(T) * kQuery);
    auto* key                = s.output<hip_bfloat16>("key", std::size_t(T) * kKv);
    auto* gate               = s.output<hip_bfloat16>("gate", std::size_t(T) * kQuery);
    auto* value              = s.output<hip_bfloat16>("value", std::size_t(T) * kKv);
    auto* q_norm             = s.add("q_norm", Role::Const, bf16_values(256, -0.3, 0.3, rng));
    auto* k_norm             = s.add("k_norm", Role::Const, bf16_values(256, -0.3, 0.3, rng));
    std::vector<std::int32_t> positions(T);
    for (unsigned t = 0; t < T; ++t) positions[t] = static_cast<std::int32_t>(4000U + 3U * t);
    auto* pos        = s.add("positions", Role::Const, positions);
    auto* qn         = s.output<hip_bfloat16>("qn", std::size_t(T) * kQuery);
    auto* kn         = s.output<hip_bfloat16>("kn", std::size_t(T) * kKv);
    const int tokens = static_cast<int>(T);
    return qualify(
        s, 3, 1,
        [&](hipStream_t st) {
            check(lin::fp8_quantize_normalized_activation({.input       = residual,
                                                           .weight      = norm,
                                                           .eps         = 1e-6F,
                                                           .unit_offset = true,
                                                           .workspace   = image,
                                                           .normalized  = nullptr,
                                                           .warm        = fp8_warm(query_key)},
                                                          st),
                  "quantize normalized");
            check(lin::fp8_small_t_pair_split({.first           = query_key.view,
                                               .second          = gate_value.view,
                                               .split           = kQuery,
                                               .first_leading   = query,
                                               .first_trailing  = key,
                                               .second_leading  = gate,
                                               .second_trailing = value},
                                              image, st),
                  "fp8 pair split");
            const Tensor positions_t(pos, DType::I32, {tokens});
            const Tensor q(query, DType::BF16, {256, 24, tokens}),
                k(key, DType::BF16, {256, 4, tokens});
            const Tensor qw(q_norm, DType::BF16, {256}), kw(k_norm, DType::BF16, {256});
            Tensor qn_t(qn, DType::BF16, {256, 24, tokens}),
                kn_t(kn, DType::BF16, {256, 4, tokens});
            ninfer::ops::qk_norm_rope(positions_t, 64, 1.0e7F, q, k, qw, kw, 1e-6F, qn_t, kn_t, st);
        },
        stream);
}

// Attention output and MLP: gated quantization -> row-scaled FP8 output projection into the
// residual -> fused RMSNorm quantization (publishing the normalized rows) -> FP8LUT4 SiLU-pair
// gate/up -> quantization of the MLP activation -> FP8LUT4 down projection accumulated into the
// residual (QuantizeGated, Fp8SmallT<Residual>, QuantizeNormalized, Fp8Lut4SmallT<false,true>,
// QuantizeActivation, Fp8Lut4SmallT<true,false>).
Outcome attention_output_mlp(unsigned T, hipStream_t stream) {
    Scenario s("attention output + MLP T" + std::to_string(T));
    Rng rng(200U + T);
    auto* gate = s.add("gate", Role::Data, bf16_values(std::size_t(T) * kQuery, -3.0, 3.0, rng));
    auto* attention =
        s.add("attention_fp32", Role::Data, f32_values(std::size_t(T) * kQuery, -1.0, 1.0, rng));
    const Fp8Rows output_weight = add_fp8_rows(s, "o_proj", kHidden, kQuery, rng);
    const auto gated_image      = add_image(s, "gated_image", T, kQuery);
    auto* residual =
        s.add("residual", Role::Data, bf16_values(std::size_t(T) * kHidden, -2.0, 2.0, rng));
    auto* norm = s.add("post_norm", Role::Const, bf16_values(kHidden, -0.2, 0.2, rng));
    const auto normalized_image = add_image(s, "normalized_image", T, kHidden);
    auto* normalized            = s.output<hip_bfloat16>("normalized", std::size_t(T) * kHidden);
    const Lut4 gate_up          = add_lut4(s, "gate_up", 2U * kIntermediate, kHidden, rng);
    auto* activation = s.output<hip_bfloat16>("mlp_activation", std::size_t(T) * kIntermediate);
    const auto activation_image = add_image(s, "activation_image", T, kIntermediate);
    const Lut4 down             = add_lut4(s, "down", kHidden, kIntermediate, rng);
    return qualify(
        s, 6, 1,
        [&](hipStream_t st) {
            check(lin::fp8_quantize_gated_activation({.gate      = gate,
                                                      .attention = attention,
                                                      .workspace = gated_image,
                                                      .warm      = fp8_warm(output_weight)},
                                                     st),
                  "quantize gated");
            check(lin::fp8_small_t_residual(output_weight.view, gated_image, residual, st),
                  "fp8 residual");
            check(
                lin::fp8_quantize_normalized_activation({.input       = residual,
                                                         .weight      = norm,
                                                         .eps         = 1e-6F,
                                                         .unit_offset = true,
                                                         .workspace   = normalized_image,
                                                         .normalized  = normalized,
                                                         .warm = lut4_warm(gate_up, 0U, 2U * kMiB)},
                                                        st),
                "quantize normalized");
            check(lin::fp8lut4_linear(gate_up.view, normalized_image,
                                      {.leading = activation, .silu_pair = true}, st),
                  "gate_up silu");
            check(lin::fp8_quantize_activation(
                      {activation, activation_image, lut4_warm(down, 0U, 2U * kMiB)}, st),
                  "quantize activation");
            check(lin::fp8lut4_linear(down.view, activation_image,
                                      {.leading = residual, .accumulate = true}, st),
                  "down accumulate");
        },
        stream);
}

// Plain quantization feeding four consumers of one image: row-scaled FP8 store, an FP8LUT4 pair
// with a second weight and feature splits on both, a plain FP8LUT4 store and an accumulating
// FP8LUT4 pair into that output (QuantizeActivation, Fp8SmallT<Store>, Fp8Lut4SmallT<false,false>
// x2, Fp8Lut4SmallT<true,false>).
Outcome projections(unsigned T, hipStream_t stream) {
    Scenario s("quantize + projections T" + std::to_string(T));
    Rng rng(300U + T);
    auto* x = s.add("x", Role::Data, bf16_values(std::size_t(T) * kHidden, -2.0, 2.0, rng));
    const auto image      = add_image(s, "image", T, kHidden);
    const Fp8Rows rows    = add_fp8_rows(s, "fp8_query_key", kAttentionRows, kHidden, rng);
    auto* y               = s.output<hip_bfloat16>("fp8_store", std::size_t(T) * kAttentionRows);
    const Lut4 query_key  = add_lut4(s, "lut4_query_key", kAttentionRows, kHidden, rng);
    const Lut4 gate_value = add_lut4(s, "lut4_gate_value", kAttentionRows, kHidden, rng);
    auto* q               = s.output<hip_bfloat16>("q_leading", std::size_t(T) * kQuery);
    auto* k               = s.output<hip_bfloat16>("k_trailing", std::size_t(T) * kKv);
    auto* g               = s.output<hip_bfloat16>("gate_leading", std::size_t(T) * kQuery);
    auto* v               = s.output<hip_bfloat16>("value_trailing", std::size_t(T) * kKv);
    const Lut4 plain      = add_lut4(s, "lut4_plain", kGdnQueryKey, kHidden, rng);
    auto* plain_out       = s.output<hip_bfloat16>("plain_out", std::size_t(T) * kGdnQueryKey);
    const Lut4 second     = add_lut4(s, "lut4_second", kGdnQueryKey, kHidden, rng);
    auto* second_out      = s.add("second_accumulator", Role::Output,
                                  bf16_values(std::size_t(T) * kGdnQueryKey, -1.0, 1.0, rng));
    return qualify(
        s, 5, 1,
        [&](hipStream_t st) {
            check(lin::fp8_quantize_activation({x, image, fp8_warm(rows)}, st), "quantize");
            check(lin::fp8_small_t_linear(rows.view, image, y, st), "fp8 store");
            check(lin::fp8lut4_linear_pair(
                      query_key.view, {.leading = q, .trailing = k, .split = kQuery},
                      gate_value.view, {.leading = g, .trailing = v, .split = kQuery}, image, st),
                  "lut4 pair split");
            check(lin::fp8lut4_linear(plain.view, image, {.leading = plain_out}, st), "lut4 plain");
            check(lin::fp8lut4_linear_pair(plain.view, {.leading = plain_out, .accumulate = true},
                                           second.view, {.leading = second_out, .accumulate = true},
                                           image, st),
                  "lut4 accumulate pair");
        },
        stream);
}

// One GDN layer at verification width T: FP8 normalized front (optionally applying this layer's
// deferred replay fold into the shared state pool) -> FP8LUT4 pair projection + recorded
// convolution (chain or tree parents, partial valid columns) -> replay-record recurrence on the
// folded state -> gated RMSNorm quantization -> FP8LUT4 output projection accumulated into the
// residual (GdnNormalizedFront, GdnPairConvRecord<T>, GdnRecord, QuantizeGatedRmsNorm,
// Fp8Lut4SmallT<true,false>). The recurrence and the pair convolution overwrite the record row the
// fold reads, as the next round does in production. Tree mode runs the (unhosted) tree recurrence
// between two lowered runs.
struct GdnCase {
    unsigned T;
    bool tree;
    unsigned valid;
    bool poison;
    int fold_batch;    // 0: no fold
    int fold_columns;  // record row 0 -> slot 1 (the round's initial slot)
    int fold_columns2; // record row 1 -> slot 2 (fold_batch 2)
};

Outcome gdn_layer(const GdnCase& c, hipStream_t stream) {
    const unsigned T = c.T;
    std::string name = "gdn layer T" + std::to_string(T) + (c.tree ? " tree" : " chain");
    if (c.valid < T) name += " valid" + std::to_string(c.valid);
    if (c.poison) name += " poisoned";
    if (c.fold_batch > 0)
        name += " fold" + std::to_string(c.fold_batch) + "x" + std::to_string(c.fold_columns) +
                (c.fold_batch > 1 ? "+" + std::to_string(c.fold_columns2) : "");
    Scenario s(name);
    Rng rng(400U + T * 7U + (c.tree ? 1U : 0U) + (c.poison ? 2U : 0U) + c.valid * 13U +
            static_cast<unsigned>(c.fold_batch) * 101U);
    constexpr unsigned kSlots           = 4;
    constexpr int kLayer                = 5;
    constexpr std::int32_t kInitialSlot = 1;

    auto residual_host = bf16_values(std::size_t(T) * kHidden, -2.0, 2.0, rng);
    if (c.poison) residual_host[std::size_t(T / 2U) * kHidden + 211U] = hip_bfloat16(NAN);
    auto* residual       = s.add("residual", Role::Data, residual_host);
    auto* norm           = s.add("input_norm", Role::Const, bf16_values(kHidden, -0.2, 0.2, rng));
    auto* a_weight       = s.add("a_proj", Role::Const,
                                 bf16_values(std::size_t(kGdnHeads) * kHidden, -0.02, 0.02, rng));
    auto* b_weight       = s.add("b_proj", Role::Const,
                                 bf16_values(std::size_t(kGdnHeads) * kHidden, -0.02, 0.02, rng));
    auto* a_log          = s.add("a_log", Role::Const, f32_values(kGdnHeads, -0.5, 0.5, rng));
    auto* dt_bias        = s.add("dt_bias", Role::Const, f32_values(kGdnHeads, -0.5, 0.5, rng));
    auto* g              = s.output<float>("g", std::size_t(T) * kGdnHeads);
    auto* beta           = s.output<float>("beta", std::size_t(T) * kGdnHeads);
    const auto image     = add_image(s, "image", T, kHidden);
    const Lut4 query_key = add_lut4(s, "query_key", kGdnQueryKey, kHidden, rng);
    const Lut4 value_z   = add_lut4(s, "value_z", kGdnValueZ, kHidden, rng);
    auto* conv_weight =
        s.add("conv_weight", Role::Const, bf16_values(4U * kGdnChannels, -0.5, 0.5, rng));
    auto* recurrent = s.add(
        "recurrent_state", Role::Data,
        f32_values(std::size_t(kSlots) * kGdnHeads * kStateDim * kStateDim, -0.006, 0.006, rng));
    auto* conv_states = s.add("conv_state", Role::Data,
                              bf16_values(std::size_t(kSlots) * 3U * kGdnChannels, -1.0, 1.0, rng));
    auto* valid       = s.add("valid_columns", Role::Const,
                              std::vector<std::int32_t>{static_cast<std::int32_t>(c.valid)});
    auto* initial     = s.add("initial_slot", Role::Const, std::vector<std::int32_t>{kInitialSlot});
    std::vector<std::int32_t> parents_host(T);
    for (unsigned t = 0; t < T; ++t)
        parents_host[t] = c.tree ? (t == 0U ? -1 : static_cast<std::int32_t>(t / 2U))
                                 : static_cast<std::int32_t>(t) - 1;
    auto* parents = s.add("parent_index", Role::Const, parents_host);

    // Replay records of the previous round (all 48 GDN layers, two record rows, width T).
    ninfer::LayoutBuilder builder;
    const ninfer::GdnReplayRecordLayout layout = ninfer::plan_gdn_replay_records(
        builder, {.layers          = 48,
                  .record_capacity = 2,
                  .width           = static_cast<std::int32_t>(T),
                  .conv_channels   = static_cast<std::int32_t>(kGdnChannels),
                  .qk_heads        = 16,
                  .value_heads     = static_cast<std::int32_t>(kGdnHeads),
                  .key_dim         = static_cast<std::int32_t>(kStateDim),
                  .value_dim       = static_cast<std::int32_t>(kStateDim)});
    const std::size_t record_bytes = builder.finish(256);
    Buffer& record_buffer =
        s.add_buffer("replay_records", Role::Data, std::vector<std::uint8_t>(record_bytes, 0U));
    const ninfer::GdnReplayRecords records({record_buffer.device, record_bytes}, layout);
    const auto offset = [&](const Tensor& t) {
        return static_cast<std::size_t>(static_cast<std::uint8_t*>(t.data) -
                                        static_cast<std::uint8_t*>(record_buffer.device));
    };
    const auto fill_bf16 = [&](const Tensor& t, double lo, double hi) {
        const auto values = bf16_values(static_cast<std::size_t>(t.numel()), lo, hi, rng);
        std::memcpy(record_buffer.host.data() + offset(t), values.data(), values.size() * 2U);
    };
    fill_bf16(records.conv, -1.0, 1.0);
    fill_bf16(records.key, -1.0, 1.0);
    fill_bf16(records.value, -1.0, 1.0);
    {
        std::vector<float> gates(static_cast<std::size_t>(records.gate.numel()));
        for (std::size_t i = 0; i < gates.size(); i += 2U) {
            gates[i]      = rng.range(-1.0, -0.01); // g
            gates[i + 1U] = rng.range(0.05, 0.95);  // beta
        }
        std::memcpy(record_buffer.host.data() + offset(records.gate), gates.data(),
                    gates.size() * 4U);
    }
    const ninfer::GdnReplayRecordLayer layer = records.layer(kLayer, 0, 1);

    auto* query         = s.output<hip_bfloat16>("query", std::size_t(T) * 2048U);
    auto* key           = s.output<hip_bfloat16>("key", std::size_t(T) * 2048U);
    auto* value         = s.output<hip_bfloat16>("value", std::size_t(T) * kGdnValue);
    auto* z             = s.output<hip_bfloat16>("z", std::size_t(T) * kGdnValue);
    auto* recurrent_out = s.output<hip_bfloat16>("recurrent_output", std::size_t(T) * kGdnValue);
    const std::size_t tree_bytes =
        ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
            static_cast<std::int32_t>(kGdnHeads), 1, static_cast<std::int32_t>(T));
    void* tree_scratch = c.tree ? s.add_buffer("tree_states", Role::Output,
                                               std::vector<std::uint8_t>(tree_bytes, 0x3CU))
                                      .device
                                : nullptr;
    auto* gated_norm   = s.add("gated_norm", Role::Const, bf16_values(kStateDim, 0.5, 1.5, rng));
    const auto output_image = add_image(s, "output_image", T, kGdnValue);
    const Lut4 out_proj     = add_lut4(s, "out_proj", kHidden, kGdnValue, rng);

    ninfer::ops::GdnDeferredFoldRows fold_rows{};
    fold_rows.row[0] = {
        .linear_state_slot = kInitialSlot, .commit_columns = c.fold_columns, .record_row = 0};
    fold_rows.row[1] = {.linear_state_slot = 2, .commit_columns = c.fold_columns2, .record_row = 1};
    auto* rows =
        s.add("fold_rows", Role::Const, std::vector<ninfer::ops::GdnDeferredFoldRows>{fold_rows});
    const int Ti    = static_cast<int>(T);
    const int slots = static_cast<int>(kSlots);
    const ninfer::ops::GdnLayerFold fold{
        .records   = &records,
        .recurrent = Tensor(recurrent, DType::FP32, {128, 128, 48, slots}),
        .conv =
            Tensor(conv_states, DType::BF16, {static_cast<std::int32_t>(kGdnChannels), 3, slots}),
        .layer = kLayer,
        .rows  = rows,
        .batch = c.fold_batch,
    };
    std::unique_ptr<ninfer::DeviceArena> arena;
    if (c.tree)
        arena = std::make_unique<ninfer::DeviceArena>(ninfer::DeviceSpan{tree_scratch, tree_bytes});

    return qualify(
        s, 5, c.tree ? 3 : 1,
        [&](hipStream_t st) {
            check(gdn::fp8_gdn_normalized_front(residual, norm, 1e-6F, true, a_weight, b_weight,
                                                a_log, dt_bias, g, beta, image, st,
                                                lut4_warm(query_key, 0U, 4U * kMiB),
                                                c.fold_batch > 0 ? &fold : nullptr),
                  "gdn front");
            check(gdn::gdn_fp8lut4_pair_conv_record_bf16(
                      image, query_key.view, value_z.view, conv_weight, conv_states, valid, initial,
                      c.tree ? parents : nullptr, static_cast<hip_bfloat16*>(layer.conv.data),
                      query, key, value, z, T, kSlots, st),
                  "gdn pair conv record");
            const Tensor q_t(query, DType::BF16, {128, 16, Ti, 1}),
                k_t(key, DType::BF16, {128, 16, Ti, 1});
            const Tensor v_t(value, DType::BF16, {128, 48, Ti, 1});
            const Tensor g_t(g, DType::FP32, {48, Ti, 1}), beta_t(beta, DType::FP32, {48, Ti, 1});
            const Tensor states_t(recurrent, DType::FP32, {128, 128, 48, slots});
            const Tensor valid_t(valid, DType::I32, {1}), initial_t(initial, DType::I32, {1});
            const Tensor parent_t(parents, DType::I32, {Ti});
            Tensor key_record = layer.key, value_record = layer.value, gate_record = layer.gate;
            Tensor out_t(recurrent_out, DType::BF16, {128, 48, Ti, 1});
            ninfer::ops::gated_delta_net_replay_record(
                q_t, k_t, v_t, g_t, beta_t, 1.0F / std::sqrt(128.0F), states_t, valid_t, initial_t,
                key_record, value_record, gate_record, out_t, st, c.tree ? &parent_t : nullptr,
                arena.get(), lut4_warm(out_proj, 2U * kMiB, 4U * kMiB));
            check(lin::fp8_quantize_gated_rmsnorm_activation(
                      {.input     = recurrent_out,
                       .gate      = z,
                       .weight    = gated_norm,
                       .eps       = 1e-6F,
                       .workspace = output_image,
                       .warm      = lut4_warm(out_proj, 0U, 2U * kMiB)},
                      st),
                  "quantize gated rmsnorm");
            check(lin::fp8lut4_linear(out_proj.view, output_image,
                                      {.leading = residual, .accumulate = true}, st),
                  "out_proj accumulate");
        },
        stream);
}

// Q/K norm + RoPE chain: 1-D RoPE of token-interleaved Q/K planes (padded token stride) -> MRoPE
// of its outputs -> 1-D RoPE again (QkNormRope<Text1D>, QkNormRope<TextMrope>).
Outcome rope_chain(unsigned T, hipStream_t stream) {
    Scenario s("qk norm rope chain T" + std::to_string(T));
    Rng rng(500U + T);
    auto* packed = s.add("packed_qk", Role::Data,
                         bf16_values(std::size_t(T) * kAttentionRows, -2.0, 2.0, rng));
    auto* q_norm = s.add("q_norm", Role::Const, bf16_values(256, -0.3, 0.3, rng));
    auto* k_norm = s.add("k_norm", Role::Const, bf16_values(256, -0.3, 0.3, rng));
    std::vector<std::int32_t> one(T), three(3U * T);
    for (unsigned t = 0; t < T; ++t) {
        one[t] = static_cast<std::int32_t>(100U + 3U * t);
        for (unsigned axis = 0; axis < 3U; ++axis)
            three[axis * T + t] = static_cast<std::int32_t>(100U + t + 7U * axis);
    }
    auto* pos1 = s.add("positions_1d", Role::Const, one);
    auto* pos3 = s.add("positions_mrope", Role::Const, three);
    hip_bfloat16* qn[3];
    hip_bfloat16* kn[3];
    for (int i = 0; i < 3; ++i) {
        qn[i] = s.output<hip_bfloat16>("qn" + std::to_string(i), std::size_t(T) * kQuery);
        kn[i] = s.output<hip_bfloat16>("kn" + std::to_string(i), std::size_t(T) * kKv);
    }
    const int Ti = static_cast<int>(T);
    return qualify(
        s, 3, 1,
        [&](hipStream_t st) {
            const Tensor p1(pos1, DType::I32, {Ti}), p3(pos3, DType::I32, {Ti, 3});
            const Tensor qw(q_norm, DType::BF16, {256}), kw(k_norm, DType::BF16, {256});
            Tensor q(packed, DType::BF16, {256, 24, Ti}),
                k(packed + kQuery, DType::BF16, {256, 4, Ti});
            const std::int64_t stride = static_cast<std::int64_t>(kAttentionRows) * 2;
            q.nb[2] = k.nb[2] = stride;
            q.nb[3] = k.nb[3] = stride * Ti;
            Tensor qt[3], kt[3];
            for (int i = 0; i < 3; ++i) {
                qt[i] = Tensor(qn[i], DType::BF16, {256, 24, Ti});
                kt[i] = Tensor(kn[i], DType::BF16, {256, 4, Ti});
            }
            ninfer::ops::qk_norm_rope(p1, 64, 1.0e7F, q, k, qw, kw, 1e-6F, qt[0], kt[0], st);
            ninfer::ops::qk_norm_rope(p3, 64, 1.0e7F, qt[0], kt[0], qw, kw, 1e-6F, qt[1], kt[1],
                                      st);
            ninfer::ops::qk_norm_rope(p1, 64, 1.0e7F, qt[1], kt[1], qw, kw, 1e-6F, qt[2], kt[2],
                                      st);
        },
        stream);
}

int select_r9700() {
    int count = 0;
    check(hipGetDeviceCount(&count), "device count");
    int selected = -1;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, ordinal), "device properties");
        if (std::string_view(properties.gcnArchName).starts_with("gfx1201") &&
            properties.warpSize == 32) {
            require(selected < 0, "multiple wave32 gfx1201 devices found");
            selected = ordinal;
        }
    }
    require(selected >= 0, "no wave32 gfx1201 device found");
    return selected;
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (const char ch : text) {
        if (ch == '"' || ch == '\\') out += '\\';
        if (ch == '\n') {
            out += "\\n";
            continue;
        }
        out += ch;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string json_path;
    if (argc == 3 && std::string_view(argv[1]) == "--out-json") {
        json_path = argv[2];
    } else if (argc != 1) {
        std::cerr << "usage: ninfer_r9700_persistent_decode_qual [--out-json FILE]\n";
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        check(hipSetDevice(select_r9700()), "hipSetDevice");
        hipStream_t stream = nullptr;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        std::vector<Outcome> outcomes;
        for (const unsigned T : {4U, 8U}) outcomes.push_back(attention_front(T, false, stream));
        outcomes.push_back(attention_front(8U, true, stream));
        for (const unsigned T : {4U, 8U}) outcomes.push_back(attention_output_mlp(T, stream));
        for (const unsigned T : {4U, 8U}) outcomes.push_back(projections(T, stream));
        for (const GdnCase& c :
             {GdnCase{4, false, 4, false, 0, 0, 0}, GdnCase{4, false, 4, false, 1, 1, 0},
              GdnCase{5, true, 3, false, 0, 0, 0}, GdnCase{6, false, 6, true, 1, 6, 0},
              GdnCase{6, true, 6, false, 1, 3, 0}, GdnCase{7, false, 5, true, 0, 0, 0},
              GdnCase{7, false, 7, false, 2, 4, 0}, GdnCase{8, false, 8, false, 0, 0, 0},
              GdnCase{8, false, 8, false, 1, 7, 0}, GdnCase{8, true, 8, true, 2, 8, 2}})
            outcomes.push_back(gdn_layer(c, stream));
        for (const unsigned T : {4U, 8U}) outcomes.push_back(rope_chain(T, stream));
        check(hipStreamDestroy(stream), "stream destroy");

        const bool pass =
            std::all_of(outcomes.begin(), outcomes.end(), [](const Outcome& o) { return o.pass; });
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!json_path.empty()) {
            std::ofstream out(json_path);
            out << "{\n  \"qualification\": \"persistent_decode_lower\",\n  \"device\": "
                   "\"gfx1201\",\n"
                << "  \"pass\": " << (pass ? "true" : "false") << ",\n  \"seconds\": " << seconds
                << ",\n  \"scenarios\": [\n";
            for (std::size_t i = 0; i < outcomes.size(); ++i) {
                const Outcome& o = outcomes[i];
                out << "    {\"name\": \"" << json_escape(o.name)
                    << "\", \"pass\": " << (o.pass ? "true" : "false")
                    << ", \"captured_nodes\": " << o.captured
                    << ", \"lowered_nodes\": " << o.lowered
                    << ", \"compared_buffers\": " << o.buffers
                    << ", \"compared_bytes\": " << o.bytes << ", \"seconds\": " << o.seconds
                    << ", \"detail\": \"" << json_escape(o.detail) << "\"}"
                    << (i + 1U == outcomes.size() ? "\n" : ",\n");
            }
            out << "  ]\n}\n";
            require(static_cast<bool>(out), "writing " + json_path);
        }
        std::cout << outcomes.size() << " scenarios, " << seconds << " s\n";
        if (!pass) {
            std::cout << "persistent decode qualification: FAIL\n";
            return 1;
        }
        std::cout << "persistent decode qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "persistent decode qualification failed: " << error.what() << '\n';
        return 1;
    }
}
