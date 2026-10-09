// Qualification of the FP8LUT4 x per-token E4M3 Linear (fp8lut4_linear) against an FP64 oracle that
// decodes the stored codes through an independently rounded codebook: every group code's table,
// small-T and prefill widths on the real Qwen3.8 projection shapes, the residual epilogue and the
// poisoned-activation contract. Optional timing (--time) on the same shapes.
#include "qualification_device_guard.h"
#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace linear = ninfer::ops::r9700::linear;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;

void check(hipError_t error, const char* what) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
}

void require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

template <class T>
struct Device {
    T* data           = nullptr;
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

    double normal() {
        return std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307 * uniform());
    }
};

double bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// Independent codebook: exhaustive E4M3 rounding of the exact rational magnitude.
std::vector<double> reference_codebook() {
    constexpr int kBase[8] = {0, 13, 27, 41, 56, 74, 94, 120};
    std::vector<double> book(256 * 8);
    for (int code = 0; code < 256; ++code) {
        const int m = code & 7, exponent = (code >> 3) - 26;
        for (int j = 0; j < 8; ++j) {
            const float exact  = static_cast<float>(std::ldexp(kBase[j] * (8 + m), exponent - 7));
            book[code * 8 + j] = oracle::decode_e4m3fn(oracle::encode_e4m3fn(exact));
        }
    }
    return book;
}

void check_table(const std::vector<double>& book) {
    const auto& table = linear::fp8lut4_magnitude_table();
    for (int code = 0; code < 256; ++code)
        for (int j = 0; j < 8; ++j)
            require(oracle::decode_e4m3fn(table[code][j]) == book[code * 8 + j],
                    "codebook word differs at group code " + std::to_string(code));
}

struct Weight {
    std::uint32_t rows, columns;
    std::vector<std::uint8_t> codes, groups;
    std::vector<float> scales;
    Device<std::uint8_t> device_codes, device_groups;
    Device<float> device_scales;

    Weight(std::uint32_t n, std::uint32_t k, Rng& rng)
        : rows(n), columns(k), codes(static_cast<std::size_t>(n) * k / 2U),
          groups(static_cast<std::size_t>(n) * k / 32U), scales(n), device_codes(codes.size()),
          device_groups(groups.size()), device_scales(n) {
        for (auto& c : codes) c = static_cast<std::uint8_t>(rng.next());
        // Group codes around the converter's operating range (E in [-8, 5]) plus rare extremes.
        for (auto& g : groups) {
            const std::uint32_t r = rng.next();
            const std::uint32_t exponent =
                (r & 63U) == 0U ? (r >> 8U) % 32U : 18U + (r >> 8U) % 14U;
            g = static_cast<std::uint8_t>((exponent << 3U) | (r >> 16U) % 8U);
        }
        for (auto& s : scales) s = static_cast<float>(std::ldexp(1.0 + rng.uniform(), -9));
        device_codes.upload(codes);

        device_groups.upload(groups);
        device_scales.upload(scales);
    }

    linear::Fp8Lut4Weight view() const {
        return {device_codes.data, device_groups.data, device_scales.data, rows, columns};
    }

    // r9700-fp8lut4-n16k64-v1: tile (r / 16) * (K / 64) + k / 64, slot 16 * ((k % 64) / 32) + r
    // % 16.
    double value(const std::vector<double>& book, std::uint32_t r, std::uint32_t k) const {
        const std::size_t tile   = static_cast<std::size_t>(r / 16U) * (columns / 64U) + k / 64U;
        const std::uint32_t slot = 16U * ((k % 64U) / 32U) + r % 16U;
        const std::uint8_t byte  = codes[tile * 512U + slot * 16U + (k % 32U) / 2U];
        const std::uint32_t code = (k & 1U) != 0U ? byte >> 4U : byte & 15U;
        const double magnitude   = book[groups[tile * 32U + slot] * 8U + (code & 7U)];
        return (code & 8U) != 0U ? -magnitude : magnitude;
    }
};

struct Activation {
    std::vector<std::uint8_t> storage_bytes;
    Device<std::uint8_t> storage;
    linear::Fp8ActivationWorkspace workspace{};

    Activation(std::uint32_t tokens, std::uint32_t columns)
        : storage(linear::fp8_activation_workspace_capacity_bytes(tokens, columns)) {
        check(linear::fp8_bind_activation_workspace(storage.data, storage.count, tokens, columns,
                                                    &workspace),
              "bind");
    }
};

void run_case(const std::vector<double>& book, std::uint32_t rows, std::uint32_t columns,
              std::uint32_t tokens, bool accumulate, bool poison, Rng& rng, hipStream_t stream,
              bool timed = false, bool graph_timing = false) {
    const std::string label = "N" + std::to_string(rows) + " K" + std::to_string(columns) + " T" +
                              std::to_string(tokens) + (accumulate ? " residual" : "") +
                              (poison ? " poisoned" : "");
    Weight weight(rows, columns, rng);
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = hip_bfloat16(static_cast<float>(rng.normal() * (i % 97U == 0U ? 20.0 : 1.0)));
    if (poison) input[input.size() / 2U] = hip_bfloat16(NAN);
    Device<hip_bfloat16> device_input(input.size());
    device_input.upload(input);
    Activation activation(tokens, columns);
    if (graph_timing)
        check(hipMemsetAsync(activation.storage.data, 0xff, activation.storage.count, stream),
              "poison represented activation storage before excluded producer");
    check(linear::fp8_quantize_activation({device_input.data, activation.workspace}, stream),
          "quantize");
    std::vector<hip_bfloat16> base(static_cast<std::size_t>(tokens) * rows);
    for (auto& b : base) b = hip_bfloat16(static_cast<float>(rng.normal()));
    Device<hip_bfloat16> output(base.size());
    output.upload(base);
    // A plain output starts as NaN: every element must be published.
    if (!accumulate)
        check(hipMemsetAsync(output.data, 0xff, output.count * sizeof(hip_bfloat16), stream),
              "poison output");
    check(linear::fp8lut4_linear(weight.view(), activation.workspace,
                                 {.leading = output.data, .accumulate = accumulate}, stream),
          "fp8lut4");
    check(hipStreamSynchronize(stream), "sync");
    constexpr int graph_batch = 32;
    hipGraph_t graph          = nullptr;
    hipGraphExec_t executable = nullptr;
    const auto launch         = [&] {
        check(linear::fp8lut4_linear(weight.view(), activation.workspace, {.leading = output.data},
                                     stream),
              "timed fp8lut4");
    };
    if (graph_timing) {
        require(timed && !accumulate && !poison, "graph timing requires plain Linear");
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
              "begin graph capture");
        for (int i = 0; i < graph_batch; ++i) launch();
        check(hipStreamEndCapture(stream, &graph), "end graph capture");
        check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");
        // No global Linear scratch exists: activation storage is a represented input, while
        // partial reductions live in LDS. Poison output so an empty/stale graph cannot pass.
        check(hipMemsetAsync(output.data, 0xff, output.count * sizeof(hip_bfloat16), stream),
              "poison graph output");
        check(hipGraphLaunch(executable, stream), "qualify graph replay");
        check(hipStreamSynchronize(stream), "sync qualified graph");
    }
    const auto got = output.download();
    if (graph_timing || !poison)
        for (const auto value : got)
            require(std::isfinite(bf16_value(value.data)), label + " unwritten/nonfinite output");
    if (poison) {
        // The nonfinite element belongs to token poisoned_token; exactly its outputs are NaN.
        const std::uint32_t poisoned_token =
            static_cast<std::uint32_t>((input.size() / 2U) / columns);
        for (std::uint32_t t = 0; t < tokens; ++t)
            for (std::uint32_t r = 0; r < rows; ++r)
                require(std::isnan(static_cast<float>(
                            got[static_cast<std::size_t>(t) * rows + r])) == (t == poisoned_token),
                        label + " per-token poisoning");
        return;
    }
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> token_scales(tokens);
    check(hipMemcpy(codes.data(), activation.workspace.codes, codes.size(), hipMemcpyDeviceToHost),
          "codes");
    check(hipMemcpy(token_scales.data(), activation.workspace.scales, tokens * sizeof(float),
                    hipMemcpyDeviceToHost),
          "scales");
    std::vector<std::uint32_t> sample_tokens;
    for (std::uint32_t t = 0; t < tokens; t += tokens > 128U ? 131U : tokens > 64U ? 7U : 1U)
        sample_tokens.push_back(t);
    sample_tokens.push_back(tokens - 1U);
    const std::uint32_t row_stride = rows > 8192U ? 97U : 13U;
    double worst                   = 0.0;
    std::vector<double> weight_row(columns);
    for (std::uint32_t r = 0; r < rows; r += row_stride) {
        for (std::uint32_t k = 0; k < columns; ++k) weight_row[k] = weight.value(book, r, k);
        for (const std::uint32_t t : sample_tokens) {
            double sum = 0.0, magnitude = 0.0;
            const std::uint8_t* a = codes.data() + static_cast<std::size_t>(t) * columns;
            for (std::uint32_t k = 0; k < columns; ++k) {
                const double product = weight_row[k] * oracle::decode_e4m3fn(a[k]);
                sum += product;
                magnitude += std::abs(product);
            }
            const double scale      = static_cast<double>(weight.scales[r]) * token_scales[t];
            double expected         = sum * scale;
            const std::size_t index = static_cast<std::size_t>(t) * rows + r;
            const double prior      = accumulate ? bf16_value(base[index].data) : 0.0;
            expected += prior;
            const double error = std::abs(bf16_value(got[index].data) - expected);
            // BF16 output rounding(s) plus FP32 accumulation of the K products.
            const double bound = (std::abs(expected) + std::abs(prior)) / 128.0 +
                                 magnitude * scale * std::ldexp(1.0, -16) + 1e-30;
            require(std::isfinite(error) && std::isfinite(bound),
                    label + " nonfinite oracle error");
            worst = std::max(worst, error / bound);
        }
    }
    require(worst <= 1.0, label + " FP64 oracle bound exceeded: " + std::to_string(worst));
    std::cout << label << ": worst/bound " << worst << "\n";
    if (timed) {
        require(!accumulate && !poison, "timing requires the nonaccumulating Linear output");
        ninfer::r9700_qualification::device_guard().check("after-linear-oracle-before-timing");
        const auto replay = [&] {
            if (graph_timing)
                check(hipGraphLaunch(executable, stream), "timed graph replay");
            else
                launch();
        };
        for (int i = 0; i < (graph_timing ? 3 : 10); ++i) replay();
        check(hipStreamSynchronize(stream), "warmup complete");
        hipEvent_t begin, end;
        check(hipEventCreate(&begin), "event");
        check(hipEventCreate(&end), "event");
        constexpr std::size_t trials = 7U;
        std::array<double, trials> samples{};
        const int iterations = graph_timing ? (tokens > 16U ? 2 : 16) : (tokens > 16U ? 20 : 200);
        for (auto& sample : samples) {
            check(hipEventRecord(begin, stream), "record");
            for (int i = 0; i < iterations; ++i) replay();
            check(hipEventRecord(end, stream), "record");
            check(hipEventSynchronize(end), "sync");
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, begin, end), "elapsed");
            sample = 1e3 * ms / (iterations * (graph_timing ? graph_batch : 1));
        }
        check(hipEventDestroy(begin), "destroy event");
        check(hipEventDestroy(end), "destroy event");
        ninfer::r9700_qualification::device_guard().check("after-linear-timing");
        std::sort(samples.begin(), samples.end());
        std::cout << "complete_linear " << label << " median_us=" << samples[trials / 2U]
                  << " min_us=" << samples.front() << " max_us=" << samples.back()
                  << " trials=" << trials << " iterations=" << iterations
                  << " execution=" << (graph_timing ? "graph" : "eager")
                  << " ops_per_replay=" << (graph_timing ? graph_batch : 1)
                  << " activation_quantize_included=false linear_global_scratch_bytes=0\n";
        if (executable != nullptr)
            check(hipGraphExecDestroy(executable), "destroy graph executable");
        if (graph != nullptr) check(hipGraphDestroy(graph), "destroy graph");
    }
}

// Two projections of one image (one launch at small T and at the fused T 33..128 widths). Either
// may split its rows: rows [0, split) and [split, N) publish to separate outputs (split 0: one
// output). Outputs start as NaN, so every element must be published; sampled rows of both are
// checked against the oracle through the splits.
void run_pair(const std::vector<double>& book, std::uint32_t rows0, std::uint32_t split0,
              std::uint32_t rows1, std::uint32_t columns, std::uint32_t tokens, Rng& rng,
              hipStream_t stream, std::uint32_t split1 = 0U, bool poison = false) {
    const std::string label =
        std::string(poison ? "poisoned " : "") + "pair N" + std::to_string(rows0) +
        (split0 != 0U ? "/" + std::to_string(split0) : "") + "+N" + std::to_string(rows1) +
        (split1 != 0U ? "/" + std::to_string(split1) : "") + " T" + std::to_string(tokens);
    Weight first(rows0, columns, rng), second(rows1, columns, rng);
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns);
    for (auto& x : input) x = hip_bfloat16(static_cast<float>(rng.normal()));
    const std::uint32_t poisoned_token = tokens / 2U;
    if (poison) input[static_cast<std::size_t>(poisoned_token) * columns + 3U] = hip_bfloat16(NAN);
    Device<hip_bfloat16> device_input(input.size());
    device_input.upload(input);
    Activation activation(tokens, columns);
    check(linear::fp8_quantize_activation({device_input.data, activation.workspace}, stream), "q");

    struct Side {
        const Weight& weight;
        std::uint32_t split;
        Device<hip_bfloat16> leading, trailing;

        Side(const Weight& w, std::uint32_t s, std::uint32_t tokens)
            : weight(w), split(s),
              leading(static_cast<std::size_t>(tokens) * (s != 0U ? s : w.rows)),
              trailing(s != 0U ? static_cast<std::size_t>(tokens) * (w.rows - s) : 1U) {}

        linear::Fp8Lut4Output output() const {
            return {leading.data, split != 0U ? trailing.data : nullptr, split, false};
        }
    };

    Side sides[2]{{first, split0, tokens}, {second, split1, tokens}};
    for (const Side& side : sides)
        for (const Device<hip_bfloat16>* buffer : {&side.leading, &side.trailing})
            check(hipMemsetAsync(buffer->data, 0xff, buffer->count * sizeof(hip_bfloat16), stream),
                  "poison pair output");
    check(linear::fp8lut4_linear_pair(first.view(), sides[0].output(), second.view(),
                                      sides[1].output(), activation.workspace, stream),
          "pair");
    check(hipStreamSynchronize(stream), "sync");
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> token_scales(tokens);
    check(hipMemcpy(codes.data(), activation.workspace.codes, codes.size(), hipMemcpyDeviceToHost),
          "c");
    check(hipMemcpy(token_scales.data(), activation.workspace.scales, tokens * 4U,
                    hipMemcpyDeviceToHost),
          "s");
    double worst = 0.0;
    for (const Side& side : sides) {
        const auto lead = side.leading.download(), trail = side.trailing.download();
        // Every element published; a poisoned run: exactly the flagged token's are the canonical
        // NaN.
        const auto published = [&](const std::vector<hip_bfloat16>& values, std::uint32_t width) {
            for (std::size_t i = 0; i < values.size(); ++i) {
                const bool flagged = poison && i / width == poisoned_token;
                require(flagged ? values[i].data == 0x7FC0U
                                : std::isfinite(bf16_value(values[i].data)),
                        label + (flagged ? " per-token poisoning" : " unwritten/nonfinite output"));
            }
        };
        published(lead, side.split != 0U ? side.split : side.weight.rows);
        if (side.split != 0U) published(trail, side.weight.rows - side.split);
        if (poison) continue;
        const std::uint32_t rows = side.weight.rows, split = side.split;
        for (std::uint32_t t = 0; t < tokens; ++t) {
            for (std::uint32_t r = 0; r < rows; r += &side == sides ? 29U : 31U) {
                const double got =
                    split == 0U ? bf16_value(lead[static_cast<std::size_t>(t) * rows + r].data)
                    : r < split
                        ? bf16_value(lead[static_cast<std::size_t>(t) * split + r].data)
                        : bf16_value(
                              trail[static_cast<std::size_t>(t) * (rows - split) + r - split].data);
                double sum = 0.0, magnitude = 0.0;
                for (std::uint32_t k = 0; k < columns; ++k) {
                    const double product =
                        side.weight.value(book, r, k) *
                        oracle::decode_e4m3fn(codes[static_cast<std::size_t>(t) * columns + k]);
                    sum += product;
                    magnitude += std::abs(product);
                }
                const double scale = static_cast<double>(side.weight.scales[r]) * token_scales[t];
                const double expected = sum * scale;
                require(std::isfinite(got) && std::isfinite(expected) && std::isfinite(magnitude),
                        label + " nonfinite paired output/oracle");
                worst = std::max(worst, std::abs(got - expected) /
                                            (std::abs(expected) / 128.0 +
                                             magnitude * scale * std::ldexp(1.0, -16) + 1e-30));
            }
        }
    }
    if (poison) {
        std::cout << label << ": poisoned token " << poisoned_token << " exact\n";
        return;
    }
    require(worst <= 1.0, label + " FP64 oracle bound exceeded: " + std::to_string(worst));
    std::cout << label << ": worst/bound " << worst << "\n";
}

// SiLU-pair epilogue: rows interleaved gate/up in 16-row tiles; output [T, rows / 2] against the
// FP64 silu(g) * u with the three BF16 roundings (g, u, output) and FP32 accumulation bounded.
void run_silu(const std::vector<double>& book, std::uint32_t rows, std::uint32_t columns,
              std::uint32_t tokens, Rng& rng, hipStream_t stream, bool every_token = false,
              bool graph_timing = false, bool poison = false) {
    const std::string label = std::string(poison ? "poisoned " : "") + "silu pair N" +
                              std::to_string(rows) + " T" + std::to_string(tokens);
    Weight weight(rows, columns, rng);
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns);
    for (auto& x : input) x = hip_bfloat16(static_cast<float>(rng.normal()));
    const std::uint32_t poisoned_token = tokens / 2U;
    if (poison) input[static_cast<std::size_t>(poisoned_token) * columns + 7U] = hip_bfloat16(NAN);
    Device<hip_bfloat16> device_input(input.size());
    device_input.upload(input);
    Activation activation(tokens, columns);
    if (graph_timing)
        check(hipMemsetAsync(activation.storage.data, 0xff, activation.storage.count, stream),
              "poison SiLU activation arena");
    check(linear::fp8_quantize_activation({device_input.data, activation.workspace}, stream), "q");
    const std::uint32_t features = rows / 2U;
    Device<hip_bfloat16> output(static_cast<std::size_t>(tokens) * features);
    check(hipMemsetAsync(output.data, 0xff, output.count * sizeof(hip_bfloat16), stream),
          "poison SiLU output");
    check(linear::fp8lut4_linear(weight.view(), activation.workspace,
                                 {.leading = output.data, .silu_pair = true}, stream),
          "silu");
    check(hipStreamSynchronize(stream), "sync");
    constexpr int graph_batch = 32;
    hipGraph_t graph          = nullptr;
    hipGraphExec_t executable = nullptr;
    if (graph_timing) {
        require(tokens == 1 && rows == 34816 && columns == 5120,
                "SiLU timing is restricted to the actual T1 gate/up cell");
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "capture SiLU");
        for (int i = 0; i < graph_batch; ++i)
            check(linear::fp8lut4_linear(weight.view(), activation.workspace,
                                         {.leading = output.data, .silu_pair = true}, stream),
                  "captured SiLU");
        check(hipStreamEndCapture(stream, &graph), "end SiLU capture");
        check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate SiLU");
        check(hipMemsetAsync(output.data, 0xff, output.count * sizeof(hip_bfloat16), stream),
              "poison SiLU output before qualified replay");
        check(hipGraphLaunch(executable, stream), "qualify SiLU replay");
        check(hipStreamSynchronize(stream), "sync SiLU replay");
    }
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> token_scales(tokens);
    check(hipMemcpy(codes.data(), activation.workspace.codes, codes.size(), hipMemcpyDeviceToHost),
          "c");
    check(hipMemcpy(token_scales.data(), activation.workspace.scales, tokens * 4U,
                    hipMemcpyDeviceToHost),
          "s");
    const auto got = output.download();
    // Every feature published; a poisoned run: exactly the flagged token's are the canonical NaN.
    for (std::size_t i = 0; i < got.size(); ++i) {
        const bool flagged = poison && i / features == poisoned_token;
        require(flagged ? got[i].data == 0x7FC0U : std::isfinite(bf16_value(got[i].data)),
                label + (flagged ? " per-token poisoning" : " unwritten/nonfinite output"));
    }
    if (poison) {
        std::cout << label << ": poisoned token " << poisoned_token << " exact\n";
        return;
    }
    std::vector<std::uint32_t> sample_tokens{0U, tokens / 2U, tokens - 1U};
    if (every_token) {
        sample_tokens.clear();
        for (std::uint32_t token = 0U; token < tokens; ++token) sample_tokens.push_back(token);
    }
    double worst = 0.0;
    for (std::uint32_t f = 0; f < features; f += 37U) {
        const std::uint32_t gate_row = (f / 8U) * 16U + f % 8U, up_row = gate_row + 8U;
        for (const std::uint32_t t : sample_tokens) {
            double g = 0.0, u = 0.0, gm = 0.0, um = 0.0;
            for (std::uint32_t k = 0; k < columns; ++k) {
                const double x =
                    oracle::decode_e4m3fn(codes[static_cast<std::size_t>(t) * columns + k]);
                const double pg = weight.value(book, gate_row, k) * x,
                             pu = weight.value(book, up_row, k) * x;
                g += pg;
                u += pu;
                gm += std::abs(pg);
                um += std::abs(pu);
            }
            const double sg = static_cast<double>(weight.scales[gate_row]) * token_scales[t];
            const double su = static_cast<double>(weight.scales[up_row]) * token_scales[t];
            g *= sg;
            u *= su;
            gm *= sg;
            um *= su;
            const double silu     = g / (1.0 + std::exp(-g));
            const double expected = silu * u;
            const double dg       = std::abs(g) / 256.0 + gm * std::ldexp(1.0, -16);
            const double du       = std::abs(u) / 256.0 + um * std::ldexp(1.0, -16);
            const double bound =
                1.5 * (1.1 * std::abs(u) * dg + std::abs(silu) * du + std::abs(expected) / 256.0) +
                1e-30;
            const double value = bf16_value(got[static_cast<std::size_t>(t) * features + f].data);
            require(std::isfinite(value) && std::isfinite(expected) && std::isfinite(bound),
                    label + " nonfinite SiLU output/oracle");
            worst = std::max(worst, std::abs(value - expected) / bound);
        }
    }
    require(worst <= 1.0, label + " FP64 oracle bound exceeded: " + std::to_string(worst));
    std::cout << label << ": worst/bound " << worst << "\n";
    if (graph_timing) {
        ninfer::r9700_qualification::device_guard().check("after-SiLU-oracle-before-timing");
        for (int i = 0; i < 3; ++i) check(hipGraphLaunch(executable, stream), "warm SiLU graph");
        check(hipStreamSynchronize(stream), "warm SiLU complete");
        hipEvent_t begin, end;
        check(hipEventCreate(&begin), "event");
        check(hipEventCreate(&end), "event");
        constexpr std::size_t trials = 7;
        constexpr int iterations     = 16;
        std::array<double, trials> samples{};
        for (auto& sample : samples) {
            check(hipEventRecord(begin, stream), "record");
            for (int i = 0; i < iterations; ++i)
                check(hipGraphLaunch(executable, stream), "timed SiLU graph");
            check(hipEventRecord(end, stream), "record");
            check(hipEventSynchronize(end), "sync");
            float ms = 0;
            check(hipEventElapsedTime(&ms, begin, end), "elapsed");
            sample = 1e3 * ms / (iterations * graph_batch);
        }
        check(hipEventDestroy(begin), "destroy event");
        check(hipEventDestroy(end), "destroy event");
        ninfer::r9700_qualification::device_guard().check("after-SiLU-timing");
        std::sort(samples.begin(), samples.end());
        std::cout << "complete_linear_silu N34816 K5120 T1 median_us=" << samples[trials / 2]
                  << " min_us=" << samples.front() << " max_us=" << samples.back()
                  << " trials=7 iterations=16 execution=graph ops_per_replay=32"
                     " activation_quantize_included=false linear_global_scratch_bytes=0\n";
        check(hipGraphExecDestroy(executable), "destroy SiLU executable");
        check(hipGraphDestroy(graph), "destroy SiLU graph");
    }
}

void time_shapes(const std::vector<double>& book, hipStream_t stream, Rng& rng) {
    struct Shape {
        std::uint32_t rows, columns;
    };

    const Shape shapes[] = {{34816, 5120}, {5120, 17408}, {12288, 5120},
                            {7168, 5120},  {5120, 6144},  {4096, 5120}};
    for (const std::uint32_t tokens : {1U, 6U, 2048U})
        for (const auto& shape : shapes)
            run_case(book, shape.rows, shape.columns, tokens, false, false, rng, stream, true);
}


} // namespace

int main(int argc, char** argv) {
    try {
        const bool silu_graph = argc == 2 && std::string(argv[1]) == "--time-silu-t1-graph";
        const bool timed      = argc == 2 && std::string(argv[1]) == "--time";
        const bool t1_only    = argc == 2 && std::string(argv[1]) == "--verify-t1";
        const bool graph_cell = argc == 5 && std::string(argv[1]) == "--time-cell-graph";
        const bool one_cell   = graph_cell || (argc == 5 && std::string(argv[1]) == "--time-cell");
        if (argc != 1 && !timed && !one_cell && !t1_only && !silu_graph)
            throw std::invalid_argument(
                "usage: fp8lut4_linear_qual [--time-silu-t1-graph | --verify-t1 | --time | "
                "--time-cell T N K | --time-cell-graph T N K]");
        std::uint32_t selected_tokens = 0, selected_rows = 0, selected_columns = 0;
        if (one_cell) {
            const auto tokens  = std::stoul(argv[2]);
            const auto rows    = std::stoul(argv[3]);
            const auto columns = std::stoul(argv[4]);
            require(tokens >= 1U && tokens <= 8192U, "time-cell T must be in 1..8192");
            require((rows == 34816U && columns == 5120U) || (rows == 5120U && columns == 17408U) ||
                        (rows == 12288U && columns == 5120U) ||
                        (rows == 7168U && columns == 5120U) ||
                        (rows == 5120U && columns == 6144U) || (rows == 4096U && columns == 5120U),
                    "time-cell N/K must select an existing timing projection");
            selected_tokens  = static_cast<std::uint32_t>(tokens);
            selected_rows    = static_cast<std::uint32_t>(rows);
            selected_columns = static_cast<std::uint32_t>(columns);
        }
        ninfer::r9700_qualification::device_guard().select();

        std::cout << "physical_packing=production\n";

        hipStream_t stream = nullptr;
        check(hipStreamCreate(&stream), "stream");
        Rng rng{0xcb4cb4ULL};
        const auto book = reference_codebook();
        check_table(book);
        if (silu_graph) {
            run_silu(book, 34816, 5120, 1, rng, stream, true, true);
            check(hipStreamDestroy(stream), "destroy stream");
            return 0;
        }
        if (t1_only) {
            run_pair(book, 7168, 6144, 7168, 5120, 1, rng, stream);
            run_silu(book, 34816, 5120, 1, rng, stream, true);
            run_case(book, 5120, 6144, 1, true, false, rng, stream);
            run_case(book, 5120, 17408, 1, true, false, rng, stream);
            run_case(book, 7168, 5120, 1, false, true, rng, stream);
            ninfer::r9700_qualification::device_guard().check("after-T1-linear-oracles");
            check(hipStreamDestroy(stream), "destroy stream");
            std::cout << "fp8lut4 T1 plain/pair/SiLU/residual/poison qualification: PASS\n";
            return 0;
        }
        if (one_cell) {
            // Both executables start from the same seed and unmodified logical host codes.
            // This fixture's independent oracle always runs before its timing window.
            run_case(book, selected_rows, selected_columns, selected_tokens, false, false, rng,
                     stream, true, graph_cell);
            check(hipStreamDestroy(stream), "destroy stream");
            std::cout << "fp8lut4 linear selected-cell qualification: PASS\n";
            return 0;
        }

        struct Shape {
            std::uint32_t rows, columns;
        };

        const Shape shapes[] = {{34816, 5120}, {5120, 17408}, {12288, 5120}, {7168, 5120},
                                {5120, 6144},  {4096, 5120},  {1280, 5120},  {5120, 25600}};
        for (const auto& shape : shapes)
            for (const std::uint32_t tokens :
                 {1U,  5U,  6U,  16U, 17U, 20U, 24U, 32U, 33U,  40U, 48U,
                  49U, 56U, 57U, 64U, 65U, 80U, 96U, 97U, 113U, 128U})
                run_case(book, shape.rows, shape.columns, tokens, false, false, rng, stream);
        // FP8LUT4 output head (N248320/K5120): C1..C8 verification widths up to T64.
        for (const std::uint32_t tokens : {1U, 6U, 16U, 24U, 32U, 40U, 49U, 64U})
            run_case(book, 248320, 5120, tokens, false, false, rng, stream);
        // Prefill CTA shapes: 256x64 (T130/200 at N<12288), 512x64 (T300/1000/2048), 256x128
        // (T640, and T <= 256 at N >= 12288), with partial token tiles.
        for (const std::uint32_t tokens : {17U, 130U, 200U, 300U, 640U, 1000U, 2048U}) {
            run_case(book, 5120, 17408, tokens, false, false, rng, stream);
            run_case(book, 7168, 5120, tokens, false, false, rng, stream);
        }
        for (const std::uint32_t tokens :
             {1U,  6U,  16U,  24U,  32U,  33U,  40U,  49U,  56U,  64U,
              65U, 96U, 100U, 113U, 128U, 130U, 200U, 300U, 640U, 2048U})
            run_silu(book, 34816, 5120, tokens, rng, stream);
        run_pair(book, 7168, 6144, 7168, 5120, 6, rng, stream);
        run_pair(book, 7168, 6144, 7168, 5120, 1, rng, stream);
        run_pair(book, 12288, 6144, 4096, 5120, 5, rng, stream);
        run_pair(book, 7168, 6144, 7168, 5120, 24, rng, stream);
        run_pair(book, 12288, 6144, 4096, 5120, 32, rng, stream);
        // T 33..64 fused deep pairs in their production layouts: attention query-key (query/key
        // split) + gate-value (gate/value split), GDN query-key + value-z (value/z split).
        run_pair(book, 7168, 6144, 7168, 5120, 33, rng, stream, 6144);
        run_pair(book, 7168, 6144, 7168, 5120, 49, rng, stream, 6144);
        run_pair(book, 7168, 6144, 7168, 5120, 64, rng, stream, 6144);
        run_pair(book, 4096, 0, 12288, 5120, 40, rng, stream, 6144);
        run_pair(book, 4096, 0, 12288, 5120, 56, rng, stream, 6144);
        run_pair(book, 12288, 6144, 4096, 5120, 64, rng, stream);
        // GDN conv-record fallback layout: query-key + value-z, both unsplit, production order.
        run_pair(book, 4096, 0, 12288, 5120, 33, rng, stream);
        run_pair(book, 4096, 0, 12288, 5120, 64, rng, stream);
        // T 65..128 deep pairs: attention 4 x 2 depth 2, GDN 8 x 1 depth 4 (five to eight tiles).
        run_pair(book, 7168, 6144, 7168, 5120, 72, rng, stream, 6144);
        run_pair(book, 7168, 6144, 7168, 5120, 128, rng, stream, 6144);
        run_pair(book, 4096, 0, 12288, 5120, 65, rng, stream, 6144);
        run_pair(book, 4096, 0, 12288, 5120, 90, rng, stream, 6144);
        run_pair(book, 4096, 0, 12288, 5120, 100, rng, stream, 6144);
        run_pair(book, 4096, 0, 12288, 5120, 128, rng, stream);
        // Poisoned tokens through the fused deep pairs.
        run_pair(book, 7168, 6144, 7168, 5120, 49, rng, stream, 6144, true);
        run_pair(book, 4096, 0, 12288, 5120, 40, rng, stream, 6144, true);
        run_pair(book, 7168, 6144, 7168, 5120, 97, rng, stream, 6144, true);
        run_pair(book, 4096, 0, 12288, 5120, 113, rng, stream, 6144, true);
        // Poisoned tokens through the SiLU epilogue: mid-T (T40) and deep (T49, T64, T65, T128).
        for (const std::uint32_t tokens : {40U, 49U, 64U, 65U, 128U})
            run_silu(book, 34816, 5120, tokens, rng, stream, false, false, true);
        run_pair(book, 7168, 6144, 7168, 5120, 71, rng, stream);
        run_pair(book, 7168, 6144, 7168, 5120, 130, rng, stream);
        run_pair(book, 12288, 6144, 4096, 5120, 200, rng, stream);
        run_pair(book, 12288, 6144, 4096, 5120, 300, rng, stream);
        run_case(book, 5120, 6144, 1, true, false, rng, stream);
        run_case(book, 5120, 17408, 1, true, false, rng, stream);
        run_case(book, 5120, 6144, 6, true, false, rng, stream);
        run_case(book, 5120, 17408, 24, true, false, rng, stream);
        run_case(book, 5120, 17408, 56, true, false, rng, stream);
        run_case(book, 5120, 6144, 64, true, false, rng, stream);
        run_case(book, 5120, 6144, 40, true, false, rng, stream); // three token tiles
        run_case(book, 5120, 17408, 49, true, true, rng, stream); // residual, poisoned token
        run_case(book, 5120, 6144, 33, true, true, rng, stream);
        run_case(book, 5120, 17408, 80, true, false, rng, stream);
        run_case(book, 5120, 6144, 128, true, false, rng, stream);
        run_case(book, 5120, 17408, 113, true, true, rng, stream); // residual, poisoned token
        run_case(book, 5120, 6144, 72, false, true, rng, stream);
        run_case(book, 7168, 5120, 50, false, true, rng, stream);
        run_case(book, 5120, 6144, 2048, true, false, rng, stream);
        // Prefill GEMM token spans (2048 tokens per wide launch, 4096 narrow): three wide spans
        // plain and SiLU, a poisoned token in the second span, a two-span narrow residual, and a
        // split pair output.
        run_case(book, 34816, 5120, 4100, false, false, rng, stream);
        run_silu(book, 34816, 5120, 4100, rng, stream);
        run_case(book, 12288, 5120, 5000, false, true, rng, stream);
        run_case(book, 5120, 17408, 4500, true, false, rng, stream);
        run_pair(book, 12288, 6144, 4096, 5120, 2200, rng, stream);
        run_case(book, 5120, 17408, 200, true, false, rng, stream);
        run_case(book, 5120, 6144, 300, true, false, rng, stream);
        run_case(book, 7168, 5120, 300, false, true, rng, stream);
        run_case(book, 7168, 5120, 1, false, true, rng, stream);
        run_case(book, 7168, 5120, 6, false, true, rng, stream);
        run_case(book, 7168, 5120, 28, false, true, rng, stream);
        run_case(book, 7168, 5120, 40, false, true, rng, stream);
        run_case(book, 7168, 5120, 64, false, true, rng, stream);
        run_case(book, 5120, 6144, 40, false, true, rng, stream);
        run_case(book, 5120, 17408, 64, false, true, rng, stream);
        run_case(book, 248320, 5120, 56, false, true, rng, stream);
        run_case(book, 7168, 5120, 300, false, true, rng, stream);
        ninfer::r9700_qualification::device_guard().check("after-linear-suite");
        if (timed) time_shapes(book, stream, rng);
        check(hipStreamDestroy(stream), "destroy stream");
        std::cout << "fp8lut4 linear qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
