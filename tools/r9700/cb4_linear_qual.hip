// Qualification of the CB4G32 x per-token E4M3 Linear (cb4_linear) against an FP64 oracle that
// decodes the stored codes through an independently rounded codebook: every group code's table,
// small-T and prefill widths on the real Qwen3.8 projection shapes, the residual epilogue and the
// poisoned-activation contract. Optional timing (--time) on the same shapes.
#include "ops/r9700/linear/cb4_linear.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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
            const float exact = static_cast<float>(std::ldexp(kBase[j] * (8 + m), exponent - 7));
            book[code * 8 + j] = oracle::decode_e4m3fn(oracle::encode_e4m3fn(exact));
        }
    }
    return book;
}

void check_table(const std::vector<double>& book) {
    const auto& table = linear::cb4_magnitude_table();
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
            const std::uint32_t exponent = (r & 63U) == 0U ? (r >> 8U) % 32U : 18U + (r >> 8U) % 14U;
            g = static_cast<std::uint8_t>((exponent << 3U) | (r >> 16U) % 8U);
        }
        for (auto& s : scales) s = static_cast<float>(std::ldexp(1.0 + rng.uniform(), -9));
        device_codes.upload(codes);
        device_groups.upload(groups);
        device_scales.upload(scales);
    }
    linear::Cb4Weight view() const {
        return {device_codes.data, device_groups.data, device_scales.data, rows, columns};
    }
    // r9700-cb4g32-n16k64-v1: tile (r / 16) * (K / 64) + k / 64, slot 16 * ((k % 64) / 32) + r % 16.
    double value(const std::vector<double>& book, std::uint32_t r, std::uint32_t k) const {
        const std::size_t tile = static_cast<std::size_t>(r / 16U) * (columns / 64U) + k / 64U;
        const std::uint32_t slot = 16U * ((k % 64U) / 32U) + r % 16U;
        const std::uint8_t byte = codes[tile * 512U + slot * 16U + (k % 32U) / 2U];
        const std::uint32_t code = (k & 1U) != 0U ? byte >> 4U : byte & 15U;
        const double magnitude = book[groups[tile * 32U + slot] * 8U + (code & 7U)];
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
                                                    &workspace), "bind");
    }
};

void run_case(const std::vector<double>& book, std::uint32_t rows, std::uint32_t columns,
              std::uint32_t tokens, bool accumulate, bool poison, Rng& rng, hipStream_t stream) {
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
    check(linear::fp8_quantize_activation({device_input.data, activation.workspace}, stream), "quantize");
    std::vector<hip_bfloat16> base(static_cast<std::size_t>(tokens) * rows);
    for (auto& b : base) b = hip_bfloat16(static_cast<float>(rng.normal()));
    Device<hip_bfloat16> output(base.size());
    output.upload(base);
    check(linear::cb4_linear(weight.view(), activation.workspace,
                             {.leading = output.data, .accumulate = accumulate}, stream), "cb4");
    check(hipStreamSynchronize(stream), "sync");
    const auto got = output.download();
    if (poison) {
        // The nonfinite element belongs to token poisoned_token; exactly its outputs are NaN.
        const std::uint32_t poisoned_token =
            static_cast<std::uint32_t>((input.size() / 2U) / columns);
        for (std::uint32_t t = 0; t < tokens; ++t)
            for (std::uint32_t r = 0; r < rows; ++r)
                require(std::isnan(static_cast<float>(got[static_cast<std::size_t>(t) * rows + r])) ==
                            (t == poisoned_token),
                        label + " per-token poisoning");
        return;
    }
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> token_scales(tokens);
    check(hipMemcpy(codes.data(), activation.workspace.codes, codes.size(), hipMemcpyDeviceToHost), "codes");
    check(hipMemcpy(token_scales.data(), activation.workspace.scales, tokens * sizeof(float),
                    hipMemcpyDeviceToHost), "scales");
    std::vector<std::uint32_t> sample_tokens;
    for (std::uint32_t t = 0; t < tokens; t += tokens > 64U ? 131U : 1U) sample_tokens.push_back(t);
    sample_tokens.push_back(tokens - 1U);
    const std::uint32_t row_stride = rows > 8192U ? 97U : 13U;
    double worst = 0.0;
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
            const double scale = static_cast<double>(weight.scales[r]) * token_scales[t];
            double expected = sum * scale;
            const std::size_t index = static_cast<std::size_t>(t) * rows + r;
            const double prior = accumulate ? bf16_value(base[index].data) : 0.0;
            expected += prior;
            const double error = std::abs(bf16_value(got[index].data) - expected);
            // BF16 output rounding(s) plus FP32 accumulation of the K products.
            const double bound = (std::abs(expected) + std::abs(prior)) / 128.0 +
                                 magnitude * scale * std::ldexp(1.0, -16) + 1e-30;
            worst = std::max(worst, error / bound);
        }
    }
    require(worst <= 1.0, label + " FP64 oracle bound exceeded: " + std::to_string(worst));
    std::cout << label << ": worst/bound " << worst << "\n";
}

// Two projections of one image in one launch; the first publishes rows [0, split) and [split, N)
// to separate outputs. Both are checked against the oracle through the split.
void run_pair(const std::vector<double>& book, std::uint32_t rows0, std::uint32_t split0,
              std::uint32_t rows1, std::uint32_t columns, std::uint32_t tokens, Rng& rng,
              hipStream_t stream) {
    const std::string label = "pair N" + std::to_string(rows0) + "/" + std::to_string(split0) +
                              "+N" + std::to_string(rows1) + " T" + std::to_string(tokens);
    Weight first(rows0, columns, rng), second(rows1, columns, rng);
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns);
    for (auto& x : input) x = hip_bfloat16(static_cast<float>(rng.normal()));
    Device<hip_bfloat16> device_input(input.size());
    device_input.upload(input);
    Activation activation(tokens, columns);
    check(linear::fp8_quantize_activation({device_input.data, activation.workspace}, stream), "q");
    Device<hip_bfloat16> leading(static_cast<std::size_t>(tokens) * split0),
        trailing(static_cast<std::size_t>(tokens) * (rows0 - split0)),
        other(static_cast<std::size_t>(tokens) * rows1);
    check(linear::cb4_linear_pair(first.view(), {leading.data, trailing.data, split0, false},
                                  second.view(), {.leading = other.data}, activation.workspace,
                                  stream), "pair");
    check(hipStreamSynchronize(stream), "sync");
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> token_scales(tokens);
    check(hipMemcpy(codes.data(), activation.workspace.codes, codes.size(), hipMemcpyDeviceToHost), "c");
    check(hipMemcpy(token_scales.data(), activation.workspace.scales, tokens * 4U, hipMemcpyDeviceToHost), "s");
    const auto lead = leading.download(), trail = trailing.download(), rest = other.download();
    double worst = 0.0;
    const auto verify = [&](const Weight& weight, std::uint32_t r, double got, std::uint32_t t) {
        double sum = 0.0, magnitude = 0.0;
        for (std::uint32_t k = 0; k < columns; ++k) {
            const double product = weight.value(book, r, k) *
                oracle::decode_e4m3fn(codes[static_cast<std::size_t>(t) * columns + k]);
            sum += product;
            magnitude += std::abs(product);
        }
        const double scale = static_cast<double>(weight.scales[r]) * token_scales[t];
        const double expected = sum * scale;
        worst = std::max(worst, std::abs(got - expected) /
                                    (std::abs(expected) / 128.0 + magnitude * scale * std::ldexp(1.0, -16) + 1e-30));
    };
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t r = 0; r < rows0; r += 29U) {
            const double got = r < split0
                ? bf16_value(lead[static_cast<std::size_t>(t) * split0 + r].data)
                : bf16_value(trail[static_cast<std::size_t>(t) * (rows0 - split0) + r - split0].data);
            verify(first, r, got, t);
        }
        for (std::uint32_t r = 0; r < rows1; r += 31U)
            verify(second, r, bf16_value(rest[static_cast<std::size_t>(t) * rows1 + r].data), t);
    }
    require(worst <= 1.0, label + " FP64 oracle bound exceeded: " + std::to_string(worst));
    std::cout << label << ": worst/bound " << worst << "\n";
}

// SiLU-pair epilogue: rows interleaved gate/up in 16-row tiles; output [T, rows / 2] against the
// FP64 silu(g) * u with the three BF16 roundings (g, u, output) and FP32 accumulation bounded.
void run_silu(const std::vector<double>& book, std::uint32_t rows, std::uint32_t columns,
              std::uint32_t tokens, Rng& rng, hipStream_t stream) {
    const std::string label = "silu pair N" + std::to_string(rows) + " T" + std::to_string(tokens);
    Weight weight(rows, columns, rng);
    std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * columns);
    for (auto& x : input) x = hip_bfloat16(static_cast<float>(rng.normal()));
    Device<hip_bfloat16> device_input(input.size());
    device_input.upload(input);
    Activation activation(tokens, columns);
    check(linear::fp8_quantize_activation({device_input.data, activation.workspace}, stream), "q");
    const std::uint32_t features = rows / 2U;
    Device<hip_bfloat16> output(static_cast<std::size_t>(tokens) * features);
    check(linear::cb4_linear(weight.view(), activation.workspace,
                             {.leading = output.data, .silu_pair = true}, stream), "silu");
    check(hipStreamSynchronize(stream), "sync");
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(tokens) * columns);
    std::vector<float> token_scales(tokens);
    check(hipMemcpy(codes.data(), activation.workspace.codes, codes.size(), hipMemcpyDeviceToHost), "c");
    check(hipMemcpy(token_scales.data(), activation.workspace.scales, tokens * 4U, hipMemcpyDeviceToHost), "s");
    const auto got = output.download();
    std::vector<std::uint32_t> sample_tokens{0U, tokens / 2U, tokens - 1U};
    double worst = 0.0;
    for (std::uint32_t f = 0; f < features; f += 37U) {
        const std::uint32_t gate_row = (f / 8U) * 16U + f % 8U, up_row = gate_row + 8U;
        for (const std::uint32_t t : sample_tokens) {
            double g = 0.0, u = 0.0, gm = 0.0, um = 0.0;
            for (std::uint32_t k = 0; k < columns; ++k) {
                const double x = oracle::decode_e4m3fn(codes[static_cast<std::size_t>(t) * columns + k]);
                const double pg = weight.value(book, gate_row, k) * x, pu = weight.value(book, up_row, k) * x;
                g += pg; u += pu; gm += std::abs(pg); um += std::abs(pu);
            }
            const double sg = static_cast<double>(weight.scales[gate_row]) * token_scales[t];
            const double su = static_cast<double>(weight.scales[up_row]) * token_scales[t];
            g *= sg; u *= su; gm *= sg; um *= su;
            const double silu = g / (1.0 + std::exp(-g));
            const double expected = silu * u;
            const double dg = std::abs(g) / 256.0 + gm * std::ldexp(1.0, -16);
            const double du = std::abs(u) / 256.0 + um * std::ldexp(1.0, -16);
            const double bound = 1.5 * (1.1 * std::abs(u) * dg + std::abs(silu) * du +
                                        std::abs(expected) / 256.0) + 1e-30;
            const double value = bf16_value(got[static_cast<std::size_t>(t) * features + f].data);
            worst = std::max(worst, std::abs(value - expected) / bound);
        }
    }
    require(worst <= 1.0, label + " FP64 oracle bound exceeded: " + std::to_string(worst));
    std::cout << label << ": worst/bound " << worst << "\n";
}

void time_shapes(hipStream_t stream, Rng& rng) {
    struct Shape { std::uint32_t rows, columns; const char* name; };
    const Shape shapes[] = {{34816, 5120, "gate/up"}, {5120, 17408, "down"}, {12288, 5120, "gdn v/z"},
                            {7168, 5120, "attention"}, {5120, 6144, "output"}, {4096, 5120, "gdn q/k"}};
    for (const std::uint32_t tokens : {1U, 6U, 2048U}) {
        for (const auto& shape : shapes) {
            Weight weight(shape.rows, shape.columns, rng);
            Activation activation(tokens, shape.columns);
            Device<hip_bfloat16> input(static_cast<std::size_t>(tokens) * shape.columns);
            check(hipMemset(input.data, 0, input.count * sizeof(hip_bfloat16)), "input");
            check(linear::fp8_quantize_activation({input.data, activation.workspace}, stream), "q");
            Device<hip_bfloat16> output(static_cast<std::size_t>(tokens) * shape.rows);
            const auto launch = [&] {
                check(linear::cb4_linear(weight.view(), activation.workspace,
                                         {.leading = output.data}, stream), "cb4");
            };
            for (int i = 0; i < 10; ++i) launch();
            hipEvent_t begin, end;
            check(hipEventCreate(&begin), "event");
            check(hipEventCreate(&end), "event");
            const int iterations = tokens > 16U ? 20 : 200;
            check(hipEventRecord(begin, stream), "record");
            for (int i = 0; i < iterations; ++i) launch();
            check(hipEventRecord(end, stream), "record");
            check(hipEventSynchronize(end), "sync");
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, begin, end), "elapsed");
            const double us = 1e3 * ms / iterations;
            const double flops = 2.0 * tokens * shape.rows * static_cast<double>(shape.columns);
            const double bytes = shape.rows * (shape.columns * (0.5 + 1.0 / 32.0) + 4.0);
            std::cout << shape.name << " T" << tokens << ": " << us << " us, "
                      << (tokens > 16U ? flops / us / 1e6 : bytes / us / 1e3)
                      << (tokens > 16U ? " TFLOP/s\n" : " GB/s\n");
            (void)hipEventDestroy(begin);
            (void)hipEventDestroy(end);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        hipStream_t stream = nullptr;
        check(hipStreamCreate(&stream), "stream");
        Rng rng{0xcb4cb4ULL};
        if (argc > 1 && std::string(argv[1]) == "--time") {
            time_shapes(stream, rng);
            return 0;
        }
        const auto book = reference_codebook();
        check_table(book);
        struct Shape { std::uint32_t rows, columns; };
        const Shape shapes[] = {{34816, 5120}, {5120, 17408}, {12288, 5120}, {7168, 5120},
                                {5120, 6144}, {4096, 5120}, {1280, 5120}, {5120, 25600}};
        for (const auto& shape : shapes)
            for (const std::uint32_t tokens : {1U, 5U, 6U, 16U})
                run_case(book, shape.rows, shape.columns, tokens, false, false, rng, stream);
        for (const std::uint32_t tokens : {17U, 130U, 2048U}) {
            run_case(book, 5120, 17408, tokens, false, false, rng, stream);
            run_case(book, 7168, 5120, tokens, false, false, rng, stream);
        }
        for (const std::uint32_t tokens : {1U, 6U, 16U, 130U, 2048U})
            run_silu(book, 34816, 5120, tokens, rng, stream);
        run_pair(book, 7168, 6144, 7168, 5120, 6, rng, stream);
        run_pair(book, 7168, 6144, 7168, 5120, 1, rng, stream);
        run_pair(book, 12288, 6144, 4096, 5120, 5, rng, stream);
        run_pair(book, 7168, 6144, 7168, 5120, 130, rng, stream);
        run_case(book, 5120, 6144, 1, true, false, rng, stream);
        run_case(book, 5120, 17408, 1, true, false, rng, stream);
        run_case(book, 5120, 6144, 6, true, false, rng, stream);
        run_case(book, 5120, 6144, 2048, true, false, rng, stream);
        run_case(book, 7168, 5120, 1, false, true, rng, stream);
        run_case(book, 7168, 5120, 6, false, true, rng, stream);
        run_case(book, 7168, 5120, 300, false, true, rng, stream);
        std::cout << "cb4 linear qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
