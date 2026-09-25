// Qualification of the A8 prefill normalized projections: the public normalized Linear (RMSNorm
// straight into the A8G64 codec, then the M128xN128 Q4 GEMM) on the Qwen3.8 gate/up boundary
// K5120 -> N34816, and the GDN output route (per-head gated RMSNorm into the codec) K6144 -> N5120.
// Evidence: an independent FP64 oracle of the complete represented formula at sampled tokens;
// an FP64 RMSNorm + codec check of every token's activation planes; exact equality with the
// production A8Q4 GEMM on the same planes; rejection of a misaligned prefill input; and timing
// against the composition it replaces (public RMSNorm with a BF16 intermediate, then Linear).
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/normalized_linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace lin = ninfer::ops::r9700::linear;

constexpr std::uint32_t kColumns = 5120;
constexpr std::uint32_t kRows = 34816;
constexpr std::uint32_t kGroup = 64;
constexpr float kEps = 1e-6F;

void check(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
    }
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
    void put(const std::vector<T>& host) {
        require(host.size() == count, "upload size");
        check(hipMemcpy(data, host.data(), count * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> get() const {
        std::vector<T> host(count);
        check(hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return host;
    }
    std::size_t bytes() const { return count * sizeof(T); }
};

struct Rng {
    std::uint64_t state;
    std::uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 32U);
    }
    double uniform() { return (next() + 0.5) / 4294967296.0; }
};

std::uint16_t half_bits(float value) { return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value)); }
double half_value(std::uint16_t bits) { return static_cast<double>(std::bit_cast<_Float16>(bits)); }

// Independent BF16 round-to-nearest-even of a double (normal range).
double bf16_rne(double x) {
    if (x == 0 || !std::isfinite(x)) return x;
    int exponent = 0;
    std::frexp(std::abs(x), &exponent);
    const double quantum = std::ldexp(1.0, std::max(exponent - 8, -133));
    return std::nearbyint(x / quantum) * quantum;
}

// A8G64 codec of one represented row: scale = FP16(amax/127) (minimum one FP16 ulp when
// amax>0), code = clamp(RNE(x/scale), +-127) with the FP32 quotient.
std::vector<double> a8g64(const std::vector<double>& row) {
    std::vector<double> decoded(row.size());
    for (std::size_t group = 0; group < row.size() / kGroup; ++group) {
        float maximum = 0.0F;
        for (std::uint32_t i = 0; i < kGroup; ++i)
            maximum = std::max(maximum, std::abs(static_cast<float>(row[group * kGroup + i])));
        std::uint16_t bits = maximum == 0.0F ? 0U : half_bits(maximum / 127.0F);
        if (maximum != 0.0F && bits == 0U) bits = 1U;
        const float scale = static_cast<float>(half_value(bits));
        for (std::uint32_t i = 0; i < kGroup; ++i) {
            const float value = static_cast<float>(row[group * kGroup + i]);
            const int code = bits == 0U ? 0 : std::clamp(
                static_cast<int>(std::nearbyint(value / scale)), -127, 127);
            decoded[group * kGroup + i] = code * static_cast<double>(scale);
        }
    }
    return decoded;
}

double weight_at(const std::vector<std::uint8_t>& codes, const std::vector<std::uint16_t>& scales,
                 std::uint32_t row, std::uint32_t column, std::uint32_t columns = kColumns) {
    const std::uint32_t groups = columns / kGroup;
    const std::uint32_t group = column / kGroup, within = column % kGroup;
    const std::size_t pair =
        (((static_cast<std::size_t>(row >> 4U) * groups + group) * 4U + within / 16U) * 16U +
         (row & 15U));
    std::uint64_t word = 0;
    std::memcpy(&word, codes.data() + pair * 8U, 8U);
    const unsigned nibble = static_cast<unsigned>(word >> ((within & 15U) * 4U)) & 15U;
    const int code = nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
    const std::size_t scale =
        (static_cast<std::size_t>(row >> 4U) * groups + group) * 16U + (row & 15U);
    return code * half_value(scales[scale]);
}

// Warmed-up, interleaved A/B timing (the GPU clock state drifts after idle host work); returns
// the medians of the two series.
std::pair<float, float> time_ab(hipStream_t stream, const auto& first, const auto& second) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "event");
    check(hipEventCreate(&end), "event");
    const auto once = [&](const auto& body) {
        check(hipEventRecord(begin, stream), "record");
        body();
        check(hipEventRecord(end, stream), "record");
        check(hipEventSynchronize(end), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, begin, end), "elapsed");
        return ms;
    };
    for (int warm = 0; warm < 40; ++warm) { (void)once(first); (void)once(second); }
    std::vector<float> a, b;
    for (int repeat = 0; repeat < 40; ++repeat) {
        a.push_back(once(first));
        b.push_back(once(second));
    }
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return {a[a.size() / 2], b[b.size() / 2]};
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--out-json",
                "usage: a8q4_normalized_linear_prefill_qual --out-json FRESH.json");
        const std::filesystem::path out = argv[2];
        require(!std::filesystem::exists(out), "output exists");
        constexpr std::uint32_t tokens = 2048;
        require(lin::a8q4g64_normalized_linear_prefill_supported(tokens),
                "T2048 is outside the prefill route");
        hipStream_t stream{};
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");

        Rng rng{0x5eedULL};
        std::vector<std::uint8_t> codes(static_cast<std::size_t>(kRows) * kColumns / 2U);
        std::vector<std::uint16_t> scales(static_cast<std::size_t>(kRows) * kColumns / kGroup);
        for (auto& byte : codes) byte = static_cast<std::uint8_t>(rng.next());
        for (auto& scale : scales) scale = half_bits(static_cast<float>(0.004 * (0.5 + rng.uniform())));
        std::vector<hip_bfloat16> norm(kColumns);
        for (auto& gain : norm) gain = hip_bfloat16((rng.uniform() - 0.3) * 0.8);
        std::vector<hip_bfloat16> input(static_cast<std::size_t>(tokens) * kColumns);
        for (std::uint32_t token = 0; token < tokens; ++token) {
            const double amplitude = 0.05 + 8.0 * rng.uniform();  // per-token RMS spread
            for (std::uint32_t k = 0; k < kColumns; ++k) {
                double value = (rng.uniform() - 0.5) * amplitude;
                if (rng.next() % 97U == 0U) value *= 12.0;  // per-group outliers
                if (token % 509U == 7U && k % 64U < 8U) value = 0.0;
                input[static_cast<std::size_t>(token) * kColumns + k] = hip_bfloat16(value);
            }
        }

        Device<hip_bfloat16> d_input(input.size()), d_norm(norm.size()),
            d_output(static_cast<std::size_t>(tokens) * kRows),
            d_control(static_cast<std::size_t>(tokens) * kRows);
        Device<std::uint8_t> d_codes(codes.size());
        Device<std::uint16_t> d_scales(scales.size());
        d_input.put(input);
        d_norm.put(norm);
        d_codes.put(codes);
        d_scales.put(scales);

        ninfer::Weight weight{};
        weight.qtype = ninfer::QType::Q4G64_F16S;
        weight.layout = ninfer::QuantLayout::Q4N16K16;
        weight.ndim = 2;
        weight.n = weight.shape[0] = weight.padded_shape[0] = kRows;
        weight.k = weight.shape[1] = weight.padded_shape[1] = kColumns;
        weight.group = weight.group_size = kGroup;
        weight.scale_dtype = ninfer::DType::FP16;
        weight.qdata = d_codes.data;
        weight.qdata_bytes = d_codes.bytes();
        weight.scales = d_scales.data;
        weight.scale_bytes = d_scales.bytes();
        weight.payload = d_codes.data;
        weight.payload_bytes = d_codes.bytes();

        const ninfer::Tensor x(d_input.data, ninfer::DType::BF16,
                               {static_cast<int>(kColumns), static_cast<int>(tokens)});
        const ninfer::Tensor gains(d_norm.data, ninfer::DType::BF16, {static_cast<int>(kColumns)});
        ninfer::Tensor y(d_output.data, ninfer::DType::BF16,
                         {static_cast<int>(kRows), static_cast<int>(tokens)});
        const std::size_t workspace_bytes =
            ninfer::ops::normalized_linear_workspace_capacity_bytes(tokens, kColumns, kRows);
        require(workspace_bytes == lin::a8q4g64_activation_workspace_capacity_bytes(tokens, kColumns),
                "prefill route still reserves a BF16 intermediate");
        Device<std::uint8_t> d_workspace(workspace_bytes);
        const auto route = [&] {
            ninfer::ops::normalized_linear(x, gains, kEps, true, weight, y,
                                           {d_workspace.data, workspace_bytes}, stream);
        };
        route();
        check(hipStreamSynchronize(stream), "sync");
        const std::vector<hip_bfloat16> output = d_output.get();

        // Misaligned prefill input is rejected, not silently read with vector loads.
        {
            const ninfer::Tensor misaligned(reinterpret_cast<std::uint8_t*>(d_input.data) + 2,
                                            ninfer::DType::BF16,
                                            {static_cast<int>(kColumns), 128});
            ninfer::Tensor small(d_control.data, ninfer::DType::BF16,
                                 {static_cast<int>(kRows), 128});
            const std::size_t small_bytes =
                ninfer::ops::normalized_linear_workspace_capacity_bytes(128, kColumns, kRows);
            bool rejected = false;
            if (lin::a8q4g64_normalized_linear_prefill_supported(128U)) {
                try {
                    ninfer::ops::normalized_linear(misaligned, gains, kEps, true, weight, small,
                                                   {d_workspace.data, small_bytes}, stream);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
            } else {
                rejected = true;  // T128 composes; nothing to reject.
            }
            require(rejected, "misaligned prefill input accepted");
        }

        // Exact equality with the production A8Q4 GEMM on the route's own planes.
        lin::A8G64ActivationWorkspace planes{};
        check(lin::a8q4g64_bind_activation_workspace(d_workspace.data, workspace_bytes, tokens,
                                                     kColumns, &planes),
              "bind planes");
        check(lin::a8q4g64_linear_prefill_cta(
                  {planes.low_codes, planes.low_code_bytes, planes.high_codes,
                   planes.high_code_bytes, planes.scales, planes.scale_bytes, planes.status,
                   d_codes.data, d_codes.bytes(), d_scales.data, d_scales.bytes(),
                   d_control.data, tokens, kRows, kColumns, kColumns},
                  stream),
              "control GEMM");
        check(hipStreamSynchronize(stream), "sync");
        require(std::memcmp(output.data(), d_control.get().data(),
                            output.size() * sizeof(hip_bfloat16)) == 0,
                "output differs from the production GEMM on the same planes");

        // FP64 RMSNorm with the explicit BF16 seam: n = BF16(x / sqrt(mean(x^2) + eps) * (1 + w)).
        const auto normalized_row = [&](std::uint32_t token) {
            std::vector<double> n(kColumns);
            double sum = 0;
            for (std::uint32_t k = 0; k < kColumns; ++k) {
                const double v = static_cast<float>(input[static_cast<std::size_t>(token) * kColumns + k]);
                sum += v * v;
            }
            const double inverse = 1.0 / std::sqrt(sum / kColumns + kEps);
            for (std::uint32_t k = 0; k < kColumns; ++k) {
                const double v = static_cast<float>(input[static_cast<std::size_t>(token) * kColumns + k]);
                n[k] = bf16_rne(v * inverse * (1.0 + static_cast<float>(norm[k])));
            }
            return n;
        };
        // Every token's planes decode within half a code step (plus one BF16 step, at most
        // |n|*2^-7, for a flipped normalization seam) of the FP64 normalized row.
        const std::vector<std::uint8_t> workspace = d_workspace.get();
        const auto plane = [&](const void* region) {
            return workspace.data() + (static_cast<const std::uint8_t*>(region) - d_workspace.data);
        };
        const std::uint8_t* low_plane = plane(planes.low_codes);
        const std::uint8_t* high_plane = plane(planes.high_codes);
        const auto* scale_plane = reinterpret_cast<const std::uint16_t*>(plane(planes.scales));
        std::uint32_t status = 0;
        std::memcpy(&status, plane(planes.status), sizeof(status));
        require(status == 0U, "prefill route reported activation status");
        double worst_plane = 0.0;
        for (std::uint32_t token = 0; token < tokens; ++token) {
            const std::vector<double> n = normalized_row(token);
            for (std::uint32_t k = 0; k < kColumns; ++k) {
                const std::size_t index = static_cast<std::size_t>(token) * kColumns + k;
                const unsigned low = (low_plane[index / 2] >> ((index & 1U) * 4U)) & 15U;
                const unsigned high_nibble = (high_plane[index / 2] >> ((index & 1U) * 4U)) & 15U;
                const int high = high_nibble >= 8U ? static_cast<int>(high_nibble) - 16
                                                   : static_cast<int>(high_nibble);
                const double scale = half_value(scale_plane[static_cast<std::size_t>(token) *
                                                                (kColumns / kGroup) + k / kGroup]);
                const double decoded = (static_cast<int>(low) + 16 * high) * scale;
                const double limit = 0.5 * scale + std::abs(n[k]) * std::ldexp(1.0, -7) + 1e-30;
                const double error = std::abs(decoded - n[k]) / limit;
                require(std::isfinite(error) && error <= 1.0,
                        "normalized A8G64 planes differ from FP64 RMSNorm at token " +
                            std::to_string(token) + " column " + std::to_string(k));
                worst_plane = std::max(worst_plane, error);
            }
        }

        // Independent FP64 oracle of the complete formula at sampled tokens, every row.
        const std::vector<std::uint32_t> sampled{0, 1, 7, 127, 128, 516, 1023, 1024, 1535, 2047};
        double worst = 0.0;
        for (const std::uint32_t token : sampled) {
            const std::vector<double> xq = a8g64(normalized_row(token));
            std::vector<double> reference(kRows);
            double sum_squares = 0;
            for (std::uint32_t row = 0; row < kRows; ++row) {
                double sum = 0;
                for (std::uint32_t k = 0; k < kColumns; ++k) sum += xq[k] * weight_at(codes, scales, row, k);
                reference[row] = sum;
                sum_squares += sum * sum;
            }
            const double rms = std::sqrt(sum_squares / kRows);
            for (std::uint32_t row = 0; row < kRows; ++row) {
                const double actual =
                    static_cast<float>(output[static_cast<std::size_t>(token) * kRows + row]);
                // Predeclared: BF16 output rounding plus a flipped normalization seam propagated
                // through one code step, bounded relative to the token's output RMS.
                const double limit = std::abs(reference[row]) * std::ldexp(1.0, -8) + 0.005 * rms;
                const double error = std::abs(actual - reference[row]) / limit;
                require(std::isfinite(error) && error <= 1.0, "FP64 oracle mismatch");
                worst = std::max(worst, error);
            }
        }

        // Timing against the composition the route replaces.
        Device<std::uint8_t> d_linear_workspace(
            ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::Q4G64_F16S, tokens, kColumns));
        Device<hip_bfloat16> d_hidden(input.size());
        ninfer::Tensor hidden(d_hidden.data, ninfer::DType::BF16,
                              {static_cast<int>(kColumns), static_cast<int>(tokens)});
        ninfer::Tensor control(d_control.data, ninfer::DType::BF16,
                               {static_cast<int>(kRows), static_cast<int>(tokens)});
        const auto composition = [&] {
            ninfer::ops::rmsnorm(x, gains, kEps, true, hidden, stream);
            ninfer::ops::linear(hidden, weight, control,
                                {d_linear_workspace.data, d_linear_workspace.bytes()}, stream);
        };
        const auto [route_ms, composition_ms] = time_ab(stream, route, composition);

        // ---- GDN output: per-head gated RMSNorm into the A8G64 codec, K6144 -> N5120. ----
        constexpr std::uint32_t kGatedColumns = 6144, kGatedRows = 5120, kHeadDim = 128;
        std::vector<std::uint8_t> gated_codes(static_cast<std::size_t>(kGatedRows) * kGatedColumns / 2U);
        std::vector<std::uint16_t> gated_scales(static_cast<std::size_t>(kGatedRows) * kGatedColumns / kGroup);
        for (auto& byte : gated_codes) byte = static_cast<std::uint8_t>(rng.next());
        for (auto& scale : gated_scales)
            scale = half_bits(static_cast<float>(0.004 * (0.5 + rng.uniform())));
        std::vector<hip_bfloat16> head_norm(kHeadDim);
        for (auto& gain : head_norm) gain = hip_bfloat16(0.2 + 1.6 * rng.uniform());
        std::vector<hip_bfloat16> recurrent(static_cast<std::size_t>(tokens) * kGatedColumns),
            gate_values(recurrent.size());
        for (std::size_t index = 0; index < recurrent.size(); ++index) {
            const double head_amplitude = 0.02 + 3.0 * ((index / kHeadDim) % 7U) / 7.0;
            recurrent[index] = hip_bfloat16((rng.uniform() - 0.5) * head_amplitude);
            gate_values[index] = hip_bfloat16((rng.uniform() - 0.5) * 12.0);
        }
        for (std::uint32_t k = 0; k < kHeadDim; ++k)  // one all-zero head row
            recurrent[static_cast<std::size_t>(5) * kGatedColumns + 3U * kHeadDim + k] =
                hip_bfloat16(0.0F);
        Device<hip_bfloat16> d_recurrent(recurrent.size()), d_gate(gate_values.size()),
            d_head_norm(head_norm.size()),
            d_gated_output(static_cast<std::size_t>(tokens) * kGatedRows),
            d_gated_control(static_cast<std::size_t>(tokens) * kGatedRows),
            d_gated_normalized(recurrent.size());
        Device<std::uint8_t> d_gated_codes(gated_codes.size());
        Device<std::uint16_t> d_gated_scales(gated_scales.size());
        d_recurrent.put(recurrent);
        d_gate.put(gate_values);
        d_head_norm.put(head_norm);
        d_gated_codes.put(gated_codes);
        d_gated_scales.put(gated_scales);
        require(lin::a8q4g64_gated_normalized_linear_prefill_supported(tokens),
                "T2048 is outside the gated prefill route");
        const std::size_t gated_workspace_bytes =
            lin::a8q4g64_activation_workspace_capacity_bytes(tokens, kGatedColumns);
        Device<std::uint8_t> d_gated_workspace(gated_workspace_bytes);
        const auto gated_route = [&] {
            check(lin::a8q4g64_gated_normalized_linear_prefill(
                      {.input = d_recurrent.data,
                       .weight_codes = d_gated_codes.data,
                       .weight_code_bytes = d_gated_codes.bytes(),
                       .weight_scales = d_gated_scales.data,
                       .weight_scale_bytes = d_gated_scales.bytes(),
                       .activation_workspace = d_gated_workspace.data,
                       .activation_workspace_bytes = gated_workspace_bytes,
                       .output = d_gated_output.data,
                       .tokens = tokens, .rows = kGatedRows, .columns = kGatedColumns,
                       .padded_columns = kGatedColumns},
                      d_head_norm.data, d_gate.data, kEps, stream),
                  "gated route");
        };
        gated_route();
        check(hipStreamSynchronize(stream), "sync");
        const std::vector<hip_bfloat16> gated_output = d_gated_output.get();
        lin::A8G64ActivationWorkspace gated_planes{};
        check(lin::a8q4g64_bind_activation_workspace(d_gated_workspace.data, gated_workspace_bytes,
                                                     tokens, kGatedColumns, &gated_planes),
              "bind gated planes");
        const lin::A8Q4G64LinearArgs gated_linear{
            gated_planes.low_codes, gated_planes.low_code_bytes, gated_planes.high_codes,
            gated_planes.high_code_bytes, gated_planes.scales, gated_planes.scale_bytes,
            gated_planes.status, d_gated_codes.data, d_gated_codes.bytes(), d_gated_scales.data,
            d_gated_scales.bytes(), d_gated_control.data, tokens, kGatedRows, kGatedColumns,
            kGatedColumns};
        check(lin::a8q4g64_linear_prefill_cta(gated_linear, stream), "gated control GEMM");
        check(hipStreamSynchronize(stream), "sync");
        require(std::memcmp(gated_output.data(), d_gated_control.get().data(),
                            gated_output.size() * sizeof(hip_bfloat16)) == 0,
                "gated output differs from the production GEMM on the same planes");
        // FP64 per-head gated RMSNorm with the BF16 seam: n = BF16(x/sqrt(mean_h(x^2)+eps)*w*silu(z)).
        const auto gated_row = [&](std::uint32_t token) {
            std::vector<double> n(kGatedColumns);
            for (std::uint32_t head = 0; head < kGatedColumns / kHeadDim; ++head) {
                const std::size_t base = static_cast<std::size_t>(token) * kGatedColumns + head * kHeadDim;
                double sum = 0;
                for (std::uint32_t k = 0; k < kHeadDim; ++k) {
                    const double v = static_cast<float>(recurrent[base + k]);
                    sum += v * v;
                }
                const double inverse = 1.0 / std::sqrt(sum / kHeadDim + kEps);
                for (std::uint32_t k = 0; k < kHeadDim; ++k) {
                    const double z = static_cast<float>(gate_values[base + k]);
                    n[head * kHeadDim + k] = bf16_rne(static_cast<float>(recurrent[base + k]) *
                                                      inverse * static_cast<float>(head_norm[k]) *
                                                      (z / (1.0 + std::exp(-z))));
                }
            }
            return n;
        };
        const std::vector<std::uint8_t> gated_workspace = d_gated_workspace.get();
        const auto gated_plane = [&](const void* region) {
            return gated_workspace.data() +
                   (static_cast<const std::uint8_t*>(region) - d_gated_workspace.data);
        };
        const std::uint8_t* gated_low = gated_plane(gated_planes.low_codes);
        const std::uint8_t* gated_high = gated_plane(gated_planes.high_codes);
        const auto* gated_scale_plane =
            reinterpret_cast<const std::uint16_t*>(gated_plane(gated_planes.scales));
        std::uint32_t gated_status = 0;
        std::memcpy(&gated_status, gated_plane(gated_planes.status), sizeof(gated_status));
        require(gated_status == 0U, "gated route reported activation status");
        double gated_worst_plane = 0.0;
        for (std::uint32_t token = 0; token < tokens; ++token) {
            const std::vector<double> n = gated_row(token);
            for (std::uint32_t k = 0; k < kGatedColumns; ++k) {
                const std::size_t index = static_cast<std::size_t>(token) * kGatedColumns + k;
                const unsigned low = (gated_low[index / 2] >> ((index & 1U) * 4U)) & 15U;
                const unsigned high_nibble = (gated_high[index / 2] >> ((index & 1U) * 4U)) & 15U;
                const int high = high_nibble >= 8U ? static_cast<int>(high_nibble) - 16
                                                   : static_cast<int>(high_nibble);
                const double scale = half_value(gated_scale_plane[static_cast<std::size_t>(token) *
                                                                      (kGatedColumns / kGroup) + k / kGroup]);
                const double decoded = (static_cast<int>(low) + 16 * high) * scale;
                const double limit = 0.5 * scale + std::abs(n[k]) * std::ldexp(1.0, -7) + 1e-30;
                const double error = std::abs(decoded - n[k]) / limit;
                require(std::isfinite(error) && error <= 1.0,
                        "gated A8G64 planes differ from FP64 gated RMSNorm at token " +
                            std::to_string(token) + " column " + std::to_string(k));
                gated_worst_plane = std::max(gated_worst_plane, error);
            }
        }
        double gated_worst = 0.0;
        for (const std::uint32_t token : sampled) {
            const std::vector<double> xq = a8g64(gated_row(token));
            std::vector<double> reference(kGatedRows);
            double sum_squares = 0;
            for (std::uint32_t row = 0; row < kGatedRows; ++row) {
                double sum = 0;
                for (std::uint32_t k = 0; k < kGatedColumns; ++k)
                    sum += xq[k] * weight_at(gated_codes, gated_scales, row, k, kGatedColumns);
                reference[row] = sum;
                sum_squares += sum * sum;
            }
            const double rms = std::sqrt(sum_squares / kGatedRows);
            for (std::uint32_t row = 0; row < kGatedRows; ++row) {
                const double actual = static_cast<float>(
                    gated_output[static_cast<std::size_t>(token) * kGatedRows + row]);
                const double limit = std::abs(reference[row]) * std::ldexp(1.0, -8) + 0.005 * rms;
                const double error = std::abs(actual - reference[row]) / limit;
                require(std::isfinite(error) && error <= 1.0, "gated FP64 oracle mismatch");
                gated_worst = std::max(gated_worst, error);
            }
        }
        Device<std::uint8_t> d_gated_linear_workspace(ninfer::ops::linear_workspace_capacity_bytes(
            ninfer::QType::Q4G64_F16S, tokens, kGatedColumns));
        ninfer::Weight gated_weight = weight;
        gated_weight.n = gated_weight.shape[0] = gated_weight.padded_shape[0] = kGatedRows;
        gated_weight.k = gated_weight.shape[1] = gated_weight.padded_shape[1] = kGatedColumns;
        gated_weight.qdata = gated_weight.payload = d_gated_codes.data;
        gated_weight.qdata_bytes = gated_weight.payload_bytes = d_gated_codes.bytes();
        gated_weight.scales = d_gated_scales.data;
        gated_weight.scale_bytes = d_gated_scales.bytes();
        const ninfer::Tensor recurrent_tensor(d_recurrent.data, ninfer::DType::BF16,
                                              {static_cast<int>(kHeadDim), 48, static_cast<int>(tokens)});
        const ninfer::Tensor gate_tensor(d_gate.data, ninfer::DType::BF16,
                                         {static_cast<int>(kHeadDim), 48, static_cast<int>(tokens)});
        const ninfer::Tensor head_norm_tensor(d_head_norm.data, ninfer::DType::BF16,
                                              {static_cast<int>(kHeadDim)});
        ninfer::Tensor gated_normalized(d_gated_normalized.data, ninfer::DType::BF16,
                                        {static_cast<int>(kHeadDim), 48, static_cast<int>(tokens)});
        ninfer::Tensor gated_control_tensor(d_gated_control.data, ninfer::DType::BF16,
                                            {static_cast<int>(kGatedRows), static_cast<int>(tokens)});
        const auto gated_composition = [&] {
            ninfer::ops::gated_rmsnorm(recurrent_tensor, head_norm_tensor, gate_tensor, kEps,
                                       gated_normalized, stream);
            ninfer::ops::linear(gated_normalized.view({static_cast<int>(kGatedColumns),
                                                       static_cast<int>(tokens)}),
                                gated_weight, gated_control_tensor,
                                {d_gated_linear_workspace.data, d_gated_linear_workspace.bytes()},
                                stream);
        };
        const auto [gated_route_ms, gated_composition_ms] =
            time_ab(stream, gated_route, gated_composition);

        std::ostringstream json;
        json << std::setprecision(9)
             << "{\"schema\":\"ninfer.r9700.a8q4-normalized-linear-prefill.v1\","
             << "\"target\":\"gfx1201\",\"public_op_tested\":true,\"tokens\":" << tokens
             << ",\"rows\":" << kRows << ",\"columns\":" << kColumns
             << ",\"oracle\":\"FP64 BF16-seam RMSNorm, A8G64 codec, decoded Q4 weights\","
             << "\"plane_criterion\":\"|decoded-n| <= scale/2 + |n|*2^-7, every token\","
             << "\"output_criterion\":\"|err| <= 2^-8|ref| + 0.005 token_rms, sampled tokens\","
             << "\"sampled_tokens\":" << sampled.size()
             << ",\"maximum_normalized_plane_error\":" << worst_plane
             << ",\"maximum_normalized_oracle_error\":" << worst
             << ",\"output_exact_vs_production_gemm_same_planes\":true"
             << ",\"misaligned_prefill_input_rejected\":true"
             << ",\"route_ms\":" << route_ms << ",\"composition_ms\":" << composition_ms
             << ",\"gated_output\":{\"rows\":" << kGatedRows << ",\"columns\":" << kGatedColumns
             << ",\"oracle\":\"FP64 per-head gated RMSNorm (BF16 seam), A8G64 codec, decoded Q4\""
             << ",\"maximum_normalized_plane_error\":" << gated_worst_plane
             << ",\"maximum_normalized_oracle_error\":" << gated_worst
             << ",\"output_exact_vs_production_gemm_same_planes\":true"
             << ",\"route_ms\":" << gated_route_ms << ",\"composition_ms\":" << gated_composition_ms
             << "}}\n";
        std::ofstream(out) << json.str();
        std::cout << json.str();
        check(hipStreamDestroy(stream), "stream");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "a8q4_normalized_linear_prefill_qual: " << error.what() << '\n';
        return 1;
    }
}
