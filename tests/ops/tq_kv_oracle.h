#pragma once

// Independent FP64 oracle of the TQ-K4-V3 growing cache (ops/r9700/kv/tq_k4v3_codec.h): the
// rotation, the per-vector encoding, decoding of stored planes back to the model basis, and
// softmax attention over decoded vectors. It shares only the codec's published constants.

#include "ops/r9700/kv/tq_k4v3_codec.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::test::tq_kv_oracle {

namespace tq                      = ninfer::ops::r9700::kv::tq;
inline constexpr std::size_t kDim = tq::kHeadDim;
using Vector                      = std::array<double, kDim>;

[[nodiscard]] inline double sign(std::size_t feature) {
    return tq::negative_sign(static_cast<std::uint32_t>(feature)) ? -1.0 : 1.0;
}

// y = H diag(s) x / 16 by the explicit matrix definition H_ij = (-1)^popcount(i & j).
[[nodiscard]] inline Vector rotate(std::span<const double> x) {
    Vector y{};
    for (std::size_t i = 0; i < kDim; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < kDim; ++j) {
            const double h = (std::popcount(i & j) & 1U) != 0U ? -1.0 : 1.0;
            sum += h * sign(j) * x[j];
        }
        y[i] = sum / 16.0;
    }
    return y;
}

// x = diag(s) H y / 16.
[[nodiscard]] inline Vector unrotate(std::span<const double> y) {
    Vector x{};
    for (std::size_t i = 0; i < kDim; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < kDim; ++j) {
            sum += ((std::popcount(i & j) & 1U) != 0U ? -1.0 : 1.0) * y[j];
        }
        x[i] = sign(i) * sum / 16.0;
    }
    return x;
}

// binary16 round-to-nearest-even of a finite nonnegative double (overflow returns 0x7C00).
[[nodiscard]] inline std::uint16_t f16_bits(double value) {
    if (value == 0.0) return 0U;
    int exponent = 0;
    (void)std::frexp(value, &exponent); // value = m * 2^exponent, m in [0.5, 1)
    const int unbiased = exponent - 1;
    const int shift    = unbiased < -14 ? -24 : unbiased - 10; // ulp exponent
    const double ulp   = std::ldexp(1.0, shift);
    double units       = value / ulp;
    double rounded     = std::nearbyint(units); // FE_TONEAREST: ties to even
    const double total = rounded * ulp;
    if (total >= 65520.0) return 0x7C00U;
    // Re-encode the rounded value exactly.
    if (total < std::ldexp(1.0, -14)) {
        return static_cast<std::uint16_t>(std::lround(total / std::ldexp(1.0, -24)));
    }
    int e               = 0;
    const double m      = std::frexp(total, &e); // total = m * 2^e, m in [0.5,1)
    const auto mantissa = static_cast<std::uint16_t>(std::lround((m * 2.0 - 1.0) * 1024.0));
    return static_cast<std::uint16_t>(((e - 1 + 15) << 10) | mantissa);
}

[[nodiscard]] inline double f16_value(std::uint16_t bits) {
    const int exponent = (bits >> 10U) & 0x1fU;
    const int mantissa = bits & 0x3ffU;
    const double magnitude =
        exponent == 0 ? std::ldexp(mantissa, -24) : std::ldexp(1024 + mantissa, exponent - 25);
    return (bits & 0x8000U) != 0U ? -magnitude : magnitude;
}

struct EncodedVector {
    std::array<std::uint8_t, kDim> index{}; // magnitude index
    std::array<bool, kDim> negative{};
    std::array<double, kDim> r{}; // level-domain magnitude before thresholding
    std::uint16_t scale = 0U;
};

// Encodes one represented vector. `bits` is 4 (K) or 3 (V).
[[nodiscard]] inline EncodedVector encode(std::span<const float> source, unsigned bits) {
    std::array<double, kDim> x{};
    double norm2 = 0.0;
    for (std::size_t i = 0; i < kDim; ++i) {
        x[i] = source[i];
        norm2 += x[i] * x[i];
    }
    const double norm = std::sqrt(norm2);
    const Vector y    = rotate(x);
    const double alpha =
        bits == 4U ? static_cast<double>(tq::kKeyAlpha) : static_cast<double>(tq::kValueAlpha);
    const std::span<const float> thresholds = bits == 4U
                                                  ? std::span<const float>(tq::kKeyThresholds)
                                                  : std::span<const float>(tq::kValueThresholds);
    const std::span<const float> levels     = bits == 4U ? std::span<const float>(tq::kKeyLevels)
                                                         : std::span<const float>(tq::kValueLevels);
    EncodedVector encoded;
    double code_norm2 = 0.0;
    for (std::size_t i = 0; i < kDim; ++i) {
        const double r     = norm > 0.0 ? std::abs(y[i]) * 16.0 * alpha / norm : 0.0;
        std::uint8_t index = 0U;
        for (const float threshold : thresholds) index += r > threshold ? 1U : 0U;
        encoded.index[i]    = index;
        encoded.negative[i] = y[i] < 0.0;
        encoded.r[i]        = r;
        code_norm2 += static_cast<double>(levels[index]) * levels[index];
    }
    encoded.scale = f16_bits(norm / std::sqrt(code_norm2));
    return encoded;
}

// FP16 scale the codec stores for given stored codes of a source vector.
[[nodiscard]] inline std::uint16_t
scale_for_codes(std::span<const float> source, std::span<const std::uint8_t> index, unsigned bits) {
    const std::span<const float> levels = bits == 4U ? std::span<const float>(tq::kKeyLevels)
                                                     : std::span<const float>(tq::kValueLevels);
    double norm2 = 0.0, code_norm2 = 0.0;
    for (std::size_t i = 0; i < kDim; ++i) {
        norm2 += static_cast<double>(source[i]) * source[i];
        code_norm2 += static_cast<double>(levels[index[i]]) * levels[index[i]];
    }
    return f16_bits(std::sqrt(norm2) / std::sqrt(code_norm2));
}

// Stored K bytes (128, low nibble = even feature) and scale to the model-basis vector.
[[nodiscard]] inline Vector decode_key(std::span<const std::uint8_t> bytes, std::uint16_t scale) {
    Vector y{};
    for (std::size_t i = 0; i < kDim; ++i) {
        const unsigned code = (bytes[i / 2U] >> ((i & 1U) * 4U)) & 0x0fU;
        const double level  = tq::kKeyLevels[code & 7U];
        y[i]                = ((code & 8U) != 0U ? -level : level) * f16_value(scale);
    }
    return unrotate(y);
}

// Stored K bytes and scale to the rotated-basis vector scale * c.
[[nodiscard]] inline Vector decode_key_rotated(std::span<const std::uint8_t> bytes,
                                               std::uint16_t scale) {
    Vector y{};
    for (std::size_t i = 0; i < kDim; ++i) {
        const unsigned code = (bytes[i / 2U] >> ((i & 1U) * 4U)) & 0x0fU;
        const double level  = tq::kKeyLevels[code & 7U];
        y[i]                = ((code & 8U) != 0U ? -level : level) * f16_value(scale);
    }
    return y;
}

// Stored K bytes to (index, negative) per feature.
inline void key_codes(std::span<const std::uint8_t> bytes, std::span<std::uint8_t> index,
                      std::span<bool> negative) {
    for (std::size_t i = 0; i < kDim; ++i) {
        const unsigned code = (bytes[i / 2U] >> ((i & 1U) * 4U)) & 0x0fU;
        index[i]            = static_cast<std::uint8_t>(code & 7U);
        negative[i]         = (code & 8U) != 0U;
    }
}

// Stored V bytes (96) to (index, negative) per feature, by the published bit placement.
inline void value_codes(std::span<const std::uint8_t> bytes, std::span<std::uint8_t> index,
                        std::span<bool> negative) {
    for (std::size_t f = 0; f < kDim; ++f) {
        const std::size_t chunk = f / 32U, local = f % 32U;
        const std::size_t word = local / 16U, within = local % 16U;
        const std::size_t lane = within % 4U, k = within / 4U;
        index[f] =
            static_cast<std::uint8_t>((bytes[chunk * 12U + word * 4U + lane] >> (2U * k)) & 3U);
        negative[f] = ((bytes[chunk * 12U + 8U + (local % 4U)] >> (local / 4U)) & 1U) != 0U;
    }
}

[[nodiscard]] inline Vector decode_value_rotated(std::span<const std::uint8_t> bytes,
                                                 std::uint16_t scale) {
    std::array<std::uint8_t, kDim> index{};
    std::array<bool, kDim> negative{};
    value_codes(bytes, index, negative);
    Vector y{};
    for (std::size_t i = 0; i < kDim; ++i) {
        const double level = tq::kValueLevels[index[i]];
        y[i]               = (negative[i] ? -level : level) * f16_value(scale);
    }
    return y;
}

[[nodiscard]] inline Vector decode_value(std::span<const std::uint8_t> bytes, std::uint16_t scale) {
    std::array<std::uint8_t, kDim> index{};
    std::array<bool, kDim> negative{};
    value_codes(bytes, index, negative);
    Vector y{};
    for (std::size_t i = 0; i < kDim; ++i) {
        const double level = tq::kValueLevels[index[i]];
        y[i]               = (negative[i] ? -level : level) * f16_value(scale);
    }
    return unrotate(y);
}

// FP64 softmax(q.k * scale) v over the selected decoded tokens of one KV head.
[[nodiscard]] inline std::vector<double>
attention_fp64(std::span<const float> query, std::span<const std::size_t> selected,
               std::span<const Vector> keys, std::span<const Vector> values, std::size_t kv_heads,
               std::size_t kv_head, double scale) {
    std::vector<double> scores(selected.size());
    double maximum = -INFINITY;
    for (std::size_t s = 0; s < selected.size(); ++s) {
        const Vector& key = keys[selected[s] * kv_heads + kv_head];
        double dot        = 0.0;
        for (std::size_t i = 0; i < kDim; ++i) dot += static_cast<double>(query[i]) * key[i];
        scores[s] = dot * scale;
        maximum   = std::max(maximum, scores[s]);
    }
    std::vector<double> output(kDim, 0.0);
    double denominator = 0.0;
    for (std::size_t s = 0; s < selected.size(); ++s) {
        const double weight = std::exp(scores[s] - maximum);
        denominator += weight;
        const Vector& value = values[selected[s] * kv_heads + kv_head];
        for (std::size_t i = 0; i < kDim; ++i) output[i] += weight * value[i];
    }
    for (double& element : output) element /= denominator;
    return output;
}

} // namespace ninfer::test::tq_kv_oracle
