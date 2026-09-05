#pragma once

// Independent, host-only numerical oracle for the proposed R9700 KV representation.
//
// Keys use the OCP E4M3FN bit pattern directly:
//   sign[7], exponent[6:3] with bias 7, fraction[2:0].
//   e=0 encodes zero/subnormals, e=15,m=0..6 encodes finite values through 448,
//   and e=15,m=7 is NaN. Encoding rejects every non-finite source, saturates finite
//   overflow to +/-448, preserves signed zero, and rounds to nearest, ties to even.
//
// Values use symmetric signed INT4 codes [-7,7]. Code -8 is reserved and rejected by
// the decoder. Each contiguous G16 or G32 group stores
//   scale = FP16_RNE(max(abs(source)) / 7)
// and code = clamp(RNE(FP32(source / decode_fp16(scale))), -7, 7). A zero group, or a finite
// nonzero group whose scale rounds down to FP16 zero, has canonical +0 scale and all-zero
// codes. Nibbles are two's-complement, low lane first.
//
// This file deliberately has no device-runtime dependency. It is a mathematical/storage
// authority for tests, not a production packer or a model-sized conversion path.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::fp8_int4_kv_oracle {

inline double decode_e4m3fn(std::uint8_t word) {
    const bool negative       = (word & 0x80U) != 0;
    const std::uint32_t exp   = (word >> 3) & 0x0fU;
    const std::uint32_t fraction = word & 0x07U;
    double magnitude = 0.0;
    if (exp == 0) {
        magnitude = static_cast<double>(fraction) * std::ldexp(1.0, -9);
    } else {
        if (exp == 15 && fraction == 7) {
            throw std::invalid_argument("FP8 E4M3FN NaN word");
        }
        magnitude = (1.0 + static_cast<double>(fraction) / 8.0) *
                    std::ldexp(1.0, static_cast<int>(exp) - 7);
    }
    return negative ? -magnitude : magnitude;
}

inline std::uint8_t encode_e4m3fn(float source) {
    if (!std::isfinite(source)) {
        throw std::invalid_argument("FP8 E4M3FN source must be finite");
    }
    const bool negative = std::signbit(source);
    const double magnitude = std::abs(static_cast<double>(source));
    if (magnitude == 0.0) { return negative ? 0x80U : 0x00U; }
    if (magnitude >= 448.0) { return static_cast<std::uint8_t>((negative ? 0x80U : 0U) | 0x7eU); }

    // Exhaustive selection is intentionally independent of any device conversion intrinsic.
    // Positive finite E4M3FN words are monotonic in [0x00,0x7e]. The word's low bit is
    // the retained significand LSB, so an even word is the ties-to-even winner.
    std::uint8_t best = 0;
    double best_distance = magnitude;
    for (std::uint16_t candidate = 1; candidate <= 0x7eU; ++candidate) {
        const double distance =
            std::abs(magnitude - decode_e4m3fn(static_cast<std::uint8_t>(candidate)));
        if (distance < best_distance ||
            (distance == best_distance && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
            best = static_cast<std::uint8_t>(candidate);
            best_distance = distance;
        }
    }
    return static_cast<std::uint8_t>((negative ? 0x80U : 0U) | best);
}

namespace detail {

inline std::uint32_t round_shift_rne(std::uint32_t value, int shift) {
    if (shift <= 0) { return value; }
    const std::uint32_t mask = (1U << shift) - 1U;
    const std::uint32_t half = 1U << (shift - 1);
    const std::uint32_t base = value >> shift;
    const std::uint32_t remainder = value & mask;
    return base + ((remainder > half || (remainder == half && (base & 1U) != 0U)) ? 1U : 0U);
}

inline std::uint16_t fp32_to_fp16_rne(float source) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(source);
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    const std::uint32_t absolute = bits & 0x7fffffffU;
    if (absolute >= 0x7f800000U) {
        const std::uint32_t mantissa = absolute & 0x007fffffU;
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa != 0 ? 0x0200U : 0U));
    }

    int exponent = static_cast<int>((absolute >> 23) & 0xffU) - 127 + 15;
    std::uint32_t mantissa = absolute & 0x007fffffU;
    if (exponent <= 0) {
        if (exponent < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000U;
        return static_cast<std::uint16_t>(sign | round_shift_rne(mantissa, 14 - exponent));
    }
    if (exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }

    std::uint32_t half_mantissa = round_shift_rne(mantissa, 13);
    if (half_mantissa == 0x0400U) {
        half_mantissa = 0;
        ++exponent;
        if (exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                      half_mantissa);
}

inline float fp16_to_fp32(std::uint16_t word) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(word) & 0x8000U) << 16;
    std::uint32_t exponent = (static_cast<std::uint32_t>(word) >> 10) & 0x1fU;
    std::uint32_t mantissa = static_cast<std::uint32_t>(word) & 0x03ffU;
    if (exponent == 0) {
        if (mantissa == 0) { return std::bit_cast<float>(sign); }
        int unbiased = -14;
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1;
            --unbiased;
        }
        mantissa &= 0x03ffU;
        return std::bit_cast<float>(sign |
                                    (static_cast<std::uint32_t>(unbiased + 127) << 23) |
                                    (mantissa << 13));
    }
    if (exponent == 31) {
        return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13));
    }
    exponent = exponent - 15 + 127;
    return std::bit_cast<float>(sign | (exponent << 23) | (mantissa << 13));
}

inline int round_nearest_even(double value) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    if (fraction < 0.5) { return static_cast<int>(lower); }
    if (fraction > 0.5) { return static_cast<int>(lower + 1.0); }
    const int lower_integer = static_cast<int>(lower);
    return (lower_integer & 1) == 0 ? lower_integer : lower_integer + 1;
}

inline void validate_shape(std::size_t size, std::size_t tokens, std::size_t dimension,
                           const char* label) {
    if (tokens == 0 || dimension == 0 ||
        tokens > std::numeric_limits<std::size_t>::max() / dimension ||
        size != tokens * dimension) {
        throw std::invalid_argument(std::string(label) + ": invalid [tokens,dimension] shape");
    }
}

} // namespace detail

struct Fp8Keys {
    std::size_t tokens{};
    std::size_t dimension{};
    std::vector<std::uint8_t> codes;

    [[nodiscard]] double at(std::size_t token, std::size_t lane) const {
        if (token >= tokens || lane >= dimension || codes.size() != tokens * dimension) {
            throw std::out_of_range("FP8 key index or storage is invalid");
        }
        return decode_e4m3fn(codes[token * dimension + lane]);
    }
};

inline Fp8Keys encode_keys(std::span<const float> source, std::size_t tokens,
                           std::size_t dimension) {
    detail::validate_shape(source.size(), tokens, dimension, "FP8 keys");
    Fp8Keys result{tokens, dimension, std::vector<std::uint8_t>(source.size())};
    std::transform(source.begin(), source.end(), result.codes.begin(), encode_e4m3fn);
    return result;
}

struct Int4Values {
    std::size_t tokens{};
    std::size_t dimension{};
    std::size_t group_size{};
    std::vector<std::uint8_t> packed_codes;
    std::vector<std::uint16_t> fp16_scales;

    [[nodiscard]] int code_at(std::size_t token, std::size_t lane) const {
        if (token >= tokens || lane >= dimension || dimension % 2 != 0 ||
            packed_codes.size() != tokens * dimension / 2) {
            throw std::out_of_range("INT4 value index or storage is invalid");
        }
        const std::uint8_t packed = packed_codes[(token * dimension + lane) / 2];
        const std::uint8_t nibble = (lane & 1U) == 0U ? packed & 0x0fU : packed >> 4;
        const int code = (nibble & 0x08U) == 0U ? static_cast<int>(nibble)
                                                : static_cast<int>(nibble) - 16;
        if (code == -8) { throw std::invalid_argument("reserved symmetric INT4 code -8"); }
        return code;
    }

    [[nodiscard]] double at(std::size_t token, std::size_t lane) const {
        if ((group_size != 16 && group_size != 32) || dimension % group_size != 0 ||
            fp16_scales.size() != tokens * (dimension / group_size)) {
            throw std::invalid_argument("INT4 value scale storage is invalid");
        }
        const std::size_t groups_per_token = dimension / group_size;
        const std::uint16_t scale_word =
            fp16_scales[token * groups_per_token + lane / group_size];
        const double scale = static_cast<double>(detail::fp16_to_fp32(scale_word));
        if (!std::isfinite(scale) || scale < 0.0) {
            throw std::invalid_argument("INT4 value scale must be finite and non-negative");
        }
        return static_cast<double>(code_at(token, lane)) * scale;
    }
};

inline Int4Values encode_values(std::span<const float> source, std::size_t tokens,
                                std::size_t dimension, std::size_t group_size) {
    detail::validate_shape(source.size(), tokens, dimension, "INT4 values");
    if ((group_size != 16 && group_size != 32) || dimension % group_size != 0 ||
        dimension % 2 != 0) {
        throw std::invalid_argument(
            "INT4 values require an even dimension divisible by G16 or G32");
    }
    const std::size_t groups_per_token = dimension / group_size;
    Int4Values result{tokens,
                      dimension,
                      group_size,
                      std::vector<std::uint8_t>(tokens * dimension / 2, 0),
                      std::vector<std::uint16_t>(tokens * groups_per_token, 0)};

    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t group = 0; group < groups_per_token; ++group) {
            const std::size_t base = token * dimension + group * group_size;
            float maximum = 0.0F;
            for (std::size_t lane = 0; lane < group_size; ++lane) {
                const float value = source[base + lane];
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("INT4 value source must be finite");
                }
                maximum = std::max(maximum, std::abs(value));
            }
            if (maximum == 0.0F) { continue; }

            // FP32 division followed by exact IEEE binary16 RNE is part of this storage contract.
            const std::uint16_t scale_word = detail::fp32_to_fp16_rne(maximum / 7.0F);
            const float scale = detail::fp16_to_fp32(scale_word);
            if (!std::isfinite(scale)) {
                throw std::overflow_error("INT4 value scale is not representable as finite FP16");
            }
            // Canonical finite-underflow representation: the result vectors were zero-filled,
            // so retain +0 scale and all-zero codes without dividing by zero.
            if (scale == 0.0F) { continue; }
            result.fp16_scales[token * groups_per_token + group] = scale_word;
            for (std::size_t lane = 0; lane < group_size; ++lane) {
                const float quotient = source[base + lane] / scale;
                const int rounded = detail::round_nearest_even(static_cast<double>(quotient));
                const int code = std::clamp(rounded, -7, 7);
                const std::uint8_t nibble = static_cast<std::uint8_t>(code) & 0x0fU;
                const std::size_t logical = base + lane;
                std::uint8_t& packed = result.packed_codes[logical / 2];
                if ((lane & 1U) == 0U) {
                    packed = static_cast<std::uint8_t>((packed & 0xf0U) | nibble);
                } else {
                    packed = static_cast<std::uint8_t>((packed & 0x0fU) | (nibble << 4));
                }
            }
        }
    }
    return result;
}

inline std::vector<double> attention_fp64(std::span<const float> represented_query,
                                          const Fp8Keys& keys, const Int4Values& values,
                                          double attention_scale) {
    if (keys.tokens == 0 || keys.tokens != values.tokens || keys.dimension != values.dimension ||
        represented_query.size() != keys.dimension || !std::isfinite(attention_scale)) {
        throw std::invalid_argument("FP8/INT4 attention operands are incompatible");
    }
    for (float value : represented_query) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("attention query must be finite");
        }
    }

    std::vector<double> scores(keys.tokens, 0.0);
    for (std::size_t token = 0; token < keys.tokens; ++token) {
        double dot = 0.0;
        for (std::size_t lane = 0; lane < keys.dimension; ++lane) {
            dot += static_cast<double>(represented_query[lane]) * keys.at(token, lane);
        }
        scores[token] = dot * attention_scale;
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    for (double& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        throw std::runtime_error("stable softmax denominator is invalid");
    }

    std::vector<double> output(keys.dimension, 0.0);
    for (std::size_t token = 0; token < keys.tokens; ++token) {
        const double probability = scores[token] / denominator;
        for (std::size_t lane = 0; lane < keys.dimension; ++lane) {
            output[lane] += probability * values.at(token, lane);
        }
    }
    return output;
}

struct GreedyResult {
    std::vector<double> attention;
    std::vector<double> logits;
    std::size_t token{};
};

inline GreedyResult attention_greedy_fp64(std::span<const float> represented_query,
                                          const Fp8Keys& keys, const Int4Values& values,
                                          double attention_scale,
                                          std::span<const float> represented_output_weight,
                                          std::size_t vocabulary,
                                          std::span<const float> represented_bias = {}) {
    GreedyResult result;
    result.attention = attention_fp64(represented_query, keys, values, attention_scale);
    const std::size_t dimension = result.attention.size();
    if (vocabulary == 0 || vocabulary > std::numeric_limits<std::size_t>::max() / dimension ||
        represented_output_weight.size() != vocabulary * dimension ||
        (!represented_bias.empty() && represented_bias.size() != vocabulary)) {
        throw std::invalid_argument("greedy projection shape is invalid");
    }
    result.logits.resize(vocabulary);
    for (std::size_t row = 0; row < vocabulary; ++row) {
        double accumulator =
            represented_bias.empty() ? 0.0 : static_cast<double>(represented_bias[row]);
        if (!std::isfinite(accumulator)) {
            throw std::invalid_argument("greedy projection inputs must be finite");
        }
        for (std::size_t lane = 0; lane < dimension; ++lane) {
            const float weight = represented_output_weight[row * dimension + lane];
            if (!std::isfinite(weight)) {
                throw std::invalid_argument("greedy projection inputs must be finite");
            }
            accumulator += result.attention[lane] * static_cast<double>(weight);
        }
        result.logits[row] = accumulator;
    }

    // Strict comparison deliberately makes an exact tie select the lowest token id.
    result.token = 0;
    for (std::size_t row = 1; row < vocabulary; ++row) {
        if (result.logits[row] > result.logits[result.token]) { result.token = row; }
    }
    return result;
}

} // namespace ninfer::test::fp8_int4_kv_oracle
