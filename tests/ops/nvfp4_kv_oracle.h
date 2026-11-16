#pragma once

// Independent host-only numerical oracle for the R9700 NVFP4 KV representation.
//
// K and V each group 16 contiguous features under one scale:
//   scale = E4M3FN_satfinite_rne(FP32_rne(max|x| / 6))
//   code  = E2M1_rne(FP32_rne(x / decode(scale)))   (zero codes when the scale decodes to zero)
// An E2M1 code holds sign bit 3 and magnitude index 0..7 of {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Midpoints
// tie to the even index, values above 6 saturate, and the sign is kept for zero magnitudes. Nibbles
// pack low lane first. Selection is exhaustive, so this file shares no threshold or conversion with
// the device codec.
//
// Host-only test reference; do not use it for production packing.

#include "ops/fp8_int4_kv_oracle.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test::nvfp4_kv_oracle {

inline constexpr std::array<double, 8> kE2m1Magnitudes{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

inline double decode_e2m1(std::uint8_t code) {
    if (code > 15U) { throw std::invalid_argument("E2M1 code exceeds four bits"); }
    const double magnitude = kE2m1Magnitudes[code & 7U];
    return (code & 8U) != 0U ? -magnitude : magnitude;
}

inline std::uint8_t encode_e2m1(float quotient) {
    if (!std::isfinite(quotient)) { throw std::invalid_argument("E2M1 source must be finite"); }
    const double magnitude = std::abs(static_cast<double>(quotient));
    std::uint8_t best      = 0;
    double best_distance   = magnitude;
    for (std::uint8_t candidate = 1; candidate < 8U; ++candidate) {
        const double distance = std::abs(magnitude - kE2m1Magnitudes[candidate]);
        if (distance < best_distance ||
            (distance == best_distance && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
            best          = candidate;
            best_distance = distance;
        }
    }
    return static_cast<std::uint8_t>(best | (std::signbit(quotient) ? 8U : 0U));
}

// One encoded plane pair over [tokens, dimension] represented values.
struct Encoded {
    std::size_t tokens{};
    std::size_t dimension{};
    std::vector<std::uint8_t> packed_codes;
    std::vector<std::uint8_t> scales;

    [[nodiscard]] double at(std::size_t token, std::size_t lane) const {
        const std::uint8_t packed = packed_codes[(token * dimension + lane) / 2U];
        const auto code =
            static_cast<std::uint8_t>((lane & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
        return decode_e2m1(code) *
               fp8_int4_kv_oracle::decode_e4m3fn(scales[token * (dimension / 16U) + lane / 16U]);
    }
};

inline Encoded encode(std::span<const float> source, std::size_t tokens, std::size_t dimension) {
    if (dimension == 0U || dimension % 16U != 0U || source.size() != tokens * dimension) {
        throw std::invalid_argument("NVFP4 source shape is invalid");
    }
    const std::size_t groups = dimension / 16U;
    Encoded result{tokens, dimension, std::vector<std::uint8_t>(tokens * dimension / 2U, 0U),
                   std::vector<std::uint8_t>(tokens * groups, 0U)};
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t group = 0; group < groups; ++group) {
            const std::size_t base = token * dimension + group * 16U;
            float maximum          = 0.0F;
            for (std::size_t lane = 0; lane < 16U; ++lane) {
                if (!std::isfinite(source[base + lane])) {
                    throw std::invalid_argument("NVFP4 source must be finite");
                }
                maximum = std::max(maximum, std::abs(source[base + lane]));
            }
            const std::uint8_t word = fp8_int4_kv_oracle::encode_e4m3fn(maximum / 6.0F);
            result.scales[token * groups + group] = word;
            const auto scale = static_cast<float>(fp8_int4_kv_oracle::decode_e4m3fn(word));
            if (scale == 0.0F) { continue; }
            for (std::size_t lane = 0; lane < 16U; ++lane) {
                const std::uint8_t code = encode_e2m1(source[base + lane] / scale);
                std::uint8_t& packed    = result.packed_codes[(base + lane) / 2U];
                packed = (lane & 1U) == 0U
                             ? static_cast<std::uint8_t>((packed & 0xf0U) | code)
                             : static_cast<std::uint8_t>((packed & 0x0fU) | (code << 4U));
            }
        }
    }
    return result;
}

// Stable FP64 attention of one represented query head over the selected logical tokens.
inline std::vector<double> attention_fp64(std::span<const float> query,
                                          std::span<const std::size_t> selected,
                                          const Encoded& keys, const Encoded& values,
                                          std::size_t kv_heads, std::size_t kv_head, double scale) {
    const std::size_t dimension = keys.dimension;
    std::vector<double> scores(selected.size());
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const std::size_t token = selected[index] * kv_heads + kv_head;
        double dot              = 0.0;
        for (std::size_t lane = 0; lane < dimension; ++lane) {
            dot += static_cast<double>(query[lane]) * keys.at(token, lane);
        }
        scores[index] = dot * scale;
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator   = 0.0;
    for (double& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }
    std::vector<double> output(dimension, 0.0);
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const std::size_t token = selected[index] * kv_heads + kv_head;
        for (std::size_t lane = 0; lane < dimension; ++lane) {
            output[lane] += scores[index] / denominator * values.at(token, lane);
        }
    }
    return output;
}

} // namespace ninfer::test::nvfp4_kv_oracle
