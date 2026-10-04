#pragma once

// Independent host oracle of the exact signed-A8G64 Q4 activation image (ninfer/ops/linear.h):
// per token row and 64-column group, scale = FP16(amax/127) (one FP16 ulp when amax > 0 rounds to
// zero; zero when the scale overflows), code = clamp(RNE(x/scale), +-127) with the FP32 quotient,
// stored as low = code & 15 and high = (code - low) / 16 nibble planes; status bit 0 reports a
// nonfinite value (encoded as zero) and bit 1 a scale overflow.

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::tools::a8g64_image {

struct Image {
    std::vector<std::uint8_t> low;
    std::vector<std::uint8_t> high;
    std::vector<std::uint16_t> scales;
    std::uint32_t status = 0;

    bool operator==(const Image&) const = default;
};

// Oracle image of represented row-major values [tokens][columns] (columns a multiple of 64).
inline Image encode(const std::vector<float>& values, std::uint32_t tokens, std::uint32_t columns) {
    if (columns % 64U != 0U || values.size() != static_cast<std::size_t>(tokens) * columns)
        throw std::invalid_argument("a8g64 oracle: rows must be whole G64 groups");
    Image image;
    image.low.assign(values.size() / 2U, 0U);
    image.high.assign(values.size() / 2U, 0U);
    image.scales.assign(values.size() / 64U, 0U);
    for (std::size_t group = 0; group < values.size() / 64U; ++group) {
        float maximum = 0.0F;
        for (std::size_t lane = 0; lane < 64U; ++lane) {
            const float x = values[group * 64U + lane];
            if (!std::isfinite(x)) {
                image.status |= 1U;
                continue;
            }
            maximum = std::max(maximum, std::fabs(x));
        }
        std::uint16_t bits = 0U;
        if (maximum != 0.0F) {
            bits = std::bit_cast<std::uint16_t>(static_cast<_Float16>(maximum / 127.0F));
            if (bits == 0U) bits = 1U;
            if (!std::isfinite(static_cast<float>(std::bit_cast<_Float16>(bits)))) {
                image.status |= 2U;
                bits = 0U;
            }
        }
        image.scales[group] = bits;
        const float scale   = static_cast<float>(std::bit_cast<_Float16>(bits));
        for (std::size_t lane = 0; lane < 64U; ++lane) {
            const std::size_t index = group * 64U + lane;
            const float x           = values[index];
            const int code =
                scale != 0.0F && std::isfinite(x)
                    ? std::clamp(static_cast<int>(std::nearbyint(x / scale)), -127, 127)
                    : 0;
            const int low        = code & 0x0f;
            const int high       = (code - low) / 16;
            const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
            image.low[index / 2U] |= static_cast<std::uint8_t>(low << shift);
            image.high[index / 2U] |= static_cast<std::uint8_t>((high & 0x0f) << shift);
        }
    }
    return image;
}

inline std::vector<float> represented(const std::vector<hip_bfloat16>& values) {
    std::vector<float> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
        result[index] = static_cast<float>(values[index]);
    return result;
}

// The image the device span holds for (tokens, columns).
inline Image read(const DeviceSpan& span, std::uint32_t tokens, std::uint32_t columns) {
    ops::r9700::linear::A8G64ActivationWorkspace bound{};
    const std::size_t bytes =
        ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    if (bytes == 0U || span.bytes < bytes ||
        ops::r9700::linear::a8q4g64_bind_activation_workspace(span.data, bytes, tokens, columns,
                                                              &bound) != hipSuccess)
        throw std::invalid_argument("a8g64 oracle: image span is invalid");
    Image image;
    image.low.resize(bound.low_code_bytes);
    image.high.resize(bound.high_code_bytes);
    image.scales.resize(bound.scale_bytes / sizeof(std::uint16_t));
    HIP_CHECK(
        hipMemcpy(image.low.data(), bound.low_codes, bound.low_code_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(image.high.data(), bound.high_codes, bound.high_code_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(image.scales.data(), bound.scales, bound.scale_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&image.status, bound.status, sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    return image;
}

// Sets the image's status word (a stale nonzero value proves the producer publishes it).
inline void poison_status(const DeviceSpan& span, std::uint32_t tokens, std::uint32_t columns,
                          hipStream_t stream) {
    ops::r9700::linear::A8G64ActivationWorkspace bound{};
    const std::size_t bytes =
        ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    if (ops::r9700::linear::a8q4g64_bind_activation_workspace(span.data, bytes, tokens, columns,
                                                              &bound) != hipSuccess)
        throw std::invalid_argument("a8g64 oracle: image span is invalid");
    HIP_CHECK(hipMemsetAsync(bound.status, 0xff, sizeof(std::uint32_t), stream));
}

inline void require_image(const Image& actual, const Image& expected, const std::string& label) {
    if (actual.low != expected.low || actual.high != expected.high ||
        actual.scales != expected.scales)
        throw std::runtime_error(label + ": activation image differs from the A8G64 oracle");
    if (actual.status != expected.status)
        throw std::runtime_error(label + ": activation status " + std::to_string(actual.status) +
                                 " != oracle " + std::to_string(expected.status));
}

} // namespace ninfer::tools::a8g64_image
