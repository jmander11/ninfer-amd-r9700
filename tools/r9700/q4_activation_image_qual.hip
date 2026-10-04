// Q4 activation image boundary (ninfer/ops/linear.h): the producers' images against the
// independent host A8G64 oracle, and every projection from an image against the ordinary Linear
// on the same input, bit for bit, at the DFlash2 drafter shapes.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"

#include "a8g64_image_oracle.h"

#include <hip/hip_bfloat16.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DeviceBuffer;
using ninfer::DeviceSpan;
using ninfer::DType;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
namespace image_oracle = ninfer::tools::a8g64_image;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

template <class T>
DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer result(values.size() * sizeof(T));
    result.copy_from_host(values.data(), result.size());
    return result;
}

template <class T>
std::vector<T> download(const DeviceBuffer& buffer) {
    std::vector<T> values(buffer.size() / sizeof(T));
    buffer.copy_to_host(values.data(), buffer.size());
    return values;
}

std::vector<hip_bfloat16> normal_rows(std::size_t count, float sigma, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0F, sigma);
    std::vector<hip_bfloat16> values(count);
    for (auto& value : values) value = hip_bfloat16(distribution(generator));
    return values;
}

bool same_bits(const std::vector<hip_bfloat16>& left, const std::vector<hip_bfloat16>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index)
        if (left[index].data != right[index].data) return false;
    return true;
}

// Arbitrary packed signed-Q4 codes and small FP16 scales in the canonical Q4N16K16 extents: the
// comparison is between two routes over the same bytes, so any codes qualify.
struct Q4Weight {
    DeviceBuffer codes;
    DeviceBuffer scales;
    Weight view;
};

Q4Weight make_weight(int rows, int columns, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows) * columns / 2U);
    for (auto& code : codes) code = static_cast<std::uint8_t>(generator());
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(rows) * (columns / 64));
    std::uniform_real_distribution<float> scale(1.0e-3F, 4.0e-3F);
    for (auto& word : scales)
        word = std::bit_cast<std::uint16_t>(static_cast<_Float16>(scale(generator)));
    Q4Weight weight{upload(codes), upload(scales), {}};
    Weight& view         = weight.view;
    view.payload         = weight.codes.data();
    view.payload_bytes   = weight.codes.size() + weight.scales.size();
    view.qtype           = QType::Q4G64_F16S;
    view.group_size      = 64;
    view.ndim            = 2;
    view.shape[0]        = rows;
    view.shape[1]        = columns;
    view.padded_shape[0] = rows;
    view.padded_shape[1] = columns;
    view.qdata           = weight.codes.data();
    view.qdata_bytes     = weight.codes.size();
    view.scales          = weight.scales.data();
    view.scale_bytes     = weight.scales.size();
    view.n               = rows;
    view.k               = columns;
    view.group           = 64;
    view.layout          = QuantLayout::Q4N16K16;
    view.scale_dtype     = DType::FP16;
    view.scale_ne[0]     = columns / 64;
    view.scale_ne[1]     = rows;
    view.scale_nb[0]     = sizeof(std::uint16_t);
    view.scale_nb[1]     = static_cast<std::int64_t>(columns / 64) * sizeof(std::uint16_t);
    return weight;
}

DeviceBuffer image_buffer(int tokens, int columns) {
    return DeviceBuffer(ninfer::ops::q4_activation_image_bytes(tokens, columns));
}

// One quantization feeding a projection must reproduce the ordinary Linear bit for bit.
void qualify_projection(ninfer::DeviceContext& device, int rows, int columns, int tokens) {
    const Q4Weight weight = make_weight(rows, columns, static_cast<std::uint32_t>(rows + columns));
    const auto input      = normal_rows(static_cast<std::size_t>(columns) * tokens, 1.0F,
                                        static_cast<std::uint32_t>(tokens * 7 + rows));
    auto d_input          = upload(input);
    DeviceBuffer d_reference(static_cast<std::size_t>(rows) * tokens * sizeof(hip_bfloat16));
    DeviceBuffer d_output(d_reference.size());
    DeviceBuffer d_linear_image = image_buffer(tokens, columns);
    DeviceBuffer d_image        = image_buffer(tokens, columns);
    Tensor x(d_input.data(), DType::BF16, {columns, tokens});
    Tensor reference(d_reference.data(), DType::BF16, {rows, tokens});
    Tensor output(d_output.data(), DType::BF16, {rows, tokens});
    const DeviceSpan image{d_image.data(), d_image.size()};
    ninfer::ops::linear(x, weight.view, reference,
                        DeviceSpan{d_linear_image.data(), d_linear_image.size()}, device.stream);
    image_oracle::poison_status(image, tokens, columns, device.stream);
    ninfer::ops::quantize_q4_activation_image(x, image, device.stream);
    ninfer::ops::linear_q4_activation_image(weight.view, output, image, device.stream);
    device.synchronize();
    const std::string label = "projection N" + std::to_string(rows) + "/K" +
                              std::to_string(columns) + "/T" + std::to_string(tokens);
    image_oracle::require_image(
        image_oracle::read(image, tokens, columns),
        image_oracle::encode(image_oracle::represented(input), tokens, columns), label);
    if (!same_bits(download<hip_bfloat16>(d_output), download<hip_bfloat16>(d_reference)))
        fail(label + ": image projection differs from Linear");
}

void qualify_rmsnorm(ninfer::DeviceContext& device, int tokens) {
    constexpr int kColumns = 5120;
    const auto input       = normal_rows(static_cast<std::size_t>(kColumns) * tokens, 0.5F,
                                         static_cast<std::uint32_t>(tokens));
    auto norm              = normal_rows(kColumns, 0.2F, 99U);
    for (auto& gain : norm) gain = hip_bfloat16(1.0F + static_cast<float>(gain));
    auto d_input = upload(input);
    auto d_norm  = upload(norm);
    DeviceBuffer d_reference(input.size() * sizeof(hip_bfloat16));
    DeviceBuffer d_normalized(d_reference.size());
    DeviceBuffer d_image = image_buffer(tokens, kColumns);
    Tensor x(d_input.data(), DType::BF16, {kColumns, tokens});
    Tensor gains(d_norm.data(), DType::BF16, {kColumns});
    Tensor reference(d_reference.data(), DType::BF16, {kColumns, tokens});
    Tensor normalized(d_normalized.data(), DType::BF16, {kColumns, tokens});
    const DeviceSpan image{d_image.data(), d_image.size()};
    const std::string label = "rmsnorm image T" + std::to_string(tokens);
    ninfer::ops::rmsnorm(x, gains, 1.0e-6F, false, reference, device.stream);
    image_oracle::poison_status(image, tokens, kColumns, device.stream);
    ninfer::ops::rmsnorm_q4_activation_image(x, gains, 1.0e-6F, false, normalized, image,
                                             device.stream);
    device.synchronize();
    const auto published = download<hip_bfloat16>(d_normalized);
    // The small route is the eager row-CTA reduction (bit-identical seam); the wide route may
    // differ by one BF16 ulp from a different FP32 reduction order.
    if (tokens <= 12 && !same_bits(published, download<hip_bfloat16>(d_reference)))
        fail(label + ": normalized rows differ from ops::rmsnorm");
    image_oracle::require_image(
        image_oracle::read(image, tokens, kColumns),
        image_oracle::encode(image_oracle::represented(published), tokens, kColumns), label);
}

void qualify_silu(ninfer::DeviceContext& device, int tokens) {
    constexpr int kColumns = 17408;
    auto gate_up           = normal_rows(static_cast<std::size_t>(2 * kColumns) * tokens, 1.5F,
                                         static_cast<std::uint32_t>(tokens + 5));
    auto d_gate_up         = upload(gate_up);
    DeviceBuffer d_reference(static_cast<std::size_t>(kColumns) * tokens * sizeof(hip_bfloat16));
    DeviceBuffer d_image = image_buffer(tokens, kColumns);
    DeviceBuffer d_completion(2 * sizeof(std::uint32_t));
    d_completion.fill(0);
    Tensor packed(d_gate_up.data(), DType::BF16, {2 * kColumns, tokens});
    Tensor reference(d_reference.data(), DType::BF16, {kColumns, tokens});
    const DeviceSpan image{d_image.data(), d_image.size()};
    const std::string label = "SiLU image T" + std::to_string(tokens);
    const ninfer::ops::Q4ActivationImageTarget target{
        image, static_cast<std::uint32_t*>(d_completion.data())};
    const auto require_idle = [&] {
        const auto words = download<std::uint32_t>(d_completion);
        if (words[0] != 0U || words[1] != 0U) fail(label + ": completion words not idle");
    };
    ninfer::ops::silu_mul(packed.slice(0, 0, kColumns), packed.slice(0, kColumns, kColumns),
                          reference, device.stream);
    image_oracle::poison_status(image, tokens, kColumns, device.stream);
    ninfer::ops::silu_mul_q4_activation_image(packed, target, device.stream);
    device.synchronize();
    image_oracle::require_image(
        image_oracle::read(image, tokens, kColumns),
        image_oracle::encode(image_oracle::represented(download<hip_bfloat16>(d_reference)), tokens,
                             kColumns),
        label);
    require_idle();
    gate_up[static_cast<std::size_t>(tokens - 1) * 2 * kColumns + kColumns + 3] =
        hip_bfloat16(std::numeric_limits<float>::quiet_NaN());
    d_gate_up.copy_from_host(gate_up.data(), d_gate_up.size());
    ninfer::ops::silu_mul_q4_activation_image(packed, target, device.stream);
    device.synchronize();
    if ((image_oracle::read(image, tokens, kColumns).status & 1U) == 0U)
        fail(label + ": a nonfinite value left the status word clear");
    require_idle();

    constexpr int iterations = 20;
    ninfer::DeviceEventTimer timer(device);
    timer.start();
    for (int iteration = 0; iteration < iterations; ++iteration)
        ninfer::ops::silu_mul_q4_activation_image(packed, target, device.stream);
    std::cout << "  " << label << " us=" << timer.stop_ms() / iterations * 1000.0F << '\n';
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        // The drafter's Q4 projections at the C1 K7 width, a status-CTA-limit width, and a
        // batched (reset-route) width.
        for (const int tokens : {8, 12, 16}) {
            qualify_projection(device, 1280, 5120, tokens);
            qualify_projection(device, 6144, 5120, tokens);
            qualify_projection(device, 5120, 4096, tokens);
            qualify_projection(device, 34816, 5120, tokens);
            qualify_projection(device, 5120, 17408, tokens);
            qualify_rmsnorm(device, tokens);
            qualify_silu(device, tokens);
        }
        qualify_projection(device, 6144, 5120, 1);
        qualify_projection(device, 6144, 5120, 384);
        std::cout << "q4_activation_image: PASS gfx1201 exact A8G64 images (quantize, RMSNorm "
                     "status CTA and reset routes, SiLU last-block status) and bit-identical image "
                     "projections "
                     "N1280/6144/34816 K5120, N5120 K4096/17408 at T1/8/12/16/384\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "q4_activation_image: FAIL: " << error.what() << '\n';
        return 1;
    }
}
