#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/linear.h"

#include "a8g64_image_oracle.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DeviceBuffer;
using ninfer::DeviceSpan;
using ninfer::DType;
using ninfer::Tensor;
namespace image_oracle = ninfer::tools::a8g64_image;

constexpr int kD          = ninfer::ops::kGroupedDynamicConvHidden;
constexpr int kGroups     = ninfer::ops::kGroupedDynamicConvGroups;
constexpr int kProjection = ninfer::ops::kGroupedDynamicConvProjRows;
constexpr float kEps      = 1.0e-6F;

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

float patterned(int first, int second, int modulus, float divisor) {
    const int centered = (first * 17 + second * 29) % modulus - modulus / 2;
    return static_cast<float>(centered) / divisor;
}

std::vector<hip_bfloat16> patterned_rows(int rows, int columns, int modulus, float divisor) {
    std::vector<hip_bfloat16> values(static_cast<std::size_t>(rows) * columns);
    for (int column = 0; column < columns; ++column)
        for (int row = 0; row < rows; ++row)
            values[static_cast<std::size_t>(column) * rows + row] =
                hip_bfloat16(patterned(column, row, modulus, divisor));
    return values;
}

double bf16_rne(double value) {
    return static_cast<float>(hip_bfloat16(static_cast<float>(value)));
}

struct Error {
    double max_abs     = 0.0;
    double relative_l2 = 0.0;
};

Error compare(const std::vector<hip_bfloat16>& actual, const std::vector<double>& expected,
              double absolute_limit, double relative_l2_limit, const std::string& label) {
    if (actual.size() != expected.size()) fail(label + ": extent mismatch");
    double squared_error = 0.0, squared_reference = 0.0;
    Error result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double got   = static_cast<float>(actual[index]);
        const double error = got - expected[index];
        if (!std::isfinite(got) || !std::isfinite(expected[index]))
            fail(label + ": nonfinite oracle/output");
        result.max_abs = std::max(result.max_abs, std::abs(error));
        squared_error += error * error;
        squared_reference += expected[index] * expected[index];
    }
    result.relative_l2 = std::sqrt(squared_error / std::max(1.0e-30, squared_reference));
    if (result.max_abs > absolute_limit || result.relative_l2 > relative_l2_limit)
        fail(label + ": FP64 oracle mismatch max_abs=" + std::to_string(result.max_abs) +
             " rel_l2=" + std::to_string(result.relative_l2));
    return result;
}

void merge(Error& total, const Error& error) {
    total.max_abs     = std::max(total.max_abs, error.max_abs);
    total.relative_l2 = std::max(total.relative_l2, error.relative_l2);
}

struct Case {
    int tokens  = 1;
    int batch   = 1;
    int columns = 1;
    std::vector<hip_bfloat16> hidden;     // [columns][kD]
    std::vector<hip_bfloat16> base;       // [2 phases][2 offsets][kD]
    std::vector<hip_bfloat16> projection; // [columns][1280]
    std::vector<hip_bfloat16> dynamic;    // [columns][2][320] (finish stash)
    std::vector<hip_bfloat16> residual;   // [columns][kD]
    std::vector<hip_bfloat16> norm;       // [kD]
};

Case make_case(int tokens, int batch) {
    Case value;
    value.tokens     = tokens;
    value.batch      = batch;
    value.columns    = tokens * batch;
    value.hidden     = patterned_rows(kD, value.columns, 37, 256.0F);
    value.base       = patterned_rows(kD, 4, 23, 1024.0F);
    value.projection = patterned_rows(kProjection, value.columns, 41, 128.0F);
    value.dynamic    = patterned_rows(2 * kGroups, value.columns, 17, 256.0F);
    value.residual   = patterned_rows(kD, value.columns, 23, 64.0F);
    value.norm       = patterned_rows(kD, 1, 29, 32.0F);
    for (auto& gain : value.norm) gain = hip_bfloat16(1.0F + static_cast<float>(gain));
    return value;
}

float at(const std::vector<hip_bfloat16>& rows, int rows_per_column, int column, int row) {
    return static_cast<float>(rows[static_cast<std::size_t>(column) * rows_per_column + row]);
}

// FP64 conv of phase `phase` with the given dynamic kernels per (column, offset, group).
template <class Dynamic>
std::vector<double> conv_oracle(const Case& value, int phase, Dynamic dynamic) {
    std::vector<double> out(static_cast<std::size_t>(kD) * value.columns);
    for (int column = 0; column < value.columns; ++column) {
        const int token = column % value.tokens;
        for (int d = 0; d < kD; ++d) {
            const int group       = d / 16;
            const double current  = at(value.hidden, kD, column, d);
            const double previous = token == 0 ? 0.0 : at(value.hidden, kD, column - 1, d);
            const double base0    = at(value.base, kD, phase * 2, d);
            const double base1    = at(value.base, kD, phase * 2 + 1, d);
            out[static_cast<std::size_t>(column) * kD + d] =
                (base0 + dynamic(column, 0, group)) * current +
                (base1 + dynamic(column, 1, group)) * previous;
        }
    }
    return out;
}

DeviceBuffer image_buffer(int columns) {
    DeviceBuffer buffer(ninfer::ops::q4_activation_image_bytes(columns, kD));
    buffer.fill(0);
    return buffer;
}

DeviceSpan span(const DeviceBuffer& buffer) { return {buffer.data(), buffer.size()}; }

// An image target with its own zeroed completion words.
struct Target {
    DeviceBuffer image;
    DeviceBuffer completion{2 * sizeof(std::uint32_t)};
    ninfer::ops::Q4ActivationImageTarget view;

    explicit Target(int columns) : image(image_buffer(columns)) {
        completion.fill(0);
        view = {span(image), static_cast<std::uint32_t*>(completion.data())};
    }
};

// Every launch must return the completion words to zero.
void require_completion_idle(const Target& target, const std::string& label) {
    const auto words = download<std::uint32_t>(target.completion);
    if (words[0] != 0U || words[1] != 0U) fail(label + ": completion words not returned to zero");
}

Error run_prepare(ninfer::DeviceContext& device, int tokens, int batch, bool graph) {
    const Case value  = make_case(tokens, batch);
    const int columns = value.columns;
    auto d_hidden     = upload(value.hidden);
    auto d_base       = upload(value.base);
    auto d_projection = upload(value.projection);
    DeviceBuffer d_prepared(value.hidden.size() * sizeof(hip_bfloat16));
    DeviceBuffer d_finish(static_cast<std::size_t>(2) * kGroups * columns * sizeof(hip_bfloat16));
    Target target(columns);
    DeviceBuffer& d_image = target.image;
    Tensor hidden(d_hidden.data(), DType::BF16, {kD, tokens, batch});
    Tensor base(d_base.data(), DType::BF16, {kD, 2, 2});
    Tensor projection(d_projection.data(), DType::BF16, {kProjection, tokens, batch});
    Tensor prepared(d_prepared.data(), DType::BF16, {kD, tokens, batch});
    Tensor finish(d_finish.data(), DType::BF16, {kGroups, 2, tokens, batch});
    const DeviceSpan image = span(d_image);

    const std::vector<double> expected =
        conv_oracle(value, 0, [&](int column, int offset, int group) {
            return static_cast<double>(
                at(value.projection, kProjection, column, offset * kGroups + group));
        });
    std::vector<double> expected_finish(static_cast<std::size_t>(2) * kGroups * columns);
    for (int column = 0; column < columns; ++column)
        for (int index = 0; index < 2 * kGroups; ++index)
            expected_finish[static_cast<std::size_t>(column) * 2 * kGroups + index] =
                at(value.projection, kProjection, column, 2 * kGroups + index);

    const auto check = [&](const std::string& label) {
        const auto actual = download<hip_bfloat16>(d_prepared);
        Error error       = compare(actual, expected, 0.02, 0.005, label + " prepared");
        merge(error, compare(download<hip_bfloat16>(d_finish), expected_finish, 0.0, 0.0,
                             label + " finish_dynamic"));
        image_oracle::require_image(
            image_oracle::read(image, static_cast<std::uint32_t>(columns), kD),
            image_oracle::encode(image_oracle::represented(actual),
                                 static_cast<std::uint32_t>(columns), kD),
            label);
        require_completion_idle(target, label);
        return error;
    };
    image_oracle::poison_status(image, columns, kD, device.stream);
    ninfer::ops::grouped_dynamic_conv_prepare(hidden, base, projection, &prepared, finish,
                                              &target.view, device.stream);
    device.synchronize();
    Error result = check("prepare T" + std::to_string(tokens) + "/B" + std::to_string(batch));
    // Image only: the same image without a BF16 publication.
    const auto published = image_oracle::read(image, columns, kD);
    d_image.fill(0);
    ninfer::ops::grouped_dynamic_conv_prepare(hidden, base, projection, nullptr, finish,
                                              &target.view, device.stream);
    device.synchronize();
    image_oracle::require_image(image_oracle::read(image, columns, kD), published,
                                "prepare image-only");
    if (graph) {
        hipGraph_t captured{};
        hipGraphExec_t executable{};
        HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeThreadLocal));
        ninfer::ops::grouped_dynamic_conv_prepare(hidden, base, projection, &prepared, finish,
                                                  &target.view, device.stream);
        HIP_CHECK(hipStreamEndCapture(device.stream, &captured));
        HIP_CHECK(hipGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            d_prepared.fill_async(0xff, device.stream);
            d_finish.fill_async(0xff, device.stream);
            HIP_CHECK(hipMemsetAsync(d_image.data(), 0xff, d_image.size(), device.stream));
            HIP_CHECK(hipGraphLaunch(executable, device.stream));
            device.synchronize();
            merge(result, check("prepare graph"));
        }
        HIP_CHECK(hipGraphExecDestroy(executable));
        HIP_CHECK(hipGraphDestroy(captured));
    }
    // A nonfinite activation sets the status word in both status modes.
    std::vector<hip_bfloat16> poisoned = value.hidden;
    poisoned[static_cast<std::size_t>(columns - 1) * kD + 77] =
        hip_bfloat16(std::numeric_limits<float>::quiet_NaN());
    d_hidden.copy_from_host(poisoned.data(), d_hidden.size());
    ninfer::ops::grouped_dynamic_conv_prepare(hidden, base, projection, nullptr, finish,
                                              &target.view, device.stream);
    device.synchronize();
    if ((image_oracle::read(image, columns, kD).status & 1U) == 0U)
        fail("prepare: a nonfinite value left the status word clear");
    return result;
}

Error run_finish(ninfer::DeviceContext& device) {
    const Case value             = make_case(4, 2);
    std::vector<double> expected = conv_oracle(value, 1, [&](int column, int offset, int group) {
        return static_cast<double>(
            at(value.dynamic, 2 * kGroups, column, offset * kGroups + group));
    });
    for (std::size_t index = 0; index < expected.size(); ++index)
        expected[index] += static_cast<float>(value.residual[index]);
    auto d_hidden  = upload(value.hidden);
    auto d_base    = upload(value.base);
    auto d_dynamic = upload(value.dynamic);
    auto d_output  = upload(value.residual);
    Tensor hidden(d_hidden.data(), DType::BF16, {kD, 4, 2});
    Tensor base(d_base.data(), DType::BF16, {kD, 2, 2});
    Tensor dynamic(d_dynamic.data(), DType::BF16, {kGroups, 2, 4, 2});
    Tensor output(d_output.data(), DType::BF16, {kD, 4, 2});
    ninfer::ops::grouped_dynamic_conv_finish(hidden, base, dynamic, output, device.stream);
    device.synchronize();
    return compare(download<hip_bfloat16>(d_output), expected, 0.02, 0.005, "finish");
}

Error run_finish_normalized(ninfer::DeviceContext& device, int tokens, int batch, bool graph) {
    const Case value  = make_case(tokens, batch);
    const int columns = value.columns;
    // r = BF16(residual + BF16(conv)), the represented finished residual (FP64 conv).
    const std::vector<double> conv = conv_oracle(value, 1, [&](int column, int offset, int group) {
        return static_cast<double>(
            at(value.dynamic, 2 * kGroups, column, offset * kGroups + group));
    });
    std::vector<double> expected_residual(conv.size());
    for (std::size_t index = 0; index < conv.size(); ++index)
        expected_residual[index] =
            bf16_rne(static_cast<float>(value.residual[index]) + bf16_rne(conv[index]));
    std::vector<double> expected_normalized(conv.size());
    for (int column = 0; column < columns; ++column) {
        double sumsq = 0.0;
        for (int d = 0; d < kD; ++d) {
            const double r = expected_residual[static_cast<std::size_t>(column) * kD + d];
            sumsq += r * r;
        }
        const double inverse = 1.0 / std::sqrt(sumsq / kD + kEps);
        for (int d = 0; d < kD; ++d) {
            const std::size_t index = static_cast<std::size_t>(column) * kD + d;
            expected_normalized[index] =
                expected_residual[index] * inverse * static_cast<float>(value.norm[d]);
        }
    }
    auto d_hidden   = upload(value.hidden);
    auto d_base     = upload(value.base);
    auto d_dynamic  = upload(value.dynamic);
    auto d_residual = upload(value.residual);
    auto d_norm     = upload(value.norm);
    DeviceBuffer d_out(value.residual.size() * sizeof(hip_bfloat16));
    DeviceBuffer d_normalized(value.residual.size() * sizeof(hip_bfloat16));
    Target target(columns);
    DeviceBuffer& d_image = target.image;
    Tensor hidden(d_hidden.data(), DType::BF16, {kD, tokens, batch});
    Tensor base(d_base.data(), DType::BF16, {kD, 2, 2});
    Tensor dynamic(d_dynamic.data(), DType::BF16, {kGroups, 2, tokens, batch});
    Tensor residual(d_residual.data(), DType::BF16, {kD, tokens, batch});
    Tensor residual_out(d_out.data(), DType::BF16, {kD, tokens, batch});
    Tensor norm(d_norm.data(), DType::BF16, {kD});
    Tensor normalized(d_normalized.data(), DType::BF16, {kD, tokens, batch});
    const DeviceSpan image = span(d_image);
    const std::string label =
        "finish_normalized T" + std::to_string(tokens) + "/B" + std::to_string(batch);
    const auto check = [&](const std::string& suffix) {
        Error error       = compare(download<hip_bfloat16>(d_out), expected_residual, 0.02, 0.005,
                                    label + suffix + " residual");
        const auto actual = download<hip_bfloat16>(d_normalized);
        merge(error,
              compare(actual, expected_normalized, 0.06, 0.005, label + suffix + " normalized"));
        image_oracle::require_image(
            image_oracle::read(image, static_cast<std::uint32_t>(columns), kD),
            image_oracle::encode(image_oracle::represented(actual),
                                 static_cast<std::uint32_t>(columns), kD),
            label + suffix);
        if (download<hip_bfloat16>(d_residual) != value.residual)
            fail(label + suffix + ": residual input changed");
        require_completion_idle(target, label + suffix);
        return error;
    };
    image_oracle::poison_status(image, columns, kD, device.stream);
    ninfer::ops::grouped_dynamic_conv_finish_normalized(hidden, base, dynamic, residual,
                                                        residual_out, norm, kEps, normalized,
                                                        &target.view, device.stream);
    device.synchronize();
    Error result = check("");
    // Without an image the BF16 outputs are unchanged.
    const auto published_residual   = download<hip_bfloat16>(d_out);
    const auto published_normalized = download<hip_bfloat16>(d_normalized);
    d_out.fill(0);
    d_normalized.fill(0);
    ninfer::ops::grouped_dynamic_conv_finish_normalized(hidden, base, dynamic, residual,
                                                        residual_out, norm, kEps, normalized,
                                                        nullptr, device.stream);
    device.synchronize();
    if (download<hip_bfloat16>(d_out) != published_residual ||
        download<hip_bfloat16>(d_normalized) != published_normalized)
        fail(label + ": BF16 outputs depend on the image request");
    if (graph) {
        hipGraph_t captured{};
        hipGraphExec_t executable{};
        HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeThreadLocal));
        ninfer::ops::grouped_dynamic_conv_finish_normalized(hidden, base, dynamic, residual,
                                                            residual_out, norm, kEps, normalized,
                                                            &target.view, device.stream);
        HIP_CHECK(hipStreamEndCapture(device.stream, &captured));
        HIP_CHECK(hipGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            d_out.fill_async(0xff, device.stream);
            d_normalized.fill_async(0xff, device.stream);
            HIP_CHECK(hipMemsetAsync(d_image.data(), 0xff, d_image.size(), device.stream));
            HIP_CHECK(hipGraphLaunch(executable, device.stream));
            device.synchronize();
            merge(result, check(" graph"));
        }
        HIP_CHECK(hipGraphExecDestroy(executable));
        HIP_CHECK(hipGraphDestroy(captured));
    }
    std::vector<hip_bfloat16> poisoned = value.residual;
    poisoned[static_cast<std::size_t>(columns - 1) * kD + 4099] =
        hip_bfloat16(std::numeric_limits<float>::infinity());
    d_residual.copy_from_host(poisoned.data(), d_residual.size());
    ninfer::ops::grouped_dynamic_conv_finish_normalized(hidden, base, dynamic, residual,
                                                        residual_out, norm, kEps, normalized,
                                                        &target.view, device.stream);
    device.synchronize();
    if ((image_oracle::read(image, columns, kD).status & 1U) == 0U)
        fail(label + ": a nonfinite residual left the status word clear");
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1) fail("usage: grouped_dynamic_conv_qual");
        (void)argv;
        ninfer::DeviceContext device;
        Error total;
        // Up to the batched 32-column width; graph replays start from a stale status word.
        merge(total, run_prepare(device, 1, 1, false));
        merge(total, run_prepare(device, 8, 1, true));
        merge(total, run_prepare(device, 4, 2, false));
        merge(total, run_prepare(device, 16, 2, true));
        merge(total, run_finish(device));
        merge(total, run_finish_normalized(device, 8, 1, true));
        merge(total, run_finish_normalized(device, 3, 4, false));
        merge(total, run_finish_normalized(device, 16, 2, true));
        std::cout << "grouped_dynamic_conv: PASS gfx1201 D5120/G320 prepare T1/B1 T8/B1 T4/B2 "
                     "T16/B2, finish T4/B2, finish_normalized T8/B1 T3/B4 T16/B2; exact A8G64 "
                     "images, last-block status with idle completion words, graph replay, "
                     "nonfinite status"
                  << " max_abs=" << total.max_abs << " rel_l2=" << total.relative_l2 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "grouped_dynamic_conv: FAIL: " << error.what() << '\n';
        return 1;
    }
}
