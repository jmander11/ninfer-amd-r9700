#include <hip/hip_runtime.h>

#include "ops/r9700/eager/eager_ops.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr std::uint32_t kFeatures = 5120U;
constexpr std::size_t kGuard = 32U;
constexpr std::uint16_t kPoison = 0x5555U;
constexpr std::size_t kPairs = 7U;
constexpr int kIterations = 256;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream message;
    message << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(message.str());
}
#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

template <class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count, std::size_t offset = 0) : count_(count), offset_(offset) {
        HIP_CHECK(hipMalloc(&data_, (count + offset) * sizeof(T)));
    }
    ~DeviceBuffer() { (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const noexcept { return data_ + offset_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
private:
    T* data_ = nullptr;
    std::size_t count_;
    std::size_t offset_;
};

std::uint16_t bf16_bits(float value) {
    std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    if ((word & 0x7f800000U) != 0x7f800000U) word += 0x7fffU + ((word >> 16U) & 1U);
    else if ((word & 0xffffU) != 0U) word |= 0x10000U;
    return static_cast<std::uint16_t>(word >> 16U);
}
hip_bfloat16 bf16(float value) { hip_bfloat16 result; result.data = bf16_bits(value); return result; }
std::uint32_t ordered(std::uint16_t bits) {
    return (bits & 0x8000U) ? 0xffffU - bits : static_cast<std::uint32_t>(bits) + 0x8000U;
}
std::uint32_t steps(std::uint16_t a, std::uint16_t b) {
    const auto x = ordered(a), y = ordered(b); return x > y ? x - y : y - x;
}
std::uint64_t round_ties_even(double value) {
    const double base = std::floor(value);
    const double tail = value - base;
    auto integer = static_cast<std::uint64_t>(base);
    if (tail > 0.5 || (tail == 0.5 && (integer & 1U) != 0U)) ++integer;
    return integer;
}
std::uint16_t bf16_bits_direct(double value) {
    if (!std::isfinite(value)) fail("nonfinite FP64 oracle value");
    const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
    const double magnitude = std::abs(value);
    if (magnitude == 0.0) return sign;
    int exponent = 0;
    const double fraction = std::frexp(magnitude, &exponent);
    int biased = exponent - 1 + 127;
    if (biased <= 0) {
        const auto mantissa = round_ties_even(std::ldexp(magnitude, 133));
        if (mantissa == 0U) return sign;
        if (mantissa >= 128U) return static_cast<std::uint16_t>(sign | 0x0080U);
        return static_cast<std::uint16_t>(sign | mantissa);
    }
    auto significand = round_ties_even(std::ldexp(fraction, 8));
    if (significand == 256U) { significand = 128U; ++biased; }
    if (biased >= 255) return static_cast<std::uint16_t>(sign | 0x7f80U);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(biased) << 7U) |
                                      static_cast<std::uint16_t>(significand - 128U));
}

template <class T> void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() < source.size()) fail("upload extent mismatch");
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}
template <class T> std::vector<T> download(const DeviceBuffer<T>& source) {
    std::vector<T> result(source.size());
    HIP_CHECK(hipMemcpy(result.data(), source.get(), result.size() * sizeof(T),
                        hipMemcpyDeviceToHost));
    return result;
}

void select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t property{};
        HIP_CHECK(hipGetDeviceProperties(&property, ordinal));
        if (!property.integrated && std::string_view(property.gcnArchName) == "gfx1201") {
            if (selected >= 0) fail("expected exactly one discrete gfx1201 device");
            selected = ordinal;
        }
    }
    if (selected != 0) fail("the sole discrete gfx1201 device must be HIP ordinal 0");
    HIP_CHECK(hipSetDevice(selected));
    hipDeviceProp_t property{};
    HIP_CHECK(hipGetDeviceProperties(&property, selected));
    if (std::string_view(property.name).find("R9700") == std::string_view::npos)
        fail("HIP device 0 is not an AMD Radeon AI PRO R9700");
}

std::vector<hip_bfloat16> make_input(std::uint32_t rows, std::uint32_t fixture) {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(rows) * kFeatures);
    constexpr std::array<float, 8> edge{0.0F, -0.0F, 0x1p-14F, -0x1p-14F,
                                        0.03125F, -0.5F, 8.0F, -31.75F};
    for (std::size_t i = 0; i < result.size(); ++i) {
        float value = 0.0F;
        if (fixture == 0U) {
            value = static_cast<float>(static_cast<std::int32_t>((i * 37U + 53U) % 509U) - 254) /
                    127.0F;
        } else if (fixture == 1U) {
            value = edge[i % edge.size()];
        } else {
            value = static_cast<float>(static_cast<std::int32_t>((i * 97U + 11U) % 8191U) - 4095) /
                    64.0F;
        }
        result[i] = bf16(value);
    }
    return result;
}
std::vector<hip_bfloat16> make_weight() {
    std::vector<hip_bfloat16> result(kFeatures);
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = bf16(static_cast<float>(static_cast<std::int32_t>((i * 29U + 17U) % 257U) - 128) /
                         509.0F);
    return result;
}

void poison(DeviceBuffer<hip_bfloat16>& storage, hipStream_t stream) {
    HIP_CHECK(hipMemsetAsync(storage.get(), 0x55, storage.size() * sizeof(hip_bfloat16), stream));
}
std::vector<hip_bfloat16> checked(DeviceBuffer<hip_bfloat16>& storage, std::size_t elements) {
    const auto raw = download(storage);
    if (raw.size() != elements + 2U * kGuard) fail("guarded extent mismatch");
    for (std::size_t i = 0; i < kGuard; ++i) {
        if (raw[i].data != kPoison || raw[kGuard + elements + i].data != kPoison)
            fail("output canary was clobbered");
    }
    return {raw.begin() + static_cast<std::ptrdiff_t>(kGuard),
            raw.begin() + static_cast<std::ptrdiff_t>(kGuard + elements)};
}

std::uint32_t require_oracle(const std::vector<hip_bfloat16>& actual,
                             const std::vector<hip_bfloat16>& input,
                             const std::vector<hip_bfloat16>& weight,
                             std::uint32_t rows, double eps, bool unit_offset) {
    std::uint32_t maximum = 0;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kFeatures;
        double sum = 0.0;
        for (std::uint32_t column = 0; column < kFeatures; ++column) {
            const double value = static_cast<float>(input[base + column]);
            sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum / kFeatures + eps);
        for (std::uint32_t column = 0; column < kFeatures; ++column) {
            const std::size_t i = base + column;
            const double expected = static_cast<float>(input[i]) * inverse *
                                    (static_cast<float>(weight[column]) +
                                     (unit_offset ? 1.0 : 0.0));
            const double observed = static_cast<float>(actual[i]);
            if (!std::isfinite(observed)) fail("nonfinite RMSNorm result");
            maximum = std::max(maximum, steps(actual[i].data, bf16_bits_direct(expected)));
        }
    }
    if (maximum > 2U) fail("route exceeds two-BF16-step FP64 oracle criterion");
    return maximum;
}

using Launch = hipError_t (*)(const hip_bfloat16*, const hip_bfloat16*, hip_bfloat16*,
                              std::uint32_t, float, bool, hipStream_t);
hipError_t incumbent(const hip_bfloat16* input, const hip_bfloat16* weight,
                     hip_bfloat16* output, std::uint32_t rows, float eps, bool unit_offset,
                     hipStream_t stream) {
    return ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
        input, weight, output, kFeatures, rows, eps, unit_offset, stream);
}
hipError_t challenger(const hip_bfloat16* input, const hip_bfloat16* weight,
                      hip_bfloat16* output, std::uint32_t rows, float eps, bool unit_offset,
                      hipStream_t stream) {
    return ninfer::ops::r9700::eager::rmsnorm_k5120_rows56_cta_qualification(
        input, weight, output, rows, eps, unit_offset, stream);
}

struct Numeric { std::uint32_t incumbent = 0; std::uint32_t challenger = 0; };
Numeric qualify(hipStream_t stream) {
    const auto weight = make_weight();
    DeviceBuffer<hip_bfloat16> d_weight(weight.size()); upload(d_weight, weight);
    Numeric result{};
    for (const auto rows : {5U, 6U}) for (const auto fixture : {0U, 1U, 2U}) {
        const auto input = make_input(rows, fixture);
        DeviceBuffer<hip_bfloat16> d_input(input.size()); upload(d_input, input);
        DeviceBuffer<hip_bfloat16> d_output(input.size() + 2U * kGuard);
        for (const bool unit_offset : {false, true}) for (const float eps : {1.0e-8F, 1.0e-6F, 1.0e-4F}) {
            for (const auto route : {Launch(incumbent), Launch(challenger)}) {
                poison(d_output, stream);
                HIP_CHECK(route(d_input.get(), d_weight.get(), d_output.get() + kGuard,
                                rows, eps, unit_offset, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                auto actual = checked(d_output, input.size());
                const auto error = require_oracle(actual, input, weight, rows, eps, unit_offset);
                (route == incumbent ? result.incumbent : result.challenger) =
                    std::max(route == incumbent ? result.incumbent : result.challenger, error);
            }
        }
    }

    // Both exact rows execute allocation-free under capture and replay from stable addresses.
    for (const auto rows : {5U, 6U}) for (const auto route : {Launch(incumbent), Launch(challenger)}) {
        const auto input = make_input(rows, 0U);
        DeviceBuffer<hip_bfloat16> d_input(input.size()); upload(d_input, input);
        DeviceBuffer<hip_bfloat16> d_output(input.size() + 2U * kGuard); poison(d_output, stream);
        hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        HIP_CHECK(route(d_input.get(), d_weight.get(), d_output.get() + kGuard,
                        rows, 1.0e-6F, true, stream));
        HIP_CHECK(hipStreamEndCapture(stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) HIP_CHECK(hipGraphLaunch(executable, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        (void)require_oracle(checked(d_output, input.size()), input, weight, rows, 1.0e-6, true);
        HIP_CHECK(hipGraphExecDestroy(executable)); HIP_CHECK(hipGraphDestroy(graph));
    }

    // The qualification boundary rejects malformed shape, status and exact-pointer aliases.
    const auto input = make_input(6U, 0U);
    DeviceBuffer<hip_bfloat16> d_input(input.size()); upload(d_input, input);
    DeviceBuffer<hip_bfloat16> d_output(input.size() + 2U * kGuard); poison(d_output, stream);
    HIP_CHECK(hipStreamSynchronize(stream)); const auto before = download(d_output);
    const auto input_before = download(d_input); const auto weight_before = download(d_weight);
    const auto expect_reject = [&](const hip_bfloat16* in, const hip_bfloat16* gain,
                                   hip_bfloat16* out, std::uint32_t rows, float eps,
                                   hipStream_t candidate_stream) {
        if (challenger(in, gain, out, rows, eps, true, candidate_stream) != hipErrorInvalidValue)
            fail("challenger accepted invalid arguments");
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto unchanged = [](const auto& lhs, const auto& rhs) {
            if (lhs.size() != rhs.size()) return false;
            for (std::size_t i = 0; i < lhs.size(); ++i)
                if (lhs[i].data != rhs[i].data) return false;
            return true;
        };
        if (!unchanged(download(d_output), before) || !unchanged(download(d_input), input_before) ||
            !unchanged(download(d_weight), weight_before))
            fail("rejected call changed guarded or aliased storage");
    };
    auto* out = d_output.get() + kGuard;
    expect_reject(nullptr, d_weight.get(), out, 6U, 1.0e-6F, stream);
    expect_reject(d_input.get(), nullptr, out, 6U, 1.0e-6F, stream);
    expect_reject(d_input.get(), d_weight.get(), nullptr, 6U, 1.0e-6F, stream);
    expect_reject(d_input.get(), d_weight.get(), out, 4U, 1.0e-6F, stream);
    expect_reject(d_input.get(), d_weight.get(), out, 7U, 1.0e-6F, stream);
    expect_reject(d_input.get(), d_weight.get(), out, 6U, 0.0F, stream);
    expect_reject(d_input.get(), d_weight.get(), out, 6U,
                  std::numeric_limits<float>::quiet_NaN(), stream);
    expect_reject(d_input.get(), d_weight.get(), out, 6U, 1.0e-6F, nullptr);
    expect_reject(d_input.get(), d_weight.get(), d_input.get(), 6U, 1.0e-6F, stream);
    expect_reject(d_input.get(), d_weight.get(), d_weight.get(), 6U, 1.0e-6F, stream);
    expect_reject(d_input.get(), d_weight.get(), d_input.get() + 1, 6U, 1.0e-6F, stream);
    expect_reject(d_input.get(), d_weight.get(), d_weight.get() + 1, 6U, 1.0e-6F, stream);
    DeviceBuffer<hip_bfloat16> d_alias(2U * input.size() + 2U * kFeatures);
    HIP_CHECK(hipMemsetAsync(d_alias.get(), 0x3c, d_alias.size() * sizeof(hip_bfloat16), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto alias_before = download(d_alias);
    const auto expect_partial_alias_reject = [&](const hip_bfloat16* in,
                                                 const hip_bfloat16* gain,
                                                 hip_bfloat16* alias_out) {
        if (challenger(in, gain, alias_out, 6U, 1.0e-6F, true, stream) != hipErrorInvalidValue)
            fail("challenger accepted a partial overlap");
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto after = download(d_alias);
        for (std::size_t i = 0; i < after.size(); ++i)
            if (after[i].data != alias_before[i].data) fail("rejected partial overlap wrote storage");
    };
    auto* const alias = d_alias.get();
    expect_partial_alias_reject(alias, d_weight.get(), alias + 1);             // output after input
    expect_partial_alias_reject(alias + 1, d_weight.get(), alias);             // output before input
    expect_partial_alias_reject(alias, d_weight.get(), alias + input.size() - 1U);
    expect_partial_alias_reject(alias + input.size() - 1U, d_weight.get(), alias);
    expect_partial_alias_reject(d_input.get(), alias, alias + 1);              // output after weight
    expect_partial_alias_reject(d_input.get(), alias + 1, alias);              // output before weight
    expect_partial_alias_reject(d_input.get(), alias, alias + kFeatures - 1U);
    expect_partial_alias_reject(d_input.get(), alias + input.size() - 1U, alias);
    return result;
}

float measure(hipStream_t stream, Launch route, const DeviceBuffer<hip_bfloat16>& input,
              const DeviceBuffer<hip_bfloat16>& weight, DeviceBuffer<hip_bfloat16>& output,
              std::uint32_t rows) {
    hipEvent_t start = nullptr, stop = nullptr;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < kIterations; ++i)
        HIP_CHECK(route(input.get(), weight.get(), output.get(), rows, 1.0e-6F, true, stream));
    HIP_CHECK(hipEventRecord(stop, stream)); HIP_CHECK(hipEventSynchronize(stop));
    float elapsed = 0.0F; HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    return elapsed / kIterations;
}

struct Cell {
    std::uint32_t rows;
    std::array<float, kPairs> incumbent_forward{}, challenger_forward{}, challenger_reverse{}, incumbent_reverse{};
};
void write_array(std::ostream& out, const auto& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}
void benchmark(hipStream_t stream, const std::filesystem::path& path, Numeric numeric) {
    const auto weight = make_weight(); DeviceBuffer<hip_bfloat16> d_weight(weight.size()); upload(d_weight, weight);
    std::array<Cell, 2> cells{{{5U}, {6U}}};
    for (auto& cell : cells) {
        const auto input = make_input(cell.rows, 0U); DeviceBuffer<hip_bfloat16> d_input(input.size()); upload(d_input, input);
        DeviceBuffer<hip_bfloat16> d_output(input.size());
        for (int warmup = 0; warmup < 8; ++warmup) {
            HIP_CHECK(incumbent(d_input.get(), d_weight.get(), d_output.get(), cell.rows, 1.0e-6F, true, stream));
            HIP_CHECK(challenger(d_input.get(), d_weight.get(), d_output.get(), cell.rows, 1.0e-6F, true, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        for (std::size_t i = 0; i < kPairs; ++i) {
            cell.incumbent_forward[i] = measure(stream, incumbent, d_input, d_weight, d_output, cell.rows);
            cell.challenger_forward[i] = measure(stream, challenger, d_input, d_weight, d_output, cell.rows);
            cell.challenger_reverse[i] = measure(stream, challenger, d_input, d_weight, d_output, cell.rows);
            cell.incumbent_reverse[i] = measure(stream, incumbent, d_input, d_weight, d_output, cell.rows);
        }
    }
    std::ostringstream out; out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "{\n\"schema\":\"ninfer.r9700.rmsnorm-k5120-rows56-cell.v1\","
           "\n\"production_dispatch_changed\":false,"
           "\n\"numeric\":{\"oracle\":\"independent FP64 represented-BF16 RMSNorm formula\","
        << "\"maximum_bf16_steps_allowed\":2,\"incumbent_maximum_bf16_steps\":" << numeric.incumbent
        << ",\"challenger_maximum_bf16_steps\":" << numeric.challenger << "},\n\"cells\":[\n";
    for (std::size_t c = 0; c < cells.size(); ++c) {
        const auto& cell = cells[c]; out << "{\"features\":5120,\"rows\":" << cell.rows
            << ",\"unit_offset\":true,\"epsilon\":9.99999997e-7,\"iterations_per_sample\":256,"
               "\"forward_reverse_pairs\":7,\"incumbent_forward_ms\":";
        write_array(out, cell.incumbent_forward); out << ",\"challenger_forward_ms\":";
        write_array(out, cell.challenger_forward); out << ",\"challenger_reverse_ms\":";
        write_array(out, cell.challenger_reverse); out << ",\"incumbent_reverse_ms\":";
        write_array(out, cell.incumbent_reverse); out << '}' << (c + 1 == cells.size() ? "\n" : ",\n");
    }
    out << "]\n}\n"; const auto payload = out.str();
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("refusing existing, symlink, or unavailable report path");
    std::size_t written = 0;
    while (written < payload.size()) {
        const auto count = ::write(descriptor, payload.data() + written, payload.size() - written);
        if (count <= 0) { (void)::close(descriptor); fail("report write failed"); }
        written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0) fail("report close failed");
}
hipError_t exact_token8(const hip_bfloat16* input, const hip_bfloat16* weight,
                       hip_bfloat16* output, std::uint32_t rows, float eps, bool unit_offset,
                       hipStream_t stream) {
    return ninfer::ops::r9700::eager::rmsnorm_k5120_token8_qualification(
        input, weight, output, rows, eps, unit_offset, stream);
}

hipError_t exact_token8_packed_epilogue(const hip_bfloat16* input, const hip_bfloat16* weight,
                               hip_bfloat16* output, std::uint32_t rows, float eps,
                               bool unit_offset, hipStream_t stream) {
    return ninfer::ops::r9700::eager::rmsnorm_k5120_token8_packed_epilogue_qualification(
        input, weight, output, rows, eps, unit_offset, stream);
}

void screen_exact_token8(const std::filesystem::path& path) {
    const std::filesystem::path pci("/sys/bus/pci/devices/0000:13:00.0");
    const auto read = [](const auto& file) {
        std::ifstream input(file); std::string value;
        if (!(input >> value)) fail("cannot read PCI qualification attribute");
        return value;
    };
    const auto power = [&] {
        if (read(pci / "power_dpm_force_performance_level") != "auto")
            fail("exact token8 screen requires auto power");
    };
    if (std::filesystem::exists(path) || std::filesystem::is_symlink(path))
        fail("refusing existing exact token8 evidence");
    if (read(pci / "vendor") != "0x1002" || read(pci / "device") != "0x7551")
        fail("exact token8 screen requires the selected R9700 PCI function");
    power(); select_r9700();
    char bus[32]{}; HIP_CHECK(hipDeviceGetPCIBusId(bus, sizeof(bus), 0));
    if (std::string_view(bus) != "0000:13:00.0") fail("HIP/PCI function mismatch");
    hipStream_t stream{}; HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    const auto weight = make_weight();
    Numeric numeric{};
    std::uint32_t packed_epilogue_error = 0;
    struct EpilogueCell {
        std::uint32_t rows;
        std::size_t offset;
        std::array<float, kPairs> incumbent_forward{}, challenger_forward{},
            packed_epilogue_forward{}, packed_epilogue_reverse{}, challenger_reverse{}, incumbent_reverse{};
    };
    std::array<EpilogueCell, 4> cells{{{5U, 0U}, {6U, 0U}, {5U, 1U}, {6U, 1U}}};
    const auto equal = [](const auto& actual, const auto& expected) {
        if (actual.size() != expected.size()) fail("exact output extent mismatch");
        for (std::size_t i = 0; i < actual.size(); ++i)
            if (actual[i].data != expected[i].data)
                fail("token8 differs from serial FP32-order incumbent at " + std::to_string(i));
    };
    for (auto& cell : cells) {
        DeviceBuffer<hip_bfloat16> d_weight(weight.size(), cell.offset);
        HIP_CHECK(hipMemcpyAsync(d_weight.get(), weight.data(), weight.size() * sizeof(hip_bfloat16),
                                hipMemcpyHostToDevice, stream));
        for (const auto fixture : {0U, 1U, 2U}) {
            const auto input = make_input(cell.rows, fixture);
            DeviceBuffer<hip_bfloat16> d_input(input.size(), cell.offset);
            DeviceBuffer<hip_bfloat16> d_output(input.size() + 2U * kGuard, cell.offset);
            for (const auto* pointer : {d_input.get(), d_weight.get(), d_output.get() + kGuard})
                if (reinterpret_cast<std::uintptr_t>(pointer) % 16U != cell.offset * sizeof(hip_bfloat16))
                    fail("fixture does not have the requested aligned or exactly two-byte alignment");
            HIP_CHECK(hipMemcpyAsync(d_input.get(), input.data(), input.size() * sizeof(hip_bfloat16),
                                    hipMemcpyHostToDevice, stream));
            for (const bool gain : {false, true}) for (const float eps : {1e-8F, 1e-6F, 1e-4F}) {
                std::vector<hip_bfloat16> serial;
                for (const auto route : {Launch(incumbent), Launch(exact_token8), Launch(exact_token8_packed_epilogue)}) {
                    poison(d_output, stream);
                    HIP_CHECK(route(d_input.get(), d_weight.get(), d_output.get() + kGuard,
                                    cell.rows, eps, gain, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                    const auto actual = checked(d_output, input.size());
                    const auto error = require_oracle(actual, input, weight, cell.rows, eps, gain);
                    if (route == incumbent) {
                        serial = actual; numeric.incumbent = std::max(numeric.incumbent, error);
                    } else {
                        equal(actual, serial);
                        if (route == exact_token8)
                            numeric.challenger = std::max(numeric.challenger, error);
                        else
                            packed_epilogue_error = std::max(packed_epilogue_error, error);
                    }
                    if (fixture == 0U && eps == 1e-6F) {
                        hipGraph_t graph{}; hipGraphExec_t executable{};
                        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
                        HIP_CHECK(route(d_input.get(), d_weight.get(), d_output.get() + kGuard,
                                        cell.rows, eps, gain, stream));
                        HIP_CHECK(hipStreamEndCapture(stream, &graph));
                        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                        for (int replay = 0; replay < 2; ++replay) {
                            poison(d_output, stream); HIP_CHECK(hipGraphLaunch(executable, stream));
                            HIP_CHECK(hipStreamSynchronize(stream));
                            equal(checked(d_output, input.size()), serial);
                        }
                        HIP_CHECK(hipGraphExecDestroy(executable)); HIP_CHECK(hipGraphDestroy(graph));
                    }
                }
            }
            equal(download(d_input), input); equal(download(d_weight), weight);
            if (fixture != 0U) continue;
            power();
            for (int warm = 0; warm < 8; ++warm)
                for (auto route : {Launch(incumbent), Launch(exact_token8), Launch(exact_token8_packed_epilogue)})
                    HIP_CHECK(route(d_input.get(), d_weight.get(), d_output.get(), cell.rows, 1e-6F, true, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            for (std::size_t i = 0; i < kPairs; ++i) {
                cell.incumbent_forward[i] = measure(stream, incumbent, d_input, d_weight, d_output, cell.rows);
                cell.challenger_forward[i] = measure(stream, exact_token8, d_input, d_weight, d_output, cell.rows);
                cell.packed_epilogue_forward[i] = measure(stream, exact_token8_packed_epilogue, d_input, d_weight, d_output, cell.rows);
                cell.packed_epilogue_reverse[i] = measure(stream, exact_token8_packed_epilogue, d_input, d_weight, d_output, cell.rows);
                cell.challenger_reverse[i] = measure(stream, exact_token8, d_input, d_weight, d_output, cell.rows);
                cell.incumbent_reverse[i] = measure(stream, incumbent, d_input, d_weight, d_output, cell.rows);
            }
            power();
        }
    }
    HIP_CHECK(hipStreamDestroy(stream)); power();
    std::ostringstream out; out << std::setprecision(9);
    out << "{\"schema\":\"ninfer.r9700.rmsnorm-exact-token8-screen.v3\","
           "\"production_dispatch_changed\":false,\"all_outputs_bit_exact\":true,"
           "\"oracle\":\"independent full FP64 represented-BF16 RMSNorm, maximum2BF16steps\","
           "\"fixtures\":3,\"gain_modes\":2,\"epsilons\":[1e-8,1e-6,1e-4],"
           "\"graph_replays\":2,\"power\":\"auto\",\"pci\":\"0000:13:00.0\","
           "\"timing\":\"warm complete RMSNorm kernel, same-stream events, 256calls/sample; serial/token8/packed_epilogue then packed_epilogue/token8/serial\","
        << "\"serial_max_bf16_steps\":" << numeric.incumbent
        << ",\"token8_max_bf16_steps\":" << numeric.challenger
        << ",\"packed_epilogue_max_bf16_steps\":" << packed_epilogue_error << ",\"cells\":[";
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto& cell = cells[i]; out << (i ? "," : "") << "{\"features\":5120,\"rows\":" << cell.rows;
        out << ",\"input_weight_output_address_mod16\":" << cell.offset * sizeof(hip_bfloat16);
        out << ",\"timing_unit_offset\":true,\"timing_epsilon\":9.99999997e-7";
        out << ",\"serial_forward_ms\":"; write_array(out, cell.incumbent_forward);
        out << ",\"token8_forward_ms\":"; write_array(out, cell.challenger_forward);
        out << ",\"packed_epilogue_forward_ms\":"; write_array(out, cell.packed_epilogue_forward);
        out << ",\"packed_epilogue_reverse_ms\":"; write_array(out, cell.packed_epilogue_reverse);
        out << ",\"token8_reverse_ms\":"; write_array(out, cell.challenger_reverse);
        out << ",\"serial_reverse_ms\":"; write_array(out, cell.incumbent_reverse); out << '}';
    }
    out << "],\"passed\":true}\n";
    const auto payload = out.str();
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) fail("refusing existing or unavailable exact token8 report");
    std::size_t written = 0;
    while (written < payload.size()) {
        const auto count = ::write(fd, payload.data() + written, payload.size() - written);
        if (count <= 0) { (void)::close(fd); fail("exact token8 report write failed"); }
        written += static_cast<std::size_t>(count);
    }
    const bool sync_failed = ::fsync(fd) != 0;
    const bool close_failed = ::close(fd) != 0;
    if (sync_failed || close_failed) fail("exact token8 report close failed");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--exact-token8-screen") {
            screen_exact_token8(argv[2]);
            std::cout << "R9700 RMSNorm exact token8 T5/T6 screen: PASS\n";
            return 0;
        }
        bool timing = false; std::filesystem::path output;
        if (argc == 4 && std::string_view(argv[1]) == "--benchmark" &&
            std::string_view(argv[2]) == "--out-json") { timing = true; output = argv[3]; }
        else if (argc != 1) fail("usage: rmsnorm_dflash_rows56_qual [--benchmark --out-json PATH | --exact-token8-screen FRESH_JSON]");
        select_r9700(); hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        const auto numeric = qualify(stream); if (timing) benchmark(stream, output, numeric);
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "R9700 RMSNorm K5120 exact rows5/6 qualification: PASS\n"; return 0;
    } catch (const std::exception& error) {
        std::cerr << "R9700 RMSNorm K5120 exact rows5/6 qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
