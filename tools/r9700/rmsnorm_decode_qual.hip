#include <hip/hip_runtime.h>

#include "ops/r9700/eager/eager_ops.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::uint32_t kFeatures = 5120U;
constexpr float kModelEps = 1.0e-6F;
constexpr std::size_t kSamples = 9U;
constexpr std::uint32_t kTimingIterations = 8192U;
constexpr std::size_t kGuardElements = 32U;
constexpr std::uint16_t kGuardBits = 0x5555U;
constexpr double kUncertaintyFactor = 3.0 * 1.4826;
constexpr double kMaximumRobustRatio = 1.01;
constexpr double kMinimumRoundSavingMs = 5.0;
constexpr std::uint32_t kOrdinaryCallsPerRound = 129U;
constexpr std::string_view kPowerPath =
    "/sys/class/drm/card2/device/power_dpm_force_performance_level";

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
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() { (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct DeviceIdentity {
    int ordinal = -1;
    std::string name;
    std::string architecture;
    int runtime_version = 0;
    int driver_version = 0;
};

struct Error {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    std::uint32_t max_bf16_steps = 0;
};

struct NumericalEvidence {
    Error incumbent;
    Error candidate;
    std::uint32_t cases_per_route = 0;
    std::uint32_t graph_replays = 0;
    std::uint32_t malformed_cases = 0;
};

std::uint16_t bf16_bits(float value) {
    std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    if ((word & 0x7f800000U) != 0x7f800000U) {
        word += 0x7fffU + ((word >> 16U) & 1U);
    } else if ((word & 0xffffU) != 0U) {
        word |= 0x10000U;
    }
    return static_cast<std::uint16_t>(word >> 16U);
}
hip_bfloat16 bf16(float value) {
    hip_bfloat16 result;
    result.data = bf16_bits(value);
    return result;
}
std::uint32_t ordered_bf16(std::uint16_t bits) {
    return (bits & 0x8000U) != 0U ? static_cast<std::uint32_t>(0xffffU - bits)
                                  : static_cast<std::uint32_t>(bits) + 0x8000U;
}
std::uint32_t bf16_steps(std::uint16_t lhs, std::uint16_t rhs) {
    const std::uint32_t a = ordered_bf16(lhs);
    const std::uint32_t b = ordered_bf16(rhs);
    return a > b ? a - b : b - a;
}

template <class T> void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() < source.size()) fail("upload extent mismatch");
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}
template <class T> std::vector<T> download(const DeviceBuffer<T>& source, std::size_t count) {
    if (source.size() < count) fail("download extent mismatch");
    std::vector<T> result(count);
    HIP_CHECK(hipMemcpy(result.data(), source.get(), count * sizeof(T), hipMemcpyDeviceToHost));
    return result;
}

void poison_output(DeviceBuffer<hip_bfloat16>& storage, hipStream_t stream) {
    HIP_CHECK(hipMemsetAsync(storage.get(), 0x55, storage.size() * sizeof(hip_bfloat16), stream));
}

std::vector<hip_bfloat16> checked_output(const DeviceBuffer<hip_bfloat16>& storage,
                                         std::size_t logical_elements) {
    const auto raw = download(storage, storage.size());
    if (storage.size() != logical_elements + 2U * kGuardElements) {
        fail("guarded output extent mismatch");
    }
    for (std::size_t index = 0; index < kGuardElements; ++index) {
        if (raw[index].data != kGuardBits ||
            raw[kGuardElements + logical_elements + index].data != kGuardBits) {
            fail("RMSNorm output guard was clobbered");
        }
    }
    return {raw.begin() + static_cast<std::ptrdiff_t>(kGuardElements),
            raw.begin() + static_cast<std::ptrdiff_t>(kGuardElements + logical_elements)};
}

std::vector<std::uint16_t> output_bits(const DeviceBuffer<hip_bfloat16>& storage) {
    const auto raw = download(storage, storage.size());
    std::vector<std::uint16_t> bits;
    bits.reserve(raw.size());
    for (const hip_bfloat16 value : raw) bits.push_back(value.data);
    return bits;
}

std::string require_auto_power() {
    std::ifstream power{std::string(kPowerPath)};
    std::string profile;
    if (!power || !std::getline(power, profile) || profile != "auto") {
        fail("performance qualification requires card2 performance profile auto");
    }
    std::string trailing;
    if (std::getline(power, trailing) && !trailing.empty()) {
        fail("performance profile file contains unexpected trailing data");
    }
    return profile;
}

DeviceIdentity select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (!properties.integrated && std::string_view(properties.gcnArchName) == "gfx1201") {
            if (selected >= 0) fail("expected exactly one discrete gfx1201 device");
            selected = ordinal;
        }
    }
    if (selected < 0) fail("no discrete gfx1201 device");
    if (selected != 0) fail("the sole discrete gfx1201 device must be HIP ordinal 0");
    HIP_CHECK(hipSetDevice(selected));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, selected));
    if (std::string_view(properties.name).find("R9700") == std::string_view::npos) {
        fail("HIP device 0 is not an AMD Radeon AI PRO R9700");
    }
    DeviceIdentity identity{selected, properties.name, properties.gcnArchName};
    HIP_CHECK(hipRuntimeGetVersion(&identity.runtime_version));
    HIP_CHECK(hipDriverGetVersion(&identity.driver_version));
    return identity;
}

std::vector<hip_bfloat16> make_input(std::uint32_t rows, std::uint32_t fixture) {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(rows) * kFeatures);
    for (std::size_t index = 0; index < result.size(); ++index) {
        float value = 0.0F;
        if (fixture == 0U) {
            const auto code = static_cast<std::int32_t>((index * 37U + 53U) % 509U) - 254;
            value = static_cast<float>(code) / 127.0F;
        } else if (fixture == 1U) {
            static constexpr std::array<float, 8> values{
                0.0F, -0.0F, 0x1p-14F, -0x1p-14F, 0.03125F, -0.5F, 8.0F, -31.75F};
            value = values[index % values.size()];
        }
        result[index] = bf16(value);
    }
    return result;
}
std::vector<hip_bfloat16> make_weight() {
    std::vector<hip_bfloat16> result(kFeatures);
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto code = static_cast<std::int32_t>((index * 29U + 17U) % 257U) - 128;
        result[index] = bf16(static_cast<float>(code) / 509.0F);
    }
    return result;
}

Error require_oracle(const std::vector<hip_bfloat16>& actual,
                     const std::vector<hip_bfloat16>& input,
                     const std::vector<hip_bfloat16>& weight, std::uint32_t rows,
                     double eps, bool unit_offset) {
    if (actual.size() != input.size()) fail("oracle extent mismatch");
    Error error;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kFeatures;
        double sumsq = 0.0;
        for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
            const double value = static_cast<float>(input[base + feature]);
            sumsq += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sumsq / kFeatures + eps);
        for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
            const std::size_t index = base + feature;
            const double gain = static_cast<float>(weight[feature]) +
                                (unit_offset ? 1.0 : 0.0);
            const double expected = static_cast<float>(input[index]) * inverse * gain;
            const double observed = static_cast<float>(actual[index]);
            if (!std::isfinite(expected) || !std::isfinite(observed)) {
                fail("nonfinite RMSNorm oracle value");
            }
            const double delta = observed - expected;
            error.max_abs = std::max(error.max_abs, std::abs(delta));
            squared_error += delta * delta;
            squared_reference += expected * expected;
            error.max_bf16_steps = std::max(
                error.max_bf16_steps, bf16_steps(actual[index].data,
                                                  bf16_bits(static_cast<float>(expected))));
        }
    }
    error.relative_l2 = std::sqrt(squared_error / std::max(1.0e-30, squared_reference));
    if (error.max_bf16_steps > 2U) {
        std::ostringstream message;
        message << "candidate exceeds FP64 oracle: steps=" << error.max_bf16_steps
                << " rel_l2=" << error.relative_l2 << " max_abs=" << error.max_abs;
        fail(message.str());
    }
    return error;
}

void launch_candidate(const DeviceBuffer<hip_bfloat16>& input,
                      const DeviceBuffer<hip_bfloat16>& weight,
                      DeviceBuffer<hip_bfloat16>& output, std::uint32_t rows, float eps,
                      bool unit_offset, hipStream_t stream) {
    HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_qualification(
        input.get(), weight.get(), output.get() + kGuardElements, rows, eps, unit_offset, stream));
}

void accumulate_error(Error& maximum, const Error& value) {
    maximum.max_bf16_steps = std::max(maximum.max_bf16_steps, value.max_bf16_steps);
    maximum.max_abs = std::max(maximum.max_abs, value.max_abs);
    maximum.relative_l2 = std::max(maximum.relative_l2, value.relative_l2);
}

NumericalEvidence qualify_numerical(hipStream_t stream) {
    const auto weight = make_weight();
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    upload(d_weight, weight);
    NumericalEvidence evidence;
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U}) {
        for (const std::uint32_t fixture : {0U, 1U, 2U}) {
            const auto input = make_input(rows, fixture);
            DeviceBuffer<hip_bfloat16> d_input(input.size());
            DeviceBuffer<hip_bfloat16> d_output(input.size() + 2U * kGuardElements);
            upload(d_input, input);
            for (const bool unit_offset : {false, true}) {
                for (const float eps : {1.0e-8F, kModelEps, 1.0e-4F}) {
                    poison_output(d_output, stream);
                    launch_candidate(d_input, d_weight, d_output, rows, eps, unit_offset, stream);
                    HIP_CHECK(hipStreamSynchronize(stream));
                    accumulate_error(evidence.candidate,
                                     require_oracle(checked_output(d_output, input.size()), input,
                                                    weight, rows, eps, unit_offset));
                    poison_output(d_output, stream);
                    HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
                        d_input.get(), d_weight.get(), d_output.get() + kGuardElements,
                        kFeatures, rows, eps, unit_offset, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                    accumulate_error(evidence.incumbent,
                                     require_oracle(checked_output(d_output, input.size()), input,
                                                    weight, rows, eps, unit_offset));
                    ++evidence.cases_per_route;
                }
            }
        }
    }

    const auto graph_input = make_input(4U, 0U);
    DeviceBuffer<hip_bfloat16> d_input(graph_input.size());
    DeviceBuffer<hip_bfloat16> d_output(graph_input.size() + 2U * kGuardElements);
    upload(d_input, graph_input);
    poison_output(d_output, stream);
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    launch_candidate(d_input, d_weight, d_output, 4U, kModelEps, true, stream);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) HIP_CHECK(hipGraphLaunch(executable, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    accumulate_error(evidence.candidate,
                     require_oracle(checked_output(d_output, graph_input.size()), graph_input,
                                    weight, 4U, kModelEps, true));
    evidence.graph_replays = 2U;
    HIP_CHECK(hipGraphExecDestroy(executable));
    HIP_CHECK(hipGraphDestroy(graph));

    using ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_qualification;
    auto reject_without_write = [&](const hip_bfloat16* in, const hip_bfloat16* gain,
                                    hip_bfloat16* out, std::uint32_t rows, float eps,
                                    hipStream_t test_stream) {
        poison_output(d_output, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto before = output_bits(d_output);
        if (rmsnorm_k5120_rows4_qualification(in, gain, out, rows, eps, true,
                                               test_stream) != hipErrorInvalidValue) {
            fail("qualification boundary accepted malformed input");
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        if (output_bits(d_output) != before) fail("malformed input changed guarded output");
        ++evidence.malformed_cases;
    };
    hip_bfloat16* const out = d_output.get() + kGuardElements;
    reject_without_write(nullptr, d_weight.get(), out, 1U, kModelEps, stream);
    reject_without_write(d_input.get(), nullptr, out, 1U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), nullptr, 1U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 0U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 5U, kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 1U, 0.0F, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 1U, -kModelEps, stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 1U,
                         std::numeric_limits<float>::quiet_NaN(), stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 1U,
                         std::numeric_limits<float>::infinity(), stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 1U,
                         -std::numeric_limits<float>::infinity(), stream);
    reject_without_write(d_input.get(), d_weight.get(), out, 1U, kModelEps, nullptr);
    std::cout << "r9700 RMSNorm K5120 rows1..4 qualification: PASS max_bf16_steps="
              << std::max(evidence.incumbent.max_bf16_steps,
                          evidence.candidate.max_bf16_steps)
              << " graph_replays=2 output_canaries=true malformed_cases="
              << evidence.malformed_cases << '\n';
    return evidence;
}

template <class Launch> double time_launch(hipStream_t stream, Launch&& launch) {
    hipEvent_t begin = nullptr, end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin, stream));
    for (std::uint32_t iteration = 0; iteration < kTimingIterations; ++iteration) {
        HIP_CHECK(launch());
    }
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    return milliseconds;
}

double median(std::array<double, kSamples> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

double mad(const std::array<double, kSamples>& values, double center) {
    std::array<double, kSamples> deviations{};
    std::transform(values.begin(), values.end(), deviations.begin(),
                   [center](double value) { return std::abs(value - center); });
    return median(deviations);
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') out << '\\' << character;
        else if (character >= 0x20U) out << character;
        else out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(character) << std::dec;
    }
    return out.str();
}

std::string executable_path() {
    std::array<char, 4096> path{};
    const ssize_t count = ::readlink("/proc/self/exe", path.data(), path.size() - 1U);
    if (count <= 0 || static_cast<std::size_t>(count) >= path.size()) {
        fail("cannot resolve qualifier executable path");
    }
    return std::string(path.data(), static_cast<std::size_t>(count));
}

bool same_owned_inode(const std::filesystem::path& path, const struct stat& owned) {
    struct stat observed{};
    return ::lstat(path.c_str(), &observed) == 0 && S_ISREG(observed.st_mode) &&
        observed.st_dev == owned.st_dev && observed.st_ino == owned.st_ino &&
        observed.st_uid == owned.st_uid;
}

void publish_atomic_exclusive(const std::filesystem::path& destination,
                              const std::string& bytes) {
    const auto absolute = std::filesystem::absolute(destination);
    const auto parent = absolute.parent_path();
    struct stat parent_stat{};
    if (::lstat(parent.c_str(), &parent_stat) != 0 || !S_ISDIR(parent_stat.st_mode) ||
        S_ISLNK(parent_stat.st_mode)) fail("report parent must be a real existing directory");
    struct stat existing{};
    if (::lstat(absolute.c_str(), &existing) == 0 || errno != ENOENT) {
        fail("benchmark output must be a fresh path");
    }
    std::filesystem::path pending;
    int descriptor = -1;
    for (unsigned attempt = 0; attempt < 100U && descriptor < 0; ++attempt) {
        pending = parent / ("." + absolute.filename().string() + "." +
                            std::to_string(static_cast<long long>(::getpid())) + "." +
                            std::to_string(attempt) + ".pending");
        descriptor = ::open(pending.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                            0600);
        if (descriptor < 0 && errno != EEXIST) fail("cannot create pending report");
    }
    if (descriptor < 0) fail("cannot allocate unique pending report");
    struct stat owned{};
    bool linked = false;
    try {
        if (::fstat(descriptor, &owned) != 0 || !S_ISREG(owned.st_mode) ||
            owned.st_uid != ::getuid() || owned.st_nlink != 1) fail("invalid pending report inode");
        std::size_t offset = 0U;
        while (offset < bytes.size()) {
            const ssize_t written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
            if (written <= 0) fail("cannot write pending report");
            offset += static_cast<std::size_t>(written);
        }
        if (::fchmod(descriptor, 0444) != 0 || ::fsync(descriptor) != 0 ||
            ::close(descriptor) != 0) fail("cannot finalize pending report");
        descriptor = -1;
        if (!same_owned_inode(pending, owned)) fail("pending report was replaced");
        if (::link(pending.c_str(), absolute.c_str()) != 0) fail("cannot atomically publish report");
        linked = true;
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0 || ::fsync(directory) != 0 || ::close(directory) != 0) {
            fail("cannot persist report directory");
        }
        if (!same_owned_inode(absolute, owned)) fail("published report inode changed");
        std::ifstream readback(absolute, std::ios::binary);
        if (!readback.is_open()) fail("cannot reopen published report");
        const std::string observed((std::istreambuf_iterator<char>(readback)),
                                   std::istreambuf_iterator<char>());
        if (readback.bad() || observed != bytes) fail("published report readback differs");
        if (::unlink(pending.c_str()) != 0) fail("cannot remove pending report link");
        const int directory_after = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory_after < 0 || ::fsync(directory_after) != 0 ||
            ::close(directory_after) != 0) fail("cannot persist final report directory");
    } catch (...) {
        if (descriptor >= 0) (void)::close(descriptor);
        if (linked && same_owned_inode(absolute, owned)) (void)::unlink(absolute.c_str());
        if (same_owned_inode(pending, owned)) (void)::unlink(pending.c_str());
        throw;
    }
}

void write_error(std::ostream& out, const Error& error) {
    out << "{\"maximum_bf16_steps\":" << error.max_bf16_steps
        << ",\"maximum_absolute_error\":" << error.max_abs
        << ",\"maximum_relative_l2\":" << error.relative_l2 << '}';
}

bool benchmark(hipStream_t stream, const std::filesystem::path& path,
               const DeviceIdentity& device, const NumericalEvidence& numerical,
               std::string_view power_before_device, std::string_view power_after_numerical) {
    const auto input = make_input(4U, 0U);
    const auto weight = make_weight();
    DeviceBuffer<hip_bfloat16> d_input(input.size()), d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_output(input.size() + 2U * kGuardElements);
    upload(d_input, input);
    upload(d_weight, weight);
    struct Row {
        std::uint32_t rows;
        std::array<double, kSamples> incumbent;
        std::array<double, kSamples> candidate;
    };
    std::array<Row, 4> rows{{{1U, {}, {}}, {2U, {}, {}}, {3U, {}, {}}, {4U, {}, {}}}};
    for (int warmup = 0; warmup < 8; ++warmup) {
        poison_output(d_output, stream);
        launch_candidate(d_input, d_weight, d_output, 4U, kModelEps, true, stream);
        HIP_CHECK(ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
            d_input.get(), d_weight.get(), d_output.get() + kGuardElements, kFeatures, 4U,
            kModelEps, true, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    (void)checked_output(d_output, input.size());
    for (Row& row : rows) {
        for (std::size_t sample = 0; sample < kSamples; ++sample) {
            auto incumbent = [&] {
                return ninfer::ops::r9700::eager::rmsnorm_incumbent_qualification(
                    d_input.get(), d_weight.get(), d_output.get() + kGuardElements, kFeatures,
                    row.rows, kModelEps, true, stream);
            };
            auto candidate = [&] {
                return ninfer::ops::r9700::eager::rmsnorm_k5120_rows4_qualification(
                    d_input.get(), d_weight.get(), d_output.get() + kGuardElements, row.rows,
                    kModelEps, true, stream);
            };
            if ((sample & 1U) == 0U) {
                row.incumbent[sample] = time_launch(stream, incumbent);
                row.candidate[sample] = time_launch(stream, candidate);
            } else {
                row.candidate[sample] = time_launch(stream, candidate);
                row.incumbent[sample] = time_launch(stream, incumbent);
            }
            if (row.incumbent[sample] < 10.0 || row.candidate[sample] < 10.0) {
                fail("timing interval is shorter than 10 ms");
            }
            (void)checked_output(d_output, input.size());
        }
    }
    const std::string power_after_timing = require_auto_power();
    bool accepted = true;
    double minimum_saving = std::numeric_limits<double>::infinity();
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\n  \"schema\": \"ninfer.r9700.rmsnorm-k5120-rows4-qualification.v2\",\n"
           << "  \"status\": \"measured\",\n"
           << "  \"command\": [\"" << json_escape(executable_path())
           << "\",\"--benchmark\",\"--out-json\",\""
           << json_escape(path.string()) << "\"],\n"
           << "  \"device\": {\"ordinal\":" << device.ordinal << ",\"name\":\""
           << json_escape(device.name) << "\",\"architecture\":\""
           << json_escape(device.architecture) << "\",\"wave_size\":32,\"runtime_version\":"
           << device.runtime_version << ",\"driver_version\":" << device.driver_version << "},\n"
           << "  \"power_profile\": {\"path\":\"" << kPowerPath
           << "\",\"required\":\"auto\",\"before_device\":\"" << power_before_device
           << "\",\"after_numerical\":\"" << power_after_numerical
           << "\",\"after_timing\":\"" << power_after_timing << "\"},\n"
           << "  \"correctness\": {\"oracle\":\"independent FP64 complete represented-BF16 RMSNorm formula; BF16 output boundary\",\"criterion\":\"maximum 2 BF16 representable steps; absolute and relative-L2 values are diagnostics\",\"rows\":[1,2,3,4],\"fixtures\":[\"ordinary\",\"mixed_magnitude\",\"zero\"],\"unit_offset\":[false,true],\"eps\":[1e-8,1e-6,1e-4],\"maximum_bf16_steps_allowed\":2,\"cases_per_route\":"
           << numerical.cases_per_route << ",\"incumbent\":";
    write_error(output, numerical.incumbent);
    output << ",\"candidate\":";
    write_error(output, numerical.candidate);
    output << ",\"graph_replays\":" << numerical.graph_replays
           << ",\"output_canaries\":true,\"malformed_cases\":"
           << numerical.malformed_cases << ",\"malformed_case_names\":[\"null_input\",\"null_weight\",\"null_output\",\"rows_zero\",\"rows_five\",\"eps_zero\",\"eps_negative\",\"eps_nan\",\"eps_positive_infinity\",\"eps_negative_infinity\",\"null_stream\"]},\n"
           << "  \"timing\": {\"stream\":\"one owned nondefault stream; both routes and events ordered on it\",\"method\":\"9 alternating paired trials; raw event intervals\",\"iterations_per_interval\":"
           << kTimingIterations << ",\"minimum_interval_ms\":10,\"uncertainty\":\"4.4478*MAD\"},\n"
           << "  \"rows\": [\n";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const Row& row = rows[index];
        std::array<double, kSamples> incumbent{}, candidate{};
        std::transform(row.incumbent.begin(), row.incumbent.end(), incumbent.begin(),
                       [](double value) { return value / kTimingIterations; });
        std::transform(row.candidate.begin(), row.candidate.end(), candidate.begin(),
                       [](double value) { return value / kTimingIterations; });
        const double im = median(incumbent), cm = median(candidate);
        const double iu = kUncertaintyFactor * mad(incumbent, im);
        const double cu = kUncertaintyFactor * mad(candidate, cm);
        if (im <= iu) fail("incumbent timing uncertainty is not positive-bounded");
        const double robust_ratio = (cm + cu) / (im - iu);
        const double saving = kOrdinaryCallsPerRound * ((im - iu) - (cm + cu));
        const bool row_pass = robust_ratio <= kMaximumRobustRatio &&
                              saving >= kMinimumRoundSavingMs;
        accepted = accepted && row_pass;
        minimum_saving = std::min(minimum_saving, saving);
        output << "    {\"rows\":" << row.rows << ",\"ordinary_calls_per_round\":"
               << kOrdinaryCallsPerRound << ",\"incumbent_median_ms\":" << im
               << ",\"candidate_median_ms\":" << cm << ",\"incumbent_mad_ms\":"
               << mad(incumbent, im) << ",\"candidate_mad_ms\":" << mad(candidate, cm)
               << ",\"incumbent_uncertainty_ms\":" << iu
               << ",\"candidate_uncertainty_ms\":" << cu
               << ",\"robust_upper_ratio\":" << robust_ratio
               << ",\"robust_round_saving_lower_ms\":" << saving
               << ",\"passed\":" << (row_pass ? "true" : "false") << ",\"trials\":[";
        for (std::size_t sample = 0; sample < kSamples; ++sample) {
            output << (sample == 0 ? "" : ",") << "{\"index\":" << sample
                   << ",\"order\":\"" << ((sample & 1U) == 0U ?
                       "incumbent,candidate" : "candidate,incumbent")
                   << "\",\"incumbent_interval_ms\":" << row.incumbent[sample]
                   << ",\"candidate_interval_ms\":" << row.candidate[sample] << '}';
        }
        output << "]}";
        output << (index + 1U == rows.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"decision\": {\"maximum_robust_ratio\":" << kMaximumRobustRatio
           << ",\"minimum_round_saving_lower_ms\":" << kMinimumRoundSavingMs
           << ",\"minimum_observed_round_saving_lower_ms\":" << minimum_saving
           << ",\"all_rows_passed\":" << (accepted ? "true" : "false")
           << ",\"accepted\":" << (accepted ? "true" : "false") << "}\n}\n";
    if (!output) fail("could not serialize benchmark report");
    publish_atomic_exclusive(path, output.str());
    return accepted;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool benchmark_mode = argc == 4 && std::string_view(argv[1]) == "--benchmark" &&
            std::string_view(argv[2]) == "--out-json";
        if (!benchmark_mode && argc != 1) {
            fail("usage: rmsnorm_decode_qual [--benchmark --out-json PATH]");
        }
        const std::string power_before_device = benchmark_mode ? require_auto_power() : "";
        const DeviceIdentity device = select_r9700();
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        const NumericalEvidence numerical = qualify_numerical(stream);
        bool accepted = true;
        if (benchmark_mode) {
            const std::string power_after_numerical = require_auto_power();
            accepted = benchmark(stream, argv[3], device, numerical, power_before_device,
                                 power_after_numerical);
        }
        HIP_CHECK(hipStreamDestroy(stream));
        if (!accepted) {
            std::cerr << "r9700 RMSNorm K5120 rows1..4 qualification: REJECTED; retained report "
                      << argv[3] << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700 RMSNorm K5120 rows1..4 qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
