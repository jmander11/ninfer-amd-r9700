// Qualification-only gfx1201 ceilings for the production A8Q4G64 prefill route.
// This executable neither links to nor changes the production runtime.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWave = 32;
constexpr unsigned kChains = 8;
constexpr std::uint64_t kIu4OpsPerInstruction = 2ULL * 16ULL * 16ULL * 32ULL;
constexpr std::uint64_t kIu8OpsPerInstruction = 2ULL * 16ULL * 16ULL * 16ULL;
constexpr std::size_t kCodeBytesPerScaleByte = 16; // Q4 G64: 32 code bytes + one FP16 scale.
constexpr std::size_t kMinimumL2Multiple = 4;
constexpr double kDefaultSeconds = 0.25;
constexpr int kDefaultTrials = 5;
constexpr std::size_t kDefaultMaximumCodeBytes = std::size_t{2} << 30;
constexpr std::string_view kRequiredArch = "gfx1201";
constexpr std::string_view kAmdVendor = "0x1002";
constexpr std::string_view kR9700Device = "0x7551";
static_assert(kThreads % kWave == 0);
static_assert(kCodeBytesPerScaleByte == 32 / sizeof(std::uint16_t));

using I32x2 = int __attribute__((ext_vector_type(2)));
using I32x8 = int __attribute__((ext_vector_type(8)));

#define HIP_CHECK(expression)                                                                    \
    do {                                                                                         \
        const hipError_t error__ = (expression);                                                 \
        if (error__ != hipSuccess) {                                                             \
            std::ostringstream message__;                                                       \
            message__ << #expression << " failed: " << hipGetErrorString(error__);              \
            throw std::runtime_error(message__.str());                                           \
        }                                                                                        \
    } while (false)

struct Options {
    std::string output;
    double seconds = kDefaultSeconds;
    int trials = kDefaultTrials;
    double code_gib = 0.0;
};

struct DeviceSelection {
    int ordinal = -1;
    std::string pci;
    std::string power_path;
    hipDeviceProp_t properties{};
};

struct Sample {
    double best_ms = 0.0;
    double median_ms = 0.0;
    std::vector<double> milliseconds;
};

struct KernelResources {
    int registers_per_thread = 0;
    std::size_t static_lds_bytes = 0;
    std::size_t private_bytes = 0;
    int maximum_threads_per_block = 0;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    std::size_t count() const { return count_; }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&value_)); }
    ~Event() { (void)hipEventDestroy(value_); }
    hipEvent_t get() const { return value_; }
private:
    hipEvent_t value_{};
};

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) return {};
    std::string value;
    std::getline(input, value);
    return trim(value);
}

std::string normalize_pci(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value.size() == 7 && value[2] == ':' && value[5] == '.') value = "0000:" + value;
    return value;
}

DeviceSelection select_device() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    const char* requested_raw = std::getenv("NINFER_R9700_PCI_BUS_ID");
    const std::string requested = requested_raw == nullptr ? "" : normalize_pci(requested_raw);
    std::vector<DeviceSelection> matches;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        char buffer[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(buffer, sizeof(buffer), ordinal));
        const std::string pci = normalize_pci(buffer);
        const std::string base = "/sys/bus/pci/devices/" + pci;
        const bool identity = read_text(base + "/vendor") == kAmdVendor &&
                              read_text(base + "/device") == kR9700Device;
        if (identity && properties.integrated == 0 && properties.warpSize == 32 &&
            std::string_view(properties.gcnArchName) == kRequiredArch &&
            (requested.empty() || requested == pci)) {
            matches.push_back(DeviceSelection{ordinal, pci,
                base + "/power_dpm_force_performance_level", properties});
        }
    }
    if (matches.size() != 1) {
        throw std::runtime_error("expected exactly one selected discrete 1002:7551 gfx1201 R9700");
    }
    HIP_CHECK(hipSetDevice(matches.front().ordinal));
    return matches.front();
}

void require_auto(const std::string& path, const char* when) {
    const std::string value = read_text(path);
    if (value != "auto") {
        throw std::runtime_error(std::string("hardware peak timing requires auto power ") + when +
                                 "; observed '" + value + "'");
    }
}

double parse_double(const char* value, const char* option) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0)
        throw std::runtime_error(std::string("invalid ") + option);
    return parsed;
}

int parse_int(const char* value, const char* option) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 1000)
        throw std::runtime_error(std::string("invalid ") + option);
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--help") {
            std::printf("Usage: %s --out-json PATH [--seconds N] [--trials N] [--code-gib N]\n",
                        argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing option value");
        if (argument == "--out-json") options.output = argv[++i];
        else if (argument == "--seconds") options.seconds = parse_double(argv[++i], "--seconds");
        else if (argument == "--trials") options.trials = parse_int(argv[++i], "--trials");
        else if (argument == "--code-gib") options.code_gib = parse_double(argv[++i], "--code-gib");
        else throw std::runtime_error("unknown option: " + std::string(argument));
    }
    if (options.output.empty()) throw std::runtime_error("--out-json is required");
    return options;
}

__global__ __launch_bounds__(kThreads) void iu4_issue_peak_kernel(const I32x2* __restrict__ inputs,
                                                                  std::int32_t* __restrict__ output,
                                                                  unsigned iterations) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned lane = threadIdx.x & 31U;
    const unsigned wave = (blockIdx.x * blockDim.x + threadIdx.x) / kWave;
    const I32x2 a = inputs[(wave * 2U) & 255U];
    const I32x2 b = inputs[(wave * 2U + 1U) & 255U];
    I32x8 accumulators[kChains];
#pragma unroll
    for (unsigned chain = 0; chain < kChains; ++chain) {
#pragma unroll
        for (unsigned element = 0; element < 8; ++element)
            accumulators[chain][element] = static_cast<int>(lane + chain + element);
    }
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (unsigned chain = 0; chain < kChains; ++chain) {
            accumulators[chain] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a, true, b, accumulators[chain], false);
        }
    }
    std::int32_t sum = 0;
#pragma unroll
    for (unsigned chain = 0; chain < kChains; ++chain) sum += accumulators[chain][lane & 7U];
    output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
#else
    (void)inputs; (void)output; (void)iterations;
#endif
}

// Topology-matched native-IU8 ceiling for one M16xN32xK64 useful W4A8 fragment. The two
// accumulator chains own the two N16 outputs and each carries four dependent K16 instructions.
// Operands are loaded once before the loop; this deliberately measures no Q4 unpack or memory path.
__global__ __launch_bounds__(kThreads) void iu8_topology_peak_kernel(
    const I32x2* __restrict__ inputs, std::int32_t* __restrict__ output, unsigned iterations) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned lane = threadIdx.x & 31U;
    const unsigned wave = (blockIdx.x * blockDim.x + threadIdx.x) / kWave;
    const I32x2 a = inputs[(wave * 2U) & 255U];
    const I32x2 b = inputs[(wave * 2U + 1U) & 255U];
    I32x8 accumulators[2];
#pragma unroll
    for (unsigned chain = 0; chain < 2; ++chain) {
#pragma unroll
        for (unsigned element = 0; element < 8; ++element)
            accumulators[chain][element] = static_cast<int>(lane + chain + element);
    }
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (unsigned quarter = 0; quarter < 4; ++quarter) {
#pragma unroll
            for (unsigned chain = 0; chain < 2; ++chain) {
                accumulators[chain] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    true, a, true, b, accumulators[chain], false);
            }
        }
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] =
        accumulators[0][lane & 7U] + accumulators[1][lane & 7U];
#else
    (void)inputs; (void)output; (void)iterations;
#endif
}

// Eight independent chains expose the register-resident native-IU8 opcode saturation ceiling.
// This is not a production topology: it separates opcode issue capacity from the four-deep
// dependency chain above and from every Q4 expansion/staging cost.
__global__ __launch_bounds__(kThreads) void iu8_saturation_peak_kernel(
    const I32x2* __restrict__ inputs, std::int32_t* __restrict__ output, unsigned iterations) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned lane = threadIdx.x & 31U;
    const unsigned wave = (blockIdx.x * blockDim.x + threadIdx.x) / kWave;
    const I32x2 a = inputs[(wave * 2U) & 255U];
    const I32x2 b = inputs[(wave * 2U + 1U) & 255U];
    I32x8 accumulators[kChains];
#pragma unroll
    for (unsigned chain = 0; chain < kChains; ++chain) {
#pragma unroll
        for (unsigned element = 0; element < 8; ++element)
            accumulators[chain][element] = static_cast<int>(lane + chain + element);
    }
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (unsigned chain = 0; chain < kChains; ++chain) {
            accumulators[chain] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                true, a, true, b, accumulators[chain], false);
        }
    }
    std::int32_t sum = 0;
#pragma unroll
    for (unsigned chain = 0; chain < kChains; ++chain) sum += accumulators[chain][lane & 7U];
    output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
#else
    (void)inputs; (void)output; (void)iterations;
#endif
}

// Match one persistent Q4G64 weight group's represented global reads: 32 packed-code bytes and
// one FP16 scale. Sixteen code bytes are read by every lane; one lane in sixteen reads sixteen
// adjacent scale bytes, preserving the exact aggregate 16:1 byte ratio with vector transactions.
__global__ __launch_bounds__(kThreads) void q4_code_scale_stream_kernel(
    const uint4* __restrict__ codes, const uint4* __restrict__ scales,
    uint4* __restrict__ checksums, std::size_t code_vectors) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    uint4 sum = make_uint4(0, 0, 0, 0);
    for (std::size_t index = tid; index < code_vectors; index += stride) {
        const uint4 code = codes[index];
        sum.x += code.x; sum.y += code.y; sum.z += code.z; sum.w += code.w;
        if ((index & (kCodeBytesPerScaleByte - 1)) == 0) {
            const uint4 scale = scales[index / kCodeBytesPerScaleByte];
            sum.x += scale.x; sum.y += scale.y; sum.z += scale.z; sum.w += scale.w;
        }
    }
    checksums[tid] = sum;
}

template <typename Kernel>
int resident_grid(Kernel kernel, int processors) {
    int blocks = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, 0));
    return std::max(processors, processors * blocks);
}

template <typename Kernel>
KernelResources kernel_resources(Kernel kernel) {
    hipFuncAttributes attributes{};
    HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(kernel)));
    return KernelResources{attributes.numRegs, attributes.sharedSizeBytes,
                           attributes.localSizeBytes, attributes.maxThreadsPerBlock};
}

template <typename Launch>
double elapsed_ms(Launch&& launch) {
    Event begin, end;
    HIP_CHECK(hipEventRecord(begin.get()));
    launch();
    HIP_CHECK(hipEventRecord(end.get()));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float value = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&value, begin.get(), end.get()));
    return value;
}

template <typename Launch>
Sample measure(int trials, Launch&& launch) {
    for (int i = 0; i < 8; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    Sample sample;
    sample.milliseconds.reserve(trials);
    for (int trial = 0; trial < trials; ++trial) sample.milliseconds.push_back(elapsed_ms(launch));
    HIP_CHECK(hipGetLastError());
    std::sort(sample.milliseconds.begin(), sample.milliseconds.end());
    sample.best_ms = sample.milliseconds.front();
    sample.median_ms = sample.milliseconds[sample.milliseconds.size() / 2];
    return sample;
}

std::string json_string(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const char c : value) {
        if (c == '"' || c == '\\') output << '\\' << c;
        else if (c == '\n') output << "\\n";
        else output << c;
    }
    output << '"';
    return output.str();
}

void write_exclusive(const std::string& path, const std::string& contents) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) throw std::runtime_error("cannot create fresh output '" + path + "': " +
                                         std::strerror(errno));
    std::size_t written = 0;
    while (written < contents.size()) {
        const ssize_t amount = ::write(fd, contents.data() + written, contents.size() - written);
        if (amount <= 0) {
            const int saved = errno; (void)::close(fd); (void)::unlink(path.c_str());
            throw std::runtime_error("output write failed: " + std::string(std::strerror(saved)));
        }
        written += static_cast<std::size_t>(amount);
    }
    if (::close(fd) != 0) throw std::runtime_error("output close failed");
}

void emit_samples(std::ostringstream& out, const Sample& sample) {
    out << '[';
    for (std::size_t i = 0; i < sample.milliseconds.size(); ++i) {
        if (i) out << ',';
        out << sample.milliseconds[i];
    }
    out << ']';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const DeviceSelection device = select_device();
        require_auto(device.power_path, "before allocation");

        std::size_t free_bytes = 0, total_bytes = 0;
        HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        std::size_t code_bytes = options.code_gib > 0.0
            ? static_cast<std::size_t>(options.code_gib * static_cast<double>(std::size_t{1} << 30))
            : std::min(kDefaultMaximumCodeBytes, free_bytes / 5);
        const std::size_t quantum = sizeof(uint4) * kCodeBytesPerScaleByte;
        code_bytes = code_bytes / quantum * quantum;
        const std::size_t scale_bytes = code_bytes / kCodeBytesPerScaleByte;
        if (code_bytes + scale_bytes < kMinimumL2Multiple *
                static_cast<std::size_t>(device.properties.l2CacheSize))
            throw std::runtime_error("Q4 code+scale working set must be at least 4x L2");
        if (code_bytes + scale_bytes > free_bytes / 2)
            throw std::runtime_error("Q4 stream buffers exceed half of free VRAM");

        const int iu4_grid = resident_grid(iu4_issue_peak_kernel,
                                            device.properties.multiProcessorCount);
        const int iu8_topology_grid = resident_grid(iu8_topology_peak_kernel,
                                                     device.properties.multiProcessorCount);
        const int iu8_saturation_grid = resident_grid(iu8_saturation_peak_kernel,
                                                       device.properties.multiProcessorCount);
        const int stream_grid = resident_grid(q4_code_scale_stream_kernel,
                                               device.properties.multiProcessorCount);
        const KernelResources iu4_resources = kernel_resources(iu4_issue_peak_kernel);
        const KernelResources iu8_topology_resources = kernel_resources(iu8_topology_peak_kernel);
        const KernelResources iu8_saturation_resources =
            kernel_resources(iu8_saturation_peak_kernel);
        DeviceBuffer<I32x2> inputs(256);
        DeviceBuffer<std::int32_t> iu4_output(static_cast<std::size_t>(iu4_grid) * kThreads);
        DeviceBuffer<std::int32_t> iu8_topology_output(
            static_cast<std::size_t>(iu8_topology_grid) * kThreads);
        DeviceBuffer<std::int32_t> iu8_saturation_output(
            static_cast<std::size_t>(iu8_saturation_grid) * kThreads);
        DeviceBuffer<uint4> codes(code_bytes / sizeof(uint4));
        DeviceBuffer<uint4> scales(scale_bytes / sizeof(uint4));
        DeviceBuffer<uint4> checksums(static_cast<std::size_t>(stream_grid) * kThreads);
        HIP_CHECK(hipMemset(inputs.get(), 0x31, inputs.count() * sizeof(I32x2)));
        HIP_CHECK(hipMemset(codes.get(), 0x5a, code_bytes));
        HIP_CHECK(hipMemset(scales.get(), 0xa5, scale_bytes));

        const unsigned calibration_iterations = 1024;
        const double iu4_calibration_ms = elapsed_ms([&] {
            iu4_issue_peak_kernel<<<iu4_grid, kThreads>>>(inputs.get(), iu4_output.get(),
                                                          calibration_iterations);
        });
        const unsigned iu4_iterations = std::max(1U, static_cast<unsigned>(std::ceil(
            options.seconds * 1000.0 / iu4_calibration_ms * calibration_iterations)));
        const Sample iu4 = measure(options.trials, [&] {
            iu4_issue_peak_kernel<<<iu4_grid, kThreads>>>(inputs.get(), iu4_output.get(),
                                                          iu4_iterations);
        });

        const double iu8_topology_calibration_ms = elapsed_ms([&] {
            iu8_topology_peak_kernel<<<iu8_topology_grid, kThreads>>>(
                inputs.get(), iu8_topology_output.get(), calibration_iterations);
        });
        const unsigned iu8_topology_iterations = std::max(1U, static_cast<unsigned>(std::ceil(
            options.seconds * 1000.0 / iu8_topology_calibration_ms * calibration_iterations)));
        const Sample iu8_topology = measure(options.trials, [&] {
            iu8_topology_peak_kernel<<<iu8_topology_grid, kThreads>>>(
                inputs.get(), iu8_topology_output.get(), iu8_topology_iterations);
        });

        const double iu8_saturation_calibration_ms = elapsed_ms([&] {
            iu8_saturation_peak_kernel<<<iu8_saturation_grid, kThreads>>>(
                inputs.get(), iu8_saturation_output.get(), calibration_iterations);
        });
        const unsigned iu8_saturation_iterations = std::max(1U, static_cast<unsigned>(std::ceil(
            options.seconds * 1000.0 / iu8_saturation_calibration_ms * calibration_iterations)));
        const Sample iu8_saturation = measure(options.trials, [&] {
            iu8_saturation_peak_kernel<<<iu8_saturation_grid, kThreads>>>(
                inputs.get(), iu8_saturation_output.get(), iu8_saturation_iterations);
        });

        const double stream_calibration_ms = elapsed_ms([&] {
            q4_code_scale_stream_kernel<<<stream_grid, kThreads>>>(
                codes.get(), scales.get(), checksums.get(), codes.count());
        });
        const unsigned stream_repeats = std::max(1U, static_cast<unsigned>(std::ceil(
            options.seconds * 1000.0 / stream_calibration_ms)));
        const Sample stream = measure(options.trials, [&] {
            for (unsigned repeat = 0; repeat < stream_repeats; ++repeat)
                q4_code_scale_stream_kernel<<<stream_grid, kThreads>>>(
                    codes.get(), scales.get(), checksums.get(), codes.count());
        });
        HIP_CHECK(hipDeviceSynchronize());
        require_auto(device.power_path, "after timing");

        const std::uint64_t waves = static_cast<std::uint64_t>(iu4_grid) * (kThreads / kWave);
        const std::uint64_t iu4_instructions = waves * iu4_iterations * kChains;
        const double issued_ops = static_cast<double>(iu4_instructions) * kIu4OpsPerInstruction;
        const double best_iu4_tops = issued_ops / (iu4.best_ms * 1.0e9);
        const double median_iu4_tops = issued_ops / (iu4.median_ms * 1.0e9);
        const double useful_iu4_w4a8_ops = issued_ops / 2.0;
        const double best_iu4_useful_w4a8_tops = useful_iu4_w4a8_ops / (iu4.best_ms * 1.0e9);
        const double median_iu4_useful_w4a8_tops =
            useful_iu4_w4a8_ops / (iu4.median_ms * 1.0e9);
        const std::uint64_t iu8_topology_waves =
            static_cast<std::uint64_t>(iu8_topology_grid) * (kThreads / kWave);
        const std::uint64_t iu8_topology_instructions =
            iu8_topology_waves * iu8_topology_iterations * 8U;
        const double iu8_topology_issued_ops =
            static_cast<double>(iu8_topology_instructions) * kIu8OpsPerInstruction;
        const double best_iu8_topology_tops =
            iu8_topology_issued_ops / (iu8_topology.best_ms * 1.0e9);
        const double median_iu8_topology_tops =
            iu8_topology_issued_ops / (iu8_topology.median_ms * 1.0e9);
        const std::uint64_t iu8_saturation_waves =
            static_cast<std::uint64_t>(iu8_saturation_grid) * (kThreads / kWave);
        const std::uint64_t iu8_saturation_instructions =
            iu8_saturation_waves * iu8_saturation_iterations * kChains;
        const double iu8_saturation_issued_ops =
            static_cast<double>(iu8_saturation_instructions) * kIu8OpsPerInstruction;
        const double best_iu8_saturation_tops =
            iu8_saturation_issued_ops / (iu8_saturation.best_ms * 1.0e9);
        const double median_iu8_saturation_tops =
            iu8_saturation_issued_ops / (iu8_saturation.median_ms * 1.0e9);
        const std::uint64_t stream_bytes =
            static_cast<std::uint64_t>(code_bytes + scale_bytes) * stream_repeats;
        const double best_stream_gbps = static_cast<double>(stream_bytes) / (stream.best_ms * 1.0e6);
        const double median_stream_gbps =
            static_cast<double>(stream_bytes) / (stream.median_ms * 1.0e6);

        int runtime_version = 0, driver_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK(hipDriverGetVersion(&driver_version));
        std::ostringstream out;
        out << std::setprecision(17)
            << "{\n  \"artifact_type\": \"ninfer_r9700_q4_hardware_peak_probe\",\n"
            << "  \"schema_version\": 2,\n  \"status\": \"passed\",\n"
            << "  \"scope\": \"qualification-only native IU4, native IU8 dependency/saturation, and represented Q4G64 global-read ceilings\",\n"
            << "  \"power_profile\": {\"path\": " << json_string(device.power_path)
            << ", \"required\": \"auto\", \"before\": \"auto\", \"after\": \"auto\"},\n"
            << "  \"hardware\": {\"device\": " << json_string(device.properties.name)
            << ", \"pci_bus_id\": " << json_string(device.pci)
            << ", \"architecture\": \"gfx1201\", \"wave_size\": 32, \"compute_units\": "
            << device.properties.multiProcessorCount << ", \"l2_bytes\": "
            << device.properties.l2CacheSize << ", \"vram_bytes\": " << total_bytes
            << ", \"hip_runtime_version\": " << runtime_version
            << ", \"hip_driver_version\": " << driver_version << "},\n"
            << "  \"iu4_issue\": {\"opcode\": \"v_wmma_i32_16x16x32_iu4\", "
               "\"operand_residency\": \"register\", \"independent_accumulator_chains\": "
            << kChains << ", \"grid_blocks\": " << iu4_grid << ", \"threads_per_block\": "
            << kThreads << ", \"kernel_iterations\": " << iu4_iterations
            << ", \"issued_instructions\": " << iu4_instructions
            << ", \"issued_operations\": " << static_cast<std::uint64_t>(issued_ops)
            << ", \"useful_w4a8_operations\": "
            << static_cast<std::uint64_t>(useful_iu4_w4a8_ops)
            << ", \"best_ms\": " << iu4.best_ms << ", \"median_ms\": " << iu4.median_ms
            << ", \"best_issued_tops\": " << best_iu4_tops
            << ", \"median_issued_tops\": " << median_iu4_tops
            << ", \"best_useful_w4a8_tops\": " << best_iu4_useful_w4a8_tops
            << ", \"median_useful_w4a8_tops\": " << median_iu4_useful_w4a8_tops
            << ", \"resources\": {\"registers_per_thread\": "
            << iu4_resources.registers_per_thread << ", \"static_lds_bytes\": "
            << iu4_resources.static_lds_bytes << ", \"private_bytes\": "
            << iu4_resources.private_bytes << ", \"maximum_threads_per_block\": "
            << iu4_resources.maximum_threads_per_block << "}, \"samples_ms\": ";
        emit_samples(out, iu4);
        out << "},\n  \"iu8_topology_issue\": {\"opcode\": \"v_wmma_i32_16x16x16_iu8\", "
               "\"operand_residency\": \"register_no_unpack_or_staging\", "
               "\"logical_useful_w4a8_fragment\": \"M16xN32xK64\", "
               "\"independent_accumulator_chains\": 2, \"dependent_instructions_per_chain\": 4, "
               "\"static_instruction_sites\": 8, \"grid_blocks\": " << iu8_topology_grid
            << ", \"threads_per_block\": " << kThreads << ", \"kernel_iterations\": "
            << iu8_topology_iterations << ", \"issued_instructions\": "
            << iu8_topology_instructions << ", \"issued_operations\": "
            << static_cast<std::uint64_t>(iu8_topology_issued_ops)
            << ", \"useful_w4a8_operations\": "
            << static_cast<std::uint64_t>(iu8_topology_issued_ops)
            << ", \"best_ms\": " << iu8_topology.best_ms << ", \"median_ms\": "
            << iu8_topology.median_ms << ", \"best_issued_tops\": "
            << best_iu8_topology_tops << ", \"median_issued_tops\": "
            << median_iu8_topology_tops << ", \"best_useful_w4a8_tops\": "
            << best_iu8_topology_tops << ", \"median_useful_w4a8_tops\": "
            << median_iu8_topology_tops
            << ", \"resources\": {\"registers_per_thread\": "
            << iu8_topology_resources.registers_per_thread << ", \"static_lds_bytes\": "
            << iu8_topology_resources.static_lds_bytes << ", \"private_bytes\": "
            << iu8_topology_resources.private_bytes << ", \"maximum_threads_per_block\": "
            << iu8_topology_resources.maximum_threads_per_block << "}, \"samples_ms\": ";
        emit_samples(out, iu8_topology);
        out << "},\n  \"iu8_saturation_issue\": {\"opcode\": \"v_wmma_i32_16x16x16_iu8\", "
               "\"operand_residency\": \"register_no_unpack_or_staging\", "
               "\"topology\": \"eight_independent_M16xN16xK16_fragments\", "
               "\"independent_accumulator_chains\": 8, \"dependent_instructions_per_chain\": 1, "
               "\"static_instruction_sites\": 8, \"grid_blocks\": " << iu8_saturation_grid
            << ", \"threads_per_block\": " << kThreads << ", \"kernel_iterations\": "
            << iu8_saturation_iterations << ", \"issued_instructions\": "
            << iu8_saturation_instructions << ", \"issued_operations\": "
            << static_cast<std::uint64_t>(iu8_saturation_issued_ops)
            << ", \"useful_integer_operations\": "
            << static_cast<std::uint64_t>(iu8_saturation_issued_ops)
            << ", \"useful_w4a8_operations\": null, "
               "\"useful_w4a8_accounting\": \"not_applicable_opcode_saturation_control\", "
            << "\"best_ms\": " << iu8_saturation.best_ms << ", \"median_ms\": "
            << iu8_saturation.median_ms << ", \"best_issued_tops\": "
            << best_iu8_saturation_tops << ", \"median_issued_tops\": "
            << median_iu8_saturation_tops
            << ", \"best_useful_w4a8_tops\": null, \"median_useful_w4a8_tops\": null, "
               "\"resources\": {\"registers_per_thread\": "
            << iu8_saturation_resources.registers_per_thread << ", \"static_lds_bytes\": "
            << iu8_saturation_resources.static_lds_bytes << ", \"private_bytes\": "
            << iu8_saturation_resources.private_bytes << ", \"maximum_threads_per_block\": "
            << iu8_saturation_resources.maximum_threads_per_block << "}, \"samples_ms\": ";
        emit_samples(out, iu8_saturation);
        out << "},\n  \"q4g64_stream\": {\"pattern\": \"separate packed-weight-code and FP16-scale planes\", "
               "\"group_size\": 64, \"code_bytes_per_group_row\": 32, "
               "\"scale_bytes_per_group_row\": 2, \"code_to_scale_byte_ratio\": 16, "
               "\"working_set_bytes\": " << code_bytes + scale_bytes
            << ", \"working_set_over_l2\": "
            << static_cast<double>(code_bytes + scale_bytes) / device.properties.l2CacheSize
            << ", \"grid_blocks\": " << stream_grid << ", \"threads_per_block\": "
            << kThreads << ", \"sweeps_per_sample\": " << stream_repeats
            << ", \"represented_read_bytes_per_sample\": " << stream_bytes
            << ", \"best_ms\": " << stream.best_ms << ", \"median_ms\": "
            << stream.median_ms << ", \"best_represented_gbps\": " << best_stream_gbps
            << ", \"median_represented_gbps\": " << median_stream_gbps
            << ", \"samples_ms\": ";
        emit_samples(out, stream);
        out << "},\n  \"interpretation\": \"Ceilings for same-session reconciliation with a8q4_prefill_traffic.py; not production-kernel utilization by themselves\"\n}\n";
        write_exclusive(options.output, out.str());
        std::printf("wrote fresh Q4 hardware peak evidence to %s\n", options.output.c_str());
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Q4 hardware peak probe failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
