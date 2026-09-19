#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

namespace {

constexpr std::uint32_t kHeads = 48;
constexpr std::uint32_t kColumns = 5120;
constexpr std::uint32_t kThreads = 256;
constexpr std::size_t kWeightElements = static_cast<std::size_t>(kHeads) * kColumns;
constexpr std::uint16_t kCanaryBits = 0x7f5a;
constexpr std::size_t kCopies = 3;
constexpr std::size_t kTrials = 12;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream out;
    out << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(out.str());
}
#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count != 0) HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    T* get() const { return data_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
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
    hipEvent_t value_ = nullptr;
};

struct GuardedBf16 {
    explicit GuardedBf16(std::size_t count) : storage(count + 2), count(count) {}
    hip_bfloat16* payload() const { return storage.get() + 1; }
    DeviceBuffer<hip_bfloat16> storage;
    std::size_t count;
};

struct GuardedFloat {
    explicit GuardedFloat(std::size_t count) : storage(count + 2), count(count) {}
    float* payload() const { return storage.get() + 1; }
    DeviceBuffer<float> storage;
    std::size_t count;
};

struct Arguments {
    const hip_bfloat16* input;
    const hip_bfloat16* a_weight;
    const hip_bfloat16* b_weight;
    const float* a_log;
    const float* dt_bias;
    hip_bfloat16* a;
    hip_bfloat16* b;
    float* g;
    float* beta;
};

__device__ __forceinline__ void control_formula(hip_bfloat16 ar, hip_bfloat16 br,
                                                 float a_log, float dt_bias,
                                                 float* g, float* beta) {
    const float av = static_cast<float>(ar) + dt_bias;
    const float softplus = av > 20.0F ? av : log1pf(expf(av));
    *g = -expf(a_log) * softplus;
    const float bv = static_cast<float>(br);
    *beta = 1.0F / (1.0F + expf(-bv));
}

__global__ void incumbent_projection_kernel(const hip_bfloat16* input,
                                             const hip_bfloat16* weights,
                                             hip_bfloat16* output) {
    const std::uint32_t row = blockIdx.x;
    float partial = 0.0F;
    const std::size_t base = static_cast<std::size_t>(row) * kColumns;
    for (std::uint32_t k = threadIdx.x; k < kColumns; k += blockDim.x) {
        partial = fmaf(static_cast<float>(input[k]), static_cast<float>(weights[base + k]), partial);
    }
    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t width = kThreads / 2; width != 0; width >>= 1) {
        if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = static_cast<hip_bfloat16>(reduction[0]);
}

__global__ void combined_projection_kernel(const hip_bfloat16* input,
                                            const hip_bfloat16* a_weight,
                                            const hip_bfloat16* b_weight,
                                            hip_bfloat16* a, hip_bfloat16* b) {
    const std::uint32_t joined_row = blockIdx.x;
    const bool second = joined_row >= kHeads;
    const std::uint32_t row = second ? joined_row - kHeads : joined_row;
    const hip_bfloat16* weights = second ? b_weight : a_weight;
    hip_bfloat16* output = second ? b : a;
    float partial = 0.0F;
    const std::size_t base = static_cast<std::size_t>(row) * kColumns;
    for (std::uint32_t k = threadIdx.x; k < kColumns; k += blockDim.x) {
        partial = fmaf(static_cast<float>(input[k]), static_cast<float>(weights[base + k]), partial);
    }
    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t width = kThreads / 2; width != 0; width >>= 1) {
        if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = static_cast<hip_bfloat16>(reduction[0]);
}

__global__ void incumbent_control_kernel(const hip_bfloat16* a, const hip_bfloat16* b,
                                          const float* a_log, const float* dt_bias,
                                          float* g, float* beta) {
    const std::uint32_t head = threadIdx.x;
    if (head < kHeads) control_formula(a[head], b[head], a_log[head], dt_bias[head],
                                       g + head, beta + head);
}

__global__ void fused_projection_control_kernel(const hip_bfloat16* input,
                                                 const hip_bfloat16* a_weight,
                                                 const hip_bfloat16* b_weight,
                                                 const float* a_log, const float* dt_bias,
                                                 hip_bfloat16* a, hip_bfloat16* b,
                                                 float* g, float* beta) {
    const std::uint32_t head = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(head) * kColumns;
    float a_partial = 0.0F;
    float b_partial = 0.0F;
    for (std::uint32_t k = threadIdx.x; k < kColumns; k += blockDim.x) {
        const float x = static_cast<float>(input[k]);
        a_partial = fmaf(x, static_cast<float>(a_weight[base + k]), a_partial);
        b_partial = fmaf(x, static_cast<float>(b_weight[base + k]), b_partial);
    }
    __shared__ float a_reduction[kThreads];
    __shared__ float b_reduction[kThreads];
    a_reduction[threadIdx.x] = a_partial;
    b_reduction[threadIdx.x] = b_partial;
    __syncthreads();
    for (std::uint32_t width = kThreads / 2; width != 0; width >>= 1) {
        if (threadIdx.x < width) {
            a_reduction[threadIdx.x] += a_reduction[threadIdx.x + width];
            b_reduction[threadIdx.x] += b_reduction[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const hip_bfloat16 ar = static_cast<hip_bfloat16>(a_reduction[0]);
        const hip_bfloat16 br = static_cast<hip_bfloat16>(b_reduction[0]);
        a[head] = ar;
        b[head] = br;
        control_formula(ar, br, a_log[head], dt_bias[head], g + head, beta + head);
    }
}

__global__ void eviction_kernel(const std::uint32_t* source, std::uint32_t* sink,
                                std::size_t count) {
    std::uint32_t value = 0;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        value ^= source[index];
    }
    if (value != 0) atomicXor(sink, value);
}

bool ranges_overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto ab = reinterpret_cast<std::uintptr_t>(a);
    const auto bb = reinterpret_cast<std::uintptr_t>(b);
    return ab < bb + b_bytes && bb < ab + a_bytes;
}

bool valid(const Arguments& x) {
    if (x.input == nullptr || x.a_weight == nullptr || x.b_weight == nullptr ||
        x.a_log == nullptr || x.dt_bias == nullptr || x.a == nullptr || x.b == nullptr ||
        x.g == nullptr || x.beta == nullptr) return false;
    const std::array<const void*, 4> outputs{x.a, x.b, x.g, x.beta};
    const std::array<std::size_t, 4> sizes{kHeads * sizeof(hip_bfloat16),
                                           kHeads * sizeof(hip_bfloat16),
                                           kHeads * sizeof(float), kHeads * sizeof(float)};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            if (ranges_overlap(outputs[i], sizes[i], outputs[j], sizes[j])) return false;
        }
    }
    const std::array<const void*, 5> inputs{x.input, x.a_weight, x.b_weight, x.a_log, x.dt_bias};
    const std::array<std::size_t, 5> input_sizes{kColumns * sizeof(hip_bfloat16),
        kWeightElements * sizeof(hip_bfloat16), kWeightElements * sizeof(hip_bfloat16),
        kHeads * sizeof(float), kHeads * sizeof(float)};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (std::size_t j = 0; j < inputs.size(); ++j) {
            if (ranges_overlap(outputs[i], sizes[i], inputs[j], input_sizes[j])) return false;
        }
    }
    return true;
}

hipError_t launch_serial(const Arguments& x, hipStream_t stream = nullptr) {
    if (!valid(x)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(incumbent_projection_kernel, dim3(kHeads), dim3(kThreads), 0, stream,
                       x.input, x.a_weight, x.a);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(incumbent_projection_kernel, dim3(kHeads), dim3(kThreads), 0, stream,
                       x.input, x.b_weight, x.b);
    status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(incumbent_control_kernel, dim3(1), dim3(kThreads), 0, stream,
                       x.a, x.b, x.a_log, x.dt_bias, x.g, x.beta);
    return hipGetLastError();
}

hipError_t launch_combined(const Arguments& x, hipStream_t stream = nullptr) {
    if (!valid(x)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(combined_projection_kernel, dim3(2 * kHeads), dim3(kThreads), 0, stream,
                       x.input, x.a_weight, x.b_weight, x.a, x.b);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(incumbent_control_kernel, dim3(1), dim3(kThreads), 0, stream,
                       x.a, x.b, x.a_log, x.dt_bias, x.g, x.beta);
    return hipGetLastError();
}

hipError_t launch_fused(const Arguments& x, hipStream_t stream = nullptr) {
    if (!valid(x)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(fused_projection_control_kernel, dim3(kHeads), dim3(kThreads), 0, stream,
                       x.input, x.a_weight, x.b_weight, x.a_log, x.dt_bias,
                       x.a, x.b, x.g, x.beta);
    return hipGetLastError();
}

std::vector<hip_bfloat16> make_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<hip_bfloat16> result(count);
    std::uint32_t state = seed;
    for (auto& value : result) {
        state = state * 1664525U + 1013904223U;
        const float centered = static_cast<float>(static_cast<int>((state >> 9) % 4097U) - 2048);
        value = static_cast<hip_bfloat16>(centered * scale);
    }
    return result;
}

template <typename T> void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}

template <typename Guard> void initialize_guard(Guard& output) {
    using T = std::remove_pointer_t<decltype(output.payload())>;
    std::vector<T> host(output.count + 2);
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        for (auto& value : host) value.data = kCanaryBits;
    } else {
        for (auto& value : host) value = -123456.75F;
    }
    upload(output.storage, host);
}

template <typename Guard> auto download_guard(const Guard& output) {
    using T = std::remove_pointer_t<decltype(output.payload())>;
    std::vector<T> host(output.count + 2);
    HIP_CHECK(hipMemcpy(host.data(), output.storage.get(), output.storage.bytes(), hipMemcpyDeviceToHost));
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        if (host.front().data != kCanaryBits || host.back().data != kCanaryBits) fail("BF16 canary changed");
    } else {
        if (host.front() != -123456.75F || host.back() != -123456.75F) fail("FP32 canary changed");
    }
    return std::vector<T>(host.begin() + 1, host.end() - 1);
}

struct Outputs {
    GuardedBf16 a{kHeads};
    GuardedBf16 b{kHeads};
    GuardedFloat g{kHeads};
    GuardedFloat beta{kHeads};
};

void initialize(Outputs& output) {
    initialize_guard(output.a); initialize_guard(output.b);
    initialize_guard(output.g); initialize_guard(output.beta);
}

Arguments arguments(const DeviceBuffer<hip_bfloat16>& input,
                    const DeviceBuffer<hip_bfloat16>& aw,
                    const DeviceBuffer<hip_bfloat16>& bw,
                    const DeviceBuffer<float>& alog, const DeviceBuffer<float>& bias,
                    Outputs& output) {
    return {input.get(), aw.get(), bw.get(), alog.get(), bias.get(), output.a.payload(),
            output.b.payload(), output.g.payload(), output.beta.payload()};
}

std::uint16_t bits(hip_bfloat16 value) { return value.data; }

struct Correctness {
    double maximum_fp64_abs = 0.0;
    double maximum_projection_fp64_abs = 0.0;
    bool exact_serial_parity = true;
};

Correctness check_correctness(const std::vector<hip_bfloat16>& input,
                              const std::vector<hip_bfloat16>& aw,
                              const std::vector<hip_bfloat16>& bw,
                              const std::vector<float>& alog,
                              const std::vector<float>& bias,
                              const std::array<Outputs*, 3>& outputs) {
    std::array<std::vector<hip_bfloat16>, 3> actual_a;
    std::array<std::vector<hip_bfloat16>, 3> actual_b;
    std::array<std::vector<float>, 3> actual_g;
    std::array<std::vector<float>, 3> actual_beta;
    for (std::size_t route = 0; route < 3; ++route) {
        actual_a[route] = download_guard(outputs[route]->a);
        actual_b[route] = download_guard(outputs[route]->b);
        actual_g[route] = download_guard(outputs[route]->g);
        actual_beta[route] = download_guard(outputs[route]->beta);
    }
    Correctness result;
    for (std::uint32_t head = 0; head < kHeads; ++head) {
        double asum = 0.0, bsum = 0.0;
        const std::size_t base = static_cast<std::size_t>(head) * kColumns;
        for (std::uint32_t k = 0; k < kColumns; ++k) {
            const double x = static_cast<float>(input[k]);
            asum = std::fma(x, static_cast<double>(static_cast<float>(aw[base + k])), asum);
            bsum = std::fma(x, static_cast<double>(static_cast<float>(bw[base + k])), bsum);
        }
        const hip_bfloat16 ar(static_cast<float>(asum));
        const hip_bfloat16 br(static_cast<float>(bsum));
        const double av = static_cast<float>(ar) + bias[head];
        const double softplus = av > 20.0 ? av : std::log1p(std::exp(av));
        const double expected_g = -std::exp(static_cast<double>(alog[head])) * softplus;
        const double expected_beta = 1.0 / (1.0 + std::exp(-static_cast<double>(static_cast<float>(br))));
        for (std::size_t route = 0; route < 3; ++route) {
            if (bits(actual_a[route][head]) != bits(actual_a[0][head]) ||
                bits(actual_b[route][head]) != bits(actual_b[0][head]) ||
                std::bit_cast<std::uint32_t>(actual_g[route][head]) != std::bit_cast<std::uint32_t>(actual_g[0][head]) ||
                std::bit_cast<std::uint32_t>(actual_beta[route][head]) != std::bit_cast<std::uint32_t>(actual_beta[0][head])) {
                result.exact_serial_parity = false;
            }
            result.maximum_projection_fp64_abs = std::max(
                result.maximum_projection_fp64_abs,
                std::abs(static_cast<double>(static_cast<float>(actual_a[route][head])) - asum));
            result.maximum_projection_fp64_abs = std::max(
                result.maximum_projection_fp64_abs,
                std::abs(static_cast<double>(static_cast<float>(actual_b[route][head])) - bsum));
            result.maximum_fp64_abs = std::max(result.maximum_fp64_abs,
                std::abs(static_cast<double>(actual_g[route][head]) - expected_g));
            result.maximum_fp64_abs = std::max(result.maximum_fp64_abs,
                std::abs(static_cast<double>(actual_beta[route][head]) - expected_beta));
        }
    }
    if (!result.exact_serial_parity) fail("candidate differs bitwise from serial BF16/control boundaries");
    // The maintained raw BF16 Linear gate permits at most one low-order BF16 unit at this fixture
    // for the parallel-FP32 versus sequential-FP64 reduction association.
    if (result.maximum_projection_fp64_abs > 0.0625) fail("projection output exceeds FP64 tolerance");
    if (result.maximum_fp64_abs > 3.0e-4) fail("control output exceeds FP64 tolerance");
    return result;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct TrialRecord {
    std::size_t trial, order, position;
    int route;
    std::size_t copy;
    float milliseconds;
};

std::string shell_quote(const std::string& value) {
    std::string result = "'";
    for (char character : value) result += character == '\'' ? "'\\''" : std::string(1, character);
    return result + "'";
}

std::string command_output(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) fail("cannot execute identity command");
    std::array<char, 4096> buffer{};
    std::string result;
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) result += buffer.data();
    if (pclose(pipe) != 0) fail("identity command failed");
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    return result;
}

std::string sha256(const std::string& path) {
    const std::string result = command_output("/usr/bin/sha256sum -- " + shell_quote(path));
    if (result.size() < 65 || result[64] != ' ') fail("malformed sha256 for " + path);
    return result.substr(0, 64);
}

std::string executable_path() {
    std::array<char, 4096> buffer{};
    const ssize_t amount = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (amount <= 0) fail("cannot resolve executable path");
    return std::string(buffer.data(), static_cast<std::size_t>(amount));
}

std::string read_text(const std::string& path) {
    std::ifstream file(path);
    if (!file) fail("cannot read " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string json_escape(const std::string& value) {
    std::string result;
    for (char character : value) {
        if (character == '\\' || character == '"') result += '\\';
        if (character == '\n') result += "\\n";
        else if (character != '\r') result += character;
    }
    return result;
}

void write_create_only(const std::string& path, const std::string& content) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) fail("create-only report open failed: " + std::string(std::strerror(errno)));
    std::size_t offset = 0;
    while (offset < content.size()) {
        const ssize_t amount = write(fd, content.data() + offset, content.size() - offset);
        if (amount < 0) {
            const int saved = errno;
            close(fd);
            fail("report write failed: " + std::string(std::strerror(saved)));
        }
        offset += static_cast<std::size_t>(amount);
    }
    if (close(fd) != 0) fail("report close failed");
}

template <typename Launch> float timed(Launch&& launch) {
    Event begin, end;
    HIP_CHECK(hipEventRecord(begin.get()));
    HIP_CHECK(launch());
    HIP_CHECK(hipEventRecord(end.get()));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, begin.get(), end.get()));
    return ms;
}

std::string read_line(const char* path) {
    std::ifstream file(path);
    std::string value;
    std::getline(file, value);
    return value;
}

void write_json(const std::string& path, const std::string& assembly_path,
                const std::string& static_receipt_path, const std::string& profile,
                const hipDeviceProp_t& properties, int runtime_version,
                const Correctness& correctness,
                const std::array<std::vector<float>, 3>& samples,
                const std::vector<TrialRecord>& records) {
    const float serial = median(samples[0]);
    const float combined = median(samples[1]);
    const float fused = median(samples[2]);
    constexpr std::uint64_t weight_bytes = 2ULL * kWeightElements * sizeof(hip_bfloat16);
    constexpr std::uint64_t input_bytes = kColumns * sizeof(hip_bfloat16);
    constexpr std::uint64_t intermediate_bytes = 2ULL * kHeads * sizeof(hip_bfloat16);
    constexpr std::uint64_t parameter_bytes = 2ULL * kHeads * sizeof(float);
    constexpr std::uint64_t result_bytes = 2ULL * kHeads * sizeof(float);
    constexpr std::uint64_t incumbent_bytes = weight_bytes + 2 * input_bytes +
        2 * intermediate_bytes + parameter_bytes + result_bytes;
    constexpr std::uint64_t combined_bytes = weight_bytes + input_bytes +
        2 * intermediate_bytes + parameter_bytes + result_bytes;
    constexpr std::uint64_t fused_bytes = weight_bytes + input_bytes +
        intermediate_bytes + parameter_bytes + result_bytes;
    static_assert(incumbent_bytes - fused_bytes == 10432);
    const double best = std::min(combined, fused);
    const std::string executable = executable_path();
    const double possible_ms_per_token = 48.0 * std::max(0.0, static_cast<double>(serial) - best);
    std::ostringstream out;
    out << std::fixed << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer.r9700-bf16-gdn-control-t1-qualification.v1\",\n"
        << "  \"status\": \"pass\",\n  \"power_profile_before\": \"" << profile << "\",\n"
        << "  \"platform\": {\"device_name\": \"" << json_escape(properties.name)
        << "\", \"gcn_arch\": \"" << json_escape(properties.gcnArchName)
        << "\", \"pci_domain\": " << properties.pciDomainID
        << ", \"pci_bus\": " << properties.pciBusID
        << ", \"pci_device\": " << properties.pciDeviceID
        << ", \"hip_runtime_version\": " << runtime_version
        << ", \"rocm_hipcc_version\": \""
        << json_escape(command_output("/opt/rocm/bin/hipcc --version 2>&1")) << "\"},\n"
        << "  \"identities\": {\"executable_path\": \"" << json_escape(executable)
        << "\", \"executable_sha256\": \"" << sha256(executable)
        << "\", \"source_path\": \"tools/r9700/bf16_gdn_control_t1_qual.hip\", \"source_sha256\": \""
        << sha256("tools/r9700/bf16_gdn_control_t1_qual.hip")
        << "\", \"checker_path\": \"tools/r9700/check_bf16_gdn_control_t1_static.py\", \"checker_sha256\": \""
        << sha256("tools/r9700/check_bf16_gdn_control_t1_static.py")
        << "\", \"assembly_path\": \"" << json_escape(assembly_path) << "\", \"assembly_sha256\": \""
        << sha256(assembly_path) << "\", \"static_receipt_path\": \""
        << json_escape(static_receipt_path) << "\", \"static_receipt_sha256\": \""
        << sha256(static_receipt_path) << "\", \"static_receipt\": \""
        << json_escape(read_text(static_receipt_path)) << "\"},\n"
        << "  \"contract\": {\"tokens\": 1, \"heads\": 48, \"columns\": 5120, "
           "\"layers_per_token\": 48, \"weight_format\": \"BF16\", "
           "\"intermediate_boundary\": \"round_each_projection_once_to_BF16_before_control\"},\n"
        << "  \"layer0\": {\"classification\": \"launch_and_cross_op_materialization\", "
        << "\"accounting_scope\": \"logical_Op_boundary_not_physical_CTA_transactions\", "
        << "\"terms_bytes\": {\"two_weights\": " << weight_bytes << ", \"one_input\": "
        << input_bytes << ", \"a_b_intermediates\": " << intermediate_bytes
        << ", \"control_parameters\": " << parameter_bytes << ", \"g_beta_results\": "
        << result_bytes << "}, \"incumbent_formula\": \"weights + 2*input + projection_writes + control_reads + parameters + results\", "
        << "\"incumbent_bytes\": " << incumbent_bytes << ", \"combined_formula\": \"weights + input + projection_writes + control_reads + parameters + results\", "
        << "\"combined_bytes\": " << combined_bytes << ", \"fused_formula\": \"weights + input + projection_writes + parameters + results\", "
        << "\"fused_bytes\": " << fused_bytes << ", \"fused_avoidable_bytes\": 10432},\n"
        << "  \"correctness\": {\"projection_fp64_max_abs\": "
        << correctness.maximum_projection_fp64_abs
        << ", \"control_fp64_max_abs\": " << correctness.maximum_fp64_abs
        << ", \"candidate_serial_bit_exact\": true, \"canaries\": \"pass\", "
           "\"malformed_and_alias_rejection\": \"pass\"},\n"
        << "  \"timing_ms_per_layer\": {\"serial_median\": " << serial
        << ", \"combined_median\": " << combined << ", \"fused_median\": " << fused << "},\n"
        << "  \"whole_token_bound\": {\"best_possible_observed_saving_ms\": "
        << possible_ms_per_token << ", \"material_threshold_ms\": 0.200000000, "
        << "\"admitted\": " << (possible_ms_per_token >= 0.2 ? "true" : "false") << "},\n"
        << "  \"schedule\": {\"trials\": 12, \"orders\": [[0,1,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[0,2,1]], "
        << "\"route_names\": [\"serial\",\"combined\",\"fused\"], \"weight_copies\": 3, \"samples_per_route_copy\": 4, "
        << "\"eviction_bytes_before_each_event\": 100663296},\n"
        << "  \"samples_ms\": {\n";
    const char* names[] = {"serial", "combined", "fused"};
    for (std::size_t route = 0; route < 3; ++route) {
        out << "    \"" << names[route] << "\": [";
        for (std::size_t i = 0; i < samples[route].size(); ++i) {
            if (i != 0) out << ", ";
            out << samples[route][i];
        }
        out << "]" << (route == 2 ? "\n" : ",\n");
    }
    out << "  },\n  \"per_copy_medians_ms\": {\n";
    for (std::size_t route = 0; route < 3; ++route) {
        out << "    \"" << names[route] << "\": [";
        for (std::size_t copy = 0; copy < kCopies; ++copy) {
            std::vector<float> selected;
            for (const auto& record : records)
                if (record.route == static_cast<int>(route) && record.copy == copy)
                    selected.push_back(record.milliseconds);
            if (selected.size() != 4) fail("route/copy schedule is not balanced");
            if (copy != 0) out << ", ";
            out << median(std::move(selected));
        }
        out << "]" << (route == 2 ? "\n" : ",\n");
    }
    out << "  },\n  \"trial_records\": [\n";
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        out << "    {\"trial\": " << record.trial << ", \"order\": " << record.order
            << ", \"position\": " << record.position << ", \"route\": " << record.route
            << ", \"copy\": " << record.copy << ", \"milliseconds\": " << record.milliseconds << "}"
            << (index + 1 == records.size() ? "\n" : ",\n");
    }
    out << "  ],\n  \"power_profile_after\": \""
        << read_line("/sys/class/drm/card2/device/power_dpm_force_performance_level") << "\"\n}\n";
    write_create_only(path, out.str());
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 7 || std::string(argv[1]) != "--output" ||
        std::string(argv[3]) != "--assembly" || std::string(argv[5]) != "--static-receipt") {
        std::cerr << "usage: " << argv[0]
                  << " --output REPORT.json --assembly FILE.s --static-receipt FILE.txt\n";
        return 2;
    }
    const std::string profile = read_line("/sys/class/drm/card2/device/power_dpm_force_performance_level");
    if (profile != "auto") fail("performance admission requires power profile auto before HIP initialization");
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    if (count != 1) fail("qualifier requires exactly one visible GPU");
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    if (std::string(properties.gcnArchName).rfind("gfx1201", 0) != 0) fail("qualifier requires gfx1201");
    int runtime_version = 0;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));

    const auto input = make_bf16(kColumns, 1, 1.0F / 1024.0F);
    const auto aw = make_bf16(kWeightElements, 3, 1.0F / 16384.0F);
    const auto bw = make_bf16(kWeightElements, 7, 1.0F / 16384.0F);
    std::vector<float> alog(kHeads), bias(kHeads);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
        alog[h] = -2.5F + static_cast<float>(h) / 64.0F;
        bias[h] = -0.75F + static_cast<float>(h) / 80.0F;
    }
    DeviceBuffer<hip_bfloat16> d_input(kColumns);
    DeviceBuffer<float> d_alog(kHeads), d_bias(kHeads);
    upload(d_input, input); upload(d_alog, alog); upload(d_bias, bias);
    std::array<DeviceBuffer<hip_bfloat16>, kCopies> d_aw{
        DeviceBuffer<hip_bfloat16>(kWeightElements), DeviceBuffer<hip_bfloat16>(kWeightElements),
        DeviceBuffer<hip_bfloat16>(kWeightElements)};
    std::array<DeviceBuffer<hip_bfloat16>, kCopies> d_bw{
        DeviceBuffer<hip_bfloat16>(kWeightElements), DeviceBuffer<hip_bfloat16>(kWeightElements),
        DeviceBuffer<hip_bfloat16>(kWeightElements)};
    for (std::size_t copy = 0; copy < kCopies; ++copy) { upload(d_aw[copy], aw); upload(d_bw[copy], bw); }

    std::array<Outputs, 3> outputs;
    for (auto& output : outputs) initialize(output);
    auto serial_args = arguments(d_input, d_aw[0], d_bw[0], d_alog, d_bias, outputs[0]);
    auto combined_args = arguments(d_input, d_aw[1], d_bw[1], d_alog, d_bias, outputs[1]);
    auto fused_args = arguments(d_input, d_aw[2], d_bw[2], d_alog, d_bias, outputs[2]);
    HIP_CHECK(launch_serial(serial_args));
    HIP_CHECK(launch_combined(combined_args));
    HIP_CHECK(launch_fused(fused_args));
    HIP_CHECK(hipDeviceSynchronize());
    const Correctness correctness = check_correctness(input, aw, bw, alog, bias,
                                                       {&outputs[0], &outputs[1], &outputs[2]});
    Arguments malformed = serial_args;
    malformed.input = nullptr;
    if (launch_serial(malformed) != hipErrorInvalidValue ||
        launch_combined(malformed) != hipErrorInvalidValue ||
        launch_fused(malformed) != hipErrorInvalidValue) fail("null input was not rejected");
    malformed = serial_args;
    malformed.b = malformed.a;
    if (launch_serial(malformed) != hipErrorInvalidValue ||
        launch_combined(malformed) != hipErrorInvalidValue ||
        launch_fused(malformed) != hipErrorInvalidValue) fail("output alias was not rejected");
    malformed = serial_args;
    malformed.a = const_cast<hip_bfloat16*>(malformed.input);
    if (launch_serial(malformed) != hipErrorInvalidValue ||
        launch_combined(malformed) != hipErrorInvalidValue ||
        launch_fused(malformed) != hipErrorInvalidValue) fail("input/output alias was not rejected");

    constexpr std::size_t kEvictionWords = 24U * 1024U * 1024U;
    DeviceBuffer<std::uint32_t> eviction(kEvictionWords), sink(1);
    HIP_CHECK(hipMemset(eviction.get(), 0xa5, eviction.bytes()));
    HIP_CHECK(hipMemset(sink.get(), 0, sink.bytes()));
    const std::array<std::array<int, 3>, 6> orders{{{{0,1,2}}, {{1,2,0}}, {{2,0,1}},
                                                   {{2,1,0}}, {{1,0,2}}, {{0,2,1}}}};
    std::array<std::vector<float>, 3> samples;
    std::vector<TrialRecord> records;
    for (std::size_t trial = 0; trial < kTrials; ++trial) {
        const auto& order = orders[trial % orders.size()];
        for (std::size_t position = 0; position < 3; ++position) {
            const int route = order[position];
            const std::size_t copy = (trial + static_cast<std::size_t>(route)) % kCopies;
            hipLaunchKernelGGL(eviction_kernel, dim3(1024), dim3(256), 0, nullptr,
                               eviction.get(), sink.get(), eviction.count());
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            initialize(outputs[route]);
            Arguments x = arguments(d_input, d_aw[copy], d_bw[copy], d_alog, d_bias, outputs[route]);
            const float elapsed = timed([&] {
                if (route == 0) return launch_serial(x);
                if (route == 1) return launch_combined(x);
                return launch_fused(x);
            });
            samples[route].push_back(elapsed);
            records.push_back({trial, trial % orders.size(), position, route, copy, elapsed});
        }
    }
    if (read_line("/sys/class/drm/card2/device/power_dpm_force_performance_level") != "auto")
        fail("power profile changed during timing");
    write_json(argv[2], argv[4], argv[6], profile, properties, runtime_version,
               correctness, samples, records);
    std::cout << "PASS wrote " << argv[2] << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
