#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
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
#include <unistd.h>
#include <vector>

namespace {

constexpr std::uint32_t kHeadDim = 256U;
constexpr std::uint32_t kQueryHeads = 24U;
constexpr std::uint32_t kKvHeads = 4U;
constexpr std::uint32_t kQueriesPerKv = 6U;
constexpr std::uint32_t kValueGroup = 16U;
constexpr std::uint32_t kPageSize = 64U;
constexpr std::uint32_t kChunk = 512U;
constexpr std::uint32_t kContext = 8192U;
constexpr std::uint32_t kChunks = kContext / kChunk;
constexpr std::uint32_t kMappedPages = kContext / kPageSize;
constexpr std::uint32_t kPhysicalPages = kMappedPages + 3U;
constexpr std::uint32_t kPhysicalTokens = kPhysicalPages * kPageSize;
constexpr std::uint32_t kRecords = kQueryHeads * kChunks;
constexpr std::uint32_t kGuard = 32U;
constexpr std::uint32_t kCanary = 0x4a123456U;
constexpr std::uint32_t kPoison = 0x7fc01234U;
constexpr std::uint32_t kSamples = 15U;
constexpr float kMinimumIntervalMs = 10.0F;
constexpr double kMaximumUpperRatio = 0.98;
constexpr std::string_view kPowerPath =
    "/sys/class/drm/card2/device/power_dpm_force_performance_level";
static_assert(kContext % kChunk == 0U);
static_assert(kQueryHeads == kKvHeads * kQueriesPerKv);

#ifndef NINFER_SOURCE_DIR
#error "NINFER_SOURCE_DIR must bind grouped-PV qualification provenance"
#endif

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) return;
    std::ostringstream out;
    out << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(out.str());
}
#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

std::string shell_quote(std::string_view value) {
    std::string result("'");
    for (const char character : value) {
        if (character == '\'') result += "'\\''";
        else result += character;
    }
    result += '\'';
    return result;
}

std::string sha256_file(const std::filesystem::path& path) {
    const std::string command = "sha256sum -- " + shell_quote(path.string());
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) fail("cannot launch sha256sum for " + path.string());
    std::array<char, 256> buffer{};
    const char* line = std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe);
    const int status = pclose(pipe);
    if (line == nullptr || status != 0) fail("cannot hash " + path.string());
    const std::string output(buffer.data());
    const std::size_t separator = output.find_first_of(" \t");
    const std::string digest = output.substr(0U, separator);
    if (digest.size() != 64U ||
        !std::all_of(digest.begin(), digest.end(), [](char value) {
            return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
        })) {
        fail("sha256sum returned an invalid digest for " + path.string());
    }
    return digest;
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char character : value) {
        switch (character) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (character < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(character) << std::dec;
            } else {
                out << static_cast<char>(character);
            }
        }
    }
    return out.str();
}

template <class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        HIP_CHECK(hipMalloc(&pointer_, elements * sizeof(T)));
    }
    ~DeviceBuffer() { (void)hipFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* data() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return elements_; }
private:
    T* pointer_ = nullptr;
    std::size_t elements_ = 0U;
};

template <class T> void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() != source.size()) fail("upload size mismatch");
    HIP_CHECK(hipMemcpy(destination.data(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}

template <class T> std::vector<T> download(const DeviceBuffer<T>& source) {
    std::vector<T> result(source.size());
    HIP_CHECK(hipMemcpy(result.data(), source.data(), result.size() * sizeof(T),
                        hipMemcpyDeviceToHost));
    return result;
}

[[nodiscard]] constexpr std::size_t value_offset(std::size_t physical, std::size_t kv_head,
                                                  std::size_t feature) noexcept {
    return feature / 2U + (kHeadDim / 2U) *
        (physical % kPageSize + kPageSize *
         (kv_head + kKvHeads * (physical / kPageSize)));
}

[[nodiscard]] constexpr std::size_t scale_offset(std::size_t physical, std::size_t kv_head,
                                                  std::size_t feature) noexcept {
    return feature / kValueGroup + (kHeadDim / kValueGroup) *
        (physical % kPageSize + kPageSize *
         (kv_head + kKvHeads * (physical / kPageSize)));
}

__device__ __forceinline__ std::size_t device_value_offset(
    std::size_t physical, std::uint32_t kv_head, std::uint32_t feature) {
    return feature / 2U + (kHeadDim / 2U) *
        (physical % kPageSize + kPageSize *
         (kv_head + kKvHeads * (physical / kPageSize)));
}

__device__ __forceinline__ std::size_t device_scale_offset(
    std::size_t physical, std::uint32_t kv_head, std::uint32_t feature) {
    return feature / kValueGroup + (kHeadDim / kValueGroup) *
        (physical % kPageSize + kPageSize *
         (kv_head + kKvHeads * (physical / kPageSize)));
}

__device__ __forceinline__ float wave_max(float value) {
#pragma unroll
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        value = fmaxf(value, __shfl_down(value, stride, 32));
    }
    return value;
}

__device__ __forceinline__ float wave_sum(float value) {
#pragma unroll
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
        value += __shfl_down(value, stride, 32);
    }
    return value;
}

// Incumbent split-512 producer: one CTA owns one (query head, chunk) record.
__global__ void split512_pv_incumbent_kernel(
    const float* scores, const std::uint8_t* values, const std::uint16_t* scales,
    const std::uint32_t* pages, float* maxima, float* sums, float* numerators) {
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t chunk = blockIdx.x;
    const std::uint32_t query_head = blockIdx.y;
    const std::uint32_t kv_head = query_head / kQueriesPerKv;
    const std::uint32_t record = query_head * kChunks + chunk;
    const std::uint32_t logical_base = chunk * kChunk;
    __shared__ float weights[kChunk];
    __shared__ std::uint32_t physical[kChunk];
    __shared__ float wave_values[kHeadDim / 32U];
    __shared__ float maximum;
    __shared__ float denominator;

    float local_maximum = -INFINITY;
    for (std::uint32_t offset = thread; offset < kChunk; offset += blockDim.x) {
        const float score = scores[query_head * kContext + logical_base + offset];
        weights[offset] = score;
        local_maximum = fmaxf(local_maximum, score);
        const std::uint32_t logical = logical_base + offset;
        physical[offset] = pages[logical / kPageSize] * kPageSize + logical % kPageSize;
    }
    local_maximum = wave_max(local_maximum);
    if (lane == 0U) wave_values[wave] = local_maximum;
    __syncthreads();
    if (thread == 0U) {
        float reduced = wave_values[0];
#pragma unroll
        for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
            reduced = fmaxf(reduced, wave_values[index]);
        }
        maximum = reduced;
    }
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t offset = thread; offset < kChunk; offset += blockDim.x) {
        const float weight = __expf(weights[offset] - maximum);
        weights[offset] = weight;
        local_sum += weight;
    }
    local_sum = wave_sum(local_sum);
    if (lane == 0U) wave_values[wave] = local_sum;
    __syncthreads();
    if (thread == 0U) {
        float reduced = 0.0F;
#pragma unroll
        for (std::uint32_t index = 0U; index < kHeadDim / 32U; ++index) {
            reduced += wave_values[index];
        }
        denominator = reduced;
        maxima[record] = maximum;
        sums[record] = reduced;
    }
    __syncthreads();

    float numerator = 0.0F;
#pragma unroll 1
    for (std::uint32_t offset = 0U; offset < kChunk; ++offset) {
        const std::uint8_t packed = values[
            device_value_offset(physical[offset], kv_head, thread)];
        int code = static_cast<int>((packed >> ((thread & 1U) * 4U)) & 0x0fU);
        code = code >= 8 ? code - 16 : code;
        const std::uint16_t scale_bits = scales[
            device_scale_offset(physical[offset], kv_head, thread)];
        numerator = fmaf(weights[offset],
                         static_cast<float>(code) * __half2float(__ushort_as_half(scale_bits)),
                         numerator);
    }
    numerators[record * kHeadDim + thread] = numerator;
}

// Challenger: one CTA owns one (KV head, chunk), translates and loads V once, and produces the
// six associated query-head records with the same per-head FP32 reduction and accumulation order.
__global__ void split512_pv_grouped_kernel(
    const float* scores, const std::uint8_t* values, const std::uint16_t* scales,
    const std::uint32_t* pages, float* maxima, float* sums, float* numerators) {
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t chunk = blockIdx.x;
    const std::uint32_t kv_head = blockIdx.y;
    const std::uint32_t logical_base = chunk * kChunk;
    __shared__ float weights[kQueriesPerKv][kChunk];
    __shared__ std::uint32_t physical[kChunk];
    __shared__ float wave_values[kHeadDim / 32U];
    __shared__ float maximum;
    __shared__ float denominator;

    for (std::uint32_t offset = thread; offset < kChunk; offset += blockDim.x) {
        const std::uint32_t logical = logical_base + offset;
        physical[offset] = pages[logical / kPageSize] * kPageSize + logical % kPageSize;
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t local_head = 0U; local_head < kQueriesPerKv; ++local_head) {
        const std::uint32_t query_head = kv_head * kQueriesPerKv + local_head;
        const std::uint32_t record = query_head * kChunks + chunk;
        float local_maximum = -INFINITY;
        for (std::uint32_t offset = thread; offset < kChunk; offset += blockDim.x) {
            const float score = scores[query_head * kContext + logical_base + offset];
            weights[local_head][offset] = score;
            local_maximum = fmaxf(local_maximum, score);
        }
        local_maximum = wave_max(local_maximum);
        if (lane == 0U) wave_values[wave] = local_maximum;
        __syncthreads();
        if (thread == 0U) {
            float reduced = wave_values[0];
#pragma unroll
            for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
                reduced = fmaxf(reduced, wave_values[index]);
            }
            maximum = reduced;
        }
        __syncthreads();
        float local_sum = 0.0F;
        for (std::uint32_t offset = thread; offset < kChunk; offset += blockDim.x) {
            const float weight = __expf(weights[local_head][offset] - maximum);
            weights[local_head][offset] = weight;
            local_sum += weight;
        }
        local_sum = wave_sum(local_sum);
        if (lane == 0U) wave_values[wave] = local_sum;
        __syncthreads();
        if (thread == 0U) {
            float reduced = 0.0F;
#pragma unroll
            for (std::uint32_t index = 0U; index < kHeadDim / 32U; ++index) {
                reduced += wave_values[index];
            }
            denominator = reduced;
            maxima[record] = maximum;
            sums[record] = reduced;
        }
        __syncthreads();
    }

    float accumulators[kQueriesPerKv]{};
#pragma unroll 1
    for (std::uint32_t offset = 0U; offset < kChunk; ++offset) {
        const std::uint8_t packed = values[
            device_value_offset(physical[offset], kv_head, thread)];
        int code = static_cast<int>((packed >> ((thread & 1U) * 4U)) & 0x0fU);
        code = code >= 8 ? code - 16 : code;
        const std::uint16_t scale_bits = scales[
            device_scale_offset(physical[offset], kv_head, thread)];
        const float represented_value =
            static_cast<float>(code) * __half2float(__ushort_as_half(scale_bits));
#pragma unroll
        for (std::uint32_t local_head = 0U; local_head < kQueriesPerKv; ++local_head) {
            accumulators[local_head] = fmaf(weights[local_head][offset], represented_value,
                                            accumulators[local_head]);
        }
    }
#pragma unroll
    for (std::uint32_t local_head = 0U; local_head < kQueriesPerKv; ++local_head) {
        const std::uint32_t query_head = kv_head * kQueriesPerKv + local_head;
        const std::uint32_t record = query_head * kChunks + chunk;
        numerators[record * kHeadDim + thread] = accumulators[local_head];
    }
}

__global__ void split512_shared_merge_kernel(
    const float* maxima, const float* sums, const float* numerators, float* output) {
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = thread & 31U;
    const std::uint32_t wave = thread >> 5U;
    const std::uint32_t query_head = blockIdx.x;
    const std::uint32_t first = query_head * kChunks;
    __shared__ float coefficients[kChunks];
    __shared__ float wave_values[kHeadDim / 32U];
    __shared__ float maximum;
    __shared__ float denominator;

    float local_maximum = -INFINITY;
    for (std::uint32_t chunk = thread; chunk < kChunks; chunk += blockDim.x) {
        local_maximum = fmaxf(local_maximum, maxima[first + chunk]);
    }
    local_maximum = wave_max(local_maximum);
    if (lane == 0U) wave_values[wave] = local_maximum;
    __syncthreads();
    if (thread == 0U) {
        float reduced = wave_values[0];
#pragma unroll
        for (std::uint32_t index = 1U; index < kHeadDim / 32U; ++index) {
            reduced = fmaxf(reduced, wave_values[index]);
        }
        maximum = reduced;
    }
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t chunk = thread; chunk < kChunks; chunk += blockDim.x) {
        const float coefficient = __expf(maxima[first + chunk] - maximum);
        coefficients[chunk] = coefficient;
        local_sum += sums[first + chunk] * coefficient;
    }
    local_sum = wave_sum(local_sum);
    if (lane == 0U) wave_values[wave] = local_sum;
    __syncthreads();
    if (thread == 0U) {
        float reduced = 0.0F;
#pragma unroll
        for (std::uint32_t index = 0U; index < kHeadDim / 32U; ++index) {
            reduced += wave_values[index];
        }
        denominator = reduced;
    }
    __syncthreads();

    float numerator = 0.0F;
#pragma unroll
    for (std::uint32_t chunk = 0U; chunk < kChunks; ++chunk) {
        numerator = fmaf(coefficients[chunk],
                         numerators[(first + chunk) * kHeadDim + thread], numerator);
    }
    output[query_head * kHeadDim + thread] = numerator / denominator;
}

struct HostFixture {
    std::vector<std::uint32_t> pages;
    std::vector<std::uint8_t> values;
    std::vector<std::uint16_t> scales;
    std::vector<float> scores;
};

HostFixture make_fixture() {
    HostFixture fixture{
        std::vector<std::uint32_t>(kMappedPages),
        std::vector<std::uint8_t>(static_cast<std::size_t>(kPhysicalTokens) * kKvHeads *
                                  kHeadDim / 2U, 0U),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kPhysicalTokens) * kKvHeads *
                                   (kHeadDim / kValueGroup), 0U),
        std::vector<float>(static_cast<std::size_t>(kQueryHeads) * kContext)};
    for (std::uint32_t page = 0U; page < kMappedPages; ++page) {
        fixture.pages[page] = (page * 7U + 2U) % kPhysicalPages;
    }
    for (std::uint32_t page = 1U; page < kMappedPages; ++page) {
        if (fixture.pages[page] == fixture.pages[page - 1U] ||
            fixture.pages[page] == fixture.pages[page - 1U] + 1U) {
            fail("page-table fixture is not fragmented");
        }
    }
    for (std::uint32_t logical = 0U; logical < kContext; ++logical) {
        const std::uint32_t physical = fixture.pages[logical / kPageSize] * kPageSize +
                                       logical % kPageSize;
        for (std::uint32_t kv_head = 0U; kv_head < kKvHeads; ++kv_head) {
            for (std::uint32_t feature = 0U; feature < kHeadDim; feature += 2U) {
                const int low = static_cast<int>((logical * 11U + kv_head * 3U +
                                                  feature * 5U) % 15U) - 7;
                const int high = static_cast<int>((logical * 13U + kv_head * 7U +
                                                   feature * 2U + 1U) % 15U) - 7;
                fixture.values[value_offset(physical, kv_head, feature)] =
                    static_cast<std::uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
            }
            for (std::uint32_t feature = 0U; feature < kHeadDim; feature += kValueGroup) {
                constexpr std::array<std::uint16_t, 8> words{
                    0x2800U, 0x2c00U, 0x3000U, 0x3400U,
                    0x3800U, 0x3a00U, 0x3c00U, 0x3e00U};
                fixture.scales[scale_offset(physical, kv_head, feature)] =
                    words[(logical * 3U + kv_head * 5U + feature / kValueGroup) % words.size()];
            }
        }
    }
    for (std::uint32_t query_head = 0U; query_head < kQueryHeads; ++query_head) {
        for (std::uint32_t logical = 0U; logical < kContext; ++logical) {
            const int code = static_cast<int>((logical * 17U + query_head * 29U +
                                               (logical / kChunk) * 11U) % 257U) - 128;
            fixture.scores[query_head * kContext + logical] =
                static_cast<float>(code) / 32.0F;
        }
    }
    return fixture;
}

float fp16_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    const std::uint32_t mantissa = bits & 0x03ffU;
    std::uint32_t result = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) return std::bit_cast<float>(sign);
        std::uint32_t normalized = mantissa;
        std::uint32_t shift = 0U;
        while ((normalized & 0x0400U) == 0U) { normalized <<= 1U; ++shift; }
        result = sign | ((127U - 15U - shift) << 23U) |
                 ((normalized & 0x03ffU) << 13U);
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        result = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(result);
}

std::vector<double> complete_oracle(const HostFixture& fixture) {
    std::vector<double> output(static_cast<std::size_t>(kQueryHeads) * kHeadDim, 0.0);
    std::vector<double> probabilities(kContext);
    for (std::uint32_t query_head = 0U; query_head < kQueryHeads; ++query_head) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::uint32_t logical = 0U; logical < kContext; ++logical) {
            maximum = std::max(maximum,
                static_cast<double>(fixture.scores[query_head * kContext + logical]));
        }
        double denominator = 0.0;
        for (std::uint32_t logical = 0U; logical < kContext; ++logical) {
            probabilities[logical] = std::exp(
                static_cast<double>(fixture.scores[query_head * kContext + logical]) - maximum);
            denominator += probabilities[logical];
        }
        const std::uint32_t kv_head = query_head / kQueriesPerKv;
        for (std::uint32_t feature = 0U; feature < kHeadDim; ++feature) {
            double numerator = 0.0;
            for (std::uint32_t logical = 0U; logical < kContext; ++logical) {
                const std::uint32_t physical = fixture.pages[logical / kPageSize] * kPageSize +
                                               logical % kPageSize;
                const std::uint8_t packed =
                    fixture.values[value_offset(physical, kv_head, feature)];
                int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
                code = code >= 8 ? code - 16 : code;
                numerator += probabilities[logical] * static_cast<double>(code) *
                    static_cast<double>(fp16_to_float(
                        fixture.scales[scale_offset(physical, kv_head, feature)]));
            }
            output[query_head * kHeadDim + feature] = numerator / denominator;
        }
    }
    return output;
}

struct GuardedFloats {
    DeviceBuffer<std::uint32_t> storage;
    std::size_t logical;
    explicit GuardedFloats(std::size_t elements)
        : storage(elements + 2U * kGuard), logical(elements) {}
    [[nodiscard]] float* data() const noexcept {
        return reinterpret_cast<float*>(storage.data() + kGuard);
    }
    void poison() {
        std::vector<std::uint32_t> bits(storage.size(), kPoison);
        std::fill_n(bits.begin(), kGuard, kCanary);
        std::fill_n(bits.end() - kGuard, kGuard, kCanary);
        upload(storage, bits);
    }
    [[nodiscard]] std::vector<float> checked(const std::string& label) const {
        const auto bits = download(storage);
        for (std::uint32_t index = 0U; index < kGuard; ++index) {
            if (bits[index] != kCanary || bits[kGuard + logical + index] != kCanary) {
                fail(label + " canary was overwritten");
            }
        }
        std::vector<float> values(logical);
        for (std::size_t index = 0U; index < logical; ++index) {
            if (bits[kGuard + index] == kPoison) fail(label + " was not fully overwritten");
            values[index] = std::bit_cast<float>(bits[kGuard + index]);
        }
        return values;
    }
};

struct Workspace {
    GuardedFloats maxima{kRecords};
    GuardedFloats sums{kRecords};
    GuardedFloats numerators{static_cast<std::size_t>(kRecords) * kHeadDim};
    GuardedFloats output{static_cast<std::size_t>(kQueryHeads) * kHeadDim};
    void poison() { maxima.poison(); sums.poison(); numerators.poison(); output.poison(); }
};

struct DeviceFixture {
    DeviceBuffer<float> scores;
    DeviceBuffer<std::uint8_t> values;
    DeviceBuffer<std::uint16_t> scales;
    DeviceBuffer<std::uint32_t> pages;
    explicit DeviceFixture(const HostFixture& fixture)
        : scores(fixture.scores.size()), values(fixture.values.size()),
          scales(fixture.scales.size()), pages(fixture.pages.size()) {
        upload(scores, fixture.scores);
        upload(values, fixture.values);
        upload(scales, fixture.scales);
        upload(pages, fixture.pages);
    }
};

enum class Producer { Incumbent, Grouped };

void launch(Producer producer, const DeviceFixture& fixture, Workspace& workspace,
            hipStream_t stream) {
    if (producer == Producer::Incumbent) {
        hipLaunchKernelGGL(split512_pv_incumbent_kernel, dim3(kChunks, kQueryHeads),
                           dim3(kHeadDim), 0, stream, fixture.scores.data(), fixture.values.data(),
                           fixture.scales.data(), fixture.pages.data(), workspace.maxima.data(),
                           workspace.sums.data(), workspace.numerators.data());
    } else {
        hipLaunchKernelGGL(split512_pv_grouped_kernel, dim3(kChunks, kKvHeads), dim3(kHeadDim),
                           0, stream, fixture.scores.data(), fixture.values.data(),
                           fixture.scales.data(), fixture.pages.data(), workspace.maxima.data(),
                           workspace.sums.data(), workspace.numerators.data());
    }
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(split512_shared_merge_kernel, dim3(kQueryHeads), dim3(kHeadDim), 0,
                       stream, workspace.maxima.data(), workspace.sums.data(),
                       workspace.numerators.data(), workspace.output.data());
    HIP_CHECK(hipGetLastError());
}

void compare_oracle(const std::vector<float>& actual, const std::vector<double>& expected,
                    const std::string& label) {
    if (actual.size() != expected.size()) fail(label + " extent mismatch");
    double max_absolute = 0.0;
    double max_relative = 0.0;
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) fail(label + " produced a nonfinite output");
        const double absolute = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double relative = absolute / std::max(std::abs(expected[index]), 1.0e-6);
        max_absolute = std::max(max_absolute, absolute);
        max_relative = std::max(max_relative, relative);
        if (absolute > 7.5e-4 && relative > 7.5e-4) {
            std::ostringstream out;
            out << label << " differs from FP64 oracle at " << index << ": actual="
                << actual[index] << " expected=" << expected[index] << " abs=" << absolute
                << " rel=" << relative;
            fail(out.str());
        }
    }
    std::cout << label << " max_abs=" << max_absolute << " max_rel=" << max_relative << '\n';
}

void require_exact(const std::vector<float>& incumbent, const std::vector<float>& grouped) {
    if (incumbent.size() != grouped.size()) fail("parity extent mismatch");
    for (std::size_t index = 0U; index < incumbent.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(incumbent[index]) !=
            std::bit_cast<std::uint32_t>(grouped[index])) {
            fail("grouped/incumbent output parity differs at " + std::to_string(index));
        }
    }
}

struct Resources {
    int registers{};
    int shared_bytes{};
    int local_bytes{};
    int maximum_threads{};
};

Resources resources(const void* kernel, const char* label, int expected_shared_bytes) {
    hipFuncAttributes attributes{};
    HIP_CHECK(hipFuncGetAttributes(&attributes, kernel));
    Resources result{attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
                     static_cast<int>(attributes.localSizeBytes), attributes.maxThreadsPerBlock};
    if (result.maximum_threads < static_cast<int>(kHeadDim) || result.local_bytes != 0 ||
        result.registers <= 0 || result.registers > 48 ||
        result.shared_bytes != expected_shared_bytes) {
        fail(std::string(label) + " fails the resource envelope");
    }
    return result;
}

float timed_interval(Producer producer, const DeviceFixture& fixture, Workspace& workspace,
                     hipStream_t stream, std::uint32_t repetitions,
                     hipEvent_t begin, hipEvent_t end) {
    HIP_CHECK(hipEventRecord(begin, stream));
    for (std::uint32_t iteration = 0U; iteration < repetitions; ++iteration) {
        launch(producer, fixture, workspace, stream);
    }
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed, begin, end));
    if (!std::isfinite(elapsed) || !(elapsed > 0.0F)) fail("invalid timing interval");
    return elapsed;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

std::string require_auto_power() {
    std::ifstream input{std::string(kPowerPath)};
    if (!input) fail("cannot read R9700 power profile");
    std::string value;
    std::getline(input, value);
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char character) {
        return character == ' ' || character == '\t' || character == '\r' || character == '\n';
    }), value.end());
    if (value != "auto") fail("performance qualification requires power profile auto");
    return value;
}

void select_device(std::string& name, std::string& architecture, int& wave_size) {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    if (count <= 0) fail("no HIP devices");
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    name = properties.name;
    architecture = properties.gcnArchName;
    wave_size = properties.warpSize;
    if (architecture != "gfx1201" || name.find("R9700") == std::string::npos ||
        wave_size != 32 || properties.integrated || properties.maxThreadsPerBlock < 256 ||
        properties.sharedMemPerBlock < 14372U) {
        fail("qualification requires the discrete R9700 gfx1201 at HIP ordinal 0");
    }
    HIP_CHECK(hipSetDevice(0));
}

void write_report(const std::string& path, const std::string& device_name,
                  const std::string& architecture, int wave_size,
                  int runtime_version, int driver_version,
                  const std::filesystem::path& executable, std::string_view executable_sha256,
                  const std::filesystem::path& source, std::string_view source_sha256,
                  const Resources& incumbent_resources, const Resources& grouped_resources,
                  const Resources& merge_resources, std::uint32_t repetitions,
                  const std::vector<double>& incumbent, const std::vector<double>& grouped,
                  const std::vector<double>& ratios, double upper_ratio, bool passed) {
    const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) fail("report path already exists or cannot be created: " + path);
    std::ostringstream out;
    out << std::setprecision(12)
        << "{\n  \"schema\": \"ninfer_r9700_split512_grouped_pv_qualification\",\n"
        << "  \"schema_version\": 1,\n"
        << "  \"disposition\": \"" << (passed ? "qualified-challenger" : "rejected-challenger")
        << "\",\n  \"device\": {\"name\": \"" << json_escape(device_name)
        << "\", \"architecture\": \"" << json_escape(architecture)
        << "\", \"wave_size\": " << wave_size << "},\n"
        << "  \"runtime\": {\"hip_runtime_version\": " << runtime_version
        << ", \"hip_driver_version\": " << driver_version << "},\n"
        << "  \"provenance\": {\"executable\": \"" << json_escape(executable.string())
        << "\", \"executable_sha256\": \"" << executable_sha256
        << "\", \"source\": \"" << json_escape(source.string())
        << "\", \"source_sha256\": \"" << source_sha256 << "\"},\n"
        << "  \"command\": \"" << json_escape(executable.string())
        << " --out-json " << json_escape(path) << "\",\n"
        << "  \"power_profile\": {\"path\": \"" << kPowerPath
        << "\", \"value\": \"auto\"},\n"
        << "  \"shape\": {\"query_rows\": 1, \"value_group\": 16, \"context\": 8192, "
           "\"head_dim\": 256, \"query_heads\": 24, \"kv_heads\": 4, "
           "\"queries_per_kv\": 6, \"chunk\": 512},\n"
        << "  \"correctness\": {\"oracle\": \"independent-complete-fp64-pv\", "
           "\"all_outputs\": true, \"exact_incumbent_parity\": true, "
           "\"poison_full_overwrite\": true, \"canaries\": true},\n";
    const auto emit_resources = [&](const char* label, const Resources& value, bool comma) {
        out << "    \"" << label << "\": {\"registers\": " << value.registers
            << ", \"shared_bytes\": " << value.shared_bytes << ", \"local_bytes\": "
            << value.local_bytes << ", \"maximum_threads\": " << value.maximum_threads
            << '}' << (comma ? ",\n" : "\n");
    };
    out << "  \"resources\": {\n";
    emit_resources("incumbent_producer", incumbent_resources, true);
    emit_resources("grouped_producer", grouped_resources, true);
    emit_resources("shared_merge", merge_resources, false);
    out << "  },\n  \"timing\": {\"samples\": " << kSamples
        << ", \"repetitions_per_interval\": " << repetitions
        << ", \"minimum_interval_ms\": " << kMinimumIntervalMs
        << ", \"upper_ratio_gate\": " << kMaximumUpperRatio << ",\n";
    const auto emit_samples = [&](const char* label, const std::vector<double>& values,
                                  bool comma) {
        out << "    \"" << label << "\": [";
        for (std::size_t index = 0U; index < values.size(); ++index) {
            if (index != 0U) out << ',';
            out << values[index];
        }
        out << ']' << (comma ? ",\n" : "\n");
    };
    emit_samples("incumbent_interval_ms", incumbent, true);
    emit_samples("grouped_interval_ms", grouped, true);
    emit_samples("paired_grouped_over_incumbent", ratios, true);
    out << "    \"incumbent_median_ms\": " << median(incumbent)
        << ",\n    \"grouped_median_ms\": " << median(grouped)
        << ",\n    \"robust_upper_ratio\": " << upper_ratio
        << "\n  },\n  \"passed\": " << (passed ? "true" : "false") << "\n}\n";
    const std::string bytes = out.str();
    const ssize_t written = write(descriptor, bytes.data(), bytes.size());
    const int close_status = close(descriptor);
    if (written != static_cast<ssize_t>(bytes.size()) || close_status != 0) {
        fail("failed to write complete qualification report");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string_view(argv[1]) != "--out-json" || argv[2][0] == '\0') {
            fail("usage: ninfer_r9700_split512_grouped_pv_qual --out-json PATH");
        }
        const std::string power = require_auto_power();
        (void)power;
        std::string device_name;
        std::string architecture;
        int wave_size = 0;
        select_device(device_name, architecture, wave_size);
        int runtime_version = 0;
        int driver_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK(hipDriverGetVersion(&driver_version));
        const std::filesystem::path executable = std::filesystem::weakly_canonical(argv[0]);
        const std::filesystem::path source =
            std::filesystem::path(NINFER_SOURCE_DIR) /
            "tools/r9700/split512_grouped_pv_qual.hip";
        if (!std::filesystem::is_regular_file(executable) ||
            !std::filesystem::is_regular_file(source)) {
            fail("qualification provenance path is not a regular file");
        }
        const std::string executable_sha256 = sha256_file(executable);
        const std::string source_sha256 = sha256_file(source);
        const HostFixture host = make_fixture();
        const std::vector<double> oracle = complete_oracle(host);
        DeviceFixture fixture(host);
        Workspace incumbent_workspace;
        Workspace grouped_workspace;
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

        incumbent_workspace.poison();
        grouped_workspace.poison();
        launch(Producer::Incumbent, fixture, incumbent_workspace, stream);
        launch(Producer::Grouped, fixture, grouped_workspace, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto incumbent_maxima = incumbent_workspace.maxima.checked("incumbent maxima");
        const auto incumbent_sums = incumbent_workspace.sums.checked("incumbent sums");
        const auto incumbent_numerators =
            incumbent_workspace.numerators.checked("incumbent numerators");
        const auto incumbent_output = incumbent_workspace.output.checked("incumbent output");
        const auto grouped_maxima = grouped_workspace.maxima.checked("grouped maxima");
        const auto grouped_sums = grouped_workspace.sums.checked("grouped sums");
        const auto grouped_numerators = grouped_workspace.numerators.checked("grouped numerators");
        const auto grouped_output = grouped_workspace.output.checked("grouped output");
        require_exact(incumbent_maxima, grouped_maxima);
        require_exact(incumbent_sums, grouped_sums);
        require_exact(incumbent_numerators, grouped_numerators);
        require_exact(incumbent_output, grouped_output);
        compare_oracle(incumbent_output, oracle, "incumbent");
        compare_oracle(grouped_output, oracle, "grouped");

        const Resources incumbent_resources = resources(
            reinterpret_cast<const void*>(split512_pv_incumbent_kernel), "incumbent producer",
            4132);
        const Resources grouped_resources = resources(
            reinterpret_cast<const void*>(split512_pv_grouped_kernel), "grouped producer", 14372);
        const Resources merge_resources = resources(
            reinterpret_cast<const void*>(split512_shared_merge_kernel), "shared merge", 104);

        for (int warmup = 0; warmup < 3; ++warmup) {
            launch(warmup & 1 ? Producer::Grouped : Producer::Incumbent,
                   fixture, warmup & 1 ? grouped_workspace : incumbent_workspace, stream);
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        hipEvent_t begin = nullptr;
        hipEvent_t end = nullptr;
        HIP_CHECK(hipEventCreate(&begin));
        HIP_CHECK(hipEventCreate(&end));
        std::uint32_t repetitions = 1U;
        while (true) {
            const float incumbent = timed_interval(Producer::Incumbent, fixture,
                incumbent_workspace, stream, repetitions, begin, end);
            const float grouped = timed_interval(Producer::Grouped, fixture,
                grouped_workspace, stream, repetitions, begin, end);
            if (incumbent >= kMinimumIntervalMs && grouped >= kMinimumIntervalMs) break;
            if (repetitions >= (1U << 20U)) fail("unable to calibrate ten-millisecond intervals");
            repetitions *= 2U;
        }
        std::vector<double> incumbent_samples;
        std::vector<double> grouped_samples;
        std::vector<double> ratios;
        incumbent_samples.reserve(kSamples);
        grouped_samples.reserve(kSamples);
        ratios.reserve(kSamples);
        for (std::uint32_t sample = 0U; sample < kSamples; ++sample) {
            float incumbent = 0.0F;
            float grouped = 0.0F;
            if ((sample & 1U) == 0U) {
                incumbent = timed_interval(Producer::Incumbent, fixture, incumbent_workspace,
                                           stream, repetitions, begin, end);
                grouped = timed_interval(Producer::Grouped, fixture, grouped_workspace,
                                         stream, repetitions, begin, end);
            } else {
                grouped = timed_interval(Producer::Grouped, fixture, grouped_workspace,
                                         stream, repetitions, begin, end);
                incumbent = timed_interval(Producer::Incumbent, fixture, incumbent_workspace,
                                           stream, repetitions, begin, end);
            }
            if (incumbent < kMinimumIntervalMs || grouped < kMinimumIntervalMs) {
                fail("a measured timing interval was shorter than ten milliseconds");
            }
            incumbent_samples.push_back(incumbent);
            grouped_samples.push_back(grouped);
            ratios.push_back(static_cast<double>(grouped) / incumbent);
        }
        HIP_CHECK(hipEventDestroy(end));
        HIP_CHECK(hipEventDestroy(begin));
        HIP_CHECK(hipStreamDestroy(stream));
        std::vector<double> ordered_ratios = ratios;
        std::sort(ordered_ratios.begin(), ordered_ratios.end());
        const std::size_t upper_index = (ordered_ratios.size() * 9U + 9U) / 10U - 1U;
        const double upper_ratio = ordered_ratios[upper_index];
        const bool passed = upper_ratio <= kMaximumUpperRatio;
        write_report(argv[2], device_name, architecture, wave_size, runtime_version,
                     driver_version, executable, executable_sha256, source, source_sha256,
                     incumbent_resources, grouped_resources, merge_resources, repetitions,
                     incumbent_samples, grouped_samples, ratios, upper_ratio, passed);
        std::cout << "grouped_pv upper_ratio=" << upper_ratio
                  << " gate=" << kMaximumUpperRatio
                  << " disposition=" << (passed ? "qualified" : "rejected") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "split512 grouped-PV qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
