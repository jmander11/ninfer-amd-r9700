#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

constexpr const char* kRequiredArch = "gfx1201";
constexpr const char* kAmdVendor = "0x1002";
constexpr const char* kR9700Device = "0x7551";
constexpr std::size_t kInt4Group = 32;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) {
        return;
    }
    std::ostringstream os;
    os << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(os.str());
}

#define HIP_CHECK(expr) hip_check((expr), #expr, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)hipFree(data_);
        }
    }

    T* get() const { return data_; }
    std::size_t count() const { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() { (void)hipEventDestroy(event_); }
    hipEvent_t get() const { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

std::string trim(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    const auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    return std::string(begin, text.end());
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return {};
    }
    std::string value;
    std::getline(input, value);
    return trim(value);
}

std::string normalize_pci_bus_id(std::string id) {
    std::transform(id.begin(), id.end(), id.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (id.size() == 7 && id[2] == ':' && id[5] == '.') {
        id = "0000:" + id;
    }
    return id;
}

std::string uuid_string(const hipUUID& uuid) {
    const bool printable = std::all_of(std::begin(uuid.bytes), std::end(uuid.bytes), [](char ch) {
        return std::isxdigit(static_cast<unsigned char>(ch)) != 0;
    });
    if (printable) {
        return std::string(std::begin(uuid.bytes), std::end(uuid.bytes));
    }
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < sizeof(uuid.bytes); ++i) {
        os << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return os.str();
}

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    const char* requested_env = std::getenv("NINFER_R9700_PCI_BUS_ID");
    const std::string requested = requested_env == nullptr ? "" : normalize_pci_bus_id(requested_env);

    struct Candidate {
        int ordinal;
        hipDeviceProp_t props;
        std::string pci;
    };
    std::vector<Candidate> candidates;

    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, ordinal));
        char pci_buffer[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_buffer, sizeof(pci_buffer), ordinal));
        const std::string pci = normalize_pci_bus_id(pci_buffer);
        const std::string sysfs = "/sys/bus/pci/devices/" + pci;
        const bool exact_pci_identity = read_text_file(sysfs + "/vendor") == kAmdVendor &&
                                        read_text_file(sysfs + "/device") == kR9700Device;
        const bool discrete = props.integrated == 0;
        const bool exact_arch = std::string_view(props.gcnArchName) == kRequiredArch;
        if (exact_pci_identity && discrete && exact_arch && (requested.empty() || requested == pci)) {
            candidates.push_back(Candidate{ordinal, props, pci});
        }
    }

    if (candidates.size() != 1) {
        std::ostringstream os;
        os << "expected exactly one discrete 1002:7551 gfx1201 device";
        if (!requested.empty()) {
            os << " at " << requested;
        }
        os << ", found " << candidates.size();
        if (candidates.size() > 1 && requested.empty()) {
            os << "; set NINFER_R9700_PCI_BUS_ID";
        }
        fail(os.str());
    }

    const auto& selected = candidates.front();
    HIP_CHECK(hipSetDevice(selected.ordinal));
    std::cout << "device: pci=" << selected.pci << " uuid=" << uuid_string(selected.props.uuid)
              << " arch=" << selected.props.gcnArchName << " name=\"" << selected.props.name
              << "\" hip_multiprocessors=" << selected.props.multiProcessorCount
              << " wave=" << selected.props.warpSize
              << " vram_mib=" << selected.props.totalGlobalMem / (1024 * 1024) << '\n';
    if (selected.props.warpSize != 32) {
        fail("gfx1201 qualification requires wave32");
    }
    return selected.ordinal;
}

std::uint8_t host_encode_e4m3(float value) {
    return __hip_cvt_float_to_fp8(value, __HIP_SATFINITE, __HIP_E4M3);
}

__global__ void fp8_encode_kernel(const float* input, std::uint8_t* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __hip_cvt_float_to_fp8(input[index], __HIP_SATFINITE, __HIP_E4M3);
    }
}

__global__ void fp8_decode_kernel(const std::uint8_t* input, float* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const __half_raw raw = __hip_cvt_fp8_to_halfraw(input[index], __HIP_E4M3);
        output[index] = __half2float(__half(raw));
    }
}

__global__ void int4_g32_encode_kernel(const float* input,
                                       std::uint8_t* packed,
                                       __half* scales,
                                       std::size_t groups) {
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t global_thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t group = global_thread >> 5U;
    if (group >= groups) {
        return;
    }

    const float value = input[group * kInt4Group + lane];
    float maximum = fabsf(value);
    for (unsigned delta = 16; delta != 0; delta >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down(maximum, delta, 32));
    }
    maximum = __shfl(maximum, 0, 32);

    const __half stored_scale = __float2half(maximum / 7.0F);
    const float scale = __half2float(stored_scale);
    int code = scale == 0.0F ? 0 : __float2int_rn(value / scale);
    code = max(-7, min(7, code));
    if (lane == 0) {
        scales[group] = stored_scale;
    }
    // RDNA DS permutes need the source lanes active at the instruction. Exchange before branching.
    const int upper = __shfl(code, lane + 1, 32);
    if ((lane & 1U) == 0) {
        packed[group * 16 + lane / 2] =
            static_cast<std::uint8_t>((code & 0x0f) | ((upper & 0x0f) << 4));
    }
}

__global__ void int4_g32_decode_kernel(const std::uint8_t* packed,
                                       const __half* scales,
                                       float* output,
                                       std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::size_t group = index / kInt4Group;
    const unsigned lane = static_cast<unsigned>(index % kInt4Group);
    const std::uint8_t byte = packed[group * 16 + lane / 2];
    const unsigned shift = (lane & 1U) * 4U;
    int code = static_cast<int>((byte >> shift) & 0x0fU);
    code = code >= 8 ? code - 16 : code;
    output[index] = static_cast<float>(code) * __half2float(scales[group]);
}

using I32x2 = int __attribute__((ext_vector_type(2)));
using F32x8 = float __attribute__((ext_vector_type(8)));

__global__ void fp8_wmma_ones_kernel(float* output) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned lane = threadIdx.x & 31U;
    const I32x2 ones = {static_cast<int>(0x38383838U), static_cast<int>(0x38383838U)};
    const F32x8 zero = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    const F32x8 result =
        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(ones, ones, zero);
    for (unsigned i = 0; i < 8; ++i) {
        output[lane * 8U + i] = result[i];
    }
#else
    (void)output;
#endif
}

__device__ __attribute__((unused)) int pack_fp8x4(const std::uint8_t* bytes) {
    return static_cast<int>(static_cast<std::uint32_t>(bytes[0]) |
                            (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                            (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                            (static_cast<std::uint32_t>(bytes[3]) << 24U));
}

// Owned gfx12 wave32 fragment contract. The independent host oracle verifies this literal map
// through the raw register dump, so neither rocWMMA nor a device-side scatter map can mask a
// transpose, K-half, accumulator-seed, or output-lane error.
__global__ void fp8_wmma_raw_layout_kernel(const std::uint8_t* matrix_a,
                                           const std::uint8_t* matrix_b,
                                           const float* accumulator_seed,
                                           float* raw_output) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const unsigned lane = threadIdx.x & 31U;
    const unsigned k_half = lane >> 4U;
    const unsigned axis = lane & 15U;
    std::uint8_t a_bytes[8];
    std::uint8_t b_bytes[8];
    F32x8 c;
    for (unsigned j = 0; j < 8; ++j) {
        a_bytes[j] = matrix_a[axis * 16U + 8U * k_half + j];
        b_bytes[j] = matrix_b[(8U * k_half + j) * 16U + axis];
        c[j] = accumulator_seed[((lane >> 4U) * 8U + j) * 16U + axis];
    }
    const I32x2 a = {pack_fp8x4(a_bytes), pack_fp8x4(a_bytes + 4)};
    const I32x2 b = {pack_fp8x4(b_bytes), pack_fp8x4(b_bytes + 4)};
    const F32x8 result =
        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    for (unsigned j = 0; j < 8; ++j) { raw_output[lane * 8U + j] = result[j]; }
#else
    (void)matrix_a;
    (void)matrix_b;
    (void)accumulator_seed;
    (void)raw_output;
#endif
}

void test_fp8_conversion() {
    namespace oracle = ninfer::test::fp8_int4_kv_oracle;
    const std::array<float, 15> values = {
        0.0F,
        -0.0F,
        0.5F,
        -0.5F,
        1.0F,
        -1.0F,
        2.0F,
        448.0F,
        1000.0F,
        -1000.0F,
        1.0625F,
        1.1875F,
        std::ldexp(1.0F, -10),
        std::nextafter(std::ldexp(1.0F, -10), 1.0F),
        std::nextafter(448.0F, 0.0F)};

    DeviceBuffer<float> d_values(values.size());
    DeviceBuffer<std::uint8_t> d_bits(values.size());
    DeviceBuffer<float> d_decoded(values.size());
    HIP_CHECK(hipMemcpy(d_values.get(), values.data(), sizeof(values), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(fp8_encode_kernel, dim3(1), dim3(32), 0, 0, d_values.get(), d_bits.get(), values.size());
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(fp8_decode_kernel, dim3(1), dim3(32), 0, 0, d_bits.get(), d_decoded.get(), values.size());
    HIP_CHECK(hipGetLastError());

    std::array<std::uint8_t, values.size()> bits{};
    std::array<float, values.size()> decoded{};
    HIP_CHECK(hipMemcpy(bits.data(), d_bits.get(), sizeof(bits), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(decoded.data(), d_decoded.get(), sizeof(decoded), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint8_t expected = oracle::encode_e4m3fn(values[i]);
        if (bits[i] != expected || bits[i] != host_encode_e4m3(values[i])) {
            std::ostringstream os;
            os << "E4M3 encode mismatch at " << i << ": got 0x" << std::hex
               << static_cast<unsigned>(bits[i]) << " expected 0x" << static_cast<unsigned>(expected);
            fail(os.str());
        }
        const float expected_decoded = static_cast<float>(oracle::decode_e4m3fn(bits[i]));
        if (decoded[i] != expected_decoded ||
            (decoded[i] == 0.0F && std::signbit(decoded[i]) != std::signbit(expected_decoded))) {
            std::ostringstream os;
            os << "E4M3 decode mismatch at " << i << ": got " << decoded[i]
               << " expected " << expected_decoded;
            fail(os.str());
        }
    }

    std::vector<std::uint8_t> finite_words;
    for (unsigned word = 0; word < 256; ++word) {
        if ((word & 0x7fU) != 0x7fU) {
            finite_words.push_back(static_cast<std::uint8_t>(word));
        }
    }
    DeviceBuffer<std::uint8_t> d_all_words(finite_words.size());
    DeviceBuffer<float> d_all_decoded(finite_words.size());
    HIP_CHECK(hipMemcpy(d_all_words.get(),
                        finite_words.data(),
                        finite_words.size(),
                        hipMemcpyHostToDevice));
    hipLaunchKernelGGL(fp8_decode_kernel,
                       dim3((finite_words.size() + 255) / 256),
                       dim3(256),
                       0,
                       0,
                       d_all_words.get(),
                       d_all_decoded.get(),
                       finite_words.size());
    HIP_CHECK(hipGetLastError());
    std::vector<float> all_decoded(finite_words.size());
    HIP_CHECK(hipMemcpy(all_decoded.data(),
                        d_all_decoded.get(),
                        all_decoded.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < finite_words.size(); ++i) {
        const float expected = static_cast<float>(oracle::decode_e4m3fn(finite_words[i]));
        if (all_decoded[i] != expected ||
            (expected == 0.0F && std::signbit(all_decoded[i]) != std::signbit(expected))) {
            fail("GPU E4M3 decode differs from independent oracle over finite word space");
        }
    }
    bool infinity_rejected = false;
    bool nan_rejected = false;
    try {
        (void)oracle::encode_e4m3fn(std::numeric_limits<float>::infinity());
    } catch (const std::invalid_argument&) {
        infinity_rejected = true;
    }
    try {
        (void)oracle::encode_e4m3fn(std::numeric_limits<float>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        nan_rejected = true;
    }
    if (!infinity_rejected || !nan_rejected) {
        fail("canonical E4M3 storage boundary must reject non-finite sources");
    }
    std::cout << "check fp8_e4m3_conversion: PASS (254 finite words, RNE edges, nonfinite rejection)\n";
}

void test_fp8_wmma() {
    DeviceBuffer<float> d_output(32 * 8);
    hipLaunchKernelGGL(fp8_wmma_ones_kernel, dim3(1), dim3(32), 0, 0, d_output.get());
    HIP_CHECK(hipGetLastError());
    std::array<float, 32 * 8> output{};
    HIP_CHECK(hipMemcpy(output.data(), d_output.get(), sizeof(output), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (output[i] != 16.0F) {
            std::ostringstream os;
            os << "FP8 WMMA mismatch at accumulator element " << i << ": got " << output[i]
               << " expected 16";
            fail(os.str());
        }
    }
    namespace oracle = ninfer::test::fp8_int4_kv_oracle;
    std::array<std::uint8_t, 16 * 16> matrix_a{};
    std::array<std::uint8_t, 16 * 16> matrix_b{};
    std::array<float, 16 * 16> seed{};
    std::array<float, 16 * 16> expected{};
    for (unsigned row = 0; row < 16; ++row) {
        for (unsigned column = 0; column < 16; ++column) {
            seed[row * 16 + column] = static_cast<float>(static_cast<int>((9U * row + 5U * column) % 17U) - 8);
        }
        for (unsigned k = 0; k < 16; ++k) {
            const float a = static_cast<float>(static_cast<int>((5U * row + 3U * k + 1U) % 15U) - 7);
            matrix_a[row * 16 + k] = oracle::encode_e4m3fn(a);
        }
    }
    for (unsigned k = 0; k < 16; ++k) {
        for (unsigned column = 0; column < 16; ++column) {
            const float b = static_cast<float>(static_cast<int>((7U * k + 11U * column + 2U) % 13U) - 6);
            matrix_b[k * 16 + column] = oracle::encode_e4m3fn(b);
        }
    }
    for (unsigned row = 0; row < 16; ++row) {
        for (unsigned column = 0; column < 16; ++column) {
            float sum = seed[row * 16 + column];
            for (unsigned k = 0; k < 16; ++k) {
                sum += static_cast<float>(oracle::decode_e4m3fn(matrix_a[row * 16 + k])) *
                       static_cast<float>(oracle::decode_e4m3fn(matrix_b[k * 16 + column]));
            }
            expected[row * 16 + column] = sum;
        }
    }
    if (expected[1U * 16U + 4U] == expected[4U * 16U + 1U]) {
        fail("asymmetric WMMA fixture unexpectedly aliases transposed outputs");
    }
    DeviceBuffer<std::uint8_t> d_matrix_a(matrix_a.size());
    DeviceBuffer<std::uint8_t> d_matrix_b(matrix_b.size());
    DeviceBuffer<float> d_seed(seed.size());
    DeviceBuffer<float> d_raw_output(expected.size());
    HIP_CHECK(hipMemcpy(d_matrix_a.get(), matrix_a.data(), matrix_a.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_matrix_b.get(), matrix_b.data(), matrix_b.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seed.get(), seed.data(), sizeof(seed), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(fp8_wmma_raw_layout_kernel,
                       dim3(1),
                       dim3(32),
                       0,
                       0,
                       d_matrix_a.get(),
                       d_matrix_b.get(),
                       d_seed.get(),
                       d_raw_output.get());
    HIP_CHECK(hipGetLastError());
    std::array<float, 16 * 16> raw_output{};
    HIP_CHECK(hipMemcpy(raw_output.data(),
                        d_raw_output.get(),
                        sizeof(raw_output),
                        hipMemcpyDeviceToHost));
    for (unsigned lane = 0; lane < 32; ++lane) {
        for (unsigned j = 0; j < 8; ++j) {
            const unsigned row = (lane >> 4U) * 8U + j;
            const unsigned column = lane & 15U;
            const float actual = raw_output[lane * 8U + j];
            const float expected_value = expected[row * 16U + column];
            if (actual != expected_value) {
                std::ostringstream os;
                os << "raw FP8 WMMA fragment mismatch at C[" << row << ',' << column
                   << "]: got " << actual << " expected " << expected_value;
                fail(os.str());
            }
        }
    }
    std::cout << "check wave32_fp8_wmma: PASS (direct opcode smoke plus owned asymmetric raw map)\n";
}

std::uint16_t half_bits(float value) {
    const _Float16 half_value = static_cast<_Float16>(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &half_value, sizeof(bits));
    return bits;
}

float half_from_bits(std::uint16_t bits) {
    _Float16 value{};
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
}

void test_int4_codec() {
    constexpr std::size_t groups = 17;
    constexpr std::size_t count = groups * kInt4Group;
    std::array<float, count> input{};
    for (std::size_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>((i * 37U + i / 11U) % 61U) - 30;
        input[i] = static_cast<float>(centered) * 0.1171875F;
    }
    std::fill_n(input.begin() + 5 * kInt4Group, kInt4Group, 0.0F);
    std::fill_n(input.begin() + 6 * kInt4Group, kInt4Group, 1.0e-8F);

    std::array<std::uint8_t, count / 2> expected_packed{};
    std::array<std::uint16_t, groups> expected_scales{};
    std::array<float, count> expected_decoded{};
    for (std::size_t group = 0; group < groups; ++group) {
        float maximum = 0.0F;
        for (std::size_t lane = 0; lane < kInt4Group; ++lane) {
            maximum = std::max(maximum, std::abs(input[group * kInt4Group + lane]));
        }
        const float raw_scale = maximum / 7.0F;
        expected_scales[group] = half_bits(raw_scale);
        const float scale = half_from_bits(expected_scales[group]);
        std::array<int, kInt4Group> codes{};
        for (std::size_t lane = 0; lane < kInt4Group; ++lane) {
            codes[lane] = scale == 0.0F
                              ? 0
                              : std::clamp(static_cast<int>(std::nearbyint(
                                               input[group * kInt4Group + lane] / scale)),
                                           -7,
                                           7);
            expected_decoded[group * kInt4Group + lane] = static_cast<float>(codes[lane]) * scale;
        }
        for (std::size_t byte = 0; byte < 16; ++byte) {
            expected_packed[group * 16 + byte] = static_cast<std::uint8_t>(
                (codes[2 * byte] & 0x0f) | ((codes[2 * byte + 1] & 0x0f) << 4));
        }
    }

    DeviceBuffer<float> d_input(count);
    DeviceBuffer<std::uint8_t> d_packed(count / 2);
    DeviceBuffer<__half> d_scales(groups);
    DeviceBuffer<float> d_decoded(count);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), sizeof(input), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(int4_g32_encode_kernel,
                       dim3(3),
                       dim3(256),
                       0,
                       0,
                       d_input.get(),
                       d_packed.get(),
                       d_scales.get(),
                       groups);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(int4_g32_decode_kernel,
                       dim3((count + 255) / 256),
                       dim3(256),
                       0,
                       0,
                       d_packed.get(),
                       d_scales.get(),
                       d_decoded.get(),
                       count);
    HIP_CHECK(hipGetLastError());

    std::array<std::uint8_t, count / 2> packed{};
    std::array<std::uint16_t, groups> scales{};
    std::array<float, count> decoded{};
    HIP_CHECK(hipMemcpy(packed.data(), d_packed.get(), sizeof(packed), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(scales.data(), d_scales.get(), sizeof(scales), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(decoded.data(), d_decoded.get(), sizeof(decoded), hipMemcpyDeviceToHost));

    if (packed != expected_packed) {
        const auto mismatch = std::mismatch(packed.begin(), packed.end(), expected_packed.begin());
        std::ostringstream os;
        os << "INT4-G32 packed codes differ from CPU oracle at byte "
           << std::distance(packed.begin(), mismatch.first) << ": got 0x" << std::hex
           << static_cast<unsigned>(*mismatch.first) << " expected 0x"
           << static_cast<unsigned>(*mismatch.second);
        fail(os.str());
    }
    if (scales != expected_scales) {
        fail("INT4-G32 FP16 scale bits differ from CPU oracle");
    }
    if (decoded != expected_decoded) {
        fail("INT4-G32 decoded values differ from CPU oracle");
    }
    std::cout << "check int4_g32_codec: PASS (adjacent signed nibbles, FP16 scale boundary, exact decode)\n";
}

template <typename Launch>
double benchmark_ms(int iterations, Launch&& launch) {
    for (int i = 0; i < 10; ++i) {
        launch();
    }
    HIP_CHECK(hipDeviceSynchronize());
    Event start;
    Event stop;
    HIP_CHECK(hipEventRecord(start.get()));
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    HIP_CHECK(hipEventRecord(stop.get()));
    HIP_CHECK(hipEventSynchronize(stop.get()));
    float elapsed_ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
    return static_cast<double>(elapsed_ms) / iterations;
}

void print_benchmark(std::string_view name, double ms, double bytes) {
    const double gib_per_s = bytes / (1024.0 * 1024.0 * 1024.0) / (ms * 1.0e-3);
    std::cout << "bench " << name << ": " << std::fixed << std::setprecision(4) << ms
              << " ms, " << std::setprecision(2) << gib_per_s << " GiB/s effective_payload\n";
}

void run_benchmarks(std::size_t count, int iterations) {
    const std::size_t groups = count / kInt4Group;
    std::vector<float> input(count);
    for (std::size_t i = 0; i < count; ++i) {
        input[i] = std::sin(static_cast<float>(i % 8192U) * 0.001953125F) * 3.0F;
    }

    DeviceBuffer<float> d_input(count);
    DeviceBuffer<float> d_output(count);
    DeviceBuffer<std::uint8_t> d_fp8(count);
    DeviceBuffer<std::uint8_t> d_int4(count / 2);
    DeviceBuffer<__half> d_scales(groups);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), count * sizeof(float), hipMemcpyHostToDevice));

    constexpr unsigned threads = 256;
    const dim3 element_grid(static_cast<unsigned>((count + threads - 1) / threads));
    const dim3 group_grid(static_cast<unsigned>((groups * 32 + threads - 1) / threads));

    const double fp8_encode_ms = benchmark_ms(iterations, [&] {
        hipLaunchKernelGGL(fp8_encode_kernel,
                           element_grid,
                           dim3(threads),
                           0,
                           0,
                           d_input.get(),
                           d_fp8.get(),
                           count);
        HIP_CHECK(hipGetLastError());
    });
    print_benchmark("fp8_e4m3_encode", fp8_encode_ms, static_cast<double>(count) * 5.0);

    const double fp8_decode_ms = benchmark_ms(iterations, [&] {
        hipLaunchKernelGGL(fp8_decode_kernel,
                           element_grid,
                           dim3(threads),
                           0,
                           0,
                           d_fp8.get(),
                           d_output.get(),
                           count);
        HIP_CHECK(hipGetLastError());
    });
    print_benchmark("fp8_e4m3_decode", fp8_decode_ms, static_cast<double>(count) * 5.0);

    const double int4_encode_ms = benchmark_ms(iterations, [&] {
        hipLaunchKernelGGL(int4_g32_encode_kernel,
                           group_grid,
                           dim3(threads),
                           0,
                           0,
                           d_input.get(),
                           d_int4.get(),
                           d_scales.get(),
                           groups);
        HIP_CHECK(hipGetLastError());
    });
    print_benchmark("int4_g32_f16scale_encode",
                    int4_encode_ms,
                    static_cast<double>(count) * (4.0 + 0.5 + 2.0 / 32.0));

    const double int4_decode_ms = benchmark_ms(iterations, [&] {
        hipLaunchKernelGGL(int4_g32_decode_kernel,
                           element_grid,
                           dim3(threads),
                           0,
                           0,
                           d_int4.get(),
                           d_scales.get(),
                           d_output.get(),
                           count);
        HIP_CHECK(hipGetLastError());
    });
    print_benchmark("int4_g32_f16scale_decode",
                    int4_decode_ms,
                    static_cast<double>(count) * (0.5 + 2.0 / 32.0 + 4.0));
}

std::size_t parse_size(const char* value, const char* option) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        fail(std::string("invalid value for ") + option + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

struct Options {
    std::size_t elements = 1U << 26U;
    int iterations = 100;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--elements" && i + 1 < argc) {
            options.elements = parse_size(argv[++i], "--elements");
        } else if (argument == "--iterations" && i + 1 < argc) {
            const std::size_t parsed = parse_size(argv[++i], "--iterations");
            if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                fail("--iterations is too large");
            }
            options.iterations = static_cast<int>(parsed);
        } else if (argument == "--help") {
            std::cout << "usage: r9700_qual [--elements N] [--iterations N]\n";
            std::exit(0);
        } else {
            fail(std::string("unknown or incomplete option: ") + argv[i]);
        }
    }
    if (options.elements % kInt4Group != 0) {
        fail("--elements must be a multiple of 32");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        select_r9700();
        std::cout << "toolchain: HIP " << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR << '.'
                  << HIP_VERSION_PATCH << ", compile_target=" << kRequiredArch << '\n';
        test_fp8_conversion();
        test_fp8_wmma();
        test_int4_codec();
        std::cout << "benchmark: elements=" << options.elements << " iterations=" << options.iterations
                  << " (ROCm 10 baseline)\n";
        run_benchmarks(options.elements, options.iterations);
        std::cout << "qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
