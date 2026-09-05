#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include "ops/r9700/kv/fp8_int4_kv_append.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/fp8_int4_kv_compact.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <algorithm>
#include <array>
#include <bit>
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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr unsigned kDimension = 256;
constexpr unsigned kQueryHeads = 24;
constexpr unsigned kKvHeads = 4;
constexpr unsigned kQueriesPerKv = kQueryHeads / kKvHeads;
constexpr unsigned kPageTokens = 64;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr const char* kRequiredArch = "gfx1201";

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
    explicit DeviceBuffer(std::size_t count) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { (void)hipFree(data_); }
    T* get() const { return data_; }

private:
    T* data_ = nullptr;
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

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.pop_back();
    }
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    return std::string(first, value.end());
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    std::string value;
    if (input) {
        std::getline(input, value);
    }
    return trim(value);
}

std::string normalize_pci(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value.size() == 7 && value[2] == ':' && value[5] == '.') {
        value = "0000:" + value;
    }
    return value;
}

void select_r9700() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    const char* requested_env = std::getenv("NINFER_R9700_PCI_BUS_ID");
    const std::string requested = requested_env == nullptr ? "" : normalize_pci(requested_env);
    struct Candidate {
        int ordinal;
        hipDeviceProp_t properties;
        std::string pci;
    };
    std::vector<Candidate> candidates;
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        char pci_buffer[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_buffer, sizeof(pci_buffer), ordinal));
        const std::string pci = normalize_pci(pci_buffer);
        const std::string sysfs = "/sys/bus/pci/devices/" + pci;
        if (properties.integrated == 0 && std::string_view(properties.gcnArchName) == kRequiredArch &&
            read_text_file(sysfs + "/vendor") == "0x1002" &&
            read_text_file(sysfs + "/device") == "0x7551" &&
            (requested.empty() || requested == pci)) {
            candidates.push_back(Candidate{ordinal, properties, pci});
        }
    }
    if (candidates.size() != 1) {
        std::ostringstream os;
        os << "expected exactly one discrete 1002:7551 gfx1201 R9700";
        if (!requested.empty()) {
            os << " at " << requested;
        }
        os << ", found " << candidates.size();
        fail(os.str());
    }
    const auto& selected = candidates.front();
    HIP_CHECK(hipSetDevice(selected.ordinal));
    if (selected.properties.warpSize != 32) {
        fail("gfx1201 KV qualification requires wave32");
    }
    std::cout << "device: pci=" << selected.pci << " arch=" << selected.properties.gcnArchName
              << " wave=" << selected.properties.warpSize
              << " vram_mib=" << selected.properties.totalGlobalMem / (1024 * 1024) << '\n';
}

template <unsigned Group, bool TokenFastest>
struct Layout {
    static_assert(Group == 16 || Group == 32);
    static constexpr unsigned kGroups = kDimension / Group;
    static constexpr unsigned kBytesPerGroup = Group / 2;

    __host__ __device__ static std::size_t key(std::size_t token,
                                               unsigned head,
                                               unsigned feature,
                                               std::size_t physical_tokens) {
        const std::size_t page = token / kPageTokens;
        const std::size_t in_page = token % kPageTokens;
        const std::size_t pages = physical_tokens / kPageTokens;
        if constexpr (TokenFastest) {
            return in_page + kPageTokens *
                                 (static_cast<std::size_t>(feature) + kDimension *
                                                                      (page + pages * head));
        }
        return feature + kDimension *
                             (in_page + kPageTokens * (static_cast<std::size_t>(head) + kKvHeads * page));
    }

    __host__ __device__ static std::size_t value_byte(std::size_t token,
                                                      unsigned head,
                                                      unsigned group,
                                                      unsigned byte,
                                                      std::size_t physical_tokens) {
        const std::size_t page = token / kPageTokens;
        const std::size_t in_page = token % kPageTokens;
        const std::size_t pages = physical_tokens / kPageTokens;
        const std::size_t feature = static_cast<std::size_t>(group) * kBytesPerGroup + byte;
        if constexpr (TokenFastest) {
            return in_page + kPageTokens *
                                 (feature + (kGroups * kBytesPerGroup) * (page + pages * head));
        }
        return feature + (kGroups * kBytesPerGroup) *
                             (in_page + kPageTokens * (static_cast<std::size_t>(head) + kKvHeads * page));
    }

    __host__ __device__ static std::size_t scale(std::size_t token,
                                                 unsigned head,
                                                 unsigned group,
                                                 std::size_t physical_tokens) {
        const std::size_t page = token / kPageTokens;
        const std::size_t in_page = token % kPageTokens;
        const std::size_t pages = physical_tokens / kPageTokens;
        if constexpr (TokenFastest) {
            return in_page + kPageTokens *
                                 (static_cast<std::size_t>(group) + kGroups * (page + pages * head));
        }
        return group + kGroups *
                           (in_page + kPageTokens * (static_cast<std::size_t>(head) + kKvHeads * page));
    }
};

std::size_t host_physical_token(std::size_t logical,
                                const std::vector<std::uint32_t>& pages,
                                std::size_t physical_tokens) {
    const std::size_t logical_page = logical / kPageTokens;
    if (logical_page >= pages.size()) {
        fail("host page oracle logical index is out of range");
    }
    const std::size_t token = static_cast<std::size_t>(pages[logical_page]) * kPageTokens +
                              logical % kPageTokens;
    if (token >= physical_tokens) {
        fail("host page oracle physical index is out of range");
    }
    return token;
}

template <unsigned Group, bool TokenFastest>
std::size_t host_key_offset(std::size_t token,
                            unsigned head,
                            unsigned feature,
                            std::size_t physical_tokens) {
    if (token >= physical_tokens || head >= kKvHeads || feature >= kDimension) {
        fail("host key layout oracle index is out of range");
    }
    return Layout<Group, TokenFastest>::key(token, head, feature, physical_tokens);
}

template <unsigned Group, bool TokenFastest>
std::size_t host_value_offset(std::size_t token,
                              unsigned head,
                              unsigned group,
                              unsigned byte,
                              std::size_t physical_tokens) {
    constexpr unsigned groups = kDimension / Group;
    constexpr unsigned bytes_per_group = Group / 2;
    if (token >= physical_tokens || head >= kKvHeads || group >= groups ||
        byte >= bytes_per_group) {
        fail("host value layout oracle index is out of range");
    }
    return Layout<Group, TokenFastest>::value_byte(token, head, group, byte, physical_tokens);
}

template <unsigned Group, bool TokenFastest>
std::size_t host_scale_offset(std::size_t token,
                              unsigned head,
                              unsigned group,
                              std::size_t physical_tokens) {
    constexpr unsigned groups = kDimension / Group;
    if (token >= physical_tokens || head >= kKvHeads || group >= groups) {
        fail("host scale layout oracle index is out of range");
    }
    return Layout<Group, TokenFastest>::scale(token, head, group, physical_tokens);
}

std::vector<std::uint32_t> make_fragmented_pages(std::size_t context) {
    const std::size_t logical_pages = (context + kPageTokens - 1) / kPageTokens;
    std::vector<std::uint32_t> pages(logical_pages);
    for (std::size_t page = 0; page < logical_pages; ++page) {
        pages[page] = static_cast<std::uint32_t>(logical_pages - page);
    }
    return pages;
}

float represented_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

std::vector<float> make_keys(std::size_t tokens) {
    std::vector<float> result(tokens * kKvHeads * kDimension);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (unsigned head = 0; head < kKvHeads; ++head) {
            for (unsigned feature = 0; feature < kDimension; ++feature) {
                const int wave = static_cast<int>((token * 13 + head * 29 + feature * 7) % 97) - 48;
                result[(token * kKvHeads + head) * kDimension + feature] =
                    represented_bf16(static_cast<float>(wave) / 32.0F);
            }
        }
    }
    return result;
}

std::vector<float> make_values(std::size_t tokens) {
    std::vector<float> result(tokens * kKvHeads * kDimension);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (unsigned head = 0; head < kKvHeads; ++head) {
            for (unsigned feature = 0; feature < kDimension; ++feature) {
                const int wave = static_cast<int>((token * 17 + head * 23 + feature * 11) % 113) - 56;
                result[(token * kKvHeads + head) * kDimension + feature] =
                    represented_bf16(static_cast<float>(wave) / 40.0F);
            }
        }
    }
    return result;
}

std::vector<float> make_query() {
    std::vector<float> result(kQueryHeads * kDimension);
    for (unsigned head = 0; head < kQueryHeads; ++head) {
        for (unsigned feature = 0; feature < kDimension; ++feature) {
            const int wave = static_cast<int>((head * 31 + feature * 19) % 89) - 44;
            result[head * kDimension + feature] =
                represented_bf16(static_cast<float>(wave) / 64.0F);
        }
    }
    return result;
}

template <unsigned Group>
void inject_codec_edge_corpus(std::vector<float>& keys, std::vector<float>& values) {
    // Keep these device cases in the full physical-plane comparison instead of testing an
    // intrinsic in isolation.  The independent host codec supplies the expected packed bytes.
    const std::size_t h0 = 0;
    const std::size_t h1 = kDimension;
    keys[h0 + 0] = -0.0F;
    keys[h0 + 1] = 0.0F;
    keys[h0 + 2] = 1000.0F;   // finite E4M3FN saturation to +448
    keys[h0 + 3] = -1000.0F;  // finite E4M3FN saturation to -448

    // Scale is exactly one because lane zero is seven.  The remaining lanes exercise the
    // RNE ties-to-even quotient boundary and canonical signed-zero value encoding.
    values[h0 + 0] = 7.0F;
    values[h0 + 1] = 0.5F;
    values[h0 + 2] = 1.5F;
    values[h0 + 3] = -0.5F;
    values[h0 + 4] = -1.5F;
    values[h0 + 5] = -0.0F;

    // A finite represented BF16 group whose FP16 scale underflows has one canonical result:
    // positive-zero scale and all-zero packed codes.  Clear the entire group first so no
    // ordinary corpus value masks this condition.
    for (unsigned lane = 0; lane < Group; ++lane) {
        values[h1 + lane] = lane == 1U ? -std::ldexp(1.0F, -126)
                                       : std::ldexp(1.0F, -126);
    }
}

// Exported PV is qualified with the represented FP32 probabilities that a stable softmax can
// produce, not a uniform-only surrogate: each head includes ordinary variation, a near-one peak,
// and finite tails near the FP32 underflow boundary.
std::vector<float> make_probability_corpus(std::size_t context) {
    std::vector<float> result(static_cast<std::size_t>(kQueryHeads) * context);
    for (unsigned head = 0; head < kQueryHeads; ++head) {
        std::vector<double> weights(context);
        double sum = 0.0;
        const std::size_t peak = (head * 37U + 11U) % context;
        for (std::size_t logical = 0; logical < context; ++logical) {
            double logit = 0.0;
            switch (head % 3U) {
                case 0: logit = logical == peak ? 0.0 : -18.0; break;
                case 1: logit = -static_cast<double>((logical * 17U + head * 13U) % 23U); break;
                default: logit = logical == peak ? 0.0 : -80.0 - static_cast<double>(logical % 5U); break;
            }
            weights[logical] = std::exp(logit);
            sum += weights[logical];
        }
        for (std::size_t logical = 0; logical < context; ++logical) {
            result[static_cast<std::size_t>(head) * context + logical] =
                static_cast<float>(weights[logical] / sum);
        }
    }
    return result;
}

template <unsigned Group, bool KTokenFastest, bool VTokenFastest, bool ScaleTokenFastest>
void append_oracle(const std::vector<float>& source_keys,
                   const std::vector<float>& source_values,
                   const std::vector<std::uint32_t>& logical_positions,
                   const std::vector<std::uint32_t>& pages,
                   std::size_t physical_tokens,
                   std::vector<std::uint8_t>& keys,
                   std::vector<std::uint8_t>& values,
                   std::vector<std::uint16_t>& scales) {
    namespace oracle = ninfer::test::fp8_int4_kv_oracle;
    const std::size_t represented_tokens = logical_positions.size() * kKvHeads;
    const oracle::Fp8Keys encoded_keys =
        oracle::encode_keys(source_keys, represented_tokens, kDimension);
    const oracle::Int4Values encoded_values =
        oracle::encode_values(source_values, represented_tokens, kDimension, Group);
    for (std::size_t input_token = 0; input_token < logical_positions.size(); ++input_token) {
        const std::size_t token =
            host_physical_token(logical_positions[input_token], pages, physical_tokens);
        for (unsigned head = 0; head < kKvHeads; ++head) {
            const std::size_t represented_token = input_token * kKvHeads + head;
            for (unsigned feature = 0; feature < kDimension; ++feature) {
                keys[host_key_offset<Group, KTokenFastest>(
                    token, head, feature, physical_tokens)] =
                    encoded_keys.codes[represented_token * kDimension + feature];
            }
            for (unsigned group = 0; group < Layout<Group, ScaleTokenFastest>::kGroups; ++group) {
                scales[host_scale_offset<Group, ScaleTokenFastest>(
                    token, head, group, physical_tokens)] =
                    encoded_values.fp16_scales[represented_token *
                                                    Layout<Group, ScaleTokenFastest>::kGroups +
                                                group];
                for (unsigned byte = 0; byte < Group / 2; ++byte) {
                    values[host_value_offset<Group, VTokenFastest>(
                        token, head, group, byte, physical_tokens)] =
                        encoded_values.packed_codes[represented_token * (kDimension / 2) +
                                                    group * (Group / 2) + byte];
                }
            }
        }
    }
}

template <unsigned Group, bool KTokenFastest, bool VTokenFastest, bool ScaleTokenFastest>
std::vector<double> attention_oracle(const std::vector<float>& query,
                                     const std::vector<std::uint8_t>& keys,
                                     const std::vector<std::uint8_t>& values,
                                     const std::vector<std::uint16_t>& scales,
                                     const std::vector<std::uint32_t>& pages,
                                     std::size_t context,
                                     std::size_t physical_tokens) {
    namespace oracle = ninfer::test::fp8_int4_kv_oracle;
    std::vector<double> output(kQueryHeads * kDimension);
    for (unsigned query_head = 0; query_head < kQueryHeads; ++query_head) {
        const unsigned kv_head = query_head / kQueriesPerKv;
        oracle::Fp8Keys logical_keys{context,
                                     kDimension,
                                     std::vector<std::uint8_t>(context * kDimension)};
        oracle::Int4Values logical_values{
            context,
            kDimension,
            Group,
            std::vector<std::uint8_t>(context * kDimension / 2),
            std::vector<std::uint16_t>(context * Layout<Group, ScaleTokenFastest>::kGroups)};
        for (std::size_t logical = 0; logical < context; ++logical) {
            const std::size_t token = host_physical_token(logical, pages, physical_tokens);
            for (unsigned feature = 0; feature < kDimension; ++feature) {
                logical_keys.codes[logical * kDimension + feature] =
                    keys[host_key_offset<Group, KTokenFastest>(
                        token, kv_head, feature, physical_tokens)];
            }
            for (unsigned group = 0; group < Layout<Group, ScaleTokenFastest>::kGroups; ++group) {
                logical_values.fp16_scales[logical * Layout<Group, ScaleTokenFastest>::kGroups + group] =
                    scales[host_scale_offset<Group, ScaleTokenFastest>(
                        token, kv_head, group, physical_tokens)];
                for (unsigned byte = 0; byte < Group / 2; ++byte) {
                    logical_values.packed_codes[logical * (kDimension / 2) + group * (Group / 2) +
                                                byte] =
                        values[host_value_offset<Group, VTokenFastest>(
                            token, kv_head, group, byte, physical_tokens)];
                }
            }
        }
        const std::span<const float> represented_query(
            query.data() + query_head * kDimension, kDimension);
        const std::vector<double> head_output = oracle::attention_fp64(
            represented_query, logical_keys, logical_values, static_cast<double>(kAttentionScale));
        std::copy(head_output.begin(),
                  head_output.end(),
                  output.begin() + query_head * kDimension);
    }
    return output;
}

template <typename Launch>
double time_ms(int iterations, Launch&& launch) {
    for (int i = 0; i < 3; ++i) {
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
    float total = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&total, start.get(), stop.get()));
    return static_cast<double>(total) / iterations;
}

template <unsigned Group, bool KTokenFastest, bool VTokenFastest, bool ScaleTokenFastest>
void run_variant(std::size_t context, int iterations, std::uint32_t query_rows,
                 bool timing_only, bool transaction_timing_only = false) {
    constexpr std::string_view key_layout_name = KTokenFastest ? "token" : "feature";
    constexpr std::string_view value_layout_name = VTokenFastest ? "token" : "feature";
    constexpr std::string_view scale_layout_name = ScaleTokenFastest ? "token" : "feature";
    const std::vector<std::uint32_t> pages = make_fragmented_pages(context);
    const std::size_t physical_pages = pages.size() + 2;
    const std::size_t physical_tokens = physical_pages * kPageTokens;
    std::vector<std::uint32_t> positions(context);
    std::iota(positions.begin(), positions.end(), 0U);
    std::vector<float> source_keys = make_keys(context);
    std::vector<float> source_values = make_values(context);
    inject_codec_edge_corpus<Group>(source_keys, source_values);
    const std::vector<float> base_query = make_query();
    std::vector<float> query(static_cast<std::size_t>(query_rows) * base_query.size());
    for (std::uint32_t row = 0; row < query_rows; ++row) {
        for (std::size_t element = 0; element < base_query.size(); ++element) {
            // Each row has independently represented BF16 Q.  This avoids accidentally proving
            // only repeated-query behavior while keeping the KV state shared, as a prefill suffix
            // does.
            const int perturbation = static_cast<int>((row * 11U + element * 3U) % 7U) - 3;
            query[static_cast<std::size_t>(row) * base_query.size() + element] =
                represented_bf16(base_query[element] + static_cast<float>(perturbation) / 1024.0F);
        }
    }
    std::vector<std::size_t> row_contexts(query_rows);
    std::vector<std::int32_t> row_positions(query_rows);
    for (std::uint32_t row = 0; row < query_rows; ++row) {
        // A descending causal prefix followed by the full frontier is the ragged prefill shape.
        // At short tails it clamps to one rather than manufacturing an invalid empty cache.
        row_contexts[row] = std::max<std::size_t>(1U, context - (query_rows - 1U - row));
        row_positions[row] = static_cast<std::int32_t>(row_contexts[row] - 1U);
    }

    const std::size_t key_size = physical_tokens * kKvHeads * kDimension;
    const std::size_t value_size = physical_tokens * kKvHeads * kDimension / 2;
    const std::size_t scale_size =
        physical_tokens * kKvHeads * Layout<Group, ScaleTokenFastest>::kGroups;
    std::vector<std::uint8_t> expected_keys(key_size, 0xa5U);
    std::vector<std::uint8_t> expected_values(value_size, 0x5aU);
    std::vector<std::uint16_t> expected_scales(scale_size, 0xa5a5U);
    append_oracle<Group, KTokenFastest, VTokenFastest, ScaleTokenFastest>(source_keys,
                                       source_values,
                                       positions,
                                       pages,
                                       physical_tokens,
                                       expected_keys,
                                       expected_values,
                                       expected_scales);

    std::vector<hip_bfloat16> bf16_source_keys;
    std::vector<hip_bfloat16> bf16_source_values;
    std::vector<hip_bfloat16> bf16_query;
    bf16_source_keys.reserve(source_keys.size());
    bf16_source_values.reserve(source_values.size());
    bf16_query.reserve(query.size());
    for (float value : source_keys) { bf16_source_keys.emplace_back(value); }
    for (float value : source_values) { bf16_source_values.emplace_back(value); }
    for (float value : query) { bf16_query.emplace_back(value); }

    DeviceBuffer<hip_bfloat16> d_source_keys(source_keys.size());
    DeviceBuffer<hip_bfloat16> d_source_values(source_values.size());
    DeviceBuffer<hip_bfloat16> d_query(query.size());
    DeviceBuffer<std::uint32_t> d_positions(positions.size());
    DeviceBuffer<std::uint32_t> d_pages(pages.size());
    DeviceBuffer<std::uint8_t> d_keys(key_size);
    DeviceBuffer<std::uint8_t> d_values(value_size);
    DeviceBuffer<std::uint16_t> d_scales(scale_size);
    DeviceBuffer<std::uint32_t> d_append_status(1);
    DeviceBuffer<float> d_output(query.size());
    DeviceBuffer<float> d_wmma_attention_output(query.size());
    DeviceBuffer<std::int32_t> d_row_positions(row_positions.size());
    DeviceBuffer<float> d_qk_scores(static_cast<std::size_t>(kQueryHeads) * context);
    DeviceBuffer<float> d_qk_wmma_scores(static_cast<std::size_t>(kQueryHeads) * context);
    DeviceBuffer<float> d_probabilities(static_cast<std::size_t>(kQueryHeads) * context);
    DeviceBuffer<float> d_pv_output(query.size());
    HIP_CHECK(hipMemcpy(d_source_keys.get(),
                        bf16_source_keys.data(),
                        bf16_source_keys.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_source_values.get(),
                        bf16_source_values.data(),
                        bf16_source_values.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_query.get(), bf16_query.data(), bf16_query.size() * sizeof(hip_bfloat16),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_positions.get(), row_positions.data(),
                        row_positions.size() * sizeof(std::int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_positions.get(),
                        positions.data(),
                        positions.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pages.get(),
                        pages.data(),
                        pages.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_keys.get(), 0xa5, key_size));
    HIP_CHECK(hipMemset(d_values.get(), 0x5a, value_size));
    HIP_CHECK(hipMemset(d_scales.get(), 0xa5, scale_size * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(d_append_status.get(), 0, sizeof(std::uint32_t)));

    if (ninfer::ops::r9700::kv::fp8_int4_kv_append(
            ninfer::ops::r9700::kv::Fp8Int4KvAppendArgs{.tokens = 0}, nullptr) !=
        hipErrorInvalidValue) {
        fail("A2 append did not reject an empty transaction");
    }
    {
        const ninfer::ops::r9700::kv::Fp8Int4KvAppendArgs invalid_layout{
            .keys = d_source_keys.get(), .values = d_source_values.get(),
            .positions = d_positions.get(), .tokens = context, .kv_heads = kKvHeads,
            .page_table = d_pages.get(), .page_table_pages = static_cast<std::uint32_t>(pages.size()),
            .physical_tokens = physical_tokens, .key_codes = d_keys.get(),
            .value_codes = d_values.get(), .value_scale_bits = d_scales.get(),
            .status = d_append_status.get(), .value_group = Group,
            .key_layout = static_cast<ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout>(255U),
        };
        if (ninfer::ops::r9700::kv::fp8_int4_kv_append(invalid_layout, nullptr) !=
            hipErrorInvalidValue) {
            fail("A2 append accepted an invalid plane layout selector");
        }
    }

    const auto launch_append = [&] {
        constexpr auto key_layout = KTokenFastest
                                          ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                          : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        constexpr auto value_layout = VTokenFastest
                                          ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                          : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        constexpr auto scale_layout = ScaleTokenFastest
                                          ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                          : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        const ninfer::ops::r9700::kv::Fp8Int4KvAppendArgs args{
            .keys             = d_source_keys.get(),
            .values           = d_source_values.get(),
            .positions        = d_positions.get(),
            .tokens           = context,
            .kv_heads         = kKvHeads,
            .page_table       = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size()),
            .physical_tokens  = physical_tokens,
            .key_codes        = d_keys.get(),
            .value_codes      = d_values.get(),
            .value_scale_bits = d_scales.get(),
            .status           = d_append_status.get(),
            .value_group      = Group,
            .key_layout       = key_layout,
            .value_layout     = value_layout,
            .value_scale_layout = scale_layout,
        };
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_append(args, nullptr));
    };
    // Production does not rewrite the full visible context. Each attention invocation first
    // appends exactly its staged decode/prefill suffix T, then consumes the resulting frontier on
    // the same ordered stream. Keep that transaction-shaped pair distinct from the full-context
    // setup append used by this fixture's byte oracle.
    const std::size_t appended_rows = std::min<std::size_t>(query_rows, context);
    const auto launch_append_suffix = [&] {
        constexpr auto key_layout = KTokenFastest
                                          ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                          : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        constexpr auto value_layout = VTokenFastest
                                          ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                          : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        constexpr auto scale_layout = ScaleTokenFastest
                                          ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                          : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
        const std::size_t first = context - appended_rows;
        const std::size_t source_offset = first * kKvHeads * kDimension;
        const ninfer::ops::r9700::kv::Fp8Int4KvAppendArgs args{
            .keys             = d_source_keys.get() + source_offset,
            .values           = d_source_values.get() + source_offset,
            .positions        = d_positions.get() + first,
            .tokens           = appended_rows,
            .first_position   = static_cast<std::uint32_t>(first),
            .kv_heads         = kKvHeads,
            .page_table       = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size()),
            .physical_tokens  = physical_tokens,
            .key_codes        = d_keys.get(),
            .value_codes      = d_values.get(),
            .value_scale_bits = d_scales.get(),
            .status           = d_append_status.get(),
            .value_group      = Group,
            .key_layout       = key_layout,
            .value_layout     = value_layout,
            .value_scale_layout = scale_layout,
        };
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_append(args, nullptr));
    };
    launch_append();
    std::uint32_t append_status = 0;
    HIP_CHECK(hipMemcpy(&append_status,
                        d_append_status.get(),
                        sizeof(append_status),
                        hipMemcpyDeviceToHost));
    if (append_status != ninfer::ops::r9700::kv::Fp8Int4KvAppendOk) {
        fail("A2 append reported a device codec/address error");
    }
    std::vector<std::uint8_t> actual_keys(key_size);
    std::vector<std::uint8_t> actual_values(value_size);
    std::vector<std::uint16_t> actual_scales(scale_size);
    HIP_CHECK(hipMemcpy(actual_keys.data(), d_keys.get(), key_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_values.data(), d_values.get(), value_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_scales.data(),
                        d_scales.get(),
                        scale_size * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost));
    if (actual_keys != expected_keys || actual_values != expected_values ||
        actual_scales != expected_scales) {
        fail("A2 append physical planes differ from independent host codec/layout oracle");
    }

    constexpr auto qk_layout = KTokenFastest
                                   ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                   : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    const auto launch_qk_vector = [&] {
        const ninfer::ops::r9700::kv::Fp8KvQkArgs args{
            .query = d_query.get(), .key_codes = d_keys.get(), .page_table = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size()), .context = context,
            .physical_tokens = physical_tokens, .query_heads = kQueryHeads, .kv_heads = kKvHeads,
            .q_to_kv = kQueriesPerKv, .attention_scale = kAttentionScale, .key_layout = qk_layout,
            .scores = d_qk_scores.get()};
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_kv_qk_vector(args, nullptr));
    };
    {
        const ninfer::ops::r9700::kv::Fp8KvQkArgs invalid_pages{
            .query = d_query.get(), .key_codes = d_keys.get(), .page_table = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size() - 1U), .context = context,
            .physical_tokens = physical_tokens, .query_heads = kQueryHeads, .kv_heads = kKvHeads,
            .q_to_kv = kQueriesPerKv, .attention_scale = kAttentionScale, .key_layout = qk_layout,
            .scores = d_qk_scores.get()};
        if (ninfer::ops::r9700::kv::fp8_kv_qk_vector(invalid_pages, nullptr) != hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_kv_qk_wmma(invalid_pages, nullptr) != hipErrorInvalidValue) {
            fail("A3 QK did not reject an undersized page table");
        }
        auto empty = invalid_pages;
        empty.page_table_pages = static_cast<std::uint32_t>(pages.size());
        empty.context = 0;
        if (ninfer::ops::r9700::kv::fp8_kv_qk_vector(empty, nullptr) != hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_kv_qk_wmma(empty, nullptr) != hipErrorInvalidValue) {
            fail("A3 QK did not reject an empty causal context");
        }
        auto nonfinite_scale = empty;
        nonfinite_scale.context = context;
        nonfinite_scale.attention_scale = std::numeric_limits<float>::infinity();
        if (ninfer::ops::r9700::kv::fp8_kv_qk_vector(nonfinite_scale, nullptr) !=
                hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_kv_qk_wmma(nonfinite_scale, nullptr) !=
                hipErrorInvalidValue) {
            fail("A3 QK accepted a nonfinite attention scale");
        }
        auto invalid_layout = empty;
        invalid_layout.context = context;
        invalid_layout.key_layout =
            static_cast<ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout>(255U);
        if (ninfer::ops::r9700::kv::fp8_kv_qk_vector(invalid_layout, nullptr) !=
                hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_kv_qk_wmma(invalid_layout, nullptr) !=
                hipErrorInvalidValue) {
            fail("A3 QK accepted an invalid K layout selector");
        }
        std::vector<std::uint32_t> corrupt_pages = pages;
        corrupt_pages.front() = static_cast<std::uint32_t>(physical_tokens / kPageTokens);
        DeviceBuffer<std::uint32_t> d_corrupt_qk_pages(corrupt_pages.size());
        HIP_CHECK(hipMemcpy(d_corrupt_qk_pages.get(), corrupt_pages.data(),
                            corrupt_pages.size() * sizeof(std::uint32_t), hipMemcpyHostToDevice));
        auto corrupt_map = empty;
        corrupt_map.context = context;
        corrupt_map.page_table = d_corrupt_qk_pages.get();
        corrupt_map.scores = d_qk_scores.get();
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_kv_qk_vector(corrupt_map, nullptr));
        corrupt_map.scores = d_qk_wmma_scores.get();
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_kv_qk_wmma(corrupt_map, nullptr));
        std::vector<float> corrupt_vector(static_cast<std::size_t>(kQueryHeads) * context);
        std::vector<float> corrupt_wmma(corrupt_vector.size());
        HIP_CHECK(hipMemcpy(corrupt_vector.data(), d_qk_scores.get(),
                            corrupt_vector.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(corrupt_wmma.data(), d_qk_wmma_scores.get(),
                            corrupt_wmma.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (unsigned head = 0; head < kQueryHeads; ++head) {
            for (std::size_t logical = 0; logical < context; ++logical) {
                const bool mapped_bad = logical < kPageTokens;
                const std::size_t index = static_cast<std::size_t>(head) * context + logical;
                if (std::isnan(corrupt_vector[index]) != mapped_bad ||
                    std::isnan(corrupt_wmma[index]) != mapped_bad) {
                    std::ostringstream os;
                    os << "A3 QK corrupt map mismatch at head=" << head << " token=" << logical
                       << " expected_nan=" << mapped_bad << " vector=" << corrupt_vector[index]
                       << " wmma=" << corrupt_wmma[index];
                    fail(os.str());
                }
            }
        }
    }
    if (!timing_only) {
        launch_qk_vector();
        std::vector<float> actual_qk(static_cast<std::size_t>(kQueryHeads) * context);
        HIP_CHECK(hipMemcpy(actual_qk.data(), d_qk_scores.get(), actual_qk.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        namespace oracle = ninfer::test::fp8_int4_kv_oracle;
        for (unsigned query_head = 0; query_head < kQueryHeads; ++query_head) {
            const unsigned kv_head = query_head / kQueriesPerKv;
            for (std::size_t logical = 0; logical < context; ++logical) {
                const std::size_t physical = host_physical_token(logical, pages, physical_tokens);
                double expected = 0.0;
                for (unsigned feature = 0; feature < kDimension; ++feature) {
                    expected += static_cast<double>(query[query_head * kDimension + feature]) *
                                oracle::decode_e4m3fn(actual_keys[host_key_offset<Group, KTokenFastest>(
                                    physical, kv_head, feature, physical_tokens)]);
                }
                expected *= static_cast<double>(kAttentionScale);
                const float actual = actual_qk[static_cast<std::size_t>(query_head) * context + logical];
                if (std::abs(static_cast<double>(actual) - expected) > 1.0e-4) {
                    std::ostringstream os;
                    os << "A3 vector QK mismatch at head=" << query_head << " token=" << logical
                       << ": got " << actual << " expected " << expected;
                    fail(os.str());
                }
            }
        }
    }

    const std::vector<float> probability_corpus = make_probability_corpus(context);
    HIP_CHECK(hipMemcpy(d_probabilities.get(), probability_corpus.data(),
                        probability_corpus.size() * sizeof(float), hipMemcpyHostToDevice));
    const ninfer::ops::r9700::kv::Int4KvPvArgs pv_args{
        .probabilities = d_probabilities.get(), .value_codes = d_values.get(),
        .value_scale_bits = d_scales.get(), .page_table = d_pages.get(),
        .page_table_pages = static_cast<std::uint32_t>(pages.size()), .context = context,
        .physical_tokens = physical_tokens, .query_heads = kQueryHeads, .kv_heads = kKvHeads,
        .q_to_kv = kQueriesPerKv, .value_group = Group,
        .value_layout = VTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .value_scale_layout = ScaleTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
        .output = d_pv_output.get()};
    {
        auto invalid_pages = pv_args;
        invalid_pages.page_table_pages = static_cast<std::uint32_t>(pages.size() - 1U);
        if (ninfer::ops::r9700::kv::int4_kv_pv_vector(invalid_pages, nullptr) != hipErrorInvalidValue) {
            fail("A3 PV did not reject an undersized page table");
        }
        auto empty = pv_args;
        empty.context = 0;
        if (ninfer::ops::r9700::kv::int4_kv_pv_vector(empty, nullptr) != hipErrorInvalidValue) {
            fail("A3 PV did not reject an empty causal context");
        }
        auto invalid_layout = pv_args;
        invalid_layout.value_layout =
            static_cast<ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout>(255U);
        if (ninfer::ops::r9700::kv::int4_kv_pv_vector(invalid_layout, nullptr) !=
            hipErrorInvalidValue) {
            fail("A3 PV accepted an invalid V layout selector");
        }
    }
    const auto launch_pv_vector = [&] {
        HIP_CHECK(ninfer::ops::r9700::kv::int4_kv_pv_vector(pv_args, nullptr));
    };
    if (!timing_only) {
        launch_pv_vector();
        std::vector<float> actual_pv(base_query.size());
        HIP_CHECK(hipMemcpy(actual_pv.data(), d_pv_output.get(), actual_pv.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        for (unsigned query_head = 0; query_head < kQueryHeads; ++query_head) {
        const unsigned kv_head = query_head / kQueriesPerKv;
        for (unsigned feature = 0; feature < kDimension; ++feature) {
            double expected = 0.0;
            for (std::size_t logical = 0; logical < context; ++logical) {
                const std::size_t physical = host_physical_token(logical, pages, physical_tokens);
                const std::uint8_t packed = actual_values[host_value_offset<Group, VTokenFastest>(
                    physical, kv_head, feature / Group, (feature % Group) / 2U, physical_tokens)];
                int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 0x0fU);
                code = code >= 8 ? code - 16 : code;
                const std::uint16_t bits = actual_scales[host_scale_offset<Group, ScaleTokenFastest>(
                    physical, kv_head, feature / Group, physical_tokens)];
                _Float16 scale{};
                std::memcpy(&scale, &bits, sizeof(bits));
                expected += static_cast<double>(probability_corpus[
                                static_cast<std::size_t>(query_head) * context + logical]) *
                            static_cast<double>(code) * static_cast<float>(scale);
            }
            const float actual = actual_pv[query_head * kDimension + feature];
            if (std::abs(static_cast<double>(actual) - expected) > 1.0e-4) {
                std::ostringstream os;
                os << "A3 vector PV mismatch at head=" << query_head << " feature=" << feature
                   << ": got " << actual << " expected " << expected;
                fail(os.str());
            }
        }
        }
    }

    const auto launch_qk_wmma = [&] {
        const ninfer::ops::r9700::kv::Fp8KvQkArgs args{
            .query = d_query.get(), .key_codes = d_keys.get(), .page_table = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size()), .context = context,
            .physical_tokens = physical_tokens, .query_heads = kQueryHeads, .kv_heads = kKvHeads,
            .q_to_kv = kQueriesPerKv, .attention_scale = kAttentionScale, .key_layout = qk_layout,
            .scores = d_qk_wmma_scores.get()};
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_kv_qk_wmma(args, nullptr));
    };
    if (!timing_only) {
        launch_qk_wmma();
        std::vector<float> actual_qk_wmma(static_cast<std::size_t>(kQueryHeads) * context);
        HIP_CHECK(hipMemcpy(actual_qk_wmma.data(), d_qk_wmma_scores.get(),
                            actual_qk_wmma.size() * sizeof(float), hipMemcpyDeviceToHost));
        namespace oracle = ninfer::test::fp8_int4_kv_oracle;
        for (unsigned query_head = 0; query_head < kQueryHeads; ++query_head) {
            const unsigned kv_head = query_head / kQueriesPerKv;
            for (std::size_t logical = 0; logical < context; ++logical) {
                const std::size_t physical = host_physical_token(logical, pages, physical_tokens);
                double expected = 0.0;
                for (unsigned feature = 0; feature < kDimension; ++feature) {
                    const std::uint8_t q_code = oracle::encode_e4m3fn(query[query_head * kDimension + feature]);
                    expected += oracle::decode_e4m3fn(q_code) *
                                oracle::decode_e4m3fn(actual_keys[host_key_offset<Group, KTokenFastest>(
                                    physical, kv_head, feature, physical_tokens)]);
                }
                expected *= static_cast<double>(kAttentionScale);
                const std::size_t index = static_cast<std::size_t>(query_head) * context + logical;
                if (std::abs(static_cast<double>(actual_qk_wmma[index]) - expected) > 2.0e-4) {
                    std::ostringstream os;
                    os << "A3 raw WMMA QK mismatch at score " << index << ": got "
                       << actual_qk_wmma[index] << " expected " << expected;
                    fail(os.str());
                }
            }
        }
    }

    const auto launch_attention = [&] {
        const ninfer::ops::r9700::kv::Fp8Int4KvAttentionArgs args{
            .query = d_query.get(), .key_codes = d_keys.get(), .value_codes = d_values.get(),
            .value_scale_bits = d_scales.get(), .page_table = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size()), .context = context,
            .physical_tokens = physical_tokens, .query_heads = kQueryHeads, .kv_heads = kKvHeads,
            .q_to_kv = kQueriesPerKv, .value_group = Group,
            .query_rows = query_rows, .row_positions = d_row_positions.get(),
            .attention_scale = kAttentionScale, .key_layout = qk_layout,
            .value_layout = VTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .value_scale_layout = ScaleTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .output = d_output.get()};
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(args, nullptr));
    };
    const auto launch_attention_wmma = [&] {
        const ninfer::ops::r9700::kv::Fp8Int4KvAttentionArgs args{
            .query = d_query.get(), .key_codes = d_keys.get(), .value_codes = d_values.get(),
            .value_scale_bits = d_scales.get(), .page_table = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size()), .context = context,
            .physical_tokens = physical_tokens, .query_heads = kQueryHeads, .kv_heads = kKvHeads,
            .q_to_kv = kQueriesPerKv, .value_group = Group,
            .query_rows = query_rows, .row_positions = d_row_positions.get(),
            .attention_scale = kAttentionScale, .key_layout = qk_layout,
            .value_layout = VTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .value_scale_layout = ScaleTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .score_workspace = d_qk_wmma_scores.get(),
            .output = d_wmma_attention_output.get()};
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(args, nullptr));
    };
    {
        const ninfer::ops::r9700::kv::Fp8Int4KvAttentionArgs invalid_args{
            .query = d_query.get(), .key_codes = d_keys.get(), .value_codes = d_values.get(),
            .value_scale_bits = d_scales.get(), .page_table = d_pages.get(),
            .page_table_pages = static_cast<std::uint32_t>(pages.size() - 1U), .context = context,
            .physical_tokens = physical_tokens, .query_heads = kQueryHeads, .kv_heads = kKvHeads,
            .q_to_kv = kQueriesPerKv, .value_group = Group,
            .query_rows = query_rows, .row_positions = d_row_positions.get(),
            .attention_scale = kAttentionScale,
            .key_layout = qk_layout,
            .value_layout = VTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .value_scale_layout = ScaleTokenFastest ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .score_workspace = d_qk_wmma_scores.get(),
            .output = d_output.get()};
        if (ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(invalid_args, nullptr) !=
                hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(invalid_args, nullptr) !=
                hipErrorInvalidValue) {
            fail("A3 attention routes did not reject an undersized page table");
        }
        auto empty = invalid_args;
        empty.page_table_pages = static_cast<std::uint32_t>(pages.size());
        empty.context = 0;
        if (ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(empty, nullptr) !=
                hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(empty, nullptr) !=
                hipErrorInvalidValue) {
            fail("A3 attention routes did not reject an empty causal context");
        }
        auto nonfinite_scale = invalid_args;
        nonfinite_scale.page_table_pages = static_cast<std::uint32_t>(pages.size());
        nonfinite_scale.attention_scale = std::numeric_limits<float>::infinity();
        if (ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(nonfinite_scale, nullptr) !=
                hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(nonfinite_scale, nullptr) !=
                hipErrorInvalidValue) {
            fail("A3 attention routes accepted a nonfinite attention scale");
        }
        auto invalid_layout = invalid_args;
        invalid_layout.page_table_pages = static_cast<std::uint32_t>(pages.size());
        invalid_layout.value_scale_layout =
            static_cast<ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout>(255U);
        if (ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(invalid_layout, nullptr) !=
                hipErrorInvalidValue ||
            ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(invalid_layout, nullptr) !=
                hipErrorInvalidValue) {
            fail("A3 attention routes accepted an invalid plane layout selector");
        }
        std::vector<std::uint32_t> corrupt_pages = pages;
        corrupt_pages.front() = static_cast<std::uint32_t>(physical_tokens / kPageTokens);
        DeviceBuffer<std::uint32_t> d_corrupt_pages(corrupt_pages.size());
        HIP_CHECK(hipMemcpy(d_corrupt_pages.get(), corrupt_pages.data(),
                            corrupt_pages.size() * sizeof(std::uint32_t), hipMemcpyHostToDevice));
        auto corrupt_args = invalid_args;
        corrupt_args.page_table = d_corrupt_pages.get();
        corrupt_args.page_table_pages = static_cast<std::uint32_t>(corrupt_pages.size());
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(corrupt_args, nullptr));
        std::vector<float> corrupt_output(query.size());
        HIP_CHECK(hipMemcpy(corrupt_output.data(), d_output.get(), corrupt_output.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        if (!std::all_of(corrupt_output.begin(), corrupt_output.end(),
                         [](float value) { return std::isnan(value); })) {
            fail("fused A3 did not expose a corrupt mapped page as invalid output");
        }
        corrupt_args.output = d_wmma_attention_output.get();
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(corrupt_args, nullptr));
        HIP_CHECK(hipMemcpy(corrupt_output.data(), d_wmma_attention_output.get(),
                            corrupt_output.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (!std::all_of(corrupt_output.begin(), corrupt_output.end(),
                         [](float value) { return std::isnan(value); })) {
            fail("WMMA A3 did not expose a corrupt mapped page as invalid output");
        }
        // Raw attention receives dynamic I32 causal positions from the schedule. A bad device
        // value must not silently turn into an empty reduction or read past the captured cache
        // frontier. It poisons only its affected row, leaving other launched rows observable.
        std::vector<std::int32_t> corrupt_positions = row_positions;
        corrupt_positions.front() = -1;
        DeviceBuffer<std::int32_t> d_corrupt_positions(corrupt_positions.size());
        HIP_CHECK(hipMemcpy(d_corrupt_positions.get(), corrupt_positions.data(),
                            corrupt_positions.size() * sizeof(std::int32_t),
                            hipMemcpyHostToDevice));
        auto corrupt_frontier_args = invalid_args;
        corrupt_frontier_args.page_table_pages = static_cast<std::uint32_t>(pages.size());
        corrupt_frontier_args.row_positions = d_corrupt_positions.get();
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(corrupt_frontier_args,
                                                                        nullptr));
        HIP_CHECK(hipMemcpy(corrupt_output.data(), d_output.get(), corrupt_output.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        const std::size_t row_width = base_query.size();
        if (!std::all_of(corrupt_output.begin(), corrupt_output.begin() + row_width,
                         [](float value) { return std::isnan(value); }) ||
            (query_rows > 1U && std::all_of(corrupt_output.begin() + row_width,
                                             corrupt_output.end(),
                                             [](float value) { return std::isnan(value); }))) {
            fail("fused A3 did not isolate a negative causal position to its row");
        }
        corrupt_frontier_args.output = d_wmma_attention_output.get();
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(corrupt_frontier_args,
                                                                      nullptr));
        HIP_CHECK(hipMemcpy(corrupt_output.data(), d_wmma_attention_output.get(),
                            corrupt_output.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (!std::all_of(corrupt_output.begin(), corrupt_output.begin() + row_width,
                         [](float value) { return std::isnan(value); }) ||
            (query_rows > 1U && std::all_of(corrupt_output.begin() + row_width,
                                             corrupt_output.end(),
                                             [](float value) { return std::isnan(value); }))) {
            fail("WMMA A3 did not isolate a negative causal position to its row");
        }
        corrupt_positions.front() = static_cast<std::int32_t>(context);
        HIP_CHECK(hipMemcpy(d_corrupt_positions.get(), corrupt_positions.data(),
                            corrupt_positions.size() * sizeof(std::int32_t),
                            hipMemcpyHostToDevice));
        corrupt_frontier_args.output = d_output.get();
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(corrupt_frontier_args,
                                                                        nullptr));
        HIP_CHECK(hipMemcpy(corrupt_output.data(), d_output.get(), corrupt_output.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        if (!std::all_of(corrupt_output.begin(), corrupt_output.begin() + row_width,
                         [](float value) { return std::isnan(value); })) {
            fail("fused A3 did not reject a causal position at its captured frontier");
        }
        corrupt_frontier_args.output = d_wmma_attention_output.get();
        HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(corrupt_frontier_args,
                                                                      nullptr));
        HIP_CHECK(hipMemcpy(corrupt_output.data(), d_wmma_attention_output.get(),
                            corrupt_output.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (!std::all_of(corrupt_output.begin(), corrupt_output.begin() + row_width,
                         [](float value) { return std::isnan(value); })) {
            fail("WMMA A3 did not reject a causal position at its captured frontier");
        }
    }
    double max_absolute = 0.0;
    double max_relative = 0.0;
    double wmma_max_absolute = 0.0;
    double wmma_max_relative = 0.0;
    if (!timing_only) {
        launch_attention();
        std::vector<float> actual_output(query.size());
        HIP_CHECK(hipMemcpy(actual_output.data(),
                            d_output.get(),
                            actual_output.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        std::vector<double> expected_output;
        expected_output.reserve(query.size());
        for (std::uint32_t row = 0; row < query_rows; ++row) {
            const auto row_begin = query.begin() + static_cast<std::size_t>(row) * base_query.size();
            const std::vector<float> row_query(row_begin, row_begin + base_query.size());
            const std::vector<double> row_expected =
                attention_oracle<Group, KTokenFastest, VTokenFastest, ScaleTokenFastest>(
                    row_query, actual_keys, actual_values, actual_scales, pages, row_contexts[row],
                    physical_tokens);
            expected_output.insert(expected_output.end(), row_expected.begin(), row_expected.end());
        }
        for (std::size_t i = 0; i < actual_output.size(); ++i) {
            const double absolute = std::abs(static_cast<double>(actual_output[i]) - expected_output[i]);
            const double relative = absolute / std::max(std::abs(expected_output[i]), 1.0e-6);
            max_absolute = std::max(max_absolute, absolute);
            max_relative = std::max(max_relative, relative);
            if (absolute > 2.0e-4 && relative > 2.0e-4) {
                std::ostringstream os;
                os << "A3 output mismatch at " << i << ": got " << actual_output[i]
                   << " expected " << expected_output[i] << " abs=" << absolute
                   << " rel=" << relative;
                fail(os.str());
            }
        }
        launch_attention_wmma();
        std::vector<float> actual_wmma_output(query.size());
        HIP_CHECK(hipMemcpy(actual_wmma_output.data(), d_wmma_attention_output.get(),
                            actual_wmma_output.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < actual_wmma_output.size(); ++i) {
            const double absolute =
                std::abs(static_cast<double>(actual_wmma_output[i]) - expected_output[i]);
            const double relative = absolute / std::max(std::abs(expected_output[i]), 1.0e-6);
            wmma_max_absolute = std::max(wmma_max_absolute, absolute);
            wmma_max_relative = std::max(wmma_max_relative, relative);
            // FP8 Q is a private native-WMMA profile checked directly against the represented
            // BF16-input oracle. This admission threshold is Op-local; model PPL/token gates
            // remain mandatory before it can replace the BF16-Q score-streaming baseline.
            if (absolute > 5.0e-3 && relative > 5.0e-3) {
                std::ostringstream os;
                os << "A3 WMMA attention mismatch at " << i << ": got "
                   << actual_wmma_output[i] << " expected " << expected_output[i]
                   << " abs=" << absolute << " rel=" << relative;
                fail(os.str());
            }
        }
    }

    // The independent exact codec and FP64-attention checks above run before any timing. Measure
    // both launches inside one event interval to expose the only latency that a legal A2+A3
    // fusion could improve; separately timed launches alone do not include their ordered-pair
    // scheduling behavior.
    const auto launch_append_attention = [&] {
        launch_append_suffix();
        launch_attention();
    };
    const double setup_append_ms = time_ms(iterations, launch_append);
    const double suffix_append_ms = time_ms(iterations, launch_append_suffix);
    const double qk_vector_ms = transaction_timing_only
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : time_ms(iterations, launch_qk_vector);
    const double qk_wmma_ms = transaction_timing_only
                                  ? std::numeric_limits<double>::quiet_NaN()
                                  : time_ms(iterations, launch_qk_wmma);
    const double pv_vector_ms = transaction_timing_only
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : time_ms(iterations, launch_pv_vector);
    const double attention_ms = time_ms(iterations, launch_attention);
    const double append_attention_ms = time_ms(iterations, launch_append_attention);
    const double attention_wmma_ms = transaction_timing_only
                                         ? std::numeric_limits<double>::quiet_NaN()
                                         : time_ms(iterations, launch_attention_wmma);

    // Long runs use the cross-page speculative path. Tail runs use the same exact three-plane
    // operation with a self-copy, which exercises a valid causal frontier without inventing
    // source positions outside the short context.
    const std::size_t compact_prefix = context >= 66 ? 55U : 0U;
    const std::vector<std::uint32_t> selected_path =
        context >= 66 ? std::vector<std::uint32_t>{0U, 3U, 8U, 9U, 10U}
                      : std::vector<std::uint32_t>{0U};
    std::vector<std::uint8_t> expected_compact_keys = actual_keys;
    std::vector<std::uint8_t> expected_compact_values = actual_values;
    std::vector<std::uint16_t> expected_compact_scales = actual_scales;
    for (std::size_t selected = 0; selected < selected_path.size(); ++selected) {
        if (selected_path[selected] < selected) {
            fail("in-place packed-tree path must be monotone source>=destination");
        }
        const std::size_t source_token =
            host_physical_token(compact_prefix + selected_path[selected], pages, physical_tokens);
        const std::size_t destination_token =
            host_physical_token(compact_prefix + selected, pages, physical_tokens);
        for (unsigned head = 0; head < kKvHeads; ++head) {
            for (unsigned feature = 0; feature < kDimension; ++feature) {
                expected_compact_keys[host_key_offset<Group, KTokenFastest>(
                    destination_token, head, feature, physical_tokens)] =
                    expected_compact_keys[host_key_offset<Group, KTokenFastest>(
                        source_token, head, feature, physical_tokens)];
            }
            for (unsigned group = 0; group < Layout<Group, ScaleTokenFastest>::kGroups; ++group) {
                expected_compact_scales[host_scale_offset<Group, ScaleTokenFastest>(
                    destination_token, head, group, physical_tokens)] =
                    expected_compact_scales[host_scale_offset<Group, ScaleTokenFastest>(
                        source_token, head, group, physical_tokens)];
                for (unsigned byte = 0; byte < Group / 2; ++byte) {
                    expected_compact_values[host_value_offset<Group, VTokenFastest>(
                        destination_token, head, group, byte, physical_tokens)] =
                        expected_compact_values[host_value_offset<Group, VTokenFastest>(
                            source_token, head, group, byte, physical_tokens)];
                }
            }
        }
    }

    DeviceBuffer<std::uint32_t> d_selected_path(selected_path.size());
    HIP_CHECK(hipMemcpy(d_selected_path.get(),
                        selected_path.data(),
                        selected_path.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
    const auto compact_key_layout = KTokenFastest
                                    ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                    : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    const auto compact_value_layout = VTokenFastest
                                      ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                      : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    const auto compact_scale_layout = ScaleTokenFastest
                                      ? ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
                                      : ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    HIP_CHECK(hipMemset(d_append_status.get(), 0, sizeof(std::uint32_t)));
    const ninfer::ops::r9700::kv::Fp8Int4KvCompactArgs compact_args{
        .prefix = compact_prefix, .selected_path = d_selected_path.get(),
        .selected_count = selected_path.size(), .page_table = d_pages.get(),
        .page_table_pages = static_cast<std::uint32_t>(pages.size()),
        .physical_tokens = physical_tokens, .kv_heads = kKvHeads, .key_codes = d_keys.get(),
        .value_codes = d_values.get(), .value_scale_bits = d_scales.get(),
        .status = d_append_status.get(), .value_group = Group, .key_layout = compact_key_layout,
        .value_layout = compact_value_layout, .value_scale_layout = compact_scale_layout};
    auto empty_compact = compact_args;
    empty_compact.selected_count = 0;
    if (ninfer::ops::r9700::kv::fp8_int4_kv_compact(empty_compact, nullptr) !=
        hipErrorInvalidValue) {
        fail("A2 compaction did not reject an empty transaction");
    }
    auto invalid_compact_layout = compact_args;
    invalid_compact_layout.value_scale_layout =
        static_cast<ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout>(255U);
    if (ninfer::ops::r9700::kv::fp8_int4_kv_compact(invalid_compact_layout, nullptr) !=
        hipErrorInvalidValue) {
        fail("A2 compaction accepted an invalid plane layout selector");
    }
    HIP_CHECK(ninfer::ops::r9700::kv::fp8_int4_kv_compact(compact_args, nullptr));
    std::uint32_t compact_status = 0;
    HIP_CHECK(hipMemcpy(&compact_status, d_append_status.get(), sizeof(compact_status),
                        hipMemcpyDeviceToHost));
    if (compact_status != ninfer::ops::r9700::kv::Fp8Int4KvAppendOk) {
        fail("A2 compact reported a device address error");
    }
    std::vector<std::uint8_t> actual_compact_keys(key_size);
    std::vector<std::uint8_t> actual_compact_values(value_size);
    std::vector<std::uint16_t> actual_compact_scales(scale_size);
    HIP_CHECK(hipMemcpy(actual_compact_keys.data(),
                        d_keys.get(),
                        key_size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_compact_values.data(),
                        d_values.get(),
                        value_size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_compact_scales.data(),
                        d_scales.get(),
                        scale_size * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost));
    if (actual_compact_keys != expected_compact_keys ||
        actual_compact_values != expected_compact_values ||
        actual_compact_scales != expected_compact_scales) {
        fail("in-place packed-tree compaction was not bit-exact across all three physical planes");
    }

    const auto read_append_status = [&] {
        std::uint32_t status = 0;
        HIP_CHECK(hipMemcpy(&status,
                            d_append_status.get(),
                            sizeof(status),
                            hipMemcpyDeviceToHost));
        return status;
    };
    std::vector<hip_bfloat16> overflowing_values(source_values.size(), hip_bfloat16(524288.0F));
    HIP_CHECK(hipMemcpy(d_source_values.get(),
                        overflowing_values.data(),
                        overflowing_values.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_append_status.get(), 0, sizeof(std::uint32_t)));
    launch_append();
    if ((read_append_status() & ninfer::ops::r9700::kv::Fp8Int4KvAppendScaleOverflow) == 0U) {
        fail("A2 append did not report an unrepresentable FP16 V scale");
    }

    std::vector<std::uint32_t> invalid_positions = positions;
    invalid_positions.front() = 1U;
    HIP_CHECK(hipMemcpy(d_positions.get(),
                        invalid_positions.data(),
                        invalid_positions.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_append_status.get(), 0, sizeof(std::uint32_t)));
    launch_append();
    if ((read_append_status() & ninfer::ops::r9700::kv::Fp8Int4KvAppendInvalidPosition) == 0U) {
        fail("A2 append did not reject a noncontiguous device position panel");
    }

    // Keep positions valid so this case reaches the independent physical-page address guard.
    // A corrupted page ID must not be conflated with the contiguous-suffix status bit.
    std::vector<std::uint32_t> invalid_pages = pages;
    invalid_pages.front() = static_cast<std::uint32_t>(physical_pages);
    HIP_CHECK(hipMemcpy(d_positions.get(),
                        positions.data(),
                        positions.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pages.get(),
                        invalid_pages.data(),
                        invalid_pages.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_append_status.get(), 0, sizeof(std::uint32_t)));
    launch_append();
    if ((read_append_status() & ninfer::ops::r9700::kv::Fp8Int4KvAppendInvalidAddress) == 0U) {
        fail("A2 append did not report an invalid physical page address");
    }
    HIP_CHECK(hipMemcpy(d_pages.get(),
                        pages.data(),
                        pages.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));

    std::vector<hip_bfloat16> nonfinite_values = bf16_source_values;
    nonfinite_values.front().data = 0x7fc0U;
    HIP_CHECK(hipMemcpy(d_source_values.get(),
                        nonfinite_values.data(),
                        nonfinite_values.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_positions.get(),
                        positions.data(),
                        positions.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_append_status.get(), 0, sizeof(std::uint32_t)));
    launch_append();
    if ((read_append_status() & ninfer::ops::r9700::kv::Fp8Int4KvAppendNonfiniteInput) == 0U) {
        fail("A2 append did not report a nonfinite BF16 source");
    }

    std::cout << "variant G" << Group << " K=" << key_layout_name << " V=" << value_layout_name
              << " S=" << scale_layout_name << ": PASS"
              << (context >= 66 ? " A2[63/64/65]=exact" : " A2=tail-exact")
              << " edge_codec=exact"
              << " rows=" << query_rows
              << " tree_compact=exact"
              << (timing_only ? " A3_oracle=prior-qualified" : " A3_oracle=exact")
              << " A3_max_abs=" << std::scientific << max_absolute
              << " A3_max_rel=" << max_relative << std::fixed
              << " A3_wmma_max_abs=" << std::scientific << wmma_max_absolute
              << " A3_wmma_max_rel=" << wmma_max_relative << std::fixed
              << " setup_append_ms=" << std::setprecision(4) << setup_append_ms
              << " suffix_append_T=" << appended_rows
              << " suffix_append_ms=" << suffix_append_ms
              << " qk_vector_ms=" << qk_vector_ms
              << " qk_wmma_ms=" << qk_wmma_ms
              << " pv_vector_ms=" << pv_vector_ms
              << " attention_ms=" << attention_ms
              << " append_attention_ms=" << append_attention_ms
              << " attention_wmma_ms=" << attention_wmma_ms << '\n';
}

std::size_t parse_positive(const char* value, const char* option) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        fail(std::string("invalid ") + option + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

struct Options {
    std::size_t context = 256;
    int iterations = 20;
    std::uint32_t rows = 1;
    bool timing_only = false;
    bool transaction_timing_only = false;
    int group = 0;
    int key_order = -1;
    int value_order = -1;
    int scale_order = -1;
};

int parse_order(const char* value, const char* option) {
    const std::string_view order(value);
    if (order == "feature") { return 0; }
    if (order == "token") { return 1; }
    fail(std::string("invalid ") + option + ": expected feature or token");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--context" && i + 1 < argc) {
            options.context = parse_positive(argv[++i], "--context");
        } else if (argument == "--iterations" && i + 1 < argc) {
            const std::size_t iterations = parse_positive(argv[++i], "--iterations");
            if (iterations > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                fail("--iterations is too large");
            }
            options.iterations = static_cast<int>(iterations);
        } else if (argument == "--rows" && i + 1 < argc) {
            const std::size_t rows = parse_positive(argv[++i], "--rows");
            if (rows > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                fail("--rows exceeds the raw U32 query-row contract");
            }
            options.rows = static_cast<std::uint32_t>(rows);
        } else if (argument == "--timing-only") {
            options.timing_only = true;
        } else if (argument == "--transaction-timing-only") {
            options.timing_only = true;
            options.transaction_timing_only = true;
        } else if (argument == "--group" && i + 1 < argc) {
            const std::size_t group = parse_positive(argv[++i], "--group");
            if (group != 16U && group != 32U) { fail("--group must be 16 or 32"); }
            options.group = static_cast<int>(group);
        } else if (argument == "--key-order" && i + 1 < argc) {
            options.key_order = parse_order(argv[++i], "--key-order");
        } else if (argument == "--value-order" && i + 1 < argc) {
            options.value_order = parse_order(argv[++i], "--value-order");
        } else if (argument == "--scale-order" && i + 1 < argc) {
            options.scale_order = parse_order(argv[++i], "--scale-order");
        } else if (argument == "--help") {
            std::cout << "usage: kv_op_qual [--context N>=1] [--rows T>=1] [--iterations N]"
                         " [--timing-only|--transaction-timing-only]"
                         " [--group 16|32 --key-order feature|token"
                         " --value-order feature|token --scale-order feature|token]\n";
            std::exit(0);
        } else {
            fail(std::string("unknown or incomplete option: ") + argv[i]);
        }
    }
    // The fused A3 candidate streams online softmax state and has no context-sized shared-score
    // allocation. Keep the qualification ceiling at the required long-context gate rather than
    // inheriting the deleted shared-memory kernel's roughly-16K limit.
    constexpr std::size_t max_context = 32768;
    if (options.context > max_context) {
        std::ostringstream os;
        os << "--context exceeds the long-context qualification limit (" << max_context << ')';
        fail(os.str());
    }
    if (options.transaction_timing_only && options.rows > options.context) {
        fail("--transaction-timing-only requires T rows to fit in the visible context");
    }
    const bool any_variant = options.group != 0 || options.key_order >= 0 ||
                             options.value_order >= 0 || options.scale_order >= 0;
    const bool complete_variant = options.group != 0 && options.key_order >= 0 &&
                                  options.value_order >= 0 && options.scale_order >= 0;
    if (any_variant && !complete_variant) {
        fail("a selected variant requires --group, --key-order, --value-order, and --scale-order");
    }
    return options;
}

template <unsigned Group, bool KeyTokenFastest, bool ValueTokenFastest,
          bool ScaleTokenFastest>
bool run_selected_variant(const Options& options) {
    if (options.group != static_cast<int>(Group) ||
        options.key_order != static_cast<int>(KeyTokenFastest) ||
        options.value_order != static_cast<int>(ValueTokenFastest) ||
        options.scale_order != static_cast<int>(ScaleTokenFastest)) {
        return false;
    }
    run_variant<Group, KeyTokenFastest, ValueTokenFastest, ScaleTokenFastest>(
        options.context, options.iterations, options.rows, options.timing_only,
        options.transaction_timing_only);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        select_r9700();
        std::cout << "kv_op: D=" << kDimension << " Hq=" << kQueryHeads << " Hkv=" << kKvHeads
                  << " page=" << kPageTokens << " context=" << options.context
                  << " rows=" << options.rows
                  << " iterations=" << options.iterations
                  << (options.timing_only ? " (ROCm 10 timing-only; prior oracle required)\n"
                                          : " (ROCm 10 full oracle)\n");
        if (options.group != 0) {
            const bool ran =
                run_selected_variant<16, false, false, false>(options) ||
                run_selected_variant<16, false, false, true>(options) ||
                run_selected_variant<16, false, true, false>(options) ||
                run_selected_variant<16, false, true, true>(options) ||
                run_selected_variant<16, true, false, false>(options) ||
                run_selected_variant<16, true, false, true>(options) ||
                run_selected_variant<16, true, true, false>(options) ||
                run_selected_variant<16, true, true, true>(options) ||
                run_selected_variant<32, false, false, false>(options) ||
                run_selected_variant<32, false, false, true>(options) ||
                run_selected_variant<32, false, true, false>(options) ||
                run_selected_variant<32, false, true, true>(options) ||
                run_selected_variant<32, true, false, false>(options) ||
                run_selected_variant<32, true, false, true>(options) ||
                run_selected_variant<32, true, true, false>(options) ||
                run_selected_variant<32, true, true, true>(options);
            if (!ran) { fail("selected layout did not match a compiled variant"); }
            std::cout << "kv_op_qualification: PASS\n";
            return 0;
        }
        // Each plane independently owns its physical order. Run the complete eight-layout
        // cross-product for both supported V groups; no all-planes-coupled shortcut is admitted.
        run_variant<16, false, false, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<16, false, false, true>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<16, false, true, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<16, false, true, true>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<16, true, false, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<16, true, false, true>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<16, true, true, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<16, true, true, true>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, false, false, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, false, false, true>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, false, true, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, false, true, true>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, true, false, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, true, false, true>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, true, true, false>(options.context, options.iterations, options.rows, options.timing_only);
        run_variant<32, true, true, true>(options.context, options.iterations, options.rows, options.timing_only);
        std::cout << "kv_op_qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv_op_qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
