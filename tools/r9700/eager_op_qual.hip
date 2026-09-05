#include <hip/hip_runtime.h>

#include "ops/r9700/eager/eager_ops.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/l2norm.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/residual_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kHidden = 5120;
constexpr std::uint32_t kVocabPhysical = 248320;
constexpr std::uint32_t kVocabValid = 248077;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) { return; }
    std::ostringstream message;
    message << file << ':' << line << ": " << expression << ": "
            << hipGetErrorString(status);
    fail(message.str());
}

#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { (void)hipFree(data_); }
    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

std::uint16_t bf16_rne_bits(float value) {
    std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t absolute = word & 0x7fffffffU;
    if ((absolute & 0x7f800000U) != 0x7f800000U) {
        word += 0x7fffU + ((word >> 16U) & 1U);
    } else if ((word & 0xffffU) != 0U) {
        word |= 0x10000U;
    }
    return static_cast<std::uint16_t>(word >> 16U);
}

float bf16_to_fp32(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

hip_bfloat16 bf16_from_bits(std::uint16_t bits) {
    hip_bfloat16 result;
    result.data = bits;
    return result;
}

std::uint16_t bf16_bits(hip_bfloat16 value) { return value.data; }

void select_r9700() {
    int devices = 0;
    HIP_CHECK(hipGetDeviceCount(&devices));
    int selected = -1;
    for (int ordinal = 0; ordinal < devices; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (properties.integrated == 0 && std::string_view(properties.gcnArchName) == "gfx1201") {
            if (selected >= 0) { fail("expected one discrete gfx1201 R9700"); }
            selected = ordinal;
        }
    }
    if (selected < 0) { fail("no discrete gfx1201 R9700 found"); }
    HIP_CHECK(hipSetDevice(selected));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, selected));
    if (properties.warpSize != 32) { fail("R9700 eager qualification requires wave32"); }
    std::cout << "device: arch=" << properties.gcnArchName << " wave=" << properties.warpSize
              << " vram_mib=" << properties.totalGlobalMem / (1024U * 1024U) << '\n';
}

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() != source.size()) { fail("host/device extent mismatch"); }
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copy_to_host(const DeviceBuffer<T>& source) {
    std::vector<T> result(source.size());
    HIP_CHECK(hipMemcpy(result.data(), source.get(), result.size() * sizeof(T),
                        hipMemcpyDeviceToHost));
    return result;
}

void require_exact(const std::vector<std::int32_t>& actual,
                   const std::vector<std::int32_t>& expected, const char* label) {
    if (actual != expected) { fail(std::string(label) + " differs from exact I32 oracle"); }
}

void require_bf16_exact(const std::vector<hip_bfloat16>& actual,
                        const std::vector<std::uint16_t>& expected, const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (bf16_bits(actual[index]) != expected[index]) {
            std::ostringstream message;
            message << label << " BF16 mismatch at " << index;
            fail(message.str());
        }
    }
}

void require_fp32_exact(const std::vector<float>& actual, const std::vector<float>& expected,
                        const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(actual[index]) !=
            std::bit_cast<std::uint32_t>(expected[index])) {
            std::ostringstream message;
            message << label << " FP32 mismatch at " << index;
            fail(message.str());
        }
    }
}

void require_bf16_close(const std::vector<hip_bfloat16>& actual,
                        const std::vector<double>& expected, const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double observed = static_cast<double>(static_cast<float>(actual[index]));
        const double absolute = std::abs(observed - expected[index]);
        const double relative = absolute / std::max(1.0e-6, std::abs(expected[index]));
        if (absolute > 2.0e-2 && relative > 2.0e-2) {
            std::ostringstream message;
            message << label << " mismatch at " << index << " observed=" << observed
                    << " expected=" << expected[index];
            fail(message.str());
        }
    }
}

std::vector<hip_bfloat16> bf16_fixture(std::size_t count, std::uint32_t seed) {
    std::vector<hip_bfloat16> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto wave = static_cast<std::int32_t>((index * 37U + seed * 13U) % 509U) - 254;
        result[index] = bf16_from_bits(bf16_rne_bits(static_cast<float>(wave) / 31.0F));
    }
    return result;
}

template <class Function>
void require_invalid_argument(Function&& function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    fail(std::string(label) + " did not reject its invalid public contract");
}

void qualify_i32(hipStream_t stream) {
    using namespace ninfer::ops::r9700::eager;
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        const std::size_t count = static_cast<std::size_t>(rows) * 17U;
        DeviceBuffer<std::int32_t> first(count);
        DeviceBuffer<std::int32_t> second(count);
        DeviceBuffer<std::int32_t> gathered(count);
        DeviceBuffer<std::int32_t> source(257);
        DeviceBuffer<std::int32_t> indices(count);
        std::vector<std::int32_t> expected(count);
        std::vector<std::int32_t> source_host(257);
        std::vector<std::int32_t> indices_host(count);
        for (std::size_t index = 0; index < count; ++index) {
            expected[index] = 4096 + static_cast<std::int32_t>(index);
            indices_host[index] = static_cast<std::int32_t>((index * 29U + rows) % source_host.size());
        }
        for (std::size_t index = 0; index < source_host.size(); ++index) {
            source_host[index] = static_cast<std::int32_t>(index * 11U - 701U);
        }
        copy_to_device(source, source_host);
        copy_to_device(indices, indices_host);
        ninfer::Tensor first_tensor(first.get(), ninfer::DType::I32,
                                    {static_cast<std::int32_t>(count)});
        ninfer::ops::fill_i32_positions(first_tensor, 4096, stream);
        HIP_CHECK(i32_copy(first.get(), second.get(), count, stream));
        HIP_CHECK(i32_offset(second.get(), first.get(), count, -17, stream));
        HIP_CHECK(i32_fill(second.get(), count, 23, stream));
        HIP_CHECK(i32_gather(source.get(), source.size(), indices.get(), gathered.get(), count, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for (std::int32_t& value : expected) { value -= 17; }
        require_exact(copy_to_host(first), expected, "I32 iota/copy/offset");
        require_exact(copy_to_host(second), std::vector<std::int32_t>(count, 23), "I32 fill");
        for (std::size_t index = 0; index < count; ++index) {
            expected[index] = source_host[static_cast<std::size_t>(indices_host[index])];
        }
        require_exact(copy_to_host(gathered), expected, "I32 gather");
    }
    if (i32_iota(nullptr, 1, 0, stream) != hipErrorInvalidValue ||
        i32_iota(reinterpret_cast<std::int32_t*>(1), 1, -1, stream) != hipErrorInvalidValue) {
        fail("I32 position boundary did not reject invalid input");
    }
}

void qualify_bf16_movement_and_cast(hipStream_t stream) {
    using namespace ninfer::ops::r9700::eager;
    constexpr std::uint32_t destination_columns = 16;
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        const std::size_t elements = static_cast<std::size_t>(kHidden) * rows;
        const std::vector<hip_bfloat16> source = bf16_fixture(elements, rows);
        std::vector<hip_bfloat16> destination(static_cast<std::size_t>(kHidden) * destination_columns,
                                              bf16_from_bits(0x7d55U));
        std::vector<std::int32_t> indices(rows);
        for (std::uint32_t row = 0; row < rows; ++row) {
            indices[row] = static_cast<std::int32_t>((rows * 3U + row * 5U) % destination_columns);
        }
        DeviceBuffer<hip_bfloat16> d_source(elements);
        DeviceBuffer<hip_bfloat16> d_destination(destination.size());
        DeviceBuffer<hip_bfloat16> d_gather(elements);
        DeviceBuffer<std::int32_t> d_indices(rows);
        copy_to_device(d_source, source);
        copy_to_device(d_destination, destination);
        copy_to_device(d_indices, indices);
        ninfer::Tensor source_tensor(d_source.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kHidden),
                                      static_cast<std::int32_t>(rows)});
        ninfer::Tensor indices_tensor(d_indices.get(), ninfer::DType::I32,
                                      {static_cast<std::int32_t>(rows)});
        ninfer::Tensor destination_tensor(d_destination.get(), ninfer::DType::BF16,
                                          {static_cast<std::int32_t>(kHidden),
                                           static_cast<std::int32_t>(destination_columns)});
        ninfer::ops::scatter(source_tensor, indices_tensor, destination_tensor, stream);
        HIP_CHECK(bf16_gather_columns(d_destination.get(), d_indices.get(), d_gather.get(), kHidden,
                                      rows, destination_columns, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for (std::uint32_t column = 0; column < rows; ++column) {
            for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
                destination[static_cast<std::size_t>(indices[column]) * kHidden + feature] =
                    source[static_cast<std::size_t>(column) * kHidden + feature];
            }
        }
        std::vector<std::uint16_t> expected_destination(destination.size());
        std::vector<std::uint16_t> expected_source(source.size());
        for (std::size_t index = 0; index < destination.size(); ++index) {
            expected_destination[index] = bf16_bits(destination[index]);
        }
        for (std::size_t index = 0; index < source.size(); ++index) {
            expected_source[index] = bf16_bits(source[index]);
        }
        require_bf16_exact(copy_to_host(d_destination), expected_destination, "BF16 scatter");
        require_bf16_exact(copy_to_host(d_gather), expected_source, "BF16 gather");

        std::vector<float> fp32(elements);
        std::vector<std::uint16_t> cast_expected(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            const std::uint32_t sign = (index & 2U) == 0U ? 0U : 0x8000U;
            const std::uint32_t exponent = 1U + static_cast<std::uint32_t>((index * 29U + rows) % 250U);
            const std::uint32_t mantissa = static_cast<std::uint32_t>((index * 97U + rows * 19U) & 0x7fU);
            const std::uint32_t upper = sign | (exponent << 7U) | mantissa;
            const std::uint32_t lower = (index & 1U) == 0U ? 0x8000U : 0x8001U;
            fp32[index] = std::bit_cast<float>((upper << 16U) | lower);
            cast_expected[index] = bf16_rne_bits(fp32[index]);
        }
        DeviceBuffer<float> d_fp32(elements);
        DeviceBuffer<hip_bfloat16> d_cast(elements);
        DeviceBuffer<float> d_roundtrip(elements);
        copy_to_device(d_fp32, fp32);
        ninfer::Tensor fp32_tensor(d_fp32.get(), ninfer::DType::FP32,
                                   {static_cast<std::int32_t>(kHidden),
                                    static_cast<std::int32_t>(rows)});
        ninfer::Tensor cast_tensor(d_cast.get(), ninfer::DType::BF16,
                                   {static_cast<std::int32_t>(kHidden),
                                    static_cast<std::int32_t>(rows)});
        ninfer::Tensor roundtrip_tensor(d_roundtrip.get(), ninfer::DType::FP32,
                                        {static_cast<std::int32_t>(kHidden),
                                         static_cast<std::int32_t>(rows)});
        ninfer::ops::cast_fp32_to_bf16(fp32_tensor, cast_tensor, stream);
        ninfer::ops::cast_bf16_to_fp32(cast_tensor, roundtrip_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_exact(copy_to_host(d_cast), cast_expected, "FP32-to-BF16 RNE");
        std::vector<float> roundtrip_expected(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            roundtrip_expected[index] = bf16_to_fp32(cast_expected[index]);
        }
        require_fp32_exact(copy_to_host(d_roundtrip), roundtrip_expected, "BF16-to-FP32 exact");

        std::vector<hip_bfloat16> residual = bf16_fixture(elements, rows + 41U);
        const std::vector<hip_bfloat16> addend = bf16_fixture(elements, rows + 83U);
        std::vector<std::uint16_t> residual_expected(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            residual_expected[index] = bf16_rne_bits(static_cast<float>(residual[index]) +
                                                      static_cast<float>(addend[index]));
        }
        DeviceBuffer<hip_bfloat16> d_residual(elements);
        DeviceBuffer<hip_bfloat16> d_addend(elements);
        copy_to_device(d_residual, residual);
        copy_to_device(d_addend, addend);
        ninfer::Tensor addend_tensor(d_addend.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kHidden),
                                      static_cast<std::int32_t>(rows)});
        ninfer::Tensor residual_tensor(d_residual.get(), ninfer::DType::BF16,
                                       {static_cast<std::int32_t>(kHidden),
                                        static_cast<std::int32_t>(rows)});
        ninfer::ops::residual_add(addend_tensor, residual_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_exact(copy_to_host(d_residual), residual_expected, "BF16 residual add");
    }
}

void qualify_mtp_pack_split(hipStream_t stream) {
    using ninfer::DType;
    using ninfer::Tensor;
    constexpr std::uint32_t packed_features = 2U * kHidden;
    constexpr std::uint32_t attn_features = 14336;
    constexpr std::array<std::uint16_t, 8> edge_bits{
        0x0000U, 0x8000U, 0x0001U, 0x7f7fU, 0xff7fU, 0x7f80U, 0xff80U, 0x7fc1U};

    for (std::uint32_t tokens = 1; tokens <= 8; ++tokens) {
        const std::size_t input_elements = static_cast<std::size_t>(kHidden) * tokens;
        std::vector<hip_bfloat16> embedding = bf16_fixture(input_elements, 701U + tokens);
        std::vector<hip_bfloat16> hidden = bf16_fixture(input_elements, 809U + tokens);
        for (std::size_t index = 0; index < edge_bits.size() && index < input_elements; ++index) {
            embedding[index] = bf16_from_bits(edge_bits[index]);
            hidden[index] = bf16_from_bits(edge_bits[edge_bits.size() - 1U - index]);
        }
        DeviceBuffer<hip_bfloat16> d_embedding(input_elements);
        DeviceBuffer<hip_bfloat16> d_hidden(input_elements);
        DeviceBuffer<hip_bfloat16> d_packed(static_cast<std::size_t>(packed_features) * tokens);
        copy_to_device(d_embedding, embedding);
        copy_to_device(d_hidden, hidden);
        Tensor embedding_tensor(d_embedding.get(), DType::BF16,
                                {static_cast<std::int32_t>(kHidden),
                                 static_cast<std::int32_t>(tokens)});
        Tensor hidden_tensor(d_hidden.get(), DType::BF16,
                             {static_cast<std::int32_t>(kHidden),
                              static_cast<std::int32_t>(tokens)});
        Tensor packed_tensor(d_packed.get(), DType::BF16,
                             {static_cast<std::int32_t>(packed_features),
                              static_cast<std::int32_t>(tokens)});
        ninfer::ops::mtp_pack_fc_input(embedding_tensor, hidden_tensor, packed_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<std::uint16_t> packed_expected(
            static_cast<std::size_t>(packed_features) * tokens);
        for (std::uint32_t token = 0; token < tokens; ++token) {
            const std::size_t input_base = static_cast<std::size_t>(token) * kHidden;
            const std::size_t output_base = static_cast<std::size_t>(token) * packed_features;
            for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
                packed_expected[output_base + feature] = bf16_bits(embedding[input_base + feature]);
                packed_expected[output_base + kHidden + feature] =
                    bf16_bits(hidden[input_base + feature]);
            }
        }
        require_bf16_exact(copy_to_host(d_packed), packed_expected,
                           "public MTP FC input pack");

        const std::size_t attn_elements = static_cast<std::size_t>(attn_features) * tokens;
        std::vector<hip_bfloat16> attn_input = bf16_fixture(attn_elements, 907U + tokens);
        for (std::size_t index = 0; index < edge_bits.size(); ++index) {
            const std::array<std::size_t, 4> boundary{
                index, 6144U + index, 7168U + index, 13312U + index};
            for (const std::size_t offset : boundary) {
                attn_input[offset] = bf16_from_bits(edge_bits[index]);
            }
        }
        DeviceBuffer<hip_bfloat16> d_attn_input(attn_elements);
        DeviceBuffer<hip_bfloat16> d_query(6144U * tokens);
        DeviceBuffer<hip_bfloat16> d_key(1024U * tokens);
        DeviceBuffer<hip_bfloat16> d_gate(6144U * tokens);
        DeviceBuffer<hip_bfloat16> d_value(1024U * tokens);
        copy_to_device(d_attn_input, attn_input);
        Tensor attn_tensor(d_attn_input.get(), DType::BF16,
                           {static_cast<std::int32_t>(attn_features),
                            static_cast<std::int32_t>(tokens)});
        Tensor query_tensor(d_query.get(), DType::BF16,
                            {256, 24, static_cast<std::int32_t>(tokens)});
        Tensor key_tensor(d_key.get(), DType::BF16,
                          {256, 4, static_cast<std::int32_t>(tokens)});
        Tensor gate_tensor(d_gate.get(), DType::BF16,
                           {256, 24, static_cast<std::int32_t>(tokens)});
        Tensor value_tensor(d_value.get(), DType::BF16,
                            {256, 4, static_cast<std::int32_t>(tokens)});
        ninfer::ops::mtp_split_attn_in(attn_tensor, query_tensor, key_tensor, gate_tensor,
                                       value_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto expected_slice = [&](std::size_t begin, std::size_t count) {
            std::vector<std::uint16_t> expected(count * tokens);
            for (std::uint32_t token = 0; token < tokens; ++token) {
                const std::size_t source_base = static_cast<std::size_t>(token) * attn_features;
                const std::size_t target_base = static_cast<std::size_t>(token) * count;
                for (std::size_t index = 0; index < count; ++index) {
                    expected[target_base + index] = bf16_bits(attn_input[source_base + begin + index]);
                }
            }
            return expected;
        };
        require_bf16_exact(copy_to_host(d_query), expected_slice(0, 6144),
                           "public MTP attention Q split");
        require_bf16_exact(copy_to_host(d_key), expected_slice(6144, 1024),
                           "public MTP attention K split");
        require_bf16_exact(copy_to_host(d_gate), expected_slice(7168, 6144),
                           "public MTP attention gate split");
        require_bf16_exact(copy_to_host(d_value), expected_slice(13312, 1024),
                           "public MTP attention V split");
    }

    DeviceBuffer<hip_bfloat16> shared_pack(4U * kHidden);
    DeviceBuffer<hip_bfloat16> pack_output(2U * kHidden);
    Tensor embedding(shared_pack.get(), DType::BF16, {static_cast<std::int32_t>(kHidden), 1});
    Tensor hidden(shared_pack.get() + kHidden, DType::BF16,
                  {static_cast<std::int32_t>(kHidden), 1});
    Tensor output(pack_output.get(), DType::BF16,
                  {static_cast<std::int32_t>(packed_features), 1});
    Tensor wrong_embedding(shared_pack.get(), DType::BF16,
                           {static_cast<std::int32_t>(kHidden - 1U), 1});
    require_invalid_argument(
        [&] { ninfer::ops::mtp_pack_fc_input(wrong_embedding, hidden, output, stream); },
        "MTP pack shape");
    Tensor hidden_overlap(shared_pack.get() + 1, DType::BF16,
                          {static_cast<std::int32_t>(kHidden), 1});
    require_invalid_argument(
        [&] { ninfer::ops::mtp_pack_fc_input(embedding, hidden_overlap, output, stream); },
        "MTP pack input alias");
    Tensor output_overlap(shared_pack.get() + 2U * kHidden - 1U, DType::BF16,
                          {static_cast<std::int32_t>(packed_features), 1});
    require_invalid_argument(
        [&] { ninfer::ops::mtp_pack_fc_input(embedding, hidden, output_overlap, stream); },
        "MTP pack output alias");

    DeviceBuffer<hip_bfloat16> split_input(attn_features);
    DeviceBuffer<hip_bfloat16> split_query(6144);
    DeviceBuffer<hip_bfloat16> split_key(1024);
    DeviceBuffer<hip_bfloat16> split_gate(6144);
    DeviceBuffer<hip_bfloat16> split_value(1024);
    Tensor attn(split_input.get(), DType::BF16, {static_cast<std::int32_t>(attn_features), 1});
    Tensor query(split_query.get(), DType::BF16, {256, 24, 1});
    Tensor key(split_key.get(), DType::BF16, {256, 4, 1});
    Tensor gate(split_gate.get(), DType::BF16, {256, 24, 1});
    Tensor value(split_value.get(), DType::BF16, {256, 4, 1});
    Tensor flattened_query(split_query.get(), DType::BF16, {6144, 1});
    require_invalid_argument(
        [&] { ninfer::ops::mtp_split_attn_in(attn, flattened_query, key, gate, value, stream); },
        "MTP split output shape");
    Tensor query_input_overlap(split_input.get() + 1, DType::BF16, {256, 24, 1});
    require_invalid_argument(
        [&] { ninfer::ops::mtp_split_attn_in(attn, query_input_overlap, key, gate, value, stream); },
        "MTP split input alias");
    Tensor key_query_overlap(split_query.get() + 512, DType::BF16, {256, 4, 1});
    require_invalid_argument(
        [&] { ninfer::ops::mtp_split_attn_in(attn, query, key_query_overlap, gate, value, stream); },
        "MTP split output alias");
}

void qualify_argmax(hipStream_t stream) {
    using namespace ninfer::ops::r9700::eager;
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        const std::size_t count = static_cast<std::size_t>(kVocabPhysical) * rows;
        std::vector<hip_bfloat16> logits(count);
        std::vector<std::int32_t> expected(rows);
        for (std::uint32_t column = 0; column < rows; ++column) {
            const std::size_t base = static_cast<std::size_t>(column) * kVocabPhysical;
            std::int32_t best_index = 0;
            float best = -std::numeric_limits<float>::infinity();
            for (std::uint32_t row = 0; row < kVocabPhysical; ++row) {
                float value = static_cast<float>((row * 17U + column * 31U) % 113U) / 16.0F - 3.0F;
                if (row == 11U || row == 57U) { value = 31.0F + static_cast<float>(column); }
                logits[base + row] = bf16_from_bits(bf16_rne_bits(value));
                if (row < kVocabValid) {
                    const float represented = static_cast<float>(logits[base + row]);
                    if (represented > best || (represented == best &&
                                               static_cast<std::int32_t>(row) < best_index)) {
                        best = represented;
                        best_index = static_cast<std::int32_t>(row);
                    }
                }
            }
            expected[column] = best_index;
        }
        DeviceBuffer<hip_bfloat16> d_logits(count);
        DeviceBuffer<std::int32_t> d_output(rows);
        copy_to_device(d_logits, logits);
        ninfer::Tensor logits_tensor(d_logits.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kVocabPhysical),
                                      static_cast<std::int32_t>(rows)});
        ninfer::Tensor output_tensor(d_output.get(), ninfer::DType::I32,
                                     {static_cast<std::int32_t>(rows)});
        ninfer::ops::argmax(logits_tensor, output_tensor, static_cast<std::int32_t>(kVocabValid),
                            stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_exact(copy_to_host(d_output), expected, "BF16 deterministic argmax");
    }
    if (argmax_bf16(nullptr, reinterpret_cast<std::int32_t*>(1), 1, 1, 1, stream) != hipErrorInvalidValue ||
        argmax_bf16(reinterpret_cast<const hip_bfloat16*>(1), reinterpret_cast<std::int32_t*>(1),
                    8, 9, 1, stream) != hipErrorInvalidValue) {
        fail("argmax boundary did not reject malformed geometry");
    }
}

void qualify_normalization_and_activations(hipStream_t stream) {
    using namespace ninfer::ops::r9700::eager;
    constexpr float eps = 1.0e-5F;
    const std::vector<hip_bfloat16> weight = bf16_fixture(kHidden, 97U);
    const std::vector<hip_bfloat16> bias = bf16_fixture(kHidden, 101U);
    DeviceBuffer<hip_bfloat16> d_weight(kHidden);
    DeviceBuffer<hip_bfloat16> d_bias(kHidden);
    copy_to_device(d_weight, weight);
    copy_to_device(d_bias, bias);
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        const std::size_t elements = static_cast<std::size_t>(kHidden) * rows;
        const std::vector<hip_bfloat16> input = bf16_fixture(elements, rows + 131U);
        const std::vector<hip_bfloat16> up = bf16_fixture(elements, rows + 149U);
        const std::vector<hip_bfloat16> gate = bf16_fixture(elements, rows + 167U);
        DeviceBuffer<hip_bfloat16> d_input(elements);
        DeviceBuffer<hip_bfloat16> d_up(elements);
        DeviceBuffer<hip_bfloat16> d_gate(elements);
        DeviceBuffer<hip_bfloat16> d_output(elements);
        DeviceBuffer<hip_bfloat16> d_mutable(elements);
        copy_to_device(d_input, input);
        copy_to_device(d_up, up);
        copy_to_device(d_gate, gate);
        ninfer::Tensor input_tensor(d_input.get(), ninfer::DType::BF16,
                                    {static_cast<std::int32_t>(kHidden),
                                     static_cast<std::int32_t>(rows)});
        ninfer::Tensor up_tensor(d_up.get(), ninfer::DType::BF16,
                                 {static_cast<std::int32_t>(kHidden),
                                  static_cast<std::int32_t>(rows)});
        ninfer::Tensor gate_tensor(d_gate.get(), ninfer::DType::BF16,
                                   {static_cast<std::int32_t>(kHidden),
                                    static_cast<std::int32_t>(rows)});
        ninfer::Tensor output_tensor(d_output.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kHidden),
                                      static_cast<std::int32_t>(rows)});
        ninfer::Tensor mutable_tensor(d_mutable.get(), ninfer::DType::BF16,
                                      {static_cast<std::int32_t>(kHidden),
                                       static_cast<std::int32_t>(rows)});
        ninfer::Tensor weight_tensor(d_weight.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kHidden)});
        ninfer::Tensor bias_tensor(d_bias.get(), ninfer::DType::BF16,
                                   {static_cast<std::int32_t>(kHidden)});

        std::vector<double> rms_expected(elements);
        std::vector<double> l2_expected(elements);
        std::vector<double> layer_expected(elements);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * kHidden;
            double sum = 0.0;
            double sumsq = 0.0;
            for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
                const double value = static_cast<float>(input[base + feature]);
                sum += value;
                sumsq += value * value;
            }
            const double rms_inverse = 1.0 / std::sqrt(sumsq / kHidden + eps);
            const double l2_inverse = 1.0 / std::sqrt(sumsq + eps);
            const double mean = sum / kHidden;
            double variance = 0.0;
            for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
                const double centered = static_cast<float>(input[base + feature]) - mean;
                variance += centered * centered;
            }
            const double layer_inverse = 1.0 / std::sqrt(variance / kHidden + eps);
            for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
                const double value = static_cast<float>(input[base + feature]);
                rms_expected[base + feature] = value * rms_inverse *
                    (1.0 + static_cast<float>(weight[feature]));
                l2_expected[base + feature] = value * l2_inverse;
                layer_expected[base + feature] = (value - mean) * layer_inverse *
                    static_cast<float>(weight[feature]) + static_cast<float>(bias[feature]);
            }
        }
        ninfer::ops::rmsnorm(input_tensor, weight_tensor, eps, true, output_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_output), rms_expected, "RMSNorm FP64 oracle");
        ninfer::ops::l2norm(input_tensor, eps, output_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_output), l2_expected, "L2Norm FP64 oracle");
        ninfer::ops::layer_norm(input_tensor, weight_tensor, bias_tensor, eps, output_tensor,
                                stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_output), layer_expected, "LayerNorm FP64 oracle");

        std::vector<double> silu_expected(elements);
        std::vector<double> sigmoid_expected(elements);
        std::vector<double> gelu_exact_expected(elements);
        std::vector<double> gelu_tanh_expected(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            const double g = static_cast<float>(gate[index]);
            const double x = static_cast<float>(up[index]);
            silu_expected[index] = (g / (1.0 + std::exp(-g))) * x;
            sigmoid_expected[index] = x / (1.0 + std::exp(-g));
            gelu_exact_expected[index] = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
            gelu_tanh_expected[index] = 0.5 * x * (1.0 + std::tanh(
                std::sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x)));
        }
        ninfer::ops::silu_mul(gate_tensor, up_tensor, output_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_output), silu_expected, "SiLU multiply FP64 oracle");

        // The Qwen MLP projection is physically [2*features, rows]; its gate/up slices retain
        // the parent row stride and are therefore non-contiguous for rows > 1. Qualify that
        // production view directly against the same independent formula.
        std::vector<hip_bfloat16> packed(elements * 2U);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const std::size_t source = static_cast<std::size_t>(row) * kHidden;
            const std::size_t destination = static_cast<std::size_t>(row) * 2U * kHidden;
            std::copy_n(gate.begin() + static_cast<std::ptrdiff_t>(source), kHidden,
                        packed.begin() + static_cast<std::ptrdiff_t>(destination));
            std::copy_n(up.begin() + static_cast<std::ptrdiff_t>(source), kHidden,
                        packed.begin() + static_cast<std::ptrdiff_t>(destination + kHidden));
        }
        DeviceBuffer<hip_bfloat16> d_packed(packed.size());
        copy_to_device(d_packed, packed);
        ninfer::Tensor packed_tensor(d_packed.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(2U * kHidden),
                                      static_cast<std::int32_t>(rows)});
        const ninfer::Tensor strided_gate = packed_tensor.slice(
            0, 0, static_cast<std::int32_t>(kHidden));
        const ninfer::Tensor strided_up = packed_tensor.slice(
            0, static_cast<std::int32_t>(kHidden), static_cast<std::int32_t>(kHidden));
        ninfer::ops::silu_mul(strided_gate, strided_up, output_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_output), silu_expected,
                           "strided SiLU multiply FP64 oracle");
        ninfer::Tensor overlapping_output(d_packed.get(), ninfer::DType::BF16,
                                          {static_cast<std::int32_t>(kHidden),
                                           static_cast<std::int32_t>(rows)});
        require_invalid_argument(
            [&] { ninfer::ops::silu_mul(strided_gate, strided_up, overlapping_output, stream); },
            "strided SiLU output/input overlap");
        copy_to_device(d_mutable, up);
        ninfer::ops::sigmoid_mul(gate_tensor, mutable_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_mutable), sigmoid_expected, "sigmoid multiply FP64 oracle");
        copy_to_device(d_mutable, up);
        ninfer::ops::gelu(mutable_tensor, ninfer::ops::GeluMode::Exact, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_mutable), gelu_exact_expected, "exact GELU FP64 oracle");
        copy_to_device(d_mutable, up);
        ninfer::ops::gelu(mutable_tensor, ninfer::ops::GeluMode::Tanh, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_mutable), gelu_tanh_expected, "tanh GELU FP64 oracle");
    }
    if (rmsnorm_bf16(nullptr, d_weight.get(), d_weight.get(), kHidden, 1, eps, false, stream) !=
            hipErrorInvalidValue ||
        l2norm_bf16(d_weight.get(), d_weight.get(), kHidden, 1,
                     std::numeric_limits<float>::infinity(), stream) != hipErrorInvalidValue ||
        gelu_bf16(d_weight.get(), 1, static_cast<GeluMode>(7), stream) != hipErrorInvalidValue) {
        fail("normalization or activation boundary did not reject malformed input");
    }
    SiluMulStridedShape mismatched_shape{};
    mismatched_shape.extent[0] = 2;
    SiluMulStridedShape overflowing_shape{};
    overflowing_shape.extent[0] = 2;
    overflowing_shape.gate_stride[0] = std::numeric_limits<std::uint64_t>::max();
    if (silu_mul_bf16_strided(d_weight.get(), d_weight.get(), d_weight.get(), 1,
                              mismatched_shape, stream) != hipErrorInvalidValue ||
        silu_mul_bf16_strided(d_weight.get(), d_weight.get(), d_weight.get(), 2,
                              overflowing_shape, stream) != hipErrorInvalidValue) {
        fail("strided SiLU boundary did not reject malformed geometry");
    }
}

void qualify_embedding(hipStream_t stream) {
    using namespace ninfer::ops::r9700::eager;
    // Allocate the actual Qwen dense BF16 table extent on device but populate only the rows that
    // this independent gather corpus addresses. This verifies high vocabulary coordinates without
    // mirroring a multi-gigabyte model table in host RAM.
    const std::size_t table_elements = static_cast<std::size_t>(kVocabPhysical) * kHidden;
    DeviceBuffer<hip_bfloat16> d_table(table_elements);
    const std::array<std::int32_t, 8> ids{0, 17, 513, 4096, 65535, 131071, 248076, 248319};
    std::array<std::vector<hip_bfloat16>, ids.size()> expected_rows;
    for (std::size_t row = 0; row < ids.size(); ++row) {
        expected_rows[row] = bf16_fixture(kHidden, static_cast<std::uint32_t>(ids[row]));
        HIP_CHECK(hipMemcpyAsync(d_table.get() + static_cast<std::size_t>(ids[row]) * kHidden,
                                 expected_rows[row].data(), expected_rows[row].size() * sizeof(hip_bfloat16),
                                 hipMemcpyHostToDevice, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        DeviceBuffer<std::int32_t> d_ids(rows);
        DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(kHidden) * rows);
        std::vector<std::int32_t> ids_host(ids.begin(), ids.begin() + rows);
        copy_to_device(d_ids, ids_host);
        ninfer::Tensor ids_tensor(d_ids.get(), ninfer::DType::I32,
                                  {static_cast<std::int32_t>(rows)});
        ninfer::Tensor output_tensor(d_output.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kHidden),
                                      static_cast<std::int32_t>(rows)});
        ninfer::Weight table{};
        table.qtype = ninfer::QType::BF16_CTRL;
        table.layout = ninfer::QuantLayout::Contiguous;
        table.ndim = 2;
        table.shape[0] = static_cast<std::int32_t>(kVocabPhysical);
        table.shape[1] = static_cast<std::int32_t>(kHidden);
        table.padded_shape[0] = table.shape[0];
        table.padded_shape[1] = table.shape[1];
        table.qdata = d_table.get();
        table.payload_bytes = table_elements * sizeof(hip_bfloat16);
        ninfer::ops::embedding(ids_tensor, table, output_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<std::uint16_t> expected(static_cast<std::size_t>(kHidden) * rows);
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
                expected[static_cast<std::size_t>(row) * kHidden + feature] =
                    bf16_bits(expected_rows[row][feature]);
            }
        }
        require_bf16_exact(copy_to_host(d_output), expected, "dense BF16 embedding gather");
    }
}

void qualify_w8_embedding_product_shape(hipStream_t stream) {
    using ninfer::DType;
    using ninfer::QType;
    using ninfer::QuantLayout;
    using ninfer::Tensor;
    using ninfer::Weight;

    constexpr std::uint32_t group = 32;
    constexpr std::uint32_t padded = kHidden;
    constexpr std::uint32_t groups = padded / group;
    constexpr std::array<std::int32_t, 8> ids{
        0, 17, 513, 4096, 65535, 131071, 248076, 248319};
    const std::size_t code_elements = static_cast<std::size_t>(kVocabPhysical) * padded;
    const std::size_t scale_elements = static_cast<std::size_t>(kVocabPhysical) * groups;
    const std::size_t scale_offset = (code_elements + 255U) / 256U * 256U;
    const std::size_t required_payload_bytes = scale_offset + scale_elements * sizeof(std::uint16_t);

    // Own the complete product extents on device, but populate only the independently checked
    // high/last rows. This proves the 64-bit row arithmetic without mirroring 1.3 GiB on the host.
    DeviceBuffer<std::uint8_t> d_codes(code_elements);
    DeviceBuffer<std::uint16_t> d_scales(scale_elements);
    DeviceBuffer<std::int32_t> d_ids(ids.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(kHidden) * ids.size());
    HIP_CHECK(hipMemsetAsync(d_codes.get(), 0, code_elements, stream));
    HIP_CHECK(hipMemsetAsync(d_scales.get(), 0, scale_elements * sizeof(std::uint16_t), stream));
    HIP_CHECK(hipMemcpyAsync(d_ids.get(), ids.data(), ids.size() * sizeof(ids.front()),
                             hipMemcpyHostToDevice, stream));

    std::vector<std::uint16_t> expected(static_cast<std::size_t>(kHidden) * ids.size());
    for (std::size_t selected = 0; selected < ids.size(); ++selected) {
        const std::uint32_t row = static_cast<std::uint32_t>(ids[selected]);
        std::vector<std::uint8_t> row_codes(padded);
        std::vector<std::uint16_t> row_scales(groups);
        for (std::uint32_t scale_group = 0; scale_group < groups; ++scale_group) {
            const float scale_value = static_cast<float>(1U + (row + 3U * scale_group) % 7U) /
                                      128.0F;
            row_scales[scale_group] =
                std::bit_cast<std::uint16_t>(static_cast<_Float16>(scale_value));
        }
        for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
            const auto code = static_cast<std::int8_t>(
                static_cast<std::int32_t>((row * 11U + feature * 17U) % 255U) - 127);
            row_codes[feature] = static_cast<std::uint8_t>(code);
            const float scale = static_cast<float>(
                std::bit_cast<_Float16>(row_scales[feature / group]));
            expected[selected * kHidden + feature] =
                bf16_rne_bits(static_cast<float>(code) * scale);
        }
        HIP_CHECK(hipMemcpyAsync(d_codes.get() + static_cast<std::size_t>(row) * padded,
                                 row_codes.data(), row_codes.size(), hipMemcpyHostToDevice,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(d_scales.get() + static_cast<std::size_t>(row) * groups,
                                 row_scales.data(), row_scales.size() * sizeof(std::uint16_t),
                                 hipMemcpyHostToDevice, stream));
    }

    Tensor ids_tensor(d_ids.get(), DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor output_tensor(d_output.get(), DType::BF16,
                         {static_cast<std::int32_t>(kHidden),
                          static_cast<std::int32_t>(ids.size())});
    Weight table{};
    table.payload = d_codes.get();
    table.payload_bytes = required_payload_bytes;
    table.qdata = d_codes.get();
    table.qdata_bytes = code_elements;
    table.scales = d_scales.get();
    table.scale_bytes = scale_elements * sizeof(std::uint16_t);
    table.qtype = QType::W8G32_F16S;
    table.group_size = table.group = group;
    table.layout = QuantLayout::RowSplit;
    table.scale_dtype = DType::FP16;
    table.ndim = 2;
    table.n = table.shape[0] = table.padded_shape[0] =
        static_cast<std::int32_t>(kVocabPhysical);
    table.k = table.shape[1] = table.padded_shape[1] =
        static_cast<std::int32_t>(kHidden);

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    ninfer::ops::embedding(ids_tensor, table, output_tensor, stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    require_bf16_exact(copy_to_host(d_output), expected, "product W8 embedding");

    Weight malformed = table;
    --malformed.payload_bytes;
    try {
        ninfer::ops::embedding(ids_tensor, malformed, output_tensor, stream);
        fail("product W8 embedding accepted a short payload");
    } catch (const std::invalid_argument&) {
    }
    malformed = table;
    malformed.padded_shape[1] += 128;
    try {
        ninfer::ops::embedding(ids_tensor, malformed, output_tensor, stream);
        fail("product W8 embedding accepted incorrect K padding");
    } catch (const std::invalid_argument&) {
    }

    const double resident_mib = static_cast<double>(
        code_elements + scale_elements * sizeof(std::uint16_t) +
        ids.size() * sizeof(std::int32_t) +
        static_cast<std::size_t>(kHidden) * ids.size() * sizeof(hip_bfloat16)) /
        static_cast<double>(1U << 20U);
    std::cout << "w8-embedding-product: shape=[" << kVocabPhysical << ',' << kHidden
              << "] ids=0/17/513/4096/65535/131071/248076/248319"
              << " exact_code_times_fp16_scale=PASS short_payload=REJECT padding=REJECT"
              << " resident_mib=" << resident_mib << " kernel_ms=" << milliseconds << '\n';
}

void qualify_public_extended_paths(hipStream_t stream) {
    using ninfer::DType;
    using ninfer::QType;
    using ninfer::QuantLayout;
    using ninfer::Tensor;
    using ninfer::Weight;

    // Scalar state and the device-scalar position offset stay ordered on the caller's stream.
    DeviceBuffer<std::int32_t> d_i32(4);
    DeviceBuffer<std::int64_t> d_i64(1);
    Tensor lhs(d_i32.get(), DType::I32, {1});
    Tensor rhs(d_i32.get() + 1, DType::I32, {1});
    Tensor sum(d_i32.get() + 2, DType::I32, {1});
    Tensor assigned(d_i32.get() + 3, DType::I32, {1});
    Tensor i64(d_i64.get(), DType::I64, {1});
    ninfer::ops::set_i32_scalar(lhs, 19, stream);
    ninfer::ops::set_i32_scalar(rhs, 23, stream);
    ninfer::ops::add_i32_scalars(lhs, rhs, sum, stream);
    ninfer::ops::assign_i32_scalar(sum, assigned, stream);
    ninfer::ops::increment_i32_scalar(assigned, stream);
    const std::vector<std::int64_t> i64_initial{9223372036854775000LL};
    copy_to_device(d_i64, i64_initial);
    ninfer::ops::increment_i64_scalar(i64, stream);

    DeviceBuffer<std::int32_t> d_positions(9);
    DeviceBuffer<std::int32_t> d_shifted(9);
    Tensor positions(d_positions.get(), DType::I32, {9});
    Tensor shifted(d_shifted.get(), DType::I32, {9});
    ninfer::ops::fill_i32_positions(positions, 500, stream);
    ninfer::ops::offset_i32_positions(positions, rhs, shifted, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    require_exact(copy_to_host(d_i32), {19, 23, 42, 43}, "public scalar transitions");
    if (copy_to_host(d_i64)[0] != i64_initial[0] + 1) {
        fail("public I64 increment differs from exact oracle");
    }
    std::vector<std::int32_t> shifted_expected(9);
    for (std::size_t index = 0; index < shifted_expected.size(); ++index) {
        shifted_expected[index] = 523 + static_cast<std::int32_t>(index);
    }
    require_exact(copy_to_host(d_shifted), shifted_expected, "public position offset");

    // Structured request/lane movement and feature extraction use exact BF16 bit oracles.
    constexpr std::uint32_t features = 64;
    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t batch = 2;
    constexpr std::uint32_t lanes_count = 3;
    const auto source_host = bf16_fixture(features * width * batch, 211U);
    std::vector<hip_bfloat16> lane_host(features * width * lanes_count,
                                        bf16_from_bits(0x5a5aU));
    const std::vector<std::int32_t> lanes{2, 0};
    const std::vector<std::int32_t> valid{3, 2};
    DeviceBuffer<hip_bfloat16> d_source(source_host.size());
    DeviceBuffer<hip_bfloat16> d_lanes(lane_host.size());
    DeviceBuffer<std::int32_t> d_lane_ids(lanes.size());
    DeviceBuffer<std::int32_t> d_valid(valid.size());
    copy_to_device(d_source, source_host);
    copy_to_device(d_lanes, lane_host);
    copy_to_device(d_lane_ids, lanes);
    copy_to_device(d_valid, valid);
    Tensor source(d_source.get(), DType::BF16,
                  {static_cast<std::int32_t>(features), static_cast<std::int32_t>(width),
                   static_cast<std::int32_t>(batch)});
    Tensor lane_storage(d_lanes.get(), DType::BF16,
                        {static_cast<std::int32_t>(features), static_cast<std::int32_t>(width),
                         static_cast<std::int32_t>(lanes_count)});
    Tensor lane_ids(d_lane_ids.get(), DType::I32, {static_cast<std::int32_t>(batch)});
    Tensor valid_columns(d_valid.get(), DType::I32, {static_cast<std::int32_t>(batch)});
    ninfer::ops::scatter_bf16_batch(source, lane_ids, valid_columns, lane_storage, stream);
    for (std::uint32_t request = 0; request < batch; ++request) {
        for (std::uint32_t column = 0; column < static_cast<std::uint32_t>(valid[request]); ++column) {
            for (std::uint32_t feature = 0; feature < features; ++feature) {
                lane_host[(static_cast<std::size_t>(lanes[request]) * width + column) * features +
                          feature] =
                    source_host[(static_cast<std::size_t>(request) * width + column) * features +
                                feature];
            }
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<std::uint16_t> lane_expected(lane_host.size());
    std::transform(lane_host.begin(), lane_host.end(), lane_expected.begin(), bf16_bits);
    require_bf16_exact(copy_to_host(d_lanes), lane_expected, "public batch scatter");

    const std::vector<std::int32_t> paths{1, 3, 3, 3, 0, 2, 3, 3};
    const std::vector<std::int32_t> counts{2, 3};
    copy_to_device(d_lane_ids, lanes);
    DeviceBuffer<std::int32_t> d_paths(paths.size());
    DeviceBuffer<std::int32_t> d_counts(counts.size());
    copy_to_device(d_paths, paths);
    copy_to_device(d_counts, counts);
    Tensor path(d_paths.get(), DType::I32,
                {static_cast<std::int32_t>(width), static_cast<std::int32_t>(batch)});
    Tensor count_tensor(d_counts.get(), DType::I32, {static_cast<std::int32_t>(batch)});
    ninfer::ops::gather_bf16_path(lane_storage, lane_ids, path, count_tensor, stream);
    for (std::uint32_t request = 0; request < batch; ++request) {
        for (std::uint32_t column = 0; column < static_cast<std::uint32_t>(counts[request]); ++column) {
            const std::uint32_t source_column = static_cast<std::uint32_t>(
                paths[static_cast<std::size_t>(request) * width + column]);
            if (source_column == column) { continue; }
            for (std::uint32_t feature = 0; feature < features; ++feature) {
                lane_host[(static_cast<std::size_t>(lanes[request]) * width + column) * features +
                          feature] =
                    lane_host[(static_cast<std::size_t>(lanes[request]) * width + source_column) *
                                  features + feature];
            }
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::transform(lane_host.begin(), lane_host.end(), lane_expected.begin(), bf16_bits);
    require_bf16_exact(copy_to_host(d_lanes), lane_expected, "public path gather");

    constexpr std::uint32_t source_features = 73;
    constexpr std::uint32_t extracted_features = 17;
    constexpr std::uint32_t extract_columns = 3;
    constexpr std::uint32_t extract_start = 11;
    const auto extract_source = bf16_fixture(source_features * extract_columns, 223U);
    DeviceBuffer<hip_bfloat16> d_extract_source(extract_source.size());
    DeviceBuffer<hip_bfloat16> d_extract(extracted_features * extract_columns);
    copy_to_device(d_extract_source, extract_source);
    Tensor extract_source_tensor(d_extract_source.get(), DType::BF16,
                                 {static_cast<std::int32_t>(source_features),
                                  static_cast<std::int32_t>(extract_columns)});
    Tensor extract_tensor(d_extract.get(), DType::BF16,
                          {static_cast<std::int32_t>(extracted_features),
                           static_cast<std::int32_t>(extract_columns)});
    ninfer::ops::extract_bf16_columns(extract_source_tensor, extract_start, extract_tensor, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<std::uint16_t> extract_expected(extracted_features * extract_columns);
    for (std::uint32_t column = 0; column < extract_columns; ++column) {
        for (std::uint32_t feature = 0; feature < extracted_features; ++feature) {
            extract_expected[static_cast<std::size_t>(column) * extracted_features + feature] =
                bf16_bits(extract_source[static_cast<std::size_t>(column) * source_features +
                                         extract_start + feature]);
        }
    }
    require_bf16_exact(copy_to_host(d_extract), extract_expected, "public feature extraction");

    // Fused and gated norms are checked against the represented-input FP64 formula. L2 dump is
    // additionally checked against the production FP32 accumulation boundary it exposes.
    constexpr std::uint32_t norm_features = 64;
    constexpr std::uint32_t norm_rows = 3;
    constexpr float eps = 1.0e-5F;
    const auto residual_input = bf16_fixture(norm_features * norm_rows, 227U);
    const auto residual_delta = bf16_fixture(norm_features * norm_rows, 229U);
    const auto norm_weight = bf16_fixture(norm_features, 233U);
    const auto norm_gate = bf16_fixture(norm_features * norm_rows, 239U);
    DeviceBuffer<hip_bfloat16> d_residual_input(residual_input.size());
    DeviceBuffer<hip_bfloat16> d_residual_delta(residual_delta.size());
    DeviceBuffer<hip_bfloat16> d_norm_weight(norm_weight.size());
    DeviceBuffer<hip_bfloat16> d_norm_gate(norm_gate.size());
    DeviceBuffer<hip_bfloat16> d_norm_output(residual_input.size());
    DeviceBuffer<hip_bfloat16> d_pair_input(residual_input.size());
    DeviceBuffer<hip_bfloat16> d_pair_output(residual_input.size());
    copy_to_device(d_residual_input, residual_input);
    copy_to_device(d_residual_delta, residual_delta);
    copy_to_device(d_norm_weight, norm_weight);
    copy_to_device(d_norm_gate, norm_gate);
    copy_to_device(d_pair_input, residual_input);
    Tensor residual_input_tensor(d_residual_input.get(), DType::BF16,
                                 {static_cast<std::int32_t>(norm_features),
                                  static_cast<std::int32_t>(norm_rows)});
    Tensor residual_delta_tensor(d_residual_delta.get(), DType::BF16,
                                 {static_cast<std::int32_t>(norm_features),
                                  static_cast<std::int32_t>(norm_rows)});
    Tensor norm_weight_tensor(d_norm_weight.get(), DType::BF16,
                              {static_cast<std::int32_t>(norm_features)});
    Tensor norm_gate_tensor(d_norm_gate.get(), DType::BF16,
                            {static_cast<std::int32_t>(norm_features),
                             static_cast<std::int32_t>(norm_rows)});
    Tensor norm_output_tensor(d_norm_output.get(), DType::BF16,
                              {static_cast<std::int32_t>(norm_features),
                               static_cast<std::int32_t>(norm_rows)});
    Tensor pair_input_tensor(d_pair_input.get(), DType::BF16,
                             {static_cast<std::int32_t>(norm_features),
                              static_cast<std::int32_t>(norm_rows)});
    Tensor pair_output_tensor(d_pair_output.get(), DType::BF16,
                              {static_cast<std::int32_t>(norm_features),
                               static_cast<std::int32_t>(norm_rows)});
    std::vector<std::uint16_t> residual_exact(residual_input.size());
    std::vector<double> residual_norm_expected(residual_input.size());
    for (std::uint32_t row = 0; row < norm_rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * norm_features;
        double sumsq = 0.0;
        for (std::uint32_t feature = 0; feature < norm_features; ++feature) {
            residual_exact[base + feature] = bf16_rne_bits(
                static_cast<float>(residual_input[base + feature]) +
                static_cast<float>(residual_delta[base + feature]));
            const double value = bf16_to_fp32(residual_exact[base + feature]);
            sumsq += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sumsq / norm_features + eps);
        for (std::uint32_t feature = 0; feature < norm_features; ++feature) {
            residual_norm_expected[base + feature] =
                bf16_to_fp32(residual_exact[base + feature]) * inverse *
                (1.0 + static_cast<float>(norm_weight[feature]));
        }
    }
    ninfer::ops::residual_rmsnorm(residual_delta_tensor, residual_input_tensor,
                                  norm_weight_tensor, eps, norm_output_tensor, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    require_bf16_exact(copy_to_host(d_residual_input), residual_exact,
                       "public fused residual publication");
    const auto fused_output = copy_to_host(d_norm_output);
    require_bf16_close(fused_output, residual_norm_expected, "public fused residual RMSNorm");
    ninfer::ops::residual_add(residual_delta_tensor, pair_input_tensor, stream);
    ninfer::ops::rmsnorm(pair_input_tensor, norm_weight_tensor, eps, true, pair_output_tensor,
                         stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    require_bf16_exact(copy_to_host(d_pair_input), residual_exact,
                       "unfused residual publication");
    std::vector<std::uint16_t> fused_bits(residual_input.size());
    for (std::size_t index = 0; index < fused_bits.size(); ++index) {
        fused_bits[index] = bf16_bits(fused_output[index]);
    }
    require_bf16_exact(copy_to_host(d_pair_output), fused_bits,
                       "fused/unfused residual RMSNorm identity");

    copy_to_device(d_residual_input, residual_input);
    std::vector<double> gated_expected(residual_input.size());
    for (std::uint32_t row = 0; row < norm_rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * norm_features;
        double sumsq = 0.0;
        for (std::uint32_t feature = 0; feature < norm_features; ++feature) {
            const double value = static_cast<float>(residual_input[base + feature]);
            sumsq += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sumsq / norm_features + eps);
        for (std::uint32_t feature = 0; feature < norm_features; ++feature) {
            const double z = static_cast<float>(norm_gate[base + feature]);
            gated_expected[base + feature] =
                static_cast<float>(residual_input[base + feature]) * inverse *
                static_cast<float>(norm_weight[feature]) * z / (1.0 + std::exp(-z));
        }
    }
    ninfer::ops::gated_rmsnorm(residual_input_tensor, norm_weight_tensor, norm_gate_tensor, eps,
                               norm_output_tensor, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    require_bf16_close(copy_to_host(d_norm_output), gated_expected, "public gated RMSNorm");

    DeviceBuffer<float> d_sumsq(norm_rows);
    DeviceBuffer<float> d_inverse(norm_rows);
    ninfer::ops::L2NormDump dump{d_sumsq.get(), d_inverse.get()};
    ninfer::ops::l2norm_dump(residual_input_tensor, eps, norm_output_tensor, stream, dump);
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto sums = copy_to_host(d_sumsq);
    const auto inverses = copy_to_host(d_inverse);
    for (std::uint32_t row = 0; row < norm_rows; ++row) {
        float sumsq = 0.0F;
        for (std::uint32_t feature = 0; feature < norm_features; ++feature) {
            const float value = static_cast<float>(residual_input[
                static_cast<std::size_t>(row) * norm_features + feature]);
            sumsq = std::fma(value, value, sumsq);
        }
        if (std::bit_cast<std::uint32_t>(sums[row]) != std::bit_cast<std::uint32_t>(sumsq) ||
            std::abs(inverses[row] - 1.0F / std::sqrt(sumsq + eps)) > 2.0e-6F) {
            fail("public L2Norm dump differs from FP32 intermediate oracle");
        }
    }

    // Every registered integer embedding codec is decoded independently from signed codes and
    // represented FP16 scales.
    constexpr std::uint32_t embed_vocab = 16;
    constexpr std::uint32_t embed_features = 96;
    constexpr std::uint32_t embed_padded = 128;
    constexpr std::uint32_t embed_rows = 3;
    constexpr std::uint32_t q6_groups = embed_padded / 64;
    constexpr std::uint32_t w8_groups = embed_padded / 32;
    const std::vector<std::int32_t> embed_ids{15, 1, 9};
    std::vector<std::uint8_t> q4_codes(embed_vocab * embed_padded / 2, 0);
    std::vector<std::uint16_t> q4_scales(embed_vocab * q6_groups);
    std::vector<std::uint8_t> q4_codes_row(q4_codes.size(), 0);
    std::vector<std::uint16_t> q4_scales_row(q4_scales.size());
    std::vector<std::uint8_t> q6_low(embed_vocab * q6_groups * 32, 0);
    std::vector<std::uint8_t> q6_high(embed_vocab * q6_groups * 16, 0);
    std::vector<std::uint16_t> q6_scales(embed_vocab * q6_groups);
    std::vector<std::uint8_t> w8_codes(embed_vocab * embed_padded);
    std::vector<std::uint16_t> w8_scales(embed_vocab * w8_groups);
    for (std::size_t logical_group = 0; logical_group < q4_scales.size(); ++logical_group) {
        const std::size_t row = logical_group / q6_groups;
        const std::size_t group = logical_group % q6_groups;
        const std::size_t scale_index =
            ((row >> 4U) * q6_groups + group) * 16U + (row & 15U);
        const std::uint16_t scale = std::bit_cast<std::uint16_t>(static_cast<_Float16>(
            0.0625F * static_cast<float>(1 + logical_group % 4)));
        q4_scales_row[logical_group] = scale;
        q4_scales[scale_index] = scale;
        for (std::uint32_t lane = 0; lane < 64; ++lane) {
            const int signed_code =
                static_cast<int>((logical_group * 13 + lane * 3) % 16) - 8;
            const std::size_t pair = lane / 16U;
            const std::size_t within = lane % 16U;
            const std::size_t word_index =
                (((row >> 4U) * q6_groups + group) * 4U + pair) * 16U +
                (row & 15U);
            auto& packed = q4_codes[word_index * 8U + within / 2U];
            packed = static_cast<std::uint8_t>(
                packed | ((static_cast<std::uint32_t>(signed_code) & 0x0fU)
                           << ((lane & 1U) * 4U)));
            auto& logical_packed = q4_codes_row[logical_group * 32U + lane / 2U];
            logical_packed = static_cast<std::uint8_t>(
                logical_packed | ((static_cast<std::uint32_t>(signed_code) & 0x0fU)
                                  << ((lane & 1U) * 4U)));
        }
    }
    for (std::size_t group = 0; group < q6_scales.size(); ++group) {
        q6_scales[group] = std::bit_cast<std::uint16_t>(static_cast<_Float16>(
            0.03125F * static_cast<float>(1 + group % 3)));
        for (std::uint32_t lane = 0; lane < 64; ++lane) {
            const int signed_code = static_cast<int>((group * 17 + lane * 5) % 64) - 32;
            const std::uint32_t code = static_cast<std::uint32_t>(signed_code) & 0x3fU;
            auto& low = q6_low[group * 32 + lane / 2];
            low = static_cast<std::uint8_t>(low | ((code & 0x0fU) << ((lane & 1U) * 4U)));
            const std::uint32_t high_position = lane * 2U;
            q6_high[group * 16 + high_position / 8U] = static_cast<std::uint8_t>(
                q6_high[group * 16 + high_position / 8U] |
                (((code >> 4U) & 0x03U) << (high_position & 7U)));
        }
    }
    for (std::size_t group = 0; group < w8_scales.size(); ++group) {
        w8_scales[group] = std::bit_cast<std::uint16_t>(static_cast<_Float16>(
            0.015625F * static_cast<float>(1 + group % 5)));
    }
    for (std::size_t index = 0; index < w8_codes.size(); ++index) {
        w8_codes[index] = static_cast<std::uint8_t>(
            static_cast<std::int8_t>(static_cast<int>((index * 11 + 7) % 255) - 127));
    }
    DeviceBuffer<std::uint8_t> d_q4_codes(q4_codes.size());
    DeviceBuffer<std::uint16_t> d_q4_scales(q4_scales.size());
    DeviceBuffer<std::uint8_t> d_q6_low(q6_low.size());
    DeviceBuffer<std::uint8_t> d_q6_high(q6_high.size());
    DeviceBuffer<std::uint16_t> d_q6_scales(q6_scales.size());
    DeviceBuffer<std::uint8_t> d_w8_codes(w8_codes.size());
    DeviceBuffer<std::uint16_t> d_w8_scales(w8_scales.size());
    DeviceBuffer<std::int32_t> d_embed_ids(embed_ids.size());
    DeviceBuffer<hip_bfloat16> d_embed_output(embed_features * embed_rows);
    copy_to_device(d_q4_codes, q4_codes);
    copy_to_device(d_q4_scales, q4_scales);
    copy_to_device(d_q6_low, q6_low);
    copy_to_device(d_q6_high, q6_high);
    copy_to_device(d_q6_scales, q6_scales);
    copy_to_device(d_w8_codes, w8_codes);
    copy_to_device(d_w8_scales, w8_scales);
    copy_to_device(d_embed_ids, embed_ids);
    Tensor embed_ids_tensor(d_embed_ids.get(), DType::I32,
                            {static_cast<std::int32_t>(embed_rows)});
    Tensor embed_output_tensor(d_embed_output.get(), DType::BF16,
                               {static_cast<std::int32_t>(embed_features),
                                static_cast<std::int32_t>(embed_rows)});
    Weight q4{};
    q4.qtype = QType::Q4G64_F16S;
    q4.layout = QuantLayout::Q4N16K16;
    q4.ndim = 2;
    q4.n = q4.shape[0] = q4.padded_shape[0] = embed_vocab;
    q4.k = q4.shape[1] = embed_features;
    q4.padded_shape[1] = embed_padded;
    q4.group_size = q4.group = 64;
    q4.scale_dtype = DType::FP16;
    q4.qdata = d_q4_codes.get();
    q4.qdata_bytes = q4_codes.size();
    q4.scales = d_q4_scales.get();
    q4.scale_bytes = q4_scales.size() * sizeof(std::uint16_t);
    q4.payload_bytes = ((q4.qdata_bytes + 255U) / 256U) * 256U + q4.scale_bytes;
    ninfer::ops::embedding(embed_ids_tensor, q4, embed_output_tensor, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<std::uint16_t> q4_expected(embed_features * embed_rows);
    for (std::uint32_t token = 0; token < embed_rows; ++token) {
        const std::size_t row = static_cast<std::size_t>(embed_ids[token]);
        for (std::uint32_t feature = 0; feature < embed_features; ++feature) {
            const std::size_t group = feature / 64U;
            const std::uint8_t packed =
                q4_codes_row[row * (embed_padded / 2U) + feature / 2U];
            const std::uint32_t nibble =
                (feature & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
            const int code = (nibble & 0x08U) != 0U ? static_cast<int>(nibble) - 16
                                                    : static_cast<int>(nibble);
            const float scale =
                static_cast<float>(std::bit_cast<_Float16>(
                    q4_scales_row[row * q6_groups + group]));
            q4_expected[static_cast<std::size_t>(token) * embed_features + feature] =
                bf16_rne_bits(static_cast<float>(code) * scale);
        }
    }
    require_bf16_exact(copy_to_host(d_embed_output), q4_expected, "public Q4 embedding");
    Weight malformed_q4 = q4;
    --malformed_q4.qdata_bytes;
    try {
        ninfer::ops::embedding(embed_ids_tensor, malformed_q4, embed_output_tensor, stream);
        fail("public Q4 embedding accepted a short code plane");
    } catch (const std::invalid_argument&) {
    }
    malformed_q4 = q4;
    malformed_q4.layout = QuantLayout::RowSplit;
    try {
        ninfer::ops::embedding(embed_ids_tensor, malformed_q4, embed_output_tensor, stream);
        fail("public Q4 embedding accepted the retired RowSplit layout");
    } catch (const std::invalid_argument&) {
    }
    Tensor aliased_q4_output(d_q4_codes.get(), DType::BF16,
                             {static_cast<std::int32_t>(embed_features),
                              static_cast<std::int32_t>(embed_rows)});
    try {
        ninfer::ops::embedding(embed_ids_tensor, q4, aliased_q4_output, stream);
        fail("public Q4 embedding accepted output/code aliasing");
    } catch (const std::invalid_argument&) {
    }

    Weight q6{};
    q6.qtype = QType::Q6G64_F16S;
    q6.layout = QuantLayout::RowSplit;
    q6.ndim = 2;
    q6.shape[0] = embed_vocab;
    q6.shape[1] = embed_features;
    q6.padded_shape[0] = embed_vocab;
    q6.padded_shape[1] = embed_padded;
    q6.group_size = q6.group = 64;
    q6.scale_dtype = DType::FP16;
    q6.qdata = d_q6_low.get();
    q6.qhigh = d_q6_high.get();
    q6.scales = d_q6_scales.get();
    q6.high_plane_bytes = q6_high.size();
    ninfer::ops::embedding(embed_ids_tensor, q6, embed_output_tensor, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<std::uint16_t> q6_expected(embed_features * embed_rows);
    for (std::uint32_t token = 0; token < embed_rows; ++token) {
        for (std::uint32_t feature = 0; feature < embed_features; ++feature) {
            const std::size_t group = static_cast<std::size_t>(embed_ids[token]) * q6_groups +
                                      feature / 64;
            const std::uint32_t lane = feature % 64;
            const std::uint8_t low_byte = q6_low[group * 32 + lane / 2];
            const std::uint32_t low = (lane & 1U) != 0U ? low_byte >> 4U : low_byte & 0x0fU;
            const std::uint32_t high_position = lane * 2U;
            const std::uint32_t high =
                (q6_high[group * 16 + high_position / 8U] >> (high_position & 7U)) & 0x03U;
            const std::uint32_t code = low | high << 4U;
            const int signed_code = (code & 0x20U) != 0U ? static_cast<int>(code) - 64 :
                                                          static_cast<int>(code);
            const float scale = static_cast<float>(std::bit_cast<_Float16>(q6_scales[group]));
            q6_expected[static_cast<std::size_t>(token) * embed_features + feature] =
                bf16_rne_bits(static_cast<float>(signed_code) * scale);
        }
    }
    require_bf16_exact(copy_to_host(d_embed_output), q6_expected, "public Q6 embedding");

    Weight w8{};
    w8.qtype = QType::W8G32_F16S;
    w8.layout = QuantLayout::RowSplit;
    w8.ndim = 2;
    w8.shape[0] = embed_vocab;
    w8.shape[1] = embed_features;
    w8.padded_shape[0] = embed_vocab;
    w8.padded_shape[1] = embed_padded;
    w8.group_size = w8.group = 32;
    w8.scale_dtype = DType::FP16;
    w8.qdata = d_w8_codes.get();
    w8.scales = d_w8_scales.get();
    ninfer::ops::embedding(embed_ids_tensor, w8, embed_output_tensor, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<std::uint16_t> w8_expected(embed_features * embed_rows);
    for (std::uint32_t token = 0; token < embed_rows; ++token) {
        const std::size_t row = static_cast<std::size_t>(embed_ids[token]);
        for (std::uint32_t feature = 0; feature < embed_features; ++feature) {
            const auto code = static_cast<std::int8_t>(w8_codes[row * embed_padded + feature]);
            const float scale = static_cast<float>(std::bit_cast<_Float16>(
                w8_scales[row * w8_groups + feature / 32]));
            w8_expected[static_cast<std::size_t>(token) * embed_features + feature] =
                bf16_rne_bits(static_cast<float>(code) * scale);
        }
    }
    require_bf16_exact(copy_to_host(d_embed_output), w8_expected, "public W8 embedding");
}

} // namespace

int main() {
    try {
        select_r9700();
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        qualify_i32(stream);
        qualify_bf16_movement_and_cast(stream);
        qualify_mtp_pack_split(stream);
        qualify_normalization_and_activations(stream);
        qualify_embedding(stream);
        qualify_w8_embedding_product_shape(stream);
        qualify_argmax(stream);
        qualify_public_extended_paths(stream);
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "r9700_eager_op_qualification: PASS rows=T1..8 D=" << kHidden
                  << " vocab=" << kVocabValid << '/' << kVocabPhysical
                  << " mtp_pack_split=exact" << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700_eager_op_qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
