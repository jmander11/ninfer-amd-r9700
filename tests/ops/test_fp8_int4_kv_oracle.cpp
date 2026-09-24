#include "ops/fp8_int4_kv_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer::test::fp8_int4_kv_oracle;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const char* message) {
    try {
        function();
        expect(false, message);
    } catch (const Exception&) {
    } catch (...) {
        expect(false, message);
    }
}

void test_e4m3fn_all_words_and_rne() {
    for (std::uint16_t raw = 0; raw < 256; ++raw) {
        const std::uint8_t word = static_cast<std::uint8_t>(raw);
        if ((word & 0x7fU) == 0x7fU) {
            expect_throws<std::invalid_argument>([&] { (void)decode_e4m3fn(word); },
                                                 "E4M3FN NaN word must be rejected");
            continue;
        }
        const double decoded = decode_e4m3fn(word);
        expect(std::isfinite(decoded), "finite E4M3FN word must decode finite");
        expect(encode_e4m3fn(static_cast<float>(decoded)) == word,
               "finite E4M3FN word must round-trip exactly");
    }

    expect(encode_e4m3fn(1.0625F) == 0x38U, "E4M3FN midpoint must tie to even lower code");
    expect(encode_e4m3fn(1.1875F) == 0x3aU, "E4M3FN midpoint must tie to even upper code");
    expect(encode_e4m3fn(std::ldexp(1.0F, -10)) == 0x00U,
           "E4M3FN half-minimum subnormal must tie to zero");
    expect(encode_e4m3fn(std::nextafter(std::ldexp(1.0F, -10), 1.0F)) == 0x01U,
           "E4M3FN value above underflow midpoint must round up");
    expect(encode_e4m3fn(1000.0F) == 0x7eU && encode_e4m3fn(-1000.0F) == 0xfeU,
           "finite E4M3FN overflow must saturate with sign");
    expect(encode_e4m3fn(-0.0F) == 0x80U && std::signbit(decode_e4m3fn(0x80U)),
           "E4M3FN must preserve signed zero");
    expect_throws<std::invalid_argument>(
        [] { (void)encode_e4m3fn(std::numeric_limits<float>::infinity()); },
        "E4M3FN infinity source must be rejected");
    expect_throws<std::invalid_argument>(
        [] { (void)encode_e4m3fn(std::numeric_limits<float>::quiet_NaN()); },
        "E4M3FN NaN source must be rejected");
}

void test_key_storage_bits() {
    const std::vector<float> source{0.0F, -0.0F, 1.0F, -1.0F, 448.0F, -448.0F, 1.0625F, 1.1875F};
    const Fp8Keys keys = encode_keys(source, 1, source.size());
    const std::vector<std::uint8_t> expected{0x00U, 0x80U, 0x38U, 0xb8U,
                                             0x7eU, 0xfeU, 0x38U, 0x3aU};
    expect(keys.codes == expected, "direct FP8 key storage bits must match the codec contract");
    expect(keys.at(0, 2) == 1.0 && keys.at(0, 4) == 448.0,
           "direct FP8 keys must decode from stored bits");
}

double squared_error(const std::vector<float>& source, const Int4Values& encoded) {
    double error = 0.0;
    for (std::size_t lane = 0; lane < source.size(); ++lane) {
        const double delta = static_cast<double>(source[lane]) - encoded.at(0, lane);
        error += delta * delta;
    }
    return error;
}

void test_int4_storage_and_groups() {
    std::vector<float> exact(16);
    for (std::size_t lane = 0; lane < exact.size(); ++lane) {
        exact[lane] = static_cast<float>(static_cast<int>(lane % 15) - 7);
    }
    const Int4Values unit = encode_values(exact, 1, 16, 16);
    expect(unit.fp16_scales.size() == 1 && unit.fp16_scales[0] == 0x3c00U,
           "INT4 unit-scale group must store exact FP16 one");
    const std::vector<std::uint8_t> expected_packed{0xa9U, 0xcbU, 0xedU, 0x0fU,
                                                    0x21U, 0x43U, 0x65U, 0x97U};
    expect(unit.packed_codes == expected_packed,
           "INT4 low-lane-first two's-complement bytes must match the storage contract");
    for (std::size_t lane = 0; lane < exact.size(); ++lane) {
        expect(unit.code_at(0, lane) == static_cast<int>(exact[lane]),
               "INT4 signed code must round-trip exactly");
        expect(unit.at(0, lane) == static_cast<double>(exact[lane]),
               "INT4 value must decode code times stored FP16 scale");
    }

    std::vector<float> fixture(32);
    for (std::size_t lane = 0; lane < 16; ++lane) {
        fixture[lane] = static_cast<float>(static_cast<int>(lane) - 8) / 8.0F;
        fixture[16 + lane] = static_cast<float>(static_cast<int>(lane) - 7) * 2.0F;
    }
    const Int4Values g16 = encode_values(fixture, 1, 32, 16);
    const Int4Values g32 = encode_values(fixture, 1, 32, 32);
    expect(g16.packed_codes.size() == 16 && g32.packed_codes.size() == 16,
           "G16 and G32 must use the same INT4 code bytes");
    expect(g16.fp16_scales.size() == 2 && g32.fp16_scales.size() == 1,
           "G16 must use two scale words where G32 uses one");
    expect(squared_error(fixture, g16) < squared_error(fixture, g32),
           "G16 must reduce error for the split-range qualification fixture");

    const Int4Values zero = encode_values(std::vector<float>(32, -0.0F), 1, 32, 32);
    expect(zero.fp16_scales[0] == 0 &&
               std::all_of(zero.packed_codes.begin(), zero.packed_codes.end(),
                           [](std::uint8_t byte) { return byte == 0; }),
           "zero group must have zero scale and codes");

    std::vector<float> underflow_source(16, std::ldexp(1.0F, -24));
    underflow_source[3] = -underflow_source[3];
    const Int4Values underflow = encode_values(underflow_source, 1, 16, 16);
    expect(underflow.fp16_scales[0] == 0x0000U &&
               std::all_of(underflow.packed_codes.begin(), underflow.packed_codes.end(),
                           [](std::uint8_t byte) { return byte == 0; }),
           "finite nonzero group with FP16 scale underflow must canonicalize to +0 and zero codes");
    expect(std::all_of(underflow_source.begin(), underflow_source.end(),
                       [](float value) { return value != 0.0F && std::isfinite(value); }),
           "scale-underflow fixture must remain finite and nonzero");

    Int4Values invalid = zero;
    invalid.packed_codes[0] = 0x08U;
    expect_throws<std::invalid_argument>([&] { (void)invalid.at(0, 0); },
                                         "reserved symmetric INT4 -8 must be rejected");
    std::vector<float> nonfinite(16, 0.0F);
    nonfinite[3] = std::numeric_limits<float>::quiet_NaN();
    expect_throws<std::invalid_argument>([&] { (void)encode_values(nonfinite, 1, 16, 16); },
                                         "non-finite INT4 source must be rejected");
    std::vector<float> overflow(16, std::numeric_limits<float>::max());
    expect_throws<std::overflow_error>([&] { (void)encode_values(overflow, 1, 16, 16); },
                                       "unrepresentable FP16 INT4 scale must be rejected");
}

void test_attention_decodes_storage_then_evaluates_fp64() {
    constexpr std::size_t dimension = 16;
    std::vector<float> query(dimension, 0.0F);
    query[0] = 1.0F;

    std::vector<float> key_source(2 * dimension, 0.0F);
    key_source[0] = 1.0625F;             // midpoint encodes to 1.0
    key_source[dimension] = -1.0625F;    // midpoint encodes to -1.0
    const Fp8Keys keys = encode_keys(key_source, 2, dimension);

    std::vector<float> value_source(2 * dimension, 0.0F);
    std::fill(value_source.begin(), value_source.begin() + dimension, 1.0F);
    std::fill(value_source.begin() + dimension, value_source.end(), -2.0F);
    const Int4Values values = encode_values(value_source, 2, dimension, 16);
    const std::vector<double> actual = attention_fp64(query, keys, values, 1.0);

    const double score0 = keys.at(0, 0);
    const double score1 = keys.at(1, 0);
    const double p0 = std::exp(score0 - std::max(score0, score1)) /
                      (std::exp(score0 - std::max(score0, score1)) +
                       std::exp(score1 - std::max(score0, score1)));
    const double expected = p0 * values.at(0, 0) + (1.0 - p0) * values.at(1, 0);
    for (double lane : actual) {
        expect(std::abs(lane - expected) <= 2.0 * std::numeric_limits<double>::epsilon(),
               "attention oracle must use decoded FP8 K and INT4 V in stable FP64 formula");
    }

    // This guards against accidentally using the pre-quantized key in the formula.
    const double source_p0 = std::exp(1.0625 - 1.0625) /
                             (std::exp(1.0625 - 1.0625) + std::exp(-1.0625 - 1.0625));
    const double source_result = source_p0 * values.at(0, 0) + (1.0 - source_p0) * values.at(1, 0);
    expect(std::abs(actual[0] - source_result) > 1.0e-4,
           "attention oracle must not consume unrepresented key source values");
}

void test_greedy_token_is_an_exact_separate_gate() {
    constexpr std::size_t dimension = 16;
    std::vector<float> query(dimension, 0.0F);
    query[0] = 1.0F;
    std::vector<float> key(dimension, 0.0F);
    key[0] = 1.0F;
    std::vector<float> value(dimension, 0.0F);
    value[0] = 1.0F;
    const Fp8Keys keys = encode_keys(key, 1, dimension);
    const Int4Values values = encode_values(value, 1, dimension, 16);

    std::vector<float> weights(3 * dimension, 0.0F);
    weights[0] = 1.0F;
    weights[dimension] = std::nextafter(1.0F, 2.0F);
    weights[2 * dimension] = -1.0F;
    const GreedyResult near_tie =
        attention_greedy_fp64(query, keys, values, 1.0, weights, 3);
    expect(near_tie.token == 1, "positive one-ULP logit margin must select token one exactly");
    expect(near_tie.logits[1] > near_tie.logits[0] &&
               near_tie.logits[1] - near_tie.logits[0] < 1.0e-5,
           "greedy fixture must exercise an adversarially small argmax margin");

    // These perturbed logits pass an ordinary absolute tolerance but select the wrong token.
    std::vector<double> tolerant_but_wrong = near_tie.logits;
    std::swap(tolerant_but_wrong[0], tolerant_but_wrong[1]);
    expect(std::abs(tolerant_but_wrong[0] - near_tie.logits[0]) < 1.0e-5 &&
               tolerant_but_wrong[0] > tolerant_but_wrong[1],
           "float tolerance must not stand in for exact greedy-token identity");

    weights[dimension] = 1.0F;
    const GreedyResult exact_tie =
        attention_greedy_fp64(query, keys, values, 1.0, weights, 3);
    expect(exact_tie.token == 0, "exact logit tie must deterministically choose lowest token id");
}

} // namespace

int main() {
    test_e4m3fn_all_words_and_rne();
    test_key_storage_bits();
    test_int4_storage_and_groups();
    test_attention_decodes_storage_then_evaluates_fp64();
    test_greedy_token_is_an_exact_separate_gate();
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " fp8-e4m3-k/int4-v host oracle correctness\n";
    return failures == 0 ? 0 : 1;
}
