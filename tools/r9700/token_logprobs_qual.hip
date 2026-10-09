#include "ninfer/ops/token_logprobs.h"
#include "core/device.h"
#include <hip/hip_runtime.h>
#include <stdexcept>


#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;

namespace {
std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float bf16_to_f32(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

// FP32 log-sum-exp over the token domain plus one FP32 subtraction at logit magnitudes below 64.
struct PointwiseCriterion {
    double absolute;
    double relative;
};

constexpr PointwiseCriterion kLogprobCriterion{1.0e-5, 1.0e-5};

int verify_pointwise(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& expected, PointwiseCriterion criterion) {
    int failures = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) >
                criterion.absolute + criterion.relative * std::abs(expected[i])) {
            std::cerr << label << " index=" << i << " actual=" << actual[i]
                      << " expected=" << expected[i] << '\n';
            ++failures;
        }
    }
    return failures;
}

class GuardedDeviceBuffer {
    unsigned char* allocation_ = nullptr;
    std::size_t bytes_;
    static constexpr std::size_t guard_ = 256;
public:
    explicit GuardedDeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        HIP_CHECK(hipMalloc(&allocation_, bytes_ + 2 * guard_));
        HIP_CHECK(hipMemset(allocation_, 0xa5, bytes_ + 2 * guard_));
    }

    ~GuardedDeviceBuffer() { (void)hipFree(allocation_); }

    GuardedDeviceBuffer(const GuardedDeviceBuffer&)            = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    void* data() const { return allocation_ + guard_; }

    void copy_from_host(const void* src, std::size_t bytes) {
        HIP_CHECK(hipMemcpy(data(), src, bytes, hipMemcpyHostToDevice));
    }

    void copy_to_host(void* dst, std::size_t bytes) const {
        HIP_CHECK(hipMemcpy(dst, data(), bytes, hipMemcpyDeviceToHost));
    }

    void fill(std::uint8_t byte) { HIP_CHECK(hipMemset(data(), byte, bytes_)); }

    int verify_guards(const char* label) const {
        std::array<unsigned char, guard_ * 2> guards{};
        HIP_CHECK(hipMemcpy(guards.data(), allocation_, guard_, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(guards.data() + guard_, allocation_ + guard_ + bytes_, guard_,
                            hipMemcpyDeviceToHost));
        if (std::all_of(guards.begin(), guards.end(), [](unsigned char v) { return v == 0xa5; })) {
            return 0;
        }
        std::cerr << label << ": output guard changed\n";
        return 1;
    }
};

constexpr std::uint8_t kUntouched = 0xcd;

enum class LogitPattern : std::uint8_t {
    Uniform,    // independent logits in [-12, 12); BF16 rounding already produces ties
    Peaked,     // a few dominant tokens over a low floor
    ZeroTies,   // the best tokens are a mix of +0 and -0 over a negative floor
    LargeEqual, // equal large logits expose cancellation in max + log(sum)
    AllEqual,   // every logit identical: ranks are the lowest ids
};

struct Case {
    std::string label;
    std::int32_t physical_rows = 0;
    std::int32_t token_domain  = 0;
    // Bound extents.
    std::int32_t columns = 0;
    std::int32_t width   = 0;
    std::int32_t batch   = 0;
    std::int32_t top     = 0;
    // Extents of the panels the bound views are cut from; larger values exercise strided views.
    std::int32_t frame_columns = 0;
    std::int32_t frame_width   = 0;
    std::int32_t frame_batch   = 0;
    bool with_counts           = false;
    bool with_columns          = false;
    bool disable_some_rows     = false;
    LogitPattern pattern       = LogitPattern::Uniform;
    std::uint32_t seed         = 1;
};

std::size_t idx2(std::int32_t i, std::int32_t b, std::int32_t extent0) {
    return static_cast<std::size_t>(b) * static_cast<std::size_t>(extent0) +
           static_cast<std::size_t>(i);
}

std::size_t idx3(std::int32_t k, std::int32_t i, std::int32_t b, std::int32_t extent0,
                 std::int32_t extent1) {
    return (static_cast<std::size_t>(b) * static_cast<std::size_t>(extent1) +
            static_cast<std::size_t>(i)) *
               static_cast<std::size_t>(extent0) +
           static_cast<std::size_t>(k);
}

std::vector<std::uint16_t> make_logits(const Case& c) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(c.physical_rows) *
                                      static_cast<std::size_t>(c.frame_columns) *
                                      static_cast<std::size_t>(c.frame_batch));
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> uniform(-12.0f, 12.0f);
    std::uniform_int_distribution<std::int32_t> any_token(0, c.token_domain - 1);
    for (std::int32_t b = 0; b < c.frame_batch; ++b) {
        for (std::int32_t col = 0; col < c.frame_columns; ++col) {
            std::uint16_t* column =
                logits.data() + idx3(0, col, b, c.physical_rows, c.frame_columns);
            for (std::int32_t v = 0; v < c.token_domain; ++v) {
                float value = 0.0f;
                switch (c.pattern) {
                case LogitPattern::Uniform:
                    value = uniform(rng);
                    break;
                case LogitPattern::Peaked:
                    value = -20.0f + 0.25f * uniform(rng);
                    break;
                case LogitPattern::ZeroTies:
                    value = -3.0f + 0.125f * uniform(rng);
                    break;
                case LogitPattern::AllEqual:
                    value = 1.5f;
                    break;
                case LogitPattern::LargeEqual:
                    value = 1048576.0f;
                    break;
                }
                column[v] = f32_to_bf16(value);
            }
            if (c.pattern == LogitPattern::Peaked) {
                for (int peak = 0; peak < 6; ++peak) {
                    column[any_token(rng)] = f32_to_bf16(20.0f - 1.75f * static_cast<float>(peak));
                }
            }
            if (c.pattern == LogitPattern::ZeroTies) {
                for (int tie = 0; tie < 2 * c.top; ++tie) {
                    column[any_token(rng)] = f32_to_bf16((tie % 2) == 0 ? 0.0f : -0.0f);
                }
            }
            // Physical padding rows are outside the token domain and must not participate.
            for (std::int32_t v = c.token_domain; v < c.physical_rows; ++v) {
                column[v] = f32_to_bf16(60.0f);
            }
        }
    }
    return logits;
}

struct SlotOracle {
    double token_logprob = 0.0;
    std::vector<std::int32_t> ids;
    std::vector<double> logprobs;
};

SlotOracle slot_oracle(const std::uint16_t* column, std::int32_t token_domain, std::int32_t token,
                       std::int32_t top) {
    std::vector<double> z(static_cast<std::size_t>(token_domain));
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int32_t v = 0; v < token_domain; ++v) {
        z[static_cast<std::size_t>(v)] = static_cast<double>(bf16_to_f32(column[v]));
        maximum                        = std::max(maximum, z[static_cast<std::size_t>(v)]);
    }
    double sum = 0.0;
    for (const double value : z) { sum += std::exp(value - maximum); }
    const double log_sum_exp = maximum + std::log(sum);

    std::vector<std::int32_t> order(static_cast<std::size_t>(token_domain));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + top, order.end(),
                      [&z](std::int32_t a, std::int32_t b) {
                          const double za = z[static_cast<std::size_t>(a)];
                          const double zb = z[static_cast<std::size_t>(b)];
                          return za > zb || (za == zb && a < b);
                      });
    SlotOracle oracle;
    oracle.token_logprob = z[static_cast<std::size_t>(token)] - log_sum_exp;
    for (std::int32_t k = 0; k < top; ++k) {
        const std::int32_t id = order[static_cast<std::size_t>(k)];
        oracle.ids.push_back(id);
        oracle.logprobs.push_back(z[static_cast<std::size_t>(id)] - log_sum_exp);
    }
    return oracle;
}

int run_case(const Case& c) {
    const std::vector<std::uint16_t> logits = make_logits(c);
    std::mt19937 rng(c.seed ^ 0x9e3779b9u);
    std::uniform_int_distribution<std::int32_t> any_token(0, c.token_domain - 1);
    std::uniform_int_distribution<std::int32_t> any_column(0, c.columns - 1);
    std::uniform_int_distribution<std::int32_t> any_count(0, c.width);

    const std::size_t slots =
        static_cast<std::size_t>(c.frame_width) * static_cast<std::size_t>(c.frame_batch);
    std::vector<std::int32_t> tokens(slots);
    std::vector<std::int32_t> columns(slots);
    // Out-of-range ids in slots the Op must skip: reading one would fault under memcheck.
    for (std::int32_t b = 0; b < c.frame_batch; ++b) {
        for (std::int32_t i = 0; i < c.frame_width; ++i) {
            tokens[idx2(i, b, c.frame_width)]  = any_token(rng);
            columns[idx2(i, b, c.frame_width)] = any_column(rng);
        }
    }
    std::vector<std::int32_t> row_enabled(static_cast<std::size_t>(c.batch), 1);
    std::vector<std::int32_t> counts(static_cast<std::size_t>(c.batch), c.width);
    for (std::int32_t b = 0; b < c.batch; ++b) {
        if (c.disable_some_rows && (b % 2) == 1) { row_enabled[static_cast<std::size_t>(b)] = 0; }
        if (c.with_counts) {
            // Row 0 keeps the full width so the widest slot is always exercised.
            counts[static_cast<std::size_t>(b)] = b == 0 ? c.width : any_count(rng);
        }
    }
    const auto active = [&](std::int32_t i, std::int32_t b) {
        return row_enabled[static_cast<std::size_t>(b)] != 0 &&
               i < counts[static_cast<std::size_t>(b)];
    };
    for (std::int32_t b = 0; b < c.batch; ++b) {
        for (std::int32_t i = 0; i < c.width; ++i) {
            if (active(i, b)) { continue; }
            tokens[idx2(i, b, c.frame_width)]  = c.physical_rows * 64;
            columns[idx2(i, b, c.frame_width)] = c.frame_columns * 64;
        }
    }

    const std::size_t top_elems = slots * static_cast<std::size_t>(c.top);
    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_tokens(slots * sizeof(std::int32_t));
    GuardedDeviceBuffer device_columns(slots * sizeof(std::int32_t));
    GuardedDeviceBuffer device_enabled(row_enabled.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_counts(counts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_token_logprob(slots * sizeof(float));
    GuardedDeviceBuffer device_top_ids(top_elems * sizeof(std::int32_t));
    GuardedDeviceBuffer device_top_logprobs(top_elems * sizeof(float));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_tokens.copy_from_host(tokens.data(), slots * sizeof(std::int32_t));
    device_columns.copy_from_host(columns.data(), slots * sizeof(std::int32_t));
    device_enabled.copy_from_host(row_enabled.data(), row_enabled.size() * sizeof(std::int32_t));
    device_counts.copy_from_host(counts.data(), counts.size() * sizeof(std::int32_t));
    device_token_logprob.fill(kUntouched);
    device_top_ids.fill(kUntouched);
    device_top_logprobs.fill(kUntouched);

    const auto bind2 = [&](GuardedDeviceBuffer& buffer, DType dtype) {
        return Tensor(buffer.data(), dtype, {c.frame_width, c.frame_batch})
            .slice(0, 0, c.width)
            .slice(1, 0, c.batch);
    };
    const auto bind3 = [&](GuardedDeviceBuffer& buffer, DType dtype) {
        return Tensor(buffer.data(), dtype, {c.top, c.frame_width, c.frame_batch})
            .slice(1, 0, c.width)
            .slice(2, 0, c.batch);
    };
    const Tensor logits_tensor =
        Tensor(device_logits.data(), DType::BF16, {c.physical_rows, c.frame_columns, c.frame_batch})
            .slice(1, 0, c.columns)
            .slice(2, 0, c.batch);
    const Tensor tokens_tensor  = bind2(device_tokens, DType::I32);
    const Tensor columns_tensor = bind2(device_columns, DType::I32);
    const Tensor enabled_tensor(device_enabled.data(), DType::I32, {c.batch});
    const Tensor counts_tensor(device_counts.data(), DType::I32, {c.batch});
    Tensor token_logprob_tensor = bind2(device_token_logprob, DType::FP32);
    Tensor top_ids_tensor       = bind3(device_top_ids, DType::I32);
    Tensor top_logprobs_tensor  = bind3(device_top_logprobs, DType::FP32);

    ops::token_logprobs(logits_tensor, tokens_tensor, enabled_tensor,
                        c.with_counts ? &counts_tensor : nullptr,
                        c.with_columns ? &columns_tensor : nullptr, c.token_domain,
                        token_logprob_tensor, top_ids_tensor, top_logprobs_tensor, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> got_token_logprob(slots);
    std::vector<std::int32_t> got_top_ids(top_elems);
    std::vector<float> got_top_logprobs(top_elems);
    device_token_logprob.copy_to_host(got_token_logprob.data(), slots * sizeof(float));
    device_top_ids.copy_to_host(got_top_ids.data(), top_elems * sizeof(std::int32_t));
    device_top_logprobs.copy_to_host(got_top_logprobs.data(), top_elems * sizeof(float));

    float untouched_f32       = 0.0f;
    std::int32_t untouched_id = 0;
    std::memset(&untouched_f32, kUntouched, sizeof(untouched_f32));
    std::memset(&untouched_id, kUntouched, sizeof(untouched_id));
    const auto same_bits = [](float a, float b) {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    };

    int failures = 0;
    std::vector<double> actual;
    std::vector<double> expected;
    for (std::int32_t b = 0; b < c.frame_batch; ++b) {
        for (std::int32_t i = 0; i < c.frame_width; ++i) {
            const bool bound = i < c.width && b < c.batch;
            if (!bound || !active(i, b)) {
                bool untouched =
                    same_bits(got_token_logprob[idx2(i, b, c.frame_width)], untouched_f32);
                for (std::int32_t k = 0; k < c.top; ++k) {
                    const std::size_t at = idx3(k, i, b, c.top, c.frame_width);
                    untouched            = untouched && got_top_ids[at] == untouched_id &&
                                           same_bits(got_top_logprobs[at], untouched_f32);
                }
                if (!untouched) {
                    std::cerr << c.label << ": inactive slot (" << i << ',' << b
                              << ") was written\n";
                    ++failures;
                }
                continue;
            }
            const std::int32_t column = c.with_columns ? columns[idx2(i, b, c.frame_width)] : i;
            const SlotOracle oracle =
                slot_oracle(logits.data() + idx3(0, column, b, c.physical_rows, c.frame_columns),
                            c.token_domain, tokens[idx2(i, b, c.frame_width)], c.top);
            actual.push_back(static_cast<double>(got_token_logprob[idx2(i, b, c.frame_width)]));
            expected.push_back(oracle.token_logprob);
            for (std::int32_t k = 0; k < c.top; ++k) {
                const std::size_t at = idx3(k, i, b, c.top, c.frame_width);
                if (got_top_ids[at] != oracle.ids[static_cast<std::size_t>(k)]) {
                    std::cerr << c.label << ": slot (" << i << ',' << b << ") rank " << k << " id "
                              << got_top_ids[at] << " expected "
                              << oracle.ids[static_cast<std::size_t>(k)] << '\n';
                    ++failures;
                }
                actual.push_back(static_cast<double>(got_top_logprobs[at]));
                expected.push_back(oracle.logprobs[static_cast<std::size_t>(k)]);
            }
        }
    }
    if (actual.empty()) {
        std::cerr << c.label << ": case has no active slot\n";
        ++failures;
    } else {
        failures += verify_pointwise(c.label, actual, expected, kLogprobCriterion);
    }
    failures += device_token_logprob.verify_guards((c.label + " token_logprob").c_str());
    failures += device_top_ids.verify_guards((c.label + " top_ids").c_str());
    failures += device_top_logprobs.verify_guards((c.label + " top_logprobs").c_str());
    return failures;
}

// The qwen3.8-27b output head: 248320 physical rows, 248077 tokens.
constexpr std::int32_t kRealRows   = 248320;
constexpr std::int32_t kRealDomain = 248077;

Case dense_case(std::string label, std::int32_t rows, std::int32_t domain, std::int32_t width,
                std::int32_t batch, std::int32_t top, LogitPattern pattern, std::uint32_t seed) {
    Case c;
    c.label         = std::move(label);
    c.physical_rows = rows;
    c.token_domain  = domain;
    c.columns = c.frame_columns = width;
    c.width = c.frame_width = width;
    c.batch = c.frame_batch = batch;
    c.top                   = top;
    c.pattern               = pattern;
    c.seed                  = seed;
    return c;
}

// Chain speculative round: column i scores token i, per-row licensed counts, bound as the
// [W,B] prefix of a [16,6] frame.
Case chain_case(std::string label, std::int32_t width, std::int32_t batch, std::uint32_t seed) {
    Case c          = dense_case(std::move(label), kRealRows, kRealDomain, width, batch,
                                 ops::kMaximumTopLogprobs, LogitPattern::Peaked, seed);
    c.frame_columns = c.frame_width = 16;
    c.frame_batch                   = 8;
    c.with_counts                   = true;
    c.disable_some_rows             = batch > 1;
    return c;
}

// Packed-tree speculative round: token i is scored at an explicit node column.
Case tree_case(std::string label, std::int32_t width, std::int32_t batch, std::uint32_t seed) {
    Case c         = chain_case(std::move(label), width, batch, seed);
    c.with_columns = true;
    c.pattern      = LogitPattern::Uniform;
    return c;
}

} // namespace

int main() {
    std::vector<Case> cases;
    // Token domains below, at, and across the per-thread strip boundary.
    cases.push_back(dense_case("domain=40", 64, 40, 3, 2, 20, LogitPattern::Uniform, 1));
    cases.push_back(dense_case("domain=20 K=20", 20, 20, 1, 1, 20, LogitPattern::Uniform, 2));
    cases.push_back(dense_case("domain=1023", 1100, 1023, 2, 1, 20, LogitPattern::Uniform, 3));
    cases.push_back(dense_case("domain=1024", 1100, 1024, 2, 1, 20, LogitPattern::Uniform, 4));
    cases.push_back(dense_case("domain=1025", 1100, 1025, 2, 1, 20, LogitPattern::Uniform, 5));
    cases.push_back(dense_case("K=1", 4096, 4000, 2, 2, 1, LogitPattern::Peaked, 6));
    cases.push_back(dense_case("K=5", 4096, 4000, 2, 2, 5, LogitPattern::Uniform, 7));
    // Ordering ties: signed zeros at the top, and a column of identical logits.
    cases.push_back(dense_case("zero ties", 8192, 8000, 2, 2, 20, LogitPattern::ZeroTies, 8));
    cases.push_back(dense_case("all equal", 4096, 4000, 1, 1, 20, LogitPattern::AllEqual, 9));
    cases.push_back(dense_case("large equal", 4096, 4000, 1, 1, 20, LogitPattern::LargeEqual, 10));
    // Real head geometry for each caller shape.
    cases.push_back(dense_case("ordinary B=1", kRealRows, kRealDomain, 1, 1,
                               ops::kMaximumTopLogprobs, LogitPattern::Peaked, 20));
    cases.push_back(dense_case("ordinary B=8", kRealRows, kRealDomain, 1, 8,
                               ops::kMaximumTopLogprobs, LogitPattern::Uniform, 21));
    cases.push_back(chain_case("chain W=4 B=1", 4, 1, 22));
    cases.push_back(chain_case("chain W=8 B=3", 8, 3, 23));
    cases.push_back(chain_case("chain W=16 B=8", 16, 8, 24));
    cases.push_back(tree_case("tree W=12 B=1", 12, 1, 25));
    cases.push_back(tree_case("tree W=12 B=4", 12, 4, 26));
    Case zero_ties_real    = chain_case("chain zero ties W=6 B=2", 6, 2, 27);
    zero_ties_real.pattern = LogitPattern::ZeroTies;
    cases.push_back(zero_ties_real);

    int failures = 0;
    for (const Case& c : cases) { failures += run_case(c); }
    if (failures != 0) {
        std::cerr << "token_logprobs failures=" << failures << '\n';
        return 1;
    }
    std::cout << "token_logprobs: PASS\n";
    return 0;
}
