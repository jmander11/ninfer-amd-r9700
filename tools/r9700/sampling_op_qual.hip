#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kPhysicalVocabulary = 248320;
constexpr int kTokenDomain = 248320;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

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

float bf16_to_float(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

hip_bfloat16 bf16(float value) {
    hip_bfloat16 result;
    result.data = bf16_rne_bits(value);
    return result;
}

unsigned long long splitmix64(unsigned long long value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

float uniform(unsigned long long seed, int position, int purpose) {
    unsigned long long key = splitmix64(
        seed ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
                0xD1B54A32D192ED03ULL));
    key = splitmix64(key ^
                     (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21));
    return static_cast<float>(static_cast<unsigned int>(key >> 40)) *
           (1.0F / 16777216.0F);
}

struct OracleResult {
    int token = -1;
    std::vector<int> support;
    double boundary_margin = 0.0;
};

OracleResult oracle_sample(const hip_bfloat16* logits, int token_domain,
                           const ninfer::ops::SamplingConfig& config,
                           const std::vector<int>* counts, int position, int purpose) {
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        float value = bf16_to_float(logits[token]);
        if (config.temperature > 0.0F && counts != nullptr) {
            const int count = (*counts)[static_cast<std::size_t>(token)];
            if (count > 0) { value -= config.presence_penalty; }
            value -= config.frequency_penalty * static_cast<float>(count);
        }
        candidates.emplace_back(value, token);
    }
    const auto better = [](const auto& left, const auto& right) {
        return left.first > right.first ||
               (left.first == right.first && left.second < right.second);
    };
    if (!(config.temperature > 0.0F)) {
        auto best = candidates.front();
        for (const auto& candidate : candidates) {
            if (better(candidate, best)) { best = candidate; }
        }
        OracleResult result;
        result.token = best.second;
        return result;
    }

    int cap = 20;
    if (config.top_k > 0 && config.top_k < 20) { cap = config.top_k; }
    cap = std::min(cap, token_domain);
    std::partial_sort(candidates.begin(), candidates.begin() + cap, candidates.end(), better);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weight(static_cast<std::size_t>(cap));
    const double maximum = static_cast<double>(candidates[0].first) /
                           static_cast<double>(config.temperature);
    double total = 0.0;
    for (int index = 0; index < cap; ++index) {
        weight[static_cast<std::size_t>(index)] =
            std::exp(static_cast<double>(candidates[static_cast<std::size_t>(index)].first) /
                         static_cast<double>(config.temperature) -
                     maximum);
        total += weight[static_cast<std::size_t>(index)];
    }
    const double min_threshold = config.min_p > 0.0F
                                     ? static_cast<double>(config.min_p) * weight[0]
                                     : -1.0;
    const double top_p_target = static_cast<double>(config.top_p) * total;
    double cumulative = 0.0;
    int support_size = 0;
    for (int index = 0; index < cap; ++index) {
        if (min_threshold >= 0.0 && weight[static_cast<std::size_t>(index)] < min_threshold) {
            break;
        }
        cumulative += weight[static_cast<std::size_t>(index)];
        support_size = index + 1;
        if (config.top_p < 1.0F && cumulative >= top_p_target) { break; }
    }
    support_size = std::max(1, support_size);
    double support_total = 0.0;
    for (int index = 0; index < support_size; ++index) {
        support_total += weight[static_cast<std::size_t>(index)];
    }
    const double goal = static_cast<double>(uniform(config.seed, position, purpose)) * support_total;
    cumulative = 0.0;
    int selected = candidates[static_cast<std::size_t>(support_size - 1)].second;
    double margin = std::numeric_limits<double>::infinity();
    OracleResult result;
    for (int index = 0; index < support_size; ++index) {
        const double previous = cumulative;
        cumulative += weight[static_cast<std::size_t>(index)];
        margin = std::min(margin, std::min(std::abs(goal - previous),
                                           std::abs(cumulative - goal)));
        result.support.push_back(candidates[static_cast<std::size_t>(index)].second);
        if (goal < cumulative) {
            selected = candidates[static_cast<std::size_t>(index)].second;
            break;
        }
    }
    result.token = selected;
    result.boundary_margin = margin;
    return result;
}

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device));
        if (properties.integrated == 0 && std::string_view(properties.gcnArchName) == "gfx1201") {
            if (selected >= 0) { fail("expected exactly one discrete gfx1201 device"); }
            selected = device;
        }
    }
    if (selected < 0) { fail("no discrete gfx1201 R9700 found"); }
    return selected;
}

std::vector<hip_bfloat16> make_logits(int rows, int physical_rows) {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(rows) * physical_rows);
    for (int row = 0; row < rows; ++row) {
        hip_bfloat16* column = result.data() + static_cast<std::size_t>(row) * physical_rows;
        for (int token = 0; token < physical_rows; ++token) {
            const int noise = (token * 17 + row * 31) % 997;
            column[token] = bf16(-12.0F + static_cast<float>(noise) / 997.0F);
        }
        // Top candidates are deliberately non-contiguous, cross 512-token partials, include an
        // exact BF16 tie, and include counted tokens whose penalty changes their rank.
        for (int rank = 0; rank < 28; ++rank) {
            const int token = 13 + rank * 7919 + row * 101;
            column[token] = bf16(8.0F - static_cast<float>(rank) * 0.125F);
        }
        column[17 + row * 101] = bf16(8.0F); // lower-id tie wins before penalties.
    }
    return result;
}

void qualify_batches(ninfer::DeviceContext& device) {
    using ninfer::ops::SamplingConfig;
    for (int rows = 1; rows <= static_cast<int>(ninfer::kMaximumConcurrency); ++rows) {
        std::vector<hip_bfloat16> logits = make_logits(rows, kPhysicalVocabulary);
        std::vector<int> positions(static_cast<std::size_t>(rows));
        std::vector<int> counts(static_cast<std::size_t>(rows) * kTokenDomain, 0);
        std::vector<SamplingConfig> configs(static_cast<std::size_t>(rows));

        ninfer::DeviceBuffer logits_device(logits.size() * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer out_device(static_cast<std::size_t>(rows) * sizeof(int));
        ninfer::DeviceBuffer position_device(static_cast<std::size_t>(rows) * sizeof(int));
        ninfer::DeviceBuffer count_device(counts.size() * sizeof(int));
        ninfer::DeviceBuffer config_device(static_cast<std::size_t>(rows) * sizeof(SamplingConfig));
        const std::size_t workspace_bytes =
            ninfer::ops::sampling_workspace_capacity_bytes(kTokenDomain, rows, rows);
        ninfer::DeviceArena workspace(workspace_bytes);

        logits_device.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
        count_device.copy_from_host(counts.data(), counts.size() * sizeof(int));
        std::vector<OracleResult> oracle(static_cast<std::size_t>(rows));
        std::vector<int> expected_counts = counts;
        for (int row = 0; row < rows; ++row) {
            positions[static_cast<std::size_t>(row)] = 4096 + row * 37;
            SamplingConfig& config = configs[static_cast<std::size_t>(row)];
            const bool greedy = rows > 1 && row == 0;
            config.temperature = greedy ? 0.0F : 0.55F + static_cast<float>(row % 3) * 0.4F;
            constexpr int top_k_cases[] = {0, 1, 7, 19, 20, 27, -3, 11};
            config.top_k = top_k_cases[row];
            config.top_p = (row % 3 == 0) ? 0.0F : ((row % 3 == 1) ? 0.72F : 1.0F);
            config.min_p = (row % 4 == 2) ? 0.35F : 0.0F;
            config.presence_penalty = greedy ? 7.0F : ((row % 3 != 1) ? 0.35F : 0.0F);
            config.frequency_penalty = greedy ? 9.0F : ((row % 3 != 0) ? 0.2F : 0.0F);
            config.seed = 0x123456789abcdef0ULL + static_cast<unsigned long long>(row * 977 + rows);
            int* row_counts_device = static_cast<int*>(count_device.data()) +
                                     static_cast<std::size_t>(row) * kTokenDomain;
            config.token_counts =
                (rows == static_cast<int>(ninfer::kMaximumConcurrency) && row == rows - 1)
                    ? nullptr
                    : row_counts_device;
            std::vector<int> row_counts(static_cast<std::size_t>(kTokenDomain), 0);
            const int counted = 13 + row * 101;
            row_counts[static_cast<std::size_t>(counted)] = 3;
            counts[static_cast<std::size_t>(row) * kTokenDomain + counted] = 3;
            expected_counts[static_cast<std::size_t>(row) * kTokenDomain + counted] = 3;
        }
        count_device.copy_from_host(counts.data(), counts.size() * sizeof(int));
        for (int row = 0; row < rows; ++row) {
            std::vector<int> row_counts(
                counts.begin() + static_cast<std::size_t>(row) * kTokenDomain,
                counts.begin() + static_cast<std::size_t>(row + 1) * kTokenDomain);
            const std::vector<int>* oracle_counts =
                configs[static_cast<std::size_t>(row)].token_counts == nullptr
                    ? nullptr
                    : &row_counts;
            oracle[static_cast<std::size_t>(row)] = oracle_sample(
                logits.data() + static_cast<std::size_t>(row) * kPhysicalVocabulary,
                kTokenDomain, configs[static_cast<std::size_t>(row)], oracle_counts,
                positions[static_cast<std::size_t>(row)], ninfer::ops::kSamplePurposeDecode);
            if (configs[static_cast<std::size_t>(row)].temperature > 0.0F) {
                if (oracle[static_cast<std::size_t>(row)].boundary_margin < 1.0e-5) {
                    fail("oracle fixture placed RNG draw too close to an inverse-CDF boundary");
                }
                if (oracle_counts != nullptr) {
                    ++expected_counts[static_cast<std::size_t>(row) * kTokenDomain +
                                      oracle[static_cast<std::size_t>(row)].token];
                }
            }
        }
        position_device.copy_from_host(positions.data(), positions.size() * sizeof(int));
        config_device.copy_from_host(configs.data(), configs.size() * sizeof(SamplingConfig));

        ninfer::Tensor logits_tensor(logits_device.data(), ninfer::DType::BF16,
                                     {kPhysicalVocabulary, rows});
        ninfer::Tensor out_tensor(out_device.data(), ninfer::DType::I32, {rows});
        ninfer::Tensor positions_tensor(position_device.data(), ninfer::DType::I32, {rows});
        ninfer::ops::sample(logits_tensor, out_tensor, kTokenDomain,
                            static_cast<const SamplingConfig*>(config_device.data()),
                            positions_tensor, ninfer::ops::kSamplePurposeDecode, workspace,
                            device.stream);
        device.synchronize();

        std::vector<int> actual(static_cast<std::size_t>(rows));
        std::vector<int> actual_counts(counts.size());
        out_device.copy_to_host(actual.data(), actual.size() * sizeof(int));
        count_device.copy_to_host(actual_counts.data(), actual_counts.size() * sizeof(int));
        for (int row = 0; row < rows; ++row) {
            if (actual[static_cast<std::size_t>(row)] != oracle[static_cast<std::size_t>(row)].token) {
                std::ostringstream message;
                message << "B=" << rows << " row=" << row << " sampled "
                        << actual[static_cast<std::size_t>(row)] << " but FP64 oracle selected "
                        << oracle[static_cast<std::size_t>(row)].token;
                fail(message.str());
            }
        }
        if (actual_counts != expected_counts) {
            fail("token-count publication differs from exact independent oracle at B=" +
                 std::to_string(rows));
        }
    }
}

void qualify_rng_and_small_route(ninfer::DeviceContext& device) {
    constexpr int tokens = 257;
    constexpr int rows = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> logits(static_cast<std::size_t>(tokens) * rows, bf16(-2.0F));
    std::vector<int> positions(rows);
    std::vector<ninfer::ops::SamplingConfig> configs(rows);
    std::vector<int> expected(rows);
    for (int row = 0; row < rows; ++row) {
        positions[row] = -17 + row * 1009;
        configs[row].temperature = 1.0F;
        configs[row].top_k = 2;
        configs[row].seed = 424242ULL + static_cast<unsigned long long>(row);
        logits[static_cast<std::size_t>(row) * tokens + 3] = bf16(0.0F);
        logits[static_cast<std::size_t>(row) * tokens + 7] = bf16(0.0F);
        expected[row] = oracle_sample(logits.data() + static_cast<std::size_t>(row) * tokens,
                                      tokens, configs[row], nullptr, positions[row],
                                      ninfer::ops::kSamplePurposeSpeculativeBonus).token;
    }
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_positions(positions.size() * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    ninfer::DeviceBuffer d_out(expected.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_positions.copy_from_host(positions.data(), positions.size() * sizeof(int));
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    ninfer::DeviceArena workspace(1);
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16, {tokens, rows});
    ninfer::Tensor out_tensor(d_out.data(), ninfer::DType::I32, {rows});
    ninfer::Tensor positions_tensor(d_positions.data(), ninfer::DType::I32, {rows});
    ninfer::ops::sample(logits_tensor, out_tensor, tokens,
                        static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
                        positions_tensor, ninfer::ops::kSamplePurposeSpeculativeBonus,
                        workspace, device.stream);
    device.synchronize();
    std::vector<int> actual(rows);
    d_out.copy_to_host(actual.data(), actual.size() * sizeof(int));
    if (actual != expected) { fail("small-route tie/counter-RNG result differs from FP64 oracle"); }
}

void qualify_validation(ninfer::DeviceContext& device) {
    ninfer::DeviceArena workspace(1);
    ninfer::Tensor absent(nullptr, ninfer::DType::BF16, {1, 1});
    ninfer::Tensor output(nullptr, ninfer::DType::I32, {1});
    ninfer::Tensor position(nullptr, ninfer::DType::I32, {1});
    try {
        ninfer::ops::sample(absent, output, 1, nullptr, position,
                            ninfer::ops::kSamplePurposeDecode, workspace, device.stream);
        fail("sampling accepted null device pointers");
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::sampling_workspace_capacity_bytes(
            1, 0, static_cast<int>(ninfer::kMaximumConcurrency));
        fail("sampling workspace accepted an invalid lane interval");
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::sampling_workspace_capacity_bytes(
            1, 1, static_cast<int>(ninfer::kMaximumConcurrency) + 1);
        fail("sampling workspace accepted a lane count above product concurrency");
    } catch (const std::invalid_argument&) {}
}

void benchmark(ninfer::DeviceContext& device) {
    constexpr int rows = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> logits = make_logits(rows, kPhysicalVocabulary);
    std::vector<int> positions(rows, 8192);
    std::vector<ninfer::ops::SamplingConfig> configs(rows);
    for (int row = 0; row < rows; ++row) {
        configs[row].temperature = 0.8F;
        configs[row].top_k = 20;
        configs[row].top_p = 0.9F;
        configs[row].min_p = 0.05F;
        configs[row].seed = static_cast<unsigned long long>(row + 1);
    }
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_positions(positions.size() * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    ninfer::DeviceBuffer d_out(rows * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_positions.copy_from_host(positions.data(), positions.size() * sizeof(int));
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    const std::size_t maximum_bytes =
        ninfer::ops::sampling_workspace_capacity_bytes(kTokenDomain, rows, rows);
    ninfer::DeviceArena workspace(maximum_bytes);
    constexpr int repeats = 50;
    for (int active_rows = 1; active_rows <= rows; ++active_rows) {
        ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                     {kPhysicalVocabulary, active_rows});
        ninfer::Tensor out_tensor(d_out.data(), ninfer::DType::I32, {active_rows});
        ninfer::Tensor positions_tensor(d_positions.data(), ninfer::DType::I32, {active_rows});
        for (int warmup = 0; warmup < 3; ++warmup) {
            ninfer::ops::sample(logits_tensor, out_tensor, kTokenDomain,
                                static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
                                positions_tensor, ninfer::ops::kSamplePurposeDecode,
                                workspace, device.stream);
        }
        ninfer::DeviceEventTimer timer(device);
        timer.start();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            ninfer::ops::sample(logits_tensor, out_tensor, kTokenDomain,
                                static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
                                positions_tensor, ninfer::ops::kSamplePurposeDecode,
                                workspace, device.stream);
        }
        timer.record_stop();
        device.synchronize();
        std::cout << "timing: vocab=" << kTokenDomain << " B=" << active_rows
                  << " positive_temperature_ms="
                  << timer.elapsed_ms() / static_cast<float>(repeats)
                  << " workspace_bytes="
                  << ninfer::ops::sampling_workspace_capacity_bytes(kTokenDomain, active_rows,
                                                                     active_rows)
                  << '\n';
    }
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(select_r9700());
        std::cout << "device: arch=" << device.props.gcnArchName << " wave="
                  << device.props.warpSize << " vram_mib="
                  << device.props.totalGlobalMem / (1024U * 1024U) << '\n';
        qualify_validation(device);
        qualify_rng_and_small_route(device);
        qualify_batches(device);
        benchmark(device);
        std::cout << "PASS: R9700 sampling B=1..4, exact token/count publication, FP64 probability oracle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
