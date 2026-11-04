#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/position.h"
#include "sampling_reference.h"
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
constexpr int kTokenDomain        = 248320;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::uint16_t bf16_rne_bits(float value) {
    std::uint32_t word           = std::bit_cast<std::uint32_t>(value);
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
    unsigned long long key =
        splitmix64(seed ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
                           0xD1B54A32D192ED03ULL));
    key = splitmix64(key ^
                     (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21));
    return static_cast<float>(static_cast<unsigned int>(key >> 40)) * (1.0F / 16777216.0F);
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
        if (!sampling_reference::eligible(token, config)) continue;
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
    cap = std::min(cap, static_cast<int>(candidates.size()));
    std::partial_sort(candidates.begin(), candidates.begin() + cap, candidates.end(), better);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weight(static_cast<std::size_t>(cap));
    const double maximum =
        static_cast<double>(candidates[0].first) / static_cast<double>(config.temperature);
    double total = 0.0;
    for (int index = 0; index < cap; ++index) {
        weight[static_cast<std::size_t>(index)] =
            std::exp(static_cast<double>(candidates[static_cast<std::size_t>(index)].first) /
                         static_cast<double>(config.temperature) -
                     maximum);
        total += weight[static_cast<std::size_t>(index)];
    }
    const double min_threshold =
        config.min_p > 0.0F ? static_cast<double>(config.min_p) * weight[0] : -1.0;
    const double top_p_target = static_cast<double>(config.top_p) * total;
    double cumulative         = 0.0;
    int support_size          = 0;
    for (int index = 0; index < cap; ++index) {
        if (min_threshold >= 0.0 && weight[static_cast<std::size_t>(index)] < min_threshold) {
            break;
        }
        cumulative += weight[static_cast<std::size_t>(index)];
        support_size = index + 1;
        if (config.top_p < 1.0F && cumulative >= top_p_target) { break; }
    }
    support_size         = std::max(1, support_size);
    double support_total = 0.0;
    for (int index = 0; index < support_size; ++index) {
        support_total += weight[static_cast<std::size_t>(index)];
    }
    const double goal =
        static_cast<double>(uniform(config.seed, position, purpose)) * support_total;
    cumulative    = 0.0;
    int selected  = candidates[static_cast<std::size_t>(support_size - 1)].second;
    double margin = std::numeric_limits<double>::infinity();
    OracleResult result;
    for (int index = 0; index < support_size; ++index) {
        const double previous = cumulative;
        cumulative += weight[static_cast<std::size_t>(index)];
        margin = std::min(margin, std::min(std::abs(goal - previous), std::abs(cumulative - goal)));
        result.support.push_back(candidates[static_cast<std::size_t>(index)].second);
        if (goal < cumulative) {
            selected = candidates[static_cast<std::size_t>(index)].second;
            break;
        }
    }
    result.token           = selected;
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
            column[token]   = bf16(-12.0F + static_cast<float>(noise) / 997.0F);
        }
        // Top candidates are deliberately non-contiguous, cross 512-token partials, include an
        // exact BF16 tie, and include counted tokens whose penalty changes their rank.
        for (int rank = 0; rank < 28; ++rank) {
            const int token = 13 + rank * 7919 + row * 101;
            column[token]   = bf16(8.0F - static_cast<float>(rank) * 0.125F);
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
            SamplingConfig& config                   = configs[static_cast<std::size_t>(row)];
            const bool greedy                        = rows > 1 && row == 0;
            config.temperature = greedy ? 0.0F : 0.55F + static_cast<float>(row % 3) * 0.4F;
            constexpr int top_k_cases[] = {0, 1, 7, 19, 20, 27, -3, 11};
            config.top_k                = top_k_cases[row];
            config.top_p                = (row % 3 == 0) ? 0.0F : ((row % 3 == 1) ? 0.72F : 1.0F);
            config.min_p                = (row % 4 == 2) ? 0.35F : 0.0F;
            config.presence_penalty     = greedy ? 7.0F : ((row % 3 != 1) ? 0.35F : 0.0F);
            config.frequency_penalty    = greedy ? 9.0F : ((row % 3 != 0) ? 0.2F : 0.0F);
            config.seed = 0x123456789abcdef0ULL + static_cast<unsigned long long>(row * 977 + rows);
            int* row_counts_device = static_cast<int*>(count_device.data()) +
                                     static_cast<std::size_t>(row) * kTokenDomain;
            config.token_counts =
                (rows == static_cast<int>(ninfer::kMaximumConcurrency) && row == rows - 1)
                    ? nullptr
                    : row_counts_device;
            std::vector<int> row_counts(static_cast<std::size_t>(kTokenDomain), 0);
            const int counted                                              = 13 + row * 101;
            row_counts[static_cast<std::size_t>(counted)]                  = 3;
            counts[static_cast<std::size_t>(row) * kTokenDomain + counted] = 3;
            expected_counts[static_cast<std::size_t>(row) * kTokenDomain + counted] = 3;
        }
        count_device.copy_from_host(counts.data(), counts.size() * sizeof(int));
        for (int row = 0; row < rows; ++row) {
            std::vector<int> row_counts(
                counts.begin() + static_cast<std::size_t>(row) * kTokenDomain,
                counts.begin() + static_cast<std::size_t>(row + 1) * kTokenDomain);
            const std::vector<int>* oracle_counts =
                configs[static_cast<std::size_t>(row)].token_counts == nullptr ? nullptr
                                                                               : &row_counts;
            oracle[static_cast<std::size_t>(row)] = oracle_sample(
                logits.data() + static_cast<std::size_t>(row) * kPhysicalVocabulary, kTokenDomain,
                configs[static_cast<std::size_t>(row)], oracle_counts,
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
            if (actual[static_cast<std::size_t>(row)] !=
                oracle[static_cast<std::size_t>(row)].token) {
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
    constexpr int rows   = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> logits(static_cast<std::size_t>(tokens) * rows, bf16(-2.0F));
    std::vector<int> positions(rows);
    std::vector<ninfer::ops::SamplingConfig> configs(rows);
    std::vector<int> expected(rows);
    for (int row = 0; row < rows; ++row) {
        positions[row]           = -17 + row * 1009;
        configs[row].temperature = 1.0F;
        configs[row].top_k       = 2;
        configs[row].seed        = 424242ULL + static_cast<unsigned long long>(row);
        logits[static_cast<std::size_t>(row) * tokens + 3] = bf16(0.0F);
        logits[static_cast<std::size_t>(row) * tokens + 7] = bf16(0.0F);
        expected[row] = oracle_sample(logits.data() + static_cast<std::size_t>(row) * tokens,
                                      tokens, configs[row], nullptr, positions[row],
                                      ninfer::ops::kSamplePurposeSpeculativeBonus)
                            .token;
    }
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_positions(positions.size() * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    ninfer::DeviceBuffer d_out(expected.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_positions.copy_from_host(positions.data(), positions.size() * sizeof(int));
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    ninfer::DeviceArena workspace(
        ninfer::ops::sampling_workspace_capacity_bytes(tokens, rows, rows));
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16, {tokens, rows});
    ninfer::Tensor out_tensor(d_out.data(), ninfer::DType::I32, {rows});
    ninfer::Tensor positions_tensor(d_positions.data(), ninfer::DType::I32, {rows});
    ninfer::ops::sample(logits_tensor, out_tensor, tokens,
                        static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
                        positions_tensor, ninfer::ops::kSamplePurposeSpeculativeBonus, workspace,
                        device.stream);
    device.synchronize();
    std::vector<int> actual(rows);
    d_out.copy_to_host(actual.data(), actual.size() * sizeof(int));
    if (actual != expected) { fail("small-route tie/counter-RNG result differs from FP64 oracle"); }
}

void qualify_p_less(ninfer::DeviceContext& device) {
    // Probability is the public contract, not a particular inverse-CDF traversal/RNG mapping.
    // Compare support and empirical frequencies to the full-domain FP64 oracle. Exact graph
    // replay for fixed public inputs is checked separately inside each case.
    constexpr int rows  = 4;
    constexpr int draws = 256;
    for (int tokens : {127, 257, kTokenDomain}) {
        for (int fixture = 0; fixture < 5; ++fixture) {
            std::vector<hip_bfloat16> logits(static_cast<std::size_t>(tokens) * rows, bf16(-24));
            std::vector<std::uint32_t> masks(static_cast<std::size_t>((tokens + 31) / 32) * rows,
                                             ~std::uint32_t{0});
            const int words = (tokens + 31) / 32;
            std::vector<ninfer::ops::SamplingConfig> configs(rows);
            std::vector<sampling_reference::Distribution> references;
            std::vector<int> positions(rows), actual(rows), previous(rows);
            std::vector<int> histogram(static_cast<std::size_t>(tokens) * rows);
            for (int row = 0; row < rows; ++row) {
                auto* column = logits.data() + static_cast<std::size_t>(row) * tokens;
                // Wide support proves no inherited top-20 truncation. Two near-equal atoms
                // distinguish the epsilon-relaxed gate from the old collision-only rule.
                if (fixture == 0 || fixture == 4)
                    for (int i = 0; i < 64; ++i) column[i] = bf16(0);
                if (fixture == 1) {
                    column[3] = bf16(0.125F);
                    column[7] = bf16(0);
                }
                if (fixture == 2) {
                    column[3] = bf16(12);
                    column[7] = bf16(0);
                }
                if (fixture == 3) std::fill(column, column + tokens, bf16(0));
                auto& cfg             = configs[row];
                cfg.temperature       = 1;
                cfg.p_less            = 1;
                cfg.top_k             = 1;
                cfg.top_p             = 0.01F;
                cfg.min_p             = 0.99F;
                cfg.presence_penalty  = 100;
                cfg.frequency_penalty = 100;
                cfg.seed              = 0x13579ULL + row;
                if (fixture == 2 || fixture == 3) cfg.typical_exclude = fixture == 2 ? 3 : 0;
                if (fixture == 4) {
                    cfg.suppressed_token_count = 2;
                    cfg.suppressed_tokens[0]   = 0;
                    cfg.suppressed_tokens[1]   = 1;
                    for (int i = 0; i < tokens; ++i)
                        if ((i + row) % 3 == 0) masks[row * words + i / 32] &= ~(1U << (i % 32));
                    cfg.typical_exclude     = 2;
                    cfg.allowed_token_words = masks.data() + row * words;
                }
                std::vector<double> represented(tokens);
                for (int i = 0; i < tokens; ++i) represented[i] = bf16_to_float(column[i]);
                references.push_back(sampling_reference::p_less(represented, cfg));
            }
            ninfer::DeviceBuffer d_logits(logits.size() * sizeof(logits[0]));
            ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
            ninfer::DeviceBuffer d_masks(masks.size() * sizeof(masks[0]));
            ninfer::DeviceBuffer d_counts(histogram.size() * sizeof(int));
            ninfer::DeviceBuffer d_positions(rows * sizeof(int)), d_out(rows * sizeof(int));
            d_logits.copy_from_host(logits.data(), logits.size() * sizeof(logits[0]));
            d_masks.copy_from_host(masks.data(), masks.size() * sizeof(masks[0]));
            d_counts.copy_from_host(histogram.data(), histogram.size() * sizeof(int));
            for (int row = 0; row < rows; ++row) {
                configs[row].token_counts = static_cast<int*>(d_counts.data()) + row * tokens;
                if (fixture == 4)
                    configs[row].allowed_token_words =
                        static_cast<std::uint32_t*>(d_masks.data()) + row * words;
            }
            d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
            ninfer::Tensor input(d_logits.data(), ninfer::DType::BF16, {tokens, rows});
            ninfer::Tensor output(d_out.data(), ninfer::DType::I32, {rows});
            ninfer::Tensor position(d_positions.data(), ninfer::DType::I32, {rows});
            ninfer::DeviceArena workspace(std::max<std::size_t>(
                1, ninfer::ops::sampling_workspace_capacity_bytes(tokens, rows, rows)));
            const auto launch = [&] {
                ninfer::ops::sample(
                    input, output, tokens,
                    static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), position,
                    ninfer::ops::kSamplePurposeDecode, workspace, device.stream);
            };
            hipGraph_t graph          = nullptr;
            hipGraphExec_t executable = nullptr;
            HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
            launch();
            HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
            HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int draw = 0; draw < draws; ++draw) {
                for (int row = 0; row < rows; ++row) positions[row] = 1009 * draw + row;
                d_positions.copy_from_host(positions.data(), positions.size() * sizeof(int));
                HIP_CHECK(hipMemsetAsync(d_out.data(), 0xff, rows * sizeof(int), device.stream));
                if (draw % 2 == 0)
                    launch();
                else
                    HIP_CHECK(hipGraphLaunch(executable, device.stream));
                device.synchronize();
                d_out.copy_to_host(actual.data(), actual.size() * sizeof(int));
                for (int row = 0; row < rows; ++row) {
                    const auto& reference = references[row];
                    if (std::find(reference.ids.begin(), reference.ids.end(), actual[row]) ==
                        reference.ids.end())
                        fail("p-less emitted outside independent support, V=" +
                             std::to_string(tokens) + " fixture=" + std::to_string(fixture));
                    ++histogram[row * tokens + actual[row]];
                }
                if (draw == 0) {
                    previous = actual;
                    HIP_CHECK(
                        hipMemsetAsync(d_out.data(), 0xff, rows * sizeof(int), device.stream));
                    HIP_CHECK(hipGraphLaunch(executable, device.stream));
                    device.synchronize();
                    d_out.copy_to_host(actual.data(), actual.size() * sizeof(int));
                    if (actual != previous) fail("p-less eager/graph fixed-input replay differs");
                    for (int row = 0; row < rows; ++row) ++histogram[row * tokens + actual[row]];
                }
            }
            HIP_CHECK(hipGraphExecDestroy(executable));
            HIP_CHECK(hipGraphDestroy(graph));
            std::vector<int> actual_counts(histogram.size());
            d_counts.copy_to_host(actual_counts.data(), actual_counts.size() * sizeof(int));
            if (actual_counts != histogram)
                fail("p-less token counts not exactly once per publication");
            for (int row = 0; row < rows; ++row) {
                const auto& reference = references[row];
                int seen              = 0;
                for (std::size_t i = 0; i < reference.ids.size(); ++i) {
                    const double p     = reference.probabilities[i];
                    const double mean  = (draws + 1) * p;
                    const int observed = histogram[row * tokens + reference.ids[i]];
                    seen += observed > 0;
                    if (std::abs(observed - mean) > 7 * std::sqrt((draws + 1) * p * (1 - p)) + 2)
                        fail("p-less frequencies disagree with independent FP64 distribution");
                }
                if (fixture == 0 && seen <= 20) fail("p-less retained obsolete top-20 cap");
            }
        }
    }
}

void qualify_nonfinite_status(ninfer::DeviceContext& device) {
    constexpr int rows = 8, physical_padding = 3;
    for (const int tokens : {127, kTokenDomain}) {
        const int physical = tokens + physical_padding, words = (tokens + 31) / 32;
        std::vector<hip_bfloat16> finite(static_cast<std::size_t>(physical) * rows, bf16(NAN));
        std::vector<std::uint32_t> masks(static_cast<std::size_t>(words) * rows, 0U);
        for (int row = 0; row < rows; ++row) {
            masks[row * words]         = 1U | (1U << 3) | (1U << 7);
            finite[row * physical + 3] = bf16(0.0F);
            finite[row * physical + 7] = bf16(1.0F);
        }
        ninfer::DeviceBuffer dl(finite.size() * sizeof(finite[0]));
        ninfer::DeviceBuffer dm(masks.size() * sizeof(masks[0]));
        ninfer::DeviceBuffer ds(rows * sizeof(std::uint32_t));
        ninfer::DeviceBuffer dc(rows * sizeof(ninfer::ops::SamplingConfig));
        ninfer::DeviceBuffer da((rows / 2) * sizeof(ninfer::ops::SamplingConfig));
        ninfer::DeviceBuffer dp(rows * sizeof(int)), dout(rows * sizeof(int));
        dm.copy_from_host(masks.data(), dm.size());
        std::vector<int> positions(rows, 17);
        dp.copy_from_host(positions.data(), dp.size());
        std::vector<ninfer::ops::SamplingConfig> configs(rows), argmax_configs(rows / 2);
        for (int row = 0; row < rows; ++row) {
            auto& cfg                  = configs[row];
            cfg.temperature            = row % 3 == 0 ? 0.0F : 1.0F;
            cfg.p_less                 = row % 3 == 2;
            cfg.top_k                  = 2;
            cfg.suppressed_token_count = 1;
            cfg.suppressed_tokens[0]   = 0;
            cfg.allowed_token_words    = static_cast<std::uint32_t*>(dm.data()) + row * words;
            cfg.nonfinite_logits       = static_cast<std::uint32_t*>(ds.data()) + row;
            if (row % 2 == 0) {
                argmax_configs[row / 2]                             = cfg;
                argmax_configs[row / 2].allowed_token_column_stride = words;
            }
        }
        dc.copy_from_host(configs.data(), dc.size());
        da.copy_from_host(argmax_configs.data(), da.size());
        ninfer::Tensor input(dl.data(), ninfer::DType::BF16, {physical, rows});
        ninfer::Tensor output(dout.data(), ninfer::DType::I32, {rows});
        ninfer::Tensor position(dp.data(), ninfer::DType::I32, {rows});
        ninfer::DeviceArena workspace(std::max<std::size_t>(
            1, ninfer::ops::sampling_workspace_capacity_bytes(tokens, rows, rows)));
        const auto sample = [&] {
            ninfer::ops::sample(
                input, output, tokens, static_cast<const ninfer::ops::SamplingConfig*>(dc.data()),
                position, ninfer::ops::kSamplePurposeDecode, workspace, device.stream);
        };
        hipGraph_t graph          = nullptr;
        hipGraphExec_t executable = nullptr;
        HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
        sample();
        HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        const auto check = [&](bool dirty) {
            device.synchronize();
            std::vector<std::uint32_t> actual(rows);
            ds.copy_to_host(actual.data(), ds.size());
            for (int row = 0; row < rows; ++row)
                if (actual[row] != (dirty ? (row < 6 ? 3U : 2U) : 0U))
                    fail("nonfinite status or configured-column offset mismatch at V=" +
                         std::to_string(tokens) + " row=" + std::to_string(row));
            std::vector<int> ids(rows);
            dout.copy_to_host(ids.data(), dout.size());
            for (const int id : ids)
                if (id < 0 || id >= tokens) fail("reported nonfinite logits yielded unsafe id");
        };
        for (const float poison : {NAN, INFINITY, -INFINITY}) {
            for (const bool all : {false, true}) {
                auto dirty = finite;
                for (int row = 0; row < 6; ++row) {
                    dirty[row * physical + 3] = bf16(poison);
                    if (all) dirty[row * physical + 7] = bf16(poison);
                }
                dl.copy_from_host(dirty.data(), dl.size());
                const std::vector<std::uint32_t> initial(rows, 2U);
                ds.copy_from_host(initial.data(), ds.size());
                HIP_CHECK(hipGraphLaunch(executable, device.stream));
                check(true); // status is sticky OR: preserve an unrelated caller-owned bit.
                ds.copy_from_host(initial.data(), ds.size());
                ninfer::ops::argmax(input, output, tokens,
                                    static_cast<const ninfer::ops::SamplingConfig*>(da.data()), 2,
                                    device.stream);
                check(true);
                // Unconfigured MTP proposal argmax also feeds an embedding gather before the
                // host can observe target failure. Its all-NaN fallback must remain in bounds.
                ninfer::ops::argmax(input, output, tokens, device.stream);
                device.synchronize();
                std::vector<int> proposal_ids(rows);
                dout.copy_to_host(proposal_ids.data(), dout.size());
                for (const int id : proposal_ids)
                    if (id < 0 || id >= tokens) fail("nonfinite proposal argmax yielded unsafe id");
                dl.copy_from_host(finite.data(), dl.size());
                HIP_CHECK(hipMemsetAsync(ds.data(), 0, ds.size(), device.stream));
                HIP_CHECK(hipGraphLaunch(executable, device.stream));
                check(false); // masked NaNs, physical padding, and prior failure stay inert.
            }
        }
        HIP_CHECK(hipGraphExecDestroy(executable));
        HIP_CHECK(hipGraphDestroy(graph));
    }
    std::cout << "nonfinite-logit status: modes/column-offsets/masked-padding/replay PASS\n";
}

void qualify_low_positive_temperature(ninfer::DeviceContext& device) {
    // The public temperature interval includes subnormal positive FP32 values. Check the
    // distribution, rather than a private inverse-CDF traversal, with NaNs outside eligibility.
    constexpr int rows = 6, draws = 256;
    for (const int tokens : {127, kTokenDomain}) {
        const int words = (tokens + 31) / 32;
        std::vector<hip_bfloat16> logits(static_cast<std::size_t>(tokens) * rows, bf16(NAN));
        std::vector<std::uint32_t> masks(static_cast<std::size_t>(words) * rows, 0U);
        std::vector<ninfer::ops::SamplingConfig> configs(rows);
        std::vector<sampling_reference::Distribution> references;
        for (int row = 0; row < rows; ++row) {
            auto& cfg               = configs[row];
            cfg.temperature         = row < 2 ? 1.0e-38F : 1.0e-40F;
            cfg.p_less              = row % 2;
            cfg.top_k               = 2;
            cfg.seed                = 0x517cc1b727220a95ULL + row;
            cfg.allowed_token_words = masks.data() + row * words;
            const int low = 3, high = tokens - 4;
            masks[row * words + low / 32] |= 1U << (low % 32);
            masks[row * words + high / 32] |= 1U << (high % 32);
            auto* column = logits.data() + static_cast<std::size_t>(row) * tokens;
            column[low]  = bf16(row < 4 ? 8.0F : 0.0F);
            column[high] = column[low];
            // One BF16 subnormal step is comparable to T: replacing an overflowing reciprocal
            // with infinity and special-casing ties would still destroy this nonzero ratio.
            if (row >= 4) column[high].data = 1U;
            std::vector<double> represented(tokens);
            for (int token = 0; token < tokens; ++token)
                represented[token] = bf16_to_float(column[token]);
            // Both fixtures retain both atoms, so the independent full-softmax oracle also
            // describes the unfiltered top-k=2 distribution.
            references.push_back(sampling_reference::p_less(represented, cfg));
            if (references.back().ids.size() != 2U) fail("low-temperature oracle lost an atom");
        }
        ninfer::DeviceBuffer dl(logits.size() * sizeof(logits[0]));
        ninfer::DeviceBuffer dm(masks.size() * sizeof(masks[0]));
        ninfer::DeviceBuffer dc(configs.size() * sizeof(configs[0]));
        ninfer::DeviceBuffer dp(rows * sizeof(int)), dout(rows * sizeof(int));
        dl.copy_from_host(logits.data(), logits.size() * sizeof(logits[0]));
        dm.copy_from_host(masks.data(), masks.size() * sizeof(masks[0]));
        for (int row = 0; row < rows; ++row)
            configs[row].allowed_token_words = static_cast<std::uint32_t*>(dm.data()) + row * words;
        dc.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
        ninfer::Tensor input(dl.data(), ninfer::DType::BF16, {tokens, rows});
        ninfer::Tensor output(dout.data(), ninfer::DType::I32, {rows});
        ninfer::Tensor position(dp.data(), ninfer::DType::I32, {rows});
        ninfer::DeviceArena workspace(std::max<std::size_t>(
            1, ninfer::ops::sampling_workspace_capacity_bytes(tokens, rows, rows)));
        std::vector<int> positions(rows), actual(rows), first_count(rows);
        for (int draw = 0; draw < draws; ++draw) {
            for (int row = 0; row < rows; ++row) positions[row] = 1009 * draw + row;
            dp.copy_from_host(positions.data(), positions.size() * sizeof(int));
            ninfer::ops::sample(
                input, output, tokens, static_cast<const ninfer::ops::SamplingConfig*>(dc.data()),
                position, ninfer::ops::kSamplePurposeDecode, workspace, device.stream);
            device.synchronize();
            dout.copy_to_host(actual.data(), actual.size() * sizeof(int));
            for (int row = 0; row < rows; ++row) {
                const auto& reference = references[row];
                if (actual[row] != reference.ids[0] && actual[row] != reference.ids[1])
                    fail("low-temperature sample escaped eligible support");
                first_count[row] += actual[row] == reference.ids[0];
            }
        }
        for (int row = 0; row < rows; ++row) {
            const double p         = references[row].probabilities[0];
            const double tolerance = 7.0 * std::sqrt(draws * p * (1.0 - p)) + 2.0;
            if (std::abs(first_count[row] - draws * p) > tolerance)
                fail("low-temperature frequencies disagree with FP64 oracle: V=" +
                     std::to_string(tokens) + " row=" + std::to_string(row));
        }
    }
    std::cout << "low-positive-temperature: finite/subnormal logits and masked NaNs PASS\n";
}

void qualify_eligibility_and_positions(ninfer::DeviceContext& device) {
    constexpr int tokens = 67, columns = 8, rows = 4, per_row = 2;
    std::vector<hip_bfloat16> logits(tokens * columns, bf16(-1));
    std::vector<std::uint32_t> masks(rows * per_row * 3, 0);
    std::vector<ninfer::ops::SamplingConfig> configs(rows);
    std::vector<int> expected(columns);
    for (int column = 0; column < columns; ++column) {
        expected[column] = 5 + column;
        masks[column * 3 + expected[column] / 32] |= 1U << (expected[column] % 32);
        masks[column * 3] |= 1U;
        logits[column * tokens]                               = bf16(100);
        configs[column / per_row].suppressed_token_count      = 1;
        configs[column / per_row].suppressed_tokens[0]        = 0;
        configs[column / per_row].allowed_token_column_stride = 3;
    }
    ninfer::DeviceBuffer dl(logits.size() * sizeof(logits[0])), dm(masks.size() * sizeof(masks[0]));
    ninfer::DeviceBuffer dc(configs.size() * sizeof(configs[0])), dout(columns * sizeof(int));
    dl.copy_from_host(logits.data(), logits.size() * sizeof(logits[0]));
    dm.copy_from_host(masks.data(), masks.size() * sizeof(masks[0]));
    for (int row = 0; row < rows; ++row)
        configs[row].allowed_token_words =
            static_cast<std::uint32_t*>(dm.data()) + row * per_row * 3;
    dc.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    ninfer::Tensor input(dl.data(), ninfer::DType::BF16, {tokens, columns});
    ninfer::Tensor output(dout.data(), ninfer::DType::I32, {columns});
    ninfer::ops::argmax(input, output, tokens,
                        static_cast<const ninfer::ops::SamplingConfig*>(dc.data()), per_row,
                        device.stream);
    device.synchronize();
    std::vector<int> actual(columns);
    dout.copy_to_host(actual.data(), actual.size() * sizeof(int));
    if (actual != expected) fail("configured argmax lost suppression or per-column eligibility");
    // Ordinary truncated sampling must obey the same eligibility domain (p-less is off).
    for (auto& config : configs) {
        config.temperature = 0.8F;
        config.top_k       = 20;
        config.top_p       = 0.9F;
        config.min_p       = 0.05F;
    }
    dc.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    std::vector<int> sample_positions(rows, 32);
    ninfer::DeviceBuffer dp(rows * sizeof(int));
    dp.copy_from_host(sample_positions.data(), sample_positions.size() * sizeof(int));
    ninfer::Tensor sample_input(dl.data(), ninfer::DType::BF16, {tokens, rows});
    ninfer::Tensor sample_output(dout.data(), ninfer::DType::I32, {rows});
    ninfer::Tensor positions(dp.data(), ninfer::DType::I32, {rows});
    ninfer::DeviceArena sample_workspace(std::max<std::size_t>(
        1, ninfer::ops::sampling_workspace_capacity_bytes(tokens, rows, rows)));
    ninfer::ops::sample(sample_input, sample_output, tokens,
                        static_cast<const ninfer::ops::SamplingConfig*>(dc.data()), positions,
                        ninfer::ops::kSamplePurposeDecode, sample_workspace, device.stream);
    device.synchronize();
    dout.copy_to_host(actual.data(), rows * sizeof(int));
    for (int row = 0; row < rows; ++row)
        if (actual[row] != expected[row * per_row])
            fail("truncated sampler lost eligibility/suppression");

    constexpr int width = 259;
    std::vector<int> source(width * rows), delta{19, -23, 0, 111}, shifted(source.size());
    for (int row = 0; row < rows; ++row)
        for (int j = 0; j < width; ++j) {
            source[row * width + j]  = 1000 * row + j;
            shifted[row * width + j] = source[row * width + j] + delta[row];
        }
    ninfer::DeviceBuffer ds(source.size() * sizeof(int)), dd(delta.size() * sizeof(int));
    ds.copy_from_host(source.data(), source.size() * sizeof(int));
    dd.copy_from_host(delta.data(), delta.size() * sizeof(int));
    ninfer::Tensor s(ds.data(), ninfer::DType::I32, {width, rows});
    ninfer::Tensor d(dd.data(), ninfer::DType::I32, {rows});
    hipGraph_t graph          = nullptr;
    hipGraphExec_t executable = nullptr;
    HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
    ninfer::ops::offset_i32_position_rows(s, d, s, device.stream);
    HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        ds.copy_from_host(source.data(), source.size() * sizeof(int));
        if (replay == 0)
            ninfer::ops::offset_i32_position_rows(s, d, s, device.stream);
        else
            HIP_CHECK(hipGraphLaunch(executable, device.stream));
        device.synchronize();
        std::vector<int> result(source.size());
        ds.copy_to_host(result.data(), result.size() * sizeof(int));
        if (result != shifted)
            fail("row-position transform differs from independent integer oracle");
    }
    HIP_CHECK(hipGraphExecDestroy(executable));
    HIP_CHECK(hipGraphDestroy(graph));
}

void qualify_validation(ninfer::DeviceContext& device) {
    ninfer::DeviceArena workspace(1);
    ninfer::Tensor absent(nullptr, ninfer::DType::BF16, {1, 1});
    ninfer::Tensor output(nullptr, ninfer::DType::I32, {1});
    ninfer::Tensor position(nullptr, ninfer::DType::I32, {1});
    try {
        ninfer::ops::sample(absent, output, 1, nullptr, position, ninfer::ops::kSamplePurposeDecode,
                            workspace, device.stream);
        fail("sampling accepted null device pointers");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::sampling_workspace_capacity_bytes(
            1, 0, static_cast<int>(ninfer::kMaximumConcurrency));
        fail("sampling workspace accepted an invalid lane interval");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::sampling_workspace_capacity_bytes(
            1, 1, static_cast<int>(ninfer::kMaximumConcurrency) + 1);
        fail("sampling workspace accepted a lane count above product concurrency");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
}

// Public greedy selection at ragged cross-tile domains. Logical maxima are
// checked independently; failed rows only promise a sticky status and safe id.
void qualify_greedy_replay(ninfer::DeviceContext& device) {
    for (const int tokens : {257, 12801, 248077}) {
        const int physical = tokens + 3, words = (tokens + 31) / 32;
        for (int rows = 1; rows <= 8; ++rows) {
            const std::size_t elements = static_cast<std::size_t>(physical) * rows;
            std::vector<hip_bfloat16> logits(elements, bf16(NAN));
            std::vector<std::uint32_t> masks(static_cast<std::size_t>(words) * rows, 0U);
            std::vector<int> counts(static_cast<std::size_t>(tokens) * rows, 3);
            std::vector<int> positions(rows, 811);
            std::vector<ninfer::ops::SamplingConfig> configs(rows);
            ninfer::DeviceBuffer dl(elements * sizeof(logits[0]));
            ninfer::DeviceBuffer dm(masks.size() * sizeof(masks[0]));
            ninfer::DeviceBuffer dh(counts.size() * sizeof(counts[0]));
            ninfer::DeviceBuffer ds(rows * sizeof(std::uint32_t));
            ninfer::DeviceBuffer dc(rows * sizeof(configs[0]));
            ninfer::DeviceBuffer dp(rows * sizeof(int)), dout(rows * sizeof(int));
            const auto bytes = ninfer::ops::sampling_workspace_capacity_bytes(tokens, rows, rows);
            ninfer::DeviceBuffer workspace(bytes);
            ninfer::DeviceArena arena(ninfer::DeviceSpan{workspace.data(), bytes});
            for (int row = 0; row < rows; ++row) {
                masks[row * words] = 1U | (1U << 7);
                masks[row * words + (tokens - 1) / 32] |= 1U << ((tokens - 1) % 32);
                auto& cfg                  = configs[row];
                cfg.temperature            = 0.0F;
                cfg.p_less                 = 1; // Positive-temperature gate must keep this greedy.
                cfg.presence_penalty       = 7.0F;
                cfg.frequency_penalty      = 9.0F;
                cfg.seed                   = 0x901324ULL + row;
                cfg.token_counts           = static_cast<int*>(dh.data()) + row * tokens;
                cfg.suppressed_token_count = 1;
                cfg.suppressed_tokens[0]   = 0;
                cfg.allowed_token_words    = static_cast<std::uint32_t*>(dm.data()) + row * words;
                cfg.nonfinite_logits       = static_cast<std::uint32_t*>(ds.data()) + row;
            }
            dm.copy_from_host(masks.data(), dm.size());
            dh.copy_from_host(counts.data(), dh.size());
            dc.copy_from_host(configs.data(), dc.size());
            dp.copy_from_host(positions.data(), dp.size());
            ninfer::Tensor input(dl.data(), ninfer::DType::BF16, {physical, rows});
            ninfer::Tensor output(dout.data(), ninfer::DType::I32, {rows});
            ninfer::Tensor position(dp.data(), ninfer::DType::I32, {rows});
            {
                const auto launch = [&] {
                    ninfer::ops::sample(input, output, tokens,
                                        static_cast<const ninfer::ops::SamplingConfig*>(dc.data()),
                                        position, ninfer::ops::kSamplePurposeDecode, arena,
                                        device.stream);
                };
                hipGraph_t graph          = nullptr;
                hipGraphExec_t executable = nullptr;
                HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
                launch();
                HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
                HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                // Include cross-tile ties and signed zero, isolated/all-eligible
                // NaN/Inf, and recovery with dirty workspace after every failure.
                for (int scenario = 0; scenario < 8; ++scenario) {
                    const bool dirty = scenario >= 1 && scenario <= 6;
                    for (int row = 0; row < rows; ++row) {
                        const float value                   = row % 2 == 0 ? 0.0F : 5.0F;
                        logits[row * physical]              = bf16(NAN); // Explicitly suppressed.
                        logits[row * physical + 7]          = bf16(value);
                        logits[row * physical + tokens - 1] = bf16(row % 2 == 0 ? -0.0F : value);
                        if (dirty) {
                            const float poison         = scenario <= 2   ? NAN
                                                         : scenario <= 4 ? INFINITY
                                                                         : -INFINITY;
                            logits[row * physical + 7] = bf16(poison);
                            if (scenario % 2 == 0)
                                logits[row * physical + tokens - 1] = bf16(poison);
                        }
                    }
                    dl.copy_from_host(logits.data(), dl.size());
                    // Unrelated status bits remain caller-owned; workspaces do
                    // not depend on a prior invocation or graph capture writes.
                    const std::vector<std::uint32_t> initial(rows, 2U);
                    for (int replay = 0; replay < 2; ++replay) {
                        ds.copy_from_host(initial.data(), ds.size());
                        workspace.fill(0xa5);
                        dout.fill(0xff);
                        if (replay == 0)
                            launch();
                        else
                            HIP_CHECK(hipGraphLaunch(executable, device.stream));
                        device.synchronize();
                        std::vector<int> actual(rows), observed_counts(counts.size());
                        std::vector<std::uint32_t> status(rows);
                        dout.copy_to_host(actual.data(), dout.size());
                        ds.copy_to_host(status.data(), ds.size());
                        dh.copy_to_host(observed_counts.data(), dh.size());
                        for (int row = 0; row < rows; ++row) {
                            if (status[row] != (dirty ? 3U : 2U)) fail("greedy status isolation");
                            if (actual[row] < 0 || actual[row] >= tokens) fail("greedy unsafe id");
                            if (!dirty && actual[row] != 7) fail("greedy exact lower-id tie");
                        }
                        if (observed_counts != counts) fail("greedy unexpectedly updated counts");
                    }
                }
                HIP_CHECK(hipGraphExecDestroy(executable));
                HIP_CHECK(hipGraphDestroy(graph));
            }
            // Supplement the independent greedy checks with fixed-input mixed
            // replay parity. Both positive-temperature paths must keep their RNG
            // draws and exactly-once count transitions when greedy rows coexist.
            for (int row = 0; row < rows; ++row) {
                configs[row].temperature = row % 3 == 0 ? 0.0F : 0.8F;
                configs[row].p_less      = row % 3 == 2;
                configs[row].top_k       = 2;
            }
            dc.copy_from_host(configs.data(), dc.size());
            std::vector<int> control;
            {
                hipGraph_t graph          = nullptr;
                hipGraphExec_t executable = nullptr;
                const auto launch         = [&] {
                    ninfer::ops::sample(input, output, tokens,
                                        static_cast<const ninfer::ops::SamplingConfig*>(dc.data()),
                                        position, ninfer::ops::kSamplePurposeDecode, arena,
                                        device.stream);
                };
                HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
                launch();
                HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
                HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                for (int replay = 0; replay < 2; ++replay) {
                    dh.copy_from_host(counts.data(), dh.size());
                    ds.fill();
                    workspace.fill(0xa5);
                    if (replay == 0)
                        launch();
                    else
                        HIP_CHECK(hipGraphLaunch(executable, device.stream));
                    device.synchronize();
                    std::vector<int> actual(rows), observed_counts(counts.size());
                    std::vector<std::uint32_t> status(rows);
                    dout.copy_to_host(actual.data(), dout.size());
                    dh.copy_to_host(observed_counts.data(), dh.size());
                    ds.copy_to_host(status.data(), ds.size());
                    if (control.empty()) control = actual;
                    if (actual != control) fail("mixed-row greedy RNG replay mismatch");
                    auto expected_counts = counts;
                    for (int row = 0; row < rows; ++row) {
                        if (status[row] != 0U) fail("mixed-row greedy false nonfinite");
                        if (actual[row] != 7 && actual[row] != tokens - 1)
                            fail("mixed-row greedy eligibility mismatch");
                        if (row % 3 == 0 && actual[row] != 7)
                            fail("mixed-row greedy exact tie mismatch");
                        if (row % 3 != 0) ++expected_counts[row * tokens + actual[row]];
                    }
                    if (observed_counts != expected_counts)
                        fail("mixed-row greedy count publication mismatch");
                }
                HIP_CHECK(hipGraphExecDestroy(executable));
                HIP_CHECK(hipGraphDestroy(graph));
            }
        }
    }
    std::cout << "greedy final reduction: "
                 "B1..8/ragged/ties/masks/nonfinite/counts/dirty-workspace/eager+graph PASS\n";
}

void benchmark(ninfer::DeviceContext& device) {
    constexpr int rows               = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> logits = make_logits(rows, kPhysicalVocabulary);
    std::vector<int> positions(rows, 8192);
    std::vector<ninfer::ops::SamplingConfig> configs(rows);
    for (int row = 0; row < rows; ++row) {
        configs[row].temperature = 0.8F;
        configs[row].top_k       = 20;
        configs[row].top_p       = 0.9F;
        configs[row].min_p       = 0.05F;
        configs[row].seed        = static_cast<unsigned long long>(row + 1);
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
                                positions_tensor, ninfer::ops::kSamplePurposeDecode, workspace,
                                device.stream);
        }
        ninfer::DeviceEventTimer timer(device);
        timer.start();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            ninfer::ops::sample(logits_tensor, out_tensor, kTokenDomain,
                                static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
                                positions_tensor, ninfer::ops::kSamplePurposeDecode, workspace,
                                device.stream);
        }
        timer.record_stop();
        device.synchronize();
        std::cout << "timing: vocab=" << kTokenDomain << " B=" << active_rows
                  << " positive_temperature_ms=" << timer.elapsed_ms() / static_cast<float>(repeats)
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
        std::cout << "device: arch=" << device.props.gcnArchName
                  << " wave=" << device.props.warpSize
                  << " vram_mib=" << device.props.totalGlobalMem / (1024U * 1024U) << '\n';
        qualify_validation(device);
        qualify_greedy_replay(device);
        qualify_nonfinite_status(device);
        qualify_low_positive_temperature(device);
        qualify_rng_and_small_route(device);
        qualify_batches(device);
        qualify_p_less(device);
        qualify_eligibility_and_positions(device);
        benchmark(device);
        std::cout << "PASS: R9700 sampling B=1..8, exact token/count publication, FP64 probability "
                     "oracle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
