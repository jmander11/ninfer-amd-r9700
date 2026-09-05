#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kRealVocabulary = 248320;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void require(bool condition, const std::string& message) {
    if (!condition) { fail(message); }
}

template <typename T>
void require_equal(const std::vector<T>& actual, const std::vector<T>& expected,
                   const std::string& label) {
    if (actual == expected) { return; }
    std::size_t index = 0;
    while (index < actual.size() && index < expected.size() && actual[index] == expected[index]) {
        ++index;
    }
    std::string detail;
    if (index < actual.size() && index < expected.size()) {
        detail = " actual=" + std::to_string(static_cast<long long>(actual[index])) +
                 " expected=" + std::to_string(static_cast<long long>(expected[index]));
    }
    fail(label + " differs at index " + std::to_string(index) + detail);
}

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

hip_bfloat16 bf16(float value) {
    hip_bfloat16 result;
    result.data = bf16_rne_bits(value);
    return result;
}

float represented(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<hip_bfloat16>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[index].data;
    }
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

template <typename T>
std::vector<T> copy_from(const ninfer::DeviceBuffer& buffer, std::size_t count) {
    std::vector<T> result(count);
    buffer.copy_to_host(result.data(), result.size() * sizeof(T));
    return result;
}

struct Distribution {
    std::vector<int> ids;
    std::vector<double> probabilities;
};

Distribution distribution_oracle(const hip_bfloat16* logits, int token_domain,
                                 const ninfer::ops::SamplingConfig& config,
                                 const std::vector<int>* counts,
                                 const std::vector<int>& overlay) {
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        float value = represented(logits[token]);
        if (config.presence_penalty != 0.0F || config.frequency_penalty != 0.0F) {
            int count = counts == nullptr ? 0 : (*counts)[static_cast<std::size_t>(token)];
            count += static_cast<int>(std::count(overlay.begin(), overlay.end(), token));
            if (count > 0) { value -= config.presence_penalty; }
            value -= config.frequency_penalty * static_cast<float>(count);
        }
        candidates.emplace_back(value, token);
    }
    const auto better = [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    };
    int cap = 20;
    if (config.top_k > 0 && config.top_k < cap) { cap = config.top_k; }
    cap = std::min(cap, token_domain);
    std::partial_sort(candidates.begin(), candidates.begin() + cap, candidates.end(), better);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weights(static_cast<std::size_t>(cap));
    const double maximum = static_cast<double>(candidates[0].first) /
                           static_cast<double>(config.temperature);
    double total = 0.0;
    for (int index = 0; index < cap; ++index) {
        weights[static_cast<std::size_t>(index)] =
            std::exp(static_cast<double>(candidates[static_cast<std::size_t>(index)].first) /
                         static_cast<double>(config.temperature) -
                     maximum);
        total += weights[static_cast<std::size_t>(index)];
    }
    const double min_threshold = config.min_p > 0.0F
                                     ? static_cast<double>(config.min_p) * weights[0]
                                     : -1.0;
    const double top_p_target = static_cast<double>(config.top_p) * total;
    double cumulative = 0.0;
    int support = 0;
    for (int index = 0; index < cap; ++index) {
        if (min_threshold >= 0.0 && weights[static_cast<std::size_t>(index)] < min_threshold) {
            break;
        }
        cumulative += weights[static_cast<std::size_t>(index)];
        support = index + 1;
        if (config.top_p < 1.0F && cumulative >= top_p_target) { break; }
    }
    support = std::max(1, support);
    double support_total = 0.0;
    for (int index = 0; index < support; ++index) {
        support_total += weights[static_cast<std::size_t>(index)];
    }
    Distribution result;
    result.ids.reserve(static_cast<std::size_t>(support));
    result.probabilities.reserve(static_cast<std::size_t>(support));
    for (int index = 0; index < support; ++index) {
        result.ids.push_back(candidates[static_cast<std::size_t>(index)].second);
        result.probabilities.push_back(weights[static_cast<std::size_t>(index)] / support_total);
    }
    return result;
}

int pick(const Distribution& distribution, int excluded, float draw) {
    double mass = 0.0;
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        if (distribution.ids[index] != excluded) { mass += distribution.probabilities[index]; }
    }
    if (!(mass > 0.0)) { return distribution.ids.front(); }
    const double target = static_cast<double>(draw) * mass;
    double cumulative = 0.0;
    int selected = distribution.ids.front();
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        if (distribution.ids[index] == excluded) { continue; }
        cumulative += distribution.probabilities[index];
        selected = distribution.ids[index];
        if (target < cumulative) { break; }
    }
    return selected;
}

float q_probability(const std::vector<int>* selector_ids, const std::vector<float>* selector_q,
                    int base, int selector_count, int token) {
    if (selector_ids == nullptr || selector_q == nullptr) { return 1.0F; }
    for (int index = 0; index < selector_count; ++index) {
        if ((*selector_ids)[static_cast<std::size_t>(base + index)] == token) {
            return (*selector_q)[static_cast<std::size_t>(base + index)];
        }
    }
    return 0.0F;
}

int pick_residual(const Distribution& distribution, const std::vector<int>* selector_ids,
                  const std::vector<float>* selector_q, int base, int selector_count,
                  int draft, float draw) {
    if (selector_ids == nullptr || selector_q == nullptr) { return pick(distribution, draft, draw); }
    std::vector<double> residual(distribution.ids.size());
    double mass = 0.0;
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        residual[index] = std::max(
            0.0, distribution.probabilities[index] -
                     static_cast<double>(q_probability(selector_ids, selector_q, base,
                                                       selector_count, distribution.ids[index])));
        mass += residual[index];
    }
    if (!(mass > 0.0)) { return draft; }
    const double target = static_cast<double>(draw) * mass;
    double cumulative = 0.0;
    int selected = distribution.ids.front();
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        if (!(residual[index] > 0.0)) { continue; }
        cumulative += residual[index];
        selected = distribution.ids[index];
        if (target < cumulative) { break; }
    }
    return selected;
}

std::vector<hip_bfloat16> make_logits(int token_domain, int columns, int batch) {
    std::vector<hip_bfloat16> result(
        static_cast<std::size_t>(token_domain) * columns * batch);
    for (int row = 0; row < batch; ++row) {
        for (int column = 0; column < columns; ++column) {
            auto* base = result.data() +
                         (static_cast<std::size_t>(row) * columns + column) * token_domain;
            for (int token = 0; token < token_domain; ++token) {
                const unsigned hash = static_cast<unsigned>(token) * 2654435761U ^
                                      static_cast<unsigned>(column * 1009 + row * 9176);
                base[token] = bf16(-10.0F + static_cast<float>(hash % 2048U) / 512.0F);
            }
            for (int rank = 0; rank < std::min(24, token_domain); ++rank) {
                const int token = (13 + rank * 7919 + column * 101 + row * 503) % token_domain;
                base[token] = bf16(9.0F - static_cast<float>(rank) * 0.1875F);
            }
        }
    }
    return result;
}

struct ChainExpected {
    std::vector<int> lengths;
    std::vector<int> anchors;
    std::vector<int> licensed;
    std::vector<int> licensed_counts;
    std::vector<int> accepted;
    std::vector<int> counts;
};

ChainExpected chain_oracle(
    int token_domain, int k, int batch, const std::vector<int>& targets,
    const std::vector<hip_bfloat16>& logits, const std::vector<int>& drafts,
    const std::vector<int>& extents, const std::vector<int>& initial_lengths,
    const std::vector<ninfer::ops::SamplingConfig>& configs,
    const std::vector<bool>& has_counts, const std::vector<int>& initial_counts,
    const std::vector<int>* selector_ids, const std::vector<float>* selector_q,
    int selector_count) {
    const int columns = k + 1;
    ChainExpected result{initial_lengths, std::vector<int>(static_cast<std::size_t>(batch), -1),
                         std::vector<int>(static_cast<std::size_t>(columns) * batch, 0),
                         std::vector<int>(static_cast<std::size_t>(batch)),
                         std::vector<int>(static_cast<std::size_t>(batch)), initial_counts};
    for (int row = 0; row < batch; ++row) {
        const int extent = std::clamp(extents[static_cast<std::size_t>(row)], 0, k);
        const auto& config = configs[static_cast<std::size_t>(row)];
        int accepted = 0;
        int terminal = 0;
        if (!(config.temperature > 0.0F)) {
            while (accepted < extent &&
                   targets[static_cast<std::size_t>(row * columns + accepted)] ==
                       drafts[static_cast<std::size_t>(row * k + accepted)]) {
                ++accepted;
            }
            terminal = targets[static_cast<std::size_t>(row * columns + accepted)];
        } else {
            std::vector<int> row_counts(
                result.counts.begin() + static_cast<std::size_t>(row) * token_domain,
                result.counts.begin() + static_cast<std::size_t>(row + 1) * token_domain);
            const std::vector<int>* counts = has_counts[static_cast<std::size_t>(row)]
                                                 ? &row_counts
                                                 : nullptr;
            for (int column = 0; column <= extent; ++column) {
                std::vector<int> overlay;
                for (int index = 0; index < column; ++index) {
                    overlay.push_back(drafts[static_cast<std::size_t>(row * k + index)]);
                }
                const Distribution distribution = distribution_oracle(
                    logits.data() +
                        (static_cast<std::size_t>(row) * columns + column) * token_domain,
                    token_domain, config, counts, overlay);
                if (column < extent) {
                    const int draft = drafts[static_cast<std::size_t>(row * k + column)];
                    double p = 0.0;
                    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
                        if (distribution.ids[index] == draft) {
                            p = distribution.probabilities[index];
                            break;
                        }
                    }
                    const int selector_base = selector_count * (column + row * k);
                    const float q = q_probability(selector_ids, selector_q, selector_base,
                                                  selector_count, draft);
                    const float draw = uniform(config.seed,
                                               initial_lengths[static_cast<std::size_t>(row)] +
                                                   column + 1,
                                               ninfer::ops::kSamplePurposeSpeculativeAccept);
                    if (q > 0.0F && draw < std::min(1.0, p / static_cast<double>(q))) {
                        accepted = column + 1;
                        continue;
                    }
                    terminal = pick_residual(
                        distribution, selector_ids, selector_q, selector_base, selector_count,
                        draft,
                        uniform(config.seed,
                                initial_lengths[static_cast<std::size_t>(row)] + column + 1,
                                ninfer::ops::kSamplePurposeSpeculativeCorrection));
                    break;
                }
                terminal = pick(
                    distribution, -1,
                    uniform(config.seed,
                            initial_lengths[static_cast<std::size_t>(row)] + extent + 1,
                            ninfer::ops::kSamplePurposeSpeculativeBonus));
            }
        }
        for (int index = 0; index < accepted; ++index) {
            result.licensed[static_cast<std::size_t>(row * columns + index)] =
                drafts[static_cast<std::size_t>(row * k + index)];
        }
        result.licensed[static_cast<std::size_t>(row * columns + accepted)] = terminal;
        result.licensed_counts[static_cast<std::size_t>(row)] = accepted + 1;
        result.accepted[static_cast<std::size_t>(row)] = accepted;
        result.anchors[static_cast<std::size_t>(row)] = terminal;
        result.lengths[static_cast<std::size_t>(row)] += accepted + 1;
        if (config.temperature > 0.0F && has_counts[static_cast<std::size_t>(row)]) {
            for (int index = 0; index <= accepted; ++index) {
                ++result.counts[static_cast<std::size_t>(row) * token_domain +
                                result.licensed[static_cast<std::size_t>(row * columns + index)]];
            }
        }
    }
    return result;
}

void run_chain_case(ninfer::DeviceContext& device, int token_domain, int k, int batch,
                    bool all_greedy, bool use_selector) {
    const int columns = k + 1;
    std::vector<int> drafts(static_cast<std::size_t>(k) * batch);
    std::vector<int> targets(static_cast<std::size_t>(columns) * batch);
    std::vector<int> extents(static_cast<std::size_t>(batch));
    std::vector<int> lengths(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        extents[static_cast<std::size_t>(row)] = (row * 3 + k) % (k + 1);
        lengths[static_cast<std::size_t>(row)] = 1000 + row * 37;
        const int desired_accept = extents[static_cast<std::size_t>(row)] == 0
                                       ? 0
                                       : row % (extents[static_cast<std::size_t>(row)] + 1);
        for (int column = 0; column < k; ++column) {
            const int draft = (17 + row * 503 + column * 7919) % token_domain;
            drafts[static_cast<std::size_t>(row * k + column)] = draft;
            targets[static_cast<std::size_t>(row * columns + column)] =
                column < desired_accept ? draft : (draft + 1) % token_domain;
        }
        targets[static_cast<std::size_t>(row * columns + k)] =
            (101 + row * 31 + k) % token_domain;
    }
    const std::vector<hip_bfloat16> logits = make_logits(token_domain, columns, batch);
    std::vector<int> counts(static_cast<std::size_t>(batch) * token_domain, 0);
    std::vector<ninfer::ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    std::vector<bool> has_counts(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        auto& config = configs[static_cast<std::size_t>(row)];
        const bool greedy = all_greedy || (batch > 1 && row == 0);
        config.temperature = greedy ? 0.0F : 0.65F + 0.2F * static_cast<float>(row % 3);
        const int top_k_cases[] = {1, 3, 7, 19, 20, 0, -1, 11};
        config.top_k = top_k_cases[row % 8];
        config.top_p = row % 3 == 0 ? 0.73F : 1.0F;
        config.min_p = row % 4 == 2 ? 0.12F : 0.0F;
        config.presence_penalty = greedy ? 0.0F : (row % 2 == 0 ? 0.4F : 0.0F);
        config.frequency_penalty = greedy ? 0.0F : (row % 3 == 1 ? 0.2F : 0.0F);
        config.seed = 0x123456789abcdef0ULL + static_cast<unsigned long long>(row * 1009 + k);
        has_counts[static_cast<std::size_t>(row)] = !greedy && row % 4 != 3;
        counts[static_cast<std::size_t>(row) * token_domain + (13 + row * 503) % token_domain] =
            2 + row;
    }
    std::vector<int> selector_ids;
    std::vector<float> selector_q;
    constexpr int selector_count = 4;
    if (use_selector) {
        selector_ids.assign(static_cast<std::size_t>(selector_count) * k * batch, 0);
        selector_q.assign(selector_ids.size(), 0.0F);
        for (int row = 0; row < batch; ++row) {
            for (int column = 0; column < k; ++column) {
                const int base = selector_count * (column + row * k);
                selector_ids[static_cast<std::size_t>(base)] =
                    drafts[static_cast<std::size_t>(row * k + column)];
                selector_q[static_cast<std::size_t>(base)] =
                    row % 3 == 1 ? 0.0F : (row % 3 == 2 ? 0.35F : 0.8F);
                for (int candidate = 1; candidate < selector_count; ++candidate) {
                    selector_ids[static_cast<std::size_t>(base + candidate)] =
                        (13 + candidate * 7919 + column * 101 + row * 503) % token_domain;
                    selector_q[static_cast<std::size_t>(base + candidate)] =
                        0.2F / static_cast<float>(candidate);
                }
            }
        }
    }

    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(lengths.size() * sizeof(int));
    ninfer::DeviceBuffer d_anchors(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_counts(counts.size() * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    ninfer::DeviceBuffer d_selector_ids(use_selector ? selector_ids.size() * sizeof(int) : 1);
    ninfer::DeviceBuffer d_selector_q(use_selector ? selector_q.size() * sizeof(float) : 1);
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
    d_counts.copy_from_host(counts.data(), counts.size() * sizeof(int));
    if (use_selector) {
        d_selector_ids.copy_from_host(selector_ids.data(), selector_ids.size() * sizeof(int));
        d_selector_q.copy_from_host(selector_q.data(), selector_q.size() * sizeof(float));
    }
    for (int row = 0; row < batch; ++row) {
        configs[static_cast<std::size_t>(row)].token_counts =
            has_counts[static_cast<std::size_t>(row)]
                ? static_cast<int*>(d_counts.data()) + static_cast<std::size_t>(row) * token_domain
                : nullptr;
    }
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    const ChainExpected expected = chain_oracle(
        token_domain, k, batch, targets, logits, drafts, extents, lengths, configs, has_counts,
        counts, use_selector ? &selector_ids : nullptr, use_selector ? &selector_q : nullptr,
        use_selector ? selector_count : 0);

    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {token_domain, columns, batch});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor licensed_count_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor selector_ids_tensor;
    ninfer::Tensor selector_q_tensor;
    if (use_selector) {
        selector_ids_tensor = ninfer::Tensor(d_selector_ids.data(), ninfer::DType::I32,
                                             {selector_count, k, batch});
        selector_q_tensor = ninfer::Tensor(d_selector_q.data(), ninfer::DType::FP32,
                                           {selector_count, k, batch});
    }
    const std::size_t workspace_bytes =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            token_domain, k, k, batch, batch);
    ninfer::DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
    ninfer::ops::speculative_accept_greedy_drafts(
        target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
        licensed_tensor, licensed_count_tensor, accepted_tensor, token_domain,
        static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
        device.stream, use_selector ? &selector_ids_tensor : nullptr,
        use_selector ? &selector_q_tensor : nullptr);
    device.synchronize();

    const std::string label = "chain V=" + std::to_string(token_domain) +
                              " K=" + std::to_string(k) + " B=" + std::to_string(batch);
    require_equal(copy_from<int>(d_lengths, batch), expected.lengths, label + " lengths");
    require_equal(copy_from<int>(d_anchors, batch), expected.anchors, label + " anchors");
    require_equal(copy_from<int>(d_licensed, static_cast<std::size_t>(columns) * batch),
                  expected.licensed, label + " licensed tokens");
    require_equal(copy_from<int>(d_licensed_counts, batch), expected.licensed_counts,
                  label + " licensed counts");
    require_equal(copy_from<int>(d_accepted, batch), expected.accepted, label + " accepted");
    require_equal(copy_from<int>(d_counts, counts.size()), expected.counts, label + " token counts");
    require_equal(copy_from<int>(d_targets, targets.size()), targets, label + " target immutable");
    require_equal(copy_from<int>(d_drafts, drafts.size()), drafts, label + " drafts immutable");
    require_equal(copy_from<int>(d_extents, extents.size()), extents, label + " extents immutable");
    require_equal(copy_from<std::uint16_t>(d_logits, logits.size()), bf16_bits(logits),
                  label + " logits immutable");
    if (use_selector) {
        require_equal(copy_from<int>(d_selector_ids, selector_ids.size()), selector_ids,
                      label + " selector ids immutable");
        require_equal(copy_from<float>(d_selector_q, selector_q.size()), selector_q,
                      label + " selector probabilities immutable");
    }
    require(workspace.used() == 0 && workspace.peak_used() == workspace_bytes,
            label + " workspace high-water differs from sizing query");
}

void qualify_prepare(ninfer::DeviceContext& device) {
    for (int k = 1; k <= 8; ++k) {
        for (int batch = 1; batch <= static_cast<int>(ninfer::kMaximumConcurrency); ++batch) {
            const int columns = k + 1;
            std::vector<int> anchors(static_cast<std::size_t>(batch));
            std::vector<int> positions(static_cast<std::size_t>(batch));
            std::vector<int> extents(static_cast<std::size_t>(batch));
            std::vector<int> drafts(static_cast<std::size_t>(k) * batch);
            std::vector<int> expected_ids(static_cast<std::size_t>(columns) * batch);
            std::vector<int> expected_positions(expected_ids.size());
            for (int row = 0; row < batch; ++row) {
                anchors[static_cast<std::size_t>(row)] = 70000 + row;
                positions[static_cast<std::size_t>(row)] = 1000 + row * 37;
                extents[static_cast<std::size_t>(row)] = (row * 3 + k) % (k + 1);
                for (int column = 0; column < k; ++column) {
                    drafts[static_cast<std::size_t>(row * k + column)] = 17 + row * 101 + column;
                }
                for (int column = 0; column < columns; ++column) {
                    expected_ids[static_cast<std::size_t>(row * columns + column)] =
                        column == 0
                            ? anchors[static_cast<std::size_t>(row)]
                            : (column <= extents[static_cast<std::size_t>(row)]
                                   ? drafts[static_cast<std::size_t>(row * k + column - 1)]
                                   : anchors[static_cast<std::size_t>(row)]);
                    expected_positions[static_cast<std::size_t>(row * columns + column)] =
                        positions[static_cast<std::size_t>(row)] +
                        std::min(column, extents[static_cast<std::size_t>(row)]);
                }
            }
            ninfer::DeviceBuffer d_anchors(anchors.size() * sizeof(int));
            ninfer::DeviceBuffer d_positions(positions.size() * sizeof(int));
            ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
            ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
            ninfer::DeviceBuffer d_ids(expected_ids.size() * sizeof(int));
            ninfer::DeviceBuffer d_output_positions(expected_positions.size() * sizeof(int));
            d_anchors.copy_from_host(anchors.data(), anchors.size() * sizeof(int));
            d_positions.copy_from_host(positions.data(), positions.size() * sizeof(int));
            d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
            d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
            ninfer::Tensor anchors_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor positions_tensor(d_positions.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor extents_tensor(d_extents.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor drafts_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
            ninfer::Tensor ids_tensor(d_ids.data(), ninfer::DType::I32, {columns, batch});
            ninfer::Tensor output_positions_tensor(d_output_positions.data(), ninfer::DType::I32,
                                                    {columns, batch});
            ninfer::ops::speculative_prepare_verify_inputs(
                anchors_tensor, drafts_tensor, positions_tensor, extents_tensor, ids_tensor,
                output_positions_tensor, device.stream);
            device.synchronize();
            require_equal(copy_from<int>(d_ids, expected_ids.size()), expected_ids,
                          "prepare ids K=" + std::to_string(k) + " B=" +
                              std::to_string(batch));
            require_equal(copy_from<int>(d_output_positions, expected_positions.size()),
                          expected_positions,
                          "prepare positions K=" + std::to_string(k) + " B=" +
                              std::to_string(batch));
            std::vector<int> sentinel(expected_ids.size(), 0x55555555);
            d_ids.copy_from_host_async(sentinel.data(), sentinel.size() * sizeof(int),
                                       device.stream);
            ninfer::ops::speculative_prepare_verify_ids(anchors_tensor, drafts_tensor,
                                                        extents_tensor, ids_tensor, device.stream);
            device.synchronize();
            require_equal(copy_from<int>(d_ids, expected_ids.size()), expected_ids,
                          "prepare ids-only K=" + std::to_string(k) + " B=" +
                              std::to_string(batch));
            require_equal(copy_from<int>(d_anchors, anchors.size()), anchors,
                          "prepare anchors immutable");
            require_equal(copy_from<int>(d_drafts, drafts.size()), drafts,
                          "prepare drafts immutable");
        }
    }
}

struct TreeExpected {
    std::vector<int> lengths;
    std::vector<int> anchors;
    std::vector<int> licensed;
    std::vector<int> licensed_counts;
    std::vector<int> accepted;
    std::vector<int> accepted_column;
    std::vector<int> path;
    std::vector<int> counts;
};

int child_for(const std::vector<int>& parents, const std::vector<int>& ids, int row, int width,
              int valid, int node, int token) {
    for (int child = node + 1; child < valid; ++child) {
        if (parents[static_cast<std::size_t>(row * width + child)] == node &&
            ids[static_cast<std::size_t>(row * width + child)] == token) {
            return child;
        }
    }
    return -1;
}

TreeExpected tree_oracle(int token_domain, int width, int batch,
                         const std::vector<int>& targets,
                         const std::vector<hip_bfloat16>& logits,
                         const std::vector<int>& ids, const std::vector<int>& parents,
                         const std::vector<int>& valid_columns,
                         const std::vector<int>& extents,
                         const std::vector<int>& initial_lengths,
                         const std::vector<ninfer::ops::SamplingConfig>& configs,
                         const std::vector<bool>& has_counts,
                         const std::vector<int>& initial_counts) {
    TreeExpected result{initial_lengths, std::vector<int>(batch, -1),
                        std::vector<int>(static_cast<std::size_t>(width) * batch, 0),
                        std::vector<int>(batch), std::vector<int>(batch),
                        std::vector<int>(batch),
                        std::vector<int>(static_cast<std::size_t>(width) * batch, 0),
                        initial_counts};
    for (int row = 0; row < batch; ++row) {
        const int valid = std::clamp(valid_columns[static_cast<std::size_t>(row)], 1, width);
        const int extent = std::clamp(extents[static_cast<std::size_t>(row)], 0, width - 1);
        const auto& config = configs[static_cast<std::size_t>(row)];
        int node = 0;
        int accepted = 0;
        int terminal = 0;
        result.path[static_cast<std::size_t>(row * width)] = 0;
        if (!(config.temperature > 0.0F)) {
            while (accepted < extent) {
                const int predicted = targets[static_cast<std::size_t>(row * width + node)];
                const int child = child_for(parents, ids, row, width, valid, node, predicted);
                if (child < 0) {
                    terminal = predicted;
                    break;
                }
                result.licensed[static_cast<std::size_t>(row * width + accepted)] = predicted;
                node = child;
                ++accepted;
                result.path[static_cast<std::size_t>(row * width + accepted)] = node;
            }
            if (accepted == extent) {
                terminal = targets[static_cast<std::size_t>(row * width + node)];
            }
        } else {
            std::vector<int> row_counts(
                result.counts.begin() + static_cast<std::size_t>(row) * token_domain,
                result.counts.begin() + static_cast<std::size_t>(row + 1) * token_domain);
            const std::vector<int>* counts = has_counts[static_cast<std::size_t>(row)]
                                                 ? &row_counts
                                                 : nullptr;
            std::vector<int> overlay;
            for (int step = 0; step <= extent; ++step) {
                const Distribution distribution = distribution_oracle(
                    logits.data() +
                        (static_cast<std::size_t>(row) * width + node) * token_domain,
                    token_domain, config, counts, overlay);
                const int sampled = pick(
                    distribution, -1,
                    uniform(config.seed,
                            initial_lengths[static_cast<std::size_t>(row)] + accepted + 1,
                            ninfer::ops::kSamplePurposeSpeculativeAccept));
                const int child = step < extent
                                      ? child_for(parents, ids, row, width, valid, node, sampled)
                                      : -1;
                if (child >= 0) {
                    result.licensed[static_cast<std::size_t>(row * width + accepted)] = sampled;
                    overlay.push_back(sampled);
                    node = child;
                    ++accepted;
                    result.path[static_cast<std::size_t>(row * width + accepted)] = node;
                } else {
                    terminal = sampled;
                    break;
                }
            }
        }
        result.licensed[static_cast<std::size_t>(row * width + accepted)] = terminal;
        result.licensed_counts[static_cast<std::size_t>(row)] = accepted + 1;
        result.accepted[static_cast<std::size_t>(row)] = accepted;
        result.accepted_column[static_cast<std::size_t>(row)] = node;
        result.anchors[static_cast<std::size_t>(row)] = terminal;
        result.lengths[static_cast<std::size_t>(row)] += accepted + 1;
        if (config.temperature > 0.0F && has_counts[static_cast<std::size_t>(row)]) {
            for (int index = 0; index <= accepted; ++index) {
                ++result.counts[static_cast<std::size_t>(row) * token_domain +
                                result.licensed[static_cast<std::size_t>(row * width + index)]];
            }
        }
    }
    return result;
}

void run_tree_case(ninfer::DeviceContext& device, int token_domain, int width, int batch,
                   bool all_greedy) {
    std::vector<int> ids(static_cast<std::size_t>(width) * batch);
    std::vector<int> parents(ids.size());
    std::vector<int> targets(ids.size());
    std::vector<int> valid(static_cast<std::size_t>(batch));
    std::vector<int> extents(static_cast<std::size_t>(batch));
    std::vector<int> lengths(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        valid[static_cast<std::size_t>(row)] = std::max(2, width - row % 3);
        extents[static_cast<std::size_t>(row)] = std::min(width - 1, (row * 5 + width) % width);
        lengths[static_cast<std::size_t>(row)] = 2048 + row * 53;
        for (int column = 0; column < width; ++column) {
            parents[static_cast<std::size_t>(row * width + column)] =
                column == 0 ? -1 : (column - 1) / 2;
            ids[static_cast<std::size_t>(row * width + column)] =
                (17 + row * 503 + column * 7919) % token_domain;
            const int first_child = column * 2 + 1;
            targets[static_cast<std::size_t>(row * width + column)] =
                first_child < valid[static_cast<std::size_t>(row)]
                    ? ids[static_cast<std::size_t>(row * width + first_child)]
                    : (101 + row * 31 + column) % token_domain;
        }
    }
    std::vector<hip_bfloat16> logits = make_logits(token_domain, width, batch);
    // Give stochastic rows a deterministic mixture of child-membership and correction columns;
    // the oracle still evaluates filtering, penalties, and RNG independently.
    for (int row = 0; row < batch; ++row) {
        for (int column = 0; column < width; ++column) {
            const int first_child = column * 2 + 1;
            const int peak = first_child < valid[static_cast<std::size_t>(row)] && column % 2 == 0
                                 ? ids[static_cast<std::size_t>(row * width + first_child)]
                                 : (211 + row * 71 + column * 13) % token_domain;
            logits[(static_cast<std::size_t>(row) * width + column) * token_domain + peak] =
                bf16(14.0F);
        }
    }
    std::vector<int> counts(static_cast<std::size_t>(batch) * token_domain, 0);
    std::vector<ninfer::ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    std::vector<bool> has_counts(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        auto& config = configs[static_cast<std::size_t>(row)];
        const bool greedy = all_greedy || (batch > 1 && row == 0);
        config.temperature = greedy ? 0.0F : 0.75F + 0.1F * static_cast<float>(row % 3);
        config.top_k = greedy ? 0 : (row % 3 == 0 ? 1 : 7);
        config.top_p = row % 2 == 0 ? 0.85F : 1.0F;
        config.min_p = row % 3 == 2 ? 0.05F : 0.0F;
        config.presence_penalty = greedy ? 0.0F : 0.2F;
        config.frequency_penalty = greedy ? 0.0F : 0.1F;
        config.seed = 0x5bd1e995ULL + static_cast<unsigned long long>(row * 997 + width);
        has_counts[static_cast<std::size_t>(row)] = !greedy && row % 4 != 3;
        counts[static_cast<std::size_t>(row) * token_domain +
               ids[static_cast<std::size_t>(row * width + 1)]] = 2;
    }

    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_ids(ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_parents(parents.size() * sizeof(int));
    ninfer::DeviceBuffer d_valid(valid.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(lengths.size() * sizeof(int));
    ninfer::DeviceBuffer d_anchors(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_licensed(ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted_column(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_path(ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_counts(counts.size() * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_ids.copy_from_host(ids.data(), ids.size() * sizeof(int));
    d_parents.copy_from_host(parents.data(), parents.size() * sizeof(int));
    d_valid.copy_from_host(valid.data(), valid.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
    d_counts.copy_from_host(counts.data(), counts.size() * sizeof(int));
    for (int row = 0; row < batch; ++row) {
        configs[static_cast<std::size_t>(row)].token_counts =
            has_counts[static_cast<std::size_t>(row)]
                ? static_cast<int*>(d_counts.data()) + static_cast<std::size_t>(row) * token_domain
                : nullptr;
    }
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    const TreeExpected expected = tree_oracle(token_domain, width, batch, targets, logits, ids,
                                              parents, valid, extents, lengths, configs,
                                              has_counts, counts);

    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {token_domain, width, batch});
    ninfer::Tensor ids_tensor(d_ids.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor parents_tensor(d_parents.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor valid_tensor(d_valid.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor extents_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor lengths_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchors_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor licensed_counts_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_column_tensor(d_accepted_column.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor path_tensor(d_path.data(), ninfer::DType::I32, {width, batch});
    const std::size_t workspace_bytes =
        ninfer::ops::speculative_accept_tree_drafts_workspace_capacity_bytes(
            token_domain, width, width, batch, batch);
    ninfer::DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
    ninfer::ops::speculative_accept_tree_drafts(
        target_tensor, logits_tensor, ids_tensor, parents_tensor, valid_tensor, extents_tensor,
        lengths_tensor, anchors_tensor, licensed_tensor, licensed_counts_tensor, accepted_tensor,
        accepted_column_tensor, path_tensor, token_domain,
        static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
        device.stream);
    device.synchronize();

    const std::string label = "tree V=" + std::to_string(token_domain) +
                              " W=" + std::to_string(width) + " B=" +
                              std::to_string(batch);
    require_equal(copy_from<int>(d_lengths, batch), expected.lengths, label + " lengths");
    require_equal(copy_from<int>(d_anchors, batch), expected.anchors, label + " anchors");
    require_equal(copy_from<int>(d_licensed, ids.size()), expected.licensed, label + " licensed");
    require_equal(copy_from<int>(d_licensed_counts, batch), expected.licensed_counts,
                  label + " licensed counts");
    require_equal(copy_from<int>(d_accepted, batch), expected.accepted, label + " accepted");
    require_equal(copy_from<int>(d_accepted_column, batch), expected.accepted_column,
                  label + " accepted columns");
    require_equal(copy_from<int>(d_path, ids.size()), expected.path, label + " fold path");
    require_equal(copy_from<int>(d_counts, counts.size()), expected.counts, label + " token counts");
    require_equal(copy_from<int>(d_ids, ids.size()), ids, label + " ids immutable");
    require_equal(copy_from<int>(d_parents, parents.size()), parents, label + " parents immutable");
    require_equal(copy_from<int>(d_targets, targets.size()), targets, label + " targets immutable");
    require_equal(copy_from<int>(d_valid, valid.size()), valid, label + " valid immutable");
    require_equal(copy_from<int>(d_extents, extents.size()), extents, label + " extents immutable");
    require_equal(copy_from<std::uint16_t>(d_logits, logits.size()), bf16_bits(logits),
                  label + " logits immutable");
    require(workspace.used() == 0 && workspace.peak_used() == workspace_bytes,
            label + " workspace high-water differs from sizing query");
    if (token_domain == kRealVocabulary) {
        for (int warmup = 0; warmup < 3; ++warmup) {
            ninfer::ops::speculative_accept_tree_drafts(
                target_tensor, logits_tensor, ids_tensor, parents_tensor, valid_tensor,
                extents_tensor, lengths_tensor, anchors_tensor, licensed_tensor,
                licensed_counts_tensor, accepted_tensor, accepted_column_tensor, path_tensor,
                token_domain,
                static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
                device.stream);
        }
        constexpr int repeats = 20;
        ninfer::DeviceEventTimer timer(device);
        timer.start();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            ninfer::ops::speculative_accept_tree_drafts(
                target_tensor, logits_tensor, ids_tensor, parents_tensor, valid_tensor,
                extents_tensor, lengths_tensor, anchors_tensor, licensed_tensor,
                licensed_counts_tensor, accepted_tensor, accepted_column_tensor, path_tensor,
                token_domain,
                static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
                device.stream);
        }
        timer.record_stop();
        device.synchronize();
        std::cout << "timing: tree V=" << token_domain << " W=" << width << " B=" << batch
                  << " ms=" << timer.elapsed_ms() / static_cast<float>(repeats)
                  << " workspace_bytes=" << workspace_bytes << '\n';
    }
}

void qualify_hidden_and_remap(ninfer::DeviceContext& device) {
    constexpr int rows = 257;
    constexpr int columns = 16;
    constexpr int batch = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> hidden(static_cast<std::size_t>(rows) * columns * batch);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index].data = static_cast<std::uint16_t>((index * 7919U + 17U) & 0xffffU);
    }
    std::vector<int> selectors(batch);
    for (int row = 0; row < batch; ++row) { selectors[static_cast<std::size_t>(row)] = row * 2; }
    std::vector<hip_bfloat16> expected(static_cast<std::size_t>(rows) * batch);
    for (int row = 0; row < batch; ++row) {
        std::copy_n(hidden.begin() +
                        (static_cast<std::size_t>(row) * columns + selectors[row]) * rows,
                    rows, expected.begin() + static_cast<std::size_t>(row) * rows);
    }
    ninfer::DeviceBuffer d_hidden(hidden.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_selectors(selectors.size() * sizeof(int));
    ninfer::DeviceBuffer d_output(expected.size() * sizeof(hip_bfloat16));
    d_hidden.copy_from_host(hidden.data(), hidden.size() * sizeof(hip_bfloat16));
    d_selectors.copy_from_host(selectors.data(), selectors.size() * sizeof(int));
    ninfer::Tensor hidden_tensor(d_hidden.data(), ninfer::DType::BF16, {rows, columns, batch});
    ninfer::Tensor selectors_tensor(d_selectors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor output_tensor(d_output.data(), ninfer::DType::BF16, {rows, batch});
    ninfer::ops::speculative_select_accepted_hidden(hidden_tensor, selectors_tensor, output_tensor,
                                                    device.stream);
    device.synchronize();
    const auto actual_bits = copy_from<std::uint16_t>(d_output, expected.size());
    std::vector<std::uint16_t> expected_bits(expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected_bits[index] = expected[index].data;
    }
    require_equal(actual_bits, expected_bits, "accepted hidden exact BF16 bits");

    constexpr int map_count = 4096;
    std::vector<int> map(map_count);
    for (int index = 0; index < map_count; ++index) {
        map[static_cast<std::size_t>(index)] = (index * 2053 + 17) & (map_count - 1);
    }
    for (const int token_count : {1, 8, 16, 257}) {
        std::vector<int> tokens(static_cast<std::size_t>(token_count));
        for (int index = 0; index < token_count; ++index) {
            tokens[static_cast<std::size_t>(index)] = (index * 7919) & (map_count - 1);
        }
        std::vector<int> remapped(tokens.size());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            remapped[index] = map[static_cast<std::size_t>(tokens[index])];
        }
        ninfer::DeviceBuffer d_map(map.size() * sizeof(int));
        ninfer::DeviceBuffer d_tokens(tokens.size() * sizeof(int));
        d_map.copy_from_host(map.data(), map.size() * sizeof(int));
        d_tokens.copy_from_host(tokens.data(), tokens.size() * sizeof(int));
        ninfer::Tensor token_tensor(d_tokens.data(), ninfer::DType::I32, {token_count});
        ninfer::ops::proposal_remap_token_ids(token_tensor, static_cast<const int*>(d_map.data()),
                                              map_count, device.stream);
        device.synchronize();
        require_equal(copy_from<int>(d_tokens, tokens.size()), remapped,
                      "proposal remap T=" + std::to_string(token_count));
        require_equal(copy_from<int>(d_map, map.size()), map, "proposal map immutable");
    }
}

void qualify_graph_replay(ninfer::DeviceContext& device) {
    constexpr int token_domain = 1024;
    constexpr int k = 2;
    constexpr int columns = 3;
    std::vector<int> targets{7, 11, 13};
    std::vector<int> drafts{7, 11};
    std::vector<int> extents{2};
    std::vector<int> lengths{100};
    std::vector<hip_bfloat16> logits(static_cast<std::size_t>(token_domain) * columns,
                                     bf16(-20.0F));
    logits[7] = bf16(20.0F);
    logits[static_cast<std::size_t>(token_domain) + 11] = bf16(20.0F);
    logits[static_cast<std::size_t>(token_domain) * 2 + 13] = bf16(20.0F);
    ninfer::ops::SamplingConfig config{};
    config.temperature = 1.0F;
    config.top_k = 1;
    config.seed = 1234;
    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(sizeof(int));
    ninfer::DeviceBuffer d_lengths(sizeof(int));
    ninfer::DeviceBuffer d_anchors(sizeof(int));
    ninfer::DeviceBuffer d_licensed(columns * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(sizeof(int));
    ninfer::DeviceBuffer d_accepted(sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_config(sizeof(config));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), sizeof(int));
    d_lengths.copy_from_host(lengths.data(), sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_config.copy_from_host(&config, sizeof(config));
    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {token_domain, columns});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {1});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {1});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {1});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns});
    ninfer::Tensor licensed_counts_tensor(d_licensed_counts.data(), ninfer::DType::I32, {1});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {1});
    const std::size_t bytes =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            token_domain, k, k, 1, 1);
    ninfer::DeviceArena workspace(bytes);

    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
    ninfer::ops::speculative_accept_greedy_drafts(
        target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
        licensed_tensor, licensed_counts_tensor, accepted_tensor, token_domain,
        static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace,
        device.stream);
    HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
        const int reset_length = 100 + replay * 17;
        d_lengths.copy_from_host(&reset_length, sizeof(int));
        HIP_CHECK(hipGraphLaunch(executable, device.stream));
        device.synchronize();
        require_equal(copy_from<int>(d_licensed, columns), std::vector<int>({7, 11, 13}),
                      "graph replay licensed tokens");
        require_equal(copy_from<int>(d_lengths, 1), std::vector<int>({reset_length + 3}),
                      "graph replay dynamic state");
    }
    HIP_CHECK(hipGraphExecDestroy(executable));
    HIP_CHECK(hipGraphDestroy(graph));
}

void qualify_validation(ninfer::DeviceContext& device) {
    try {
        (void)ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            1, 0, 8, 1, static_cast<int>(ninfer::kMaximumConcurrency));
        fail("chain workspace accepted K=0");
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::speculative_accept_tree_drafts_workspace_capacity_bytes(
            1, 1, 16, 1, static_cast<int>(ninfer::kMaximumConcurrency));
        fail("tree workspace accepted W=1");
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::speculative_accept_tree_drafts_workspace_capacity_bytes(
            1, 2, 16, 1, static_cast<int>(ninfer::kMaximumConcurrency) + 1);
        fail("tree workspace accepted B above product concurrency");
    } catch (const std::invalid_argument&) {}

    ninfer::DeviceBuffer d_i32(64);
    ninfer::Tensor vector(d_i32.data(), ninfer::DType::I32, {1});
    ninfer::Tensor matrix(d_i32.data(), ninfer::DType::I32, {2});
    try {
        ninfer::ops::speculative_prepare_verify_ids(vector, vector, vector, matrix, nullptr);
        fail("prepare accepted a null stream");
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::ops::speculative_prepare_verify_ids(vector, vector, vector, matrix, device.stream);
        fail("prepare accepted overlapping tensors");
    } catch (const std::invalid_argument&) {}

    constexpr int token_domain = 1024;
    ninfer::DeviceBuffer d_target(2 * sizeof(int));
    ninfer::DeviceBuffer d_logits(2 * token_domain * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_draft(sizeof(int));
    ninfer::DeviceBuffer d_extent(sizeof(int));
    ninfer::DeviceBuffer d_length(sizeof(int));
    ninfer::DeviceBuffer d_anchor(sizeof(int));
    ninfer::DeviceBuffer d_licensed(2 * sizeof(int));
    ninfer::DeviceBuffer d_licensed_count(sizeof(int));
    ninfer::DeviceBuffer d_accepted(sizeof(int));
    ninfer::DeviceBuffer d_config(sizeof(ninfer::ops::SamplingConfig));
    ninfer::DeviceBuffer d_selector_ids(sizeof(int));
    const std::vector<int> targets{7, 11};
    const std::vector<int> drafts{7};
    const std::vector<int> one{1};
    const std::vector<hip_bfloat16> logits(2 * token_domain, bf16(0.0F));
    ninfer::ops::SamplingConfig config{};
    config.temperature = 1.0F;
    config.top_k = 1;
    d_target.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_draft.copy_from_host(drafts.data(), sizeof(int));
    d_extent.copy_from_host(one.data(), sizeof(int));
    d_length.copy_from_host(one.data(), sizeof(int));
    d_config.copy_from_host(&config, sizeof(config));
    ninfer::Tensor target(d_target.data(), ninfer::DType::I32, {2});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16, {token_domain, 2});
    ninfer::Tensor draft(d_draft.data(), ninfer::DType::I32, {1});
    ninfer::Tensor extent(d_extent.data(), ninfer::DType::I32, {1});
    ninfer::Tensor length(d_length.data(), ninfer::DType::I32, {1});
    ninfer::Tensor anchor(d_anchor.data(), ninfer::DType::I32, {1});
    ninfer::Tensor licensed(d_licensed.data(), ninfer::DType::I32, {2});
    ninfer::Tensor licensed_count(d_licensed_count.data(), ninfer::DType::I32, {1});
    ninfer::Tensor accepted(d_accepted.data(), ninfer::DType::I32, {1});
    ninfer::Tensor selector_ids(d_selector_ids.data(), ninfer::DType::I32, {1, 1, 1});
    const std::size_t needed =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            token_domain, 1, 1, 1, 1);
    ninfer::DeviceArena workspace(needed);
    try {
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, licensed, licensed_count,
            accepted, token_domain, nullptr, workspace, device.stream);
        fail("chain acceptance accepted a null config pointer");
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, licensed, licensed_count,
            accepted, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace,
            device.stream, &selector_ids, nullptr);
        fail("chain acceptance accepted an unpaired selector");
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::DeviceArena undersized(needed - 1);
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, licensed, licensed_count,
            accepted, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), undersized,
            device.stream);
        fail("chain acceptance accepted undersized workspace");
    } catch (const std::bad_alloc&) {}
    try {
        ninfer::Tensor aliased_output(d_target.data(), ninfer::DType::I32, {2});
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, aliased_output,
            licensed_count, accepted, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace,
            device.stream);
        fail("chain acceptance accepted output/input aliasing");
    } catch (const std::invalid_argument&) {}

    try {
        ninfer::Tensor tokens(d_target.data(), ninfer::DType::I32, {2});
        ninfer::ops::proposal_remap_token_ids(tokens, static_cast<const int*>(d_target.data()), 2,
                                              device.stream);
        fail("proposal remap accepted an overlapping id map");
    } catch (const std::invalid_argument&) {}
}

void benchmark(ninfer::DeviceContext& device) {
    constexpr int token_domain = kRealVocabulary;
    constexpr int k = 8;
    constexpr int columns = k + 1;
    constexpr int batch = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> logits = make_logits(token_domain, columns, batch);
    std::vector<int> targets(static_cast<std::size_t>(columns) * batch, 0);
    std::vector<int> drafts(static_cast<std::size_t>(k) * batch, 0);
    std::vector<int> extents(batch, k);
    std::vector<int> lengths(batch, 8192);
    std::vector<ninfer::ops::SamplingConfig> configs(batch);
    for (int row = 0; row < batch; ++row) {
        configs[static_cast<std::size_t>(row)].temperature = 0.8F;
        configs[static_cast<std::size_t>(row)].top_k = 20;
        configs[static_cast<std::size_t>(row)].top_p = 0.9F;
        configs[static_cast<std::size_t>(row)].min_p = 0.05F;
        configs[static_cast<std::size_t>(row)].seed = static_cast<unsigned long long>(row + 1);
    }
    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(lengths.size() * sizeof(int));
    ninfer::DeviceBuffer d_anchors(batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(batch * sizeof(int));
    ninfer::DeviceBuffer d_accepted(batch * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {token_domain, columns, batch});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor licensed_counts_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    const std::size_t bytes =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            token_domain, k, k, batch, batch);
    ninfer::DeviceArena workspace(bytes);
    for (int warmup = 0; warmup < 3; ++warmup) {
        ninfer::ops::speculative_accept_greedy_drafts(
            target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor,
            anchor_tensor, licensed_tensor, licensed_counts_tensor, accepted_tensor,
            token_domain, static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
            workspace, device.stream);
    }
    constexpr int repeats = 20;
    ninfer::DeviceEventTimer timer(device);
    timer.start();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        ninfer::ops::speculative_accept_greedy_drafts(
            target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor,
            anchor_tensor, licensed_tensor, licensed_counts_tensor, accepted_tensor,
            token_domain, static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
            workspace, device.stream);
    }
    timer.record_stop();
    device.synchronize();
    std::cout << "timing: chain V=" << token_domain << " K=" << k << " B=" << batch
              << " ms=" << timer.elapsed_ms() / static_cast<float>(repeats)
              << " workspace_bytes=" << bytes << '\n';
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(select_r9700());
        std::cout << "device: arch=" << device.props.gcnArchName << " wave="
                  << device.props.warpSize << '\n';
        qualify_prepare(device);
        for (int k = 1; k <= 8; ++k) {
            for (int batch = 1; batch <= static_cast<int>(ninfer::kMaximumConcurrency); ++batch) {
                run_chain_case(device, 64, k, batch, true, false);
            }
        }
        run_chain_case(device, 257, 8, static_cast<int>(ninfer::kMaximumConcurrency), false, true);
        run_chain_case(device, 4099, 8, static_cast<int>(ninfer::kMaximumConcurrency), false, true);
        run_chain_case(device, kRealVocabulary, 8,
                       static_cast<int>(ninfer::kMaximumConcurrency), false, true);
        for (int width = 2; width <= 16; ++width) {
            for (int batch = 1; batch <= static_cast<int>(ninfer::kMaximumConcurrency); ++batch) {
                run_tree_case(device, 64, width, batch, true);
            }
        }
        run_tree_case(device, 257, 16, static_cast<int>(ninfer::kMaximumConcurrency), false);
        run_tree_case(device, 4099, 16, static_cast<int>(ninfer::kMaximumConcurrency), false);
        run_tree_case(device, kRealVocabulary, 12,
                      static_cast<int>(ninfer::kMaximumConcurrency), false);
        qualify_hidden_and_remap(device);
        qualify_graph_replay(device);
        qualify_validation(device);
        benchmark(device);
        std::cout << "speculative_round: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "speculative_round: FAIL: " << error.what() << '\n';
        return 1;
    }
}
