#pragma once

#include "ninfer/ops/sampling.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Independent host oracle: full logical FP64 softmax from represented public logits.
// No production reduction, staging representation, or private workspace is imported.
namespace sampling_reference {
struct Distribution {
    std::vector<int> ids;
    std::vector<double> probabilities;
};
inline bool eligible(int token, const ninfer::ops::SamplingConfig& config) {
    for (int i = 0; i < config.suppressed_token_count; ++i)
        if (config.suppressed_tokens[i] == token) return false;
    return config.allowed_token_words == nullptr ||
           ((config.allowed_token_words[token / 32] >> (token % 32)) & 1U) != 0;
}
inline Distribution p_less(const std::vector<double>& logits,
                           const ninfer::ops::SamplingConfig& config) {
    std::vector<int> order;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i)
        if (eligible(i, config)) order.push_back(i);
    if (order.empty()) throw std::runtime_error("oracle requires a nonempty eligible domain");
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    if (!(config.temperature > 0)) return {{order[0]}, {1.0}};
    std::vector<double> weights(logits.size());
    double total = 0;
    for (int i : order) {
        weights[i] = std::exp((logits[i] - logits[order[0]]) / config.temperature);
        total += weights[i];
    }
    double collision = 0;
    for (int i : order) {
        weights[i] /= total;
        collision += weights[i] * weights[i];
    }
    // Epsilon relaxes only the collision threshold, never the effective-support floor.
    const double threshold = std::max(collision * std::exp(-2.0 * 0.0625 / config.temperature),
                                      1.0 / 1024.0);
    Distribution result;
    double retained = 0;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        if (eligible(i, config) && weights[i] >= threshold && i != config.typical_exclude) {
            result.ids.push_back(i);
            result.probabilities.push_back(weights[i]);
            retained += weights[i];
        }
    }
    if (result.ids.empty()) {
        const int fallback = order[0] == config.typical_exclude && order.size() > 1
                                 ? order[1] : order[0];
        return {{fallback}, {1.0}};
    }
    for (double& p : result.probabilities) p /= retained;
    return result;
}
} // namespace sampling_reference
