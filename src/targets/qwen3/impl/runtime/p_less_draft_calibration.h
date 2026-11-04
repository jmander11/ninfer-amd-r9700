#pragma once

// Host policy for the DFlash2 p-less proposal temperature.
//
// Each hop draws from q_j = softmax(score_j / T_d) over 16 selector candidates and verifies
// against target law p'_j. Re-tempering to T' only needs q_j^(T_d/T') renormalized, so the
// accept Op scores every grid temperature on the same verify columns without more drafter work:
// alpha_j(T') = sum_c min(p'_j(c), q'_j(c)). Comparing temperatures needs no exploration.
//
// For each target temperature T, keep discounted prefix acceptance sums
// S_j(T') = prod_{i<=j} alpha_i(T') over rounds that drafted hop j. Predict the yield of a
// k-draft chain as 1 + sum_{j<k} mean S_j(T') and choose the grid temperature with greatest
// yield. Use the Variant's prior until kWarmRounds of evidence; ties favor the nearest prior.
// This estimate scores token verification. Block verification accepts at least as much, and
// offline replay ranked proposal laws identically under both.
//
// The product estimate has lower variance than exact block-verification acceptance:
// P(tau >= j) = E[W_{j-1} * sum_c min(q'_j(c), p_{j-1} p'_j(c))], where p is the carried
// verifier weight and W is the realized prefix's importance weight under T'. The exact estimate
// is unbiased for every T', but its per-round spread was about ten times larger on recorded
// rounds and it chose worse temperatures over this policy's memory window.

#include "ninfer/ops/p_less_proposal_calibration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3 {

inline constexpr std::size_t kPLessCalibrationGrid =
    static_cast<std::size_t>(ops::kPLessProposalCalibrationTemperatureCount);
// Drafted hops per row; the Program checks it covers qwen3::kDFlashDecodeMaximumDrafts.
inline constexpr std::size_t kPLessCalibrationHops = 15;
// Distinct p-less target temperatures tracked at once; the least recently observed is replaced.
inline constexpr std::size_t kPLessCalibrationBuckets = 8;
// Per-round discount: an effective memory of 256 rounds (a few seconds of decode) follows
// content shifts such as thinking to answer while averaging the per-round noise of alpha.
inline constexpr double kPLessCalibrationDiscount = 1.0 - 1.0 / 256.0;
// Discounted hop-0 rounds before the calibration replaces the prior.
inline constexpr double kPLessCalibrationWarmRounds = 8.0;
// The kernel's FP32 overlap may exceed 1 by rounding.
inline constexpr float kPLessCalibrationRoundingSlack = 1.0e-3f;
// Predicted lengths within this relative distance of the best are not separated by the evidence.
inline constexpr double kPLessCalibrationTieTolerance = 1.0e-6;

struct PLessDraftCalibrationBucket {
    float temperature       = 0.0f;
    std::uint64_t last_used = 0; // 0: empty
    // weight[j]: discounted rounds that drafted hop j; prefix[j][g]: discounted sum of S_j(g).
    std::array<double, kPLessCalibrationHops> weight{};
    std::array<std::array<double, kPLessCalibrationGrid>, kPLessCalibrationHops> prefix{};
};

struct PLessDraftCalibration {
    std::array<PLessDraftCalibrationBucket, kPLessCalibrationBuckets> buckets{};
    std::uint64_t clock = 0;
};

[[nodiscard]] inline const PLessDraftCalibrationBucket*
p_less_calibration_find(const PLessDraftCalibration& calibration, float temperature) {
    for (const PLessDraftCalibrationBucket& bucket : calibration.buckets) {
        if (bucket.last_used != 0 && bucket.temperature == temperature) { return &bucket; }
    }
    return nullptr;
}

// The draft temperature for a k-draft chain round of a request at p-less temperature T.
[[nodiscard]] inline float
p_less_calibrated_draft_temperature(const PLessDraftCalibration& calibration, float temperature,
                                    std::uint32_t k, float prior) {
    const PLessDraftCalibrationBucket* bucket = p_less_calibration_find(calibration, temperature);
    if (bucket == nullptr || bucket->weight[0] < kPLessCalibrationWarmRounds || k == 0) {
        return prior;
    }
    const std::size_t hops = std::min<std::size_t>(k, kPLessCalibrationHops);
    std::array<double, kPLessCalibrationGrid> lengths{};
    double best_length = 0.0;
    for (std::size_t g = 0; g < kPLessCalibrationGrid; ++g) {
        lengths[g] = 1.0;
        for (std::size_t j = 0; j < hops; ++j) {
            // A hop no round has drafted adds nothing to any grid temperature.
            if (bucket->weight[j] > 0.0) { lengths[g] += bucket->prefix[j][g] / bucket->weight[j]; }
        }
        best_length = std::max(best_length, lengths[g]);
    }
    // Among the temperatures tied with the best, the nearest to the prior in log temperature.
    std::size_t best     = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::size_t g = 0; g < kPLessCalibrationGrid; ++g) {
        if (lengths[g] < best_length * (1.0 - kPLessCalibrationTieTolerance)) { continue; }
        const double distance =
            std::abs(std::log(static_cast<double>(ops::kPLessProposalCalibrationTemperatures[g])) -
                     std::log(static_cast<double>(prior)));
        if (distance < best_distance) {
            best_distance = distance;
            best          = g;
        }
    }
    return ops::kPLessProposalCalibrationTemperatures[best];
}

// Records one row of a calibrated chain round: alphas is the row's [K][G] slice of the accept op's
// proposal calibration (hop-major, grid fastest) and extent the row's drafted hop count.
// Throws std::runtime_error when a drafted hop's entry is not a probability.
inline void p_less_calibration_observe(PLessDraftCalibration& calibration, float temperature,
                                       std::span<const float> alphas, std::uint32_t extent) {
    if (extent == 0) { return; }
    if (extent > kPLessCalibrationHops || alphas.size() < extent * kPLessCalibrationGrid) {
        throw std::invalid_argument("p-less draft calibration row is shorter than its extent");
    }
    PLessDraftCalibrationBucket* bucket = nullptr;
    for (PLessDraftCalibrationBucket& candidate : calibration.buckets) {
        if (candidate.last_used != 0 && candidate.temperature == temperature) {
            bucket = &candidate;
            break;
        }
        if (bucket == nullptr || candidate.last_used < bucket->last_used) { bucket = &candidate; }
    }
    if (bucket->last_used == 0 || bucket->temperature != temperature) {
        *bucket             = PLessDraftCalibrationBucket{};
        bucket->temperature = temperature;
    }
    bucket->last_used = ++calibration.clock;
    for (std::size_t j = 0; j < kPLessCalibrationHops; ++j) {
        bucket->weight[j] *= kPLessCalibrationDiscount;
        for (double& sum : bucket->prefix[j]) { sum *= kPLessCalibrationDiscount; }
    }
    std::array<double, kPLessCalibrationGrid> running{};
    running.fill(1.0);
    for (std::size_t j = 0; j < extent; ++j) {
        bucket->weight[j] += 1.0;
        for (std::size_t g = 0; g < kPLessCalibrationGrid; ++g) {
            const float alpha = alphas[j * kPLessCalibrationGrid + g];
            if (!std::isfinite(alpha) || alpha < 0.0f ||
                alpha > 1.0f + kPLessCalibrationRoundingSlack) {
                throw std::runtime_error("DFlash p-less proposal calibration is not a probability");
            }
            running[g] *= std::min(static_cast<double>(alpha), 1.0);
            bucket->prefix[j][g] += running[g];
        }
    }
}

} // namespace ninfer::targets::qwen3
