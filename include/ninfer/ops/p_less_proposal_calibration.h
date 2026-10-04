#pragma once

// HIP-free constants of the DFlash p-less proposal calibration, shared by the accept Op
// (ninfer/ops/speculative_round.h) and the host policy that consumes its output.

#include <array>
#include <cstdint>

namespace ninfer::ops {

// Candidate DFlash p-less draft temperatures scored by the accept op's proposal calibration.
inline constexpr std::int32_t kPLessProposalCalibrationTemperatureCount = 8;
inline constexpr std::array<float, kPLessProposalCalibrationTemperatureCount>
    kPLessProposalCalibrationTemperatures = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f, 1.25f};

// The grid passed by value to the calibration kernel.
struct PLessProposalCalibrationGrid {
    float temperature[kPLessProposalCalibrationTemperatureCount];
};

} // namespace ninfer::ops
