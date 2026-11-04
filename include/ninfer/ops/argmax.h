#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <hip/hip_runtime.h> // hipStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 * Finite logits define the mathematical result. If an all-NaN column has no ordered maximum,
 * writes token zero as a safe placeholder for speculative proposal embedding gathers. Target
 * token publication uses the configured overload's nonfinite status instead of accepting that
 * placeholder as a successful inference result.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, hipStream_t stream);

// Greedy selection with row-local suppression and per-column eligibility bitsets.
// Config b owns each consecutive
// `columns_per_config` columns; temperature, filters, penalties, RNG, and counts are ignored.
// Each row's config must leave at least one token in valid_rows eligible.
// The optional SamplingConfig nonfinite_logits status is indexed by the within-config column;
// eligible nonfinite logits set its bit zero. With status reporting, an all-NaN column writes a
// safe in-domain placeholder; the caller must reject the status before publishing any token.
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
            const SamplingConfig* configs, std::int32_t columns_per_config, hipStream_t stream);

} // namespace ninfer::ops
