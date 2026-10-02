#pragma once

#include "core/arena.h"

#include <hip/hip_runtime_api.h>

namespace ninfer::ops {

/**
 * Op: persistent_decode_lower
 *
 * Rewrites a captured single-stream Device Graph in place: every run of at least two consecutive
 * kernel nodes whose kernels the persistent decode kernel hosts becomes one persistent decode
 * kernel node, which executes the run's launches as phases in capture order, one grid-wide
 * barrier between consecutive phases. Each phase runs the captured launch's own kernel body on
 * virtual CTAs with the captured arguments, block and grid indices, so every hosted launch
 * produces exactly the bytes it produces as its own kernel node; a launch's CTAs keep CTA
 * semantics (no ordering or co-residency among them). A launch's cache-warm CTAs
 * (core/cache_warm.h), which write nothing, are not run; the kernel prefetches the next phase's
 * weights and instructions while it waits at a barrier instead. The hosted kernels are those of
 * a single-sequence decode graph (one 16-token tile per projection).
 *
 * The graph is a single chain (each node has at most one dependency and one dependent); other
 * nodes (copies, fills, kernels the persistent kernel does not host) stay as they are. Returns the
 * device storage the rewritten nodes reference (phase tables, captured arguments and the grid
 * barrier word); the caller keeps it alive with the graph and every executable instantiated from
 * or updated with it. Executables of graphs lowered from identical node sequences can be updated
 * from one another.
 */
[[nodiscard]] DeviceBuffer persistent_decode_lower(hipGraph_t graph);

} // namespace ninfer::ops
