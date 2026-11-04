#pragma once

#include "core/arena.h"

#include <hip/hip_runtime_api.h>

#include <memory>
#include <vector>

namespace ninfer::ops {

using PersistentDecodeImageOwners = std::vector<std::shared_ptr<DeviceArena>>;

// Startup storage for byte-identical lowered phase images. Definitions retain shared ownership
// of their image after this construction-time owner is destroyed.
class PersistentDecodeStorage {
public:
    PersistentDecodeStorage();
    ~PersistentDecodeStorage();
    PersistentDecodeStorage(PersistentDecodeStorage&&) noexcept;
    PersistentDecodeStorage& operator=(PersistentDecodeStorage&&) noexcept;
    PersistentDecodeStorage(const PersistentDecodeStorage&)            = delete;
    PersistentDecodeStorage& operator=(const PersistentDecodeStorage&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend PersistentDecodeImageOwners persistent_decode_lower(hipGraph_t,
                                                               PersistentDecodeStorage&);
};

/**
 * Op: persistent_decode_lower
 *
 * Rewrites a captured single-stream Device Graph in place: every run of at least two consecutive
 * kernel nodes whose kernels the persistent decode kernel hosts becomes one child graph
 * containing a cooperative persistent kernel, which executes the run's launches as phases in
 * capture order, one grid-wide barrier between consecutive phases. Each phase runs the captured
 * launch's kernel body on
 * virtual CTAs with the captured arguments, block and grid indices, or a re-tiled body of the
 * same Op over its own virtual grid (the GDN record: one CTA per sequence and head), so every
 * hosted launch produces exactly the bytes it produces as its own kernel node; a launch's CTAs
 * keep CTA semantics (no ordering or co-residency among them). A launch's cache-warm CTAs
 * (core/cache_warm.h), which write nothing, are not run; the kernel prefetches the next phase's
 * weights and instructions while it waits at a barrier instead. The hosted kernels are those of
 * a single-sequence decode graph (one 16-token tile per projection); a hosted kernel launched
 * past its hosted shape (a GDN record wider than 8 columns) stays a kernel node.
 *
 * The graph is a single chain (each node has at most one dependency and one dependent); other
 * nodes (copies, fills, kernels the persistent kernel does not host) stay as they are. Returns the
 * shared device arenas the rewritten nodes reference (phase tables, captured arguments and grid
 * barrier word); the caller keeps it alive with the graph and every executable instantiated from
 * it. Byte-identical runs share a phase image within the supplied startup owner; images are
 * packed into fixed-address arenas. Graphs sharing an image must execute on one ordered stream
 * without overlap, because their grid-barrier counter is shared too. The represented phases and
 * their order are unchanged.
 * These kernels require cooperative dispatch; occupancy alone is insufficient for whole-grid
 * synchronization. Before replay, the caller must order other streams' already-submitted
 * compute/transfer kernels before the resident grid; the Engine joins load, copy, and disk
 * scatter work at its single worker's round boundary.
 */
[[nodiscard]] PersistentDecodeImageOwners persistent_decode_lower(hipGraph_t graph,
                                                                  PersistentDecodeStorage& images);

} // namespace ninfer::ops
