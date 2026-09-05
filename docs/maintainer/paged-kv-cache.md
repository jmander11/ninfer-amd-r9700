# FP8-K/INT4-V paged context store

This document defines the growing key/value state for the sole Qwen3.8-27B R9700 product. It is
the storage and publication authority used by the fixed-concurrency runtime, attention Ops,
prefix retention, and host-RAM spill. Kernel arithmetic is defined by the attention and Op
authorities; scheduling is defined by `concurrent-inference-architecture.md`.

## 1. Product contract

The Text and MTP growing caches have exactly three persistent planes:

- K: one OCP FP8 E4M3FN byte per feature;
- V: signed canonical INT4, two codes per byte; and
- V scale: one IEEE FP16 scale for each fixed feature group.

There is no K scale, K mean, homogeneous cache dtype, runtime cache-format selector, or alternate
growing-cache path. Represented BF16 K/V inputs are quantized only when appended. Attention reads
the stored representation directly; it never gathers the cache into a request-contiguous buffer.

The product is one resident model on one `gfx1201` wave32 device with startup-fixed concurrency
`C=1..4`. A single request may consume most of the shared physical page pool. Active, retained,
and speculative provisional state use the same entitlement accounting.

The cache owns storage and physical mapping, not model scheduling. It does not interpret request
priority, MTP acceptance, DFlash selection, or GDN state. The target runtime owns those semantics
and changes the published cache frontier only through the transaction API.

## 2. Granularities and geometry

Three independent granularities must not be conflated:

| Granularity | Meaning | Contract |
|---|---|---|
| allocation | payload acquired or released together | one 64-token page group |
| publication | logical positions visible to consumers | one token |
| reuse | positions with complete continuation state | target checkpoint only |

A page boundary is neither an attention boundary nor a reusable-prefix boundary. A valid frontier
may end at any token inside a page. Truncating cache publication does not prove that the rest of the
model state can resume there; the target must also own a matching complete checkpoint.

For the Qwen3.8 Text geometry, `D=256`, `Hkv=4`, `Lfull=16`, page size `P=64`, and provisional value
group `G=16`:

```text
bytes_per_token_per_layer = Hkv * (D + D/2 + 2*D/G)
                          = 4 * (256 + 128 + 32)
                          = 1664 bytes

Text bytes_per_token      = 16 * 1664 = 26624 bytes
Text page_group_bytes     = 64 * 26624 = 1.625 MiB
MTP page_group_bytes      = 64 * 1664  = 0.1015625 MiB
```

`G=16` is the current whole-attention timing leader, not a finalized public artifact decision.
The real paired quality, capacity, and whole-inference Pareto gate must compare G16 and G32 before
the losing branch is deleted.

## 3. Pools

Startup constructs a fixed set of typed pools:

```text
ordinary or DFlash Engine:
    Text FP8-K/INT4-V pool

MTP Engine:
    Text FP8-K/INT4-V pool
    MTP FP8-K/INT4-V pool
```

Text and MTP are physically separate because they have different layer counts and publication
frontiers. Each pool nevertheless uses the same allocator semantics and current codec profile.

The selected DFlash2 target has no DFlash growing full-context layers. Its local, rewrite, and
staging state is fixed BF16 cyclic storage with separate ownership. GDN convolution and recurrence
state, Vision intermediates, and operator-transient query K/V are also outside this store.

## 4. Capacity resolution

Let:

```text
S      = EngineOptions.max_context
C      = EngineOptions.max_concurrency
P      = 64
L      = ceil(S/P) logical pages per allocation
M_min  = max(L,C)
M_max  = C*L
M      = resolved Text physical page-group count
```

An explicit capacity converts tokens to `ceil(tokens/P)` and must lie in `[M_min,M_max]`. It must
also name at least `S` tokens. Automatic capacity evaluates the complete target layout after
weights are resident:

```text
B(M)   = persistent state + workspace + request transient + Device Graph allowance
B_min  = B(M_min)
B_step = B(M_min+1) - B(M_min)
M      = min(M_max, M_min + floor((F-R-B_min)/B_step))
```

`F` is device memory available after weights and `R` is the configured sizing headroom. The CLI
and server default to 1 GiB. The same production layout builder supplies `B_min` and `B_step`; the
common resolver does not duplicate model dimensions or bytes-per-token formulas.

The runtime reports configured `S`, resolved `M*P`, page counts, reservation bytes, headroom, and
planned slack. Tail capacity created by page rounding is storage padding and never permits a
sequence frontier beyond `S`.

The MTP pool has `M + C*ceil((K-1)/P)` physical page groups for startup-fixed draft window `K`.
The extra physical headroom supports provisional MTP growth; it does not increase any sequence's
logical context ceiling. DFlash adds no growing pool.

### Historical all-Q4 capacity measurement

The superseded schema-v19 reports bind the qualified fixed physical plane candidate used by the earlier model
campaign (token-fastest K,
feature-fastest V, feature-fastest V-scale), the byte-identical all-Q4 artifact, and the compiled
cache group. The strict automatic-capacity validator resolves:

| C | G16 tokens | G16 constraint | G32 tokens | G32 constraint |
|---:|---:|---|---:|---|
| 1 | 262,144 | model context | 262,144 | model context |
| 2 | 524,288 | model context | 524,288 | model context |
| 3 | 570,304 | device memory | 593,152 | device memory |
| 4 | 558,080 | device memory | 580,416 | device memory |
| 5 | 545,856 | device memory | 567,680 | device memory |
| 6 | 533,632 | device memory | 555,008 | device memory |
| 7 | 521,408 | device memory | 542,272 | device memory |
| 8 | 509,184 | device memory | 529,536 | device memory |

The retained historical schema-v12 directories are
`profiles/bench/pareto-capacity-markerfree-layout-all-q4-g16-20260903/` and
`profiles/bench/pareto-capacity-markerfree-layout-all-q4-g32-20260903/`. Their manifest SHA-256
values are `be1d5727948512560f28a7e149547714f9b80dca96169646a2eef0310a883feb` and
`c7e5020b03cd104440011fcc17095e3166defdc59045274192e6b930b027d922`; both bind artifact SHA-256
`19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9`. The respective benchmark
SHA-256 values are `86763d6d3ac2ff8fc27a3815f3816aaba44a11acfaf4ebf54ca5d8991ddc7d65`
and `2c9bb5a64f0e25a307f3f2b783a527ed6b42d1671c6e826d7ca98545052e9b31`.

These once superseded every earlier capacity generation, but are themselves historical after the
C=1..4 product-cap migration and cannot enter the current selection. Fresh exact C=1..4 capacity
matrices are required for both eligible recipes, both cache groups, and each candidate execution
profile. The still earlier generations are the
`pareto-capacity-layout-*` pair with always-on nested ROCTX markers, the completed G16 and
interrupted G32 `pareto-capacity-unprofiled-layout-*` intermediate runs, and the schema-v11
manifest/schema-v18 report `pareto-capacity-max-*` pair. All old raw evidence remains unchanged as
historical evidence; none is valid input to the current selection frontier.

## 5. Three-plane physical layout

Every layer has three required planes. A page-group ID identifies the same 64 logical positions
across all three planes and all layers in that pool. There is one I32 block-table row per allocation,
not a table per plane, layer, or head.

Each plane independently selects one of two closed physical orders:

- feature-fastest, page-major; or
- token-fastest, head-major.

The startup profile fixes all three orders. Requests and runtime modes cannot select layouts.
The current provisional profile is:

| Plane | Physical order |
|---|---|
| K FP8 | token-fastest, head-major |
| V INT4 | feature-fastest, page-major |
| V-scale FP16 | feature-fastest, page-major |

The generic pool records plane extents, element type, alignment, slab order, and intra-page order.
The typed wrapper validates exact K/V/scale geometry before exposing a layer view. Storage contains
no optional fourth plane.

For a logical position `p`:

```text
logical_page = p / 64
token_offset = p % 64
physical_page = block_table[logical_page]
```

The plane layout then maps `(feature-or-group, token_offset, head, physical_page)` to bytes. The
same physical page ID is used for K, V, and V scale. Page IDs are pool-local I32 values; no device
pointer is stored in the table.

The layout sweep may instantiate all eight independent plane-order combinations and G16/G32 in
qualification tools. Production startup instantiates only one compile-time profile, currently the
provisional G16 profile above. After the real model gate, losing groups and layout-dispatch branches
must be removed.

## 6. Allocation, entitlement, and mapping

A `PagedKVAllocation` owns:

- a page entitlement charged immediately against its pool;
- an ordered logical-to-physical page vector;
- at most one bound block-table row; and
- a mapping generation incremented whenever the mapping or row identity changes.

Reservation guarantees capacity but does not materialize pages. Materialization obtains pool-local
page IDs, zeroes the new page groups, and publishes their IDs to the bound table row on the ordered
stream. Mappings need not be physically contiguous. Duplicate physical IDs within a production
allocation are invalid.

Entitlement, materialized-page count, and published token frontier are separate values. Shrinking
publication need not immediately release a partial tail page. Trimming complete trailing pages
invalidates old reads and returns their IDs to the pool. Recycled storage is not visible until a
new owner has written and published it.

One sequence state owns independent Text and optional MTP allocations plus their publication
objects. A retained state transfers ownership rather than sharing writable pages. The current
product does not implement page reference counts, copy-on-write fan-out, active offload, or shared
writable prefixes.

## 7. Publication state and transactions

Each sequence/pool publication records:

- `valid_frontier`;
- health/poison state;
- whether a transaction is open; and
- a monotonically increasing transaction generation.

Only one mutation transaction may be open for a publication. Opening a transaction invalidates
previous layer-read capabilities. The caller supplies fixed-address device status and position
workspace; segmented MTP transactions also supply a cursor. Cache Ops do not allocate memory,
stage hidden positions, or synchronize the host internally.

### 7.1 Ordinary append

An ordinary append represents a contiguous suffix beginning exactly at the closed frontier. One
transaction launches every layer exactly once on one ordered HIP stream. Each layer consumes
represented BF16 K/V, writes the three physical planes, and exposes only a same-stream pending
read. Commit is legal only after every layer launched, the stream completed, and device status was
zero. Commit then advances the frontier once for the whole pool.

### 7.2 Device-prefix and segmented MTP append

A device-prefix transaction captures a fixed maximum panel plus a device I32 active count. Zero is
valid and produces no visible append. The host does not read the count between graph nodes.

One MTP round uses a segmented transaction spanning alignment and every autoregressive segment.
A device base frontier initializes a fixed cursor and remains immutable for the lifetime of that
transaction. Speculative acceptance advances a separate fixed-address frontier vector; it must not
reuse or mutate the transaction base storage. Each segment validates and advances the private
cursor on the ordered stream. A dynamic device table-row selector is permitted only within the
allocation-owned row set. Eager and Device Graph routes use the same addresses and one post-round
status/cursor resolution. Resolution publishes the complete Text verify suffix and only the
licensed MTP suffix.

### 7.3 Failure

The append codec ORs failure bits for invalid positions/counts/addressing, nonfinite represented
inputs, or unrepresentable FP16 V scales. A failed or abandoned launched transaction:

- does not advance the public frontier;
- closes every pending read;
- poisons only the affected sequence publication; and
- prevents stale written bytes from entering the readable domain.

Host validation failures before launch leave publication and bytes unchanged. Recovery uses a
known-good checkpoint or destroys the sequence; it never edits the frontier around an unresolved
device failure.

## 8. Read capabilities and attention

Attention cannot construct a cache read from raw pointers. It receives a non-forgeable
`PagedKVLayerRead` containing the exact allocation, layer view, bound row, mapped-page count,
mapping generation, transaction generation, and visible frontier.

A committed read is valid only while publication remains closed with the same generation and
frontier. A pending read is created only after that layer's append launch and is valid only on the
same ordered stream. This permits a decoder layer to attend newly written K/V before the all-layer
transaction commits without exposing provisional state elsewhere.

Mapping materialize/trim/bind/unbind, a new transaction, commit, abort, compaction, or row rebinding
invalidates older capabilities. Cross-stream pending reads and capabilities from another allocation
are rejected before dispatch.

The attention leaf traverses logical positions, translates each page once per shared work unit,
decodes FP8 K and signed INT4 V with FP16 scales, and evaluates causal or packed-tree visibility.
Malformed device positions/table rows make only the affected output row conspicuously NaN. A
cached-only read never writes cache storage.

## 9. Compaction and speculative rollback

Compaction is a transaction over every layer. It copies the exact packed three-plane representation,
not dequantized values. The caller supplies a retained prefix and selected suffix path with unique,
in-range positions. Commit publishes the compacted frontier only after all layers and device status
pass.

When accepted tokens already occupy their final contiguous positions, zero-copy rollback uses
publication truncation instead. Bytes beyond the retained frontier may remain in a tail page but
are invalid until a later append fully overwrites the matching K code, V code, and V-scale groups.

MTP acceptance, cancellation, and prefix reuse therefore change cache visibility through a single
transaction authority; direct frontier repair is prohibited.

## 10. Prefix retention and host-RAM spill

A reusable bundle includes every state component needed to continue, not just Text KV:

- Text allocation, publication, and exact physical mapping;
- optional MTP allocation/publication;
- GDN convolution and FP32 recurrence state;
- hidden/replay/checkpoint state; and
- fixed DFlash cyclic state when that backend is active.

The optional pinned-host tier is an exclusive FIFO of completed retained bundles not currently on
a device lane. Capture is D2H on the copy stream and keeps source pages mapped until its event
completes. Restore validates the entire image before any H2D write, chooses a free lane/mapping,
writes exact physical bytes, then orders compute after the copy event immediately before prefill.

The RAM image version binds an FP8-K/INT4-V semantic fingerprint containing codec version, page
size, physical/logical capacity, layer/head geometry, value group, and all plane orders. A mismatch
rejects before any destination byte changes. Text and MTP fingerprints are independent.

Host capacity is fixed by `--kv-ram-capacity`; `off` disables retained FIFO spill but not live-lane
checkpoint state. Captures that do not fit are dropped without blocking admission. Active requests
are never offloaded. Logged occupancy counts live host residents, not retired in-flight buffers.

## 11. Fixed-concurrency and Device Graph rules

The runtime owns one startup-fixed block-table row per possible active lane and prepares exact
batch definitions for `B=1..C`. Graph definitions bind stable pool, table, control, status, cursor,
and workspace bases. Mapping content may change between replays, but the addresses do not.

Cross-page materialization happens before replay on the same execution lane. Serving never captures
new graphs or allocates cache workspace. There is no per-row submission, per-page kernel launch,
mapping-dependent recapture, or padding of `B=1` to maximum concurrency.

Graph capture records the same append/attention/state transitions used eagerly. It does not waive
transaction commit, poisoning, or publication rules. Exact eager/graph state bytes are required at
the semantic boundary.

## 12. Prohibited implementations

- Gather paged KV into a contiguous scratch cache before attention.
- Store per-plane, per-layer, or per-head block tables.
- Store 64-bit device pointers in block tables.
- Perform one independent page-table lookup per scalar lane when a workgroup can share it.
- Launch one kernel per physical page.
- Assume adjacent logical pages have adjacent physical IDs.
- Retain a continuous-cache fallback or runtime cache-format branch.
- Allocate, repack, or read device status back inside an Op.
- Publish a frontier before all layers and device status succeed.
- Reuse a read capability across a mapping/publication generation change.
- Make K/V/scale planes independently writable by multiple heads or transactions.

## 13. Correctness qualification

The mathematical oracle evaluates attention from logical positions and independently decoded
represented cache values. It does not reproduce production page traversal, staging casts, or
reduction order. The exact codec oracle independently derives every physical byte offset and packed
code.

Required coverage for any changed cache/attention route includes:

- identity, contiguous-offset, and fragmented-permutation mappings;
- logical positions 0, 1, 31, 32, 63, 64, 65, 127, and 128 as applicable;
- append beginning mid-page and crossing one or more pages;
- G16 and G32 while group selection remains open;
- all eight independent plane-order combinations while layout selection remains open;
- exact FP8 saturation/signed-zero bytes, INT4 RNE ties, FP16 scale underflow/overflow;
- zero/full/partial device-prefix counts and segmented MTP alignment/AR append;
- same-stream pending read, committed read, cross-stream/stale-generation rejection;
- in-place tree compaction and zero-copy rollback;
- malformed/aliased mappings, invalid device row/position/count, nonfinite input, poison isolation;
- multiple allocations/table rows in one batched invocation;
- mapping changes between Device Graph replays; and
- exact host spill/restore plus semantic-fingerprint no-write rejection.

Floating output is compared to the independent FP64 formula using a criterion appropriate to the
selected implementation profile. Codec, mapping, state, and packed-cache transformations are exact.

## 14. Performance qualification

Performance decisions use HIP-event timings on an otherwise idle R9700 after correctness passes.
Short routes require repeated interleaved measurements and absolute latency; long routes use
repeatable medians. Profiler data explains a measured route but does not select one by itself.

Representative coverage is:

- suffix append and complete attention at T=1..8 and T=9/17/128;
- context 1K, 4K, 8K, and 32K;
- G16/G32 and every plane-order candidate while selection is open;
- fragmented as well as identity mappings;
- ordinary and speculative fixed-width execution; and
- final whole-Engine prefill/decode at C=1..4 once a real artifact exists.

The 32-point cache-layout sweep passed the independent layout and attention oracles. The current
G16/token-K/feature-V/feature-scale profile has normalized mean latency 1.004041, mean rank 2.188,
and 14 wins. It is provisional because operator latency cannot replace model quality, resolved
capacity, and whole-inference evidence.

If complete evidence leaves the G16 and G32 layouts for the selected byte-identical artifact
non-dominated, production does not retain both. It selects the layout with the best worst-cell
whole-inference throughput ratio after normalizing every required cell to its frontier best.
Exact ties resolve by worst-cell normalized capacity, then worst quality-tier budget consumption,
then canonical static cache/execution-profile identity when the measured objective vectors are identical. The
operator sweep above is qualification evidence and is not a tie-breaker for the static product
profile.

Measured suffix append costs about 5 to 8 microseconds at T=1..8 and 37 to 46 microseconds at T=128,
while complete attention ranges from roughly 0.66 ms at 1K/T1 to 301 ms at 32K/T128. Even a
physically impossible zero-cost append fusion is therefore below 0.8 percent at 1K and below 0.21
percent at longer contexts. A single ordinary HIP launch cannot globally order all independent
append writers before all attention readers, and per-query-head encoding violates single-writer
transaction ownership. Separate ordered append followed by the selected attention family is the
qualified architecture. At context 8,192 and above, ordinary T=1 and fixed-width T=4 use the
three-stage split-512 leaf with caller-owned score/partial/merge storage. Below that boundary,
ordinary T=1/context>=64 and T=2/context>=320 use FP8-Q/K WMMA plus FP32 score/softmax and exact
vector PV; remaining shapes use fused QK/online-FP32-softmax/PV. The T=4 split route accepts the
same causal, packed-tree, and device-active-row metadata as the fused leaf. Metadata-bearing T=1
retains the fused leaf at every context because no such split-512 form was admitted.

Final admission additionally requires:

1. paired BF16-reference 8K and 32K PPL;
2. exact greedy-token comparison at every scored position;
3. eager/Device Graph state and output parity;
4. speculative acceptance/publication checks; and
5. complete prefill/decode and C=1..4 throughput with valid attribution.

The complete BF16 source checkpoint and supported Python environment are present, and the paired
quality evidence is retained. Remaining final admission work is selection-dependent physical
eager/Device Graph parity, speculative acceptance/publication, and attributed prefill/decode
throughput across C=1..4 on the selected profile. No earlier device/backend result is product
evidence.

## 15. Fixed contract versus tunable profile

Fixed architecture:

- FP8 E4M3FN K, signed INT4 V, FP16 V scale, three planes only;
- `P=64`, pool-local I32 page-group IDs, one allocation-owned block-table row;
- Text/MTP physical separation and target-derived capacity coupling;
- generation-bound read capabilities and transaction-only publication;
- no gather, fallback, hidden allocation, runtime repack, or runtime format selection; and
- fixed-address exact-B Device Graph execution.

Currently tunable until the real-model gate:

- G16 versus G32;
- the three plane orders;
- attention tile/split/vector/LDS staging and wave assignment; and
- append workgroup geometry, provided it preserves one writer per packed destination.

Changing page size, code semantics, scale type, pool grouping, page-ID model, publication ownership,
or introducing another cache representation is an architecture change, not kernel tuning.
