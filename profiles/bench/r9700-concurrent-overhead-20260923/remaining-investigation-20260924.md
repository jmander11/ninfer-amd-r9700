# Remaining decode targets: investigation, not promotion

User requested investigation of concurrent projection tiling, attention PV and GDN
recording. Production remains a24d51f9; no kernel, precision, weights or build changed.
R9700 gfx1201, auto, installed cap26/Q4 DFlash, dense G16, P4096/G128, chunk2048.
Fresh C3K4 trace passed same-C ordinary tokens, with no memory-scope pressure events;
C4 uses the completed final trace. Both have31 steady main-graph rounds. Trace
durations are attribution only; retained unprofiled rates remain190.96/181.89tok/s.

| Owner | C3 us/call | C4 us/call | C3/C4 calls per steady round |
|---|---:|---:|---:|
| Gate/up N34816/K5120 | 180.50 | 324.83 | 67 / 67 |
| Down N5120/K17408 | 93.76 | 94.10 | 67 / 67 |
| Attention PV | 176.04 | 170.33 | 48 / 64 |
| GDN recurrence/record | 52.45 | 78.18 | 48 / 48 |

## 1. Gate/up cross-tile weight reuse: highest priority

T15 uses one16-row tile; T20 uses two. The current independent waves issue the
same weight/scale loads for both tiles and compute padded tail entries. This is
not proof of doubled DRAM traffic: cache can reuse the addresses. Gate/up grows
1.80x while down remains flat, so do not blanket-rewrite every T>16 projection.
Gate/up alone adds9.67ms per steady round and owns30.01% of final C4 kernel service.
The old T5/T6 bandwidth exclusion does not decide this T20 duplicated-load question.

Next bounded candidate: one wave processes both M tiles, reusing each loaded
weight/scale payload across independent accumulators; preserve ascending G64
arithmetic and current one-successor overlap. Mask the unused tail's epilogue.
It reduces weight-load instructions but does not halve WMMA work. Register pressure,
lost occupancy and premature load waits can erase the benefit. This is not rejected
two-group lookahead or split-K. First screen exact T20 gate/up, with T15 control;
extend to T18/24 only after a material result. Use complete BF16-to-A8-to-Q4 Linear
oracle/codec/guard/graph checks, inspect IU4 and spills, then unprofiled Op timing.
Only a qualified winner warrants whole C4K4 timing; C3 must remain unchanged.

A hypothetical20% gate/up reduction saves4.35ms/steady round, or6.0% of kernel
service, not6.0% of end-to-end time. Recovering the full9.67ms difference is an
optimistic comparison, not a forecast. C3/C4 unprofiled acceptance lengths also
differ4.021/3.864; therefore the aggregate throughput difference is not solely tiling.

## 2. PV adjacent-feature reuse: second priority

Current stage32 assigns one feature per lane. Adjacent features load the same
packed INT4 byte, scale and probability, then select opposite nibbles. Assign two
adjacent features to each lane, reuse those operands and preserve two independent
ascending-token FP32 FMA chains. Feature CTAs fall8→4; query rows remain separate.
This reduces repeated load/decode instruction instances, not minimum cache bytes
or total FMAs. It is distinct from rejected all-row sharing, stage64 and explicit
page-pointer candidates. Do not repeat those candidates unchanged.

Current emitted PV is111VGPR/48SGPR, wave32, zero LDS/private scratch. Inspect the
paired-feature code before timing: added accumulators and half the CTAs may reduce
latency hiding. C4 has more PV calls, not slower individual calls; there is no
measured C4-specific PV regression. At15.03% service share, a hypothetical20%
PV improvement saves3.0% of kernel service. Require the same independent complete
attention oracle/profile bounds, exact serial/graph, page/odd-tail guards, and
warm/cold complete-attention timing W4/5/6 at short and4K contexts. Do not infer
an attention win from nibble-unpack instruction counts alone.

## 3. GDN exact-tree normalization: third priority

`record_kernel<false>` computes the recurrence and stores compact replay records;
it does not write full state snapshots per token. Removing snapshots is therefore
not a valid optimization. Its1024-thread CTA retains state in registers, while
normalizing Q/K through repeated shared reductions and block barriers every token.
Current compile-bound wave-QK candidate is off; its changed FMA/reduction order
must not be silently promoted as an exact replacement.

A distinct exact-tree mechanism: lanes l<32 independently square inputs at
l,l+32,l+64,l+96 with separate multiplies, form (a0+a2)+(a1+a3), then shuffle-down
16,8,4,2,1. This reproduces the current stride64→32→16→... sum tree, avoiding
most block barriers without changing rsqrt, recurrence, key/value/gate records or
commit/rollback. Retain barriers needed for shared operand visibility. Inspect
the actual ISA for accidental contraction/reassociation; source equivalence alone
is insufficient. Public independent FP64 output/state checks plus exact outputs,
records and replayed FP32 state are required before timing.

The52.45→78.18us C3/C4 growth motivates synchronization/scheduling investigation
but does not prove barrier stalls. At5.17% service share, a20–30% Op win saves
1.0–1.55% of service. A bounded screen is appropriate; a larger state architecture
rewrite is not justified. Preserve all supported extent and zero/partial-tail behavior.

## Evidence and limits

One fresh trace only; no conversion, compilation, power change, PMC capture or
candidate implementation. No new speedup is claimed. Physical DRAM bytes and stall
attribution remain unmeasured here; instruction/source observations are not counters.
The three mechanisms above merit separate qualification, not automatic promotion.
Independent CPU review agrees on the PV/GDN mechanisms and risks.

Fresh invocation: `run_cell.py --label investigate-c3-k4-trace-20260924
--concurrency 3 --draft 4 --trace` under a serial systemd user scope with20GiB
MemoryHigh,24GiB MemoryMax and zero swap. Exact command/receipt/report are adjacent.
Fresh raw database: `investigate-c3-k4-trace-20260924/raw/r2d2/1565862_results.db`.
Use `analyze_trace.py` with that explicit path and a new output path; do not glob
for a latest database. `investigate_remaining.py` joins the two attribution reports
and retained unprofiled acceptance into `remaining-investigation-20260924.json`.
Helpers create outputs exclusively; use fresh output identities for new captures.
