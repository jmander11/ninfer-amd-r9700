# DFlash T5/T6 down scale-gather qualification

Qualification only; no product route or split-K selector changes. Build:
`make -C tools/r9700 build/dflash_down_scale_gather_qual build/dflash_down_scale_gather.s build/a8q4_gdn_pair_t1.s`.
Then use this package's `commands.sh --prepare`, independent review, and `--preflight`.
Preflight performs CPU/static work in temporary storage and never launches HIP or creates attempt-1.
Only `--measure` launches the complete numerical/timing qualifier on the R9700 in auto.

The boundary accepts represented BF16 [T,17408] and Q4N16K16/G64 [5120,17408], freshly
prepares exact A8G64 scratch, then produces BF16 [T,5120]. T is exactly 5 or 6.
The challenger gathers each token's strided scale into one wave lane and broadcasts it,
replacing 5/6 serialized scale-load/wait phases with one gather. Four native IU4 WMMAs
per G64 group, signed integer decomposition, 272 serial FP32 group FMAs per output,
and final BF16 rounding are preserved. No extra global workspace or weight traffic.
Broadcasts compile to ds_bpermute plus dependency waits; their cost is part of timing.

The independent oracle decodes stored signed Q4 nibbles and exact FP16 scales, evaluates
the exact A8 represented-input dot in FP64, and retains the unrounded result. Its BF16
output criterion is relative L2 <=0.01 and a gross pointwise cap <=0.01*max_abs(reference)+1e-5,
with finite outputs required. This permits near-zero cancellation without weakening exact
candidate/incumbent BF16 parity. The A8 codec/status is independently exact. Dense distinct
token rows, negative/rescaled rows, zero rows, input/weight immutability, guards, malformed
bindings, and captured poison/stale/finite recovery precede timing.

Timing uses 3 address-distinct allocations and 24 balanced cold graph-replay pairs per width.
80 MiB scrub is outside each timed interval; both arms include fresh codec preparation.
Each width must win on every allocation and have paired ratio mean+2SE <1. The lower
two-standard-error bound on 64 times the per-Op saving must exceed 1% of the retained matched
round: T5 92.12005557 ms, T6 93.21258994 ms. Thus required lower savings are 14.3938/14.5645
microseconds per Op. These are historical projection denominators, not a current throughput claim.
A valid loser closes completed/rejected with retained receipts. No broad sweep follows.

Sources, binaries, assembly, four matched whole control reports and scripts are bound by plan.json.
Embedded ISA must match source instruction order/resources, show the gather/broadcast mechanism,
and preserve zero scratch/spills and occupancy 16. Immutable attempt-1 records process exits,
static/oracle/timing outputs, summary, closure and result.sha256. Passing authorizes only
preparation of exact-token and matched whole K4/W5 and K5/W6 gates, never production routing.
