# Benchmarks

The native R9700 product exposes two whole-product benchmark executables. Configure with
`-DNINFER_BUILD_BENCHMARKS=ON`; kernel candidates remain qualified and timed by the focused
physical gfx1201 executables under `tools/r9700` and do not enter this graph.

`ninfer_bench` drives the public Engine for prefill/decode and fixed-concurrency measurements.
`ninfer_qwen3_8_27b_mtp_round_bench` drives the exact target Program for MTP round attribution.
Both require a real converted Qwen3.8-27B R9700 artifact for execution.

The `pareto` matrix retains phase-separated prefill and batched decode. Its
`pareto-whole` supplement submits fresh prompts and measures end-to-end makespan/output throughput;
`pareto-feasibility` asks the Engine to prove that the required 32K workload fits with standard
headroom and records whether device memory or the configured logical context ceiling binds. It is
not a maximum-capacity measurement and must not populate a Pareto capacity objective.
`pareto-capacity` instead sets the per-request ceiling to the model-native 262,144 tokens. Its
automatic result is the exact effective maximum: device memory may bind below `C*262144`, while
the model context binds when the full addressable curve fits. All three evidence classes are
required for a complete model Pareto record.

Use the procedure in `docs/maintainer/kernel-iteration.md`. The maintained standalone entry points
include:

```bash
cmake -S . -B build-r9700 -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-r9700 --target \
  ninfer_bench ninfer_qwen3_8_27b_mtp_round_bench --parallel 8
```

Run each resulting executable directly on the selected R9700. These programs combine independent
exact or FP64 qualification with unprofiled HIP-event timing at their declared real shapes. Their
accepted routes, options, baseline measurements, and ISA/resource commands are documented in
`tools/r9700/README.md`.

Profiler interception is for one named attribution question and never selects a route. The
benchmark's `--profile-measured` boundary drives ROCm selected-region collection, excluding model
load, graph priming, and warmup. For a selected whole-inference matrix point, use
`tools/bench/prepare_whole_profile.py`; it binds the generated command to the exact completed
matrix, artifact, executable, report, concurrency, and workload. The maintained lower-level
trace/counter recipes are:

```bash
make -C tools/r9700 profile-trace
make -C tools/r9700 profile-attention-trace
make -C tools/r9700 profile-pmc
```

The MTP benchmark is target-private because it measures an internal round boundary. All general
inference and concurrency claims use `ninfer_bench` through the public Engine. Neither executable
selects a kernel route by itself; final claims require the real artifact and the gates described
in `docs/performance.md`.

Schema-v18 JSON may retain per-repetition/per-lane greedy token IDs with `--retain-token-ids`;
this flag is JSON-only and is intended for exact same-artifact route comparison rather than normal
throughput output. The `tools/bench/run_ninfer_bench_matrix.py --preset dflash-pareto` campaign
uses it to gate ordinary-versus-DFlash token identity while separately retaining DFlash acceptance
and speed. Its two syncing proposal/selector diagnostics retain an exact repeated proposal/target
trace and are isolated from performance rows.
Before the larger Pareto campaign, use `--preset dflash-shortlist` to compare every supported
DFlash draft length on one representative 8K-context, 256-token decode through the public Engine.
The converted all-Q4 and mixed DFlash companions remain evaluation identities until base-profile
selection. Under the C=1..4 product limit, both base recipes have complete retained pre-promotion
native-context capacity controls at every supported concurrency; the historical mixed C7/C8
failures no longer exclude that recipe. Current selection still requires all C=1..4 capacity cells
to be rerun against the promoted executable and selected prefill chunk. Assign the physical DFlash
campaign only after the corresponding base recipe and cache group win that selection.
The preset is deliberately fixed at C=1: one shared ordinary control, K=1..11 performance rows
with two measured repetitions after one warmup, and one syncing first-reject diagnostic per K.
Every DFlash command uses the optimized proposal head and an explicit verification width. K=1..5
use W=K+1 single-block chains, K=6/7 use W=12 single-block packed trees, and K=8..11 use W=K+1
two-block chains.

```bash
python3 tools/bench/run_ninfer_bench_matrix.py \
  --preset dflash-shortlist \
  --weights out/qwen3.8-27b-r9700-q4g64-dflash2-q4-eval.ninfer \
  --expected-kv-value-group 16 \
  --expected-q4-activation-bits 8 \
  --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --output-dir profiles/bench/dflash-shortlist-c1
```

`dflash-shortlist.json` binds the artifact and executable hashes, requested and resolved K/W and
topology, memory/capacity, exact ordinary-control parity, accepted tokens per round,
target-equivalent generated tokens/s, fallback rate, and first-reject rate. Synced diagnostic
timings are discarded and never copied into performance fields. The speed ranking admits only
complete exact-parity rows. The separate non-dominated list keeps speed, acceptance, fallback,
and repair tradeoffs intact and supplies the K candidates for matched C=1 and C=4 follow-up;
there is intentionally no single blended score.
