# Bounded-panel attention qualification and admission

Prepared experiment package. It neither builds executables nor runs a profiler.
The existing trace established that fallback attention accounted for
39087.580/43838.812 ms (89.16%) of measured kernel duration. This package qualifies
the replacement and measures its unprofiled effect before another chunk campaign.

```sh
bash profiles/bench/r9700-chunked-attention-candidate-20260921/commands.sh preflight
bash profiles/bench/r9700-chunked-attention-candidate-20260921/commands.sh qualification
bash profiles/bench/r9700-chunked-attention-candidate-20260921/commands.sh whole
```

Preflight is read-only and launches no subprocess. It validates the retained
baseline report and frozen baseline executable/artifact/corpus identities, then
records identities of the candidate benchmark, qualifier, ISA, routing/planner
sources, and build configuration. Missing prerequisites stop the stage.
The baseline build is `build-r9700-selection-dense-g16-20260921`; the candidate
build is `build-r9700-chunked-attention-candidate-g16-20260921`. The shared artifact
is the exact receipt-bound `out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer`.

Qualification first applies the existing static checker to QK Bk16/Bk32,
maximum, and G16/G32 PV production specializations in
`qualification/dense_prefill_attention.s` under the candidate build. It then runs
`src/ninfer_r9700_runtime_planner_qual --host-split512-routing`, the no-argument
`qualification/dense_prefill_attention_qual`, and no-argument
`src/ninfer_r9700_full_attention_qual`. The independent FP64 qualifier includes
both value groups, appended 1024/1537 rows at context8192, full8192 panels,
1025/8192 rows at context32768, and128 rows at context262144; it also covers
panel tails, active rows/canaries, fragmented pages and graph replay. The public
leaf check covers the typed caller boundary for appended attention.
Any static or numerical failure stops qualification and leaves no passing result.

Under the root's exclusive GPU lease, each GPU stage initially requires auto
power and idle resident-memory/activity readings on PCI0000:13:00.0, then binds
HIP device0 to that PCI function. The first launch checks auto power without
rechecking the transient busy sample caused by its own HIP probe. Later launches
follow completed owned children and require released resident memory plus auto
power; they do not reject a stale sampled busy percentage from the prior child.
The paused matrix must remain paused through these stages.

Whole measurement requires a passing qualification result with identical inputs.
It runs three repetitions and one warmup for each of:

- Candidate C1 P8192, chunk1024, draft0, compared with the retained matching
  three-repetition baseline (184.8640144 mean tok/s; median44.35115479 seconds).
- Fresh baseline and candidate C1 P2048, chunk4096, draft0, to check the existing
  single-chunk fast route.

All runs use the same corpus and artifact, fixed FP8-K/INT4-V cache and no profiler.
Admission requires candidate P8192 median prefill duration at most0.90 times the
retained baseline median and candidate P2048 median at most1.02 times the fresh
baseline median. Reports retain every repetition and any emitted token evidence;
no cross-chunk token identity gate is imposed here. Arithmetic qualification and
the subsequent model quality campaign own those distinct checks.

The create-only admission authority is `whole/result.json` inside this package:
`artifact_type=ninfer_chunked_attention_whole_prefill_comparison`, schema1,
`status=admitted` or `status=not_admitted`. It binds `inputs` (source/build and
executable identities) and the passing `qualification/result.json` hash, records
all distributions, medians and thresholds, and includes both duration ratios.
A threshold failure preserves the result and returns nonzero. Borderline outcomes
require root review; there are no automatic retries. No new chunk campaign should
freeze or run unless this exact authority says `admitted` and its source/build
bindings still match. Existing qualification/whole namespaces are never reused.

CPU tests:

```sh
/home/battlefront/.local/bin/python3.11 profiles/bench/r9700-chunked-attention-candidate-20260921/test_experiment.py
```
