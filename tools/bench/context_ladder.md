# Matched context speed and PPL ladders

`context_ladder.py` runs **serial C1** prefill measurements with explicit binary, artifact,
corpus, chunk and compiled attention profile. It supports arbitrary increasing context lengths,
retains each command/log/report, and resumes completed matching cells. A failed/incomplete cell
is preserved: inspect it, then use a fresh directory rather than overwriting it. `--collect-only`
validates existing reports without executing anything; imported binary/power provenance remains
the responsibility of the original package. `manifest.json` records the workload;
`ladder.json` contains completed cells. Model path, size and reported identity are retained without
re-reading a large weight payload to hash it. Keep artifacts immutable during a campaign.

Stop Compose first. Keep inference, profiling, compilation and conversion serial; build with
12 jobs (maximum14). The runner requires the R9700's power mode already be `auto`, and never changes it.
It does not stop other GPU applications or restart the server. Do not run another GPU job alongside it.

## Speed

From the repository root, with the currently installed artifact:

```bash
model=/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/qwen3.8-27b-r9700-q4-fp8-selective-cap-n16k16-dflash2-q4-eval.ninfer
ids=profiles/bench/r9700-compact-mixed-delivery-20260923/code.ids
python3.11 tools/bench/context_ladder.py run --kind speed \
  --binary build-r9700/bench/ninfer_bench --attention dense \
  --weights "$model" --ids "$ids" --contexts 8192 16384 32768 65536 \
  --chunk 2048 --max-context 131200 --spec dflash --draft-tokens 5 --lm-head-draft \
  --allow-cyclic-speed-corpus --out profiles/bench/context-dense
```

This matches the deployment's loaded DFlash backend, but `-p` measures prefill only: no speculative
round is executed. Default warmup0/repetitions1 is a screening measurement, not a statistical
ceiling claim. Use identical `--warmup` and `--repetitions` on both sides when confirming a win.
The retained128K dense attempt was aborted by the user after >46 minutes.
Do not repeat that unchanged run; start with8K/32K when testing a new candidate.
The September25 baseline is retained at `profiles/bench/r9700-prefill-context-sweep-20260925`;
use that as `--out` with `--collect-only --contexts 8192 16384 32768 65536`
and otherwise identical arguments to collect the four completed points without rerunning.

Average rate is active request prefill tokens/second, excluding model loading. Tail rate comes
from the Engine's existing last-second window: completed chunk work is prorated over that window
and the token count is rounded. It approximates late-context rate, not an instantaneous kernel
measurement. Every repetition's rate/window is retained; the summary averages repetition rates.

The example intentionally cycles4096 code tokens, matching the existing baseline. Repetition can
affect XAttention sparsity, so any speedup on this input is **only a cyclic-corpus result**.
For representative XAttention selection, supply a natural `.ids` corpus at least as long as the
largest context and omit `--allow-cyclic-speed-corpus`. Run both profiles on the same input.

## XAttention build and comparison

Use a separate qualification build with the same source, compiler, artifact, activation and cache
settings as the dense control. The only intended build differences are:

```text
NINFER_R9700_XATTENTION_QUALIFICATION=ON
NINFER_R9700_XATTENTION_STRIDE=16
NINFER_R9700_XATTENTION_TAU_PERMILLE=900
```

Configure the separate build with the dense build's remaining options explicitly; do not assume
CMake defaults match its activation profile. Build `ninfer_bench` and `ninfer-ppl` there, serially,
with `cmake --build <qualification-build> --parallel 12 --target ninfer_bench ninfer-ppl`.
This is not a product runtime attention flag and does not promote XAttention.

Repeat the speed command with the qualification binary, `--attention b128-s16-tau900`, and a
fresh output directory. Then:

```bash
python3.11 tools/bench/context_ladder.py compare \
  --dense profiles/bench/context-dense/ladder.json \
  --candidate profiles/bench/context-xattention/ladder.json \
  --out profiles/bench/context-speed-comparison.json
```

Comparison requires matching complete workloads, corpus content, weight identity, activation/cache
profiles, hardware and runtime. It rechecks retained reports and reports average and tail speedup.
If a future kernel/compiler change invalidates the old control, run a fresh matched pair; a preserved
historical baseline does not become a same-source A/B simply because the harness can read it.

## Matched PPL

Quality is a separate run using **natural token IDs**, never the cyclic timing workload. Choose
one explicit corpus per pair; repeat for additional domains. The same first P IDs and scored
positions `skip .. P-2` are used by both binaries. For example, once a sufficiently long local
natural corpus exists (nothing is downloaded or generated implicitly):

```bash
quality_ids=/absolute/path/to/natural-long-document.ids
python3.11 tools/bench/context_ladder.py run --kind ppl \
  --binary build-r9700/apps/ninfer-ppl --attention dense \
  --weights "$model" --ids "$quality_ids" --contexts 8192 32768 \
  --chunk 2048 --skip half --out profiles/ppl/context-dense
python3.11 tools/bench/context_ladder.py run --kind ppl \
  --binary /absolute/path/to/qualification-build/apps/ninfer-ppl \
  --attention b128-s16-tau900 --weights "$model" --ids "$quality_ids" \
  --contexts 8192 32768 --chunk 2048 --skip half --out profiles/ppl/context-xattention
python3.11 tools/bench/context_ladder.py compare \
  --dense profiles/ppl/context-dense/ladder.json \
  --candidate profiles/ppl/context-xattention/ladder.json \
  --out profiles/ppl/context-quality-comparison.json
```

PPL runs teacher-forced **prefill, speculation off**, retaining JSON, per-position `.nllf32` and
`.argmaxi32`. Collection checks lengths, finiteness and summary consistency. Comparison reports
mean-NLL delta, relative PPL, worst per-position delta and newly severe positions at the scorer's
reported threshold. It deliberately does not invent a quality threshold or admission decision;
the live ledger owns those gates, including long-context retrieval before production cutover.

The frozen5090 reference remains under `tools/ppl/fixtures/nvfp4-5090-20260922/`, with corpus IDs,
scores and provenance. Its4K prefill/decode spans are not8K–128K quality references. Do not compare
PPL across different inputs, score windows or schedules. The legacy `tools/ppl/compare_xattention.py`
still owns the distinct fixed8K/32K BF16-source G16/G32 admission campaign; this exploratory ladder
does not replace or manufacture its gate artifacts.

## Focused trace

Profile one point **after** unprofiled timing, using the same binary/model/corpus/chunk. Keep the
profile separate from the speed ladder. Example32K capture (fresh output directory):

```bash
trace=profiles/rocprof/context-dense32k
test ! -e "$trace"
/opt/rocm/bin/rocprofv3 --selected-regions -f csv rocpd -d "$trace" -o context \
  --marker-trace --kernel-trace --memory-copy-trace -- \
  build-r9700/bench/ninfer_bench --weights "$model" --corpus "$ids" \
  --device 0 --concurrency 1 --whole-pg 32768,16 --prefill-chunk 2048 \
  --max-ctx 131200 --kv-capacity workload --spec dflash --draft-tokens 5 --lm-head-draft \
  --warmup 0 -r 1 --profile-measured --output json --output-file "$trace/report.json"
python3.11 tools/bench/analyze_context_trace.py \
  --database "$trace/context_results.db" --markers "$trace/context_marker_api_trace.csv" \
  --out "$trace/attribution.json"
```

Use the qualification binary and a separate trace directory for XAttention. The analyzer groups
all kernel symbols (not just dense kernels), resources and launch sizes, plus chunk service totals.
Its scope is one C1 request on one queue with graph decode after prefill. Graph0 includes prefill
and host-launched setup; chunk windows use successive start markers, with the final window ending
at the first graph dispatch. These are service-attribution windows, not exact wall-clock phases.
No bandwidth, cache-hit or stall claim follows from a kernel trace without hardware counters.
