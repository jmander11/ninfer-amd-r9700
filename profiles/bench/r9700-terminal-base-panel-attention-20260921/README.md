# Current terminal-base continuation

STOPPED: capacity found unaccounted FP8 library allocations and was interrupted.
See `capacity/closure.json`. Preserve this namespace; do not run its capacity,
whole, controls or select commands. A fresh resource-accounting recovery package
will reuse the eight completed non-hybrid capacity matrices. The commands below
describe the retained package, not the current next launch.

Prepared commands, not measured capacity, whole results, or terminal admission.
Run only after root review and publication of the six current quality authorities.
No DFlash companion, artifact conversion, final materialization, or XAttention
cutover is performed. The controls stage checks exact graph/eager public tokens;
final artifact and XAttention cutover retain their declared post-selection gates.

Inputs are the validated
`profiles/bench/prefill-chunk-selection-panel-attention-20260921.json` authority
(currently chunk2048; the value is read, never overridden), frozen
`profiles/bench/r9700-chunk-selection-panel-attention-20260921/inputs.json`, and
`profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/quality-authorities-receipt-bound-n16k16.json`.
Missing quality publication rejects preflight without creating outputs or launching
subprocesses. Preflight revalidates the six quality campaigns and selection, then
compares actual three N16 artifacts, four benchmark/planner pairs and benchmark
corpus against relevant frozen identities. It also binds quality artifacts and
selection source evidence to those inputs; unrelated frozen tools are not gated.

Run from the repository root, sequentially:

```sh
bash profiles/bench/r9700-terminal-base-panel-attention-20260921/commands.sh preflight
bash profiles/bench/r9700-terminal-base-panel-attention-20260921/commands.sh capacity
bash profiles/bench/r9700-terminal-base-panel-attention-20260921/commands.sh whole
bash profiles/bench/r9700-terminal-base-panel-attention-20260921/controls.sh preflight
bash profiles/bench/r9700-terminal-base-panel-attention-20260921/controls.sh run
bash profiles/bench/r9700-terminal-base-panel-attention-20260921/commands.sh select
```

`capacity`, `whole`, and `controls.sh run` require root's exclusive GPU lease and the R9700 in auto
power mode. Existing matrix runner checks auto mode and physical PCI device
identity; it never changes power settings. Python is explicitly
`/home/battlefront/.local/bin/python3.11`. Builds are
`build-r9700-selection-panel-{dense,xattention}-g{16,32}-20260921` with `--no-build`.
The four-role candidates use each matching host planner's workspace authority.

Each stage owns a create-only directory named `capacity`, `whole`, `controls`, or `select`.
All attempts, stdout/stderr and matrix-owned reports/failures are retained. Existing
stage directories prevent rerun, including failed attempts: there is no automatic
cleanup, overwrite, resume, or silent retry. Three authority-file bindings are
recorded per stage and checked across continuation stages. A failed attempt needs
an explicit follow-up decision, not an unreviewed command change.

`capacity` invokes the existing schema-v14 `pareto-capacity` preset for all twelve
recipe/group/attention combinations, each with C1..4, selected chunk, draft0 and
automatic KV capacity at the model-native context ceiling. Nonzero runner exits
are collected without dropping remaining profiles. After all twelve attempts,
the existing `validate_post_chunk_capacity_campaign` validates every outcome and
requires symmetric dense/XAttention eligibility. Malformed or unstructured failures
reject validation; measured capacity exclusions remain evidence.

`whole` revalidates capacity evidence and intersects its eligible identities with
replayed per-group quality eligibility. A sparse quality failure does not discard
its passing dense control or change its declared quality tier.
The existing `pareto-whole` preset supplies spec-none ordinary 8192/32768 prompts,
256 generated tokens, three repetitions, one warmup, and C1..4. It uses the exact
same group-specific executable/artifact as capacity. A failed whole matrix stops
the stage and preserves its results. All public token IDs are retained; the benchmark
already collects them in every run, so retaining JSON adds no timed work.

`controls.sh preflight` is read-only and requires the completed whole stage.
`controls.sh run` reuses every eligible graph report and runs its exact benchmark,
artifact, corpus, group, chunk and C1..4 workload with only graph disabled, one
repetition, zero warmup and a new output path. Every lane's 257 public tokens must
match every retained graph repetition exactly at both lengths. These eager runs
are correctness evidence, not timing admission. Inputs, commands, comparisons and
failures are create-only; no failed attempt is resumed or replaced.

`select` revalidates capacity evidence, resolves six quality campaign paths from
the published authority map, and supplies all twelve candidates to
`assemble_pareto.py --require-xattention-dense-controls`. Quality- or capacity-ineligible
candidates receive `-` for whole evidence, preserving exclusions. Existing
classification and exclusive `terminal_selection_io.publish` produce unchanged
schema-v7 authority at `select/result.json`, bound to `select/pareto-input.json`.
`select` also reopens and recomputes the exact controls before creating outputs.
Missing whole evidence, invalid controls, or an invalid terminal result cannot be
published. Publication is a base-selection result, not final artifact cutover.

CPU checks (no GPU):

```sh
/home/battlefront/.local/bin/python3.11 profiles/bench/r9700-terminal-base-panel-attention-20260921/test_campaign.py
/home/battlefront/.local/bin/python3.11 profiles/bench/r9700-terminal-base-panel-attention-20260921/test_controls.py
```
