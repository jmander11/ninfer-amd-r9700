# Shared FP8-context capacity recovery

This create-only successor retains the eight successful nonhybrid capacity matrices
from `r9700-terminal-base-panel-attention-20260921`. Its stopped hybrid OOMs and
partial outputs remain untouched and are not capacity exclusions.

Only four hybrid matrices (G16/G32 × dense/XAttention, C1–4) are rerun. Whole timings
and graph/eager controls are fresh for every capacity-and-quality-eligible candidate.
The valid mixed-XAttention numerical exclusions remain excluded. All twelve candidates
and the same published quality/chunk-2048 authorities remain in final selection.

Accounted physical proofs are now complete: `qualification-accounted/` passes all
fifteen prepared profiles with zero sampled BF16-step error and zero binding allocation.
`startup-accounted/result.json` passes with 5,213,519,872 bytes actually free versus
5,202,160,896 planned slack (11,358,976 bytes extra). These stages must not be rerun;
the instructions below describe their retained producer contract. Capacity, whole,
controls and selection are separate remaining stages.

## Prerequisites and ordering

The first `qualification/` and standalone model-startup attempt are historical
diagnostics. The first real C1 report exposed a 1,223,936-byte startup accounting
shortfall despite removing the former 3.9 GB library-resource gap. It is not a passing
gate and must not be overwritten or silently accepted.

The corrected builds are explicit:

- `build-r9700-fp8-accounted-g16-20260921`
- `build-r9700-fp8-accounted-g32-20260921`
- `build-r9700-fp8-accounted-xattention-g16-20260921`
- `build-r9700-fp8-accounted-xattention-g32-20260921`

After the accounted implementation and oracle receive independent review, root runs
`bash profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921/commands.sh qualify`.
This requires a completed dense-G16 qualifier build, exact original compiled options,
idle R9700/auto power and the correct HIP PCI binding. It freezes executable, source,
cache and command identities before launching the public-Op qualifier, and writes only
`qualification-accounted/`. All fifteen prepared algorithm profiles must match their
independent-context controls; sampled FP64 eager and two poisoned graph replays must
pass at both oracle widths for three real FP8 shapes. Memory snapshots are diagnostic,
not substitutes for the real-model accounting gate. Preparation must precede workspace
binding, an unbound run must be rejected, and binding must allocate zero further bytes.

An independent source reviewer writes the explicit source-review JSON with the helper's
`ninfer_r9700_fp8_context_source_review` schema, `unchanged_arithmetic` status, fixed
scope and hashes of the actual allocation/accounting source files. This is a semantic
review prerequisite; hashes alone do not authorize numerical evidence reuse.

Root then runs `commands.sh startup --review /absolute/path/to/reviewed-accounting-source.json`.
This pre-freeze stage writes `startup-accounted/`, binding the actual artifact, new
benchmark/planner, chunk, qualified executable receipt and source review before its
single C1 model process. It validates the full ordinary capacity report and requires
actual startup free memory at least its planned slack. No unexplained allowance or
tolerance is introduced, and failed reports remain retained without a passing result.

Freeze uses both explicit real proof paths:

```sh
bash profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921/commands.sh freeze \
  --review /absolute/path/to/reviewed-accounting-source.json \
  --model-startup profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921/startup-accounted/result.json
```

Those paths are required local completed proofs, not placeholders accepted by the tool.
`freeze` exclusively publishes `resource-recovery.json` only after validation. It binds
the review, successful Op and real-model proofs, original authorities, original stopped
attempt, all eight retained matrix/report sets, and four precise old→new hybrid
benchmark/planner identities. Original artifact receipts and all compiled options must
remain exact. It does not transfer capacity or performance results to the new binaries.

Then run `commands.sh preflight` (read-only), `commands.sh capacity`,
`commands.sh whole`, the local `controls.py run` with Python 3.11, and finally
`commands.sh select`. Each execution namespace is create-only; no stage uses `--resume`
or removes failed outputs. Capacity recombines eight original plus four new matrices,
revalidates actual startup slack, and retains genuine capacity exclusions. Capacity,
whole, and controls use the same binary/planner for each profile. The resource bridge
is carried into and revalidated by schema-v7 terminal selection.

No chunk sweep, numerical-quality campaign, artifact conversion, DFlash timing, or
production cutover is launched by this package. Root serializes all GPU execution.
