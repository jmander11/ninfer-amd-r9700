# Selected-chunk numerical quality campaign

Reference and all six quality campaigns completed on2026-09-21. Ten of twelve profiles pass;
mixed XAttention fails its strict accuracy severe-position budget at32K for G16/G32.
The original quality stage exited1 and its results are preserved. Do not rerun reference
or quality: the repaired publisher validates measured exclusions alongside passing evidence.
This is numerical eligibility, not final production selection.
All stages require the real, validated
`profiles/bench/prefill-chunk-selection-panel-attention-20260921.json` authority.
The scripts never choose or substitute a chunk. Candidate scorers come from
`build-r9700-selection-panel-{dense,xattention}-g{16,32}-20260921/apps/ninfer-ppl`.

Stage commands from the repository root (reference/quality are completed, not restart commands):

```sh
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh preflight
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh reference
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh quality
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh publish
```

`preflight` is read-only, launches no subprocess, and validates the selected chunk,
corpus, three receipt-bound N16/K16 artifacts, and available reference evidence.
The actual artifacts and four PPL scorer hashes must match the chunk campaign's
frozen `profiles/bench/r9700-chunk-selection-panel-attention-20260921/inputs.json`.
The selected-chunk source artifacts and benchmark identities must match that same
receipt. Unrelated frozen tools are not prerequisites of this quality package.
A missing selected-chunk authority stops it before these dependent checks.
Existing output directories remain untouched; `quality` requires all six new output
directories to be absent. Interrupted attempts must be preserved or explicitly moved
before retrying. No in-place resume or automatic cleanup is performed.

At chunk 4096, `reference` validates the retained deterministic 20260904 BF16
campaign and its fresh-process repeat comparison, including the reused cells and
sidecars. It launches no reference execution. Other selected chunks first run the
full-span GDN oracle probe under the explicit reference interpreter. The stage
requires `all_pass is True`, row extents `[4095,4096]`, and matching interpreter
path/hash provenance before scoring. Its create-only
`bf16-chunk{selected_chunk}-gdn-full-span.json` remains preserved even on failure;
an existing probe blocks another attempt. Do not run a duplicate manual probe.
After this gate, the stage requires two
fresh deterministic BF16 campaigns followed by `compare_bf16_repeats.py`, before
candidate quality can execute. For those chunks, supply an existing suitable
interpreter explicitly to the reference stage:

```sh
env -u PYTHONPATH LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
  bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh reference \
  --reference-python /ssdpool2nvme/local_llm/.venv-ninfer-r9700-py311/bin/python
```

The selected CPU interpreter `/home/battlefront/.local/bin/python3.11` has no torch
and is sufficient for retained-reference reuse, candidate orchestration, and
publication. Root installed `/ssdpool2nvme/local_llm/.venv-ninfer-r9700-py311/bin/python`
with Python 3.11.16, ROCm torch 2.9.1+rocm7.2.4.git39497456, and Triton
3.5.1+rocm7.2.4.gita272dfa8. `pip check` and CPU-only imports passed;
gfx1201 is listed. The full-span GDN oracle and fresh chunk2048 BF16 A/B have now
passed on the GPU; both lengths have exactly matching NLL/token sidecars.
This package does
not install dependencies or change a Python or shared-library environment.

`quality` runs six separate all-Q4, mixed-Q4/W8, and four-role Q4/FP8 campaigns,
each for dense and XAttention, G16 and G32, and the runner's default 8192/32768
token lengths. The mixed recipe uses the accuracy tier (0.02 mean-NLL gate);
all-Q4 and four-role use capacity-speed (log(1.05)). Every campaign uses freshly
executed candidate scores and the validated BF16 authority, fixed FP8-K/INT4-V
cache, prefill-only, spec-none execution. Failed campaigns retain their reports;
the repaired stage distinguishes replay-validated numerical exclusions from malformed
or incomplete evidence. A numerical exclusion cannot be reclassified into a looser tier.

`publish` reopens each campaign through the existing Pareto quality validator,
recomputes finite aligned sidecar metrics and original fixed-tier gates, retains each
group's true eligibility, binds the exact artifact receipt and selected chunk, and uses the
existing exclusive quality authority publisher. It writes
`quality-authorities-receipt-bound-n16k16.json` only after validation.

CPU regression checks:

```sh
/home/battlefront/.local/bin/python3.11 profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/test_campaign.py
```
