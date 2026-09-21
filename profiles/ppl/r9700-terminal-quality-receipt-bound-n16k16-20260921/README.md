# Selected-chunk numerical quality campaign

Prepared commands only; no numerical results or selection are asserted by this package.
All stages require the real, validated
`profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260921.json` authority.
The scripts never choose or substitute a chunk. Candidate scorers come from
`build-r9700-selection-{dense,xattention}-g{16,32}-20260921/apps/ninfer-ppl`.

Run from the repository root:

```sh
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh preflight
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh reference
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh quality
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh publish
```

`preflight` is read-only, launches no subprocess, and validates the selected chunk,
corpus, three receipt-bound N16/K16 artifacts, and available reference evidence.
The actual artifacts and four PPL scorer hashes must match the chunk campaign's
frozen `profiles/bench/r9700-chunk-selection-receipt-bound-n16k16-20260921/inputs.json`.
The selected-chunk source artifacts and benchmark identities must match that same
receipt. Unrelated frozen tools are not prerequisites of this quality package.
A missing selected-chunk authority stops it before these dependent checks.
Existing output directories remain untouched; `quality` requires all six new output
directories to be absent. Interrupted attempts must be preserved or explicitly moved
before retrying. No in-place resume or automatic cleanup is performed.

At chunk 4096, `reference` validates the retained deterministic 20260904 BF16
campaign and its fresh-process repeat comparison, including the reused cells and
sidecars. It launches no reference execution. Other selected chunks require two
fresh deterministic BF16 campaigns followed by `compare_bf16_repeats.py`, before
candidate quality can execute. For those chunks, supply an existing suitable
interpreter explicitly to the reference stage:

```sh
bash profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/commands.sh reference --reference-python /path/to/python3.11
```

The selected CPU interpreter `/home/battlefront/.local/bin/python3.11` has no torch
and is sufficient for retained-reference reuse, candidate orchestration, and
publication. The available `/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python`
is Python 3.12 with ROCm torch 2.9.1 (requiring its library environment); it does not
satisfy the requested Python 3.11 fresh-reference prerequisite. This package does
not install dependencies or change a Python or shared-library environment.

`quality` runs six separate all-Q4, mixed-Q4/W8, and four-role Q4/FP8 campaigns,
each for dense and XAttention, G16 and G32, and the runner's default 8192/32768
token lengths. The mixed recipe uses the accuracy tier (0.02 mean-NLL gate);
all-Q4 and four-role use capacity-speed (log(1.05)). Every campaign uses freshly
executed candidate scores and the validated BF16 authority, fixed FP8-K/INT4-V
cache, prefill-only, spec-none execution. Failed campaigns retain their reports;
the stage tries all six and returns failure if any fails.

`publish` reopens each campaign through the existing Pareto quality validator,
binds both groups to the exact artifact receipt and selected chunk, and uses the
existing exclusive quality authority publisher. It writes
`quality-authorities-receipt-bound-n16k16.json` only after validation.

CPU regression checks:

```sh
/home/battlefront/.local/bin/python3.11 profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/test_campaign.py
```
