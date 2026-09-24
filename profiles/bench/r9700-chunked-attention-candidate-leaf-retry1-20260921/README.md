# Public-leaf fixture retry

Stage entry waits at most ten seconds for an actual zero-busy, low-residency sample;
transient activity samples do not trigger repeated artifact preflight. Sustained activity,
occupied VRAM, or non-auto power still blocks launch. Owned-child VRAM teardown also has a
bounded ten-second wait before subsequent model loads.

The original bounded-panel package retains passing static, host-routing and full
G16/G32 raw FP64 qualification. Its public-leaf fixture incorrectly supplied zero
workspace for appended dense attention and failed with hipInvalidValue. Only that
fixture source and public-leaf executable are replaced for this retry. The original
failed qualification namespace remains untouched.

```sh
bash profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/commands.sh preflight
bash profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/commands.sh qualification
bash profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/commands.sh whole
```

Preflight is read-only. It imports the original experiment implementation and
requires its original input identities to match except `public_leaf` and
`sources[tools/r9700/full_attention_leaf_qual.hip]`. It verifies all24 retained raw
PASS cases and the final PASS, host-routing PASS and all five static stage records,
then binds their evidence files by hash. Unchanged production source, benchmark,
raw qualifier, ISA, artifacts and build settings are required.

Qualification launches only the corrected no-argument
`build-r9700-chunked-attention-candidate-g16-20260921/src/ninfer_r9700_full_attention_qual`.
It uses the original PCI/auto/idle policy under the root's exclusive GPU lease.
It does not rerun passed raw, host or static checks. Successful publication records
the retained evidence alongside the new public-leaf pass in a create-only
`qualification/result.json` here.

Whole measurement reuses the original implementation, baseline and candidate
benchmarks unchanged. It runs candidate P8192/chunk1024 and fresh matched
baseline/candidate P2048/chunk4096, each with three repetitions and one warmup.
The admission thresholds remain median duration ratios at most0.90 and1.02,
respectively. The fresh admission path is this package's `whole/result.json`,
with the existing `ninfer_chunked_attention_whole_prefill_comparison` schema1 and
`status=admitted` or `status=not_admitted`. Its qualification identity binds both
the retained passes and corrected leaf result. Successor campaigns must reference
this retry authority, not the original failed package.

No GPU execution occurs during package preparation or CPU tests:

```sh
/home/battlefront/.local/bin/python3.11 profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/test_retry.py
```

The first whole stage completed candidate P8192 (1184.242683 mean tok/s;
6.870248486, 6.901032133, 6.982186078 seconds), then stopped before baseline P2048
because its exited child's VRAM was still being reclaimed. The retry wrapper now
allows up to10 seconds, polling every0.1 seconds, for owned-child residency to
fall below1 GiB while retaining the auto power requirement. Initial external idle
checks remain strict. The original experiment implementation stays unchanged.

Complete only the two missing P2048 runs with:

```sh
bash profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/commands.sh finish-whole
```

This stage validates the retained whole inputs, qualification and completed P8192
report, creates a new `whole-completion/` for baseline/candidate P2048 measurements,
and combines them with the retained P8192 measurements using the original admission
function. It creates the previously absent `whole/result.json`, including the
continuation reason and evidence identities. It neither overwrites previous files
nor reruns P8192, and refuses an existing completion directory or final result.
