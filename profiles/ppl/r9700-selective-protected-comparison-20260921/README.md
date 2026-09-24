# Selective-protected AMD PPL comparison

User-requested bounded experiment, separate from the paused terminal speed matrix.
Start from the exact all-Q4 N16K16 artifact and copy unchanged payloads byte-for-byte.
Promote from the original Qwen3.8-27B BF16 source:

- W8G32: token embedding and full output head.
- BF16: attention query_key and gate_value at layers3,7,11,15,19,23; attention output3,7;
  GDN output4.
- Row-scaled E4M3 FP8: attention output11; attention query_key and gate_value27,31,51;
  MLP gate_up and down62,63.

These are28 physical objects, corresponding to the NVFP4 base protections plus eight additional
logical selective matrices. Retain Vision, MTP, optimized draft head and token map unchanged.
This is an AMD Q4 experiment, not a byte-identical NVFP4 conversion or a production selection.
Estimated artifact size17.678GB, excluding any DFlash companion and runtime allocations.

Fresh build: `build-r9700-selective-protected-g16-20260921`.
Artifact: `out/qwen3.8-27b-r9700-q4-selective-protected-n16k16-eval.ninfer`.
PPL: dense/G16, A8 Q4/W8, FP8 QK, C1, chunk2048, prefill schedule, skip-half, spec-none,
8192 and32768 prompt tokens (4095/16383 scored positions).

Retained numerical baseline owner:
`profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921`.
Reuse `bf16-chunk2048-a/results.json` plus `bf16-chunk2048-repeat.json` and existing
`all_q4_dense_quality/results.json` / `four_role_dense_quality/results.json` G16 cells.
Reusing these measurements requires review that incumbent numerical paths remain unchanged.
Compare complete aligned raw NLL and argmax sidecars; report PPL, paired NLL differences,
new severe positions and BF16 greedy flips. Flips across weight recipes are diagnostic, not
an exact-token gate. Evaluate both existing accuracy and capacity-speed criteria explicitly.
No broad benchmark campaign, XAttention work, DFlash conversion or decode sweep is authorized here.

Artifact created and completed-output validation passed: 17,678,295,040 bytes,
SHA-256 `42fede52c8cd8ddcb77db68f00bca285c097df9c0f0dca1cedd89f1dd9adbd9a`.
Its adjacent `.conversion.json` binds the base and the copied/re-encoded objects.
Actual-artifact C++ binder, registry and workspace host checks passed. Physical numerical
checks passed for FP8 N5120/K6144,17408 and BF16 N7168/K5120,N5120/K6144 at T1/2047/2048,
including eager and two poisoned graph replays. See `fp8-qualification.json` and
`bf16-qualification.log`. These are numerical checks, not speed claims.
The first whole-model launch rejected an FP8 attention-output binding before scoring:
the family caller passed the previously unused route-token argument instead of the actual
layer index. The leaf contract/caller now require that index, preserving exact weight identity.
The successful 8K PPL rerun exercises the corrected integration route; no failed-attempt
measurements were reused. Independent review confirmed unchanged incumbent arithmetic.

Reproduction (fresh destinations only; the documented artifact already exists):

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/ssdpool2nvme/local_llm/ninfer/out/numerical-reference-venv/lib/python3.11/site-packages \
/home/battlefront/.local/bin/python3.11 -m tools.convert.qwen3_8_27b_r9700.convert_selective_protected \
  --base out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --model /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --out out/qwen3.8-27b-r9700-q4-selective-protected-n16k16-eval.ninfer

cmake -S . -B build-r9700-selective-protected-g16-20260921 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_BUILD_R9700_CORE_QUALIFIER=ON \
  -DNINFER_R9700_KV_VALUE_GROUP=16 -DNINFER_R9700_Q4_ACTIVATION_BITS=8 \
  -DNINFER_R9700_W8_ACTIVATION_BITS=8 -DNINFER_R9700_FP8_QK_WMMA=1 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION=OFF \
  -DNINFER_R9700_XATTENTION_STRIDE=16 -DNINFER_R9700_XATTENTION_TAU_PERMILLE=1000
cmake --build build-r9700-selective-protected-g16-20260921 --target ninfer-ppl \
  ninfer_r9700_registry_qual ninfer_r9700_target_binding_qual \
  ninfer_r9700_fp8_execution_state_contract_test ninfer_r9700_fp8_gate_up_qual \
  ninfer_r9700_linear_op_qual -j 6
bash profiles/ppl/r9700-selective-protected-comparison-20260921/commands.sh ppl
```

## Results

Completed matched dense/G16 PPL, lower is better. Sizes are decimal GB, excluding runtime
allocations. Baseline/reference measurements were reused after raw-sidecar validation and
review of unchanged incumbent arithmetic; only the selective recipe was newly scored.

| Recipe | File GB | 8K PPL | 32K PPL | New severe positions 8K/32K |
|---|---:|---:|---:|---:|
| Independent BF16 reference | — | 6.463888 | 5.632511 | — |
| All-Q4 | 15.172833 | 6.723230 | 5.891989 | 7 / 36 |
| Selective-protected | 17.678295 | 6.641352 | 5.804673 | 9 / 26 |
| Four-role FP8/Q4 | 21.553549 | 6.614002 | 5.797376 | 5 / 28 |

The selective artifact is 17.98% smaller than four-role and 16.51% larger than all-Q4.
Its paired mean-NLL deltas versus BF16 are 0.02708459 / 0.03010804. All three recipes
pass capacity-speed quality at both lengths (mean delta <= ln(1.05), new-severe budgets
11/41). All three fail the stricter accuracy tier (mean delta <=0.02, severe budgets5/17).
The selective recipe improves average PPL over all-Q4 but has more new severe positions at8K;
it does not dominate every quality metric and does not pass strict accuracy. Four-role retains
lower average PPL at both lengths. No quality waiver, production selection, DFlash claim,
or speed comparison follows from this experiment.

`comparison.json` includes paired NLL statistics, per-tier decisions and BF16-greedy diagnostics.
`candidate/` retains native reports and aligned NLL/argmax sidecars. Successful whole-model
scoring closes the attention-output layer-index integration regression described above.
Status: requested model creation and PPL comparison complete; other campaigns remain paused.
