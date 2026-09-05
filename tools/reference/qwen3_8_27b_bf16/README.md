# Qwen3.8-27B checkpoint-direct BF16 scorer

This scorer is the independent numerical baseline for the R9700 PPL and exact-token gate. It
opens the original complete 18-shard BF16 safetensors checkpoint directly. It does not import or
consume a `.ninfer` artifact, packed weights, the product Engine, or a product kernel. The default
`bf16-reference` profile also has no quantized cache. Two separate cache-only diagnostic profiles,
`bf16-source-kv-g16` and `bf16-source-kv-g32`, keep every source BF16 weight and the same model
formula while replacing only the full-attention cache append/use representation.

The evaluator uses a layer-major causal topological order. It stages one source layer on the
accelerator, evaluates the complete represented token sequence through that layer, and releases
the weights before loading the next layer. This keeps the 51.75-GiB source checkpoint off the
32-GiB device while avoiding token-major CPU/GPU offload, which would reload the checkpoint for
every decode token. At 32K, the persistent hidden sequence is about 0.31 GiB; current-layer BF16
KV, FP32 GDN state, weights, and bounded activations remain separately owned.

The baseline mathematical profile is explicit: source weights and linear outputs are BF16;
residual, normalization, RoPE trigonometry, attention score/softmax/PV, GDN gates, and GDN state
updates use FP32 before their specified BF16 boundaries. Full attention retains BF16 K/V without
quantization.
The GDN recurrence starts from exact zero state. Qwen3.8's interleaved query/output-gate projection,
GDN q/k/v slices, depthwise convolution orientation, zero-centered GDN norm, and untied output head
are mapped directly from the Hugging Face source names.

The cache-only diagnostics change exactly one boundary in each of the sixteen full-attention
layers. After source-BF16 projection, normalization, and RoPE, represented K is appended as direct
OCP E4M3FN with round-to-nearest-even and finite saturation at +/-448. Represented projected V is
appended as symmetric signed INT4 codes in [-7,7], low lane first, with one
`FP16_RNE(max(abs(V))/7)` scale per contiguous G16 or G32 feature group. Zero groups and finite
groups whose scale underflows to FP16 zero use canonical positive-zero scales and all-zero codes.
At each attention use, the stored K and V prefix is decoded to FP32 and consumed by the unchanged
FP32 score, softmax, and PV formula. Query is not quantized. GDN state and every non-cache boundary
remain identical to `bf16-reference`.

The vectorized diagnostic codec is local PyTorch code, not a call to the product implementation.
Its CPU tests compare exact key bytes, value nibbles, and FP16 scale bits with the independent
dependency-free scalar authority in `kv_codec.py`, including ties, signed zero, finite saturation,
zero groups, scale underflow, scale overflow, and G16/G32 grouping.

Install the explicit dependencies from `requirements.txt` using a ROCm-enabled PyTorch build.
Multi-token GDN spans use FLA's fixed fused-recurrent implementation; single-token spans and the
independent short-tensor authority use the project's explicit FP32 recurrence. No
`accelerate`, Transformers auto-loader, CPU offload, or network access is used. Then run:

```bash
source /path/to/rocm-python-venv/bin/activate
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Activating the environment is required when `ppl.py` is passed to `tools/ppl/run.py`, because the
runner executes that scorer path directly and its environment-selected Python must provide the
three declared dependencies.

Every ordinary scorer process now has one mandatory deterministic execution profile. Before
importing PyTorch, FLA, or the backend, it sets `TORCH_BLAS_PREFER_HIPBLASLT=0`,
`ROCBLAS_DEFAULT_ATOMICS_MODE=0`, `FLA_USE_FAST_OPS=0`, disables PyTorch TunableOp, hipBLASLt,
launch/copy serialization, and allocator no-cache mode, and fixes the TF32 and Triton
fusion/interpreter/kernel-override controls. The FLA setting fixes its imported exponent
implementation for the fused recurrence. Triton architecture overrides, rocBLAS Tensile override
paths, and custom PyTorch allocator configurations must be absent. A caller-supplied conflicting
or forbidden value is an error, not an alternate route. It then enables strict, non-warn-only
PyTorch deterministic algorithms and verifies highest matmul precision, disabled TunableOp, and
resolved hipBLAS before constructing the checkpoint backend. The ordinary JSON binds this profile,
the complete canonical environment, deterministic/reduction settings, interpreter and package
records, scorer and FLA Python-tree hashes, and actual gfx1201 device identity as
`execution_provenance`. Profile `rocm-hipblas-no-atomics-strict-deterministic-pv-gdn-v3` binds
the full-attention PV implementation as explicit FP32 `torch.mm` over absolute cache rows in
ascending fixed 8,192-row chunks, including the final partial chunk, with ordered FP32 in-place
accumulation. It also binds the exact GDN dispatch: the project FP32 recurrence at T=1 and
`fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` at every T>1 span, with BF16 Q/K/V,
FP32 gates/state, BF16 output, and FP32 final state. A result without both exact structured
`attention_pv` and `gdn_recurrence` identities is not reusable BF16 authority. This rejects all
v2 results produced by the superseded multi-kernel FLA chunk route. It also records and validates
the effective Triton target plus AMD lowering controls for buffer operations/atomics, global/local
prefetch, async copy, block ping-pong, in-thread transpose, and packed-float scalarization. A prior
v3 report that lacks this structured boundary is not reusable.

Before admitting the fused-recurrent route at production prefill spans, run its focused numerical
probe once under the same ROCm environment:

```bash
/path/to/rocm-python-venv/bin/python \
  tools/reference/qwen3_8_27b_bf16/gdn_full_span_probe.py \
  --device 0 \
  --out-json profiles/ppl/bf16-gdn-full-span-probe.json
```

The probe executes the scorer-owned fused route at exactly T=4,095 and T=4,096 from deterministic
host-owned represented BF16 Q/K/V and FP32 decay, beta, and initial state. Its independent serial
FP64 oracle follows every recurrence row for five selected value-head/value-feature columns; each
prediction and output still sums all 128 key features. It checks five output rows per column and
all 128 elements of every selected final-state column, while avoiding the full 48x128x128 FP64
state trajectory. The diagnostic schema records complete input and result hashes, sampled-oracle
hashes/errors, the exact fused-route identity, source hashes, and the mandatory execution
provenance. `quality_evidence=false`: this validates the operator boundary but does not replace
fresh exact untraced scorer repeats.

```bash
/path/to/rocm-python-venv/bin/python tools/reference/qwen3_8_27b_bf16/ppl.py \
  --weights /path/to/complete/Qwen3.8-27B-BF16 \
  --ids tools/ppl/corpus.ids \
  --scheme bf16-reference \
  --schedule prefill \
  --skip half \
  --tokens 8192 \
  --prefill-chunk 4096 \
  --device 0 \
  --out-json profiles/ppl/bf16-reference-rocm10/prefill-8192.json

/path/to/rocm-python-venv/bin/python tools/reference/qwen3_8_27b_bf16/ppl.py \
  --weights /path/to/complete/Qwen3.8-27B-BF16 \
  --ids tools/ppl/corpus.ids \
  --scheme bf16-reference \
  --schedule prefill \
  --skip half \
  --tokens 32768 \
  --prefill-chunk 4096 \
  --device 0 \
  --out-json profiles/ppl/bf16-reference-rocm10/prefill-32768.json
```

For selection authority, run two fresh reference-only schema-v6 campaigns so each independently
covers both 8K and 32K without `--trace-json`, then validate their raw reports, provenance, and
sidecars together:

```bash
python3 tools/ppl/compare_bf16_repeats.py \
  --first profiles/ppl/bf16-reference-deterministic-a/results.json \
  --second profiles/ppl/bf16-reference-deterministic-b/results.json \
  --out profiles/ppl/bf16-reference-deterministic-repeat-comparison.json
```

The comparator accepts only reference-only prefill campaigns, reopens each raw result and hashed
sidecar through the campaign validator, requires identical source/scorer/environment/workload
identity, and reports exactness separately for 8K and 32K. Its comparison report is determinism
evidence; either validated input campaign supplies the reusable BF16 quality cells. Pass that
report as `--bf16-repeat-comparison` alongside every `--reuse-bf16-campaign`; candidate campaigns
and terminal Pareto assembly reject a reusable realization that is not one of the exact pair.

### Opt-in stage trace for nondeterminism localization

The scorer can separately hash bounded first-64-row and last-64-row snapshots after the
embedding, every attention/GDN mixer, every MLP, final normalization, and the first and last
logit slices. Because the first reproduced divergence was the last-row window after layer 3,
schema v8 additionally retains only that layer's last 64 rows after normalized input, Q/gate, K,
and V projection, Q/K RoPE, attention, gated flatten, output projection, and residual update. Run
the same command twice as fresh processes, using distinct output paths:

```bash
/path/to/rocm-python-venv/bin/python tools/reference/qwen3_8_27b_bf16/ppl.py \
  --weights /path/to/complete/Qwen3.8-27B-BF16 \
  --ids tools/ppl/corpus.ids \
  --scheme bf16-reference \
  --schedule prefill \
  --skip half \
  --tokens 4097 \
  --prefill-chunk 4096 \
  --device 0 \
  --out-json profiles/ppl/bf16-stage-trace-4097-a.json \
  --trace-json profiles/ppl/bf16-stage-trace-4097-a.trace.json

/path/to/rocm-python-venv/bin/python tools/reference/qwen3_8_27b_bf16/ppl.py \
  --weights /path/to/complete/Qwen3.8-27B-BF16 \
  --ids tools/ppl/corpus.ids \
  --scheme bf16-reference \
  --schedule prefill \
  --skip half \
  --tokens 4097 \
  --prefill-chunk 4096 \
  --device 0 \
  --out-json profiles/ppl/bf16-stage-trace-4097-b.json \
  --trace-json profiles/ppl/bf16-stage-trace-4097-b.trace.json

python3 tools/reference/qwen3_8_27b_bf16/compare_stage_traces.py \
  --first profiles/ppl/bf16-stage-trace-4097-a.trace.json \
  --second profiles/ppl/bf16-stage-trace-4097-b.trace.json \
  --out profiles/ppl/bf16-stage-trace-4097-comparison.json
```

The 4,097-token starting point gives the scorer exactly 4,096 hidden rows and exercises one full
configured chunk. If the comparison is exact while the retained campaign still demonstrates
nondeterminism, repeat both fresh-process runs and the comparison with `--tokens 8192` and new
`8192-a`/`8192-b` paths. That escalation exercises the next scorer span and carried GDN state. Do
not start at 8,192: the smaller run is the first-divergence authority when it already reproduces
the issue.

An embedding mismatch points before the layer stack. Otherwise the first mismatching
`layer-NN.attention` or `layer-NN.gdn` snapshot identifies the mixer boundary; a first mismatch at
`layer-NN.mlp` identifies the post-mixer path; matching final-normalization snapshots followed by
a logits mismatch identifies the output head. Matching checkpoint and logits hashes with differing
score hashes localizes the issue to final score reduction. The first/last row pair distinguishes
an immediately global difference from one accumulated through the causal suffix. Within layer 3,
the ordered `layer-03.attention-detail.*.last` checkpoints identify the first differing semantic
operation without synchronizing or copying intermediates during scoring.
For each of layer 3's four KV heads, schema v8 further retains masked FP32 QK scores, FP32 softmax
probabilities, and FP32 PV results for those final query rows. The following aggregate
`attended-output` checkpoint is the corresponding BF16 PV boundary. Causal negative infinity is
expected only in the masked-QK checkpoints; NaN and positive infinity are always rejected.
The paired provenance also binds PyTorch's deterministic-algorithm state, the effective
`ROCBLAS_DEFAULT_ATOMICS_MODE` and `TORCH_BLAS_PREFER_HIPBLASLT` environment settings, and the
resolved result of `torch.backends.cuda.preferred_blas_library()`. PyTorch exposes CUDA-named enum
members on ROCm, so the trace records their canonical ROCm meanings as `default`, `hipblas`,
`hipblaslt`, or `ck`. Schema v8 also binds the deterministic execution-profile name, the exact
structured attention-PV and GDN recurrence identities, disabled TunableOp, highest matmul precision,
the canonical/forbidden process environment, actual gfx1201 identity, and resolved Triton/AMD
code-generation controls.

Before changing the scorer's long-prefix PV implementation, isolate that operation with
`pv_determinism_probe.py`. Each invocation is one fresh process. It constructs the same FP32
`[32,6,S] @ [S,256]` geometry from closed-form host operands for
`S=8192,8193,16384,32768`, runs both the current one-shot einsum and the proposed absolute
ascending 8192-row `torch.mm` chunks, and records input/output hashes, timings, implementation
provenance, and an independent serial FP64 check of sampled outputs. Run it twice, then compare
the process-isolated reports without launching another GPU workload:

```bash
/path/to/rocm-python-venv/bin/python \
  tools/reference/qwen3_8_27b_bf16/pv_determinism_probe.py \
  --device 0 --out-json profiles/ppl/bf16-pv-probe-a.json
/path/to/rocm-python-venv/bin/python \
  tools/reference/qwen3_8_27b_bf16/pv_determinism_probe.py \
  --device 0 --out-json profiles/ppl/bf16-pv-probe-b.json
python3 tools/reference/qwen3_8_27b_bf16/pv_determinism_probe.py \
  --compare profiles/ppl/bf16-pv-probe-a.json profiles/ppl/bf16-pv-probe-b.json \
  --out-json profiles/ppl/bf16-pv-probe-comparison.json
```

The comparison reports repeatability separately for each route and source extent. It is
diagnostic evidence only: the scorer remains unchanged until the fixed-chunk route is exact in
fresh processes and passes the sampled FP64 oracle at all four extents.

Stage traces and comparison reports declare `quality_evidence=false`. They are diagnostic-only:
do not use them as a BF16 quality authority, a reusable PPL campaign, or a selection input. The
ordinary `--out-json` result remains separate. Omitting `--trace-json` performs no checkpoint
capture, while retaining the mandatory deterministic execution and provenance contract above.

Run the paired 8K cache-only diagnostics from the same environment and source checkpoint:

```bash
tools/reference/qwen3_8_27b_bf16/ppl.py \
  --weights /path/to/complete/Qwen3.8-27B-BF16 \
  --ids tools/ppl/corpus.ids \
  --scheme bf16-source-kv-g16 \
  --schedule prefill \
  --skip half \
  --tokens 8192 \
  --prefill-chunk 4096 \
  --device 0 \
  --no-device-graph \
  --out-json profiles/ppl/bf16-cache-only-rocm10/prefill-8192-g16.json

tools/reference/qwen3_8_27b_bf16/ppl.py \
  --weights /path/to/complete/Qwen3.8-27B-BF16 \
  --ids tools/ppl/corpus.ids \
  --scheme bf16-source-kv-g32 \
  --schedule prefill \
  --skip half \
  --tokens 8192 \
  --prefill-chunk 4096 \
  --device 0 \
  --no-device-graph \
  --out-json profiles/ppl/bf16-cache-only-rocm10/prefill-8192-g32.json
```

These are attribution diagnostics, not alternate product profiles and not replacements for the
BF16 quality authority. Comparing each diagnostic directly with `bf16-reference` isolates cache
representation error under unchanged source BF16 weights; comparing G16 with G32 is supplementary.

Both schedule labels process the source oracle in identical `--prefill-chunk` spans. `decode`
changes only the reported/scored start to `max(skip,1)`; the paired product executable is what
exercises T=1 kernels. This avoids reloading or relaunching the BF16 oracle once per suffix token
while preserving the identical causal target distribution. Both schedules emit one little-endian
float32 NLL and one little-endian signed-int32 greedy token for every scored position. NLL and
argmax use exactly vocabulary rows `[0,248077)`.

`--spec mtp --draft-tokens 1..5` is accepted only as a comparison label because MTP cannot change
the target model's teacher-forced distribution; JSON records `speculative_execution=false`.
Device Graph execution is not part of this authority: `--no-device-graph` is accepted and
`--device-graph` is rejected. The JSON descriptor is published only after both sidecars, and an
existing result triplet is removed before checkpoint preflight. Every descriptor records the
unchanged `bf16-source` weights identity, source hashes, and complete deterministic execution
provenance. It also records `weight_format`,
`formula_profile`, `cache_only_diagnostic`, `cache_diagnostic_scope`, `kv_format`,
`kv_value_group`, the exact key/value codec labels, and the append/use boundaries, so a cache-only
trace cannot be mistaken for either the BF16 authority or a product result.

The scorer checks the exact root/text configuration, the 1,199-entry/55,562,855,904-byte source
index, all 851 required text tensor names and BF16 shapes, the exact 18-file shard set, source
config/index hashes, every indexed shard's SHA-256, and the represented token-ID hash. Shard hashing
reads the complete checkpoint before each fresh score; this is intentional source provenance and
does not initialize or allocate the accelerator.
The maintainer-provided source must contain all 18 shards. Partial checkouts fail during source
preflight before importing PyTorch or touching the device.
The current CPU-only dependency, tensor-metadata, and complete shard-hash preflight is retained at
`profiles/ppl/bf16-source-rocm10-preflight.json`.
