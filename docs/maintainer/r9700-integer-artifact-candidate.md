# R9700 integer artifact candidate

This is the conversion contract for one provisional Qwen3.8-27B R9700
integer candidate. It makes direct-BF16 source validation and deterministic
weight packing concrete while the final R9700 weight recipe remains unselected.
It is accepted by the in-tree sole-target runtime so the real PPL/token/performance
gates can execute, but it is not a selected product artifact contract and does
not establish model accuracy or performance.

## Candidate identity and status

```text
model_id   = qwen3.8-27b
weights_id = r9700-int-candidate
target_key = qwen3_8_27b_r9700
recipe_id  = r9700-w8g32-candidate-v0
status     = provisional evaluation identity
```

The Engine accepts this one identity only to run the required independent
BF16-reference Op/model parity, FP8-K/INT4-V quality, capacity, and
whole-inference performance gates. It must not be renamed or advertised as the
selected product recipe until those gates pass. After the artifact is atomically closed, its
conversion report records the exact output byte count and SHA-256 together with
`weight_recipe_selected: false`; that field is an intentional guard against
treating a successful byte conversion as a selected model profile.

## Exact inventory

The candidate has the exact Qwen3.8-27B object sequence: six frontend
resources followed by 1,118 tensors, in the existing Qwen3.8 source order.
Every logical tensor name and shape is identical to that source inventory.
Its only storage decision is explicit and mechanical:

| Persistent roles | Stored format | Count |
|---|---:|---:|
| Existing direct BF16 roles | BF16 | 582 |
| Existing direct FP32 roles | FP32 | 96 |
| Existing direct I32 roles | I32 | 1 |
| Every role formerly Q4, Q5, Q6, or W8 | W8G32_F16S | 439 |

The 439 W8 roles are a bring-up candidate, not a claim that W8/G32 is the
eventual R9700 matrix recipe. No prior artifact payload is accepted by this path.

The separately qualified Q4G64 challenger is not encoded by this provisional artifact. Its
complete raw Linear boundary uses direct row-split signed Q4 weights, caller-owned signed A4G64
activation storage, FP16 scales, and native gfx1201 signed-INT4 WMMA. At the real
`[7168,5120]` projection shape it passes exact activation-code and complete-output FP64 formula
checks for decode T=1..8 and prefill T=16/32/64/128, and is materially faster than this W8 route
on the isolated synthetic shape gate. That result admits the separately identified Q4 evaluators
to real-source quality and whole-inference comparison; it does not alter this provisional
W8 artifact's identity, inventory, converter, or unselected status.

A8 is the production-style compile default for every Q4-bearing identity and consumes those same
Q4G64 artifact planes without changing their identity. The retained A4 route requires an explicit
evaluator build. A8 stores each signed A8G64 code as unsigned low and signed high nibbles,
executes both with native IU4 WMMA, and recombines `low + 16 * high` exactly in I32 before FP32
scale composition and one BF16 output rounding. Exact activation images and every independent
FP64-formula output pass at `[7168,5120]`, T=1..8/16/32/64/128, with rotating coverage of all 64
packed K-lane positions. Both A4 and A8 carry quantizer status into the ordered WMMA consumer;
invalid input produces exact BF16 NaN outputs asynchronously rather than silently using zero
codes. Direct separately compiled Tensor/WorkspaceArena qualifiers protect selection, exact
caller-owned workspace allocation, and rewind for both activation widths. Its represented-source
maximum error is 0.015625 versus A4's 0.212891, at an 8.7--24.3 percent complete-Op latency cost
across the tested shapes. This admits A8 to matched real-model evaluation while keeping A4
reproducible; it does not select a persistent Q4 recipe for production.

## Direct BF16 source and byte contract

The converter consumes one complete Qwen3.8-27B BF16 checkout: the pinned
configuration, all source tensors named by the established Qwen3.8 recipe,
and all six frontend resource files. It opens every required safetensors shard
during preflight before it creates an output. A partial checkout is rejected.

Candidate W8 matrices use the registered `row-split-k128-v1` storage layout:

```text
K physical extent = round_up(K, 128)
group             = 32 represented BF16 source values
scale             = FP16_RNE(max(abs(group))/127)
code              = clamp[-127,127](RNE_even(value / represented scale))
zero group        = zero scale and zero codes
nonzero FP16 underflow = smallest positive FP16 subnormal scale
```

Codes occupy the base plane as signed int8 bytes in row-major K order. The
FP16 scale plane begins at the next 256-byte boundary and is row-major by
`[row][K/32 group]`; K padding is zero before quantization. NaN, infinity, or
a finite source group whose scale cannot be represented as a finite positive
FP16 word rejects conversion.

The standard-library scalar reference at
`tools/convert/qwen3_8_27b_r9700/codec.py` is independent of Torch and defines
this packing contract for synthetic-byte testing. The actual converter uses the
existing vectorized row-split encoder only after source BF16 materialization.

## Invocation and non-claims

After a complete source checkout and the project Python environment are
available, invoke the target-specific converter with an explicit Torch device:

```text
python3 -m tools.convert.qwen3_8_27b_r9700.build_draft_ranking --corpus tools/ppl/corpus.ids --out out/qwen3_8_27b_draft_ranking.i64
python3 -m tools.convert.qwen3_8_27b_r9700.convert --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <new-artifact.ninfer> --device <cpu-or-rocm-torch-device>
```

The builder derives exactly one little-endian I64 total-frequency row from explicitly named,
manifest-validated Qwen3.8 token corpora and writes a JSON provenance sidecar. Converter preflight
requires that sibling sidecar, revalidates every named corpus/manifest, and re-derives the exact
row before creating artifact output. The builder accepts the current unique PPL corpus but rejects
the tiled benchmark corpus, whose repeated throughput padding
would bias frequency counts. It validates decimal IDs, the `0..248076` tokenizer domain, declared
token count, and SHA-256. Accepted manifests are exactly the schema-2 `ninfer_ppl_corpus` contract
for `qwen3.8-27b`, with both special-token insertion and chat templating explicitly disabled;
an optional tokenizer model identity must also name Qwen3.8. Special-ID force-inclusion remains
the converter's responsibility. The
repository supplies no retired-model default. The converter refuses to overwrite an existing
output. It does not accept a quantized shell, does not runtime-repack weights, and does not append
DFlash2. The future selected converter must either replace this candidate identity and recipe
atomically or delete this candidate path after its evidence is captured.

The separately compiled W8 activation evaluator directly binds these unchanged W8G32 planes and
selects between represented-BF16 and dynamic signed-A8G32 execution at measured per-shape
crossovers. A8 owns caller-provided code/FP16-scale/status workspace and uses native signed-INT8
WMMA. The mixed inventory uses A8 from T3 for `[7168,5120]`, `[12288,5120]`, `[14336,5120]`, and
`[5120,17408]`; from T4 for `[4608,4608]`, `[5120,4608]`, `[248320,5120]`, `[5120,6144]`, and
`[5120,10240]`; from T32 for `[34816,5120]`; and conservatively from T64 for all three 1152-row
Vision shapes. Unknown all-W8-only shapes stay on exact execution. This adaptive BF16/A8 profile
does not change artifact bytes or identity. A matched 8K source-MSE mixed-artifact comparison
reduced scorer time from 264.179 to 133.504 seconds and improved PPL from 6.544746 to 6.538677;
against BF16 it has +0.012077 mean NLL and three new severe positions. It is therefore the selected
W8 execution profile, while a separately compiled A16 build remains the control.

The FP8-K/INT4-V cache is runtime state, so its G16/G32 group and plane orders are not encoded in
this weight artifact's identity or object metadata. Paired cache comparisons use separately
compiled runtime profiles and may hold this exact candidate artifact constant.

## DFlash2 Q4 evaluation companions

Three explicit evaluation identities append the source DFlash2 checkpoint to the corresponding A8
base artifact without changing its base matrix formats:

- `qwen3.8-27b/r9700-q4g64-n16k16-dflash2-q4-eval` extends the all-Q4G64 base;
- `qwen3.8-27b/r9700-q4-w8-mse-n16k16-dflash2-q4-eval` extends the source-MSE mixed Q4G64/W8G32 base.
- `qwen3.8-27b/r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval` extends the
  authority-bound four-role rowwise-FP8/all-other-Q4G64 base without changing its 144 FP8 roles.

The source checkpoint has exactly 81 BF16 tensors. Conversion fuses query/key/value and gate/up in
the same semantic order consumed by the runtime, producing 66 DFlash objects: 32 persistent
Q4G64-FP16-scale matrices and 34 direct BF16 tensors. The two selector codebooks remain the
model-specified BF16 `[248320,256]` tables. Norms, convolution base kernels, and other non-matrix
state also remain BF16. The Q4 matrices use the compile-selected adaptive A8G64 route; this is an
activation intermediate and does not create a second persistent artifact format. No DFlash matrix
is assigned W8 speculatively. A W8 fallback is admissible only after matched draft quality or
whole-inference attribution identifies a sensitive semantic family.

The DFlash addition is 1,209,469,440 tensor bytes: 954,654,720 Q4 bytes plus 254,814,720 BF16
bytes. It is 56,156,288 bytes smaller than the inspected retired NVFP4 companion segment, which
used the same 32-matrix/34-BF16-object split and occupied 1,265,625,728 bytes inside its
19,589,713,920-byte artifact. The projected bound tensor arenas are 16,369,271,200 bytes for the
all-Q4 companion and
24,077,646,752 bytes for the mixed companion; including exact binder alignment they are
16,369,285,120 and 24,077,660,672 bytes. The hybrid companion projects to a 22,750,001,152-byte
device arena. Projected artifact file sizes are 16,382,310,912, 24,090,686,464, and
22,763,026,944 bytes respectively. The largest DFlash Q4 activation is the `[5120,25600]` feature projection,
so both profiles reserve caller-owned A8G64 codes, FP16 scales, and status for K=25600. The binder
rejects any DFlash identity when its complete companion inventory is absent. It binds packed
weights and BF16 state directly and performs no runtime repack or hidden allocation. The grouped
dynamic-convolution projection and both chain/tree selector projections admit Q4 explicitly, size
their nested Linear activation image inside their public workspace contract, and pass the same
caller-owned arena through to Linear. Family DFlash planning derives those child qtypes from the
resolved evaluation identity rather than assuming W8.

Dependency-light preflight validates the exact source/config identity and both base inventories
before any output is opened:

```text
python3 -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 --base out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 --preflight-only
python3 -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 --base out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 --preflight-only
python3 -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 --base out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 --preflight-only
```

Remove `--preflight-only`, add an explicit `--out` and Torch `--device`, and run the conversion.
The converter copies the base payload byte-for-byte and
encodes DFlash source BF16 directly into its final persistent representation. The source
`model.safetensors` SHA-256 is
`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`; its config and README
hashes are retained in conversion provenance. A completed conversion also hashes the exact base
artifact before copying and the completed output artifact before writing its sibling report, so
both large-file identities are preserved with the source provenance. The retained conversions
completed on 2026-09-03: the all-Q4 artifact is 16,382,310,912 bytes with SHA-256
`ba39608b8a70e78e038e29154983c45017437803d9a77ea308a288c7908f9cdc`, and the mixed artifact is
24,090,686,464 bytes with SHA-256
`c2dcb265beae6a08052bb83b4e0fc1ef9e19eb743e0c1d58675477129379a43c`. Their sibling
`.conversion.json` reports bind those outputs to the exact base and source identities. These are
evaluation identities: draft-token quality, acceptance, resolved capacity, and whole-inference
speed remain open selection gates.

The two existing complete conversions are below. The hybrid companion remains intentionally
unmaterialized until schema-v7 selects its base branch; it must then be converted from those exact
hybrid bytes and bind their conversion receipt before any DFlash campaign can begin.

```text
python3 -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 --base out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 --out out/qwen3.8-27b-r9700-q4g64-n16k16-dflash2-q4-eval.ninfer --device cuda
python3 -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 --base out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 --out out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-dflash2-q4-eval.ninfer --device cuda
```

Before artifact publication the converter atomically writes a sibling
`.conversion.pending.json` receipt containing the exact base hash, source hashes, output identity,
and converter environment. After the artifact is published it validates the complete identity,
object plan, byte size, and SHA-256, updates that receipt, and atomically publishes the final
`.conversion.json` report. If the process stops after the artifact is complete but before the
report is published, resume only report finalization instead of reconverting:

```text
python3 -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 --base BASE.ninfer --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 --out COMPLETED.ninfer --finalize-report
```

When no pending receipt survives, pass the independently retained output digest with
`--expected-output-sha256`. Finalization rejects a changed base/source, output identity, object
plan, size, or digest before recreating the atomic report.

## Evaluation candidates

Eight additional evaluation-only identities preserve the exact same 1,124 object names, ordering,
shapes, source/frontend validation, and draft-ranking provenance. They are registered in the C++
target only for real-model evaluation; none selects the production recipe:

| converter | weights identity | quantized formats | tensor bytes | device arena bytes | projected artifact bytes |
|---|---|---|---:|---:|---:|
| `convert_q4.py` | `r9700-q4g64-n16k16-eval` | 439 Q4G64 | 15,159,801,760 | 15,159,815,680 | 15,172,829,184 |
| `convert_q4_w8.py` | `r9700-q4-w8-n16k16-eval` | 183 Q4G64, 256 W8G32 | 22,868,177,312 | 22,868,191,232 | 22,881,204,736 |
| `convert_q4_w8_mse.py` | `r9700-q4-w8-mse-n16k16-eval` | 183 source-MSE Q4G64, 256 source-MSE W8G32 | 22,868,177,312 | 22,868,191,232 | 22,881,204,736 |
| `convert.py` | `r9700-int-candidate` | 439 W8G32 | 30,260,413,792 | 30,260,425,984 | 30,273,439,488 |
| `convert_w8_mse.py` | `r9700-w8g32-mse-eval` | 439 source-MSE-refined W8G32 | 30,260,413,792 | 30,260,425,984 | 30,273,439,488 |
| `convert_w8_bf16_embedding.py` | `r9700-w8-bf16-embed-eval` | 438 W8G32, BF16 token embedding | 31,452,349,792 | 31,452,361,984 | 31,465,375,488 |
| `convert_w8_bf16_attention_qk.py` | `r9700-w8-bf16-attn-qk-eval` | 423 W8G32, 16 BF16 full-attention query/key | 30,810,916,192 | 30,810,928,384 | 30,823,941,888 |
| `convert_w8_bf16_attention_vo.py` | `r9700-w8-bf16-attn-vo-eval` | 407 W8G32, 32 BF16 full-attention value/output | 31,282,775,392 | 31,282,787,584 | 31,295,801,088 |
| `convert_w8_bf16_gdn_qk.py` | `r9700-w8-bf16-gdn-qk-eval` | 391 W8G32, 48 BF16 GDN query/key | 31,204,132,192 | 31,204,144,384 | 31,217,157,888 |

The first is the native Q4 capacity floor. The second keeps only roles originally assigned Q4 at
Q4G64 and promotes every source-Q5/Q6/W8 role to W8G32, providing an accuracy/capacity compromise
without making Q5/Q6 primary formats. Q5G64 and Q6G64 remain fallback measurements only if native
Q4G64 and W8G32 cannot meet quality and capacity together. Projected artifact sizes include the
current hash-qualified frontend resources, object alignment, and directory; they are byte-plan
facts, not speed, PPL, quality-admission, or selection claims.

The same-format mixed MSE control changes neither role assignment nor physical layout. For every
represented-BF16 Q4G64 or W8G32 group, the canonical absmax scale remains the first candidate and
eight deterministic alternating signed-code/least-squares-scale steps add candidates rounded to
FP16 before code selection. The earliest minimum decoded-weight SSE wins. Quantization consumes
only the original source weights; activation calibration, draft-ranking frequencies or corpus
tokens, PPL/argmax sidecars, and GPU results are excluded from the scale objective. The ordinary
artifact-wide draft-head ranking preflight remains required. Its stable identity and conversion
JSON preserve the exact source/frontend/ranking provenance and byte plan for later paired PPL
reporting.

The completed matched 8K G16 comparison rejects all-Q4+A4 at +0.135526 mean NLL and 800
BF16-greedy flips and mixed Q4/W8 at +0.046148 and 420 flips. All-W8 is much closer at
+0.002053 and 90 diagnostic flips and meets the current 8K quality guardrails. Restoring only the BF16 output head
regresses to +0.002425 and 92 flips; that measured artifact and its sidecars remain evidence, but
its converter, binder profile, and accepted identity are removed. The active isolated BF16
evaluators remain quality-eligible while retaining diagnostic flips: token embedding is +0.001624 / 80, full-attention
value/output +0.001098 / 84, GDN query/key +0.001375 / 98, and full-attention query/key
+0.001996 / 81. Mixed Q4/W8+A8 is also quality-eligible at +0.013815 mean NLL and two new
severe positions. The source-MSE mixed artifact improves to +0.012077 and three with adaptive-A8
W8 execution; its represented-BF16 W8 control remains retained at +0.013005 and three. Both meet
the accuracy tier. All-Q4+A8 meets the capacity-speed tier at +0.039509 and nine new severe
positions. Both A8 model profiles remain supported optimized comparison profiles. The complete comparison is recorded in
`profiles/ppl/8k-candidate-comparison.md`; none of these results alone selects a recipe.

Quality eligibility requires complete finite aligned sidecars and an explicit tier. The accuracy
tier uses paired mean-NLL delta at most 0.02 and at most
`max(4, ceil(0.001 * scored positions))` new NLL-at-least-10 positions. The capacity-speed tier
uses at most `ln(1.05) = 0.048790164` mean-NLL delta and
`ceil(0.0025 * scored positions)` new severe positions. BF16 greedy-token flip count and rate
remain diagnostics. Selection then retains every non-dominated
profile after matched 8K/32K and C=1..4 measurement: A dominates B only when it is no worse in
mean-NLL delta, new-severe-position rate, resolved capacity, and every required whole-inference
throughput cell, and is strictly better in at least one. PPL scorer wall time is never a throughput
objective.

Non-dominance does not leave either the recipe or production cache layout selectable. Quality is
an admission constraint. The classifier first retains one cache/execution winner for each recipe,
then selects one terminal production winner across those recipe winners. Both stages maximize the
minimum globally normalized whole-inference speed ratio, then the minimum normalized resolved
capacity, then minimize the worst declared-tier quality-budget fraction. Canonical artifact and
static-profile identity resolves only a complete measured tie. Selection uses retained per-cell
means exactly; raw repetition spread is retained as evidence but does not create a tolerance or
alternate rank. The schema-v7 record retains the complete frontier and per-recipe winners, then
emits `terminal_production_selection` with rule
`global_maximin_whole_then_capacity_then_quality_then_canonical_v1`, normalized ratios, one
terminal artifact/cache/execution winner, and decisive stage.
This rule weights no
workload arbitrarily and follows the product priority of end-to-end performance after supported
quality and capacity constraints are satisfied.

The earlier layout-bearing all-Q4 capacity evidence is exact historical measurement but does not
choose the cache group and is not current C=1..4 product evidence. The schema-v12 G16/G32 manifests
and schema-v19 reports, including C=1..8 maxima,
binding constraints, artifact hash, executable hashes, and superseded-run boundary, are recorded
in `paged-kv-cache.md` and `performance.md`. Matched phase and whole-inference evidence remains the
missing input to the schema-v7 selection record. The restored mixed recipe likewise requires fresh
C=1..4 capacity/whole evidence for dense and XAttention G16/G32 candidates, plus matched sparse
quality, so the final comparison covers both weight recipes rather than only all-Q4.

The same-size `r9700-w8g32-mse-eval` candidate changes only the source-to-W8 scale objective.
For each represented-BF16 G32 group, it includes the canonical FP16(absmax/127) baseline and
follows eight deterministic alternating least-squares steps: RNE/clamped codes are formed from
the stored FP16 scale, then the decoded-weight SSE-optimal scale for those fixed codes is
canonicalized back to FP16. The earliest minimum-SSE candidate is selected, so the objective
cannot regress a group relative to the baseline. It consumes no calibration text, PPL labels,
argmax sidecars, or GPU results. Complete representative source tensors improve decoded-weight
SSE by 4.94--5.33% for Text families and 2.65% for Vision QKV; deterministic 4,096-row vocabulary
samples improve token embedding by 5.33% and output head by 5.16%, with about 2.3% of Text codes
changing. Format, byte size, binder profile, and runtime W8 dispatch are unchanged. This evidence
admits the candidate to real quality and Pareto evaluation; it does not select it.

The source inventory provides a non-arbitrary precision ranking. Its only Q6 roles are token
embedding and output head; attention gate/value, attention/GDN output, GDN value/z, and MLP down
are Q5; query/key and MLP gate/up are Q4. Once the Q6 output head failed, the remaining complete
promotion choices within 1,191,936,000 bytes are:

| coherent role set | source tier | W8-to-BF16 increase |
|---|---|---:|
| token embedding | Q6 | 1,191,936,000 |
| all full-attention gate/value plus output projections | Q5 | 1,022,361,600 |
| all GDN query/key projections | Q4 | 943,718,400 |
| all full-attention query/key projections | Q4 | 550,502,400 |

Partial GDN-Q5 or MLP-down promotion would require choosing layers without layer-sensitivity
evidence. The BF16 token embedding is therefore the strongest next isolated candidate: it is the
remaining highest source tier and its represented values enter every input token before all 64
residual layers. Every other matrix, including the rejected output head, remains on the qualified
W8 route.

The full-attention Q5 value/output family is registered as a coherent isolated evaluator. Across
all 16 full-attention layers it restores both the fused
gate/value projection and attention output projection, increasing the all-W8 payload by exactly
1,022,361,600 bytes. It does not select individual layers and does not change query/key, GDN, MLP,
vocabulary, MTP, or Vision matrices.

Another coherent fallback restores GDN query/key in every one of the 48 GDN layers. It is the
complete source-Q4 GDN query/key family attributed above, not a selected subset of layers, and
adds exactly 943,718,400 bytes over all-W8. GDN value/z and output, full attention, MLP,
vocabulary, MTP, and Vision remain W8.

The full-attention query/key evaluator restores that fused projection in all 16 full-attention
layers. This is the complete semantic family rather than a post-hoc split of the value/output
evaluator, adds 550,502,400 bytes over all-W8, and leaves every gate/value, attention output, GDN,
MLP, vocabulary, MTP, and Vision matrix W8.

One BF16 vocabulary matrix occupies 2,542,796,800 bytes rather than
1,350,860,800 bytes in W8G32, so restoring both vocabulary matrices would make the device arena
32,644,297,984 bytes. On a 32 GiB device that leaves 1,715,440,384 bytes before runtime and only
641,698,560 bytes after the default 1 GiB sizing headroom. The main G32 FP8-K/INT4-V cache alone is
25,600 bytes per token, or 838,860,800 bytes at 32K, so the two-BF16-vocabulary candidate cannot
support 32K even before MTP KV, fixed GDN state, graph allocations, and workspace. The
embedding-only arena is 31,452,361,984 bytes and leaves 1,833,634,560 bytes after the same sizing
headroom. The attention-QK evaluator leaves 2,475,068,160 bytes after that headroom, the attention
value/output fallback leaves 2,003,208,960 bytes, and the GDN-QK
fallback leaves 2,081,852,160 bytes. Actual resolved capacity remains a runtime measurement, not a
claim here.

The registered binder maps every matrix role to its identity-owned persistent format and the
runtime consumes both planes directly. Q4 calls use caller-owned activation workspace sized for
the separately compiled A4G64 or A8G64 evaluation profile in all Text, MTP, scoring, DFlash-head,
and Vision schedules; they neither allocate nor repack weights at runtime. PPL results report
`q4_activation_bits`, `w8_activation_bits`, and the enabled FP8-Q/K classifier plus its exact T1/T2
context thresholds, so distinct evaluator binaries cannot be confused over the same artifact. The
W8 field names the compile profile; its A8 profile still retains the measured represented-BF16
crossover at smaller shapes.
The all-W8 identity retains its qualified W8 dispatch and zero linear scratch. The
BF16-embedding identity also needs no linear scratch: the binder maps only
`text/token_embedding` to BF16 and the existing BF16 embedding gather consumes it directly, with
no runtime allocation or repacking. The attention fallback likewise has zero quantization
workspace: its 32 promoted matrices use the existing direct BF16 Linear path.
The attention-QK evaluator similarly routes only its 16 promoted matrices through that BF16 path.
The GDN-QK fallback likewise routes its 48 promoted matrices through that BF16 Linear path with
zero quantization workspace.

These evaluation paths are explicit rather than selected by a recipe string:

```text
python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4 --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <q4-eval.ninfer> --device <cpu-or-rocm-torch-device>
python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4_w8 --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <q4-w8-eval.ninfer> --device <cpu-or-rocm-torch-device>
python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4_w8_mse --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <q4-w8-mse-eval.ninfer> --device <cpu-or-rocm-torch-device>
python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_mse --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <w8g32-mse-eval.ninfer> --device <cpu-or-rocm-torch-device>
python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_embedding --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <w8-bf16-embedding-eval.ninfer> --device <cpu-or-rocm-torch-device>
python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_attention_qk --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <w8-bf16-attention-qk-eval.ninfer> --device <cpu-or-rocm-torch-device>
python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_attention_vo --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <w8-bf16-attention-vo-eval.ninfer> --device <cpu-or-rocm-torch-device>
python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_gdn_qk --model <complete-bf16-dir> --draft-ranking out/qwen3_8_27b_draft_ranking.i64 --out <w8-bf16-gdn-qk-eval.ninfer> --device <cpu-or-rocm-torch-device>
```
