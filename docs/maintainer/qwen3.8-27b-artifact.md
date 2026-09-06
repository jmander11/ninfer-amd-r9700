# Qwen3.8-27B R9700 artifact

The sole target owns 1,124 ordered objects: six exact frontend resources and 1,118 tensors. During
migration it admits nine explicit evaluation identities under target key `qwen3_8_27b_r9700`:
the all-W8 `r9700-int-candidate`, all-Q4 `r9700-q4g64-n16k16-eval`, and mixed
`r9700-q4-w8-n16k16-eval`. The same-format `r9700-q4-w8-mse-n16k16-eval` retains that mixed plan and changes
only its source-derived scale objective. `r9700-w8-bf16-embed-eval` retains all-W8 matrices except for a
direct source-BF16 token embedding, and `r9700-w8-bf16-attn-vo-eval`, which restores the complete
full-attention gate/value and output Q5 role family. None is a final recipe selection.
`r9700-w8-bf16-attn-qk-eval` restores query/key in all 16 full-attention layers, while
`r9700-w8-bf16-gdn-qk-eval` restores GDN query/key in all 48 GDN layers.
The same-size `r9700-w8g32-mse-eval` retains the all-W8 layout and runtime while selecting each
stored FP16 group scale by a deterministic source-only decoded-weight SSE objective.

The provisional tensor counts are 582 BF16, 96 FP32, one I32, and 439 W8G32. Conversion starts only
from the complete official BF16 source. It uses the target-owned inventory and source recipe under
`tools/convert/qwen3_8_27b_r9700`, validates checkpoint dimensions, all source tensors and shards,
six exact frontend resource hashes, draft-head shortlist provenance, and every planned object
before opening the output.

The optimized shortlist is not recipe-dependent: every registered identity stores
`text/draft_head` as `Q4G64_F16S[131072,5120]` in `r9700-q4g64-n16-k16-v1`, with packed signed codes
followed by FP16 G64 scales. Startup validates that exact object even when speculation is disabled
and materializes it only for optimized MTP/DFlash. Materialization exposes the resident artifact
planes directly to Linear; it does not repack weights. The execution-side activation boundary is
compile-time A8G64 over represented BF16 input, not a second artifact format or runtime selector.

```bash
python3 -m tools.convert.qwen3_8_27b_r9700.build_draft_ranking \
  --corpus tools/ppl/corpus.ids \
  --out out/qwen3_8_27b_draft_ranking.i64

python3 -m tools.convert.qwen3_8_27b_r9700.convert \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_candidate.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_mse \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3.8-27b-r9700-w8g32-mse-eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4 \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_q4_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4_w8 \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_q4_w8_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4_w8_mse \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_q4_w8_mse_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_embedding \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_w8_bf16_embedding_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_attention_qk \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_w8_bf16_attention_qk_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_attention_vo \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_w8_bf16_attention_vo_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_gdn_qk \
  --model /path/to/complete/Qwen3.8-27B-BF16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out out/qwen3_8_27b_r9700_w8_bf16_gdn_qk_eval.ninfer \
  --device cpu
```

The eight additional commands are registered evaluation-only lanes. `r9700-w8g32-mse-eval`
uses the same 439 W8G32 tensor plan, 30,260,413,792 tensor bytes, 30,260,425,984-byte device
arena, and projected 30,273,439,488-byte artifact as the baseline all-W8 identity. It refines
only source-derived group scales and does not change runtime dispatch. `r9700-q4g64-n16k16-eval` stores all
439 non-direct matrices as Q4G64: 15,159,801,760 tensor bytes, a 15,159,815,680-byte device arena,
and a projected 15,172,829,184-byte complete artifact with the hash-qualified frontend resources.
`r9700-q4-w8-n16k16-eval` keeps the 183 source-Q4 roles at Q4G64 and promotes all remaining 256 matrix
roles to W8G32: 22,868,177,312 tensor bytes, a 22,868,191,232-byte device arena, and a projected
22,881,204,736-byte complete artifact. `r9700-q4-w8-mse-n16k16-eval` has those exact formats, counts,
layouts, and byte totals; both its Q4G64 and W8G32 groups use deterministic source-only MSE scale
selection. That scale objective consumes no activation, draft-ranking/corpus-token, PPL, argmax,
or GPU measurement input; the ordinary artifact-wide draft-head ranking preflight remains required.
The 439-W8G32 candidate occupies 30,260,413,792 tensor
bytes. The W8/BF16-embedding evaluator stores 438 W8G32 matrices and the represented source-BF16
`text/token_embedding`: 31,452,349,792 tensor bytes, a 31,452,361,984-byte device arena, and a
projected 31,465,375,488-byte complete artifact. The attention query/key evaluator has 16 BF16
and 423 W8 matrices: 30,810,916,192 tensor bytes, a 30,810,928,384-byte device arena, and a
projected 30,823,941,888-byte complete artifact; 2,475,068,160 bytes remain after default
headroom. The attention value/output evaluator has 32 BF16
and 407 W8 matrices: 31,282,775,392 tensor bytes, a 31,282,787,584-byte device arena, and a
projected 31,295,801,088-byte complete artifact. It leaves 2,003,208,960 bytes after the default
1 GiB sizing headroom on a 32 GiB R9700. The GDN-QK evaluator has 48 BF16 and 391 W8 matrices:
31,204,132,192 tensor bytes, a 31,204,144,384-byte device arena, and a projected
31,217,157,888-byte complete artifact; 2,081,852,160 bytes remain after default headroom. Q5/Q6
are reserved as fallbacks if native Q4 and
W8 cannot satisfy quality and capacity together; they are not primary candidate formats. These
are representation sizes, not speed, quality, or selection claims.

The first measured 8K comparison rejects all-Q4: its mean NLL increases by 0.1355 and 800 greedy
positions flip relative to the BF16 scorer. All-W8 is much closer at +0.002053 mean NLL but still
flips 90 greedy positions, and the short mixed-Q4/W8 result is already poor. A measured BF16 output
head did not help: +0.002425 mean NLL and 92 flips, so that identity is rejected and no longer
registered. The remaining source-Q6 role is the token embedding; unlike the final projection, it
feeds every input through all 64 residual layers. The active BF16-embedding evaluator isolates that
role while leaving `text/output_head` and all internal matrices W8. Restoring both vocabulary
matrices would add another 1,191,936,000 resident bytes and leave only 641,698,560 bytes
after the default 1 GiB sizing headroom on a 32 GiB R9700. That is smaller than the main 32K G32
cache payload alone (838,860,800 bytes), before MTP KV, fixed state, graphs, or workspace.
The attention value/output evaluator is independently runnable. It promotes
all 16 full-attention gate/value and output pairs—the complete source-Q5 value/output family—rather
than choosing layers without sensitivity evidence; query/key, every GDN/MLP matrix, vocabulary,
MTP, and Vision remain W8.
The full-attention query/key evaluator independently promotes that complete fused projection in
all 16 full-attention layers without splitting the value/output family after observing its result.
Gate/value, attention output, every GDN/MLP matrix, vocabulary, MTP, and Vision remain W8.
The GDN fallback similarly promotes the complete GDN query/key family across all 48 GDN layers
instead of choosing individual layers. GDN value/z and output, every attention/MLP matrix,
vocabulary, MTP, and Vision remain W8.

The completed matched 8K G16 evidence retains BF16-greedy differences as diagnostics. Token
embedding produces +0.001624 mean NLL / 80 flips, full-attention value/output
+0.001098 / 84, GDN query/key +0.001375 / 98, and full-attention query/key +0.001996 / 81.
All-W8 is +0.002053 / 90. With A4 activations, all-Q4 and mixed Q4/W8 fail the accuracy tier at
+0.135526 / 800 and +0.046148 / 420. The matched A8 route improves all-Q4 to +0.039509 / 450
(PPL 6.720524) and mixed Q4/W8 to +0.013815 / 237 (PPL 6.550048). The same-format source-MSE
mixed artifact with adaptive-A8 W8 execution is the Q4-containing leader at +0.012077 / 234
(PPL 6.538677); its represented-BF16 W8 control remains retained at +0.013005 / 234 (PPL
6.544746). Its three new NLL-at-least-10 positions are within the five-position 8K budget, so it
is quality-eligible. The
all-Q4+A8 row meets the capacity-speed tier at +0.039509 mean NLL and nine new severe positions.
Both A8 profiles retain their quality evidence. Under the C=1..4 product cap, both recipes and
both G16/G32 cache groups remain capacity candidates. The earlier mixed-recipe C7/C8 startup
failures are retained as out-of-scope stress evidence and no longer exclude it. Fresh exact C=1..4
capacity and whole-inference evidence is required for selection; the earlier C=1..8 manifests are
historical rather than current product evidence.
Their historical mixed C=1..4 rows resolved G16 to 262,144/314,112/301,888/289,664 tokens and
G32 to 262,144/326,656/313,984/301,248 tokens; these are retained facts, not reusable admission
manifests.

The C++ binder consumes Q4G64/W8G32 planes directly according to each explicit identity. Q4G64
uses the Q4-only `r9700-q4g64-n16-k16-v1` persistent order; W8G32 remains row-split. The converter
and `transcode_q4_n16k16.py` are the only layout writers, while runtime binding never repacks. The
runtime quantizes represented BF16 activations into caller-owned, compile-selected A4G64 or A8G64
evaluation scratch and launches the qualified native signed-INT4 WMMA route without hidden
allocation or runtime weight repacking.
All-W8 keeps its previously qualified workspace-free dispatch. The BF16 token table goes directly
through the existing BF16 embedding gather, and BF16 attention projections use the existing BF16
Linear path; neither route repacks or copies weights to a private allocation.
The family schedule reserves the
maximum candidate scratch for Text, MTP, scoring, DFlash-head, and Vision call sites.

The three legacy evaluation artifacts can be migrated without source shards and without modifying
them. The transcoder has a closed identity map: all-Q4, mixed source-MSE Q4/W8, and four-role
FP8/Q4 map only to their corresponding `-n16k16-eval` identities. Inspect exact identity,
inventory, offsets, formats, layouts, sizes, and projected identity without reading payload bytes
or creating an output with:

```bash
python3 -m tools.convert.qwen3_8_27b_r9700.transcode_q4_n16k16 \
  --source out/qwen3.8-27b-r9700-q4g64-eval.ninfer --preflight-only
python3 -m tools.convert.qwen3_8_27b_r9700.transcode_q4_n16k16 \
  --source out/qwen3.8-27b-r9700-q4-w8-mse-eval.ninfer --preflight-only
python3 -m tools.convert.qwen3_8_27b_r9700.transcode_q4_n16k16 \
  --source out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-eval.ninfer --preflight-only
```

An actual migration additionally requires a distinct nonexistent output path, for example:

```bash
python3 -m tools.convert.qwen3_8_27b_r9700.transcode_q4_n16k16 \
  --source out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-eval.ninfer \
  --output out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer
```

The staging tool copies every non-Q4 payload exactly, transposes only Q4 code/scale storage,
verifies exact logical code and scale hashes plus non-Q4 payload hashes before create-only
publication, then verifies the published whole-file hash and inode. It refuses to replace either
input or an existing destination. The all-Q4 and mixed conversions use the same command shape with
outputs `qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer` and
`qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer`, respectively.

Every selectable N16/K16 artifact requires an adjacent create-only migration receipt. The retained
four-role receipt remains the authority published by the transcoder. The all-Q4 and mixed receipts
have also been published with the common producer and are immutable create-only authorities; do not
rerun receipt publication or replace any of the three adjacent receipt paths.

Receipt validation reopens the exact legacy source and its conversion receipt, the frozen
transcoder, and the migrated artifact directory. It requires the registered source and N16 object
plans, exact artifact hashes and sizes, and the logical-Q4/non-Q4 verification recorded by the
publisher. Benchmark, selection, Pareto, and DFlash authorities carry the same normalized receipt
identity; no recipe may borrow another recipe's migration receipt.

The retained all-Q4 receipt is adjacent to
`out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer`, SHA-256
`a8567a6b25176aac1f2106bcac3131b74d887e1a5be2c98da23c38a3a7870e54`; it binds source plan
`d77d47a8cc0e005c50a6488fdde7b539365a81443b59993113c023a8dcabf25c` to N16 plan
`bff624cbfda357d7c8b355530824682b1625c68c1f909ddb8b4241c2b3d21ec6` across 1,124 objects,
including 439 Q4 objects. The retained mixed receipt is adjacent to
`out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer`, SHA-256
`1298ba51b2e80c225663814771a706afcbbe014e03f0bdefd4aa7dbf1cc8551c`; it binds source plan
`c9392556ae633dde553ed74d328c4f04cc6a1cde8e17c166f271ead192b6884c` to N16 plan
`14b80eb8a8200112169ef34bda4520d8b50233935bc9aa1c79f02b674ef36fb3` across 1,124 objects,
including 183 Q4 objects. Both receipts bind transcoder SHA-256
`e988d0ecc7d20a12728aa8313a71221eeea9c30dfff5d998020a826baab52801` and common receipt
producer SHA-256 `3fb4f58e376e5c8dbd333f06ef146796410eb7829de4130c408d8adc5a615d64`.
The earlier occupied prepared screen/finalist roots are not upgraded in place. Any subsequent
receipt-bound preparation uses fresh names ending in `-receipt-bound-n16k16-20260905` and creates
a fresh campaign, pipeline, and selection authority after the prefill practical-ceiling gate.

The retained migrated artifact is
`out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer`, identity
`qwen3.8-27b/r9700-q4g64-f8e4m3-four-role-n16k16-eval`, 21,553,549,312 bytes, SHA-256
`040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2`. Its immutable legacy
source is 21,553,545,216 bytes with SHA-256
`1dfe9626fd6412592f87480a2f6934e8494a4b267693a25831dff490959542ce`. A production-parser
reopen confirmed the exact ordered 1,124-object selected inventory and 295 Q4 descriptors; exact
logical Q4 code/scale hashes and every non-Q4 payload hash matched the source, which remained
unchanged.

The three explicit DFlash2 evaluation identities are
`qwen3.8-27b/r9700-q4g64-n16k16-dflash2-q4-eval` and
`qwen3.8-27b/r9700-q4-w8-mse-n16k16-dflash2-q4-eval`, plus
`qwen3.8-27b/r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval`. They extend their named base inventories with the
complete 66-object DFlash plan: 32 Q4G64-FP16-scale matrices plus 34 BF16 objects, including both
BF16 selector codebooks. Base objects retain their identity-owned Q4G64/W8G32 or selected
four-role rowwise-FP8 formats. DFlash Q4
calls use the same compile-selected adaptive A8G64 execution intermediate and caller-owned
workspace as base Q4 calls; K=25600 determines the enlarged workspace maximum. The binder requires
the complete DFlash inventory for these identities and admits no DFlash objects under other
registered evaluation identities, preventing a byte-compatible base artifact from being silently
reinterpreted as a DFlash package. DFlash grouped-convolution and selector child projections use
the profile-derived Q4 type and the same caller-owned workspace; neither child may invoke the
workspace-free Linear overload.
The DFlash companion corresponding to the recipe selected by the base C=1..4 Pareto decision is
eligible for physical DFlash shortlist, capacity, phase, and whole-inference campaigns. The other
converted companions remain valid provenance and binder evidence but receive no duplicate
physical campaign. The hybrid companion is not yet materialized; if selected, its conversion must
bind the exact hybrid base receipt before those fresh DFlash gates begin.
The production DFlash K/W is not encoded in either evaluation artifact and cannot be chosen from a
filename or shortlist rank. It is selected only by the schema-v3 provenance-bound record emitted by
`tools/bench/assemble_dflash_selection.py` after the base schema-v7
`terminal_production_selection`, complete
shortlist-frontier C=1..4 capacity accounting, and every capacity-eligible 22-point DFlash matrix.

The ranking is exactly one 248,320-column little-endian I64 total-frequency row. The builder accepts
explicit `.ids` paths, discovers each sibling manifest, and validates Qwen3.8 tokenizer identity,
decimal token IDs in `0..248076`, token count, and payload SHA-256 before summing each independent
corpus once. It emits a JSON provenance sidecar with every input and output hash. Converter
preflight requires the sibling sidecar, revalidates every named corpus and its manifest, and
byte-compares a freshly derived row before opening artifact output. The non-tiled PPL corpus is the
only currently valid repository input. The benchmark corpus is tiled throughput
padding and is rejected because counting its repetitions would bias the shortlist. The builder
does not force special IDs; that remains the converter's tokenizer-owned step. Retired-model counts
remain invalid provenance. The artifact output is never overwritten and, after its atomic close,
receives a conversion report containing the output path/size/SHA-256, source/checkpoint, exact
ranking path/size/SHA-256, recipe, object, environment, and timing provenance. The candidate
identity is usable only for the real comparison gates. It
must be renamed and made final only after BF16-reference operator/model parity, paired 8K and 32K
quality guardrails, same-candidate eager/Device-Graph parity, resolved capacity, and
whole-inference performance establish the non-dominated weight recipes and fixed cache layout.

Cache group and plane order are compile-time runtime-state profiles, not tensor descriptors or
artifact identity fields. Separate G16/G32 evaluator builds may consume this same candidate artifact
when the weight recipe is held constant.

`r9700-integer-artifact-candidate.md` defines the live candidate details and outstanding external
gate. `artifact-container.md`, `tensor-formats.md`, and `storage-layouts.md` define generic framing
and representation.
