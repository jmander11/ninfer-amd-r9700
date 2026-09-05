# NInfer for Radeon AI PRO R9700

NInfer is a from-scratch C++/HIP inference engine specialized for one AMD Radeon AI PRO R9700
(`gfx1201`, wave32) and the Qwen3.8-27B model. It supports local CLI generation, OpenAI and
Anthropic compatible HTTP serving, teacher-forced perplexity scoring, fixed concurrency from one to
four requests, Vision input, MTP, DFlash2, prefix reuse, host-RAM prefix spill, and Device Graphs.

There is no compatibility backend and no runtime cache-format selector. The growing
Text/MTP cache has one fixed represented format:

- keys: FP8 E4M3FN;
- values: signed canonical INT4;
- value scales: FP16, one scale per fixed feature group;
- attention softmax and accumulation: FP32;
- explicit model-boundary outputs: BF16 where required by the Qwen formula.

FP64 is used only by independent qualification oracles, never by production attention.

## Migration status

The HIP core, artifact reader/materializer, typed cache, sole-target runtime, public Engine, CLI,
server, PPL executable, and a broad set of native Ops build for `gfx1201`. Physical R9700
qualifiers cover cache bytes/lifecycle, full Text attention, speculative transitions, DFlash2,
Vision, Linear, GDN, sampling, persistent state, and fixed C=1..4 runtime planning.

The currently accepted artifact identity is deliberately provisional:

```text
model_id   = qwen3.8-27b
weights_id = r9700-int-candidate
target_key = qwen3_8_27b_r9700
recipe     = W8G32 candidate
```

It exists so the real model gates can run. It is not yet a selected or published product artifact.
Final weight-recipe and G16/G32 cache-layout selection require paired BF16-reference quality,
resolved-capacity, and matched whole-inference performance results. BF16 greedy-token differences
are retained as diagnostics rather than a zero-difference gate.
Those inputs are not committed to this repository.

## Requirements

- 64-bit Linux;
- AMD Radeon AI PRO R9700;
- a coherent ROCm 10 installation with `gfx1201` support;
- CMake 3.28 or newer and Ninja;
- a C++20 compiler;
- FFmpeg development libraries;
- libcurl development files for the CLI/server media-acquisition path;
- Python 3.11, PyTorch, and safetensors only when converting or running Python reference tooling.

The build rejects every HIP architecture other than `gfx1201`.

## Build

```sh
cmake -S . -B build-r9700 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON
cmake --build build-r9700 --parallel
```

The product executables are:

```text
build-r9700/apps/ninfer
build-r9700/apps/ninfer-serve
build-r9700/apps/ninfer-ppl
```

Use each executable's `--help` output as the exact option/default authority.

## Artifact conversion

The provisional converter consumes a complete BF16 source directory directly. It preflights every
required shard and frontend resource before creating output, writes final bound layouts, and never
relies on runtime weight repacking.

```sh
python3 -m tools.convert.qwen3_8_27b_r9700.build_draft_ranking \
  --corpus tools/ppl/corpus.ids \
  --out out/qwen3_8_27b_draft_ranking.i64

python3 -m tools.convert.qwen3_8_27b_r9700.convert \
  --model /path/to/complete-qwen3.8-27b-bf16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out models/qwen3_8_27b_r9700_candidate.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4 \
  --model /path/to/complete-qwen3.8-27b-bf16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out models/qwen3_8_27b_r9700_q4_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4_w8 \
  --model /path/to/complete-qwen3.8-27b-bf16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out models/qwen3_8_27b_r9700_q4_w8_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_q4_w8_mse \
  --model /path/to/complete-qwen3.8-27b-bf16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out models/qwen3_8_27b_r9700_q4_w8_mse_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_embedding \
  --model /path/to/complete-qwen3.8-27b-bf16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out models/qwen3_8_27b_r9700_w8_bf16_embedding_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_attention_vo \
  --model /path/to/complete-qwen3.8-27b-bf16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out models/qwen3_8_27b_r9700_w8_bf16_attention_vo_eval.ninfer \
  --device cpu

python3 -m tools.convert.qwen3_8_27b_r9700.convert_w8_bf16_gdn_qk \
  --model /path/to/complete-qwen3.8-27b-bf16 \
  --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
  --out models/qwen3_8_27b_r9700_w8_bf16_gdn_qk_eval.ninfer \
  --device cpu
```

The ranking builder validates the explicitly named Qwen3.8 token IDs, count, domain, and SHA-256
against the sibling manifest and emits exactly one little-endian I64 frequency row plus its JSON
provenance sidecar. Converter preflight requires that sibling sidecar, revalidates every named
corpus and manifest, and re-derives the row before opening the artifact. The current PPL corpus is
unique source text and is valid ranking evidence; the
benchmark corpus is deliberately excluded because its manifest says it is tiled for throughput and
would bias frequencies. Tokenizer special-ID force-inclusion remains converter-owned. Retired-model
ranking fixtures are not valid provenance. The converter refuses partial checkpoints and existing
output files. `convert_q4.py` writes the registered evaluation-only all-Q4G64 capacity artifact (15,159,801,760
tensor bytes); `convert_q4_w8.py` preserves source-Q4 roles and promotes every source-Q5/Q6/W8 role
to W8G32 (22,868,177,312 tensor bytes); `convert_q4_w8_mse.py` keeps that exact format and byte
plan while refining both Q4G64 and W8G32 scales from the represented source weights alone;
`convert.py` retains the all-W8G32 evaluator
(30,260,413,792 tensor bytes); and `convert_w8_bf16_embedding.py` keeps those W8 matrices except for
the direct source-BF16 token embedding (31,452,349,792 tensor bytes). The prior BF16-output-head
evaluator was removed after its 8K result failed to improve all-W8. Q5/Q6 remain fallback
measurements rather than primary artifacts; `convert_w8_bf16_attention_vo.py` is the coherent Q5
fallback with all 16 full-attention gate/value and output pairs in BF16 (31,282,775,392 tensor bytes),
while `convert_w8_bf16_gdn_qk.py` is the complete 48-layer GDN query/key fallback
(31,204,132,192 tensor bytes),
and none of these evaluator identities selects the production recipe. DFlash2 companion objects
are optional in the base artifact but are required when
starting the Engine with `--spec dflash`. See
`docs/maintainer/r9700-integer-artifact-candidate.md` for the exact current inventory and codec.

## CLI

Text generation:

```sh
build-r9700/apps/ninfer models/qwen3_8_27b_r9700_candidate.ninfer \
  --prompt "Explain wave32 matrix instructions." \
  --max-new 256
```

Structured messages and media:

```sh
build-r9700/apps/ninfer models/qwen3_8_27b_r9700_candidate.ninfer \
  --messages examples/cli/messages/scenario_translation_markdown.json \
  --vision \
  --max-new 256
```

MTP and DFlash2 are startup-fixed backends. DFlash2 is the preferred speculative path for the
R9700 product and is the only backend with remaining feature/performance work. MTP remains a
supported, already-implemented path whose cache, row-view, state, and exact-execution behavior are
kept under regression coverage; no new MTP optimization is required for product completion.

```sh
build-r9700/apps/ninfer models/qwen3_8_27b_r9700_dflash_candidate.ninfer \
  --prompt "Write a short proof." \
  --spec dflash --draft-tokens 4
```

The CLI streams answer content to stdout and diagnostics/reasoning to stderr. It accepts exactly one
prompt source: `--prompt` or `--messages`.

## HTTP server

```sh
build-r9700/apps/ninfer-serve models/qwen3_8_27b_r9700_candidate.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 8192 \
  --kv-capacity auto \
  --max-concurrency 2
```

The server exposes OpenAI Responses, OpenAI Chat Completions, and Anthropic Messages protocol
surfaces. Request concurrency is startup-fixed in `[1,4]`; excess requests enter a bounded FIFO and
are never preempted. See `docs/serving.md` for endpoint and streaming semantics.

## Perplexity and Pareto gates

Encode a corpus once:

```sh
build-r9700/apps/ninfer-ppl \
  --encode \
  --weights models/qwen3_8_27b_r9700_candidate.ninfer \
  --text corpus.txt \
  --ids corpus.ids
```

Score it:

```sh
build-r9700/apps/ninfer-ppl \
  --weights models/qwen3_8_27b_r9700_candidate.ninfer \
  --ids corpus.ids \
  --schedule prefill \
  --out-json profiles/bench/r9700-ppl.json
```

The report includes per-token NLL and greedy argmax IDs. Final admission uses the paired campaign
runner described in `tools/ppl/README.md`: its schema-v6 result binds the candidate, scorers,
validated corpus, commands, and sidecar hashes; requires explicit PPL and new-severe-position
guardrails; and retains BF16 greedy flips as a diagnostic. Prefill/decode schedules require an
explicit measured per-token NLL bound and report their greedy flips diagnostically because private
attention precision may differ. Same-route graph/eager, speculative/ordinary, and draft-window
variants retain exact-token parity. The
reference is a separate BF16-source scorer; changing an Engine cache flag is not a reference path.
G16 and G32 are separate compile-time evaluator/Engine profiles over runtime state, not separate
`.ninfer` identities, and the selection campaign requires their candidate artifacts to be
byte-identical. Speculative proposal acceptance and whole-path latency are measured separately by
the schema-v14 native benchmark matrix, whose schema-v20 reports identify the compiled KV group
and exact plane layouts, Q4/W8
activation profiles, the exact FP8-Q/K crossover classifier, and the compile-bound dense or
XAttention qualification identity.

## Correctness and performance

Focused physical qualifiers are built under `build-r9700/src/`. The live completion ledger and
latest verified commands are in `plans/r9700-autonomous-todos.md`.

Real-model 8K PPL evidence is retained for every evaluated weight/activation profile. The BF16
reference is 6.460181; the best Q4-containing profile is the 183-Q4/256-W8 source-MSE artifact with
A8G64 Q4 and adaptive-A8G32 W8 execution at 6.538677. Its +0.012077 mean-NLL delta and three new
severe positions meet the 8K accuracy tier; the identical artifact's represented-BF16 W8 control
is retained at 6.544746. All-Q4+A8 meets the capacity-speed tier at +0.039509 mean NLL and nine new
severe positions. BF16-greedy differences are diagnostic. No end-to-end R9700
tokens-per-second result is published for a final artifact. Pareto selection still requires current
matched dense/sparse quality, post-promotion C=1..4 capacity, 32K graph/eager parity, phase and
whole-inference measurement, and relevant profiler attribution on an otherwise idle R9700.
Exact-v3 BF16 authority and the dense all-Q4 quality rebase are complete; mixed and sparse quality
plus all post-promotion capacity gates remain open.

ROCm tracing and ISA/resource inspection use `rocprofv3`, `rocprof-compute` when its installed release
supports `gfx1201`, HIP events, and LLVM disassembly. Radeon GPU Profiler is optional. The currently
installed profiler stack does not expose complete VALU/LDS/stall or absolute request-size counters
for `gfx1201`; zero values from those events are not accepted as evidence. Dispatch-level GL2C/TCP
hit counters are usable while the card is held in `profile_standard`, but those pinned-clock timings
are attribution-only and production performance is measured under `auto`.

## Documentation

- `docs/cli.md`: CLI behavior and options.
- `docs/serving.md`: HTTP contracts and server operation.
- `docs/maintainer/qwen3.8-27b-model.md`: exact model and family-runtime semantics.
- `docs/maintainer/paged-kv-cache.md`: typed cache ownership and capacity.
- `docs/maintainer/softmax-attention.md`: Text/MTP and DFlash attention ownership.
- `docs/maintainer/op-development.md`: numerical-oracle and Op admission rules.
- `docs/maintainer/kernel-iteration.md`: gfx1201 kernel optimization procedure.

## Scope

NInfer is intentionally not a general inference framework. It supports one registered model/device
contract, one resident model instance, and small fixed concurrency. Additional GPUs, checkpoint
families, compatibility backends, plugin discovery, preemptive scheduling, and runtime weight
repacking are outside the product.
