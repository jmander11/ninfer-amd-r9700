# NInfer AMD — Radeon AI PRO R9700

Native C++/HIP inference for **Qwen3.8-27B on one Radeon AI PRO R9700** (`gfx1201`,
wave32). Supports CLI generation, OpenAI/Anthropic-compatible serving, Vision,
DFlash2 speculative decoding, fixed concurrency of 1–4 requests, prefix caching,
Device Graphs and perplexity scoring. This is a target-specific AMD engine, not
a general multi-model or multi-GPU framework.

Serving includes constrained JSON/schema output, required or named tool calls,
optional restart-persistent Responses history, and candidate scoring through
`POST /v1/score`. `GET /metrics` (Prometheus) and `GET /metrics.json` expose the process
snapshot. Protocols, options, scoring semantics and metrics are in `docs/serving.md`.

## Performance

One R9700, ROCm 10, Release build, power `auto`, Device Graphs, dense G16 attention.
Code workload: **4,096 prompt + 128 output tokens**, chunk 2,048, maximum context
4,240; median of three repetitions after one warmup. Rates measure inference
phases, excluding model loading—not end-to-end request throughput.

### DFlash2 decode · 2026-09-24

| Concurrent requests | 4 drafts, aggregate tok/s | 5 drafts, aggregate tok/s | Best tok/s per request |
|---|---:|---:|---:|
| 1 | — | **105.15** | 105.15 |
| 2 | 146.25 | **149.40** | 74.70 |
| 3 | **191.05** | 178.29 | 63.68 |
| 4 | **203.26** | 186.88 | 50.82 |

Per-request rates are aggregate divided by concurrency. C1/four drafts was not
remeasured. Five drafts wins at C1–2 and four at C3–4 on this workload; other
prompts may differ. C4 adaptive reaches **201.04 aggregate tok/s**. All 24 final
repetitions and 44 cold-transition cases matched ordinary greedy tokens exactly at that time;
since 2026-09-25 verification attention keeps BF16 queries (ordinary decode uses FP8 queries), so
greedy DFlash can differ from greedy ordinary decode on near-ties. See `docs/performance.md`.

### Prefill · 2026-09-26 (FP8LUT4 artifact)

Single-request prefill, chunk 2,048, same host and power: **8K 3,352, 32K 2,894,
64K 2,455 tok/s** (context ladder). Six-waves-per-SIMD attention and GEMM kernels
with interleaved weight staging added +12% at 32K and 64K over the first FP8LUT4
build in a same-session A/B, with bit-identical PPL. That build was itself +16% to +36% over
the previous Q4 artifact (8K 2,046, 32K 2,075, 64K 1,804). Ordinary and DFlash2
decode were unchanged within noise by the FP8LUT4 migration. Results are
workload-specific, not a claimed hardware ceiling. Methodology, quality checks
and evidence: `docs/performance.md`.

## Model and precision

- **16.75 GB DFlash-enabled artifact** (decimal file size, not VRAM),
  `qwen3.8-27b/r9700-fp8lut4` (the admitted production weights).
- Text-layer weights and output head in FP8LUT4 (4-bit codes with a per-32 codebook of exact E4M3
  values, FP32 row scale; GPTQ-rounded on real-session calibration except attention query/key);
  21 protected attention projections in FP8; Q4 embedding, MTP and DFlash with BF16 codebooks.
- Measured closer to the BF16 source than the 5090 NVFP4 build on every 8K/32K prefill and
  decode PPL cell.
- Per-token FP8 E4M3 activations for every Text projection (prefill, ordinary
  decode and DFlash verification); Q4 A8 for the head, MTP and drafter.
- Fixed cache: FP8 E4M3FN keys, INT4 values, FP16 value scales. DFlash state is BF16.

Against a BF16 reference at 8K, the mean NLL increase is +0.023 (prefill), about a third below
the previous Q4 artifact. At 4K it is 0.024 nats/token below the 5090 NVFP4 build on the same
wiki/technical/code windows (about 2.4% lower PPL). NIAH exact-answer retrieval passes 8K-128K and 240K (the
opencode compaction point, 241K-token prompts) at five positions, standard and multikey.
Admission against the source-BF16 reference (2026-09-27): +0.023/+0.018 at 8K/32K prefill, closer
to BF16 than the 5090 NVFP4 build on the same positions (+0.032/+0.027); graph/eager decode and
whole-inference greedy tokens are exact. This is not universal quality equivalence.

Artifacts are not bundled. The installed artifact and its conversion receipt are
under `/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-fp8lut4/`.
Conversion and binding details: `docs/maintainer/qwen3.8-27b-artifact.md` (FP8LUT4 Text recipe).

## Build

Requires 64-bit Linux, R9700, ROCm 10 with `gfx1201`, CMake ≥3.28, Ninja, C++20,
FFmpeg and libcurl development libraries. Python reference/conversion tools also
require Python 3.11, PyTorch and safetensors.

```sh
cmake -S . -B build-r9700 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 \
  -DNINFER_R9700_Q4_PREFILL_A4_FAMILIES=0 \
  -DNINFER_R9700_W8_ACTIVATION_BITS=8
cmake --build build-r9700 --parallel 4
```

Only `gfx1201` is supported. Serialize GPU work with the shared lock in `AGENTS.md`.
Each agent may run one build at a time, capped at 12 compiler jobs by default (14 maximum).

For the native ROCm Docker image, dedicated development container and GPU device
passthrough commands, see `docs/containers.md`. No NVIDIA container runtime is needed.
`compose.yaml` defaults to temperature1.5, DFlash,4GiB RAM prefix cache and32GiB
persistent disk cache; copy `.env.example` to `.env` and set your local paths.

## Run

Use a DFlash-enabled `.ninfer` artifact for speculative generation:

```sh
build-r9700/apps/ninfer /path/to/model-dflash.ninfer \
  --prompt "Explain wave32 matrix instructions." \
  --spec dflash --draft-tokens 5 --max-new 256

build-r9700/apps/ninfer-serve /path/to/model.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity auto --max-concurrency 2
```

Also built: `build-r9700/apps/ninfer-ppl` and `build-r9700/bench/ninfer_bench`.
Use each executable's `--help` for exact options. Benchmark reproduction commands
and reports: `profiles/bench/r9700-remaining-candidates-20260924/`.

## Documentation

- `docs/cli.md` — generation, sampling, media, speculative decoding and prefix storage.
- `docs/serving.md` — OpenAI/Anthropic endpoints and request lifecycle.
- `docs/performance.md` — benchmarks, recipes, quality comparisons and limitations.
- `docs/README.md` — architecture, artifact formats, conversion and kernel development.
- `plans/r9700-autonomous-todos.md` — live development work items.
