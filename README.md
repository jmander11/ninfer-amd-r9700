# NInfer AMD — Radeon AI PRO R9700

Native C++/HIP inference for **Qwen3.8-27B on one Radeon AI PRO R9700** (`gfx1201`,
wave32). Supports CLI generation, OpenAI/Anthropic-compatible serving, Vision,
DFlash2 speculative decoding, fixed concurrency of 1–4 requests, prefix caching,
Device Graphs and perplexity scoring. This is a target-specific AMD engine, not
a general multi-model or multi-GPU framework.

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

### Prefill and ordinary decode · 2026-09-25

| C1, no speculation | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| 4K prompt (workload above) | **2,173.84** | 30.18 |

Longer single-request prefill (cycled code corpus, chunk 2,048, one screen each):
**8K 1,976, 32K 1,765, 64K 1,514 tok/s**. DFlash2 prefill was not remeasured.
Keep chunk **2,048**: tested 4,096 chunks were 3.6–4.2% slower. Results are
workload-specific, not a claimed hardware ceiling. Methodology, quality checks
and committed evidence: `docs/performance.md`.

## Model and precision

- **15.79 GB base / 17.00 GB DFlash-enabled artifact** (decimal file sizes, not VRAM).
- Selective Q4/FP8 weights, Q4 embedding/output head; Q4 DFlash with BF16 codebooks.
- A8 activations for every Q4 operation (prefill, ordinary decode and DFlash
  verification); FP8 activations for the protected FP8 projections.
- Fixed cache: FP8 E4M3FN keys, INT4 values, FP16 value scales. DFlash state is BF16.

On three matched 4K texts, prefill PPL is within **−1.37% to +0.06% of the 5090
NVFP4 reference** (worst: code, +0.06%). This is not universal quality equivalence;
the separate BF16-source production-admission campaign remains unfinished.

Artifacts are not bundled. The installed recipe and creation receipts are under
`/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/`.
Conversion and binding details: `docs/maintainer/r9700-integer-artifact-candidate.md`.

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

Only `gfx1201` is supported. Serialize builds, model conversion and GPU jobs on
the shared host; never use uncapped build parallelism (maximum 14 jobs).

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
- `plans/r9700-autonomous-todos.md` — current development status and paused campaigns.
