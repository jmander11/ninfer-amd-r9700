# NInfer AMD — Radeon AI PRO R9700

C++/HIP inference for **Qwen3.8-27B on one AMD Radeon AI PRO R9700**, with **1–16 concurrent requests** (1–12 with DFlash2).

An AMD port of [Neroued/NInfer](https://github.com/Neroued/ninfer), distributed under Apache-2.0. Model weights are separate; see `docs/acknowledgements.md` for credits.

## What it does

- **Chat interfaces.** Run from the terminal or connect a client that uses OpenAI Chat Completions, Responses or Anthropic Messages. Answers stream as they are generated, with reasoning kept separate from the answer.
- **Tools and structured answers.** Let the model choose a tool or require a specific one, with incremental tool-call streams. Constrain answers to JSON, a JSON Schema, an EBNF grammar, a full-match regex or literal choices.
- **Images and video.** Add `--vision` at startup to accept media alongside text.
- **Speculative decoding.** DFlash2 proposes tokens for the main model to check together, adapting up to seven drafts. It is the default for up to twelve requests; MTP supports up to sixteen.
- **Conversation history.** Cached prefixes avoid processing the same text again. Keep them in GPU memory, add a RAM cache, or save them to SSD for reuse after a restart.
- **Generation controls and scoring.** Adjust sampling, choose greedy decoding, inspect token probabilities or compare candidate answers. `ninfer-ppl` measures how well the model predicts text.
- **Server activity.** Prometheus metrics, JSON statistics and request logs report throughput, memory use, cache hits and draft acceptance.

## Built for the R9700

HIP kernels and Device Graphs are tuned for `gfx1201` (wave32). The engine batches up to sixteen requests with MTP or without speculation, and up to twelve with DFlash. With DFlash, it can process a new prompt while other requests generate. GPU memory is planned at startup, with workspace shared between execution phases.

**LQ4: 4-bit codebook weights decoded to FP8 for computation.** Most Text projections and the output head use LQ4; 21 sensitive projections stay in FP8. The DFlash-enabled artifact, `qwen3.8-27b/r9700-lq4`, is **16.75 GB** on disk. This is a decimal size, not a VRAM requirement. The default paged KV cache uses NVFP4 E2M1 keys and values with E4M3 scales per 16 features. FP8 keys with INT4 values and FP16 scales remain opt-in (`--kv-cache fp8-int4`), as does TQ-K4-V3 (`--kv-cache tq-k4-v3`). DFlash keeps its own BF16 state.

For image/video inference, select the **17.42 GB** `qwen3.8-27b/r9700-lq4-vision-bf16` artifact and add `--vision`. It keeps the admitted Text weights byte-exact and uses original BF16 Vision matrices, adding about 640 MiB to storage and Vision-enabled weight memory. Vision-off loading has the same memory plan. Conversion and accuracy/performance evidence are in `docs/maintainer/qwen3.8-27b-artifact.md`.

This port combines NInfer features with AMD-specific kernels, weight/cache formats and memory tuning. Port decisions: `docs/maintainer/upstream-sync.md`.

## Context and capacity

Each request can use up to **262,144 tokens** for prompt and answer combined, subject to the shared GPU cache capacity below.

We recommend enabling the RAM cache, for example `--kv-ram-capacity 8192` (MiB). Without it, each request reserves GPU cache for its entire possible answer up front. With it, each request reserves its prompt plus up to 8,192 answer tokens at a time, then extends the reservation as needed. This lets more requests start sooner. If active requests together outgrow the GPU cache, the engine moves some to RAM at a safe point and resumes them when space is available. The same RAM keeps finished conversation prefixes for reuse.

The default cache is **NVFP4**: 4-bit floating-point keys and values with one 8-bit scale per 16 features, using 288 bytes per token per KV head. **TurboQuant** is the opt-in `--kv-cache tq-k4-v3` cache, called "TQ" below. It rotates each key and value vector with a Hadamard transform, then stores 4-bit keys and 3-bit values (228 bytes). From five requests up, where shared cache capacity rather than the 262,144-token request limit is the constraint, TurboQuant holds about **1.26x** as many tokens. At four requests it already reaches the full 4 × 262,144.

Shared cache capacity in tokens, measured 2026-11-16. Settings: automatic sizing, the default 32 MiB headroom, RAM/SSD and Vision off. Adaptive drafting uses up to seven drafts for DFlash and up to three for MTP:

| Concurrent requests (C) | DFlash K7, NVFP4 | DFlash K7, TQ | MTP K3, NVFP4 | MTP K3, TQ |
|---:|---:|---:|---:|---:|
| 1 | 262,144 | 262,144 | 262,144 | 262,144 |
| 2 | 524,288 | 524,288 | 524,288 | 524,288 |
| 3 | 786,432 | 786,432 | 786,432 | 786,432 |
| 4 | 902,656 | 1,048,576 | 905,472 | 1,048,576 |
| 5 | 890,944 | 1,125,312 | 894,848 | 1,130,240 |
| 6 | 879,232 | 1,110,528 | 884,096 | 1,116,608 |
| 7 | 867,520 | 1,095,680 | 873,216 | 1,102,976 |
| 8 | 855,808 | 1,080,896 | 862,464 | 1,089,408 |
| 9 | 842,368 | 1,064,064 | 850,752 | 1,074,624 |
| 10 | 829,696 | 1,048,064 | 840,000 | 1,060,992 |
| 11 | 817,024 | 1,032,064 | 829,184 | 1,047,488 |
| 12 | 804,288 | 1,015,936 | 818,368 | 1,033,728 |
| 13 | Unsupported | Unsupported | 807,488 | 1,020,096 |
| 14 | Unsupported | Unsupported | 796,800 | 1,006,400 |
| 15 | Unsupported | Unsupported | 785,984 | 992,896 |
| 16 | Unsupported | Unsupported | 775,168 | 979,200 |

**TurboQuant trade-offs.** It raises perplexity by 0.2–0.6% against NVFP4 on the 8K and 32K test cells. With only the cache quantized against a BF16 reference, it adds 0.6% (32K) and 1.1% (64K), where NVFP4 adds 0.2% and 0.3%. On a 60-case long-context retrieval suite (32K to 260K), NVFP4 finds all 60 and TurboQuant 59. Its one miss, a multi-value needle at the start of a 260K prompt, also fails on the 5090 NVFP4 build. Speed is about the same; see below. The older `--kv-cache fp8-int4` (FP8 keys, INT4 values) remains available with about 0.69x NVFP4's capacity.

## Performance and quality

Measured 2026-11-16 on one R9700, using the same build for both caches. Settings: Release build, Device Graphs, power `auto`, chunk 2048 and optimized proposal heads. Each cell shows NVFP4 / TQ.

**Prompt processing (prefill), one request:**

| Prompt tokens | Average tokens/s |
|---:|---:|
| 8,192 | 3,456 / 3,444 |
| 32,768 | 2,959 / 2,953 |
| 65,536 | 2,506 / 2,486 |
| 131,072 | 1,923 / 1,897 |
| 245,760 | 1,364 / 1,339 |

**Answer generation.** Both tables use p-less sampling at the default temperature of 1.5 and 512 output tokens per request, including reasoning. Runs use a 65,536-token request limit, with RAM/SSD and Vision off. Rates exclude prompt processing. C>1 rates are totals across requests while all are generating; divide by C for the per-request rate. Each cell is the mean of three measured waves after a warmup; DFlash C1 uses twelve.

TurboQuant is **0.1% slower** than NVFP4 on average, using the geometric mean over the 20 DFlash and 25 MTP cells. Individual cells differ by up to ±5% in either direction. This is sampling noise: each run accepts a different number of drafted tokens. Draft acceptance is the same for both caches.

**DFlash, adaptive up to seven drafts:**

| Workload | Prompt tokens | C1 tokens/s | C6 aggregate tokens/s | C8 aggregate tokens/s | C12 aggregate tokens/s |
|---|---:|---:|---:|---:|---:|
| English story | 146–147 | 85 / 88 | 369 / 352 | 450 / 443 | 603 / 594 |
| Python package | 172–173 | 136 / 131 | 585 / 616 | 698 / 729 | 839 / 865 |
| Math reasoning | 346–347 | 124 / 120 | 505 / 501 | 643 / 616 | 771 / 766 |
| Document summary 8K | 8,194 | 98 / 95 | 380 / 384 | 467 / 456 | 612 / 610 |
| Document summary 32K | 32,778 | 90 / 92 | 329 / 338 | 378 / 391 | 487 / 487 |

**MTP, adaptive up to three drafts:**

| Workload | Prompt tokens | C1 tokens/s | C6 aggregate tokens/s | C8 aggregate tokens/s | C12 aggregate tokens/s | C16 aggregate tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| English story | 146–147 | 70 / 71 | 314 / 310 | 392 / 392 | 526 / 522 | 620 / 608 |
| Python package | 172–173 | 98 / 98 | 454 / 468 | 560 / 569 | 773 / 778 | 924 / 910 |
| Math reasoning | 346–347 | 84 / 83 | 371 / 373 | 456 / 468 | 612 / 625 | 742 / 725 |
| Document summary 8K | 8,194 | 87 / 83 | 369 / 369 | 451 / 441 | 596 / 590 | 694 / 663 |
| Document summary 32K | 32,778 | 80 / 83 | 315 / 317 | 364 / 374 | 464 / 464 | 501 / 503 |

Prompt content and sampling affect draft acceptance and speed. Answer quality is not scored. The prefill table uses a separate greedy token-corpus workload. See `docs/performance.md` for methodology and `bench/results/readme-workloads-kv-cache-20261116.json` for per-cell results and acceptance.

The stored weight recipe’s original admission cells covered standard 8K/32K prefill and ordinary decode. They measured smaller prediction-loss increases over BF16 than the fixed 5090 NVFP4 reference. The updated verification partitioning changes floating-point rounding. Its separate NLL measurements and 103/103 exact retrieval answers are recorded in `docs/performance.md`. Historical standard and multikey needle retrieval passed through 240K; multikey also passed at 260K. These measurements cover prediction loss and retrieval. Test conditions and limitations are documented with the results.

## Build

On Ubuntu 24.04, after installing the supported ROCm 10 toolchain:

```sh
sudo apt-get install build-essential cmake ninja-build git patch pkg-config python3 curl \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev libzstd-dev

git clone https://github.com/jmander11/ninfer-amd-r9700.git
cd ninfer-amd-r9700
```

The build requires 64-bit Linux, the R9700, ROCm 10 with `gfx1201` and rocprofiler SDK ROCTx, CMake ≥3.28, Ninja, C++20, Python 3, pkg-config, and FFmpeg, libcurl and zstd development libraries. CMake fetches a pinned XGrammar dependency. Python conversion/reference tools additionally need Python 3.11, PyTorch and safetensors.

```sh
cmake -S . -B build-r9700 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-r9700 --parallel 8 \
  --target ninfer ninfer-serve ninfer-ppl ninfer_bench
```

Only `gfx1201` is supported; Ninja compile/link concurrency is capped at 8 jobs. Model artifacts are supplied separately. **A public model download is not yet available.** The commands below require the selected converted artifact; `docs/model-artifact.md` records its identity, checksum and conversion references.

## Run

Use the selected DFlash-enabled production `.ninfer` artifact:

```sh
build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --prompt "Explain wave32 matrix instructions." \
  --max-context 8192 --max-new 256

build-r9700/apps/ninfer-serve /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --max-concurrency 6 --kv-ram-capacity 8192
```

CLI and server default to DFlash with the optimized draft head and adaptive drafting up to seven tokens. `--spec none` disables speculation; `--no-lm-head-draft` selects the full proposal head. Server KV capacity defaults to `auto`. Use `--draft-tokens N --no-adaptive-draft` for a fixed length.

For C13..16 use `--spec mtp --draft-tokens 3 --adaptive-draft --lm-head-draft`, or `--spec none`.

P-less sampling uses temperature **1.5** by default; `--temperature` overrides it. It ignores top-p/top-k/min-p and penalties; `--no-p-less-sampling` enables those controls. Use `--greedy` for argmax. Vision and speculation settings are fixed at startup. Each executable's `--help` lists all options.

To check the server from a second terminal:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128}'
```

For a localhost container, copy `.env.example` to `.env` and set your model, ROCm and self-contained Python paths, then:

```sh
mkdir -p .cache/ninfer
docker compose -f compose.yaml -f compose.local.yaml up -d --wait
```

This publishes `127.0.0.1:8080` and requires Compose ≥2.24.4. Defaults: C6, adaptive DFlash up to K7, temperature 1.5, 262,144-token context, 8 GiB RAM prefix cache and 25 GiB persistent disk cache. `docs/containers.md` covers prerequisites and the separate external-proxy deployment.

## Documentation

- `docs/cli.md` — input, sampling, media, speculation and prefix storage.
- `docs/serving.md` — HTTP schemas, tools, constraints, scoring, state and metrics.
- `docs/containers.md` — container prerequisites, deployment and verification.
- `docs/performance.md` — capacity, benchmarks, quality gates and limitations.
- `docs/README.md` — architecture, artifact formats, conversion and kernel development.
