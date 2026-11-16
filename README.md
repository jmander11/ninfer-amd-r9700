# NInfer AMD — Radeon AI PRO R9700

C++/HIP inference for **Qwen3.8-27B on one AMD Radeon AI PRO R9700**, with **1–16 concurrent requests** (1–12 with DFlash2).

An AMD port of [Neroued/NInfer](https://github.com/Neroued/ninfer), distributed under Apache-2.0. Model weights are separate; see `docs/acknowledgements.md` for credits.

## What it does

- **Works with chat clients.** Run from the terminal or connect a client that uses OpenAI Chat Completions, Responses or Anthropic Messages. Answers stream as they are generated, with reasoning kept separate from the answer.
- **Calls tools and returns structured answers.** Let the model choose a tool or require a specific one, with incremental tool-call streams. Constrain answers to JSON, a JSON Schema, an EBNF grammar, a full-match regex or literal choices.
- **Reads images and video.** Add `--vision` at startup to accept media alongside text.
- **Speculative decoding.** DFlash2 proposes tokens for the main model to check together, adapting up to seven drafts. It is the default for up to twelve requests; MTP supports up to sixteen.
- **Reuses conversation history.** Cached prefixes avoid processing the same text again. Keep them in GPU memory, add a RAM cache, or save them to SSD for reuse after a restart.
- **Controls generation and scores text.** Adjust sampling, choose greedy decoding, inspect token probabilities or compare candidate answers. `ninfer-ppl` measures how well the model predicts text.
- **Reports server activity.** Prometheus metrics, JSON statistics and request logs report throughput, memory use, cache hits and draft acceptance.

## Built for the R9700

HIP kernels and Device Graphs are tuned for `gfx1201` (wave32). The engine batches up to sixteen requests with MTP or without speculation, and up to twelve with DFlash, where it can process a new prompt while other requests generate. GPU memory is planned at startup, with workspace shared between execution phases.

**LQ4: 4-bit codebook weights decoded to FP8 for computation.** Most Text projections and the output head use LQ4; 21 sensitive projections stay in FP8. The DFlash-enabled artifact is **16.75 GB** on disk (decimal; this is not a VRAM requirement), with identity `qwen3.8-27b/r9700-lq4`. The default paged KV cache uses NVFP4 E2M1 keys and values with E4M3 scales per 16 features. FP8 keys with INT4 values and FP16 scales remain opt-in (`--kv-cache fp8-int4`), as does TQ-K4-V3 (`--kv-cache tq-k4-v3`). DFlash keeps its own BF16 state.

For image/video inference, select the **17.42 GB** `qwen3.8-27b/r9700-lq4-vision-bf16` artifact and add `--vision`. It keeps the admitted Text weights byte-exact and uses original BF16 Vision matrices, adding about 640 MiB to storage and Vision-enabled weight memory. Vision-off loading has the same memory plan. Conversion and accuracy/performance evidence are in `docs/maintainer/qwen3.8-27b-artifact.md`.

This port combines NInfer features with AMD-specific kernels, weight/cache formats and memory tuning. Port decisions: `docs/maintainer/upstream-sync.md`.

## Context and capacity

Each request can use up to **262,144 tokens** for prompt and answer combined, subject to the shared GPU cache capacity below.

Shared cache capacity with automatic sizing, adaptive drafting (DFlash up to seven drafts, MTP up to three), optimized proposal heads, chunk 2048, Vision off, RAM/SSD off and Compose graph settings (HIP 7.15.26333), measured 2026-11-16 at the default 32 MiB headroom. NVFP4 (E2M1 keys and values with E4M3 scales per 16 features, 288 bytes per head-token) is the production default. The opt-in `--kv-cache tq-k4-v3` cache stores Hadamard-rotated 4-bit keys and 3-bit values in 228 bytes per head-token, about 1.26x as many tokens once shared capacity rather than the 262,144-token request ceiling is the limit:

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

TQ costs some quality and a little speed. Against NVFP4 on the engine PPL cells it raises perplexity 0.5% / 0.3% (prefill 8K / 32K) and 0.6% / 0.2% (decode 8K / 32K); in the cache-only comparison with BF16 it adds 0.59% (32K WikiText) and 1.13% (64K natural text) where NVFP4 adds 0.18% and 0.30%. Needle retrieval on a TQ server passed 59 of 60 cases from 8K to 260K (the miss, a multi-value needle at the start of a 260K prompt, also misses on the 5090 NVFP4 build). NVFP4 passes the BF16-source PPL gate (no worse than the 5090 NVFP4 build on every cell) and 108/108 needle retrievals from 8K to 260K. `--kv-cache fp8-int4` (FP8 keys, INT4 values) remains selectable at about 0.69x NVFP4 capacity; its tables are in `docs/performance.md`.

Capacity depends on available VRAM and enabled features; automatic sizing leaves 32 MiB headroom by default (raise `--kv-capacity-headroom` when a desktop or another process shares the GPU). These are allocated shared capacities, checked at startup and with one short request per lane; every cell kept the headroom physically free, although some repeated startups used up to about 20 MiB more device memory (see `docs/performance.md`). Filling the lanes does not raise device memory use: NVFP4 DFlash C4 with 4 x 222.6K-token prompts (98.6% of capacity) and TQ DFlash C12 with 12 x 82K-token prompts (97.2%) kept the same physical free memory from startup through prefill and decode. RAM can separately back suspended active continuations under KV pressure. Test conditions and settings: `docs/performance.md`; startup summaries for every cell: `bench/results/shared-kv-20261116.json`.

## Performance and quality

One R9700, Release build, Device Graphs, power `auto`, chunk 2048 and optimized proposal heads, measured 2026-11-16.

**Prompt processing (prefill), one request (NVFP4 / TQ cache):**

| Prompt tokens | Average tokens/s |
|---:|---:|
| 8,192 | 3,456 / 3,444 |
| 32,768 | 2,959 / 2,953 |
| 65,536 | 2,506 / 2,486 |
| 131,072 | 1,923 / 1,897 |
| 245,760 | 1,364 / 1,339 |

**DFlash answer generation with the default sampling settings (NVFP4 / TQ cache):**

| Workload | Prompt tokens | C1 tokens/s | C6 aggregate tokens/s | C8 aggregate tokens/s | C12 aggregate tokens/s |
|---|---:|---:|---:|---:|---:|
| English story | 146–147 | 85 / 88 | 369 / 352 | 450 / 443 | 603 / 594 |
| Python package | 172–173 | 136 / 131 | 585 / 616 | 698 / 729 | 839 / 865 |
| Math reasoning | 346–347 | 124 / 120 | 505 / 501 | 643 / 616 | 771 / 766 |
| Document summary 8K | 8,194 | 98 / 95 | 380 / 384 | 467 / 456 | 612 / 610 |
| Document summary 32K | 32,778 | 90 / 92 | 329 / 338 | 378 / 391 | 487 / 487 |

P-less temperature 1.5, adaptive DFlash up to seven drafts, 512 output tokens per request (including reasoning). Means of three measured waves after one warmup; C1 uses twelve waves per cache.

**MTP answer generation (NVFP4 / TQ cache):**

| Workload | Prompt tokens | C1 tokens/s | C6 aggregate tokens/s | C8 aggregate tokens/s | C12 aggregate tokens/s | C16 aggregate tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| English story | 146–147 | 70 / 71 | 314 / 310 | 392 / 392 | 526 / 522 | 620 / 608 |
| Python package | 172–173 | 98 / 98 | 454 / 468 | 560 / 569 | 773 / 778 | 924 / 910 |
| Math reasoning | 346–347 | 84 / 83 | 371 / 373 | 456 / 468 | 612 / 625 | 742 / 725 |
| Document summary 8K | 8,194 | 87 / 83 | 369 / 369 | 451 / 441 | 596 / 590 | 694 / 663 |
| Document summary 32K | 32,778 | 80 / 83 | 315 / 317 | 364 / 374 | 464 / 464 | 501 / 503 |

Adaptive MTP up to three drafts, otherwise the same settings with three measured waves per cell.

Both tables were measured 2026-11-16 on one build for both caches, with a 65,536-token per-request ceiling, automatic shared KV sizing, and RAM/SSD and Vision off. Rates exclude prompt processing: each prompt is cached before timing, and C>1 counts only intervals in which every request is decoding. Divide an aggregate rate by concurrency for the average per active request. TQ throughput matches NVFP4 within sampling noise (geometric mean 0.999 for DFlash and for MTP; single cells 0.95–1.05), and pooled first-draft acceptance is equal (0.75–0.77 for both caches at every concurrency). Prompt content and sampling affect draft acceptance and speed, and answer quality is not scored. Prefill above uses a separate greedy token-corpus workload. Methodology and variation: `docs/performance.md`. Machine-readable results, including per-cell acceptance: `bench/results/readme-workloads-kv-cache-20261116.json`.

The stored weight recipe’s original standard 8K/32K prefill and ordinary-decode admission cells measured smaller prediction-loss increases over BF16 than the fixed 5090 NVFP4 reference. The updated verification partitioning changes floating-point rounding; its separate NLL measurements and 103/103 exact retrieval answers are recorded in `docs/performance.md`. Historical standard and multikey needle retrieval passed through 240K; multikey also passed at 260K. These are prediction-loss and retrieval measurements; test conditions and limitations are documented with the results.

## Build

On Ubuntu 24.04, after installing the supported ROCm 10 toolchain:

```sh
sudo apt-get install build-essential cmake ninja-build git patch pkg-config python3 curl \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev libzstd-dev

git clone https://github.com/jmander11/ninfer-amd-r9700.git
cd ninfer-amd-r9700
```

Requires 64-bit Linux, the R9700, ROCm 10 with `gfx1201` and rocprofiler SDK ROCTx, CMake ≥3.28, Ninja, C++20, Python 3, pkg-config, and FFmpeg, libcurl and zstd development libraries. CMake fetches a pinned XGrammar dependency. Python conversion/reference tools additionally need Python 3.11, PyTorch and safetensors.

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
  --max-context 262144 --max-concurrency 6
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

This publishes `127.0.0.1:8080` and needs Compose ≥2.24.4. Defaults: C6, adaptive DFlash up to K7, temperature 1.5, 262,144-token context, 8 GiB RAM prefix cache and 25 GiB persistent disk cache. `docs/containers.md` covers prerequisites and the separate external-proxy deployment.

## Documentation

- `docs/cli.md` — input, sampling, media, speculation and prefix storage.
- `docs/serving.md` — HTTP schemas, tools, constraints, scoring, state and metrics.
- `docs/containers.md` — container prerequisites, deployment and verification.
- `docs/performance.md` — capacity, benchmarks, quality gates and limitations.
- `docs/README.md` — architecture, artifact formats, conversion and kernel development.
