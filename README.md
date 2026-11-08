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

**LQ4: 4-bit codebook weights decoded to FP8 for computation.** Most Text projections and the output head use LQ4; 21 sensitive projections stay in FP8. The DFlash-enabled artifact is **16.75 GB** on disk (decimal; this is not a VRAM requirement), with identity `qwen3.8-27b/r9700-lq4`. The paged KV cache stores FP8 keys, INT4 values and FP16 scales. DFlash keeps its own BF16 state.

For image/video inference, select the **17.42 GB** `qwen3.8-27b/r9700-lq4-vision-bf16` artifact and add `--vision`. It keeps the admitted Text weights byte-exact and uses original BF16 Vision matrices, adding about 640 MiB to storage and Vision-enabled weight memory. Vision-off loading has the same memory plan. Conversion and accuracy/performance evidence are in `docs/maintainer/qwen3.8-27b-artifact.md`.

This port combines NInfer features with AMD-specific kernels, weight/cache formats and memory tuning. Port decisions: `docs/maintainer/upstream-sync.md`.

## Context and capacity

Each request can use up to **262,144 tokens** for prompt and answer combined, subject to the shared GPU cache capacity below.

Shared cache capacity with automatic sizing, adaptive drafting (DFlash up to seven drafts, MTP up to three), optimized proposal heads, chunk 2048, Vision off, RAM/SSD off and Compose graph settings (HIP 7.15.26333), measured 2026-11-08 for both caches at the default 32 MiB headroom. The default cache is FP8 keys with INT4 values; the opt-in NVFP4 cache (`--kv-cache nvfp4`: E2M1 keys and values with E4M3 scales per 16 features) stores about 1.44x as many tokens:

| Concurrent requests (C) | DFlash K7, default | DFlash K7, NVFP4 | MTP K3, default | MTP K3, NVFP4 |
|---:|---:|---:|---:|---:|
| 1 | 262,144 | 262,144 | 262,144 | 262,144 |
| 2 | 524,288 | 524,288 | 524,288 | 524,288 |
| 3 | 633,536 | 786,432 | 647,424 | 786,432 |
| 4 | 625,408 | 902,656 | 639,936 | 923,776 |
| 5 | 617,280 | 890,944 | 632,576 | 913,088 |
| 6 | 609,088 | 879,232 | 625,024 | 902,336 |
| 7 | 600,960 | 867,520 | 617,536 | 891,520 |
| 8 | 592,768 | 855,808 | 610,048 | 880,768 |
| 9 | 583,168 | 842,368 | 601,600 | 868,992 |
| 10 | 574,400 | 829,696 | 594,112 | 858,240 |
| 11 | 565,632 | 817,024 | 586,688 | 847,424 |
| 12 | 556,800 | 804,288 | 579,136 | 836,544 |
| 13 | Unsupported | Unsupported | 571,648 | 825,792 |
| 14 | Unsupported | Unsupported | 564,224 | 815,040 |
| 15 | Unsupported | Unsupported | 556,736 | 804,224 |
| 16 | Unsupported | Unsupported | 549,312 | 793,472 |

NVFP4 passes the same BF16-source PPL gate as the default (no worse than the 5090 NVFP4 build on every cell) and 108/108 needle retrievals from 8K to 260K; its README workload throughput is on par with the default (geometric mean +0.8% DFlash, +1.2% MTP; 32K summaries +3–6%) with equal draft acceptance. Default remains `fp8-int4`; see `docs/performance.md`.

Capacity depends on available VRAM and enabled features; automatic sizing leaves 32 MiB headroom by default (raise `--kv-capacity-headroom` when a desktop or another process shares the GPU). These are allocated shared capacities, checked at startup and with short decode, rather than an all-lane near-capacity qualification; every cell kept the headroom physically free. RAM can separately back suspended active continuations under KV pressure. Test conditions and settings: `docs/performance.md`; startup summaries for every cell: `bench/results/shared-kv-20261108.json`.

## Performance and quality

One R9700, Release build, Device Graphs, power `auto`, chunk 2048 and optimized proposal heads. Prefill samples were measured 2026-10-17; generation dates are given below.

**Prompt processing (prefill), one request:**

| Prompt tokens | Average tokens/s |
|---:|---:|
| 8,192 | 3,375 |
| 32,768 | 2,903 |
| 65,536 | 2,477 |
| 131,072 | 1,907 |
| 245,760 | 1,354 |

**DFlash answer generation with the default sampling settings (default / NVFP4 cache):**

| Workload | Prompt tokens | C1 tokens/s | C6 aggregate tokens/s | C8 aggregate tokens/s | C12 aggregate tokens/s |
|---|---:|---:|---:|---:|---:|
| English story | 146–147 | 86 / 85 | 362 / 356 | 446 / 443 | 593 / 601 |
| Python package | 172–173 | 134 / 136 | 620 / 555 | 700 / 718 | 866 / 850 |
| Math reasoning | 346–347 | 122 / 125 | 510 / 528 | 619 / 648 | 768 / 758 |
| Document summary 8K | 8,194 | 90 / 92 | 365 / 383 | 452 / 450 | 582 / 587 |
| Document summary 32K | 32,779 | 91 / 90 | 313 / 324 | 367 / 379 | 459 / 479 |

P-less temperature 1.5, adaptive DFlash up to seven drafts, 512 output tokens per request (including reasoning). Means of three measured waves after one warmup; C1 uses twelve waves per cache.

**MTP answer generation (default / NVFP4 cache):**

| Workload | Prompt tokens | C1 tokens/s | C6 aggregate tokens/s | C8 aggregate tokens/s | C12 aggregate tokens/s | C16 aggregate tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| English story | 146–147 | 71 / 69 | 311 / 317 | 389 / 397 | 528 / 534 | 619 / 630 |
| Python package | 172–173 | 100 / 97 | 473 / 448 | 569 / 573 | 770 / 784 | 908 / 928 |
| Math reasoning | 346–347 | 86 / 85 | 374 / 373 | 459 / 464 | 611 / 616 | 731 / 731 |
| Document summary 8K | 8,194 | 81 / 78 | 337 / 345 | 408 / 417 | 530 / 540 | 615 / 627 |
| Document summary 32K | 32,779 | 81 / 84 | 289 / 305 | 339 / 357 | 418 / 443 | 469 / 490 |

Adaptive MTP up to three drafts, otherwise the same settings with three measured waves per cell.

Both tables were measured 2026-11-07 on one build for both caches, with a 65,536-token per-request ceiling, automatic shared KV sizing, and RAM/SSD and Vision off. Rates exclude prompt processing: each prompt is cached before timing, and C>1 counts only intervals in which every request is decoding. Divide an aggregate rate by concurrency for the average per active request. Across all cells NVFP4 throughput is 0.8% (DFlash) and 1.2% (MTP) higher by geometric mean; 32K summaries gain 3–6%. Pooled first-draft acceptance is equal within sampling error at every concurrency; the largest single-cell difference (DFlash C6 Python) follows that cell's sampled acceptance at temperature 1.5. Prompt content and sampling affect draft acceptance and speed, and answer quality is not scored. Prefill above uses a separate greedy token-corpus workload. Methodology and variation: `docs/performance.md`. Machine-readable results, including per-cell acceptance: `bench/results/readme-workloads-kv-cache-20261107.json`.

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
