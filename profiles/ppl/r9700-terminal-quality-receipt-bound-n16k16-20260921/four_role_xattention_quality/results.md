# Perplexity: qwen3.8-27b

- weight inputs: {'bf16-reference': '/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16', 'r9700-g16': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer', 'r9700-g32': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer'}
- candidate artifact: {'path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer', 'bytes': 21553549312, 'sha256': '040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2', 'model_id': 'qwen3.8-27b', 'weights_id': 'r9700-q4g64-f8e4m3-four-role-n16k16-eval', 'conversion_receipt': {'path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer.conversion.json', 'sha256': '2c5cb9f405ba581f685a96b1a2965ad3ff49b31280ebdd15d19158b6f4779c59', 'recipe_id': 'r9700-q4g64-f8e4m3-four-role-n16k16-eval-v1', 'object_plan_sha256': '82692559ecd3293e68b3ea3e19df8b93b690d046bb764a476bf59178bd637227', 'source_artifact_sha256': '1dfe9626fd6412592f87480a2f6934e8494a4b267693a25831dff490959542ce', 'source_receipt_sha256': 'a3bea3a6866b83bdb6ad9a382ca44b98ad009674c19a585d67ddb0f6a1ea9df7', 'transcoder_sha256': 'e988d0ecc7d20a12728aa8313a71221eeea9c30dfff5d998020a826baab52801', 'selection_sha256': 'b2ceeb63c581c0f26aab5a4d8c0958da34d836fcc5c47d377bce709eaf37e3e8', 'source_index_sha256': '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df', 'source_ranking_sha256': '205b2d6c5b58d946c87b425578da13b5cfc32a0abe1ff201e95b47d032bf004e'}, 'g16_path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer', 'g32_path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer'}
- lengths: [8192, 32768]
- skip default: half
- device graphs: on unless a cell sets device_graph=false
- terrible token: nll >= 10.0
- independent baseline: `bf16-reference` per (length, schedule, spec)
- decode spec: none (draft 0) unless a cell sets spec=none; prefill lane is spec-free

| length | schedule | scheme | spec | graph | skip | scored | greedy exact (diagnostic) | flips | flip rate | mean_nll | max_nll | severe | severe Δ | new / budget | ppl | Δ mean_nll | Δabs p50 | p95 | p99 | max | Δ 1σ | σ nll | noise | quality gate |
|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 8192 | prefill | `bf16-reference` | none | off | 4096 | 4095 | yes | 0 | 0.000000 | 1.866231 | 15.1419 | 46 | 0 | - / - | 6.4639 | 0.000000 | - | - | - | - | - | 2.345 |  | reference |
| 8192 | prefill | `r9700-g16` | none | on | 4096 | 4095 | NO | 412 | 0.100611 | 1.888789 | 15.0005 | 44 | -2 | 5 / 11 | 6.6114 | 0.022559 | 0.075674 | 0.613297 | 1.045879 | 4.132200 | 0.004480 | 2.366 |  | PASS |
| 8192 | prefill | `r9700-g32` | none | on | 4096 | 4095 | NO | 409 | 0.099878 | 1.890633 | 14.8970 | 42 | -4 | 5 / 11 | 6.6236 | 0.024402 | 0.073631 | 0.613704 | 1.123997 | 4.142681 | 0.004593 | 2.369 |  | PASS |
| 32768 | prefill | `bf16-reference` | none | off | 16384 | 16383 | yes | 0 | 0.000000 | 1.728555 | 19.1647 | 152 | 0 | - / - | 5.6325 | 0.000000 | - | - | - | - | - | 2.245 |  | reference |
| 32768 | prefill | `r9700-g16` | none | on | 16384 | 16383 | NO | 1486 | 0.090704 | 1.756447 | 19.9220 | 157 | 5 | 22 / 41 | 5.7918 | 0.027892 | 0.079826 | 0.626231 | 1.097333 | 5.072336 | 0.002327 | 2.278 |  | PASS |
| 32768 | prefill | `r9700-g32` | none | on | 16384 | 16383 | NO | 1487 | 0.090765 | 1.756392 | 20.2873 | 162 | 10 | 23 / 41 | 5.7915 | 0.027837 | 0.079374 | 0.640725 | 1.118008 | 5.259331 | 0.002355 | 2.280 |  | PASS |

_Greedy exactness and flip rate against BF16 are diagnostics, not quality gates. Prefill/decode schedule flips are also diagnostic because the selected attention routes may use different qualified private precision; their finite aligned NLL sidecars must meet the explicitly supplied bound. Graph/eager, MTP/ordinary, and draft-window execution variants remain exact-token comparisons._

_Noise columns: `σ nll` is the per-token NLL std for the cell; `Δ 1σ` is the SE of the per-token paired Δnll vs the independent BF16 reference (index-aligned tokens, from the .nllf32 sidecars). `noise` marks |Δ| ≤ 2·Δ1σ — the delta is not resolved above the per-token noise floor._
