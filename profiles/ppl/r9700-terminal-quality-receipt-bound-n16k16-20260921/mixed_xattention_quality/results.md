# Perplexity: qwen3.8-27b

- weight inputs: {'bf16-reference': '/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16', 'r9700-g16': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer', 'r9700-g32': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer'}
- candidate artifact: {'path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer', 'bytes': 22881204736, 'sha256': 'b021b71f70790215ccd9a287195c69b94f7c5ebf9ac0b9bb90bd62c3f09b3bcc', 'model_id': 'qwen3.8-27b', 'weights_id': 'r9700-q4-w8-mse-n16k16-eval', 'conversion_receipt': {'path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer.conversion.json', 'sha256': '1298ba51b2e80c225663814771a706afcbbe014e03f0bdefd4aa7dbf1cc8551c', 'recipe_id': 'r9700-source-q4-n16k16-promoted-w8-source-mse8-eval-v1', 'object_plan_sha256': '14b80eb8a8200112169ef34bda4520d8b50233935bc9aa1c79f02b674ef36fb3', 'source_artifact_sha256': '8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd', 'source_receipt_sha256': '5d9a30ec76ce9f557195d554a7f8d71ad803fb4af5b115f2f887b7622de66d89', 'transcoder_sha256': 'e988d0ecc7d20a12728aa8313a71221eeea9c30dfff5d998020a826baab52801', 'receipt_producer_sha256': '3fb4f58e376e5c8dbd333f06ef146796410eb7829de4130c408d8adc5a615d64'}, 'g16_path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer', 'g32_path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer'}
- lengths: [8192, 32768]
- skip default: half
- device graphs: on unless a cell sets device_graph=false
- terrible token: nll >= 10.0
- independent baseline: `bf16-reference` per (length, schedule, spec)
- decode spec: none (draft 0) unless a cell sets spec=none; prefill lane is spec-free

| length | schedule | scheme | spec | graph | skip | scored | greedy exact (diagnostic) | flips | flip rate | mean_nll | max_nll | severe | severe Δ | new / budget | ppl | Δ mean_nll | Δabs p50 | p95 | p99 | max | Δ 1σ | σ nll | noise | quality gate |
|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 8192 | prefill | `bf16-reference` | none | off | 4096 | 4095 | yes | 0 | 0.000000 | 1.866231 | 15.1419 | 46 | 0 | - / - | 6.4639 | 0.000000 | - | - | - | - | - | 2.345 |  | reference |
| 8192 | prefill | `r9700-g16` | none | on | 4096 | 4095 | NO | 266 | 0.064957 | 1.878788 | 14.9698 | 44 | -2 | 3 / 5 | 6.5456 | 0.012557 | 0.054270 | 0.448853 | 0.754741 | 2.350851 | 0.003371 | 2.361 |  | PASS |
| 8192 | prefill | `r9700-g32` | none | on | 4096 | 4095 | NO | 262 | 0.063980 | 1.879787 | 15.0484 | 45 | -1 | 5 / 5 | 6.5521 | 0.013556 | 0.051758 | 0.460624 | 0.825662 | 2.993017 | 0.003517 | 2.366 |  | PASS |
| 32768 | prefill | `bf16-reference` | none | off | 16384 | 16383 | yes | 0 | 0.000000 | 1.728555 | 19.1647 | 152 | 0 | - / - | 5.6325 | 0.000000 | - | - | - | - | - | 2.245 |  | reference |
| 32768 | prefill | `r9700-g16` | none | on | 16384 | 16383 | NO | 1039 | 0.063419 | 1.745616 | 19.4592 | 160 | 8 | 19 / 17 | 5.7294 | 0.017060 | 0.052807 | 0.486198 | 0.908323 | 4.713537 | 0.001848 | 2.275 |  | FAIL |
| 32768 | prefill | `r9700-g32` | none | on | 16384 | 16383 | NO | 1044 | 0.063725 | 1.746204 | 19.5374 | 158 | 6 | 18 / 17 | 5.7328 | 0.017649 | 0.052433 | 0.490411 | 0.911727 | 4.517784 | 0.001862 | 2.276 |  | FAIL |

_Greedy exactness and flip rate against BF16 are diagnostics, not quality gates. Prefill/decode schedule flips are also diagnostic because the selected attention routes may use different qualified private precision; their finite aligned NLL sidecars must meet the explicitly supplied bound. Graph/eager, MTP/ordinary, and draft-window execution variants remain exact-token comparisons._

_Noise columns: `σ nll` is the per-token NLL std for the cell; `Δ 1σ` is the SE of the per-token paired Δnll vs the independent BF16 reference (index-aligned tokens, from the .nllf32 sidecars). `noise` marks |Δ| ≤ 2·Δ1σ — the delta is not resolved above the per-token noise floor._
