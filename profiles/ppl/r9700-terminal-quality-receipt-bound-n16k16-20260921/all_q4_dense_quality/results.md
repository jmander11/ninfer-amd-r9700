# Perplexity: qwen3.8-27b

- weight inputs: {'bf16-reference': '/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16', 'r9700-g16': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer', 'r9700-g32': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer'}
- candidate artifact: {'path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer', 'bytes': 15172833280, 'sha256': '60719c5d5bfe978376c7de94db460bcd82d965ec447870cc47f0a02bf8d59953', 'model_id': 'qwen3.8-27b', 'weights_id': 'r9700-q4g64-n16k16-eval', 'conversion_receipt': {'path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer.conversion.json', 'sha256': 'a8567a6b25176aac1f2106bcac3131b74d887e1a5be2c98da23c38a3a7870e54', 'recipe_id': 'r9700-all-q4g64-n16k16-eval-v1', 'object_plan_sha256': 'bff624cbfda357d7c8b355530824682b1625c68c1f909ddb8b4241c2b3d21ec6', 'source_artifact_sha256': '19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9', 'source_receipt_sha256': 'aa6861cc31c7db6aa06335b111507f4802d702c6e83ef8fc96c0714b9dc477f9', 'transcoder_sha256': 'e988d0ecc7d20a12728aa8313a71221eeea9c30dfff5d998020a826baab52801', 'receipt_producer_sha256': '3fb4f58e376e5c8dbd333f06ef146796410eb7829de4130c408d8adc5a615d64'}, 'g16_path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer', 'g32_path': '/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer'}
- lengths: [8192, 32768]
- skip default: half
- device graphs: on unless a cell sets device_graph=false
- terrible token: nll >= 10.0
- independent baseline: `bf16-reference` per (length, schedule, spec)
- decode spec: none (draft 0) unless a cell sets spec=none; prefill lane is spec-free

| length | schedule | scheme | spec | graph | skip | scored | greedy exact (diagnostic) | flips | flip rate | mean_nll | max_nll | severe | severe Δ | new / budget | ppl | Δ mean_nll | Δabs p50 | p95 | p99 | max | Δ 1σ | σ nll | noise | quality gate |
|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 8192 | prefill | `bf16-reference` | none | off | 4096 | 4095 | yes | 0 | 0.000000 | 1.866231 | 15.1419 | 46 | 0 | - / - | 6.4639 | 0.000000 | - | - | - | - | - | 2.345 |  | reference |
| 8192 | prefill | `r9700-g16` | none | on | 4096 | 4095 | NO | 433 | 0.105739 | 1.905569 | 14.7764 | 45 | -1 | 7 / 11 | 6.7232 | 0.039338 | 0.090332 | 0.671484 | 1.182965 | 3.476297 | 0.005055 | 2.376 |  | PASS |
| 8192 | prefill | `r9700-g32` | none | on | 4096 | 4095 | NO | 439 | 0.107204 | 1.906206 | 14.7241 | 49 | 3 | 10 / 11 | 6.7275 | 0.039975 | 0.090796 | 0.662266 | 1.255538 | 3.533892 | 0.005119 | 2.379 |  | PASS |
| 32768 | prefill | `bf16-reference` | none | off | 16384 | 16383 | yes | 0 | 0.000000 | 1.728555 | 19.1647 | 152 | 0 | - / - | 5.6325 | 0.000000 | - | - | - | - | - | 2.245 |  | reference |
| 32768 | prefill | `r9700-g16` | none | on | 16384 | 16383 | NO | 1676 | 0.102301 | 1.773594 | 19.4604 | 163 | 11 | 36 / 41 | 5.8920 | 0.045038 | 0.089186 | 0.739880 | 1.382116 | 4.942131 | 0.002815 | 2.296 |  | PASS |
| 32768 | prefill | `r9700-g32` | none | on | 16384 | 16383 | NO | 1672 | 0.102057 | 1.774640 | 19.7413 | 165 | 13 | 35 / 41 | 5.8982 | 0.046085 | 0.090893 | 0.748249 | 1.382972 | 4.787981 | 0.002824 | 2.299 |  | PASS |

_Greedy exactness and flip rate against BF16 are diagnostics, not quality gates. Prefill/decode schedule flips are also diagnostic because the selected attention routes may use different qualified private precision; their finite aligned NLL sidecars must meet the explicitly supplied bound. Graph/eager, MTP/ordinary, and draft-window execution variants remain exact-token comparisons._

_Noise columns: `σ nll` is the per-token NLL std for the cell; `Δ 1σ` is the SE of the per-token paired Δnll vs the independent BF16 reference (index-aligned tokens, from the .nllf32 sidecars). `noise` marks |Δ| ≤ 2·Δ1σ — the delta is not resolved above the per-token noise floor._
