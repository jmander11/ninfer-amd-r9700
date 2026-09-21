# Perplexity: qwen3.8-27b

- weight inputs: {'bf16-reference': '/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16'}
- candidate artifact: None
- lengths: [8192, 32768]
- skip default: half
- device graphs: on unless a cell sets device_graph=false
- terrible token: nll >= 10.0
- independent baseline: `bf16-reference` per (length, schedule, spec)
- decode spec: none (draft 0) unless a cell sets spec=none; prefill lane is spec-free

| length | schedule | scheme | spec | graph | skip | scored | greedy exact (diagnostic) | flips | flip rate | mean_nll | max_nll | severe | severe Δ | new / budget | ppl | Δ mean_nll | Δabs p50 | p95 | p99 | max | Δ 1σ | σ nll | noise | quality gate |
|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 8192 | prefill | `bf16-reference` | none | off | 4096 | 4095 | yes | 0 | 0.000000 | 1.866231 | 15.1419 | 46 | 0 | - / - | 6.4639 | 0.000000 | - | - | - | - | - | 2.345 |  | reference |
| 32768 | prefill | `bf16-reference` | none | off | 16384 | 16383 | yes | 0 | 0.000000 | 1.728555 | 19.1647 | 152 | 0 | - / - | 5.6325 | 0.000000 | - | - | - | - | - | 2.245 |  | reference |

_Greedy exactness and flip rate against BF16 are diagnostics, not quality gates. Prefill/decode schedule flips are also diagnostic because the selected attention routes may use different qualified private precision; their finite aligned NLL sidecars must meet the explicitly supplied bound. Graph/eager, MTP/ordinary, and draft-window execution variants remain exact-token comparisons._

_Noise columns: `σ nll` is the per-token NLL std for the cell; `Δ 1σ` is the SE of the per-token paired Δnll vs the independent BF16 reference (index-aligned tokens, from the .nllf32 sidecars). `noise` marks |Δ| ≤ 2·Δ1σ — the delta is not resolved above the per-token noise floor._
