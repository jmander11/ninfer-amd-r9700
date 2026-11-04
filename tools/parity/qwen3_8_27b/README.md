# Qwen3.8-27B R9700 Vision parity tool

The diagnostic captures 471 represented boundaries per media item: preprocessing, patch projection, all 27 Vision blocks and the merger. It compares HIP with the artifact-native implementation reference, reruns operations from captured inputs to isolate local error, checks MRoPE metadata, and compares 333 Vision tensors and native tower outputs directly with the source BF16 model. Each local Linear oracle evaluates the complete dot in FP64 from represented BF16 inputs and exact BF16 weights or stored signed codes times scales; it does not reproduce private activation quantization or round decoded weights to BF16. The local attention oracle evaluates segmented softmax(Q Kᵀ / √D)V in FP64, retaining FP64 output. CPU schedule references use the same mathematical attention followed by their explicit BF16 boundary; library BF16 SDPA remains a supplementary implementation comparison through the source Hugging Face final check. Use the existing AMD artifact binder; NVFP4 artifacts are unsupported.

Run with an explicitly selected artifact and existing Python 3.11 reference environment:

```bash
cmake --build build-r9700 --target ninfer_qwen3_8_27b_vision_trace
source .ninfer-local.env
flock --exclusive "$NINFER_GPU_LOCK" "$NINFER_PYTHON" -m tools.parity.qwen3_8_27b.vision \
  --weights /absolute/path/to/selected-r9700.ninfer \
  --model-dir /path/to/models/qwen3.8-27b-bf16 \
  --messages examples/cli/messages/image_chart.json \
  --trace-exe build-r9700/tests/ninfer_qwen3_8_27b_vision_trace \
  --device cpu --output /absolute/path/to/new-vision-report.json
```

The selected-artifact command planner in `tools/bench/prepare_selected_vision_diagnostic.py` requires this tracer in the selected build's `tests/` directory. Its schema-2 plan records and validates the executable, artifact, source receipt, prepared input and diagnostic implementation. The completion validator requires all 471 captures in each of the three comparison profiles and reevaluates their numerical gates from the metrics; historical four-capture reports are not accepted or reinterpreted.

The native tracer requires sole R9700 ownership; `--device` selects only the Python reference device. The existing environment must provide torch, safetensors and Transformers; nothing is downloaded.

`--trace-dir` consumes an existing manifest; otherwise native traces are temporary. `--prepared-input` reuses a previously frozen single-image reference input, still checking its metadata against the native trace. The tracer checks exact traced/untraced output and aggregate and per-item agreement.

The version-3 report names the direct native/source comparison `production_vs_source_bf16`. Final embeddings must have relative L2 error at most 5% and cosine at least 0.999 against the BF16 source. This engineering admission target supplements the unchanged source-weight row/column/group guards and per-Op checks; it does not establish visual task accuracy by itself. The prior 25%/0.97 final gate and the artifact-reference/source substitution are superseded. A failed diagnostic remains a failed diagnostic even when final embeddings meet this target.

Reports preserve source provenance, finite shapes, whole-tensor, worst-token, worst-feature and stored-group metrics with explicit local and accumulated criteria. Reports never overwrite existing files. These diagnostics do not replace independent Op or model-quality qualification.

Metric fault tests check that dropped tokens, features and quantization groups cannot hide in aggregate averages: `python3.11 -m pytest tools/parity/qwen3_8_27b/test_vision.py tools/parity/qwen3_8_27b/test_vision_parity_metrics.py`.

## Generation and sampling diagnostics

Build `ninfer_qwen3_8_27b_generation_trace` with `NINFER_BUILD_BENCHMARKS=ON`. The public-Engine diagnostic accepts `ARTIFACT MANIFEST.json NEW_OUTPUT_DIR`; its executable is under `build-r9700/bench/`. The manifest has an `engine` object with `max_context`, `max_concurrency`, `prefill_chunk`, `backend` (`dflash` or `none`), `draft_tokens`, `adaptive_draft`, `device_graph`, `recovery`, and optional `draft_temperature`. Its `requests` array contains objects with `label`, `path` (the frozen request JSON), and optional `max_tokens`, `p_less`, `temperature`, and integer `seed` overrides. This saved-fixture adapter supports text-only messages, top-level `enable_thinking`, `chat_template_kwargs.preserve_thinking`, and function tools with auto choice and explicit parameter objects. It inserts `strict: false` when the function field is absent or null, matching HTTP tool normalization before template rendering; the frozen request file remains unchanged. It does not claim general HTTP-schema equivalence, including server system prepends or other template options. Check the prepared prompt IDs before interpreting a replay comparison.

For an exact token-prefix probe, replace a request's `path` with `token_ids_path`, pointing to a nonempty JSON array of vocabulary IDs. This calls `Engine::prepare_tokens` directly, without message/template reconstruction; it makes no HTTP prompt-equivalence claim. Set `raw: true` and `use_model_stops: false` to inspect an unrestricted next-token ranking at that prefix, with `max_tokens: 1`, `temperature: 0`, and the desired `p_less`/`seed` settings. `raw` defaults to false and `use_model_stops` to true on either route. Prefix reuse and cache writes remain disabled. Raw output bypasses structured channel interpretation and generation recovery; it does not force a token or install an eligibility mask.

The diagnostic preserves structured output by default and records prompt/generated IDs, resolved sampling, raw target T=1 top-20 logprobs, content, reasoning, tool calls, errors, and recovery attempts. Stop-token publication and special-token preservation expose terminal IDs without requiring raw output. Raw target logprobs are not the actual temperature-scaled, masked p-less probabilities.

Set `NINFER_SAMPLING_CAPTURE_PREFIX` to a new path prefix to capture actual sampling inputs for rounds that license a watched token. The default watch list is `248046,248044`; `NINFER_SAMPLING_CAPTURE_TOKENS` accepts an explicit comma-separated list of nonnegative IDs. This diagnostic requires eager execution (`device_graph: false`) and supports ordinary sampling and chain DFlash verification; tree verification is rejected. It synchronously reads device state and is not performance evidence or a reproduction of a historical Device Graph execution. The existing `NINFER_DFLASH_DECISION_TRACE_OUT` top-two trace remains independently available; either diagnostic can be enabled alone.

Each watched round writes `PREFIX-N.json`, `PREFIX-N.bf16`, and, when a domain mask exists, `PREFIX-N.mask.u32`. Files must not already exist. The manifest is written last. BF16 little-endian rows contain only the logical vocabulary; the manifest records the source physical row count and pitch. Mask words are captured from the actual sampling configuration before round masks are released, with each live column materialized even when the source mask is broadcast. The manifest also records suppressed IDs, first-column exclusion, exact uint64 seed, positions, licensed IDs, and, for DFlash, all live target columns, drafts, represented selector probabilities, and accepted count. A watched speculative round can contain licensed tokens beyond the public output cutoff: correlate it with the Engine's published IDs and terminal result before attributing a stop.

Run the independent CPU verifier with the selected Python 3.11 interpreter:

```bash
"$NINFER_PYTHON" -m tools.parity.qwen3_8_27b.sampling_capture \
  /absolute/path/to/PREFIX-0.json --output /absolute/path/to/new-oracle-report.json
```

The verifier evaluates the complete eligible-domain softmax and collision probability in FP64, applies the p-less floor, exclusion and Dirac fallback, and checks Algorithm 2 against the exact counter RNG. It reports each watched ID's actual target sampling probability separately from its raw unmasked T=1 logprob, as well as support, acceptance and CDF margins. Greedy captures are supported; positive-temperature non-p-less and nonzero-penalty captures are explicitly rejected (token counts are not captured). Deterministic CDF comparison targets the full-vocabulary product route; small-vocabulary single-workgroup CDF ordering is not a claimed supported replay profile.

FP32 transcendental/reduction/CDF association can move boundary decisions, so an oracle mismatch alone is not proof of a kernel defect. Inspect the reported margins and eligibility before making that claim. A matched selection establishes that captured eager decision, not that a model-selected EOS is semantically appropriate. Statistical tests with frozen draft tokens and varying verifier RNG do not establish the unconditional target distribution. CPU regression checks are in `test_sampling_capture.py`; the GPU empty-support transition regression belongs to `ninfer_r9700_speculative_round_qual`.

The generation-trace manifest accepts `stream: true` per request to capture native published deltas with channel, timestamp, original logprobs and typed tool events. Result records also include recovery events and cumulative framing-repair/discarded-content counters. These diagnostic captures are not throughput benchmarks or external task-success evidence.
