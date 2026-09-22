# Qwen3.8-27B R9700 Vision parity tool

The diagnostic captures 471 represented boundaries per media item: preprocessing, patch
projection, all 27 Vision blocks and the merger. It compares HIP with the artifact-native
reference, reruns operations from captured inputs to isolate local error, checks MRoPE metadata,
and compares 333 Vision tensors and the full tower with the source BF16 model. The existing AMD
artifact binder remains authoritative; no NVFP4 artifact is admitted.

Run with an explicitly selected artifact and existing Python 3.11 reference environment:

```bash
cmake --build build-r9700 --target ninfer_qwen3_8_27b_vision_trace
python3.11 -m tools.parity.qwen3_8_27b.vision \
  --weights /absolute/path/to/selected-r9700.ninfer \
  --model-dir /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --messages examples/cli/messages/image_chart.json \
  --trace-exe build-r9700/tests/ninfer_qwen3_8_27b_vision_trace \
  --device cpu --output /absolute/path/to/new-vision-report.json
```

The selected-artifact command planner in `tools/bench/prepare_selected_vision_diagnostic.py`
requires this tracer in the selected build's `tests/` directory. Its schema-2 plan closes over
the executable, artifact, source receipt, prepared input, and diagnostic implementation. The
completion validator requires all 471 captures in each of the three comparison profiles and
reevaluates their numerical gates from the metrics; historical four-capture reports are not
accepted or reinterpreted.

The native tracer requires sole R9700 ownership; `--device` selects only the Python reference
device. The existing environment must provide torch, safetensors and Transformers; nothing is
downloaded. `--trace-dir` consumes an existing manifest; otherwise native traces and reference
activation dumps are temporary. `--prepared-input` reuses a previously frozen single-image
reference input, still checking its metadata against the native trace. The tracer checks exact traced/untraced output and aggregate/
per-item agreement. Reports preserve source provenance, finite shapes, whole-tensor, worst-token,
worst-feature and stored-group metrics with explicit local and accumulated criteria. Publication
is create-only. These diagnostics do not replace independent Op or model-quality qualification.

Metric fault tests check that dropped tokens, features and quantization groups cannot hide in
aggregate averages: `python3.11 -m pytest tools/parity/qwen3_8_27b/test_vision.py
tools/parity/qwen3_8_27b/test_vision_parity_metrics.py`.
