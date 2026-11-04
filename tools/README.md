# NInfer maintainer tools

Tools for converting and inspecting artifacts, checking numerical accuracy, benchmarking and testing the Qwen3.8-27B R9700 server.

## Artifact conversion

The base converter reads the complete official BF16 checkpoint and writes a W8G32 evaluation artifact. It validates all 1,118 tensors and six frontend resources before opening the output. This is a conversion and qualification base; the selected production recipe is LQ4. See `docs/maintainer/qwen3.8-27b-artifact.md` for production conversion.

```bash
python3 -m tools.convert.qwen3_8_27b_r9700.convert \
  --model /path/to/Qwen3.8-27B-BF16 \
  --draft-ranking /path/to/qwen3.8-draft-ranking.i64 \
  --out out/qwen3_8_27b_r9700_candidate.ninfer \
  --device cpu

python3 -m tools.artifact.inspect \
  out/qwen3_8_27b_r9700_candidate.ninfer --objects
```

The complete BF16 source, a Qwen3.8-derived draft-token frequency ranking, and a Python 3.11 environment with Torch and safetensors are local maintainer prerequisites. No retired-model ranking is bundled or accepted implicitly. Conversion never uses a prior quantized artifact as its source. The ranking is a row-major little-endian I64 matrix with 248,320 columns and the total frequency vector in its first row; the conversion report records its resolved path, size, and SHA-256.

## Python reference and parity

`tools/reference/qwen3_8_27b` diagnoses Text, Vision and MTP using the 1,124-object candidate artifact. It evaluates packed weights and the fixed FP8-E4M3FN-K/INT4-G16-V/FP16-scale cache.

For independent comparisons with the BF16 source, use `tools/reference/qwen3_8_27b_bf16` for Text and `tools/parity/qwen3_8_27b` for Vision. Real-model numerical checks require the complete BF16 checkpoint and converted artifact.

## R9700 qualification

Each gfx1201 Op qualifier compares the production kernel with an independent oracle, then measures real model shapes with unprofiled HIP events. ISA and resource checks inspect the emitted code object.

```bash
make -C tools/r9700 build/eager_op_qual
tools/r9700/build/eager_op_qual
make -C tools/r9700 eager-isa
```

See tools/r9700/README.md, docs/maintainer/kernel-iteration.md, and bench/README.md.

## Performance evidence assembly

The selection tools combine benchmark and quality results in four steps:

1. `bench/run_ninfer_bench_matrix.py` runs benchmarks and writes reports and matrix manifests.
2. `bench/select_prefill_chunk.py` chooses one global prefill chunk from twelve candidates.
3. `ppl/assemble_pareto.py` and `ppl/pareto.py` compare base profiles using that chunk and select G16 or G32.
4. `bench/assemble_dflash_selection.py` selects DFlash draft/verify lengths from the shortlist, capacity results, eligible speed matrices, token parity and generated-behavior checks. It records the winner, alternatives and exclusions.

Commands are in `bench/README.md`.

## Serving smoke

After starting `build-r9700/apps/ninfer-serve`, run the protocol smoke client against the local endpoint. The resident server and every product application use the fixed FP8-K/INT4-V cache.

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.8-27b
```
