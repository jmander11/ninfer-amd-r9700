# NInfer maintainer tools

`tools/` owns conversion, inspection, numerical qualification, benchmarking, and serving smoke
workflows for the sole Qwen3.8-27B R9700 product.

## Artifact conversion

The current converter starts from the complete official BF16 checkpoint and writes the provisional
W8G32 evaluation artifact. It validates all 1,118 tensor objects and six frontend resources before
opening the output. This identity remains provisional until real-model PPL, exact-token, and
whole-inference gates choose the final integer recipe.

```bash
python3 -m tools.convert.qwen3_8_27b_r9700.convert \
  --model /path/to/Qwen3.8-27B-BF16 \
  --draft-ranking /path/to/qwen3.8-draft-ranking.i64 \
  --out out/qwen3_8_27b_r9700_candidate.ninfer \
  --device cpu

python3 -m tools.artifact.inspect \
  out/qwen3_8_27b_r9700_candidate.ninfer --objects
```

The complete BF16 source, a Qwen3.8-derived draft-token frequency ranking, and a Python 3.11
environment with Torch and safetensors are local maintainer prerequisites. No retired-model
ranking is bundled or accepted implicitly. Conversion never uses a prior quantized artifact as
its source. The ranking is a row-major little-endian I64 matrix with 248,320 columns and the total
frequency vector in its first row; the conversion report records its resolved path, size, and
SHA-256.

## Python reference and parity

`tools/reference/qwen3_8_27b` is the independent artifact-native Text, Vision, and MTP diagnostic
over the exact 1,124-object candidate binding. It evaluates the represented packed weights and the
fixed FP8-E4M3FN-K/INT4-G16-V/FP16-scale cache; it is not the source-BF16 authority. The
checkpoint-direct Text authority is `tools/reference/qwen3_8_27b_bf16`, while the source-BF16
Vision diagnostic is `tools/parity/qwen3_8_27b`. Real-model numerical claims require the complete
BF16 checkpoint and converted candidate artifact.

## R9700 qualification

The gfx1201 workflow builds an owning HIP Op qualifier, checks it against an independent oracle,
then measures the same real model shape with unprofiled HIP events. ISA and resource checks inspect
the emitted gfx1201 code object.

```bash
make -C tools/r9700 build/eager_op_qual
tools/r9700/build/eager_op_qual
make -C tools/r9700 eager-isa
```

See tools/r9700/README.md, docs/maintainer/kernel-iteration.md, and bench/README.md.

## Performance evidence assembly

`bench/run_ninfer_bench_matrix.py` owns physical schema-v20 benchmark reports and schema-v14
matrix manifests. `bench/select_prefill_chunk.py` owns the twelve-candidate global chunk decision;
`ppl/assemble_pareto.py` plus `ppl/pareto.py` bind that authority into the schema-v7 base static-profile
decision. After that record fixes G16 or G32, `bench/assemble_dflash_selection.py` is the sole owner
of the schema-v2 DFlash K/W decision: it binds the shortlist, every frontier capacity campaign,
every eligible full DFlash matrix, exact exclusions, and generated-behavior gates into one retained
winner and full frontier. See bench/README.md for the copy-ready commands.

## Serving smoke

After starting `build-r9700/apps/ninfer-serve`, run the protocol smoke client against the local
endpoint. The resident server and every product application use the fixed FP8-K/INT4-V cache.

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.8-27b
```
