# Tests

The CTest graph is the native Radeon AI PRO R9700 product graph. It links the same
`ninfer_r9700_core`, `ninfer_engine`, and `ninfer_serve` targets as the applications and registers
the production-owned `tools/r9700` qualification executables. It does not compile an alternate
device backend or duplicate production kernels inside tests.

## Organization

- `ops/test_fp8_int4_kv_oracle.cpp` is the independent exact host authority for FP8 E4M3FN keys,
  signed INT4 values, and FP16 value scales.
- `artifact/` covers generic container framing and the supported direct and row-split layouts.
- `targets/qwen3/` covers family-owned runtime mechanisms used by Qwen3.8-27B without
  requiring checkpoint resources.
- protocol tests cover the advertised OpenAI, Anthropic, Responses, request-log, and server-option
  behavior.
- `tools/r9700/*_qual` owns physical gfx1201 operator, cache, transaction, runtime, and Engine
  qualification against independent exact or FP64 oracles.

Old per-operator device fixtures are intentionally absent. They staged a second implementation
graph and no longer represented the kernels linked into the sole product. Real-model token,
perplexity, and whole-inference performance gates live in the R9700 qualification workflow and
require the complete BF16-derived product artifact.

## Build and run

```bash
cmake -S . -B build-r9700 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DBUILD_TESTING=ON
cmake --build build-r9700 --parallel
ctest --test-dir build-r9700 --output-on-failure
```

The tests expect the selected physical gfx1201 device. Absence of the supported R9700 is a test
failure, not a request to select another backend. The self-contained artifact tests create only
small temporary fixtures and need no checkpoint.

Run only the physical qualification set with:

```bash
ctest --test-dir build-r9700 -L r9700 --output-on-failure
```

Run the retained Python container and report-consumer contracts with an explicitly selected Python
interpreter that provides pytest, Torch, and safetensors. Fail the dependency check before starting
the suite rather than falling back to the unqualified system interpreter:

```bash
TEST_PYTHON=/absolute/path/to/python-with-pytest-torch-and-safetensors
"${TEST_PYTHON}" -c 'import pytest, safetensors, torch'
NINFER_RUN_R9700_CODEC_TESTS=0 "${TEST_PYTHON}" -m pytest \
  tests/artifact \
  tests/test_bench_matrix.py \
  tests/test_serve_corpus.py \
  tools/convert/qwen3_8_27b_r9700/test_codec.py \
  tools/convert/qwen3_8_27b_r9700/test_vectorized_codec.py
```

The Python codec tests are independent converter/storage checks. They do not substitute for the
physical operator qualifiers or the real-model perplexity and exact-token gates.

## Admission rule

A permanent test must protect supported observable behavior or a realistic regression: exact
artifact framing, converter transforms, numerical operator output, state publication, Engine
request behavior, or an external protocol. Source-shape scans, compatibility spellings, deleted
formats, and performance thresholds do not belong in the permanent suite. Performance evidence is
recorded by the benchmark and profiler workflow at the level of the claim.
