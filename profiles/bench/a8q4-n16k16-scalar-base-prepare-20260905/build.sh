#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if (( $# != 0 )); then echo "build.sh accepts no arguments" >&2; exit 2; fi
HIPCC=/opt/rocm/bin/hipcc
PY=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python
COMMON=(-O3 -std=c++20 --offload-arch=gfx1201 -Wall -Wextra -Werror -Wno-unused-command-line-argument -Isrc -Itools/r9700)
OBJECT=build-r9700/src/CMakeFiles/ninfer_r9700_core.dir/ops/r9700/linear/r9700_linear.hip.o
FATBIN=tools/r9700/build/a8q4_n16k16_scalar_base_fatbin.bin
BASE=tools/r9700/build/a8q4_n16k16_scalar_base_baseline.hsaco
CAND=tools/r9700/build/a8q4_n16k16_scalar_base_candidate.hsaco
BUNDLE=tools/r9700/build/a8q4_n16k16_scalar_base_candidate.bundle
rm -f "$FATBIN" "$BASE" "$CAND" "$BUNDLE" tools/r9700/build/a8q4_n16k16_scalar_base_qual.s tools/r9700/build/a8q4_n16k16_scalar_base_candidate.notes tools/r9700/build/a8q4_n16k16_scalar_base_baseline.objdump tools/r9700/build/a8q4_n16k16_scalar_base_candidate.objdump tools/r9700/build/a8q4_n16k16_scalar_base_harness
/usr/bin/objcopy --dump-section .hip_fatbin="$FATBIN" "$OBJECT"
"$PY" -m tools.r9700.patch_a8q4_n16k16_weight_ht_assembly --fatbin "$FATBIN" --baseline "$BASE"
"$HIPCC" "${COMMON[@]}" --genco tools/r9700/a8q4_n16k16_scalar_base_qual.hip -o "$BUNDLE"
/opt/rocm/core-10.0/lib/llvm/bin/clang-offload-bundler -type=o -unbundle \
  -input="$BUNDLE" -output="$CAND" -targets=hipv4-amdgcn-amd-amdhsa--gfx1201
"$HIPCC" "${COMMON[@]}" -S tools/r9700/a8q4_n16k16_scalar_base_qual.hip -o tools/r9700/build/a8q4_n16k16_scalar_base_qual.s
/opt/rocm/core-10.0/lib/llvm/bin/llvm-readobj --notes "$CAND" > tools/r9700/build/a8q4_n16k16_scalar_base_candidate.notes
/opt/rocm/core-10.0/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201 "$BASE" > tools/r9700/build/a8q4_n16k16_scalar_base_baseline.objdump
/opt/rocm/core-10.0/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201 "$CAND" > tools/r9700/build/a8q4_n16k16_scalar_base_candidate.objdump
"$HIPCC" "${COMMON[@]}" tools/r9700/a8q4_n16k16_scalar_base_harness.hip -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -o tools/r9700/build/a8q4_n16k16_scalar_base_harness
"$PY" -m tools.r9700.check_a8q4_n16k16_scalar_base_static --source tools/r9700/a8q4_n16k16_scalar_base_qual.hip --assembly tools/r9700/build/a8q4_n16k16_scalar_base_qual.s --baseline "$BASE" --candidate "$CAND"
"$PY" -m py_compile tools/r9700/check_a8q4_n16k16_scalar_base_static.py tools/r9700/run_a8q4_n16k16_scalar_base_gate.py
"$PY" -m unittest tools.r9700.test_check_a8q4_n16k16_scalar_base_static tools.r9700.test_run_a8q4_n16k16_scalar_base_gate
