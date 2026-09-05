#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if (( $# != 0 )); then
  echo "build.sh accepts no arguments" >&2
  exit 2
fi
HIPCC=/opt/rocm/bin/hipcc
PY=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python
COMMON=(-O3 -std=c++20 --offload-arch=gfx1201 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-const-variable -Isrc -Itools/r9700)
OBJECT=build-r9700/src/CMakeFiles/ninfer_r9700_core.dir/ops/r9700/linear/r9700_linear.hip.o
rm -f tools/r9700/build/a8q4_n16k16_weight_ht_fatbin.bin \
  tools/r9700/build/a8q4_n16k16_weight_ht_baseline.hsaco \
  tools/r9700/build/a8q4_n16k16_weight_ht_candidate.hsaco
/usr/bin/objcopy --dump-section \
  .hip_fatbin=tools/r9700/build/a8q4_n16k16_weight_ht_fatbin.bin "$OBJECT"
"$PY" -m tools.r9700.patch_a8q4_n16k16_weight_ht_assembly \
  --fatbin tools/r9700/build/a8q4_n16k16_weight_ht_fatbin.bin \
  --baseline tools/r9700/build/a8q4_n16k16_weight_ht_baseline.hsaco
/opt/rocm/core-10.0/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201 \
  tools/r9700/build/a8q4_n16k16_weight_ht_baseline.hsaco \
  > tools/r9700/build/a8q4_n16k16_weight_ht_baseline.objdump
"$PY" -m tools.r9700.patch_a8q4_n16k16_weight_ht_assembly \
  --fatbin tools/r9700/build/a8q4_n16k16_weight_ht_fatbin.bin \
  --disassembly tools/r9700/build/a8q4_n16k16_weight_ht_baseline.objdump \
  --baseline tools/r9700/build/a8q4_n16k16_weight_ht_baseline.hsaco \
  --candidate tools/r9700/build/a8q4_n16k16_weight_ht_candidate.hsaco
/opt/rocm/core-10.0/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201 \
  tools/r9700/build/a8q4_n16k16_weight_ht_candidate.hsaco \
  > tools/r9700/build/a8q4_n16k16_weight_ht_candidate.objdump
/opt/rocm/core-10.0/lib/llvm/bin/llvm-readobj --notes \
  tools/r9700/build/a8q4_n16k16_weight_ht_candidate.hsaco \
  > tools/r9700/build/a8q4_n16k16_weight_ht_candidate.notes
"$HIPCC" "${COMMON[@]}" \
  tools/r9700/a8q4_n16k16_weight_ht_harness.hip \
  -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
  -o tools/r9700/build/a8q4_n16k16_weight_ht_harness
"$PY" -m tools.r9700.check_a8q4_n16k16_weight_ht_static \
  --baseline tools/r9700/build/a8q4_n16k16_weight_ht_baseline.hsaco \
  --candidate tools/r9700/build/a8q4_n16k16_weight_ht_candidate.hsaco \
  --baseline-disassembly tools/r9700/build/a8q4_n16k16_weight_ht_baseline.objdump \
  --candidate-disassembly tools/r9700/build/a8q4_n16k16_weight_ht_candidate.objdump \
  --notes tools/r9700/build/a8q4_n16k16_weight_ht_candidate.notes
"$PY" -m py_compile \
  tools/r9700/check_a8q4_n16k16_weight_ht_static.py \
  tools/r9700/patch_a8q4_n16k16_weight_ht_assembly.py \
  tools/r9700/run_a8q4_n16k16_weight_ht_gate.py \
  tools/r9700/validate_a8q4_n16k16_weight_ht_report.py
"$PY" -m unittest \
  tools.r9700.test_check_a8q4_n16k16_weight_ht_static \
  tools.r9700.test_run_a8q4_n16k16_weight_ht_gate
