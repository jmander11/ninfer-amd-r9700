#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
plan_dir="$root/profiles/bench/r9700-fp8-gate-up-m128n256-retained-reopen-20260906"
assembly="$plan_dir/retained-fp8-gate-up-m128n256.s"
object="$plan_dir/retained-fp8-gate-up-m128n256.hip.o"
attempt_dir="$plan_dir/attempt-4"
resource_report="$attempt_dir/resource-report.json"
resource_stderr="$attempt_dir/resource-stderr.log"
resource_exit_code_file="$attempt_dir/resource-exit-code.txt"
report="$attempt_dir/report.json"
stderr_log="$attempt_dir/stderr.log"
exit_code_file="$attempt_dir/exit-code.txt"
decision="$attempt_dir/decision.json"
executable_receipt="$attempt_dir/qualification-executable.sha256"
ldd_report="$attempt_dir/ldd.txt"
runtime_library_path=/opt/rocm/core-10.0/lib

cd "$root"
sha256sum --check "$plan_dir/prepared.sha256"
if [[ -e "$attempt_dir" ]]; then
  echo "refusing to overwrite qualification attempt: $attempt_dir" >&2
  exit 2
fi
for output in "$resource_report" "$resource_stderr" "$resource_exit_code_file" "$report" "$stderr_log" "$exit_code_file" "$decision" "$executable_receipt" "$ldd_report"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite qualification output: $output" >&2
    exit 2
  fi
done
mkdir "$attempt_dir"

python3 "$plan_dir/qualification-source/validate_retained.py" \
  --assembly "$assembly" --object "$object" >/dev/null

build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT
executable="$build_dir/fp8_gate_up_m128n256_retained_qualifier"
/opt/rocm/bin/hipcc --offload-arch=gfx1201 -std=c++20 -O3 -DNDEBUG \
  -I"$plan_dir/qualification-source" \
  -DNINFER_FP8_GATE_UP_CANDIDATE_HEADER='"fp8_gate_up_m128n256_retained_abi.h"' \
  -DNINFER_FP8_GATE_UP_CANDIDATE_NAMESPACE=ninfer::qualification::fp8_gate_up_m128n256_retained \
  -DNINFER_FP8_GATE_UP_EXPECT_RUNTIME_REPORTED_REGISTERS=101 \
  -DNINFER_FP8_GATE_UP_STATIC_LDS_BYTES=24576 \
  -DNINFER_FP8_GATE_UP_EXACT_RESOURCES=1 \
  -DNINFER_FP8_GATE_UP_MAX_MATRIX_MS=3.18103603125 \
  -DNINFER_FP8_GATE_UP_MAX_COMPLETE_MS=3.2809701875 \
  -DNINFER_FP8_GATE_UP_FIXED_COMPLETE_OVERHEAD_MS=0.09993415625 \
  -DNINFER_FP8_GATE_UP_MIN_SAVING_64_MS=51.438603 \
  -DNINFER_FP8_GATE_UP_MAX_ORDER_RELATIVE_DELTA=0.05 \
  -DNINFER_FP8_GATE_UP_REPORT_SCHEMA='"ninfer.r9700.fp8_gate_up_m128n256_retained_ab.v1"' \
  -DNINFER_FP8_GATE_UP_HARNESS_NAME='"fp8_gate_up_m128n256_retained_qualifier"' \
  -x hip "$plan_dir/qualification-source/fp8_gate_up_harness.hip" -x none "$object" \
  -lhipblaslt -o "$executable"
sha256sum "$executable" >"$executable_receipt"
LD_LIBRARY_PATH="$runtime_library_path" ldd "$executable" >"$ldd_report"
if grep -F 'not found' "$ldd_report" >/dev/null; then
  echo "qualification executable has an unresolved shared library; attempt retained" >&2
  exit 2
fi
grep -F "libhipblaslt.so.1 => $runtime_library_path/libhipblaslt.so.1" "$ldd_report" >/dev/null
grep -F "libamdhip64.so.7 => $runtime_library_path/libamdhip64.so.7" "$ldd_report" >/dev/null

set +e
LD_LIBRARY_PATH="$runtime_library_path" "$executable" --resources \
  >"$resource_report" 2>"$resource_stderr"
resource_exit=$?
set -e
printf '%s\n' "$resource_exit" >"$resource_exit_code_file"
python3 "$plan_dir/qualification-source/validate_retained.py" \
  --assembly "$assembly" --object "$object" --resource-report "$resource_report" \
  --resource-stderr "$resource_stderr" --resource-exit-code "$resource_exit" >/dev/null

set +e
LD_LIBRARY_PATH="$runtime_library_path" "$executable" --benchmark >"$report" 2>"$stderr_log"
qualification_exit=$?
set -e
printf '%s\n' "$qualification_exit" >"$exit_code_file"
if [[ "$qualification_exit" -ne 0 && "$qualification_exit" -ne 1 ]]; then
  echo "qualifier exited unexpectedly with $qualification_exit; outputs retained" >&2
  exit 2
fi

python3 "$plan_dir/qualification-source/validate_retained.py" \
  --assembly "$assembly" --object "$object" --resource-report "$resource_report" \
  --resource-stderr "$resource_stderr" --resource-exit-code "$resource_exit" --report "$report" \
  --stderr "$stderr_log" --ldd "$ldd_report" --exit-code "$qualification_exit" \
  --output "$decision"
echo "validated retained-candidate decision: $decision"
