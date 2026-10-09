#!/usr/bin/env bash
# Full R9700 promotion checks. This validates a committed candidate; it never merges or pushes.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: scripts/master-gates.sh ARTIFACT.ninfer'
  echo 'Run every master gate on a clean candidate containing origin/master. See docs/maintainer/merging-to-master.md.'
  exit 0
fi
(($# == 1)) || { echo 'Supply one explicit production R9700 .ninfer artifact.' >&2; exit 2; }
artifact="$(realpath "$1")"
[[ "$artifact" == *.ninfer && -s "$artifact" ]] || { echo 'Artifact is missing or empty.' >&2; exit 2; }
cd "$repo_root"
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit the candidate before running master gates.' >&2; exit 1; }
git fetch origin master
candidate="$(git rev-parse HEAD)"
master="$(git rev-parse origin/master)"
git merge-base --is-ancestor "$master" "$candidate" || {
  echo 'Merge origin/master into the working branch first, then rerun.' >&2; exit 1;
}
jobs="${NINFER_DEV_JOBS:-8}"
[[ "$jobs" =~ ^[1-8]$ ]] || { echo 'NINFER_DEV_JOBS must be 1..8.' >&2; exit 2; }
python="${NINFER_PYTHON:-python3.11}"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 11); import pytest, torch, safetensors'
build_dir="$repo_root/build-r9700"
asan_dir="$repo_root/build-r9700-asan"
gpu_lock=/ssdpool2nvme/local_llm/.ninfer-coordination/gpu.lock
report_dir="$repo_root/profiles/master-gates/$candidate"
mkdir -p "$report_dir"
export NINFER_R9700_WEIGHTS="$artifact"
export NINFER_DEV_JOBS="$jobs"
export NINFER_BUILD_DIR="$build_dir"
export ASAN_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
{
  printf 'candidate=%s\nmaster=%s\nartifact=%s\n' "$candidate" "$master" "$artifact"
  uname -sr
  /opt/rocm/llvm/bin/clang++ --version
} > "$report_dir/environment.txt"
# Each gate retains its command and output, and stops promotion on failure.
gate() {
  local name="$1"
  shift
  printf 'Running %s\n' "$name"
  printf '%q ' "$@" > "$report_dir/$name.command"
  printf '\n' >> "$report_dir/$name.command"
  if "$@" > "$report_dir/$name.log" 2>&1; then
    printf '%s PASS\n' "$name" >> "$report_dir/results.txt"
  else
    printf '%s FAIL\n' "$name" >> "$report_dir/results.txt"
    tail -60 "$report_dir/$name.log" >&2
    return 1
  fi
}
# CTest considers skipped tests successful; a master gate requires that each selected test ran.
ctest_gate() {
  local name="$1" tree="$2"
  shift 2
  gate "$name" flock --exclusive "$gpu_lock" ctest --test-dir "$tree" \
    --output-on-failure --no-tests=error --output-junit "$report_dir/$name.xml" "$@"
  require_no_skips "$report_dir/$name.xml"
}
require_no_skips() {
  "$python" - "$1" <<'PYXML'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = list(root.iter("testcase"))
if not cases or any(case.find("skipped") is not None for case in cases):
    raise SystemExit("Gate did not execute every selected test; promotion is blocked")
PYXML
}
: > "$report_dir/results.txt"
gate 01-hygiene pre-commit run --all-files
[[ -z "$(git status --porcelain)" ]] || { echo 'A hook rewrote files; commit fixes and restart.' >&2; exit 1; }
gate 02-configure cmake -S "$repo_root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  "-DPython3_EXECUTABLE=$python" -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_WARNINGS_AS_ERRORS=ON -DNINFER_SANITIZE=
gate 02-build cmake --build "$build_dir" --parallel "$jobs"
gate 03-clang-tidy "$python" scripts/run-clang-tidy.py --build-dir "$build_dir"
ctest_gate 04-unit "$build_dir" -LE real
gate 04-python flock --exclusive "$gpu_lock" env NINFER_RUN_R9700_CODEC_TESTS=1 "$python" -m pytest \
  "--junitxml=$report_dir/04-python.xml" \
  tests/artifact tests/test_bench_matrix.py tests/test_serve_corpus.py \
  tools/convert/qwen3_8_27b_r9700/test_codec.py \
  tools/convert/qwen3_8_27b_r9700/test_vectorized_codec.py
require_no_skips "$report_dir/04-python.xml"
ctest_gate 05-real "$build_dir" -L real
gate 05-cache-cancel flock --exclusive "$gpu_lock" \
  "$build_dir/src/ninfer_r9700_engine_cache_cancel_qual" "$artifact"
gate 05-recovery-mtp flock --exclusive "$gpu_lock" \
  "$build_dir/src/ninfer_r9700_recovery_kv_qual" "$artifact" mtp
gate 05-recovery-dflash flock --exclusive "$gpu_lock" \
  "$build_dir/src/ninfer_r9700_recovery_kv_qual" "$artifact" dflash
gate 06-memcheck bash scripts/gpu-check.sh memcheck
gate 07-racecheck bash scripts/gpu-check.sh racecheck
gate 08-initcheck bash scripts/gpu-check.sh initcheck
gate 09-asan-configure cmake -S "$repo_root" -B "$asan_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  "-DPython3_EXECUTABLE=$python" -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_SANITIZE=address,undefined
gate 09-asan-build cmake --build "$asan_dir" --parallel "$jobs"
export ASAN_OPTIONS=protect_shadow_gap=0
ctest_gate 09-asan "$asan_dir" -LE r9700
[[ "$(git rev-parse HEAD)" == "$candidate" && -z "$(git status --porcelain)" ]] || {
  echo 'Candidate changed during the gates; restart on the committed candidate.' >&2; exit 1;
}
git fetch origin master
[[ "$(git rev-parse origin/master)" == "$master" ]] || {
  echo 'Master moved; reconcile it and restart the gates.' >&2; exit 1;
}
printf 'All master gates passed for %s. Evidence: %s\n' "$candidate" "$report_dir"
