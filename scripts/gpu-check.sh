#!/usr/bin/env bash
# R9700 device checks over the `gpucheck` CTest set (tools/r9700/gpu_check; the gfx1201 counterpart
# of compute-sanitizer memcheck, initcheck, and racecheck).
#
#   scripts/gpu-check.sh memcheck  [-- CTEST_ARGS]   guard pages around every hipMalloc (build-r9700)
#   scripts/gpu-check.sh initcheck [-- CTEST_ARGS]   NaN/-1 poison in every hipMalloc (build-r9700)
#   scripts/gpu-check.sh racecheck [-- CTEST_ARGS]   LDS race instrumentation (build-r9700-racecheck)
#
# Each test runs under its own lease of the shared R9700 GPU lock. A finding fails the test: memcheck by a GPU
# memory fault, initcheck by the test's own oracle, racecheck by exit status 86 with a report.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-}"
case "$mode" in
  memcheck|initcheck|racecheck) shift ;;
  *) echo 'Usage: scripts/gpu-check.sh memcheck|initcheck|racecheck [-- CTEST_ARGS]' >&2; exit 2 ;;
esac
ctest_args=()
if (($#)); then
  [[ "$1" == "--" ]] || { echo "Unknown option: $1; pass CTest arguments after --." >&2; exit 2; }
  shift
  ctest_args=("$@")
fi
jobs="${NINFER_DEV_JOBS:-8}"
[[ "$jobs" =~ ^[1-8]$ ]] || { echo 'NINFER_DEV_JOBS must be 1..8.' >&2; exit 2; }
gpu_lock="${NINFER_GPU_LOCK:-/ssdpool2nvme/local_llm/.ninfer-coordination/gpu.lock}"
tools="$repo_root/tools/r9700/gpu_check"

make -C "$tools" --no-print-directory
if [[ "$mode" == racecheck ]]; then
  build_dir="${NINFER_RACECHECK_BUILD_DIR:-$repo_root/build-r9700-racecheck}"
  if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    cmake -S "$repo_root" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      "-DCMAKE_HIP_COMPILER_LAUNCHER=python3.11;$tools/hip_racecheck_launcher.py" \
      -DNINFER_BUILD_APPS=OFF -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_R9700_QUALIFIER_TIMEOUT=3600
  fi
  # The launcher and its tools are not Ninja dependencies: rebuild every HIP object when they
  # change, or when no stamp records which tools built the tree.
  stamp="$build_dir/.gpu-check-tools.stamp"
  if [[ ! -f "$stamp" ]] || [[ -n "$(find "$tools/build" "$tools/hip_racecheck_launcher.py" \
      -newer "$stamp" -type f -print -quit)" ]]; then
    cmake --build "$build_dir" --target clean
  fi
else
  build_dir="${NINFER_BUILD_DIR:-$repo_root/build-r9700}"
  test -f "$build_dir/CMakeCache.txt" || { echo "Configure $build_dir first." >&2; exit 1; }
fi
cmake -S "$repo_root" -B "$build_dir" >/dev/null  # current labels before listing the set
mapfile -t targets < <(ctest --test-dir "$build_dir" -N -L gpucheck "${ctest_args[@]}" |
  sed -n 's/^ *Test *#[0-9]*: //p')
((${#targets[@]})) || { echo "No gpucheck tests are registered in $build_dir." >&2; exit 1; }
cmake --build "$build_dir" --parallel "$jobs" --target "${targets[@]}"
[[ "$mode" == racecheck ]] && touch "$build_dir/.gpu-check-tools.stamp"

export LD_PRELOAD="$tools/build/libninfer_gpu_check.so${LD_PRELOAD:+:$LD_PRELOAD}"
export NINFER_GPU_CHECK="$mode"
# One GPU-lock lease per test, so other sessions can interleave during a long racecheck set.
failed=()
for test in "${targets[@]}"; do
  flock --exclusive "$gpu_lock" \
    ctest --test-dir "$build_dir" --output-on-failure -R "^${test}\$" || failed+=("$test")
done
if ((${#failed[@]})); then
  printf 'gpu-check %s failed: %s\n' "$mode" "${failed[*]}" >&2
  exit 1
fi
echo "gpu-check $mode: ${#targets[@]} tests passed"
