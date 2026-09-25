#!/usr/bin/env bash
# Host tests by default. --gpu explicitly enables physical qualifiers; --real names one artifact.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
builder="${NINFER_DEV_CONTAINER:-ninfer-r9700-builder}"
build_dir="${NINFER_BUILD_DIR:-$repo_root/build-r9700}"
python="${NINFER_PYTHON:-python3.11}"
gpu=0
run_python=0
use_builder=0
artifact=""
ctest_args=()
while (($#)); do
  case "$1" in
    --gpu) gpu=1; shift ;;
    --python) run_python=1; shift ;;
    --builder) use_builder=1; shift ;;
    --real) artifact="${2:?--real requires an explicit artifact path}"; shift 2 ;;
    --print-weights) printf 'Explicit artifact: %s\n' "${NINFER_ARTIFACT:-not configured}"; exit 0 ;;
    --help|-h)
      echo 'Usage: scripts/run-unit-tests.sh [--builder] [--gpu] [--python] [--real ARTIFACT.ninfer] [-- CTEST_ARGS]'
      echo 'Default: existing native build-r9700 host tests. --builder uses the dedicated container; paths then refer to that container.'
      exit 0 ;;
    --) shift; ctest_args+=("$@"); break ;;
    *) echo "Unknown option: $1; pass CTest arguments after --." >&2; exit 2 ;;
  esac
done
jobs="${NINFER_DEV_JOBS:-12}"
[[ "$jobs" =~ ^([1-9]|1[0-4])$ ]] || { echo 'NINFER_DEV_JOBS must be 1..14.' >&2; exit 2; }
if ((use_builder)); then
  bash "$repo_root/scripts/dev-setup.sh"
  args=()
  ((gpu)) && args+=(--gpu)
  ((run_python)) && args+=(--python)
  [[ -n "$artifact" ]] && args+=(--real "$artifact")
  docker exec -e NINFER_BUILD_DIR=/build -e NINFER_PYTHON=/opt/python311/bin/python3.11 \
    -e "NINFER_DEV_JOBS=$jobs" \
    -e "NINFER_DRM_RENDER_NODE=${NINFER_DRM_RENDER_NODE:-/dev/dri/renderD128}" \
    -e "NINFER_MIN_FREE_VRAM_GIB=${NINFER_MIN_FREE_VRAM_GIB:-20}" \
    -w /src "$builder" bash /src/scripts/run-unit-tests.sh "${args[@]}" -- "${ctest_args[@]}"
  exit 0
fi
test -f "$build_dir/CMakeCache.txt" || { echo "Configure $build_dir with BUILD_TESTING=ON first." >&2; exit 1; }
if [[ -n "$artifact" ]]; then
  [[ "$artifact" == *.ninfer && -s "$artifact" ]] || { echo 'Explicit .ninfer artifact is missing or empty.' >&2; exit 1; }
fi
if ((gpu)) || [[ -n "$artifact" ]]; then
  node="${NINFER_DRM_RENDER_NODE:-/dev/dri/renderD128}"
  device_sysfs="/sys/class/drm/$(basename "$node")/device"
  read -r total < "$device_sysfs/mem_info_vram_total"
  read -r used < "$device_sysfs/mem_info_vram_used"
  minimum="${NINFER_MIN_FREE_VRAM_GIB:-20}"
  [[ "$minimum" =~ ^[0-9]+$ ]] || { echo 'VRAM minimum must be an integer GiB value.' >&2; exit 1; }
  ((total - used >= minimum * 1024 * 1024 * 1024)) || {
    echo "Insufficient free R9700 VRAM; stop the resident workload before testing (need $minimum GiB)." >&2; exit 1;
  }
  exec 9>"$build_dir/.r9700-tests.lock"
  flock -n 9 || { echo 'Another test runner owns this build GPU lane.' >&2; exit 1; }
fi
cmake --build "$build_dir" --parallel "$jobs"
selection=()
((gpu)) || selection+=(-LE r9700)
ctest --test-dir "$build_dir" --output-on-failure "${selection[@]}" "${ctest_args[@]}"
if [[ -n "$artifact" ]]; then
  "$build_dir/src/ninfer_r9700_engine_cache_cancel_qual" "$artifact"
fi
if ((run_python)); then
  "$python" -c 'import sys; assert sys.version_info[:2] == (3, 11); import pytest, torch'
  cd "$repo_root"
  "$python" -m pytest tests/artifact tests/test_bench_matrix.py tests/test_serve_corpus.py
fi
