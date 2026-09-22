#!/usr/bin/env bash
# Build/start the dedicated gfx1201 builder. See docs/containers.md.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: scripts/dev-setup.sh'
  echo 'Requires local ROCm 10 and self-contained Python 3.11; optional NINFER_MODELS_DIR mounts read-only.'
  exit 0
fi
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
builder="${NINFER_DEV_CONTAINER:-ninfer-r9700-builder}"
image="${NINFER_BUILDER_IMAGE:-local/ninfer-r9700-builder:local}"
volume="${NINFER_BUILD_VOLUME:-ninfer-r9700-build-cache}"
rocm_context="${NINFER_ROCM_CONTEXT:-/opt/rocm/core-10.0}"
render_node="${NINFER_DRM_RENDER_NODE:-/dev/dri/renderD128}"
command -v docker >/dev/null
if [[ ${NINFER_REBUILD_BUILDER:-0} == 1 ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
  python_context="${NINFER_PYTHON_CONTEXT:-}"
  if [[ -z "$python_context" ]]; then
    python_executable="$(realpath "$(command -v python3.11)")"
    python_context="$(dirname "$(dirname "$python_executable")")"
  fi
  test -x "$rocm_context/lib/llvm/bin/clang++"
  test -x "$python_context/bin/python3.11"
  docker build --target build --tag "$image" \
    --build-context "rocm=$rocm_context" --build-context "python311=$python_context" "$repo_root"
fi
platform="$(docker image inspect -f '{{index .Config.Labels "org.ninfer.platform"}}' "$image")"
[[ "$platform" == gfx1201 ]] || { echo 'Builder image is not marked gfx1201.' >&2; exit 1; }
if docker inspect --type container "$builder" >/dev/null 2>&1; then
  mounted="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/src"}}{{.Source}}{{end}}{{end}}' "$builder")"
  [[ "$mounted" == "$repo_root" ]] || { echo "Existing builder mounts another checkout: $mounted" >&2; exit 1; }
  platform="$(docker inspect -f '{{index .Config.Labels "org.ninfer.platform"}}' "$builder")"
  [[ "$platform" == gfx1201 ]] || { echo 'Existing builder is not marked gfx1201.' >&2; exit 1; }
else
  test -c /dev/kfd
  test -c "$render_node"
  docker volume create "$volume" >/dev/null
  args=(--name "$builder" --device /dev/kfd --device "$render_node"
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g "$render_node")"
    --shm-size 1g -v "$repo_root:/src:rw" -v "$volume:/build" -w /src)
  if [[ -n ${NINFER_MODELS_DIR:-} ]]; then
    models_dir="$(realpath "$NINFER_MODELS_DIR")"
    test -d "$models_dir"
    args+=(-v "$models_dir:/models:ro")
  fi
  docker create "${args[@]}" "$image" sleep infinity >/dev/null
fi
if [[ $(docker inspect -f '{{.State.Running}}' "$builder") != true ]]; then
  docker start "$builder" >/dev/null
fi
docker exec "$builder" cmake -S /src -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=ON -DPython3_EXECUTABLE=/opt/python311/bin/python3.11 \
  -DNINFER_BUILD_BENCHMARKS="${NINFER_BUILD_BENCHMARKS:-OFF}"
echo "Builder $builder ready: $repo_root at /src; volume $volume at /build."
