#!/usr/bin/env bash
# Incremental app export/deploy for dedicated gfx1201 images and containers.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
builder="${NINFER_DEV_CONTAINER:-ninfer-r9700-builder}"
container="${NINFER_CONTAINER:-ninfer-r9700}"
image="${NINFER_IMAGE:-local/ninfer-r9700:local}"
out="${NINFER_HOT_OUT:-$repo_root/out/hot-patch-r9700}"
restart=1
image_only=0
export_only=0
while (($#)); do
  case "$1" in
    --no-restart) restart=0; shift ;;
    --image-only) image_only=1; shift ;;
    --export-only) export_only=1; shift ;;
    --help|-h)
      echo 'Usage: scripts/hot-patch.sh [--no-restart | --image-only | --export-only]'
      echo 'Build/export CLI, server and PPL; update existing gfx1201 image/container. Existing image is tagged -rollback.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
((image_only && export_only)) && { echo 'Choose image-only or export-only.' >&2; exit 2; }
bash "$repo_root/scripts/dev-setup.sh"
docker exec "$builder" cmake --build /build --parallel "${NINFER_DEV_JOBS:-$(nproc)}" \
  --target ninfer ninfer-serve ninfer-ppl
mkdir -p "$out"
for app in ninfer ninfer-serve ninfer-ppl; do
  docker cp "$builder:/build/apps/$app" "$out/$app"
  chmod +x "$out/$app"
done
echo "Exported binaries into $out"
((export_only)) && exit 0

check_platform() {
  local kind="$1" name="$2" platform
  platform="$(docker inspect --type "$kind" -f '{{index .Config.Labels "org.ninfer.platform"}}' "$name")"
  [[ "$platform" == gfx1201 ]] || {
    echo "Refusing to patch $kind $name: missing gfx1201 platform label." >&2; exit 1;
  }
}
copy_apps() {
  local target="$1" app
  for app in ninfer ninfer-serve ninfer-ppl; do
    docker cp "$out/$app" "$target:/usr/local/bin/$app"
  done
}

image_exists=0
container_exists=0
docker image inspect "$image" >/dev/null 2>&1 && image_exists=1
docker container inspect "$container" >/dev/null 2>&1 && container_exists=1
if ((image_exists)); then check_platform image "$image"; fi
if ((!image_only && container_exists)); then check_platform container "$container"; fi
if ((image_only && !image_exists)); then
  echo "Runtime image $image is missing; build Dockerfile runtime first." >&2; exit 1
fi
temporary_container=""
cleanup() {
  if [[ -n "$temporary_container" ]]; then docker rm -f "$temporary_container" >/dev/null; fi
}
trap cleanup EXIT
if ((image_exists)); then
  temporary_container="$(docker create "$image")"
  copy_apps "$temporary_container"
  docker tag "$image" "$image-rollback"
  docker commit -m 'R9700 incremental app update' "$temporary_container" "$image" >/dev/null
  docker rm "$temporary_container" >/dev/null
  temporary_container=""
  echo "Updated $image; previous image kept as $image-rollback"
fi
if ((!image_only && container_exists)); then
  running="$(docker inspect -f '{{.State.Running}}' "$container")"
  if [[ "$running" == true ]]; then
    # Rename a new inode over each binary: never overwrite a running executable.
    for app in ninfer ninfer-serve ninfer-ppl; do
      docker cp "$out/$app" "$container:/usr/local/bin/$app.hot-patch"
      docker exec "$container" mv -f "/usr/local/bin/$app.hot-patch" "/usr/local/bin/$app"
    done
  else
    copy_apps "$container"
  fi
  if ((restart)) && [[ "$running" == true ]]; then
    docker restart "$container" >/dev/null
    echo "Restarted $container"
  else
    echo "Updated $container binaries; the next process start will use them."
  fi
elif ((!image_exists)); then
  echo "Nothing deployed: image $image and container $container are absent. Exports remain in $out." >&2
  exit 1
fi
