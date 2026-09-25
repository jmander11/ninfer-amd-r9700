# R9700 containers and test runner

## Compose: server with persistent prefix caching

`compose.yaml` builds this checkout and runs the selected DFlash artifact with
temperature **1.5**, **4 GiB pinned RAM prefix cache**, **32 GiB disk cache**, chunk
2048, C1/K5 and a32768-token context limit. Cache sizes are capacities, not upfront
disk allocations. A bind mount stores disk prefixes under
`/ssdpool2nvme/local_llm/cache_r9700/prefix`, beside (not shared with) the 5090
cache. Override its parent with `NINFER_DISK_CACHE_DIR`. This host directory
survives container replacement and `docker compose down --volumes`.
Disk cache is prefix reuse, not active-context offload. RAM contents do not survive
restart; durable disk entries can be restored for the same model fingerprint.
Requests may override temperature; p-less sampling remains enabled by default.

```sh
cp .env.example .env
# Edit the explicit model, ROCm and self-contained Python paths for your machine.
mkdir -p /ssdpool2nvme/local_llm/cache_r9700
docker compose config --quiet
docker compose build server
docker compose up -d --no-build server
docker compose logs -f server
```

Buildx is required to build, not to run an already-built image. It is Docker's
BuildKit frontend: it supplies build caching, multi-stage builds and the named
local ROCm/Python contexts used here. Compose is the separate plugin that manages
the service's devices, ports, mounts and lifecycle. Both must be installed for
the complete workflow below.

The image compiles a snapshot of **local working-tree files** using CMake and
Ninja, not `make`, and copies the resulting apps into the runtime image. It does
not pull Git, live-mount source, or compile at server startup. After changing or
pulling source, stop the server, run `docker compose build server`, then
`docker compose up -d --no-build server`; Docker reuses unchanged build layers.
Do not overlap image compilation with inference or other heavy jobs.

`docker compose up -d` normally reuses an existing image; it can build a missing
image but does not detect source edits as a rebuild requirement. Use
`docker compose up -d --build server` to explicitly build before starting.
Stop an already-running server first on this shared host.

For a memory-bounded build, create a dedicated Buildx builder once (this does not
change Docker's globally selected builder):

```sh
docker buildx create --name ninfer-r9700-build --driver docker-container \
  --driver-opt memory=24g,memory-swap=24g,cpu-period=100000,cpu-quota=1400000,default-load=true
docker compose build --builder ninfer-r9700-build server
docker compose up -d --no-build server
```

The compile-job default remains4. Compose's service memory limit applies to the
running server, not image compilation; the dedicated builder supplies the latter
limit. Reuse it for subsequent builds rather than creating it again.

The default listener is host loopback port8080, with no API key. Change the bind
address only behind suitable access controls. `.env` can override model path,
port, context, C1..4, draft count and cache capacities. For the measured workload,
use K5 at C1–2 and K4 at C3–4. The runtime is bounded to24GiB host RAM with no
container swap; this limit does not count GPU VRAM. Cache/runtime allocations
must fit that bound. Existing NVIDIA containers are not modified.

Docker references: https://docs.docker.com/reference/compose-file/build/ and
https://docs.docker.com/reference/cli/docker/buildx/build/.
Build/restart semantics: https://docs.docker.com/reference/cli/docker/compose/up/;
builder limits: https://docs.docker.com/build/builders/drivers/docker-container/.

## Image prerequisites

The Dockerfile builds only `gfx1201` and the Qwen3.8-27B product. It copies the
maintainer's self-contained ROCm 10 tree through BuildKit's `rocm` named context,
and a self-contained Python 3.11 distribution through `python311`. No model or
converted artifact is downloaded or copied into an image. Docker build installs
Ubuntu build/runtime packages and CMake fetches the pinned CPU grammar dependency.

Docker Buildx is required for these named contexts. On this Ubuntu host with the
Docker apt repository configured, install the missing plugin once:

```sh
sudo apt-get install docker-buildx-plugin
docker buildx version
```

```sh
docker buildx build --load --target runtime --tag local/ninfer-r9700:local \
  --build-context rocm=/opt/rocm/core-10.0 \
  --build-context python311=/absolute/path/to/self-contained-python-3.11 .
```

Image builds default to four compile jobs; `--build-arg NINFER_BUILD_JOBS=8`
overrides this within the enforced range 1–14. Build and GPU/model jobs must run
serially on the shared host. These are native AMD images: no NVIDIA Container
Toolkit or `--gpus all` is needed.

Both contexts must contain their `bin` and `lib` directories; do not provide a
single executable, an external-symlink virtual environment, or the parent of a
versioned ROCm installation. The runtime retains the same ROCm tree used to link
the apps. The host provides the AMD kernel driver and `/dev/kfd` plus the R9700's
DRM render node. The default render node is `/dev/dri/renderD128`; select the
actual R9700 node with `NINFER_DRM_RENDER_NODE` if it differs.

## Builder

```sh
NINFER_PYTHON_CONTEXT=/absolute/path/to/self-contained-python-3.11 \
NINFER_MODELS_DIR=/absolute/path/to/local-models \
  bash scripts/dev-setup.sh
```

Defaults are `local/ninfer-r9700-builder:local`, container `ninfer-r9700-builder`,
and volume `ninfer-r9700-build-cache`. Override them with `NINFER_BUILDER_IMAGE`,
`NINFER_DEV_CONTAINER`, and `NINFER_BUILD_VOLUME`. `NINFER_ROCM_CONTEXT` defaults
to `/opt/rocm/core-10.0`. If the Python context is omitted, setup resolves the
installed `python3.11` executable and takes its distribution root. Setup mounts
the checkout at `/src`, the build volume at `/build`, and the optional models
directory read-only at `/models`. It checks existing container ownership of the
checkout and never repurposes another builder. Set `NINFER_REBUILD_BUILDER=1` to
rebuild the image; an existing container continues to use its original image
until the user recreates it. No packages are installed into an existing container.

Setup configures the mounted build volume. Compile the applications explicitly:

```sh
docker exec ninfer-r9700-builder cmake --build /build --parallel 4 \
  --target ninfer ninfer-serve ninfer-ppl
docker exec ninfer-r9700-builder /build/apps/ninfer --help
```

This is the incremental development path: CMake/Ninja retain objects in `/build`
and rebuild affected dependencies after edits to the live `/src` checkout. Native
`cmake --build build-r9700 --parallel 4 --target ninfer-serve` is incremental too.
The Compose runtime image is separate: its Dockerfile caches complete build layers,
but a changed source snapshot invalidates the compile layer and recompiles the apps.
Building in the development container does not update the Compose runtime image.

For source-only edits, the AMD-adapted `scripts/hot-patch.sh` also deploys the
incrementally built CLI/server/PPL apps. For this Compose service:

```sh
docker compose stop server
set -a
. ./.env
set +a
NINFER_IMAGE=local/ninfer-r9700:compose NINFER_DEV_JOBS=4 \
  bash scripts/hot-patch.sh --image-only
docker compose up -d --no-build --wait server
```

The first invocation creates the development builder if absent; later invocations
reuse its persistent objects. The previous runtime image receives a `-rollback`
tag. This helper was inspected, not executed in the Compose validation below.
Dockerfile, toolchain or runtime-library changes still require a full image build.

For benchmarks, set `NINFER_BUILD_BENCHMARKS=ON` when running setup, then build
the `ninfer_bench` target; its executable is `/build/bench/ninfer_bench`.
The runtime image contains the three applications, not the benchmark or test tools.

## Tests

There is a registered CTest suite for host contracts and physical GPU Ops/runtime
qualifiers, plus separate real-model and live-HTTP checks. No single command proves
every model/context/precision combination. Run GPU tests only after stopping the
server and any other GPU owner; never overlap them with builds or model conversion.

```sh
docker compose stop server
# Existing native build, rebuilt first; all registered host and GPU CTests, serially:
bash scripts/run-unit-tests.sh --gpu -- --parallel 1
# Real-artifact RAM continuation/cancellation (ordinary greedy execution):
bash scripts/run-unit-tests.sh --real /absolute/path/to/exact-model.ninfer -- --parallel 1
# Restart/persistence proof requires a fresh directory whose parent already exists:
build-r9700/src/ninfer_r9700_engine_cache_cancel_qual /absolute/path/to/exact-model.ninfer \
  --disk-dir /absolute/path/to/new-cache-qualification
# Start Compose only after the other tests finish, then exercise real HTTP routes:
docker compose up -d --no-build server
python3.11 tools/smoke/serve_cache.py --base-url http://127.0.0.1:8080
```

The HTTP smoke sends Chat, Responses, Anthropic and streaming requests without a
temperature override, checks generated output and both cache-stat objects, and
reports cache observations. It does not prove disk restoration; the separate
restart qualifier explicitly requires a disk hit and cold-output parity.

Validation on2026-09-24: rebuilt native gfx1201 suite86/86 passed, including physical
GPU tests. Corrected stale qualifier expectations for temperature2.0, K<=5/widthK+1,
W4 WMMA graph routing and reused selector scratch; engine arithmetic was unchanged.
The selected17.00GB DFlash artifact served all four HTTP smoke routes using the
Compose defaults (only host port/cache path differed). Request logs confirm
temperature1.5, RAM restoration and a12-token disk restore after clean restart.
The separate ordinary-greedy Engine qualifier passed cancellation/RAM continuation
and disk restart with38 reused tokens/16 output tokens exact to cold execution.
Local native request logs are under
`profiles/bench/r9700-compose-validation-20260924/`.
Compose configuration and actual image/runtime validation also pass after installing
Buildx. `local/ninfer-r9700:compose` was built from this checkout with the local
ROCm/Python contexts and a24GiB/no-swap builder. All three apps compiled; the
Compose server became healthy, passed Chat/Responses/Anthropic/SSE, logged
temperature1.5, restored RAM prefixes and restored12 prompt tokens from disk after
restart (22 generated tokens). The image and stopped service
are retained; `docker compose up -d` can now start this tested image without building.
Generated profiler dumps and local qualifier binaries are excluded from build context.

The final Compose mount was separately verified as a writable bind from
`/ssdpool2nvme/local_llm/cache_r9700` to `/cache`. All four HTTP smoke routes and
RAM reuse pass; after restart, the request restores12 prompt tokens from
`host_disk` and generates21 tokens. An observed disk save was84ms. These are
functional observations, not a controlled storage benchmark. The earlier named
volume on Docker's ZFS data root had2.8–3.8s writes; that unused volume is retained,
not migrated or deleted. No 5090 cache or Docker data-root settings were changed.

The native runner uses an already configured `build-r9700` with `BUILD_TESTING=ON`;
`NINFER_BUILD_DIR` changes that explicit path. `--builder` selects the dedicated
builder and its `/build`. Host tests run by default. GPU tests must be requested
explicitly, and the runner does not stop a resident server or other GPU owner.

```sh
bash scripts/run-unit-tests.sh -- --stop-on-failure
bash scripts/run-unit-tests.sh --builder --gpu -- -R ninfer_kv_disk_cache_test
bash scripts/run-unit-tests.sh --builder --real /models/exact-selected-artifact.ninfer
NINFER_PYTHON=/absolute/path/to/python3.11 bash scripts/run-unit-tests.sh --python
```

`--real` runs the public-Engine ordinary cancellation and RAM continuation
qualifier with that exact artifact (one active request, greedy, 16-token outputs).
It does not infer a model filename or silently skip a missing artifact. Paths
passed with `--builder` are container paths. `--gpu` enables registered physical
tests; it does not select a cache dtype. All product tests use FP8 E4M3FN keys,
signed INT4 values and FP16 value scales.

The same built qualifier has a separate explicit SSD restart mode:

```sh
build-r9700/src/ninfer_r9700_engine_cache_cancel_qual /absolute/path/to/selected.ninfer \
  --disk-dir /absolute/path/to/new-cache-qualification
```

It uses one lane, 16-token greedy requests, 1 GiB RAM and 1 GiB SSD cache, closes
and reopens the public Engine, requires an SSD restore, and compares with cold
output. The parent directory must exist; the named directory must not exist.
Its cache files are retained on success or failure and are never removed by the test.

The physical-test preflight reads the selected DRM device's VRAM counters and
requires 20 GiB free by default; `NINFER_MIN_FREE_VRAM_GIB` can set a different
explicit requirement for a focused test. A build-directory lock prevents two
instances of this runner sharing that build from using the GPU concurrently.
CTest also uses the `r9700_device` resource lock. These locks do not coordinate
unrelated benchmark processes: the maintainer still schedules the sole GPU.

`--python` additionally runs the artifact, benchmark-matrix and serving-corpus
Python tests with Python 3.11. The selected interpreter must already provide
`pytest` and `torch`; the runner does not install or upgrade dependencies.
`NINFER_DEV_JOBS` controls setup, test-runner and hot-patch build parallelism
(default 4, enforced range 1–14). CTest arguments follow `--`.

## Runtime and incremental app deployment

```sh
docker run --name ninfer-r9700 --device /dev/kfd --device /dev/dri/renderD128 \
  -v /absolute/path/to/local-models:/models:ro -p 8080:8080 \
  local/ninfer-r9700:local ninfer-serve /models/exact-selected-artifact.ninfer \
  --host 0.0.0.0 --port 8080 --max-concurrency 1
bash scripts/hot-patch.sh --export-only
bash scripts/hot-patch.sh --image-only
bash scripts/hot-patch.sh --no-restart
```

`hot-patch.sh` builds CLI, server and PPL in the dedicated builder. Exports go to
`out/hot-patch-r9700` (`NINFER_HOT_OUT` overrides it). Without an export-only flag,
it updates the existing `local/ninfer-r9700:local` image and `ninfer-r9700`
container; `NINFER_IMAGE` and `NINFER_CONTAINER` select explicit alternatives.
Every target must carry `org.ninfer.platform=gfx1201`, which this Dockerfile sets.
It refuses an unmarked or different-platform target, preserving existing NVIDIA
deployments. The prior image is tagged with the `-rollback` suffix. By default a
running target container restarts; `--no-restart` leaves its current process
running, and `--image-only` changes only the image. A stopped container is never
started automatically. Dockerfile, toolchain or dependency changes require a full
image rebuild rather than an app-only hot patch. No container workflow changes
the startup-fixed one-to-four active-request contract.
