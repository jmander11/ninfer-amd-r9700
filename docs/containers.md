# R9700 containers and test runner

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

For benchmarks, set `NINFER_BUILD_BENCHMARKS=ON` when running setup, then build
the `ninfer_bench` target; its executable is `/build/bench/ninfer_bench`.
The runtime image contains the three applications, not the benchmark or test tools.

## Tests

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
