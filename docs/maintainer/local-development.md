# Local maintainer setup

Copy `.ninfer-local.env.example` to `.ninfer-local.env` and set your Python 3.11 interpreter, BF16 tokenizer directory, R9700 PCI address, DRM render node and shared GPU lock. These host settings stay out of Git and Docker builds. The file is shell code; only source a copy you control.

```sh
cp .ninfer-local.env.example .ninfer-local.env
# Edit the paths and physical device identity before use.
source .ninfer-local.env
mkdir -p "$(dirname "$NINFER_GPU_LOCK")"
flock --exclusive "$NINFER_GPU_LOCK" command-to-run-on-the-r9700
```

Use the same lock path across every checkout sharing the card. Keep the lock file; removing it breaks coordination with existing processes. Builds do not need the GPU lock. The master-gates and GPU-check runners load the local file and require `NINFER_GPU_LOCK`. The unit-test runner loads it too; wrap GPU/real runs in the shared flock shown above. An optional ignored `AGENTS.local.md` can describe host-specific prerequisites for local agents.

Set `NINFER_R9700_PCI_BUS_ID` to the R9700's actual domain:bus:device.function; physical qualifiers require it. `NINFER_DRM_RENDER_NODE` selects the matching DRM render device. Benchmark tools derive the power-profile path from that PCI address or render node. `NINFER_POWER_PROFILE` can name an explicit sysfs power attribute. These settings identify the device; they do not change clocks, voltage or power policy. Record the measured policy with performance results.

`NINFER_OFFICIAL_TOKENIZER_DIR` is required for tests using the official tokenizer. Without it, those resource-dependent sections report a skip; synthetic tokenizer coverage still runs. Promotion must supply prerequisites and reject skipped selected tests. For manual CTest runs, export the variable before starting CTest.

Conversion and campaign tools take checkpoint/interpreter paths as explicit arguments. Use their `--help`; historical `profiles/` inputs are local prerequisites, not bundled defaults or downloads. Restore old evidence from the private archive using the instructions in `../../bench/results/historical-evidence.json` when needed. Docker Compose has its own ignored `.env`; do not replace an existing deployment's model or cache paths when copying the public example.

Keep raw profiler databases local: they capture process environments, hostnames, machine identifiers and connection/session metadata. Publish reviewed summaries and small anonymized fixtures. Keep private audit reports, redaction rules and recovery bundles outside the repository.
