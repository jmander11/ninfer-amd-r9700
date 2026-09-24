# Python 3.11 ROCm reference dependency plan

Research date: 2026-09-21. Initial research did not download or install wheels.
Root subsequently installed `/ssdpool2nvme/local_llm/.venv-ninfer-r9700-py311`:
`pip check` and CPU-only imports passed with Python 3.11.16, torch
2.9.1+rocm7.2.4.git39497456, Triton 3.5.1+rocm7.2.4.gita272dfa8,
numpy 2.4.6, and safetensors 0.8.0. Static target flags include gfx1201;
GPU remained uninitialized. GPU qualification has not run. Launch remains subject
to root review and exclusive GPU ownership.

## Exact available pair

AMD's official ROCm 7.2.4 directory lists both CPython 3.11 Linux x86-64 wheels:

| Distribution | Exact wheel | Published bytes |
| --- | --- | ---: |
| torch | torch-2.9.1+rocm7.2.4.lw.git39497456-cp311-cp311-linux_x86_64.whl | 1650547330 |
| triton | triton-3.5.1+rocm7.2.4.gita272dfa8-cp311-cp311-linux_x86_64.whl | 284445474 |

Both exact URLs answered HTTP 200 to HEAD requests and matched the index lengths.
These are the CP311 counterparts of the currently installed CP312 pair. The
installed torch distribution metadata requires exactly
`triton==3.5.1+rocm7.2.4.gita272dfa8` on Linux x86-64.

AMD ROCm 7.2.4's compatibility matrix lists gfx1201 and PyTorch 2.9.1. AMD's
Radeon Linux matrix also lists R9700, with PyTorch 2.9.1 and Triton 3.5.1 production
support in the documented 7.2.1 Radeon release. The installed CP312 counterpart's
static compiled-target query reports `gfx1201` without device initialization:

```text
torch.__version__: 2.9.1+rocm7.2.4.git39497456
torch._C._cuda_getArchFlags(): gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1200 gfx1201 gfx950 gfx1151 gfx1150
torch.cuda.is_initialized(): False before and after
```

Together these establish an official CP311 package candidate matching the existing
native-gfx1201 stack. The CP311 wheel's embedded code was not inspected because
wheel downloads were explicitly out of scope. Its import, compiled target list,
and actual deterministic scorer execution must still pass after installation.
No gfx override is needed or proposed; the scorer rejects architecture overrides.
Host ROCm 10 versus wheel-build ROCm 7.2.4 dynamic-library compatibility also remains
a runtime check, even though the installed matching CP312 build already imports
with the selected local library paths.

## Scorer dependencies

`tools/reference/qwen3_8_27b_bf16/requirements.txt` requires torch>=2.7,
safetensors>=0.5 and flash-linear-attention>=0.5.1. `backend.py` imports torch,
torch.nn.functional, `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`,
and `safetensors.safe_open`. The deterministic provenance additionally imports
Triton and records its lowering controls and FLA package/source identities.

Use the installed FLA versions: flash-linear-attention==0.5.2 and fla-core==0.5.2.
Their metadata permits Python>=3.10; FLA requires transformers>=4.45.0 and fla-core
requires einops. Existing versions transformers==5.16.1 and einops==0.8.2 satisfy
this. NumPy==2.4.6 is already verified under local Python 3.11; the installed
ROCm CP312 environment's NumPy 2.5.2 requires Python>=3.12 and must not be copied.
Safetensors==0.8.0 uses an abi3 wheel compatible with Python 3.11. Torch's ordinary
dependencies (filelock, typing-extensions, sympy, networkx, jinja2, fsspec) remain
resolved normally. This scorer does not require torchvision or torchaudio.

## Setup commands — retained installation recipe

Create an isolated environment only when selected chunk !=4096 and this dependency
installation is authorized.
Use direct wheel URLs in the same resolver transaction so a general-index torch
or Triton cannot silently replace the intended ROCm pair. Do not point PYTHONPATH
at the separate CUDA numerical-reference environment for this ROCm environment.

```bash
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
reference_env=/ssdpool2nvme/local_llm/.venv-ninfer-r9700-py311
test ! -e "$reference_env"
/home/battlefront/.local/bin/python3.11 -m venv "$reference_env"
"$reference_env/bin/python" -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp311-cp311-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.5.1%2Brocm7.2.4.gita272dfa8-cp311-cp311-linux_x86_64.whl' \
  'numpy==2.4.6' 'safetensors==0.8.0' \
  'flash-linear-attention==0.5.2' 'fla-core==0.5.2' \
  'einops==0.8.2' 'transformers==5.16.1'
"$reference_env/bin/python" -m pip check

# This import/static-target probe does not request a device or execute GPU math.
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
  "$reference_env/bin/python" - <<'PY'
import sys
from tools.reference.qwen3_8_27b_bf16.protocol import establish_deterministic_environment
establish_deterministic_environment()
import torch
import triton
import numpy
import safetensors
assert sys.version_info[:2] == (3, 11)
assert torch.version.hip
assert 'gfx1201' in torch._C._cuda_getArchFlags().split()
assert not torch.cuda.is_initialized()
print(sys.version, torch.__version__, triton.__version__, numpy.__version__, safetensors.__version__)
PY
```

FLA may itself inspect accelerator availability during import; therefore separately
check `from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule` only
when the root GPU lease allows it. The torch-only
static-target query was the actual no-device probe performed during this research;
FLA was inspected through installed metadata and source, not imported here.

After installation, only for selected chunk !=4096 and when the GPU is exclusively
available after review, execute the reference stage. It runs the mandatory
full-span GDN numerical probe before selected-chunk fresh-process BF16 A/B scoring:

```bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
reference_env=/ssdpool2nvme/local_llm/.venv-ninfer-r9700-py311
quality_package=profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921
env -u PYTHONPATH LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
  bash "$quality_package/commands.sh" reference --reference-python "$reference_env/bin/python"
```

Do not run a duplicate manual probe. The stage creates
`bf16-chunk{selected_chunk}-gdn-full-span.json` and refuses existing output,
including failed probes. The probe can exit zero with `all_pass=false`; the stage
explicitly gates `all_pass is True`, exact row extents `[4095,4096]`, and matching
interpreter path/hash provenance before either reference campaign.
Its independent sampled FP64 oracle is not replaced by the campaign's repeat
comparison. The reference stage already runs both 8K/32K A/B processes and the
exact sidecar comparison; no additional GDN determinism/PV diagnostic campaign is
needed. A changed interpreter/runtime cannot inherit the old reference execution
identity. If chunk4096 is selected, validated retained BF16 reuse still needs no
new ROCm Python environment or new probe.

## Official sources checked

- AMD exact wheel directory: https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/
- Torch CP311: https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp311-cp311-linux_x86_64.whl
- Triton CP311: https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.5.1%2Brocm7.2.4.gita272dfa8-cp311-cp311-linux_x86_64.whl
- AMD ROCm 7.2.4 compatibility: https://rocm.docs.amd.com/en/docs-7.2.4/compatibility/compatibility-matrix.html
- AMD Radeon Linux compatibility: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html
- AMD Radeon installation guidance: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html
- Official PyTorch ROCm 7.2 index, checked but not selected because its published versions differ from the installed 2.9.1 build: https://download.pytorch.org/whl/rocm7.2/torch/

Local pairing evidence: installed CP312 torch/triton `METADATA` and
`direct_url.json` under `/ssdpool2nvme/local_llm/.venv-ninfer-r9700/lib/python3.12/site-packages`.
