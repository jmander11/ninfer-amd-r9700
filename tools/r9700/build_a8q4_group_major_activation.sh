#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$repo"
mkdir -p tools/r9700/build
/opt/rocm/llvm/bin/clang++ --offload-device-only --offload-arch=gfx1201 -O3 -std=c++20 -x hip \
  tools/r9700/a8q4_group_major_activation_qual.hip -Itools/r9700 \
  -S -o tools/r9700/build/a8q4_group_major_activation_qual.s
PYTHON=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python
"$PYTHON" -m tools.r9700.check_a8q4_group_major_activation_static \
  --source tools/r9700/a8q4_group_major_activation_qual.hip \
  --assembly tools/r9700/build/a8q4_group_major_activation_qual.s
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx1201 -std=c++20 \
  tools/r9700/a8q4_group_major_activation_harness.hip \
  tools/r9700/a8q4_group_major_activation_qual.hip \
  src/ops/r9700/linear/r9700_linear.hip \
  src/ops/r9700/eager/eager_ops.hip \
  -Itools/r9700 -Isrc -Iinclude \
  -o tools/r9700/build/a8q4_group_major_activation_harness
