#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
case "${1:-}" in
  dense-g16) group=16; sparse=OFF; tau=1000; suffix=-g16 ;;
  dense-g32) group=32; sparse=OFF; tau=900; suffix=-g32 ;;
  xattention-g16) group=16; sparse=ON; tau=900; suffix=-xattention-g16 ;;
  xattention-g32) group=32; sparse=ON; tau=900; suffix=-xattention-g32 ;;
  *) echo 'usage: build.sh dense-g16|dense-g32|xattention-g16|xattention-g32' >&2; exit 2 ;;
esac
readonly phase_build="build-r9700-phase-sum${suffix}-20260921"
test ! -e "$phase_build" && test ! -L "$phase_build"
cmake -S . -B "$phase_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_BUILD_R9700_CORE_QUALIFIER=ON \
  -DNINFER_R9700_KV_VALUE_GROUP="$group" \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
  -DNINFER_R9700_FP8_QK_WMMA=1 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION="$sparse" \
  -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE="$tau"
cmake --build "$phase_build" -j4 --target ninfer_bench \
  ninfer_r9700_runtime_planner_qual ninfer_bench_support_test
