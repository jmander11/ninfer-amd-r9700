# Real-model XAttention keep distributions

This opt-in target-private diagnostic measures real-model keep distributions, not speed or
numerical accuracy. It captures actual B64-page keep counts for every head/B128 query block
before shared workspace reuse. One eager C1 prefill covers all 16 full-attention layers per
selected 2048-token chunk. Existing BF16-source quality remains the numerical gate;
uninstrumented whole runs remain the timing gate. All diagnostic outputs are timing-ineligible.

Root may run these CPU commands after source review. Never rebuild frozen recovery or selection
executables for this diagnostic; the builds below are separate.

```sh
for group in 16 32; do
  cmake -S . -B "build-r9700-xattention-keep-diagnostic-g${group}-20260921" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_HIP_FLAGS=-DNINFER_R9700_XATTENTION_KEEP_DIAGNOSTIC=1 \
    -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_BUILD_R9700_CORE_QUALIFIER=ON \
    -DNINFER_R9700_KV_VALUE_GROUP="$group" \
    -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
    -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_R9700_XATTENTION_QUALIFICATION=ON \
    -DNINFER_R9700_XATTENTION_STRIDE=16 -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900
  cmake --build "build-r9700-xattention-keep-diagnostic-g${group}-20260921" --target ninfer_bench -j6
done
```

Read-only preflight compares compiled arithmetic options with the panel build, rejects stale
executables, replays selected-chunk/quality authorities, and binds exact artifacts/corpus:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/battlefront/.local/bin/python3.11 \
  tools/bench/xattention_keep_distribution.py preflight
```

After independent review, root alone launches eight serial diagnostics: all-Q4/four-role,
sparse G16/G32, P8192/P32768. Mixed XAttention is quality-ineligible and excluded.

```sh
PYTHONDONTWRITEBYTECODE=1 /home/battlefront/.local/bin/python3.11 \
  tools/bench/xattention_keep_distribution.py run \
  --output profiles/bench/r9700-xattention-real-keep-distribution-20260921
```

Output root and traces are create-only. Failed commands/logs/partial traces are retained, never
resumed. `NINFER_XATTENTION_KEEP_TRACE` enables the compile-gated hook; capture rejects Device
Graph streams. Validation requires complete dispatch coverage, causal keep-count bounds and
nontrivial sparse selection. Counts are retained B64 pages per shared B128 query block, not
individual query-token arithmetic or a roofline. Do not use diagnostic benchmark durations.
