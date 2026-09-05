#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 0 ]] || { echo 'run.sh accepts no arguments' >&2; exit 2; }

readonly root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$root/profiles/bench/r9700-n16k16-q4-cu-mask-causal-repeat1-prepare-20260905"
readonly plan="$package/plan.json"
readonly validator="$package/validate.py"
readonly power=/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level
readonly bench="$root/build-r9700-dense-selection-g16/bench/ninfer_bench"
readonly artifact="$root/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer"
readonly corpus="$root/bench/fixtures/bench_corpus.ids"
readonly peak="$root/tools/r9700/build/q4_hardware_peak_probe"
readonly rocprof=/opt/rocm/core-10.0/bin/rocprofv3
readonly python=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python

cd "$root"
[[ -z "${HSA_CU_MASK+x}" ]] || { echo 'HSA_CU_MASK must be unset before launching the package' >&2; exit 2; }
sha256sum --check --strict "$package/prepared.sha256"
"$python" "$validator" --plan "$plan" --preflight

for arm in full64 half32 quarter16; do
  for leaf in "peak-$arm.json" "benchmark-$arm.json" "power-$arm-before.txt" \
              "power-$arm-after.txt" "raw-$arm"; do
    [[ ! -e "$package/$leaf" && ! -L "$package/$leaf" ]] || {
      echo "refusing existing output $package/$leaf" >&2; exit 2;
    }
  done
done
[[ ! -e "$package/result.json" && ! -L "$package/result.json" ]] || {
  echo 'refusing existing final result' >&2; exit 2;
}
[[ "$(cat "$power")" == auto ]]
trap '[[ "$(cat "$power")" == auto ]]' EXIT

run_arm() {
  local arm=$1 mask=$2
  local -a mask_env=(env -u HSA_CU_MASK)
  if [[ -n "$mask" ]]; then mask_env=(env "HSA_CU_MASK=$mask"); fi
  [[ "$(cat "$power")" == auto ]]
  printf '%s\n' auto >"$package/power-$arm-before.txt"
  "${mask_env[@]}" "$peak" --out-json "$package/peak-$arm.json" \
    --seconds 0.2 --trials 5 --code-gib 0.25
  "${mask_env[@]}" "$rocprof" --selected-regions -f csv rocpd \
    -d "$package/raw-$arm" -o "cu-mask-$arm" --marker-trace --kernel-trace \
    --kernel-include-regex 'a8q4g64_linear_prefill_cta_kernel' -- \
    "$bench" --weights "$artifact" --corpus "$corpus" --device 0 \
      --concurrency 1 -p 2048 --prefill-chunk 4096 --draft-tokens 0 \
      --output json --output-file "$package/benchmark-$arm.json" \
      -r 1 --warmup 1 --profile-measured
  [[ "$(cat "$power")" == auto ]]
  printf '%s\n' auto >"$package/power-$arm-after.txt"
}

run_arm full64 ''
run_arm half32 '0:0-31'
run_arm quarter16 '0:0-15'

sha256sum --check --strict "$package/prepared.sha256"
"$python" "$validator" --plan "$plan" --output "$package/result.json"
[[ "$(cat "$power")" == auto ]]
trap - EXIT
