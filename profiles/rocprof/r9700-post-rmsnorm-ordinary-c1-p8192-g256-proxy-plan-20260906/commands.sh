#!/usr/bin/env bash
set -euo pipefail

plan_dir=/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/rocprof/r9700-post-rmsnorm-ordinary-c1-p8192-g256-proxy-plan-20260906
power=/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level
bench=/ssdpool2nvme/local_llm/ninfer-amd-r9700/build-r9700-rmsnorm-production-final-20260906/bench/ninfer_bench
artifact=/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer
corpus=/ssdpool2nvme/local_llm/ninfer-amd-r9700/bench/fixtures/bench_corpus.ids
current_after=

test "$(sha256sum "$bench" | cut -d' ' -f1)" = a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425
test "$(sha256sum "$artifact" | cut -d' ' -f1)" = 60719c5d5bfe978376c7de94db460bcd82d965ec447870cc47f0a02bf8d59953
test "$(sha256sum "$corpus" | cut -d' ' -f1)" = 27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002
test "$(sha256sum /ssdpool2nvme/local_llm/ninfer-amd-r9700/tools/r9700/build/hbm_bandwidth_probe | cut -d' ' -f1)" = 060e110bb3bf69678d3e81b52f3c3c457d17770f8210e2e165f3494169ade51b
test "$(sha256sum /opt/rocm/core-10.0/bin/rocprofv3 | cut -d' ' -f1)" = 2d220ded2167097e480e6866a916b575548d56bed8a823a4e505f7133868f0a6
test "$(sha256sum /opt/rocm/core-10.0/bin/rocprofv3-avail | cut -d' ' -f1)" = b079e32c759b3564a7750b7c274697537f2e2894f76ddaa7f17f139325571d75
test "$(cat "$power")" = auto
sudo -v

restore_auto() {
  printf '%s\n' auto | sudo tee "$power" >/dev/null
  test "$(cat "$power")" = auto
  if [[ -n "$current_after" ]]; then printf '%s\n' auto >"$current_after"; fi
}
trap restore_auto EXIT

# Preflight both bounded counter sets under the required profiling power state.
printf '%s\n' profile_standard | sudo tee "$power" >/dev/null
test "$(cat "$power")" = profile_standard
/opt/rocm/bin/rocprofv3-avail --device 0 pmc-check SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAVE_CYCLES SQ_WAVES GL2C_HIT GL2C_MISS TCP_REQ TCP_REQ_MISS GL2C_EA_RDREQ GRBM_GUI_ACTIVE TA_TA_BUSY GL2C_MC_WRREQ_STALL
/opt/rocm/bin/rocprofv3-avail --device 0 pmc-check SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM SQ_INSTS_LDS SQ_INSTS_TEX_LOAD SQ_INSTS_TEX_STORE SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_WAVES GRBM_GUI_ACTIVE TA_TA_BUSY
restore_auto

run_pass() {
  local label=$1; shift
  local report="$plan_dir/benchmark-$label.json"
  local raw="$plan_dir/raw-$label"
  local before="$plan_dir/power-$label-before.txt"
  local after="$plan_dir/power-$label-after.txt"
  current_after=$after
  test ! -e "$raw" && test ! -e "$report" && test ! -e "$before" && test ! -e "$after"
  test "$(cat "$power")" = auto
  printf '%s\n' profile_standard | sudo tee "$power" >/dev/null
  test "$(cat "$power")" = profile_standard
  printf '%s\n' profile_standard >"$before"
  /opt/rocm/core-10.0/bin/rocprofv3 --selected-regions -f csv rocpd -d "$raw" -o "post-rmsnorm-$label" --marker-trace --kernel-trace --pmc "$@" -- \
    "$bench" --weights "$artifact" --corpus "$corpus" --device 0 --concurrency 1 --whole-pg 8192,256 --prefill-chunk 4096 --draft-tokens 0 --retain-token-ids --output json --output-file "$report" -r 1 --warmup 1 --profile-measured
  restore_auto
  current_after=
}

run_pass cache-wait SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAVE_CYCLES SQ_WAVES GL2C_HIT GL2C_MISS TCP_REQ TCP_REQ_MISS GL2C_EA_RDREQ GRBM_GUI_ACTIVE TA_TA_BUSY GL2C_MC_WRREQ_STALL
run_pass issue-lds SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM SQ_INSTS_LDS SQ_INSTS_TEX_LOAD SQ_INSTS_TEX_STORE SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_WAVES GRBM_GUI_ACTIVE TA_TA_BUSY
restore_auto

# Same-session auto-state streaming ceiling; this is not inference traffic.
current_after="$plan_dir/power-stream-after.txt"
test ! -e "$plan_dir/stream-probe.stdout" && test ! -e "$plan_dir/stream-probe.stderr"
test ! -e "$plan_dir/power-stream-before.txt" && test ! -e "$current_after"
test "$(cat "$power")" = auto
printf '%s\n' auto >"$plan_dir/power-stream-before.txt"
/ssdpool2nvme/local_llm/ninfer-amd-r9700/tools/r9700/build/hbm_bandwidth_probe --size-gib 4 --trials 5 >"$plan_dir/stream-probe.stdout" 2>"$plan_dir/stream-probe.stderr"
test "$(cat "$power")" = auto
printf '%s\n' auto >"$current_after"
current_after=
trap - EXIT
python3 /ssdpool2nvme/local_llm/ninfer-amd-r9700/tools/bench/analyze_post_rmsnorm_decode_proxy.py --plan "$plan_dir/plan.json" --root "$plan_dir" --out "$plan_dir/analysis.json"
