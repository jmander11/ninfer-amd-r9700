#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/post-terminal-focused-verification-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly resolver="$package/resolve.py"
readonly route="$package/route.pending.json"
readonly route_recheck="$package/route.recheck.pending.json"
readonly report_pending="$package/gpu-verification.pending.json"
readonly report="$package/gpu-verification.json"

if [[ $# -ne 1 || ( $1 != --run-cpu && $1 != --execute-gpu-verification ) ]]; then
  echo "usage: $0 --run-cpu|--execute-gpu-verification" >&2
  exit 2
fi
readonly mode=$1
cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
if [[ ! -f "$selection" ]]; then
  echo "validated terminal schema-v7 selection is not available: $selection" >&2
  exit 1
fi
: "${TEST_PYTHON:?set TEST_PYTHON to an explicit Python interpreter with pytest, Torch, and safetensors}"
[[ $TEST_PYTHON == /* && -f $TEST_PYTHON && -x $TEST_PYTHON ]]
# Preserve the absolute launcher: resolving a venv's bin/python symlink selects the base
# interpreter and silently discards pyvenv.cfg/package discovery.
readonly test_python=$TEST_PYTHON

python3 - "$route" "$route_recheck" "$report_pending" "$report" "$mode" <<'PY'
import sys
from pathlib import Path

from tools.bench.focused_verification_io import require_absent

paths = [Path(path) for path in sys.argv[1:3]]
if sys.argv[5] == "--execute-gpu-verification":
    paths.extend(Path(path) for path in sys.argv[3:5])
require_absent(paths)
PY
route_owner=
route_recheck_owner=
cleanup_transient_routes() {
  python3 - "$route" "$route_owner" "$route_recheck" "$route_recheck_owner" <<'PY'
import sys
from pathlib import Path

from tools.bench.focused_verification_io import unlink_if_owned

for raw_path, raw_owner in zip(sys.argv[1::2], sys.argv[2::2], strict=True):
    if raw_owner:
        device, inode = map(int, raw_owner.split(":"))
        unlink_if_owned(Path(raw_path), (device, inode))
PY
}
route_owner=$(python3 "$resolver" --selection "$selection" --test-python "$test_python" --out "$route")
[[ $route_owner =~ ^[0-9]+:[0-9]+$ ]]
trap cleanup_transient_routes EXIT

mapfile -d '' -t selected < <(python3 - "$route" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in (value["build_directory"], value["execution_profile"]["xattention_profile"]):
    print(item, end="\0")
PY
)
[[ ${#selected[@]} -eq 2 ]]
readonly build=${selected[0]}
readonly attention=${selected[1]}
[[ -d "$build" ]]

host_targets=(
  ninfer_public_api_test ninfer_fp8_int4_kv_oracle_test ninfer_xattention_oracle_test
  ninfer_artifact_reader_test ninfer_tensor_test ninfer_admission_policy_test
  ninfer_sampling_defaults_test ninfer_gdn_replay_records_test
  ninfer_r9700_linear_prefill_dispatch_test ninfer_r9700_fp8_activation_contract_test
  ninfer_r9700_fp8_execution_state_contract_test ninfer_qwen3_runtime_mechanisms_test
  ninfer_qwen3_score_index_test ninfer_cli_options_test ninfer_bench_support_test
)
schema_targets=(
  ninfer_openai_schema_test ninfer_responses_schema_test ninfer_response_store_test
  ninfer_anthropic_schema_test ninfer_tool_call_parser_test ninfer_serve_options_test
  ninfer_request_log_test ninfer_http_error_handler_test
)
device_names=(
  core artifact kv_capacity registry target_binding kv linear state speculative_round eager gdn
  full_attention target_variant_gdn mtp_round runtime_planner engine_boundary
)
if [[ $mode == --execute-gpu-verification && $attention == b128-s16-tau900 ]]; then
  device_names+=(xattention)
fi
device_targets=()
for name in "${device_names[@]}"; do device_targets+=("ninfer_r9700_${name}_qual"); done

# Selection builds may initially contain only ninfer_bench. Materialize exactly the focused test
# targets before asking CTest to prove their availability; never build or run the broad test graph.
build_targets=("${host_targets[@]}" "${schema_targets[@]}")
if [[ $mode == --execute-gpu-verification ]]; then
  build_targets+=("${device_targets[@]}")
fi
cmake --build "$build" -j4 --target "${build_targets[@]}"

readonly host_regex='^ninfer_(public_api|fp8_int4_kv_oracle|xattention_oracle|artifact_reader|tensor|admission_policy|sampling_defaults|gdn_replay_records|r9700_linear_prefill_dispatch|r9700_fp8_activation_contract|r9700_fp8_execution_state_contract|qwen3_runtime_mechanisms|qwen3_score_index|cli_options|bench_support)_test$'
readonly schema_regex='^ninfer_(openai_schema|responses_schema|response_store|anthropic_schema|tool_call_parser|serve_options|request_log|http_error_handler)_test$'

require_count() {
  local regex=$1 expected=$2
  local count
  count=$(ctest --test-dir "$build" -N -R "$regex" | sed -n 's/^  Test  *#[0-9][0-9]*: /x/p' | wc -l)
  [[ $count -eq $expected ]]
}
require_count "$host_regex" 15
require_count "$schema_regex" 8
ctest --test-dir "$build" --output-on-failure -R "$host_regex"
ctest --test-dir "$build" --output-on-failure -R "$schema_regex"

NINFER_RUN_R9700_CODEC_TESTS=0 "$test_python" -m pytest \
  tests/artifact/test_container.py \
  tests/artifact/test_layouts.py \
  tests/test_bench_matrix.py \
  tests/test_serve_corpus.py \
  tools/bench/test_post_terminal_focused_verification.py \
  tools/bench/test_focused_verification_io.py \
  tools/bench/test_run_ninfer_bench_matrix.py \
  tools/ppl/test_assemble_pareto.py \
  tools/ppl/test_pareto.py

if [[ $mode == --run-cpu ]]; then
  sha256sum --check --strict "$package/prepared.sha256"
  route_recheck_owner=$(python3 "$resolver" --selection "$selection" --test-python "$test_python" --out "$route_recheck")
  [[ $route_recheck_owner =~ ^[0-9]+:[0-9]+$ ]]
  cmp --silent "$route" "$route_recheck"
  exit 0
fi

device_alt=$(IFS='|'; echo "${device_names[*]}")
device_regex="^ninfer_r9700_(${device_alt})_qual$"
# CTest automatically adds the 13 target-binding fixture producers required by the selected
# target_binding qualifier. They are part of the artifact-integration contract, not a broad sweep.
require_count "$device_regex" "$(( ${#device_names[@]} + 13 ))"
ctest --test-dir "$build" --output-on-failure -R "$device_regex"
sha256sum --check --strict "$package/prepared.sha256"
route_recheck_owner=$(python3 "$resolver" --selection "$selection" --test-python "$test_python" --out "$route_recheck")
[[ $route_recheck_owner =~ ^[0-9]+:[0-9]+$ ]]
cmp --silent "$route" "$route_recheck"
python3 - "$route" "$package/prepared.sha256" "$report_pending" "$report" "$(( ${#device_names[@]} + 13 ))" <<'PY'
import sys
from pathlib import Path

from tools.bench.focused_verification_io import publish_success

publish_success(
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
)
PY
