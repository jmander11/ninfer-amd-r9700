#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/low-context-selected-ladder-20260905"
readonly terminal_package="$repo/profiles/bench/terminal-static-selection-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly runner="$repo/tools/bench/run_ninfer_bench_matrix.py"
readonly validator="$repo/tools/bench/validate_low_context_prefill.py"
readonly corpus="$repo/bench/fixtures/bench_corpus.ids"
readonly ladder="$repo/profiles/bench/low-context-prefill-selected-20260905"
readonly evaluation="$repo/profiles/bench/low-context-prefill-evaluation-20260905.json"
readonly pending_route="$package/route.pending.json"
readonly pending_evaluation="$package/evaluation.pending.json"
readonly power=/sys/class/drm/card2/device/power_dpm_force_performance_level

if [[ $# -ne 1 || $1 != --execute-gpu-campaign ]]; then
  echo "usage: $0 --execute-gpu-campaign" >&2
  exit 2
fi
cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
sha256sum --check --strict "$terminal_package/prepared.sha256"
[[ -f "$selection" ]]
[[ ! -e "$ladder" && ! -L "$ladder" && ! -e "$evaluation" && ! -L "$evaluation" ]]
[[ ! -e "$pending_route" && ! -L "$pending_route" \
   && ! -e "$pending_evaluation" && ! -L "$pending_evaluation" ]]
test "$(cat "$power")" = auto

python3 - "$selection" "$pending_route" <<'PY'
import json
import sys
from pathlib import Path

from tools.bench.validate_low_context_prefill import resolve_selected_dense_route

selection, output = map(Path, sys.argv[1:])
value = resolve_selected_dense_route(selection)
with output.open("x", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2)
    handle.write("\n")
PY

mapfile -d '' -t route < <(python3 - "$pending_route" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in (
    value["artifact"]["path"], value["executable"]["path"],
    str(value["value_group"]), str(value["selected_prefill_chunk"]),
    value["weights_id"], value["hybrid_width_tool"] or "",
):
    print(item, end="\0")
PY
)
[[ ${#route[@]} -eq 6 ]]
readonly artifact=${route[0]}
readonly executable=${route[1]}
readonly value_group=${route[2]}
readonly selected_chunk=${route[3]}
readonly weights_id=${route[4]}
readonly hybrid_width_tool=${route[5]}
hybrid_args=()
if [[ $weights_id == r9700-q4g64-f8e4m3-four-role-eval ]]; then
  [[ -n $hybrid_width_tool ]]
  hybrid_args=(--require-fp8-hybrid --hybrid-width-tool "$hybrid_width_tool")
else
  [[ -z $hybrid_width_tool ]]
fi

python3 "$runner" --preset low-context-prefill \
  --bench "$executable" --no-build --weights "$artifact" --corpus "$corpus" \
  --prefill-chunk "$selected_chunk" --device 0 --concurrency 1 \
  --expected-kv-value-group "$value_group" \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 --expected-xattention-profile dense \
  "${hybrid_args[@]}" --output-dir "$ladder"
test "$(cat "$power")" = auto

validation_status=0
python3 "$validator" \
  --manifest "$ladder/manifest.json" --selection "$selection" \
  --executable "$executable" --artifact "$artifact" \
  --min-p2048-tok-s 2000 --out "$pending_evaluation" || validation_status=$?
# A complete authority below the explicit performance floor intentionally returns one. Any
# validation failure occurs before the exclusive output is written and therefore stays fatal.
[[ $validation_status -eq 0 || ( $validation_status -eq 1 && -f "$pending_evaluation" ) ]]
test "$(cat "$power")" = auto

python3 - "$pending_route" "$pending_evaluation" "$ladder/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from tools.bench.validate_low_context_prefill import validate_ladder

route_path, evaluation_path, manifest_path = map(Path, sys.argv[1:])
route = json.loads(route_path.read_text(encoding="utf-8"))
actual = json.loads(evaluation_path.read_text(encoding="utf-8"))
expected = validate_ladder(
    manifest_path, 2000.0, Path(route["executable"]["path"]),
    Path(route["artifact"]["path"]), Path(route["terminal_selection"]["path"]),
)
if actual != expected:
    raise SystemExit("pending low-context evaluation differs from fresh recomputation")
PY

# Publish only the complete, recomputed authority. The publisher fsyncs the private inode,
# hard-links it without replacement, revalidates the published bytes and every dependency, then
# fsyncs the parent directory. On failure it removes only the exact inode it observed.
python3 -m tools.bench.publish_low_context_prefill \
  --pending "$pending_evaluation" --out "$evaluation" \
  --manifest "$ladder/manifest.json" --selection "$selection" \
  --executable "$executable" --artifact "$artifact" \
  --min-p2048-tok-s 2000 --transient "$pending_route"
exit "$validation_status"
