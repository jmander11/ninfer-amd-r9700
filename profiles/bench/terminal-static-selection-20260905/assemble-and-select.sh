#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/terminal-static-selection-20260905"
readonly post_chunk="$repo/profiles/bench/post-chunk-twelve-candidate-20260905"
readonly selection="$repo/profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"
readonly capacity_validation="$post_chunk/executed-capacity-validation.json"
readonly assembled="$repo/profiles/bench/pareto-input-post-promotion-20260905.json"
readonly result="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly pending_assembled="$package/pareto-input.pending.json"
readonly pending_result="$package/pareto-result.pending.json"

if [[ $# -ne 6 ]]; then
  echo "usage: $0 ALL_Q4_DENSE_QUALITY ALL_Q4_XATTENTION_QUALITY MIXED_DENSE_QUALITY MIXED_XATTENTION_QUALITY FOUR_ROLE_DENSE_QUALITY FOUR_ROLE_XATTENTION_QUALITY" >&2
  exit 2
fi
readonly all_q4_dense_quality=$(realpath -e "$1")
readonly all_q4_xattention_quality=$(realpath -e "$2")
readonly mixed_dense_quality=$(realpath -e "$3")
readonly mixed_xattention_quality=$(realpath -e "$4")
readonly four_role_dense_quality=$(realpath -e "$5")
readonly four_role_xattention_quality=$(realpath -e "$6")

cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
sha256sum --check --strict "$post_chunk/prepared.sha256"
python3 - "$assembled" "$result" "$pending_assembled" "$pending_result" <<'PY'
import sys
from pathlib import Path

from tools.ppl.terminal_selection_io import require_absent

require_absent([Path(argument) for argument in sys.argv[1:]])
PY
python3 -c 'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; validate_selection_record(Path("profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"))'
readonly selection_sha256=$(sha256sum "$selection" | cut -d ' ' -f1)

whole_arg() {
  python3 - "$capacity_validation" "$1" "$2" "$3" "$4" <<'PY'
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
identity = [sys.argv[2], int(sys.argv[3]), sys.argv[4]]
print(sys.argv[5] if identity in validation["whole_eligible_identities"] else "-")
PY
}

python3 tools/ppl/assemble_pareto.py \
  --require-xattention-dense-controls \
  --prefill-chunk-selection "$selection" \
  --post-chunk-capacity-validation "$capacity_validation" \
  --candidate dense-g16 r9700-q4g64-n16k16-eval 16 "$all_q4_dense_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-dense-all-q4-g16-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-n16k16-eval 16 dense "$repo/profiles/bench/pareto-whole-post-promotion-dense-all-q4-g16-receipt-bound-n16k16-20260905")" \
  --candidate dense-g32 r9700-q4g64-n16k16-eval 32 "$all_q4_dense_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-dense-all-q4-g32-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-n16k16-eval 32 dense "$repo/profiles/bench/pareto-whole-post-promotion-dense-all-q4-g32-receipt-bound-n16k16-20260905")" \
  --candidate xattention-g16 r9700-q4g64-n16k16-eval 16 "$all_q4_xattention_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-all-q4-g16-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-n16k16-eval 16 b128-s16-tau900 "$repo/profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-all-q4-g16-receipt-bound-n16k16-20260905")" \
  --candidate xattention-g32 r9700-q4g64-n16k16-eval 32 "$all_q4_xattention_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-all-q4-g32-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-n16k16-eval 32 b128-s16-tau900 "$repo/profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-all-q4-g32-receipt-bound-n16k16-20260905")" \
  --candidate mixed-dense-g16 r9700-q4-w8-mse-n16k16-eval 16 "$mixed_dense_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-dense-mixed-g16-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4-w8-mse-n16k16-eval 16 dense "$repo/profiles/bench/pareto-whole-post-promotion-dense-mixed-g16-receipt-bound-n16k16-20260905")" \
  --candidate mixed-dense-g32 r9700-q4-w8-mse-n16k16-eval 32 "$mixed_dense_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-dense-mixed-g32-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4-w8-mse-n16k16-eval 32 dense "$repo/profiles/bench/pareto-whole-post-promotion-dense-mixed-g32-receipt-bound-n16k16-20260905")" \
  --candidate mixed-xattention-g16 r9700-q4-w8-mse-n16k16-eval 16 "$mixed_xattention_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-mixed-g16-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4-w8-mse-n16k16-eval 16 b128-s16-tau900 "$repo/profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-mixed-g16-receipt-bound-n16k16-20260905")" \
  --candidate mixed-xattention-g32 r9700-q4-w8-mse-n16k16-eval 32 "$mixed_xattention_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-mixed-g32-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4-w8-mse-n16k16-eval 32 b128-s16-tau900 "$repo/profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-mixed-g32-receipt-bound-n16k16-20260905")" \
  --candidate four-role-dense-g16 r9700-q4g64-f8e4m3-four-role-n16k16-eval 16 "$four_role_dense_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-dense-four-role-g16-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-f8e4m3-four-role-n16k16-eval 16 dense "$repo/profiles/bench/pareto-whole-post-promotion-dense-four-role-g16-receipt-bound-n16k16-20260905")" \
  --candidate four-role-dense-g32 r9700-q4g64-f8e4m3-four-role-n16k16-eval 32 "$four_role_dense_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-dense-four-role-g32-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-f8e4m3-four-role-n16k16-eval 32 dense "$repo/profiles/bench/pareto-whole-post-promotion-dense-four-role-g32-receipt-bound-n16k16-20260905")" \
  --candidate four-role-xattention-g16 r9700-q4g64-f8e4m3-four-role-n16k16-eval 16 "$four_role_xattention_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-four-role-g16-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-f8e4m3-four-role-n16k16-eval 16 b128-s16-tau900 "$repo/profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-four-role-g16-receipt-bound-n16k16-20260905")" \
  --candidate four-role-xattention-g32 r9700-q4g64-f8e4m3-four-role-n16k16-eval 32 "$four_role_xattention_quality" "$repo/profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-four-role-g32-receipt-bound-n16k16-20260905" "$(whole_arg r9700-q4g64-f8e4m3-four-role-n16k16-eval 32 b128-s16-tau900 "$repo/profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-four-role-g32-receipt-bound-n16k16-20260905")" \
  --out "$pending_assembled"

# The terminal result must bind the eventual global input pathname, while both sets of bytes
# remain private until construction succeeds. Classification itself does not depend on that
# pathname; final validation below opens and recomputes the newly published input.
python3 - "$pending_assembled" "$assembled" "$pending_result" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.ppl.pareto import classify, load_payload

pending_input, published_input, pending_result = map(Path, sys.argv[1:])
source = load_payload(pending_input.read_text(encoding="utf-8"))
value = classify(source)
value["pareto_input"] = {
    "path": str(published_input.resolve()),
    "sha256": hashlib.sha256(pending_input.read_bytes()).hexdigest(),
}
pending_result.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
PY

# Hard links publish exact prepared bytes atomically and fail rather than overwriting a path
# created after the initial absence check. The terminal result is never visible until it has
# independently reopened and recomputed the published input.
sha256sum --check --strict "$package/prepared.sha256"
sha256sum --check --strict "$post_chunk/prepared.sha256"
[[ $(sha256sum "$selection" | cut -d ' ' -f1) == "$selection_sha256" ]]
python3 - "$pending_assembled" "$assembled" "$pending_result" "$result" "$selection" "$selection_sha256" <<'PY'
import sys
from pathlib import Path

from tools.ppl.terminal_selection_io import publish

publish(
    *(Path(argument) for argument in sys.argv[1:5]),
    guard_path=Path(sys.argv[5]), guard_sha256=sys.argv[6],
)
PY
python3 -c 'import json; from pathlib import Path; from tools.ppl.pareto import validate_terminal_production_authority; value=json.loads(Path("profiles/bench/pareto-result-post-promotion-20260905.json").read_text()); validate_terminal_production_authority(value)'
[[ $(sha256sum "$selection" | cut -d ' ' -f1) == "$selection_sha256" ]]
