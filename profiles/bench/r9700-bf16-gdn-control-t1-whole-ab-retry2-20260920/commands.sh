#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry2-20260920"
python=/home/battlefront/.local/bin/python3.11
test "$PWD" = "$root"

case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    "$python" -B -c 'import ast, pathlib, sys; [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in sys.argv[1:]]' "$package/prepare.py" "$package/run.py"
    "$python" -B "$package/prepare.py" --preflight
    ;;
  --measure)
    test "$#" -eq 1
    "$python" -B -c 'import ast, pathlib, sys; [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in sys.argv[1:]]' "$package/prepare.py" "$package/run.py"
    "$python" -B "$package/prepare.py" --write-plan
    "$python" -B "$package/run.py"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
