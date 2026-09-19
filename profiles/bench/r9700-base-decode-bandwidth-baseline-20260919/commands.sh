#!/usr/bin/env bash
set -euo pipefail
root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-base-decode-bandwidth-baseline-20260919"
test "$PWD" = "$root"
case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    python3 "$package/validate.py"
    python3 "$package/preflight.py" --mode all
    ;;
  --measure)
    test "$#" -eq 1
    python3 "$package/validate.py"
    python3 "$package/preflight.py" --mode measure
    python3 "$package/run.py"
    ;;
  --profile)
    test "$#" -eq 1
    python3 "$package/validate.py"
    bash "$package/profile.sh"
    ;;
  *)
    echo "usage: $0 --preflight|--measure|--profile" >&2
    exit 2
    ;;
esac
