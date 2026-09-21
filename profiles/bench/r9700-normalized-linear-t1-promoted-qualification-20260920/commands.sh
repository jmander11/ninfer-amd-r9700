#!/usr/bin/env bash
set -euo pipefail
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec /home/battlefront/.local/bin/python3.11 "$package/run.py" "$@"
