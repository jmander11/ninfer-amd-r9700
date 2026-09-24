#!/usr/bin/env bash
# Exact invocation for the C1/P129+G27 DFlash K4/W5 after-parity screen.
# Run from the package directory. Requires the retained 6fe53d53 build and the R9700.
set -euo pipefail
cd "$(dirname "$0")"

# Non-mutating preflight (validates plan, build receipt, artifact, corpus, DSO, PCI, power, CPU tests)
/usr/bin/python3 run.py --preflight-only

# Full execution (creates results/, runs both arms, analyzes, closes)
/usr/bin/python3 run.py
