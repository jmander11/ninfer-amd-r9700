#!/usr/bin/env bash
# Sourced by maintainer runners; this is trusted, private host configuration.
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/../.ninfer-local.env" ]]; then
  # shellcheck source=/dev/null
  source "$(dirname "${BASH_SOURCE[0]}")/../.ninfer-local.env"
fi
