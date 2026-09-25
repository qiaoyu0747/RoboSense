#!/usr/bin/env bash
set -euo pipefail
predictions="${1:?Usage: fit_router.sh PREDICTIONS OUTPUT}"
output="${2:?Usage: fit_router.sh PREDICTIONS OUTPUT}"
python -m robosense.cli fit-router --predictions "$predictions" --output "$output" --budget 0.25
