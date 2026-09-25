#!/usr/bin/env bash
set -euo pipefail
predictions="${1:?Usage: evaluate.sh PREDICTIONS OUTPUT}"
output="${2:?Usage: evaluate.sh PREDICTIONS OUTPUT}"
python -m robosense.cli evaluate --predictions "$predictions" --output "$output"
