#!/usr/bin/env bash
set -euo pipefail
predictions="${1:?Usage: adapt_edge.sh PREDICTIONS ROUTER OUTPUT_DIR CLOUD_THRESHOLD}"
router="${2:?Usage: adapt_edge.sh PREDICTIONS ROUTER OUTPUT_DIR CLOUD_THRESHOLD}"
output="${3:?Usage: adapt_edge.sh PREDICTIONS ROUTER OUTPUT_DIR CLOUD_THRESHOLD}"
threshold="${4:?Usage: adapt_edge.sh PREDICTIONS ROUTER OUTPUT_DIR CLOUD_THRESHOLD}"
python -m robosense.cli adapt --predictions "$predictions" --router "$router" \
  --output-dir "$output" --cloud-threshold "$threshold" --budget 0.25 --positive-only
