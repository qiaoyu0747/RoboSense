#!/usr/bin/env bash
set -euo pipefail
: "${ROBOSENSE_EDGE_MODEL:=Qwen/Qwen2.5-Omni-3B}"
: "${ROBOSENSE_OUTPUT_ROOT:?Set ROBOSENSE_OUTPUT_ROOT}"
dataset="${1:?Usage: train_edge.sh DATASET}"
export ROBOSENSE_MODEL="$ROBOSENSE_EDGE_MODEL"
export ROBOSENSE_RUN_DIR="$ROBOSENSE_OUTPUT_ROOT/$dataset/edge_3b_ft"
python -m robosense.cli train --role edge --config "configs/$dataset.yaml"
