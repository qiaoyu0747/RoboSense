#!/usr/bin/env bash
set -euo pipefail
: "${ROBOSENSE_CLOUD_MODEL:=Qwen/Qwen2.5-Omni-7B}"
: "${ROBOSENSE_OUTPUT_ROOT:?Set ROBOSENSE_OUTPUT_ROOT}"
dataset="${1:?Usage: train_cloud.sh DATASET}"
export ROBOSENSE_MODEL="$ROBOSENSE_CLOUD_MODEL"
export ROBOSENSE_RUN_DIR="$ROBOSENSE_OUTPUT_ROOT/$dataset/cloud_7b_ft"
python -m robosense.cli train --role cloud --config "configs/$dataset.yaml"
