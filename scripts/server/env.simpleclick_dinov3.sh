#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/code/simpleclick+dinov3}"
DATA_ROOT="${DATA_ROOT:-$HOME/code/data}"

resolve_sbd_root() {
  local candidates=(
    "$DATA_ROOT/SBD/dataset"
    "$DATA_ROOT/SBD/benchmark_RELEASE/dataset"
    "$DATA_ROOT/SBD"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  printf '%s\n' "$DATA_ROOT/SBD"
}

export PROJECT_ROOT
export PYTHON_BIN="${PYTHON_BIN:-python}"
export DEVICE="${DEVICE:-cuda:0}"
export CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/weights/simpleclick_models/cocolvis_vit_base.pth}"
export DINOV3_REPO="${DINOV3_REPO:-$PROJECT_ROOT/third_party/dinov3}"
export DINOV3_MODEL="${DINOV3_MODEL:-dinov3_vitl16}"
export DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-$PROJECT_ROOT/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
export SBD_ROOT="${SBD_ROOT:-$(resolve_sbd_root)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/SBD_granularity_refine_full}"
export PREVIEW_OUTPUT_ROOT="${PREVIEW_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/SBD_granularity_refine_preview}"
export EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/granularity_refine}"
export PREVIEW_MAX_IMAGES="${PREVIEW_MAX_IMAGES:-2}"
export PREVIEW_PER_SPLIT="${PREVIEW_PER_SPLIT:-2}"
export LOG_EVERY_IMAGES="${LOG_EVERY_IMAGES:-25}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"

cat <<EOF
PROJECT_ROOT=$PROJECT_ROOT
SBD_ROOT=$SBD_ROOT
OUTPUT_ROOT=$OUTPUT_ROOT
EXP_ROOT=$EXP_ROOT
CHECKPOINT=$CHECKPOINT
DINOV3_REPO=$DINOV3_REPO
DINOV3_MODEL=$DINOV3_MODEL
DINOV3_WEIGHTS=$DINOV3_WEIGHTS
DEVICE=$DEVICE
EOF
