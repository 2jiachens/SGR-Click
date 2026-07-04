#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/weights/simpleclick_models/cocolvis_vit_base.pth}"
DINOV3_REPO="${DINOV3_REPO:-$PROJECT_ROOT/third_party/dinov3}"
DINOV3_MODEL="${DINOV3_MODEL:-dinov3_vitl16}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-$PROJECT_ROOT/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
SBD_ROOT="${SBD_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/SBD_granularity_refine_full}"
EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/granularity_refine}"
PREVIEW_OUTPUT_ROOT="${PREVIEW_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/SBD_granularity_refine_preview}"
PREVIEW_MAX_IMAGES="${PREVIEW_MAX_IMAGES:-2}"
PREVIEW_PER_SPLIT="${PREVIEW_PER_SPLIT:-2}"
LOG_EVERY_IMAGES="${LOG_EVERY_IMAGES:-25}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server/run_granularity_refine_pipeline.sh [preview|build|train|all]

Required environment variables:
  SBD_ROOT       Path to benchmark_RELEASE/dataset on the server.

Optional environment variables:
  PROJECT_ROOT, PYTHON_BIN, DEVICE, CHECKPOINT, DINOV3_REPO, DINOV3_MODEL, DINOV3_WEIGHTS
  OUTPUT_ROOT, PREVIEW_OUTPUT_ROOT, EXP_ROOT, PREVIEW_MAX_IMAGES, PREVIEW_PER_SPLIT
  LOG_EVERY_IMAGES, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_NUM_WORKERS
EOF
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ -z "$path" ]]; then
    echo "Missing required path: $label" >&2
    exit 1
  fi
  if [[ ! -e "$path" ]]; then
    echo "Path does not exist for $label: $path" >&2
    exit 1
  fi
}

run_preview() {
  mkdir -p "$PREVIEW_OUTPUT_ROOT"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_sbd_granularity_refine_dataset.py" \
    --sbd-root "$SBD_ROOT" \
    --output-root "$PREVIEW_OUTPUT_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --dinov3-repo "$DINOV3_REPO" \
    --dinov3-model "$DINOV3_MODEL" \
    --dinov3-weights "$DINOV3_WEIGHTS" \
    --max-images-per-split "$PREVIEW_MAX_IMAGES" \
    --preview-per-split "$PREVIEW_PER_SPLIT" \
    --device "$DEVICE"
}

run_build() {
  mkdir -p "$OUTPUT_ROOT"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_sbd_granularity_refine_dataset.py" \
    --sbd-root "$SBD_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --dinov3-repo "$DINOV3_REPO" \
    --dinov3-model "$DINOV3_MODEL" \
    --dinov3-weights "$DINOV3_WEIGHTS" \
    --log-every-images "$LOG_EVERY_IMAGES" \
    --device "$DEVICE"
}

run_train() {
  mkdir -p "$EXP_ROOT"
  "$PYTHON_BIN" "$PROJECT_ROOT/train_granularity_refine.py" \
    --dataset-root "$OUTPUT_ROOT" \
    --exp-root "$EXP_ROOT" \
    --epochs "$TRAIN_EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --num-workers "$TRAIN_NUM_WORKERS" \
    --device "$DEVICE" \
    --save-vis
}

case "$STAGE" in
  preview|build|train|all)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 1
    ;;
esac

require_path "PROJECT_ROOT" "$PROJECT_ROOT"
require_path "SBD_ROOT" "$SBD_ROOT"
require_path "CHECKPOINT" "$CHECKPOINT"
require_path "DINOV3_REPO" "$DINOV3_REPO"
require_path "DINOV3_WEIGHTS" "$DINOV3_WEIGHTS"

if [[ "$STAGE" == "preview" || "$STAGE" == "all" ]]; then
  run_preview
fi

if [[ "$STAGE" == "build" || "$STAGE" == "all" ]]; then
  run_build
fi

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  run_train
fi
