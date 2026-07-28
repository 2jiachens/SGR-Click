# Granularity Recovery Runtime Checklist

This file records the artifacts required to run the maintained SGR-Click GUI.

## 0. Verified Local Snapshot

Verified in the current local workspace:

- `weights/simpleclick_models/cocolvis_vit_base.pth`
- `weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `experiments/granularity_recovery/best.pth`
- `experiments/granularity_recovery/latest.pth`
- `third_party/dinov3`

The launcher check currently resolves to:

- recovery checkpoint: `experiments/granularity_recovery/best.pth`

## 1. Local Files That Must Stay On Disk

The two backbone checkpoints are required locally and intentionally not pushed
to Git. The SGR-Click recovery checkpoint is included in the repository.

### Base Interactive Segmentation Checkpoint

- `weights/simpleclick_models/cocolvis_vit_base.pth`

Purpose:

- Original SimpleClick one-click / multi-click segmentation backbone.

### DINOv3 Weights

- `weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

Purpose:

- Builds the click-rooted semantic hierarchy used by the granularity slider.

### Granularity Recovery Checkpoint

Preferred path:

- `experiments/granularity_recovery/best.pth`

Fallback path:

- `experiments/granularity_recovery/latest.pth`

Purpose:

- Drives the single-network granularity recovery path across the slider range.

### Local DINOv3 Repository

Preferred path:

- `third_party/dinov3`

Alternative:

- set environment variable `DINOV3_REPO`

Purpose:

- Provides the DINOv3 code used by `isegm/inference/dino_granularity.py`.

## 2. Optional Environment Overrides

You can override the default runtime paths with:

- `DINOV3_REPO`
- `DINOV3_WEIGHTS`
- `GRANULARITY_RECOVERY_CHECKPOINT`

## 3. Minimal Runtime Entry

The current maintained runtime path is:

- GUI entry: `granularity_recovery/demo_granularity_recovery.py`
- GUI logic: `granularity_recovery/gui_app.py`
- controller: `granularity_recovery/gui_controller.py`
- DINO hierarchy builder: `isegm/inference/dino_granularity.py`
- recovery inference: `granularity_recovery/inference.py`
- model definition: `granularity_recovery/model.py`
