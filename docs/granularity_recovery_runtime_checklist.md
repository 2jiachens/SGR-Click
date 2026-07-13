# Granularity Recovery Runtime Checklist

This file records the local artifacts required to run the current maintained GUI:

- `start_granularity_recovery_gui.bat`

Older local experiment lines were removed from this repository. A backup of the
older code line remains available at:

- [2jiachens/simpleclick-dinov3](https://github.com/2jiachens/simpleclick-dinov3)

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

These files are required locally and are intentionally not pushed to git:

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

- `SIMPLECLICK_DINOV3_ENV`
- `DINOV3_REPO`
- `DINOV3_WEIGHTS`
- `GRANULARITY_RECOVERY_CHECKPOINT`

## 3. Minimal Runtime Entry

The current maintained runtime path is:

- launcher: `start_granularity_recovery_gui.bat`
- GUI entry: `granularity_recovery/demo_granularity_recovery.py`
- GUI logic: `granularity_recovery/gui_app.py`
- controller: `granularity_recovery/gui_controller.py`
- DINO hierarchy builder: `isegm/inference/dino_granularity.py`
- recovery inference: `granularity_recovery/inference.py`
- model definition: `granularity_recovery/model.py`

## 4. If You Need the Older Version Again

Use the GitHub backup instead of trying to recover deleted local files:

- old launcher: `start_simpleclick_gui.bat`
- old no-refine launcher: `start_simpleclick_gui_no_refine_vitl.bat`
- old dataset builder: `scripts/build_sbd_granularity_refine_dataset.py`
- old trainer: `train_granularity_refine.py`

Backup repository:

- [2jiachens/simpleclick-dinov3](https://github.com/2jiachens/simpleclick-dinov3)
