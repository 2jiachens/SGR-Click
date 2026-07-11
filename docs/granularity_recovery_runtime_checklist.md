# Granularity Recovery Runtime Checklist

This workspace has been cleaned to keep only the files needed to run:

- `start_granularity_refine_rerun_gui.bat`

The old SimpleClick-only GUI, the old non-rerun granularity-refine checkpoint line, and the independent semantic-growth experiment line were removed from this workspace. The older SimpleClick code line was backed up to GitHub:

- [2jiachens/simpleclick-dinov3](https://github.com/2jiachens/simpleclick-dinov3)

## 0. Verified current local snapshot

Verified in the current local workspace:

- `weights/simpleclick_models/cocolvis_vit_base.pth`
- `weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `experiments/granularity_refine_rerun/best.pth`
- `experiments/granularity_refine_rerun/latest.pth`
- `third_party/dinov3`

The launcher check currently resolves to:

- rerun base checkpoint: `experiments/granularity_refine_rerun/best.pth`

## 1. Local files that must stay on disk

These files are required locally and are intentionally not pushed to git:

### Base interactive segmentation checkpoint

- `weights/simpleclick_models/cocolvis_vit_base.pth`

Purpose:

- Original SimpleClick one-click / multi-click segmentation backbone.

### DINOv3 weights

- `weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

Purpose:

- Builds the click-rooted semantic hierarchy used by the granularity slider.

### Rerun base granularity refine checkpoint

Preferred path:

- `experiments/granularity_refine_rerun/best.pth`

Fallback path:

- `experiments/granularity_refine_rerun/latest.pth`

Purpose:

- Refines the low-granularity branch with the rerun checkpoint so that the slider keeps the original SimpleClick-style local interaction feel while avoiding the older base checkpoint line.

### Local DINOv3 repository

Preferred path:

- `third_party/dinov3`

Alternative:

- set environment variable `DINOV3_REPO`

Purpose:

- Provides the DINOv3 code used by `isegm/inference/dino_granularity.py`.

## 2. Optional environment overrides

You can override the default runtime paths with:

- `SIMPLECLICK_DINOV3_ENV`
- `DINOV3_REPO`
- `DINOV3_WEIGHTS`
- `GRANULARITY_REFINE_RERUN_CHECKPOINT`

## 3. Minimal runtime entry

The cleaned workspace is centered on:

- launcher: `start_granularity_refine_rerun_gui.bat`
- GUI entry: `granularity_recovery/demo_granularity_recovery.py`
- GUI logic: `granularity_recovery/gui_app.py`
- controller: `granularity_recovery/gui_controller.py`
- DINO hierarchy builder: `isegm/inference/dino_granularity.py`
- base refine inference: `isegm/inference/granularity_refine_inference.py`

## 4. If you ever need the old version again

Use the GitHub backup instead of trying to recover deleted local files:

- old launcher: `start_simpleclick_gui.bat`
- old no-refine launcher: `start_simpleclick_gui_no_refine_vitl.bat`
- old dataset builder: `scripts/build_sbd_granularity_refine_dataset.py`
- old trainer: `train_granularity_refine.py`

Backup repository:

- [2jiachens/simpleclick-dinov3](https://github.com/2jiachens/simpleclick-dinov3)
