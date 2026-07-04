# Legacy Backup Guide

This document records what each launcher and training script means in the current local backup snapshot.
It is intended as a stable reference before the workspace moves to a newer direction.

## 1. Main code lines in this backup

There are two different lines in this repository:

- `granularity_refine`
  This is the older and more practical line built on top of the original `demo.py` GUI.
  It keeps the original SimpleClick interaction style and adds a learned refinement module on top of
  the one-click DINOv3 granularity hierarchy.
- `granularity_recovery`
  This is a later experimental line for "semantic pivot + whole-object recovery".
  It is kept here as an isolated experiment, but it is not the main legacy version.

If you later clean the workspace and only want to keep one old baseline, the recommended baseline in this backup is:

- GUI: `start_simpleclick_gui.bat`
- Dataset builder: `scripts/build_sbd_granularity_refine_dataset.py`
- Trainer: `train_granularity_refine.py`

## 2. Launcher scripts

### `start_simpleclick_gui.bat`

Meaning:

- Main legacy GUI launcher.
- Base interactive predictor is the original SimpleClick checkpoint:
  - `weights/simpleclick_models/cocolvis_vit_base.pth`
- If present, it additionally loads:
  - `experiments/granularity_refine/best.pth`

Behavior:

- First positive click builds a DINOv3 hierarchy.
- The slider controls granularity growth.
- The optional `granularity_refine` checkpoint makes the intermediate slider results look more like
  corrected SimpleClick outputs instead of raw DINO masks.

Recommended use:

- This is the old version to preserve if you want a backup before moving on to the new project direction.

### `start_simpleclick_gui_no_refine_vitl.bat`

Meaning:

- Ablation launcher for the same GUI without the learned granularity refine model.
- Uses:
  - SimpleClick base checkpoint
  - DINOv3 ViT-L weights
- Explicitly disables the refine module with `--no-granularity-refine-model`.

Behavior:

- Lets you observe the raw one-click DINOv3 granularity hierarchy.
- Useful for checking how much improvement comes from the trained refine model.

### `start_granularity_recovery_gui.bat`

Meaning:

- Launcher for the later experimental `granularity_recovery` line.
- Uses:
  - SimpleClick base checkpoint
  - a base granularity checkpoint from `experiments/granularity_refine`
  - a recovery checkpoint from `experiments/granularity_recovery`

Behavior:

- Splits the slider into a low-granularity preservation stage and a high-granularity recovery stage.
- This is not the original legacy line. It is a separate experiment kept for reference.

## 3. Training and dataset scripts

### `scripts/build_sbd_granularity_refine_dataset.py`

Meaning:

- Builder for the old `granularity_refine` dataset.
- Takes SBD instance masks and generates:
  - first-click SimpleClick probability
  - DINOv3 runtime heuristic masks at multiple slider values
  - corrected teacher masks and probabilities

Output format:

- Matches `SBD_granularity_refine_full`
- NPZ keys include:
  - `simple_prob`
  - `heuristic_masks`
  - `corrected_probs`
  - `granularities`
  - `positive_click_map`

### `train_granularity_refine.py`

Meaning:

- Trainer for the old `granularity_refine` model only.
- It consumes the dataset produced by `scripts/build_sbd_granularity_refine_dataset.py`.

Model meaning:

- The network learns to map:
  - image
  - first-click SimpleClick probability
  - DINOv3 heuristic mask
  - click map
  - granularity value
  to a corrected slider result.

Use this trainer with:

- dataset root like `SBD_granularity_refine_full`
- checkpoints saved to something like `experiments/granularity_refine_*`

### `granularity_recovery/build_granularity_recovery_dataset.py`

Meaning:

- Builder for the later experimental recovery line.
- Creates base teachers, recovery teachers, and training teachers around a semantic pivot.

This is not the old area-ratio-style backup dataset builder.

### `granularity_recovery/train_granularity_recovery.py`

Meaning:

- Trainer for the experimental recovery branch.
- Expects the new `granularity_recovery_dataset` format.

Do not mix it with `SBD_granularity_refine_full`.

## 4. Core code files

### `demo.py`

- Entry point for the old main GUI.
- Can run:
  - pure SimpleClick
  - SimpleClick + DINOv3 hierarchy
  - SimpleClick + DINOv3 hierarchy + granularity refine checkpoint

### `isegm/inference/dino_granularity.py`

- Builds the one-click DINOv3 semantic hierarchy used by the slider.

### `isegm/inference/granularity_refine_inference.py`

- Loads the old `granularity_refine` checkpoint at inference time.

### `isegm/model/granularity_refine_model.py`

- U-Net style model used in the old learned slider correction branch.

### `isegm/data/datasets/granularity_refine.py`

- Dataset loader for the old `granularity_refine` NPZ format.

## 5. Weights that stay local and are not pushed

These are intentionally excluded from git and should stay on local disk:

- `weights/simpleclick_models/cocolvis_vit_base.pth`
- `weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- any files under `experiments/` such as:
  - `experiments/granularity_refine/best.pth`
  - `experiments/granularity_refine/latest.pth`
  - `experiments/granularity_recovery/best.pth`

If you later rebuild a clean workspace, restore these paths first before running the launchers.

## 6. What this backup is for

This GitHub backup is meant to preserve:

- the old SimpleClick + DINOv3 granularity GUI line
- the old dataset builder and trainer for `granularity_refine`
- the later isolated `granularity_recovery` experiment for reference

It is not meant to archive:

- local training outputs
- cached weights
- server export packages
- paper drafts
- temporary evaluation folders
