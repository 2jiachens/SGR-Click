## Granularity Recovery

This folder contains the maintained granularity-recovery runtime and training code
used by the current GUI launcher and evaluation pipeline.

Files:

- `build_granularity_recovery_dataset.py`
  Builds the SBD-derived fine-tuning dataset for the recovery network.
- `dataset.py`
  Loads the generated `.npz` records for training.
- `model.py`
  Lightweight single-network U-Net used across the full slider range.
- `inference.py`
  Runtime inference wrapper for the trained recovery model.
- `train_granularity_recovery.py`
  Training entrypoint. By default it writes checkpoints to
  `experiments/granularity_recovery`, matching the GUI/evaluation checkpoint path.
- `granularity_mapping.py`
  Splits the display slider into hierarchy-space and model-space granularity values.
- `gui_controller.py`
  Applies the trained granularity-recovery model over the full slider range.
- `gui_app.py`
  Standalone Tk GUI.
- `demo_granularity_recovery.py`
  Runnable GUI entrypoint.

Folder responsibilities:

- dataset generation for recovery training
- checkpoint training and validation
- runtime GUI inference
- slider-conditioned granularity control

The standalone evaluation rules remain under `evaluate/`.
