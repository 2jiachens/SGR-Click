## Granularity Recovery

This folder contains the granularity recovery training and runtime code used by the
current GUI launcher and evaluation pipeline.

Files:

- `build_granularity_recovery_dataset.py`
  Builds the SBD-derived fine-tuning dataset for the recovery head. The generated
  samples contain first-click SimpleClick outputs, DINOv3 slider masks, base
  teachers, high-granularity recovery teachers, and progressive training teachers.
- `dataset.py`
  Loads the generated `.npz` records for training.
- `model.py`
  Lightweight single-network U-Net used for the full granularity range.
- `inference.py`
  Runtime inference wrapper for the single granularity-recovery network.
- `train_granularity_recovery.py`
  Training entrypoint. By default it writes checkpoints to
  `experiments/granularity_recovery`, matching the GUI/evaluation checkpoint path.
- `granularity_mapping.py`
  Splits the display slider into hierarchy-space and model-space granularity values.
- `gui_controller.py`
  Applies one granularity-conditioned network over the full slider range.
- `gui_app.py`
  Standalone Tk GUI.
- `demo_granularity_recovery.py`
  Runnable GUI entrypoint.

Purpose:

- data generation: fine-tuning + high-granularity recovery, not evaluation-set
  extraction
- runtime: preserve low-granularity local behavior and recover toward the whole
  object at high granularity

The standalone evaluation rules remain under `evaluate/`.
