## Granularity Recovery

This folder contains standalone files for the proposed
"base granularity preservation + high-granularity whole-object recovery"
direction.

Files:

- `build_granularity_recovery_dataset.py`
  Builds a dataset with:
  - first-click SimpleClick probability
  - runtime DINOv3 slider masks at multiple granularities
  - base teachers that preserve the original low-granularity interaction style
  - recovery teachers that expand high-granularity results toward the full GT
  - training teachers that switch from base to recovery supervision after `g > 0.5`
- `dataset.py`
  Loads both new-format and legacy-format generated datasets.
- `model.py`
  Standalone recovery U-Net used only for the high-granularity enhancement branch.
- `train_granularity_recovery.py`
  Standalone training entrypoint.
- `inference.py`
  Minimal checkpoint loader for batched inference.
- `gui_controller.py`
  Standalone click controller that keeps the original base-granularity feel for `g <= 0.5`
  and applies recovery enhancement for `g > 0.5`.
- `gui_app.py`
  Standalone Tk GUI built without modifying the existing demo.
- `demo_granularity_recovery.py`
  Runnable independent GUI entrypoint.

The implementation is intentionally isolated from the existing training
and inference pipeline. It imports the current repository code, but does
not modify any existing module.
