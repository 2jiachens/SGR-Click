## Granularity Recovery

This folder now keeps only the runtime files required by the current GUI launcher:

- `start_granularity_refine_rerun_gui.bat`

Files:

- `granularity_mapping.py`
  Splits the display slider into hierarchy-space and model-space granularity values.
- `gui_controller.py`
  Keeps the rerun base-granularity feel for `g <= 0.5`
  and uses the same base branch over the full slider range.
- `gui_app.py`
  Standalone Tk GUI.
- `demo_granularity_recovery.py`
  Runnable GUI entrypoint.

The older dataset builders, trainers, and legacy GUI launchers were removed from this local workspace
after being backed up to GitHub. The backup repository is recorded in:

- `docs/granularity_recovery_runtime_checklist.md`
