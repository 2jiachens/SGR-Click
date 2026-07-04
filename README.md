# SimpleClick DINOv3 Granularity Recovery

This local workspace has been cleaned to keep only the code needed for the current GUI entry:

- `start_granularity_recovery_gui.bat`

The current runnable line is:

- original SimpleClick interactive backbone
- DINOv3 click-rooted semantic hierarchy
- base granularity refine checkpoint for low-granularity interaction feel
- granularity recovery checkpoint for high-granularity whole-object recovery

## Main entry

Run:

```powershell
start_granularity_recovery_gui.bat
```

Or check the launcher without starting the GUI:

```powershell
start_granularity_recovery_gui.bat --check
```

## Required local runtime files

Read:

- [docs/granularity_recovery_runtime_checklist.md](docs/granularity_recovery_runtime_checklist.md)

That file records:

- which local checkpoints must remain on disk
- which environment variables can override default paths
- which old files were intentionally removed from this workspace
- where the old backup repository lives

## Current code path

The minimal current path is:

- launcher: `start_granularity_recovery_gui.bat`
- GUI entry: `granularity_recovery/demo_granularity_recovery.py`
- GUI/controller: `granularity_recovery/gui_app.py`, `granularity_recovery/gui_controller.py`
- DINO hierarchy: `isegm/inference/dino_granularity.py`
- base branch inference: `isegm/inference/granularity_refine_inference.py`
- recovery branch inference: `granularity_recovery/inference.py`

## Old version backup

The deleted old line has already been backed up to:

- [2jiachens/simpleclick-dinov3](https://github.com/2jiachens/simpleclick-dinov3)
