# SimpleClick DINOv3 Granularity Recovery

This repository contains the current code line for click-grounded semantic
granularity control built on top of SimpleClick and DINOv3.

The maintained runtime entry is:

- `start_granularity_recovery_gui.bat`

The current pipeline is:

- SimpleClick interactive backbone
- DINOv3 click-rooted semantic hierarchy
- a single granularity-recovery network for full-range slider control

## Repository Layout

- `granularity_recovery/`
  runtime GUI, dataset builder, training code, inference wrapper, and model
- `evaluate/`
  evaluation entrypoints, slider protocols, fixed sample lists, and server run templates
- `isegm/`
  SimpleClick-based interaction, model, and utility code
- `models/`
  model config files used by the interactive backbone
- `docs/`
  runtime checklist and training notes
- `paper/`
  local paper drafts and figures

Detailed directory notes:

- [docs/repository_layout.md](docs/repository_layout.md)

## Quick Start

Launch the GUI:

```powershell
start_granularity_recovery_gui.bat
```

Check the launcher without opening the GUI:

```powershell
start_granularity_recovery_gui.bat --check
```

## Required Local Artifacts

This repository does not include large checkpoints or datasets.

Required local files are documented in:

- [docs/granularity_recovery_runtime_checklist.md](docs/granularity_recovery_runtime_checklist.md)

That checklist records:

- which checkpoints must remain on disk
- which environment variables can override default paths
- where the local DINOv3 repository is expected

## Training and Data Generation

The main recovery training code lives in:

- [granularity_recovery/build_granularity_recovery_dataset.py](granularity_recovery/build_granularity_recovery_dataset.py)
- [granularity_recovery/train_granularity_recovery.py](granularity_recovery/train_granularity_recovery.py)
- [granularity_recovery/dataset.py](granularity_recovery/dataset.py)

Supporting notes:

- [docs/granularity_refine_training_plan.md](docs/granularity_refine_training_plan.md)
- [docs/server_training.md](docs/server_training.md)

## Evaluation

The maintained evaluation entry is:

- [evaluate/run_evaluation.py](evaluate/run_evaluation.py)

This directory also contains:

- fixed sample lists
- slider protocols
- result-merging scripts
- server command templates

See:

- [evaluate/README.md](evaluate/README.md)

## Backup of Older Code Line

An older backup repository remains available at:

- [2jiachens/simpleclick-dinov3](https://github.com/2jiachens/simpleclick-dinov3)
