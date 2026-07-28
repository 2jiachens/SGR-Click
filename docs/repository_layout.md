# Repository Layout

This document describes the cleaned repository structure kept for the paper
release.

## Top-Level Directories

- `granularity_recovery/`: the maintained granularity-recovery pipeline,
  including dataset generation, training, inference, and GUI control logic.
- `interactive_demo/`: interactive application components used by the local GUI.
- `isegm/`: base interactive segmentation framework and dataset wrappers used
  by both the SimpleClick baseline and the recovery extension.
- `models/`: backbone and model definitions required by the base system.
- `evaluate/`: evaluation entrypoints, fixed sample lists, and reusable
  evaluation protocols.
- `experiments/`: the released SGR-Click recovery checkpoint and ignored local
  training artifacts.
- `weights/`: external pretrained weights such as DINOv3 and SimpleClick
  checkpoints. Keep locally; do not rely on them being versioned in Git.
- `outputs/`: generated predictions, evaluation dumps, and temporary visual
  exports.
- `docs/`: runtime, training, and repository documentation.
- `paper/`: manuscript sources and figure assets.
- `third_party/`: vendored or minimally wrapped third-party components required
  by the current code path.

## Top-Level Files

- `config.yml`: local runtime configuration.
- `requirements.txt`: Python dependency list.
- `README.md`: paper-oriented project overview and quick start.

## What To Keep Locally

For a runnable local environment, the minimal external artifacts are:

- `weights/simpleclick_models/*.pth`
- `weights/dinov3/*.pth`
- `experiments/granularity_recovery/best.pth` (included)

If you want a lighter public repository, keep code and docs in Git, but store
large checkpoints and datasets outside the repository tree or distribute them
separately.
