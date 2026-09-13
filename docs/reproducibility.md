# Reproducibility and Method Scope

## Claim boundary

SGR-Click is a training-free composition of two **pretrained, frozen** models.
It has zero *task-specific trainable* parameters and requires no additional
optimization. This wording is intentionally distinct from “zero learned
parameters”: the SimpleClick and DINOv3 backbones contain parameters learned
during their original pretraining.

The maintained method consists of four stages:

1. a first positive click produces a SimpleClick probability prior;
2. frozen DINOv3 features define a click-rooted semantic hierarchy;
3. a scalar control selects a cumulative hierarchy state; and
4. frozen SimpleClick projects the selected state to a pixel-level boundary.

## Confirmatory protocols

- Use the fixed sample keys in `evaluate/paper_samples.json`.
- Use the slider states specified by `evaluate/paper_protocols.json`; hierarchical
  comparison uses the shared 11-state grid from 0.0 to 1.0.
- Choose thresholds before inspecting test-set results and report all samples,
  including the number and reasons for skipped cases.
- Compare methods on identical sample keys and semantic state budgets.
- Use ascending traversal as the primary protocol. Slider states are computed
  independently; descending traversal is retained only as an order-invariance
  diagnostic.

## Component ablations

| Setting | DINOv3 hierarchy | Boundary projection | Interpretation |
| --- | --- | --- | --- |
| `full` | yes | yes | complete training-free method |
| `frozen_hierarchy` | yes | no | contribution of pixel-boundary projection |
| `simple_prior` | no | no | contribution of semantic hierarchy control |

The deleted recovery U-Net is not part of the maintained method and must not be
reported as an SGR-Click component.

## Statistical reporting

Report point estimates with the number of valid and skipped samples. For method
comparisons, use paired resampling over the intersection of sample keys and
report a 95% confidence interval for the paired difference. Do not infer an
improvement from overlapping results produced on different subsets. Alongside
LR/HSR and mBIoU, report semantic-order rate, backward-area rate, and observable
state count so that coverage cannot hide an unstable slider trajectory.

## Run records

Each maintained runner seeds Python, NumPy, and PyTorch and disables cuDNN
benchmarking. It writes `run_manifest.json` with:

- command-line arguments and seed;
- Git commit and dirty-worktree status;
- Python, NumPy, PyTorch, CUDA, cuDNN, and GPU identity; and
- paths, byte sizes, and SHA-256 digests of available checkpoints and protocol
  files.

Archive the manifest with raw per-sample JSON, summaries, and table-generation
outputs. A dirty-worktree run is valid for development but should not be the
sole source of a final paper result unless the exact patch is also archived.
