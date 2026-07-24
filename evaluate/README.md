# Evaluation

This directory stores the maintained evaluation protocol for the paper-facing
granularity recovery pipeline. It contains only code, rules, and fixed sample
lists. Datasets, checkpoints, and generated results are expected to live
outside the repository.

## Metrics

All targets are evaluated at multiple slider values unless a specific protocol
overrides the grid.

Reported metrics use the same per-instance aggregation rule adopted by the
project:

- `NoC@85`: for each object, take the minimum click count that reaches IoU
  `0.85` over all tested slider values, then average across objects.
- `NoC@90`: identical aggregation at IoU `0.90`.
- `IoU@1`: for each object, record the best first-click IoU over all tested
  slider values, then average across objects.

The default click cap is `20`.

## ESS-10 Protocol

`--ess10` enables a separate effective-semantic-state oracle for staircase
outputs. It scans the DINO hierarchy over the semantic slider range
`[0.5, 1.0]`, records the first display value of every distinct semantic stage,
and retains at most ten stages with the largest hierarchy-mask transitions.
The 0.5 root state and a distinct endpoint stage are retained. If fewer than
ten semantic stages exist, only the unused slots are filled uniformly below
0.5. Candidate construction does not inspect GT; GT is used only for the
standard first click and final metric computation.

ESS-10 writes a resumable `ess10.json` for each target and must use a separate
output directory:

```bash
python -u evaluate/run_evaluation.py \
  --task slider \
  --config-path evaluate/config_server_eval.yml \
  --checkpoint weights/simpleclick_models/cocolvis_vit_base.pth \
  --granularity-recovery-checkpoint experiments/granularity_recovery/best.pth \
  --ess10 --ess10-scan-step 0.01 --ess10-max-states 10 \
  --output-dir /root/autodl-tmp/eval_outputs/granularity_recovery_main_ess10_vitb
```

Keep the fixed `{0.1, ..., 1.0}` protocol for direct comparison with UniGraCo.
ESS-10 is a staircase-aware supplementary protocol unless the same state
selection rule is also applied to the competing method.

## Supported Targets

Standard datasets:

- `GrabCut`, `Berkeley`, `SBD`, `DAVIS`, `PascalVOC`
- Report `NoC@85`, `NoC@90`, and `IoU@1`.

PascalPart:

- `head`, `wheel`, `torso`, `all`
- Category subsets follow fixed sample keys.
- Part labels use the intended semantic target rather than finer descendants.

PartImageNet:

- `head`, `body`, `foot`, `all`
- `all` uses a fixed random-1000 sample list.

## Main Files

- `run_evaluation.py`: main evaluation entry for the current repository.
- `merge_slider_metrics.py`: merges partial runs and applies the best-slider
  aggregation rule.
- `config_server_eval.yml`: dataset and checkpoint paths for server-side runs.
- `protocols/`: predefined slider schedules and dataset protocols.
- `sample_keys/`: fixed sample subsets for reproducible part evaluation.
- `server_runs/`: command templates used for split server execution.

## Split Evaluation Notes

The repository keeps the previous multi-server split command templates for
large-scale runs. If a dataset is evaluated across several slider shards, merge
the partial outputs with `merge_slider_metrics.py` after all jobs finish.

Example:

```bash
python evaluate/merge_slider_metrics.py \
  outputs/server_slider_eval_vitl_part_body/PartImageNet__body \
  outputs/server_slider_eval_vitl_part_body_g080/PartImageNet__body \
  outputs/server_slider_eval_vitl_part_body_g090/PartImageNet__body \
  outputs/server_slider_eval_vitl_part_body_g100/PartImageNet__body
```

## External Artifacts

Before running evaluation, prepare:

- base interactive-segmentation checkpoints under `weights/simpleclick_models/`
- the trained recovery checkpoint under `experiments/granularity_recovery/`
- dataset roots referenced by `config_server_eval.yml`
- optional fixed evaluation subsets such as the PartImageNet export used in the
  paper experiments
