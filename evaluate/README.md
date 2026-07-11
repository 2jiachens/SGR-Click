# Evaluation Rules

This directory contains the evaluation protocol used for the three-server runs.
It is intentionally code-and-rule only: datasets, checkpoints, and generated
outputs are not stored here.

## Metrics

All targets are evaluated at slider values from 0% to 100% in 10% increments
unless a split protocol says otherwise.

The reported metrics are aggregated across sliders per image:

- `NoC@85`: for each image/object, take the best click count that reaches IoU
  0.85 across all evaluated sliders, then average over images.
- `NoC@90`: same rule at IoU 0.90.
- `IoU@1`: for each image/object, take the maximum first-click IoU across all
  evaluated sliders, then average over images.

The click cap is 20.

## Dataset Targets

Standard datasets:

- `GrabCut`, `Berkeley`, `SBD`, `DAVIS`, `PascalVOC`
- Report `NoC@85`, `NoC@90`, and `IoU@1`.

PascalPart:

- `head`, `wheel`, `torso`, `all`
- Class targets use exact largest-part samples. For example, `head` does not
  include finer child parts such as `eye`.
- `all` is a fixed random-1000 sample list.

PartImageNet:

- `head`, `body`, `foot`, `all`
- `all` is a fixed random-1000 sample list.

## Main Files

- `run_slider_evaluation.py`: evaluation runner.
- `merge_slider_metrics.py`: post-hoc metric merger that applies the per-image
  best-slider rule.
- `config_server_eval.yml`: server dataset path config.
- `protocols/`: slider protocols.
- `sample_keys/`: fixed target sample lists.
- `server_runs/`: command templates matching the final three-server split.

## Three-Server Split Used

The final distributed rule is:

- Server 1:
  - `SBD` head split: `g000-g040`
  - `PartImageNet/body`: `g000-g070`
  - `PartImageNet/body`: `g090`
- Server 2:
  - `PartImageNet/head`: all sliders
  - `PartImageNet/body`: `g080`
- Server 3:
  - `SBD` tail split: `g050-g100`
  - `PartImageNet/foot,all`: all sliders
  - `PartImageNet/body`: `g100`

After the split runs finish, merge `PartImageNet/body` with:

```bash
python evaluate/merge_slider_metrics.py \
  outputs/server_slider_eval_vitl_part_body/PartImageNet__body \
  outputs/server_slider_eval_vitl_part_body_g080/PartImageNet__body \
  outputs/server_slider_eval_vitl_part_body_g090/PartImageNet__body \
  outputs/server_slider_eval_vitl_part_body_g100/PartImageNet__body
```

For SBD ViT-L split results:

```bash
python evaluate/merge_slider_metrics.py \
  outputs/server_slider_eval_vitl/SBD \
  outputs/server_slider_eval_vitl_sbd_tail/SBD
```

## Required External Artifacts

The repository should not include large artifacts. Before running evaluation,
place these on the server:

- checkpoints under `weights/simpleclick_models/`
- granularity refiner under `experiments/granularity_refine_rerun/best.pth`
- datasets under the paths configured in `config_server_eval.yml`
- PartImageNet evaluation subset at `partimagenet_head_body_foot_eval/`

