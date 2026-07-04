# Granularity Refine Training Plan

## Goal

Train a lightweight refinement module for the first-click granularity slider without changing the original SimpleClick multi-click logic.

The module learns:

- input: image + first-click SimpleClick response + current DINOv3 slider mask + granularity scalar
- target: the improved mask obtained after one automatic corrective click

This matches the observed behavior in the current system:

- the slider result is sometimes unnatural
- one extra positive or negative click often fixes it dramatically

## Dataset split

Use only `SBD`.

- `train`: official `train.txt` minus a deterministic `10%` carve-out
- `val`: deterministic `10%` carve-out from official `train.txt`
- `test`: official `val.txt`

Why:

- validation stays disjoint from training
- official SBD validation remains untouched as the final held-out test split
- this is close to the "train on SBD only" setting you wanted

## Processed sample definition

Each processed sample is one SBD instance object.

Stored fields:

- `gt_mask`: binary ground-truth object mask
- `simple_prob`: first-click SimpleClick probability
- `heuristic_masks`: current DINOv3 slider masks at fixed granularities
- `corrected_probs`: pseudo teacher after one automatic corrective click
- `corrective_clicks`: the auto-generated corrective click for each granularity
- `positive_click_map`: Gaussian map of the first positive click
- `effective_level_mask`: which granularity slots are genuinely distinct
- `effective_level_indices`: the distinct granularity indices kept for supervision
- `num_effective_levels`: number of distinct levels in this sample
- `positive_click`: `[x, y]`
- `granularities`: slider values used in this sample

## Trainable module

`GranularityRefineUNet`

- input channels: `7`
- channels:
  - RGB: `3`
  - first-click SimpleClick probability: `1`
  - current heuristic slider mask: `1`
  - first-click Gaussian map: `1`
  - granularity scalar map: `1`
- output: refined mask logits

## Loss

The current training code uses:

- soft BCE to `corrected_probs`
- Dice loss to thresholded teacher mask
- small BCE term to `gt_mask`

This keeps the model close to:

- the pseudo-improved mask
- the true object extent

## Preview dataset generation

```powershell
conda run -n simpleclick-dinov3 python scripts\build_sbd_granularity_refine_dataset.py `
  --sbd-root D:\数据集\SBD\benchmark_RELEASE\dataset `
  --output-root D:\数据集\SBD_granularity_refine_preview `
  --max-images-per-split 2 `
  --preview-per-split 2 `
  --device cuda:0
```

## Full dataset generation

```powershell
conda run -n simpleclick-dinov3 python scripts\build_sbd_granularity_refine_dataset.py `
  --sbd-root D:\数据集\SBD\benchmark_RELEASE\dataset `
  --output-root D:\数据集\SBD_granularity_refine_full `
  --log-every-images 25 `
  --device cuda:0
```

Notes:

- rerun the same command to resume from existing `.npz` files
- use `--overwrite` only when you want to rebuild the output directory from scratch

## Training

Do not start yet unless the processed dataset looks correct.

```powershell
conda run -n simpleclick-dinov3 python train_granularity_refine.py `
  --dataset-root D:\数据集\SBD_granularity_refine_full `
  --exp-root D:\数据集\experiments\granularity_refine `
  --epochs 20 `
  --batch-size 2 `
  --device cuda:0 `
  --save-vis
```

## Expected integration after training

Inference should remain:

- first positive click: SimpleClick + DINOv3 slider + refine head
- second and later clicks: original SimpleClick multi-click path

So the trained module only replaces the unstable one-click slider output, not the proven multi-click behavior.
