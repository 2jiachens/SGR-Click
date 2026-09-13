# SGR-Click

SGR-Click adds continuous semantic granularity control to interactive
segmentation through a single slider. The method reuses frozen pretrained
models and introduces **zero task-specific trainable parameters**; no
additional training is required.

## Pipeline

1. **SimpleClick** generates a click prior from the first positive click.
2. **DINOv3** builds a click-rooted semantic hierarchy via agglomerative
   clustering and greedy frontier expansion on position-debiased features.
   Low-evidence semantic additions are rejected with aggregate soft
   SimpleClick evidence, preventing background from entering before the object.
3. The **granularity slider** selects a cumulative hierarchy mask and hands the
   endpoint to a soft whole-object semantic completion field.
4. **SimpleClick boundary projection** refines the blocky DINOv3-patch
   mask into a pixel-accurate result using synthetic interior clicks.

## Quick Start

```bash
# 1. Create environment
conda create -n sgr-click python=3.10 -y && conda activate sgr-click
pip install -r requirements.txt

# 2. Clone DINOv3
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3

# 3. Download pretrained weights (see Model Weights below), then launch
start_granularity_recovery_no_training_gui.bat
```

## Model Weights

Two frozen pretrained backbones from their original authors:

| Component | Expected local path | Official source |
| --- | --- | --- |
| SimpleClick ViT-B | `weights/simpleclick_models/cocolvis_vit_base.pth` | [SimpleClick weights](https://drive.google.com/drive/folders/1zVhZefCjsTBxvyxnYMVnbkrNeRCH6y9Y) |
| DINOv3 ViT-L/16 | `weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | [Official DINOv3 repo](https://github.com/facebookresearch/dinov3) |

Override paths via `DINOV3_REPO` and `DINOV3_WEIGHTS` environment variables.

## Repository Layout

- `granularity_recovery/` — GUI entry, controller, and app
- `isegm/` — SimpleClick interaction, model, and utility code
- `isegm/inference/dino_granularity.py` — DINOv3 hierarchy construction (core algorithm)
- `evaluate/` — paper evaluation entrypoints
- `models/` — model config files for the interactive backbone

## Evaluation

```bash
python evaluate/run_evaluation.py

# Per-image visible-staircase NoC used by the default model
python evaluate/run_staircase_noc.py --datasets GrabCut,Berkeley
```

See `evaluate/README.md` for protocols and fixed sample lists.

Every maintained evaluation writes a `run_manifest.json` containing the exact
arguments, Git revision and dirty-worktree status, software/CUDA environment,
random seed, and SHA-256 identities of the protocol files and checkpoints.
See `docs/reproducibility.md` for the method boundary, ablation definitions,
and reporting checklist.
