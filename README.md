# SGR-Click

SGR-Click is a click-grounded semantic granularity recovery framework built on
top of SimpleClick and DINOv3. It provides continuous target-scope control
through a semantic granularity slider while retaining standard interactive
segmentation refinement.

The maintained runtime entry is
`granularity_recovery/demo_granularity_recovery.py`.

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

Create the environment and install the Python dependencies:

```bash
conda create -n sgr-click python=3.10 -y
conda activate sgr-click
pip install -r requirements.txt
```

Clone the official DINOv3 implementation:

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

Place the external checkpoints at the paths listed below, then launch the GUI:

```bash
python granularity_recovery/demo_granularity_recovery.py \
  --checkpoint weights/simpleclick_models/cocolvis_vit_base.pth
```

## Model Weights

The trained SGR-Click recovery checkpoint is included in this repository:

- `experiments/granularity_recovery/best.pth`
- SHA-256: `781cd9bf6cba3b6e2f37d4567d76f604baf71784d73992036ff539eaa91f1e9c`

The two large pretrained backbones are distributed by their original authors
and are not duplicated in this repository:

| Component | Expected local path | Official source |
| --- | --- | --- |
| SimpleClick ViT-B | `weights/simpleclick_models/cocolvis_vit_base.pth` | [SimpleClick model weights](https://drive.google.com/drive/folders/1zVhZefCjsTBxvyxnYMVnbkrNeRCH6y9Y?usp=sharing) |
| DINOv3 ViT-L/16 | `weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | [Official DINOv3 repository](https://github.com/facebookresearch/dinov3) |

Alternative locations can be supplied through `DINOV3_REPO`,
`DINOV3_WEIGHTS`, and `GRANULARITY_RECOVERY_CHECKPOINT`.

## Training and Data Generation

The main recovery training code lives in:

- [granularity_recovery/build_granularity_recovery_dataset.py](granularity_recovery/build_granularity_recovery_dataset.py)
- [granularity_recovery/train_granularity_recovery.py](granularity_recovery/train_granularity_recovery.py)
- [granularity_recovery/dataset.py](granularity_recovery/dataset.py)

Supporting notes:

- [docs/granularity_refine_training_plan.md](docs/granularity_refine_training_plan.md)

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

## Upstream and Project History

SGR-Click is developed from SimpleClick. The original upstream implementation
is available at:

- [uncbiag/SimpleClick](https://github.com/uncbiag/SimpleClick)
