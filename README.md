# SGR-Click

**Semantic granularity recovery for click-based interactive segmentation.**

SGR-Click first recovers representative semantic scopes from a positive click,
then constructs a mask path between them. The granularity control `g` navigates
that path; **Next stage** moves directly to the next semantic anchor.

This repository provides both released models:

| Model | Semantic features | Semantic parameters | Total parameters |
| :--- | :--- | ---: | ---: |
| **SGR-Click-TF** | Frozen DINOv3 ViT-L/16 | 303.15M | 401.18M |
| **SGR-Click-Lite** | Trained head sharing the interaction backbone | 2.41M | 100.43M |

Both use a frozen SimpleClick ViT-B interaction backbone and Euclidean ordering
between anchors. Lite does not load DINOv3 at inference time.

## Installation

Python 3.10 is recommended. Install a CUDA-enabled PyTorch/torchvision pair
appropriate for your machine, then install this package:

```bash
git clone https://github.com/2jiachens/SGR-Click.git
cd SGR-Click
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install PyTorch and torchvision using https://pytorch.org/get-started/locally/
python -m pip install -e ".[dev]"
```

The code retains the SimpleClick checkpoint namespace (`isegm`) for compatibility.
Use a separate environment from UniGraCo, which defines the same namespace.
The desktop demo requires Tk; on Ubuntu, install `python3-tk` if it is absent.
See [the tested environment](docs/environment.md).

## Weights

The trained Lite semantic head is included in
[`checkpoints/sgr_click_lite.pt`](checkpoints/sgr_click_lite.pt).
Download the interaction backbone from the official SimpleClick model release.
TF additionally requires the official DINOv3 repository and ViT-L/16 checkpoint.

See **[weight preparation and SHA-256 checks](docs/weights.md)** before running.
Large third-party checkpoints and datasets are not bundled.

## Interactive demo

```bash
sgr-click demo --model lite
sgr-click demo --model tf
```

Open an image and left-click the target. Move the granularity slider to inspect
the recovery path, or use **Previous stage / Next stage** to select anchors.
Add positive clicks with the left mouse button and negative clicks with the
right mouse button to correct the selected mask. Undo restores the preceding
interaction state.

## Image inference

```bash
sgr-click predict --model lite --image image.jpg --click 240 160 \
  --granularity 0.6 --output outputs/example
```

Coordinates are `(x, y)` in the input image. Output contains `mask.png`,
`anchors.npz`, and `prediction.json` with the actual anchor granularities.
All commands accept explicit `--backbone`, `--student`, `--dino-repo`,
`--dino-weights`, and `--device` paths/options.

```python
from sgr_click.model import ModelPaths, build_model

controller, provider = build_model(
    "lite",
    ModelPaths("checkpoints/cocolvis_vit_base.pth", "checkpoints/sgr_click_lite.pt"),
)
controller.set_image(rgb_uint8_image)
controller.add_click(x, y, True)
controller.set_granularity(0.6)
mask = controller.current_object_prob > 0.5
provider.close()
```

## Reproduce evaluation

Configure dataset locations in [`configs/datasets.yaml`](configs/datasets.yaml).
Inference and evaluation use **the same `build_model` function**.

```bash
# Semantic coverage and slider diagnostics on the remaining evaluation subset
sgr-evaluate semantic --model lite --dataset PascalPart \
  --split remaining --output outputs/lite/semantic/pascalpart
sgr-evaluate semantic --model tf --dataset PartImageNet \
  --split remaining --output outputs/tf/semantic/partimagenet

# NoC@85/90: official ten-branch and best-first-101 oracle protocols
sgr-evaluate noc --model lite --dataset GrabCut --protocol both \
  --output outputs/lite/noc/grabcut
sgr-evaluate noc --model tf --dataset PascalPart --category head \
  --protocol both --output outputs/tf/noc/pascalpart_head
```

The evaluators save sample-level records, summaries, configuration, source hashes,
and weight hashes. Interrupted runs resume only with matching settings.
`--limit 1` is a smoke test; omit it for full evaluation.
`--shard-id` and `--num-shards` support disjoint workers.

**[Evaluation guide](docs/evaluation.md)** covers all datasets, sample selection,
metric definitions, failure accounting, NoC selection rules, and
summary commands. Oracle NoC measures GT-assisted scope selection, not measured
human interaction time.

## Repository layout

```text
checkpoints/           Released Lite head and weight manifest
configs/               Dataset path configuration
docs/                  Installation, weights, and evaluation instructions
licenses/              Third-party license notices
scripts/               Asset verification and batch evaluation
src/
  sgr_click/
    model.py           Shared TF/Lite construction
    runtime/           Semantic anchors, projection, and mask paths
    gui/               English interactive demo
    evaluation/        Data readers, metrics, and reproducible protocols
  isegm/               SimpleClick backbone and predictor compatibility code
tests/                 Path, cache, data, and protocol checks
```

Run tests with `python -m pytest`. They use synthetic inputs and do not require
model downloads. See [validation notes](docs/validation.md) for real-image
regression checks.

## Acknowledgments and license

Built on [SimpleClick](https://github.com/uncbiag/SimpleClick) and
[DINOv3](https://github.com/facebookresearch/dinov3). The ten-branch NoC
implementation and click simulator are attributed to
[UniGraCo](https://github.com/Zhao-Yian/UniGraCo).
See [THIRD_PARTY.md](THIRD_PARTY.md) for license boundaries and source attribution.
