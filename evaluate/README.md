# Paper Evaluation

This directory contains the maintained code and fixed sample lists required to
reproduce the experiments reported in the paper. Datasets, checkpoints, and
generated outputs are stored outside this directory.

## Experiments

### Standard ten-point evaluation

`run_evaluation.py` evaluates SGR-Click at the ten slider values
`{0.1, 0.2, ..., 1.0}` and reports NoC@85, NoC@90, and IoU@1. It supports the
five whole-object datasets, PascalPart targets, and PartImageNet targets used
by the paper.

```bash
python evaluate/run_evaluation.py \
  --config-path config.yml \
  --protocol standard_ten \
  --checkpoint weights/simpleclick_models/cocolvis_vit_base.pth \
  --gui-slider-protocol \
  --output-dir outputs/standard_ten_vitb
```

Use `cocolvis_vit_large.pth` for the ViT-L run. `merge_slider_metrics.py`
merges non-overlapping sample or slider shards without changing the
per-instance best-slider aggregation.

### Hierarchical semantic evaluation

The paper evaluates complete slider trajectories on fixed 1,000-sample subsets
of SPIN, PascalPart, and PartImageNet. The maintained runners archive 11 states
(`0.00-1.00` at a `0.10` interval) so that the finest endpoint remains available
for diagnostics. The main comparison with UniGraCo follows its official fixed
ten-point protocol and rescored the saved trajectories on `0.10-1.00`, without
new inference. The 21-state scan used only by the response visualization is
enabled explicitly with `--scan-step 0.05`. Metrics are LR@60, HSR@60, LR@70,
HSR@70, and mBIoU.

SGR-Click:

```bash
python evaluate/run_semantic_coverage.py \
  --dataset SPIN \
  --backbone ViT-B \
  --output-dir outputs/hierarchical_sgr_vitb_spin
```

Each control value is evaluated independently from the same click-conditioned
semantic hierarchy. Use `--traversal-order descending` only as an
order-invariance diagnostic; quantitative comparisons use ascending order.

UniGraCo:

Clone the [official UniGraCo repository](https://github.com/Zhao-Yian/UniGraCo)
to `third_party/UniGraCo` and download its official SimpleClick and LoRA
checkpoints before running the comparison:

```bash
python evaluate/run_unigraco_semantic_coverage.py \
  --dataset SPIN \
  --backbone ViT-B \
  --official-repo third_party/UniGraCo \
  --checkpoint weights/simpleclick_models/sbd_vit_base.pth \
  --lora-checkpoint weights/unigraco/GraCo_base_lora.pth \
  --output-dir outputs/hierarchical_unigraco_vitb_spin
```

Use `--sample-shard-id` and `--num-sample-shards` for deterministic,
non-overlapping distributed evaluation.

### Module ablations

`run_evaluation.py --ablation-mode` supports the training-free method's three
well-defined settings: `full`, `frozen_hierarchy` (without SimpleClick boundary
projection), and `simple_prior` (without the DINOv3 hierarchy). Select
`--protocol ablation_refinement` for the fixed `0.50` refinement experiment or
`--protocol ablation_expansion` for the fixed `1.00` expansion experiment.
`run_semantic_coverage.py --ablation-mode frozen_hierarchy` archives the
training-free semantic path on the same 11-state diagnostic grid; the paper's
ablation table is rescored on the fixed ten-point subset.
`analyze_semantic_path_ablation.py` performs the paired comparison with the
complete ViT-B model.

## Reproducibility records

Maintained runners use seed `0` by default (override with `--seed`) and write a
`run_manifest.json` before model loading. The manifest records the CLI,
normalized arguments, Git revision and dirty status, Python/PyTorch/CUDA/GPU
environment, and SHA-256 digests for available checkpoints and protocol files.
Keep this file with every reported result directory.

### Revision Round 3 experiments

`run_sam_multimask_semantic_coverage.py` evaluates the official Meta SAM
ViT-L model with one positive point and `multimask_output=True`. It reports
both the three native area-sorted candidates and their nearest-neighbor
repetition over 11 controls. Repetition does not create additional masks, so
coverage metrics for the two mappings are expected to match.

`profile_interactive_cost.py` measures first-click latency, slider-update
latency, peak allocated GPU memory, and frozen/trainable parameter counts. Run
each method/backbone combination in a separate process on the same GPU.

`summarize_revision_round3.py` validates paired sample keys, recomputes SOR
from saved trajectories, calculates paired bootstrap intervals, and writes the
SAM comparison, training-free component ablation, and cost-profile tables under
`outputs/05_revision_round3/summary`.

## Maintained Files

- `run_evaluation.py`: standard ten-point evaluation and module ablations.
- `merge_slider_metrics.py`: deterministic merge and metric aggregation.
- `run_semantic_coverage.py`: SGR-Click hierarchical semantic evaluation.
- `run_unigraco_semantic_coverage.py`: UniGraCo hierarchical semantic evaluation.
- `run_sam_multimask_semantic_coverage.py`: official SAM multimask baseline.
- `profile_interactive_cost.py`: same-device interactive cost profiler.
- `summarize_revision_round3.py`: Round 3 result validation and table export.
- `semantic_coverage.py`: LR, HSR, and mBIoU definitions and aggregation.
- `semantic_coverage_data.py`: SPIN, PascalPart, and PartImageNet hierarchy loaders.
- `paper_samples.json`: all fixed paper sample lists and their metadata.
- `paper_protocols.json`: the standard ten-point and two ablation protocols.
- `../config.yml`: portable dataset and checkpoint path configuration.

Generated results, caches, server queue scripts, deprecated protocols, and
unused comparison methods are intentionally excluded.
