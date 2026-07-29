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
  --granularity-recovery-checkpoint experiments/granularity_recovery/best.pth \
  --gui-slider-protocol \
  --output-dir outputs/standard_ten_vitb
```

Use `cocolvis_vit_large.pth` for the ViT-L run. `merge_slider_metrics.py`
merges non-overlapping sample or slider shards without changing the
per-instance best-slider aggregation.

### Hierarchical semantic evaluation

The paper evaluates complete slider trajectories on fixed 1,000-sample subsets
of SPIN, PascalPart, and PartImageNet. Both methods scan `0.00-1.00` at a
`0.05` interval. Metrics are LR@60, HSR@60, LR@70, HSR@70, and mBIoU; all are
higher-is-better.

SGR-Click:

```bash
python evaluate/run_semantic_coverage.py \
  --dataset SPIN \
  --backbone ViT-B \
  --output-dir outputs/hierarchical_sgr_vitb_spin
```

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

`run_evaluation.py --ablation-mode` supports the paper's `simple_prior`,
`frozen_hierarchy`, `recovery_only`, and `full` settings. Select
`--protocol ablation_refinement` for the fixed `0.50` refinement experiment or
`--protocol ablation_expansion` for the fixed `1.00` expansion experiment.

## Maintained Files

- `run_evaluation.py`: standard ten-point evaluation and module ablations.
- `merge_slider_metrics.py`: deterministic merge and metric aggregation.
- `run_semantic_coverage.py`: SGR-Click hierarchical semantic evaluation.
- `run_unigraco_semantic_coverage.py`: UniGraCo hierarchical semantic evaluation.
- `semantic_coverage.py`: LR, HSR, and mBIoU definitions and aggregation.
- `semantic_coverage_data.py`: SPIN, PascalPart, and PartImageNet hierarchy loaders.
- `paper_samples.json`: all fixed paper sample lists and their metadata.
- `paper_protocols.json`: the standard ten-point and two ablation protocols.
- `../config.yml`: portable dataset and checkpoint path configuration.

Generated results, caches, server queue scripts, deprecated protocols, and
unused comparison methods are intentionally excluded.
