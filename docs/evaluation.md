# Evaluation reproduction

## Data preparation

Use the original annotations. Edit `configs/datasets.yaml`; relative paths are
resolved against that file. Expected locations:

| Key | Contents |
| :--- | :--- |
| `pascalpart_annotations` | `*.mat` files from Pascal-Part `Annotations_Part` |
| `pascalpart_images` | Pascal VOC 2012 `JPEGImages` |
| `partimagenet_zip` | Original `PartImageNet_Seg.zip` |
| `spin_annotations` | `spin_test_subparts.json`, `spin_test_parts.json`, `spin_test_wholes.json` |
| `grabcut` | `data_GT/`, `boundary_GT/` |
| `berkeley` | `images/`, `masks/` |
| `davis` | DAVIS345 `images/`, `masks/` |
| `sbd` | SBD `benchmark_RELEASE/dataset` |
| `pascalvoc` | VOC2012 with images, segmentation masks, and split files |

PartImageNet reads ZIP members under `PartImageNet/images/test/` and
`PartImageNet/annotations/test/`. Its part IDs are **0–39** and background is
**40**. The whole target is the union of valid part labels; it is not the
separate whole-object PNG. SPIN reuses PartImageNet image files.

Fixed metadata is in `src/sgr_click/evaluation/paper_samples.json`.
Pascal-Part and PartImageNet categories refer to target groups, not aggregate
scores. In particular, `all` is a separate 1,000-target group.

## Semantic evaluation

Run both models on SPIN, PascalPart, and PartImageNet:

```bash
python scripts/evaluate_all.py --task semantic --model lite
python scripts/evaluate_all.py --task semantic --model tf
```

Each dataset contains 1,000 hierarchy examples. The development split is the
first 100 under ascending SHA-256 ordering of
`compact-mechanism-v1|DATASET|SAMPLE_KEY`; the remaining split contains 900.
The batch script uses `remaining`. Set `--split development` or `--split full`
to select the other pools. Keep development and remaining results separate.

For each example, an official distance-transform click simulator places the
first positive click in the finest target. The model builds one path and queries
all **101 values** `0, 0.01, ..., 1`. Predictions are thresholded at `0.5`;
there is no cumulative-union postprocessing.

- **mBIoU:** maximize IoU over queried masks for each annotated scope, then
  average over scopes within an example and over examples.
- **SOR:** for strictly nested, nonidentical target chains, choose the earliest
  IoU-maximizing query for each target. Count success only if all selected
  indices increase strictly. The main SOR macro-average uses Pascal-Part and
  PartImageNet; report SPIN separately.
- **SWR:** let each adjacent XOR pixel count be normalized by the union of all
  101 masks. Count five-step windows whose accumulated normalized change is
  at most `0.005`, with tolerance `1e-12`.
- **JSR:** fraction of adjacent steps with normalized change greater than
  `0.05 + 1e-12`.

The known invalid-seed construction failure remains a zero mBIoU example and,
when eligible, a zero SOR example. Its undefined SWR/JSR are excluded. Missing
data, dependency errors, and unexpected runtime errors stop the run.

Raw JSON metrics are fractions. Multiply by 100 for percentage tables.
The main mBIoU/SWR/JSR macro-average weights the three datasets equally.

## NoC evaluation

```bash
python scripts/evaluate_all.py --task noc --model lite
python scripts/evaluate_all.py --task noc --model tf
```

This covers 13 groups / 19,403 targets:

| Group | Targets |
| :--- | ---: |
| GrabCut / Berkeley / DAVIS / SBD / PascalVOC | 50 / 100 / 345 / 6,671 / 3,417 |
| PascalPart head / wheel / torso / all | 590 / 134 / 554 / 1,000 |
| PartImageNet head / body / foot / all | 2,029 / 2,316 / 1,197 / 1,000 |

All traces count the first physical click, allow up to 20 clicks, and report
NoC@85 and NoC@90 plus failure counts. Failure is recorded as 20 clicks, even
if a trace terminates earlier under branch pruning.

**Ten-branch oracle.** Restore the same first-click state for each
`g = 0.1, ..., 1.0`. Execute the UniGraCo oracle's branch-pruning and selection
rules: select the shortest executed trace, break ties by first-click IoU, then
by earlier branch order. Part targets use separate stopping thresholds for
85 and 90. Whole-object targets use the 90-threshold trace for both reports.
The discarded branch attempts are not charged to NoC.

**Best-first-101 oracle.** Scan the first-click masks over all 101 granularities.
Select the earliest maximum-GT-IoU mask, then correct that chosen mask with
subsequent clicks. No further granularity search occurs. The evaluator checks
that the selected mask actually reaches the predictor as the correction input.

Both are GT-assisted oracle protocols. They do not measure a user's time or
the number of attempts required to choose a granularity.

Use `--validate-branches` to additionally replay branches and compare selected
states with fresh first-click paths. This is useful for a small verification
run, but adds inference work.

## Summaries

Aggregate disjoint run folders:

```bash
sgr-summarize outputs/lite/semantic/remaining/* --output outputs/lite/semantic_summary.json
```

On shells without wildcard expansion, list the run folders explicitly.
The summarizer rejects duplicate targets and incompatible run configurations.
It reports the six-part-category NoC macro-average separately from dataset
tables; `all` groups and whole-object groups do not enter that macro-average.

`records.jsonl` contains per-target scores, selected granularities, traces,
click coordinates, and failure flags. `manifest.json` records input options,
metadata, source and checkpoint hashes. Reusing an output directory with
different settings raises an error.

## Comparison methods

Use the official [UniGraCo](https://github.com/Zhao-Yian/UniGraCo) and
[UnSAMv2](https://github.com/yujunwei04/UnSAMv2) releases in separate environments.
Match the sample metadata, original image resolution, first click, 101-query
grid, target definitions, and oracle rules above. The packaged evaluation
commands construct the two SGR-Click models; they do not substitute an SGR
controller for either comparison model.
