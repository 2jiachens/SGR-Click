from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granularity_recovery.dataset import GranularityRecoveryDataset


COMMON_REQUIRED_KEYS = {
    "gt_mask",
    "simple_prob",
    "global_level_masks",
    "global_level_ious_to_gt",
    "best_global_level_index",
    "completion_target_indices",
    "positive_click_map",
    "positive_click",
    "effective_level_mask",
    "effective_level_indices",
    "num_effective_levels",
    "granularities",
}

NEW_TEACHER_KEYS = {
    "base_teacher_probs",
    "base_teacher_masks",
    "base_corrective_clicks",
    "recovery_teacher_probs",
    "recovery_teacher_masks",
    "recovery_corrective_clicks",
    "training_teacher_probs",
    "training_teacher_masks",
}

LEGACY_TEACHER_KEYS = {
    "refine_teacher_probs",
    "refine_teacher_masks",
    "refine_corrective_clicks",
    "completion_teacher_probs",
    "completion_teacher_masks",
    "completion_corrective_clicks",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--sample-limit", type=int, default=0, help="0 means check every sample in each split.")
    parser.add_argument("--load-limit", type=int, default=8, help="How many samples per split to load via Dataset.")
    return parser.parse_args()


def ensure(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_rows(rows: list[dict], sample_limit: int) -> Iterable[dict]:
    if sample_limit <= 0:
        return rows
    return rows[: min(sample_limit, len(rows))]


def check_npz(sample_name: str, npz_path: Path):
    ensure(npz_path.exists(), f"{sample_name}: missing npz {npz_path}")
    blob = np.load(npz_path)
    missing_common = COMMON_REQUIRED_KEYS.difference(blob.files)
    ensure(not missing_common, f"{sample_name}: missing keys {sorted(missing_common)}")
    has_new = NEW_TEACHER_KEYS.issubset(blob.files)
    has_legacy = LEGACY_TEACHER_KEYS.issubset(blob.files)
    ensure(
        has_new or has_legacy,
        f"{sample_name}: missing teacher keys for both new and legacy formats",
    )

    gt_mask = blob["gt_mask"]
    simple_prob = blob["simple_prob"]
    click_map = blob["positive_click_map"]
    level_masks = blob["global_level_masks"]
    level_ious = blob["global_level_ious_to_gt"]
    base_probs = blob["base_teacher_probs"] if "base_teacher_probs" in blob.files else blob["refine_teacher_probs"]
    base_masks = blob["base_teacher_masks"] if "base_teacher_masks" in blob.files else blob["refine_teacher_masks"]
    base_clicks = blob["base_corrective_clicks"] if "base_corrective_clicks" in blob.files else blob["refine_corrective_clicks"]
    recovery_probs = blob["recovery_teacher_probs"] if "recovery_teacher_probs" in blob.files else blob["completion_teacher_probs"]
    recovery_masks = blob["recovery_teacher_masks"] if "recovery_teacher_masks" in blob.files else blob["completion_teacher_masks"]
    recovery_clicks = blob["recovery_corrective_clicks"] if "recovery_corrective_clicks" in blob.files else blob["completion_corrective_clicks"]
    training_probs = blob["training_teacher_probs"] if "training_teacher_probs" in blob.files else np.where(
        blob["granularities"][:, None, None] <= 0.5,
        base_probs,
        recovery_probs,
    )
    training_masks = blob["training_teacher_masks"] if "training_teacher_masks" in blob.files else (training_probs >= 0.49).astype(np.uint8)
    completion_target_indices = blob["completion_target_indices"]
    effective_level_mask = blob["effective_level_mask"]
    effective_level_indices = blob["effective_level_indices"]
    num_effective_levels = int(blob["num_effective_levels"])
    granularities = blob["granularities"]
    best_global_level_index = int(blob["best_global_level_index"])

    ensure(gt_mask.ndim == 2, f"{sample_name}: gt_mask must be HxW")
    ensure(simple_prob.shape == gt_mask.shape, f"{sample_name}: simple_prob shape mismatch")
    ensure(click_map.shape == gt_mask.shape, f"{sample_name}: click_map shape mismatch")

    ensure(level_masks.ndim == 3, f"{sample_name}: global_level_masks must be LxHxW")
    num_levels, height, width = level_masks.shape
    ensure((height, width) == gt_mask.shape, f"{sample_name}: level mask spatial shape mismatch")
    ensure(level_ious.shape == (num_levels,), f"{sample_name}: global_level_ious_to_gt shape mismatch")
    ensure(base_probs.shape == level_masks.shape, f"{sample_name}: base_teacher_probs shape mismatch")
    ensure(base_masks.shape == level_masks.shape, f"{sample_name}: base_teacher_masks shape mismatch")
    ensure(recovery_probs.shape == level_masks.shape, f"{sample_name}: recovery_teacher_probs shape mismatch")
    ensure(recovery_masks.shape == level_masks.shape, f"{sample_name}: recovery_teacher_masks shape mismatch")
    ensure(training_probs.shape == level_masks.shape, f"{sample_name}: training_teacher_probs shape mismatch")
    ensure(training_masks.shape == level_masks.shape, f"{sample_name}: training_teacher_masks shape mismatch")
    ensure(completion_target_indices.shape == (num_levels,), f"{sample_name}: completion_target_indices shape mismatch")
    ensure(effective_level_mask.shape == (num_levels,), f"{sample_name}: effective_level_mask shape mismatch")
    ensure(granularities.shape == (num_levels,), f"{sample_name}: granularities shape mismatch")
    ensure(0 <= best_global_level_index < num_levels, f"{sample_name}: best_global_level_index out of range")

    ensure(base_clicks.ndim == 3 and base_clicks.shape[0] == num_levels and base_clicks.shape[2] == 3,
           f"{sample_name}: base_corrective_clicks shape mismatch")
    ensure(recovery_clicks.ndim == 3 and recovery_clicks.shape[0] == num_levels and recovery_clicks.shape[2] == 3,
           f"{sample_name}: recovery_corrective_clicks shape mismatch")

    ensure(0 <= num_effective_levels <= num_levels, f"{sample_name}: num_effective_levels out of range")
    ensure(len(effective_level_indices) >= num_effective_levels, f"{sample_name}: effective_level_indices too short")
    valid_effective = effective_level_indices[:num_effective_levels]
    if len(valid_effective) > 0:
        ensure(np.all((0 <= valid_effective) & (valid_effective < num_levels)),
               f"{sample_name}: effective_level_indices out of range")

    valid_targets = completion_target_indices[completion_target_indices >= 0]
    if len(valid_targets) > 0:
        ensure(np.all(valid_targets < num_levels), f"{sample_name}: completion_target_indices out of range")


def check_split(dataset_root: Path, split: str, sample_limit: int, load_limit: int):
    metadata_path = dataset_root / "metadata" / f"{split}.jsonl"
    ensure(metadata_path.exists(), f"{split}: missing metadata file {metadata_path}")
    rows = read_jsonl(metadata_path)
    ensure(len(rows) > 0, f"{split}: metadata is empty")

    checked = 0
    for row in iter_rows(rows, sample_limit):
        sample_name = row["sample_name"]
        image_path = Path(row["image_path"])
        npz_path = Path(row["npz_path"])
        ensure(image_path.exists(), f"{sample_name}: missing image {image_path}")
        check_npz(sample_name, npz_path)
        checked += 1

    dataset = GranularityRecoveryDataset(str(dataset_root), split=split)
    load_count = min(load_limit, len(dataset))
    for index in range(load_count):
        sample = dataset[index]
        ensure(sample["inputs"].ndim == 4, f"{split}[{index}]: inputs tensor shape mismatch")
        ensure(sample["teacher_probs"].shape == sample["teacher_masks"].shape,
               f"{split}[{index}]: teacher tensor shape mismatch after load")
        ensure(sample["gt_masks"].shape == sample["teacher_masks"].shape,
               f"{split}[{index}]: gt tensor shape mismatch after load")
        ensure(sample["inputs"].shape[0] == sample["teacher_probs"].shape[0],
               f"{split}[{index}]: variant count mismatch after load")

    return {
        "metadata_rows": len(rows),
        "npz_checked": checked,
        "dataset_loaded": load_count,
    }


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    ensure(dataset_root.exists(), f"dataset root does not exist: {dataset_root}")

    summary = {}
    for split in args.splits:
        summary[split] = check_split(dataset_root, split, args.sample_limit, args.load_limit)

    print(json.dumps({
        "status": "ok",
        "dataset_root": str(dataset_root),
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
