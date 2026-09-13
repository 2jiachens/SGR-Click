from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.semantic_coverage import (
    hierarchical_semantic_metrics,
    semantic_trajectory_diagnostics,
    summarize_hierarchical_records,
)
from evaluate.semantic_coverage_data import SemanticHierarchyDataset
from isegm.inference.clicker import Clicker


MAPPING_NAMES = ("direct3", "area_expanded11")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["PascalPart", "PartImageNet", "SPIN"],
        required=True,
    )
    parser.add_argument(
        "--category",
        choices=["all", "head", "torso", "wheel", "body", "foot"],
        default="all",
        help="Category-aligned subset. SPIN supports only all.",
    )
    parser.add_argument("--sam-repo", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", default="vit_l", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pascalpart-annotations-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "PascalPart" / "parts" / "Annotations_Part",
    )
    parser.add_argument(
        "--pascalpart-images-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "VOCdevkit" / "VOC2012" / "JPEGImages",
    )
    parser.add_argument(
        "--partimagenet-zip",
        type=Path,
        default=REPO_ROOT / "datasets" / "PartImageNet_Seg.zip",
    )
    parser.add_argument(
        "--spin-annotations-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "SPIN" / "spin_jsons_for_coco",
    )
    parser.add_argument("--sample-shard-id", type=int, default=0)
    parser.add_argument("--num-sample-shards", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--recall-thresholds", default="0.60,0.70")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refresh-every", type=int, default=16)
    args = parser.parse_args()
    allowed_categories = {
        "PascalPart": {"all", "head", "torso", "wheel"},
        "PartImageNet": {"all", "head", "body", "foot"},
        "SPIN": {"all"},
    }
    if args.category not in allowed_categories[args.dataset]:
        parser.error(f"category {args.category!r} is invalid for {args.dataset}")
    if args.num_sample_shards < 1:
        parser.error("--num-sample-shards must be positive")
    if not 0 <= args.sample_shard_id < args.num_sample_shards:
        parser.error("--sample-shard-id must be in [0, num shards)")
    args.recall_thresholds = tuple(
        float(value) for value in args.recall_thresholds.split(",") if value.strip()
    )
    return args


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_sam_predictor(args: argparse.Namespace):
    sam_repo = str(args.sam_repo.expanduser().resolve())
    if sam_repo not in sys.path:
        sys.path.insert(0, sam_repo)
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise RuntimeError(
            "Could not import the official facebookresearch/segment-anything "
            f"package from {sam_repo}"
        ) from error

    model = sam_model_registry[args.model_type](
        checkpoint=str(args.sam_checkpoint.expanduser().resolve())
    )
    model.to(device=torch.device(args.device))
    model.eval()
    return SamPredictor(model)


def choose_first_positive_click(click_target: np.ndarray) -> tuple[int, int]:
    clicker = Clicker(gt_mask=np.asarray(click_target, dtype=np.int32))
    clicker.make_next_click(np.zeros_like(click_target, dtype=bool))
    click = clicker.clicks_list[-1]
    if not click.is_positive:
        raise RuntimeError("the first click selected from the fine target is not positive")
    y, x = click.coords
    return int(x), int(y)


def area_sort_masks(
    masks: Sequence[np.ndarray],
    predicted_ious: Sequence[float],
) -> tuple[list[np.ndarray], list[float], list[int]]:
    order = sorted(
        range(len(masks)),
        key=lambda index: (int(np.count_nonzero(masks[index])), index),
    )
    return (
        [np.asarray(masks[index], dtype=bool) for index in order],
        [float(predicted_ious[index]) for index in order],
        [int(index) for index in order],
    )


def mapping_indices(name: str, candidate_count: int) -> list[int]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if name == "direct3":
        return list(range(candidate_count))
    if name == "area_expanded11":
        controls = np.linspace(0.0, 1.0, 11, dtype=np.float64)
        return np.floor(controls * (candidate_count - 1) + 0.5).astype(int).tolist()
    raise ValueError(f"unknown mapping: {name}")


def controls_for_mapping(name: str, candidate_count: int) -> list[float]:
    if name == "direct3":
        return np.linspace(0.0, 1.0, candidate_count, dtype=np.float64).tolist()
    if name == "area_expanded11":
        return np.linspace(0.0, 1.0, 11, dtype=np.float64).tolist()
    raise ValueError(f"unknown mapping: {name}")


def score_mapping(
    name: str,
    area_sorted_masks: Sequence[np.ndarray],
    semantic_targets: Sequence[np.ndarray],
    recall_thresholds: Sequence[float],
) -> dict[str, object]:
    indices = mapping_indices(name, len(area_sorted_masks))
    predictions = [area_sorted_masks[index] for index in indices]
    metrics = hierarchical_semantic_metrics(
        predictions,
        semantic_targets,
        recall_thresholds=recall_thresholds,
    )
    prediction_areas = [int(np.count_nonzero(mask)) for mask in predictions]
    diagnostics = semantic_trajectory_diagnostics(
        metrics["matched_prediction_indices"],
        prediction_areas,
        metrics["target_prediction_ious"],
    )
    return {
        "selected_controls": controls_for_mapping(name, len(area_sorted_masks)),
        "candidate_indices": indices,
        "prediction_areas": prediction_areas,
        "sor": float(diagnostics["semantic_order_success"]),
        **metrics,
    }


def summarize_mapping_records(
    records: Sequence[dict[str, object]],
    mapping_name: str,
    recall_thresholds: Sequence[float],
) -> dict[str, object]:
    rows = [record["mappings"][mapping_name] for record in records]
    summary = summarize_hierarchical_records(
        rows,
        recall_thresholds=recall_thresholds,
    )
    summary["sor"] = (
        float(np.mean([float(row["sor"]) for row in rows])) if rows else None
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "semantic_coverage.partial.json"
    result_path = args.output_dir / "semantic_coverage.json"
    if result_path.exists():
        print(f"[already complete] {result_path}", flush=True)
        return

    partial = (
        json.loads(partial_path.read_text(encoding="utf-8"))
        if partial_path.exists()
        else {"records": [], "skipped": []}
    )
    records = list(partial.get("records", []))
    skipped = list(partial.get("skipped", []))
    completed = {str(record["sample_key"]) for record in [*records, *skipped]}

    dataset = SemanticHierarchyDataset(args)
    samples = dataset.samples()[args.sample_shard_id :: args.num_sample_shards]
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    predictor = build_sam_predictor(args)
    started = time.time()

    try:
        for index, sample in enumerate(samples, start=1):
            sample_key = str(sample["sample_key"])
            if sample_key in completed:
                continue
            print(f"[sample start] {index}/{len(samples)} {sample_key}", flush=True)
            try:
                image, targets, target_names = dataset.load(sample)
                x, y = choose_first_positive_click(targets[0])
                predictor.set_image(image)
                masks, predicted_ious, _ = predictor.predict(
                    point_coords=np.asarray([[x, y]], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int32),
                    multimask_output=True,
                    return_logits=False,
                )
                sorted_masks, sorted_ious, original_order = area_sort_masks(
                    masks,
                    predicted_ious,
                )
                if len(sorted_masks) != 3:
                    raise RuntimeError(
                        f"SAM multimask_output returned {len(sorted_masks)} masks, expected 3"
                    )
                record = {
                    "sample_key": sample_key,
                    "sample_id": str(sample["sample_id"]),
                    "target_names": target_names,
                    "target_areas": [int(np.count_nonzero(mask)) for mask in targets],
                    "click_xy": [x, y],
                    "sam_original_indices_in_area_order": original_order,
                    "sam_predicted_ious_in_area_order": sorted_ious,
                    "sam_candidate_areas": [
                        int(np.count_nonzero(mask)) for mask in sorted_masks
                    ],
                    "mappings": {
                        name: score_mapping(
                            name,
                            sorted_masks,
                            targets,
                            args.recall_thresholds,
                        )
                        for name in MAPPING_NAMES
                    },
                }
                records.append(record)
                completed.add(sample_key)
                direct = record["mappings"]["direct3"]
                print(
                    f"[sample done] LR60={direct['lr60']:.3f} "
                    f"HSR60={direct['hsr60']:.3f} LR70={direct['lr70']:.3f} "
                    f"mBIoU={direct['m_biou']:.3f} SOR={direct['sor']:.3f}",
                    flush=True,
                )
            except Exception as error:
                skipped.append(
                    {
                        "sample_key": sample_key,
                        "sample_id": str(sample["sample_id"]),
                        "error": repr(error),
                    }
                )
                completed.add(sample_key)
                print(f"[sample skipped] {sample_key}: {error!r}", flush=True)
            write_json_atomic(partial_path, {"records": records, "skipped": skipped})
            if args.refresh_every > 0 and index % args.refresh_every == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        dataset.close()

    payload = {
        "protocol": {
            "name": "SAM ViT-L multimask semantic-level coverage",
            "model": "SAM",
            "model_type": args.model_type,
            "sam_repo": str(args.sam_repo.expanduser().resolve()),
            "sam_checkpoint": str(args.sam_checkpoint.expanduser().resolve()),
            "sam_checkpoint_sha256": sha256_file(
                args.sam_checkpoint.expanduser().resolve()
            ),
            "dataset": args.dataset,
            "category": args.category,
            "prompt": "one positive click selected from the finest annotated target",
            "candidate_order": "ascending binary-mask area",
            "mappings": {
                "direct3": "three area-sorted native SAM candidates",
                "area_expanded11": (
                    "nearest-neighbor resampling of the three area-sorted candidates "
                    "onto 11 controls; repeated masks do not create new candidates"
                ),
            },
            "recall_thresholds": list(args.recall_thresholds),
            "sample_shard_id": args.sample_shard_id,
            "num_sample_shards": args.num_sample_shards,
        },
        "summaries": {
            name: summarize_mapping_records(records, name, args.recall_thresholds)
            for name in MAPPING_NAMES
        },
        "records": records,
        "skipped": skipped,
        "elapsed_sec": time.time() - started,
    }
    write_json_atomic(result_path, payload)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(payload["summaries"], indent=2), flush=True)
    print(f"[complete] {result_path}", flush=True)


if __name__ == "__main__":
    main()
