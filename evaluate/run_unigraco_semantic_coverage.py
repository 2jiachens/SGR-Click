from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.semantic_coverage import (
    hierarchical_semantic_metrics,
    summarize_hierarchical_records,
)
from evaluate.semantic_coverage_data import SemanticHierarchyDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["PascalPart", "PartImageNet", "SPIN"],
        required=True,
    )
    parser.add_argument("--backbone", choices=["ViT-B", "ViT-L"], required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
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
    parser.add_argument("--scan-step", type=float, default=0.05)
    parser.add_argument("--recall-thresholds", default="0.60,0.70")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
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


def load_official_runtime(args: argparse.Namespace):
    official_repo = str(args.official_repo.resolve())
    sys.path.insert(0, official_repo)
    from isegm.inference import utils as official_utils
    from isegm.inference.predictors import get_predictor
    from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference

    device = torch.device(args.device)
    model = official_utils.load_is_model(
        args.checkpoint,
        device,
        eval_ritm=False,
        lora_checkpoint=args.lora_checkpoint,
    )
    interpolate_pos_embed_inference(model.backbone, (448, 448), device)
    predictor = get_predictor(
        model,
        "NoBRS",
        device,
        prob_thresh=0.5,
        predictor_params={},
        zoom_in_params={
            "skip_clicks": -1,
            "target_size": (448, 448),
            "expansion_ratio": 1.4,
        },
    )
    return predictor


def generate_unigraco_candidates(
    predictor,
    image: np.ndarray,
    click_target: np.ndarray,
    scan_step: float,
):
    from isegm.inference.clicker import Clicker

    clicker = Clicker(gt_mask=np.asarray(click_target, dtype=np.int32))
    clicker.make_next_click(np.zeros_like(click_target, dtype=bool))
    first_click = clicker.clicks_list[-1]
    predictor.set_input_image(image)

    scan_count = max(2, int(round(1.0 / scan_step)) + 1)
    controls = np.linspace(0.0, 1.0, scan_count, dtype=np.float64)
    masks = []
    with torch.no_grad():
        for control in controls:
            predictor.prev_prediction = torch.zeros_like(predictor.original_image[:, :1, :, :])
            probability = predictor.get_prediction(clicker, gra=float(control))
            masks.append((probability > 0.5).copy())

    return (
        masks,
        controls.astype(float).tolist(),
        (int(first_click.coords[1]), int(first_click.coords[0])),
    )


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
    records = list(partial["records"])
    skipped = list(partial["skipped"])
    completed = {str(record["sample_key"]) for record in [*records, *skipped]}
    dataset = SemanticHierarchyDataset(args)
    samples = dataset.samples()[args.sample_shard_id::args.num_sample_shards]
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]

    predictor = load_official_runtime(args)
    started = time.time()
    try:
        for index, sample in enumerate(samples, start=1):
            sample_key = str(sample["sample_key"])
            if sample_key in completed:
                continue
            print(f"[sample start] {index}/{len(samples)} {sample_key}", flush=True)
            try:
                image, targets, target_names = dataset.load(sample)
                predictions, controls, click = generate_unigraco_candidates(
                    predictor,
                    image,
                    targets[0],
                    scan_step=args.scan_step,
                )
                metrics = hierarchical_semantic_metrics(
                    predictions,
                    targets,
                    recall_thresholds=args.recall_thresholds,
                )
                record = {
                    "sample_key": sample_key,
                    "sample_id": str(sample["sample_id"]),
                    "target_names": target_names,
                    "selected_controls": controls,
                    "prediction_areas": [
                        int(np.count_nonzero(mask)) for mask in predictions
                    ],
                    "target_areas": [int(np.count_nonzero(mask)) for mask in targets],
                    "click_xy": list(click),
                    **metrics,
                }
                records.append(record)
                completed.add(sample_key)
                print(
                    f"[sample done] LR60={record['lr60']:.3f} "
                    f"HSR60={record['hsr60']:.3f} LR70={record['lr70']:.3f} "
                    f"HSR70={record['hsr70']:.3f} mBIoU={record['m_biou']:.3f}",
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
            if index % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        dataset.close()

    payload = {
        "protocol": {
            "name": "Annotated Semantic-Level Coverage",
            "model": "UniGraCo",
            "backbone": args.backbone,
            "dataset": args.dataset,
            "official_repo": str(args.official_repo),
            "semantic_targets": (
                ["subpart", "part", "whole_object"]
                if args.dataset == "SPIN"
                else ["part", "whole_object"]
            ),
            "native_control_range": [0.0, 1.0],
            "candidate_states": "complete uniformly sampled control grid",
            "recall_thresholds": list(args.recall_thresholds),
            "scan_step": args.scan_step,
            "sample_shard_id": args.sample_shard_id,
            "num_sample_shards": args.num_sample_shards,
        },
        "summary": summarize_hierarchical_records(
            records,
            recall_thresholds=args.recall_thresholds,
        ),
        "records": records,
        "skipped": skipped,
        "elapsed_sec": time.time() - started,
    }
    write_json_atomic(result_path, payload)
    print(f"[complete] {result_path}", flush=True)


if __name__ == "__main__":
    main()
