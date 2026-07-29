from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.run_evaluation import (
    build_controller,
    build_runtime_components,
    resolve_default_dinov3_repo,
    resolve_default_dinov3_weights,
)
from evaluate.semantic_coverage import (
    hierarchical_semantic_metrics,
    summarize_hierarchical_records,
)
from evaluate.semantic_coverage_data import SemanticHierarchyDataset
from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.utils import exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["PascalPart", "PartImageNet", "SPIN"],
        required=True,
    )
    parser.add_argument("--backbone", choices=["ViT-B", "ViT-L"], required=True)
    parser.add_argument("--config-path", default="config.yml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="SimpleClick checkpoint. Defaults to the matching cocolvis ViT-B/ViT-L checkpoint.",
    )
    parser.add_argument("--granularity-recovery-checkpoint", default="experiments/granularity_recovery/best.pth")
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
    parser.add_argument("--refresh-every", type=int, default=8)
    args = parser.parse_args()
    if args.num_sample_shards < 1:
        parser.error("--num-sample-shards must be positive")
    if not 0 <= args.sample_shard_id < args.num_sample_shards:
        parser.error("--sample-shard-id must be in [0, num shards)")
    if not 0.0 < args.scan_step <= 1.0:
        parser.error("--scan-step must be in (0, 1]")
    args.recall_thresholds = tuple(
        float(value) for value in args.recall_thresholds.split(",") if value.strip()
    )
    if args.checkpoint is None:
        checkpoint_name = {
            "ViT-B": "cocolvis_vit_base.pth",
            "ViT-L": "cocolvis_vit_large.pth",
        }[args.backbone]
        args.checkpoint = f"weights/simpleclick_models/{checkpoint_name}"
    return args


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_sgr_runtime(args: argparse.Namespace):
    runtime_args = SimpleNamespace(
        ablation_mode="full",
        checkpoint=args.checkpoint,
        granularity_recovery_checkpoint=args.granularity_recovery_checkpoint,
        device=torch.device(args.device),
        dinov3_repo=resolve_default_dinov3_repo(),
        dinov3_model="dinov3_vitl16",
        dinov3_weights=resolve_default_dinov3_weights("dinov3_vitl16"),
        dinov3_image_size=768,
        granularity_candidate_threshold=0.50,
        granularity_max_hole_ratio=0.02,
        pred_thresh=0.5,
        limit_longest_size=800,
        refine_strength=0.65,
    )
    cfg = exp.load_config_file(args.config_path, return_edict=True)
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, runtime_args.checkpoint)
    model = utils.load_is_model(
        checkpoint_path,
        runtime_args.device,
        eval_ritm=False,
        cpu_dist_maps=True,
    )
    refiner, recovery = build_runtime_components(runtime_args)
    controller = build_controller(runtime_args, model, refiner, recovery)
    return controller


def generate_sgr_candidates(
    controller,
    image: np.ndarray,
    click_target: np.ndarray,
    scan_step: float,
):
    controller.set_image(image)
    clicker = Clicker(gt_mask=np.asarray(click_target, dtype=np.int32))
    clicker.make_next_click(np.zeros_like(click_target, dtype=bool))
    first_click = clicker.clicks_list[-1]
    y, x = first_click.coords
    controller._requested_granularity = 0.50
    controller.add_click(int(x), int(y), bool(first_click.is_positive))

    count = int(round(1.0 / scan_step))
    controls = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    masks = []
    for control in controls:
        controller.set_granularity(float(control))
        probability = controller.current_object_prob
        masks.append((probability > controller.prob_thresh).copy())

    controller.reset_last_object(update_image=False)
    return masks, controls.astype(float).tolist(), (int(x), int(y))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "semantic_coverage.partial.json"
    result_path = args.output_dir / "semantic_coverage.json"
    if result_path.exists():
        print(f"[already complete] {result_path}", flush=True)
        return

    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        records = list(partial.get("records", []))
        skipped = list(partial.get("skipped", []))
    else:
        records, skipped = [], []
    completed = {str(record["sample_key"]) for record in [*records, *skipped]}

    dataset = SemanticHierarchyDataset(args)
    samples = dataset.samples()
    samples = samples[args.sample_shard_id::args.num_sample_shards]
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    controller = build_sgr_runtime(args)
    started = time.time()

    try:
        for index, sample in enumerate(samples, start=1):
            sample_key = str(sample["sample_key"])
            if sample_key in completed:
                continue
            print(f"[sample start] {index}/{len(samples)} {sample_key}", flush=True)
            try:
                image, targets, target_names = dataset.load(sample)
                predictions, controls, click = generate_sgr_candidates(
                    controller,
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
                    "prediction_areas": [int(np.count_nonzero(mask)) for mask in predictions],
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
                skipped.append({"sample_key": sample_key, "sample_id": str(sample["sample_id"]), "error": repr(error)})
                completed.add(sample_key)
                print(f"[sample skipped] {sample_key}: {error!r}", flush=True)
            write_json_atomic(partial_path, {"records": records, "skipped": skipped})
            if index % args.refresh_every == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        dataset.close()

    payload = {
        "protocol": {
            "name": "Annotated Semantic-Level Coverage",
            "model": "SGR-Click",
            "backbone": args.backbone,
            "dataset": args.dataset,
            "semantic_targets": (
                ["subpart", "part", "whole_object"]
                if args.dataset == "SPIN"
                else ["part", "whole_object"]
            ),
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
