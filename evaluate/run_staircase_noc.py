from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import run_evaluation as evaluation
from isegm.inference.clicker import Clicker
from isegm.utils import exp
from granularity_recovery.soft_semantic_controller import SoftSemanticFusionController
from isegm.inference.soft_dino_semantic import SoftDinoSemanticRefiner

VISIBLE_STEP_MAX_IOU = 0.97
DEFAULT_BUILD_RUNTIME_COMPONENTS = evaluation.build_runtime_components


def build_runtime_components(args):
    return DEFAULT_BUILD_RUNTIME_COMPONENTS(args)


def build_controller(args, simpleclick_model, granularity_refiner):
    return SoftSemanticFusionController(
        simpleclick_model,
        args.device,
        predictor_params={
            "brs_mode": "NoBRS",
            "prob_thresh": args.pred_thresh,
            "zoom_in_params": {
                "skip_clicks": -1,
                "target_size": (448, 448),
                "expansion_ratio": 1.4,
            },
            "predictor_params": {
                "net_clicks_limit": None,
                "max_size": args.limit_longest_size,
            },
            "brs_opt_func_params": {"min_iou_diff": 1e-3},
            "lbfgs_params": {"maxfun": 20},
        },
        update_image_callback=lambda reset_canvas=False: None,
        granularity_refiner=granularity_refiner,
        prob_thresh=args.pred_thresh,
        enable_boundary_projection=args.ablation_mode not in {
            "frozen_hierarchy",
            "simple_prior",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NoC over per-image meaningful semantic staircase levels."
    )
    parser.add_argument("--config-path", default="config.yml")
    parser.add_argument("--checkpoint", default="weights/simpleclick_models/cocolvis_vit_base.pth")
    parser.add_argument("--datasets", default="GrabCut,Berkeley")
    parser.add_argument(
        "--pascalpart-targets",
        default="",
        help="Optional comma-separated PascalPart targets.",
    )
    parser.add_argument(
        "--partimagenet-targets",
        default="",
        help="Optional comma-separated PartImageNet targets.",
    )
    parser.add_argument("--output-dir", default="outputs/sgr_click_visible_staircase_noc")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument(
        "--sample-start-index",
        type=int,
        default=0,
        help="Zero-based inclusive index in the dataset sample list.",
    )
    parser.add_argument(
        "--sample-end-index",
        type=int,
        default=0,
        help="Zero-based exclusive index; 0 means the end of the dataset.",
    )
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--noc-thresholds", default="0.85,0.90")
    parser.add_argument("--pred-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit-longest-size", type=int, default=800)
    parser.add_argument("--dinov3-repo", default=None)
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-weights", default=None)
    parser.add_argument("--dinov3-image-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ablation-mode",
        choices=[
            "full",
            "frozen_hierarchy",
            "external_hierarchy",
            "soft_no_evidence_filter",
            "simple_prior",
        ],
        default="full",
    )
    args = parser.parse_args()
    if args.sample_start_index < 0:
        parser.error("--sample-start-index must be non-negative")
    if args.sample_end_index < 0:
        parser.error("--sample-end-index must be non-negative")
    if args.sample_end_index and args.sample_end_index <= args.sample_start_index:
        parser.error("--sample-end-index must exceed --sample-start-index")
    args.device = torch.device("cpu" if args.cpu else args.device)
    if args.dinov3_repo is None:
        args.dinov3_repo = evaluation.resolve_default_dinov3_repo()
    if args.dinov3_weights is None:
        args.dinov3_weights = evaluation.resolve_default_dinov3_weights(args.dinov3_model)
    return args


def discover_staircase(controller, image: np.ndarray, gt_mask: np.ndarray) -> list[float]:
    """Use coarse root samples and visually distinct GUI-tail masks."""
    controller.set_image(image)
    controller._requested_granularity = 0.50
    clicker = Clicker(gt_mask=gt_mask)
    clicker.make_next_click(np.zeros_like(gt_mask, dtype=bool))
    first = clicker.clicks_list[-1]
    y, x = first.coords
    controller.add_click(int(x), int(y), bool(first.is_positive))

    refiner = controller.granularity_refiner
    if refiner is None or not refiner.ready:
        controller.reset_last_object(update_image=False)
        return [0.5]

    # The 0-50 root phase contains many tiny internal expansions that are not
    # perceived as separate semantic results in normal GUI use.  Evaluate it at
    # a fixed 10-point granularity interval instead.
    values: list[float] = [index / 100.0 for index in range(0, 51, 10)]

    # In the semantic tail, compare the final mask shown by the GUI.  Ignore
    # tiny boundary changes whose mask IoU with the previous accepted display
    # exceeds 0.97; they are internal refinements rather than perceptible
    # semantic staircase results.  The endpoint is always retained.
    controller.set_granularity(0.50)
    previous_display = controller.current_object_prob > controller.prob_thresh
    for slider_index in range(51, 100):
        value = slider_index / 100.0
        controller.set_granularity(value)
        display = controller.current_object_prob > controller.prob_thresh
        intersection = int(np.logical_and(previous_display, display).sum())
        union = int(np.logical_or(previous_display, display).sum())
        display_iou = intersection / max(union, 1)
        if display_iou <= VISIBLE_STEP_MAX_IOU:
            values.append(value)
            previous_display = display.copy()

    values.append(1.0)

    controller.reset_last_object(update_image=False)
    return values


def best_result_for_threshold(trajectories: list[dict], threshold: float, max_clicks: int) -> dict:
    scored = []
    for trajectory in trajectories:
        noc = evaluation.compute_realized_noc(trajectory["ious"], threshold, max_clicks)
        reached = any(float(iou) >= threshold for iou in trajectory["ious"])
        scored.append(
            (
                0 if reached else 1,
                noc,
                -float(trajectory["ious"][0]),
                float(trajectory["granularity"]),
            )
        )
    failed, noc, _, granularity = min(scored)
    return {
        "noc": int(noc),
        "granularity": float(granularity),
        "reached": not bool(failed),
    }


def sample_identity(sample: dict) -> str:
    identity = {
        "sample_id": sample.get("sample_id"),
        "object_id": sample.get("object_id"),
        "part_id": sample.get("part_id"),
        "part_name": sample.get("part_name"),
        "source_values": sample.get("source_values"),
    }
    return json.dumps(identity, ensure_ascii=False, sort_keys=True)


def summarize(records: list[dict], thresholds: list[float]) -> dict:
    if not records:
        summary = {
            "samples": 0,
            "visible_step_max_iou": VISIBLE_STEP_MAX_IOU,
            "mean_staircase_levels": None,
            "min_staircase_levels": None,
            "max_staircase_levels": None,
            "iou_at_1": None,
        }
        for threshold in thresholds:
            key = evaluation.threshold_key(threshold)
            summary[f"noc{key}"] = None
            summary[f"failures_at_max_clicks_{key}"] = 0
        return summary
    summary = {
        "samples": len(records),
        "visible_step_max_iou": VISIBLE_STEP_MAX_IOU,
        "mean_staircase_levels": mean(row["staircase_level_count"] for row in records),
        "min_staircase_levels": min(row["staircase_level_count"] for row in records),
        "max_staircase_levels": max(row["staircase_level_count"] for row in records),
        "iou_at_1": mean(row["best_iou_at_1"] for row in records),
    }
    for threshold in thresholds:
        key = evaluation.threshold_key(threshold)
        summary[f"noc{key}"] = mean(row[f"best_{key}"]["noc"] for row in records)
        summary[f"failures_at_max_clicks_{key}"] = sum(
            not row[f"best_{key}"]["reached"] for row in records
        )
    return summary


def main() -> None:
    args = parse_args()
    evaluation.seed_everything(args.seed)
    evaluation.build_runtime_components = build_runtime_components
    evaluation.build_controller = build_controller
    cfg = exp.load_config_file(args.config_path, return_edict=True)
    runtime_args, model, refiner, controller = evaluation.load_runtime(args, cfg)
    del model, refiner

    thresholds = [float(value) for value in evaluation.split_csv(args.noc_thresholds)]
    stop_threshold = max(thresholds)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate = {}
    targets = [
        (dataset_name, evaluation.iter_standard_samples(dataset_name, cfg))
        for dataset_name in evaluation.split_csv(args.datasets)
    ]
    targets.extend(
        (f"PascalPart/{target}", evaluation.iter_pascalpart_samples(target, cfg))
        for target in evaluation.split_csv(args.pascalpart_targets)
    )
    targets.extend(
        (f"PartImageNet/{target}", evaluation.iter_partimagenet_samples(target, cfg))
        for target in evaluation.split_csv(args.partimagenet_targets)
    )

    for dataset_name, sample_iterator in targets:
        all_samples = list(sample_iterator)
        if args.sample_limit > 0:
            all_samples = all_samples[: args.sample_limit]
        sample_start = min(args.sample_start_index, len(all_samples))
        sample_end = (
            min(args.sample_end_index, len(all_samples))
            if args.sample_end_index > 0
            else len(all_samples)
        )
        samples = all_samples[sample_start:sample_end]
        target_dir = output_root / dataset_name.replace("/", "__")
        target_dir.mkdir(parents=True, exist_ok=True)
        records_path = target_dir / "records.json"
        records = (
            json.loads(records_path.read_text(encoding="utf-8"))
            if records_path.is_file()
            else []
        )
        completed = {sample_identity(row) for row in records}

        for local_index, sample in enumerate(samples, start=1):
            sample_index = sample_start + local_index
            sample_key = sample_identity(sample)
            if sample_key in completed:
                print(
                    f"[resume] {dataset_name} {sample_index}/{len(all_samples)} "
                    f"sample={sample['sample_id']}",
                    flush=True,
                )
                continue
            staircase = discover_staircase(controller, sample["image"], sample["gt_mask"])
            print(
                f"[staircase] {dataset_name} {sample_index}/{len(all_samples)} "
                f"sample={sample['sample_id']} levels={len(staircase)}",
                flush=True,
            )
            trajectories = []
            for value in staircase:
                result = evaluation.evaluate_fixed_granularity(
                    controller,
                    sample["image"],
                    sample["gt_mask"],
                    granularity=value,
                    max_clicks=args.max_clicks,
                    iou_threshold=stop_threshold,
                    gui_slider_protocol=True,
                )
                trajectories.append(result)

            row = {
                "dataset": dataset_name,
                "sample_id": sample["sample_id"],
                "staircase_values": staircase,
                "staircase_level_count": len(staircase),
                "max_clicks": args.max_clicks,
                "best_iou_at_1": max(float(item["ious"][0]) for item in trajectories),
                "trajectories": trajectories,
            }
            for identity_field in (
                "object_id",
                "part_id",
                "part_name",
                "source_values",
            ):
                if identity_field in sample:
                    row[identity_field] = sample[identity_field]
            for threshold in thresholds:
                key = evaluation.threshold_key(threshold)
                row[f"best_{key}"] = best_result_for_threshold(
                    trajectories, threshold, args.max_clicks
                )
            records.append(row)
            completed.add(sample_key)
            evaluation.write_json_atomic(records_path, records)

        dataset_summary = summarize(records, thresholds)
        evaluation.write_json_atomic(target_dir / "summary.json", dataset_summary)
        aggregate[dataset_name] = dataset_summary
        print(f"[dataset done] {dataset_name} {json.dumps(dataset_summary)}", flush=True)

    evaluation.write_json_atomic(output_root / "summary.json", aggregate)


if __name__ == "__main__":
    main()
