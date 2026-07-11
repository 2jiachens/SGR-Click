from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.inference.dino_granularity import DinoGranularityRefiner
from isegm.inference.granularity_refine_inference import GranularityRefineInference
from isegm.utils import exp
from granularity_recovery.gui_controller import GranularityRecoveryController


def resolve_default_dinov3_repo() -> str:
    env_path = os.environ.get("DINOV3_REPO")
    if env_path:
        return env_path
    return str(REPO_ROOT / "third_party" / "dinov3")


def resolve_default_dinov3_weights(model_name: str) -> str:
    env_path = os.environ.get("DINOV3_WEIGHTS")
    if env_path:
        return env_path
    weights_dir = REPO_ROOT / "weights" / "dinov3"
    default_names = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    }
    return str(weights_dir / default_names[model_name])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weights/simpleclick_models/cocolvis_vit_base.pth")
    parser.add_argument("--base-granularity-checkpoint", type=str, required=True)
    parser.add_argument("--config-path", type=str, default="config.yml")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="PascalPart root that contains parts/Annotations_Part")
    parser.add_argument("--voc-images-dir", type=str, default=None,
                        help="VOC JPEGImages directory, e.g. .../VOC2012/JPEGImages")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Text file listing sample ids, one per line")
    parser.add_argument("--part-name", type=str, default="head")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--first-click-granularity", type=float, default=0.50)
    parser.add_argument("--oracle-granularity", action="store_true", default=False,
                        help="Select the first-click granularity per sample by sweeping candidate slider values.")
    parser.add_argument("--oracle-granularity-values", type=str, default="0.00,0.08,0.18,0.32,0.50,0.68,0.82,0.92,1.00",
                        help="Comma-separated candidate slider values used when --oracle-granularity is enabled.")
    parser.add_argument("--pred-thresh", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--limit-longest-size", type=int, default=800)
    parser.add_argument("--dinov3-repo", type=str, default=None)
    parser.add_argument("--dinov3-model", type=str, default="dinov3_vitl16",
                        choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"])
    parser.add_argument("--dinov3-weights", type=str, default=None)
    parser.add_argument("--dinov3-image-size", type=int, default=768)
    parser.add_argument("--granularity-work-size", type=int, default=512)
    parser.add_argument("--granularity-segments", type=int, default=1000)
    parser.add_argument("--granularity-candidate-threshold", type=float, default=0.50)
    parser.add_argument("--granularity-max-hole-ratio", type=float, default=0.04)
    parser.add_argument("--granularity-slic-compactness", type=float, default=6.0)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--refresh-every", type=int, default=8,
                        help="Rebuild the runtime controller every N evaluated samples to reduce long-run state buildup.")
    parser.add_argument("--skip-sample-ids", type=str, default="",
                        help="Comma-separated Pascal sample ids to skip, e.g. 2008_000271,2008_000345")
    parser.add_argument("--candidate-offset", type=int, default=0,
                        help="Skip the first N candidate part instances after sorting.")
    parser.add_argument("--sample-keys-file", type=str, default=None,
                        help="Optional JSON file containing an explicit ordered sample list to evaluate.")
    args = parser.parse_args()

    if args.dinov3_repo is None:
        args.dinov3_repo = resolve_default_dinov3_repo()
    if args.dinov3_weights is None:
        args.dinov3_weights = resolve_default_dinov3_weights(args.dinov3_model)
    if args.cpu:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)
    return args


def load_part_samples(dataset_root: Path, voc_images_dir: Path, split_file: Path, part_name: str) -> List[Dict[str, object]]:
    anno_dir = dataset_root / "parts" / "Annotations_Part"
    sample_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples: List[Dict[str, object]] = []

    for sample_id in sample_ids:
        mat_path = anno_dir / f"{sample_id}.mat"
        image_path = voc_images_dir / f"{sample_id}.jpg"
        if not mat_path.exists() or not image_path.exists():
            continue

        ann = loadmat(str(mat_path))["anno"]["objects"][0, 0]
        for object_id in range(ann.shape[1]):
            parts = ann["parts"][0, object_id]
            for part_id in range(parts.shape[1]):
                current_part_name = parts["part_name"][0, part_id][0].split("_")[0]
                if current_part_name != part_name:
                    continue
                samples.append({
                    "sample_id": sample_id,
                    "object_id": int(object_id),
                    "part_id": int(part_id),
                    "mat_path": str(mat_path),
                    "image_path": str(image_path),
                    "part_name": current_part_name,
                })

    samples.sort(key=lambda item: (item["sample_id"], item["object_id"], item["part_id"]))
    return samples


def auto_find_child_dir(root: Path, target_name: str) -> Path | None:
    if not root.exists():
        return None
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    for child in children:
        try:
            if child.is_dir() and child.name == target_name:
                return child
            nested = child / target_name
            if nested.is_dir():
                return nested
        except OSError:
            continue
    return None


def resolve_dataset_paths(args):
    drive_root = Path("D:/")

    dataset_root = Path(args.dataset_root) if args.dataset_root else auto_find_child_dir(drive_root, "PascalPart")
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError("PascalPart root was not found. Pass --dataset-root explicitly.")

    if args.voc_images_dir:
        voc_images_dir = Path(args.voc_images_dir)
    else:
        voc_root = auto_find_child_dir(drive_root, "VOCdevkit")
        if voc_root is None:
            raise FileNotFoundError("VOCdevkit root was not found. Pass --voc-images-dir explicitly.")
        voc_images_dir = voc_root / "VOC2012" / "JPEGImages"
    if not voc_images_dir.exists():
        raise FileNotFoundError(f"VOC images dir was not found: {voc_images_dir}")

    if args.split_file:
        split_file = Path(args.split_file)
    else:
        split_file = voc_images_dir.parent / "ImageSets" / "Segmentation" / "val.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file was not found: {split_file}")

    return dataset_root, voc_images_dir, split_file


def load_sample(sample: Dict[str, object]):
    image_bytes = np.fromfile(sample["image_path"], dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {sample['image_path']}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ann = loadmat(sample["mat_path"])["anno"]["objects"][0, 0]
    gt_mask = ann["parts"][0, int(sample["object_id"])]["mask"][0, int(sample["part_id"])]
    gt_mask = (gt_mask > 0).astype(np.int32)
    return image, gt_mask


def normalize_sample_key(sample: Dict[str, object]) -> str:
    return f"{sample['sample_id']}:{int(sample['object_id'])}:{int(sample['part_id'])}"


def load_explicit_samples(sample_keys_file: Path, dataset_root: Path, voc_images_dir: Path) -> List[Dict[str, object]]:
    payload = json.loads(sample_keys_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("sample-keys-file must contain a JSON list.")

    anno_dir = dataset_root / "parts" / "Annotations_Part"
    explicit_samples: List[Dict[str, object]] = []
    for item in payload:
        sample_id = str(item["sample_id"])
        object_id = int(item["object_id"])
        part_id = int(item["part_id"])
        explicit_samples.append({
            "sample_id": sample_id,
            "object_id": object_id,
            "part_id": part_id,
            "mat_path": str(anno_dir / f"{sample_id}.mat"),
            "image_path": str(voc_images_dir / f"{sample_id}.jpg"),
            "part_name": str(item.get("part_name", "")),
        })
    return explicit_samples


def parse_granularity_values(values_text: str) -> List[float]:
    values: List[float] = []
    for raw in values_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        values.append(min(1.0, max(0.0, value)))
    if not values:
        raise RuntimeError("oracle granularity candidate list is empty.")
    return values


def build_runtime_components(args):
    granularity_refiner = DinoGranularityRefiner(
        repo_path=args.dinov3_repo,
        weights_path=args.dinov3_weights,
        device=args.device,
        model_name=args.dinov3_model,
        image_size=args.dinov3_image_size,
        work_size=args.granularity_work_size,
        n_segments=args.granularity_segments,
        candidate_threshold=args.granularity_candidate_threshold,
        max_hole_ratio=args.granularity_max_hole_ratio,
        slic_compactness=args.granularity_slic_compactness,
        cluster_tau=0.60,
    )
    base_granularity_predictor = GranularityRefineInference(
        checkpoint_path=args.base_granularity_checkpoint,
        device=args.device,
    )
    return granularity_refiner, base_granularity_predictor


def build_controller(args, simpleclick_model, granularity_refiner, base_granularity_predictor):
    controller = GranularityRecoveryController(
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
        base_granularity_predictor=base_granularity_predictor,
        prob_thresh=args.pred_thresh,
    )
    return controller


def compute_realized_noc(ious: List[float], iou_threshold: float, max_clicks: int) -> int:
    for click_index, iou in enumerate(ious[:max_clicks], start=1):
        if float(iou) >= iou_threshold:
            return click_index
    return int(max_clicks)


def evaluate_fixed_granularity(controller: GranularityRecoveryController, image: np.ndarray, gt_mask: np.ndarray,
                               granularity: float, max_clicks: int, iou_threshold: float) -> Dict[str, object]:
    controller.set_image(image)
    controller._requested_granularity = float(granularity)

    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask, dtype=bool)
    ious: List[float] = []
    clicks: List[Dict[str, object]] = []

    for click_index in range(max_clicks):
        clicker.make_next_click(pred_mask)
        current_click = clicker.clicks_list[-1]
        y, x = current_click.coords

        controller.add_click(int(x), int(y), bool(current_click.is_positive))

        current_prob = controller.current_object_prob
        if current_prob is None:
            current_prob = np.zeros_like(gt_mask, dtype=np.float32)
        pred_mask = current_prob > controller.prob_thresh
        iou = float(utils.get_iou(gt_mask, pred_mask))
        ious.append(iou)
        clicks.append({
            "index": click_index + 1,
            "x": int(x),
            "y": int(y),
            "is_positive": bool(current_click.is_positive),
            "iou": iou,
        })
        if iou >= iou_threshold:
            break

    controller.reset_last_object(update_image=False)
    return {
        "granularity": float(granularity),
        "ious": ious,
        "clicks": clicks,
        "realized_noc": compute_realized_noc(ious, iou_threshold, max_clicks),
        "iou_at_1": float(ious[0]) if ious else 0.0,
        "reached_threshold": any(float(iou) >= iou_threshold for iou in ious),
    }


def evaluate_sample(controller: GranularityRecoveryController, image: np.ndarray, gt_mask: np.ndarray,
                    first_click_granularity: float, max_clicks: int, iou_threshold: float,
                    oracle_granularity: bool = False,
                    oracle_granularity_values: List[float] | None = None) -> Dict[str, object]:
    candidate_values = [float(first_click_granularity)]
    if oracle_granularity:
        candidate_values = oracle_granularity_values or [float(first_click_granularity)]

    candidate_results = [
        evaluate_fixed_granularity(
            controller,
            image,
            gt_mask,
            granularity=float(granularity),
            max_clicks=max_clicks,
            iou_threshold=iou_threshold,
        )
        for granularity in candidate_values
    ]
    candidate_results.sort(key=lambda item: float(item["granularity"]))

    selected_candidate = min(
        candidate_results,
        key=lambda item: (int(item["realized_noc"]), -float(item["iou_at_1"])),
    )
    best_iou1_candidate = max(
        candidate_results,
        key=lambda item: (float(item["iou_at_1"]), -int(item["realized_noc"])),
    )

    return {
        "ious": list(selected_candidate["ious"]),
        "clicks": list(selected_candidate["clicks"]),
        "selected_granularity": float(selected_candidate["granularity"]),
        "selected_noc": int(selected_candidate["realized_noc"]),
        "selected_iou_at_1": float(selected_candidate["iou_at_1"]),
        "best_iou_at_1": float(best_iou1_candidate["iou_at_1"]),
        "best_iou_at_1_granularity": float(best_iou1_candidate["granularity"]),
        "oracle_candidates": candidate_results,
    }


def main():
    args = parse_args()
    oracle_granularity_values = parse_granularity_values(args.oracle_granularity_values)
    cfg = exp.load_config_file(args.config_path, return_edict=True)
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    simpleclick_model = utils.load_is_model(checkpoint_path, args.device, eval_ritm=False, cpu_dist_maps=True)

    granularity_refiner, base_granularity_predictor = build_runtime_components(args)
    controller = build_controller(args, simpleclick_model, granularity_refiner, base_granularity_predictor)

    dataset_root, voc_images_dir, split_file = resolve_dataset_paths(args)
    if args.sample_keys_file:
        samples = load_explicit_samples(Path(args.sample_keys_file), dataset_root, voc_images_dir)
    else:
        samples = load_part_samples(dataset_root, voc_images_dir, split_file, args.part_name)
        skip_sample_ids = {item.strip() for item in args.skip_sample_ids.split(",") if item.strip()}
        if skip_sample_ids:
            samples = [sample for sample in samples if sample["sample_id"] not in skip_sample_ids]
        if args.candidate_offset > 0:
            samples = samples[args.candidate_offset:]
    if not samples:
        raise RuntimeError("No matching PascalPart samples were found for the requested part and split.")

    target_sample_count = min(args.sample_limit, len(samples)) if args.sample_limit > 0 else len(samples)
    print(f"[eval] part={args.part_name} candidates={len(samples)} target_samples={target_sample_count} "
          f"first_click_granularity={args.first_click_granularity:.2f} "
          f"oracle_granularity={args.oracle_granularity}", flush=True)
    all_ious: List[np.ndarray] = []
    records: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    start_time = time.time()

    for candidate_index, sample in enumerate(samples, start=1):
        if len(all_ious) >= target_sample_count:
            break
        if args.refresh_every > 0 and len(all_ious) > 0 and len(all_ious) % args.refresh_every == 0:
            del controller
            gc.collect()
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            controller = build_controller(args, simpleclick_model, granularity_refiner, base_granularity_predictor)
        try:
            image, gt_mask = load_sample(sample)
            result = evaluate_sample(
                controller,
                image,
                gt_mask,
                first_click_granularity=args.first_click_granularity,
                max_clicks=args.max_clicks,
                iou_threshold=args.iou_threshold,
                oracle_granularity=args.oracle_granularity,
                oracle_granularity_values=oracle_granularity_values,
            )
        except Exception as error:
            skipped.append({
                **sample,
                "error": str(error),
            })
            print(f"[skip {candidate_index:03d}] {sample['sample_id']} obj={sample['object_id']} part={sample['part_id']} "
                  f"error={error}", flush=True)
            continue

        ious = np.asarray(result["ious"], dtype=np.float32)
        all_ious.append(ious)
        records.append({
            **sample,
            "sample_key": normalize_sample_key(sample),
            "ious": [float(x) for x in result["ious"]],
            "clicks": result["clicks"],
            "selected_granularity": float(result["selected_granularity"]),
            "selected_noc": int(result["selected_noc"]),
            "selected_iou_at_1": float(result["selected_iou_at_1"]),
            "best_iou_at_1": float(result["best_iou_at_1"]),
            "best_iou_at_1_granularity": float(result["best_iou_at_1_granularity"]),
            "oracle_candidates": result["oracle_candidates"],
        })
        final_iou = float(ious[-1]) if len(ious) > 0 else 0.0
        first_iou = float(ious[0]) if len(ious) > 0 else 0.0
        print(f"[{len(all_ious):03d}/{target_sample_count:03d}] {sample['sample_id']} obj={sample['object_id']} part={sample['part_id']} "
              f"IoU@1={first_iou:.4f} IoU@last={final_iou:.4f} clicks={len(ious)} "
              f"g*={float(result['selected_granularity']):.2f} "
              f"maxIoU1@{float(result['best_iou_at_1_granularity']):.2f}={float(result['best_iou_at_1']):.4f}",
              flush=True)

        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_ious:
        raise RuntimeError("All candidate samples failed during evaluation.")

    elapsed_time = time.time() - start_time
    noc85, noc85_std, over_max = utils.compute_noc_metric(all_ious, [args.iou_threshold], max_clicks=args.max_clicks)
    mean_iou1 = float(np.mean([float(record["selected_iou_at_1"]) for record in records]))
    mean_max_iou1 = float(np.mean([float(record["best_iou_at_1"]) for record in records]))
    mean_clicks = float(np.mean([len(ious) for ious in all_ious]))

    print("", flush=True)
    print("Summary", flush=True)
    print(f"part={args.part_name}", flush=True)
    print(f"evaluated_samples={len(all_ious)}", flush=True)
    print(f"skipped_samples={len(skipped)}", flush=True)
    print(f"IoU threshold={args.iou_threshold:.2f}", flush=True)
    print(f"first_click_granularity={args.first_click_granularity:.2f}", flush=True)
    print(f"oracle_granularity={args.oracle_granularity}", flush=True)
    print(f"mean selected IoU@1={mean_iou1:.4f}", flush=True)
    print(f"mean max IoU@1={mean_max_iou1:.4f}", flush=True)
    print(f"NoC@{int(round(args.iou_threshold * 100))}={noc85[0]:.4f}", flush=True)
    print(f"NoC std={noc85_std[0]:.4f}", flush=True)
    print(f'>={args.max_clicks} failures={over_max[0]}', flush=True)
    print(f"mean realized clicks={mean_clicks:.4f}", flush=True)
    print(f"elapsed_sec={elapsed_time:.2f}", flush=True)

    if args.save_json:
        payload = {
            "part_name": args.part_name,
            "sample_count": len(samples),
            "iou_threshold": args.iou_threshold,
            "first_click_granularity": args.first_click_granularity,
            "oracle_granularity": bool(args.oracle_granularity),
            "oracle_granularity_values": oracle_granularity_values,
            "noc": float(noc85[0]),
            "noc_std": float(noc85_std[0]),
            "failures_at_max_clicks": int(over_max[0]),
            "mean_iou_at_1": mean_iou1,
            "mean_max_iou_at_1": mean_max_iou1,
            "mean_realized_clicks": mean_clicks,
            "elapsed_sec": elapsed_time,
            "samples": records,
            "skipped": skipped,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={save_path}", flush=True)


if __name__ == "__main__":
    main()
