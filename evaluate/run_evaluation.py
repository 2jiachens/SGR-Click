from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List

import cv2
import numpy as np
import torch
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granularity_recovery.gui_controller import GranularityRecoveryController
from granularity_recovery.inference import GranularityRecoveryInference
from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.inference.dino_granularity import DinoGranularityRefiner
from isegm.utils import exp


STANDARD_DATASETS = ["GrabCut", "Berkeley", "SBD", "DAVIS", "PascalVOC"]
PASCALPART_TARGETS = ["head", "wheel", "torso", "all"]
PARTIMAGENET_TARGETS = ["head", "body", "foot", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["slider", "pascalpart"], default="slider")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--protocol-path", default="evaluate/protocols/all_sliders.json")
    parser.add_argument("--checkpoint", default="weights/simpleclick_models/cocolvis_vit_base.pth")
    parser.add_argument("--granularity-recovery-checkpoint", default="experiments/granularity_recovery/best.pth")
    parser.add_argument("--output-dir", default="outputs/server_slider_eval")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--datasets", default=",".join(STANDARD_DATASETS))
    parser.add_argument("--pascalpart-targets", default=",".join(PASCALPART_TARGETS))
    parser.add_argument("--partimagenet-targets", default=",".join(PARTIMAGENET_TARGETS))
    parser.add_argument("--dataset-root", default=None,
                        help="PascalPart root that contains parts/Annotations_Part")
    parser.add_argument("--voc-images-dir", default=None,
                        help="VOC JPEGImages directory, e.g. .../VOC2012/JPEGImages")
    parser.add_argument("--split-file", default=None,
                        help="Text file listing sample ids, one per line")
    parser.add_argument("--sample-keys-file", default=None,
                        help="Optional JSON file containing an explicit ordered PascalPart sample list.")
    parser.add_argument("--part-name", default="head")
    parser.add_argument("--sample-limit", type=int, default=0, help="Debug cap per target; 0 means no cap.")
    parser.add_argument("--candidate-offset", type=int, default=0,
                        help="PascalPart-only: skip the first N candidate instances after sorting.")
    parser.add_argument("--skip-sample-ids", default="",
                        help="PascalPart-only: comma-separated sample ids to skip.")
    parser.add_argument("--skip-samples-file", default="", help="Slider-only: optional newline list of target,sample_id entries to skip.")
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--noc-thresholds", default="0.85,0.90")
    parser.add_argument("--first-click-granularity", type=float, default=0.50)
    parser.add_argument("--oracle-granularity", action="store_true", default=False)
    parser.add_argument("--oracle-granularity-values", default="0.00,0.08,0.18,0.32,0.50,0.68,0.82,0.92,1.00")
    parser.add_argument("--pred-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit-longest-size", type=int, default=800)
    parser.add_argument("--dinov3-repo", default=None)
    parser.add_argument("--dinov3-model", default="dinov3_vitl16", choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"])
    parser.add_argument("--dinov3-weights", default=None)
    parser.add_argument("--dinov3-image-size", type=int, default=768)
    parser.add_argument("--granularity-candidate-threshold", type=float, default=0.50)
    parser.add_argument("--granularity-max-hole-ratio", type=float, default=0.04)
    parser.add_argument("--refresh-every", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    if args.config_path is None:
        args.config_path = "evaluate/config_server_eval.yml" if args.task == "slider" else "config.yml"
    if args.dinov3_repo is None:
        args.dinov3_repo = resolve_default_dinov3_repo()
    if args.dinov3_weights is None:
        args.dinov3_weights = resolve_default_dinov3_weights(args.dinov3_model)
    args.device = torch.device("cpu" if args.cpu else args.device)
    return args


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


def split_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_granularity_values(values_text: str) -> List[float]:
    values = [min(1.0, max(0.0, float(raw.strip()))) for raw in values_text.split(",") if raw.strip()]
    if not values:
        raise RuntimeError("oracle granularity candidate list is empty.")
    return values


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_runtime_components(args):
    granularity_refiner = DinoGranularityRefiner(
        repo_path=args.dinov3_repo,
        weights_path=args.dinov3_weights,
        device=args.device,
        model_name=args.dinov3_model,
        image_size=args.dinov3_image_size,
        candidate_threshold=args.granularity_candidate_threshold,
        max_hole_ratio=args.granularity_max_hole_ratio,
        cluster_tau=0.60,
    )
    granularity_predictor = GranularityRecoveryInference(
        checkpoint_path=args.granularity_recovery_checkpoint,
        device=args.device,
    )
    return granularity_refiner, granularity_predictor


def build_controller(args, simpleclick_model, granularity_refiner, granularity_predictor):
    return GranularityRecoveryController(
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
        granularity_predictor=granularity_predictor,
        prob_thresh=args.pred_thresh,
    )


def compute_realized_noc(ious: List[float], iou_threshold: float, max_clicks: int) -> int:
    for click_index, iou in enumerate(ious[:max_clicks], start=1):
        if float(iou) >= iou_threshold:
            return click_index
    return int(max_clicks)


def evaluate_fixed_granularity(
    controller: GranularityRecoveryController,
    image: np.ndarray,
    gt_mask: np.ndarray,
    granularity: float,
    max_clicks: int,
    iou_threshold: float,
) -> Dict[str, object]:
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


def load_runtime(args: argparse.Namespace, cfg):
    runtime_args = SimpleNamespace(**vars(args))
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    simpleclick_model = utils.load_is_model(checkpoint_path, runtime_args.device, eval_ritm=False, cpu_dist_maps=True)
    granularity_refiner, granularity_predictor = build_runtime_components(runtime_args)
    controller = build_controller(runtime_args, simpleclick_model, granularity_refiner, granularity_predictor)
    return runtime_args, simpleclick_model, granularity_refiner, granularity_predictor, controller


def load_skip_samples(path_text: str) -> set[tuple[str, str]]:
    if not path_text:
        return set()
    path = Path(path_text)
    if not path.exists():
        return set()
    skip_samples: set[tuple[str, str]] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in re.split(r"[,\s]+", line, maxsplit=1) if part]
        if len(parts) >= 2:
            skip_samples.add((parts[0], parts[1]))
        elif len(parts) == 1:
            skip_samples.add(("", parts[0]))
    return skip_samples


def filter_skip_samples(target_name: str, samples: List[Dict[str, object]], skip_samples: set[tuple[str, str]]) -> List[Dict[str, object]]:
    if not skip_samples:
        return samples
    kept = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        if (target_name, sample_id) in skip_samples or ("", sample_id) in skip_samples:
            print(f"[skip-list] {target_name} sample={sample_id}", flush=True)
            continue
        kept.append(sample)
    return kept


def load_pascalpart_mask(sample: Dict[str, object], dataset_root: Path, voc_images_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    sample_id = str(sample["sample_id"])
    mat_path = dataset_root / "parts" / "Annotations_Part" / f"{sample_id}.mat"
    image_path = voc_images_dir / f"{sample_id}.jpg"
    image = read_image(image_path)
    ann = loadmat(str(mat_path))["anno"]["objects"][0, 0]
    gt_mask = ann["parts"][0, int(sample["object_id"])]["mask"][0, int(sample["part_id"])]
    return image, (gt_mask > 0).astype(np.int32)


def load_partimagenet_mask(row: Dict[str, str], eval_root: Path) -> tuple[np.ndarray, np.ndarray]:
    image = read_image(eval_root / row["image_path"])
    mask = cv2.imread(str(eval_root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(row["mask_path"])
    return image, (mask > 0).astype(np.int32)


def iter_standard_samples(dataset_name: str, cfg) -> Iterator[Dict[str, object]]:
    dataset = utils.get_dataset(dataset_name, cfg)
    if dataset is None:
        raise RuntimeError(f"Unknown dataset: {dataset_name}")
    for index in range(len(dataset)):
        sample = dataset.get_sample(index)
        for object_id in sample.objects_ids:
            yield {
                "dataset": dataset_name,
                "sample_id": str(getattr(sample, "sample_id", index)),
                "object_id": int(object_id),
                "image": sample.image,
                "gt_mask": sample.gt_mask(object_id),
            }


def iter_pascalpart_samples(target: str, cfg) -> Iterator[Dict[str, object]]:
    keys_path = REPO_ROOT / "evaluate" / "sample_keys" / f"pascalpart_{target}_{'largest' if target != 'all' else 'random1000'}.json"
    payload = json.loads(keys_path.read_text(encoding="utf-8"))
    dataset_root = (REPO_ROOT / cfg.PASCALPART_PATH).resolve()
    voc_images_dir = (REPO_ROOT / cfg.PASCALPART_VOC_IMAGES_DIR).resolve()
    for item in payload:
        image, gt_mask = load_pascalpart_mask(item, dataset_root, voc_images_dir)
        yield {
            "dataset": f"PascalPart/{target}",
            "sample_id": str(item["sample_id"]),
            "object_id": int(item["object_id"]),
            "part_id": int(item["part_id"]),
            "part_name": str(item["part_name"]),
            "image": image,
            "gt_mask": gt_mask,
        }


def iter_partimagenet_samples(target: str, cfg) -> Iterator[Dict[str, object]]:
    name = f"partimagenet_{target if target != 'all' else 'all_random1000'}.csv"
    csv_path = REPO_ROOT / "evaluate" / "sample_keys" / name
    eval_root = (REPO_ROOT / cfg.PARTIMAGENET_EVAL_PATH).resolve()
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image, gt_mask = load_partimagenet_mask(row, eval_root)
            yield {
                "dataset": f"PartImageNet/{target}",
                "sample_id": row["sample_id"],
                "part_name": row["class_name"],
                "source_values": row["source_values"],
                "image": image,
                "gt_mask": gt_mask,
            }


def threshold_key(threshold: float) -> str:
    return str(int(round(threshold * 100)))


def summarize(records: List[Dict[str, object]], thresholds: List[float], max_clicks: int) -> Dict[str, object]:
    if not records:
        summary = {"samples": 0, "iou_at_1": None}
        for threshold in thresholds:
            key = threshold_key(threshold)
            summary[f"noc{key}"] = None
            summary[f"failures_at_max_clicks_{key}"] = 0
        summary["failures_at_max_clicks"] = 0
        return summary

    noc_values = {threshold: [] for threshold in thresholds}
    failures = {threshold: 0 for threshold in thresholds}
    iou1_values = []
    for record in records:
        ious = [float(v) for v in record["ious"]]
        for threshold in thresholds:
            noc = compute_realized_noc(ious, threshold, max_clicks)
            noc_values[threshold].append(noc)
            failures[threshold] += int(not any(iou >= threshold for iou in ious))
        iou1_values.append(ious[0] if ious else 0.0)

    summary = {"samples": len(records), "iou_at_1": float(np.mean(iou1_values))}
    for threshold in thresholds:
        key = threshold_key(threshold)
        summary[f"noc{key}"] = float(np.mean(noc_values[threshold]))
        summary[f"failures_at_max_clicks_{key}"] = int(failures[threshold])
    if 0.85 in thresholds:
        summary["failures_at_max_clicks"] = summary["failures_at_max_clicks_85"]
    return summary


def fmt_metric(value: object) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def evaluate_target(
    target_name: str,
    sample_iter: Iterable[Dict[str, object]],
    slider_values: List[float],
    args: argparse.Namespace,
    runtime_args,
    simpleclick_model,
    granularity_refiner,
    granularity_predictor,
    controller,
    output_dir: Path,
) -> None:
    target_dir = output_dir / target_name.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_summary_path = target_dir / "summary.json"
    if args.resume and target_summary_path.exists():
        print(f"[resume skip target] {target_name} -> {target_summary_path}", flush=True)
        return

    samples = list(sample_iter)
    samples = filter_skip_samples(target_name, samples, load_skip_samples(args.skip_samples_file))
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    noc_thresholds = [float(v) for v in split_csv(args.noc_thresholds)]
    stop_iou_threshold = max([args.iou_threshold, *noc_thresholds])

    target_summaries: Dict[str, object] = {}
    print(f"[target start] {target_name} samples={len(samples)} sliders={slider_values}", flush=True)
    for slider in slider_values:
        slider_tag = f"g{int(round(slider * 100)):03d}"
        result_path = target_dir / f"{slider_tag}.json"
        if args.resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            target_summaries[slider_tag] = payload["summary"]
            print(f"[resume skip slider] {target_name} {slider_tag}", flush=True)
            continue

        records = []
        skipped = []
        start = time.time()
        for index, sample in enumerate(samples, start=1):
            if args.refresh_every > 0 and index > 1 and (index - 1) % args.refresh_every == 0:
                del controller
                gc.collect()
                if runtime_args.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                controller = build_controller(runtime_args, simpleclick_model, granularity_refiner, granularity_predictor)

            print(f"[sample start] {target_name} {slider_tag} {index}/{len(samples)} sample={sample.get('sample_id')}", flush=True)
            try:
                result = evaluate_fixed_granularity(
                    controller,
                    sample["image"],
                    sample["gt_mask"],
                    granularity=slider,
                    max_clicks=args.max_clicks,
                    iou_threshold=stop_iou_threshold,
                )
                record = {k: v for k, v in sample.items() if k not in {"image", "gt_mask"}}
                record.update(result)
                records.append(record)
                ious = result["ious"]
                print(
                    f"[{target_name} {slider_tag} {index}/{len(samples)}] sample={sample.get('sample_id')} "
                    f"IoU@1={(ious[0] if ious else 0.0):.4f} last={(ious[-1] if ious else 0.0):.4f} clicks={len(ious)}",
                    flush=True,
                )
            except Exception as error:
                skipped.append({"sample_id": sample.get("sample_id"), "error": str(error)})
                print(f"[skip] {target_name} {slider_tag} sample={sample.get('sample_id')} error={error}", flush=True)

            if runtime_args.device.type == "cuda":
                torch.cuda.empty_cache()

        summary = summarize(records, noc_thresholds, args.max_clicks)
        summary.update({"target": target_name, "slider": slider, "elapsed_sec": time.time() - start, "skipped": len(skipped)})
        result_path.write_text(
            json.dumps({"summary": summary, "samples": records, "skipped": skipped}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        target_summaries[slider_tag] = summary
        noc_text = " ".join(
            f"NoC@{threshold_key(threshold)}={fmt_metric(summary.get(f'noc{threshold_key(threshold)}'))}"
            for threshold in noc_thresholds
        )
        print(f"[slider done] {target_name} {slider_tag} {noc_text} IoU@1={fmt_metric(summary['iou_at_1'])} skipped={len(skipped)}", flush=True)

    target_summary_path.write_text(json.dumps(target_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[target done] {target_name} -> {target_summary_path}", flush=True)


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


def resolve_pascalpart_paths(args):
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


def load_pascalpart_candidates(dataset_root: Path, voc_images_dir: Path, split_file: Path, part_name: str) -> List[Dict[str, object]]:
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


def load_pascalpart_sample(sample: Dict[str, object]):
    image = read_image(Path(sample["image_path"]))
    ann = loadmat(sample["mat_path"])["anno"]["objects"][0, 0]
    gt_mask = ann["parts"][0, int(sample["object_id"])]["mask"][0, int(sample["part_id"])]
    return image, (gt_mask > 0).astype(np.int32)


def normalize_sample_key(sample: Dict[str, object]) -> str:
    return f"{sample['sample_id']}:{int(sample['object_id'])}:{int(sample['part_id'])}"


def load_explicit_pascalpart_samples(sample_keys_file: Path, dataset_root: Path, voc_images_dir: Path) -> List[Dict[str, object]]:
    payload = json.loads(sample_keys_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("sample-keys-file must contain a JSON list.")
    anno_dir = dataset_root / "parts" / "Annotations_Part"
    return [
        {
            "sample_id": str(item["sample_id"]),
            "object_id": int(item["object_id"]),
            "part_id": int(item["part_id"]),
            "mat_path": str(anno_dir / f"{item['sample_id']}.mat"),
            "image_path": str(voc_images_dir / f"{item['sample_id']}.jpg"),
            "part_name": str(item.get("part_name", "")),
        }
        for item in payload
    ]


def evaluate_pascalpart_sample(controller, image: np.ndarray, gt_mask: np.ndarray, args) -> Dict[str, object]:
    candidate_values = [float(args.first_click_granularity)]
    if args.oracle_granularity:
        candidate_values = parse_granularity_values(args.oracle_granularity_values)

    candidate_results = [
        evaluate_fixed_granularity(
            controller,
            image,
            gt_mask,
            granularity=float(granularity),
            max_clicks=args.max_clicks,
            iou_threshold=args.iou_threshold,
        )
        for granularity in candidate_values
    ]
    candidate_results.sort(key=lambda item: float(item["granularity"]))

    selected_candidate = min(candidate_results, key=lambda item: (int(item["realized_noc"]), -float(item["iou_at_1"])))
    best_iou1_candidate = max(candidate_results, key=lambda item: (float(item["iou_at_1"]), -int(item["realized_noc"])))
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


def run_slider_evaluation(args: argparse.Namespace) -> None:
    cfg = exp.load_config_file(args.config_path, return_edict=True)
    protocol = json.loads(Path(args.protocol_path).read_text(encoding="utf-8"))
    slider_values = [float(v) for v in protocol["slider_values"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_args, simpleclick_model, granularity_refiner, granularity_predictor, controller = load_runtime(args, cfg)

    for dataset_name in split_csv(args.datasets):
        evaluate_target(dataset_name, iter_standard_samples(dataset_name, cfg), slider_values, args, runtime_args,
                        simpleclick_model, granularity_refiner, granularity_predictor, controller, output_dir)
    for target in split_csv(args.pascalpart_targets):
        evaluate_target(f"PascalPart/{target}", iter_pascalpart_samples(target, cfg), slider_values, args, runtime_args,
                        simpleclick_model, granularity_refiner, granularity_predictor, controller, output_dir)
    for target in split_csv(args.partimagenet_targets):
        evaluate_target(f"PartImageNet/{target}", iter_partimagenet_samples(target, cfg), slider_values, args, runtime_args,
                        simpleclick_model, granularity_refiner, granularity_predictor, controller, output_dir)

    print(f"[all done] output_dir={output_dir}", flush=True)


def run_pascalpart_evaluation(args: argparse.Namespace) -> None:
    cfg = exp.load_config_file(args.config_path, return_edict=True)
    runtime_args, simpleclick_model, granularity_refiner, granularity_predictor, controller = load_runtime(args, cfg)

    dataset_root, voc_images_dir, split_file = resolve_pascalpart_paths(args)
    if args.sample_keys_file:
        samples = load_explicit_pascalpart_samples(Path(args.sample_keys_file), dataset_root, voc_images_dir)
    else:
        samples = load_pascalpart_candidates(dataset_root, voc_images_dir, split_file, args.part_name)
        skip_sample_ids = {item.strip() for item in args.skip_sample_ids.split(",") if item.strip()}
        if skip_sample_ids:
            samples = [sample for sample in samples if sample["sample_id"] not in skip_sample_ids]
        if args.candidate_offset > 0:
            samples = samples[args.candidate_offset:]
    if not samples:
        raise RuntimeError("No matching PascalPart samples were found for the requested part and split.")

    target_sample_count = min(args.sample_limit, len(samples)) if args.sample_limit > 0 else len(samples)
    print(
        f"[eval] part={args.part_name} candidates={len(samples)} target_samples={target_sample_count} "
        f"first_click_granularity={args.first_click_granularity:.2f} oracle_granularity={args.oracle_granularity}",
        flush=True,
    )
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
            if runtime_args.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            controller = build_controller(runtime_args, simpleclick_model, granularity_refiner, granularity_predictor)
        try:
            image, gt_mask = load_pascalpart_sample(sample)
            result = evaluate_pascalpart_sample(controller, image, gt_mask, args)
        except Exception as error:
            skipped.append({**sample, "error": str(error)})
            print(
                f"[skip {candidate_index:03d}] {sample['sample_id']} obj={sample['object_id']} part={sample['part_id']} error={error}",
                flush=True,
            )
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
        print(
            f"[{len(all_ious):03d}/{target_sample_count:03d}] {sample['sample_id']} obj={sample['object_id']} "
            f"part={sample['part_id']} IoU@1={first_iou:.4f} IoU@last={final_iou:.4f} clicks={len(ious)} "
            f"g*={float(result['selected_granularity']):.2f} "
            f"maxIoU1@{float(result['best_iou_at_1_granularity']):.2f}={float(result['best_iou_at_1']):.4f}",
            flush=True,
        )

        if runtime_args.device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_ious:
        raise RuntimeError("All candidate samples failed during evaluation.")

    elapsed_time = time.time() - start_time
    noc_values, noc_std, over_max = utils.compute_noc_metric(all_ious, [args.iou_threshold], max_clicks=args.max_clicks)
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
    print(f"NoC@{int(round(args.iou_threshold * 100))}={noc_values[0]:.4f}", flush=True)
    print(f"NoC std={noc_std[0]:.4f}", flush=True)
    print(f">={args.max_clicks} failures={over_max[0]}", flush=True)
    print(f"mean realized clicks={mean_clicks:.4f}", flush=True)
    print(f"elapsed_sec={elapsed_time:.2f}", flush=True)

    if args.save_json:
        payload = {
            "part_name": args.part_name,
            "sample_count": len(samples),
            "iou_threshold": args.iou_threshold,
            "first_click_granularity": args.first_click_granularity,
            "oracle_granularity": bool(args.oracle_granularity),
            "oracle_granularity_values": parse_granularity_values(args.oracle_granularity_values),
            "noc": float(noc_values[0]),
            "noc_std": float(noc_std[0]),
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


def main() -> None:
    args = parse_args()
    if args.task == "pascalpart":
        run_pascalpart_evaluation(args)
    else:
        run_slider_evaluation(args)


if __name__ == "__main__":
    main()
