from __future__ import annotations

import argparse
import csv
import gc
import json
import random
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

from granularity_recovery.evaluate_part_noc85 import (
    build_controller,
    build_runtime_components,
    compute_realized_noc,
    evaluate_fixed_granularity,
    resolve_default_dinov3_repo,
    resolve_default_dinov3_weights,
)
from isegm.inference import utils
from isegm.utils import exp


STANDARD_DATASETS = ["GrabCut", "Berkeley", "SBD", "DAVIS", "PascalVOC"]
PASCALPART_TARGETS = ["head", "wheel", "torso", "all"]
PARTIMAGENET_TARGETS = ["head", "body", "foot", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="evaluate/config_server_eval.yml")
    parser.add_argument("--protocol-path", default="evaluate/protocols/all_sliders.json")
    parser.add_argument("--checkpoint", default="weights/simpleclick_models/cocolvis_vit_base.pth")
    parser.add_argument("--base-granularity-checkpoint", default="experiments/granularity_refine_rerun/best.pth")
    parser.add_argument("--output-dir", default="outputs/server_slider_eval")
    parser.add_argument("--datasets", default=",".join(STANDARD_DATASETS))
    parser.add_argument("--pascalpart-targets", default=",".join(PASCALPART_TARGETS))
    parser.add_argument("--partimagenet-targets", default=",".join(PARTIMAGENET_TARGETS))
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--noc-thresholds", default="0.85,0.90")
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
    parser.add_argument("--sample-limit", type=int, default=0, help="Debug cap per target; 0 means no cap.")
    parser.add_argument("--skip-samples-file", default="", help="Optional newline list of target,sample_id entries to skip.")
    return parser.parse_args()


def split_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


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


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image, gt_mask = load_partimagenet_mask(row, eval_root)
            yield {
                "dataset": f"PartImageNet/{target}",
                "sample_id": row["sample_id"],
                "part_name": row["class_name"],
                "source_values": row["source_values"],
                "image": image,
                "gt_mask": gt_mask,
            }


def load_runtime(args: argparse.Namespace, cfg):
    runtime_args = SimpleNamespace(**vars(args))
    runtime_args.device = torch.device("cpu" if args.cpu else args.device)
    runtime_args.dinov3_repo = args.dinov3_repo or resolve_default_dinov3_repo()
    runtime_args.dinov3_weights = args.dinov3_weights or resolve_default_dinov3_weights(args.dinov3_model)

    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    simpleclick_model = utils.load_is_model(checkpoint_path, runtime_args.device, eval_ritm=False, cpu_dist_maps=True)
    granularity_refiner, base_granularity_predictor = build_runtime_components(runtime_args)
    controller = build_controller(runtime_args, simpleclick_model, granularity_refiner, base_granularity_predictor)
    return runtime_args, simpleclick_model, granularity_refiner, base_granularity_predictor, controller


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

    summary = {
        "samples": len(records),
        "iou_at_1": float(np.mean(iou1_values)),
    }
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
    base_granularity_predictor,
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
                controller = build_controller(runtime_args, simpleclick_model, granularity_refiner, base_granularity_predictor)

            print(
                f"[sample start] {target_name} {slider_tag} {index}/{len(samples)} "
                f"sample={sample.get('sample_id')}",
                flush=True,
            )
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
                    f"[{target_name} {slider_tag} {index}/{len(samples)}] "
                    f"sample={sample.get('sample_id')} IoU@1={(ious[0] if ious else 0.0):.4f} "
                    f"last={(ious[-1] if ious else 0.0):.4f} clicks={len(ious)}",
                    flush=True,
                )
            except Exception as error:
                skipped.append({"sample_id": sample.get("sample_id"), "error": str(error)})
                print(f"[skip] {target_name} {slider_tag} sample={sample.get('sample_id')} error={error}", flush=True)

            if runtime_args.device.type == "cuda":
                torch.cuda.empty_cache()

        summary = summarize(records, noc_thresholds, args.max_clicks)
        summary.update({
            "target": target_name,
            "slider": slider,
            "elapsed_sec": time.time() - start,
            "skipped": len(skipped),
        })
        result_path.write_text(
            json.dumps({"summary": summary, "samples": records, "skipped": skipped}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        target_summaries[slider_tag] = summary
        noc_text = " ".join(
            f"NoC@{threshold_key(threshold)}={fmt_metric(summary.get(f'noc{threshold_key(threshold)}'))}"
            for threshold in noc_thresholds
        )
        print(
            f"[slider done] {target_name} {slider_tag} "
            f"{noc_text} "
            f"IoU@1={fmt_metric(summary['iou_at_1'])} skipped={len(skipped)}",
            flush=True,
        )

    target_summary_path.write_text(json.dumps(target_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[target done] {target_name} -> {target_summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = exp.load_config_file(args.config_path, return_edict=True)
    protocol = json.loads(Path(args.protocol_path).read_text(encoding="utf-8"))
    slider_values = [float(v) for v in protocol["slider_values"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime(args, cfg)
    runtime_args, simpleclick_model, granularity_refiner, base_granularity_predictor, controller = runtime

    for dataset_name in split_csv(args.datasets):
        evaluate_target(
            dataset_name,
            iter_standard_samples(dataset_name, cfg),
            slider_values,
            args,
            runtime_args,
            simpleclick_model,
            granularity_refiner,
            base_granularity_predictor,
            controller,
            output_dir,
        )

    for target in split_csv(args.pascalpart_targets):
        evaluate_target(
            f"PascalPart/{target}",
            iter_pascalpart_samples(target, cfg),
            slider_values,
            args,
            runtime_args,
            simpleclick_model,
            granularity_refiner,
            base_granularity_predictor,
            controller,
            output_dir,
        )

    for target in split_csv(args.partimagenet_targets):
        evaluate_target(
            f"PartImageNet/{target}",
            iter_partimagenet_samples(target, cfg),
            slider_values,
            args,
            runtime_args,
            simpleclick_model,
            granularity_refiner,
            base_granularity_predictor,
            controller,
            output_dir,
        )

    print(f"[all done] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
