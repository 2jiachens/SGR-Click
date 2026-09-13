from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import zipfile
from functools import lru_cache
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
from granularity_recovery.soft_semantic_controller import SoftSemanticFusionController
from evaluate.reproducibility import seed_everything, write_run_manifest
from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.inference.dino_granularity import DinoGranularityRefiner
from isegm.inference.soft_dino_semantic import SoftDinoSemanticRefiner
from isegm.inference.direct_dino_fusion import DirectDinoFusionRefiner
from isegm.utils import exp


STANDARD_DATASETS = ["GrabCut", "Berkeley", "SBD", "DAVIS", "PascalVOC"]
PASCALPART_TARGETS = ["head", "wheel", "torso", "all"]
PARTIMAGENET_TARGETS = ["head", "body", "foot", "all"]
PAPER_SAMPLES_PATH = REPO_ROOT / "evaluate" / "paper_samples.json"
PAPER_PROTOCOLS_PATH = REPO_ROOT / "evaluate" / "paper_protocols.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="config.yml")
    parser.add_argument(
        "--protocol",
        choices=["standard_ten", "ablation_refinement", "ablation_expansion"],
        default="standard_ten",
    )
    parser.add_argument("--checkpoint", default="weights/simpleclick_models/cocolvis_vit_base.pth")
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
        help=(
            "Training-free component ablation: full hierarchy plus boundary "
            "projection; hierarchy without projection; or the single-granularity "
            "SimpleClick prior."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/standard_ten")
    parser.add_argument("--datasets", default=",".join(STANDARD_DATASETS))
    parser.add_argument(
        "--pascalpart-targets",
        default="",
        help="Optional comma-separated PascalPart targets; disabled by default.",
    )
    parser.add_argument(
        "--partimagenet-targets",
        default="",
        help="Optional comma-separated PartImageNet targets; disabled by default.",
    )
    parser.add_argument("--sample-limit", type=int, default=0, help="Debug cap per target; 0 means no cap.")
    parser.add_argument("--sample-shard-id", type=int, default=0)
    parser.add_argument("--num-sample-shards", type=int, default=1)
    parser.add_argument("--skip-samples-file", default="", help="Slider-only: optional newline list of target,sample_id entries to skip.")
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--noc-thresholds", default="0.85,0.90")
    parser.add_argument("--pred-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit-longest-size", type=int, default=800)
    parser.add_argument("--dinov3-repo", default=None)
    parser.add_argument("--dinov3-model", default="dinov3_vitl16", choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"])
    parser.add_argument("--dinov3-weights", default=None)
    parser.add_argument("--dinov3-image-size", type=int, default=768)
    parser.add_argument(
        "--positional-debias-mode",
        choices=["none", "zero", "random"],
        default="zero",
    )
    parser.add_argument("--positional-debias-seed", type=int, default=0)
    parser.add_argument(
        "--gui-slider-protocol",
        action="store_true",
        default=False,
        help=(
            "Create the standard first click at slider 0.50, then move the "
            "slider to the requested value before measuring the first-click mask."
        ),
    )
    parser.add_argument("--refresh-every", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    if args.num_sample_shards < 1:
        parser.error("--num-sample-shards must be >= 1")
    if not 0 <= args.sample_shard_id < args.num_sample_shards:
        parser.error("--sample-shard-id must satisfy 0 <= id < num shards")
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


@lru_cache(maxsize=1)
def load_paper_samples() -> Dict[str, object]:
    return json.loads(PAPER_SAMPLES_PATH.read_text(encoding="utf-8"))


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_runtime_components(args):
    if args.ablation_mode == "simple_prior":
        return None

    if args.ablation_mode == "external_hierarchy":
        return DinoGranularityRefiner(
            repo_path=args.dinov3_repo,
            weights_path=args.dinov3_weights,
            device=args.device,
            model_name=args.dinov3_model,
            image_size=args.dinov3_image_size,
            candidate_threshold=0.50,
            positional_debias_mode=getattr(args, "positional_debias_mode", "zero"),
            positional_debias_seed=getattr(args, "positional_debias_seed", 0),
            cluster_tau=0.60,
        )

    if args.ablation_mode == "direct_dino_fusion":
        return DirectDinoFusionRefiner(
            repo_path=args.dinov3_repo,
            weights_path=args.dinov3_weights,
            device=args.device,
            model_name=args.dinov3_model,
            image_size=args.dinov3_image_size,
            positional_debias_mode=getattr(args, "positional_debias_mode", "zero"),
            positional_debias_seed=getattr(args, "positional_debias_seed", 0),
            prototype_radius=getattr(args, "prototype_radius", 2.5),
            similarity_temperature=getattr(args, "similarity_temperature", 0.08),
            logit_gain=getattr(args, "soft_logit_gain", 4.5),
        )

    granularity_refiner = SoftDinoSemanticRefiner(
        repo_path=args.dinov3_repo,
        weights_path=args.dinov3_weights,
        device=args.device,
        model_name=args.dinov3_model,
        image_size=args.dinov3_image_size,
        positional_debias_mode=getattr(args, "positional_debias_mode", "zero"),
        positional_debias_seed=getattr(args, "positional_debias_seed", 0),
        expansion_evidence_floor=(
            0.0
            if args.ablation_mode == "soft_no_evidence_filter"
            else getattr(args, "expansion_evidence_floor", 0.15)
        ),
        cluster_tau=getattr(args, "cluster_tau", 0.60),
        soft_logit_gain=getattr(args, "soft_logit_gain", 4.5),
        prototype_radius=getattr(args, "prototype_radius", 2.5),
        similarity_temperature=getattr(args, "similarity_temperature", 0.08),
        enable_connected_support=args.ablation_mode != "soft_no_connected_support",
        enable_soft_endpoint=args.ablation_mode not in {
            "soft_no_endpoint",
            "soft_no_endpoint_no_projection",
        },
    )
    return granularity_refiner


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
            "direct_dino_fusion",
            "soft_no_endpoint_no_projection",
        },
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
    gui_slider_protocol: bool = False,
) -> Dict[str, object]:
    controller.set_image(image)
    controller._requested_granularity = 0.50 if gui_slider_protocol else float(granularity)

    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask, dtype=bool)
    ious: List[float] = []
    clicks: List[Dict[str, object]] = []

    for click_index in range(max_clicks):
        clicker.make_next_click(pred_mask)
        current_click = clicker.clicks_list[-1]
        y, x = current_click.coords

        controller.add_click(int(x), int(y), bool(current_click.is_positive))
        if click_index == 0 and gui_slider_protocol:
            controller.set_granularity(float(granularity))

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
    # The legacy Cython CPU distance-map implementation uses a fixed-size
    # propagation queue that can overflow with many clicks and corrupt the
    # process heap.  The native PyTorch implementation computes the same
    # coordinate features without that unsafe queue.
    simpleclick_model = utils.load_is_model(
        checkpoint_path,
        runtime_args.device,
        eval_ritm=False,
        cpu_dist_maps=False,
    )
    granularity_refiner = build_runtime_components(runtime_args)
    controller = build_controller(runtime_args, simpleclick_model, granularity_refiner)
    return runtime_args, simpleclick_model, granularity_refiner, controller


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


def select_sample_shard(samples: List[Dict[str, object]], shard_id: int, num_shards: int) -> List[Dict[str, object]]:
    if num_shards == 1:
        return samples
    return samples[shard_id::num_shards]


def evaluation_sample_key(sample: Dict[str, object]) -> str:
    identity = {
        "sample_id": sample.get("sample_id"),
        "object_id": sample.get("object_id"),
        "part_id": sample.get("part_id"),
        "part_name": sample.get("part_name"),
        "source_values": sample.get("source_values"),
    }
    return json.dumps(identity, ensure_ascii=False, sort_keys=True)


def write_json_atomic(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_pascalpart_mask(sample: Dict[str, object], dataset_root: Path, voc_images_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    sample_id = str(sample["sample_id"])
    annotation_candidates = [
        dataset_root / "parts" / "Annotations_Part" / f"{sample_id}.mat",
        dataset_root / "Annotations_Part" / f"{sample_id}.mat",
        dataset_root / f"{sample_id}.mat",
    ]
    image_candidates = [
        voc_images_dir / f"{sample_id}.jpg",
        voc_images_dir / "train" / f"{sample_id}.jpg",
        voc_images_dir / "val" / f"{sample_id}.jpg",
    ]
    mat_path = next(
        (path for path in annotation_candidates if path.is_file()),
        annotation_candidates[0],
    )
    image_path = next(
        (path for path in image_candidates if path.is_file()), image_candidates[0]
    )
    image = read_image(image_path)
    ann = loadmat(str(mat_path))["anno"]["objects"][0, 0]
    gt_mask = ann["parts"][0, int(sample["object_id"])]["mask"][0, int(sample["part_id"])]
    return image, (gt_mask > 0).astype(np.int32)


@lru_cache(maxsize=1)
def open_partimagenet_archive(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def load_partimagenet_mask(
    row: Dict[str, str],
    eval_root: Path,
    source_zip: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if source_zip is not None and source_zip.is_file():
        archive = open_partimagenet_archive(str(source_zip))
        split = str(row["split"])
        sample_id = str(row["sample_id"])
        image_payload = archive.read(
            f"PartImageNet/images/{split}/{sample_id}.JPEG"
        )
        annotation_payload = archive.read(
            f"PartImageNet/annotations/{split}/{sample_id}.png"
        )
        image_bgr = cv2.imdecode(
            np.frombuffer(image_payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        labels = cv2.imdecode(
            np.frombuffer(annotation_payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if image_bgr is None or labels is None:
            raise RuntimeError(f"failed to decode PartImageNet sample {sample_id}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        source_values = [
            int(value)
            for value in str(row["source_values"]).replace(";", ",").split(",")
            if value.strip()
        ]
        return image, np.isin(labels, source_values).astype(np.int32)

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
    payload = load_paper_samples()["pascalpart"][target]
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
    eval_root = (REPO_ROOT / cfg.PARTIMAGENET_EVAL_PATH).resolve()
    source_zip_value = getattr(cfg, "PARTIMAGENET_SOURCE_ZIP", "")
    source_zip = (
        (REPO_ROOT / source_zip_value).resolve() if source_zip_value else None
    )
    for row in load_paper_samples()["partimagenet"][target]:
        image, gt_mask = load_partimagenet_mask(row, eval_root, source_zip)
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
    samples = select_sample_shard(samples, args.sample_shard_id, args.num_sample_shards)
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
        partial_path = target_dir / f"{slider_tag}.partial.json"
        if args.resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            target_summaries[slider_tag] = payload["summary"]
            print(f"[resume skip slider] {target_name} {slider_tag}", flush=True)
            continue

        records = []
        skipped = []
        if args.resume and partial_path.exists():
            partial_payload = json.loads(partial_path.read_text(encoding="utf-8"))
            records = list(partial_payload.get("samples", []))
            skipped = list(partial_payload.get("skipped", []))
            print(
                f"[resume partial slider] {target_name} {slider_tag} "
                f"records={len(records)} skipped={len(skipped)}",
                flush=True,
            )
        completed_keys = {
            evaluation_sample_key(record)
            for record in [*records, *skipped]
        }
        newly_processed = 0
        start = time.time()
        for index, sample in enumerate(samples, start=1):
            if evaluation_sample_key(sample) in completed_keys:
                continue
            if args.refresh_every > 0 and index > 1 and (index - 1) % args.refresh_every == 0:
                del controller
                gc.collect()
                if runtime_args.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                controller = build_controller(
                    runtime_args,
                    simpleclick_model,
                    granularity_refiner,
                )

            print(f"[sample start] {target_name} {slider_tag} {index}/{len(samples)} sample={sample.get('sample_id')}", flush=True)
            try:
                result = evaluate_fixed_granularity(
                    controller,
                    sample["image"],
                    sample["gt_mask"],
                    granularity=slider,
                    max_clicks=args.max_clicks,
                    iou_threshold=stop_iou_threshold,
                    gui_slider_protocol=args.gui_slider_protocol,
                )
                record = {k: v for k, v in sample.items() if k not in {"image", "gt_mask"}}
                record.update(result)
                records.append(record)
                completed_keys.add(evaluation_sample_key(record))
                ious = result["ious"]
                print(
                    f"[{target_name} {slider_tag} {index}/{len(samples)}] sample={sample.get('sample_id')} "
                    f"IoU@1={(ious[0] if ious else 0.0):.4f} last={(ious[-1] if ious else 0.0):.4f} clicks={len(ious)}",
                    flush=True,
                )
            except Exception as error:
                skipped_record = {
                    key: value
                    for key, value in sample.items()
                    if key not in {"image", "gt_mask"}
                }
                skipped_record["error"] = str(error)
                skipped.append(skipped_record)
                completed_keys.add(evaluation_sample_key(skipped_record))
                print(f"[skip] {target_name} {slider_tag} sample={sample.get('sample_id')} error={error}", flush=True)

            newly_processed += 1
            if newly_processed % 25 == 0:
                write_json_atomic(
                    partial_path,
                    {"samples": records, "skipped": skipped},
                )

            if runtime_args.device.type == "cuda":
                torch.cuda.empty_cache()

        summary = summarize(records, noc_thresholds, args.max_clicks)
        summary.update({
            "target": target_name,
            "slider": slider,
            "ablation_mode": args.ablation_mode,
            "sample_shard_id": args.sample_shard_id,
            "num_sample_shards": args.num_sample_shards,
            "elapsed_sec": time.time() - start,
            "skipped": len(skipped),
        })
        write_json_atomic(
            result_path,
            {"summary": summary, "samples": records, "skipped": skipped},
        )
        partial_path.unlink(missing_ok=True)
        target_summaries[slider_tag] = summary
        noc_text = " ".join(
            f"NoC@{threshold_key(threshold)}={fmt_metric(summary.get(f'noc{threshold_key(threshold)}'))}"
            for threshold in noc_thresholds
        )
        print(f"[slider done] {target_name} {slider_tag} {noc_text} IoU@1={fmt_metric(summary['iou_at_1'])} skipped={len(skipped)}", flush=True)

    target_summary_path.write_text(json.dumps(target_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[target done] {target_name} -> {target_summary_path}", flush=True)


def run_slider_evaluation(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    cfg = exp.load_config_file(args.config_path, return_edict=True)
    protocols = json.loads(PAPER_PROTOCOLS_PATH.read_text(encoding="utf-8"))
    protocol = protocols[args.protocol]
    slider_values = [float(v) for v in protocol["slider_values"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir,
        args,
        REPO_ROOT,
        artifacts=[
            args.config_path,
            PAPER_PROTOCOLS_PATH,
            PAPER_SAMPLES_PATH,
            args.checkpoint,
            args.dinov3_weights,
        ],
    )

    runtime_args, simpleclick_model, granularity_refiner, controller = load_runtime(args, cfg)

    aggregate_targets: Dict[str, object] = {}

    def run_target(target_name, samples):
        evaluate_target(
            target_name,
            samples,
            slider_values,
            args,
            runtime_args,
            simpleclick_model,
            granularity_refiner,
            controller,
            output_dir,
        )
        summary_path = output_dir / target_name.replace("/", "__") / "summary.json"
        if summary_path.exists():
            aggregate_targets[target_name] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

    for dataset_name in split_csv(args.datasets):
        run_target(dataset_name, iter_standard_samples(dataset_name, cfg))
    for target in split_csv(args.pascalpart_targets):
        run_target(f"PascalPart/{target}", iter_pascalpart_samples(target, cfg))
    for target in split_csv(args.partimagenet_targets):
        run_target(f"PartImageNet/{target}", iter_partimagenet_samples(target, cfg))

    write_json_atomic(
        output_dir / "aggregate.json",
        {
            "protocol": args.protocol,
            "checkpoint": args.checkpoint,
            "datasets": split_csv(args.datasets),
            "pascalpart_targets": split_csv(args.pascalpart_targets),
            "partimagenet_targets": split_csv(args.partimagenet_targets),
            "targets": aggregate_targets,
        },
    )
    print(f"[all done] output_dir={output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    run_slider_evaluation(args)


if __name__ == "__main__":
    main()
