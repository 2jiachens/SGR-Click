from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isegm.inference import clicker, utils
from isegm.inference.dino_granularity import DinoGranularityRefiner
from isegm.inference.predictors import get_predictor
from granularity_recovery.granularity_mapping import (
    map_display_granularities_for_hierarchy,
    map_display_granularities_for_model,
)


DEFAULT_GRANULARITIES = (0.0, 0.08, 0.18, 0.32, 0.50, 0.68, 0.82, 0.92, 1.0)
DEFAULT_MAX_REFINE_CLICKS = 2
DEFAULT_MAX_COMPLETION_CLICKS = 2
DEFAULT_FIRST_CLICK_VARIANTS = 2


def resolve_default_checkpoint():
    env_path = os.environ.get("SIMPLECLICK_CHECKPOINT")
    if env_path:
        return env_path
    return str(REPO_ROOT / "weights" / "simpleclick_models" / "cocolvis_vit_base.pth")


def resolve_default_dinov3_repo():
    env_path = os.environ.get("DINOV3_REPO")
    if env_path:
        return env_path
    return str(REPO_ROOT / "third_party" / "dinov3")


def resolve_default_dinov3_weights(model_name):
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
    parser.add_argument("--sbd-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dinov3-repo", type=str, default=None)
    parser.add_argument("--dinov3-model", type=str, default="dinov3_vitl16",
                        choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"])
    parser.add_argument("--dinov3-weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--granularities", type=float, nargs="+", default=list(DEFAULT_GRANULARITIES))
    parser.add_argument("--first-click-variants", type=int, default=DEFAULT_FIRST_CLICK_VARIANTS)
    parser.add_argument("--min-object-area", type=int, default=400)
    parser.add_argument("--preview-per-split", type=int, default=6)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    parser.add_argument("--candidate-threshold", type=float, default=0.50)
    parser.add_argument("--max-hole-ratio", type=float, default=0.04)
    parser.add_argument("--work-size", type=int, default=512)
    parser.add_argument("--segments", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--log-every-images", type=int, default=25)
    parser.add_argument("--max-refine-clicks", type=int, default=DEFAULT_MAX_REFINE_CLICKS)
    parser.add_argument("--max-completion-clicks", type=int, default=DEFAULT_MAX_COMPLETION_CLICKS)
    parser.add_argument("--max-completion-candidates", type=int, default=4)
    parser.add_argument("--max-completion-future-levels", type=int, default=2)
    parser.add_argument("--completion-min-iou-gain", type=float, default=0.003)
    parser.add_argument("--completion-max-fp-ratio", type=float, default=0.03)
    parser.add_argument("--completion-min-candidate-precision", type=float, default=0.60)
    parser.add_argument("--completion-min-added-precision", type=float, default=0.70)
    parser.add_argument("--max-completion-gt-clicks", type=int, default=2)
    parser.add_argument("--completion-target-recall", type=float, default=0.90)
    parser.add_argument("--completion-overflow-trigger-ratio", type=float, default=0.005)
    parser.add_argument("--max-level-extra-gt-clicks", type=int, default=3)
    parser.add_argument("--max-level-target-recall", type=float, default=0.98)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = resolve_default_checkpoint()
    if args.dinov3_repo is None:
        args.dinov3_repo = resolve_default_dinov3_repo()
    if args.dinov3_weights is None:
        args.dinov3_weights = resolve_default_dinov3_weights(args.dinov3_model)
    return args


def read_split_ids(split_file: Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_sbd_dataset_root(path: Path) -> Path:
    candidates = [path, path / "benchmark_RELEASE" / "dataset", path / "dataset"]
    for candidate in candidates:
        if (candidate / "img").is_dir() and (candidate / "inst").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate SBD dataset root.")


def deterministic_train_val_split(image_ids: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    scored = []
    for image_id in image_ids:
        score = int(hashlib.md5(image_id.encode("utf-8")).hexdigest(), 16)
        scored.append((score, image_id))
    scored.sort()
    val_count = max(1, int(round(len(scored) * val_ratio)))
    val_ids = [image_id for _, image_id in scored[:val_count]]
    train_ids = [image_id for _, image_id in scored[val_count:]]
    return train_ids, val_ids


def ensure_dirs(root: Path):
    for rel in [
        "records/train",
        "records/val",
        "records/test",
        "granularity_recovery_previews/train",
        "granularity_recovery_previews/val",
        "granularity_recovery_previews/test",
        "splits",
        "metadata",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray):
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def load_sbd_image_and_instances(dataset_root: Path, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
    image_path = dataset_root / "img" / f"{image_id}.jpg"
    inst_path = dataset_root / "inst" / f"{image_id}.mat"
    image = imread_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"Missing SBD image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    instances_mask = loadmat(str(inst_path))["GTinst"][0][0][0].astype(np.int32)
    return image, instances_mask


def instance_ids_from_mask(instances_mask: np.ndarray) -> List[int]:
    ids = np.unique(instances_mask)
    return [int(x) for x in ids.tolist() if x > 0]


def pick_first_positive_click(gt_mask: np.ndarray) -> Tuple[int, int]:
    padded = np.pad(gt_mask.astype(np.uint8), ((1, 1), (1, 1)), mode="constant")
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return int(x), int(y)


def _odd_kernel_size(area: int, scale: float, minimum: int, maximum: int) -> int:
    size = int(round(np.sqrt(max(area, 1)) * scale))
    size = max(minimum, min(size, maximum))
    if size % 2 == 0:
        size += 1
    return size


def _select_farthest_click(mask: np.ndarray, reference_points: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    coords_yx = np.argwhere(mask > 0)
    if coords_yx.size == 0:
        return None
    coords_xy = coords_yx[:, [1, 0]].astype(np.float32)
    if not reference_points:
        choice = coords_xy[len(coords_xy) // 2]
        return int(choice[0]), int(choice[1])

    ref_xy = np.asarray(reference_points, dtype=np.float32)
    sq_dist = ((coords_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2)
    score = sq_dist.min(axis=1)
    best_index = int(np.argmax(score))
    best_xy = coords_xy[best_index]
    return int(best_xy[0]), int(best_xy[1])


def sample_first_positive_clicks(
    gt_mask: np.ndarray,
    max_variants: int = DEFAULT_FIRST_CLICK_VARIANTS,
) -> List[Tuple[Tuple[int, int], str]]:
    max_variants = max(int(max_variants), 1)
    area = int(gt_mask.sum())
    center_click = pick_first_positive_click(gt_mask)

    samples: List[Tuple[Tuple[int, int], str]] = []
    used = set()

    def register(click_xy: Optional[Tuple[int, int]], click_role: str):
        if click_xy is None:
            return
        key = (int(click_xy[0]), int(click_xy[1]))
        if key in used:
            return
        used.add(key)
        samples.append((key, click_role))

    register(center_click, "center")
    if len(samples) >= max_variants:
        return samples

    safe_kernel = _odd_kernel_size(area, scale=0.08, minimum=7, maximum=25)
    safe_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_kernel, safe_kernel))
    safe_inner_mask = cv2.erode(gt_mask.astype(np.uint8), safe_structure, iterations=1)
    if int(safe_inner_mask.sum()) < 16:
        fallback_kernel = _odd_kernel_size(area, scale=0.05, minimum=5, maximum=17)
        fallback_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fallback_kernel, fallback_kernel))
        safe_inner_mask = cv2.erode(gt_mask.astype(np.uint8), fallback_structure, iterations=1)
    if int(safe_inner_mask.sum()) < 16:
        safe_inner_mask = gt_mask.astype(np.uint8)

    relaxed_kernel = _odd_kernel_size(area, scale=0.04, minimum=5, maximum=15)
    relaxed_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (relaxed_kernel, relaxed_kernel))
    relaxed_inner_mask = cv2.erode(gt_mask.astype(np.uint8), relaxed_structure, iterations=1)
    if int(relaxed_inner_mask.sum()) < 16:
        relaxed_inner_mask = safe_inner_mask.copy()

    register(_select_farthest_click(safe_inner_mask, [center_click]), "interior_offset")
    if len(samples) >= max_variants:
        return samples

    register(
        _select_farthest_click(relaxed_inner_mask, [click_xy for click_xy, _ in samples]),
        "extreme_interior",
    )
    if len(samples) >= max_variants:
        return samples

    while len(samples) < max_variants:
        spread_click = _select_farthest_click(relaxed_inner_mask, [click_xy for click_xy, _ in samples])
        before = len(samples)
        register(spread_click, f"interior_spread{len(samples)}")
        if len(samples) == before:
            break
    return samples


def click_to_map(height: int, width: int, click_xy: Tuple[int, int], sigma: float = 10.0) -> np.ndarray:
    x, y = click_xy
    yy, xx = np.mgrid[0:height, 0:width]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    return np.exp(-distance_sq / (2.0 * sigma * sigma)).astype(np.float32)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy().astype(np.float32)
    alpha = 0.45
    color_arr = np.asarray(color, dtype=np.float32)
    canvas[mask > 0] = (1.0 - alpha) * canvas[mask > 0] + alpha * color_arr
    return canvas.astype(np.uint8)


def overlay_delta(image: np.ndarray, base_mask: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    canvas = image.copy().astype(np.float32)
    added = np.logical_and(refined_mask > 0, base_mask <= 0)
    removed = np.logical_and(base_mask > 0, refined_mask <= 0)
    kept = np.logical_and(base_mask > 0, refined_mask > 0)
    canvas[kept] = 0.70 * canvas[kept] + 0.30 * np.asarray([255, 180, 0], dtype=np.float32)
    canvas[added] = 0.45 * canvas[added] + 0.55 * np.asarray([0, 220, 255], dtype=np.float32)
    canvas[removed] = 0.45 * canvas[removed] + 0.55 * np.asarray([255, 32, 32], dtype=np.float32)
    return canvas.astype(np.uint8)


def draw_clicks(
    image: np.ndarray,
    first_click_xy: Tuple[int, int],
    corrective_clicks: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    first_x, first_y = int(first_click_xy[0]), int(first_click_xy[1])
    cv2.circle(canvas, (first_x, first_y), 7, (0, 255, 0), 2)
    cv2.circle(canvas, (first_x, first_y), 2, (0, 255, 0), -1)
    for item in corrective_clicks:
        if int(item[0]) < 0:
            continue
        click_x, click_y, is_positive = int(item[0]), int(item[1]), int(item[2])
        color = (0, 255, 255) if is_positive > 0 else (255, 0, 0)
        cv2.circle(canvas, (click_x, click_y), 6, color, 2)
        cv2.line(canvas, (click_x - 4, click_y), (click_x + 4, click_y), color, 2)
        cv2.line(canvas, (click_x, click_y - 4), (click_x, click_y + 4), color, 2)
    return canvas


def probability_heatmap(prob: np.ndarray) -> np.ndarray:
    prob_uint8 = np.clip(prob * 255.0, 0.0, 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def append_jsonl(path: Path, record: Dict):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_teacher_prob_triplet(blob) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_teacher_probs = (
        blob["base_teacher_probs"].astype(np.float32)
        if "base_teacher_probs" in blob.files
        else blob["refine_teacher_probs"].astype(np.float32)
    )
    recovery_teacher_probs = (
        blob["recovery_teacher_probs"].astype(np.float32)
        if "recovery_teacher_probs" in blob.files
        else blob["completion_teacher_probs"].astype(np.float32)
    )
    training_teacher_probs = (
        blob["training_teacher_probs"].astype(np.float32)
        if "training_teacher_probs" in blob.files
        else np.where(
            blob["granularities"].astype(np.float32)[:, None, None] <= 0.5,
            base_teacher_probs,
            recovery_teacher_probs,
        )
    )
    return base_teacher_probs, recovery_teacher_probs, training_teacher_probs


def load_existing_metadata(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records[item["sample_name"]] = item
    return records


def write_split_files(output_root: Path, split_to_ids: Dict[str, List[str]]):
    for split_name, image_ids in split_to_ids.items():
        split_path = output_root / "splits" / f"{split_name}.txt"
        with open(split_path, "w", encoding="utf-8") as handle:
            for image_id in image_ids:
                handle.write(f"{image_id}\n")


def run_simpleclick_prediction(predictor, image: np.ndarray, clicks: List[clicker.Click], prev_prediction=None) -> np.ndarray:
    local_clicker = clicker.Clicker()
    predictor.set_input_image(image)
    if prev_prediction is not None:
        predictor.prev_prediction = torch.from_numpy(prev_prediction.astype(np.float32)).to(
            predictor.device
        ).unsqueeze(0).unsqueeze(0)
    for current_click in clicks:
        local_clicker.add_click(current_click)
    return predictor.get_prediction(local_clicker)


def build_runtime_slider_masks(
    refiner: DinoGranularityRefiner,
    image: np.ndarray,
    simple_prob: np.ndarray,
    click_xy: Tuple[int, int],
    granularities: Sequence[float],
) -> List[np.ndarray]:
    refiner.prepare_image(image)
    refiner.build(image, simple_prob.astype(np.float32), click_xy)
    return [refiner.mask_at(float(granularity)).astype(np.uint8) for granularity in granularities]


def compute_ious(masks: Sequence[np.ndarray], gt_mask: np.ndarray) -> np.ndarray:
    gt = gt_mask.astype(bool)
    values = []
    for mask in masks:
        pred = mask.astype(bool)
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        values.append(float(inter / max(union, 1)))
    return np.asarray(values, dtype=np.float32)


def compute_single_iou(mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / max(union, 1))


def compute_false_positive_ratio(mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = mask.astype(bool)
    gt = gt_mask.astype(bool)
    false_positive = np.logical_and(pred, np.logical_not(gt)).sum()
    gt_area = gt.sum()
    return float(false_positive / max(gt_area, 1))


def compute_gt_recall(mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = mask.astype(bool)
    gt = gt_mask.astype(bool)
    true_positive = np.logical_and(pred, gt).sum()
    gt_area = gt.sum()
    return float(true_positive / max(gt_area, 1))


def build_click_local_core(
    mask: np.ndarray,
    click_xy: Tuple[int, int],
    display_granularity: float,
) -> np.ndarray:
    mask_bool = mask.astype(bool)
    height, width = mask_bool.shape
    click_x = int(np.clip(click_xy[0], 0, width - 1))
    click_y = int(np.clip(click_xy[1], 0, height - 1))

    component_source = mask_bool.astype(np.uint8)
    if not component_source[click_y, click_x]:
        repaired = component_source.copy()
        repaired[click_y, click_x] = 1
        component_source = repaired
    _, labels = cv2.connectedComponents(component_source, connectivity=8)
    component_label = int(labels[click_y, click_x])
    component = labels == component_label if component_label > 0 else component_source.astype(bool)

    local = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
    radius = max(2, int(round(3 + 5 * local)))
    disk = np.zeros_like(component_source, dtype=np.uint8)
    cv2.circle(disk, (click_x, click_y), radius, 1, -1)
    local_core = np.logical_and(component, disk.astype(bool))
    if int(local_core.sum()) < 9:
        dilated = cv2.dilate(disk, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        local_core = np.logical_and(component, dilated)
    if int(local_core.sum()) < 9:
        local_core = component
    return local_core.astype(np.uint8)


def tighten_low_granularity_teacher(
    probability: np.ndarray,
    target_mask: np.ndarray,
    click_xy: Tuple[int, int],
    display_granularity: float,
) -> np.ndarray:
    display_granularity = float(display_granularity)
    if display_granularity > 0.18:
        return probability.astype(np.float32)

    target_bool = target_mask.astype(bool)
    if not np.any(target_bool):
        return probability.astype(np.float32)

    local = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
    if display_granularity <= 0.10:
        support = build_click_local_core(target_bool, click_xy, display_granularity).astype(bool)
    else:
        dilation_size = max(3, int(round(4 + 6 * local)))
        if dilation_size % 2 == 0:
            dilation_size += 1
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        support = cv2.dilate(target_bool.astype(np.uint8), dilation_kernel, iterations=1).astype(bool)
        support |= build_click_local_core(target_bool, click_xy, display_granularity).astype(bool)

    constrained = np.clip(probability.astype(np.float32), 0.0, 1.0) * support.astype(np.float32)
    target_prior = support.astype(np.float32) * (0.94 - 0.10 * local)
    return np.maximum(constrained, target_prior).astype(np.float32)


def pick_component_center(mask: np.ndarray) -> Tuple[int, int]:
    padded = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), mode="constant")
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return int(x), int(y)


def correct_towards_target(
    predictor,
    image: np.ndarray,
    first_click: clicker.Click,
    target_mask: np.ndarray,
    init_prev_mask: np.ndarray,
    max_clicks: int = DEFAULT_MAX_REFINE_CLICKS,
    pred_thresh: float = 0.49,
):
    clicks = [first_click.copy()]
    corrected_prob = run_simpleclick_prediction(
        predictor,
        image,
        clicks,
        prev_prediction=init_prev_mask.astype(np.float32),
    ).astype(np.float32)

    click_log = []
    target_clicker = clicker.Clicker(gt_mask=target_mask.astype(np.uint8))
    target_clicker.add_click(first_click.copy())

    previous_iou = compute_single_iou(corrected_prob >= pred_thresh, target_mask)
    for _ in range(max_clicks):
        pred_mask = corrected_prob >= pred_thresh
        target_clicker.make_next_click(pred_mask)
        corrective_click = target_clicker.get_clicks()[-1]
        click_log.append([
            int(corrective_click.coords[1]),
            int(corrective_click.coords[0]),
            int(corrective_click.is_positive),
        ])
        clicks.append(corrective_click)

        updated_prob = run_simpleclick_prediction(
            predictor,
            image,
            clicks,
            prev_prediction=init_prev_mask.astype(np.float32),
        ).astype(np.float32)
        updated_mask = updated_prob >= pred_thresh
        current_iou = compute_single_iou(updated_mask, target_mask)
        corrected_prob = updated_prob
        if current_iou >= 0.97 or (current_iou - previous_iou) < 0.005:
            break
        previous_iou = current_iou

    padded_clicks = np.full((max_clicks, 3), -1, dtype=np.int16)
    if click_log:
        padded_clicks[:len(click_log)] = np.asarray(click_log, dtype=np.int16)
    return corrected_prob, padded_clicks


def apply_click_log(
    predictor,
    image: np.ndarray,
    first_click: clicker.Click,
    click_log: List[List[int]],
    init_prev_mask: np.ndarray,
) -> np.ndarray:
    clicks = [first_click.copy()]
    for click_x, click_y, is_positive in click_log:
        clicks.append(clicker.Click(is_positive=bool(is_positive), coords=(int(click_y), int(click_x))))
    return run_simpleclick_prediction(
        predictor,
        image,
        clicks,
        prev_prediction=init_prev_mask.astype(np.float32),
    ).astype(np.float32)


def cleanup_overflow_with_negative_clicks(
    predictor,
    image: np.ndarray,
    first_click: clicker.Click,
    gt_mask: np.ndarray,
    probability: np.ndarray,
    click_log: List[List[int]],
    init_prev_mask: np.ndarray,
    pred_thresh: float,
    target_fp_ratio: float,
    max_cleanup_rounds: int,
    min_fp_drop: float,
    max_recall_drop: float,
) -> Tuple[np.ndarray, List[List[int]]]:
    best_prob = probability.astype(np.float32)
    best_click_log = [item[:] for item in click_log]
    gt_bool = gt_mask.astype(bool)

    for _ in range(max_cleanup_rounds):
        current_mask = best_prob >= pred_thresh
        current_fp_ratio = compute_false_positive_ratio(current_mask, gt_mask)
        if current_fp_ratio <= target_fp_ratio:
            break
        overflow_mask = np.logical_and(current_mask, np.logical_not(gt_bool)).astype(np.uint8)
        if not np.any(overflow_mask):
            break

        neg_click_xy = pick_component_center(overflow_mask)
        trial_click_log = best_click_log + [[int(neg_click_xy[0]), int(neg_click_xy[1]), 0]]
        trial_prob = apply_click_log(
            predictor,
            image,
            first_click,
            trial_click_log,
            init_prev_mask,
        )
        trial_mask = trial_prob >= pred_thresh
        trial_fp_ratio = compute_false_positive_ratio(trial_mask, gt_mask)
        current_recall = compute_gt_recall(current_mask, gt_mask)
        trial_recall = compute_gt_recall(trial_mask, gt_mask)
        if trial_fp_ratio < current_fp_ratio - min_fp_drop and trial_recall >= current_recall - max_recall_drop:
            best_prob = trial_prob
            best_click_log = trial_click_log
            continue
        break

    return best_prob.astype(np.float32), best_click_log


def build_progressive_mask_between(
    anchor_mask: np.ndarray,
    target_mask: np.ndarray,
    progress: float,
) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, 1.0))
    target_bool = target_mask.astype(bool)
    if progress >= 1.0:
        return target_bool.astype(np.uint8)

    anchor = np.logical_and(anchor_mask.astype(bool), target_bool)
    if not np.any(anchor):
        anchor = target_bool.copy()
    if progress <= 0.0 or np.array_equal(anchor, target_bool):
        return anchor.astype(np.uint8)

    missing = np.logical_and(target_bool, np.logical_not(anchor))
    if not np.any(missing):
        return anchor.astype(np.uint8)

    distance = cv2.distanceTransform((np.logical_not(anchor)).astype(np.uint8), cv2.DIST_L2, 5)
    missing_distances = distance[missing]
    threshold = float(np.quantile(missing_distances, progress))
    expanded_missing = np.logical_and(missing, distance <= threshold + 1e-6)
    progressive = np.logical_or(anchor, expanded_missing)
    return progressive.astype(np.uint8)


def build_progressive_gt_mask(
    anchor_mask: np.ndarray,
    gt_mask: np.ndarray,
    progress: float,
) -> np.ndarray:
    return build_progressive_mask_between(anchor_mask, gt_mask, progress)


def build_progressive_training_teacher(
    base_prob: np.ndarray,
    recovery_prob: np.ndarray,
    gt_mask: np.ndarray,
    click_xy: Tuple[int, int],
    display_granularity: float,
    semantic_pivot_prob: np.ndarray,
    semantic_pivot_mask: np.ndarray,
    pred_thresh: float = 0.49,
) -> np.ndarray:
    display_granularity = float(display_granularity)
    base_mask = base_prob >= pred_thresh
    base_target = np.logical_and(base_mask, gt_mask > 0).astype(np.uint8)
    if not np.any(base_target):
        base_target = base_mask.astype(np.uint8)

    pivot_mask = semantic_pivot_mask.astype(np.uint8)
    if not np.any(pivot_mask):
        pivot_mask = base_target.copy()
    pivot_prob = semantic_pivot_prob.astype(np.float32)

    local_core = build_click_local_core(
        pivot_mask if np.any(pivot_mask) else (base_target if np.any(base_target) else gt_mask.astype(np.uint8)),
        click_xy,
        min(display_granularity, 0.10),
    )

    if display_granularity <= 0.5:
        local = float(np.clip(display_granularity / 0.5, 0.0, 1.0))
        progress = local ** 0.72
        progressive_mask = build_progressive_mask_between(local_core, pivot_mask, progress)
        progressive_bool = progressive_mask.astype(bool)
        teacher_source = np.maximum(np.clip(base_prob.astype(np.float32), 0.0, 1.0), np.clip(pivot_prob, 0.0, 1.0))
        teacher = teacher_source * progressive_bool.astype(np.float32)
        target_floor = progressive_bool.astype(np.float32) * (0.88 + 0.10 * progress)
        return np.maximum(teacher, target_floor).astype(np.float32)

    if display_granularity >= 0.95:
        return (recovery_prob * gt_mask.astype(np.float32)).astype(np.float32)

    local = float(np.clip((display_granularity - 0.5) / 0.45, 0.0, 1.0))
    progress = local ** 1.15
    progressive_mask = build_progressive_gt_mask(pivot_mask, gt_mask, progress)
    progressive_bool = progressive_mask.astype(bool)

    teacher = np.maximum(np.maximum(base_prob.astype(np.float32), recovery_prob.astype(np.float32)), pivot_prob)
    teacher = teacher * progressive_bool.astype(np.float32)
    target_floor = progressive_bool.astype(np.float32) * (0.82 + 0.10 * progress)
    teacher = np.maximum(teacher, target_floor)
    return (teacher * gt_mask.astype(np.float32)).astype(np.float32)


def build_semantic_completion_candidates(
    base_mask: np.ndarray,
    gt_mask: np.ndarray,
    global_level_masks: np.ndarray,
    effective_level_indices: np.ndarray,
    current_level_index: int,
    min_candidate_precision: float = 0.60,
    max_future_levels: int = 2,
) -> List[Tuple[float, np.ndarray, Tuple[int, int]]]:
    base_mask = base_mask.astype(bool)
    missing_gt = np.logical_and(gt_mask.astype(bool), np.logical_not(base_mask))
    base_dilated = cv2.dilate(base_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    candidates = []
    seen_signatures = set()

    sorted_effective = [int(x) for x in effective_level_indices.tolist() if int(x) > current_level_index]
    if max_future_levels > 0:
        sorted_effective = sorted_effective[:max_future_levels]
    for order, level_index in enumerate(sorted_effective, start=1):
        semantic_mask = global_level_masks[level_index].astype(bool)
        delta_mask = np.logical_and(semantic_mask, np.logical_not(base_mask)).astype(np.uint8)
        num_components, labels = cv2.connectedComponents(delta_mask)
        for component_id in range(1, num_components):
            component = labels == component_id
            if not np.any(component):
                continue
            if not np.any(np.logical_and(component, base_dilated)):
                continue
            gt_overlap = np.logical_and(component, missing_gt).sum()
            if gt_overlap <= 0:
                continue
            signature = component.tobytes()
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            component_uint8 = component.astype(np.uint8)
            click_xy = pick_component_center(component_uint8)
            touch_count = float(
                np.logical_and(
                    cv2.dilate(component_uint8, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool),
                    base_mask,
                ).sum()
            )
            area = float(component_uint8.sum())
            gt_precision = float(gt_overlap / max(area, 1.0))
            gt_recall = float(gt_overlap / max(missing_gt.sum(), 1.0))
            if gt_precision < min_candidate_precision:
                continue
            score = (
                1.5 * gt_precision
                + 1.5 * gt_recall
                + (1.0 / order)
                + 0.002 * min(touch_count, 200.0)
                - 0.00001 * area
            )
            candidates.append((score, component_uint8, click_xy))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def build_completion_teacher_from_candidates(
    predictor,
    image: np.ndarray,
    first_click: clicker.Click,
    gt_mask: np.ndarray,
    base_prob: np.ndarray,
    init_prev_mask: np.ndarray,
    candidate_components: List[Tuple[float, np.ndarray, Tuple[int, int]]],
    max_completion_clicks: int = DEFAULT_MAX_COMPLETION_CLICKS,
    max_candidates: int = 4,
    min_iou_gain: float = 0.003,
    max_fp_ratio: float = 0.03,
    min_added_precision: float = 0.70,
    max_gt_clicks: Optional[int] = 2,
    target_recall: float = 0.90,
    overflow_trigger_ratio: float = 0.005,
    guided_until_target: bool = False,
    pred_thresh: float = 0.49,
) -> Tuple[np.ndarray, np.ndarray]:
    best_prob = base_prob.astype(np.float32).copy()
    base_mask = init_prev_mask.astype(np.float32) >= pred_thresh
    gt_bool = gt_mask.astype(bool)
    best_iou = compute_single_iou(best_prob >= pred_thresh, gt_mask)
    best_fp_ratio = compute_false_positive_ratio(best_prob >= pred_thresh, gt_mask)
    best_click_log: List[List[int]] = []
    ranked_candidates = candidate_components[:max_candidates]

    num_prefix = min(len(ranked_candidates), max_completion_clicks)
    for prefix_len in range(1, num_prefix + 1):
        prefix = ranked_candidates[:prefix_len]
        trial_click_log: List[List[int]] = []
        trial_pseudo_clicks: List[clicker.Click] = []
        for _, _, click_xy in prefix:
            trial_click_log.append([int(click_xy[0]), int(click_xy[1]), 1])
            trial_pseudo_clicks.append(clicker.Click(is_positive=True, coords=(click_xy[1], click_xy[0])))

        trial_clicks = [first_click.copy()] + [item.copy() for item in trial_pseudo_clicks]
        trial_prob = run_simpleclick_prediction(
            predictor,
            image,
            trial_clicks,
            prev_prediction=init_prev_mask.astype(np.float32),
        ).astype(np.float32)
        trial_mask = trial_prob >= pred_thresh
        trial_iou = compute_single_iou(trial_mask, gt_mask)
        trial_fp_ratio = compute_false_positive_ratio(trial_mask, gt_mask)
        added_mask = np.logical_and(trial_mask, np.logical_not(base_mask))
        added_area = float(added_mask.sum())
        added_precision = float(np.logical_and(added_mask, gt_bool).sum() / max(added_area, 1.0))

        if (
            trial_iou > best_iou + min_iou_gain
            and trial_fp_ratio <= max_fp_ratio
            and trial_fp_ratio <= best_fp_ratio + 0.01
            and added_precision >= min_added_precision
        ):
            best_prob = trial_prob
            best_iou = trial_iou
            best_fp_ratio = trial_fp_ratio
            best_click_log = trial_click_log

    best_mask = best_prob >= pred_thresh
    fp_mask = np.logical_and(best_mask, np.logical_not(gt_bool)).astype(np.uint8)
    if np.any(fp_mask):
        neg_click_xy = pick_component_center(fp_mask)
        negative_click = clicker.Click(is_positive=False, coords=(neg_click_xy[1], neg_click_xy[0]))
        cleanup_clicks = [first_click.copy()]
        for click_x, click_y, is_positive in best_click_log:
            del is_positive
            cleanup_clicks.append(clicker.Click(is_positive=True, coords=(click_y, click_x)))
        cleanup_clicks.append(negative_click)
        cleanup_prob = run_simpleclick_prediction(
            predictor,
            image,
            cleanup_clicks,
            prev_prediction=init_prev_mask.astype(np.float32),
        ).astype(np.float32)
        cleanup_mask = cleanup_prob >= pred_thresh
        cleanup_iou = compute_single_iou(cleanup_mask, gt_mask)
        cleanup_fp_ratio = compute_false_positive_ratio(cleanup_mask, gt_mask)
        if cleanup_fp_ratio < best_fp_ratio and cleanup_iou >= best_iou - 0.002:
            best_prob = cleanup_prob
            best_iou = cleanup_iou
            best_fp_ratio = cleanup_fp_ratio
            if len(best_click_log) < max_completion_clicks:
                best_click_log.append([int(neg_click_xy[0]), int(neg_click_xy[1]), 0])

    guided_rounds = 0
    repeated_guided_failures = 0
    attempted_guided_actions = set()
    while True:
        current_mask = best_prob >= pred_thresh
        current_recall = compute_gt_recall(current_mask, gt_mask)
        current_fp_ratio = compute_false_positive_ratio(current_mask, gt_mask)
        missing_mask = np.logical_and(gt_bool, np.logical_not(current_mask)).astype(np.uint8)
        overflow_mask = np.logical_and(current_mask, np.logical_not(gt_bool)).astype(np.uint8)

        if current_recall >= target_recall and current_fp_ratio <= overflow_trigger_ratio:
            break
        if not np.any(missing_mask) and not np.any(overflow_mask):
            break
        if not guided_until_target and max_gt_clicks is not None and guided_rounds >= max_gt_clicks:
            break

        if np.any(overflow_mask) and current_fp_ratio > overflow_trigger_ratio:
            overflow_click_xy = pick_component_center(overflow_mask)
            guided_click = clicker.Click(
                is_positive=False,
                coords=(overflow_click_xy[1], overflow_click_xy[0]),
            )
        elif np.any(missing_mask):
            missing_click_xy = pick_component_center(missing_mask)
            guided_click = clicker.Click(
                is_positive=True,
                coords=(missing_click_xy[1], missing_click_xy[0]),
            )
        elif np.any(overflow_mask):
            overflow_click_xy = pick_component_center(overflow_mask)
            guided_click = clicker.Click(
                is_positive=False,
                coords=(overflow_click_xy[1], overflow_click_xy[0]),
            )
        else:
            break

        guided_action = (
            int(guided_click.coords[1]),
            int(guided_click.coords[0]),
            int(guided_click.is_positive),
        )
        if guided_action in attempted_guided_actions:
            break
        attempted_guided_actions.add(guided_action)

        current_clicks = [first_click.copy()]
        for click_x, click_y, is_positive in best_click_log:
            current_clicks.append(clicker.Click(is_positive=bool(is_positive), coords=(int(click_y), int(click_x))))

        guided_trial_clicks = current_clicks + [guided_click]
        guided_prob = run_simpleclick_prediction(
            predictor,
            image,
            guided_trial_clicks,
            prev_prediction=init_prev_mask.astype(np.float32),
        ).astype(np.float32)
        guided_mask = guided_prob >= pred_thresh
        guided_iou = compute_single_iou(guided_mask, gt_mask)
        guided_fp_ratio = compute_false_positive_ratio(guided_mask, gt_mask)
        guided_recall = compute_gt_recall(guided_mask, gt_mask)

        accept_guided = False
        if guided_click.is_positive:
            added_mask = np.logical_and(guided_mask, np.logical_not(current_mask))
            added_area = float(added_mask.sum())
            added_precision = float(np.logical_and(added_mask, gt_bool).sum() / max(added_area, 1.0))
            if guided_until_target:
                accept_guided = (
                    guided_recall > current_recall + 0.001
                    and guided_fp_ratio <= max(max_fp_ratio + 0.02, current_fp_ratio + 0.02)
                    and added_precision >= max(min_added_precision - 0.20, 0.35)
                )
            else:
                accept_guided = (
                    guided_recall > current_recall + 0.005
                    and guided_fp_ratio <= max_fp_ratio + 0.005
                    and added_precision >= max(min_added_precision - 0.10, 0.50)
                )
        else:
            if guided_until_target:
                accept_guided = (
                    guided_fp_ratio < current_fp_ratio - 0.0005
                    or (guided_fp_ratio <= current_fp_ratio and guided_iou >= best_iou - 0.001)
                )
            else:
                accept_guided = guided_fp_ratio < current_fp_ratio and guided_iou >= best_iou - 0.003

        if not accept_guided:
            repeated_guided_failures += 1
            if repeated_guided_failures >= 2:
                break
            continue

        best_prob = guided_prob
        best_iou = guided_iou
        best_fp_ratio = guided_fp_ratio
        best_click_log.append([
            int(guided_click.coords[1]),
            int(guided_click.coords[0]),
            int(guided_click.is_positive),
        ])
        guided_rounds += 1
        repeated_guided_failures = 0

    strict_fp_ratio = max_fp_ratio if guided_until_target else min(max_fp_ratio, 0.012)
    best_prob, best_click_log = cleanup_overflow_with_negative_clicks(
        predictor,
        image,
        first_click,
        gt_mask,
        best_prob,
        best_click_log,
        init_prev_mask,
        pred_thresh=pred_thresh,
        target_fp_ratio=strict_fp_ratio,
        max_cleanup_rounds=3 if not guided_until_target else 2,
        min_fp_drop=0.001 if not guided_until_target else 0.0005,
        max_recall_drop=0.01 if not guided_until_target else 0.02,
    )

    if not best_click_log:
        return best_prob, np.full((0, 3), -1, dtype=np.int16)
    return best_prob, np.asarray(best_click_log, dtype=np.int16)


def compute_effective_levels(level_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat_signatures = [level_masks[index].tobytes() for index in range(level_masks.shape[0])]
    unique_signature_order = []
    signature_to_level = {}
    effective_level_mask = np.zeros(level_masks.shape[0], dtype=np.uint8)
    for level_index, signature in enumerate(flat_signatures):
        if signature not in signature_to_level:
            signature_to_level[signature] = len(unique_signature_order)
            unique_signature_order.append(level_index)
            effective_level_mask[level_index] = 1
    return effective_level_mask, np.asarray(unique_signature_order, dtype=np.int16)


def build_completion_target_indices(level_masks: np.ndarray, effective_level_indices: np.ndarray) -> np.ndarray:
    next_indices = np.full((level_masks.shape[0],), -1, dtype=np.int16)
    effective_set = set(int(x) for x in effective_level_indices.tolist())
    for level_index in range(level_masks.shape[0]):
        for candidate_index in range(level_index + 1, level_masks.shape[0]):
            if candidate_index in effective_set:
                next_indices[level_index] = np.int16(candidate_index)
                break
    return next_indices


def save_preview(
    output_path: Path,
    image: np.ndarray,
    gt_mask: np.ndarray,
    simple_prob: np.ndarray,
    global_level_masks: np.ndarray,
    base_teacher_probs: np.ndarray,
    recovery_teacher_probs: np.ndarray,
    training_teacher_probs: np.ndarray,
):
    rows = []
    num_levels = int(global_level_masks.shape[0])
    mid_level_index = min(num_levels - 1, max(0, num_levels // 2))
    last_level_index = num_levels - 1
    first_row = [
        image,
        overlay_mask(image, gt_mask, (0, 255, 0)),
        overlay_mask(image, simple_prob >= 0.49, (255, 255, 0)),
        overlay_mask(image, base_teacher_probs[0] >= 0.49, (255, 128, 0)),
        overlay_mask(image, training_teacher_probs[mid_level_index] >= 0.49, (64, 160, 255)),
        overlay_mask(image, recovery_teacher_probs[last_level_index] >= 0.49, (200, 64, 200)),
        overlay_delta(image, simple_prob >= 0.49, recovery_teacher_probs[last_level_index] >= 0.49),
    ]
    rows.append(np.concatenate(first_row, axis=1))

    for level_index in range(num_levels):
        level_mask = global_level_masks[level_index]
        base_prob = base_teacher_probs[level_index]
        recovery_prob = recovery_teacher_probs[level_index]
        training_prob = training_teacher_probs[level_index]
        level_row = [
            overlay_mask(image, level_mask > 0, (255, 128, 0)),
            overlay_mask(image, base_prob >= 0.49, (255, 196, 0)),
            overlay_delta(image, level_mask, base_prob >= 0.49),
            overlay_mask(image, training_prob >= 0.49, (64, 160, 255)),
            overlay_delta(image, base_prob >= 0.49, training_prob >= 0.49),
            overlay_mask(image, recovery_prob >= 0.49, (200, 64, 200)),
            overlay_delta(image, training_prob >= 0.49, gt_mask > 0),
        ]
        rows.append(np.concatenate(level_row, axis=1))

    max_w = max(row.shape[1] for row in rows)
    normalized_rows = []
    for row in rows:
        if row.shape[1] != max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=row.dtype)
            row = np.concatenate([row, pad], axis=1)
        normalized_rows.append(row)
    canvas = np.concatenate(normalized_rows, axis=0)
    imwrite_unicode(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def generate_record(
    predictor,
    refiner: DinoGranularityRefiner,
    image: np.ndarray,
    gt_mask: np.ndarray,
    click_xy: Tuple[int, int],
    display_granularities: Sequence[float],
    max_refine_clicks: int,
    max_completion_clicks: int,
    max_completion_candidates: int,
    max_completion_future_levels: int,
    completion_min_iou_gain: float,
    completion_max_fp_ratio: float,
    completion_min_candidate_precision: float,
    completion_min_added_precision: float,
    max_completion_gt_clicks: int,
    completion_target_recall: float,
    completion_overflow_trigger_ratio: float,
    max_level_extra_gt_clicks: int,
    max_level_target_recall: float,
) -> Dict:
    height, width = gt_mask.shape
    positive_click = clicker.Click(is_positive=True, coords=(click_xy[1], click_xy[0]))
    simple_prob = run_simpleclick_prediction(predictor, image, [positive_click]).astype(np.float32)
    display_granularities_array = np.asarray(display_granularities, dtype=np.float32)
    hierarchy_granularities = map_display_granularities_for_hierarchy(display_granularities_array)
    model_granularities = map_display_granularities_for_model(display_granularities_array)

    global_level_masks = build_runtime_slider_masks(
        refiner=refiner,
        image=image,
        simple_prob=simple_prob,
        click_xy=click_xy,
        granularities=hierarchy_granularities.tolist(),
    )
    global_level_masks = np.stack(global_level_masks, axis=0).astype(np.uint8)
    global_level_ious = compute_ious(global_level_masks, gt_mask)
    best_global_level_index = int(np.argmax(global_level_ious))
    effective_level_mask, effective_level_indices = compute_effective_levels(global_level_masks)
    completion_target_indices = build_completion_target_indices(global_level_masks, effective_level_indices)

    num_levels, height, width = global_level_masks.shape
    base_teacher_probs = np.zeros((num_levels, height, width), dtype=np.float32)
    base_teacher_masks = np.zeros((num_levels, height, width), dtype=np.uint8)
    base_corrective_clicks = np.full((num_levels, max_refine_clicks, 3), -1, dtype=np.int16)
    recovery_teacher_probs = np.zeros((num_levels, height, width), dtype=np.float32)
    recovery_teacher_masks = np.zeros((num_levels, height, width), dtype=np.uint8)
    completion_click_logs: List[np.ndarray] = [np.full((0, 3), -1, dtype=np.int16) for _ in range(num_levels)]

    signature_to_result: Dict[bytes, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    effective_rank_map = {int(level_index): rank for rank, level_index in enumerate(effective_level_indices.tolist())}
    total_effective = max(len(effective_level_indices), 1)
    for level_index in range(num_levels):
        level_mask = global_level_masks[level_index]
        display_granularity = float(display_granularities_array[level_index])
        signature = level_mask.tobytes()
        if signature in signature_to_result:
            cached = signature_to_result[signature]
            base_teacher_probs[level_index] = cached[0]
            base_teacher_masks[level_index] = cached[1]
            base_corrective_clicks[level_index] = cached[2]
            recovery_teacher_probs[level_index] = cached[3]
            recovery_teacher_masks[level_index] = cached[4]
            cached_completion_clicks = cached[5]
            completion_click_logs[level_index] = cached_completion_clicks.copy()
            continue

        base_prob, base_clicks = correct_towards_target(
            predictor,
            image,
            positive_click,
            target_mask=level_mask,
            init_prev_mask=level_mask,
            max_clicks=max_refine_clicks,
        )
        base_prob = tighten_low_granularity_teacher(
            base_prob,
            level_mask,
            click_xy,
            display_granularity,
        )
        base_mask = (base_prob >= 0.49).astype(np.uint8)
        base_recall = compute_gt_recall(base_mask, gt_mask)
        level_rank = effective_rank_map.get(level_index, total_effective - 1)
        progress = float(level_rank / max(total_effective - 1, 1))
        level_target_recall = base_recall + progress * max(completion_target_recall - base_recall, 0.0)
        level_target_recall = float(min(max(level_target_recall, base_recall), 0.999))
        level_gt_clicks = max_completion_gt_clicks
        if level_rank >= total_effective - 1:
            level_target_recall = max(level_target_recall, max_level_target_recall)
            level_gt_clicks = max(level_gt_clicks, max_completion_gt_clicks + max_level_extra_gt_clicks)

        completion_candidates = build_semantic_completion_candidates(
            base_mask=base_mask,
            gt_mask=gt_mask,
            global_level_masks=global_level_masks,
            effective_level_indices=effective_level_indices,
            current_level_index=level_index,
            min_candidate_precision=completion_min_candidate_precision,
            max_future_levels=max_completion_future_levels,
        )
        recovery_prob, recovery_clicks = build_completion_teacher_from_candidates(
            predictor,
            image,
            positive_click,
            gt_mask=gt_mask,
            base_prob=base_prob,
            init_prev_mask=base_prob,
            candidate_components=completion_candidates,
            max_completion_clicks=max_completion_clicks,
            max_candidates=max_completion_candidates,
            min_iou_gain=completion_min_iou_gain,
            max_fp_ratio=completion_max_fp_ratio,
            min_added_precision=completion_min_added_precision,
            max_gt_clicks=level_gt_clicks,
            target_recall=level_target_recall,
            overflow_trigger_ratio=completion_overflow_trigger_ratio,
            guided_until_target=(level_rank >= total_effective - 1),
            pred_thresh=0.49,
        )
        recovery_mask = (recovery_prob >= 0.49).astype(np.uint8)
        if level_rank >= total_effective - 1:
            recovery_mask = np.logical_and(recovery_mask > 0, gt_mask > 0).astype(np.uint8)
            max_level_recall = compute_gt_recall(recovery_mask, gt_mask)
            if max_level_recall < max_level_target_recall:
                missing_gt = np.logical_and(gt_mask > 0, recovery_mask <= 0).astype(np.float32)
                recovery_prob = np.maximum(recovery_prob, missing_gt * 0.90).astype(np.float32)
                recovery_mask = (recovery_prob >= 0.49).astype(np.uint8)
                recovery_mask = np.logical_and(recovery_mask > 0, gt_mask > 0).astype(np.uint8)
            recovery_prob = recovery_prob * gt_mask.astype(np.float32)

        base_teacher_probs[level_index] = base_prob.astype(np.float32)
        base_teacher_masks[level_index] = base_mask
        base_corrective_clicks[level_index] = base_clicks
        recovery_teacher_probs[level_index] = recovery_prob.astype(np.float32)
        recovery_teacher_masks[level_index] = recovery_mask
        completion_click_logs[level_index] = recovery_clicks.copy()
        signature_to_result[signature] = (
            base_prob.astype(np.float32),
            base_mask,
            base_clicks,
            recovery_prob.astype(np.float32),
            recovery_mask,
            recovery_clicks,
        )

    completion_log_len = max(1, max(clicks.shape[0] for clicks in completion_click_logs))
    recovery_corrective_clicks = np.full((num_levels, completion_log_len, 3), -1, dtype=np.int16)
    for level_index, recovery_clicks in enumerate(completion_click_logs):
        if recovery_clicks.size == 0:
            continue
        recovery_corrective_clicks[level_index, :recovery_clicks.shape[0]] = recovery_clicks

    pivot_level_index = int(np.argmin(np.abs(display_granularities_array - 0.5)))
    semantic_pivot_prob = base_teacher_probs[pivot_level_index].astype(np.float32)
    semantic_pivot_mask = (semantic_pivot_prob >= 0.49).astype(np.uint8)
    if not np.any(semantic_pivot_mask):
        semantic_pivot_mask = (base_teacher_probs[min(pivot_level_index, num_levels - 1)] >= 0.49).astype(np.uint8)

    training_teacher_probs = np.zeros_like(base_teacher_probs, dtype=np.float32)
    for level_index, display_granularity in enumerate(display_granularities_array.tolist()):
        training_teacher_probs[level_index] = build_progressive_training_teacher(
            base_teacher_probs[level_index],
            recovery_teacher_probs[level_index],
            gt_mask,
            click_xy,
            float(display_granularity),
            semantic_pivot_prob,
            semantic_pivot_mask,
            pred_thresh=0.49,
        )
    training_teacher_masks = (training_teacher_probs >= 0.49).astype(np.uint8)

    return {
        "gt_mask": gt_mask.astype(np.uint8),
        "simple_prob": simple_prob.astype(np.float16),
        "global_level_masks": global_level_masks.astype(np.uint8),
        "global_level_ious_to_gt": global_level_ious.astype(np.float32),
        "best_global_level_index": np.int16(best_global_level_index),
        "base_teacher_probs": base_teacher_probs.astype(np.float16),
        "base_teacher_masks": base_teacher_masks.astype(np.uint8),
        "base_corrective_clicks": base_corrective_clicks.astype(np.int16),
        "recovery_teacher_probs": recovery_teacher_probs.astype(np.float16),
        "recovery_teacher_masks": recovery_teacher_masks.astype(np.uint8),
        "recovery_corrective_clicks": recovery_corrective_clicks.astype(np.int16),
        "training_teacher_probs": training_teacher_probs.astype(np.float16),
        "training_teacher_masks": training_teacher_masks.astype(np.uint8),
        "completion_target_indices": completion_target_indices.astype(np.int16),
        "positive_click_map": click_to_map(height, width, click_xy),
        "positive_click": np.asarray(click_xy, dtype=np.int16),
        "effective_level_mask": effective_level_mask.astype(np.uint8),
        "effective_level_indices": effective_level_indices.astype(np.int16),
        "num_effective_levels": np.int16(len(effective_level_indices)),
        "granularities": display_granularities_array,
        "model_granularities": model_granularities.astype(np.float32),
        "hierarchy_granularities": hierarchy_granularities.astype(np.float32),
    }


def print_progress(split_name: str, processed_images: int, total_images: int,
                   built_records: int, skipped_records: int, missing_images: int):
    print(
        f"[{split_name}] images={processed_images}/{total_images} "
        f"built={built_records} skipped={skipped_records} missing={missing_images}",
        flush=True,
    )


def main():
    args = parse_args()
    dataset_root = resolve_sbd_dataset_root(Path(args.sbd_root).expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dirs(output_root)

    official_train_ids = read_split_ids(dataset_root / "train.txt")
    official_test_ids = read_split_ids(dataset_root / "val.txt")
    train_ids, val_ids = deterministic_train_val_split(official_train_ids, args.val_ratio)
    split_to_ids = {"train": train_ids, "val": val_ids, "test": official_test_ids}
    write_split_files(output_root, split_to_ids)

    device = torch.device(args.device)
    model = utils.load_is_model(args.checkpoint, device, eval_ritm=False, cpu_dist_maps=True)
    predictor = get_predictor(
        model,
        brs_mode="NoBRS",
        device=device,
        with_flip=True,
        zoom_in_params={
            "skip_clicks": -1,
            "target_size": (448, 448),
            "expansion_ratio": 1.4,
        },
        predictor_params={
            "optimize_after_n_clicks": 1,
            "max_size": 800,
        },
    )
    refiner = DinoGranularityRefiner(
        repo_path=args.dinov3_repo,
        weights_path=args.dinov3_weights,
        device=device,
        model_name=args.dinov3_model,
        image_size=args.image_size,
        work_size=args.work_size,
        n_segments=args.segments,
        candidate_threshold=args.candidate_threshold,
        max_hole_ratio=args.max_hole_ratio,
    )

    granularities = [float(x) for x in args.granularities]
    summary = {
        "sbd_root": str(dataset_root),
        "checkpoint": args.checkpoint,
        "dinov3_repo": args.dinov3_repo,
        "dinov3_model": args.dinov3_model,
        "dinov3_weights": args.dinov3_weights,
        "granularities": granularities,
        "max_refine_clicks": args.max_refine_clicks,
        "max_completion_clicks": args.max_completion_clicks,
        "max_completion_candidates": args.max_completion_candidates,
        "max_completion_future_levels": args.max_completion_future_levels,
        "completion_min_iou_gain": args.completion_min_iou_gain,
        "completion_max_fp_ratio": args.completion_max_fp_ratio,
        "completion_min_candidate_precision": args.completion_min_candidate_precision,
        "completion_min_added_precision": args.completion_min_added_precision,
        "max_completion_gt_clicks": args.max_completion_gt_clicks,
        "completion_target_recall": args.completion_target_recall,
        "completion_overflow_trigger_ratio": args.completion_overflow_trigger_ratio,
        "max_level_extra_gt_clicks": args.max_level_extra_gt_clicks,
        "max_level_target_recall": args.max_level_target_recall,
        "val_ratio_from_official_train": args.val_ratio,
        "splits": {},
    }

    for split_name, image_ids in split_to_ids.items():
        split_jsonl = output_root / "metadata" / f"{split_name}.jsonl"
        if args.overwrite and split_jsonl.exists():
            split_jsonl.unlink()
        existing_metadata = load_existing_metadata(split_jsonl) if args.resume and not args.overwrite else {}

        split_records = 0
        split_skipped = 0
        preview_budget = args.preview_per_split
        processed_images = 0
        missing_images = 0
        total_images = len(image_ids) if args.max_images_per_split <= 0 else min(len(image_ids), args.max_images_per_split)

        print(f"[{split_name}] start total_images={total_images}", flush=True)

        for image_id in image_ids:
            if args.max_images_per_split > 0 and processed_images >= args.max_images_per_split:
                break
            try:
                image, instances_mask = load_sbd_image_and_instances(dataset_root, image_id)
            except FileNotFoundError:
                missing_images += 1
                continue
            processed_images += 1
            refiner.prepare_image(image)

            for instance_id in instance_ids_from_mask(instances_mask):
                gt_mask = (instances_mask == instance_id).astype(np.uint8)
                if int(gt_mask.sum()) < args.min_object_area:
                    continue

                click_variants = sample_first_positive_clicks(gt_mask, max_variants=args.first_click_variants)
                for click_variant_index, (click_xy, click_role) in enumerate(click_variants):
                    sample_name = f"{image_id}_inst{instance_id:03d}_click{click_variant_index:02d}_{click_role}"
                    npz_path = output_root / "records" / split_name / f"{sample_name}.npz"
                    if args.resume and not args.overwrite and npz_path.exists():
                        split_skipped += 1
                        if sample_name not in existing_metadata:
                            meta = {
                                "sample_name": sample_name,
                                "split": split_name,
                                "image_id": image_id,
                                "instance_id": instance_id,
                                "click_variant_index": int(click_variant_index),
                                "click_variant_role": click_role,
                                "image_path": str(dataset_root / "img" / f"{image_id}.jpg"),
                                "npz_path": str(npz_path),
                                "height": int(image.shape[0]),
                                "width": int(image.shape[1]),
                                "positive_click_xy": [int(click_xy[0]), int(click_xy[1])],
                                "granularities": granularities,
                            }
                            append_jsonl(split_jsonl, meta)
                            existing_metadata[sample_name] = meta
                        if preview_budget > 0:
                            preview_path = output_root / "granularity_recovery_previews" / split_name / f"{sample_name}_granularity_recovery.jpg"
                            if not preview_path.exists():
                                with np.load(npz_path) as record_tensors:
                                    base_teacher_probs, recovery_teacher_probs, training_teacher_probs = resolve_teacher_prob_triplet(record_tensors)
                                    save_preview(
                                        output_path=preview_path,
                                        image=image,
                                        gt_mask=record_tensors["gt_mask"],
                                        simple_prob=record_tensors["simple_prob"].astype(np.float32),
                                        global_level_masks=record_tensors["global_level_masks"],
                                        base_teacher_probs=base_teacher_probs,
                                        recovery_teacher_probs=recovery_teacher_probs,
                                        training_teacher_probs=training_teacher_probs,
                                    )
                            preview_budget -= 1
                        continue

                    record_tensors = generate_record(
                        predictor=predictor,
                        refiner=refiner,
                        image=image,
                        gt_mask=gt_mask,
                        click_xy=click_xy,
                        display_granularities=granularities,
                        max_refine_clicks=args.max_refine_clicks,
                        max_completion_clicks=args.max_completion_clicks,
                        max_completion_candidates=args.max_completion_candidates,
                        max_completion_future_levels=args.max_completion_future_levels,
                        completion_min_iou_gain=args.completion_min_iou_gain,
                        completion_max_fp_ratio=args.completion_max_fp_ratio,
                        completion_min_candidate_precision=args.completion_min_candidate_precision,
                        completion_min_added_precision=args.completion_min_added_precision,
                        max_completion_gt_clicks=args.max_completion_gt_clicks,
                        completion_target_recall=args.completion_target_recall,
                        completion_overflow_trigger_ratio=args.completion_overflow_trigger_ratio,
                        max_level_extra_gt_clicks=args.max_level_extra_gt_clicks,
                        max_level_target_recall=args.max_level_target_recall,
                    )
                    np.savez_compressed(npz_path, **record_tensors)

                    meta = {
                        "sample_name": sample_name,
                        "split": split_name,
                        "image_id": image_id,
                        "instance_id": instance_id,
                        "click_variant_index": int(click_variant_index),
                        "click_variant_role": click_role,
                        "image_path": str(dataset_root / "img" / f"{image_id}.jpg"),
                        "npz_path": str(npz_path),
                        "height": int(image.shape[0]),
                        "width": int(image.shape[1]),
                        "positive_click_xy": [int(click_xy[0]), int(click_xy[1])],
                        "granularities": granularities,
                        "best_global_level_index": int(record_tensors["best_global_level_index"]),
                        "num_effective_levels": int(record_tensors["num_effective_levels"]),
                        "effective_level_indices": record_tensors["effective_level_indices"].tolist(),
                    }
                    append_jsonl(split_jsonl, meta)
                    existing_metadata[sample_name] = meta
                    split_records += 1

                    if preview_budget > 0:
                        preview_path = output_root / "granularity_recovery_previews" / split_name / f"{sample_name}_granularity_recovery.jpg"
                        save_preview(
                            output_path=preview_path,
                            image=image,
                            gt_mask=gt_mask,
                            simple_prob=record_tensors["simple_prob"].astype(np.float32),
                            global_level_masks=record_tensors["global_level_masks"],
                            base_teacher_probs=record_tensors["base_teacher_probs"].astype(np.float32),
                            recovery_teacher_probs=record_tensors["recovery_teacher_probs"].astype(np.float32),
                            training_teacher_probs=record_tensors["training_teacher_probs"].astype(np.float32),
                        )
                        preview_budget -= 1

            if processed_images % max(args.log_every_images, 1) == 0:
                print_progress(split_name, processed_images, total_images, split_records, split_skipped, missing_images)

        summary["splits"][split_name] = {
            "num_images": processed_images,
            "num_records": split_records,
            "num_skipped_records": split_skipped,
            "num_missing_images": missing_images,
            "metadata": str(split_jsonl),
        }
        print_progress(split_name, processed_images, total_images, split_records, split_skipped, missing_images)

    with open(output_root / "metadata" / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join([
            "# Granularity Recovery Dataset",
            "",
            "Generated from the current SimpleClick + DINOv3 granularity-recovery pipeline.",
            "",
            "Each `.npz` sample contains:",
            "- `gt_mask`: binary SBD instance mask",
            "- `simple_prob`: first-click SimpleClick probability map",
            "- `global_level_masks`: actual runtime DINOv3 slider masks at fixed granularities",
            "- `global_level_ious_to_gt`: IoU of each level against the instance GT",
            "- `best_global_level_index`: best GT-overlap level for analysis and preview",
            "- `base_teacher_probs`: per-level base teachers that preserve the original low-granularity interaction style",
            "- `base_teacher_masks`: binarized base teachers",
            "- `base_corrective_clicks`: corrective clicks used to refine raw DINO slider masks into SimpleClick-style local results",
            "- `recovery_teacher_probs`: high-granularity recovery teachers that pull the mask toward the full GT while remaining SimpleClick-like",
            "- `recovery_teacher_masks`: binarized recovery teachers",
            "- `recovery_corrective_clicks`: accepted semantic pseudo clicks and GT-guided corrective clicks used for whole-object recovery",
            "- `training_teacher_probs`: actual supervision used for each level, with `50%` as the semantic pivot and continuous expansion on both sides of that pivot",
            "- `training_teacher_masks`: binarized training teachers",
            "- `completion_target_indices`: next larger effective level index for analysis only",
            "- `positive_click_map`: Gaussian map of the first click",
            "- `positive_click`: first click `[x, y]`",
            "- `effective_level_mask`: which level slots are genuinely distinct",
            "- `effective_level_indices`: distinct level indices",
            "- `num_effective_levels`: number of distinct levels in this sample",
            "- `granularities`: slider values used for this sample",
            "- `model_granularities`: remapped granularity values fed into the recovery network",
            "- `hierarchy_granularities`: remapped granularity values used to query the DINOv3 hierarchy",
            "",
            "Preview layout:",
            "- Each instance may appear multiple times with different first-click variants such as center, interior-offset, and extreme-interior clicks.",
            "- `50%` is treated as the semantic pivot level: `0%->50%` expands from a tiny click-local core to the clicked semantic unit, and `50%->100%` expands from that semantic unit toward the full GT.",
            "- First row: image, GT, first-click SimpleClick, base teacher at 0%, mid-level training teacher, final recovery teacher, and SimpleClick-to-recovery delta",
            "- Each following row: DINO level mask, base teacher, level-to-base delta, training teacher, base-to-training delta, recovery teacher, and training-to-GT gap",
        ]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
