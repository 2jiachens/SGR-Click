import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


DEFAULT_GRANULARITIES = (0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0)
MAX_CORRECTIVE_CLICKS = 8
SEARCH_GRANULARITY_STEP = 0.02


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
    parser.add_argument("--sbd-root", type=str, required=True,
                        help="Path to benchmark_RELEASE/dataset.")
    parser.add_argument("--output-root", type=str, required=True,
                        help="Directory where processed records will be stored.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dinov3-repo", type=str, default=None)
    parser.add_argument("--dinov3-model", type=str, default="dinov3_vitl16",
                        choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"],
                        help="DINOv3 architecture matching the selected weights.")
    parser.add_argument("--dinov3-weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of official SBD train split carved out as validation.")
    parser.add_argument("--granularities", type=float, nargs="+",
                        default=list(DEFAULT_GRANULARITIES))
    parser.add_argument("--min-object-area", type=int, default=400)
    parser.add_argument("--preview-per-split", type=int, default=6)
    parser.add_argument("--max-images-per-split", type=int, default=0,
                        help="0 means process the whole split. Useful for preview generation.")
    parser.add_argument("--candidate-threshold", type=float, default=0.50)
    parser.add_argument("--max-hole-ratio", type=float, default=0.04)
    parser.add_argument("--work-size", type=int, default=512)
    parser.add_argument("--segments", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip finished samples under output-root and continue.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Rebuild output files instead of resuming.")
    parser.add_argument("--log-every-images", type=int, default=25,
                        help="Print split progress every N loaded images.")
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
    candidates = [
        path,
        path / "benchmark_RELEASE" / "dataset",
        path / "dataset",
    ]
    for candidate in candidates:
        if (candidate / "img").is_dir() and (candidate / "inst").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate SBD dataset root. Expected either the dataset directory itself "
        "or a parent containing benchmark_RELEASE/dataset."
    )


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
        "previews/train",
        "previews/val",
        "previews/test",
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


def save_preview(
    output_path: Path,
    image: np.ndarray,
    gt_mask: np.ndarray,
    simple_prob: np.ndarray,
    heuristic_masks: np.ndarray,
    corrected_probs: np.ndarray,
    granularities: List[float],
):
    preview_tiles = [image]
    preview_tiles.append(overlay_mask(image, gt_mask, (0, 255, 0)))
    preview_tiles.append(overlay_mask(image, simple_prob >= 0.49, (255, 255, 0)))

    for index, granularity in enumerate(granularities):
        del granularity
        preview_tiles.append(overlay_mask(image, heuristic_masks[index] > 0, (255, 128, 0)))
        preview_tiles.append(overlay_mask(image, corrected_probs[index] >= 0.49, (255, 0, 0)))

    rows = []
    row_width = 3
    for start in range(0, len(preview_tiles), row_width):
        row = preview_tiles[start:start + row_width]
        max_h = max(tile.shape[0] for tile in row)
        padded = []
        for tile in row:
            if tile.shape[0] != max_h:
                pad = np.zeros((max_h - tile.shape[0], tile.shape[1], 3), dtype=tile.dtype)
                tile = np.concatenate([tile, pad], axis=0)
            padded.append(tile)
        rows.append(np.concatenate(padded, axis=1))
    canvas = np.concatenate(rows, axis=0)
    imwrite_unicode(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def build_runtime_slider_masks(
    refiner: DinoGranularityRefiner,
    image: np.ndarray,
    simple_prob: np.ndarray,
    click_xy: Tuple[int, int],
    granularities: List[float],
) -> List[np.ndarray]:
    refiner.prepare_image(image)
    refiner.build(
        image,
        simple_prob.astype(np.float32),
        click_xy,
    )
    masks = []
    for granularity in granularities:
        masks.append(refiner.mask_at(float(granularity)).astype(np.uint8))
    return masks


def build_search_granularities(target_granularities: List[float]) -> List[float]:
    start = min(min(float(x) for x in target_granularities), 0.01)
    dense = np.arange(start, 1.0 + SEARCH_GRANULARITY_STEP * 0.5, SEARCH_GRANULARITY_STEP, dtype=np.float32)
    merged = np.unique(np.concatenate([dense, np.asarray(target_granularities, dtype=np.float32)]))
    return [float(x) for x in merged.tolist()]


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


def _component_at_click(mask: np.ndarray, click_xy: Tuple[int, int]) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    if int(mask_uint8.sum()) == 0:
        return mask_uint8

    num_labels, labels = cv2.connectedComponents(mask_uint8)
    click_x = int(np.clip(click_xy[0], 0, mask.shape[1] - 1))
    click_y = int(np.clip(click_xy[1], 0, mask.shape[0] - 1))
    click_label = int(labels[click_y, click_x])
    if click_label > 0:
        return (labels == click_label).astype(np.uint8)

    coords_yx = np.argwhere(mask_uint8 > 0)
    if coords_yx.size == 0:
        return mask_uint8
    sq_dist = (coords_yx[:, 1] - click_x) ** 2 + (coords_yx[:, 0] - click_y) ** 2
    nearest_y, nearest_x = coords_yx[int(np.argmin(sq_dist))]
    nearest_label = int(labels[int(nearest_y), int(nearest_x)])
    if nearest_label > 0:
        return (labels == nearest_label).astype(np.uint8)
    return mask_uint8


def _click_disk(mask_shape: Tuple[int, int], click_xy: Tuple[int, int], radius: int) -> np.ndarray:
    disk = np.zeros(mask_shape, dtype=np.uint8)
    cv2.circle(disk, (int(click_xy[0]), int(click_xy[1])), int(max(radius, 1)), 1, thickness=-1)
    return disk


def component_area_ratio(mask: np.ndarray, gt_mask: np.ndarray, click_xy: Tuple[int, int]) -> float:
    gt_component = _component_at_click(gt_mask, click_xy)
    gt_area = max(int(gt_component.sum()), 1)
    pred_component = _component_at_click(mask, click_xy)
    return float(int(pred_component.sum()) / gt_area)


def pick_click_xy_from_mask(
    mask: np.ndarray,
    used_points: set[Tuple[int, int]] | None = None,
    min_distance: int = 6,
) -> Tuple[int, int] | None:
    mask_uint8 = (mask > 0).astype(np.uint8)
    if int(mask_uint8.sum()) == 0:
        return None

    working = mask_uint8.copy()
    used_points = used_points or set()
    min_distance = max(int(min_distance), 0)

    while int(working.sum()) > 0:
        padded = np.pad(working, ((1, 1), (1, 1)), mode="constant")
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        x = int(x)
        y = int(y)

        is_reused = False
        for used_x, used_y in used_points:
            if (used_x - x) ** 2 + (used_y - y) ** 2 < (min_distance ** 2):
                is_reused = True
                break
        if not is_reused:
            return x, y

        cv2.circle(working, (x, y), min_distance, 0, thickness=-1)
    return None


def select_slider_masks_for_sample(
    heuristic_masks: List[np.ndarray],
    search_granularities: List[float],
    gt_mask: np.ndarray,
    click_xy: Tuple[int, int],
    target_coverages: List[float],
) -> Tuple[List[np.ndarray], List[float]]:
    coverage_ratios = np.asarray(
        [component_area_ratio(mask, gt_mask, click_xy) for mask in heuristic_masks],
        dtype=np.float32,
    )
    selected_masks: List[np.ndarray] = []
    selected_granularities: List[float] = []
    last_index = 0

    for target in target_coverages:
        candidate_slice = coverage_ratios[last_index:]
        if candidate_slice.size == 0:
            best_index = len(heuristic_masks) - 1
        else:
            relative_index = int(np.argmin(np.abs(candidate_slice - float(target))))
            best_index = last_index + relative_index
        selected_masks.append(heuristic_masks[best_index].copy())
        selected_granularities.append(float(search_granularities[best_index]))
        last_index = best_index

    return selected_masks, selected_granularities


def build_low_granularity_support(
    target_mask: np.ndarray,
    click_xy: Tuple[int, int],
    granularity: float,
) -> np.ndarray:
    component = _component_at_click(target_mask, click_xy)
    area = int(component.sum())
    if area == 0:
        return component

    if granularity <= 0.05:
        erosion_scale, dilation_scale, click_scale = 0.16, 0.08, 0.08
    elif granularity <= 0.10:
        erosion_scale, dilation_scale, click_scale = 0.13, 0.10, 0.10
    else:
        erosion_scale, dilation_scale, click_scale = 0.09, 0.13, 0.12

    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel_size(area, erosion_scale, 7, 31), _odd_kernel_size(area, erosion_scale, 7, 31)),
    )
    core = cv2.erode(component, erosion_kernel, iterations=1)
    if int(core.sum()) < 9:
        core = component.copy()

    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel_size(area, dilation_scale, 5, 25), _odd_kernel_size(area, dilation_scale, 5, 25)),
    )
    expanded = cv2.dilate(core, dilation_kernel, iterations=1)
    click_radius = _odd_kernel_size(area, click_scale, 5, 21) // 2
    click_disk = _click_disk(component.shape, click_xy, click_radius)
    support = np.logical_and(expanded > 0, component > 0)
    support = np.logical_or(support, np.logical_and(click_disk > 0, component > 0))
    return support.astype(np.uint8)


def correct_towards_coverage(
    predictor,
    image: np.ndarray,
    first_click: clicker.Click,
    gt_mask: np.ndarray,
    init_prev_mask: np.ndarray,
    click_xy: Tuple[int, int],
    target_coverage: float,
    max_clicks: int = MAX_CORRECTIVE_CLICKS,
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
    gt_component = _component_at_click(gt_mask, click_xy)
    gt_area = max(int(gt_component.sum()), 1)
    best_distance = float("inf")
    stale_steps = 0
    used_points = {(int(click_xy[0]), int(click_xy[1]))}
    target_value = float(target_coverage)
    stop_tolerance = 0.005 if target_value >= 0.95 else (0.01 if target_value >= 0.80 else 0.03)
    stale_limit = 8 if target_value >= 0.95 else (4 if target_value >= 0.80 else 2)

    for _ in range(max_clicks):
        pred_mask = corrected_prob >= pred_thresh
        pred_component = _component_at_click(pred_mask.astype(np.uint8), click_xy)
        current_coverage = float(int(pred_component.sum()) / gt_area)
        current_distance = abs(current_coverage - float(target_coverage))
        if current_distance < best_distance - 1e-4:
            best_distance = current_distance
            stale_steps = 0
        else:
            stale_steps += 1

        if current_distance <= stop_tolerance or stale_steps >= stale_limit:
            break

        allow_positive = target_value >= 0.20
        if current_coverage > target_value:
            error_region = np.logical_and(pred_component > 0, gt_component == 0).astype(np.uint8)
            if int(error_region.sum()) == 0:
                error_region = pred_component.astype(np.uint8)
            click_xy_next = pick_click_xy_from_mask(error_region, used_points=used_points, min_distance=5)
            click_is_positive = False
        else:
            if not allow_positive:
                break
            error_region = np.logical_and(gt_component > 0, pred_component == 0).astype(np.uint8)
            if int(error_region.sum()) == 0:
                break
            click_xy_next = pick_click_xy_from_mask(error_region, used_points=used_points, min_distance=5)
            click_is_positive = True

        if click_xy_next is None:
            break

        corrective_click = clicker.Click(
            is_positive=click_is_positive,
            coords=(int(click_xy_next[1]), int(click_xy_next[0])),
        )
        click_log.append([
            int(corrective_click.coords[1]),
            int(corrective_click.coords[0]),
            int(corrective_click.is_positive),
        ])
        used_points.add((int(corrective_click.coords[1]), int(corrective_click.coords[0])))
        clicks.append(corrective_click)

        updated_prob = run_simpleclick_prediction(
            predictor,
            image,
            clicks,
            prev_prediction=init_prev_mask.astype(np.float32),
        ).astype(np.float32)
        corrected_prob = updated_prob

    padded_clicks = np.full((max_clicks, 3), -1, dtype=np.int16)
    if click_log:
        padded_clicks[:len(click_log)] = np.asarray(click_log, dtype=np.int16)
    return corrected_prob, padded_clicks


def generate_record(
    predictor,
    refiner: DinoGranularityRefiner,
    image: np.ndarray,
    gt_mask: np.ndarray,
    click_xy: Tuple[int, int],
    granularities: List[float],
) -> Dict:
    height, width = gt_mask.shape
    positive_click = clicker.Click(is_positive=True, coords=(click_xy[1], click_xy[0]))
    simple_prob = run_simpleclick_prediction(predictor, image, [positive_click]).astype(np.float32)

    search_granularities = build_search_granularities(granularities)
    search_masks = build_runtime_slider_masks(
        refiner=refiner,
        image=image,
        simple_prob=simple_prob,
        click_xy=click_xy,
        granularities=search_granularities,
    )
    heuristic_masks, heuristic_granularities = select_slider_masks_for_sample(
        heuristic_masks=search_masks,
        search_granularities=search_granularities,
        gt_mask=gt_mask,
        click_xy=click_xy,
        target_coverages=granularities,
    )
    corrected_probs = []
    corrective_clicks = []

    for heuristic_mask, target_coverage in zip(heuristic_masks, granularities):
        corrected_prob, padded_clicks = correct_towards_coverage(
            predictor,
            image,
            positive_click,
            gt_mask,
            heuristic_mask,
            click_xy=click_xy,
            target_coverage=float(target_coverage),
            max_clicks=MAX_CORRECTIVE_CLICKS,
        )
        corrected_probs.append(corrected_prob)
        corrective_clicks.append(padded_clicks)

    heuristic_masks = np.stack(heuristic_masks, axis=0).astype(np.uint8)
    corrected_probs = np.stack(corrected_probs, axis=0).astype(np.float16)
    corrective_clicks = np.stack(corrective_clicks, axis=0).astype(np.int16)
    actual_granularities = np.asarray(
        [
            component_area_ratio(corrected_probs[index] >= 0.49, gt_mask, click_xy)
            for index in range(len(granularities))
        ],
        dtype=np.float32,
    )
    flat_signatures = [heuristic_masks[index].tobytes() for index in range(len(granularities))]
    unique_signature_order = []
    signature_to_level = {}
    effective_level_mask = np.zeros(len(granularities), dtype=np.uint8)
    for level_index, signature in enumerate(flat_signatures):
        if signature not in signature_to_level:
            signature_to_level[signature] = len(unique_signature_order)
            unique_signature_order.append(level_index)
            effective_level_mask[level_index] = 1
    effective_level_indices = np.asarray(unique_signature_order, dtype=np.int16)
    num_effective_levels = np.int16(len(unique_signature_order))

    return {
        "gt_mask": gt_mask.astype(np.uint8),
        "simple_prob": simple_prob.astype(np.float16),
        "heuristic_masks": heuristic_masks,
        "corrected_probs": corrected_probs,
        "corrective_clicks": corrective_clicks,
        "positive_click_map": click_to_map(height, width, click_xy),
        "effective_level_mask": effective_level_mask,
        "effective_level_indices": effective_level_indices,
        "num_effective_levels": num_effective_levels,
        "granularities": actual_granularities,
        "target_granularities": np.asarray(granularities, dtype=np.float32),
        "heuristic_granularities": np.asarray(heuristic_granularities, dtype=np.float32),
    }


def write_split_files(output_root: Path, split_to_ids: Dict[str, List[str]]):
    for split_name, image_ids in split_to_ids.items():
        split_path = output_root / "splits" / f"{split_name}.txt"
        with open(split_path, "w", encoding="utf-8") as handle:
            for image_id in image_ids:
                handle.write(f"{image_id}\n")


def append_jsonl(path: Path, record: Dict):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    split_to_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": official_test_ids,
    }
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

                sample_name = f"{image_id}_inst{instance_id:03d}"
                npz_path = output_root / "records" / split_name / f"{sample_name}.npz"
                if args.resume and not args.overwrite and npz_path.exists():
                    split_skipped += 1
                    if sample_name not in existing_metadata:
                        click_xy = pick_first_positive_click(gt_mask)
                        meta = {
                            "sample_name": sample_name,
                            "split": split_name,
                            "image_id": image_id,
                            "instance_id": instance_id,
                            "image_path": str(dataset_root / "img" / f"{image_id}.jpg"),
                            "npz_path": str(npz_path),
                            "height": int(image.shape[0]),
                            "width": int(image.shape[1]),
                            "positive_click_xy": [int(click_xy[0]), int(click_xy[1])],
                            "granularities": granularities,
                            "target_granularities": granularities,
                        }
                        append_jsonl(split_jsonl, meta)
                        existing_metadata[sample_name] = meta
                    continue

                click_xy = pick_first_positive_click(gt_mask)
                record_tensors = generate_record(
                    predictor=predictor,
                    refiner=refiner,
                    image=image,
                    gt_mask=gt_mask,
                    click_xy=click_xy,
                    granularities=granularities,
                )

                np.savez_compressed(
                    npz_path,
                    gt_mask=record_tensors["gt_mask"],
                    simple_prob=record_tensors["simple_prob"],
                    heuristic_masks=record_tensors["heuristic_masks"],
                    corrected_probs=record_tensors["corrected_probs"],
                    corrective_clicks=record_tensors["corrective_clicks"],
                    positive_click_map=record_tensors["positive_click_map"],
                    effective_level_mask=record_tensors["effective_level_mask"],
                    effective_level_indices=record_tensors["effective_level_indices"],
                    num_effective_levels=record_tensors["num_effective_levels"],
                    granularities=record_tensors["granularities"],
                    target_granularities=record_tensors["target_granularities"],
                    heuristic_granularities=record_tensors["heuristic_granularities"],
                    positive_click=np.asarray(click_xy, dtype=np.int16),
                )

                meta = {
                    "sample_name": sample_name,
                    "split": split_name,
                    "image_id": image_id,
                    "instance_id": instance_id,
                    "image_path": str(dataset_root / "img" / f"{image_id}.jpg"),
                    "npz_path": str(npz_path),
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                    "positive_click_xy": [int(click_xy[0]), int(click_xy[1])],
                    "granularities": record_tensors["granularities"].astype(np.float32).tolist(),
                    "target_granularities": record_tensors["target_granularities"].astype(np.float32).tolist(),
                    "heuristic_granularities": record_tensors["heuristic_granularities"].astype(np.float32).tolist(),
                    "num_effective_levels": int(record_tensors["num_effective_levels"]),
                    "effective_level_indices": record_tensors["effective_level_indices"].tolist(),
                }
                append_jsonl(split_jsonl, meta)
                existing_metadata[sample_name] = meta
                split_records += 1

                if preview_budget > 0:
                    preview_path = output_root / "previews" / split_name / f"{sample_name}.jpg"
                    save_preview(
                        preview_path,
                        image=image,
                        gt_mask=gt_mask,
                        simple_prob=record_tensors["simple_prob"].astype(np.float32),
                        heuristic_masks=record_tensors["heuristic_masks"],
                        corrected_probs=record_tensors["corrected_probs"].astype(np.float32),
                        granularities=granularities,
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
            "# SBD Granularity Refine Dataset",
            "",
            "Generated from the current SimpleClick + DINOv3 pipeline.",
            "",
            "Each `.npz` sample contains:",
            "- `gt_mask`: binary SBD instance mask",
            "- `simple_prob`: first-click SimpleClick probability map",
            "- `heuristic_masks`: actual runtime no-refine DINOv3 slider masks at fixed granularities",
            "- `corrected_probs`: pseudo teacher that keeps each slider stage but reshapes it to look more like SimpleClick",
            "- `corrective_clicks`: `[x, y, is_positive]` for each granularity",
            "- `positive_click_map`: Gaussian map of the first click",
            "- `effective_level_mask`: which granularity slots are genuinely distinct",
            "- `effective_level_indices`: the distinct granularity indices kept for supervision",
            "- `num_effective_levels`: number of distinct levels in this sample",
            "- `positive_click`: first click `[x, y]`",
            "- `granularities`: the slider values used for this sample",
            "",
            "Splits:",
            "- `train`: official SBD train minus a deterministic validation carve-out",
            "- `val`: deterministic subset of official SBD train",
            "- `test`: official SBD val",
        ]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
