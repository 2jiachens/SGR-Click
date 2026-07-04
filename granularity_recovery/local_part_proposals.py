from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class LocalProposalResult:
    patch_box_xyxy: Tuple[int, int, int, int]
    parent_patch_mask: np.ndarray
    candidates: List[np.ndarray]
    descriptors: List[Dict[str, float]]


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(x1, 0, max(width - 1, 0)))
    y1 = int(np.clip(y1, 0, max(height - 1, 0)))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def _extract_click_local_semantic_region(
    parent_mask: np.ndarray,
    click_xy: Tuple[int, int],
    semantic_radius: int,
) -> np.ndarray:
    click_x, click_y = click_xy
    height, width = parent_mask.shape
    click_x = int(np.clip(click_x, 0, width - 1))
    click_y = int(np.clip(click_y, 0, height - 1))

    yy, xx = np.mgrid[0:height, 0:width]
    local_support = ((xx - click_x) ** 2 + (yy - click_y) ** 2) <= semantic_radius ** 2
    seed_region = parent_mask.astype(bool) & local_support
    if not np.any(seed_region):
        seed_region = parent_mask.astype(bool).copy()

    seed_region = seed_region.astype(np.uint8)
    if seed_region[click_y, click_x] == 0:
        cv2.circle(seed_region, (click_x, click_y), 2, 1, -1)

    _, labels = cv2.connectedComponents(seed_region, connectivity=8)
    label = int(labels[click_y, click_x])
    if label <= 0:
        fallback = np.zeros_like(parent_mask, dtype=np.uint8)
        cv2.circle(fallback, (click_x, click_y), 3, 1, -1)
        return fallback > 0
    return labels == label


def _compute_patch_box(parent_mask: np.ndarray, click_xy: Tuple[int, int], patch_size: int, margin: int) -> Tuple[int, int, int, int]:
    height, width = parent_mask.shape
    click_x, click_y = click_xy
    ys, xs = np.where(parent_mask)
    # Base rule: always start from a click-centered fixed local window.
    base_size = int(round(patch_size))
    max_size = int(round(1.25 * patch_size))
    min_size = int(round(0.85 * patch_size))
    target_size = int(np.clip(base_size, min_size, max_size))

    center_x = int(click_x)
    center_y = int(click_y)

    if len(xs) > 0:
        # DINOv3 already influenced the parent mask. Use only its local support near
        # the click to adjust the window center slightly, instead of letting the full
        # parent extent dominate the crop size.
        semantic_radius = max(10, int(round(0.28 * patch_size)))
        local_semantic = _extract_click_local_semantic_region(parent_mask, click_xy, semantic_radius=semantic_radius)
        sys, sxs = np.where(local_semantic)
        if len(sxs) > 0:
            semantic_cx = float(sxs.mean())
            semantic_cy = float(sys.mean())
            center_x = int(round(0.80 * click_x + 0.20 * semantic_cx))
            center_y = int(round(0.80 * click_y + 0.20 * semantic_cy))

            semantic_w = int(sxs.max()) + 1 - int(sxs.min())
            semantic_h = int(sys.max()) + 1 - int(sys.min())
            semantic_extent = max(semantic_w, semantic_h) + 2 * margin
            target_size = int(np.clip(max(base_size, semantic_extent), min_size, max_size))

    half = int(round(target_size / 2.0))
    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + target_size
    y2 = y1 + target_size
    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)

    if len(xs) > 0:
        crop_mask = parent_mask[y1:y2, x1:x2]
        if crop_mask.sum() < max(9, 0.03 * parent_mask.sum()):
            # If the crop misses almost all parent support, keep the same size but
            # recenter slightly toward the local parent centroid near the click.
            local_semantic = _extract_click_local_semantic_region(
                parent_mask,
                click_xy,
                semantic_radius=max(12, int(round(0.38 * patch_size))),
            )
            sys, sxs = np.where(local_semantic)
            if len(sxs) > 0:
                center_x = int(round(0.70 * click_x + 0.30 * float(sxs.mean())))
                center_y = int(round(0.70 * click_y + 0.30 * float(sys.mean())))
                x1 = center_x - half
                y1 = center_y - half
                x2 = x1 + target_size
                y2 = y1 + target_size
                x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
    return x1, y1, x2, y2


def _component_from_seed(binary_mask: np.ndarray, click_xy: Tuple[int, int]) -> np.ndarray:
    binary_mask = binary_mask.astype(bool)
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    click_x, click_y = click_xy
    height, width = binary_mask.shape
    click_x = int(np.clip(click_x, 0, width - 1))
    click_y = int(np.clip(click_y, 0, height - 1))

    if not binary_mask[click_y, click_x]:
        repaired = binary_mask.astype(np.uint8)
        cv2.circle(repaired, (click_x, click_y), 2, 1, -1)
        binary_mask = repaired > 0

    _, labels = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
    label = int(labels[click_y, click_x])
    if label <= 0:
        seed = np.zeros_like(binary_mask, dtype=np.uint8)
        cv2.circle(seed, (click_x, click_y), 3, 1, -1)
        return seed.astype(bool)
    return labels == label


def _deduplicate_masks(masks: Sequence[np.ndarray], min_area: int) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    seen = set()
    for mask in masks:
        mask = mask.astype(bool)
        if int(mask.sum()) < min_area:
            continue
        signature = mask.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mask)
    return unique


def _distance_to_click(height: int, width: int, click_xy: Tuple[int, int]) -> np.ndarray:
    click_x, click_y = click_xy
    yy, xx = np.mgrid[0:height, 0:width]
    return np.sqrt((xx - click_x) ** 2 + (yy - click_y) ** 2)


def _click_centered_probability_candidates(simple_prob_patch: np.ndarray, parent_mask: np.ndarray, click_xy_patch: Tuple[int, int]) -> List[np.ndarray]:
    click_x, click_y = click_xy_patch
    seed_prob = float(simple_prob_patch[click_y, click_x])
    distance = _distance_to_click(parent_mask.shape[0], parent_mask.shape[1], click_xy_patch)
    parent_area = max(int(parent_mask.sum()), 1)
    base_radius = max(5.0, np.sqrt(parent_area / np.pi) * 0.12)
    candidates = []

    for radius_scale in (0.55, 0.80, 1.10, 1.55):
        radius = base_radius * radius_scale
        local_support = distance <= radius
        for delta in (0.02, 0.07, 0.12):
            threshold = max(0.10, seed_prob - delta)
            binary = parent_mask & local_support & (simple_prob_patch >= threshold)
            candidates.append(_component_from_seed(binary, click_xy_patch))
    return candidates


def _color_similarity_candidates(image_patch: np.ndarray, parent_mask: np.ndarray, click_xy_patch: Tuple[int, int]) -> List[np.ndarray]:
    click_x, click_y = click_xy_patch
    lab = cv2.cvtColor(image_patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    seed = lab[click_y, click_x]
    delta = np.linalg.norm(lab - seed[None, None, :], axis=2)
    distance = _distance_to_click(parent_mask.shape[0], parent_mask.shape[1], click_xy_patch)

    candidates = []
    for threshold, radius_scale in ((6.0, 0.65), (9.0, 0.90), (13.0, 1.20), (18.0, 1.60)):
        local_support = distance <= max(6.0, radius_scale * 14.0)
        binary = (delta <= threshold) & parent_mask & local_support
        candidates.append(_component_from_seed(binary, click_xy_patch))
    return candidates


def _distance_candidates(parent_mask: np.ndarray, click_xy_patch: Tuple[int, int]) -> List[np.ndarray]:
    distance = _distance_to_click(parent_mask.shape[0], parent_mask.shape[1], click_xy_patch)
    parent_area = max(int(parent_mask.sum()), 1)
    base_radius = max(4.0, np.sqrt(parent_area / np.pi) * 0.10)
    candidates = []
    for scale in (0.45, 0.65, 0.95, 1.30, 1.70):
        binary = (distance <= base_radius * scale) & parent_mask
        candidates.append(_component_from_seed(binary, click_xy_patch))
    return candidates


def _edge_barrier_candidates(image_patch: np.ndarray, parent_mask: np.ndarray, click_xy_patch: Tuple[int, int]) -> List[np.ndarray]:
    gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient = gradient / max(float(gradient.max()), 1e-6)
    distance = _distance_to_click(parent_mask.shape[0], parent_mask.shape[1], click_xy_patch)

    candidates = []
    for grad_thr, dist_thr in ((0.10, 10.0), (0.16, 15.0), (0.22, 22.0)):
        barrier_free = gradient <= grad_thr
        local_support = distance <= dist_thr
        binary = parent_mask & barrier_free & local_support
        candidates.append(_component_from_seed(binary, click_xy_patch))
    return candidates


def _simpleclick_band_candidates(simple_prob_patch: np.ndarray, parent_mask: np.ndarray, click_xy_patch: Tuple[int, int]) -> List[np.ndarray]:
    click_x, click_y = click_xy_patch
    seed_prob = float(simple_prob_patch[click_y, click_x])
    candidates = []
    for low, high in (
        (max(0.10, seed_prob - 0.20), max(0.18, seed_prob - 0.05)),
        (max(0.12, seed_prob - 0.12), min(1.0, seed_prob + 0.05)),
        (0.18, 0.55),
    ):
        band = (simple_prob_patch >= low) & (simple_prob_patch <= high) & parent_mask
        candidates.append(_component_from_seed(band, click_xy_patch))
    return candidates


def _describe_mask(mask: np.ndarray, parent_mask: np.ndarray, simple_prob_patch: np.ndarray) -> Dict[str, float]:
    parent_area = max(float(parent_mask.sum()), 1.0)
    area = float(mask.sum())
    overlap_simple = float(np.logical_and(mask, simple_prob_patch >= 0.49).sum())
    mean_prob = float(simple_prob_patch[mask].mean()) if np.any(mask) else 0.0
    return {
        "area": area,
        "area_ratio_to_parent": area / parent_area,
        "simple_overlap_ratio": overlap_simple / max(area, 1.0),
        "mean_simple_prob": mean_prob,
    }


def _subsample_candidates(candidates: List[np.ndarray], max_candidates: int) -> List[np.ndarray]:
    if len(candidates) <= max_candidates:
        return candidates
    candidates = sorted(candidates, key=lambda mask: int(mask.sum()))
    if max_candidates <= 4:
        indices = np.linspace(0, len(candidates) - 1, max_candidates).round().astype(int).tolist()
        return [candidates[index] for index in indices]

    keep = []
    smallest = list(range(min(4, len(candidates))))
    largest = list(range(max(len(candidates) - 2, 0), len(candidates)))
    middle_count = max_candidates - len(set(smallest + largest))
    middle_pool = list(range(min(4, len(candidates)), max(len(candidates) - 2, min(4, len(candidates)))))
    if middle_count > 0 and middle_pool:
        middle = np.linspace(0, len(middle_pool) - 1, middle_count).round().astype(int).tolist()
        middle = [middle_pool[index] for index in middle]
    else:
        middle = []
    for idx in smallest + middle + largest:
        if idx not in keep and 0 <= idx < len(candidates):
            keep.append(idx)
    keep = keep[:max_candidates]
    return [candidates[index] for index in keep]


def build_local_part_proposals(
    image: np.ndarray,
    simple_prob: np.ndarray,
    parent_mask: np.ndarray,
    click_xy: Tuple[int, int],
    patch_size: int = 128,
    margin: int = 12,
    max_candidates: int = 12,
    min_area: int = 9,
) -> LocalProposalResult:
    x1, y1, x2, y2 = _compute_patch_box(parent_mask, click_xy, patch_size=patch_size, margin=margin)
    image_patch = image[y1:y2, x1:x2]
    simple_prob_patch = simple_prob[y1:y2, x1:x2].astype(np.float32)
    parent_patch_mask = parent_mask[y1:y2, x1:x2].astype(bool)

    click_xy_patch = (int(click_xy[0] - x1), int(click_xy[1] - y1))
    seed_mask = np.zeros_like(parent_patch_mask, dtype=np.uint8)
    cv2.circle(seed_mask, click_xy_patch, 2, 1, -1)

    raw_candidates: List[np.ndarray] = [
        np.logical_and(seed_mask > 0, parent_patch_mask),
    ]
    raw_candidates.extend(_distance_candidates(parent_patch_mask, click_xy_patch))
    raw_candidates.extend(_click_centered_probability_candidates(simple_prob_patch, parent_patch_mask, click_xy_patch))
    raw_candidates.extend(_simpleclick_band_candidates(simple_prob_patch, parent_patch_mask, click_xy_patch))
    raw_candidates.extend(_color_similarity_candidates(image_patch, parent_patch_mask, click_xy_patch))
    raw_candidates.extend(_edge_barrier_candidates(image_patch, parent_patch_mask, click_xy_patch))
    raw_candidates.append(parent_patch_mask.copy())

    candidates = _deduplicate_masks(raw_candidates, min_area=min_area)
    candidates = _subsample_candidates(candidates, max_candidates=max_candidates)
    descriptors = [_describe_mask(mask, parent_patch_mask, simple_prob_patch) for mask in candidates]

    return LocalProposalResult(
        patch_box_xyxy=(x1, y1, x2, y2),
        parent_patch_mask=parent_patch_mask,
        candidates=candidates,
        descriptors=descriptors,
    )
