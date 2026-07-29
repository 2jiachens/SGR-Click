from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_RECALL_THRESHOLDS = (0.60, 0.70)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    union = int(np.logical_or(first_mask, second_mask).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first_mask, second_mask).sum()) / float(union)


def hierarchical_semantic_metrics(
    predictions: Sequence[np.ndarray],
    semantic_targets: Sequence[np.ndarray],
    recall_thresholds: Sequence[float] = DEFAULT_RECALL_THRESHOLDS,
) -> dict[str, object]:
    """Score only annotated hierarchy levels, ordered from fine to coarse."""
    if not predictions:
        raise ValueError("at least one prediction is required")
    if not semantic_targets:
        raise ValueError("at least one semantic target is required")
    if any(not 0.0 <= float(threshold) <= 1.0 for threshold in recall_thresholds):
        raise ValueError("recall thresholds must be in [0, 1]")

    target_by_prediction = np.asarray(
        [
            [mask_iou(target, prediction) for prediction in predictions]
            for target in semantic_targets
        ],
        dtype=np.float64,
    )
    matched_indices = target_by_prediction.argmax(axis=1)
    best_ious = target_by_prediction[
        np.arange(len(semantic_targets)),
        matched_indices,
    ]
    metrics: dict[str, object] = {
        "m_biou": float(np.mean(best_ious)),
        "best_ious": best_ious.tolist(),
        "matched_prediction_indices": matched_indices.astype(int).tolist(),
        "target_prediction_ious": target_by_prediction.tolist(),
    }
    for threshold in recall_thresholds:
        suffix = f"{round(float(threshold) * 100):02d}"
        hits = best_ious >= float(threshold)
        metrics[f"lr{suffix}"] = float(np.mean(hits))
        metrics[f"hsr{suffix}"] = float(np.all(hits))
    return metrics


def summarize_hierarchical_records(
    records: Sequence[dict[str, object]],
    recall_thresholds: Sequence[float] = DEFAULT_RECALL_THRESHOLDS,
) -> dict[str, object]:
    metric_names = ["m_biou"]
    for threshold in recall_thresholds:
        suffix = f"{round(float(threshold) * 100):02d}"
        metric_names.extend((f"lr{suffix}", f"hsr{suffix}"))
    if not records:
        return {"samples": 0, **{name: None for name in metric_names}}
    return {
        "samples": len(records),
        **{
            name: float(np.mean([float(record[name]) for record in records]))
            for name in metric_names
        },
    }
