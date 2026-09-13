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


def semantic_trajectory_diagnostics(
    matched_prediction_indices: Sequence[int],
    prediction_areas: Sequence[int],
    target_prediction_ious: Sequence[Sequence[float]],
    signature_decimals: int = 12,
) -> dict[str, float | int]:
    """Diagnose ordering, reverse area changes, and observable state changes.

    The observable signature combines mask area with IoU against every annotated
    semantic target. Distinct signatures are a conservative lower bound on the
    number of distinct binary masks because two masks can share the same area and
    target overlaps.
    """
    matched = np.asarray(matched_prediction_indices, dtype=np.int64)
    areas = np.asarray(prediction_areas, dtype=np.int64)
    target_ious = np.asarray(target_prediction_ious, dtype=np.float64)

    if matched.ndim != 1 or matched.size == 0:
        raise ValueError("matched_prediction_indices must be a non-empty vector")
    if areas.ndim != 1 or areas.size < 2:
        raise ValueError("prediction_areas must contain at least two states")
    if target_ious.ndim != 2 or target_ious.shape[1] != areas.size:
        raise ValueError("target_prediction_ious must be [levels, states]")
    if target_ious.shape[0] != matched.size:
        raise ValueError("one matched index is required per semantic level")

    semantic_order_success = float(np.all(np.diff(matched) > 0))
    area_deltas = np.diff(areas.astype(np.float64))
    previous_areas = np.maximum(areas[:-1].astype(np.float64), 1.0)
    backward_area_rate = float(
        np.mean(np.maximum(-area_deltas, 0.0) / previous_areas)
    )
    rounded_ious = np.round(target_ious, decimals=int(signature_decimals))
    signatures = [
        (int(areas[state]), *rounded_ious[:, state].tolist())
        for state in range(areas.size)
    ]
    observable_state_count = len(set(signatures))

    return {
        "semantic_order_success": semantic_order_success,
        "backward_area_rate": backward_area_rate,
        "observable_state_count": int(observable_state_count),
    }


def rescore_semantic_record(
    record: dict[str, object],
    state_indices: Sequence[int],
    recall_thresholds: Sequence[float] = DEFAULT_RECALL_THRESHOLDS,
) -> dict[str, float | int]:
    """Rescore a saved trajectory on a shared subset of slider states."""
    indices = np.asarray(state_indices, dtype=np.int64)
    target_ious = np.asarray(record["target_prediction_ious"], dtype=np.float64)
    areas = np.asarray(record["prediction_areas"], dtype=np.int64)
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("state_indices must contain at least two indices")
    if indices.min() < 0 or indices.max() >= target_ious.shape[1]:
        raise ValueError("state index is outside the saved trajectory")

    selected_ious = target_ious[:, indices]
    selected_areas = areas[indices]
    matched = selected_ious.argmax(axis=1)
    best_ious = selected_ious[
        np.arange(selected_ious.shape[0]),
        matched,
    ]
    diagnostics = semantic_trajectory_diagnostics(
        matched,
        selected_areas,
        selected_ious,
    )
    diagnostics["semantic_order_rate"] = diagnostics.pop(
        "semantic_order_success"
    )
    result: dict[str, float | int] = {
        "m_biou": float(np.mean(best_ious)),
        **diagnostics,
    }
    for threshold in recall_thresholds:
        suffix = f"{round(float(threshold) * 100):02d}"
        hits = best_ious >= float(threshold)
        result[f"lr{suffix}"] = float(np.mean(hits))
        result[f"hsr{suffix}"] = float(np.all(hits))
    return result


def summarize_semantic_trajectory_records(
    records: Sequence[dict[str, object]],
) -> dict[str, float | int | None]:
    if not records:
        return {
            "samples": 0,
            "semantic_order_rate": None,
            "backward_area_rate": None,
            "observable_state_count": None,
        }

    diagnostics = [
        semantic_trajectory_diagnostics(
            record["matched_prediction_indices"],
            record["prediction_areas"],
            record["target_prediction_ious"],
        )
        for record in records
    ]
    names = tuple(diagnostics[0])
    summary = {
        name: float(np.mean([float(item[name]) for item in diagnostics]))
        for name in names
    }
    summary["semantic_order_rate"] = summary.pop("semantic_order_success")
    return {"samples": len(records), **summary}
