from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


LOW_HIERARCHY_POWER = 1.9
LOW_MODEL_POWER = 1.7
HIGH_HIERARCHY_POWER = 1.8
HIGH_MODEL_POWER = 1.6


def split_display_granularity(display_granularity: float) -> Tuple[float, float]:
    value = float(np.clip(display_granularity, 0.0, 1.0))
    if value <= 0.5:
        local = value / 0.5
        hierarchy = 0.5 * (local ** LOW_HIERARCHY_POWER)
        model = 0.5 * (local ** LOW_MODEL_POWER)
    else:
        local = (value - 0.5) / 0.5
        hierarchy = 0.5 + 0.5 * (local ** HIGH_HIERARCHY_POWER)
        model = 0.5 + 0.5 * (local ** HIGH_MODEL_POWER)
    return float(hierarchy), float(model)


def map_display_granularity_for_hierarchy(display_granularity: float) -> float:
    hierarchy, _ = split_display_granularity(display_granularity)
    return hierarchy


def map_display_granularity_for_model(display_granularity: float) -> float:
    _, model = split_display_granularity(display_granularity)
    return model


def map_display_granularities_for_hierarchy(display_granularities: Iterable[float]) -> np.ndarray:
    return np.asarray(
        [map_display_granularity_for_hierarchy(value) for value in display_granularities],
        dtype=np.float32,
    )


def map_display_granularities_for_model(display_granularities: Iterable[float]) -> np.ndarray:
    return np.asarray(
        [map_display_granularity_for_model(value) for value in display_granularities],
        dtype=np.float32,
    )
