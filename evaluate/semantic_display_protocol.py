"""Shared display protocol for semantic-granularity model comparisons."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DISPLAY_MODES = ("raw", "ascending_union")


def apply_display_protocol(
    masks: Sequence[np.ndarray],
    mode: str,
) -> list[np.ndarray]:
    """Apply the same post-hoc display rule to an ascending control sequence.

    ``raw`` keeps every independently queried control unchanged.
    ``ascending_union`` emulates a monotone ascending display cache with a
    pixelwise cumulative union.  The operation is deliberately model-agnostic.
    """
    if mode not in DISPLAY_MODES:
        raise ValueError(f"unknown display mode: {mode!r}")
    if not masks:
        raise ValueError("at least one mask is required")

    binary_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    reference_shape = binary_masks[0].shape
    if any(mask.shape != reference_shape for mask in binary_masks):
        raise ValueError("all masks must have the same shape")
    if mode == "raw":
        return [mask.copy() for mask in binary_masks]

    displayed: list[np.ndarray] = []
    accumulated = np.zeros(reference_shape, dtype=bool)
    for mask in binary_masks:
        accumulated = np.logical_or(accumulated, mask)
        displayed.append(accumulated.copy())
    return displayed
