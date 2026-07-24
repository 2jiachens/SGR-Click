from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class MaskState:
    granularity: float
    mask: np.ndarray


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum()) / float(union)


def collapse_equivalent_states(
    states: Iterable[MaskState],
    equivalent_iou: float = 0.995,
) -> List[MaskState]:
    """Collapse adjacent masks into plateaus without consulting ground truth."""
    ordered = sorted(states, key=lambda state: float(state.granularity))
    if not ordered:
        return []

    plateaus: List[List[MaskState]] = [[ordered[0]]]
    for state in ordered[1:]:
        if mask_iou(plateaus[-1][-1].mask, state.mask) >= equivalent_iou:
            plateaus[-1].append(state)
        else:
            plateaus.append([state])

    collapsed = []
    for plateau in plateaus:
        center = 0.5 * (plateau[0].granularity + plateau[-1].granularity)
        representative = min(plateau, key=lambda state: abs(state.granularity - center))
        collapsed.append(representative)
    return collapsed


def select_effective_states(
    states: Sequence[MaskState],
    max_states: int = 10,
    required_granularities: Sequence[float] = (0.0, 0.5, 1.0),
) -> List[MaskState]:
    """Select states uniformly in cumulative mask-change distance."""
    ordered = sorted(states, key=lambda state: float(state.granularity))
    if max_states <= 0:
        raise ValueError("max_states must be positive.")
    if len(ordered) <= max_states:
        return ordered

    cumulative = np.zeros(len(ordered), dtype=np.float64)
    for index in range(1, len(ordered)):
        cumulative[index] = cumulative[index - 1] + 1.0 - mask_iou(
            ordered[index - 1].mask,
            ordered[index].mask,
        )

    selected = {
        min(
            range(len(ordered)),
            key=lambda index: abs(ordered[index].granularity - required),
        )
        for required in required_granularities
    }
    selected.update((0, len(ordered) - 1))

    while len(selected) < max_states:
        remaining = [index for index in range(len(ordered)) if index not in selected]
        if not remaining:
            break
        next_index = max(
            remaining,
            key=lambda index: (
                min(abs(cumulative[index] - cumulative[current]) for current in selected),
                -index,
            ),
        )
        selected.add(next_index)

    return [ordered[index] for index in sorted(selected)]


def select_anchor_preserving_states(
    scanned_states: Sequence[MaskState],
    anchor_granularities: Sequence[float],
    max_states: int = 10,
    equivalent_iou: float = 0.995,
) -> List[MaskState]:
    """Replace only exactly duplicated anchor masks, then fill freed slots."""
    ordered = sorted(scanned_states, key=lambda state: float(state.granularity))
    if not ordered:
        return []
    if len(anchor_granularities) > max_states:
        raise ValueError("anchor count cannot exceed max_states.")

    anchor_states = [
        min(ordered, key=lambda state: abs(state.granularity - anchor))
        for anchor in anchor_granularities
    ]
    selected: List[MaskState] = []
    for state in anchor_states:
        if not any(np.array_equal(state.mask, current.mask) for current in selected):
            selected.append(state)

    candidates = collapse_equivalent_states(ordered, equivalent_iou=equivalent_iou)
    while len(selected) < max_states:
        remaining = [
            state
            for state in candidates
            if not any(np.array_equal(state.mask, current.mask) for current in selected)
        ]
        if not remaining:
            break
        next_state = max(
            remaining,
            key=lambda state: (
                min(1.0 - mask_iou(state.mask, current.mask) for current in selected),
                min(abs(state.granularity - current.granularity) for current in selected),
                -state.granularity,
            ),
        )
        selected.append(next_state)

    return sorted(selected, key=lambda state: float(state.granularity))


def collapse_exact_semantic_stages(states: Sequence[MaskState]) -> List[MaskState]:
    """Keep the first slider value at which each semantic hierarchy stage appears."""
    ordered = sorted(states, key=lambda state: float(state.granularity))
    collapsed: List[MaskState] = []
    for state in ordered:
        if not collapsed or not np.array_equal(state.mask, collapsed[-1].mask):
            collapsed.append(state)
    return collapsed


def select_semantic_transition_states(
    stages: Sequence[MaskState],
    max_states: int = 10,
) -> List[MaskState]:
    """Keep endpoints and the semantic stages with the largest hierarchy jumps."""
    ordered = sorted(stages, key=lambda state: float(state.granularity))
    if max_states <= 0:
        raise ValueError("max_states must be positive.")
    if len(ordered) <= max_states:
        return ordered

    selected = {0, len(ordered) - 1}
    transition_scores = [
        (1.0 - mask_iou(ordered[index - 1].mask, ordered[index].mask), index)
        for index in range(1, len(ordered))
    ]
    for _, index in sorted(transition_scores, key=lambda item: (-item[0], item[1])):
        selected.add(index)
        if len(selected) >= max_states:
            break
    return [ordered[index] for index in sorted(selected)]


def build_ess10_states(
    states: Iterable[MaskState],
    max_states: int = 10,
    equivalent_iou: float = 0.995,
) -> List[MaskState]:
    collapsed = collapse_equivalent_states(states, equivalent_iou=equivalent_iou)
    return select_effective_states(collapsed, max_states=max_states)
