import numpy as np
import pytest

from evaluate.semantic_coverage import (
    hierarchical_semantic_metrics,
    summarize_hierarchical_records,
)


def square(start: int, stop: int) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[start:stop, start:stop] = True
    return mask


def test_hierarchical_metrics_score_recall_success():
    subpart = square(3, 5)
    part = square(2, 6)
    whole = square(1, 7)
    metrics = hierarchical_semantic_metrics(
        [subpart, part, whole],
        [subpart, part, whole],
    )
    assert metrics["lr60"] == 1.0
    assert metrics["hsr60"] == 1.0
    assert metrics["lr70"] == 1.0
    assert metrics["hsr70"] == 1.0
    assert metrics["m_biou"] == 1.0


def test_hierarchy_success_requires_every_level():
    subpart = square(3, 5)
    part = square(2, 6)
    whole = square(0, 8)
    metrics = hierarchical_semantic_metrics(
        [subpart, part],
        [subpart, part, whole],
    )
    assert metrics["lr60"] == pytest.approx(2.0 / 3.0)
    assert metrics["hsr60"] == 0.0


def test_hierarchical_summary_averages_records():
    summary = summarize_hierarchical_records(
        [
            {"lr60": 1.0, "hsr60": 1.0, "lr70": 0.5, "hsr70": 0.0, "m_biou": 0.8},
            {"lr60": 0.5, "hsr60": 0.0, "lr70": 0.0, "hsr70": 0.0, "m_biou": 0.6},
        ]
    )
    assert summary["samples"] == 2
    assert summary["lr60"] == pytest.approx(0.75)
    assert summary["hsr60"] == pytest.approx(0.5)
    assert summary["m_biou"] == pytest.approx(0.7)
