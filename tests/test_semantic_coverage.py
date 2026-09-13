import numpy as np
import pytest

from evaluate.semantic_coverage import (
    hierarchical_semantic_metrics,
    rescore_semantic_record,
    semantic_trajectory_diagnostics,
    summarize_hierarchical_records,
    summarize_semantic_trajectory_records,
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


def test_semantic_trajectory_diagnostics():
    diagnostics = semantic_trajectory_diagnostics(
        matched_prediction_indices=[1, 2, 4],
        prediction_areas=[10, 20, 18, 18, 40],
        target_prediction_ious=[
            [0.1, 0.8, 0.6, 0.6, 0.2],
            [0.1, 0.4, 0.9, 0.9, 0.5],
            [0.0, 0.1, 0.2, 0.2, 0.9],
        ],
    )
    assert diagnostics["semantic_order_success"] == 1.0
    assert diagnostics["backward_area_rate"] == pytest.approx(0.025)
    assert diagnostics["observable_state_count"] == 4


def test_semantic_trajectory_order_rejects_ties():
    diagnostics = semantic_trajectory_diagnostics(
        matched_prediction_indices=[1, 1],
        prediction_areas=[5, 10],
        target_prediction_ious=[[0.2, 0.8], [0.1, 0.9]],
    )
    assert diagnostics["semantic_order_success"] == 0.0


def test_rescore_semantic_record_uses_shared_state_budget():
    record = {
        "prediction_areas": [10, 15, 20, 25, 30],
        "target_prediction_ious": [
            [0.8, 0.9, 0.5, 0.2, 0.1],
            [0.1, 0.2, 0.4, 0.9, 0.8],
        ],
    }
    metrics = rescore_semantic_record(record, [0, 2, 4])
    assert metrics["semantic_order_rate"] == 1.0
    assert metrics["m_biou"] == pytest.approx(0.8)


def test_summarize_semantic_trajectory_records():
    records = [
        {
            "matched_prediction_indices": [0, 1],
            "prediction_areas": [10, 20],
            "target_prediction_ious": [[0.8, 0.1], [0.1, 0.8]],
        },
        {
            "matched_prediction_indices": [1, 0],
            "prediction_areas": [20, 10],
            "target_prediction_ious": [[0.1, 0.8], [0.8, 0.1]],
        },
    ]
    summary = summarize_semantic_trajectory_records(records)
    assert summary["samples"] == 2
    assert summary["semantic_order_rate"] == 0.5
    assert summary["backward_area_rate"] == pytest.approx(0.25)
