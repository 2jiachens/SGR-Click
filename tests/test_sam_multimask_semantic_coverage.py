import numpy as np
import pytest

pytest.importorskip("torch")

from evaluate.run_sam_multimask_semantic_coverage import (
    area_sort_masks,
    mapping_indices,
    score_mapping,
)


def test_area_sort_masks_keeps_scores_and_original_indices_aligned():
    masks = [
        np.ones((4, 4), dtype=bool),
        np.eye(4, dtype=bool),
        np.ones((2, 2), dtype=bool),
    ]
    scores = [0.1, 0.2, 0.3]

    sorted_masks, sorted_scores, original_indices = area_sort_masks(masks, scores)

    assert [int(mask.sum()) for mask in sorted_masks] == [4, 4, 16]
    assert sorted_scores == [0.2, 0.3, 0.1]
    assert original_indices == [1, 2, 0]


def test_area_expanded_mapping_repeats_only_native_candidates():
    indices = mapping_indices("area_expanded11", 3)

    assert len(indices) == 11
    assert indices == sorted(indices)
    assert set(indices) == {0, 1, 2}


def test_direct_and_expanded_mappings_have_equal_coverage():
    fine = np.zeros((6, 6), dtype=bool)
    fine[2:4, 2:4] = True
    middle = np.zeros((6, 6), dtype=bool)
    middle[1:5, 1:5] = True
    coarse = np.ones((6, 6), dtype=bool)
    candidates = [fine, middle, coarse]

    direct = score_mapping("direct3", candidates, candidates, (0.60, 0.70))
    expanded = score_mapping(
        "area_expanded11",
        candidates,
        candidates,
        (0.60, 0.70),
    )

    for metric in ("lr60", "hsr60", "lr70", "hsr70", "m_biou", "sor"):
        assert direct[metric] == expanded[metric]
