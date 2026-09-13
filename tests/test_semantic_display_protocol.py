import numpy as np
import pytest

from evaluate.semantic_display_protocol import apply_display_protocol


def test_raw_returns_independent_copies():
    masks = [np.array([[1, 0]], dtype=bool), np.array([[0, 1]], dtype=bool)]
    displayed = apply_display_protocol(masks, "raw")
    assert np.array_equal(displayed[0], [[True, False]])
    assert np.array_equal(displayed[1], [[False, True]])
    assert not np.shares_memory(displayed[0], masks[0])


def test_ascending_union_never_removes_pixels():
    masks = [
        np.array([[1, 0, 0]], dtype=bool),
        np.array([[0, 1, 0]], dtype=bool),
        np.array([[0, 0, 1]], dtype=bool),
    ]
    displayed = apply_display_protocol(masks, "ascending_union")
    assert [mask.sum() for mask in displayed] == [1, 2, 3]
    assert np.array_equal(displayed[-1], [[True, True, True]])


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        apply_display_protocol([np.zeros((2, 2)), np.zeros((3, 3))], "raw")
