import numpy as np
import pytest
from isegm.inference.anchored_scope_path import AnchoredScopePath
from sgr_click.runtime.simple_distance_path import apply
from sgr_click.evaluation.noc_protocols import best_tick, threshold_metrics
from sgr_click.evaluation.slider_metrics import slider_diagnostics
from sgr_click.evaluation.cli import strict_chain, subset_samples
from sgr_click.evaluation.semantic_coverage_data import load_paper_samples
from sgr_click.model import ModelPaths


def test_euclidean_transition_endpoints_and_single_updates():
    a = np.zeros((32, 32), bool)
    b = a.copy()
    a[8:20, 5:20] = True
    b[6:27, 9:29] = True
    image = np.zeros((32, 32, 3), np.uint8)
    path = AnchoredScopePath([a, b], image, a.astype(np.float32),
                             merge_contour_duplicates=False)
    path = apply(path, (10, 10))
    masks = np.stack([path.mask_at(t / 100) > 0.5 for t in range(101)])
    np.testing.assert_array_equal(masks[0], a)
    np.testing.assert_array_equal(masks[-1], b)
    assert np.all(masks[:, a & b])
    assert np.all(np.count_nonzero(masks[1:] ^ masks[:-1], axis=0) <= 1)
    for tick, mask in zip(path.anchor_ticks, path.masks):
        np.testing.assert_array_equal(masks[tick], mask)


def test_oracle_tie_and_failed_noc():
    values = np.zeros(101)
    values[21] = values[38] = 0.8
    assert best_tick(values) == 21
    assert threshold_metrics([0.1, 0.86, 0.91]) == {
        "85": {"noc": 2, "reached": True},
        "90": {"noc": 3, "reached": True},
    }
    assert threshold_metrics([0.1, 0.2])["85"] == {"noc": 20, "reached": False}
    with pytest.raises(ValueError):
        best_tick([0.1, 0.2])


def test_plateau_and_jump_diagnostics():
    masks = np.zeros((101, 10, 10), bool)
    masks[:, :2, :2] = True
    masks[50:] = True
    score = slider_diagnostics(masks)
    assert score["jump_destination_ticks"] == [50]
    assert score["jump_step_rate"] == 0.01
    assert score["stagnant_window_rate"] == 91 / 96


def test_sor_excludes_equal_and_nonnested_targets():
    a = np.zeros((10, 10), bool)
    a[2:4, 2:4] = True
    b = a.copy()
    b[4:8, 4:8] = True
    assert strict_chain([a, b])
    assert not strict_chain([a, a])
    assert not strict_chain([b, a])


def test_paper_groups_and_fixed_split():
    samples = load_paper_samples()
    assert {k: len(v) for k, v in samples["partimagenet"].items()} == {
        "head": 2029, "body": 2316, "foot": 1197, "all": 1000,
    }
    rows = samples["spin"]["samples"]
    development = subset_samples("SPIN", rows, "development")
    remaining = subset_samples("SPIN", rows, "remaining")
    assert len(development) == 100 and len(remaining) == 900
    assert not {s["sample_key"] for s in development} & {s["sample_key"] for s in remaining}
    assert development == subset_samples("SPIN", rows[::-1], "development")


def test_missing_weights_fail_before_model_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="SimpleClick"):
        ModelPaths(tmp_path / "missing.pth").validate("lite")
