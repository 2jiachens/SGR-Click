import numpy as np

from isegm.inference.soft_dino_semantic import fuse_probabilities


def test_similar_low_probability_pixel_is_raised():
    simple = np.asarray([[0.6, 0.3, 0.3]], dtype=np.float32)
    support = np.asarray([[0.9, 0.9, 0.0]], dtype=np.float32)
    corrected = fuse_probabilities(simple, support)

    assert np.isclose(corrected[0, 1], simple[0, 1])
    assert corrected[0, 2] < simple[0, 2]


def test_wider_semantic_support_suppresses_fewer_pixels():
    simple = np.linspace(0.05, 0.95, 20, dtype=np.float32).reshape(4, 5)
    narrow_support = np.zeros((4, 5), dtype=np.float32)
    narrow_support[:, :2] = 1.0
    broad_support = narrow_support.copy()
    broad_support[:, 2:4] = 1.0
    narrow = fuse_probabilities(simple, narrow_support)
    broad = fuse_probabilities(simple, broad_support)

    assert np.all(broad >= narrow)


def test_fusion_stays_inside_probability_range():
    simple = np.asarray([[0.0, 0.1, 0.9, 1.0]], dtype=np.float32)
    support = np.asarray([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    corrected = fuse_probabilities(simple, support)

    assert np.all(corrected > 0.0)
    assert np.all(corrected < 1.0)
