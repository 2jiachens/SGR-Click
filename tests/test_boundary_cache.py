"""Exact-output checks for final deployment boundary computation reuse."""
import unittest

import numpy as np

from sgr_click.runtime import fast_first_click as fast
from sgr_click.runtime.edge_refinement import refine_boundary as original


class BoundaryCacheTest(unittest.TestCase):
    def test_exact_masks_with_holes_borders_and_click_constraints(self):
        rng = np.random.default_rng(101)
        image = rng.integers(0, 256, (72, 88, 3), dtype=np.uint8)
        yy, xx = np.indices(image.shape[:2])
        masks = [
            (xx-44)**2+(yy-36)**2 < 24**2,
            ((xx-44)**2+(yy-36)**2 < 24**2) & ((xx-44)**2+(yy-36)**2 > 8**2),
            (xx < 39) & (yy < 52),
            (abs(xx-yy) < 3),
            np.zeros((72, 88), dtype=bool),
            np.ones((72, 88), dtype=bool),
        ]
        cache = fast.BoundaryCache(original)
        for mask in masks:
            for radius, strength, sigma in [(5, 4., 12.), (3, 2., 8.)]:
                kwargs = dict(radius=radius, strength=strength, color_sigma=sigma,
                              positive_points=[(40, 35)], negative_points=[(87, 71)],
                              protected_foreground=mask & (xx < 30))
                expected = original(image, mask, **kwargs)
                np.testing.assert_array_equal(cache(image, mask, **kwargs), expected)
                cached = cache(image, mask, **kwargs)
                np.testing.assert_array_equal(cached, expected)
                cached[:] = ~cached
                np.testing.assert_array_equal(cache(image, mask, **kwargs), expected)
        self.assertEqual(cache.hits, len(masks)*4)

    def test_image_changes_and_mutated_masks_do_not_reuse_stale_cuts(self):
        rng = np.random.default_rng(9)
        image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64,64), dtype=bool)
        mask[8:52, 8:48] = True
        cache = fast.BoundaryCache(original)
        cache(image, mask)
        mask[16:24, 16:24] = False
        np.testing.assert_array_equal(cache(image, mask), original(image, mask))
        image2 = image[::-1].copy()
        np.testing.assert_array_equal(cache(image2, mask), original(image2, mask))
        image2[:] = image
        cache.clear()  # controller.set_image invalidates even for the same object
        np.testing.assert_array_equal(cache(image2, mask), original(image2, mask))

    def test_linkage_reuse_preserves_threshold_cuts_and_detects_changed_features(self):
        from isegm.inference.dino_granularity import DinoGranularityRefiner as cls
        source = cls.__new__(cls)
        cache = fast.LinkageCache(source)
        rng = np.random.default_rng(62)
        features = rng.normal(size=(50, 16)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        for tau in [0.8, 0.6, 0.2]:
            expected = cls.__new__(cls)._agglomerative_clustering(features.copy(), tau)
            np.testing.assert_array_equal(cache.cluster(source, features.copy(), tau), expected)
        self.assertEqual(cache.hits, 2)
        changed = features.copy()
        changed[4] = changed[0]
        expected = cls.__new__(cls)._agglomerative_clustering(changed.copy(), .6)
        np.testing.assert_array_equal(cache.cluster(source, changed, .6), expected)
        self.assertEqual(cache.hits, 2)
        cache.clear()
        np.testing.assert_array_equal(cache.cluster(source, features.copy(), .6),
                                      cls.__new__(cls)._agglomerative_clustering(features.copy(), .6))


if __name__ == "__main__":
    unittest.main()
