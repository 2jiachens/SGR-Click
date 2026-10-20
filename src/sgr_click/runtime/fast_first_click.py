"""Reuse image affinities and identical boundary cuts without changing masks."""
import ast
import hashlib
import inspect
from collections import OrderedDict
from types import MethodType

import cv2
import numpy as np


class BoundaryCache:
    """Bounded cache for the frozen boundary refiner's pure computations."""

    def __init__(self, original):
        self.image = None
        self.affinities = {}
        self.results = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.signature = inspect.signature(original)
        # Keep the original graph construction and solver exactly. Replace only
        # the image-only Lab conversion and four directional affinity arrays.
        tree = ast.parse(inspect.getsource(original))
        replacements = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == "lab":
                node.value = ast.Constant(None)
                replacements += 1
            elif target.id == "weight":
                node.value = ast.parse(
                    "_cached_weights(image, strength, color_sigma, dy, dx)",
                    mode="eval",
                ).body
                replacements += 1
        if replacements != 2:
            raise RuntimeError("Boundary refiner changed; cache adapter needs review.")
        namespace = dict(original.__globals__, _cached_weights=self.weights)
        exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
        self.compute = namespace[original.__name__]

    def clear(self):
        self.image = None
        self.affinities.clear()
        self.results.clear()

    def weights(self, image, strength, sigma, dy, dx):
        key = (strength, sigma)
        if key not in self.affinities:
            lab = cv2.cvtColor(
                np.clip(image, 0, 255).astype(np.float32) / 255.,
                cv2.COLOR_RGB2LAB,
            )
            h, w = image.shape[:2]
            directions = {}
            for sy, sx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                a = (slice(0, h - sy), slice(max(0, -sx), min(w, w - sx)))
                b = (slice(sy, h), slice(max(0, sx), min(w, w + sx)))
                directions[sy, sx] = (
                    .03 + strength * np.exp(
                        -np.sum((lab[a] - lab[b]) ** 2, axis=2) / (2 * sigma ** 2)
                    )
                ) / np.hypot(sy, sx)
            # Runtime uses one setting; bound memory if callers vary settings.
            self.affinities.clear()
            self.affinities[key] = directions
        return self.affinities[key][dy, dx]

    @staticmethod
    def array_key(value):
        if value is None:
            return None
        array = np.asarray(value, dtype=bool)
        return array.shape, hashlib.sha256(array.tobytes()).digest()

    def __call__(self, image, mask, *args, **kwargs):
        if image is not self.image:
            self.clear()
            self.image = image
        arguments = self.signature.bind(image, mask, *args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        key = (
            self.array_key(mask),
            values["radius"], values["strength"], values["color_sigma"],
            tuple(map(tuple, values["positive_points"])),
            tuple(map(tuple, values["negative_points"])),
            self.array_key(values["protected_foreground"]),
        )
        if key in self.results:
            self.hits += 1
            self.results.move_to_end(key)
            return self.results[key].copy()
        self.misses += 1
        result = self.compute(image, mask, *args, **kwargs)
        self.results[key] = result.copy()
        if len(self.results) > 48:
            self.results.popitem(last=False)
        return result


class LinkageCache:
    """Reuse a fitted linkage tree when rebuilt descriptors are bit-identical."""

    def __init__(self, source):
        self.original = source._agglomerative_clustering
        self.features = None
        self.tree = None
        self.hits = 0

    def clear(self):
        self.features = None
        self.tree = None

    def cluster(self, source, features, tau):
        if getattr(source, "_clustering_features", None) is not features:
            if (self.features is not None and features.dtype == self.features.dtype
                    and np.array_equal(features, self.features)):
                source._clustering_features = features
                source._clustering_tree = self.tree
                self.hits += 1
            result = self.original(features, tau)
            self.features = features.copy()
            self.tree = source._clustering_tree
            return result
        return self.original(features, tau)


def install(controller, source=None):
    """Install after the Euclidean path hook; invalidate on every new image."""
    if getattr(controller, "fast_first_click_installed", False):
        return
    # Each deployed model runs in its own process. The original controller
    # resolves this function through launch_boundary's module globals.
    from . import boundary as launch_boundary
    from . import edge_refinement

    cache = BoundaryCache(edge_refinement.refine_boundary)
    launch_boundary.refine_boundary = cache
    linkages = []
    if source is not None:
        for provider in (source, getattr(source, "_external_trajectory", None)):
            if provider is not None and hasattr(provider, "_agglomerative_clustering"):
                linkage = LinkageCache(provider)
                provider._agglomerative_clustering = MethodType(linkage.cluster, provider)
                linkages.append(linkage)
    base = type(controller)

    class CachedBoundaryController(base):
        def set_image(self, image):
            cache.clear()
            for linkage in linkages:
                linkage.clear()
            return super().set_image(image)

    controller.__class__ = CachedBoundaryController
    controller.boundary_computation_cache = cache
    controller.linkage_computation_caches = linkages
    controller.fast_first_click_installed = True
