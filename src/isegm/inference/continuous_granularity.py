"""Nested pixel-level scope interpolation; no weights and no slider inference."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric


class ContinuousScopePath:
    """Keep semantic anchor masks and fill each added shell by geodesic growth.

    g measures normalized equivalent radius (sqrt of foreground area), rather
    than log area. All pixels have a fixed arrival value, making queries nested
    and independent of the direction/history of slider interaction.
    """

    def __init__(self, masks, image, probability, area_power=0.5):
        if not 0 < area_power <= 1:
            raise ValueError('area_power must be in (0, 1]')
        probability = np.asarray(probability, dtype=np.float32)
        if probability.ndim != 2 or image.shape[:2] != probability.shape:
            raise ValueError('image and probability shapes must match')
        if not np.isfinite(probability).all():
            raise ValueError('probability must be finite')
        self.shape = probability.shape
        self.area_power = float(area_power)
        self.arrival = np.full(self.shape, np.inf, dtype=np.float32)
        cumulative = np.zeros(self.shape, dtype=bool)
        self.anchors = []
        for mask in masks:
            if np.shape(mask) != self.shape:
                raise ValueError('anchor shapes must match probability')
            current = cumulative | (np.asarray(mask) > 0.5)
            if current.any() and (not self.anchors or not np.array_equal(current, cumulative)):
                self.anchors.append(current)
                cumulative = current
        if not self.anchors:
            raise ValueError('at least one nonempty anchor is required')
        self.areas = np.array([m.sum() for m in self.anchors], dtype=np.int64)
        minimum, maximum = float(self.areas[0]), float(self.areas[-1])
        self.anchor_controls = self._control(self.areas.astype(float), minimum, maximum)
        self.arrival[self.anchors[0]] = 0
        # Image evidence only orders pixels INSIDE each accepted semantic shell;
        # it cannot add background outside the final anchor.
        gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        scale = max(float(np.quantile(gradient, .95)), .02)
        cost = 1 + 2 * np.minimum(gradient / scale, 3) + 2 * (1 - np.clip(probability, 0, 1))
        for previous, current in zip(self.anchors[:-1], self.anchors[1:]):
            shell = current & ~previous
            yy, xx = np.where(shell)
            y0, y1 = max(int(yy.min()) - 1, 0), min(int(yy.max()) + 2, self.shape[0])
            x0, x1 = max(int(xx.min()) - 1, 0), min(int(xx.max()) + 2, self.shape[1])
            region = np.s_[y0:y1, x0:x1]
            local_shell = shell[region]
            seeds = ndimage.binary_dilation(previous, structure=np.ones((3, 3)))[region] & local_shell
            # Disconnected semantic additions are allowed by the source hierarchy.
            # Give each disconnected component its nearest entry, explicitly;
            # do not fabricate a foreground bridge through background.
            labels, count = ndimage.label(local_shell, structure=np.ones((3, 3)))
            seeded = set(np.unique(labels[seeds]))
            missing = [i for i in range(1, count + 1) if i not in seeded]
            if missing:
                distance = ndimage.distance_transform_edt(~previous)[region]
                for label in missing:
                    candidates = np.where(labels == label, distance, np.inf)
                    seeds.flat[int(candidates.argmin())] = True
            costs = np.where(local_shell, cost[region], np.inf)
            distances, _ = MCP_Geometric(costs, fully_connected=True).find_costs(np.argwhere(seeds).tolist())
            local_indices = np.flatnonzero(local_shell)
            order = np.argsort(distances.flat[local_indices], kind='stable')
            ordered = local_indices[order]
            counts = int(previous.sum()) + np.arange(1, len(ordered) + 1)
            values = self._control(counts.astype(float), minimum, maximum)
            local_arrival = self.arrival[region].copy()
            local_arrival.flat[ordered] = values.astype(np.float32)
            self.arrival[region] = local_arrival
        # Keep only the first mask and arrival field; do not retain O(KHW) masks.
        self.root = self.anchors[0].copy()
        del self.anchors

    def _control(self, area, minimum, maximum):
        denominator = maximum ** self.area_power - minimum ** self.area_power
        if denominator <= 0:
            return np.zeros_like(area, dtype=float)
        return (area ** self.area_power - minimum ** self.area_power) / denominator

    def mask_at(self, g):
        if not np.isfinite(g):
            raise ValueError('granularity must be finite')
        return (self.arrival <= np.float32(np.clip(g, 0, 1))).astype(np.float32)


class ContinuousGranularityRefiner:
    """Merge equivalent legacy semantic anchors, with a pixel-level local prefix."""
    continuous = True

    def __init__(self, source, area_power=.5):
        self.source = source
        self.area_power = area_power
        self.project_anchor = None
        self.reset()

    def reset(self):
        self.path = None
        self.source.reset()

    @property
    def ready(self):
        return self.path is not None

    def warmup(self):
        return self.source.warmup()

    def prepare_image(self, image, **kwargs):
        return self.source.prepare_image(image, **kwargs)

    def build(self, image, simple_probability, click_xy, **kwargs):
        self.path = None
        self.source.build(image, simple_probability, click_xy, **kwargs)
        from isegm.inference.anchored_scope_path import AnchoredScopePath, same_object_scope
        import hashlib
        levels, controls, projection_cache = [], [], {}
        # Old controls below 50 are patch-hop growth, not semantic waypoints.
        # Their square boundaries must not constrain the new local prefix.
        # Consolidate equivalent raw scopes BEFORE the expensive neural
        # projection. Keep the one/two-synthetic-click regimes separate and
        # retain the exact soft g=100 input as the last representative.
        candidates=[]
        reference=None
        previous_key=None
        for tick in range(50, 101):
            old_g = tick / 100.
            stage = self.source.mask_at(old_g).astype(np.float32)
            regime = 1 if old_g <= .505 else 2
            key=(regime,hashlib.sha256(stage.tobytes()).digest())
            if (candidates and candidates[-1][2]==regime
                    and (key==previous_key or same_object_scope(reference,stage>.5))):
                candidates[-1]=(stage,old_g,regime)
            else:
                candidates.append((stage,old_g,regime))
                reference=stage>.5
            previous_key=key
        self.projection_candidate_count=len(candidates)
        for stage,old_g,regime in candidates:
            # Keep the semantic display exact: one synthetic point at old 50,
            # two above old 50.5; preserve soft endpoint input at old 100.
            if self.project_anchor is not None and old_g >= .5:
                key = (regime, hashlib.sha256(stage.tobytes()).digest())
                if key not in projection_cache:
                    projection_cache[key] = self.project_anchor(stage, old_g) > .5
                mask = projection_cache[key]
            else:
                mask = stage > .5
            levels.append(mask)
            controls.append(old_g)
        first_semantic = np.asarray(levels[0], dtype=bool)
        x, y = map(int, click_xy)
        if not first_semantic[y, x]:
            # Do not silently insert pixels into a preserved semantic anchor.
            raise ValueError('The first semantic mask does not contain the positive click')
        start = np.zeros(first_semantic.shape, dtype=np.uint8)
        cv2.circle(start, (x, y), 2, 1, thickness=-1)
        start = start.astype(bool) & first_semantic
        if np.array_equal(start, first_semantic) and int(first_semantic.sum()) > 1:
            start[:] = False
            start[y, x] = True
        levels.insert(0, start)
        controls.insert(0, 0.0)
        self.path = AnchoredScopePath(levels, image, simple_probability, controls, keep_start_duplicate=True)
        self.path.has_local_start = True
        self.path.first_semantic_index = 1
        return self.mask_at(0)

    def root_mask(self):
        if not self.ready:
            raise RuntimeError('Build a scope path before querying it')
        return self.path.root.copy()

    def mask_at(self, g):
        if not self.ready:
            raise RuntimeError('Build a scope path before querying it')
        return self.path.mask_at(g)
