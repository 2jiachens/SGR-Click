"""Euclidean pixel order, retaining the selected semantic anchors and timing."""
import copy
import numpy as np
from scipy import ndimage


def distance_order(seed, changed, click_xy):
    indices = np.flatnonzero(changed)
    if not len(indices):
        return indices
    if seed.any():
        distance = ndimage.distance_transform_edt(~seed).flat[indices]
    else:
        y, x = np.unravel_index(indices, seed.shape)
        distance = (x-click_xy[0])**2 + (y-click_xy[1])**2
    return indices[np.argsort(distance, kind='stable')]


def apply(path, click_xy):
    from isegm.inference.anchored_scope_path import AnchoredScopePath
    from .quadratic_local_prefix import apply as quadratic
    result = AnchoredScopePath.__new__(AnchoredScopePath)
    for name, value in vars(path).items():
        if name not in ['mask_at', 'quadratic_local_prefix', 'transitions']:
            setattr(result, name, copy.copy(value))
    result.masks = [a.copy() for a in path.masks]
    result.anchor_ticks = path.anchor_ticks.copy()
    result.anchor_controls = path.anchor_controls.copy()
    result.transitions = [(distance_order(a, b & ~a, click_xy),
                           distance_order(b, a & ~b, click_xy)[::-1])
                          for a, b in zip(result.masks[:-1], result.masks[1:])]
    result = quadratic(result)[0]
    result.simple_distance_applied = True
    return result


def install(controller):
    base = type(controller)
    holder = dict(building_first=False)

    class SimpleDistanceController(base):
        def _apply_simple_distance(self):
            if (holder['building_first'] or len(self.clicker) != 1
                    or not self.granularity_ready or not self.clicker.get_clicks()[0].is_positive):
                return
            path = self.granularity_refiner.path
            if getattr(path, 'simple_distance_applied', False):
                return
            y, x = self.clicker.get_clicks()[0].coords
            self.granularity_refiner.path = apply(path, (x, y))
            self.set_granularity(self.requested_granularity)

        def add_click(self, x, y, is_positive):
            first = len(self.clicker) == 0 and is_positive and self._init_mask is None
            if first: holder['building_first'] = True
            try:
                result = super().add_click(x, y, is_positive)
            finally:
                if first: holder['building_first'] = False
            if first: self._apply_simple_distance()
            return result

        def _finalize_display_path(self):
            result = super()._finalize_display_path()
            self._apply_simple_distance()
            return result

    controller.__class__ = SimpleDistanceController
    return controller
