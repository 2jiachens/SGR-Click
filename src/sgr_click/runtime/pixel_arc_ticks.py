"""Allocate slider intervals in proportion to their predicted pixel changes."""
import copy
import numpy as np


def reallocate(path, variant='pixel_arc'):
    details = dict(variant=variant, semantic_stages=len(path.masks) - 1,
                   original_ticks=path.anchor_ticks.tolist())
    if variant == 'base' or len(path.masks) <= 2:
        details['changed'] = False
        return path, details
    assert variant in ['pixel_arc', 'pixel_arc_semantic']
    gaps = len(path.masks) - 1
    assert 1 <= gaps <= 100
    prefix = 1 if variant == 'pixel_arc_semantic' else 0
    intervals = gaps - prefix
    start = int(path.anchor_ticks[prefix])
    budget = 100 - start
    assert intervals <= budget
    changes = np.array([np.count_nonzero(a ^ b)
                        for a, b in zip(path.masks[prefix:-1], path.masks[prefix+1:])], float)
    weights = changes / changes.sum() if changes.sum() else np.full(intervals, 1 / intervals)
    # Every semantic anchor remains reachable at an integer tick. The rest
    # follows total pixel traversal length; no GT or metric cutoff is used.
    ideal = weights * (budget - intervals)
    extra = np.floor(ideal).astype(int)
    remaining = budget - intervals - int(extra.sum())
    extra[np.argsort(-(ideal - extra), kind='stable')[:remaining]] += 1
    result = copy.copy(path)
    result.transitions = list(path.transitions)
    result.anchor_ticks = np.r_[path.anchor_ticks[:prefix+1], start + np.cumsum(extra + 1)]
    result.anchor_controls = result.anchor_ticks / 100.
    assert result.anchor_ticks[-1] == 100
    assert np.all(np.diff(result.anchor_ticks) >= 1)
    assert all(np.array_equal(a, b) for a, b in zip(path.masks, result.masks))
    details.update(changed=not np.array_equal(path.anchor_ticks, result.anchor_ticks),
                   new_ticks=result.anchor_ticks.tolist(), pixel_changes=changes.astype(int).tolist(),
                   unchanged_prefix_end_tick=start)
    return result, details


def install(controller, holder):
    base = type(controller)

    class PixelArcController(base):
        def _apply_pixel_arc(self):
            if (len(self.clicker) == 1 and self.granularity_ready
                    and self.clicker.get_clicks()[0].is_positive):
                path, details = reallocate(self.granularity_refiner.path, holder['variant'])
                self.granularity_refiner.path = path
                holder['details'] = details
                self.set_granularity(self.requested_granularity)

        def add_click(self, x, y, is_positive):
            result = super().add_click(x, y, is_positive)
            self._apply_pixel_arc()
            return result

        def _finalize_display_path(self):
            result = super()._finalize_display_path()
            self._apply_pixel_arc()
            return result

    controller.__class__ = PixelArcController
    return controller
