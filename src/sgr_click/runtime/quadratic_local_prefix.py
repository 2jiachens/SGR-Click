"""Spend more of the existing local transition near the user's initial point.

The fixed quadratic curve changes only interpolation to the first semantic
anchor. It preserves pixel order, every anchor, ticks, and all later scopes.
Original single-semantic-scope paths remain exactly unchanged.
"""
import copy
import types


def apply(path):
    if len(path.masks) <= 2 or getattr(path, 'quadratic_local_prefix', False):
        return path, dict(changed=False)
    end = float(path.anchor_controls[1])
    assert end > 0
    original = path.mask_at
    result = copy.copy(path)
    result.transitions = list(path.transitions)

    def mask_at(self, granularity):
        value = float(granularity)
        if 0 < value < end:
            value = end * (value / end) ** 2
        return original(value)

    result.mask_at = types.MethodType(mask_at, result)
    result.quadratic_local_prefix = True
    return result, dict(changed=True, fixed_power=2, prefix_end_tick=int(path.anchor_ticks[1]),
                        anchor_masks_unchanged=True, later_transitions_unchanged=True)


def install(controller, holder):
    base = type(controller)

    class QuadraticLocalController(base):
        def _apply_local_prefix(self):
            if (holder['enabled'] and len(self.clicker)==1 and self.granularity_ready
                    and self.clicker.get_clicks()[0].is_positive):
                path, details = apply(self.granularity_refiner.path)
                self.granularity_refiner.path = path
                holder['details'] = details
                self.set_granularity(self.requested_granularity)

        def add_click(self, x, y, is_positive):
            first = len(self.clicker) == 0 and is_positive and self._init_mask is None
            result = super().add_click(x, y, is_positive)
            if first:
                self._apply_local_prefix()
            return result

        def _finalize_display_path(self):
            result = super()._finalize_display_path()
            # Confidence changes rebuild the display path from its unrefined
            # scopes. Apply the same fixed curve to that newly finalized path.
            self._apply_local_prefix()
            return result

    controller.__class__ = QuadraticLocalController
    return controller
