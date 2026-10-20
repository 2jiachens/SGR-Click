"""Preserve finalized binary auxiliary scopes when a confidence cutoff changes."""
import copy
import numpy as np


def install(controller):
    base = type(controller)

    class HardScopeThresholdController(base):
        def add_click(self, x, y, is_positive):
            first = len(self.clicker) == 0 and is_positive and self._init_mask is None
            result = super().add_click(x, y, is_positive)
            if first and self.granularity_ready:
                path = self.granularity_refiner.path
                probabilities = getattr(path, 'display_probabilities', ())
                path.hard_scope_probabilities = bool(probabilities) and all(
                    np.all((probability == 0) | (probability == 1))
                    for probability in probabilities)
            return result

        def _finalize_display_path(self):
            path = self.granularity_refiner.path
            if (len(self.clicker) == 1 and self.granularity_ready
                    and getattr(path, 'display_finalized', False)
                    and getattr(path, 'hard_scope_probabilities', False)
                    and 0 <= self.prob_thresh < 1):
                # These scopes already expose hard 0/1 probabilities. Changing
                # a cutoff below 1 cannot change their binary masks. Refining
                # them again would alter predictions rather than threshold them.
                updated = copy.copy(path)
                updated.display_threshold = self.prob_thresh
                self.granularity_refiner.path = updated
                self.set_granularity(self.requested_granularity)
                return
            return super()._finalize_display_path()

    controller.__class__ = HardScopeThresholdController
    return controller
