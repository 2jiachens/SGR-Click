from __future__ import annotations

import numpy as np

from sgr_click.gui.gui_controller import GranularityRecoveryController


class SoftSemanticFusionController(GranularityRecoveryController):
    """Default SGR-Click controller preserving continuous semantic probability."""

    def add_click(self, x, y, is_positive):
        continuous = getattr(self.granularity_refiner, "continuous", False)
        path = self.granularity_refiner.path if continuous else None
        granularity = self._requested_granularity
        super().add_click(x, y, is_positive)
        if continuous and self.states:
            self.states[-1]["scope_path"] = path
            self.states[-1]["scope_granularity"] = granularity

    def undo_click(self):
        if getattr(self.granularity_refiner, "continuous", False) and self.states:
            previous = self.states[-1]
            self.granularity_refiner.path = previous.get("scope_path")
            self._requested_granularity = previous.get("scope_granularity", 0.0)
        return super().undo_click()

    def _rebuild_hierarchy(self, simple_pred):
        if getattr(self.granularity_refiner, "continuous", False):
            self.granularity_refiner.project_anchor = (
                self._project_boundary if self.enable_boundary_projection else None
            )
        return super()._rebuild_hierarchy(simple_pred)

    def _constrain_hierarchy_mask(
        self, simple_pred, hierarchy_mask, display_granularity=None
    ):
        del simple_pred, display_granularity
        return np.clip(hierarchy_mask.astype(np.float32), 0.0, 1.0)

    def _apply_slider_consistency(self, simple_pred, mask, display_granularity):
        # The logit fusion is already monotonic in granularity.  The legacy
        # binary-mask consistency operation would destroy the soft probabilities.
        del simple_pred, display_granularity
        return np.clip(mask.astype(np.float32), 0.0, 1.0)

    def _project_boundary(self, stage_prob, display_granularity):
        projected = super()._project_boundary(stage_prob, display_granularity)
        self._last_projection_changed_mask = not np.array_equal(
            projected > self.prob_thresh,
            stage_prob > self.prob_thresh,
        )
        return projected

    def _maybe_apply_boundary_projection(self, stage_prob, display_granularity):
        if getattr(self.granularity_refiner, "continuous", False):
            return stage_prob
        projected = super()._maybe_apply_boundary_projection(
            stage_prob, float(display_granularity)
        )

        projected_mask = projected > self.prob_thresh
        return np.where(
            projected_mask,
            np.maximum(projected, self.prob_thresh + 0.03),
            projected * 0.25,
        ).astype(np.float32)
