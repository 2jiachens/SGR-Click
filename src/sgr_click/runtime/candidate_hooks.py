import copy

import cv2

import numpy as np

from scipy import ndimage

from skimage.graph import MCP_Geometric

def install_correction_probe(controller):
    """Change only correction output/state; the first-click path is untouched."""
    from .quality_controller import synchronize_prediction
    base = type(controller)

    class CorrectionProbe(base):
        @property
        def current_object_prob(self):
            if self._probe_correction == 'native' and len(self.clicker) > 1:
                return self.raw_object_prob
            return super().current_object_prob

        def add_click(self, x, y, is_positive):
            state = None
            if self._probe_correction in {'sync', 'native'} and len(self.clicker) and self._init_mask is None:
                state = copy.deepcopy(self.predictor.get_states())
                synchronize_prediction(self.predictor, self.current_object_prob)
            result = super().add_click(x, y, is_positive)
            if state is not None and self.states:
                self.states[-1]['predictor'] = state
            return result

    controller._probe_correction = 'base'
    controller.__class__ = CorrectionProbe
    return controller
