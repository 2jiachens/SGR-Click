"""Small, inference-only revisions. No target labels enter this module."""
import copy
import hashlib
import cv2
from collections import OrderedDict
import numpy as np
import torch
from .fast_boundary import crop_solver
from .quality_edge import refine_boundary as confidence_boundary

fast_confidence_boundary = crop_solver(confidence_boundary)


def install_exact_crop():
    from . import boundary as launch_boundary
    if not getattr(launch_boundary.refine_boundary, '_quality_crop', False):
        fast = crop_solver(launch_boundary.refine_boundary)
        fast._quality_crop = True
        launch_boundary.refine_boundary = fast


def synchronize_prediction(predictor, probability):
    """Use the same full-resolution prediction for the mask input and ZoomIn."""
    p = np.asarray(probability, dtype=np.float32)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('Expected a finite H x W probability map')
    predictor.prev_prediction = torch.from_numpy(p.copy()).to(predictor.device)[None, None]
    zoom = predictor.zoom_in
    if zoom is not None:
        zoom._prev_probs = p[None, None].copy()
        # Recompute from the selected scope and *all* user clicks. Reusing the
        # crop from the pre-selection prediction can truncate a new scope.
        zoom._object_roi = None
        zoom._roi_image = None


def insert_prior(masks, probability, click_xy):
    """Retain the native prediction only at a compatible position in the path.

    Never reorder existing anchors or enlarge them by cumulative union. The
    insertion must fit their area order and cannot introduce more scope reversal
    than the pair it replaces. All tests use predictions, never ground truth.
    """
    p = np.asarray(probability) > .5
    x, y = map(int, click_xy)
    if not p[y, x] or not p.any() or len(masks) >= 101:
        return masks, None
    if any(np.array_equal(p, m) for m in masks):
        return masks, next(i for i, m in enumerate(masks) if np.array_equal(p, m))
    candidates = []
    for i, (a, b) in enumerate(zip(masks[:-1], masks[1:])):
        if not a.sum() <= p.sum() <= b.sum():
            continue
        old_reversal = np.count_nonzero(a & ~b)
        new_reversal = np.count_nonzero(a & ~p) + np.count_nonzero(p & ~b)
        if new_reversal > old_reversal:
            continue
        detour = np.count_nonzero(a ^ p) + np.count_nonzero(p ^ b)
        candidates.append((detour, i + 1))
    if not candidates:
        return masks, None
    _, at = min(candidates)
    return masks[:at] + [p.copy()] + masks[at:], at


def revised_controller(base, variant='corrected'):
    if variant not in {'sync', 'native', 'full', 'aligned', 'corrected', 'polarity'}:
        raise ValueError(variant)

    class QualityController(base):
        def __init__(self, *args, **kwargs):
            install_exact_crop()
            self._quality_path_ready = False
            self._quality_edge_cache = OrderedDict()
            self._quality_lab = None
            self._quality_constraints = {}
            super().__init__(*args, **kwargs)

        def set_image(self, image):
            self._quality_path_ready = False
            self._quality_edge_cache = OrderedDict()
            self._quality_lab = None
            self._quality_constraints = {}
            return super().set_image(image)

        def reset_last_object(self, *args, **kwargs):
            self._quality_path_ready = False
            self._quality_edge_cache = OrderedDict()
            self._quality_constraints = {}
            return super().reset_last_object(*args, **kwargs)

        def _constrain_correction(self, probability):
            constraint = self._quality_constraints.get(len(self.clicker)) if variant == 'polarity' else None
            if constraint is None or probability is None:
                return probability
            previous, positive = constraint
            return np.maximum(probability, previous) if positive else np.minimum(probability, previous)

        @property
        def raw_object_prob(self):
            return self._constrain_correction(super().raw_object_prob)

        @property
        def current_object_prob(self):
            # Neural multi-click predictions already contain image-conditioned
            # contour probabilities. Avoid repeatedly shrinking them with a
            # second geometric objective. Initial path remains semantic-guided.
            if variant in {'aligned', 'corrected', 'polarity'} and len(self.clicker) > 1:
                raw = self.raw_object_prob
                if raw is None:
                    return raw
                clicks = tuple((int(c.coords[1]), int(c.coords[0]), bool(c.is_positive)) for c in self.clicker.get_clicks())
                key = (hashlib.sha256(raw.tobytes()).digest(), clicks)
                if key not in self._quality_edge_cache:
                    if self._quality_lab is None:
                        self._quality_lab = cv2.cvtColor(self.image.astype(np.float32)/255., cv2.COLOR_RGB2LAB)
                    mask = raw > self.prob_thresh
                    refined = fast_confidence_boundary(self.image, mask, probability=raw, lab_image=self._quality_lab,
                        positive_points=[(x,y) for x,y,p in clicks if p], negative_points=[(x,y) for x,y,p in clicks if not p],
                        protected_foreground=self.probs_history[-1][0] > self.prob_thresh)
                    result = raw.copy()
                    result[refined & ~mask] = self.prob_thresh + .03
                    result[mask & ~refined] = self.prob_thresh - .03
                    self._quality_edge_cache[key] = self._constrain_correction(result)
                    if len(self._quality_edge_cache)>8:self._quality_edge_cache.popitem(last=False)
                return self._quality_edge_cache[key]
            if variant in {'native', 'full'} and len(self.clicker) > 1:
                return self.raw_object_prob
            if variant == 'full' and self._quality_path_ready and self.granularity_ready:
                return self.raw_object_prob
            return super().current_object_prob

        def _project_boundary(self, stage_prob, display_granularity):
            if variant != 'aligned':
                return super()._project_boundary(stage_prob, display_granularity)
            # Projection has the same state-handoff issue as a user correction:
            # its crop must be derived from the candidate being projected.
            old_state = copy.deepcopy(self.predictor.get_states())
            try:
                synchronize_prediction(self.predictor, stage_prob)
                return super()._project_boundary(stage_prob, display_granularity)
            finally:
                self.predictor.set_states(old_state)

        def add_click(self, x, y, is_positive):
            old_ready = self._quality_path_ready
            previous_state = copy.deepcopy(self.predictor.get_states())
            if len(self.clicker) and self._init_mask is None:
                probability = self.current_object_prob
                if probability is not None:
                    if variant == 'polarity':
                        self._quality_constraints[len(self.clicker)+1] = (probability.copy(), bool(is_positive))
                    synchronize_prediction(self.predictor, probability)
            result = super().add_click(x, y, is_positive)
            if self.states:
                self.states[-1]['predictor'] = previous_state
                self.states[-1]['quality_path_ready'] = old_ready
            if variant == 'full' and len(self.clicker) == 1 and self.granularity_ready:
                self._finish_anchor_path()
            return result

        def _finish_anchor_path(self):
            from isegm.inference.anchored_scope_path import AnchoredScopePath
            refiner = self.granularity_refiner
            old_path = refiner.path
            requested = self.requested_granularity
            masks = []
            # Apply semantic and boundary correction once at semantic anchors.
            # Pixel traversal then connects these finalized scopes; slider
            # queries require no graph cut and remain independent of history.
            for g in old_path.anchor_controls:
                self.set_granularity(float(g))
                masks.append((super().current_object_prob > self.prob_thresh).copy())
            masks, prior_index = insert_prior(masks, self.simple_probs_history[-1], self._first_positive_click())
            path = AnchoredScopePath(masks, self.image, self.simple_probs_history[-1], keep_start_duplicate=True)
            path.has_local_start = True
            path.first_semantic_index = 1
            path.prior_index = prior_index
            refiner.path = path
            self._quality_path_ready = True
            self.set_granularity(requested)

        def undo_click(self):
            if self.states:
                self._quality_path_ready = self.states[-1].get('quality_path_ready', False)
            return super().undo_click()

    QualityController.__name__ = 'QualityController'
    return QualityController


def apply_to_controller(controller, variant):
    if variant == 'baseline':
        return controller
    install_exact_crop()
    controller.__class__ = revised_controller(type(controller), variant)
    controller._quality_path_ready = False
    controller._quality_edge_cache = OrderedDict()
    controller._quality_lab = None
    controller._quality_constraints = {}
    return controller
