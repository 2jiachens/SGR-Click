from __future__ import annotations

import cv2
import numpy as np
import torch
import warnings

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class GranularityRecoveryController:
    """Manage clicks, mask history, semantic paths, and predictor projection."""

    @staticmethod
    def report_error(title, message):
        raise RuntimeError(title + ": " + message)

    @staticmethod
    def report_warning(title, message):
        warnings.warn(title + ": " + message, stacklevel=2)

    def __init__(
        self,
        net,
        device,
        predictor_params,
        update_image_callback,
        granularity_refiner=None,
        prob_thresh: float = 0.5,
        enable_boundary_projection: bool = True,
    ):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.simple_probs_history = []
        self.click_context_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None
        self._requested_granularity = 0.0

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.granularity_refiner = granularity_refiner
        self.enable_boundary_projection = bool(enable_boundary_projection)
        self.reset_predictor()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        if self.granularity_refiner is not None:
            self.granularity_refiner.prepare_image(image)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            self.report_warning(
                "Warning",
                "A segmentation mask must have the same sizes as the current image!",
            )
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = (
            torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        )
        self.clicker.click_indx_offset = 1

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device, **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.simple_probs_history = []
        self.click_context_history = []
        self._requested_granularity = 0.0
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if self.granularity_refiner is not None:
            self.granularity_refiner.reset()
        if update_image:
            self.update_image_callback()

    def add_click(self, x, y, is_positive):
        self.states.append({
            "clicker": self.clicker.get_state(),
            "predictor": self.predictor.get_states(),
        })

        if self._is_transitioning_from_granularity_to_multiclick():
            # Read the selected slider scope before resetting the control.
            # Otherwise the next click starts from the g=0 seed, discarding
            # the mask the user actually selected.
            current_prob = self.current_object_prob.astype(np.float32)
            self.predictor.prev_prediction = (
                torch.from_numpy(current_prob).to(self.device).unsqueeze(0).unsqueeze(0)
            )

        if len(self.clicker.get_clicks()) >= 1:
            self._requested_granularity = 0.0

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        simple_pred = pred.copy()

        if self._should_use_granularity_refiner():
            try:
                hierarchy_mask = self._rebuild_hierarchy(simple_pred)
                if self._requested_granularity > 0.0:
                    hierarchy_mask = self.granularity_refiner.mask_at(
                        self._requested_granularity
                    )
                pred = self._constrain_hierarchy_mask(
                    simple_pred,
                    hierarchy_mask,
                    display_granularity=self._requested_granularity,
                )
                click_context = {"mode": "granularity", "simple_pred": simple_pred.copy()}
            except Exception as error:
                self.granularity_refiner.reset()
                self.report_error("Semantic hierarchy error", str(error))
                return
        elif self.granularity_refiner is not None:
            self.granularity_refiner.reset()
            click_context = {"mode": "simpleclick", "simple_pred": simple_pred.copy()}
        else:
            click_context = {"mode": "simpleclick", "simple_pred": simple_pred.copy()}

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))
        self.simple_probs_history.append(simple_pred)
        self.click_context_history.append(click_context)

        if click_context["mode"] == "granularity":
            pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
            self.probs_history[-1] = (self.probs_history[-1][0], pred)

        self.update_image_callback()

    def set_granularity(self, value):
        self._requested_granularity = float(np.clip(value, 0.0, 1.0))
        if not self.probs_history or not self.click_context_history:
            return

        click_context = self.click_context_history[-1]
        if click_context["mode"] != "granularity":
            return
        if self.granularity_refiner is None or not self.granularity_refiner.ready:
            return

        try:
            mask = self._mask_for_granularity(
                self.simple_probs_history[-1],
                self._requested_granularity,
            )
            mask = self._apply_slider_consistency(
                self.simple_probs_history[-1],
                mask,
                self._requested_granularity,
            )
            if self._requested_granularity >= 0.5:
                mask = self._maybe_apply_boundary_projection(
                    mask, self._requested_granularity
                )
            mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        except Exception as error:
            self.granularity_refiner.reset()
            self.report_error("Semantic hierarchy error", str(error))
            return

        current_total, _ = self.probs_history[-1]
        self.probs_history[-1] = (current_total, mask.astype(np.float32))
        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state["clicker"])
        self.predictor.set_states(prev_state["predictor"])
        self.probs_history.pop()
        if self.simple_probs_history:
            self.simple_probs_history.pop()
        if self.click_context_history:
            self.click_context_history.pop()

        if not self.probs_history:
            self.reset_init_mask()
            if self.granularity_refiner is not None:
                self.granularity_refiner.reset()
            self._requested_granularity = 0.0
        elif self._should_use_granularity_refiner():
            try:
                mask = self._mask_for_granularity(
                    self.simple_probs_history[-1],
                    self._requested_granularity,
                )
                current_total, _ = self.probs_history[-1]
                self.probs_history[-1] = (current_total, mask.astype(np.float32))
            except Exception as error:
                self.granularity_refiner.reset()
                self.report_error("Semantic hierarchy error", str(error))
        elif self.granularity_refiner is not None:
            self.granularity_refiner.reset()

        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return
        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return
        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])
        self.clicker.reset_clicks()
        self.simple_probs_history = []
        self.reset_predictor()
        self.reset_init_mask()
        if self.granularity_refiner is not None:
            self.granularity_refiner.reset()
        self.update_image_callback()

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def requested_granularity(self):
        return self._requested_granularity

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def granularity_ready(self):
        if not self.probs_history or not self.click_context_history:
            return False
        mode = self.click_context_history[-1].get("mode")
        if mode == "granularity":
            return self.granularity_refiner is not None and self.granularity_refiner.ready
        return False

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[
                self.current_object_prob > self.prob_thresh
            ] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(
            self.image,
            mask=results_mask_for_vis,
            alpha=alpha_blend,
            clicks_list=self.clicker.clicks_list,
            radius=click_radius,
        )
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(
                vis, mask=results_mask_for_vis, alpha=alpha_blend,
            )
        return vis

    # ------------------------------------------------------------------
    #  Internal: click helpers
    # ------------------------------------------------------------------

    def _first_positive_click(self):
        for c in self.clicker.get_clicks():
            if c.is_positive:
                y, x = c.coords
                return x, y
        return None

    def _positive_clicks(self):
        return [
            (c.coords[1], c.coords[0])
            for c in self.clicker.get_clicks()
            if c.is_positive
        ]

    # ------------------------------------------------------------------
    #  Internal: DINOv3 hierarchy
    # ------------------------------------------------------------------

    def _should_use_granularity_refiner(self):
        if self.granularity_refiner is None or self._init_mask is not None:
            return False
        clicks = self.clicker.get_clicks()
        return len(clicks) == 1 and clicks[0].is_positive

    def _rebuild_hierarchy(self, simple_pred):
        first_click = self._first_positive_click()
        if first_click is None:
            raise RuntimeError(
                "No positive click is available for DINOv3 hierarchy construction."
            )
        return self.granularity_refiner.build(
            self.image, simple_pred, first_click, semantic_overdrive=True,
        )

    def _is_transitioning_from_granularity_to_multiclick(self):
        if (
            self.granularity_refiner is None
            or not self.granularity_refiner.ready
            or self._init_mask is not None
        ):
            return False
        clicks = self.clicker.get_clicks()
        return (
            len(clicks) == 1
            and clicks[0].is_positive
            and self.current_object_prob is not None
        )

    # ------------------------------------------------------------------
    #  Internal: mask synthesis
    # ------------------------------------------------------------------

    def _mask_for_granularity(self, simple_pred, display_granularity):
        display_granularity = float(np.clip(display_granularity, 0.0, 1.0))
        hierarchy_mask = self.granularity_refiner.mask_at(display_granularity)
        return self._constrain_hierarchy_mask(
            simple_pred, hierarchy_mask, display_granularity=display_granularity,
        )

    def _constrain_hierarchy_mask(self, simple_pred, hierarchy_mask, display_granularity=None):
        simple_prob = np.clip(simple_pred.astype(np.float32), 0.0, 1.0)
        hierarchy_prob = np.clip(hierarchy_mask.astype(np.float32), 0.0, 1.0)
        positive_clicks = self._positive_clicks()
        final_should_expand = self._final_should_expand_beyond_simple(
            simple_prob, hierarchy_prob, positive_clicks,
        )

        # At endpoint where DINOv3 did not actually expand beyond SimpleClick,
        # return the SimpleClick result directly.
        if (
            display_granularity is not None
            and float(display_granularity) >= 0.995
            and not final_should_expand
        ):
            return simple_prob.astype(np.float32)

        constrained = self._constrain_prediction(
            hierarchy_prob, simple_prob, positive_clicks,
            hierarchy_mask=hierarchy_mask,
            display_granularity=display_granularity,
            final_should_expand=final_should_expand,
        )

        # Keep SimpleClick component as a floor at the endpoint.
        if (
            display_granularity is not None
            and float(display_granularity) >= 0.995
            and final_should_expand
        ):
            simple_binary = simple_prob > self.prob_thresh
            if positive_clicks:
                simple_binary = self._keep_positive_components(
                    simple_binary, simple_binary, positive_clicks,
                )
            simple_floor = np.where(
                simple_binary,
                np.maximum(simple_prob, self.prob_thresh + 0.03),
                0.0,
            ).astype(np.float32)
            constrained = np.maximum(constrained.astype(np.float32), simple_floor)

        return constrained

    def _maybe_apply_boundary_projection(self, stage_prob, display_granularity):
        if self.granularity_refiner is None or not self.enable_boundary_projection:
            return stage_prob.astype(np.float32)
        return self._project_boundary(stage_prob, display_granularity)

    # ------------------------------------------------------------------
    #  Internal: slider consistency (UI)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_endpoint(g):
        return g <= 0.005 or g >= 0.995

    def _compute_reference_slider_mask(self, simple_pred, display_granularity):
        key = round(float(np.clip(display_granularity, 0.0, 1.0)), 3)
        return self._mask_for_granularity(simple_pred, key).astype(np.float32)

    def _apply_slider_consistency(self, simple_pred, mask, display_granularity):
        mask = mask.astype(np.float32)
        display_granularity = float(display_granularity)
        if self._is_endpoint(display_granularity):
            return mask

        upper_mask = self._compute_reference_slider_mask(simple_pred, 1.0)
        mask = np.minimum(mask, upper_mask)

        mid_mask = self._compute_reference_slider_mask(simple_pred, 0.5)
        if display_granularity <= 0.5:
            mask = np.minimum(mask, mid_mask)
        else:
            mask = np.maximum(mask, mid_mask)

        return mask

    # ------------------------------------------------------------------
    #  Internal: boundary projection
    # ------------------------------------------------------------------

    def _project_boundary(self, stage_prob, display_granularity):
        """Use frozen SimpleClick with synthetic clicks to refine mask boundaries.

        The hierarchy mask defines the target semantic region but has blocky
        DINOv3-patch edges.  We erode the mask, place synthetic positive clicks
        in the interior, and pass the mask as prev_mask to SimpleClick for a
        single refinement pass.
        """
        target_binary = stage_prob > self.prob_thresh
        first_click = self._first_positive_click()
        if first_click is None or int(target_binary.sum()) < 16:
            return stage_prob.astype(np.float32)

        height, width = target_binary.shape
        first_x = int(np.clip(first_click[0], 0, width - 1))
        first_y = int(np.clip(first_click[1], 0, height - 1))

        eroded = cv2.erode(
            target_binary.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
        if not np.any(eroded):
            eroded = target_binary.copy()

        synthetic_points = []
        available = eroded.astype(np.uint8)
        cv2.circle(available, (first_x, first_y), 8, 0, -1)
        num_synthetic = 1 if display_granularity <= 0.505 else 2
        suppression_radius = max(10, int(round(0.10 * np.hypot(height, width))))
        for _ in range(num_synthetic):
            if not np.any(available):
                break
            dist = cv2.distanceTransform(available, cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
            if dist[y, x] <= 0.0:
                break
            synthetic_points.append((int(x), int(y)))
            cv2.circle(available, (int(x), int(y)), suppression_radius, 0, -1)

        if not synthetic_points:
            return stage_prob.astype(np.float32)

        predictor_state = self.predictor.get_states()
        temp_clicker = clicker.Clicker()
        temp_clicker.add_click(clicker.Click(is_positive=True, coords=(first_y, first_x)))
        for x, y in synthetic_points:
            temp_clicker.add_click(clicker.Click(is_positive=True, coords=(y, x)))

        try:
            prev_mask = (
                torch.from_numpy(stage_prob.astype(np.float32))
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            projected = self.predictor.get_prediction(
                temp_clicker, prev_mask=prev_mask,
            ).astype(np.float32)
        finally:
            self.predictor.set_states(predictor_state)

        allowed_support = self._dilate_mask(target_binary, 11)
        projected_binary = np.logical_and(projected > self.prob_thresh, allowed_support)
        projected_binary = self._keep_positive_components(
            projected_binary, target_binary, [(first_x, first_y)],
        )

        target_area = max(int(target_binary.sum()), 1)
        projected_area = int(projected_binary.sum())
        # Reject implausible projections (too much shrinkage or expansion).
        if projected_area < int(0.55 * target_area) or projected_area > int(1.35 * target_area):
            return stage_prob.astype(np.float32)

        return np.where(
            projected_binary,
            np.maximum(projected, self.prob_thresh + 0.03),
            projected * 0.25,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    #  Internal: constrain prediction to click component
    # ------------------------------------------------------------------

    def _constrain_prediction(
        self,
        prediction,
        probability,
        positive_clicks,
        hierarchy_mask=None,
        display_granularity=None,
        final_should_expand=True,
    ):
        simple_support = self._click_component(probability, positive_clicks, 0.49)
        strict_simple_support = probability > self.prob_thresh
        if positive_clicks:
            strict_simple_support = self._keep_positive_components(
                strict_simple_support, simple_support, positive_clicks,
            )
        if not np.any(strict_simple_support):
            strict_simple_support = simple_support
        allowed_support = simple_support

        if hierarchy_mask is not None and display_granularity is not None:
            hierarchy_binary = hierarchy_mask > 0.5
            if np.any(hierarchy_binary):
                hierarchy_component = self._keep_positive_components(
                    hierarchy_binary, hierarchy_binary, positive_clicks,
                )
                if not np.any(hierarchy_component):
                    hierarchy_component = hierarchy_binary

                if not final_should_expand:
                    if display_granularity < 0.18:
                        local_core = self._click_local_core(
                            strict_simple_support, positive_clicks, display_granularity,
                        )
                        allowed_support = np.logical_and(local_core, strict_simple_support)
                        if not np.any(allowed_support):
                            allowed_support = strict_simple_support
                        allowed_support = self._keep_positive_components(
                            allowed_support, strict_simple_support, positive_clicks,
                        )
                    elif display_granularity < 0.50:
                        allowed_support = strict_simple_support
                        allowed_support = self._keep_positive_components(
                            allowed_support, strict_simple_support, positive_clicks,
                        )
                        allowed_support = np.logical_and(allowed_support, strict_simple_support)
                    else:
                        allowed_support = strict_simple_support
                        allowed_support = self._keep_positive_components(
                            allowed_support, strict_simple_support, positive_clicks,
                        )
                        allowed_support = np.logical_and(allowed_support, strict_simple_support)
                elif display_granularity < 0.18:
                    local = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
                    local_core = self._click_local_core(
                        hierarchy_component, positive_clicks, display_granularity,
                    )
                    simple_local = self._dilate_mask(simple_support, 1 + local)
                    if display_granularity <= 0.14:
                        allowed_support = np.logical_and(local_core, simple_local)
                        if not np.any(allowed_support):
                            allowed_support = local_core
                    else:
                        local_support = self._dilate_mask(hierarchy_component, 1 + local)
                        allowed_support = np.logical_or(
                            np.logical_and(local_support, simple_local), local_core,
                        )
                        allowed_support = self._keep_positive_components(
                            allowed_support, local_support, positive_clicks,
                        )
                elif display_granularity < 0.50:
                    t = float(np.clip((display_granularity - 0.18) / 0.32, 0.0, 1.0))
                    expanded_simple = self._dilate_mask(simple_support, 3 + 2 * t)
                    allowed_support = np.logical_or(expanded_simple, hierarchy_component)
                    allowed_support = self._keep_positive_components(
                        allowed_support, hierarchy_component, positive_clicks,
                    )
                else:
                    allowed_support = hierarchy_component

        if not np.any(allowed_support):
            return prediction.astype(np.float32)

        prediction = prediction.astype(np.float32)
        constrained = prediction * allowed_support.astype(np.float32)
        positive_region = constrained > self.prob_thresh
        if np.any(positive_region):
            positive_region = self._keep_positive_components(
                positive_region, allowed_support, positive_clicks,
            )
            positive_region = self._smooth_visible_region(
                positive_region, allowed_support, positive_clicks,
            )
            positive_region = np.logical_and(positive_region, allowed_support)
            constrained = np.where(
                positive_region,
                np.maximum(constrained, self.prob_thresh + 0.03),
                constrained * 0.35,
            ).astype(np.float32)
        return constrained.astype(np.float32)

    # ------------------------------------------------------------------
    #  Internal: morphology and component utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _click_component(probability, positive_clicks, threshold):
        if not positive_clicks:
            return probability >= threshold

        height, width = probability.shape
        click_probs = [
            probability[
                int(np.clip(y, 0, height - 1)), int(np.clip(x, 0, width - 1))
            ]
            for x, y in positive_clicks
        ]
        threshold = min(float(threshold), max(0.08, float(max(click_probs)) - 0.04))

        for t in (threshold, 0.50, 0.40, 0.30, 0.20, 0.10):
            binary = probability >= t
            _, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
            support = np.zeros_like(binary, dtype=bool)
            for x, y in positive_clicks:
                x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
                label = int(labels[y, x])
                if label > 0:
                    support |= labels == label
            if support.sum() >= 16:
                return support

        seed = np.zeros_like(probability, dtype=np.uint8)
        for x, y in positive_clicks:
            x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
            cv2.circle(seed, (x, y), 3, 1, -1)
        return seed.astype(bool)

    @staticmethod
    def _keep_positive_components(mask, fallback_mask, positive_clicks):
        if not positive_clicks:
            return mask

        _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        kept = np.zeros_like(mask, dtype=bool)
        height, width = mask.shape
        for x, y in positive_clicks:
            x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
            label = int(labels[y, x])
            if label > 0:
                kept |= labels == label
        if np.any(kept):
            return kept

        _, labels = cv2.connectedComponents(fallback_mask.astype(np.uint8), connectivity=8)
        for x, y in positive_clicks:
            x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
            label = int(labels[y, x])
            if label > 0:
                kept |= labels == label
        return kept if np.any(kept) else mask

    @staticmethod
    def _click_local_core(mask, positive_clicks, display_granularity):
        if not positive_clicks:
            return mask.astype(bool)

        mask_uint8 = mask.astype(np.uint8)
        height, width = mask_uint8.shape
        click_x, click_y = positive_clicks[0]
        click_x = int(np.clip(click_x, 0, width - 1))
        click_y = int(np.clip(click_y, 0, height - 1))
        if not mask_uint8[click_y, click_x]:
            repaired = mask_uint8.copy()
            repaired[click_y, click_x] = 1
            mask_uint8 = repaired

        _, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
        label = int(labels[click_y, click_x])
        component = labels == label if label > 0 else mask_uint8.astype(bool)

        local = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
        radius = max(2, int(round(3 + 5 * local)))
        disk = np.zeros_like(mask_uint8, dtype=np.uint8)
        cv2.circle(disk, (click_x, click_y), radius, 1, -1)
        local_core = np.logical_and(component, disk.astype(bool))
        if int(local_core.sum()) < 9:
            dilated = cv2.dilate(
                disk, np.ones((3, 3), dtype=np.uint8), iterations=1,
            ).astype(bool)
            local_core = np.logical_and(component, dilated)
        if int(local_core.sum()) < 9:
            local_core = component
        return local_core.astype(bool)

    @staticmethod
    def _dilate_mask(mask, kernel_size):
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    def _smooth_visible_region(self, binary_mask, fallback_mask, positive_clicks):
        mask_uint8 = binary_mask.astype(np.uint8)
        support_mask = fallback_mask.astype(bool)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, k3, iterations=1)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, k3, iterations=1)
        smoothed = cv2.medianBlur(smoothed * 255, 5) > 127
        smoothed = np.logical_and(smoothed, support_mask)
        smoothed = self._keep_positive_components(smoothed, support_mask, positive_clicks)
        if not np.any(smoothed):
            return binary_mask.astype(bool)
        smoothed = cv2.morphologyEx(
            smoothed.astype(np.uint8), cv2.MORPH_CLOSE, k5, iterations=1,
        ).astype(bool)
        smoothed = np.logical_and(smoothed, support_mask)
        smoothed = self._keep_positive_components(smoothed, support_mask, positive_clicks)
        return smoothed.astype(bool)

    def _expansion_need(self, simple_prob, hierarchy_prob, positive_clicks):
        simple_binary = simple_prob > self.prob_thresh
        hierarchy_binary = hierarchy_prob > self.prob_thresh
        if positive_clicks:
            simple_binary = self._keep_positive_components(
                simple_binary, simple_binary, positive_clicks,
            )
            hierarchy_binary = self._keep_positive_components(
                hierarchy_binary, hierarchy_binary, positive_clicks,
            )

        hierarchy_area = int(hierarchy_binary.sum())
        if hierarchy_area <= 0:
            return 0.0

        simple_area = int(simple_binary.sum())
        intersection = int(np.logical_and(simple_binary, hierarchy_binary).sum())
        union = int(np.logical_or(simple_binary, hierarchy_binary).sum())
        added_area = int(
            np.logical_and(hierarchy_binary, np.logical_not(simple_binary)).sum()
        )

        added_ratio = added_area / max(hierarchy_area, 1)
        overlap_iou = intersection / max(union, 1)
        size_gap = max(0.0, (hierarchy_area - simple_area) / max(hierarchy_area, 1))

        need_from_added = float(np.clip((added_ratio - 0.02) / 0.20, 0.0, 1.0))
        need_from_gap = float(np.clip((size_gap - 0.03) / 0.25, 0.0, 1.0))
        suppress = float(np.clip((overlap_iou - 0.72) / 0.20, 0.0, 1.0))

        return float(np.clip(max(need_from_added, need_from_gap) * (1.0 - 0.85 * suppress), 0.0, 1.0))

    def _final_should_expand_beyond_simple(self, simple_prob, hierarchy_prob, positive_clicks):
        simple_binary = simple_prob > self.prob_thresh
        hierarchy_binary = hierarchy_prob > self.prob_thresh
        if positive_clicks:
            simple_binary = self._keep_positive_components(
                simple_binary, simple_binary, positive_clicks,
            )
            hierarchy_binary = self._keep_positive_components(
                hierarchy_binary, hierarchy_binary, positive_clicks,
            )

        intersection = int(np.logical_and(simple_binary, hierarchy_binary).sum())
        union = int(np.logical_or(simple_binary, hierarchy_binary).sum())
        overlap_iou = intersection / max(union, 1)

        expansion_need = self._expansion_need(simple_prob, hierarchy_prob, positive_clicks)
        added_region = np.logical_and(hierarchy_binary, np.logical_not(simple_binary))
        added_area = int(added_region.sum())
        simple_shell = self._dilate_mask(simple_binary, 9)
        far_added_area = int(
            np.logical_and(added_region, np.logical_not(simple_shell)).sum()
        )
        far_added_ratio = far_added_area / max(added_area, 1)

        if expansion_need < 0.16 and overlap_iou >= 0.78:
            return False
        if expansion_need < 0.08:
            return False
        if far_added_ratio > 0.55 and overlap_iou >= 0.60:
            return False
        return True
