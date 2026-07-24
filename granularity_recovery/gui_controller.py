from __future__ import annotations

import cv2
import numpy as np
import torch
from tkinter import messagebox

from granularity_recovery.granularity_mapping import split_display_granularity
from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class GranularityRecoveryController:
    def __init__(
        self,
        net,
        device,
        predictor_params,
        update_image_callback,
        granularity_refiner=None,
        granularity_predictor=None,
        prob_thresh=0.5,
        refine_strength=0.65,
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
        self._last_slider_granularity = None
        self._last_slider_mask = None
        self._canonical_slider_masks = {}
        self._display_slider_masks = {}
        self._refined_semantic_stage_cache = {}
        self._last_refined_high_granularity = None
        self._last_refined_high_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.granularity_refiner = granularity_refiner
        self.granularity_predictor = granularity_predictor
        self.refine_strength = float(np.clip(refine_strength, 0.0, 1.25))
        self.reset_predictor()

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
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
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
        self._last_slider_granularity = None
        self._last_slider_mask = None
        self._canonical_slider_masks = {}
        self._display_slider_masks = {}
        self._refined_semantic_stage_cache = {}
        self._last_refined_high_granularity = None
        self._last_refined_high_mask = None
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if self.granularity_refiner is not None:
            self.granularity_refiner.reset()
        if update_image:
            self.update_image_callback()

    def add_click(self, x, y, is_positive):
        self._last_slider_granularity = None
        self._last_slider_mask = None
        self._canonical_slider_masks = {}
        self._display_slider_masks = {}
        self._refined_semantic_stage_cache = {}
        self._last_refined_high_granularity = None
        self._last_refined_high_mask = None
        previous_slider_value = self._requested_granularity
        previous_click_count = len(self.clicker.get_clicks())
        if previous_click_count >= 1:
            self._requested_granularity = 0.0
        display_granularity = self._requested_granularity
        hierarchy_granularity, model_granularity = self._split_granularity(display_granularity)

        self.states.append({
            "clicker": self.clicker.get_state(),
            "predictor": self.predictor.get_states(),
        })

        if self._is_transitioning_from_granularity_to_multiclick():
            current_prob = self.current_object_prob.astype(np.float32)
            self.predictor.prev_prediction = torch.from_numpy(current_prob).to(self.device).unsqueeze(0).unsqueeze(0)

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        simple_pred = pred.copy()
        if self._should_use_granularity_refiner():
            try:
                hierarchy_mask = self._rebuild_granularity_from_prediction(simple_pred)
                if display_granularity > 0.0:
                    hierarchy_mask = self.granularity_refiner.mask_at(hierarchy_granularity)
                pred = self._apply_granularity_model(
                    simple_pred,
                    hierarchy_mask,
                    model_granularity,
                    display_granularity=display_granularity,
                )
                click_context = {
                    "mode": "granularity",
                    "simple_pred": simple_pred.copy(),
                }
            except Exception as error:
                self._requested_granularity = previous_slider_value
                self.granularity_refiner.reset()
                messagebox.showerror("DINOv3 hierarchy error", str(error))
                return
        elif self.granularity_refiner is not None:
            self.granularity_refiner.reset()
            click_context = {
                "mode": "simpleclick",
                "simple_pred": simple_pred.copy(),
            }
        else:
            click_context = {
                "mode": "simpleclick",
                "simple_pred": simple_pred.copy(),
            }

        torch.cuda.empty_cache()
        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))
        self.simple_probs_history.append(simple_pred)
        self.click_context_history.append(click_context)
        if click_context["mode"] == "granularity":
            pred = self._apply_final_slider_monotonicity(pred, display_granularity)
            self.probs_history[-1] = (self.probs_history[-1][0], pred)
            self._last_slider_granularity = display_granularity
            self._last_slider_mask = pred.astype(np.float32).copy()
        self.update_image_callback()

    @staticmethod
    def _split_granularity(display_granularity):
        return split_display_granularity(display_granularity)

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
            mask = self._mask_for_display_granularity(
                self.simple_probs_history[-1],
                self._requested_granularity,
            )
            mask = self._apply_slider_consistency(
                self.simple_probs_history[-1],
                mask,
                self._requested_granularity,
            )
            if 0.495 <= self._requested_granularity <= 1.0:
                mask = self._enhance_high_granularity_display(
                    self.simple_probs_history[-1],
                    mask,
                    self._requested_granularity,
                )
            mask = self._apply_final_slider_monotonicity(
                mask,
                self._requested_granularity,
            )
        except Exception as error:
            self.granularity_refiner.reset()
            messagebox.showerror("DINOv3 hierarchy error", str(error))
            return
        current_total, _ = self.probs_history[-1]
        self.probs_history[-1] = (current_total, mask.astype(np.float32))
        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        self._last_slider_granularity = None
        self._last_slider_mask = None
        self._canonical_slider_masks = {}
        self._display_slider_masks = {}
        self._refined_semantic_stage_cache = {}
        self._last_refined_high_granularity = None
        self._last_refined_high_mask = None
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
            granularity = self._requested_granularity
            try:
                mask = self._mask_for_display_granularity(self.simple_probs_history[-1], granularity)
                current_total, _ = self.probs_history[-1]
                self.probs_history[-1] = (current_total, mask.astype(np.float32))
            except Exception as error:
                self.granularity_refiner.reset()
                messagebox.showerror("DINOv3 hierarchy error", str(error))
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
        self._last_slider_granularity = None
        self._last_slider_mask = None
        self._canonical_slider_masks = {}
        self._display_slider_masks = {}
        self._refined_semantic_stage_cache = {}
        self._last_refined_high_granularity = None
        self._last_refined_high_mask = None
        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])
        self.clicker.reset_clicks()
        self.simple_probs_history = []
        self.reset_predictor()
        self.reset_init_mask()
        if self.granularity_refiner is not None:
            self.granularity_refiner.reset()
        self.update_image_callback()

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
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
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
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)
        return vis

    def _first_positive_click(self):
        for current_click in self.clicker.get_clicks():
            if current_click.is_positive:
                y, x = current_click.coords
                return x, y
        return None

    def _positive_clicks(self):
        return [
            (current_click.coords[1], current_click.coords[0])
            for current_click in self.clicker.get_clicks()
            if current_click.is_positive
        ]

    def _rebuild_granularity_from_prediction(self, simple_pred):
        first_click = self._first_positive_click()
        if first_click is None:
            raise RuntimeError("No positive click is available for DINOv3 hierarchy construction.")
        return self.granularity_refiner.build(
            self.image,
            simple_pred,
            first_click,
            # Keep the clicked root block unchanged, but let the high-granularity
            # tail search beyond the first-click SimpleClick component.
            semantic_overdrive=True,
        )

    def _mask_for_display_granularity(self, simple_pred, display_granularity):
        display_granularity = float(np.clip(display_granularity, 0.0, 1.0))
        hierarchy_granularity, model_granularity = self._split_granularity(display_granularity)
        hierarchy_mask = self.granularity_refiner.mask_at(hierarchy_granularity)
        return self._apply_granularity_model(
            simple_pred,
            hierarchy_mask,
            model_granularity,
            display_granularity=display_granularity,
        )

    @staticmethod
    def _is_canonical_slider_endpoint(display_granularity):
        display_granularity = float(display_granularity)
        return display_granularity <= 0.005 or display_granularity >= 0.995

    def _get_canonical_slider_mask(self, simple_pred, display_granularity):
        key = round(float(np.clip(display_granularity, 0.0, 1.0)), 3)
        if key not in self._canonical_slider_masks:
            self._canonical_slider_masks[key] = self._mask_for_display_granularity(simple_pred, key).astype(np.float32)
        return self._canonical_slider_masks[key].copy()

    def _apply_slider_consistency(self, simple_pred, mask, display_granularity):
        mask = mask.astype(np.float32)
        display_granularity = float(display_granularity)
        if self._is_canonical_slider_endpoint(display_granularity):
            return mask

        upper_mask = self._get_canonical_slider_mask(simple_pred, 1.0)
        mask = np.minimum(mask, upper_mask)

        mid_mask = self._get_canonical_slider_mask(simple_pred, 0.5)
        if display_granularity <= 0.5:
            mask = np.minimum(mask, mid_mask)
        else:
            mask = np.maximum(mask, mid_mask)

        return mask

    def _apply_final_slider_monotonicity(self, mask, display_granularity):
        """Keep final displayed masks nested after all refinement operations."""
        key = round(float(np.clip(display_granularity, 0.0, 1.0)), 3)
        cached = self._display_slider_masks.get(key)
        if cached is not None:
            return cached.copy()

        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        for lower_key in sorted(current for current in self._display_slider_masks if current < key):
            mask = np.maximum(mask, self._display_slider_masks[lower_key])
        for upper_key in sorted(
            (current for current in self._display_slider_masks if current > key),
            reverse=True,
        ):
            mask = np.minimum(mask, self._display_slider_masks[upper_key])

        self._display_slider_masks[key] = mask.copy()
        return mask

    def _should_use_granularity_refiner(self):
        if self.granularity_refiner is None or self._init_mask is not None:
            return False
        clicks = self.clicker.get_clicks()
        return len(clicks) == 1 and clicks[0].is_positive

    def _enhance_high_granularity_display(self, simple_pred, baseline_mask, display_granularity):
        if self.granularity_predictor is None or self.granularity_refiner is None:
            return baseline_mask.astype(np.float32)

        first_click = self._first_positive_click()
        if first_click is None:
            return baseline_mask.astype(np.float32)

        display_granularity = float(np.clip(display_granularity, 0.50, 1.0))
        is_midpoint = abs(display_granularity - 0.50) <= 0.005
        is_endpoint = display_granularity >= 0.995
        hierarchy_granularity, model_granularity = self._split_granularity(display_granularity)
        hierarchy_prob = np.clip(
            self.granularity_refiner.mask_at(hierarchy_granularity).astype(np.float32),
            0.0,
            1.0,
        )
        hierarchy_binary = hierarchy_prob > self.prob_thresh
        simple_prob = np.clip(simple_pred.astype(np.float32), 0.0, 1.0)
        baseline_prob = np.clip(baseline_mask.astype(np.float32), 0.0, 1.0)
        baseline_binary = baseline_prob > self.prob_thresh
        stage_key = np.packbits(baseline_binary.reshape(-1)).tobytes()
        cached_stage = self._refined_semantic_stage_cache.get(stage_key)
        if cached_stage is not None:
            cached_mask = cached_stage[1].copy()
            self._last_refined_high_granularity = display_granularity
            self._last_refined_high_mask = cached_mask.copy()
            return cached_mask

        current_area = int(baseline_binary.sum())
        previous_stage = None
        if (
            self._last_refined_high_granularity is not None
            and self._last_refined_high_mask is not None
            and display_granularity > self._last_refined_high_granularity + 1e-8
        ):
            previous_stage = (
                int((self._last_refined_high_mask > self.prob_thresh).sum()),
                self._last_refined_high_mask.copy(),
            )
        else:
            for previous_binary, previous_prob in self._refined_semantic_stage_cache.values():
                previous_area = int(previous_binary.sum())
                if previous_area <= 0 or previous_area > current_area:
                    continue
                retained_ratio = float(np.logical_and(previous_binary, baseline_binary).sum()) / previous_area
                if retained_ratio >= 0.98 and (
                    previous_stage is None or previous_area > previous_stage[0]
                ):
                    previous_stage = (previous_area, previous_prob)

        if is_midpoint:
            anchor_prob = simple_prob
        elif is_endpoint:
            # At the open-search endpoint, let the trained completion head reject
            # newly exposed background instead of inheriting a hard coarse mask.
            anchor_prob = baseline_prob
        else:
            anchor_prob = np.where(baseline_binary, 0.90, 0.05).astype(np.float32)
            _, model_granularity = self._split_granularity(0.50)
        learned_prob = self.granularity_predictor.predict(
            self.image,
            anchor_prob,
            hierarchy_prob,
            first_click,
            float(model_granularity),
        ).astype(np.float32)

        fixed_core_prob = None
        if not is_midpoint:
            midpoint_baseline = self._get_canonical_slider_mask(simple_pred, 0.50)
            midpoint_key = np.packbits(
                (midpoint_baseline > self.prob_thresh).reshape(-1)
            ).tobytes()
            midpoint_cached = self._refined_semantic_stage_cache.get(midpoint_key)
            if midpoint_cached is None:
                fixed_core_prob = self._enhance_high_granularity_display(
                    simple_pred,
                    midpoint_baseline,
                    0.50,
                )
            else:
                fixed_core_prob = midpoint_cached[1].copy()

        positive_clicks = self._positive_clicks()
        final_should_expand = self._final_should_expand_beyond_simple(
            simple_prob,
            hierarchy_prob,
            positive_clicks,
        )

        progress = float(np.clip((0.50 - 0.18) / 0.82, 0.0, 1.0))
        strength = self.refine_strength * (0.55 + 0.45 * progress)
        if not is_midpoint and not is_endpoint:
            strength = min(1.0, 1.15 * strength)
        delta = learned_prob - anchor_prob
        if is_midpoint:
            reference_prob = hierarchy_prob if final_should_expand else simple_prob
        else:
            reference_prob = baseline_prob
        reference_binary = reference_prob > self.prob_thresh
        boundary_band = self._boundary_band(reference_binary, erosion_kernel_size=5)
        outer_boundary_band = np.logical_and(
            self._dilate_mask(reference_binary, 5),
            np.logical_not(reference_binary),
        )
        positive_scale = np.where(
            boundary_band,
            1.60 * strength,
            np.where(
                reference_binary,
                1.10 * strength,
                np.where(outer_boundary_band, 0.45 * strength, 0.0),
            ),
        ).astype(np.float32)
        negative_scale = min(1.0, 0.95 + 0.05 * strength)
        delta = np.where(delta >= 0.0, positive_scale * delta, negative_scale * delta)
        proposal = np.clip(anchor_prob + delta, 0.0, 1.0)

        support_seed = reference_binary
        if positive_clicks:
            support_seed = self._keep_positive_components(
                support_seed,
                support_seed,
                positive_clicks,
            )
        if not np.any(support_seed):
            return baseline_prob
        allowed_support = self._dilate_mask(support_seed, 5 if is_midpoint else 7)
        allowed_support = self._keep_positive_components(
            allowed_support,
            support_seed,
            positive_clicks,
        )

        constrained = proposal * allowed_support.astype(np.float32)
        visible_region = constrained > self.prob_thresh
        visible_region = self._keep_positive_components(
            visible_region,
            allowed_support,
            positive_clicks,
        )
        visible_region = self._smooth_visible_region(
            visible_region,
            allowed_support,
            positive_clicks,
        )
        if not np.any(visible_region):
            return baseline_prob

        if not is_midpoint:
            smoothed = cv2.medianBlur(visible_region.astype(np.uint8) * 255, 7) > 127
            smoothed = np.logical_and(smoothed, allowed_support)
            smoothed = self._keep_positive_components(
                smoothed,
                allowed_support,
                positive_clicks,
            )
            if np.any(smoothed):
                visible_region = smoothed
        else:
            # The learned head may be overly conservative for a local root
            # block. Preserve that semantic target and use SimpleClick below to
            # project its coarse patch outline onto image boundaries.
            visible_region = np.logical_or(visible_region, hierarchy_binary)
            constrained = np.where(
                hierarchy_binary,
                np.maximum(constrained, self.prob_thresh + 0.03),
                constrained,
            )

        if fixed_core_prob is not None:
            fixed_core_binary = fixed_core_prob > self.prob_thresh
            # The midpoint refinement is immutable. High-granularity stages may
            # only append regions supported by both the completion head and the
            # first-click soft probability; they cannot rewrite the core edge.
            if is_endpoint:
                # Endpoint selection has already stopped at the last coherent
                # semantic stage. Keep that full target for pixel projection.
                accepted_addition = np.logical_and(
                    hierarchy_binary,
                    np.logical_not(fixed_core_binary),
                )
            else:
                high_progress = float(np.clip((display_granularity - 0.50) / 0.50, 0.0, 1.0))
                addition_score = 0.82 * learned_prob + 0.18 * simple_prob
                addition_threshold = 0.50 + 0.04 * high_progress
                accepted_addition = np.logical_and.reduce((
                    visible_region,
                    hierarchy_binary,
                    np.logical_not(fixed_core_binary),
                    addition_score >= addition_threshold,
                ))
            connection_seed = self._dilate_mask(fixed_core_binary, 5)
            connected_growth = np.logical_or(connection_seed, accepted_addition)
            connected_growth = self._keep_positive_components(
                connected_growth,
                connection_seed,
                positive_clicks,
            )
            accepted_addition = np.logical_and(accepted_addition, connected_growth)
            visible_region = np.logical_or(fixed_core_binary, accepted_addition)
            constrained = np.where(fixed_core_binary, fixed_core_prob, constrained)

        if previous_stage is not None:
            previous_prob = previous_stage[1]
            locked_region = previous_prob > self.prob_thresh
            visible_region = np.logical_or(visible_region, locked_region)
            constrained = np.where(locked_region, previous_prob, constrained)

        result = np.where(
            visible_region,
            np.maximum(constrained, self.prob_thresh + 0.03),
            constrained * 0.35,
        ).astype(np.float32)
        result = self._project_semantic_stage_with_simpleclick(
            result,
            display_granularity,
        )
        self._refined_semantic_stage_cache[stage_key] = (
            baseline_binary.copy(),
            result.copy(),
        )
        self._last_refined_high_granularity = display_granularity
        self._last_refined_high_mask = result.copy()
        return result

    def _project_semantic_stage_with_simpleclick(self, stage_prob, display_granularity):
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
        num_synthetic_clicks = 1 if display_granularity <= 0.505 else 2
        suppression_radius = max(10, int(round(0.10 * np.hypot(height, width))))
        for _ in range(num_synthetic_clicks):
            if not np.any(available):
                break
            distance_to_edge = cv2.distanceTransform(available, cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(np.argmax(distance_to_edge)), distance_to_edge.shape)
            if distance_to_edge[y, x] <= 0.0:
                break
            synthetic_points.append((int(x), int(y)))
            cv2.circle(available, (int(x), int(y)), suppression_radius, 0, -1)

        if not synthetic_points:
            return stage_prob.astype(np.float32)

        predictor_state = self.predictor.get_states()
        temporary_clicker = clicker.Clicker()
        temporary_clicker.add_click(clicker.Click(is_positive=True, coords=(first_y, first_x)))
        for x, y in synthetic_points:
            temporary_clicker.add_click(clicker.Click(is_positive=True, coords=(y, x)))

        try:
            prev_mask = torch.from_numpy(stage_prob.astype(np.float32)).to(self.device).unsqueeze(0).unsqueeze(0)
            projected = self.predictor.get_prediction(temporary_clicker, prev_mask=prev_mask).astype(np.float32)
        finally:
            self.predictor.set_states(predictor_state)

        allowed_support = self._dilate_mask(target_binary, 11)
        projected_binary = np.logical_and(projected > self.prob_thresh, allowed_support)
        projected_binary = self._keep_positive_components(
            projected_binary,
            target_binary,
            [(first_x, first_y)],
        )
        target_area = max(int(target_binary.sum()), 1)
        projected_area = int(projected_binary.sum())
        if projected_area < int(0.55 * target_area) or projected_area > int(1.35 * target_area):
            return stage_prob.astype(np.float32)

        return np.where(
            projected_binary,
            np.maximum(projected, self.prob_thresh + 0.03),
            projected * 0.25,
        ).astype(np.float32)

    def _is_transitioning_from_granularity_to_multiclick(self):
        if self.granularity_refiner is None or not self.granularity_refiner.ready or self._init_mask is not None:
            return False
        clicks = self.clicker.get_clicks()
        return len(clicks) == 1 and clicks[0].is_positive and self.current_object_prob is not None

    def _apply_granularity_model(self, simple_pred, hierarchy_mask, granularity, display_granularity=None):
        simple_prob = np.clip(simple_pred.astype(np.float32), 0.0, 1.0)
        hierarchy_prob = np.clip(hierarchy_mask.astype(np.float32), 0.0, 1.0)
        positive_clicks = self._positive_clicks()
        first_click = self._first_positive_click()
        if first_click is None:
            return hierarchy_prob
        final_should_expand = self._final_should_expand_beyond_simple(
            simple_prob,
            hierarchy_prob,
            positive_clicks,
        )
        if self.granularity_predictor is None:
            refined = hierarchy_prob
        else:
            anchor_prob = simple_prob
            if display_granularity is not None:
                display_granularity = float(np.clip(display_granularity, 0.0, 1.0))
                if display_granularity < 0.18:
                    anchor_prob = simple_prob
                elif display_granularity < 0.50:
                    transition = float(np.clip((display_granularity - 0.18) / 0.32, 0.0, 1.0))
                    anchor_prob = np.clip(
                        (1.0 - transition) * simple_prob + transition * hierarchy_prob,
                        0.0,
                        1.0,
                    )
                else:
                    transition = float(np.clip((display_granularity - 0.50) / 0.50, 0.0, 1.0))
                    hierarchy_weight = 0.88 + 0.08 * transition
                    anchor_prob = np.clip(
                        (1.0 - hierarchy_weight) * simple_prob + hierarchy_weight * hierarchy_prob,
                        0.0,
                        1.0,
                    )
            refined = self.granularity_predictor.predict(
                self.image,
                anchor_prob,
                hierarchy_prob,
                first_click,
                float(granularity),
            )
            if display_granularity is not None and display_granularity >= 0.18:
                strength = self._refine_strength_at(display_granularity)
                delta = refined.astype(np.float32) - anchor_prob
                hierarchy_binary = hierarchy_prob > self.prob_thresh
                boundary_band = self._boundary_band(hierarchy_binary, erosion_kernel_size=5)
                inside_object = hierarchy_binary
                positive_scale_map = np.where(
                    boundary_band,
                    1.35 * strength,
                    np.where(
                        inside_object,
                        0.55 * strength,
                        0.0,
                    ),
                ).astype(np.float32)
                negative_scale = min(1.0, 0.95 + 0.05 * strength)
                delta = np.where(delta >= 0.0, positive_scale_map * delta, negative_scale * delta)
                refined = np.clip(anchor_prob + delta, 0.0, 1.0)
        if display_granularity is not None and display_granularity < 0.18:
            anchor_ratio = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
            refined = (
                anchor_ratio * refined.astype(np.float32)
                + (1.0 - anchor_ratio) * hierarchy_prob
            )
        if (
            display_granularity is not None
            and float(display_granularity) >= 0.995
            and not final_should_expand
        ):
            return simple_prob.astype(np.float32)
        constrained = self._constrain_granularity_prediction(
            refined,
            simple_prob,
            positive_clicks,
            hierarchy_mask=hierarchy_mask,
            display_granularity=display_granularity,
            final_should_expand=final_should_expand,
        )
        if (
            display_granularity is not None
            and float(display_granularity) >= 0.995
            and final_should_expand
        ):
            simple_binary = simple_prob > self.prob_thresh
            if positive_clicks:
                simple_binary = self._keep_positive_components(simple_binary, simple_binary, positive_clicks)
            simple_floor = np.where(
                simple_binary,
                np.maximum(simple_prob, self.prob_thresh + 0.03),
                0.0,
            ).astype(np.float32)
            constrained = np.maximum(constrained.astype(np.float32), simple_floor)
        return constrained

    @staticmethod
    def _click_connected_probability_component(probability, positive_clicks, threshold):
        if not positive_clicks:
            return probability >= threshold

        height, width = probability.shape
        click_probs = [
            probability[int(np.clip(y, 0, height - 1)), int(np.clip(x, 0, width - 1))]
            for x, y in positive_clicks
        ]
        threshold = min(float(threshold), max(0.08, float(max(click_probs)) - 0.04))

        for current_threshold in (threshold, 0.50, 0.40, 0.30, 0.20, 0.10):
            binary = probability >= current_threshold
            _, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
            support = np.zeros_like(binary, dtype=bool)
            for x, y in positive_clicks:
                x = int(np.clip(x, 0, width - 1))
                y = int(np.clip(y, 0, height - 1))
                label = int(labels[y, x])
                if label > 0:
                    support |= labels == label
            if support.sum() >= 16:
                return support

        seed = np.zeros_like(probability, dtype=np.uint8)
        for x, y in positive_clicks:
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
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
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            label = int(labels[y, x])
            if label > 0:
                kept |= labels == label

        if np.any(kept):
            return kept

        _, labels = cv2.connectedComponents(fallback_mask.astype(np.uint8), connectivity=8)
        for x, y in positive_clicks:
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
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
            dilated = cv2.dilate(disk, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
            local_core = np.logical_and(component, dilated)
        if int(local_core.sum()) < 9:
            local_core = component
        return local_core.astype(bool)

    @staticmethod
    def _dilate_mask(mask, kernel_size):
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    @staticmethod
    def _boundary_band(mask, erosion_kernel_size=5):
        mask_uint8 = mask.astype(np.uint8)
        kernel_size = max(1, int(erosion_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask_uint8, kernel, iterations=1).astype(bool)
        return np.logical_and(mask.astype(bool), np.logical_not(eroded))

    def _smooth_visible_region(self, binary_mask, fallback_mask, positive_clicks):
        mask_uint8 = binary_mask.astype(np.uint8)
        support_mask = fallback_mask.astype(bool)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel3, iterations=1)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel3, iterations=1)
        smoothed = cv2.medianBlur(smoothed * 255, 5) > 127
        smoothed = np.logical_and(smoothed, support_mask)
        smoothed = self._keep_positive_components(smoothed, support_mask, positive_clicks)
        if not np.any(smoothed):
            return binary_mask.astype(bool)
        smoothed = cv2.morphologyEx(smoothed.astype(np.uint8), cv2.MORPH_CLOSE, kernel5, iterations=1).astype(bool)
        smoothed = np.logical_and(smoothed, support_mask)
        smoothed = self._keep_positive_components(smoothed, support_mask, positive_clicks)
        return smoothed.astype(bool)

    def _expansion_need(self, simple_prob, hierarchy_prob, positive_clicks):
        simple_binary = simple_prob > self.prob_thresh
        hierarchy_binary = hierarchy_prob > self.prob_thresh

        if positive_clicks:
            simple_binary = self._keep_positive_components(simple_binary, simple_binary, positive_clicks)
            hierarchy_binary = self._keep_positive_components(hierarchy_binary, hierarchy_binary, positive_clicks)

        hierarchy_area = int(hierarchy_binary.sum())
        if hierarchy_area <= 0:
            return 0.0

        simple_area = int(simple_binary.sum())
        intersection = int(np.logical_and(simple_binary, hierarchy_binary).sum())
        union = int(np.logical_or(simple_binary, hierarchy_binary).sum())
        added_area = int(np.logical_and(hierarchy_binary, np.logical_not(simple_binary)).sum())

        added_ratio = added_area / max(hierarchy_area, 1)
        overlap_iou = intersection / max(union, 1)
        size_gap = max(0.0, (hierarchy_area - simple_area) / max(hierarchy_area, 1))

        need_from_added = float(np.clip((added_ratio - 0.02) / 0.20, 0.0, 1.0))
        need_from_gap = float(np.clip((size_gap - 0.03) / 0.25, 0.0, 1.0))
        suppress_from_agreement = float(np.clip((overlap_iou - 0.72) / 0.20, 0.0, 1.0))

        expansion_need = max(need_from_added, need_from_gap) * (1.0 - 0.85 * suppress_from_agreement)
        return float(np.clip(expansion_need, 0.0, 1.0))

    def _final_should_expand_beyond_simple(self, simple_prob, hierarchy_prob, positive_clicks):
        simple_binary = simple_prob > self.prob_thresh
        hierarchy_binary = hierarchy_prob > self.prob_thresh
        if positive_clicks:
            simple_binary = self._keep_positive_components(simple_binary, simple_binary, positive_clicks)
            hierarchy_binary = self._keep_positive_components(hierarchy_binary, hierarchy_binary, positive_clicks)

        intersection = int(np.logical_and(simple_binary, hierarchy_binary).sum())
        union = int(np.logical_or(simple_binary, hierarchy_binary).sum())
        overlap_iou = intersection / max(union, 1)
        expansion_need = self._expansion_need(simple_prob, hierarchy_prob, positive_clicks)
        added_region = np.logical_and(hierarchy_binary, np.logical_not(simple_binary))
        added_area = int(added_region.sum())
        simple_shell = self._dilate_mask(simple_binary, 9)
        far_added_area = int(np.logical_and(added_region, np.logical_not(simple_shell)).sum())
        far_added_ratio = far_added_area / max(added_area, 1)

        if expansion_need < 0.16 and overlap_iou >= 0.78:
            return False
        if expansion_need < 0.08:
            return False
        if far_added_ratio > 0.55 and overlap_iou >= 0.60:
            return False
        return True

    def _should_expand_beyond_simple(self, simple_prob, hierarchy_prob, positive_clicks, display_granularity):
        if display_granularity is None or float(display_granularity) < 0.995:
            return True
        return self._final_should_expand_beyond_simple(simple_prob, hierarchy_prob, positive_clicks)

    def _refine_strength_at(self, display_granularity):
        if display_granularity is None:
            return self.refine_strength
        display_granularity = float(np.clip(display_granularity, 0.0, 1.0))
        if display_granularity < 0.18:
            return 0.0
        progress = float(np.clip((display_granularity - 0.18) / 0.82, 0.0, 1.0))
        return self.refine_strength * (0.35 + 0.65 * progress)

    def _constrain_granularity_prediction(
        self,
        prediction,
        probability,
        positive_clicks,
        hierarchy_mask=None,
        display_granularity=None,
        final_should_expand=True,
    ):
        simple_support = self._click_connected_probability_component(probability, positive_clicks, 0.49)
        strict_simple_support = probability > self.prob_thresh
        if positive_clicks:
            strict_simple_support = self._keep_positive_components(
                strict_simple_support,
                simple_support,
                positive_clicks,
            )
        if not np.any(strict_simple_support):
            strict_simple_support = simple_support
        allowed_support = simple_support
        if hierarchy_mask is not None and display_granularity is not None:
            hierarchy_binary = hierarchy_mask > 0.5
            if np.any(hierarchy_binary):
                hierarchy_component = self._keep_positive_components(hierarchy_binary, hierarchy_binary, positive_clicks)
                if not np.any(hierarchy_component):
                    hierarchy_component = hierarchy_binary
                if not final_should_expand:
                    if display_granularity < 0.18:
                        local_core = self._click_local_core(
                            strict_simple_support,
                            positive_clicks,
                            display_granularity,
                        )
                        allowed_support = np.logical_and(local_core, strict_simple_support)
                        if not np.any(allowed_support):
                            allowed_support = strict_simple_support
                        allowed_support = self._keep_positive_components(allowed_support, strict_simple_support, positive_clicks)
                    elif display_granularity < 0.50:
                        allowed_support = strict_simple_support
                        allowed_support = self._keep_positive_components(allowed_support, strict_simple_support, positive_clicks)
                    else:
                        allowed_support = strict_simple_support
                        allowed_support = self._keep_positive_components(allowed_support, strict_simple_support, positive_clicks)
                    allowed_support = np.logical_and(allowed_support, strict_simple_support)
                elif display_granularity < 0.18:
                    local = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
                    local_core = self._click_local_core(hierarchy_component, positive_clicks, display_granularity)
                    simple_local = self._dilate_mask(simple_support, 1 + local)
                    if display_granularity <= 0.14:
                        allowed_support = np.logical_and(local_core, simple_local)
                        if not np.any(allowed_support):
                            allowed_support = local_core
                    else:
                        local_support = self._dilate_mask(hierarchy_component, 1 + local)
                        allowed_support = np.logical_or(np.logical_and(local_support, simple_local), local_core)
                        allowed_support = self._keep_positive_components(allowed_support, local_support, positive_clicks)
                elif display_granularity < 0.50:
                    transition = float(np.clip((display_granularity - 0.18) / 0.32, 0.0, 1.0))
                    expanded_simple = self._dilate_mask(simple_support, 3 + 2 * transition)
                    allowed_support = np.logical_or(expanded_simple, hierarchy_component)
                    allowed_support = self._keep_positive_components(allowed_support, hierarchy_component, positive_clicks)
                else:
                    allowed_support = hierarchy_component
        if not np.any(allowed_support):
            return prediction.astype(np.float32)

        prediction = prediction.astype(np.float32)
        constrained = prediction * allowed_support.astype(np.float32)
        positive_region = constrained > self.prob_thresh
        if np.any(positive_region):
            positive_region = self._keep_positive_components(positive_region, allowed_support, positive_clicks)
            positive_region = self._smooth_visible_region(positive_region, allowed_support, positive_clicks)
            positive_region = np.logical_and(positive_region, allowed_support)
            constrained = np.where(
                positive_region,
                np.maximum(constrained, self.prob_thresh + 0.03),
                constrained * 0.35,
            ).astype(np.float32)
        return constrained.astype(np.float32)
