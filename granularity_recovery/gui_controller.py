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
        base_granularity_predictor=None,
        prob_thresh=0.5,
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

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.granularity_refiner = granularity_refiner
        self.base_granularity_predictor = base_granularity_predictor
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
                if display_granularity <= 0.5:
                    pred = self._apply_base_granularity_model(
                        simple_pred,
                        hierarchy_mask,
                        model_granularity,
                        display_granularity=display_granularity,
                    )
                else:
                    pred = self._apply_recovery_model(
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
            mask = self._apply_slider_monotonicity(mask, self._requested_granularity)
        except Exception as error:
            self.granularity_refiner.reset()
            messagebox.showerror("DINOv3 hierarchy error", str(error))
            return
        current_total, _ = self.probs_history[-1]
        self.probs_history[-1] = (current_total, mask.astype(np.float32))
        self.update_image_callback()

    def set_granularity_detail(self, cluster_tau: float):
        if self.granularity_refiner is None:
            return
        changed = self.granularity_refiner.set_cluster_tau(cluster_tau)
        if not changed or not self._should_use_granularity_refiner() or not self.simple_probs_history:
            return
        self._last_slider_granularity = None
        self._last_slider_mask = None
        self.set_granularity(self._requested_granularity)

    def undo_click(self):
        if not self.states:
            return

        self._last_slider_granularity = None
        self._last_slider_mask = None
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
        )

    def _mask_for_display_granularity(self, simple_pred, display_granularity):
        display_granularity = float(np.clip(display_granularity, 0.0, 1.0))
        hierarchy_granularity, model_granularity = self._split_granularity(display_granularity)
        hierarchy_mask = self.granularity_refiner.mask_at(hierarchy_granularity)
        if display_granularity <= 0.5:
            return self._apply_base_granularity_model(
                simple_pred,
                hierarchy_mask,
                model_granularity,
                display_granularity=display_granularity,
            )
        return self._apply_recovery_model(
            simple_pred,
            hierarchy_mask,
            model_granularity,
            display_granularity=display_granularity,
        )

    def _apply_slider_monotonicity(self, mask, display_granularity):
        mask = mask.astype(np.float32)
        display_granularity = float(display_granularity)
        if (
            self._last_slider_granularity is not None
            and self._last_slider_mask is not None
            and display_granularity >= self._last_slider_granularity - 1e-6
            and self._last_slider_mask.shape == mask.shape
        ):
            mask = np.maximum(mask, self._last_slider_mask.astype(np.float32))
        self._last_slider_granularity = display_granularity
        self._last_slider_mask = mask.copy()
        return mask

    def _should_use_granularity_refiner(self):
        if self.granularity_refiner is None or self._init_mask is not None:
            return False
        clicks = self.clicker.get_clicks()
        return len(clicks) == 1 and clicks[0].is_positive

    def _is_transitioning_from_granularity_to_multiclick(self):
        if self.granularity_refiner is None or not self.granularity_refiner.ready or self._init_mask is not None:
            return False
        clicks = self.clicker.get_clicks()
        return len(clicks) == 1 and clicks[0].is_positive and self.current_object_prob is not None

    def _apply_base_granularity_model(self, simple_pred, hierarchy_mask, granularity, display_granularity=None):
        probability = np.clip(simple_pred.astype(np.float32), 0.0, 1.0)
        positive_clicks = self._positive_clicks()
        first_click = self._first_positive_click()
        if first_click is None:
            return hierarchy_mask.astype(np.float32)
        if self.base_granularity_predictor is None:
            refined = hierarchy_mask.astype(np.float32)
        else:
            refined = self.base_granularity_predictor.predict(
                self.image,
                probability,
                hierarchy_mask.astype(np.float32),
                first_click,
                float(granularity),
            )
        if display_granularity is not None and display_granularity < 0.18:
            anchor_ratio = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
            refined = (
                anchor_ratio * refined.astype(np.float32)
                + (1.0 - anchor_ratio) * hierarchy_mask.astype(np.float32)
            )
        return self._constrain_base_prediction(
            refined,
            probability,
            positive_clicks,
            hierarchy_mask=hierarchy_mask,
            display_granularity=display_granularity,
        )

    def _apply_recovery_model(self, simple_pred, hierarchy_mask, granularity, display_granularity=None):
        return self._apply_base_granularity_model(
            simple_pred,
            hierarchy_mask,
            granularity,
            display_granularity=display_granularity,
        )

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

    def _constrain_base_prediction(self, prediction, probability, positive_clicks, hierarchy_mask=None, display_granularity=None):
        simple_support = self._click_connected_probability_component(probability, positive_clicks, 0.49)
        allowed_support = simple_support
        if hierarchy_mask is not None and display_granularity is not None and display_granularity < 0.18:
            hierarchy_binary = hierarchy_mask > 0.5
            if np.any(hierarchy_binary):
                local = float(np.clip(display_granularity / 0.18, 0.0, 1.0))
                local_core = self._click_local_core(hierarchy_binary, positive_clicks, display_granularity)
                if display_granularity <= 0.10:
                    allowed_support = local_core
                else:
                    kernel_size = max(3, int(round(4 + 6 * local)))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    local_support = cv2.dilate(hierarchy_binary.astype(np.uint8), kernel, iterations=1).astype(bool)
                    allowed_support = np.logical_or(local_support, local_core)
                    allowed_support = self._keep_positive_components(allowed_support, local_support, positive_clicks)
        if not np.any(allowed_support):
            return prediction.astype(np.float32)

        prediction = prediction.astype(np.float32)
        constrained = prediction * allowed_support.astype(np.float32)
        positive_region = constrained > self.prob_thresh
        if np.any(positive_region):
            positive_region = self._keep_positive_components(positive_region, allowed_support, positive_clicks)
            constrained *= positive_region.astype(np.float32)
        return constrained.astype(np.float32)
