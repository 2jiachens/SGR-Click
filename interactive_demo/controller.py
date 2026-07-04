import torch
import numpy as np
import cv2
from tkinter import messagebox

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5,
                 granularity_refiner=None, granularity_refine_predictor=None):
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
        self.granularity_refine_predictor = granularity_refine_predictor
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

    def add_click(self, x, y, is_positive):
        previous_slider_value = self._requested_granularity
        previous_object_prob = None if self.current_object_prob is None else self.current_object_prob.astype(np.float32).copy()
        previous_click_count = len(self.clicker.get_clicks())

        if previous_click_count >= 1:
            self._requested_granularity = 0.0
        granularity = self._requested_granularity

        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
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
                if granularity > 0.0:
                    hierarchy_mask = self.granularity_refiner.mask_at(granularity)
                pred = self._apply_granularity_refine_model(simple_pred, hierarchy_mask, granularity)
                click_context = {
                    'mode': 'granularity',
                    'simple_pred': simple_pred.copy(),
                }
            except Exception as error:
                self._requested_granularity = previous_slider_value
                self.granularity_refiner.reset()
                messagebox.showerror("DINOv3 hierarchy error", str(error))
                return
        elif self.granularity_refiner is not None:
            self.granularity_refiner.reset()
            click_context = {
                'mode': 'simpleclick',
                'simple_pred': simple_pred.copy(),
            }
        else:
            click_context = {
                'mode': 'simpleclick',
                'simple_pred': simple_pred.copy(),
            }

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))
        self.simple_probs_history.append(simple_pred)
        self.click_context_history.append(click_context)

        self.update_image_callback()

    def set_granularity(self, value):
        self._requested_granularity = float(np.clip(value, 0.0, 1.0))
        if not self.probs_history or not self.click_context_history:
            return

        click_context = self.click_context_history[-1]
        if click_context['mode'] == 'granularity':
            if self.granularity_refiner is None or not self.granularity_refiner.ready:
                return
            hierarchy_mask = self.granularity_refiner.mask_at(self._requested_granularity)
            mask = self._apply_granularity_refine_model(
                self.simple_probs_history[-1],
                hierarchy_mask,
                self._requested_granularity,
            )
        else:
            return

        current_total, _ = self.probs_history[-1]
        self.probs_history[-1] = (current_total, mask.astype(np.float32))
        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
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
                mask = self._rebuild_granularity_from_prediction(self.simple_probs_history[-1])
                if granularity > 0.0:
                    mask = self.granularity_refiner.mask_at(granularity)
                mask = self._apply_granularity_refine_model(self.simple_probs_history[-1], mask, granularity)
                current_total, _ = self.probs_history[-1]
                self.probs_history[-1] = (current_total, mask.astype(np.float32))
            except Exception as error:
                self.granularity_refiner.reset()
                messagebox.showerror("DINOv3 hierarchy error", str(error))
        elif self.granularity_refiner is not None:
            self.granularity_refiner.reset()
        self.update_image_callback()

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

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

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

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def requested_granularity(self):
        return self._requested_granularity

    def set_granularity_detail(self, cluster_tau: float):
        if self.granularity_refiner is None:
            return
        changed = self.granularity_refiner.set_cluster_tau(cluster_tau)
        if not changed or not self._should_use_granularity_refiner() or not self.simple_probs_history:
            return
        try:
            hierarchy_mask = self._rebuild_granularity_from_prediction(self.simple_probs_history[-1])
            if self._requested_granularity > 0.0:
                hierarchy_mask = self.granularity_refiner.mask_at(self._requested_granularity)
            mask = self._apply_granularity_refine_model(
                self.simple_probs_history[-1],
                hierarchy_mask,
                self._requested_granularity,
            )
            current_total, _ = self.probs_history[-1]
            self.probs_history[-1] = (current_total, mask.astype(np.float32))
            self.update_image_callback()
        except Exception as error:
            self.granularity_refiner.reset()
            messagebox.showerror("DINOv3 hierarchy error", str(error))

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

    def _negative_clicks(self):
        return [
            (current_click.coords[1], current_click.coords[0])
            for current_click in self.clicker.get_clicks()
            if not current_click.is_positive
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

    def _apply_granularity_refine_model(self, simple_pred, hierarchy_mask, granularity):
        if self.granularity_refine_predictor is None:
            return hierarchy_mask.astype(np.float32)
        probability = np.clip(simple_pred.astype(np.float32), 0.0, 1.0)
        positive_clicks = self._positive_clicks()
        first_click = self._first_positive_click()
        if first_click is None:
            return hierarchy_mask.astype(np.float32)
        refined = self.granularity_refine_predictor.predict(
            self.image,
            probability,
            hierarchy_mask.astype(np.float32),
            first_click,
            float(granularity),
        )
        return self._constrain_prediction_to_simpleclick(refined, probability, positive_clicks)

    @staticmethod
    def _click_connected_probability_component(probability, positive_clicks, threshold):
        if not positive_clicks:
            return probability >= threshold

        height, width = probability.shape
        click_probs = [
            probability[
                int(np.clip(y, 0, height - 1)),
                int(np.clip(x, 0, width - 1)),
            ]
            for x, y in positive_clicks
        ]
        threshold = min(float(threshold), max(0.08, float(max(click_probs)) - 0.04))

        for current_threshold in (threshold, 0.50, 0.40, 0.30, 0.20, 0.10):
            binary = probability >= current_threshold
            labels_count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
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

        labels_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
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

        labels_count, labels = cv2.connectedComponents(fallback_mask.astype(np.uint8), connectivity=8)
        for x, y in positive_clicks:
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            label = int(labels[y, x])
            if label > 0:
                kept |= labels == label
        return kept if np.any(kept) else mask

    def _constrain_prediction_to_simpleclick(self, prediction, probability, positive_clicks):
        support = self._click_connected_probability_component(probability, positive_clicks, 0.49)
        if not np.any(support):
            return prediction.astype(np.float32)

        prediction = prediction.astype(np.float32)
        constrained = prediction * support.astype(np.float32)
        if prediction.ndim == 2:
            positive_region = constrained > self.prob_thresh
            if np.any(positive_region):
                positive_region = self._keep_positive_components(positive_region, support, positive_clicks)
                constrained *= positive_region.astype(np.float32)
        return constrained.astype(np.float32)

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def granularity_ready(self):
        if not self.probs_history or not self.click_context_history:
            return False
        mode = self.click_context_history[-1].get('mode')
        if mode == 'granularity':
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
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
