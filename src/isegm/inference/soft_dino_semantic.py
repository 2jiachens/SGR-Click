from __future__ import annotations

import heapq
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch

from isegm.inference.dino_granularity import DinoGranularityRefiner

_SOFT_LOGIT_GAIN = 4.5


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def fuse_probabilities(
    simple_probability: np.ndarray,
    semantic_support: np.ndarray,
    logit_gain: float = _SOFT_LOGIT_GAIN,
) -> np.ndarray:
    """Build the unchanged narrow endpoint used by the previous experiment."""
    simple = np.clip(simple_probability.astype(np.float32), 1e-4, 1.0 - 1e-4)
    support = np.clip(semantic_support.astype(np.float32), 0.0, 1.0)
    simple_logit = np.log(simple) - np.log1p(-simple)
    uncertainty = 4.0 * simple * (1.0 - simple)
    correction = np.minimum(support - 0.5, 0.0)
    corrected_logit = simple_logit + float(logit_gain) * uncertainty * correction
    return _sigmoid(corrected_logit).astype(np.float32)


class SoftDinoSemanticRefiner(DinoGranularityRefiner):
    """SGR-Click hierarchy with a soft whole-object semantic endpoint.

    The external scaffold supplies familiar intermediate masks; the custom
    DINOv3 path supplies the high-granularity whole-object result.
    """

    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        device: torch.device,
        model_name: str = "dinov3_vitl16",
        image_size: int = 768,
        positional_debias_components: int = 500,
        positional_debias_mode: str = "zero",
        positional_debias_seed: int = 0,
        prototype_radius: float = 2.5,
        similarity_temperature: float = 0.08,
        expansion_evidence_floor: float = 0.15,
        cluster_tau: float = 0.60,
        soft_logit_gain: float = _SOFT_LOGIT_GAIN,
        enable_connected_support: bool = True,
        enable_soft_endpoint: bool = True,
        frontier_score_mode: str = "full",
    ):
        # Initialize only the state needed by the inherited frozen-feature
        # utilities.  No candidate-mask configuration or state is created.
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.device = device
        self.model_name = model_name
        self.image_size = int(image_size)
        self.positional_debias_components = int(positional_debias_components)
        self.positional_debias_mode = positional_debias_mode
        self.positional_debias_seed = int(positional_debias_seed)
        self.encoder = None
        self._positional_basis_cache = {}
        self._prepared_image_key = None
        self._prepared_semantic_feature = None
        self._external_trajectory = DinoGranularityRefiner(
            repo_path=str(self.repo_path),
            weights_path=str(self.weights_path),
            device=self.device,
            model_name=self.model_name,
            image_size=self.image_size,
            # Match the external launcher's hierarchy scaffold.  The inherited
            # mask convention is used directly; this experiment has no
            # separate candidate-threshold configuration.
            positional_debias_components=self.positional_debias_components,
            positional_debias_mode=self.positional_debias_mode,
            positional_debias_seed=self.positional_debias_seed,
            cluster_tau=float(cluster_tau),
            frontier_score_mode=frontier_score_mode,
        )
        self.reset()
        self.prototype_radius = float(prototype_radius)
        self.similarity_temperature = float(similarity_temperature)
        self.expansion_evidence_floor = float(expansion_evidence_floor)
        self.soft_logit_gain = float(soft_logit_gain)
        self.enable_connected_support = bool(enable_connected_support)
        self.enable_soft_endpoint = bool(enable_soft_endpoint)

    def reset(self):
        self._clustering_features = None
        self._clustering_tree = None
        self._original_size = None
        self._current_granularity = 0.0
        self._simple_probability = None
        self._base_probability = None
        self._base_patch_mask = None
        self._semantic_support = None
        self._patch_arrival = None
        self._final_patch_mask = None
        self._final_target_image = None
        self._primary_image_mask = None
        self._semantic_levels = None
        self._external_ready = False
        self._external_evidence_audit = []

    @property
    def ready(self) -> bool:
        return self._external_ready

    def root_mask(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("The soft DINOv3 semantic field has not been built yet.")
        if not self.ready:
            raise RuntimeError("The external-model trajectory has not been built yet.")
        return self._external_trajectory.root_mask()

    @staticmethod
    def _patch_coordinate(
        click_xy: Tuple[int, int], image_shape: Tuple[int, int], grid_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        x, y = click_xy
        image_h, image_w = image_shape
        grid_h, grid_w = grid_shape
        patch_x = int(np.clip(round(x * grid_w / image_w), 0, grid_w - 1))
        patch_y = int(np.clip(round(y * grid_h / image_h), 0, grid_h - 1))
        return patch_x, patch_y

    def _target_prototype(
        self,
        features: np.ndarray,
        probability_patch: np.ndarray,
        positive_clicks: Iterable[Tuple[int, int]],
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, list[Tuple[int, int]], np.ndarray]:
        grid_h, grid_w, _ = features.shape
        yy, xx = np.mgrid[0:grid_h, 0:grid_w]
        weights = np.zeros((grid_h, grid_w), dtype=np.float32)
        roots = []
        radius_sq = max(self.prototype_radius, 0.5) ** 2

        for click_xy in positive_clicks:
            patch_x, patch_y = self._patch_coordinate(
                click_xy, image_shape, (grid_h, grid_w)
            )
            roots.append((patch_y, patch_x))
            spatial = np.exp(
                -((xx - patch_x) ** 2 + (yy - patch_y) ** 2) / (2.0 * radius_sq)
            ).astype(np.float32)
            # This is deliberately soft: every local probability contributes;
            # there is no probability threshold or candidate mask.
            confidence = 0.15 + 0.85 * np.square(probability_patch)
            weights += spatial * confidence
            weights[patch_y, patch_x] += 2.0

        if not roots:
            raise ValueError("At least one positive click is required.")
        weight_sum = max(float(weights.sum()), 1e-6)
        prototype = (features * weights[..., None]).sum(axis=(0, 1)) / weight_sum
        prototype /= max(float(np.linalg.norm(prototype)), 1e-6)
        return prototype.astype(np.float32), roots, weights

    def _connected_semantic_support(
        self,
        features: np.ndarray,
        similarities: np.ndarray,
        roots: list[Tuple[int, int]],
        positive_weights: np.ndarray,
        probability_patch: np.ndarray,
    ) -> np.ndarray:
        grid_h, grid_w, _ = features.shape
        positive_reference = float(
            (similarities * positive_weights).sum() / max(float(positive_weights.sum()), 1e-6)
        )
        low_probability = probability_patch <= float(np.quantile(probability_patch, 0.35))
        if np.any(low_probability):
            negative_reference = float(np.quantile(similarities[low_probability], 0.75))
        else:
            negative_reference = float(np.quantile(similarities, 0.50))
        center = min(
            positive_reference - 0.03,
            0.5 * (positive_reference + negative_reference),
        )
        temperature = max(self.similarity_temperature, 1e-3)
        node_support = _sigmoid((similarities - center) / temperature).astype(np.float32)

        if not self.enable_connected_support:
            return node_support

        # Widest-path propagation keeps a remote look-alike out unless a chain of
        # mutually similar neighbouring patches connects it to a positive click.
        connected = np.zeros((grid_h, grid_w), dtype=np.float32)
        finalized = np.zeros((grid_h, grid_w), dtype=bool)
        queue: list[Tuple[float, int, int]] = []
        for root_y, root_x in roots:
            connected[root_y, root_x] = 1.0
            heapq.heappush(queue, (-1.0, root_y, root_x))

        while queue:
            negative_score, y, x = heapq.heappop(queue)
            score = -negative_score
            if finalized[y, x]:
                continue
            if score + 1e-8 < float(connected[y, x]):
                continue
            # With a max-priority queue, the first accepted pop is the optimal
            # widest-path value.  Finalizing it bounds work to O(E log V).
            finalized[y, x] = True
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if next_y < 0 or next_y >= grid_h or next_x < 0 or next_x >= grid_w:
                    continue
                if finalized[next_y, next_x]:
                    continue
                edge_cosine = float(np.dot(features[y, x], features[next_y, next_x]))
                edge_support = float(
                    _sigmoid(np.asarray((edge_cosine - center) / temperature))
                )
                path_score = min(score, float(node_support[next_y, next_x]), edge_support)
                if path_score > float(connected[next_y, next_x]) + 1e-8:
                    connected[next_y, next_x] = path_score
                    heapq.heappush(queue, (-path_score, next_y, next_x))

        return np.sqrt(np.clip(node_support * connected, 0.0, 1.0)).astype(np.float32)

    @staticmethod
    def _root_components(
        labels: np.ndarray, roots: list[Tuple[int, int]]
    ) -> np.ndarray:
        selected = np.zeros_like(labels, dtype=bool)
        for root_y, root_x in roots:
            root_cluster = (labels == int(labels[root_y, root_x])).astype(np.uint8)
            _, components = cv2.connectedComponents(root_cluster, connectivity=4)
            selected |= components == components[root_y, root_x]
        return selected

    def _build_multiscale_target(
        self,
        features: np.ndarray,
        roots: list[Tuple[int, int]],
        probability_patch: np.ndarray,
    ) -> Tuple[list[np.ndarray], np.ndarray]:
        grid_h, grid_w, channels = features.shape
        flat_features = features.reshape(grid_h * grid_w, channels)
        tau_schedule = (
            0.80, 0.70, 0.60, 0.50, 0.40, 0.30,
            0.28, 0.26, 0.24, 0.22, 0.20,
        )
        levels: list[np.ndarray] = []
        cumulative = np.zeros((grid_h, grid_w), dtype=bool)
        for tau in tau_schedule:
            labels = self._agglomerative_clustering(flat_features, tau).reshape(
                grid_h, grid_w
            )
            component = self._root_components(labels, roots)
            area_ratio = float(component.mean())
            # Very low tau often collapses every patch into one cluster.  That
            # state contains no object information and must not become g=1.
            if area_ratio >= 0.95:
                break
            added = np.logical_and(component, np.logical_not(cumulative))
            if levels and np.any(added):
                added_probability = float(probability_patch[added].mean())
                # Audit a proposed semantic merge with aggregate soft
                # SimpleClick evidence.  This is not a per-pixel threshold and
                # does not construct or clip to a candidate mask.
                if added_probability < self.expansion_evidence_floor:
                    break
            cumulative = np.logical_or(cumulative, component)
            if not levels or not np.array_equal(levels[-1], cumulative):
                levels.append(cumulative.copy())

        if not levels:
            fallback = np.zeros((grid_h, grid_w), dtype=bool)
            for root_y, root_x in roots:
                fallback[root_y, root_x] = True
            levels = [fallback]
        return levels, levels[-1]

    @staticmethod
    def _arrival_inside_target(
        target: np.ndarray, roots: list[Tuple[int, int]]
    ) -> np.ndarray:
        grid_h, grid_w = target.shape
        arrival = np.full((grid_h, grid_w), np.inf, dtype=np.float32)
        queue: list[Tuple[int, int]] = []
        for root_y, root_x in roots:
            if target[root_y, root_x]:
                arrival[root_y, root_x] = 0.0
                queue.append((root_y, root_x))
        cursor = 0
        while cursor < len(queue):
            y, x = queue[cursor]
            cursor += 1
            next_distance = float(arrival[y, x] + 1.0)
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if next_y < 0 or next_y >= grid_h or next_x < 0 or next_x >= grid_w:
                    continue
                if not target[next_y, next_x] or np.isfinite(arrival[next_y, next_x]):
                    continue
                arrival[next_y, next_x] = next_distance
                queue.append((next_y, next_x))
        return arrival

    def _expansion_support_at(self, granularity: float) -> np.ndarray:
        # Keep the previous click-rooted graph-distance path, but expose it over
        # the complete slider so the intermediate stages no longer disappear.
        local = float(np.clip(granularity, 0.0, 1.0))
        finite = np.isfinite(self._patch_arrival)
        maximum = max(float(self._patch_arrival[finite].max()), 1.0)
        normalized = np.where(finite, self._patch_arrival / maximum, np.inf)
        # A one-patch transition band avoids blocky probability discontinuities.
        transition = max(1.0 / maximum, 1e-3)
        patch_support = np.clip(
            (local + transition - normalized) / (2.0 * transition), 0.0, 1.0
        )
        patch_support *= self._final_patch_mask.astype(np.float32)
        # The zero-endpoint region is already represented by base_probability;
        # only newly reached patches receive expansion evidence.
        patch_support *= 1.0 - self._base_patch_mask.astype(np.float32)
        height, width = self._original_size
        support = cv2.resize(
            patch_support.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(support, 0.0, 1.0).astype(np.float32)

    def _filter_external_tail_levels(
        self,
        trajectory: DinoGranularityRefiner,
        simple_probability: np.ndarray,
    ) -> None:
        """Reject low-evidence semantic additions after the click-root phase.

        The external hierarchy is cumulative.  A wrong component admitted once
        therefore survives through every later slider position.  Audit each new
        connected component with its aggregate soft SimpleClick probability and
        block only components below the existing expansion evidence floor.
        Root levels are kept exactly as generated by the external model.
        """
        raw_levels = trajectory._level_masks
        if raw_levels is None or len(raw_levels) == 0:
            return

        root_count = max(1, min(int(trajectory._root_level_count), len(raw_levels)))
        probability = np.clip(simple_probability.astype(np.float32), 0.0, 1.0)
        accepted = np.zeros_like(raw_levels[0], dtype=bool)
        blocked = np.zeros_like(raw_levels[0], dtype=bool)
        filtered_levels: list[np.ndarray] = []
        filtered_root_count = 0
        audit = []

        for level_index, raw_level in enumerate(raw_levels):
            raw = raw_level.astype(bool)
            if level_index >= root_count:
                proposed = raw & ~accepted & ~blocked
                component_count, component_labels = cv2.connectedComponents(
                    proposed.astype(np.uint8), connectivity=8
                )
                for component_id in range(1, component_count):
                    component = component_labels == component_id
                    area = int(component.sum())
                    if area == 0:
                        continue
                    mean_probability = float(probability[component].mean())
                    keep = mean_probability >= self.expansion_evidence_floor
                    audit.append(
                        {
                            "source_level": level_index,
                            "area": area,
                            "mean_probability": mean_probability,
                            "kept": keep,
                        }
                    )
                    if not keep:
                        blocked |= component

            candidate = accepted | (raw & ~blocked)
            accepted = candidate
            if filtered_levels and np.array_equal(filtered_levels[-1], candidate):
                continue
            filtered_levels.append(candidate.copy())
            if level_index < root_count:
                filtered_root_count = len(filtered_levels)

        trajectory._level_masks = filtered_levels
        trajectory._level_areas = np.asarray(
            [float(mask.sum()) for mask in filtered_levels], dtype=np.float32
        )
        trajectory._level_costs = np.arange(len(filtered_levels), dtype=np.float64)
        trajectory._root_level_count = min(filtered_root_count, len(filtered_levels))
        trajectory._start_level = 0
        self._external_evidence_audit = audit

    def build(
        self,
        image: np.ndarray,
        simple_probability: np.ndarray,
        click_xy: Tuple[int, int],
        positive_clicks: Iterable[Tuple[int, int]] = None,
        negative_clicks: Iterable[Tuple[int, int]] = None,
        candidate_override: np.ndarray = None,
        semantic_overdrive: bool = False,
    ) -> np.ndarray:
        del candidate_override, semantic_overdrive
        self.reset()
        self._original_size = image.shape[:2]
        feature_tensor = self.prepare_image(image)
        features = feature_tensor.permute(1, 2, 0).float().cpu().numpy()
        features /= np.maximum(np.linalg.norm(features, axis=2, keepdims=True), 1e-6)
        grid_h, grid_w = features.shape[:2]

        probability_patch = cv2.resize(
            np.clip(simple_probability.astype(np.float32), 0.0, 1.0),
            (grid_w, grid_h),
            interpolation=cv2.INTER_LINEAR,
        )
        clicks = list(positive_clicks) if positive_clicks is not None else [click_xy]
        prototype, roots, positive_weights = self._target_prototype(
            features, probability_patch, clicks, image.shape[:2]
        )
        similarities = np.einsum("hwc,c->hw", features, prototype).astype(np.float32)
        patch_support = self._connected_semantic_support(
            features, similarities, roots, positive_weights, probability_patch
        )

        self._semantic_levels, self._final_patch_mask = self._build_multiscale_target(
            features, roots, probability_patch
        )
        self._patch_arrival = self._arrival_inside_target(
            self._final_patch_mask, roots
        )

        height, width = image.shape[:2]
        self._final_target_image = cv2.resize(
            self._final_patch_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        self._semantic_support = cv2.resize(
            patch_support, (width, height), interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)
        self._semantic_support = np.clip(self._semantic_support, 0.0, 1.0)
        self._simple_probability = np.clip(
            simple_probability.astype(np.float32), 0.0, 1.0
        )
        self._base_probability = fuse_probabilities(
            self._simple_probability,
            self._semantic_support,
            logit_gain=self.soft_logit_gain,
        )
        self._base_patch_mask = cv2.resize(
            (self._base_probability >= 0.5).astype(np.uint8),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        self._base_patch_mask &= self._final_patch_mask

        # Build the exact external-model level sequence with the same cached
        # DINO tensor, then retain the soft hierarchy only for the final handoff.
        trajectory = self._external_trajectory
        trajectory.encoder = self.encoder
        trajectory._positional_basis_cache = self._positional_basis_cache
        trajectory._prepared_image_key = trajectory._image_cache_key(image)
        trajectory._prepared_semantic_feature = feature_tensor
        trajectory.build(
            image,
            simple_probability,
            click_xy,
            positive_clicks=positive_clicks,
            negative_clicks=negative_clicks,
            semantic_overdrive=True,
        )
        self._filter_external_tail_levels(trajectory, simple_probability)
        # The clicked SimpleClick component is the primary object.  It must be
        # complete before any secondary component in the soft endpoint can
        # appear (dog first, person later in the reported failure case).
        self._primary_image_mask = self._click_component(
            self._simple_probability, click_xy, 0.50
        )
        self._external_ready = True
        self._current_granularity = 0.0
        return self.mask_at(0.0)

    def _soft_probability_at(self, granularity: float) -> np.ndarray:
        if granularity <= 0.0:
            return self._base_probability.copy()
        support = self._expansion_support_at(granularity)
        base = np.clip(self._base_probability, 1e-4, 1.0 - 1e-4)
        base_logit = np.log(base) - np.log1p(-base)
        return _sigmoid(base_logit + self.soft_logit_gain * support).astype(np.float32)

    def mask_at(self, granularity: float) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("The external-model trajectory has not been built yet.")
        self._current_granularity = float(np.clip(granularity, 0.0, 1.0))
        external = self._external_trajectory.mask_at(
            self._current_granularity
        ).astype(np.float32)
        if self._current_granularity < 1.0 or not self.enable_soft_endpoint:
            return external
        primary_floor = self._primary_image_mask.astype(np.float32) * 0.53
        final_support = self._expansion_support_at(1.0)
        added_probability = np.zeros_like(external, dtype=np.float32)
        added_probability[self._final_target_image] = (
            0.53 * final_support[self._final_target_image]
        )
        return np.maximum.reduce(
            (
                external,
                self._soft_probability_at(1.0),
                primary_floor,
                added_probability,
            )
        ).astype(np.float32)

    @property
    def semantic_support(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("The soft DINOv3 semantic field has not been built yet.")
        return self._semantic_support.copy()
