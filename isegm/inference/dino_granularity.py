from __future__ import annotations

import heapq
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic
from sklearn.cluster import AgglomerativeClustering


class DinoGranularityRefiner:
    """Build a frozen-DINOv3, click-rooted hierarchy from one SimpleClick mask."""

    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        device: torch.device,
        model_name: str = "dinov3_vitl16",
        image_size: int = 768,
        candidate_threshold: float = 0.50,
        positional_debias_components: int = 500,
        positional_debias_mode: str = "zero",
        positional_debias_seed: int = 0,
        cluster_tau: float = 0.60,
    ):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.device = device
        self.model_name = model_name
        self.image_size = image_size
        self.candidate_threshold = candidate_threshold
        self.positional_debias_components = positional_debias_components
        if positional_debias_mode not in {"none", "zero", "random"}:
            raise ValueError(
                "positional_debias_mode must be one of: none, zero, random"
            )
        self.positional_debias_mode = positional_debias_mode
        self.positional_debias_seed = int(positional_debias_seed)
        self.cluster_tau = cluster_tau

        self.encoder = None
        self._positional_basis_cache = {}
        self._prepared_image_key = None
        self._prepared_semantic_feature = None
        self.capture_frontier_scores = False
        self.reset()

    def reset(self):
        self._labels = None
        self._candidate = None
        self._original_candidate = None
        self._arrival = None
        self._primary_arrival = None
        self._graph_hops = None
        self._level_costs = None
        self._level_areas = None
        self._level_masks = None
        self._root_level_count = 0
        self._start_level = 0
        self._original_size = None
        self._current_granularity = 0.0
        self._semantic_component_map = None
        self._semantic_component_features = None
        self._semantic_component_centroids = None
        self._semantic_component_neighbors = None
        self._semantic_image_size = None
        self._frontier_score_trace = []

    @staticmethod
    def _image_cache_key(image: np.ndarray) -> Tuple[int, Tuple[int, ...], str]:
        return id(image), tuple(image.shape), str(image.dtype)

    def prepare_image(self, image: np.ndarray, force: bool = False) -> torch.Tensor:
        image_key = self._image_cache_key(image)
        if force or self._prepared_semantic_feature is None or self._prepared_image_key != image_key:
            self._prepared_semantic_feature = self._extract_clustering_semantic_feature(image)
            self._prepared_image_key = image_key
        return self._prepared_semantic_feature

    def set_cluster_tau(self, tau: float) -> bool:
        tau = float(tau)
        if abs(self.cluster_tau - tau) < 1e-8:
            return False
        self.cluster_tau = tau
        return True

    @property
    def ready(self) -> bool:
        has_levels = self._level_costs is not None and len(self._level_costs) > 0
        has_masks = self._level_masks is not None and len(self._level_masks) > 0
        return has_levels or has_masks

    @property
    def current_granularity(self) -> float:
        return self._current_granularity

    def root_mask(self) -> np.ndarray:
        if not self.ready or self._level_masks is None or len(self._level_masks) == 0:
            raise RuntimeError("The DINOv3 hierarchy has not been built yet.")
        root_level_count = max(1, min(int(self._root_level_count), len(self._level_masks)))
        return self._level_masks[root_level_count - 1].copy()

    def _load_encoder(self):
        if self.encoder is not None:
            return
        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"DINOv3 repository not found: {self.repo_path}")
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {self.weights_path}")

        self.encoder = torch.hub.load(
            str(self.repo_path),
            self.model_name,
            source="local",
            weights=str(self.weights_path),
        )
        self.encoder.eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def warmup(self):
        """Load DINOv3 and precompute image-independent startup state."""
        self._load_encoder()
        channels = getattr(self.encoder, "embed_dim", None)
        if channels is None:
            patch_embed = getattr(self.encoder, "patch_embed", None)
            projection = getattr(patch_embed, "proj", None)
            channels = getattr(projection, "out_channels", None)
        if channels is not None:
            self._build_positional_basis(
                self.image_size,
                self.image_size,
                int(channels),
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def _build_positional_basis(self, input_height: int, input_width: int, channels: int) -> torch.Tensor:
        cache_key = (
            input_height,
            input_width,
            channels,
            self.positional_debias_mode,
            self.positional_debias_seed,
        )
        if cache_key in self._positional_basis_cache:
            return self._positional_basis_cache[cache_key]

        if self.positional_debias_mode == "none":
            basis = torch.empty(channels, 0, device=self.device, dtype=torch.float32)
            self._positional_basis_cache[cache_key] = basis
            return basis

        if self.positional_debias_mode == "zero":
            probe_image = torch.zeros(
                1, 3, input_height, input_width,
                device=self.device, dtype=torch.float32,
            )
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.positional_debias_seed)
            probe_image = torch.rand(
                1, 3, input_height, input_width,
                device=self.device, dtype=torch.float32,
                generator=generator,
            )
        mean = probe_image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = probe_image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        probe_image = (probe_image - mean) / std

        probe_feature = self.encoder.get_intermediate_layers(
            probe_image,
            n=[11],
            reshape=True,
        )[0][0]
        probe_feature = F.normalize(probe_feature, p=2, dim=0)

        max_components = min(
            self.positional_debias_components,
            channels,
            probe_feature.shape[1] * probe_feature.shape[2],
        )
        if max_components <= 0:
            basis = torch.empty(channels, 0, device=self.device, dtype=probe_feature.dtype)
        else:
            matrix = probe_feature.reshape(channels, -1)
            matrix = matrix - matrix.mean(dim=1, keepdim=True)
            basis, _, _ = torch.linalg.svd(matrix, full_matrices=False)
            basis = basis[:, :max_components].contiguous()

        self._positional_basis_cache[cache_key] = basis
        return basis

    @staticmethod
    def _debias_feature_map(feature_map: torch.Tensor, positional_basis: torch.Tensor) -> torch.Tensor:
        channels, height, width = feature_map.shape
        if positional_basis.numel() == 0:
            return F.normalize(feature_map, p=2, dim=0)

        flat = feature_map.reshape(channels, height * width)
        basis = positional_basis.to(flat.device, dtype=flat.dtype)
        projection_perp = torch.eye(channels, device=flat.device, dtype=flat.dtype) - basis @ basis.T
        debiased = projection_perp @ flat
        return F.normalize(debiased.reshape(channels, height, width), p=2, dim=0)

    @torch.inference_mode()
    def _extract_features(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        self._load_encoder()
        height, width = image.shape[:2]
        scale = self.image_size / max(height, width)
        input_height = max(16, int(round(height * scale / 16)) * 16)
        input_width = max(16, int(round(width * scale / 16)) * 16)

        tensor = torch.from_numpy(image).to(self.device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = F.interpolate(tensor, (input_height, input_width), mode="bilinear", align_corners=False)
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        early, middle, semantic = self.encoder.get_intermediate_layers(
            tensor,
            n=[3, 7, 11],
            reshape=True,
        )
        channels = int(semantic[0].shape[0])
        positional_basis = self._build_positional_basis(input_height, input_width, channels)

        early_map = self._debias_feature_map(F.normalize(early[0], dim=0), positional_basis)
        middle_map = self._debias_feature_map(F.normalize(middle[0], dim=0), positional_basis)
        semantic_map = self._debias_feature_map(F.normalize(semantic[0], dim=0), positional_basis)
        fine = F.normalize(0.5 * early_map + 0.5 * middle_map, dim=0)
        return fine, semantic_map

    @torch.inference_mode()
    def _extract_clustering_semantic_feature(self, image: np.ndarray) -> torch.Tensor:
        # Match scripts/cluster_dinov3_images.py as closely as possible so the
        # interactive root semantic block follows the same coarse segmentation.
        self._load_encoder()
        height, width = image.shape[:2]
        scale = self.image_size / max(height, width)
        input_height = max(16, int(round(height * scale / 16)) * 16)
        input_width = max(16, int(round(width * scale / 16)) * 16)

        tensor = torch.from_numpy(image).to(self.device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = F.interpolate(tensor, (input_height, input_width), mode="bilinear", align_corners=False)
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        semantic = self.encoder.get_intermediate_layers(tensor, n=1, reshape=True)[0][0]
        semantic = F.normalize(semantic, dim=0)
        positional_basis = self._build_positional_basis(self.image_size, self.image_size, int(semantic.shape[0]))
        return self._debias_feature_map(semantic, positional_basis)

    @staticmethod
    def _resize_preserving_aspect(image: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, size, interpolation=interpolation)

    @staticmethod
    def _click_component(probability: np.ndarray, click_xy: Tuple[int, int], threshold: float) -> np.ndarray:
        x, y = click_xy
        x = int(np.clip(x, 0, probability.shape[1] - 1))
        y = int(np.clip(y, 0, probability.shape[0] - 1))

        for current_threshold in (threshold, 0.10, 0.05):
            binary = (probability >= current_threshold).astype(np.uint8)
            binary[y, x] = 1
            _, components = cv2.connectedComponents(binary, connectivity=8)
            component = components == components[y, x]
            if component.sum() >= 16:
                return component

        fallback = np.zeros_like(probability, dtype=np.uint8)
        cv2.circle(fallback, (x, y), 6, 1, -1)
        return fallback.astype(bool)

    @staticmethod
    def _region_means(values: np.ndarray, labels: np.ndarray, num_regions: int) -> np.ndarray:
        flat_labels = labels.reshape(-1)
        counts = np.bincount(flat_labels, minlength=num_regions).clip(min=1).astype(np.float32)
        flat_values = values.reshape(-1, values.shape[-1])
        means = np.empty((num_regions, flat_values.shape[1]), dtype=np.float32)
        for channel in range(flat_values.shape[1]):
            means[:, channel] = np.bincount(
                flat_labels,
                weights=flat_values[:, channel],
                minlength=num_regions,
            ) / counts
        return means

    @staticmethod
    def _feature_region_means(
        feature_map: torch.Tensor,
        labels: np.ndarray,
        num_regions: int,
    ) -> np.ndarray:
        work_height, work_width = labels.shape
        resized = F.interpolate(
            feature_map.unsqueeze(0),
            (work_height, work_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        flat_features = resized.permute(1, 2, 0).reshape(-1, resized.shape[0])
        flat_labels = torch.from_numpy(labels.reshape(-1)).to(feature_map.device, dtype=torch.long)
        sums = torch.zeros(
            (num_regions, flat_features.shape[1]),
            dtype=flat_features.dtype,
            device=feature_map.device,
        )
        sums.index_add_(0, flat_labels, flat_features)
        counts = torch.bincount(flat_labels, minlength=num_regions).clamp_min(1).unsqueeze(1)
        means = F.normalize(sums / counts, dim=1)
        return means.float().cpu().numpy()

    @staticmethod
    def _adjacent_pairs(labels: np.ndarray, gradient: np.ndarray) -> Dict[Tuple[int, int], float]:
        pair_sums: Dict[Tuple[int, int], float] = {}
        pair_counts: Dict[Tuple[int, int], int] = {}

        def collect(left: np.ndarray, right: np.ndarray, boundary: np.ndarray):
            changed = left != right
            first = left[changed]
            second = right[changed]
            values = boundary[changed]
            for a, b, value in zip(first.tolist(), second.tolist(), values.tolist()):
                pair = (a, b) if a < b else (b, a)
                pair_sums[pair] = pair_sums.get(pair, 0.0) + float(value)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        collect(labels[:, :-1], labels[:, 1:], gradient[:, 1:])
        collect(labels[:-1, :], labels[1:, :], gradient[1:, :])
        return {pair: pair_sums[pair] / pair_counts[pair] for pair in pair_sums}

    @staticmethod
    def _additive_arrival(
        roots: Iterable[int],
        active: np.ndarray,
        edges: Iterable[Tuple[int, int, float]],
        num_regions: int,
    ) -> np.ndarray:
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_regions)]
        for first, second, cost in edges:
            if active[first] and active[second]:
                adjacency[first].append((second, cost))
                adjacency[second].append((first, cost))

        arrival = np.full(num_regions, np.inf, dtype=np.float64)
        roots = sorted(set(int(root) for root in roots if active[int(root)]))
        for root in roots:
            arrival[root] = 0.0
        queue = [(0.0, root) for root in roots]
        heapq.heapify(queue)
        while queue:
            current_cost, current = heapq.heappop(queue)
            if current_cost > arrival[current] + 1e-12:
                continue
            for neighbor, edge_cost in adjacency[current]:
                new_cost = current_cost + edge_cost
                if new_cost + 1e-12 < arrival[neighbor]:
                    arrival[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor))
        return arrival

    @staticmethod
    def _shortest_graph_hops(
        roots: Iterable[int],
        active: np.ndarray,
        edges: Iterable[Tuple[int, int, float]],
        num_regions: int,
    ) -> np.ndarray:
        adjacency: List[List[int]] = [[] for _ in range(num_regions)]
        for first, second, _ in edges:
            if active[first] and active[second]:
                adjacency[first].append(second)
                adjacency[second].append(first)

        hops = np.full(num_regions, np.inf, dtype=np.float64)
        queue = sorted(set(int(root) for root in roots))
        for root in queue:
            hops[root] = 0.0
        index = 0
        while index < len(queue):
            current = queue[index]
            index += 1
            for neighbor in adjacency[current]:
                if not np.isfinite(hops[neighbor]):
                    hops[neighbor] = hops[current] + 1.0
                    queue.append(neighbor)
        return hops

    @staticmethod
    def _first_stable_level(
        level_costs: np.ndarray,
        level_areas: np.ndarray,
    ) -> int:
        if len(level_costs) <= 2:
            return 0
        total_area = max(float(level_areas[-1]), 1.0)
        minimum_area = max(16.0, total_area * 0.0005)
        minimum_index = int(np.searchsorted(level_areas, minimum_area, side="left"))

        gaps = np.diff(level_costs)
        local_gaps = gaps[minimum_index:]
        median = float(np.median(local_gaps))
        mad = float(np.median(np.abs(local_gaps - median)))
        jump_threshold = median + 1.5 * max(mad, 1e-6)
        for index in range(minimum_index, len(gaps)):
            if gaps[index] > jump_threshold:
                return index
        return min(minimum_index, len(level_costs) - 1)

    @staticmethod
    def _semantic_kmeans(features: np.ndarray, num_clusters: int, num_iters: int = 12) -> np.ndarray:
        num_points = features.shape[0]
        if num_points <= num_clusters:
            return np.arange(num_points, dtype=np.int32)

        norms = np.linalg.norm(features, axis=1)
        first = int(np.argmax(norms))
        centers = [features[first]]
        distances = np.full(num_points, np.inf, dtype=np.float32)
        for _ in range(1, num_clusters):
            current = centers[-1]
            current_dist = 1.0 - features.dot(current)
            distances = np.minimum(distances, current_dist)
            next_index = int(np.argmax(distances))
            centers.append(features[next_index])
        centers = np.stack(centers, axis=0).astype(np.float32)

        assignments = np.zeros(num_points, dtype=np.int32)
        for _ in range(num_iters):
            similarity = features @ centers.T
            assignments = np.argmax(similarity, axis=1).astype(np.int32)
            for cluster_index in range(num_clusters):
                members = features[assignments == cluster_index]
                if len(members) == 0:
                    continue
                center = members.mean(axis=0)
                centers[cluster_index] = center / max(np.linalg.norm(center), 1e-6)
        return assignments

    @staticmethod
    def _component_mask(mask: np.ndarray, y: int, x: int) -> np.ndarray:
        if not mask[y, x]:
            repaired = mask.copy()
            repaired[y, x] = True
            mask = repaired
        _, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        component_id = components[y, x]
        return components == component_id

    @staticmethod
    def _proposal_score(
        cluster_index: int,
        root_cluster: int,
        cluster_features: np.ndarray,
        cluster_areas: np.ndarray,
        root_area: float,
        root_feature: np.ndarray,
        root_centroid: np.ndarray,
        cluster_centroids: np.ndarray,
        cluster_boundary_ratio: np.ndarray,
        cluster_touch_ratio: np.ndarray,
    ) -> Tuple[float, float, float, float, int]:
        semantic_distance = 1.0 - float(np.dot(cluster_features[cluster_index], root_feature))
        area_ratio = abs(np.log(max(float(cluster_areas[cluster_index]), 1.0) / max(root_area, 1.0)))
        centroid_distance = float(np.linalg.norm(cluster_centroids[cluster_index] - root_centroid))
        boundary_penalty = float(cluster_boundary_ratio[cluster_index])
        touch_bonus = -float(cluster_touch_ratio[cluster_index])
        if cluster_index == root_cluster:
            return (-1.0, 0.0, 0.0, 0.0, cluster_index)
        return (
            semantic_distance + 0.12 * centroid_distance,
            area_ratio,
            boundary_penalty,
            touch_bonus,
            cluster_index,
        )

    @staticmethod
    def _upsample_patch_mask(
        patch_mask: np.ndarray,
        output_size: Tuple[int, int],
        simple_probability: np.ndarray,
        candidate_mask: np.ndarray,
        relax_to_probability: bool = True,
        hard_clip_candidate: bool = True,
    ) -> np.ndarray:
        width, height = output_size
        del simple_probability, relax_to_probability
        candidate = candidate_mask.astype(bool)

        # Preserve the semantic selection from the patch grid, then smooth only
        # a thin image-space boundary band so we reduce blockiness without changing
        # which semantic region the slider picked.
        hard_mask = cv2.resize(
            patch_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        soft_mask = cv2.resize(
            patch_mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(hard_mask.astype(np.uint8), kernel, iterations=1) > 0
        eroded = cv2.erode(hard_mask.astype(np.uint8), kernel, iterations=1) > 0
        boundary_band = dilated ^ eroded

        up_mask = hard_mask.copy()
        up_mask[boundary_band] = soft_mask[boundary_band] >= 0.50

        up_mask = cv2.morphologyEx(up_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
        if hard_clip_candidate:
            up_mask &= candidate
        return up_mask

    @staticmethod
    def _agglomerative_clustering(features: np.ndarray, tau: float) -> np.ndarray:
        similarity = np.clip(features @ features.T, -1.0, 1.0)
        distance = (1.0 - similarity).astype(np.float32)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=float(1.0 - tau),
        )
        return clustering.fit_predict(distance).astype(np.int32)

    @staticmethod
    def _enforce_monotonic_levels(level_masks: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        if not level_masks:
            return [], np.empty((0,), dtype=np.float32)

        monotonic_masks: List[np.ndarray] = []
        level_areas: List[float] = []
        cumulative = np.zeros_like(level_masks[0], dtype=bool)
        for mask in level_masks:
            cumulative = np.logical_or(cumulative, mask.astype(bool))
            if monotonic_masks and np.array_equal(monotonic_masks[-1], cumulative):
                continue
            monotonic_masks.append(cumulative.copy())
            level_areas.append(float(cumulative.sum()))
        return monotonic_masks, np.asarray(level_areas, dtype=np.float32)

    def propose_semantic_clicks(
        self,
        current_mask: np.ndarray,
        hierarchy_mask: np.ndarray,
        max_clicks: int = 1,
        min_missing_pixels: int = 2,
    ) -> List[Tuple[int, int]]:
        if (
            self._semantic_component_map is None
            or self._semantic_component_features is None
            or self._semantic_component_centroids is None
            or self._semantic_image_size is None
        ):
            return []

        image_height, image_width = self._semantic_image_size
        if current_mask.shape != (image_height, image_width) or hierarchy_mask.shape != (image_height, image_width):
            return []

        component_map = self._semantic_component_map
        grid_h, grid_w = component_map.shape
        current_patch = cv2.resize(
            current_mask.astype(np.uint8),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        hierarchy_patch = cv2.resize(
            hierarchy_mask.astype(np.uint8),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        missing_patch = hierarchy_patch & np.logical_not(current_patch)
        if not np.any(missing_patch):
            return []

        valid = component_map >= 0
        num_components = int(self._semantic_component_features.shape[0])
        flat_components = component_map[valid].reshape(-1)
        current_counts = np.bincount(
            flat_components,
            weights=current_patch[valid].reshape(-1).astype(np.float32),
            minlength=num_components,
        )
        hierarchy_counts = np.bincount(
            flat_components,
            weights=hierarchy_patch[valid].reshape(-1).astype(np.float32),
            minlength=num_components,
        )
        missing_counts = np.bincount(
            flat_components,
            weights=missing_patch[valid].reshape(-1).astype(np.float32),
            minlength=num_components,
        )

        current_component_indices = np.flatnonzero(current_counts > 0)
        if len(current_component_indices) == 0:
            return []
        current_dilated = cv2.dilate(current_patch.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        near_current_patch = current_dilated & np.logical_not(current_patch)
        current_distance = cv2.distanceTransform(np.logical_not(current_patch).astype(np.uint8), cv2.DIST_L2, 5)
        weights = current_counts[current_component_indices].astype(np.float32)
        prototype = np.average(self._semantic_component_features[current_component_indices], axis=0, weights=weights)
        prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-6)

        current_set = set(int(index) for index in current_component_indices.tolist())
        neighbor_sets = self._semantic_component_neighbors or {}
        candidate_indices = [
            int(index)
            for index in np.flatnonzero(missing_counts >= float(min_missing_pixels)).tolist()
            if hierarchy_counts[int(index)] > 0
        ]
        if not candidate_indices:
            return []

        current_centroid = np.average(
            self._semantic_component_centroids[current_component_indices],
            axis=0,
            weights=weights,
        )
        scored = []
        containing_scored = []
        touching_scored = []
        for component_index in candidate_indices:
            feature = self._semantic_component_features[component_index]
            similarity = float(np.dot(feature, prototype))
            missing_ratio = float(missing_counts[component_index] / max(hierarchy_counts[component_index], 1.0))
            current_ratio = float(current_counts[component_index] / max(hierarchy_counts[component_index], 1.0))
            is_touching = any(neighbor in current_set for neighbor in neighbor_sets.get(component_index, ()))
            component_missing = (component_map == component_index) & missing_patch
            touch_pixels = int(np.logical_and(component_missing, near_current_patch).sum())
            touch_ratio = float(touch_pixels / max(float(component_missing.sum()), 1.0))
            component_distances = current_distance[component_missing]
            nearest_distance = float(component_distances.min()) if component_distances.size else 1e3
            mean_distance = float(component_distances.mean()) if component_distances.size else 1e3
            centroid_distance = float(np.linalg.norm(self._semantic_component_centroids[component_index] - current_centroid))
            score = (
                0.42 * touch_ratio
                + 0.24 * (1.0 if is_touching else 0.0)
                + 0.18 * similarity
                + 0.10 * missing_ratio
                - 0.12 * np.log1p(nearest_distance)
                - 0.04 * np.log1p(mean_distance)
                - 0.05 * centroid_distance
            )
            is_containing = current_counts[component_index] > 0
            if is_containing:
                score = (
                    1.00
                    + 0.38 * current_ratio
                    + 0.24 * missing_ratio
                    + 0.18 * similarity
                    - 0.10 * np.log1p(mean_distance)
                )
            row = (
                score,
                1 if is_containing else 0,
                touch_ratio,
                similarity,
                -nearest_distance,
                missing_counts[component_index],
                component_index,
            )
            scored.append(row)
            if is_containing:
                containing_scored.append(row)
            if is_touching or touch_pixels > 0:
                touching_scored.append(row)
        if containing_scored:
            scored = containing_scored
        elif touching_scored:
            scored = touching_scored
        scored.sort(reverse=True)

        clicks: List[Tuple[int, int]] = []
        missing_image = hierarchy_mask.astype(bool) & np.logical_not(current_mask.astype(bool))
        for row in scored[: max(int(max_clicks) * 3, 1)]:
            is_containing = bool(row[1])
            component_index = int(row[-1])
            component_missing = (component_map == component_index) & missing_patch
            if not np.any(component_missing):
                continue
            component_region = (component_map == component_index) & hierarchy_patch
            if is_containing and np.any(component_region):
                region_ys, region_xs = np.where(component_region)
                center_y = float(region_ys.mean())
                center_x = float(region_xs.mean())
                missing_ys, missing_xs = np.where(component_missing)
                nearest = int(np.argmin((missing_xs - center_x) ** 2 + (missing_ys - center_y) ** 2))
                grid_y = float(missing_ys[nearest])
                grid_x = float(missing_xs[nearest])
            else:
                contact = component_missing & near_current_patch
                if np.any(contact):
                    contact_dist = current_distance.copy()
                    contact_dist[~contact] = np.inf
                    grid_y, grid_x = np.unravel_index(int(np.argmin(contact_dist)), contact_dist.shape)
                else:
                    ys, xs = np.where(component_missing)
                    grid_y = float(ys.mean())
                    grid_x = float(xs.mean())
            image_x = int(np.clip(round((grid_x + 0.5) * image_width / grid_w), 0, image_width - 1))
            image_y = int(np.clip(round((grid_y + 0.5) * image_height / grid_h), 0, image_height - 1))
            if missing_image[image_y, image_x]:
                clicks.append((image_x, image_y))
            else:
                yy, xx = np.where(missing_image)
                if len(xx) == 0:
                    yy, xx = np.where(hierarchy_mask.astype(bool))
                if len(xx) == 0:
                    continue
                nearest = int(np.argmin((xx - image_x) ** 2 + (yy - image_y) ** 2))
                clicks.append((int(xx[nearest]), int(yy[nearest])))
            if len(clicks) >= max_clicks:
                break
        return clicks

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
        self.reset()
        self._original_size = image.shape[:2]
        semantic_feature = self.prepare_image(image)

        if positive_clicks is None:
            positive_clicks = [click_xy]
        positive_clicks = list(positive_clicks)

        if candidate_override is not None:
            if candidate_override.shape != simple_probability.shape:
                raise ValueError("Candidate override must match the prediction size.")
            original_candidate = candidate_override.astype(bool).copy()
        else:
            original_candidate = np.zeros_like(simple_probability, dtype=bool)
            for positive_click in positive_clicks:
                original_candidate |= self._click_component(
                    simple_probability.astype(np.float32),
                    positive_click,
                    self.candidate_threshold,
                )
        self._original_candidate = original_candidate.copy()
        probability_patch = cv2.resize(
            simple_probability.astype(np.float32),
            (semantic_feature.shape[2], semantic_feature.shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )
        candidate_patch = cv2.resize(
            original_candidate.astype(np.uint8),
            (semantic_feature.shape[2], semantic_feature.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        channels, grid_h, grid_w = semantic_feature.shape
        patch_features = semantic_feature.reshape(channels, grid_h * grid_w).permute(1, 0).float().cpu().numpy()
        patch_labels = self._agglomerative_clustering(patch_features, self.cluster_tau).reshape(grid_h, grid_w)

        click_x = int(np.clip(round(click_xy[0] * grid_w / image.shape[1]), 0, grid_w - 1))
        click_y = int(np.clip(round(click_xy[1] * grid_h / image.shape[0]), 0, grid_h - 1))
        root_cluster_label = int(patch_labels[click_y, click_x])
        root_cluster_mask = patch_labels == root_cluster_label
        root_component = self._component_mask(root_cluster_mask, click_y, click_x)

        flat_patch_labels = patch_labels.reshape(-1)
        unique_clusters = np.unique(flat_patch_labels)
        cluster_features = {}
        cluster_probabilities = {}
        for cluster_label in unique_clusters.tolist():
            cluster_mask = flat_patch_labels == cluster_label
            cluster_feature = patch_features[cluster_mask].mean(axis=0)
            cluster_features[int(cluster_label)] = cluster_feature / max(np.linalg.norm(cluster_feature), 1e-6)
            cluster_probabilities[int(cluster_label)] = float(probability_patch.reshape(-1)[cluster_mask].mean())

        component_map = np.full((grid_h, grid_w), -1, dtype=np.int32)
        component_cluster = []
        component_features = []
        component_probabilities = []
        component_centroids = []
        component_candidate_overlap = []
        component_pixel_masks = []
        component_index = 0
        for cluster_label in unique_clusters.tolist():
            cluster_mask = (patch_labels == int(cluster_label)).astype(np.uint8)
            _, components = cv2.connectedComponents(cluster_mask, connectivity=4)
            for label_index in range(1, int(components.max()) + 1):
                component_mask = components == label_index
                component_map[component_mask] = component_index
                component_cluster.append(int(cluster_label))
                component_features.append(cluster_features[int(cluster_label)])
                component_probabilities.append(float(probability_patch[component_mask].mean()))
                ys, xs = np.where(component_mask)
                component_centroids.append(
                    np.asarray(
                        [
                            float(ys.mean()) / max(grid_h - 1, 1),
                            float(xs.mean()) / max(grid_w - 1, 1),
                        ],
                        dtype=np.float32,
                    )
                )
                component_candidate_overlap.append(float(candidate_patch[component_mask].mean()))
                component_pixel_masks.append(component_mask)
                component_index += 1

        num_components = component_index
        root_component_index = int(component_map[click_y, click_x])
        if root_component_index < 0:
            raise RuntimeError("Failed to locate the clicked patch component.")

        component_neighbors = {index: set() for index in range(num_components)}
        for y in range(grid_h):
            for x in range(grid_w - 1):
                first = int(component_map[y, x])
                second = int(component_map[y, x + 1])
                if first >= 0 and second >= 0 and first != second:
                    component_neighbors[first].add(second)
                    component_neighbors[second].add(first)
        for y in range(grid_h - 1):
            for x in range(grid_w):
                first = int(component_map[y, x])
                second = int(component_map[y + 1, x])
                if first >= 0 and second >= 0 and first != second:
                    component_neighbors[first].add(second)
                    component_neighbors[second].add(first)

        self._semantic_component_map = component_map.copy()
        self._semantic_component_features = np.stack(component_features, axis=0).astype(np.float32)
        self._semantic_component_centroids = np.stack(component_centroids, axis=0).astype(np.float32)
        self._semantic_component_neighbors = {
            int(index): tuple(int(neighbor) for neighbor in sorted(neighbors))
            for index, neighbors in component_neighbors.items()
        }
        self._semantic_image_size = image.shape[:2]

        root_feature = component_features[root_component_index]
        root_probability = component_probabilities[root_component_index]
        root_centroid = component_centroids[root_component_index]

        def _add_level_from_patch_mask(
            patch_mask: np.ndarray,
            level_masks: List[np.ndarray],
            level_areas: List[float],
            *,
            relax_to_probability: bool,
            hard_clip_candidate: bool,
        ):
            up_mask = self._upsample_patch_mask(
                patch_mask,
                (image.shape[1], image.shape[0]),
                simple_probability,
                original_candidate,
                relax_to_probability=relax_to_probability,
                hard_clip_candidate=hard_clip_candidate,
            )
            area = float(up_mask.sum())
            if area < 16:
                return
            signature = up_mask.tobytes()
            if any(existing.tobytes() == signature for existing in level_masks):
                return
            if level_areas and area - level_areas[-1] < 8.0:
                return
            level_masks.append(up_mask)
            level_areas.append(area)

        level_masks: List[np.ndarray] = []
        level_areas: List[float] = []

        # Make the very first hierarchy level visually anchored at the click
        # center instead of exposing the full clicked patch block immediately.
        seed_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
        seed_radius = max(1, int(round(min(grid_h, grid_w) * 0.010)))
        cv2.circle(seed_mask, (click_x, click_y), seed_radius, 1, -1)
        seed_mask = seed_mask.astype(bool)
        _add_level_from_patch_mask(
            seed_mask,
            level_masks,
            level_areas,
            relax_to_probability=False,
            hard_clip_candidate=False,
        )

        root_hops = np.full((grid_h, grid_w), -1, dtype=np.int32)
        queue = [(click_y, click_x)]
        root_hops[click_y, click_x] = 0
        queue_index = 0
        while queue_index < len(queue):
            current_y, current_x = queue[queue_index]
            queue_index += 1
            for delta_y, delta_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_y = current_y + delta_y
                next_x = current_x + delta_x
                if next_y < 0 or next_y >= grid_h or next_x < 0 or next_x >= grid_w:
                    continue
                if not root_component[next_y, next_x] or root_hops[next_y, next_x] >= 0:
                    continue
                root_hops[next_y, next_x] = root_hops[current_y, current_x] + 1
                queue.append((next_y, next_x))

        unique_hops = sorted(int(hop) for hop in np.unique(root_hops[root_hops >= 0]).tolist())
        if not unique_hops:
            unique_hops = [0]
        for hop in unique_hops:
            _add_level_from_patch_mask(
                root_component & (root_hops >= 0) & (root_hops <= hop),
                level_masks,
                level_areas,
                relax_to_probability=False,
                hard_clip_candidate=False,
            )
        _add_level_from_patch_mask(
            root_component,
            level_masks,
            level_areas,
            relax_to_probability=False,
            hard_clip_candidate=False,
        )

        self._root_level_count = len(level_masks)

        selected_components = [root_component_index]
        selected_set = {root_component_index}
        frontier = set(component_neighbors[root_component_index])

        def _current_region_feature() -> np.ndarray:
            features = np.stack([component_features[index] for index in selected_components], axis=0)
            weights = np.asarray(
                [max(component_candidate_overlap[index], 0.05) * max(component_probabilities[index], 0.05)
                 for index in selected_components],
                dtype=np.float32,
            )
            prototype = np.average(features, axis=0, weights=weights)
            return prototype / max(np.linalg.norm(prototype), 1e-6)

        def _current_region_area() -> float:
            return float(sum(component_pixel_masks[index].sum() for index in selected_components))

        def _component_score(component_idx: int, region_feature: np.ndarray, region_area: float) -> Tuple[float, float, float, float, int]:
            semantic_similarity = float(np.dot(component_features[component_idx], region_feature))
            semantic_distance = 1.0 - semantic_similarity
            probability_bonus = component_probabilities[component_idx]
            candidate_bonus = component_candidate_overlap[component_idx]
            centroid_distance = float(np.linalg.norm(component_centroids[component_idx] - root_centroid))
            area_ratio = float(component_pixel_masks[component_idx].sum()) / max(region_area, 1.0)
            area_penalty = max(0.0, area_ratio - 0.35)
            touch_count = sum(1 for neighbor in component_neighbors[component_idx] if neighbor in selected_set)
            touch_bonus = min(touch_count, 3) * 0.05
            score = (
                semantic_distance
                - 0.22 * probability_bonus
                - 0.18 * candidate_bonus
                + 0.08 * centroid_distance
                + 0.20 * area_penalty
                - touch_bonus
            )
            return (
                score,
                semantic_distance,
                -probability_bonus,
                area_penalty,
                int(component_idx),
            )

        def _cumulative_patch_mask() -> np.ndarray:
            patch_mask = np.zeros((grid_h, grid_w), dtype=bool)
            for component_idx in selected_components:
                patch_mask |= component_pixel_masks[component_idx]
            return self._component_mask(patch_mask, click_y, click_x)

        image_patch_area = float(grid_h * grid_w)
        candidate_patch_area = max(float(candidate_patch.sum()), 1.0)
        # Open the high-granularity search beyond SimpleClick without allowing a
        # small clicked component to grow through most of the image background.
        max_overdrive_area = min(
            image_patch_area * 0.62,
            max(candidate_patch_area * 12.0, candidate_patch_area + image_patch_area * 0.15),
        )
        while frontier:
            region_feature = _current_region_feature()
            region_area = _current_region_area()
            candidate_frontier = sorted(frontier)
            candidate_scores = {
                int(component_idx): _component_score(
                    component_idx, region_feature, region_area
                )
                for component_idx in candidate_frontier
            }
            best_component = min(
                candidate_frontier,
                key=lambda component_idx: candidate_scores[component_idx],
            )
            best_score = candidate_scores[best_component]
            if self.capture_frontier_scores:
                self._frontier_score_trace.append(
                    {
                        "step": len(selected_components) - 1,
                        "selected_component": int(best_component),
                        "candidates": [
                            {
                                "component": int(component_idx),
                                "score": float(candidate_scores[component_idx][0]),
                            }
                            for component_idx in candidate_frontier
                        ],
                    }
                )
            if semantic_overdrive:
                component_area = float(component_pixel_masks[best_component].sum())
                semantic_distance = float(best_score[1])
                component_probability = float(component_probabilities[best_component])
                component_overlap = float(component_candidate_overlap[best_component])
                if region_area + component_area > max_overdrive_area:
                    break
                if (
                    len(selected_components) > max(3, self._root_level_count + 1)
                    and semantic_distance > 0.82
                    and component_probability < 0.04
                    and component_overlap < 0.005
                ):
                    break
            frontier.remove(best_component)
            if best_component in selected_set:
                continue
            selected_set.add(best_component)
            selected_components.append(best_component)

            cumulative_patch_mask = _cumulative_patch_mask()
            _add_level_from_patch_mask(
                cumulative_patch_mask,
                level_masks,
                level_areas,
                relax_to_probability=True,
                hard_clip_candidate=not semantic_overdrive,
            )
            for neighbor in component_neighbors[best_component]:
                if neighbor not in selected_set:
                    frontier.add(neighbor)

        if not level_masks:
            fallback = original_candidate.copy()
            level_masks = [fallback]
            level_areas = [float(fallback.sum())]
            self._root_level_count = 1
        elif level_areas[-1] < float(original_candidate.sum()) * 0.95:
            level_masks.append(original_candidate.copy())
            level_areas.append(float(original_candidate.sum()))

        if semantic_overdrive and len(level_masks) > self._root_level_count:
            root_index = max(0, self._root_level_count - 1)
            root_area = max(float(level_areas[root_index]), 1.0)
            candidate_area = max(float(original_candidate.sum()), 1.0)
            candidate_to_root = candidate_area / root_area

            if candidate_to_root >= 2.5:
                # A first-click prediction spanning several root blocks is
                # already an object-level proposal. End at the semantic stage
                # that best agrees with it instead of growing into background.
                best_index = root_index
                best_score = -1.0
                for level_index in range(root_index, len(level_masks)):
                    current = level_masks[level_index]
                    intersection = float(np.logical_and(current, original_candidate).sum())
                    union = float(np.logical_or(current, original_candidate).sum())
                    iou = intersection / max(union, 1.0)
                    outside_ratio = float(np.logical_and(current, np.logical_not(original_candidate)).sum()) / max(
                        float(current.sum()),
                        1.0,
                    )
                    score = iou - 0.20 * outside_ratio
                    if score > best_score + 1e-8:
                        best_score = score
                        best_index = level_index
                level_masks = level_masks[:best_index + 1]
                level_areas = level_areas[:best_index + 1]
            else:
                # For genuinely local first-click masks, retain open semantic
                # completion but stop when growth leaves a stable plateau via a
                # sudden large component, which is usually background.
                increments = np.diff(np.asarray(level_areas, dtype=np.float64))
                stop_index = len(level_masks)
                minimum_region_area = max(candidate_area * 4.0, root_area * 6.0)
                for level_index in range(root_index + 7, len(level_masks)):
                    previous = increments[max(root_index, level_index - 6):level_index - 1]
                    if previous.size < 4 or float(level_areas[level_index - 1]) < minimum_region_area:
                        continue
                    typical_increment = max(float(np.median(previous)), image.shape[0] * image.shape[1] * 0.001)
                    current_increment = float(increments[level_index - 1])
                    if current_increment > max(4.0 * typical_increment, image.shape[0] * image.shape[1] * 0.01):
                        stop_index = level_index
                        break
                level_masks = level_masks[:stop_index]
                level_areas = level_areas[:stop_index]

        level_masks, self._level_areas = self._enforce_monotonic_levels(level_masks)
        self._root_level_count = min(self._root_level_count, len(level_masks))

        self._labels = None
        self._candidate = None
        self._arrival = None
        self._primary_arrival = None
        self._graph_hops = None
        self._level_costs = np.arange(len(level_masks), dtype=np.float64)
        self._level_masks = level_masks
        self._start_level = 0
        self._current_granularity = 0.0
        return self.mask_at(0.0)

    def mask_at(self, granularity: float) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("The DINOv3 hierarchy has not been built yet.")
        granularity = float(np.clip(granularity, 0.0, 1.0))
        self._current_granularity = granularity

        if self._level_masks is not None and len(self._level_masks) > 0:
            if len(self._level_masks) == 1:
                return self._level_masks[0].copy()
            root_level_count = max(1, min(int(self._root_level_count), len(self._level_masks)))
            phase_boundary = 0.50 if root_level_count < len(self._level_masks) else 1.0

            if granularity <= phase_boundary:
                local_granularity = granularity / max(phase_boundary, 1e-6)
                root_areas = self._level_areas[:root_level_count]
                minimum_area = max(float(root_areas[0]), 1.0)
                maximum_area = max(float(root_areas[-1]), minimum_area)
                target_area = minimum_area * (maximum_area / minimum_area) ** local_granularity
                level = int(np.searchsorted(root_areas, target_area, side="left"))
                level = int(np.clip(level, 0, root_level_count - 1))
            else:
                if root_level_count >= len(self._level_masks):
                    level = len(self._level_masks) - 1
                else:
                    local_granularity = (granularity - phase_boundary) / max(1.0 - phase_boundary, 1e-6)
                    tail_areas = self._level_areas[root_level_count - 1:]
                    minimum_area = max(float(tail_areas[0]), 1.0)
                    maximum_area = max(float(tail_areas[-1]), minimum_area)
                    target_area = minimum_area * (maximum_area / minimum_area) ** local_granularity
                    tail_level = int(np.searchsorted(tail_areas, target_area, side="left"))
                    tail_level = int(np.clip(tail_level, 0, len(tail_areas) - 1))
                    level = (root_level_count - 1) + tail_level
            return self._level_masks[level].copy()

        minimum_area = max(float(self._level_areas[self._start_level]), 1.0)
        maximum_area = max(float(self._level_areas[-1]), minimum_area)
        target_area = minimum_area * (maximum_area / minimum_area) ** granularity
        level = int(np.searchsorted(self._level_areas, target_area, side="left"))
        level = int(np.clip(level, self._start_level, len(self._level_costs) - 1))
        threshold = self._level_costs[level]
        mask = (self._arrival[self._labels] <= threshold) & self._candidate
        if granularity >= 1.0:
            mask = self._candidate.copy()

        height, width = self._original_size
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask & self._original_candidate
