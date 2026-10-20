"""Merge contour-only semantic duplicates, then interpolate between anchors."""
import numpy as np
import cv2
from scipy import ndimage
from isegm.inference.continuous_granularity import ContinuousScopePath


def same_semantic_stage(a, b, minimum_iou=.95, contour_radius=3, allow_fragment_noise=False,
                        relative_contour_radius=0.):
    """Require both high overlap and changes confined to a narrow contour band.

    Area alone would merge different parts of an object. IoU alone would also
    discard small new parts far from the contour of a large object.
    """
    changed = a ^ b
    difference = int(np.count_nonzero(changed))
    if not difference:
        return True
    union_area = int(np.count_nonzero(a | b))
    if difference > (1. - minimum_iou) * union_area:
        return False
    contour_radius=max(contour_radius, int(round(
        np.sqrt(min(np.count_nonzero(a),np.count_nonzero(b))/np.pi)*relative_contour_radius)))
    structure = np.ones((3, 3), dtype=bool)
    shared_band = np.ones(a.shape, dtype=bool)
    kernel=np.ones((2*contour_radius+1,2*contour_radius+1),dtype=np.uint8)
    for mask in (a, b):
        # Square morphology equals repeated 3x3 dilation/erosion, but OpenCV
        # uses an efficient rectangular filter even for large object contours.
        pixels=mask.astype(np.uint8)
        band=cv2.dilate(pixels,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        band &= cv2.erode(pixels,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)==0
        shared_band &= band
    beyond_contour = changed & ~shared_band
    if not beyond_contour.any():
        return True
    if not allow_fragment_noise or np.count_nonzero(beyond_contour) > .005 * union_area:
        return False
    # The soft endpoint may introduce isolated one-pixel fragments far from
    # the object. These do not define a new semantic scope. Keep real regions
    # and thin object parts: both off-contour noise and each fragment are bounded,
    # and every fragment must be only one pixel thick.
    # Measure the ENTIRE changed component, not just its off-contour tip;
    # otherwise a long newly added thin part could masquerade as tiny noise.
    labels, _ = ndimage.label(changed, structure)
    noise_ids = np.unique(labels[beyond_contour])
    sizes = np.bincount(labels.ravel())[noise_ids]
    noise = np.isin(labels, noise_ids)
    return (sizes.max() <= 16
            and ndimage.distance_transform_edt(noise).max() <= 1.)


def same_object_scope(a,b):
    """Treat small contour changes of the same object as one semantic scope.

    Allow up to 10% mask difference, confined to 30% of the smaller object's
    equivalent radius. A new distant part or solid region still fails the
    contour/fragment checks. No image identity or class label is used.
    """
    return same_semantic_stage(a,b,minimum_iou=.90,contour_radius=6,
        allow_fragment_noise=True,relative_contour_radius=.30)


class AnchoredScopePath:
    def __init__(self, masks, image, probability, legacy_controls=None, keep_start_duplicate=False,
                 merge_contour_duplicates=True, contour_radius=3, allow_fragment_noise=False,
                 minimum_iou=.95, relative_contour_radius=0.):
        self.masks, self.legacy_controls = [], []
        self.anchor_groups = []
        controls = list(legacy_controls) if legacy_controls is not None else list(range(len(masks)))
        if len(controls) != len(masks):
            raise ValueError('Each anchor must have a legacy control')
        group_reference = None
        for index, (mask, g) in enumerate(zip(masks, controls)):
            mask = np.asarray(mask) > .5
            if mask.shape != probability.shape or not mask.any():
                raise ValueError('Anchors must be nonempty and match the image')
            # Keep the local click seed separate from the first semantic stage.
            new_group = not self.masks or (keep_start_duplicate and index == 1)
            if not new_group:
                identical = np.array_equal(mask, self.masks[-1])
                equivalent = merge_contour_duplicates and same_semantic_stage(
                    group_reference, mask, contour_radius=contour_radius,
                    allow_fragment_noise=allow_fragment_noise,minimum_iou=minimum_iou,
                    relative_contour_radius=relative_contour_radius)
                if identical or equivalent:
                    self.anchor_groups[-1].append(float(g))
                    if merge_contour_duplicates:
                        # Use the last contour in a group, preserving the final
                        # endpoint without a union that could invent foreground.
                        self.masks[-1] = mask.copy()
                        self.legacy_controls[-1] = float(g)
                    continue
            self.masks.append(mask.copy())
            self.legacy_controls.append(float(g))
            self.anchor_groups.append([float(g)])
            # Compare with the group's FIRST mask, preventing small successive
            # expansions from chaining into the loss of a real semantic stage.
            group_reference = mask.copy()
        self.input_anchor_count = len(controls)
        if not self.masks or len(self.masks) > 101:
            raise ValueError('The integer 0-100 slider supports 1 to 101 anchors')
        self.root = self.masks[0].copy()
        self.areas = np.asarray([m.sum() for m in self.masks])
        self.anchor_ticks = self._allocate_ticks()
        self.anchor_controls = self.anchor_ticks / 100.
        # Most projected paths are used only to collect exact display anchors,
        # then replaced. Do not build their pixel traversal a second time.
        self.transitions = [None] * (len(self.masks)-1)
        self._image=image
        self._probability=probability

    def _transition(self,index):
        if self.transitions[index] is None:
            a,b=self.masks[index:index+2]
            union=a|b
            add=self._growth_order(a,union,b&~a,self._image,self._probability)
            remove=self._growth_order(b,union,a&~b,self._image,self._probability)[::-1]
            self.transitions[index]=(add,remove)
        return self.transitions[index]

    def _allocate_ticks(self):
        gaps = len(self.masks) - 1
        if not gaps:
            return np.array([0], dtype=int)
        changes = np.sqrt([np.count_nonzero(a ^ b) for a, b in zip(self.masks[:-1], self.masks[1:])])
        # Reserve an integer tick for each old result; share remaining room
        # equally across semantic intervals and by sqrt pixel-change magnitude.
        weights = .5 / gaps + .5 * changes / changes.sum() if changes.sum() else np.full(gaps, 1. / gaps)
        ideal = weights * (100 - gaps)
        extra = np.floor(ideal).astype(int)
        remaining = 100 - gaps - int(extra.sum())
        extra[np.argsort(-(ideal - extra), kind='stable')[:remaining]] += 1
        return np.r_[0, np.cumsum(extra + 1)]

    @staticmethod
    def _growth_order(seed, union, changed, image, probability):
        indices = np.flatnonzero(changed)
        if not len(indices):
            return indices
        path = ContinuousScopePath([seed, union], image, probability)
        return indices[np.argsort(path.arrival.flat[indices], kind='stable')]

    def mask_at(self, g):
        if not np.isfinite(g):
            raise ValueError('Granularity must be finite')
        tick = float(np.clip(g, 0, 1)) * 100
        exact = np.flatnonzero(np.isclose(self.anchor_ticks, tick, rtol=0, atol=1e-6))
        if len(exact):
            return self.masks[int(exact[0])].astype(np.float32)
        if tick >= self.anchor_ticks[-1]:
            return self.masks[-1].astype(np.float32)
        left = int(np.searchsorted(self.anchor_ticks, tick) - 1)
        fraction = (tick - self.anchor_ticks[left]) / (self.anchor_ticks[left + 1] - self.anchor_ticks[left])
        add, remove = self._transition(left)
        result = self.masks[left].copy()
        result.flat[add[:int(np.floor(fraction * len(add)))]] = True
        result.flat[remove[:int(np.floor(fraction * len(remove)))]] = False
        return result.astype(np.float32)
