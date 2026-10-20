"""Automatic DINO foreground/background competition for semantic expansion."""
import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def build_semantic_margin(features, probability, endpoint):
    features = np.asarray(features, dtype=np.float32)
    gh, gw = features.shape[:2]
    p = cv2.resize(np.asarray(probability, np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    extent = cv2.resize(np.asarray(endpoint, np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    fg = p > max(.5, float(np.quantile(p, .80)))
    outside = extent < .05
    bg = outside & (p <= float(np.quantile(p[outside], .7))) if outside.any() else p <= float(np.quantile(p, .2))
    if fg.sum() < 3:
        fg = p >= max(float(np.quantile(p, .95)), .1)
    if bg.sum() < 3:
        bg = p <= float(np.quantile(p, .2))
    if not fg.any() or not bg.any():
        return None

    def prototypes(selected):
        values = features[selected].copy()
        with threadpool_limits(limits=4):
            centers = KMeans(n_clusters=min(8, len(values)), n_init=3,
                             random_state=0).fit(values).cluster_centers_
        return centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6)

    sf = np.max(features @ prototypes(fg).T, axis=2)
    sb = np.max(features @ prototypes(bg).T, axis=2)
    return cv2.resize(sf-sb, (probability.shape[1], probability.shape[0]), interpolation=cv2.INTER_LINEAR)


def build_expansion_support(probability, endpoint, margin):
    """Reject added connected regions whose average evidence favors background.

    Decisions cover complete added regions rather than fixing individual DINO
    patches as foreground. RGB boundary refinement then corrects the contour.
    The result is computed once per click and shared by every slider value.
    """
    allowed = np.ones(probability.shape, dtype=bool)
    if margin is None:
        return allowed
    corridor = probability > .5
    regions, count = ndimage.label(np.asarray(endpoint, bool) & ~corridor, structure=np.ones((3, 3)))
    if count:
        means = ndimage.mean(margin, regions, np.arange(1, count+1))
        rejected = np.r_[False, np.asarray(means) < 0.]
        allowed[rejected[regions]] = False
    return allowed


def refine_semantic_expansion(image, mask, probability, margin,
                              positive_points=(), negative_points=(), protected_foreground=None,
                              allowed_support=None):
    """P supplies reliable seeds; DINO decides whether novel parts can expand.

    All thresholds are shared across images. No class/image identifiers, GT,
    additional clicks, learned weights or further DINO forward passes are used.
    """
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if margin is None or mask.sum() < 32:
        return mask.copy()
    if image.shape != (*mask.shape, 3) or probability.shape != mask.shape or margin.shape != mask.shape:
        raise ValueError('Image, mask, probability and semantic margin shapes differ')
    if allowed_support is None:
        allowed_support = build_expansion_support(probability, mask, margin)
    result = mask & allowed_support
    if protected_foreground is not None:
        protected = np.asarray(protected_foreground, dtype=bool)
        if protected.shape != mask.shape:
            raise ValueError('Protected mask shape differs')
        result[protected] = True
    for x, y in positive_points:
        x, y = int(x), int(y)
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            result[y, x] = True
    for x, y in negative_points:
        x, y = int(x), int(y)
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            result[y, x] = False
    return result
