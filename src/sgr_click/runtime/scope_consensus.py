"""Reject tiny local fragments lacking first-click segmentation support."""
import numpy as np


def prune(path,image,probability,click):
    from isegm.inference.anchored_scope_path import AnchoredScopePath
    levels=list(path.masks[1:]);native=probability>.5;removed=0
    while len(levels)>1:
        fine=levels[0];coarse=levels[-1]
        # The independently computed user-click segmentation must be much
        # larger and contain the fragment. This is a prediction consistency
        # test, not a minimum area rule applied to ground-truth parts.
        if (fine.sum()>=.03*coarse.sum() or native.sum()<=8*fine.sum()
                or np.count_nonzero(fine&native)<.90*fine.sum()):break
        levels.pop(0);removed+=1
    if not removed:return path,dict(removed=0)
    result=AnchoredScopePath([path.masks[0]]+levels,image,probability,keep_start_duplicate=True,
        contour_radius=6,allow_fragment_noise=True,minimum_iou=.90,relative_contour_radius=.30)
    result.has_local_start=True;result.first_semantic_index=1
    result.display_finalized=True;result.display_threshold=.5
    result.display_probabilities=[m.astype(np.float32) for m in result.masks]
    result.unrefined_path=getattr(path,'unrefined_path',path)
    return result,dict(removed=removed,first_click_prediction_area=int(native.sum()))
