"""Keep a broad coherent region even when it lies in a permissive contour band."""
import cv2,numpy as np

def install(source,model):
    from isegm.inference import anchored_scope_path as paths
    original=paths.same_semantic_stage;state=dict(variant='base')
    def equivalent(a,b,**kwargs):
        same=original(a,b,**kwargs)
        if not same or state['variant']=='base':return same
        changed=a^b
        if not changed.any():return True
        ys,xs=np.where(changed)
        crop=np.pad(changed[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.uint8),1)
        radius=float(cv2.distanceTransform(crop,cv2.DIST_L2,5).max())
        limit={'blob4':4.,'blob6':6.}[state['variant']]
        # Thin boundary jitter still merges. A solid region wider than twice
        # this radius retains its own scope. No categories or GT are consulted.
        return radius<=limit
    paths.same_semantic_stage=equivalent
    return state
