"""Exact spatial cropping of the existing narrow-band solver."""
import numpy as np


def crop_solver(solver):
    def refine(image, mask, radius=5, **kwargs):
        mask=np.asarray(mask,dtype=bool)
        yy,xx=np.nonzero(mask)
        if not len(yy) or radius<1:
            return solver(image,mask,radius=radius,**kwargs)
        # Include every foreground pixel and a fixed-background collar. No
        # variable node or edge incident to it is removed from the graph.
        pad=int(radius)+2
        y0=max(int(yy.min())-pad,0);y1=min(int(yy.max())+pad+1,mask.shape[0])
        x0=max(int(xx.min())-pad,0);x1=min(int(xx.max())+pad+1,mask.shape[1])
        for name in ['positive_points','negative_points']:
            for x,y in kwargs.get(name,()):
                if not (x0<=x<x1 and y0<=y<y1):
                    return solver(image,mask,radius=radius,**kwargs)
        if (y1-y0)*(x1-x0)>=.9*mask.size:
            return solver(image,mask,radius=radius,**kwargs)
        local=dict(kwargs)
        for name in ['positive_points','negative_points']:
            local[name]=[(x-x0,y-y0) for x,y in kwargs.get(name,())]
        for name in ['protected_foreground','probability','lab_image']:
            if local.get(name) is not None:local[name]=local[name][y0:y1,x0:x1]
        # Keep a true background cell in the crop; all-foreground early exits
        # in the reference solver otherwise have different semantics.
        sub=mask[y0:y1,x0:x1]
        if sub.all():return solver(image,mask,radius=radius,**kwargs)
        result=mask.copy()
        result[y0:y1,x0:x1]=solver(image[y0:y1,x0:x1],sub,radius=radius,**local)
        return result
    return refine
