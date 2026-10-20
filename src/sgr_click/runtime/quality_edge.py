"""Image-guided cut confined to a narrow band around an existing mask."""
import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from skimage.morphology import skeletonize

def refine_boundary(image,mask,radius=5,strength=4.,color_sigma=12.,positive_points=(),negative_points=(),protected_foreground=None,probability=None,lab_image=None):
    image=np.asarray(image)
    mask=np.asarray(mask,dtype=bool)
    if image.shape!=(*mask.shape,3):raise ValueError('RGB image and mask shapes differ')
    if radius<1 or not np.isfinite(strength) or strength<=0 or not np.isfinite(color_sigma) or color_sigma<=0:raise ValueError('Invalid boundary parameters')
    if not mask.any() or mask.all() or mask.sum()<32:return mask.copy()
    inside=ndimage.distance_transform_edt(mask)
    outside=ndimage.distance_transform_edt(~mask)
    band=(inside<=radius)&(outside<=radius)
    # Hold the centerline of thin foreground structures, as well as the core.
    fixed_fg=(inside>radius)|(skeletonize(mask)&(inside<=radius))
    fixed_bg=outside>radius
    if probability is not None:
        probability=np.asarray(probability,dtype=np.float32)
        if probability.shape!=mask.shape or not np.isfinite(probability).all():
            raise ValueError('Invalid probability map')
        # Preserve reliable neural labels and solve only the uncertain portion
        # of the existing narrow band. A user click overrides either label.
        fixed_fg |= probability >= .95
        fixed_bg |= probability <= .05
    if protected_foreground is not None:
        protected_foreground=np.asarray(protected_foreground,dtype=bool)
        if protected_foreground.shape!=mask.shape:raise ValueError('Protected mask shape differs')
        fixed_fg|=protected_foreground&mask
    for points,fixed,opposite in [(positive_points,fixed_fg,fixed_bg),(negative_points,fixed_bg,fixed_fg)]:
        for x,y in points:
            if 0<=int(y)<mask.shape[0] and 0<=int(x)<mask.shape[1]:
                fixed[int(y),int(x)]=True
                opposite[int(y),int(x)]=False
    if np.any(fixed_fg&fixed_bg):return mask.copy()
    free=band&~fixed_fg&~fixed_bg
    n=int(free.sum())
    if n==0:
        result=mask.copy();result[fixed_fg]=True;result[fixed_bg]=False
        return result
    ids=np.full(mask.shape,-1,dtype=np.int32);ids[free]=np.arange(n,dtype=np.int32)
    lab=cv2.cvtColor(np.clip(image,0,255).astype(np.float32)/255.,cv2.COLOR_RGB2LAB) if lab_image is None else lab_image
    signed=inside-outside
    d_bg=np.maximum(signed[free],0)*.45
    d_fg=np.maximum(-signed[free],0)*.45
    if probability is not None:
        p=np.clip(probability[free],1e-6,1-1e-6)
        d_bg-=np.log1p(-p)
        d_fg-=np.log(p)
    rows=[];cols=[];values=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        h,w=mask.shape
        a=(slice(0,h-dy),slice(max(0,-dx),min(w,w-dx)))
        b=(slice(dy,h),slice(max(0,dx),min(w,w+dx)))
        weight=(.03+strength*np.exp(-np.sum((lab[a]-lab[b])**2,axis=2)/(2*color_sigma**2)))/np.hypot(dy,dx)
        ia,ib=ids[a],ids[b]
        both=(ia>=0)&(ib>=0)
        rows.extend([ia[both],ib[both]]);cols.extend([ib[both],ia[both]]);values.extend([weight[both],weight[both]])
        for src,dst,adj_fg,adj_bg in [(ia,ib,fixed_fg[b],fixed_bg[b]),(ib,ia,fixed_fg[a],fixed_bg[a])]:
            selected=(src>=0)&(dst<0)
            np.add.at(d_bg,src[selected&adj_fg],weight[selected&adj_fg])
            np.add.at(d_fg,src[selected&adj_bg],weight[selected&adj_bg])
    nodes=np.arange(n,dtype=np.int32)
    rows.extend([np.full(n,n),nodes]);cols.extend([nodes,np.full(n,n+1)]);values.extend([d_bg,d_fg])
    capacity=np.maximum(np.rint(np.concatenate(values)*1024),0).astype(np.int64)
    graph=coo_matrix((capacity,(np.concatenate(rows),np.concatenate(cols))),shape=(n+2,n+2)).tocsr()
    graph.eliminate_zeros()
    flow=maximum_flow(graph,n,n+1,method='dinic')
    residual=graph-flow.flow
    residual.data=(residual.data>0).astype(np.int64);residual.eliminate_zeros()
    reachable=breadth_first_order(residual,n,directed=True,return_predecessors=False)
    labels=np.zeros(n+2,dtype=bool);labels[reachable]=True
    result=mask.copy();result[free]=labels[:n]
    result[fixed_fg]=True;result[fixed_bg]=False
    return result
