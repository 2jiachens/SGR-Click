"""Semantic filtering and boundary projection of anchor masks."""
import hashlib
import time
from collections import OrderedDict
import numpy as np
import torch
from sgr_click.gui.soft_semantic_controller import SoftSemanticFusionController
from .edge_refinement import refine_boundary
from .semantic_refinement import build_semantic_margin, build_expansion_support, refine_semantic_expansion

variant = "anchors"

class EdgeController(SoftSemanticFusionController):
    def __init__(self,*args,**kwargs):
        self.edge_cache=OrderedDict();self.edge_last_ms=0.
        self.semantic_cache=OrderedDict();self.semantic_last_ms=0.
        self._finalizing_display_path=False
        super().__init__(*args,**kwargs)

    def set_image(self,image):
        self.edge_cache.clear();self.semantic_cache.clear()
        return super().set_image(image)

    def reset_last_object(self,*args,**kwargs):
        self.edge_cache.clear();self.semantic_cache.clear()
        return super().reset_last_object(*args,**kwargs)

    @property
    def raw_object_prob(self):return super().current_object_prob

    def _semantic_context(self,initial):
        key=(self.prob_thresh,hashlib.sha256(initial.tobytes()).digest())
        if key not in self.semantic_cache:
            start=time.perf_counter()
            refiner=self.granularity_refiner
            source=getattr(refiner,'source',refiner)
            features=source._prepared_semantic_feature.permute(1,2,0).float().cpu().numpy()
            path=getattr(refiner,'path',None)
            if path is not None:
                endpoint=path.masks[-1]
            else:
                old_g=source._current_granularity
                endpoint=self._maybe_apply_boundary_projection(source.mask_at(1.),1.)>self.prob_thresh
                source._current_granularity=old_g
            margin=build_semantic_margin(features,initial,endpoint)
            allowed=build_expansion_support(initial,endpoint,margin)
            self.semantic_cache[key]=(margin,allowed)
            self.semantic_last_ms=(time.perf_counter()-start)*1000
            if len(self.semantic_cache)>4:self.semantic_cache.popitem(last=False)
        self.semantic_cache.move_to_end(key)
        return self.semantic_cache[key]

    @property
    def current_object_prob(self):
        path=getattr(self.granularity_refiner,'path',None)
        if (variant=='anchors' and not self._finalizing_display_path
                and len(self.clicker)==1 and self.granularity_ready
                and getattr(path,'display_finalized',False)):
            if path.display_threshold!=self.prob_thresh:
                self._finalize_display_path()
                path=self.granularity_refiner.path
            exact=np.flatnonzero(np.isclose(path.anchor_controls,self.requested_granularity,rtol=0,atol=1e-6))
            if len(exact):return path.display_probabilities[int(exact[0])].copy()
            # The interpolation already connects corrected contours. A
            # second graph cut could recreate duplicate displayed stages.
            return super().current_object_prob
        raw=super().current_object_prob
        if raw is None or self.image is None:return raw
        mask=raw>self.prob_thresh
        if mask.sum()<32:return raw
        clicks=tuple((int(c.coords[1]),int(c.coords[0]),bool(c.is_positive)) for c in self.clicker.get_clicks())
        protected=self.probs_history[-1][0]>self.prob_thresh
        use_expansion=len(clicks)==1 and clicks[0][2] and self.granularity_ready and bool(self.simple_probs_history)
        initial=self.simple_probs_history[-1] if use_expansion else None
        key=(id(self.image),self.prob_thresh,use_expansion,hashlib.sha256(mask.tobytes()).digest(),hashlib.sha256(protected.tobytes()).digest(),clicks,
             hashlib.sha256(initial.tobytes()).digest() if initial is not None else None)
        if key not in self.edge_cache:
            start=time.perf_counter()
            positive=[(x,y) for x,y,p in clicks if p]
            negative=[(x,y) for x,y,p in clicks if not p]
            refined=mask
            if use_expansion:
                margin,allowed=self._semantic_context(initial)
                refined=refine_semantic_expansion(self.image,mask,initial,margin,positive,negative,protected,allowed)
            # Every displayed state is fully refined, including while the
            # slider moves. Never render an intermediate contour first.
            refined=refine_boundary(self.image,refined,positive_points=positive,negative_points=negative,protected_foreground=protected)
            self.edge_cache[key]=refined
            self.edge_last_ms=(time.perf_counter()-start)*1000
            if len(self.edge_cache)>16:self.edge_cache.popitem(last=False)
        self.edge_cache.move_to_end(key)
        refined=self.edge_cache[key]
        result=raw.copy()
        # Preserve soft probabilities except where the binary contour moves.
        result[refined&~mask]=min(1.,self.prob_thresh+.03)
        result[mask&~refined]=max(0.,self.prob_thresh-.03)
        return result

    def _finalize_display_path(self):
        """Compare what is displayed AFTER semantic filtering and graph cut.

        Projection-only anchors can differ yet become the same object once
        the display filters run. Finalize once per click/threshold, preserve
        endpoint probabilities, and interpolate only the retained scopes.
        """
        from isegm.inference.anchored_scope_path import AnchoredScopePath
        refiner=self.granularity_refiner
        old_path=refiner.path
        source_path=getattr(old_path,'unrefined_path',old_path)
        requested=self.requested_granularity
        old_probability=self.probs_history[-1]
        callback=self.update_image_callback
        probabilities=[]
        self._finalizing_display_path=True
        self.update_image_callback=lambda reset_canvas=False:None
        try:
            refiner.path=source_path
            for g in source_path.anchor_controls:
                self.set_granularity(float(g))
                probabilities.append(self.current_object_prob.copy())
            # Several projections of the same object may move the contour
            # farther than the graph-cut band. Scale tolerance with the
            # object, retaining genuine new parts beyond that contour.
            path=AnchoredScopePath(
                [p>self.prob_thresh for p in probabilities],self.image,
                self.simple_probs_history[-1],source_path.legacy_controls,
                keep_start_duplicate=True,contour_radius=6,allow_fragment_noise=True,
                minimum_iou=.90,relative_contour_radius=.30)
            by_control=dict(zip(source_path.legacy_controls,probabilities))
            path.display_probabilities=[by_control[g] for g in path.legacy_controls]
            path.has_local_start=True
            path.first_semantic_index=1
            path.display_finalized=True
            path.display_threshold=self.prob_thresh
            path.unrefined_path=source_path
            path.semantic_stages_before_display_merge=len(source_path.masks)-1
            refiner.path=path
            self.set_granularity(requested)
        except Exception:
            refiner.path=old_path
            self._requested_granularity=requested
            self.probs_history[-1]=old_probability
            raise
        finally:
            self._finalizing_display_path=False
            self.update_image_callback=callback

    def add_click(self,x,y,is_positive):
        # The legacy controller resets its trajectory on a second click but
        # does not retain it for undo. Keep the existing arrays by reference;
        # reset replaces these fields rather than mutating the arrays.
        legacy_state=None
        if variant=='original' and self.granularity_refiner.ready:
            legacy_state=(self.granularity_refiner.__dict__.copy(),self.requested_granularity)
        if len(self.clicker)>0 and self._init_mask is None:
            probability=self.current_object_prob
            if probability is not None:
                self.predictor.prev_prediction=torch.from_numpy(probability.astype(np.float32)).to(self.device)[None,None]
        first_anchor=variant=='anchors' and len(self.clicker)==0 and is_positive and self._init_mask is None
        callback=self.update_image_callback
        if first_anchor:self.update_image_callback=lambda reset_canvas=False:None
        try:
            result=super().add_click(x,y,is_positive)
            if first_anchor and self.granularity_ready:
                self._finalize_display_path()
        finally:
            self.update_image_callback=callback
        if variant=='original' and self.states:
            self.states[-1]['edge_legacy_state']=legacy_state
        if len(self.clicker)==1 and is_positive and self.granularity_ready and self.simple_probs_history:
            self._semantic_context(self.simple_probs_history[-1])
        if first_anchor:self.update_image_callback()
        return result

    def undo_click(self):
        legacy_state=self.states[-1].get('edge_legacy_state') if variant=='original' and self.states else None
        if legacy_state is not None:
            self.granularity_refiner.__dict__.update(legacy_state[0])
            self._requested_granularity=legacy_state[1]
        result=super().undo_click()
        if legacy_state is not None and self.granularity_ready:
            self.set_granularity(legacy_state[1])
        return result
