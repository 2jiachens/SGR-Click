"""Prediction-only scope consolidation using the existing first-click mask.

Near-complete foreground scopes may share one representative. A separately
declared small-prefix rule is evaluated as a candidate, never a GT-based gate.
"""
import numpy as np

VARIANTS={
    'near_only':None,
    'near_frag10':(.10,6.,.90),
    'near_frag20':(.20,4.,.80),
    'near80_support85_frag10':(.10,6.,.90),
    'near85_support85_frag10':(.10,6.,.90),
    'near80_ambiguous90_frag10':(.10,6.,.90),
    'near80_ambiguous90_nativebest_frag10':(.10,6.,.90),
    'near80_protected97_nativebest_frag10':(.10,6.,.90),
}
NEAR_RULES={'near80_support85_frag10':(.80,.85),'near85_support85_frag10':(.85,.85),
    'near80_ambiguous90_frag10':(.80,.85),'near80_ambiguous90_nativebest_frag10':(.80,.85),
    'near80_protected97_nativebest_frag10':(.80,.85)}


def consolidate(path,image,probability,click,variant):
    assert variant in VARIANTS
    near_iou,near_support=NEAR_RULES.get(variant,(.90,.90))
    native_best=variant in ['near80_ambiguous90_nativebest_frag10','near80_protected97_nativebest_frag10']
    protected=variant=='near80_protected97_nativebest_frag10'
    details=dict(variant=variant,near_iou=near_iou,near_support=near_support,changed=False,single_scope_preserved=len(path.masks)<=2,
        removed_prefix=0,removed_near_scopes=0,mask_union_used=False,
        representative_rule='native_soft_iou' if native_best else 'last',single_scope_endpoint_preserved=False,
        foreground_addition_protection=.97 if protected else None,protected_pairs=0)
    if len(path.masks)<=2:return path,details
    foreground=probability>.5;area=int(foreground.sum());x,y=map(int,click)
    if not area or not foreground[y,x]:return path,dict(details,reason='No positive-click native foreground')
    levels=list(path.masks[1:]);endpoint=levels[-1]
    def coverage(a,b):return np.count_nonzero(a&b)/max(int(a.sum()),1)
    prefix=VARIANTS[variant]
    if prefix is not None and coverage(foreground,endpoint)>=.90 and int(endpoint.sum())<=2*area:
        fraction,ratio,support=prefix
        while len(levels)>1:
            fine=levels[0];fine_area=int(fine.sum())
            if (fine_area>=fraction*int(endpoint.sum()) or area<=ratio*fine_area
                    or coverage(fine,foreground)<support):break
            levels.pop(0);details['removed_prefix']+=1
    def equivalent(a,b):
        union=int(np.count_nonzero(a|b));iou=np.count_nonzero(a&b)/max(union,1)
        supported=bool(iou>=near_iou and int(a.sum())<=1.5*area and int(b.sum())<=1.5*area
            and coverage(a,foreground)>=near_support and coverage(b,foreground)>=near_support
            and coverage(foreground,a)>=near_support and coverage(foreground,b)>=near_support)
        if not supported or variant not in ['near80_ambiguous90_frag10','near80_ambiguous90_nativebest_frag10','near80_protected97_nativebest_frag10']:return supported
        addition=b&~a
        # Strongly predicted additions can represent a genuine part even when
        # both scopes cover over 90% of the native object. Apply this safeguard
        # before the strict near-duplicate shortcut; no labels are consulted.
        if protected and addition.any() and coverage(a,foreground)>=.97 and coverage(addition,foreground)>=.97:
            details['protected_pairs']+=1
            return False
        strict=(iou>=.90 and coverage(a,foreground)>=.90 and coverage(b,foreground)>=.90
            and coverage(foreground,a)>=.90 and coverage(foreground,b)>=.90)
        if strict:return True
        # Wider merging is reserved for uncertain predicted contour additions.
        # Strong native foreground additions can be genuine object parts.
        return bool(addition.any() and coverage(addition,foreground)<.90)
    predicted=np.clip(probability,0,1).astype(np.float64) if native_best else None
    predicted_mass=float(predicted.sum()) if native_best else 0.
    def representative_score(mask):
        intersection=float(predicted[mask].sum())
        return intersection/max(float(mask.sum())+predicted_mass-intersection,1e-8)
    retained=[];reference=None;representative_scores=[]
    for mask in levels:
        if retained and equivalent(reference,mask):
            score=representative_score(mask) if native_best else 0.
            if not native_best or score>=representative_scores[-1]:
                retained[-1]=mask;representative_scores[-1]=score
            details['removed_near_scopes']+=1
        else:
            retained.append(mask);reference=mask
            representative_scores.append(representative_score(mask) if native_best else 0.)
    if native_best and len(retained)==1:
        retained[-1]=endpoint;details['single_scope_endpoint_preserved']=True
    if not details['removed_prefix'] and not details['removed_near_scopes']:return path,details
    from isegm.inference.anchored_scope_path import AnchoredScopePath
    result=AnchoredScopePath([path.masks[0],*retained],image,probability,keep_start_duplicate=True,
        merge_contour_duplicates=not protected,
        contour_radius=6,allow_fragment_noise=True,minimum_iou=.90,relative_contour_radius=.30)
    if native_best and len(result.masks)==2 and not np.array_equal(result.masks[-1],endpoint):
        result=AnchoredScopePath([path.masks[0],endpoint],image,probability,keep_start_duplicate=True,
            contour_radius=6,allow_fragment_noise=True,minimum_iou=.90,relative_contour_radius=.30)
        details['single_scope_endpoint_preserved']=True
    result.has_local_start=True;result.first_semantic_index=1
    result.display_finalized=True;result.display_threshold=.5
    result.display_probabilities=[m.astype(np.float32) for m in result.masks]
    result.hard_scope_probabilities=True;result.original_consensus_applied=True
    result.native_consensus_applied=True;result.unrefined_path=getattr(path,'unrefined_path',path)
    endpoint_same=bool(np.array_equal(result.masks[-1],endpoint))
    if not native_best:np.testing.assert_array_equal(result.masks[-1],endpoint)
    if native_best and len(result.masks)==2:assert endpoint_same
    details.update(changed=True,native_area=area,resulting_semantic_stages=len(result.masks)-1,
        original_semantic_stages=len(path.masks)-1,endpoint_unchanged=endpoint_same,parameters_added=0,
        representative_scores=representative_scores if native_best else None,
        constructor_contour_deduplication=not protected)
    return result,details


def install(controller,holder):
    base=type(controller)
    class NativeConsensusController(base):
        def _apply_native_consensus(self):
            if (holder['variant']=='control' or len(self.clicker)!=1 or not self.granularity_ready
                    or not self.clicker.get_clicks()[0].is_positive):return
            path=self.granularity_refiner.path
            if getattr(path,'native_consensus_applied',False):return
            click=self.clicker.get_clicks()[0].coords
            path,details=consolidate(path,self.image,self.simple_probs_history[-1],(click[1],click[0]),holder['variant'])
            path.native_consensus_applied=True;holder['details']=details
            self.granularity_refiner.path=path;self.set_granularity(self.requested_granularity)
        def add_click(self,x,y,is_positive):
            first=len(self.clicker)==0 and is_positive and self._init_mask is None
            result=super().add_click(x,y,is_positive)
            if first:self._apply_native_consensus()
            return result
        def _finalize_display_path(self):
            result=super()._finalize_display_path();self._apply_native_consensus();return result
    controller.__class__=NativeConsensusController
    return controller
