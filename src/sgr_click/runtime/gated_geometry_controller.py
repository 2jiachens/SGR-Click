"""Enable solid-region deduplication protection after the unchanged base gate."""
import torch.nn.functional as F
from .scope_consensus import prune

def install(controller,source,holder,geometry):
    base=type(controller)
    class GeometryController(base):
        def add_click(self,x,y,is_positive):
            first=len(self.clicker)==0 and is_positive and self._init_mask is None
            if first:geometry['variant']='base'
            result=super().add_click(x,y,is_positive)
            if not first or not self.granularity_ready:return result
            variant=holder['variant'];original=self.granularity_refiner.path
            details=dict(base_stages=len(original.masks)-1,centered=False)
            if variant!='base':
                original,consensus=prune(original,self.image,self.simple_probs_history[-1],(x,y))
                self.granularity_refiner.path=original;details['consensus']=consensus
                details['base_stages']=len(original.masks)-1
            if variant!='base' and details['base_stages']>1:
                geometry['variant']='blob4' if variant=='gated_center_blob4' else 'base'
                feature=source._prepared_semantic_feature
                centered=F.normalize(feature-.75*feature.mean((1,2),keepdim=True),dim=0)
                source._prepared_semantic_feature=centered
                self.semantic_cache.clear();self.edge_cache.clear()
                try:
                    self._rebuild_hierarchy(self.simple_probs_history[-1]);self._finalize_display_path()
                    assert source._prepared_semantic_feature is centered,'prepare_image replaced the centered descriptor'
                    details['centered']=True
                    details['centered_descriptor_preserved']=True
                except ValueError:
                    self.granularity_refiner.path=original;source._prepared_semantic_feature=feature
                    details['fallback']='invalid centered scope'
            details['dedup_variant']=geometry['variant'];holder['details']=details
            self.set_granularity(self.requested_granularity)
            return result
    controller.__class__=GeometryController
    return controller
