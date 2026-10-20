"""Prune only tiny predicted fragments; retain original student descriptors."""
from .scope_consensus import prune


def install(controller,holder):
    base=type(controller)

    class PrunedOriginalController(base):
        def _apply_original_consensus(self):
            if (not holder['enabled'] or len(self.clicker)!=1 or not self.granularity_ready
                    or not self.clicker.get_clicks()[0].is_positive):return
            path=self.granularity_refiner.path
            if getattr(path,'original_consensus_applied',False):return
            click=self.clicker.get_clicks()[0].coords
            changed,details=prune(path,self.image,self.simple_probs_history[-1],(click[1],click[0]))
            changed.original_consensus_applied=True
            self.granularity_refiner.path=changed
            holder['details']=details
            self.set_granularity(self.requested_granularity)

        def add_click(self,x,y,is_positive):
            first=len(self.clicker)==0 and is_positive and self._init_mask is None
            result=super().add_click(x,y,is_positive)
            if first:self._apply_original_consensus()
            return result

        def _finalize_display_path(self):
            result=super()._finalize_display_path()
            self._apply_original_consensus()
            return result

    controller.__class__=PrunedOriginalController
    return controller
