"""DINO-free descriptor provider, extracted unchanged from the delivered runtime."""
import torch
from .semantic_student import FeatureTap
from isegm.inference.soft_dino_semantic import SoftDinoSemanticRefiner

def forbid_encoder(*args, **kwargs):
    raise RuntimeError("The lightweight version must not load a DINO encoder")

class DistilledSemanticRefiner(SoftDinoSemanticRefiner):
    def __init__(self,model,student,device,**kwargs):
        super().__init__('.', '.', device,**kwargs)
        self.student=student.eval().to(device);self.image_shape=None;self.student_calls=0
        for p in self.student.parameters():p.requires_grad_(False)
        self._external_trajectory._load_encoder=forbid_encoder
        self.tap=FeatureTap(model,self.capture)
    def _load_encoder(self):forbid_encoder()
    def warmup(self):return None
    def prepare_image(self,image,force=False):
        key=self._image_cache_key(image)
        if force or key!=self._prepared_image_key:
            self._prepared_semantic_feature=None;self._prepared_image_key=key;self.image_shape=image.shape[:2]
        return self._prepared_semantic_feature
    def begin_first_click(self,image):
        self.prepare_image(image,force=True);self.tap.begin()
    @torch.inference_mode()
    def capture(self,features):
        if self.image_shape is None:raise RuntimeError('Image must be set before first prediction')
        h,w=self.image_shape;scale=self.image_size/max(h,w)
        size=(max(1,int(round(h*scale/16))),max(1,int(round(w*scale/16))))
        self._prepared_semantic_feature=self.student(features,size)[0].detach()
        self.student_calls+=1
        self.tap.values={}
    def build(self,*args,**kwargs):
        if self._prepared_semantic_feature is None:raise RuntimeError('First SimpleClick feature capture is missing')
        result=super().build(*args,**kwargs)
        assert self.encoder is None and self._external_trajectory.encoder is None
        return result
    def close(self):self.tap.close()
