"""A small descriptor student reusing the original SimpleClick forward pass."""
import torch
from torch import nn
from torch.nn import functional as F

class ChannelNorm(nn.Module):
    def forward(self,x):return F.layer_norm(x.permute(0,2,3,1),(x.shape[1],)).permute(0,3,1,2)


class ContextBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.local=nn.Sequential(ChannelNorm(),nn.Conv2d(width,width,3,padding=1,groups=width),nn.GELU(),nn.Conv2d(width,width,1))
        self.mlp=nn.Sequential(ChannelNorm(),nn.Conv2d(width,width*3,1),nn.GELU(),nn.Conv2d(width*3,width,1))
        self.scale=nn.Parameter(torch.full((2,width,1,1),.1))
    def forward(self,x):
        x=x+self.scale[0]*self.local(x)
        return x+self.scale[1]*self.mlp(x)

class ContextSemanticStudent(nn.Module):
    """Extra mapping capacity and pooled context, still sharing the image backbone."""
    def __init__(self):
        super().__init__();width=256
        self.projections=nn.ModuleList([nn.Sequential(ChannelNorm(),nn.Conv2d(c,192,1),nn.GELU()) for c in [768,768,256]])
        self.fuse=nn.Conv2d(576,width,1)
        self.blocks=nn.ModuleList([ContextBlock(width) for _ in range(3)])
        self.context_norm=nn.LayerNorm(width)
        self.context=nn.MultiheadAttention(width,8,batch_first=True)
        self.context_scale=nn.Parameter(torch.full((1,width,1,1),.1))
        self.output=nn.Sequential(ChannelNorm(),nn.Conv2d(width,1024,1))
    def forward(self,features,size):
        maps=[F.interpolate(layer(value.float()),size=size,mode='bilinear',align_corners=False) for layer,value in zip(self.projections,features)]
        x=self.blocks[0](self.fuse(torch.cat(maps,1)))
        pool_size=(min(12,size[0]),min(12,size[1]))
        tokens=F.adaptive_avg_pool2d(x,pool_size).flatten(2).transpose(1,2)
        tokens=self.context_norm(tokens)
        context=self.context(tokens,tokens,tokens,need_weights=False)[0]
        context=context.transpose(1,2).reshape(len(x),x.shape[1],*pool_size)
        x=x+self.context_scale*F.interpolate(context,size=size,mode='bilinear',align_corners=False)
        for block in self.blocks[1:]:x=block(x)
        return F.normalize(self.output(x).float(),dim=1)

def make_student(architecture='context'):
    if architecture != 'context':
        raise ValueError('The released Lite head uses the context architecture')
    return ContextSemanticStudent()

class FeatureTap:
    """Capture only the first prediction, then ignore internal anchor projections."""
    def __init__(self,model,callback=None):
        self.model=model;self.callback=callback;self.enabled=False;self.values={};self.calls=0
        self.handles=[model.backbone.blocks[5].register_forward_hook(self.middle),model.neck.register_forward_pre_hook(self.last),model.head.fusion_conv.register_forward_hook(self.decoder)]
    def begin(self):self.enabled=True;self.values={};self.calls=0
    @staticmethod
    def merge_flip(value):
        value=value.detach().float()
        if len(value)==2:return (value[:1]+value[1:].flip(-1))/2
        if len(value)!=1:raise ValueError(('Unexpected prediction batch',value.shape))
        return value
    def middle(self,module,inputs,output):
        if not self.enabled:return
        h,w=self.model.backbone.patch_embed.grid_size
        assert output.shape[1]==h*w,output.shape
        value=output.transpose(1,2).reshape(len(output),output.shape[2],h,w)
        self.values['middle']=self.merge_flip(value)
    def last(self,module,inputs):
        if self.enabled:self.values['last']=self.merge_flip(inputs[0])
    def decoder(self,module,inputs,output):
        if not self.enabled:return
        self.values['decoder']=F.avg_pool2d(self.merge_flip(output),2)
        self.enabled=False;self.calls+=1
        if self.callback is not None:self.callback(self.features())
    def features(self):return [self.values[k] for k in ['middle','last','decoder']]
    def close(self):
        for handle in self.handles:handle.remove()
