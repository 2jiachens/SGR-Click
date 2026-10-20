# Model weights

## SGR-Click-Lite

The released semantic head is included as `checkpoints/sgr_click_lite.pt`.
It contains the context-head `state_dict`, architecture name, and epoch (16).
No DINOv3 weights are needed to run Lite.

Download **cocolvis_vit_base.pth** from the official
[SimpleClick model release](https://github.com/uncbiag/SimpleClick#model-weights)
and place it in `checkpoints/`. This is the COCO+LVIS ViT-B checkpoint, not the
SBD checkpoint or a larger ViT variant.

```text
checkpoints/
  cocolvis_vit_base.pth
  sgr_click_lite.pt
  manifest.json
```

```bash
python scripts/verify_weights.py --model lite
sgr-click demo --model lite
```

## SGR-Click-TF

TF uses the same interaction checkpoint plus DINOv3 ViT-L/16:

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
git -C third_party/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
```

Obtain `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` through the
[official DINOv3 pretrained model instructions](https://github.com/facebookresearch/dinov3#pretrained-models).
Follow the upstream model access and license terms. Place the file in
`checkpoints/`, then run:

```bash
python scripts/verify_weights.py --model tf
sgr-click demo --model tf
```

The Lite head is not needed for TF. The checksums and byte counts of all three
checkpoints are published in `checkpoints/manifest.json`.

Custom locations can be passed to every demo and evaluation command:

```bash
sgr-click demo --model tf \
  --backbone /path/to/cocolvis_vit_base.pth \
  --dino-repo /path/to/dinov3 \
  --dino-weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The interaction checkpoint uses the upstream SimpleClick serialization format.
Only load checkpoint files from trusted sources. The Lite head is loaded with
PyTorch's restricted `weights_only=True` loader.
