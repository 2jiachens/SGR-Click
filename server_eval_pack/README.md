# Server Evaluation Pack

This folder fixes the requested server-side evaluation protocol.

After extracting the archive, use this layout:

```text
<bundle_root>/
  simpleclick+dinov3/
  GrabCut/
  Berkeley/
  DAVIS345/
  SBD/
  VOCdevkit/
  PascalPart/
  PartImageNet/
```

Run evaluation from `simpleclick+dinov3` and use:

```bash
--config-path server_eval_pack/config_server_eval.yml
```

Protocol:

- Slider values: `0.0, 0.1, ..., 1.0`.
- Metrics per slider: `NoC@85` and `IoU@1`.
- Standard datasets: GrabCut, Berkeley, SBD val, DAVIS, Pascal VOC val.
- PascalPart targets: `head`, `wheel`, `torso`, and `all`.
- PartImageNet targets: `head`, `body`, `foot`, and `all`.
- `all` subsets are deterministic random 1000-sample lists with seed `20260706`.

PascalPart category policy:

- Use exact top-level part labels only.
- `head` does not include `eye`, `nose`, `mouth`, etc.
- `wheel` does not include `fwheel` or `bwheel`.
- When multiple exact matches exist for the same image and class, keep the largest mask.

PartImageNet data:

- `partimagenet_head_body_foot_eval` contains binary masks and a manifest extracted from PartImageNet `test`.
- Per-class CSV manifests and the deterministic `all` subset are in `sample_keys/`.
