#!/usr/bin/env bash
set -euo pipefail
cd /root/code/simpleclick+dinov3

python -u evaluate/run_evaluation.py --task slider \
  --config-path evaluate/config_server_eval.yml \
  --protocol-path evaluate/protocols/all_sliders.json \
  --output-dir outputs/server_slider_eval \
  --checkpoint weights/simpleclick_models/cocolvis_vit_base.pth \
  --datasets "" \
  --pascalpart-targets "" \
  --partimagenet-targets head,body,foot,all \
  --noc-thresholds 0.85 \
  --skip-samples-file outputs/server_slider_eval/partimagenet_skip_samples.txt \
  --resume
