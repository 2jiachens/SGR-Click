#!/usr/bin/env bash
set -euo pipefail
cd /root/code/simpleclick+dinov3

python -u evaluate/run_slider_evaluation.py \
  --config-path evaluate/config_server_eval.yml \
  --protocol-path evaluate/protocols/sbd_tail_g050_g100.json \
  --output-dir outputs/server_slider_eval_vitl_sbd_tail \
  --checkpoint weights/simpleclick_models/cocolvis_vit_large.pth \
  --datasets SBD \
  --pascalpart-targets "" \
  --partimagenet-targets "" \
  --noc-thresholds 0.85,0.90 \
  --resume
