#!/usr/bin/env bash
set -euo pipefail
cd /root/code/simpleclick+dinov3

python -u evaluate/run_evaluation.py --task slider \
  --config-path evaluate/config_server_eval.yml \
  --protocol-path evaluate/protocols/part_body_g090.json \
  --output-dir outputs/server_slider_eval_vitl_part_body_g090 \
  --checkpoint weights/simpleclick_models/cocolvis_vit_large.pth \
  --datasets "" \
  --pascalpart-targets "" \
  --partimagenet-targets body \
  --noc-thresholds 0.85 \
  --skip-samples-file outputs/server_slider_eval_vitl_part_body_g090/partimagenet_body_g090_skip_samples.txt \
  --resume
