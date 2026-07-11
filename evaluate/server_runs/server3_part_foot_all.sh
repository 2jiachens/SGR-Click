#!/usr/bin/env bash
set -euo pipefail
cd /root/code/simpleclick+dinov3

python -u evaluate/run_slider_evaluation.py \
  --config-path evaluate/config_server_eval.yml \
  --protocol-path evaluate/protocols/all_sliders.json \
  --output-dir outputs/server_slider_eval_vitl_part_foot_all \
  --checkpoint weights/simpleclick_models/cocolvis_vit_large.pth \
  --datasets "" \
  --pascalpart-targets "" \
  --partimagenet-targets foot,all \
  --noc-thresholds 0.85 \
  --skip-samples-file outputs/server_slider_eval_vitl_part_foot_all/partimagenet_foot_all_skip_samples.txt \
  --resume
