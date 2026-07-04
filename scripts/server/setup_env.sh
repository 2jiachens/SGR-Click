#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-simpleclick-dinov3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu121}"
REQ_FILE="$PROJECT_ROOT/scripts/server/requirements-server.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda or Mambaforge on the server first." >&2
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Missing requirements file: $REQ_FILE" >&2
  exit 1
fi

case "$TORCH_CHANNEL" in
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  cu118)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    ;;
  cu121)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  cu124)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    ;;
  *)
    echo "Unsupported TORCH_CHANNEL: $TORCH_CHANNEL" >&2
    echo "Use one of: cpu, cu118, cu121, cu124" >&2
    exit 1
    ;;
esac

if ! conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

run_in_env() {
  conda run --no-capture-output -n "$ENV_NAME" "$@"
}

run_in_env python -m pip install --upgrade pip wheel
run_in_env python -m pip install "setuptools<70"
run_in_env python -m pip install \
  torch==2.3.1 \
  torchvision==0.18.1 \
  --index-url "$TORCH_INDEX_URL"
run_in_env python -m pip install --no-build-isolation mmcv==1.6.2
run_in_env python -m pip install -r "$REQ_FILE"

# Verify the core runtime and force the Cython distance-map helper to build once.
(
  cd "$PROJECT_ROOT"
  run_in_env python -c "import torch, torchvision, cv2, albumentations, timm, scipy, skimage; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); import isegm.utils.cython.dist_maps as _dist_maps; import isegm.inference.dino_granularity as _dino; import train_granularity_refine as _train; print('imports_ok', True)"
)

cat <<EOF
Environment is ready.

Activate it with:
  conda activate $ENV_NAME

Recommended next steps on the server:
  source "$PROJECT_ROOT/scripts/server/env.simpleclick_dinov3.sh"
  bash "$PROJECT_ROOT/scripts/server/run_granularity_refine_pipeline.sh" preview
EOF
