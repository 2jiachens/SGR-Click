# Server Deployment And Training

This project can run directly on a Linux GPU server. The main requirements are:

- Python 3.10 or 3.11
- CUDA-capable PyTorch matching the server driver
- the local DINOv3 repository checked out on the server
- SimpleClick checkpoint and DINOv3 weights present on the server
- SBD extracted as `benchmark_RELEASE/dataset`

## Recommended server layout

```text
/data/projects/simpleclick+dinov3
/data/datasets/SBD/benchmark_RELEASE/dataset
/data/third_party/dinov3
```

If you are already using VS Code Remote SSH, the simplest deployment path is:

```bash
cd /data/projects
git clone <your-repo-url> simpleclick+dinov3
cd simpleclick+dinov3
```

If the server does not pull directly from git, sync the current workspace from Windows:

```bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'weights' \
  --exclude 'experiments' \
  /local/path/simpleclick+dinov3/ \
  user@server:/data/projects/simpleclick+dinov3/
```

Then upload the model weights separately to:

```text
/data/projects/simpleclick+dinov3/weights/simpleclick_models/cocolvis_vit_base.pth
/data/projects/simpleclick+dinov3/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## Environment setup

Create a clean conda environment on the server:

```bash
conda create -n simpleclick-dinov3 python=3.10 -y
conda activate simpleclick-dinov3
```

Install PyTorch first with the wheel matching the server CUDA runtime. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```bash
cd /data/projects/simpleclick+dinov3
pip install -r requirements.txt
```

The dataset builder now resolves defaults from repository-relative paths or these environment variables:

- `SIMPLECLICK_CHECKPOINT`
- `DINOV3_REPO`
- `DINOV3_WEIGHTS`

## Quick preview build

Run a small preview first and inspect `previews/` before launching the full build:

```bash
cd /data/projects/simpleclick+dinov3
export SBD_ROOT=/data/datasets/SBD/benchmark_RELEASE/dataset
export DINOV3_REPO=/data/third_party/dinov3

python granularity_recovery/build_granularity_recovery_dataset.py \
  --sbd-root "$SBD_ROOT" \
  --output-root outputs/SBD_granularity_recovery_preview \
  --max-images-per-split 2 \
  --preview-per-split 2 \
  --device cuda:0
```

Preview output defaults to:

```text
/data/projects/simpleclick+dinov3/outputs/SBD_granularity_recovery_preview
```

## Full dataset build

```bash
cd /data/projects/simpleclick+dinov3
mkdir -p logs
export SBD_ROOT=/data/datasets/SBD/benchmark_RELEASE/dataset
export DINOV3_REPO=/data/third_party/dinov3
export OUTPUT_ROOT=/data/projects/simpleclick+dinov3/outputs/SBD_granularity_recovery_full
export DEVICE=cuda:0

nohup python granularity_recovery/build_granularity_recovery_dataset.py \
  --sbd-root "$SBD_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --log-every-images 25 \
  --device "$DEVICE" \
  > logs/build_granularity_dataset.log 2>&1 &
tail -f logs/build_granularity_dataset.log
```

The builder is resumable. Re-running the same command continues from existing `.npz` files.

## Training

After the full build finishes:

```bash
cd /data/projects/simpleclick+dinov3
mkdir -p logs
export OUTPUT_ROOT=/data/projects/simpleclick+dinov3/outputs/SBD_granularity_recovery_full
export EXP_ROOT=/data/projects/simpleclick+dinov3/experiments/granularity_recovery
export DEVICE=cuda:0
export TRAIN_BATCH_SIZE=2
export TRAIN_EPOCHS=20

nohup python granularity_recovery/train_granularity_recovery.py \
  --dataset-root "$OUTPUT_ROOT" \
  --exp-root "$EXP_ROOT" \
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --device "$DEVICE" \
  --save-vis \
  > logs/train_granularity_recovery.log 2>&1 &
tail -f logs/train_granularity_recovery.log
```

Checkpoints are written to:

```text
/data/projects/simpleclick+dinov3/experiments/granularity_recovery
```

## Single-command end-to-end run

If preview inspection is already done and paths are correct:

```bash
cd /data/projects/simpleclick+dinov3
mkdir -p logs
export SBD_ROOT=/data/datasets/SBD/benchmark_RELEASE/dataset
export DINOV3_REPO=/data/third_party/dinov3

nohup python granularity_recovery/build_granularity_recovery_dataset.py \
  --sbd-root "$SBD_ROOT" \
  --output-root outputs/SBD_granularity_recovery_full \
  --log-every-images 25 \
  --device cuda:0 \
  > logs/granularity_pipeline.log 2>&1 &
tail -f logs/granularity_pipeline.log
```

## Common failure points

- `third_party/dinov3` missing on the server: pass `DINOV3_REPO` explicitly.
- PyTorch CUDA wheel mismatch: reinstall `torch` and `torchvision` for the server driver.
- `tkinter` missing: only affects `demo.py`, not dataset build or training.
- OOM during training: lower `TRAIN_BATCH_SIZE` to `1`.
