# Runtime environment

The release was checked on Windows with Python 3.10 and an NVIDIA RTX 4060 Laptop
GPU. The source is portable Python; CUDA is the default device, and `--device cpu`
is available for inference. Full GPU regression and evaluation smoke tests were
performed; full CPU performance has not been benchmarked.

Versions used for the release checks:

| Package | Version |
| :--- | :--- |
| torch | 2.12.0+cu126 |
| torchvision | 0.27.0+cu126 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| OpenCV | 4.10.0 |
| scikit-learn | 1.7.2 |
| scikit-image | 0.24.0 |
| timm | 0.6.11 |
| mmcv | 1.6.2 |
| albumentations | 0.5.2 |
| einops | 0.8.2 |
| PyYAML | 6.0.2 |
| pycocotools | 2.0.11 |

Install a mutually compatible torch/torchvision pair for your CUDA driver.
The package dependencies keep the legacy SimpleClick-compatible `timm`, `mmcv`,
and `albumentations` versions. MMCV's compiled CUDA operators are not needed.
`numpy<2` is intentional for compatibility with these packages.

The desktop interface uses Tkinter and Pillow. No browser service is required.
Dataset image decoding supports Unicode filesystem paths, although all repository
filenames, source text, and interface labels are English.

`requirements-repro.txt` pins the numerical and image-processing libraries used
for the checks. Install it after selecting the appropriate torch/torchvision
build:

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e ".[dev]"
```
