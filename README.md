# DeBCR

Denoising, deblurring, and optical deconvolution using a physics-informed neural
network for light microscopy.

This branch contains a PyTorch-only DeBCR package. TensorFlow/Keras runtime code,
vendored CSBDeep code, and TensorFlow checkpoint weights have been removed from
the package path. Existing TensorFlow checkpoints are not converted; recover them
from the original `main` branch if needed.

## Installation

The project uses `pyproject.toml`, Hatchling, and `uv`. On non-macOS platforms,
`uv` is configured to resolve `torch` and `torchvision` from the PyTorch CUDA
`cu130` wheel index.

```bash
uv sync
```

For editable development in an existing environment:

```bash
uv pip install -e .
```

## Data Format

Training and prediction use NumPy `.npz` files.

Training files must contain:

- `low`: degraded input images
- `gt`: ground truth images

Prediction files must contain:

- `low`: degraded input images

Supported layouts are:

- `H, W`
- `N, H, W`
- `N, H, W, C`
- `N, C, H, W`

Arrays are converted to channel-first PyTorch tensors. By default, every sample
and channel is min/max rescaled to `[0, 1]`.

## Training

Training uses PyTorch Lightning.

```bash
debcr-train \
  --train data/2D_denoising/train \
  --val data/2D_denoising/val \
  --output-dir checkpoints/2D_denoising \
  --epochs 100 \
  --batch-size 4
```

Checkpoints are written as Lightning `.ckpt` files. `last.ckpt` is always
written; when validation data is provided, the lowest-`val_loss` checkpoint is
written as `best.ckpt`.

Useful options:

```bash
debcr-train --help
```

## Prediction

Run prediction on one `.npz` file or a directory of `.npz` files:

```bash
debcr-predict \
  --input data/2D_denoising/test \
  --checkpoint checkpoints/2D_denoising/best.ckpt \
  --output-dir results/2D_denoising
```

Outputs are written as `results*.npz` files containing:

- `pred`: restored full-resolution prediction

Useful options:

```bash
debcr-predict --help
```

## ONNX Export

Export trained Lightning checkpoints for native integration:

```bash
debcr-export-onnx \
  --checkpoint checkpoints/2D_denoising/best.ckpt \
  --output debcr.onnx \
  --height 128 \
  --width-px 128
```

The default ONNX graph has:

- input `x0`: `float32` tensor shaped `N, C, H, W`
- output `pred`: full-resolution `float32` tensor shaped `N, C, H, W`

Batch, height, and width are dynamic by default. Use `--fixed-shape` for a
static graph. Use `--mimo` to export all three model outputs as `z0`, `z2`, and
`z4`.

ONNX export requires the optional dependencies:

```bash
uv sync --extra onnx
```

## Native Integration

Two minimal ONNX inference paths are included:

- [debcr-cpp](debcr-cpp/README.md): C++17 + ONNX Runtime
- [debcr-rs](debcr-rs/README.md): Rust + `ort`

Both examples read raw `float32` NCHW input, run `x0 -> pred`, and write raw
`float32` output. They are intentionally small so application code can replace
the raw-file I/O with microscopy image loading while keeping the ONNX session
setup.

## Python API

```python
import torch
from debcr import DeBCR
from debcr.losses import MIMOMSEFFTLoss

model = DeBCR()
x0 = torch.rand(2, 1, 128, 128)

# Convenience path: half and quarter scales are derived internally.
z0, z2, z4 = model(x0)

# Explicit path.
x2 = torch.nn.functional.avg_pool2d(x0, kernel_size=2, stride=2)
x4 = torch.nn.functional.avg_pool2d(x0, kernel_size=4, stride=4)
z0, z2, z4 = model(x0, x2, x4)

criterion = MIMOMSEFFTLoss()
loss = criterion([z0, z2, z4], [x0, x2, x4])
loss.backward()
```

## Citation

If you use this software, please cite the project metadata in
[CITATION.cff](CITATION.CFF).

## License

DeBCR is licensed under the [MIT license](LICENSE).
