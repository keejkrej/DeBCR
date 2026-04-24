"""NPZ data loading utilities for DeBCR."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset


def collect_npz_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() != ".npz":
            raise ValueError(f"Expected an .npz file, got {root}")
        return [root]
    if root.is_dir():
        files = sorted(root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {root}")
        return files
    raise FileNotFoundError(root)


def to_nchw(array: np.ndarray) -> np.ndarray:
    """Convert common microscopy array layouts to ``N, C, H, W``."""

    x = np.asarray(array)
    if x.ndim == 2:
        x = x[None, None, :, :]
    elif x.ndim == 3:
        x = x[:, None, :, :]
    elif x.ndim == 4:
        if x.shape[1] <= 4 and x.shape[-1] > 4:
            pass
        elif x.shape[-1] <= 4:
            x = np.moveaxis(x, -1, 1)
        else:
            raise ValueError(
                "Ambiguous 4D array layout; expected NCHW with C<=4 or NHWC with C<=4"
            )
    else:
        raise ValueError(f"Expected a 2D, 3D, or 4D array, got shape {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float32)


def rescale_01(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-sample, per-channel min/max rescaling to ``[0, 1]``."""

    x = np.asarray(array, dtype=np.float32)
    mn = np.amin(x, axis=(-2, -1), keepdims=True)
    mx = np.amax(x, axis=(-2, -1), keepdims=True)
    return np.nan_to_num((x - mn) / np.maximum(mx - mn, eps)).astype(np.float32)


def load_npz_pair(path: str | Path, rescale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "low" not in data or "gt" not in data:
            raise KeyError(f"{path} must contain 'low' and 'gt' arrays")
        low = to_nchw(data["low"])
        gt = to_nchw(data["gt"])
    if low.shape != gt.shape:
        raise ValueError(f"{path} low/gt shape mismatch: {low.shape} vs {gt.shape}")
    if rescale:
        low = rescale_01(low)
        gt = rescale_01(gt)
    return low, gt


def load_npz_low(path: str | Path, rescale: bool = True) -> np.ndarray:
    with np.load(path) as data:
        if "low" not in data:
            raise KeyError(f"{path} must contain a 'low' array")
        low = to_nchw(data["low"])
    return rescale_01(low) if rescale else low


def make_mimo_scales(x: Tensor) -> list[Tensor]:
    """Return full, half, and quarter image tensors."""

    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(x.shape)}")
    return [
        x,
        F.avg_pool2d(x, kernel_size=2, stride=2),
        F.avg_pool2d(x, kernel_size=4, stride=4),
    ]


class NPZImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset for one or more NPZ files containing ``low`` and ``gt`` arrays."""

    def __init__(self, paths: str | Path | Iterable[str | Path], rescale: bool = True) -> None:
        if isinstance(paths, (str, Path)):
            files = collect_npz_files(paths)
        else:
            files = [Path(path) for path in paths]
        lows: list[np.ndarray] = []
        gts: list[np.ndarray] = []
        for path in files:
            low, gt = load_npz_pair(path, rescale=rescale)
            lows.append(low)
            gts.append(gt)
        self.low = torch.from_numpy(np.concatenate(lows, axis=0))
        self.gt = torch.from_numpy(np.concatenate(gts, axis=0))

    def __len__(self) -> int:
        return int(self.low.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.low[index], self.gt[index]
