"""Metric helpers for DeBCR predictions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def psnr(pred: Tensor, target: Tensor, data_range: float = 1.0, eps: float = 1e-12) -> Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=pred.device, dtype=pred.dtype) / (mse + eps))


def _to_numpy_images(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] <= 4:
        x = np.moveaxis(x, 1, -1)
    if x.ndim == 2:
        x = x[None, ..., None]
    if x.ndim == 3:
        x = x[..., None]
    return x


def ssim(pred: Tensor | np.ndarray, target: Tensor | np.ndarray, data_range: float = 1.0) -> float:
    """Average SSIM over a batch.

    ``scikit-image`` is used when available. A clear error is raised otherwise.
    """

    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError("SSIM requires scikit-image to be installed") from exc

    pred_np = _to_numpy_images(pred)
    target_np = _to_numpy_images(target)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Shape mismatch: {pred_np.shape} vs {target_np.shape}")

    scores: list[float] = []
    for pred_img, target_img in zip(pred_np, target_np):
        if pred_img.shape[-1] == 1:
            scores.append(
                float(
                    structural_similarity(
                        target_img[..., 0],
                        pred_img[..., 0],
                        data_range=data_range,
                    )
                )
            )
        else:
            scores.append(
                float(
                    structural_similarity(
                        target_img,
                        pred_img,
                        channel_axis=-1,
                        data_range=data_range,
                    )
                )
            )
    return float(np.mean(scores)) if scores else math.nan


@torch.no_grad()
def restoration_metrics(pred: Tensor, target: Tensor, data_range: float = 1.0) -> dict[str, Any]:
    return {
        "psnr": float(psnr(pred, target, data_range=data_range).detach().cpu()),
        "rmse": float(rmse(pred, target).detach().cpu()),
        "ssim": ssim(pred, target, data_range=data_range),
    }

