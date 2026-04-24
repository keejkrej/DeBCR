"""Loss functions for PyTorch DeBCR training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def mse_fft_loss(target: Tensor, pred: Tensor, fft_weight: float = 0.5) -> Tensor:
    """MSE plus an L1 frequency-domain term, matching the legacy DeBCR loss."""

    mse = F.mse_loss(pred, target)
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    fft = torch.mean(torch.abs(target_fft - pred_fft))
    return mse + fft_weight * fft


def mimo_mse_fft_loss(
    targets: Sequence[Tensor],
    preds: Sequence[Tensor],
    fft_weight: float = 0.5,
) -> Tensor:
    """Sum ``mse_fft_loss`` over the full, half, and quarter outputs."""

    if len(targets) != len(preds):
        raise ValueError(f"Expected equal target/pred lengths, got {len(targets)} and {len(preds)}")
    return sum(mse_fft_loss(y, y_hat, fft_weight=fft_weight) for y, y_hat in zip(targets, preds))


class MIMOMSEFFTLoss(nn.Module):
    """Module wrapper for the DeBCR MIMO loss."""

    def __init__(self, fft_weight: float = 0.5) -> None:
        super().__init__()
        self.fft_weight = fft_weight

    def forward(self, preds: Sequence[Tensor], targets: Sequence[Tensor]) -> Tensor:
        return mimo_mse_fft_loss(targets, preds, fft_weight=self.fft_weight)

