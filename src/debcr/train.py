"""Training entrypoint for PyTorch Lightning DeBCR."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback

from .data import NPZImageDataset, make_mimo_scales
from .losses import MIMOMSEFFTLoss
from .metrics import psnr
from .models import DeBCR

ImageLoader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _trainer_device_kwargs(name: str) -> dict[str, Any]:
    if name == "auto":
        return {"accelerator": "auto", "devices": "auto"}
    if name == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    if name == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    if name.startswith("cuda:"):
        return {"accelerator": "gpu", "devices": [int(name.split(":", maxsplit=1)[1])]}
    return {"accelerator": name, "devices": 1}


def save_checkpoint(path: Path, model: DeBCR, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config.__dict__,
            "args": vars(args),
        },
        path,
    )


class DeBCRDataModule(L.LightningDataModule):
    """Lightning data module for NPZ low/gt training datasets."""

    def __init__(
        self,
        train_path: str,
        val_path: str | None,
        batch_size: int,
        num_workers: int,
        rescale: bool,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rescale = rescale
        self.pin_memory = pin_memory
        self.train_data: NPZImageDataset | None = None
        self.val_data: NPZImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and self.train_data is None:
            self.train_data = NPZImageDataset(self.train_path, rescale=self.rescale)
        if stage in (None, "fit", "validate") and self.val_path and self.val_data is None:
            self.val_data = NPZImageDataset(self.val_path, rescale=self.rescale)

    def train_dataloader(self) -> ImageLoader:
        if self.train_data is None:
            raise RuntimeError("DeBCRDataModule.setup('fit') must run before train_dataloader().")
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> ImageLoader | list[ImageLoader]:
        if self.val_data is None:
            return []
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class DeBCRLightningModule(L.LightningModule):
    """Lightning module that owns the DeBCR model, loss, metrics, and optimizer."""

    def __init__(self, model: DeBCR, lr: float, fft_weight: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = MIMOMSEFFTLoss(fft_weight=fft_weight)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        low, gt = batch
        targets = make_mimo_scales(gt)
        preds = self.model(low)
        loss = self.criterion(preds, targets)
        batch_size = int(low.shape[0])

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_psnr",
            psnr(preds[0].detach(), gt),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class DeBCRValidationLightningModule(DeBCRLightningModule):
    """DeBCR Lightning module with validation enabled."""

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "val")


class LegacyCheckpointCallback(Callback):
    """Write checkpoints in the existing debcr-train .pt format."""

    def __init__(self, output_dir: str | Path, args: argparse.Namespace, has_val: bool) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.args = args
        self.has_val = has_val
        self.best_val = float("inf")

    def _save(self, trainer: L.Trainer, pl_module: DeBCRLightningModule, name: str) -> None:
        if not trainer.is_global_zero:
            return
        if not trainer.optimizers:
            raise RuntimeError("Cannot save checkpoint before the optimizer is initialized.")
        save_checkpoint(
            self.output_dir / name,
            pl_module.model,
            trainer.optimizers[0],
            trainer.current_epoch + 1,
            self.args,
        )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not isinstance(pl_module, DeBCRLightningModule):
            raise TypeError(f"Expected DeBCRLightningModule, got {type(pl_module)!r}")
        if not self.has_val:
            self._save(trainer, pl_module, "last.pt")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if not isinstance(pl_module, DeBCRLightningModule):
            raise TypeError(f"Expected DeBCRLightningModule, got {type(pl_module)!r}")

        self._save(trainer, pl_module, "last.pt")
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        current = float(val_loss.detach().cpu())
        if current < self.best_val:
            self.best_val = current
            self._save(trainer, pl_module, "best.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PyTorch DeBCR on NPZ low/gt datasets.")
    parser.add_argument("--train", required=True, help="Training NPZ file or directory of NPZ files.")
    parser.add_argument("--val", help="Validation NPZ file or directory of NPZ files.")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for last.pt and best.pt.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--growth", type=int, default=16)
    parser.add_argument("--dense-layers", type=int, default=3)
    parser.add_argument("--fft-weight", type=float, default=0.5)
    parser.add_argument("--no-rescale", action="store_true", help="Disable per-sample [0, 1] rescaling.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = _device(args.device)

    datamodule = DeBCRDataModule(
        train_path=args.train,
        val_path=args.val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescale=not args.no_rescale,
        pin_memory=device.type == "cuda",
    )
    datamodule.setup("fit")
    if datamodule.train_data is None:
        raise RuntimeError("Training data was not initialized.")
    sample_low, sample_gt = datamodule.train_data[0]
    model = DeBCR(
        in_channels=int(sample_low.shape[0]),
        out_channels=int(sample_gt.shape[0]),
        width=args.width,
        blocks=args.blocks,
        growth=args.growth,
        dense_layers=args.dense_layers,
    )
    lightning_module_cls = DeBCRValidationLightningModule if args.val else DeBCRLightningModule
    lightning_model = lightning_module_cls(model, lr=args.lr, fft_weight=args.fft_weight)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[LegacyCheckpointCallback(args.output_dir, args, has_val=args.val is not None)],
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=1,
        num_sanity_val_steps=2 if args.val else 0,
        **_trainer_device_kwargs(args.device),
    )
    if args.val:
        trainer.fit(lightning_model, datamodule=datamodule)
    else:
        trainer.fit(lightning_model, train_dataloaders=datamodule.train_dataloader())


if __name__ == "__main__":
    main()
