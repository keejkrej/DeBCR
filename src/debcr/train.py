"""Training entrypoint for PyTorch DeBCR."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NPZImageDataset, make_mimo_scales
from .losses import MIMOMSEFFTLoss
from .metrics import psnr
from .models import DeBCR


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def run_epoch(
    model: DeBCR,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: MIMOMSEFFTLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_psnr = 0.0
    total_items = 0

    for low, gt in tqdm(loader, leave=False):
        low = low.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        targets = make_mimo_scales(gt)

        if train:
            optimizer.zero_grad(set_to_none=True)
        preds = model(low)
        loss = criterion(preds, targets)
        if train:
            loss.backward()
            optimizer.step()

        batch = int(low.shape[0])
        total_loss += float(loss.detach().cpu()) * batch
        total_psnr += float(psnr(preds[0].detach(), gt).detach().cpu()) * batch
        total_items += batch

    return total_loss / max(total_items, 1), total_psnr / max(total_items, 1)


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

    train_data = NPZImageDataset(args.train, rescale=not args.no_rescale)
    sample_low, sample_gt = train_data[0]
    model = DeBCR(
        in_channels=int(sample_low.shape[0]),
        out_channels=int(sample_gt.shape[0]),
        width=args.width,
        blocks=args.blocks,
        growth=args.growth,
        dense_layers=args.dense_layers,
    ).to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if args.val:
        val_loader = DataLoader(
            NPZImageDataset(args.val, rescale=not args.no_rescale),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MIMOMSEFFTLoss(fft_weight=args.fft_weight)
    output_dir = Path(args.output_dir)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_psnr = run_epoch(model, train_loader, criterion, device, optimizer)
        line = f"epoch {epoch:04d} train_loss={train_loss:.6f} train_psnr={train_psnr:.3f}"
        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_psnr = run_epoch(model, val_loader, criterion, device)
            line += f" val_loss={val_loss:.6f} val_psnr={val_psnr:.3f}"
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args)
        print(line)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, args)


if __name__ == "__main__":
    main()

