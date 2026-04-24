"""Prediction entrypoint for PyTorch DeBCR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import collect_npz_files, load_npz_low
from .models import DeBCR


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Lightning checkpoint from debcr-train: {path}")

    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    config = dict(hyper_parameters.get("model_config") or {})
    if not config:
        raise ValueError(f"Checkpoint is missing model_config hyperparameters: {path}")

    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    if not state_dict:
        raise ValueError(f"Checkpoint has no DeBCR model weights: {path}")
    return state_dict, config


def _to_output_array(pred: torch.Tensor) -> np.ndarray:
    arr = pred.detach().cpu().numpy()
    if arr.shape[1] == 1:
        return arr[:, 0]
    return np.moveaxis(arr, 1, -1)


def predict_file(
    model: DeBCR,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    rescale: bool,
) -> None:
    low = torch.from_numpy(load_npz_low(input_path, rescale=rescale))
    loader = DataLoader(TensorDataset(low), batch_size=batch_size, shuffle=False)
    preds: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            preds.append(model(batch)[0].cpu())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, pred=_to_output_array(torch.cat(preds, dim=0)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PyTorch DeBCR prediction on NPZ low arrays.")
    parser.add_argument("--input", required=True, help="Single NPZ file or directory of NPZ files.")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint from debcr-train.")
    parser.add_argument("--output-dir", default="results", help="Directory for results*.npz files.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--no-rescale", action="store_true", help="Disable per-sample [0, 1] rescaling.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = _device(args.device)
    files = collect_npz_files(args.input)

    state_dict, config = _load_checkpoint(args.checkpoint, device)
    first_low = load_npz_low(files[0], rescale=not args.no_rescale)
    if int(first_low.shape[1]) != int(config["in_channels"]):
        raise ValueError(
            f"Input has {first_low.shape[1]} channel(s), but checkpoint expects {config['in_channels']}."
        )

    model = DeBCR(**config).to(device)
    model.load_state_dict(state_dict)

    output_dir = Path(args.output_dir)
    for idx, path in enumerate(files):
        suffix = "" if len(files) == 1 else f"_{idx}"
        output_path = output_dir / f"results{suffix}.npz"
        predict_file(
            model,
            path,
            output_path,
            device=device,
            batch_size=args.batch_size,
            rescale=not args.no_rescale,
        )
        print(f"{path} -> {output_path}")


if __name__ == "__main__":
    main()
