"""Export DeBCR checkpoints to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from .models import DeBCR


class FullResolutionWrapper(nn.Module):
    """ONNX-friendly wrapper returning only the full-resolution prediction."""

    def __init__(self, model: DeBCR) -> None:
        super().__init__()
        self.model = model

    def forward(self, x0: Tensor) -> Tensor:
        return self.model(x0)[0]


class MIMOWrapper(nn.Module):
    """ONNX-friendly wrapper returning all DeBCR scales."""

    def __init__(self, model: DeBCR) -> None:
        super().__init__()
        self.model = model

    def forward(self, x0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z0, z2, z4 = self.model(x0)
        return z0, z2, z4


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(checkpoint_path: str | Path, device: torch.device) -> DeBCR:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Lightning checkpoint from debcr-train: {checkpoint_path}")

    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    config = dict(hyper_parameters.get("model_config") or {})
    if not config:
        raise ValueError(f"Checkpoint is missing model_config hyperparameters: {checkpoint_path}")

    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    if not state_dict:
        raise ValueError(f"Checkpoint has no DeBCR model weights: {checkpoint_path}")

    model = DeBCR(**config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Lightning DeBCR checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint from debcr-train.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--height", type=int, default=128, help="Example input height for tracing.")
    parser.add_argument("--width-px", type=int, default=128, help="Example input width for tracing.")
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch size for tracing.")
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda', 'auto', or another torch device string.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--mimo", action="store_true", help="Export z0, z2, and z4 outputs instead of only pred.")
    parser.add_argument("--fixed-shape", action="store_true", help="Disable dynamic batch/height/width axes.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = _device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=device)
    wrapper: nn.Module = MIMOWrapper(model) if args.mimo else FullResolutionWrapper(model)
    wrapper.eval()

    example = torch.randn(
        args.batch_size,
        model.config.in_channels,
        args.height,
        args.width_px,
        device=device,
    )
    output_names = ["z0", "z2", "z4"] if args.mimo else ["pred"]
    dynamic_axes = None
    if not args.fixed_shape:
        dynamic_axes = {"x0": {0: "batch", 2: "height", 3: "width"}}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})
        dynamic_axes[output_names[0]].update({2: "height", 3: "width"})

    torch.onnx.export(
        wrapper,
        example,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["x0"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
