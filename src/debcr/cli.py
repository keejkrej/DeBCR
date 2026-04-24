"""Console script entrypoints."""

from __future__ import annotations


def main_train() -> None:
    from .train import main

    main()


def main_predict() -> None:
    from .predict import main

    main()


def main_export_onnx() -> None:
    from .export_onnx import main

    main()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch DeBCR command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Use debcr-train for the full training CLI.")
    subparsers.add_parser("predict", help="Use debcr-predict for the full prediction CLI.")
    subparsers.add_parser("export-onnx", help="Use debcr-export-onnx for the full ONNX export CLI.")
    args, extra = parser.parse_known_args()
    if args.command == "train":
        from .train import main as train_main

        train_main(extra)
    elif args.command == "predict":
        from .predict import main as predict_main

        predict_main(extra)
    elif args.command == "export-onnx":
        from .export_onnx import main as export_onnx_main

        export_onnx_main(extra)
