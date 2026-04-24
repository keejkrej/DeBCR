# DeBCR ONNX Rust Inference

This is a minimal Rust integration path for an exported DeBCR ONNX model. It
uses the `ort` crate, reads a raw `float32` NCHW tensor, runs the model input
named `x0`, and writes the full-resolution output named `pred` as raw
`float32`.

Export a model from the Python package first:

```bash
debcr-export-onnx --checkpoint checkpoints/last.pt --output debcr.onnx --height 128 --width-px 128
```

Run:

```bash
cargo run --release -- debcr.onnx input.f32 output.f32 1 1 128 128
```

The example keeps I/O deliberately simple. Applications can replace the raw-file
loading with their own microscopy image pipeline and reuse the session setup.
