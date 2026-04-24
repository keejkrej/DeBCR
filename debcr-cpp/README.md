# DeBCR ONNX C++ Inference

This is a minimal ONNX Runtime integration path for an exported DeBCR model.
It reads a raw `float32` NCHW tensor, runs the model input named `x0`, and writes
the full-resolution output named `pred` as raw `float32`.

Export a model from the Python package first:

```bash
debcr-export-onnx --checkpoint checkpoints/last.pt --output debcr.onnx --height 128 --width-px 128
```

Build with a packaged ONNX Runtime CMake config or set `ONNXRUNTIME_ROOT` to an
ONNX Runtime release directory containing `include/` and `lib/`.

```bash
cmake -S debcr-cpp -B debcr-cpp/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build debcr-cpp/build --config Release
```

Run:

```bash
debcr-cpp debcr.onnx input.f32 output.f32 1 1 128 128
```

The binary is intentionally dependency-light. Real applications can replace the
raw-file loading with their image stack or microscopy data pipeline and keep the
session setup unchanged.
