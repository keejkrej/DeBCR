use std::{env, fs, path::Path};

use anyhow::{bail, Context, Result};
use ndarray::Array4;
use ort::{inputs, session::Session, value::Tensor};

fn parse_dim(text: &str, name: &str) -> Result<usize> {
    let value: usize = text
        .parse()
        .with_context(|| format!("invalid {name} dimension"))?;
    if value == 0 {
        bail!("{name} must be positive");
    }
    Ok(value)
}

fn read_f32_file(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let expected = count
        .checked_mul(std::mem::size_of::<f32>())
        .context("tensor byte count overflow")?;
    if bytes.len() != expected {
        bail!(
            "{} has {} bytes, expected {} bytes",
            path.display(),
            bytes.len(),
            expected
        );
    }

    let mut values = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn write_f32_file(path: &Path, values: impl IntoIterator<Item = f32>) -> Result<()> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 8 {
        bail!("usage: debcr-rs <model.onnx> <input.f32> <output.f32> <N> <C> <H> <W>");
    }

    let model_path = Path::new(&args[1]);
    let input_path = Path::new(&args[2]);
    let output_path = Path::new(&args[3]);
    let n = parse_dim(&args[4], "N")?;
    let c = parse_dim(&args[5], "C")?;
    let h = parse_dim(&args[6], "H")?;
    let w = parse_dim(&args[7], "W")?;
    let count = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w))
        .context("tensor element count overflow")?;

    let values = read_f32_file(input_path, count)?;
    let input = Array4::from_shape_vec((n, c, h, w), values).context("invalid input shape")?;
    let input_tensor = Tensor::from_array(input).context("failed to create ONNX input tensor")?;

    let mut session = Session::builder()
        .context("failed to create ONNX Runtime session builder")?
        .commit_from_file(model_path)
        .with_context(|| format!("failed to load {}", model_path.display()))?;

    let outputs = session
        .run(inputs! {
            "x0" => input_tensor
        })
        .context("ONNX inference failed")?;
    let (_shape, pred) = outputs["pred"]
        .try_extract_tensor::<f32>()
        .context("failed to read pred output tensor")?;

    write_f32_file(output_path, pred.iter().copied())?;
    println!("wrote {} float32 values to {}", pred.len(), output_path.display());
    Ok(())
}
