use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use rusttorch::{
    Device, Kind, Result, Tensor,
    interop::{load_state_dict, save_state_dict},
    nn, optim,
};
use serde_json::{Value, json};
use tch::nn::VarStore;

#[test]
#[ignore = "run through scripts/run-python-parity.sh"]
fn bidirectional_python_parity() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference_dir = PathBuf::from(
        std::env::var_os("RUSTTORCH_PYTHON_REFERENCE_DIR")
            .expect("run this ignored test through scripts/run-python-parity.sh"),
    );
    let reference: Value = serde_json::from_slice(
        &fs::read(reference_dir.join("reference.json")).expect("read Python reference JSON"),
    )
    .expect("parse Python reference JSON");
    let input = tensor_from_json(&reference["input"], true);
    let weights = reference_dir.join("pytorch_linear.safetensors");

    let (var_store, layer) = loaded_linear(&weights)?;
    let output = layer.forward(&input)?;
    let linear_loss = output.f_square()?.f_mean(Kind::Float)?;
    linear_loss.backward();

    let targets = Tensor::from_slice(&[2i64, 0]);
    let plain_input = tensor_from_json(&reference["input"], false);
    let logits = layer.forward(&plain_input)?;
    let cross_entropy = nn::functional::cross_entropy(&logits, &targets)?;
    let mse_target = Tensor::from_slice(&[0.0f32, 0.5, -0.5, 1.0, -1.0, 0.25]).view([2, 3]);
    let mse = nn::functional::mse_loss(&logits, &mse_target)?;

    let linear = json!({
        "forward": tensor_json(&output),
        "loss": linear_loss.double_value(&[]),
        "input_grad": tensor_json(&input.grad()),
        "weight_grad": tensor_json(&layer.weight().grad()),
        "bias_grad": tensor_json(&layer.bias().expect("linear bias").grad()),
    });
    let sgd = optimizer_step(&weights, &reference, OptimizerKind::Sgd)?;
    let adam = optimizer_step(&weights, &reference, OptimizerKind::Adam)?;
    let residual = residual_result(&reference)?;

    let rust_results = json!({
        "linear": linear,
        "losses": {
            "cross_entropy": cross_entropy.double_value(&[]),
            "mse": mse.double_value(&[]),
        },
        "sgd": sgd,
        "adam": adam,
        "residual": residual,
    });
    let rust_results_path = reference_dir.join("rust_results.json");
    fs::write(
        &rust_results_path,
        serde_json::to_vec_pretty(&rust_results).expect("serialize Rust parity results"),
    )
    .expect("write Rust parity results");

    let rust_weights = reference_dir.join("rusttorch_linear.safetensors");
    save_state_dict(&rust_weights, &var_store)?;
    let rust_io = reference_dir.join("rust_linear_io.json");
    fs::write(
        &rust_io,
        serde_json::to_vec_pretty(&json!({
            "input": reference["input"],
            "output": tensor_json(&output),
        }))
        .expect("serialize Rust linear IO"),
    )
    .expect("write Rust linear IO");

    run_python(
        &root,
        "scripts/verify-rusttorch-load.py",
        &[&rust_results_path, &reference_dir.join("reference.json")],
    );
    run_python(
        &root,
        "scripts/verify-pytorch-load.py",
        &[&rust_weights, &rust_io],
    );
    Ok(())
}

fn loaded_linear(weights: &Path) -> Result<(VarStore, nn::Linear)> {
    let var_store = VarStore::new(Device::Cpu);
    let layer = nn::LinearConfig::new(4, 3).build(&var_store.root())?;
    let report = load_state_dict(weights, &var_store)?;
    assert_eq!(report.loaded, ["bias", "weight"]);
    Ok((var_store, layer))
}

#[derive(Clone, Copy)]
enum OptimizerKind {
    Sgd,
    Adam,
}

fn optimizer_step(weights: &Path, reference: &Value, kind: OptimizerKind) -> Result<Value> {
    let (var_store, layer) = loaded_linear(weights)?;
    let input = tensor_from_json(&reference["input"], false);
    let target = Tensor::from_slice(&[2i64, 0]);
    let loss = nn::functional::cross_entropy(&layer.forward(&input)?, &target)?;
    match kind {
        OptimizerKind::Sgd => optim::Sgd::builder()
            .learning_rate(0.05)
            .build(&var_store)?
            .backward_step(&loss)?,
        OptimizerKind::Adam => optim::Adam::builder()
            .learning_rate(0.01)
            .build(&var_store)?
            .backward_step(&loss)?,
    }
    Ok(json!({
        "loss": loss.double_value(&[]),
        "state": state_json(&var_store),
    }))
}

fn residual_result(reference: &Value) -> Result<Value> {
    let var_store = VarStore::new(Device::Cpu);
    let main = nn::LinearConfig::new(4, 4)
        .bias(false)
        .build(&(var_store.root() / "main"))?;
    let skip = nn::LinearConfig::new(4, 4)
        .bias(false)
        .build(&(var_store.root() / "skip"))?;
    assign(
        main.weight(),
        &[
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5,
        ],
        [4, 4],
    )?;
    assign(
        skip.weight(),
        &[
            0.0, 0.1, 0.0, -0.2, 0.3, 0.0, 0.2, 0.0, 0.0, -0.4, 0.0, 0.1, 0.2, 0.0, -0.3, 0.0,
        ],
        [4, 4],
    )?;
    let input = tensor_from_json(&reference["input"], true);
    let output = nn::functional::relu(&main.forward(&input)?)?.f_add(&skip.forward(&input)?)?;
    let loss = output.f_square()?.f_mean(Kind::Float)?;
    loss.backward();
    Ok(json!({
        "forward": tensor_json(&output),
        "loss": loss.double_value(&[]),
        "input_grad": tensor_json(&input.grad()),
        "parameter_grads": {
            "main.weight": tensor_json(&main.weight().grad()),
            "skip.weight": tensor_json(&skip.weight().grad()),
        },
    }))
}

fn assign<const N: usize>(tensor: &Tensor, values: &[f32], shape: [i64; N]) -> Result<()> {
    let source = Tensor::from_slice(values).view(shape.as_slice());
    let mut destination = tensor.shallow_clone();
    rusttorch::no_grad(|| destination.f_copy_(&source))?;
    Ok(())
}

fn state_json(var_store: &VarStore) -> Value {
    var_store
        .variables()
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|(name, tensor)| (name, tensor_json(&tensor)))
        .collect()
}

fn tensor_from_json(value: &Value, requires_grad: bool) -> Tensor {
    let rows = value.as_array().expect("tensor JSON rows");
    let columns = rows
        .first()
        .expect("non-empty tensor")
        .as_array()
        .expect("tensor JSON columns");
    let flat = rows
        .iter()
        .flat_map(|row| row.as_array().expect("tensor JSON row"))
        .map(|value| value.as_f64().expect("numeric tensor JSON") as f32)
        .collect::<Vec<_>>();
    Tensor::from_slice(&flat)
        .view([rows.len() as i64, columns.len() as i64])
        .set_requires_grad(requires_grad)
}

fn tensor_json(tensor: &Tensor) -> Value {
    match tensor.size().len() {
        0 => json!(tensor.double_value(&[])),
        1 => json!(Vec::<f32>::try_from(tensor).expect("convert rank-1 tensor")),
        2 => json!(Vec::<Vec<f32>>::try_from(tensor).expect("convert rank-2 tensor")),
        rank => panic!("parity helper does not support rank {rank}"),
    }
}

fn run_python(root: &Path, script: &str, arguments: &[&Path]) {
    let python = root.join(".venv/bin/python");
    let status = Command::new(python)
        .arg(root.join(script))
        .args(arguments)
        .status()
        .expect("run Python parity verifier");
    assert!(status.success(), "Python verifier {script} failed");
}
