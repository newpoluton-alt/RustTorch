use std::{env, path::PathBuf};

use rusttorch::{
    DeviceSpec, Result, RustTorchError, Tensor,
    nn::{Sequential, functional},
    no_grad,
    optim::Adam,
};

struct Options {
    device: DeviceSpec,
    save: Option<PathBuf>,
}

fn main() -> Result<()> {
    let options = parse_options()?;
    let mut model = Sequential::builder()
        .linear(2, 3)
        .relu()
        .linear(3, 1)
        .build(options.device)?;
    assign_model_weights(&model)?;
    let mut optimizer = Adam::builder()
        .learning_rate(1e-2)
        .build(model.var_store())?;
    let device = model.device();
    let inputs = Tensor::from_slice(&[1.0_f32, 2.0, 2.0, -1.0])
        .reshape([2, 2])
        .f_to_device(device)?;
    let targets = Tensor::from_slice(&[0.5_f32, -1.0])
        .reshape([2, 1])
        .f_to_device(device)?;

    model.train();
    let predictions = model.forward(&inputs)?;
    let loss = functional::mse_loss(&predictions, &targets)?;
    let loss_value = loss.f_double_value(&[])?;
    optimizer.backward_step(&loss)?;

    println!("device={device:?} loss={loss_value:.6}");
    if let Some(path) = options.save {
        model.save_weights(&path)?;
        println!("saved {}", path.display());
    }
    Ok(())
}

fn assign_model_weights(model: &Sequential) -> Result<()> {
    let variables = model.var_store().variables();
    assign(
        variables
            .get("0.weight")
            .expect("builder guarantees first weight"),
        &[1.0, 0.0, 0.0, 1.0, 0.5, -0.5],
        &[3, 2],
    )?;
    assign(
        variables
            .get("0.bias")
            .expect("builder guarantees first bias"),
        &[0.0, 0.0, 0.0],
        &[3],
    )?;
    assign(
        variables
            .get("2.weight")
            .expect("builder guarantees second weight"),
        &[1.0, -1.0, 0.5],
        &[1, 3],
    )?;
    assign(
        variables
            .get("2.bias")
            .expect("builder guarantees second bias"),
        &[0.0],
        &[1],
    )
}

fn assign(target: &Tensor, values: &[f32], shape: &[i64]) -> Result<()> {
    let source = Tensor::from_slice(values)
        .reshape(shape)
        .f_to_device(target.device())?;
    let mut target = target.shallow_clone();
    no_grad(|| target.f_copy_(&source))?;
    Ok(())
}

fn parse_options() -> Result<Options> {
    let mut options = Options {
        device: DeviceSpec::Cpu,
        save: None,
    };
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device" => {
                let value = arguments.next().ok_or_else(|| {
                    invalid_argument("--device", "requires auto, cpu, mps, cuda, or cuda:N")
                })?;
                options.device = parse_device(&value)?;
            }
            "--save" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| invalid_argument("--save", "requires a path"))?;
                options.save = Some(value.into());
            }
            _ => {
                return Err(invalid_argument(
                    "arguments",
                    &format!("unknown option `{argument}`"),
                ));
            }
        }
    }
    Ok(options)
}

fn parse_device(value: &str) -> Result<DeviceSpec> {
    match value {
        "auto" => Ok(DeviceSpec::Auto),
        "cpu" => Ok(DeviceSpec::Cpu),
        "mps" => Ok(DeviceSpec::Mps),
        "cuda" => Ok(DeviceSpec::Cuda(0)),
        _ => value
            .strip_prefix("cuda:")
            .and_then(|index| index.parse().ok())
            .map(DeviceSpec::Cuda)
            .ok_or_else(|| {
                invalid_argument("--device", "expected auto, cpu, mps, cuda, or cuda:N")
            }),
    }
}

fn invalid_argument(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.to_owned(),
    }
}
