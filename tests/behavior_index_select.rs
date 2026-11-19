use std::{env, fs};

use rusttorch::{Device, Kind, Result, RustTorchError, Tensor};
use serde_json::{Value, json};

fn probe_device() -> Device {
    match env::var("RUSTTORCH_BEHAVIOR_DEVICE").as_deref() {
        Ok("cpu") | Err(env::VarError::NotPresent) => Device::Cpu,
        Ok("mps") => Device::Mps,
        Ok("cuda") => Device::Cuda(0),
        other => panic!("unexpected behavior probe device {other:?}"),
    }
}

fn write_observation(device: Device, metadata: Value, values: Value) {
    if let Some(path) = env::var_os("RUSTTORCH_BEHAVIOR_OUTPUT") {
        let device_name = match device {
            Device::Cpu => "cpu",
            Device::Mps => "mps",
            Device::Cuda(0) => "cuda",
            other => panic!("unexpected result device {other:?}"),
        };
        let observed = json!({
            "device": device_name, "dtype": "float32", "layout": "strided",
            "metadata": metadata, "values": values,
        });
        fs::write(path, observed.to_string()).expect("write behavior probe observation");
    }
}

#[test]
fn repeated_rows_backward() -> Result<()> {
    let device = probe_device();
    let input = Tensor::f_from_slice(&[1_f32, 2., 3., 4., 5., 6.])?
        .f_reshape([3, 2])?
        .f_to_device(device)?
        .set_requires_grad(true);
    let indices = Tensor::f_from_slice(&[2_i64, 0, 2])?.f_to_device(device)?;
    let selected = rusttorch::tensor::select_rows(&input, &indices)?;
    selected.f_square()?.f_sum(Kind::Float)?.f_backward()?;
    let grad = input.grad();

    for tensor in [&selected, &grad] {
        assert_eq!(tensor.size(), [3, 2]);
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(tensor.device(), device);
        assert!(tensor.is_contiguous());
    }
    let elements = |tensor: &Tensor| -> Result<Vec<f32>> {
        Ok(Vec::try_from(
            tensor.f_to_device(Device::Cpu)?.f_view([-1])?,
        )?)
    };
    let values = elements(&selected)?;
    let gradient = elements(&grad)?;
    assert_eq!(values, [5., 6., 1., 2., 5., 6.]);
    assert_eq!(gradient, [2., 4., 0., 0., 20., 24.]);

    write_observation(
        selected.device(),
        json!({ "shape": selected.size(), "stride": selected.stride(),
                "gradient_shape": grad.size(), "gradient_stride": grad.stride() }),
        json!({ "selected": values, "gradient": gradient }),
    );
    Ok(())
}

#[test]
fn rejects_float_indices() -> Result<()> {
    let device = probe_device();
    let input = Tensor::f_from_slice(&[1_f32, 2.])?.f_to_device(device)?;
    let indices = Tensor::f_from_slice(&[1_f32])?.f_to_device(device)?;
    let error =
        rusttorch::tensor::select_rows(&input, &indices).expect_err("float index must fail");
    assert!(
        matches!(error, RustTorchError::InvalidConfiguration {
        field: "select_rows.ids", ref reason,
    } if reason.contains("dtype")),
        "unexpected error: {error}"
    );
    let after = Vec::<f32>::try_from(input.f_to_device(Device::Cpu)?)?;
    assert_eq!(after, [1., 2.]);
    write_observation(
        input.device(),
        json!({ "error": "index_dtype", "shape": input.size() }),
        json!({ "input_after_error": after }),
    );
    Ok(())
}
