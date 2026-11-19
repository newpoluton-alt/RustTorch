use std::{env, fs};

use rusttorch::{Device, Kind, Result, Tensor, nn::Module};
use serde_json::json;

#[test]
fn trainable_noncontiguous_token_features() -> Result<()> {
    let requested = env::var("RUSTTORCH_BEHAVIOR_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = match requested.as_str() {
        "cpu" => Device::Cpu,
        "mps" => Device::Mps,
        "cuda" => Device::Cuda(0),
        other => panic!("unsupported behavior test device {other}"),
    };
    let store = rusttorch::nn::VarStore::new(device);
    let mut norm: rusttorch::nn::RmsNorm =
        rusttorch::nn::RmsNormConfig::new([2, 3]).build(&(store.root() / "tokens"))?;
    let weight = store.variables().remove("tokens.weight").unwrap();
    let scale = Tensor::f_arange_start(1, 7, (Kind::Float, device))?.f_reshape([2, 3])? / 4.;
    rusttorch::no_grad(|| weight.shallow_clone().f_copy_(&scale))?;
    let x = Tensor::f_arange_start(1, 13, (Kind::Float, device))?
        .f_reshape([2, 3, 2])?
        .f_transpose(1, 2)?
        .set_requires_grad(true);
    assert!(!x.is_contiguous());

    let output = rusttorch::nn::RmsNorm::forward(&norm, &x)?;
    let functional = rusttorch::nn::functional::rms_norm(&x, &[2, 3], Some(&weight), None)?;
    assert!(output.f_equal(&functional)?);
    assert!((output.double_value(&[0, 0, 0]) - 0.25 / (91_f64 / 6.).sqrt()).abs() < 1e-6);
    output.f_sum(Kind::Float)?.f_backward()?;
    let input_grad = x.grad();
    let weight_grad = weight.grad();
    for tensor in [&output, &input_grad, &weight_grad] {
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(tensor.device(), device);
        assert!(!tensor.is_sparse());
    }
    rusttorch::nn::RmsNorm::reset_parameters(&mut norm)?;
    assert!(weight.f_equal(&weight.f_ones_like()?)?);
    if let Some(path) = env::var_os("RUSTTORCH_BEHAVIOR_OUTPUT") {
        let values = |tensor: &Tensor| -> Result<Vec<f32>> {
            Ok(Vec::try_from(
                tensor.f_to_device(Device::Cpu)?.f_reshape([-1])?,
            )?)
        };
        let observation = json!({
            "device": requested, "dtype": "float32", "layout": "strided",
            "metadata": {
                "shape": output.size(), "input_stride": x.stride(), "output_stride": output.stride(),
                "input_gradient_stride": input_grad.stride(), "weight_shape": weight.size()
            },
            "values": {"output": values(&output)?, "input_gradient": values(&input_grad)?,
                       "weight_gradient": values(&weight_grad)?, "reset_weight": values(&weight)?}
        });
        fs::write(path, observation.to_string()).expect("write observed RMS normalization result");
    }
    Ok(())
}
