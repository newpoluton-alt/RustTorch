use rusttorch::{Device, Kind, Result, Tensor, nn::*, no_grad};
use serde_json::Value;

fn input(shape: &[i64]) -> Tensor {
    (Tensor::arange(shape.iter().product::<i64>(), (Kind::Double, Device::Cpu)).reshape(shape)
        / 7.0
        - 1.0)
        .set_requires_grad(true)
}

fn double_store() -> VarStore {
    let mut store = VarStore::new(Device::Cpu);
    store.set_kind(Kind::Double);
    store
}

fn gradients(output: &Tensor) -> Result<()> {
    let factors = Tensor::arange(output.numel() as i64, (Kind::Double, Device::Cpu))
        .reshape(output.size())
        / 10.0
        + 0.2;
    output.f_mul(&factors)?.f_sum(Kind::Double)?.f_backward()?;
    Ok(())
}

fn assert_values(tensor: &Tensor, expected: &[f64]) {
    let actual = Vec::<f64>::try_from(&tensor.reshape([-1])).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 2e-8 * (1.0 + b.abs()),
            "index {i}: {a} != {b}"
        );
    }
}

fn close(tensor: &Tensor, expected: &Value) {
    let values: Vec<f64> = expected
        .as_array()
        .expect("reference array")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_values(tensor, &values);
}

#[test]
fn batch_norm_tracks_unbiased_variance_cumulative_average_and_eval() -> Result<()> {
    let store = double_store();
    let norm = BatchNormConfig::<1>::new(2)
        .momentum(None)
        .build(&store.root())?;
    let x = Tensor::from_slice(&[1_f64, 2., 3., 6.])
        .reshape([2, 2])
        .set_requires_grad(true);
    let y = norm.forward_t(&x, true)?;
    assert_values(norm.running_mean().unwrap(), &[2., 4.]);
    assert_values(norm.running_var().unwrap(), &[2., 8.]);
    let _ = norm.forward_t(&(x.detach() * 2.0), true)?;
    assert_values(norm.running_mean().unwrap(), &[3., 6.]);
    assert_values(norm.running_var().unwrap(), &[5., 20.]);
    assert_eq!(norm.num_batches_tracked().unwrap().int64_value(&[]), 2);
    let evaluation = norm.forward(&x)?;
    assert_values(
        &evaluation,
        &[
            -2.0 / (5.0_f64 + 1e-5).sqrt(),
            -4.0 / (20.0_f64 + 1e-5).sqrt(),
            0.,
            0.,
        ],
    );
    assert_eq!(norm.num_batches_tracked().unwrap().int64_value(&[]), 2);
    gradients(&y)?;
    assert!(x.grad().defined());
    assert!(norm.weight().unwrap().grad().defined());
    assert_eq!(store.variables().len(), 5);
    assert_eq!(store.trainable_variables().len(), 2);
    assert_eq!(norm.running_mean().unwrap().kind(), Kind::Double);
    assert_eq!(norm.num_batches_tracked().unwrap().kind(), Kind::Int64);
    norm.reset_running_stats()?;
    assert_values(norm.running_mean().unwrap(), &[0., 0.]);
    assert_values(norm.running_var().unwrap(), &[1., 1.]);
    assert_eq!(norm.num_batches_tracked().unwrap().int64_value(&[]), 0);
    Ok(())
}

#[test]
fn normalization_untracked_modes_unbatched_inputs_and_state_roundtrip() -> Result<()> {
    let store = double_store();
    let norm = BatchNormConfig::<1>::new(3)
        .affine(false)
        .track_running_stats(false)
        .build(&store.root())?;
    let x = input(&[2, 3, 4]);
    let train = norm.forward_t(&x, true)?;
    let eval = norm.forward(&x)?;
    assert!(train.allclose(&eval, 1e-10, 1e-10, false));
    assert!(store.variables().is_empty());
    let instance = InstanceNormConfig::<2>::new(3).build(&store.root())?;
    let image = input(&[3, 3, 4]);
    let batched = instance.forward(&image.unsqueeze(0))?;
    assert!(
        instance
            .forward(&image)?
            .allclose(&batched.squeeze_dim(0), 1e-10, 1e-10, false)
    );
    assert!(store.variables().is_empty());

    let tracked = BatchNormConfig::<2>::new(3)
        .bias(false)
        .build(&(store.root() / "norm"))?;
    let _ = tracked.forward_t(&image.unsqueeze(0), true)?;
    let file =
        std::env::temp_dir().join(format!("rusttorch-norm-{}.safetensors", std::process::id()));
    rusttorch::interop::save_state_dict(&file, &store)?;
    let loaded_store = double_store();
    let loaded = BatchNormConfig::<2>::new(3)
        .bias(false)
        .build(&(loaded_store.root() / "norm"))?;
    rusttorch::interop::load_state_dict(&file, &loaded_store)?;
    std::fs::remove_file(file).expect("remove norm checkpoint");
    assert!(loaded.forward(&image.unsqueeze(0))?.allclose(
        &tracked.forward(&image.unsqueeze(0))?,
        1e-10,
        1e-10,
        false
    ));
    assert_eq!(loaded.num_batches_tracked().unwrap().int64_value(&[]), 1);
    assert_eq!(loaded.num_batches_tracked().unwrap().kind(), Kind::Int64);
    assert_eq!(loaded_store.trainable_variables().len(), 1);
    Ok(())
}

#[test]
fn sequential_spatial_factory_trains_and_restores_registered_buffers() -> Result<()> {
    let build = || {
        Sequential::builder()
            .conv2d(ConvConfig::new(1, 2, [3, 3]).padding([1, 1]))
            .layer(|path| BatchNormConfig::<2>::new(2).build(path))
            .relu()
            .layer(|_| MaxPool2d::new([2, 2]))
            .layer(|_| AdaptiveAvgPool2d::new([1, 1]))
            .flatten(1, -1)
            .linear(2, 1)
            .build(rusttorch::DeviceSpec::Cpu)
    };
    let mut model = build()?;
    let x = input(&[2, 1, 6, 6]).to_kind(Kind::Float);
    model.train();
    let prediction = model.forward(&x)?;
    assert_eq!(prediction.size(), [2, 1]);
    prediction.square().mean(Kind::Float).f_backward()?;
    assert!(
        model
            .var_store()
            .trainable_variables()
            .iter()
            .all(|p| p.grad().defined())
    );
    let counter = &model.var_store().variables()["1.num_batches_tracked"];
    assert_eq!(counter.int64_value(&[]), 1);
    model.eval();
    let expected = model.forward(&x)?;
    let file = std::env::temp_dir().join(format!(
        "rusttorch-spatial-sequential-{}.safetensors",
        std::process::id()
    ));
    model.save_weights(&file)?;
    let mut restored = build()?;
    restored.load_weights(&file)?;
    std::fs::remove_file(file).expect("remove spatial model checkpoint");
    restored.eval();
    assert!(expected.allclose(&restored.forward(&x)?, 1e-6, 1e-6, false));
    assert_eq!(
        restored.var_store().variables()["1.num_batches_tracked"].int64_value(&[]),
        1
    );
    Ok(())
}

#[test]
fn normalization_validates_channels_dimensions_and_options() -> Result<()> {
    let store = double_store();
    assert!(BatchNormConfig::<0>::new(2).build(&store.root()).is_err());
    assert!(
        InstanceNormConfig::<4>::new(2)
            .build(&store.root())
            .is_err()
    );
    assert!(BatchNormConfig::<1>::new(0).build(&store.root()).is_err());
    assert!(
        BatchNormConfig::<1>::new(2)
            .eps(-1.)
            .build(&store.root())
            .is_err()
    );
    assert!(
        BatchNormConfig::<1>::new(2)
            .momentum(Some(f64::NAN))
            .build(&store.root())
            .is_err()
    );
    assert!(GroupNormConfig::new(0, 4).build(&store.root()).is_err());
    assert!(GroupNormConfig::new(3, 4).build(&store.root()).is_err());
    let batch = BatchNormConfig::<2>::new(2).build(&store.root())?;
    assert!(batch.forward(&Tensor::new()).is_err());
    assert!(batch.forward(&input(&[2, 2, 3])).is_err());
    assert!(batch.forward_t(&input(&[1, 2, 1, 1]), true).is_err());
    assert!(batch.forward(&input(&[2, 3, 4, 4])).is_err());
    let instance = InstanceNormConfig::<1>::new(2).build(&(store.root() / "instance"))?;
    assert!(instance.forward(&input(&[1, 2, 1])).is_err());
    assert!(instance.forward(&input(&[1, 3, 4])).is_err());
    let group = GroupNormConfig::new(2, 2).build(&(store.root() / "group"))?;
    assert!(group.forward(&input(&[1, 2, 1])).is_err());
    assert!(group.forward(&Tensor::new()).is_err());
    let mut integer_store = VarStore::new(Device::Cpu);
    integer_store.set_kind(Kind::Int64);
    assert!(
        BatchNormConfig::<1>::new(2)
            .build(&integer_store.root())
            .is_err()
    );
    Ok(())
}

#[test]
fn transposed_convolution_initialization_output_size_and_validation() -> Result<()> {
    let store = double_store();
    let layer = ConvTransposeConfig::new(2, 4, [3])
        .groups(2)
        .stride([2])
        .build(&store.root())?;
    assert_eq!(layer.weight().size(), [2, 2, 3]);
    assert!(layer.weight().abs().max().double_value(&[]) <= 1.0 / 6_f64.sqrt());
    assert!(layer.bias().unwrap().abs().max().double_value(&[]) <= 1.0 / 6_f64.sqrt());
    let x = input(&[2, 3]);
    assert_eq!(layer.forward(&x)?.size(), [4, 7]);
    assert_eq!(layer.forward_with_output_size(&x, &[8])?.size(), [4, 8]);
    assert_eq!(layer.forward_with_output_size(&x, &[4, 8])?.size(), [4, 8]);
    assert!(layer.forward_with_output_size(&x, &[9]).is_err());
    assert!(layer.forward_with_output_size(&x, &[]).is_err());
    assert!(layer.forward(&Tensor::new()).is_err());
    assert!(layer.forward(&input(&[1, 3, 4])).is_err());
    assert!(
        ConvTransposeConfig::new(1, 1, [2])
            .output_padding([1])
            .build(&store.root())
            .is_err()
    );
    assert!(
        ConvTransposeConfig::new(1, 1, [2])
            .output_padding([-1])
            .build(&store.root())
            .is_err()
    );
    assert!(
        ConvTransposeConfig::new(1, 1, [2])
            .stride([0])
            .build(&store.root())
            .is_err()
    );
    assert!(
        ConvTransposeConfig::new(1, 1, [2])
            .groups(2)
            .build(&store.root())
            .is_err()
    );
    assert!(
        ConvTransposeConfig::new(1, 1, [i64::MAX, 2])
            .build(&store.root())
            .is_err()
    );
    assert!(
        ConvTransposeConfig::new(1, 1, [])
            .build(&store.root())
            .is_err()
    );
    Ok(())
}

#[test]
fn pooling_known_values_indices_unbatched_gradients_and_validation() -> Result<()> {
    let x = Tensor::from_slice(&[1_f64, 3., 2., 4.])
        .reshape([1, 4])
        .set_requires_grad(true);
    let (values, indices) = MaxPool1d::new([2])?.forward_with_indices(&x)?;
    assert_values(&values, &[3., 4.]);
    assert_eq!(Vec::<i64>::try_from(&indices.flatten(0, -1))?, [1, 3]);
    values.sum(Kind::Double).f_backward()?;
    assert_values(&x.grad(), &[0., 1., 0., 1.]);
    assert_values(&AvgPool1d::new([2])?.forward(&x)?, &[2., 3.]);
    assert_values(&AdaptiveAvgPool1d::new([1])?.forward(&x)?, &[2.5]);
    let (max, indices) = AdaptiveMaxPool1d::new([1])?.forward_with_indices(&x)?;
    assert_values(&max, &[4.]);
    assert_eq!(indices.int64_value(&[0, 0]), 3);
    assert!(MaxPool1d::new([0]).is_err());
    assert!(MaxPool::<0>::new([]).is_err());
    assert!(AvgPool1d::new([2])?.stride([0]).forward(&x).is_err());
    assert!(MaxPool1d::new([2])?.padding([2]).forward(&x).is_err());
    assert!(MaxPool1d::new([2])?.dilation([0]).forward(&x).is_err());
    assert!(
        AvgPool1d::new([2])?
            .divisor_override(Some(2))
            .forward(&x)
            .is_err()
    );
    assert!(
        AvgPool2d::new([2, 2])?
            .divisor_override(Some(0))
            .forward(&input(&[1, 4, 4]))
            .is_err()
    );
    assert!(AdaptiveAvgPool1d::new([0]).is_err());
    assert!(AdaptiveMaxPool::<4>::new([1; 4]).is_err());
    assert!(MaxPool1d::new([2])?.forward(&Tensor::new()).is_err());
    assert!(
        AdaptiveAvgPool1d::new([1])?
            .forward(&Tensor::new())
            .is_err()
    );
    Ok(())
}

#[test]
fn activations_handle_zero_gradients_large_values_and_invalid_options() -> Result<()> {
    let x = Tensor::from_slice(&[-1000_f64, -1., 0., 1., 1000.]).set_requires_grad(true);
    let elu = ELU::new(1.2)?.forward(&x)?;
    elu.sum(Kind::Double).f_backward()?;
    assert_values(&x.grad(), &[0., 1.2 / std::f64::consts::E, 1.2, 1., 1.]);
    assert!(elu.isfinite().all().int64_value(&[]) != 0);
    let zero = Tensor::from_slice(&[0_f64]).set_requires_grad(true);
    LeakyReLU::new(0.2)?
        .forward(&zero)?
        .sum(Kind::Double)
        .f_backward()?;
    assert_values(&zero.grad(), &[0.2]);
    assert_values(
        &Softmax::new(-1).forward(&Tensor::from_slice(&[0_f64, 0.]))?,
        &[0.5, 0.5],
    );
    assert!(ELU::new(f64::NAN).is_err());
    assert!(LeakyReLU::new(f64::INFINITY).is_err());
    assert!(
        LeakyReLU::default()
            .forward(&Tensor::from_slice(&[1_i64]))
            .is_err()
    );
    assert!(
        ELU::default()
            .forward(&Tensor::from_slice(&[1_i64]))
            .is_err()
    );
    assert!(Softmax::new(9).forward(&x).is_err());
    assert!(Sigmoid.forward(&Tensor::new()).is_err());
    assert!(Tanh.forward(&Tensor::new()).is_err());
    assert!(SiLU.forward(&Tensor::new()).is_err());
    assert!(LogSoftmax::new(0).forward(&Tensor::new()).is_err());
    Ok(())
}

fn normalization_parity<const D: usize>(reference: &Value) -> Result<()> {
    for instance in [false, true] {
        for cumulative in [false, true] {
            let expected = &reference[format!(
                "{}{D}_{}",
                if instance { "instance" } else { "batch" },
                if cumulative { "True" } else { "False" }
            )];
            let store = double_store();
            let mut shape = vec![2, 3];
            shape.extend([2; D]);
            let x = input(&shape);
            let momentum = if cumulative { None } else { Some(0.1) };
            macro_rules! check_norm {
                ($norm:expr) => {{
                    let norm = $norm;
                    let output = norm.forward_t(&x, true)?;
                    close(&output, &expected["first"]);
                    gradients(&output)?;
                    let second = norm.forward_t(&(x.detach() * 2.0 + 1.0), true)?;
                    close(&second, &expected["second"]);
                    close(&norm.forward(&x.detach())?, &expected["eval"]);
                    close(&x.grad(), &expected["input_grad"]);
                    close(&norm.weight().unwrap().grad(), &expected["weight_grad"]);
                    close(&norm.bias().unwrap().grad(), &expected["bias_grad"]);
                    close(norm.running_mean().unwrap(), &expected["running_mean"]);
                    close(norm.running_var().unwrap(), &expected["running_var"]);
                    assert_eq!(
                        norm.num_batches_tracked().unwrap().int64_value(&[]),
                        expected["count"].as_i64().unwrap()
                    );
                }};
            }
            if instance {
                check_norm!(
                    InstanceNormConfig::<D>::new(3)
                        .affine(true)
                        .track_running_stats(true)
                        .momentum(momentum)
                        .build(&store.root())?
                );
            } else {
                check_norm!(
                    BatchNormConfig::<D>::new(3)
                        .momentum(momentum)
                        .build(&store.root())?
                );
            }
        }
    }
    Ok(())
}

fn transpose_parity<const D: usize>(reference: &Value) -> Result<()> {
    let expected = &reference[format!("transpose{D}")];
    let store = double_store();
    tch::manual_seed(700 + D as i64);
    let layer = ConvTransposeConfig::new(2, 4, [2; D])
        .stride([2; D])
        .padding([1; D])
        .output_padding([1; D])
        .dilation([2; D])
        .groups(2)
        .build(&store.root())?;
    close(layer.weight(), &expected["initial_weight"]);
    close(layer.bias().unwrap(), &expected["initial_bias"]);
    no_grad(|| -> Result<()> {
        let weight = Tensor::arange(layer.weight().numel() as i64, (Kind::Double, Device::Cpu))
            .reshape(layer.weight().size())
            / 11.0
            - 0.2;
        layer.weight().shallow_clone().f_copy_(&weight)?;
        layer
            .bias()
            .unwrap()
            .shallow_clone()
            .f_copy_(&Tensor::from_slice(&[-0.2_f64, -0.1, 0., 0.1]))?;
        Ok(())
    })?;
    let mut shape = vec![2, 2];
    shape.extend([3; D]);
    let x = input(&shape);
    let output = layer.forward(&x)?;
    close(&output, &expected["output"]);
    assert_eq!(serde_json::json!(output.size()), expected["shape"]);
    gradients(&output)?;
    close(&x.grad(), &expected["input_grad"]);
    close(&layer.weight().grad(), &expected["weight_grad"]);
    close(&layer.bias().unwrap().grad(), &expected["bias_grad"]);
    close(
        &layer.forward_with_output_size(&x.detach(), &[5; D])?,
        &expected["requested"],
    );
    Ok(())
}

fn pooling_parity<const D: usize>(reference: &Value) -> Result<()> {
    for kind in ["max", "avg", "adaptive_max", "adaptive_avg"] {
        let expected = &reference[format!("{kind}{D}")];
        let mut shape = vec![2, 2];
        shape.extend([5; D]);
        let x = input(&shape);
        let (output, indices) = match kind {
            "max" => {
                let (o, i) = MaxPool::new([3; D])?
                    .stride([2; D])
                    .padding([1; D])
                    .dilation([2; D])
                    .ceil_mode(true)
                    .forward_with_indices(&x)?;
                (o, Some(i))
            }
            "avg" => (
                AvgPool::new([3; D])?
                    .stride([2; D])
                    .padding([1; D])
                    .ceil_mode(true)
                    .count_include_pad(false)
                    .divisor_override(if D == 1 { None } else { Some(4) })
                    .forward(&x)?,
                None,
            ),
            "adaptive_max" => {
                let (o, i) = AdaptiveMaxPool::new([2; D])?.forward_with_indices(&x)?;
                (o, Some(i))
            }
            _ => (AdaptiveAvgPool::new([2; D])?.forward(&x)?, None),
        };
        close(&output, &expected["output"]);
        assert_eq!(serde_json::json!(output.size()), expected["shape"]);
        gradients(&output)?;
        close(&x.grad(), &expected["input_grad"]);
        if let Some(indices) = indices {
            assert_eq!(
                serde_json::json!(Vec::<i64>::try_from(&indices.reshape([-1]))?),
                expected["indices"]
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "run through scripts/run-python-parity.sh after generating spatial.json"]
fn spatial_layers_match_pinned_python_outputs_gradients_and_buffers() -> Result<()> {
    let directory = std::env::var_os("RUSTTORCH_PYTHON_REFERENCE_DIR")
        .expect("set RUSTTORCH_PYTHON_REFERENCE_DIR");
    let reference: Value = serde_json::from_slice(
        &std::fs::read(std::path::PathBuf::from(directory).join("spatial.json"))
            .expect("generate spatial.json"),
    )
    .expect("valid JSON");
    assert_eq!(
        reference["commit"],
        "cf30153c4c131c8164ee7798e5022d810682e2cb"
    );
    normalization_parity::<1>(&reference)?;
    normalization_parity::<2>(&reference)?;
    normalization_parity::<3>(&reference)?;
    transpose_parity::<1>(&reference)?;
    transpose_parity::<2>(&reference)?;
    transpose_parity::<3>(&reference)?;
    pooling_parity::<1>(&reference)?;
    pooling_parity::<2>(&reference)?;
    pooling_parity::<3>(&reference)?;
    let store = double_store();
    let group = GroupNormConfig::new(3, 6)
        .eps(1e-4)
        .bias(false)
        .build(&store.root())?;
    let x = input(&[2, 6, 2, 3]);
    let output = group.forward(&x)?;
    close(&output, &reference["group"]["output"]);
    gradients(&output)?;
    close(&x.grad(), &reference["group"]["input_grad"]);
    close(
        &group.weight().unwrap().grad(),
        &reference["group"]["weight_grad"],
    );
    let activations: Vec<(&str, Box<dyn Module>)> = vec![
        ("sigmoid", Box::new(Sigmoid)),
        ("tanh", Box::new(Tanh)),
        ("silu", Box::new(SiLU)),
        ("softmax", Box::new(Softmax::new(-1))),
        ("log_softmax", Box::new(LogSoftmax::new(-1))),
        ("leaky_relu", Box::new(LeakyReLU::new(0.2)?)),
        ("elu", Box::new(ELU::new(1.2)?)),
    ];
    for (name, layer) in activations {
        let x =
            Tensor::from_slice(&[-1000_f64, -2., -0.1, 0., 0.2, 2., 1000.]).set_requires_grad(true);
        let output = layer.forward(&x)?;
        close(&output, &reference[name]["output"]);
        gradients(&output)?;
        close(&x.grad(), &reference[name]["input_grad"]);
    }
    Ok(())
}
