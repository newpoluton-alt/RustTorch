use rusttorch::{
    Device, DeviceSpec, Kind, Result, Tensor,
    nn::{
        ConvConfig, EmbeddingConfig, LayerNormConfig, Module, ParameterPath, Sequential, VarStore,
        functional,
    },
    no_grad,
};

fn assign(target: &Tensor, values: &[f32]) -> Result<()> {
    let source = Tensor::from_slice(values).reshape(target.size());
    no_grad(|| target.shallow_clone().f_copy_(&source))?;
    Ok(())
}

fn assert_close(actual: &Tensor, expected: &[f32]) {
    let actual = Vec::<f32>::try_from(&actual.reshape([-1])).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn grouped_convolution_respects_stride_padding_dilation_and_gradients() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let path: ParameterPath<'_> = store.root() / "conv";
    let layer = ConvConfig::new(2, 2, [2])
        .groups(2)
        .stride([2])
        .padding([1])
        .dilation([2])
        .build(&path)?;
    assign(layer.weight(), &[1., 2., 3., 4.])?;
    assign(layer.bias().unwrap(), &[0.5, -0.5])?;
    let input = Tensor::from_slice(&[1_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
        .reshape([1, 2, 6])
        .set_requires_grad(true);
    let output = layer.forward(&input)?;
    assert_eq!(output.size(), [1, 2, 3]);
    assert_close(&output, &[4.5, 10.5, 16.5, 31.5, 63.5, 77.5]);
    output.sum(Kind::Float).f_backward()?;
    assert_close(
        &input.grad(),
        &[0., 3., 0., 3., 0., 2., 0., 7., 0., 7., 0., 4.],
    );
    assert_close(&layer.weight().grad(), &[6., 12., 18., 30.]);
    assert_close(&layer.bias().unwrap().grad(), &[3., 3.]);
    assert_eq!(store.variables().len(), 2);
    assert!(store.variables().contains_key("conv.weight"));
    Ok(())
}

#[test]
fn image_and_volume_convolutions_accept_unbatched_inputs() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let image = ConvConfig::new(1, 2, [2, 2])
        .bias(false)
        .build(&(store.root() / "image"))?;
    assign(image.weight(), &[1.; 8])?;
    let input = Tensor::ones([1, 3, 3], (Kind::Float, Device::Cpu));
    let output = image.forward(&input)?;
    assert_eq!(output.size(), [2, 2, 2]);
    assert_close(&output, &[4.; 8]);
    assert!(image.bias().is_none());

    let volume = ConvConfig::new(1, 2, [2, 2, 2])
        .bias(false)
        .build(&(store.root() / "volume"))?;
    assign(volume.weight(), &[1.; 16])?;
    let input = Tensor::ones([1, 3, 3, 3], (Kind::Float, Device::Cpu));
    let output = volume.forward(&input)?;
    assert_eq!(output.size(), [2, 2, 2, 2]);
    assert_close(&output, &[8.; 16]);
    Ok(())
}

#[test]
fn convolution_initialization_uses_fan_in_and_rejects_invalid_configuration() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let layer = ConvConfig::new(4, 32, [3, 2])
        .groups(2)
        .build(&store.root())?;
    let bound = 1.0 / 12_f64.sqrt();
    assert!(layer.weight().abs().max().double_value(&[]) <= bound);
    let bias = layer.bias().unwrap();
    assert!(bias.abs().max().double_value(&[]) <= bound);
    assert!(bias.abs().sum(Kind::Float).double_value(&[]) > 0.);
    for config in [
        ConvConfig::new(2, 4, [0]),
        ConvConfig::new(0, 4, [3]),
        ConvConfig::new(2, 4, [3]).groups(0),
        ConvConfig::new(2, 3, [3]).groups(2),
        ConvConfig::new(2, 4, [3]).stride([0]),
        ConvConfig::new(2, 4, [3]).padding([-1]),
        ConvConfig::new(2, 4, [3]).dilation([0]),
    ] {
        assert!(config.build(&store.root()).is_err());
    }
    assert!(ConvConfig::new(1, 1, []).build(&store.root()).is_err());
    assert!(ConvConfig::new(1, 1, [1; 4]).build(&store.root()).is_err());
    assert!(
        ConvConfig::new(2, 1, [i64::MAX, 2])
            .build(&store.root())
            .is_err()
    );
    let input = Tensor::ones([1, 2, 5], (Kind::Float, Device::Cpu));
    let weight = Tensor::ones([4, 2, 3], (Kind::Float, Device::Cpu));
    assert!(functional::conv1d(&input, &weight, None, [1], [0], [1], 0).is_err());
    assert!(functional::conv1d(&input, &weight, None, [1], [0], [1], 2).is_err());
    assert!(
        functional::conv1d(
            &input.squeeze_dim(0).get(0),
            &weight,
            None,
            [1],
            [0],
            [1],
            1
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn layer_norm_normalizes_trailing_features_and_optional_affine_parameters() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let layer = LayerNormConfig::new([2])
        .eps(0.)
        .build(&(store.root() / "affine"))?;
    assign(layer.weight().unwrap(), &[2., 3.])?;
    assign(layer.bias().unwrap(), &[0.5, -0.5])?;
    let input = Tensor::from_slice(&[1_f32, 3., 2., 4.])
        .reshape([2, 2])
        .set_requires_grad(true);
    let output = layer.forward(&input)?;
    assert_close(&output, &[-1.5, 2.5, -1.5, 2.5]);
    output.sum(Kind::Float).f_backward()?;
    assert_close(&layer.weight().unwrap().grad(), &[-2., 2.]);
    assert_close(&layer.bias().unwrap().grad(), &[2., 2.]);
    assert!(input.grad().defined());
    let plain = LayerNormConfig::new([2])
        .elementwise_affine(false)
        .eps(0.)
        .build(&(store.root() / "plain"))?;
    assert!(plain.weight().is_none());
    assert!(plain.bias().is_none());
    assert_close(&plain.forward(&input)?, &[-1., 1., -1., 1.]);
    let scaled = LayerNormConfig::new([2])
        .bias(false)
        .build(&(store.root() / "scale"))?;
    assert!(scaled.weight().is_some());
    assert!(scaled.bias().is_none());
    assert!(LayerNormConfig::new([]).build(&store.root()).is_err());
    assert!(LayerNormConfig::new([0]).build(&store.root()).is_err());
    assert!(
        LayerNormConfig::new([2])
            .eps(f64::NAN)
            .build(&store.root())
            .is_err()
    );
    assert!(
        LayerNormConfig::new([2])
            .eps(-1.)
            .build(&store.root())
            .is_err()
    );
    assert!(functional::layer_norm(&input, &[3], None, None, 1e-5).is_err());
    Ok(())
}

#[test]
fn embedding_padding_frequency_scaling_sparse_gradients_and_indices_are_checked() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let layer = EmbeddingConfig::new(4, 2)
        .padding_idx(-1)
        .scale_grad_by_freq(true)
        .build(&(store.root() / "dense"))?;
    assert_close(&layer.weight().get(3), &[0., 0.]);
    assign(layer.weight(), &[1., 2., 3., 4., 5., 6., 7., 8.])?;
    let indices = Tensor::from_slice(&[1_i64, 1, 3, 2]).reshape([2, 2]);
    let output = layer.forward(&indices)?;
    assert_eq!(output.size(), [2, 2, 2]);
    assert_close(&output, &[3., 4., 3., 4., 7., 8., 5., 6.]);
    output.sum(Kind::Float).f_backward()?;
    assert_close(&layer.weight().grad(), &[0., 0., 1., 1., 1., 1., 0., 0.]);

    let sparse = EmbeddingConfig::new(4, 2)
        .sparse(true)
        .padding_idx(0)
        .build(&(store.root() / "sparse"))?;
    sparse
        .forward(&Tensor::from_slice(&[0_i64, 2, 2]))?
        .sum(Kind::Float)
        .f_backward()?;
    assert!(sparse.weight().grad().is_sparse());
    assert_close(
        &sparse.weight().grad().to_dense(None, false),
        &[0., 0., 0., 0., 2., 2., 0., 0.],
    );
    assert!(EmbeddingConfig::new(0, 2).build(&store.root()).is_err());
    assert!(EmbeddingConfig::new(2, 0).build(&store.root()).is_err());
    assert!(
        EmbeddingConfig::new(2, 2)
            .padding_idx(2)
            .build(&store.root())
            .is_err()
    );
    assert!(
        EmbeddingConfig::new(2, 2)
            .padding_idx(-3)
            .build(&store.root())
            .is_err()
    );
    assert!(
        EmbeddingConfig::new(2, 2)
            .sparse(true)
            .scale_grad_by_freq(true)
            .build(&store.root())
            .is_err()
    );
    assert!(layer.forward(&Tensor::from_slice(&[4_i64])).is_err());
    assert!(layer.forward(&Tensor::from_slice(&[-1_i64])).is_err());
    assert!(layer.forward(&Tensor::from_slice(&[1_f32])).is_err());
    Ok(())
}

#[test]
fn sequential_builds_image_and_token_models_with_registered_parameters() -> Result<()> {
    let image = Sequential::builder()
        .conv2d(ConvConfig::new(3, 4, [3, 3]).padding([1, 1]))
        .relu()
        .flatten(1, -1)
        .linear(4 * 4 * 4, 2)
        .build(DeviceSpec::Cpu)?;
    let input = Tensor::zeros([2, 3, 4, 4], (Kind::Float, Device::Cpu));
    assert_eq!(image.forward(&input)?.size(), [2, 2]);
    assert!(image.var_store().variables().contains_key("0.weight"));
    assert!(image.var_store().variables().contains_key("3.weight"));
    let token = Sequential::builder()
        .embedding(EmbeddingConfig::new(10, 4).padding_idx(0))
        .layer_norm(LayerNormConfig::new([4]))
        .linear(4, 2)
        .build(DeviceSpec::Cpu)?;
    let input = Tensor::from_slice(&[0_i64, 2, 3, 1]).reshape([2, 2]);
    let output = token.forward(&input)?;
    assert_eq!(output.size(), [2, 2, 2]);
    output.square().sum(Kind::Float).f_backward()?;
    assert!(token.var_store().variables()["0.weight"].grad().defined());
    assert!(token.var_store().variables()["1.weight"].grad().defined());
    let sequence = Sequential::builder()
        .conv1d(ConvConfig::new(1, 2, [2]))
        .build(DeviceSpec::Cpu)?;
    assert_eq!(
        sequence
            .forward(&Tensor::ones([1, 1, 3], (Kind::Float, Device::Cpu)))?
            .size(),
        [1, 2, 2]
    );
    let volume = Sequential::builder()
        .conv3d(ConvConfig::new(1, 2, [2; 3]))
        .build(DeviceSpec::Cpu)?;
    assert_eq!(
        volume
            .forward(&Tensor::ones([1, 1, 3, 3, 3], (Kind::Float, Device::Cpu)))?
            .size(),
        [1, 2, 2, 2, 2]
    );
    Ok(())
}

#[test]
fn invalid_parameter_dtype_returns_errors_without_poisoning_the_store() -> Result<()> {
    for kind in [Kind::Int64, Kind::Bool, Kind::QInt8] {
        let mut store = VarStore::new(Device::Cpu);
        store.set_kind(kind);
        assert!(ConvConfig::new(1, 2, [3]).build(&store.root()).is_err());
        assert!(matches!(
            LayerNormConfig::new([2]).build(&store.root()),
            Err(rusttorch::RustTorchError::InvalidConfiguration {
                field: "parameter dtype",
                ..
            })
        ));

        assert!(EmbeddingConfig::new(3, 2).build(&store.root()).is_err());
        assert!(
            rusttorch::nn::LinearConfig::new(2, 3)
                .build(&store.root())
                .is_err()
        );
        assert!(store.variables().is_empty());
        let plain = LayerNormConfig::new([2])
            .elementwise_affine(false)
            .build(&store.root())?;
        let input = Tensor::ones([2], (Kind::Float, Device::Cpu));
        assert_eq!(plain.forward(&input)?.size(), [2]);
        store.set_kind(Kind::Float);
        LayerNormConfig::new([2]).build(&store.root())?;
        assert_eq!(store.variables().len(), 2);
    }
    Ok(())
}

#[test]
fn undefined_tensors_are_rejected_without_metadata_panics() -> Result<()> {
    let undefined = Tensor::new();
    let input = Tensor::ones([1, 1, 3], (Kind::Float, Device::Cpu));
    let weight = Tensor::ones([1, 1, 2], (Kind::Float, Device::Cpu));
    assert!(functional::conv1d(&undefined, &weight, None, [1], [0], [1], 1).is_err());
    assert!(functional::conv1d(&input, &undefined, None, [1], [0], [1], 1).is_err());
    assert!(functional::conv1d(&input, &weight, Some(&undefined), [1], [0], [1], 1).is_err());
    assert!(functional::layer_norm(&undefined, &[3], None, None, 1e-5).is_err());
    assert!(functional::layer_norm(&undefined, &[3], Some(&weight), None, 1e-5).is_err());
    assert!(functional::layer_norm(&input, &[3], Some(&undefined), None, 1e-5).is_err());
    assert!(functional::layer_norm(&input, &[3], None, Some(&undefined), 1e-5).is_err());
    let ids = Tensor::from_slice(&[1_i64]);
    let weight = Tensor::ones([3, 2], (Kind::Float, Device::Cpu));
    assert!(functional::embedding(&undefined, &weight, None, false, false).is_err());
    assert!(functional::embedding(&ids, &undefined, None, false, false).is_err());
    assert!(functional::linear(&undefined, &weight, None).is_err());
    assert!(functional::linear(&input, &undefined, None).is_err());
    assert!(functional::mse_loss(&undefined, &input).is_err());
    assert!(functional::mse_loss(&input, &undefined).is_err());
    assert!(functional::cross_entropy(&undefined, &ids).is_err());
    assert!(functional::cross_entropy(&input, &undefined).is_err());
    let sequence = Sequential::builder().build(DeviceSpec::Cpu)?;
    assert!(sequence.forward(&undefined).is_err());
    assert!(
        matches!(rusttorch::device::ensure_device("test", &undefined, Device::Cpu),
        Err(rusttorch::RustTorchError::InvalidDimensions { context, .. }) if context == "test")
    );
    Ok(())
}
