use rusttorch::{Device, Kind, Result, RustTorchError, Tensor, optim};
use tch::nn::{Init, VarStore};

#[test]
fn adamw_applies_decoupled_decay_and_tracks_moments() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let parameter = store.root().var("weight", &[1], Init::Const(2.0));
    let mut optimizer = optim::AdamW::builder()
        .learning_rate(0.1)
        .betas(0.0, 0.0)
        .eps(0.0)
        .weight_decay(0.2)
        .amsgrad(true)
        .build(&store)?;
    optimizer.backward_step(&parameter.f_sum(Kind::Float)?)?;
    assert!((parameter.double_value(&[0]) - 1.86).abs() < 1e-6);
    optimizer.set_learning_rate(0.05)?;
    optimizer.backward_step(&parameter.f_sum(Kind::Float)?)?;
    assert!((parameter.double_value(&[0]) - 1.7914).abs() < 1e-6);
    Ok(())
}

#[test]
fn rmsprop_uses_squared_gradient_average() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let parameter = store.root().var("weight", &[1], Init::Const(2.0));
    let mut optimizer = optim::RmsProp::builder()
        .learning_rate(0.1)
        .alpha(0.5)
        .eps(0.0)
        .build(&store)?;
    optimizer.backward_step(&parameter.f_square()?.f_sum(Kind::Float)?)?;
    assert!((parameter.double_value(&[0]) - (2.0 - 0.4 / 8_f64.sqrt())).abs() < 1e-6);
    Ok(())
}

#[test]
fn optimizer_configuration_rejects_nonfinite_and_negative_values() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let _parameter = store.root().var("weight", &[1], Init::Const(1.0));
    for value in [f64::NAN, f64::INFINITY, -0.1] {
        assert!(
            optim::AdamW::builder()
                .learning_rate(value)
                .build(&store)
                .is_err()
        );
        assert!(optim::AdamW::builder().eps(value).build(&store).is_err());
        assert!(
            optim::AdamW::builder()
                .weight_decay(value)
                .build(&store)
                .is_err()
        );
        assert!(
            optim::RmsProp::builder()
                .learning_rate(value)
                .build(&store)
                .is_err()
        );
        assert!(
            optim::RmsProp::builder()
                .alpha(value)
                .build(&store)
                .is_err()
        );
        assert!(optim::RmsProp::builder().eps(value).build(&store).is_err());
        assert!(
            optim::RmsProp::builder()
                .momentum(value)
                .build(&store)
                .is_err()
        );
        assert!(
            optim::RmsProp::builder()
                .weight_decay(value)
                .build(&store)
                .is_err()
        );
    }
    for (beta1, beta2) in [(1.0, 0.9), (0.9, 1.0), (f64::NAN, 0.9)] {
        assert!(
            optim::AdamW::builder()
                .betas(beta1, beta2)
                .build(&store)
                .is_err()
        );
    }
    let mut optimizer = optim::AdamW::builder().build(&store)?;
    optimizer.clip_grad_norm(1.0)?;
    optimizer.clip_grad_value(1.0)?;
    assert!(optimizer.set_learning_rate(f64::NAN).is_err());
    assert!(optimizer.clip_grad_norm(-1.0).is_err());
    assert!(optimizer.clip_grad_value(f64::INFINITY).is_err());
    assert!(matches!(
        optimizer.backward_step(&Tensor::new()),
        Err(RustTorchError::InvalidConfiguration { .. })
    ));
    Ok(())
}

#[test]
fn gradient_clipping_controls_the_update() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let parameter = store.root().var("weight", &[2], Init::Const(2.0));
    let mut optimizer = optim::Sgd::builder().learning_rate(1.0).build(&store)?;
    parameter.f_square()?.f_sum(Kind::Float)?.backward();
    optimizer.clip_grad_value(1.0)?;
    assert_eq!(Vec::<f32>::try_from(parameter.grad())?, vec![1.0, 1.0]);
    optimizer.clip_grad_norm(0.5)?;
    assert!((parameter.grad().norm().double_value(&[]) - 0.5).abs() < 1e-6);
    optimizer.step();
    let expected = 2.0 - 0.5 / 2_f64.sqrt();
    assert!((parameter.double_value(&[0]) - expected).abs() < 1e-6);
    Ok(())
}
