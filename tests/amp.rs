use rusttorch::{
    Device, Kind, Result, Tensor,
    amp::{GradScaler, GradScalerConfig, autocast, autocast_for},
    nn::VarStore,
    optim::Sgd,
};

fn model() -> Result<(VarStore, Tensor, rusttorch::optim::Optimizer)> {
    let store = VarStore::new(Device::Cpu);
    let value = store.root().var("weight", &[1], tch::nn::Init::Const(2.));
    let optimizer = Sgd::builder().learning_rate(0.1).build(&store)?;
    Ok((store, value, optimizer))
}

#[test]
fn scaling_preserves_updates_skips_nonfinite_steps_and_replays_state() -> Result<()> {
    let (_store, value, mut optimizer) = model()?;
    let mut scaler = GradScalerConfig {
        initial_scale: 8.,
        growth_interval: 2,
        ..Default::default()
    }
    .build()?;
    for expected in [1.6, 1.28] {
        optimizer.try_zero_grad()?;
        scaler
            .scale(&value.f_square()?.f_sum(Kind::Float)?)?
            .f_backward()?;
        scaler.unscale(&optimizer)?;
        assert!(scaler.step(&mut optimizer)?);
        scaler.update()?;
        assert!((value.double_value(&[0]) - expected).abs() < 1e-6);
    }
    assert_eq!(scaler.get_scale(), 16.);
    optimizer.try_zero_grad()?;
    scaler
        .scale(&value.f_mul_scalar(f64::INFINITY)?.f_sum(Kind::Float)?)?
        .f_backward()?;
    let before = value.double_value(&[0]);
    assert!(!scaler.step(&mut optimizer)?);
    scaler.update()?;
    assert_eq!(value.double_value(&[0]), before);
    assert_eq!(scaler.get_scale(), 8.);
    let state = scaler.state_dict()?;
    let bytes = serde_json::to_vec(&state).unwrap();
    let state = serde_json::from_slice(&bytes).unwrap();
    let mut restored = GradScaler::default();
    restored.load_state_dict(&state)?;
    assert_eq!(restored.state_dict()?, scaler.state_dict()?);
    Ok(())
}

#[test]
fn scaler_enforces_step_order_and_validates_before_restore() -> Result<()> {
    let (_store, value, mut optimizer) = model()?;
    let (_second_store, _second, mut second) = model()?;
    let mut scaler = GradScaler::default();
    assert!(scaler.update().is_err());
    assert!(scaler.unscale(&optimizer).is_err());
    scaler.scale(&value.sum(Kind::Float))?.f_backward()?;
    scaler.unscale(&optimizer)?;
    assert!(scaler.unscale(&optimizer).is_err());
    assert!(scaler.state_dict().is_err());
    assert!(scaler.scale(&value.sum(Kind::Float)).is_err());
    assert!(scaler.step(&mut second).is_err());
    scaler.step(&mut optimizer)?;
    assert!(scaler.step(&mut optimizer).is_err());
    scaler.update()?;
    let saved = scaler.state_dict()?;
    let mut invalid = saved.clone();
    invalid.scale = f64::NAN;
    assert!(scaler.load_state_dict(&invalid).is_err());
    assert_eq!(scaler.state_dict()?, saved);
    invalid = saved.clone();
    invalid.schema_version = 9;
    assert!(scaler.load_state_dict(&invalid).is_err());
    for scale in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(
            GradScalerConfig {
                initial_scale: scale,
                ..Default::default()
            }
            .build()
            .is_err()
        );
    }
    for growth in [0., 1., f64::NAN] {
        assert!(
            GradScalerConfig {
                growth_factor: growth,
                ..Default::default()
            }
            .build()
            .is_err()
        );
    }
    assert!(
        GradScalerConfig {
            growth_interval: 0,
            ..Default::default()
        }
        .build()
        .is_err()
    );
    assert!(
        GradScalerConfig {
            backoff_factor: 1.,
            ..Default::default()
        }
        .build()
        .is_err()
    );
    Ok(())
}

#[test]
fn scaler_supports_accumulation_clipping_and_disabled_execution() -> Result<()> {
    let (_store, value, mut optimizer) = model()?;
    let mut scaler = GradScalerConfig {
        enabled: false,
        ..Default::default()
    }
    .build()?;
    for _ in 0..2 {
        scaler
            .scale(&value.sum(Kind::Float).f_div_scalar(2.)?)?
            .f_backward()?;
    }
    scaler.unscale(&optimizer)?;
    optimizer.clip_grad_value(0.25)?;
    scaler.step(&mut optimizer)?;
    scaler.update()?;
    assert_eq!(scaler.get_scale(), 1.);
    assert!((value.double_value(&[0]) - 1.975).abs() < 1e-6);
    Ok(())
}

#[test]
fn autocast_preserves_results_and_strict_device_policy() -> Result<()> {
    assert_eq!(autocast(false, || 42), 42);
    assert_eq!(autocast_for(Device::Cpu, false, || Ok(7))?, 7);
    let mut invoked = false;
    assert!(
        autocast_for(Device::Cpu, true, || {
            invoked = true;
            Ok(())
        })
        .is_err()
    );
    assert!(!invoked);
    let panic = std::panic::catch_unwind(|| autocast(true, || panic!("expected unwind")));
    assert!(panic.is_err());
    assert_eq!(autocast(false, || 9), 9);
    Ok(())
}

#[test]
#[ignore = "run through scripts/run-python-parity.sh"]
fn grad_scaler_matches_python_growth_backoff_and_updates() -> Result<()> {
    let directory = std::env::var_os("RUSTTORCH_PYTHON_REFERENCE_DIR").expect("parity environment");
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::path::PathBuf::from(directory).join("training.json")).unwrap(),
    )
    .unwrap();
    for (key, config) in [
        (
            "scaler",
            GradScalerConfig {
                initial_scale: 8.,
                growth_interval: 2,
                ..Default::default()
            },
        ),
        (
            "scaler_custom",
            GradScalerConfig {
                initial_scale: 3.3,
                growth_interval: 1,
                growth_factor: 1.1,
                backoff_factor: 0.7,
                enabled: true,
            },
        ),
    ] {
        let (_store, value, mut optimizer) = model()?;
        let mut scaler = config.build()?;
        for (index, nonfinite) in [false, false, true, false].into_iter().enumerate() {
            optimizer.try_zero_grad()?;
            let loss = if nonfinite {
                value.f_mul_scalar(f64::INFINITY)?
            } else {
                value.f_square()?
            }
            .f_sum(Kind::Float)?;
            scaler.scale(&loss)?.f_backward()?;
            scaler.unscale(&optimizer)?;
            assert_eq!(scaler.step(&mut optimizer)?, !nonfinite);
            scaler.update()?;
            let expected = &reference[key][index];
            assert!(
                (value.double_value(&[0]) - expected["parameter"].as_f64().unwrap()).abs() < 1e-6
            );
            assert_eq!(
                scaler.get_scale(),
                expected["scale"].as_f64().unwrap(),
                "{key} step {index}"
            );
            assert_eq!(
                scaler.state_dict()?.growth_tracker,
                expected["tracker"].as_u64().unwrap()
            );
        }
    }
    Ok(())
}

#[test]
fn cuda_autocast_restores_dtype_after_nesting_and_unwind() -> Result<()> {
    if !rusttorch::available_devices().cuda {
        eprintln!("skipped CUDA autocast: no available CUDA device");
        return Ok(());
    }
    let x = Tensor::ones([4, 4], (Kind::Float, Device::Cuda(0)));
    autocast_for(Device::Cuda(0), true, || {
        assert_eq!(x.f_matmul(&x)?.kind(), Kind::Half);
        assert_eq!(autocast(false, || x.f_matmul(&x))?.kind(), Kind::Float);
        assert_eq!(x.f_matmul(&x)?.kind(), Kind::Half);
        Ok(())
    })?;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        autocast(true, || panic!("test CUDA autocast scope"));
    }));
    assert!(panic.is_err());
    assert_eq!(x.f_matmul(&x)?.kind(), Kind::Float);
    Ok(())
}
