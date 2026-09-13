use rusttorch::{
    Device, Kind, Result, Tensor,
    testing::{self, CloseOptions, GradcheckOptions},
};

#[test]
fn tensor_comparisons_check_metadata_nan_infinity_and_exact_integers() -> Result<()> {
    let a = Tensor::from_slice(&[1_f64, f64::NAN, f64::INFINITY, -f64::INFINITY]);
    let b = Tensor::from_slice(&[1.1_f64, f64::NAN, f64::INFINITY, f64::INFINITY]);
    let report = testing::compare(&a, &b, CloseOptions::default())?;
    assert_eq!(report.mismatches, 3);
    assert_eq!(report.first_mismatch, Some(vec![0]));
    assert_eq!(
        testing::compare(
            &a,
            &a,
            CloseOptions {
                equal_nan: true,
                ..Default::default()
            }
        )?
        .mismatches,
        0
    );
    assert!(testing::assert_close(&a, &b, CloseOptions::default()).is_err());
    let ints = Tensor::from_slice(&[9_007_199_254_740_992_i64]);
    let different = Tensor::from_slice(&[9_007_199_254_740_993_i64]);
    assert_eq!(
        testing::compare(&ints, &different, CloseOptions::default())?.mismatches,
        1
    );
    assert!(testing::compare(&ints, &ints.reshape([1, 1]), CloseOptions::default()).is_err());
    assert!(testing::compare(&ints, &ints.to_kind(Kind::Double), CloseOptions::default()).is_err());
    assert!(
        testing::compare(
            &ints,
            &ints,
            CloseOptions {
                rtol: f64::NAN,
                ..Default::default()
            }
        )
        .is_err()
    );
    let empty = Tensor::zeros([0], (Kind::Float, Device::Cpu));
    assert_eq!(
        testing::compare(&empty, &empty, CloseOptions::default())?.elements,
        0
    );
    let scalar = Tensor::from(1_f64);
    assert_eq!(
        testing::compare(&scalar, &Tensor::from(2_f64), CloseOptions::default())?.first_mismatch,
        Some(vec![])
    );
    Ok(())
}

#[test]
fn finite_differences_accept_correct_gradients_and_reject_surrogates_and_oversized_work()
-> Result<()> {
    let input = Tensor::from_slice(&[0.2_f64, -0.4]).set_requires_grad(true);
    testing::gradcheck(|x| Ok(x.f_tanh()?), &input, GradcheckOptions::default())?;
    let strided = Tensor::from_slice(&[0.2_f64, -0.4, 0.1, 0.3])
        .reshape([2, 2])
        .transpose(0, 1);
    assert!(!strided.is_contiguous());
    testing::gradcheck(|x| Ok(x.f_tanh()?), &strided, GradcheckOptions::default())?;
    testing::gradcheck(
        |x| Ok(x.f_square()?.f_sum(Kind::Double)?),
        &input,
        GradcheckOptions::default(),
    )?;
    assert!(!input.grad().defined());
    assert!(
        testing::gradcheck(
            |x| rusttorch::autograd::with_surrogate_gradient(&x.f_square()?, x),
            &input,
            GradcheckOptions::default()
        )
        .is_err()
    );
    assert!(
        testing::gradcheck(
            |x| Ok(x.f_tanh()?),
            &input,
            GradcheckOptions {
                max_jacobian_elements: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        testing::gradcheck(
            |x| Ok(x.shallow_clone()),
            &input.to_kind(Kind::Float),
            GradcheckOptions::default()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn local_noise_resumes_without_global_rng_and_rejects_limits_without_advancing() -> Result<()> {
    use rusttorch::{
        data::ResourceLimits,
        reproducibility::{TensorRng, TensorRngState},
    };
    let mut rng = TensorRng::new(31);
    let first = rng.normal(&[3, 4], (Kind::Double, Device::Cpu))?;
    assert_eq!(first.size(), [3, 4]);
    let state = rng.state_dict();
    let json = serde_json::to_string(&state).unwrap();
    let mut restored = TensorRng::from_state(serde_json::from_str(&json).unwrap())?;
    let expected = rng.uniform(&[128], (Kind::Float, Device::Cpu))?;
    // A global reseed does not affect the task-local stream.
    rusttorch::manual_seed(99);
    assert!(expected.f_equal(&restored.uniform(&[128], (Kind::Float, Device::Cpu))?)?);
    assert!(expected.f_ge(0.)?.f_all()?.f_int64_value(&[])? != 0);
    assert!(expected.f_lt(1.)?.f_all()?.f_int64_value(&[])? != 0);
    let mut bounded = TensorRng::new(0).with_limits(ResourceLimits {
        max_tensor_elements: 2,
        ..Default::default()
    });
    let saved = bounded.state_dict();
    assert!(bounded.normal(&[3], (Kind::Float, Device::Cpu)).is_err());
    assert!(bounded.normal(&[-1], (Kind::Float, Device::Cpu)).is_err());
    assert!(bounded.normal(&[1], (Kind::Int64, Device::Cpu)).is_err());
    assert_eq!(saved, bounded.state_dict());
    let invalid: TensorRngState =
        serde_json::from_str("{\"version\":2,\"seed\":0,\"next_stream\":0}").unwrap();
    assert!(TensorRng::from_state(invalid).is_err());
    Ok(())
}

#[test]
fn profiles_are_bounded_thread_safe_and_export_only_completed_regions() -> Result<()> {
    use rusttorch::profiling::Profiler;
    let profile = Profiler::new(4)?;
    let outer = profile.span("outer")?;
    assert!(profile.chrome_trace().is_err());
    assert!(profile.clear().is_err());
    std::thread::scope(|scope| {
        scope.spawn(|| profile.record("worker", || Ok(())).unwrap());
    });
    profile.record("quoted\"name", || Ok(()))?;
    drop(outer);
    let error = profile.record("failure", || -> Result<()> {
        Err(rusttorch::RustTorchError::GraphValidation("test".into()))
    });
    assert!(error.is_err());
    let called = std::cell::Cell::new(false);
    assert!(
        profile
            .record("too many", || {
                called.set(true);
                Ok(())
            })
            .is_err()
    );
    assert!(!called.get());
    let json: serde_json::Value = serde_json::from_str(&profile.chrome_trace()?).unwrap();
    let events = json["traceEvents"].as_array().unwrap();
    assert_eq!(events.len(), 4);
    assert_ne!(events[0]["tid"], events[1]["tid"]);
    assert_eq!(events[2]["name"], "quoted\"name");
    assert_eq!(events[3]["args"]["outcome"], "error");
    assert!(
        events
            .iter()
            .all(|event| event["ph"] == "X" && event["dur"].as_f64().unwrap() >= 0.)
    );
    profile.clear()?;
    assert!(
        serde_json::from_str::<serde_json::Value>(&profile.chrome_trace()?).unwrap()["traceEvents"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    Ok(())
}

#[test]
fn benchmarks_warm_up_and_synchronize_each_completed_invocation() -> Result<()> {
    use rusttorch::profiling::{BenchmarkOptions, benchmark};
    use std::cell::Cell;
    let work = Cell::new(0);
    let sync = Cell::new(0);
    let sample = benchmark(
        BenchmarkOptions {
            warmup: 2,
            samples: 4,
        },
        || {
            work.set(work.get() + 1);
            Ok(work.get())
        },
        || {
            sync.set(sync.get() + 1);
            Ok(())
        },
    )?;
    assert_eq!(work.get(), 6);
    assert_eq!(sync.get(), 12);
    assert_eq!(sample.seconds.len(), 4);
    assert!(sample.minimum <= sample.median && sample.median <= sample.maximum);
    assert!(sample.minimum <= sample.mean && sample.mean <= sample.maximum);
    assert!(sample.standard_deviation.is_finite());
    assert!(
        benchmark(
            BenchmarkOptions {
                warmup: 0,
                samples: 0
            },
            || Ok(()),
            || Ok(())
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn gradient_check_rejects_matching_infinite_derivatives() {
    let result = testing::gradcheck(
        |x| {
            let forward = x.f_sign()?.f_mul_scalar(1e308)?;
            let surrogate = x.f_mul_scalar(1e308)?.f_mul_scalar(2.)?;
            rusttorch::autograd::with_surrogate_gradient(&forward, &surrogate)
        },
        &Tensor::from_slice(&[0_f64]),
        GradcheckOptions {
            epsilon: 1e-308,
            ..Default::default()
        },
    );
    assert!(
        result.is_err(),
        "two overflowed Jacobians cannot establish a valid derivative"
    );
}

#[test]
#[ignore = "requires the locked PyTorch reference environment"]
fn pinned_python_comparisons_and_gradcheck_match() -> Result<()> {
    let python = std::env::var("RUSTTORCH_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = std::process::Command::new(python)
        .arg("tests/python_reference/framework_tools.py")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let actual =
        Tensor::from_slice(&[1_f64, f64::NAN, f64::INFINITY, -f64::INFINITY]).reshape([2, 2]);
    let expected =
        Tensor::from_slice(&[1.1_f64, f64::NAN, f64::INFINITY, f64::INFINITY]).reshape([2, 2]);
    for (i, equal_nan) in [false, true].into_iter().enumerate() {
        let report = testing::compare(
            &actual,
            &expected,
            CloseOptions {
                equal_nan,
                ..Default::default()
            },
        )?;
        assert_eq!(
            report.mismatches as u64,
            reference["comparisons"][i]["mismatches"].as_u64().unwrap()
        );
        assert_eq!(
            serde_json::to_value(report.first_mismatch).unwrap(),
            reference["comparisons"][i]["first"]
        );
    }
    let x = Tensor::from_slice(&[0.2_f64, -0.4, 0.1, 0.3])
        .reshape([2, 2])
        .transpose(0, 1);
    let good = testing::gradcheck(|x| Ok(x.f_tanh()?), &x, GradcheckOptions::default()).is_ok();
    let bad = testing::gradcheck(
        |x| rusttorch::autograd::with_surrogate_gradient(&x.f_square()?, x),
        &x,
        GradcheckOptions::default(),
    )
    .is_ok();
    assert_eq!(good, reference["gradcheck_good"].as_bool().unwrap());
    assert_eq!(bad, reference["gradcheck_bad"].as_bool().unwrap());
    Ok(())
}
