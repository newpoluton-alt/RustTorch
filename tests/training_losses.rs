use rusttorch::{Device, Kind, Reduction, Result, Tensor, nn::functional as f};

#[test]
fn regression_losses_cover_reductions_transitions_and_gradients() -> Result<()> {
    let input = Tensor::from_slice(&[1_f32, 3.]).set_requires_grad(true);
    let target = Tensor::from_slice(&[0_f32, 1.]);
    assert_eq!(
        Vec::<f32>::try_from(f::mse_loss_with_reduction(
            &input,
            &target,
            Reduction::None
        )?)?,
        [1., 4.]
    );
    assert_eq!(
        f::mse_loss_with_reduction(&input, &target, Reduction::Sum)?.double_value(&[]),
        5.
    );
    assert_eq!(
        f::l1_loss(&input, &target, Reduction::Mean)?.double_value(&[]),
        1.5
    );
    assert_eq!(
        f::smooth_l1_loss(&input, &target, 1., Reduction::Sum)?.double_value(&[]),
        2.
    );
    assert_eq!(
        f::smooth_l1_loss(&input, &target, 0., Reduction::Sum)?.double_value(&[]),
        3.
    );
    let huber = f::huber_loss(&input, &target, 2., Reduction::Sum)?;
    assert_eq!(huber.double_value(&[]), 2.5);
    huber.f_backward()?;
    assert_eq!(Vec::<f32>::try_from(input.grad())?, [1., 2.]);
    Ok(())
}

#[test]
fn binary_and_weighted_classification_losses_backpropagate() -> Result<()> {
    let input = Tensor::from_slice(&[0_f32, 0.]).set_requires_grad(true);
    let target = Tensor::from_slice(&[0_f32, 1.]);
    let weights = Tensor::from_slice(&[1_f32, 3.]);
    let positive = Tensor::from_slice(&[2_f32]);
    let loss = f::binary_cross_entropy_with_logits(
        &input,
        &target,
        Some(&weights),
        Some(&positive),
        Reduction::Sum,
    )?;
    assert!((loss.double_value(&[]) - 7. * 2_f64.ln()).abs() < 1e-6);
    loss.f_backward()?;
    assert_eq!(Vec::<f32>::try_from(input.grad())?, [0.5, -3.]);
    let probability = Tensor::from_slice(&[0.5_f32, 0.5]);
    assert!(
        (f::binary_cross_entropy(&probability, &target, None, Reduction::Mean)?.double_value(&[])
            - 2_f64.ln())
        .abs()
            < 1e-6
    );

    let logits = Tensor::zeros([3, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let classes = Tensor::from_slice(&[0_i64, -1, 1]);
    let options = f::CrossEntropyOptions {
        weight: Some(&weights),
        ignore_index: -1,
        reduction: Reduction::None,
        label_smoothing: 0.2,
    };
    let result = f::cross_entropy_with_options(&logits, &classes, options)?;
    assert_eq!(result.size(), [3]);
    assert_eq!(result.double_value(&[1]), 0.);
    result.sum(Kind::Float).f_backward()?;
    assert_eq!(Vec::<f32>::try_from(logits.grad().get(1))?, [0., 0.]);
    let log_probs = logits.f_log_softmax(-1, Kind::Float)?;
    let nll = f::nll_loss(&log_probs, &classes, Some(&weights), -1, Reduction::Mean)?;
    assert!((nll.double_value(&[]) - 2_f64.ln()).abs() < 1e-6);
    Ok(())
}

#[test]
fn loss_validation_rejects_invalid_settings_and_undefined_tensors() {
    let input = Tensor::ones([2], (Kind::Float, Device::Cpu));
    for threshold in [-1., f64::NAN, f64::INFINITY] {
        assert!(f::smooth_l1_loss(&input, &input, threshold, Reduction::Mean).is_err());
        assert!(f::huber_loss(&input, &input, threshold, Reduction::Mean).is_err());
    }
    assert!(f::huber_loss(&input, &input, 0., Reduction::Mean).is_err());
    for smoothing in [-0.1, 1.1, f64::NAN] {
        assert!(
            f::cross_entropy_with_options(
                &input,
                &input,
                f::CrossEntropyOptions {
                    label_smoothing: smoothing,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    assert!(f::mse_loss_with_reduction(&input, &input, Reduction::Other(3)).is_err());
    assert!(f::l1_loss(&Tensor::new(), &input, Reduction::Mean).is_err());
    assert!(f::binary_cross_entropy(&input, &Tensor::new(), None, Reduction::Mean).is_err());
    assert!(
        f::binary_cross_entropy_with_logits(
            &input,
            &input,
            Some(&Tensor::new()),
            None,
            Reduction::Mean
        )
        .is_err()
    );
    assert!(
        f::binary_cross_entropy_with_logits(
            &input,
            &input,
            None,
            Some(&Tensor::new()),
            Reduction::Mean
        )
        .is_err()
    );
    assert!(
        f::binary_cross_entropy_with_logits(&input, &input.get(0), None, None, Reduction::Mean)
            .is_err()
    );
    assert!(f::nll_loss(&input, &input, Some(&Tensor::new()), -100, Reduction::Mean).is_err());
}

#[test]
#[ignore = "run through scripts/run-python-parity.sh"]
fn configured_losses_match_python_values_and_gradients() -> Result<()> {
    use serde_json::Value;
    let directory = std::env::var_os("RUSTTORCH_PYTHON_REFERENCE_DIR").expect("parity environment");
    let reference: Value = serde_json::from_slice(
        &std::fs::read(std::path::PathBuf::from(directory).join("training.json")).unwrap(),
    )
    .unwrap();
    for name in [
        "mse_none",
        "l1_mean",
        "smooth_l1",
        "huber",
        "bce",
        "bce_logits",
        "ce",
        "ce_probabilities",
        "nll",
    ] {
        let input = match name {
            "bce" => Tensor::from_slice(&[0.2_f32, 0.7, 0.9]),
            "bce_logits" => Tensor::from_slice(&[-1_f32, 0.5, 2.]),
            "ce" | "ce_probabilities" | "nll" => {
                Tensor::from_slice(&[1_f32, -1., 2., 0.5, 1.5, -0.5, -2., 1., 0.]).reshape([3, 3])
            }
            _ => Tensor::from_slice(&[-1.5_f32, 0.25, 2., 4.]),
        }
        .set_requires_grad(true);
        let target = Tensor::from_slice(&[0.5_f32, -0.25, 1., 2.]);
        let binary = Tensor::from_slice(&[0_f32, 1., 0.]);
        let weights = Tensor::from_slice(&[1_f32, 2., 0.5]);
        let positive = Tensor::from_slice(&[2_f32]);
        let classes = Tensor::from_slice(&[2_i64, -1, 0]);
        let class_weights = Tensor::from_slice(&[0.5_f32, 2., 1.]);
        let probabilities =
            Tensor::from_slice(&[0.1_f32, 0.2, 0.7, 0., 0.5, 0.5, 1., 0., 0.]).reshape([3, 3]);
        let loss = match name {
            "mse_none" => f::mse_loss_with_reduction(&input, &target, Reduction::None)?,
            "l1_mean" => f::l1_loss(&input, &target, Reduction::Mean)?,
            "smooth_l1" => f::smooth_l1_loss(&input, &target, 0.75, Reduction::Sum)?,
            "huber" => f::huber_loss(&input, &target, 1.5, Reduction::Sum)?,
            "bce" => f::binary_cross_entropy(&input, &binary, Some(&weights), Reduction::Sum)?,
            "bce_logits" => f::binary_cross_entropy_with_logits(
                &input,
                &binary,
                Some(&weights),
                Some(&positive),
                Reduction::Sum,
            )?,
            "ce" => f::cross_entropy_with_options(
                &input,
                &classes,
                f::CrossEntropyOptions {
                    weight: Some(&class_weights),
                    ignore_index: -1,
                    label_smoothing: 0.15,
                    ..Default::default()
                },
            )?,
            "ce_probabilities" => f::cross_entropy_with_options(
                &input,
                &probabilities,
                f::CrossEntropyOptions {
                    weight: Some(&class_weights),
                    label_smoothing: 0.1,
                    reduction: Reduction::Sum,
                    ..Default::default()
                },
            )?,
            "nll" => f::nll_loss(
                &input.log_softmax(-1, Kind::Float),
                &classes,
                Some(&class_weights),
                -1,
                Reduction::Mean,
            )?,
            _ => unreachable!(),
        };
        loss.sum(Kind::Float).f_backward()?;
        let expected = &reference["losses"][name];
        fn flatten(value: &Value, out: &mut Vec<f64>) {
            if let Some(items) = value.as_array() {
                for item in items {
                    flatten(item, out);
                }
            } else {
                out.push(value.as_f64().expect("numeric reference"));
            }
        }
        for (field, tensor) in [("loss", loss), ("grad", input.grad())] {
            let mut expected_values = Vec::new();
            flatten(&expected[field], &mut expected_values);
            let actual = Vec::<f32>::try_from(tensor.reshape([-1]))?;
            assert_eq!(actual.len(), expected_values.len(), "{name}.{field}");
            for (actual, expected) in actual.into_iter().zip(expected_values) {
                assert!(
                    (actual as f64 - expected).abs() <= 1e-6 + 1e-5 * expected.abs(),
                    "{name}.{field}: {actual} != {expected}"
                );
            }
        }
    }
    Ok(())
}
