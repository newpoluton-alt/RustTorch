use rusttorch::{
    Device, Kind, Result, Tensor,
    autograd::{self, GradOptions},
    distributions::{Bernoulli, Categorical, Normal},
};

fn close(actual: &Tensor, expected: &[f64]) {
    let values = Vec::<f64>::try_from(actual.to_kind(Kind::Double).reshape([-1])).unwrap();
    assert_eq!(values.len(), expected.len());
    for (a, b) in values.iter().zip(expected) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}

#[test]
fn continuous_samples_and_scores_have_the_expected_gradient_boundary() -> Result<()> {
    let mean = Tensor::from_slice(&[0_f64, 2.]).set_requires_grad(true);
    let scale = Tensor::from(0.5_f64).set_requires_grad(true);
    let distribution = Normal::new(&mean, &scale)?;
    assert_eq!(distribution.batch_shape(), [2]);
    assert_eq!(distribution.rsample(&[3, 4])?.size(), [3, 4, 2]);
    assert!(!distribution.sample(&[2])?.requires_grad());
    let draws = distribution.rsample(&[5])?;
    let grads = autograd::grad(
        &draws.sum(Kind::Double),
        &[&mean, &scale],
        GradOptions::default(),
    )?;
    close(&grads[0], &[5., 5.]);
    assert!(grads[1].isfinite().all().int64_value(&[]) != 0);
    close(&distribution.variance()?, &[0.25, 0.25]);
    close(
        &distribution.log_prob(&mean)?,
        &[-0.5 * (2. * std::f64::consts::PI).ln() + 2_f64.ln(); 2],
    );
    assert_eq!(distribution.sample(&[0])?.size(), [0, 2]);
    assert!(distribution.sample(&[-1]).is_err());
    Ok(())
}

#[test]
fn discrete_distributions_broadcast_scores_and_preserve_parameter_gradients() -> Result<()> {
    let probabilities = Tensor::from_slice(&[0_f64, 1., 0.25]).set_requires_grad(true);
    let bernoulli = Bernoulli::from_probs(&probabilities)?;
    assert!(!bernoulli.sample(&[2])?.requires_grad());
    close(&bernoulli.sample(&[])?.narrow(0, 0, 2), &[0., 1.]);
    close(
        &bernoulli.log_prob(&Tensor::from(1_f64))?.narrow(0, 2, 1),
        &[0.25_f64.ln()],
    );
    let logits = Tensor::from_slice(&[0_f64, 0., 0., 1., 2., 3.])
        .reshape([2, 3])
        .set_requires_grad(true);
    let classes = Categorical::from_logits(&logits)?;
    let labels = Tensor::from_slice(&[0_i64, 2, 1, 0]).reshape([2, 2]);
    let scores = classes.log_prob(&labels)?;
    assert_eq!(scores.size(), [2, 2]);
    assert_eq!(classes.sample(&[4, 5])?.size(), [4, 5, 2]);
    assert_eq!(classes.sample(&[0])?.size(), [0, 2]);
    let gradients = autograd::grad(
        &scores.sum(Kind::Double),
        &[&logits],
        GradOptions::default(),
    )?;
    close(
        &gradients[0].sum_dim_intlist([-1].as_slice(), false, Kind::Double),
        &[0., 0.],
    );
    let certain = Categorical::from_probs(&Tensor::from_slice(&[0_f64, 7.]))?;
    close(&certain.sample(&[5])?, &[1.; 5]);
    assert!(certain.entropy()?.double_value(&[]).is_finite());
    for (kind, extent) in [
        (Kind::Float, 3e38),
        (Kind::Half, 60000.),
        (Kind::Double, 1e308),
    ] {
        let extreme =
            Categorical::from_logits(&Tensor::from_slice(&[-extent, extent]).to_kind(kind))?;
        close(&extreme.entropy()?, &[0.]);
    }
    let empty = Categorical::from_logits(&Tensor::zeros([0, 3], (Kind::Double, Device::Cpu)))?;
    assert_eq!(empty.sample(&[2])?.size(), [2, 0]);
    Ok(())
}

#[test]
fn invalid_distribution_parameters_and_observations_return_errors() -> Result<()> {
    for scale in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(Normal::new(&Tensor::from(0_f64), &Tensor::from(scale)).is_err());
    }
    assert!(Normal::new(&Tensor::from(0_f64), &Tensor::from(1_f32)).is_err());
    assert!(Bernoulli::from_probs(&Tensor::from(-0.1_f64)).is_err());
    assert!(Bernoulli::from_logits(&Tensor::from(f64::INFINITY)).is_err());
    assert!(
        Bernoulli::from_probs(&Tensor::from(0.5_f64))?
            .log_prob(&Tensor::from(0.5_f64))
            .is_err()
    );
    for p in [
        Tensor::from(1_f64),
        Tensor::from_slice(&[0_f64, 0.]),
        Tensor::from_slice(&[-1_f64, 2.]),
        Tensor::zeros([0], (Kind::Double, Device::Cpu)),
        Tensor::new(),
    ] {
        assert!(Categorical::from_probs(&p).is_err());
    }
    let categorical = Categorical::from_logits(&Tensor::from_slice(&[1_f64, 2.]))?;
    assert!(categorical.log_prob(&Tensor::from(2_i64)).is_err());
    assert!(categorical.log_prob(&Tensor::from(1_f64)).is_err());
    assert!(categorical.sample(&[i64::MAX, 2]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires scripts/run-python-parity.sh"]
fn distribution_scores_samples_and_gradients_match_pinned_python() -> Result<()> {
    let root = std::env::var("RUSTTORCH_PYTHON_REFERENCE_DIR").expect("run the parity script");
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::path::Path::new(&root).join("differentiation.json")).unwrap(),
    )
    .unwrap();
    fn flatten(v: &serde_json::Value, out: &mut Vec<f64>) {
        if let Some(a) = v.as_array() {
            for v in a {
                flatten(v, out);
            }
        } else {
            out.push(v.as_f64().unwrap());
        }
    }
    let compare = |actual: &Tensor, expected: &serde_json::Value| {
        let mut flat = vec![];
        flatten(expected, &mut flat);
        close(actual, &flat);
    };
    let mean = Tensor::from_slice(&[0.1_f64, -0.3]).set_requires_grad(true);
    let scale = Tensor::from(0.7_f64).set_requires_grad(true);
    let normal = Normal::new(&mean, &scale)?;
    rusttorch::manual_seed(847);
    let draws = normal.rsample(&[3])?;
    compare(&draws, &reference["normal_draws"]);
    for (i, g) in autograd::grad(
        &draws.square().sum(Kind::Double),
        &[&mean, &scale],
        GradOptions::default(),
    )?
    .iter()
    .enumerate()
    {
        compare(g, &reference["normal_draw_grads"][i]);
    }
    compare(
        &normal.log_prob(&Tensor::from_slice(&[0.4_f64, -1.2]).reshape([2, 1]))?,
        &reference["normal_scores"],
    );
    compare(&normal.entropy()?, &reference["normal_entropy"]);
    for family in ["bernoulli", "categorical"] {
        for parameter in ["probs", "logits"] {
            let p = Tensor::from_slice(if parameter == "probs" {
                &[0.2_f64, 0.3, 0.5, 0.1, 0.7, 0.2]
            } else {
                &[-1.2_f64, 0.3, 2.1, 0.1, 0.7, -0.2]
            })
            .reshape([2, 3])
            .set_requires_grad(true);
            let (scores, entropy, samples) = if family == "bernoulli" {
                let distribution = if parameter == "probs" {
                    Bernoulli::from_probs(&p)?
                } else {
                    Bernoulli::from_logits(&p)?
                };
                rusttorch::manual_seed(847);
                (
                    distribution.log_prob(&Tensor::from_slice(&[1_f64, 0., 1.]))?,
                    distribution.entropy()?,
                    distribution.sample(&[3])?,
                )
            } else {
                let distribution = if parameter == "probs" {
                    Categorical::from_probs(&p)?
                } else {
                    Categorical::from_logits(&p)?
                };
                rusttorch::manual_seed(847);
                (
                    distribution.log_prob(&Tensor::from_slice(&[0_i64, 2]).reshape([2, 1]))?,
                    distribution.entropy()?,
                    distribution.sample(&[3])?,
                )
            };
            let item = &reference[format!("{family}_{parameter}")];
            compare(&scores, &item["scores"]);
            compare(&entropy, &item["entropy"]);
            compare(&samples, &item["samples"]);
            compare(
                &autograd::grad(
                    &scores.f_add(&entropy)?.f_sum(Kind::Double)?,
                    &[&p],
                    GradOptions::default(),
                )?
                .remove(0),
                &item["grad"],
            );
        }
    }
    Ok(())
}
