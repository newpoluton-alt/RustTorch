use std::process::Command;

use rusttorch::{
    Device, Kind, Result, Tensor,
    nn::{VarStore, functional},
    no_grad,
    optim::{
        self, CosineAnnealingLr, ExponentialLr, MultiStepLr, Optimizer, OptimizerState,
        ReduceLrOnPlateau, StepLr, ThresholdMode,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const ALGORITHMS: [&str; 7] = [
    "adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax",
];

fn model(name: &str) -> Result<(VarStore, Tensor, Tensor, Optimizer)> {
    let mut store = VarStore::new(Device::Cpu);
    store.set_kind(Kind::Double);
    let p = store
        .root()
        .f_var_copy("p", &Tensor::from_slice(&[0.4_f64, -0.7, 1.2]))?;
    let q = store
        .root()
        .set_group(3)
        .f_var_copy("q", &Tensor::from_slice(&[-0.2_f64, 0.5]))?;
    let mut optimizer = match name {
        "adam" => optim::Adam::builder()
            .learning_rate(0.03)
            .weight_decay(0.02)
            .betas(0.8, 0.91)
            .eps(1e-6)
            .amsgrad(true)
            .build(&store)?,
        "adamw" => optim::AdamW::builder()
            .learning_rate(0.03)
            .weight_decay(0.02)
            .betas(0.8, 0.91)
            .eps(1e-6)
            .amsgrad(true)
            .build(&store)?,
        "sgd" => optim::Sgd::builder()
            .learning_rate(0.03)
            .weight_decay(0.02)
            .momentum(0.8)
            .nesterov(true)
            .build(&store)?,
        "rmsprop" => optim::RmsProp::builder()
            .learning_rate(0.03)
            .weight_decay(0.02)
            .alpha(0.8)
            .eps(1e-5)
            .momentum(0.7)
            .centered(true)
            .build(&store)?,
        "adagrad" => optim::Adagrad::builder()
            .learning_rate(0.03)
            .weight_decay(0.02)
            .lr_decay(0.03)
            .initial_accumulator_value(0.2)
            .eps(1e-6)
            .build(&store)?,
        "adadelta" => optim::Adadelta::builder()
            .learning_rate(0.03)
            .weight_decay(0.02)
            .rho(0.8)
            .eps(1e-5)
            .build(&store)?,
        "adamax" => optim::Adamax::builder()
            .learning_rate(0.03)
            .weight_decay(0.02)
            .betas(0.8, 0.91)
            .eps(1e-6)
            .build(&store)?,
        _ => unreachable!(),
    };
    optimizer.set_group_learning_rate(3, 0.012)?;
    optimizer.set_group_weight_decay(3, 0.)?;
    Ok((store, p, q, optimizer))
}

fn train_step(optimizer: &mut Optimizer, p: &Tensor, q: &Tensor, step: usize) -> Result<()> {
    if step == 3 {
        optimizer.set_group_learning_rate(3, 0.007)?;
    }
    if step == 6 {
        optimizer.set_learning_rate(0.009)?;
    }
    optimizer.try_zero_grad()?;
    assert!(!p.grad().defined() && !q.grad().defined());
    let mut loss = Tensor::zeros([], (Kind::Double, Device::Cpu));
    if step != 5 {
        let gradient = Tensor::from_slice(&[
            (step + 1) as f64 * 0.1,
            -0.2,
            (step % 3) as f64 * 0.15 - 0.15,
        ]);
        loss = loss.f_add(&p.f_mul(&gradient)?.f_sum(Kind::Double)?)?;
    }
    if step != 1 && step != 4 {
        let gradient = Tensor::from_slice(&[0.3, -0.1 * (step + 1) as f64]);
        loss = loss.f_add(&q.f_mul(&gradient)?.f_sum(Kind::Double)?)?;
    }
    loss.f_backward()?;
    optimizer.try_step()
}

fn weights(p: &Tensor, q: &Tensor) -> Result<Vec<f64>> {
    let mut values = Vec::<f64>::try_from(p)?;
    values.extend(Vec::<f64>::try_from(q)?);
    Ok(values)
}

fn assign(target: &Tensor, source: &Tensor) -> Result<()> {
    no_grad(|| target.shallow_clone().f_copy_(source))?;
    Ok(())
}

fn json_roundtrip(state: &OptimizerState) -> OptimizerState {
    serde_json::from_slice(&serde_json::to_vec(state).unwrap()).unwrap()
}

#[test]
fn all_optimizer_moments_and_group_settings_resume_exactly_at_every_step() -> Result<()> {
    for name in ALGORITHMS {
        let (_, p, q, mut uninterrupted) = model(name)?;
        for step in 0..8 {
            train_step(&mut uninterrupted, &p, &q, step)?;
        }
        let expected_weights = weights(&p, &q)?;
        let expected_state = uninterrupted.state_dict()?;
        for boundary in 0..=8 {
            let (_, p, q, mut original) = model(name)?;
            for step in 0..boundary {
                train_step(&mut original, &p, &q, step)?;
            }
            let saved = json_roundtrip(&original.state_dict()?);
            let (_, rp, rq, mut resumed) = model(name)?;
            assign(&rp, &p)?;
            assign(&rq, &q)?;
            resumed.load_state_dict(&saved)?;
            for step in boundary..8 {
                train_step(&mut resumed, &rp, &rq, step)?;
            }
            assert_eq!(
                weights(&rp, &rq)?,
                expected_weights,
                "{name} boundary {boundary}"
            );
            assert_eq!(
                resumed.state_dict()?,
                expected_state,
                "{name} boundary {boundary}"
            );
        }
    }
    Ok(())
}

#[test]
fn optimizer_checkpoint_validation_rejects_corruption_before_any_mutation() -> Result<()> {
    let (_, p, q, mut optimizer) = model("adam")?;
    train_step(&mut optimizer, &p, &q, 0)?;
    let valid = optimizer.state_dict()?;
    let source = serde_json::to_value(&valid).unwrap();
    for field in [
        "version",
        "algorithm",
        "name",
        "shape",
        "dtype",
        "group",
        "membership",
        "group_count",
        "rate",
        "counter",
        "slot",
        "payload",
        "moment_shape",
    ] {
        let mut bad = source.clone();
        match field {
            "version" => bad["schema_version"] = 999.into(),
            "algorithm" => {
                bad["algorithm"] =
                    serde_json::json!({"Sgd":{"momentum":0.,"dampening":0.,"nesterov":false}})
            }
            "name" => {
                let value = bad["parameters"]
                    .as_object_mut()
                    .unwrap()
                    .remove("p")
                    .unwrap();
                bad["parameters"]["different"] = value;
            }
            "shape" => bad["parameters"]["q"]["shape"] = serde_json::json!([3]),
            "dtype" => bad["parameters"]["q"]["dtype"] = "Float32".into(),
            "group" => bad["parameters"]["q"]["group"] = 0.into(),
            "membership" => bad["groups"][1]["parameters"] = serde_json::json!(["p"]),
            "group_count" => {
                let duplicate = bad["groups"][0].clone();
                bad["groups"].as_array_mut().unwrap().push(duplicate);
            }
            "rate" => bad["groups"][1]["learning_rate"] = (-0.01).into(),
            "counter" => bad["parameters"]["q"]["step"] = 7.into(),
            "slot" => {
                bad["parameters"]["q"]["slots"]
                    .as_object_mut()
                    .unwrap()
                    .remove("exp_avg");
            }
            "payload" => {
                bad["parameters"]["q"]["slots"]["exp_avg"]["bytes"] = serde_json::json!([0])
            }
            "moment_shape" => {
                bad["parameters"]["q"]["slots"]["exp_avg"]["shape"] = serde_json::json!([-1])
            }
            _ => unreachable!(),
        }
        let decoded: OptimizerState = serde_json::from_value(bad).unwrap();
        let before = weights(&p, &q)?;
        assert!(
            optimizer.load_state_dict(&decoded).is_err(),
            "accepted {field}"
        );
        assert_eq!(optimizer.state_dict()?, valid, "mutated state for {field}");
        assert_eq!(weights(&p, &q)?, before, "mutated weights for {field}");
    }
    Ok(())
}

#[test]
fn dynamic_registration_and_undefined_gradients_preserve_group_and_skip_semantics() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let p = store.root().f_ones("p", &[1])?;
    let mut optimizer = optim::Sgd::builder()
        .momentum(0.9)
        .weight_decay(0.1)
        .build(&store)?;
    optimizer.backward_step(&p.f_sum(Kind::Float)?)?;
    let q = store.root().set_group(7).f_ones("q", &[1])?;
    assert_eq!(optimizer.trainable_variables().len(), 2);
    optimizer.try_zero_grad()?;
    assert!(!p.grad().defined());
    let before = p.double_value(&[0]);
    q.f_sum(Kind::Float)?.f_backward()?;
    optimizer.try_step()?;
    assert_eq!(p.double_value(&[0]), before);
    assert!(q.double_value(&[0]) < 1.);
    assert_eq!(
        optimizer
            .parameter_groups()
            .iter()
            .map(|g| g.id)
            .collect::<Vec<_>>(),
        [0, 7]
    );
    assert!(optimizer.set_group_learning_rate(8, 0.1).is_err());
    assert!(optimizer.set_group_weight_decay(7, f64::NAN).is_err());
    Ok(())
}

#[test]
fn gradient_validation_and_sparse_rejection_happen_before_any_update() -> Result<()> {
    let (_, p, q, mut optimizer) = model("adam")?;
    p.f_sum(Kind::Double)?
        .f_add(&q.f_sum(Kind::Double)?)?
        .f_backward()?;
    let before = optimizer.state_dict()?;
    let original = weights(&p, &q)?;
    q.grad()
        .f_set_data(&Tensor::zeros([3], (Kind::Double, Device::Cpu)))?;
    assert!(optimizer.try_step().is_err());
    assert_eq!(weights(&p, &q)?, original);
    assert_eq!(optimizer.state_dict()?, before);

    let store = VarStore::new(Device::Cpu);
    let dense = store.root().f_ones("a", &[1])?;
    let embedding = store.root().f_ones("z", &[3, 2])?;
    let mut adam = optim::Adam::builder().build(&store)?;
    let sparse = functional::embedding(
        &Tensor::from_slice(&[1_i64, 1]),
        &embedding,
        None,
        false,
        true,
    )?;
    dense
        .f_sum(Kind::Float)?
        .f_add(&sparse.f_sum(Kind::Float)?)?
        .f_backward()?;
    let before = adam.state_dict()?;
    assert!(adam.try_step().is_err());
    assert_eq!(dense.double_value(&[0]), 1.);
    assert_eq!(adam.state_dict()?, before);
    let mut sgd = optim::Sgd::builder().learning_rate(0.1).build(&store)?;
    sgd.try_step()?;
    assert!((embedding.double_value(&[1, 0]) - 0.8).abs() < 1e-6);
    assert_eq!(embedding.double_value(&[0, 0]), 1.);
    Ok(())
}

#[test]
fn checkpoint_capture_rejects_parameter_dtype_or_shape_drift() -> Result<()> {
    let mut store = VarStore::new(Device::Cpu);
    let parameter = store.root().f_ones("weight", &[2])?;
    let mut optimizer = optim::Adam::builder().build(&store)?;
    optimizer.backward_step(&parameter.f_sum(Kind::Float)?)?;
    let valid = optimizer.state_dict()?;
    store.double();
    assert!(optimizer.state_dict().is_err());
    store.float();
    assert_eq!(optimizer.state_dict()?, valid);
    no_grad(|| {
        parameter
            .shallow_clone()
            .f_set_data(&Tensor::ones([3], (Kind::Float, Device::Cpu)))
    })?;
    assert!(optimizer.state_dict().is_err());
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Schedule {
    Step(StepLr),
    Exponential(ExponentialLr),
    Multi(MultiStepLr),
    Cosine(CosineAnnealingLr),
    Plateau(ReduceLrOnPlateau),
}
impl Schedule {
    fn new(name: &str, optimizer: &mut Optimizer) -> Result<Self> {
        Ok(match name {
            "step" => Self::Step(StepLr::new(optimizer, 2, 0.5)?),
            "exponential" => Self::Exponential(ExponentialLr::new(optimizer, 0.8)?),
            "multi" => Self::Multi(MultiStepLr::new(optimizer, vec![0, 2, 2, 5], 0.5)?),
            "cosine" => Self::Cosine(CosineAnnealingLr::new(optimizer, 3, 0.005)?),
            "plateau" => Self::Plateau(
                ReduceLrOnPlateau::new(optimizer)?
                    .factor(0.5)?
                    .patience(1)
                    .threshold(0.01, ThresholdMode::Absolute)?
                    .cooldown(1)
                    .min_learning_rates(vec![0.02, 0.005])?
                    .eps(1e-10)?,
            ),
            _ => unreachable!(),
        })
    }
    fn step(&mut self, optimizer: &mut Optimizer, metric: f64) -> Result<()> {
        match self {
            Self::Step(s) => s.step(optimizer),
            Self::Exponential(s) => s.step(optimizer),
            Self::Multi(s) => s.step(optimizer),
            Self::Cosine(s) => s.step(optimizer),
            Self::Plateau(s) => s.step(optimizer, metric),
        }
    }
}
const SCHEDULES: [&str; 5] = ["step", "exponential", "multi", "cosine", "plateau"];
const METRICS: [f64; 10] = [1., 1., 1., 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8];

fn scheduler_model(name: &str) -> Result<(Optimizer, Schedule)> {
    let (_, _, _, mut optimizer) = model("sgd")?;
    optimizer.set_group_learning_rate(0, 0.1)?;
    optimizer.set_group_learning_rate(3, 0.04)?;
    let schedule = Schedule::new(name, &mut optimizer)?;
    Ok((optimizer, schedule))
}
fn rates(optimizer: &Optimizer) -> Vec<f64> {
    optimizer
        .parameter_groups()
        .iter()
        .map(|g| g.learning_rate)
        .collect()
}

#[test]
fn all_schedulers_resume_exactly_at_every_epoch_including_plateau_cooldown() -> Result<()> {
    for name in SCHEDULES {
        let (mut optimizer, mut schedule) = scheduler_model(name)?;
        for metric in METRICS {
            optimizer.try_step()?;
            schedule.step(&mut optimizer, metric)?;
        }
        let expected_rates = rates(&optimizer);
        let expected = serde_json::to_value(&schedule).unwrap();
        for boundary in 0..=METRICS.len() {
            let (mut optimizer, mut schedule) = scheduler_model(name)?;
            for &metric in &METRICS[..boundary] {
                optimizer.try_step()?;
                schedule.step(&mut optimizer, metric)?;
            }
            let saved_optimizer = json_roundtrip(&optimizer.state_dict()?);
            let saved_schedule = serde_json::to_vec(&schedule).unwrap();
            let (mut resumed, _) = scheduler_model(name)?;
            resumed.load_state_dict(&saved_optimizer)?;
            let mut schedule: Schedule = serde_json::from_slice(&saved_schedule).unwrap();
            for &metric in &METRICS[boundary..] {
                resumed.try_step()?;
                schedule.step(&mut resumed, metric)?;
            }
            assert_eq!(rates(&resumed), expected_rates, "{name} epoch {boundary}");
            assert_eq!(
                serde_json::to_value(schedule).unwrap(),
                expected,
                "{name} epoch {boundary}"
            );
        }
    }
    Ok(())
}

#[test]
fn malformed_scheduler_state_and_overflow_cannot_partially_update_groups() -> Result<()> {
    let (mut optimizer, _) = scheduler_model("step")?;
    let schedule = StepLr::new(&optimizer, 1, 0.5)?;
    for field in [
        "schema_version",
        "epoch",
        "group_ids",
        "base_rates",
        "step_size",
        "gamma",
    ] {
        let mut bad = serde_json::to_value(&schedule).unwrap();
        match field {
            "schema_version" => bad["state"][field] = 2.into(),
            "epoch" => bad["state"][field] = u64::MAX.into(),
            "group_ids" => bad["state"][field] = serde_json::json!([0, 99]),
            "base_rates" => bad["state"][field] = serde_json::json!([]),
            "step_size" => bad[field] = 0.into(),
            "gamma" => bad[field] = (-0.1).into(),
            _ => unreachable!(),
        }
        let mut bad: StepLr = serde_json::from_value(bad).unwrap();
        let prior = rates(&optimizer);
        assert!(bad.step(&mut optimizer).is_err(), "accepted {field}");
        assert_eq!(rates(&optimizer), prior);
    }
    optimizer.set_group_learning_rate(3, f64::MAX)?;
    let prior = rates(&optimizer);
    let mut overflow = ExponentialLr::new(&optimizer, 2.)?;
    assert!(overflow.step(&mut optimizer).is_err());
    assert_eq!(rates(&optimizer), prior);
    assert_eq!(overflow.epoch(), 0);
    let mut plateau = ReduceLrOnPlateau::new(&optimizer)?;
    assert!(plateau.step(&mut optimizer, f64::NAN).is_err());
    assert_eq!(plateau.epoch(), 0);
    Ok(())
}

fn assert_close(actual: &[f64], expected: &Value, context: &str) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (&actual, expected) in actual.iter().zip(expected) {
        let expected = expected.as_f64().unwrap();
        assert!(
            (actual - expected).abs() < 2e-10,
            "{context}: {actual} vs {expected}"
        );
    }
}

#[test]
#[ignore = "requires locked Python; run through scripts/run-python-parity.sh"]
fn pinned_python_optimizer_and_scheduler_parity() -> Result<()> {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/python_reference/optim_state.py"
        ))
        .output()
        .expect("run locked Python optimizer reference");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference: Value = serde_json::from_slice(&output.stdout).unwrap();
    for name in ALGORITHMS {
        let (_, p, q, mut optimizer) = model(name)?;
        for step in 0..8 {
            train_step(&mut optimizer, &p, &q, step)?;
            assert_close(
                &weights(&p, &q)?,
                &reference["optimizers"][name]["trajectory"][step],
                &format!("{name} step {step}"),
            );
        }
        let state = serde_json::to_value(optimizer.state_dict()?).unwrap();
        for parameter in ["p", "q"] {
            for (slot, tensor) in state["parameters"][parameter]["slots"].as_object().unwrap() {
                let bytes = tensor["bytes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect::<Vec<_>>();
                let values = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|chunk| f64::from_le_bytes(*chunk))
                    .collect::<Vec<_>>();
                assert_close(
                    &values,
                    &reference["optimizers"][name]["states"][parameter][slot],
                    &format!("{name} {parameter} {slot}"),
                );
            }
        }
    }
    for name in SCHEDULES {
        let (mut optimizer, mut schedule) = scheduler_model(name)?;
        assert_close(&rates(&optimizer), &reference["schedulers"][name][0], name);
        for (index, metric) in METRICS.into_iter().enumerate() {
            optimizer.try_step()?;
            schedule.step(&mut optimizer, metric)?;
            assert_close(
                &rates(&optimizer),
                &reference["schedulers"][name][index + 1],
                name,
            );
        }
    }
    Ok(())
}
