use rusttorch::{
    Device, DeviceSpec, Kind, Reduction, RustTorchError, Tensor,
    nn::{
        self, Dropout, Flatten, Gelu, Identity, LinearConfig, Module, ReLU, Sequential, functional,
    },
    no_grad,
    optim::{Adam, Sgd},
};
use tch::nn::{Init, VarStore};

fn tensor(values: &[f32], shape: &[i64]) -> Tensor {
    Tensor::from_slice(values).reshape(shape)
}

fn values(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&tensor.reshape([-1])).expect("test tensor must convert to f32 values")
}

fn assert_close(actual: &Tensor, expected: &[f32], tolerance: f32) {
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }
}

fn assign(target: &Tensor, source: &[f32], shape: &[i64]) {
    assert_eq!(target.size(), shape);
    let mut target = target.shallow_clone();
    no_grad(|| target.f_copy_(&tensor(source, shape)))
        .expect("deterministic test values must copy");
}

#[test]
fn tensor_core_types_are_reexported() {
    let value = Tensor::from_slice(&[1.0_f32, 2.0]);
    let _: Reduction = Reduction::Mean;

    assert_eq!(value.kind(), Kind::Float);
    assert_eq!(value.device(), Device::Cpu);
}

#[test]
fn linear_bias_and_no_bias_have_expected_shapes_and_values() {
    let var_store = VarStore::new(Device::Cpu);
    let biased = LinearConfig::new(2, 2)
        .build(&(var_store.root() / "biased"))
        .expect("biased linear must build");
    assign(biased.weight(), &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assign(biased.bias().expect("bias must exist"), &[0.5, -0.5], &[2]);

    let input = tensor(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
    let output = biased.forward(&input).expect("biased forward must work");
    assert_eq!(output.size(), [2, 2]);
    assert_close(&output, &[5.5, 10.5, 0.5, -1.5], 1e-6);

    let no_bias = LinearConfig::new(2, 2)
        .bias(false)
        .build(&(var_store.root() / "no_bias"))
        .expect("bias-free linear must build");
    assign(no_bias.weight(), &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(no_bias.bias().is_none());
    let output = no_bias
        .forward(&input)
        .expect("bias-free forward must work");
    assert_eq!(output.size(), [2, 2]);
    assert_close(&output, &[5.0, 11.0, 0.0, -1.0], 1e-6);
}

#[test]
fn gradients_flow_through_linear() {
    let var_store = VarStore::new(Device::Cpu);
    let linear = LinearConfig::new(2, 1)
        .bias(false)
        .build(&(var_store.root() / "linear"))
        .expect("linear must build");
    assign(linear.weight(), &[2.0, -3.0], &[1, 2]);
    let input = tensor(&[1.0, 4.0], &[1, 2]).set_requires_grad(true);

    linear
        .forward(&input)
        .expect("linear forward must work")
        .sum(Kind::Float)
        .f_backward()
        .expect("backward must work");

    assert_close(&input.grad(), &[2.0, -3.0], 1e-6);
    assert_close(&linear.weight().grad(), &[1.0, 4.0], 1e-6);
}

#[test]
fn gradients_accumulate_across_eager_residual_branches() {
    let var_store = VarStore::new(Device::Cpu);
    let left = LinearConfig::new(2, 2)
        .bias(false)
        .build(&(var_store.root() / "left"))
        .expect("left branch must build");
    let right = LinearConfig::new(2, 2)
        .bias(false)
        .build(&(var_store.root() / "right"))
        .expect("right branch must build");
    assign(left.weight(), &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assign(right.weight(), &[2.0, 0.0, 0.0, 2.0], &[2, 2]);
    let input = tensor(&[1.0, 2.0], &[1, 2]).set_requires_grad(true);

    let branched = left
        .forward(&input)
        .expect("left branch must run")
        .f_add(&right.forward(&input).expect("right branch must run"))
        .expect("branches must add")
        .f_add(&input)
        .expect("residual must add");
    branched
        .sum(Kind::Float)
        .f_backward()
        .expect("backward must work");

    assert_close(&input.grad(), &[4.0, 4.0], 1e-6);
    assert_close(&left.weight().grad(), &[1.0, 2.0, 1.0, 2.0], 1e-6);
    assert_close(&right.weight().grad(), &[1.0, 2.0, 1.0, 2.0], 1e-6);
}

#[test]
fn identity_relu_gelu_and_flatten_match_expected_behavior() {
    let input = tensor(&[-1.0, 0.0, 1.0], &[3]);
    assert_close(
        &Identity.forward(&input).expect("identity must work"),
        &[-1.0, 0.0, 1.0],
        0.0,
    );
    assert_close(
        &ReLU.forward(&input).expect("ReLU must work"),
        &[0.0, 0.0, 1.0],
        0.0,
    );
    assert_close(
        &Gelu::default().forward(&input).expect("GELU must work"),
        &[-0.158_655_26, 0.0, 0.841_344_7],
        1e-5,
    );

    let input = Tensor::arange(24, (Kind::Float, Device::Cpu)).reshape([2, 3, 4]);
    let output = Flatten::default()
        .forward(&input)
        .expect("flatten must work");
    assert_eq!(output.size(), [2, 12]);
    assert_close(&output, &values(&input), 0.0);
}

#[test]
fn dropout_boundary_probabilities_and_eval_gradients_are_correct() {
    let input = Tensor::ones([4], (Kind::Float, Device::Cpu));
    assert_close(
        &Dropout::new(0.0)
            .expect("p=0 must be valid")
            .forward_t(&input, true)
            .expect("p=0 training must work"),
        &[1.0; 4],
        0.0,
    );
    let dropout = Dropout::new(1.0).expect("p=1 must be valid");
    assert_close(
        &dropout
            .forward_t(&input, true)
            .expect("p=1 training must work"),
        &[0.0; 4],
        0.0,
    );

    let input = input.set_requires_grad(true);
    let evaluation = dropout
        .forward_t(&input, false)
        .expect("dropout evaluation must work");
    assert_close(&evaluation, &[1.0; 4], 0.0);
    assert!(evaluation.requires_grad());
    evaluation
        .sum(Kind::Float)
        .f_backward()
        .expect("evaluation backward must work");
    assert_close(&input.grad(), &[1.0; 4], 0.0);
}

#[test]
fn sequential_uses_numeric_parameter_names_and_tracks_mode() {
    let mut model = Sequential::builder()
        .linear(2, 2)
        .relu()
        .dropout(1.0)
        .linear(2, 1)
        .build(DeviceSpec::Cpu)
        .expect("sequential model must build");
    let variables = model.var_store().variables();
    let mut names = variables.keys().cloned().collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, ["0.bias", "0.weight", "3.bias", "3.weight"]);

    assign(
        variables.get("0.weight").expect("first weight must exist"),
        &[1.0, 0.0, 0.0, 1.0],
        &[2, 2],
    );
    assign(
        variables.get("0.bias").expect("first bias must exist"),
        &[0.0, 0.0],
        &[2],
    );
    assign(
        variables.get("3.weight").expect("second weight must exist"),
        &[1.0, 1.0],
        &[1, 2],
    );
    assign(
        variables.get("3.bias").expect("second bias must exist"),
        &[0.0],
        &[1],
    );

    let input = tensor(&[1.0, 2.0], &[1, 2]);
    assert!(model.is_training());
    assert_close(
        &model.forward(&input).expect("training forward must work"),
        &[0.0],
        0.0,
    );
    model.eval();
    assert!(!model.is_training());
    assert_close(
        &model.forward(&input).expect("evaluation forward must work"),
        &[3.0],
        1e-6,
    );
    model.train();
    assert!(model.is_training());
}

#[test]
fn mse_and_cross_entropy_have_deterministic_values() {
    let mse = functional::mse_loss(&tensor(&[1.0, 2.0], &[2]), &tensor(&[0.0, 4.0], &[2]))
        .expect("MSE must work");
    assert!((mse.double_value(&[]) - 2.5).abs() < 1e-7);

    let cross_entropy =
        functional::cross_entropy(&tensor(&[2.0, 0.0], &[1, 2]), &Tensor::from_slice(&[0_i64]))
            .expect("cross entropy must work");
    assert!((cross_entropy.double_value(&[]) - 0.126_928_05).abs() < 1e-6);
}

#[test]
fn sgd_and_adam_each_update_a_parameter_once() {
    let sgd_store = VarStore::new(Device::Cpu);
    let sgd_weight = sgd_store.root().var("weight", &[1], Init::Const(1.0));
    let mut sgd = Sgd::builder()
        .learning_rate(0.1)
        .build(&sgd_store)
        .expect("SGD must build");
    let loss = functional::mse_loss(&sgd_weight, &Tensor::zeros([1], (Kind::Float, Device::Cpu)))
        .expect("SGD loss must build");
    sgd.zero_grad();
    loss.f_backward().expect("SGD backward must work");
    sgd.step();
    assert_close(&sgd_weight, &[0.8], 1e-6);

    let adam_store = VarStore::new(Device::Cpu);
    let adam_weight = adam_store.root().var("weight", &[1], Init::Const(1.0));
    let mut adam = Adam::builder()
        .learning_rate(0.1)
        .build(&adam_store)
        .expect("Adam must build");
    let loss = functional::mse_loss(
        &adam_weight,
        &Tensor::zeros([1], (Kind::Float, Device::Cpu)),
    )
    .expect("Adam loss must build");
    adam.backward_step(&loss).expect("Adam step must work");
    assert_close(&adam_weight, &[0.9], 1e-5);
}

#[test]
fn invalid_layer_dropout_and_optimizer_configs_are_rejected() {
    let var_store = VarStore::new(Device::Cpu);
    assert!(matches!(
        LinearConfig::new(-1, 2).build(&var_store.root()),
        Err(RustTorchError::InvalidConfiguration {
            field: "in_features",
            ..
        })
    ));
    for probability in [-0.1, 1.1, f64::NAN] {
        assert!(matches!(
            Dropout::new(probability),
            Err(RustTorchError::InvalidConfiguration {
                field: "dropout probability",
                ..
            })
        ));
    }
    assert!(matches!(
        Adam::builder().learning_rate(f64::NAN).build(&var_store),
        Err(RustTorchError::InvalidConfiguration {
            field: "learning_rate",
            ..
        })
    ));
    assert!(matches!(
        Adam::builder().fused(true).build(&var_store),
        Err(RustTorchError::UnsupportedOption {
            component: "Adam",
            option: "fused"
        })
    ));
    assert!(matches!(
        Sgd::builder().nesterov(true).build(&var_store),
        Err(RustTorchError::InvalidConfiguration {
            field: "nesterov",
            ..
        })
    ));
}

#[test]
fn no_grad_and_detach_stop_gradient_recording() {
    let input = tensor(&[1.0, 2.0], &[2]).set_requires_grad(true);
    let recorded = input.f_mul_scalar(3.0).expect("recorded branch must work");
    let unrecorded = no_grad(|| input.f_mul_scalar(2.0)).expect("no-grad branch must work");
    let detached = input.f_detach().expect("detach must work");

    assert!(recorded.requires_grad());
    assert!(!unrecorded.requires_grad());
    assert!(!detached.requires_grad());

    recorded
        .f_add(&detached)
        .expect("branches must add")
        .f_add(&unrecorded)
        .expect("no-grad branch must add")
        .sum(Kind::Float)
        .f_backward()
        .expect("backward must work");
    assert_close(&input.grad(), &[3.0, 3.0], 1e-6);
}

#[test]
fn short_linear_constructor_uses_bias() {
    let var_store = VarStore::new(Device::Cpu);
    let linear = nn::linear(&var_store.root(), 3, 2).expect("short constructor must build");
    assert_eq!(linear.weight().size(), [2, 3]);
    assert_eq!(linear.bias().expect("default bias must exist").size(), [2]);
}
