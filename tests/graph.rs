use rusttorch::{
    Device, DeviceSpec, Kind, RustTorchError, Tensor, available_devices,
    graph::{
        DeadNodeElimination, Dim, GraphBuilder, GraphInputs, GraphModule, GraphPass,
        ShapePropagation, TensorSpec, Validation,
    },
    no_grad,
};

fn float_tensor(values: &[f32], shape: &[i64], device: Device) -> Tensor {
    Tensor::from_slice(values).reshape(shape).to_device(device)
}

fn values(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&tensor.to_device(Device::Cpu).reshape([-1]))
        .expect("test tensor must convert to f32 values")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual} (tolerance {tolerance})",
        );
    }
}

fn set_variable(model: &GraphModule, name: &str, shape: &[i64], data: &[f32]) {
    let variables = model.var_store().variables();
    let mut target = variables
        .get(name)
        .unwrap_or_else(|| panic!("missing graph variable `{name}`"))
        .shallow_clone();
    let source = float_tensor(data, shape, model.device());
    no_grad(|| target.f_copy_(&source)).expect("deterministic graph weights must copy");
}

fn identity_model() -> GraphModule {
    let mut builder = GraphBuilder::new();
    let input = builder
        .input(
            "features",
            TensorSpec::new().known_dimensions([1, 2]).kind(Kind::Float),
        )
        .expect("input must build");
    let identity = builder
        .identity("identity", input)
        .expect("identity must build");
    builder
        .output("result", identity)
        .expect("output must build")
        .build(DeviceSpec::Cpu)
        .expect("identity model must build")
}

#[test]
fn all_public_graph_operations_execute_with_expected_values() {
    let mut builder = GraphBuilder::new();
    let features = builder
        .input(
            "features",
            TensorSpec::new().known_dimensions([1, 2]).kind(Kind::Float),
        )
        .expect("features input must build");
    let other = builder
        .input(
            "other",
            TensorSpec::new().known_dimensions([1, 2]).kind(Kind::Float),
        )
        .expect("other input must build");
    let mse_target = builder
        .input(
            "mse_target",
            TensorSpec::new().known_dimensions([1, 6]).kind(Kind::Float),
        )
        .expect("MSE target must build");
    let labels = builder
        .input(
            "labels",
            TensorSpec::new().known_dimensions([1]).kind(Kind::Int64),
        )
        .expect("class target must build");

    let identity = builder
        .identity("identity", features)
        .expect("identity must build");
    let linear = builder
        .linear("projection", identity, 2, 2)
        .expect("linear must build");
    let relu = builder.relu("relu", linear).expect("ReLU must build");
    let gelu = builder.gelu("gelu", relu).expect("GELU must build");
    let dropout = builder
        .dropout("dropout", gelu, 0.0)
        .expect("dropout must build");
    let flatten = builder
        .flatten("flatten", dropout, 0, -1)
        .expect("flatten must build");
    let add = builder.add("add", features, other).expect("add must build");
    let subtract = builder
        .subtract("subtract", features, other)
        .expect("subtract must build");
    let multiply = builder
        .multiply("multiply", features, other)
        .expect("multiply must build");
    let concatenate = builder
        .concatenate("concatenate", vec![add, subtract, multiply], 1)
        .expect("concatenate must build");
    let mse = builder
        .mse_loss("mse", concatenate, mse_target)
        .expect("MSE must build");
    let cross_entropy = builder
        .cross_entropy_loss("cross_entropy", linear, labels)
        .expect("cross-entropy must build");

    for (name, value) in [
        ("identity_result", identity),
        ("linear_result", linear),
        ("relu_result", relu),
        ("gelu_result", gelu),
        ("dropout_result", dropout),
        ("flatten_result", flatten),
        ("add_result", add),
        ("subtract_result", subtract),
        ("multiply_result", multiply),
        ("concatenate_result", concatenate),
        ("mse_result", mse),
        ("cross_entropy_result", cross_entropy),
    ] {
        builder
            .add_output(name, value)
            .unwrap_or_else(|error| panic!("output `{name}` must build: {error}"));
    }

    let model = builder
        .build(DeviceSpec::Cpu)
        .expect("all-operations graph must build");
    set_variable(&model, "projection.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    set_variable(&model, "projection.bias", &[2], &[0.0, 0.0]);

    let inputs = GraphInputs::new()
        .with("features", float_tensor(&[-1.0, 2.0], &[1, 2], Device::Cpu))
        .expect("features must insert")
        .with("other", float_tensor(&[3.0, 4.0], &[1, 2], Device::Cpu))
        .expect("other must insert")
        .with("mse_target", float_tensor(&[0.0; 6], &[1, 6], Device::Cpu))
        .expect("MSE target must insert")
        .with("labels", Tensor::from_slice(&[1i64]))
        .expect("labels must insert");
    let outputs = model.forward(inputs).expect("graph forward must succeed");

    assert_eq!(values(outputs.get("identity_result").unwrap()), [-1.0, 2.0]);
    assert_eq!(values(outputs.get("linear_result").unwrap()), [-1.0, 2.0]);
    assert_eq!(values(outputs.get("relu_result").unwrap()), [0.0, 2.0]);
    assert_close(
        &values(outputs.get("gelu_result").unwrap()),
        &[0.0, 1.954_499_7],
        1e-6,
    );
    assert_close(
        &values(outputs.get("dropout_result").unwrap()),
        &[0.0, 1.954_499_7],
        1e-6,
    );
    assert_eq!(outputs.get("flatten_result").unwrap().size(), [2]);
    assert_close(
        &values(outputs.get("flatten_result").unwrap()),
        &[0.0, 1.954_499_7],
        1e-6,
    );
    assert_eq!(values(outputs.get("add_result").unwrap()), [2.0, 6.0]);
    assert_eq!(
        values(outputs.get("subtract_result").unwrap()),
        [-4.0, -2.0],
    );
    assert_eq!(values(outputs.get("multiply_result").unwrap()), [-3.0, 8.0],);
    assert_eq!(
        values(outputs.get("concatenate_result").unwrap()),
        [2.0, 6.0, -4.0, -2.0, -3.0, 8.0],
    );
    assert_close(
        &values(outputs.get("mse_result").unwrap()),
        &[133.0 / 6.0],
        1e-5,
    );
    assert_close(
        &values(outputs.get("cross_entropy_result").unwrap()),
        &[0.048_587_35],
        1e-6,
    );
}

#[test]
fn residual_branch_accumulates_input_and_all_linear_parameter_gradients() {
    let mut builder = GraphBuilder::new();
    let features = builder
        .input(
            "features",
            TensorSpec::new().known_dimensions([1, 2]).kind(Kind::Float),
        )
        .expect("input must build");
    let main = builder
        .linear("main", features, 2, 2)
        .expect("main branch must build");
    let main = builder
        .relu("main_relu", main)
        .expect("main activation must build");
    let skip = builder
        .linear("skip", features, 2, 2)
        .expect("skip branch must build");
    let residual = builder
        .add("residual", main, skip)
        .expect("residual add must build");
    let model = builder
        .output("result", residual)
        .expect("output must build")
        .build(DeviceSpec::Cpu)
        .expect("residual graph must build");

    set_variable(&model, "main.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    set_variable(&model, "main.bias", &[2], &[0.0, 0.0]);
    set_variable(&model, "skip.weight", &[2, 2], &[2.0, 0.0, 0.0, 2.0]);
    set_variable(&model, "skip.bias", &[2], &[0.0, 0.0]);

    let input = float_tensor(&[1.0, 2.0], &[1, 2], Device::Cpu).set_requires_grad(true);
    let outputs = model
        .forward(
            GraphInputs::new()
                .with("features", input.shallow_clone())
                .expect("input must insert"),
        )
        .expect("residual forward must succeed");
    let output = outputs.get("result").expect("result must exist");
    assert_eq!(values(output), [3.0, 6.0]);
    output.sum(Kind::Float).backward();

    assert_eq!(values(&input.grad()), [3.0, 3.0]);
    let variables = model.var_store().variables();
    assert_eq!(variables.len(), 4);
    for (name, expected) in [
        ("main.bias", vec![1.0, 1.0]),
        ("main.weight", vec![1.0, 2.0, 1.0, 2.0]),
        ("skip.bias", vec![1.0, 1.0]),
        ("skip.weight", vec![1.0, 2.0, 1.0, 2.0]),
    ] {
        let gradient = variables
            .get(name)
            .unwrap_or_else(|| panic!("missing graph variable `{name}`"))
            .grad();
        assert!(gradient.defined(), "`{name}` did not receive a gradient");
        assert_eq!(values(&gradient), expected, "wrong gradient for `{name}`");
    }
}

#[test]
fn graph_inputs_reject_duplicate_names() {
    let duplicate = GraphInputs::new()
        .with("features", float_tensor(&[1.0, 2.0], &[1, 2], Device::Cpu))
        .expect("first input must insert")
        .with("features", float_tensor(&[3.0, 4.0], &[1, 2], Device::Cpu));

    assert!(matches!(
        duplicate,
        Err(RustTorchError::DuplicateName(name)) if name == "features"
    ));
}

#[test]
fn forward_rejects_missing_and_unexpected_inputs() {
    let model = identity_model();
    assert!(matches!(
        model.forward(GraphInputs::new()),
        Err(RustTorchError::MissingGraphInput(name)) if name == "features"
    ));

    let inputs = GraphInputs::new()
        .with("features", float_tensor(&[1.0, 2.0], &[1, 2], Device::Cpu))
        .expect("required input must insert")
        .with("extra", float_tensor(&[0.0], &[1], Device::Cpu))
        .expect("extra input must insert");
    assert!(matches!(
        model.forward(inputs),
        Err(RustTorchError::UnexpectedGraphInput(name)) if name == "extra"
    ));
}

#[test]
fn builder_rejects_a_value_from_another_graph() {
    let mut owner = GraphBuilder::new();
    let foreign = owner
        .input("owner_input", TensorSpec::new().rank(1))
        .expect("owner input must build");
    let mut receiver = GraphBuilder::new();

    assert!(matches!(
        receiver.identity("invalid_identity", foreign),
        Err(RustTorchError::GraphValidation(_))
    ));
}

#[test]
fn builder_rejects_duplicate_node_names() {
    let mut builder = GraphBuilder::new();
    let input = builder
        .input("duplicate", TensorSpec::new().rank(1))
        .expect("input must build");

    assert!(matches!(
        builder.identity("duplicate", input),
        Err(RustTorchError::DuplicateName(name)) if name == "duplicate"
    ));
}

#[test]
fn graph_without_an_output_is_rejected() {
    let mut builder = GraphBuilder::new();
    builder
        .input("features", TensorSpec::new().rank(1))
        .expect("input must build");

    assert!(matches!(
        builder.finish(),
        Err(RustTorchError::GraphValidation(_))
    ));
}

#[test]
fn runtime_rejects_dtype_rank_and_known_dimension_mismatches() {
    let model = identity_model();
    let int_input = GraphInputs::new()
        .with("features", Tensor::from_slice(&[1i64, 2]).reshape([1, 2]))
        .expect("integer input must insert");
    assert!(matches!(
        model.forward(int_input),
        Err(RustTorchError::DtypeMismatch {
            expected: Kind::Float,
            actual: Kind::Int64,
            ..
        })
    ));

    let rank_one = GraphInputs::new()
        .with("features", float_tensor(&[1.0, 2.0], &[2], Device::Cpu))
        .expect("rank-one input must insert");
    assert!(matches!(
        model.forward(rank_one),
        Err(RustTorchError::InvalidDimensions { .. })
    ));

    let wrong_dimension = GraphInputs::new()
        .with(
            "features",
            float_tensor(&[1.0, 2.0, 3.0], &[1, 3], Device::Cpu),
        )
        .expect("wrong-size input must insert");
    assert!(matches!(
        model.forward(wrong_dimension),
        Err(RustTorchError::InvalidDimensions { .. })
    ));
}

#[test]
fn runtime_rejects_an_input_on_another_available_device() {
    let capabilities = available_devices();
    let other_device = if capabilities.cuda {
        Device::Cuda(0)
    } else if capabilities.mps {
        Device::Mps
    } else {
        eprintln!(
            "skipped graph device-mismatch execution: linked LibTorch reports no usable CUDA or MPS backend"
        );
        return;
    };
    let model = identity_model();
    let inputs = GraphInputs::new()
        .with("features", float_tensor(&[1.0, 2.0], &[1, 2], other_device))
        .expect("accelerator input must insert");

    assert!(matches!(
        model.forward(inputs),
        Err(RustTorchError::DeviceMismatch {
            expected: Device::Cpu,
            actual,
            ..
        }) if actual == other_device
    ));
}

#[test]
fn dropout_follows_train_and_eval_mode_without_rng_assumptions() {
    let mut builder = GraphBuilder::new();
    let features = builder
        .input(
            "features",
            TensorSpec::new().known_dimensions([2, 2]).kind(Kind::Float),
        )
        .expect("input must build");
    let dropout = builder
        .dropout("dropout", features, 1.0)
        .expect("dropout must build");
    let mut model = builder
        .output("result", dropout)
        .expect("output must build")
        .build(DeviceSpec::Cpu)
        .expect("dropout graph must build");

    assert!(model.is_training());
    let training = model
        .forward(
            GraphInputs::new()
                .with("features", float_tensor(&[1.0; 4], &[2, 2], Device::Cpu))
                .expect("training input must insert"),
        )
        .expect("training forward must succeed");
    assert_eq!(values(training.get("result").unwrap()), [0.0; 4]);

    model.eval();
    assert!(!model.is_training());
    let evaluation = model
        .forward(
            GraphInputs::new()
                .with("features", float_tensor(&[1.0; 4], &[2, 2], Device::Cpu))
                .expect("evaluation input must insert"),
        )
        .expect("evaluation forward must succeed");
    assert_eq!(values(evaluation.get("result").unwrap()), [1.0; 4]);

    model.train();
    assert!(model.is_training());
}

#[test]
fn summary_and_dot_are_deterministic_and_describe_the_graph() {
    let mut builder = GraphBuilder::new();
    let features = builder
        .input(
            "features",
            TensorSpec::new().known_dimensions([1, 2]).kind(Kind::Float),
        )
        .expect("input must build");
    let projection = builder
        .linear("projection", features, 2, 2)
        .expect("linear must build");
    let activation = builder
        .relu("activation", projection)
        .expect("ReLU must build");
    let graph = builder
        .output("result", activation)
        .expect("output must build")
        .finish()
        .expect("graph must finish");

    let summary = graph.summary();
    assert_eq!(summary, graph.summary());
    assert!(summary.starts_with("id  name  op  inputs  shape  kind  parameters\n"));
    assert!(summary.contains("features  Input"));
    assert!(summary.contains("projection  Linear"));
    assert!(summary.lines().any(|line| line.ends_with("  6")));
    assert!(summary.contains("activation  ReLU"));
    assert!(summary.contains("result  Output"));

    let expected_dot = concat!(
        "digraph rusttorch {\n",
        "  n0 [label=\"features\\nInput\"];\n",
        "  n1 [label=\"projection\\nLinear\"];\n",
        "  n2 [label=\"activation\\nReLU\"];\n",
        "  n3 [label=\"result\\nOutput\"];\n",
        "  n0 -> n1;\n",
        "  n1 -> n2;\n",
        "  n2 -> n3;\n",
        "}\n",
    );
    assert_eq!(graph.to_dot(), expected_dot);
    assert_eq!(graph.to_dot(), expected_dot);
}

#[test]
fn finish_eliminates_dead_nodes_and_propagates_flatten_linear_shapes() {
    let mut builder = GraphBuilder::new();
    let features = builder
        .input(
            "features",
            TensorSpec::new()
                .known_dimensions([2, 3, 4])
                .kind(Kind::Float),
        )
        .expect("input must build");
    let _dead = builder
        .relu("dead_branch", features)
        .expect("dead branch must build");
    let flattened = builder
        .flatten("flattened", features, 1, -1)
        .expect("flatten must build");
    let projected = builder
        .linear_config("projection", flattened, 12, 5, false)
        .expect("linear must build");
    let mut graph = builder
        .output("result", projected)
        .expect("output must build")
        .finish()
        .expect("graph must finish");

    let dead = graph
        .nodes()
        .iter()
        .find(|node| node.name == "dead_branch")
        .expect("dead node must remain inspectable");
    assert!(!dead.active);
    let flattened = graph
        .nodes()
        .iter()
        .find(|node| node.name == "flattened")
        .expect("flatten node must exist");
    assert_eq!(
        flattened.spec.dims(),
        Some(&[Dim::Known(2), Dim::Known(12)][..])
    );
    let projection = graph
        .nodes()
        .iter()
        .find(|node| node.name == "projection")
        .expect("linear node must exist");
    assert_eq!(
        projection.spec.dims(),
        Some(&[Dim::Known(2), Dim::Known(5)][..])
    );
    assert!(
        graph
            .topological_order()
            .expect("topological order must exist")
            .iter()
            .all(|id| *id != dead.id)
    );

    Validation
        .run(&mut graph)
        .expect("public validation pass must succeed");
    assert!(
        DeadNodeElimination
            .run(&mut graph)
            .expect("dead-node pass must be idempotent")
            .changed_nodes
            .is_empty()
    );
    assert!(
        ShapePropagation
            .run(&mut graph)
            .expect("shape pass must be idempotent")
            .changed_nodes
            .is_empty()
    );
}

#[test]
fn binary_shape_propagation_preserves_dimension_order() {
    let mut builder = GraphBuilder::new();
    let left = builder
        .input(
            "left",
            TensorSpec::new()
                .known_dimensions([2, 3, 4])
                .kind(Kind::Float),
        )
        .expect("left input must build");
    let right = builder
        .input(
            "right",
            TensorSpec::new()
                .known_dimensions([1, 3, 1])
                .kind(Kind::Float),
        )
        .expect("right input must build");
    let product = builder
        .multiply("product", left, right)
        .expect("broadcast multiply must build");
    let graph = builder
        .output("result", product)
        .expect("output must build")
        .finish()
        .expect("graph must finish");
    let product = graph
        .nodes()
        .iter()
        .find(|node| node.name == "product")
        .expect("product node must exist");

    assert_eq!(
        product.spec.dims(),
        Some(&[Dim::Known(2), Dim::Known(3), Dim::Known(4)][..]),
    );
}
