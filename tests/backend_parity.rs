use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

use rusttorch::{
    Device, DeviceSpec, Kind, Tensor, available_devices,
    graph::{GraphBuilder, GraphInputs, GraphModule, TensorSpec},
    nn::{Sequential, functional},
    no_grad,
    optim::{Adam, Sgd},
};

const CPU_CUDA_RTOL: f32 = 1e-5;
const CPU_CUDA_ATOL: f32 = 1e-5;
const MPS_RTOL: f32 = 1e-4;
const MPS_ATOL: f32 = 1e-4;

#[derive(Debug)]
struct GraphRun {
    output: Vec<f32>,
    loss: f32,
    input_gradient: Vec<f32>,
    parameter_gradients: BTreeMap<String, Vec<f32>>,
}

#[derive(Debug)]
struct EagerRun {
    output: Vec<f32>,
    cross_entropy: f32,
    mse: f32,
    input_gradient: Vec<f32>,
    parameter_gradients: BTreeMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Copy)]
enum OptimizerKind {
    Sgd,
    Adam,
}

struct TempSafetensors(PathBuf);

fn backend_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl TempSafetensors {
    fn new(label: &str) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "rusttorch-backend-{label}-{}-{id}.safetensors",
            process::id(),
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempSafetensors {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn float_tensor(values: &[f32], shape: &[i64], device: Device) -> Tensor {
    Tensor::from_slice(values).reshape(shape).to_device(device)
}

fn values(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&tensor.to_device(Device::Cpu).reshape([-1]))
        .expect("test tensor must convert to f32 values")
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

fn parameter_values(model: &GraphModule) -> BTreeMap<String, Vec<f32>> {
    model
        .var_store()
        .variables()
        .into_iter()
        .map(|(name, variable)| (name, values(&variable)))
        .collect()
}

fn zero_variables(model: &GraphModule) {
    for (_, variable) in model.var_store().variables() {
        let mut variable = variable.shallow_clone();
        let _ = no_grad(|| variable.f_zero_()).expect("test variable must zero");
    }
}

fn residual_model(device: DeviceSpec) -> GraphModule {
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
        .build(device)
        .expect("residual graph must build on reported device");

    set_variable(&model, "main.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    set_variable(&model, "main.bias", &[2], &[0.0, 0.0]);
    set_variable(&model, "skip.weight", &[2, 2], &[2.0, 0.0, 0.0, 2.0]);
    set_variable(&model, "skip.bias", &[2], &[0.0, 0.0]);
    model
}

fn eager_model(device: DeviceSpec) -> Sequential {
    let model = Sequential::builder()
        .linear(2, 3)
        .relu()
        .gelu()
        .build(device)
        .expect("eager model must build on reported device");
    for (name, shape, data) in [
        (
            "0.weight",
            &[3, 2][..],
            &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0][..],
        ),
        ("0.bias", &[3][..], &[0.0, 0.0, 0.0][..]),
    ] {
        let variables = model.var_store().variables();
        let mut target = variables
            .get(name)
            .unwrap_or_else(|| panic!("missing eager variable `{name}`"))
            .shallow_clone();
        let source = float_tensor(data, shape, model.device());
        no_grad(|| target.f_copy_(&source)).expect("deterministic eager weights must copy");
    }
    model
}

fn run_eager_losses(device_spec: DeviceSpec) -> EagerRun {
    let model = eager_model(device_spec);
    let device = model.device();
    let input = float_tensor(&[1.0, 2.0], &[1, 2], device).set_requires_grad(true);
    let output = model
        .forward(&input)
        .expect("eager forward must succeed on reported device");
    assert_eq!(output.device(), device);
    let class_target = Tensor::from_slice(&[1_i64]).to_device(device);
    let mse_target = float_tensor(&[0.5, 1.5, 0.0], &[1, 3], device);
    let cross_entropy = functional::cross_entropy(&output, &class_target)
        .expect("cross-entropy must execute on reported device");
    let mse =
        functional::mse_loss(&output, &mse_target).expect("MSE must execute on reported device");
    cross_entropy
        .f_add(&mse)
        .expect("losses must add")
        .backward();

    let input_gradient = input.grad();
    assert!(input_gradient.defined());
    assert_eq!(input_gradient.device(), device);
    let mut parameter_gradients = BTreeMap::new();
    for (name, variable) in model.var_store().variables() {
        let gradient = variable.grad();
        assert!(gradient.defined(), "`{name}` did not receive a gradient");
        assert_eq!(gradient.device(), device);
        parameter_gradients.insert(name, values(&gradient));
    }

    EagerRun {
        output: values(&output),
        cross_entropy: f32::try_from(cross_entropy.to_device(Device::Cpu))
            .expect("cross-entropy must be scalar f32"),
        mse: f32::try_from(mse.to_device(Device::Cpu)).expect("MSE must be scalar f32"),
        input_gradient: values(&input_gradient),
        parameter_gradients,
    }
}

fn run_residual_graph(device_spec: DeviceSpec) -> GraphRun {
    let model = residual_model(device_spec);
    let device = model.device();
    let input = float_tensor(&[1.0, 2.0], &[1, 2], device).set_requires_grad(true);
    let outputs = model
        .forward(
            GraphInputs::new()
                .with("features", input.shallow_clone())
                .expect("input must insert"),
        )
        .expect("graph forward must succeed");
    let output = outputs.get("result").expect("result must exist");
    assert_eq!(output.device(), device);
    let loss = output.sum(Kind::Float);
    loss.backward();

    let input_gradient = input.grad();
    assert!(input_gradient.defined());
    assert_eq!(input_gradient.device(), device);
    let mut parameter_gradients = BTreeMap::new();
    for (name, variable) in model.var_store().variables() {
        let gradient = variable.grad();
        assert!(gradient.defined(), "`{name}` did not receive a gradient");
        assert_eq!(
            gradient.device(),
            device,
            "wrong gradient device for `{name}`"
        );
        parameter_gradients.insert(name, values(&gradient));
    }

    GraphRun {
        output: values(output),
        loss: f32::try_from(loss.to_device(Device::Cpu)).expect("loss must be scalar f32"),
        input_gradient: values(&input_gradient),
        parameter_gradients,
    }
}

fn run_optimizer_step(
    device_spec: DeviceSpec,
    optimizer_kind: OptimizerKind,
) -> BTreeMap<String, Vec<f32>> {
    let model = residual_model(device_spec);
    let input = float_tensor(&[1.0, 2.0], &[1, 2], model.device());
    let outputs = model
        .forward(
            GraphInputs::new()
                .with("features", input)
                .expect("input must insert"),
        )
        .expect("optimizer graph forward must succeed");
    let loss = outputs.get("result").unwrap().sum(Kind::Float);
    let mut optimizer = match optimizer_kind {
        OptimizerKind::Sgd => Sgd::builder()
            .learning_rate(0.1)
            .build(model.var_store())
            .expect("SGD must build"),
        OptimizerKind::Adam => Adam::builder()
            .learning_rate(1e-3)
            .build(model.var_store())
            .expect("Adam must build"),
    };
    optimizer
        .backward_step(&loss)
        .expect("optimizer step must succeed");
    parameter_values(&model)
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], rtol: f32, atol: f32) {
    assert_eq!(actual.len(), expected.len(), "shape mismatch for {label}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}]: expected {expected}, got {actual}, tolerance {tolerance}",
        );
    }
}

fn assert_run_matches(backend: &str, actual: &GraphRun, expected: &GraphRun, rtol: f32, atol: f32) {
    assert_close(
        &format!("{backend} output"),
        &actual.output,
        &expected.output,
        rtol,
        atol,
    );
    assert_close(
        &format!("{backend} loss"),
        &[actual.loss],
        &[expected.loss],
        rtol,
        atol,
    );
    assert_close(
        &format!("{backend} input gradient"),
        &actual.input_gradient,
        &expected.input_gradient,
        rtol,
        atol,
    );
    assert_eq!(
        actual.parameter_gradients.keys().collect::<Vec<_>>(),
        expected.parameter_gradients.keys().collect::<Vec<_>>(),
    );
    for (name, expected_gradient) in &expected.parameter_gradients {
        assert_close(
            &format!("{backend} `{name}` gradient"),
            &actual.parameter_gradients[name],
            expected_gradient,
            rtol,
            atol,
        );
    }
}

fn assert_eager_run_matches(
    backend: &str,
    actual: &EagerRun,
    expected: &EagerRun,
    rtol: f32,
    atol: f32,
) {
    for (label, actual, expected) in [
        (
            "output",
            actual.output.as_slice(),
            expected.output.as_slice(),
        ),
        (
            "input gradient",
            actual.input_gradient.as_slice(),
            expected.input_gradient.as_slice(),
        ),
    ] {
        assert_close(
            &format!("{backend} eager {label}"),
            actual,
            expected,
            rtol,
            atol,
        );
    }
    assert_close(
        &format!("{backend} cross-entropy"),
        &[actual.cross_entropy],
        &[expected.cross_entropy],
        rtol,
        atol,
    );
    assert_close(
        &format!("{backend} MSE"),
        &[actual.mse],
        &[expected.mse],
        rtol,
        atol,
    );
    assert_parameter_maps_close(
        &format!("{backend} eager gradient"),
        &actual.parameter_gradients,
        &expected.parameter_gradients,
        rtol,
        atol,
    );
}

fn assert_parameter_maps_close(
    backend: &str,
    actual: &BTreeMap<String, Vec<f32>>,
    expected: &BTreeMap<String, Vec<f32>>,
    rtol: f32,
    atol: f32,
) {
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (name, expected_values) in expected {
        assert_close(
            &format!("{backend} `{name}`"),
            &actual[name],
            expected_values,
            rtol,
            atol,
        );
    }
}

#[test]
fn deterministic_residual_graph_matches_cpu_on_available_backends() {
    let _backend_guard = backend_test_lock();
    let cpu = run_residual_graph(DeviceSpec::Cpu);
    assert_close("CPU output", &cpu.output, &[3.0, 6.0], 0.0, 1e-6);
    assert_close("CPU loss", &[cpu.loss], &[9.0], 0.0, 1e-6);
    assert_close(
        "CPU input gradient",
        &cpu.input_gradient,
        &[3.0, 3.0],
        0.0,
        1e-6,
    );
    for (name, expected) in [
        ("main.bias", &[1.0, 1.0][..]),
        ("main.weight", &[1.0, 2.0, 1.0, 2.0][..]),
        ("skip.bias", &[1.0, 1.0][..]),
        ("skip.weight", &[1.0, 2.0, 1.0, 2.0][..]),
    ] {
        assert_close(
            &format!("CPU `{name}` gradient"),
            &cpu.parameter_gradients[name],
            expected,
            0.0,
            1e-6,
        );
    }

    let capabilities = available_devices();
    if capabilities.cuda {
        let cuda = run_residual_graph(DeviceSpec::Cuda(0));
        assert_run_matches("CUDA", &cuda, &cpu, CPU_CUDA_RTOL, CPU_CUDA_ATOL);
    } else {
        eprintln!(
            "skipped CUDA graph parity: linked LibTorch reports no usable CUDA device (count={})",
            capabilities.cuda_device_count,
        );
    }

    if capabilities.mps {
        let mps = run_residual_graph(DeviceSpec::Mps);
        assert_run_matches("MPS", &mps, &cpu, MPS_RTOL, MPS_ATOL);
    } else {
        eprintln!(
            "skipped MPS graph parity: linked LibTorch or the current machine reports no usable MPS device"
        );
    }
}

#[test]
fn eager_linear_activations_and_losses_match_cpu_on_available_backends() {
    let _backend_guard = backend_test_lock();
    let cpu = run_eager_losses(DeviceSpec::Cpu);
    let capabilities = available_devices();

    if capabilities.cuda {
        let cuda = run_eager_losses(DeviceSpec::Cuda(0));
        assert_eager_run_matches("CUDA", &cuda, &cpu, CPU_CUDA_RTOL, CPU_CUDA_ATOL);
        assert!(
            eager_model(DeviceSpec::Cuda(0))
                .forward(&float_tensor(&[1.0, 2.0], &[1, 2], Device::Cpu))
                .is_err(),
            "CUDA model must reject a CPU input"
        );
    } else {
        eprintln!(
            "skipped CUDA eager parity: linked LibTorch reports no usable CUDA device (count={})",
            capabilities.cuda_device_count,
        );
    }

    if capabilities.mps {
        let mps = run_eager_losses(DeviceSpec::Mps);
        assert_eager_run_matches("MPS", &mps, &cpu, MPS_RTOL, MPS_ATOL);
        assert!(
            eager_model(DeviceSpec::Mps)
                .forward(&float_tensor(&[1.0, 2.0], &[1, 2], Device::Cpu))
                .is_err(),
            "MPS model must reject a CPU input"
        );
    } else {
        eprintln!(
            "skipped MPS eager parity: linked LibTorch or the current machine reports no usable MPS device"
        );
    }
}

#[test]
fn one_sgd_and_adam_step_match_cpu_on_available_backends() {
    let _backend_guard = backend_test_lock();
    let cpu_sgd = run_optimizer_step(DeviceSpec::Cpu, OptimizerKind::Sgd);
    let cpu_adam = run_optimizer_step(DeviceSpec::Cpu, OptimizerKind::Adam);
    for (name, expected) in [
        ("main.bias", &[-0.1, -0.1][..]),
        ("main.weight", &[0.9, -0.2, -0.1, 0.8][..]),
        ("skip.bias", &[-0.1, -0.1][..]),
        ("skip.weight", &[1.9, -0.2, -0.1, 1.8][..]),
    ] {
        assert_close(
            &format!("CPU SGD `{name}`"),
            &cpu_sgd[name],
            expected,
            0.0,
            1e-6,
        );
    }
    for (name, expected) in [
        ("main.bias", &[-0.001, -0.001][..]),
        ("main.weight", &[0.999, -0.001, -0.001, 0.999][..]),
        ("skip.bias", &[-0.001, -0.001][..]),
        ("skip.weight", &[1.999, -0.001, -0.001, 1.999][..]),
    ] {
        assert_close(
            &format!("CPU Adam `{name}`"),
            &cpu_adam[name],
            expected,
            0.0,
            1e-6,
        );
    }

    let capabilities = available_devices();
    if capabilities.cuda {
        let cuda_sgd = run_optimizer_step(DeviceSpec::Cuda(0), OptimizerKind::Sgd);
        let cuda_adam = run_optimizer_step(DeviceSpec::Cuda(0), OptimizerKind::Adam);
        assert_parameter_maps_close(
            "CUDA SGD",
            &cuda_sgd,
            &cpu_sgd,
            CPU_CUDA_RTOL,
            CPU_CUDA_ATOL,
        );
        assert_parameter_maps_close(
            "CUDA Adam",
            &cuda_adam,
            &cpu_adam,
            CPU_CUDA_RTOL,
            CPU_CUDA_ATOL,
        );
    } else {
        eprintln!(
            "skipped CUDA optimizer parity: linked LibTorch reports no usable CUDA device (count={})",
            capabilities.cuda_device_count,
        );
    }

    if capabilities.mps {
        let mps_sgd = run_optimizer_step(DeviceSpec::Mps, OptimizerKind::Sgd);
        let mps_adam = run_optimizer_step(DeviceSpec::Mps, OptimizerKind::Adam);
        assert_parameter_maps_close("MPS SGD", &mps_sgd, &cpu_sgd, MPS_RTOL, MPS_ATOL);
        assert_parameter_maps_close("MPS Adam", &mps_adam, &cpu_adam, MPS_RTOL, MPS_ATOL);
    } else {
        eprintln!(
            "skipped MPS optimizer parity: linked LibTorch or the current machine reports no usable MPS device"
        );
    }
}

fn assert_safetensors_and_movement(device_spec: DeviceSpec, device: Device, backend: &str) {
    let cpu_file = TempSafetensors::new(&format!("cpu-to-{backend}"));
    let accelerator_file = TempSafetensors::new(&format!("{backend}-to-cpu"));
    let cpu_source = residual_model(DeviceSpec::Cpu);
    let expected = parameter_values(&cpu_source);
    cpu_source
        .save_weights(cpu_file.path())
        .expect("CPU SafeTensors save must succeed");

    let accelerator_target = residual_model(device_spec);
    zero_variables(&accelerator_target);
    let report = accelerator_target
        .load_weights(cpu_file.path())
        .expect("CPU SafeTensors must load into accelerator model");
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert_eq!(accelerator_target.device(), device);
    for variable in accelerator_target.var_store().variables().into_values() {
        assert_eq!(variable.device(), device);
    }
    assert_parameter_maps_close(
        &format!("CPU to {backend}"),
        &parameter_values(&accelerator_target),
        &expected,
        0.0,
        0.0,
    );

    accelerator_target
        .save_weights(accelerator_file.path())
        .expect("accelerator SafeTensors save must succeed");
    let cpu_target = residual_model(DeviceSpec::Cpu);
    zero_variables(&cpu_target);
    cpu_target
        .load_weights(accelerator_file.path())
        .expect("accelerator SafeTensors must load into CPU model");
    assert_parameter_maps_close(
        &format!("{backend} to CPU"),
        &parameter_values(&cpu_target),
        &expected,
        0.0,
        0.0,
    );

    let mut moving = residual_model(DeviceSpec::Cpu);
    moving.eval();
    moving
        .to_device(device_spec)
        .expect("CPU model must move to reported accelerator backend");
    assert_eq!(moving.device(), device);
    assert!(!moving.is_training());
    for variable in moving.var_store().variables().into_values() {
        assert_eq!(variable.device(), device);
    }
    assert_parameter_maps_close(
        &format!("CPU moved to {backend}"),
        &parameter_values(&moving),
        &expected,
        0.0,
        0.0,
    );
    let accelerator_output = moving
        .forward(
            GraphInputs::new()
                .with("features", float_tensor(&[1.0, 2.0], &[1, 2], device))
                .expect("accelerator input must insert"),
        )
        .expect("moved model must execute on accelerator");
    assert_close(
        &format!("moved {backend} output"),
        &values(accelerator_output.get("result").unwrap()),
        &[3.0, 6.0],
        if device == Device::Mps {
            MPS_RTOL
        } else {
            CPU_CUDA_RTOL
        },
        if device == Device::Mps {
            MPS_ATOL
        } else {
            CPU_CUDA_ATOL
        },
    );

    moving
        .to_device(DeviceSpec::Cpu)
        .expect("MPS model must move back to CPU");
    assert_eq!(moving.device(), Device::Cpu);
    assert!(!moving.is_training());
    assert_parameter_maps_close(
        "MPS moved to CPU",
        &parameter_values(&moving),
        &expected,
        0.0,
        0.0,
    );
    let cpu_output = moving
        .forward(
            GraphInputs::new()
                .with("features", float_tensor(&[1.0, 2.0], &[1, 2], Device::Cpu))
                .expect("CPU input must insert"),
        )
        .expect("moved model must execute after returning to CPU");
    assert_eq!(values(cpu_output.get("result").unwrap()), [3.0, 6.0]);
}

#[test]
fn accelerator_safetensors_and_cpu_movement_preserve_model_state() {
    let _backend_guard = backend_test_lock();
    let capabilities = available_devices();
    if capabilities.cuda {
        assert_safetensors_and_movement(DeviceSpec::Cuda(0), Device::Cuda(0), "CUDA");
    } else {
        eprintln!(
            "skipped CUDA SafeTensors/movement parity: linked LibTorch reports no usable CUDA device (count={})",
            capabilities.cuda_device_count,
        );
    }
    if capabilities.mps {
        assert_safetensors_and_movement(DeviceSpec::Mps, Device::Mps, "MPS");
    } else {
        eprintln!(
            "skipped MPS SafeTensors/movement parity: linked LibTorch or the current machine reports no usable MPS device"
        );
    }
}
