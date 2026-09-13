# Check, measure and resume a training application

These tools help you verify a new operation, locate slow application stages,
repeat random preprocessing and restart a training job at the next update.
They work with the same tensors, models, optimizers and loaders used elsewhere
in RustTorch.

## Verify a custom loss or transform

Start with a small Float64 input where the function is smooth. `gradcheck`
compares the derivatives recorded by automatic differentiation with central
finite differences. It catches mistakes such as an accidental detach or a
surrogate gradient that does not match the forward calculation.

```rust
use rusttorch::{Kind, Tensor, testing::{gradcheck, GradcheckOptions}};

let prediction = Tensor::from_slice(&[0.2_f64, -0.4]);
gradcheck(
    |x| Ok(x.f_tanh()?.f_square()?.f_mean(Kind::Double)?),
    &prediction,
    GradcheckOptions::default(),
)?;
# Ok::<(), rusttorch::RustTorchError>(())
```

The check accepts nonempty, finite CPU Float64 inputs and outputs and limits
the full Jacobian size. Use a deterministic function without input mutation;
dropout, random augmentation, integer operations and nondifferentiable points
need different tests. Existing `.grad()` buffers are preserved.

## Compare a refactored implementation

`assert_close` returns an error with the mismatch count and first location.
Shapes and dtypes must match, so an extra batch axis cannot be hidden by
broadcasting. Floating/complex values use the tolerances below; integer and
Boolean values compare exactly. NaNs are unequal unless explicitly enabled.

```rust
use rusttorch::{Tensor, testing::{assert_close, CloseOptions}};

let input = Tensor::from_slice(&[-1_f32, 0., 2.]);
let expected = Tensor::from_slice(&[0_f32, 0., 2.]);
assert_close(&input.f_relu()?, &expected, CloseOptions {
    rtol: 1e-5,
    atol: 1e-7,
    ..Default::default()
})?;
# Ok::<(), rusttorch::RustTorchError>(())
```

When intentionally comparing devices, set `check_device: false`; detached
copies are compared on CPU. This transfer belongs to correctness checking,
not to a claim about accelerator performance.

## Find slow application stages

Record regions such as data loading, forward calculation, backward calculation
and optimizer updates. The result is a bounded Chrome trace JSON file that can
be opened in [Perfetto](https://ui.perfetto.dev/).

```rust
use rusttorch::{Tensor, profiling::Profiler};

let profile = Profiler::new(1_000)?;
let input = Tensor::from_slice(&[1_f32, 2., 3.]);
let prediction = profile.record("forward", || Ok(input.f_square()?))?;
let _loss = profile.record("loss", || Ok(prediction.f_mean(prediction.kind())?))?;
let trace_json = profile.chrome_trace()?;
assert!(trace_json.contains("forward"));
// Persist trace_json to a .json file when collecting a real application trace.
# Ok::<(), rusttorch::RustTorchError>(())
```

Regions measure host wall time, including scheduling and waits. They do not
intercept LibTorch operators, record allocation stacks or produce GPU kernel
timings. A GPU launch may finish after a host region ends: synchronize the
required device work within the region when completed execution is the
measurement you need. Finish active spans before exporting or clearing a
window. Capacity exhaustion is an explicit error before another recorded
closure executes.

## Benchmark equivalent work

Warm up the operation, retain raw samples, and compare identical inputs and
runtime settings. Report the build profile, hardware, backend, thread count,
warmup count and sample count alongside the distribution.

```rust
use rusttorch::{Device, Kind, Tensor, profiling::{benchmark, BenchmarkOptions}};

let matrix = Tensor::f_ones([32, 32], (Kind::Float, Device::Cpu))?;
let measurement = benchmark(
    BenchmarkOptions { warmup: 3, samples: 10 },
    || Ok(matrix.f_matmul(&matrix)?),
    || Ok(()), // CPU work completes synchronously.
)?;
assert_eq!(measurement.seconds.len(), 10);
assert!(measurement.minimum <= measurement.median);
# Ok::<(), rusttorch::RustTorchError>(())
```

The synchronization closure runs before and after each call, including warmup.
For asynchronous work, provide a real backend synchronization operation or
include a required result transfer to CPU in the work being measured. State
whether transfer time is included. One sample times one closure call; group
very small operations inside the closure to reduce timer overhead. Returned
objects are kept alive through synchronization and timing, then dropped.

## Repeat application noise after a restart

Use `TensorRng` for random noise that should be isolated from other threads and
LibTorch's global generator. Each successful call uses one independent ChaCha12
stream. Save its state alongside the corresponding training step.

```rust
use rusttorch::{Device, Kind, reproducibility::TensorRng};

let mut noise = TensorRng::new(42);
let _first_batch = noise.normal(&[8, 4], (Kind::Float, Device::Cpu))?;
let saved = noise.state_dict();
let expected_next = noise.uniform(&[8, 4], (Kind::Float, Device::Cpu))?;

let mut restored = TensorRng::from_state(saved)?;
let actual_next = restored.uniform(&[8, 4], (Kind::Float, Device::Cpu))?;
assert!(expected_next.f_equal(&actual_next)?);
# Ok::<(), rusttorch::RustTorchError>(())
```

Samples are generated on CPU and transferred to the requested device. Uniform
values lie in `[0, 1)`; normals have zero mean and unit variance. The sequence is
a RustTorch contract, separate from PyTorch's generator sequence. Exact normal
replay targets the same build/platform because system math can differ in the
last bits. Invalid calls leave the saved position unchanged.

`RuntimeConfig` instead sets LibTorch's global seed and CPU thread count at
startup. Apply it before concurrent work; it does not snapshot an advanced
native RNG stream or make every backend kernel deterministic. Loader sampling
and `TaskContext` transforms retain their existing independent seed contracts.

## Save model, optimizer and application state together

`checkpoint::save` writes one `.rtckpt` artifact containing SafeTensors weights,
typed optimizer state and your JSON-serializable application state. Publication
is atomic and refuses to overwrite an existing path. Choose a new path for each
retained training boundary.

```rust,no_run
use rusttorch::{DeviceSpec, nn::Sequential, optim::Adam,
    checkpoint::{save, TrainingCheckpoint, CheckpointLimits},
    reproducibility::{TensorRng, TensorRngState}};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Progress {
    completed_updates: u64,
    noise: TensorRngState,
}

let model = Sequential::builder().linear(4, 1).build(DeviceSpec::Cpu)?;
let mut optimizer = Adam::builder().build(model.var_store())?;
let noise = TensorRng::new(42);
let progress = Progress { completed_updates: 0, noise: noise.state_dict() };
let limits = CheckpointLimits::default();
save("step-0.rtckpt", model.var_store(), &mut optimizer, &progress, limits)?;

// A new process rebuilds this architecture and its optimizer first.
let restored = Sequential::builder().linear(4, 1).build(DeviceSpec::Cpu)?;
let mut restored_optimizer = Adam::builder().build(restored.var_store())?;
let checkpoint = TrainingCheckpoint::<Progress>::read("step-0.rtckpt", limits)?;
let progress = checkpoint.restore(restored.var_store(), &mut restored_optimizer)?;
let _noise = TensorRng::from_state(progress.noise)?;
assert_eq!(progress.completed_updates, 0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a full training job, include the scheduler, `GradScalerState`, an exact
`LoaderState` or `StreamLoaderState`, and any preprocessing configuration in
your application state. Capture the loader after a consumed batch and the
model/optimizer after its completed update. Validate dataset identity and build
the resumed loader from `checkpoint.state()` before restoring model state.
Recreate training/evaluation mode explicitly. The checkpoint stores state, not
Rust code, architecture or arbitrary native RNG snapshots.

Format, tensor and optimizer validation and staging precede live-state changes.
Use a fresh model if a failed native device copy must not affect a running model.
The archive reader enforces finite byte/tensor limits, validates CRCs, rejects
unknown members and never extracts or executes files. Application JSON remains
typed, so `#[serde(deny_unknown_fields)]` is useful for your own schema.

After training, use the [deployment guide](https://docs.rs/rusttorch/0.4.0/rusttorch/tutorials/deployment/) to produce an inference
artifact. Keep a training checkpoint for continuation and choose PT2, ONNX or
TorchScript according to the receiving application's supported operations.
