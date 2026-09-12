# Models and training

A RustTorch model is ordinary Rust code that transforms tensors. Use
`Sequential` for a straight chain of layers, or implement `nn::Module` when
inputs branch, layers share parameters, or your application needs a named model
type. Trainable tensors live in `nn::VarStore`; attach an optimizer to that
store so it can update every registered parameter.

## Choose layers for your input

| Input or task | Useful layers | Typical shape |
|---|---|---|
| Numeric features | `Linear`, ReLU, Dropout | `[batch, features]` |
| Time series | `Conv1d`, `Rnn`, `Gru`, `Lstm` | `[batch, channels, length]` |
| Images | `Conv2d`, `BatchNorm2d`, `MaxPool2d` | `[batch, channels, height, width]` |
| Volumes | `Conv3d` | `[batch, channels, depth, height, width]` |
| Token IDs or categories | `Embedding` | Integer IDs → `[..., embedding_dim]` |
| Normalize feature vectors | `LayerNorm` | Normalize the configured trailing dimensions |
| Normalize spatial features | Batch/InstanceNorm1d/2d/3d, GroupNorm | Channel axis after the batch axis |
| Reduce or expand spatial resolution | Max/Avg/Adaptive pools, ConvTranspose1d/2d/3d | One, two or three spatial axes |
| Learn relationships between tokens | MultiheadAttention, TransformerEncoder/Decoder | Sequence and batch layouts are configurable |

Convolution configuration controls per-axis kernels, stride, zero padding,
dilation, groups, and bias. Embedding padding IDs can reserve a row that does
not receive gradients. Layer normalization can learn scale and bias, or use
only input statistics. See the [API reference](https://docs.rs/rusttorch/latest/rusttorch/nn/)
for each configuration's defaults, shape requirements, and examples.

## Build a model with a residual connection

Register layers under meaningful paths to give saved parameters stable names.
The input, model parameters, and training targets must be on the same device.

```rust
use rusttorch::{Device, Kind, Result, Tensor};
use rusttorch::nn::{Linear, LinearConfig, Module, VarStore, functional};

struct ResidualBlock {
    projection: Linear,
}

impl Module for ResidualBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = functional::relu(&self.projection.forward(input)?)?;
        Ok(hidden.f_add(input)?)
    }
}

fn main() -> Result<()> {
    let parameters = VarStore::new(Device::Cpu);
    let model = ResidualBlock {
        projection: LinearConfig::new(4, 4)
            .build(&(parameters.root() / "projection"))?,
    };
    let input = Tensor::f_ones([2, 4], (Kind::Float, Device::Cpu))?;
    assert_eq!(model.forward(&input)?.size(), [2, 4]);
    assert!(parameters.variables().contains_key("projection.weight"));
    Ok(())
}
```

For mode-dependent layers, implement `forward_t` and pass its `training` flag
to child modules. Keep the `VarStore` alongside the model for optimization,
device movement, and weight persistence.

## Train an image classifier

Combine convolution, batch normalization and pooling to extract image features.
Adaptive pooling gives the classifier a fixed-width input even when image sizes
change between batches. Here two synthetic grayscale images demonstrate the full
forward/loss/update path. Replace them with batches from your dataset for real
training; the final layer has one output per class.

```rust
use rusttorch::{DeviceSpec, Kind, Result, Tensor};
use rusttorch::nn::{AdaptiveAvgPool2d, BatchNormConfig, ConvConfig, MaxPool2d, Sequential, functional};
use rusttorch::optim::AdamW;

fn main() -> Result<()> {
    let model = Sequential::builder()
        .conv2d(ConvConfig::new(1, 4, [3, 3]).padding([1, 1]))
        .layer(|path| BatchNormConfig::<2>::new(4).build(path))
        .relu()
        .layer(|_| MaxPool2d::new([2, 2]))
        .layer(|_| AdaptiveAvgPool2d::new([1, 1]))
        .flatten(1, -1)
        .linear(4, 2)
        .build(DeviceSpec::Cpu)?;
    let images = Tensor::f_zeros([2, 1, 8, 8], (Kind::Float, model.device()))?;
    let classes = Tensor::from_slice(&[0_i64, 1]);
    let mut optimizer = AdamW::builder().build(model.var_store())?;
    let logits = model.forward(&images)?;
    assert_eq!(logits.size(), [2, 2]);
    optimizer.backward_step(&functional::cross_entropy(&logits, &classes)?)?;
    Ok(())
}
```

Batch normalization updates registered running statistics during training and
uses them during evaluation. Group normalization always uses the current input;
it can be useful when batches are small. Instance normalization computes each
sample's spatial statistics and defaults to no affine parameters or running
statistics. Use `SequentialBuilder::layer` to insert any `Module` through a
factory that receives its registered parameter path.

## Choose a loss and optimizer

| Target | Loss | Input |
|---|---|---|
| Continuous values | `mse_loss`, `l1_loss` | Predictions and target values |
| Regression with large residuals | `smooth_l1_loss`, `huber_loss` | Predictions and targets; tune beta or delta |
| Exclusive classes | `cross_entropy`, `cross_entropy_with_options` | Raw logits and integer labels or probability targets |
| Binary or independent labels | `binary_cross_entropy_with_logits` | Raw logits and target probabilities |
| Already computed probabilities | `binary_cross_entropy` | Values in `[0, 1]` and targets |
| Already computed log probabilities | `nll_loss` | Log probabilities and integer class labels |

`mse_loss` and `cross_entropy` use mean reduction. Configurable variants accept
`Reduction::None`, `Mean`, or `Sum`. Reduce a non-scalar loss before the optimizer
step. Cross-entropy options include class weights, ignored class indices, and
label smoothing; BCE with logits also accepts positive-class weights. Do not
apply softmax before cross-entropy.

```rust
use rusttorch::{Reduction, Result, Tensor, nn::functional::{self, CrossEntropyOptions}};
fn main() -> Result<()> {
    let logits = Tensor::from_slice(&[2_f32, -1., 0., 1.]).f_reshape([2, 2])?;
    let labels = Tensor::from_slice(&[0_i64, -1]);
    let loss = functional::cross_entropy_with_options(&logits, &labels,
        CrossEntropyOptions { ignore_index: -1, label_smoothing: 0.1,
            reduction: Reduction::Mean, ..Default::default() })?;
    assert!(loss.double_value(&[]).is_finite());
    Ok(())
}
```

| Optimizer | Use it when | Default learning rate |
|---|---|---:|
| `Adam` | You want adaptive updates without decoupled weight decay | `0.001` |
| `AdamW` | You want adaptive updates with decoupled regularization | `0.001` |
| `Sgd` | You want direct gradient updates and optional momentum | `0.001` |
| `RmsProp` | You want updates scaled by a moving squared-gradient average | `0.01` |
| `Adagrad` | You want step sizes based on accumulated squared gradients | `0.01` |
| `Adadelta` | You want moving averages of both gradients and updates | `1.0` |
| `Adamax` | You want adaptive updates using an infinity-norm accumulator | `0.002` |

These are starting configurations, not guarantees of convergence. Tune the
learning rate against a validation set. Adam and RMSprop use coupled L2 decay;
AdamW applies its decay directly to parameters. RMSprop supports momentum and
centered variance estimates. Optimizers preserve their internal state between
steps, so construct them once before the training loop.

## Accumulate and clip gradients

`backward_step` clears gradients, differentiates the scalar loss, and updates
parameters. For microbatch accumulation or clipping, separate those operations.
With equally sized microbatches, divide each mean loss by the number of
microbatches before backward. For unequal sizes, weight losses by their sample
counts instead.

```rust
use rusttorch::{DeviceSpec, Kind, Result, Tensor};
use rusttorch::nn::{Sequential, functional};
use rusttorch::optim::AdamW;

fn main() -> Result<()> {
    let model = Sequential::builder().linear(2, 1).build(DeviceSpec::Cpu)?;
    let mut optimizer = AdamW::builder().learning_rate(0.001)
        .weight_decay(0.01).build(model.var_store())?;
    let input = Tensor::f_ones([4, 2], (Kind::Float, model.device()))?;
    let target = Tensor::f_zeros([4, 1], (Kind::Float, model.device()))?;

    optimizer.try_zero_grad()?;
    for _ in 0..2 {
        let loss = functional::mse_loss(&model.forward(&input)?, &target)?;
        loss.f_div_scalar(2.0)?.backward();
    }
    optimizer.clip_grad_norm(1.0)?;
    optimizer.try_step()?;
    optimizer.set_learning_rate(0.0005)?;
    Ok(())
}
```

Use `try_zero_grad` and `try_step` to propagate runtime failures with `?`.
The older `zero_grad` and `step` conveniences panic on a runtime error. Clearing
gradients removes them; parameters that receive no gradient in the next backward
pass keep their weights and optimizer moments unchanged.

`clip_grad_norm` limits the combined L2 norm; `clip_grad_value` clamps individual
elements. Apply clipping after backward and before `step`. Learning-rate changes
affect all optimizer parameter groups and retain accumulated moments. These
clipping methods are intended for dense gradients; native errors propagate as
`Result` failures. A zero learning rate suppresses parameter updates while the
optimizer can still update its internal statistics.

## Evaluate and persist a model

Switch a `Sequential` model to `eval()` to disable training behavior such as
dropout, and use `no_grad(|| model.forward(&input))` to avoid recording an
autograd graph. These are separate choices: evaluation mode alone still allows
gradients, which can be useful for input attribution.

`model.save_weights("model.safetensors")` stores named parameter values. To load
them, build the same architecture and call
`model.load_weights("model.safetensors")`. Custom models use
`interop::save_state_dict(path, &parameters)` and
`interop::load_state_dict(path, &parameters)` with their `VarStore`.

Weight files include registered normalization buffers as well as trainable
parameters. A weight file does not contain the model's Rust code, optimizer
state, or input pipeline position. Loader checkpoints are separate typed state; use the
[data guide](../crates/rusttorch-data/README.md) for supported resume modes and
the [interoperability guide](model-interoperability.md) for weight naming,
validation, and exchange.


## Set different learning rates and schedule training

Assign parameter groups while constructing the model. For example, use a lower
learning rate for an existing feature extractor and a higher one for a new head:

```rust
use rusttorch::{Device, Result, nn::{LinearConfig, VarStore}, optim::{AdamW, StepLr}};
fn main() -> Result<()> {
    let parameters = VarStore::new(Device::Cpu);
    let _features = LinearConfig::new(8, 4).build(&(parameters.root().set_group(0) / "features"))?;
    let _head = LinearConfig::new(4, 2).build(&(parameters.root().set_group(1) / "head"))?;
    let mut optimizer = AdamW::builder().build(&parameters)?;
    optimizer.set_group_learning_rate(0, 0.0001)?;
    optimizer.set_group_learning_rate(1, 0.001)?;
    optimizer.set_group_weight_decay(1, 0.0)?;
    let mut schedule = StepLr::new(&optimizer, 10, 0.1)?;
    // Call after the epoch's optimizer updates:
    schedule.step(&mut optimizer)?;
    assert_eq!(schedule.epoch(), 1);
    Ok(())
}
```

Construct the scheduler after choosing the initial group rates. `StepLr` decays
at fixed intervals, `MultiStepLr` at specified epochs, `ExponentialLr` at every
epoch, and `CosineAnnealingLr` along a cosine curve. `ReduceLrOnPlateau` consumes
a validation metric and supports min/max mode, patience, threshold, cooldown,
and a minimum rate per group. Step schedulers after optimizer updates; for
validation scheduling, call `step(&mut optimizer, metric)` after evaluation.
Keep the set of parameter groups fixed while using a scheduler.

## Scale gradients for mixed precision

Use `amp::autocast_for(Device::Cuda(index), true, closure)` around the forward
pass and loss on an available CUDA device. CUDA autocast selects native kernel
dtypes; model parameters normally remain `Float`. `autocast_for` rejects enabled
CPU and MPS policies. A disabled scope runs normally, making device selection
explicit in portable applications.

`GradScaler` also works with full-precision CPU tensors. The update sequence is:

```rust
use rusttorch::{DeviceSpec, Kind, Result, Tensor, amp::GradScaler,
    nn::{Sequential, functional}, optim::AdamW};
fn main() -> Result<()> {
    let model = Sequential::builder().linear(2, 1).build(DeviceSpec::Cpu)?;
    let mut optimizer = AdamW::builder().build(model.var_store())?;
    let mut scaler = GradScaler::default();
    let input = Tensor::ones([4, 2], (Kind::Float, model.device()));
    let target = Tensor::zeros([4, 1], (Kind::Float, model.device()));
    optimizer.try_zero_grad()?;
    let loss = functional::mse_loss(&model.forward(&input)?, &target)?;
    scaler.scale(&loss)?.f_backward()?;
    scaler.unscale(&optimizer)?;
    optimizer.clip_grad_norm(1.0)?;
    let applied = scaler.step(&mut optimizer)?;
    scaler.update()?;
    assert!(applied);
    Ok(())
}
```

The scaler skips an update if gradients contain NaN or infinity and reduces its
scale; successful intervals grow it. Call `update` after both applied and skipped
steps. Unscale once before clipping, and finish the update before saving state.
Multiple scaled backward calls may accumulate before unscale. One scaler cycle
coordinates one optimizer, using dense gradients. Scaling uses normal positive
`f32` magnitudes and clamps backoff at their lower bound. CUDA execution needs
CUDA hardware; CPU scaler tests do not establish CUDA numerical coverage.

## Save a training run and resume the next update

Save a completed update boundary: weights and buffers with SafeTensors,
`optimizer.state_dict()` for named moments, configuration and parameter groups,
the scheduler itself with serde, and `scaler.state_dict()` for scale and growth
tracking. Deserialize `OptimizerState` and call `load_state_dict` on an optimizer
attached to the reconstructed model. Load scaler state through its validated
`load_state_dict` method, and resume with the saved scheduler.

For JSON storage, enable `serde_json`'s `float_roundtrip` feature in your
application so floating-point configuration survives parsing exactly:

```toml
[dependencies]
serde_json = { version = "1", features = ["float_roundtrip"] }
```

The runnable [training checkpoint example](../examples/training_checkpoint.rs)
saves a classifier after two epochs, restores all four components, and checks
that its next update exactly matches uninterrupted training:

```sh
cargo run --example training_checkpoint
```

Use a fresh checkpoint directory and publish its completion marker only after
all components are saved. This example uses fixed data and deterministic layers.
A shuffled or stochastic training job must also restore its data-loader position
and random state; optimizer state alone cannot reproduce those inputs. Optimizer
checkpoints use a versioned RustTorch format, not Python pickle/state dictionaries.
Move the model to its final device before constructing or restoring its optimizer.

For token classification, explicit recurrent state, masking, and sequence-to-
sequence models, continue with [sequence models](sequence-models.md).
