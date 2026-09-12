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
| Time series | `Conv1d` | `[batch, channels, length]` |
| Images | `Conv2d` | `[batch, channels, height, width]` |
| Volumes | `Conv3d` | `[batch, channels, depth, height, width]` |
| Token IDs or categories | `Embedding` | Integer IDs → `[..., embedding_dim]` |
| Normalize feature vectors | `LayerNorm` | Normalize the configured trailing dimensions |

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

For small fixed-size images, start with convolutional features and a dense
classification head. Here two synthetic grayscale images demonstrate the full
forward/loss/update path. Replace them with batches from your dataset for real
training; the final layer has one output per class.

```rust
use rusttorch::{DeviceSpec, Kind, Result, Tensor};
use rusttorch::nn::{ConvConfig, Sequential, functional};
use rusttorch::optim::AdamW;

fn main() -> Result<()> {
    let model = Sequential::builder()
        .conv2d(ConvConfig::new(1, 4, [3, 3]).padding([1, 1]))
        .relu()
        .flatten(1, -1)
        .linear(4 * 8 * 8, 2)
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

## Choose a loss and optimizer

For regression, `functional::mse_loss` compares predictions and targets with
mean reduction. For classification, `functional::cross_entropy` takes raw
logits and `Int64` class indices. Do not apply softmax before cross-entropy.

| Optimizer | Use it when | Default learning rate |
|---|---|---:|
| `Adam` | You want adaptive updates without decoupled weight decay | `0.001` |
| `AdamW` | You want adaptive updates with decoupled regularization | `0.001` |
| `Sgd` | You want direct gradient updates and optional momentum | `0.001` |
| `RmsProp` | You want updates scaled by a moving squared-gradient average | `0.01` |

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

    optimizer.zero_grad();
    for _ in 0..2 {
        let loss = functional::mse_loss(&model.forward(&input)?, &target)?;
        loss.f_div_scalar(2.0)?.backward();
    }
    optimizer.clip_grad_norm(1.0)?;
    optimizer.step();
    optimizer.set_learning_rate(0.0005)?;
    Ok(())
}
```

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

A weight file does not contain the model's Rust code, optimizer state, or input
pipeline position. Loader checkpoints are separate typed state; use the
[data guide](../crates/rusttorch-data/README.md) for supported resume modes and
the [interoperability guide](model-interoperability.md) for weight naming,
validation, and exchange.
