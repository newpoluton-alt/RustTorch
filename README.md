<div align="center">

# RustTorch

**Build, train, and run neural networks in Rust.**

[![crates.io](https://img.shields.io/crates/v/rusttorch.svg)](https://crates.io/crates/rusttorch)
[![docs.rs](https://img.shields.io/docsrs/rusttorch)](https://docs.rs/rusttorch)
[![license](https://img.shields.io/crates/l/rusttorch.svg)](#license)
[![CI](https://github.com/newpoluton-alt/RustTorch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/newpoluton-alt/RustTorch/actions/workflows/ci.yml)

[Get started](#installation) · [Train a model](#example-train-a-regressor) ·
[Guides](#documentation) · [Roadmap](docs/roadmap.md) · [Contribute](CONTRIBUTING.md)

</div>

RustTorch brings tensor computation, automatic differentiation, neural-network
layers, optimizers, and typed data pipelines to Rust. Write ordinary Rust
functions, compose layers, and train on CPU or an available CUDA or MPS device.
LibTorch supplies the numerical kernels; RustTorch supplies model construction,
validation, data loading, and weight management.

Use it for regression and classification, convolutional feature extraction,
learned token embeddings, streaming training data, and inference inside a Rust
application. The API is evolving in the 0.x series. The
[roadmap](docs/roadmap.md) separates available functionality from future work.

## Installation

Install the 0.2 release from crates.io. Rust 1.88 or newer is required:

```sh
cargo install rusttorch-cli --version 0.2.0
cargo new rusttorch-demo
cd rusttorch-demo
cargo add rusttorch@0.2
rusttorch setup --backend auto
```

Setup selects a compatible native runtime and checks your project. The first
setup can download a large LibTorch archive. Use `--backend cpu` to select CPU
or `--backend cuda-12.6` for the managed CUDA distribution on a supported host.
See [platform setup](docs/platform-support.md) for prerequisites and runtime
library paths.

This source guide includes model and optimizer APIs added after 0.2.0. To run
its examples before the next release, use a checkout containing these changes
and point your application at it:

```sh
cargo add rusttorch --path /absolute/path/to/RustTorch
```

For the released 0.2 API, use the [published documentation](https://docs.rs/rusttorch/0.2.0).

## Example: train a regressor

Put this in `src/main.rs`, then run `cargo run`. Each row contains two numeric
features; the model learns one target value per row.

```rust
use rusttorch::{DeviceSpec, Result, Tensor, no_grad};
use rusttorch::nn::{Sequential, functional};
use rusttorch::optim::AdamW;

fn main() -> Result<()> {
    let mut model = Sequential::builder()
        .linear(2, 16)
        .relu()
        .linear(16, 1)
        .build(DeviceSpec::Auto)?;
    let inputs = Tensor::from_slice(&[0_f32, 0., 0., 1., 1., 0., 1., 1.])
        .f_reshape([4, 2])?.f_to_device(model.device())?;
    let targets = Tensor::from_slice(&[0_f32, 1., 1., 2.])
        .f_reshape([4, 1])?.f_to_device(model.device())?;
    let mut optimizer = AdamW::builder()
        .learning_rate(0.01)
        .build(model.var_store())?;

    model.train();
    for _ in 0..200 {
        let predictions = model.forward(&inputs)?;
        let loss = functional::mse_loss(&predictions, &targets)?;
        optimizer.backward_step(&loss)?;
    }

    model.eval();
    let predictions = no_grad(|| model.forward(&inputs))?;
    println!("predictions: {predictions:?}");
    model.save_weights("regressor.safetensors")?;
    Ok(())
}
```

For classification, make the final layer output one logit per class and use
`functional::cross_entropy` with `Int64` class indices. For inference, rebuild
the same architecture, call `load_weights("regressor.safetensors")`, switch to
`eval()`, and run the forward pass inside `no_grad`. Evaluation mode controls
layers such as dropout; `no_grad` controls gradient recording.

The [training guide](docs/training.md) covers custom models, optimizer choice,
gradient accumulation and clipping, learning-rate changes, and saving weights.

## Example: batch a dataset

A dataset returns owned samples and its own error type. The borrowed loader is
useful for a small dataset or for debugging a preprocessing pipeline.

```rust
use std::convert::Infallible;
use rusttorch::data::{DataLoader, Dataset, SequentialSampler};

struct Rows(Vec<f32>);
impl Dataset for Rows {
    type Sample = f32;
    type Error = Infallible;
    fn len(&self) -> usize { self.0.len() }
    fn get(&self, index: usize) -> Result<f32, Infallible> {
        Ok(self.0[index])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Rows(vec![2.0, 3.0, 5.0]);
    let batches = DataLoader::new(&data, SequentialSampler::new(data.len()), 2, false)?
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(batches, [vec![2.0, 3.0], vec![5.0]]);
    Ok(())
}
```

Use `DataLoader::builder(dataset)` for reusable epochs, shuffling, transforms,
worker threads, and prefetching. Use `StreamDataLoader` for explicitly sharded
streams. See the [data guide](crates/rusttorch-data/README.md) for worker
configuration, distributed sampling, memory budgets, and the supported exact
checkpoint/resume combinations.

## Features

| Need | RustTorch API |
|---|---|
| Tensor math and gradients | `Tensor`, `Kind`, `no_grad`, and fallible tensor operations |
| Dense models | `nn::Linear`, `Sequential`, ReLU, GELU, Dropout, and Flatten |
| Spatial or sequence features | `nn::Conv1d`, `Conv2d`, and `Conv3d` |
| Feature normalization | `nn::LayerNorm` |
| Learned categorical or token features | `nn::Embedding` |
| Model training | MSE, cross-entropy, Adam, AdamW, RMSprop, SGD, and gradient clipping |
| Input pipelines | Datasets, samplers, collation, bounded workers, streams, and checkpoints |
| Weight persistence | SafeTensors with strict validation and explicit name mappings |
| Model inspection | Named graph inputs, validation, summaries, and DOT diagrams |

`download-libtorch` is enabled by default and acquires the compatible native
runtime. For documentation builds without a native runtime, disable defaults
and enable `doc-only`. **`doc-only` cannot run a model.**

```toml
[dependencies]
rusttorch = { version = "0.2", default-features = false, features = ["doc-only"] }
```

## Native runtime

RustTorch currently uses `tch` 0.26.0 and LibTorch 2.13.0. LibTorch is the C++
numerical library also used by PyTorch; your Rust application does not need a
Python training loop. You can use the managed download, an installed LibTorch
selected with `LIBTORCH`, or Python `torch` 2.13.0 selected with
`LIBTORCH_USE_PYTORCH=1`. The platform loader must find the selected shared
libraries when your executable runs.

`DeviceSpec::Auto` selects CUDA, then MPS, then CPU according to runtime
availability. An explicit unavailable device returns an error. Consult the
[device guide](docs/device-system.md), [CUDA guide](docs/cuda-support.md), and
[MPS guide](docs/mps-support.md) for backend requirements.

## Documentation

| Guide | Start here when you want to… |
|---|---|
| [Rust API reference](https://docs.rs/rusttorch) | Find types, methods, defaults, and Rust examples |
| [Models and training](docs/training.md) | Build a custom model and control its training loop |
| [Data pipelines](crates/rusttorch-data/README.md) | Load, transform, batch, and resume training data |
| [Graph guide](docs/graph-system.md) | Inspect named inputs, branches, and execution order |
| [Model interoperability](docs/model-interoperability.md) | Exchange weights with another model implementation |
| [Platform setup](docs/platform-support.md) | Configure the native runtime for your machine |
| [Architecture](docs/architecture.md) | Understand ownership and execution boundaries |
| [Roadmap](docs/roadmap.md) | See implementation priorities and remaining capabilities |
| [Compatibility evidence](docs/api-coverage.md) | Check exact tested scopes and source references |

PyTorch is the behavioral reference for compatibility tests and weight exchange.
It appears in provenance and interoperability guides for that reason. The
long-term goal is to provide its framework functionality through RustTorch
APIs; [current coverage](docs/api-coverage.md) is narrower. Python runtime
features need explicit Rust equivalents, and planned functionality is not a
support claim.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing an API change. It covers
issue-first discussion, focused tests, examples, compatibility evidence, source
attribution, and DCO sign-off. The [maintainer guide](docs/maintainer-guide.md)
describes review and release checks.

[Code of conduct](CODE_OF_CONDUCT.md) · [Governance](GOVERNANCE.md) ·
[Support](SUPPORT.md) · [Security](SECURITY.md)

## License

RustTorch is available under the [MIT License](LICENSE-MIT) or
[Apache License 2.0](LICENSE-APACHE), at your option. Dependencies and upstream
behavioral references retain their own terms; see
[third-party notices](THIRD_PARTY_NOTICES.md). RustTorch is independent of the
PyTorch Foundation.
