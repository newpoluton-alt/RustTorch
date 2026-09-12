# rusttorch-core

Tensor computation, automatic differentiation, device selection, and shared
error types for RustTorch. Use this small package when you need tensor math
without the model-building or data-loading facade; use `rusttorch` when you
also need layers and optimizers.

## Installation

```toml
[dependencies]
rusttorch-core = { version = "0.2", features = ["download-libtorch"] }
```

## Features

This package has no default features. `download-libtorch` lets `tch` acquire
its compatible LibTorch 2.13.0 runtime. `doc-only` is for checks and rustdoc;
it intentionally does not provide a runtime for an executable.

## Native runtime

An executable needs LibTorch/PyTorch 2.13.0, matching `tch` 0.26.0. Use
`download-libtorch`, or disable default features and build with either
`LIBTORCH_USE_PYTORCH=1` against an installed Python `torch` 2.13.0 or
`LIBTORCH=/absolute/path/to/libtorch`. The platform dynamic loader must find
the selected runtime's shared libraries when the executable runs.

## Example

```rust
use rusttorch_core::{Device, Kind, Result, Tensor};

fn main() -> Result<()> {
    let input = Tensor::from_slice(&[2_f32, 3.0]).set_requires_grad(true);
    let loss = input.f_square()?.f_mean(Kind::Float)?;
    loss.f_backward()?;
    assert_eq!(Vec::<f32>::try_from(input.grad())?, vec![2.0, 3.0]);
    assert_eq!(input.device(), Device::Cpu);
    Ok(())
}
```

Use `resolve_device(DeviceSpec::Auto)` to choose an available accelerator or
CPU, and create or move tensors onto that device. Use `no_grad` for inference
or parameter assignment. The [API guide](https://docs.rs/rusttorch-core) has
examples of matrix multiplication, backend selection, and gradients.
