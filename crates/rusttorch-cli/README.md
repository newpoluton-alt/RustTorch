# RustTorch setup CLI

## Installation

`rusttorch-cli` installs the `rusttorch` bootstrap command. Run it in a Cargo
project after adding the RustTorch library dependency:

```sh
cargo install --git https://github.com/newpoluton-alt/RustTorch rusttorch-cli
cargo add rusttorch --git https://github.com/newpoluton-alt/RustTorch
rusttorch setup --backend auto
cargo run
```

## Features

`rusttorch-cli` has no Cargo features. The application it configures uses
RustTorch's `download-libtorch` feature by default; `doc-only` is only for
library checks and rustdoc, not an executable runtime.

## Native runtime

The CLI itself does not link LibTorch. The configured application needs
LibTorch/PyTorch 2.13.0, matching `tch` 0.26.0: use the default downloaded
runtime, or build with `LIBTORCH_USE_PYTORCH=1` against Python `torch` 2.13.0
or `LIBTORCH=/absolute/path/to/libtorch`. Its dynamic loader must find the
selected shared libraries at runtime.

## Example

The accepted setup commands are exactly:

```sh
rusttorch setup --backend auto
rusttorch setup --backend cpu
rusttorch setup --backend cuda-12.6
```

Setup locates the Cargo workspace root, writes project-local managed settings
for CPU or CUDA, and runs `cargo check`. It does not install globally managed
LibTorch files, NVIDIA drivers, or CUDA toolkits. `auto` preserves an active
LibTorch/Python/TORCH selection; otherwise it chooses CUDA 12.6 only on a
Linux or Windows NVIDIA system with a compatible driver, and chooses CPU on
the remaining supported systems. The macOS CPU LibTorch distribution can use
MPS when supported.

See the [RustTorch README](https://github.com/newpoluton-alt/RustTorch) for
configuration ownership, target isolation, offline setup, and dynamic-loader
caveats.
