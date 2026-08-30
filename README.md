# RustTorch

RustTorch is an unofficial Rust frontend for PyTorch-style eager machine
learning. The Cargo package is `rusttorch`; Rust code imports the
library as `rusttorch`. Version 0.1 is an MVP and does not yet promise API or
graph-format stability.

`tch` supplies Rust bindings and LibTorch performs tensor storage, kernels,
automatic differentiation, and backend execution. RustTorch adds a focused,
idiomatic model API, SafeTensors state interchange, device selection, and an
optional explicit graph for inspection and validation. It is not affiliated
with or endorsed by the PyTorch Foundation.

## Installation

Add the package to your application:

```toml
[dependencies]
rusttorch = "0.1"
```

RustTorch links dynamically to PyTorch/LibTorch 2.13.0. Adding the Cargo
dependency does not install LibTorch. Before building an application, choose
one of the setups supported by `tch`:

- Install Python PyTorch 2.13.0 in an activated environment and set
  `LIBTORCH_USE_PYTORCH=1` while building the application.
- Install the matching standalone LibTorch 2.13.0 distribution and set
  `LIBTORCH` to its absolute installation path.

When running the application, the operating-system loader must be able to find
LibTorch. If needed, add its `lib` directory (or the Python package's
`torch/lib` directory) to
`LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS. CUDA applications
must use a CUDA-enabled LibTorch build compatible with the installed NVIDIA
driver. RustTorch does not install LibTorch, CUDA, or a driver.

The public API is documented on
[docs.rs](https://docs.rs/rusttorch). Design, compatibility, backend,
and interoperability guides live under [`docs/`](docs/). The source is hosted
in the public [RustTorch GitHub repository](https://github.com/newpoluton-alt/RustTorch).

## Compatibility

- `tch`: 0.26.0
- PyTorch/LibTorch: 2.13.0
- behavioral reference: PyTorch tag v2.13.0, commit `cf30153`
- local project Python: 3.14.7 with torch 2.13.0, safetensors 0.8.0,
  and NumPy 2.5.2
- Rust: 1.88 or newer

The current development host is macOS 26.5.2 on arm64. Python reports that MPS
is built and available and CUDA is unavailable. The Rust tests independently
execute the linked LibTorch backend; Python availability is not treated as a
Rust pass. CUDA requires a CUDA-enabled LibTorch distribution and a compatible
NVIDIA driver on Linux or Windows.

The CPU interoperability harness is verified: Python SafeTensors load strictly
in Rust and match Linear forward/input/parameter gradients, cross-entropy, MSE,
one SGD step, one Adam step, and residual forward/input/parameter gradients.
Rust SafeTensors load strictly back into Python and forward output matches.
On this host, Rust MPS forward/backward, gradients, SGD, Adam, CPU↔MPS model
movement, and SafeTensors state transfer pass against the CPU reference. CUDA
is unavailable and is reported as skipped, not passed.

## Contributor setup

Create the project-local environment if it is absent, then activate the
platform setup from the repository root:

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python3 \
  'torch==2.13.0' 'safetensors==0.8.0' 'numpy==2.5.2'
. scripts/dev-env.sh
```

These repository scripts use the active project `.venv` as LibTorch via
`LIBTORCH_USE_PYTORCH=1`. They do not edit shell startup files or global Python
installations. They are contributor conveniences, not an installation step for
applications consuming RustTorch from crates.io. See
[platform support](docs/platform-support.md) for the equivalent consumer setup
and the repository's Linux CPU and CUDA helpers.

## Contributor checks

```sh
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo doc --no-deps
scripts/check-backends.sh
scripts/run-python-parity.sh
```

Run examples with `cargo run --example <name>`. Hardware tests are conditional:
an unavailable backend must be reported as skipped, never as passed.

## Eager API

The intended primary style is ordinary eager Rust code:

```rust,no_run
use rusttorch::{nn, optim, DeviceSpec, Kind, Tensor};

fn train_step() -> rusttorch::Result<()> {
    rusttorch::manual_seed(42);
    let mut model = nn::Sequential::builder()
        .linear(4, 16)
        .relu()
        .dropout(0.1)
        .linear(16, 3)
        .build(DeviceSpec::Auto)?;
    let device = model.device();
    let inputs = Tensor::randn([32, 4], (Kind::Float, device));
    let targets = Tensor::randint(3, [32], (Kind::Int64, device));
    let mut optimizer = optim::Adam::builder()
        .learning_rate(1e-3)
        .build(model.var_store())?;
    model.train();
    let loss = nn::functional::cross_entropy(&model.forward(&inputs)?, &targets)?;
    optimizer.backward_step(&loss)?;
    model.save_weights("model.safetensors")?;
    Ok(())
}
```

The MVP surface covers Linear, Identity, ReLU, GELU, Dropout, Flatten,
Sequential, MSE, cross-entropy, Adam, and SGD. LibTorch owns the dynamic
autograd graph. Calling `eval()` changes training-sensitive modules such as
Dropout; it does not disable gradients.

Custom eager modules do not need to construct a RustTorch graph. The optional
Graph IR supports named inputs and outputs, branching, residual addition,
validation, summaries, and Graphviz DOT generation while still executing
normal `tch::Tensor` operations.

## Devices

`DeviceSpec::Auto` selects CUDA device 0 when usable, otherwise MPS when
usable, otherwise CPU. Explicit device requests fail when unavailable; they do
not silently fall back. Inputs are not silently moved to a model device.

```rust,ignore
model.to_device(DeviceSpec::Cpu)?;
model.to_device(DeviceSpec::Mps)?;
model.to_device(DeviceSpec::Cuda(0))?;
```

Rebuild an optimizer after moving a model unless the implementation explicitly
documents and verifies optimizer-state movement.

## PyTorch weight interchange

SafeTensors is the portable weight contract. Architecture and parameter names
must be equivalent or explicitly mapped.

RustTorch 0.1 modules register parameter tensors. The state helpers serialize
the named tensors returned by `tch::nn::VarStore::variables`; there is not yet
a dedicated RustTorch persistent-buffer API or a tested tied-parameter alias
contract.

Python to Rust:

```python
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")
```

```rust,ignore
model.load_weights("model.safetensors")?;
```

Rust to Python:

```rust,ignore
model.save_weights("model.safetensors")?;
```

```python
from safetensors.torch import load_file
model.load_state_dict(load_file("model.safetensors"), strict=True)
```

See [model interoperability](docs/model-interoperability.md) and
[state-dict naming](docs/state-dict-naming.md). The helper scripts under
`scripts/` create and verify a small deterministic parity model.

## Model formats and limitations

- `.safetensors`: the only RustTorch 0.1 model-state format.
- `.pt`/`.bin` state dictionaries: not supported by the RustTorch 0.1 API;
  pickle must be treated as untrusted.
- TorchScript: not exposed by RustTorch 0.1. Applications that need opaque
  legacy inference can use `tch::CModule` directly.
- `.pt2`/`torch.export`: roadmap only.
- `.rusttorch`: graph-plus-weights package concept only; not implemented.
- `torch.save(model)`: arbitrary Python-pickled whole models are not portable
  to Rust and are not supported.

The MVP does not implement replacement autograd, `torch.compile`, custom
kernels, distributed training, full PyTorch operator coverage, convolutional,
recurrent, or transformer layers, quantization, or cross-language optimizer
checkpoint resume.

## License

Original RustTorch code is dual-licensed under MIT or Apache-2.0. PyTorch and
other dependencies retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
