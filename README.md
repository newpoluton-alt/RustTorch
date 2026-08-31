<div align="center">

# RustTorch

**easy to use and easy to implement, but crazy fast**

An unofficial, eager-first Rust frontend over LibTorch.

[![crates.io](https://img.shields.io/crates/v/rusttorch.svg)](https://crates.io/crates/rusttorch)
[![docs.rs](https://img.shields.io/docsrs/rusttorch)](https://docs.rs/rusttorch)
[![license](https://img.shields.io/crates/l/rusttorch.svg)](#license)
[![MSRV](https://img.shields.io/crates/msrv/rusttorch.svg)](Cargo.toml)
[![CI](https://github.com/newpoluton-alt/RustTorch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/newpoluton-alt/RustTorch/actions/workflows/ci.yml)

[Quick start](#quick-start) · [Capabilities](#current-capabilities) ·
[Runtime setup](#runtime-setup) · [Documentation](#documentation) ·
[Contributing](#contributing) · [License](#license)

</div>

> [!IMPORTANT]
> RustTorch is early-stage software with a deliberately small API. It does not
> provide full PyTorch parity, and the 0.x series does not promise API or graph
> format stability. Use the [canonical compatibility ledger](compat/pytorch_api.toml)
> and its [generated coverage page](docs/api-coverage.md) to evaluate
> implemented scope.

RustTorch offers fallible Rust APIs for common eager model code while LibTorch
owns tensor storage, kernels, automatic differentiation, and backend execution.
The project credo guides API and implementation choices; it is not an
unqualified performance claim.

## Quick start

`rusttorch-cli` is not published yet, so install both packages from the current
Git source:

```sh
cargo install --git https://github.com/newpoluton-alt/RustTorch rusttorch-cli
cargo new rusttorch-demo
cd rusttorch-demo
cargo add rusttorch --git https://github.com/newpoluton-alt/RustTorch
rusttorch setup --backend auto
```

Replace `src/main.rs` with this eager example:

```rust
use rusttorch::nn::Sequential;
use rusttorch::{DeviceSpec, Kind, Result, Tensor};

fn main() -> Result<()> {
    let model = Sequential::builder()
        .linear(2, 4)
        .relu()
        .linear(4, 1)
        .build(DeviceSpec::Auto)?;
    let input = Tensor::f_zeros([8, 2], (Kind::Float, model.device()))?;
    let output = model.forward(&input)?;
    assert_eq!(output.size(), [8, 1]);
    Ok(())
}
```

Then run it:

```sh
cargo run
```

The first setup may download a large official LibTorch artifact into Cargo
build storage. RustTorch links LibTorch dynamically, so the platform loader
must also be able to find its shared libraries at runtime.

## Current capabilities

| Area | Implemented scope |
|---|---|
| Eager models | `Linear`, `Identity`, ReLU, GELU, Dropout, Flatten, and `Sequential` |
| Training | LibTorch autograd, MSE and cross-entropy losses, Adam, and SGD |
| Devices | Explicit CPU, CUDA, and MPS requests plus checked automatic selection |
| State interchange | Strict, non-strict, mapped, and dry-run SafeTensors loading |
| Graphs | Optional named-input graph API with branching, validation, summaries, and DOT output |
| Data loading | Fallible map datasets and streams, seeded local sampling, batching, and custom collation |
| Runtime | Project-local managed CPU or CUDA 12.6 setup over official LibTorch artifacts |

The machine-readable [compatibility ledger](compat/pytorch_api.toml) is the
canonical inventory; [API coverage](docs/api-coverage.md) is its generated
human-readable view. Every entry has an exact scope and one of `supported`,
`partial`, `planned`, `python_only`, or `not_supported`. `supported` applies
only to the written scope. Rows cover differently sized capabilities, so their
count is not converted into a support percentage.

Backend availability depends on the linked LibTorch build and host; explicit
unavailable device requests return errors rather than silently falling back.

## Runtime setup

Run one of the three supported commands inside a Cargo project:

```sh
rusttorch setup --backend auto
rusttorch setup --backend cpu
rusttorch setup --backend cuda-12.6
```

- `auto` preserves an active Python, system LibTorch, or CUDA selector.
  Otherwise it chooses CUDA 12.6 on a compatible Linux or Windows NVIDIA host
  and CPU on the remaining supported hosts.
- `cpu` selects the managed CPU distribution. On supported macOS systems that
  distribution can expose MPS.
- `cuda-12.6` selects `cu126` on Linux or Windows after checking the NVIDIA
  driver. It never installs or changes drivers or CUDA toolkits.

Setup locates the Cargo workspace root, keeps managed CPU and CUDA artifacts
separate, writes project-local Cargo settings, and runs `cargo check`. Existing
Python, system, and offline LibTorch workflows remain available. See
[platform setup](docs/platform-support.md) and
[CUDA support](docs/cuda-support.md) for selectors, driver floors, target
isolation, retry behavior, and dynamic-loader requirements.

## Documentation

| Guide | What it covers |
|---|---|
| [API documentation](https://docs.rs/rusttorch) | Public Rust types and functions |
| [Compatibility ledger](compat/pytorch_api.toml) | Canonical machine-readable scopes and evidence |
| [Compatibility coverage](docs/api-coverage.md) | Generated status view of PyTorch API areas |
| [Architecture](docs/architecture.md) | Eager frontend and LibTorch boundary |
| [Platform support](docs/platform-support.md) | Runtime, devices, and system/Python setup |
| [Backend evidence](docs/backend-parity.md) | Hardware-specific validation and parity scope |
| [Graph system](docs/graph-system.md) | Optional graph construction, validation, and execution |
| [Model interoperability](docs/model-interoperability.md) | SafeTensors exchange with PyTorch |
| [Porting policy](docs/porting-policy.md) | Source attribution and compatibility rules |
| [Release provenance](docs/releasing.md) | Exact package subjects and verification procedure |
| [Governance](GOVERNANCE.md) | Roles, decisions, security, and release authority |
| [Support](SUPPORT.md) | Usage questions and issue routing |
| [Security](SECURITY.md) | Supported versions and private vulnerability reporting |

## Limitations

RustTorch currently has no convolutional, recurrent, or transformer module
surface; distributed training; quantization; replacement autograd;
`torch.compile`; or custom-kernel framework. SafeTensors is the supported
model-state format. Python pickle models, TorchScript, `torch.export`, and
cross-language optimizer checkpoint resume are not exposed by the current API.
Data loading is currently single-threaded; workers, prefetch, pinned memory,
distributed sampling, and loader checkpoint/resume remain planned.

## Contributing

Contributions are welcome. Start with the
[contribution guide](CONTRIBUTING.md), keep claims within tested scope, and add
the narrowest evidence that proves a change. Report vulnerabilities through
the project's [security policy](SECURITY.md), never a public issue.

## License

Original RustTorch code is available under either the [MIT License](LICENSE-MIT)
or the [Apache License 2.0](LICENSE-APACHE), at your option. PyTorch, LibTorch,
`tch`, and other dependencies retain their own licenses and attribution; see
[third-party notices](THIRD_PARTY_NOTICES.md).

RustTorch is not affiliated with or endorsed by the PyTorch Foundation.
