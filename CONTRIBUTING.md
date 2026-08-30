# Contributing

RustTorch ports high-level PyTorch behavior and delegates numerical work to
`tch`/LibTorch. Before adding code, confirm the behavior is not already exposed
by `tch` or the standard library.

## Development

```sh
. scripts/dev-env.sh
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
cargo doc --workspace --no-deps
cargo check -p rusttorch --all-targets --no-default-features --features tch/doc-only
```

Run `scripts/run-python-parity.sh` for changes affecting PyTorch defaults,
initialization, gradients, optimizers, naming, or serialization. Run backend
checks only on available hardware and report unavailable hardware as skipped.

## Change requirements

- Keep the eager API primary; do not force custom models through Graph IR.
- Reuse `tch::Tensor`, `tch::nn::VarStore`, LibTorch autograd, and backend
  kernels.
- Use structured errors at recoverable boundaries and avoid public `unsafe`.
- Preserve PyTorch-compatible state names where practical; transformations and
  mappings must be explicit.
- Add the narrowest test that fails without the behavior.
- Update `compat/pytorch_api.toml` and API coverage for public API changes.
- Cite the PyTorch source path and v2.13.0/`cf30153` when substantially adapting
  high-level logic, and update `THIRD_PARTY_NOTICES.md`.
- Do not commit virtual environments, downloaded LibTorch builds, model
  artifacts, credentials, or vendored PyTorch source.

Contributions are accepted under both the MIT and Apache-2.0 licenses.
