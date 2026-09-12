# MPS support

MPS is the Metal Performance Shaders backend supplied by PyTorch/LibTorch on
compatible Apple systems. RustTorch adds no custom Metal kernels.

Use MPS to run a model on the GPU of a supported Apple system. Create the model
with `DeviceSpec::Mps`, then construct or move input tensors onto
`model.device()`. Use `DeviceSpec::Auto` when CPU fallback is appropriate for
your application.

Capability detection prefers a safe availability API from the pinned `tch`.
Where no direct helper exists, RustTorch performs and caches one tiny fallible
operation on `Device::Mps`. Backend failure reports unavailable without panic,
large allocation, or Python runtime invocation.

`DeviceSpec::Mps` is strict: unavailable MPS and unsupported operations are
errors, not silent CPU fallback. Model parameters, inputs, outputs, and
gradients remain on MPS unless movement is explicit. RustTorch 0.1 has no
dedicated persistent-buffer API.

Conditional tests cover tensor creation, eager and graph forward/backward,
Linear, ReLU, GELU, residual Add, cross-entropy, MSE, Adam, SGD, SafeTensors,
CPU↔MPS movement, mismatch errors, and output/gradient device. Deterministic
weights and inputs are compared with CPU using documented tolerances.

An earlier macOS 26.5.2 arm64 development run passed forward/backward,
gradients, one Adam and SGD step,
SafeTensors CPU↔MPS transfer, and model movement. CUDA was skipped. MPS
hardware tests are serialized because concurrent LibTorch MPS test execution
was unstable on that host. The 2026-09-12 macOS 26.6.2 validation run passed
the four existing Rust CPU/MPS backend test cases with host hardware access.
New layer and optimizer families have CPU evidence only; see
[backend evidence](backend-parity.md).

Activate and inspect the local setup with:

```sh
. scripts/dev-env-macos.sh
scripts/check-backends.sh
```

The Python probe is useful setup evidence; only Rust execution proves RustTorch
MPS support.
