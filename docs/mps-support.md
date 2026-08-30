# MPS support

MPS is the Metal Performance Shaders backend supplied by PyTorch/LibTorch on
compatible Apple systems. RustTorch adds no custom Metal kernels.

The development host is macOS 26.5.2 arm64. Its Python torch 2.13.0 reports MPS
built and available. Independent Rust tests successfully create and operate on
MPS tensors through the linked LibTorch.

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

On the development host, forward/backward, gradients, one Adam and SGD step,
SafeTensors CPU↔MPS transfer, and model movement pass. CUDA is unavailable and
is reported as skipped. MPS hardware tests are serialized because concurrent
LibTorch MPS test execution was unstable on this host.

Activate and inspect the local setup with:

```sh
. scripts/dev-env-macos.sh
scripts/check-backends.sh
```

The Python probe is useful setup evidence; only Rust execution proves RustTorch
MPS support.
