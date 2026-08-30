# CUDA support

RustTorch CUDA execution is LibTorch CUDA execution. The crate contains no CUDA
kernels and does not pin or install a CUDA toolkit.

Requirements:

- Linux or Windows on a supported NVIDIA system;
- a CUDA-enabled PyTorch/LibTorch build compatible with `tch` 0.26.0;
- a compatible NVIDIA driver/runtime;
- a requested device index smaller than the reported device count.

`DeviceSpec::Cuda(index)` fails clearly when those conditions are not met.
`DeviceSpec::Auto` chooses CUDA:0 only after availability is verified. Model
parameters, inputs, outputs, losses, and gradients stay on CUDA; implicit CPU
fallback is not enabled.

The CUDA validation matrix conditionally covers tensor creation, eager and
Graph IR forward/backward, Adam, SGD, device mismatch, output/gradient device,
and CPU↔CUDA weight loading. CPU is the numerical reference and deterministic
assigned weights avoid relying on matching backend RNG streams.

The macOS arm64 development host has no CUDA. CUDA tests there must report a
skip reason. CUDA support is not “passed” until the same Rust tests execute on
a suitable NVIDIA host.

Activate a prepared Linux environment with:

```sh
. scripts/dev-env-linux-cuda.sh
scripts/check-backends.sh
```
