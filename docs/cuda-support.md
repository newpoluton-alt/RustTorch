# CUDA support

RustTorch CUDA execution is LibTorch CUDA execution. The crate contains no CUDA
kernels. Setup never installs or modifies an NVIDIA driver or CUDA toolkit.

The managed command is:

```sh
rusttorch setup --backend cuda-12.6
```

This exact backend maps to the official `cu126` LibTorch artifact and forces
`TORCH_CUDA_VERSION=cu126` in project configuration. The minimum detected
driver is 525.60.13 on Linux and 528.33.0 on Windows. `auto` chooses this
backend only when `nvidia-smi` reports a compatible driver; an older, missing,
or unparseable driver makes `auto` choose CPU. Explicit CUDA setup returns an
error instead. The probe is selection only: the Rust hardware tests are the
evidence that CUDA execution actually works.

Requirements:

- Linux or Windows on a supported NVIDIA system;
- a CUDA-enabled PyTorch/LibTorch build compatible with `tch` 0.26.0;
- a compatible NVIDIA driver/runtime;
- a requested device index smaller than the reported device count.

`DeviceSpec::Cuda(index)` fails clearly when those conditions are not met.
`DeviceSpec::Auto` chooses CUDA:0 only after availability is verified. Model
parameters, inputs, outputs, losses, and gradients stay on CUDA; implicit CPU
fallback is not enabled.

Managed CUDA artifacts use `target/rusttorch/cuda-12.6`, separate from the
managed CPU target. Setup rejects `CARGO_TARGET_DIR` and
`CARGO_BUILD_TARGET_DIR`; later raw Cargo target-directory overrides can still
bypass this isolation. An already active LibTorch/Python/TORCH selector is
preserved by `auto` and rejected by explicit CUDA setup. RustTorch does not
promise that acquisition occurs exactly once, and the operating-system loader
must be able to find the selected LibTorch shared libraries at runtime.

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
