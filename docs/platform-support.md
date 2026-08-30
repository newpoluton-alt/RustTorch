# Platform support

RustTorch uses the backend capabilities of the linked PyTorch/LibTorch build.
Backend availability is a runtime fact, not a crate feature claim.

| Platform | Backend | Status rule |
|---|---|---|
| macOS arm64 | CPU | passed on the development host |
| compatible Apple macOS | MPS | passed on the development host; verify each host/build |
| Linux/Windows NVIDIA | CUDA | conditional; requires CUDA LibTorch and driver |
| other supported LibTorch hosts | CPU | expected; verify locally |

The development host is macOS 26.5.2 arm64. Its project Python environment is
Python 3.14.7, torch 2.13.0, safetensors 0.8.0, and NumPy 2.5.2. Python reports
MPS built and available and CUDA unavailable. Independent Rust tests passed on
CPU and MPS, including eager and graph forward/backward, gradients, Adam, SGD,
model movement, and SafeTensors transfer. CUDA was unavailable and was skipped;
it has not passed on this host.

## Application setup

Applications consuming RustTorch from crates.io must provide a matching
PyTorch/LibTorch 2.13.0 installation. Either activate a Python environment with
PyTorch 2.13.0 and set `LIBTORCH_USE_PYTORCH=1`, or set `LIBTORCH` to an
absolute standalone LibTorch installation. Ensure the platform dynamic loader
can find that installation's shared libraries when the application runs.

## Contributor environment entry points

From the repository root, source one script:

```sh
. scripts/dev-env.sh              # dispatch by host OS
. scripts/dev-env-macos.sh
. scripts/dev-env-linux-cpu.sh
. scripts/dev-env-linux-cuda.sh
```

These repository-only helpers require `.venv`, verify torch 2.13.0, select it
with `LIBTORCH_USE_PYTORCH=1`, and derive library paths from the installed
package. They do not modify global Python or shell configuration. The CUDA
script does not install or pin a CUDA toolkit.

`scripts/check-backends.sh` performs small Python/LibTorch probes. Rust tests
are the authoritative proof that RustTorch itself works on a backend.

Unavailable hardware tests must print a skip reason. A skipped backend is
neither passed nor failed.
