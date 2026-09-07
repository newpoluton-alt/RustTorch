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

The normal crates.io flow is:

```sh
cargo install rusttorch-cli
cargo add rusttorch
rusttorch setup --backend auto
cargo run
```

`rusttorch setup` locates the Cargo workspace root, safely edits the active
root `.cargo/config` or `.cargo/config.toml` for a managed backend, then runs
`cargo check`. It is project-local bootstrap, not a global installer. Existing
unrelated Cargo settings are preserved, and a failed check leaves the managed
configuration available for a retry. Preconfigured mode writes no config.

Use `rusttorch setup --backend cpu` to force the CPU distribution, or
`rusttorch setup --backend cuda-12.6` on Linux/Windows to force CUDA 12.6.
macOS downloads CPU LibTorch; that distribution can use MPS when the host and
LibTorch build support it.

`auto` preserves active `LIBTORCH_USE_PYTORCH`, `LIBTORCH`,
`LIBTORCH_INCLUDE`, `LIBTORCH_LIB`, or nonempty `TORCH_CUDA_VERSION` selectors,
plus Linux `/usr/lib/libtorch.so`. Without one, it chooses CUDA 12.6 only when
a Linux/Windows NVIDIA driver meets the compatibility floor; otherwise it
chooses CPU on supported hosts. Driver probing selects an artifact only.
Hardware tests, not the probe, prove that a backend executes correctly.

Setup reads selectors from the workspace root's applicable Cargo configuration,
its ancestors, and `CARGO_HOME` (defaulting to the user's `.cargo` directory).
Legacy `config` takes precedence over `config.toml`; Cargo environment tables
merge by field and honor process environment and `force` precedence. Explicit
backends reject an existing runtime before driver probing or configuration writes.
The CLI recognizes its own managed CUDA selector when switching backends.
Config includes, conflicting string/table definitions, and inactive user-owned
CUDA entries fail closed with instructions to resolve the configuration first.

Managed CPU and CUDA use separate `target/rusttorch/cpu` and
`target/rusttorch/cuda-12.6` directories. Managed CUDA forces `cu126`; ordinary
CPU builds must keep `TORCH_CUDA_VERSION` unset. Setup rejects a present
`CARGO_TARGET_DIR` or `CARGO_BUILD_TARGET_DIR` before a managed write. Later
raw Cargo `--target-dir` and environment overrides can bypass the isolation.

For Python, system, or offline LibTorch, use
`rusttorch = { version = "0.1.0", default-features = false }` and set either
`LIBTORCH_USE_PYTORCH=1` or `LIBTORCH=/absolute/path/to/libtorch` while
building. Cargo `--offline` works once the dependency and compatible LibTorch
installation are local. The platform dynamic loader must still find the
selected installation's shared libraries when the application runs.

On Windows with the MSVC target and pinned torch 2.13 / torch-sys 0.26 combination,
use LLVM's `clang-cl` with C++20 and exception handling: the PyTorch headers use
C++20 constructs, while torch-sys defaults to C++17. MSVC's C++20 parser also
rejects the binding's legacy `module` typedef. With LLVM and the Visual Studio
C++ build tools installed, set these in PowerShell:

```powershell
$env:CXX = "clang-cl"
$env:CXXFLAGS = "$env:CXXFLAGS /std:c++20 /EHsc"
```

CI applies these settings for Windows runtime tests. LLVM documents the
[MSVC-compatible Clang driver](https://clang.llvm.org/docs/UsersManual.html#clang-cl).
When using the Python
backend, activate `.venv` so Cargo's subprocesses use its Python executable,
and add the installed `torch/lib` directory to `PATH` for DLL loading.

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
