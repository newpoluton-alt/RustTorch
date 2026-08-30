# Managed LibTorch Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a normal RustTorch dependency acquire the pinned official LibTorch runtime automatically and provide the three approved `rusttorch setup` commands without requiring LibTorch to build the setup binary.

**Architecture:** The root crate forwards a default Cargo feature to the existing `tch` downloader. A separate lightweight workspace package builds the `rusttorch` bootstrap binary without `tch`, chooses a concrete backend, safely persists project-local Cargo settings, and runs Cargo so the upstream build script performs acquisition.

**Tech Stack:** Rust 1.88, Cargo workspaces/features, `tch`/`torch-sys` 0.26.0, `toml_edit`, standard-library process and platform APIs.

**Spec:** `docs/superpowers/specs/2026-08-30-pytorch-compatibility-program.md`

## Global Constraints

- Product credo: “easy to use and easy to implement, but crazy fast.”
- PyTorch/LibTorch is exactly 2.13.0 through `tch` 0.26.0.
- `rusttorch setup` must build before LibTorch exists and therefore must not depend on `tch` or `rusttorch`.
- `cuda-12.6` is Linux/Windows only and maps exactly to `TORCH_CUDA_VERSION=cu126`.
- Setup never installs or modifies an NVIDIA driver.
- CPU and CUDA builds use different target directories.
- Existing unrelated Cargo configuration must be preserved.
- docs.rs must remain network-free.

---

### Task 1: Forward automatic LibTorch acquisition

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`

**Interfaces:**
- Produces: root feature `download-libtorch = ["tch/download-libtorch"]` enabled by default.
- Produces: workspace member `crates/rusttorch-cli` with root as the default member.
- Produces: docs.rs build with `no-default-features = true` and `features = ["tch/doc-only"]`.

- [ ] **Step 1: Record the failing metadata assertion**

Run this before editing:

```sh
cargo metadata --no-deps --format-version 1
```

Verify the root package has no `download-libtorch` feature and no workspace CLI member.

- [ ] **Step 2: Add the workspace and feature configuration**

Add these root manifest sections and exclude the companion package from the root `.crate` archive:

```toml
[workspace]
members = ["crates/rusttorch-cli"]
default-members = ["."]
resolver = "2"

[features]
default = ["download-libtorch"]
download-libtorch = ["tch/download-libtorch"]

[package.metadata.docs.rs]
no-default-features = true
features = ["tch/doc-only"]
```

Add `"/crates/rusttorch-cli"` to the existing package `exclude` list.

- [ ] **Step 3: Verify metadata and lock resolution**

Run:

```sh
cargo metadata --no-deps --format-version 1
cargo check -p rusttorch --all-targets --features tch/doc-only
```

Expected: metadata lists the forwarded/default feature and the network-free check succeeds.

- [ ] **Step 4: Commit**

```sh
git add Cargo.toml Cargo.lock
git commit -m "feat: acquire LibTorch through a default feature"
```

### Task 2: Parse setup requests and resolve a backend

**Files:**
- Create: `crates/rusttorch-cli/Cargo.toml`
- Create: `crates/rusttorch-cli/src/main.rs`

**Interfaces:**
- Produces: package `rusttorch-cli`, binary `rusttorch`.
- Produces: `BackendRequest::{Auto, Cpu, Cuda126}` parsed from the exact accepted strings.
- Produces: `ResolvedBackend::{Preconfigured, Cpu, Cuda126}`.
- Produces: pure `resolve_backend(request, platform, configured, driver)` logic used by tests and `main`.

- [ ] **Step 1: Write parser and backend-resolution tests**

Add inline tests covering these assertions:

```rust
assert_eq!(parse_backend("auto").unwrap(), BackendRequest::Auto);
assert_eq!(parse_backend("cpu").unwrap(), BackendRequest::Cpu);
assert_eq!(parse_backend("cuda-12.6").unwrap(), BackendRequest::Cuda126);
assert!(parse_backend("cuda").is_err());

assert_eq!(
    resolve_backend(BackendRequest::Auto, Platform::Macos, false, None).unwrap(),
    ResolvedBackend::Cpu,
);
assert_eq!(
    resolve_backend(
        BackendRequest::Auto,
        Platform::Linux,
        false,
        Some(DriverVersion::new(525, 60, 13)),
    )
    .unwrap(),
    ResolvedBackend::Cuda126,
);
assert!(resolve_backend(BackendRequest::Cuda126, Platform::Macos, false, None).is_err());
```

Also test Linux driver `525.59.0` and Windows driver `528.32.0` are rejected for explicit CUDA, while `auto` falls back to CPU.

- [ ] **Step 2: Run the tests and confirm the package is absent**

Run:

```sh
cargo test -p rusttorch-cli
```

Expected: FAIL because the workspace package does not exist yet.

- [ ] **Step 3: Implement the minimal package and pure selection logic**

Use this manifest shape:

```toml
[package]
name = "rusttorch-cli"
version = "0.1.0"
edition = "2024"
rust-version = "1.88"
description = "Managed LibTorch setup for RustTorch projects"
repository = "https://github.com/newpoluton-alt/RustTorch"
license = "MIT OR Apache-2.0"
publish = ["crates-io"]

[[bin]]
name = "rusttorch"
path = "src/main.rs"

[dependencies]
toml_edit = "0.22"
tempfile = "3"
```

Parse only `setup --backend <value>` plus `--help` and `--version` with
`std::env::args_os`. Detect active `LIBTORCH_USE_PYTORCH`, `LIBTORCH`,
`LIBTORCH_INCLUDE`, or `LIBTORCH_LIB`. Parse the first driver line returned by:

```text
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

CUDA 12.x minimums are `525.60.13` on Linux and `528.33.0` on Windows.

- [ ] **Step 4: Run the focused tests**

Run:

```sh
cargo test -p rusttorch-cli
```

Expected: all parser, platform, preconfiguration, and driver tests pass.

- [ ] **Step 5: Commit**

```sh
git add crates/rusttorch-cli Cargo.lock
git commit -m "feat: resolve RustTorch setup backends"
```

### Task 3: Persist backend configuration and trigger acquisition

**Files:**
- Modify: `crates/rusttorch-cli/src/main.rs`

**Interfaces:**
- Consumes: `ResolvedBackend` from Task 2.
- Produces: `configure_project(root, backend) -> Result<PathBuf, CliError>`.
- Produces: `cargo_check_spec(root, backend) -> CargoCheckSpec` for a testable command description.
- Produces: project-local `.cargo/config.toml` entries for backend-specific `build.target-dir` and CUDA selection.

- [ ] **Step 1: Write configuration-preservation tests**

Create a temporary Cargo project in each test and assert:

```rust
let original = "[alias]\nfast = \"check\"\n";
fs::write(root.join(".cargo/config.toml"), original).unwrap();
configure_project(&root, ResolvedBackend::Cuda126).unwrap();
let document = fs::read_to_string(root.join(".cargo/config.toml")).unwrap();
assert!(document.contains("fast = \"check\""));
assert!(document.contains("target/rusttorch/cuda-12.6"));
assert!(document.contains("TORCH_CUDA_VERSION"));
assert!(document.contains("cu126"));
```

Add a CPU-switch test proving the CUDA environment key is removed and the
target directory becomes `target/rusttorch/cpu`. Add a test proving an absent
`Cargo.toml` is rejected before any file is written.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```sh
cargo test -p rusttorch-cli configuration
```

Expected: FAIL because configuration functions do not exist.

- [ ] **Step 3: Implement safe TOML editing**

Use `toml_edit::DocumentMut` to preserve unrelated keys. Set:

```toml
[build]
target-dir = "target/rusttorch/cuda-12.6" # or cpu

[env]
RUSTTORCH_BACKEND = "cuda-12.6"          # or cpu
TORCH_CUDA_VERSION = "cu126"             # CUDA only
```

For CPU remove only the `TORCH_CUDA_VERSION` key previously managed by setup.
Use `tempfile::NamedTempFile` in the `.cargo` directory, flush and sync it,
then persist it over the destination so an interrupted write does not truncate
existing configuration.

- [ ] **Step 4: Implement Cargo execution**

Represent the command as data first, then execute:

```rust
Command::new("cargo")
    .arg("check")
    .current_dir(project_root)
    .status()
```

For CUDA set `TORCH_CUDA_VERSION=cu126`; for CPU remove it from the child
environment. Return an error when Cargo exits unsuccessfully. Print the
resolved backend, project configuration path, and that the first acquisition
can be large before starting Cargo.

- [ ] **Step 5: Run CLI tests and smoke-test help**

Run:

```sh
cargo test -p rusttorch-cli
cargo run -p rusttorch-cli -- --help
```

Expected: tests pass and help shows the three exact setup commands.

- [ ] **Step 6: Commit**

```sh
git add crates/rusttorch-cli/src/main.rs Cargo.lock
git commit -m "feat: configure and acquire the selected LibTorch runtime"
```

### Task 4: Document and package the setup experience

**Files:**
- Modify: `README.md`
- Modify: `docs/platform-support.md`
- Modify: `docs/cuda-support.md`
- Modify: `CONTRIBUTING.md`
- Modify: `CHANGELOG.md`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: the three CLI commands and forwarded feature.
- Produces: consumer, offline, system-LibTorch, CUDA-driver, and contributor instructions.

- [ ] **Step 1: Replace the manual-only installation section**

Document this primary flow:

```sh
cargo install rusttorch-cli
cargo add rusttorch
rusttorch setup --backend auto
cargo run
```

Document explicit CPU/CUDA commands, that macOS CPU builds expose MPS when
supported, and the `default-features = false` escape hatch for system/Python
LibTorch.

- [ ] **Step 2: Add contributor and third-party notes**

Change workspace checks to `--workspace`. Record that `torch-sys` downloads
official artifacts, CUDA drivers remain external, and the package does not
redistribute downloaded binaries inside its `.crate` archive.

- [ ] **Step 3: Run packaging checks**

Run:

```sh
cargo fmt --all -- --check
cargo test -p rusttorch-cli
cargo check -p rusttorch --all-targets --features tch/doc-only
cargo package -p rusttorch --locked --no-verify --list
cargo package -p rusttorch-cli --locked --list
```

Expected: both packages contain only their source/docs and no downloaded runtime.

- [ ] **Step 4: Commit**

```sh
git add README.md docs/platform-support.md docs/cuda-support.md CONTRIBUTING.md CHANGELOG.md THIRD_PARTY_NOTICES.md
git commit -m "docs: explain managed LibTorch setup"
```
