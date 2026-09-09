# Data Workspace Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the shared runtime contracts and existing data implementation into publishable `rusttorch-core` and `rusttorch-data` packages without changing any existing `rusttorch` public path or behavior.

**Architecture:** The root `rusttorch` package remains the facade. `rusttorch-core` owns the current error, device, tensor re-export, and LibTorch feature forwarding; `rusttorch-data` depends only on core and owns the current loader implementation. The facade re-exports both packages, and release/CI policy treats all workspace packages as one synchronized version set.

**Tech Stack:** Cargo workspaces, Rust 2024, Rust 1.88 MSRV, `tch` 0.26.0, `thiserror` 2.0, `rand` 0.8, `rand_chacha` 0.3, UV-managed Python policy tests, GitHub Actions, SLSA generic provenance.

**Spec:** `docs/superpowers/specs/2026-08-31-unified-data-ecosystem-design.md`

## Global Constraints

- Package and crate names remain lowercase: `rusttorch`, `rusttorch-core`, and `rusttorch-data`; the product name in prose remains `RustTorch`.
- Every package uses version `0.1.0`, edition `2024`, and `rust-version = "1.88"` until a separately reviewed version bump.
- `tch` remains exactly `0.26.0`; the pinned PyTorch/LibTorch target remains `2.13.0` at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Root `default = ["download-libtorch"]` behavior remains unchanged.
- Root `doc-only` forwards to `tch/doc-only`; docs.rs remains network-free with default features disabled.
- Existing `rusttorch::data`, `rusttorch::device`, `rusttorch::error`, `rusttorch::Result`, `rusttorch::RustTorchError`, tensor re-exports, and `manual_seed` remain source-compatible.
- Existing `DataLoader::new`, `DataLoader::with_collate`, `batches`, and sampler behavior remain byte-for-byte equivalent at this milestone.
- Lower packages disable dependency default features and never select a LibTorch acquisition strategy implicitly.
- Before running task commands, source `. scripts/dev-env.sh`; it selects the locked UV environment, sets `LIBTORCH_USE_PYTORCH=1`, and configures the platform library path. `doc-only` is reserved for `cargo check`, Clippy, and rustdoc because it intentionally does not link a runtime.
- Crate archives contain no native runtime, downloaded dataset, Python environment, generated model, or paid-service dependency.
- Every commit uses DCO sign-off and leaves formatting, Clippy, tests, rustdoc, package inspection, and compatibility validation green.

---

### Task 1: Extract `rusttorch-core` without changing facade paths

**Files:**
- Create: `crates/rusttorch-core/Cargo.toml`
- Create: `crates/rusttorch-core/README.md`
- Create: `crates/rusttorch-core/src/lib.rs`
- Create from current implementation: `crates/rusttorch-core/src/error.rs`
- Create from current implementation: `crates/rusttorch-core/src/device.rs`
- Create: `crates/rusttorch-core/tests/public_api.rs`
- Modify: `Cargo.toml`
- Modify: `crates/rusttorch-cli/Cargo.toml`
- Modify: `src/lib.rs`
- Replace implementation with facade shim: `src/error.rs`
- Replace implementation with facade shim: `src/device.rs`
- Modify mechanically: imports in `src/graph/mod.rs`, `src/interop/mod.rs`, `src/nn/mod.rs`, `src/nn/functional.rs`, and `src/optim.rs` only if the facade re-export does not already satisfy them

**Interfaces:**
- Produces: `rusttorch_core::{Result, RustTorchError, DeviceCapabilities, DeviceSpec, available_devices, resolve_device}` and the documented `rusttorch_core::device::ensure_device` validator.
- Produces: `rusttorch_core::{Device, Kind, Reduction, Tensor, no_grad, no_grad_guard, manual_seed}`.
- Preserves: the same symbols at every current `rusttorch` root/module path.
- Produces features: `download-libtorch -> tch/download-libtorch` and `doc-only -> tch/doc-only`, with no core default feature.

- [ ] **Step 1: Write direct-package and facade-compatibility tests**

Create `crates/rusttorch-core/tests/public_api.rs` with compile-time type use and one runtime CPU assertion:

```rust
use rusttorch_core::{Device, DeviceSpec, Kind, Result, Tensor, resolve_device};

#[test]
fn core_exports_the_shared_runtime_contract() -> Result<()> {
    let device = resolve_device(DeviceSpec::Cpu)?;
    assert_eq!(device, Device::Cpu);
    let tensor = Tensor::f_zeros([2], (Kind::Float, device))?;
    assert_eq!(tensor.size(), [2]);
    Ok(())
}
```

Add a facade test to `tests/device.rs` that assigns the same value through both paths:

```rust
let direct: rusttorch_core::DeviceSpec = rusttorch::DeviceSpec::Cpu;
let facade: rusttorch::device::DeviceSpec = direct;
assert_eq!(facade, rusttorch::DeviceSpec::Cpu);
```

- [ ] **Step 2: Run the focused tests and verify the package is absent**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-core --test public_api
```

Expected: Cargo fails because package `rusttorch-core` does not exist.

- [ ] **Step 3: Add synchronized workspace metadata and the core manifest**

Add these root workspace tables and dependencies, retaining the root package metadata and benchmark:

```toml
[workspace]
members = ["crates/*"]
default-members = ["."]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.88"
repository = "https://github.com/newpoluton-alt/RustTorch"
license = "MIT OR Apache-2.0"

[workspace.dependencies]
tch = { version = "=0.26.0", default-features = false }
thiserror = "2.0"
rand = "0.8"
rand_chacha = "0.3"
rusttorch-core = { version = "=0.1.0", path = "crates/rusttorch-core", default-features = false }
```

Replace the root and CLI package's duplicated version/edition/MSRV/repository/
license values with workspace inheritance while retaining their descriptions,
readmes, publish settings, keywords, categories, and targets:

```toml
version.workspace = true
edition.workspace = true
rust-version.workspace = true
repository.workspace = true
license.workspace = true
```

Create `crates/rusttorch-core/Cargo.toml`:

```toml
[package]
name = "rusttorch-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
repository.workspace = true
license.workspace = true
description = "Shared runtime, device, tensor, and error contracts for RustTorch"
readme = "README.md"
publish = ["crates-io"]

[features]
default = []
download-libtorch = ["tch/download-libtorch"]
doc-only = ["tch/doc-only"]

[dependencies]
tch.workspace = true
thiserror.workspace = true

[package.metadata.docs.rs]
no-default-features = true
features = ["doc-only"]
```

Change the root feature/dependency forwarding to:

```toml
[features]
default = ["download-libtorch"]
download-libtorch = ["rusttorch-core/download-libtorch"]
doc-only = ["rusttorch-core/doc-only"]

[dependencies]
rusttorch-core.workspace = true
tch.workspace = true
```

Replace root docs.rs metadata with the facade feature name rather than a
dependency-feature path:

```toml
[package.metadata.docs.rs]
no-default-features = true
features = ["doc-only"]
```

- [ ] **Step 4: Move the shared implementation and replace facade definitions with re-exports**

Move the complete current contents of `src/error.rs` and `src/device.rs` into the matching core files, changing only crate-relative imports. Make `ensure_device` public, give it complete rustdoc for its context/tensor/expected-device contract and mismatch error, and treat it as a supported API instead of hiding an undocumented cross-crate symbol. Create `crates/rusttorch-core/src/lib.rs`:

```rust
//! Shared runtime contracts for the RustTorch workspace.

#![deny(missing_docs)]

pub mod device;
pub mod error;

pub use device::{DeviceCapabilities, DeviceSpec, available_devices, resolve_device};
pub use error::{Result, RustTorchError};
pub use tch::{Device, Kind, Reduction, Tensor, no_grad, no_grad_guard};

/// Seeds LibTorch's random number generator.
pub fn manual_seed(seed: i64) {
    tch::manual_seed(seed);
}
```

Keep the root module declarations. Replace `src/error.rs` with:

```rust
//! Errors returned by the fallible RustTorch API.

pub use rusttorch_core::error::{Result, RustTorchError};
```

Replace `src/device.rs` with:

```rust
//! Runtime device discovery and strict device selection.

pub use rusttorch_core::device::{
    DeviceCapabilities, DeviceSpec, available_devices, ensure_device, resolve_device,
};
```

Keep the current root re-exports from the `device` and `error` facade modules.
Replace only the direct `tch` tensor re-export and root `manual_seed` function
with:

```rust
pub use rusttorch_core::{
    Device, Kind, Reduction, Tensor, manual_seed, no_grad, no_grad_guard,
};
```

- [ ] **Step 5: Run core, facade, feature, and documentation checks**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-core --all-targets
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch --all-targets
cargo check -p rusttorch --all-targets --locked
RUSTDOCFLAGS="-D warnings" cargo doc -p rusttorch-core --no-deps --no-default-features --features doc-only
RUSTDOCFLAGS="-D warnings" cargo doc -p rusttorch --no-deps --no-default-features --features doc-only
```

Expected: all tests pass, both explicit feature paths compile, and rustdoc reports no warnings.

- [ ] **Step 6: Commit**

```sh
git add Cargo.toml Cargo.lock src crates/rusttorch-core tests/device.rs
git commit -s -m "refactor: extract RustTorch core contracts"
```

### Task 2: Extract `rusttorch-data` and preserve every current data path

**Files:**
- Create: `crates/rusttorch-data/Cargo.toml`
- Create: `crates/rusttorch-data/README.md`
- Create: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/current_api.rs`
- Replace implementation with facade shim: `src/data.rs`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify rustdoc imports only: current data examples moved into `crates/rusttorch-data/src/lib.rs`

**Interfaces:**
- Produces the existing data surface directly as `rusttorch_data::*`.
- Preserves the same surface as `rusttorch::data::*` through a one-line facade re-export.
- Keeps construction errors as `rusttorch_core::RustTorchError` and iterator errors as the original dataset/collator error type.
- Preserves the exact public generic `DataLoader<'a, D, S, C, B, E>` type and constructors.

- [ ] **Step 1: Write direct/facade identity tests before moving code**

Create `crates/rusttorch-data/tests/current_api.rs` with a minimal dataset and verify direct-package behavior:

```rust
use std::convert::Infallible;
use rusttorch_data::{DataLoader, Dataset, SequentialSampler};

struct Rows([usize; 3]);

impl Dataset for Rows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize { self.0.len() }
    fn get(&self, index: usize) -> Result<usize, Infallible> { Ok(self.0[index]) }
}

#[test]
fn direct_package_matches_the_existing_facade_loader() {
    let rows = Rows([2, 3, 5]);
    let batches = DataLoader::new(&rows, SequentialSampler::new(3), 2, false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches, vec![vec![2, 3], vec![5]]);
}
```

Add this type-identity assertion to the root `tests/data.rs`:

```rust
let direct = rusttorch_data::SequentialSampler::new(2).collect::<Vec<_>>();
let facade = rusttorch::data::SequentialSampler::new(2).collect::<Vec<_>>();
assert_eq!(direct, facade);
```

- [ ] **Step 2: Run the direct-package test and verify failure**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test current_api
```

Expected: Cargo fails because package `rusttorch-data` does not exist.

- [ ] **Step 3: Add the data package manifest and facade dependency**

Add to `[workspace.dependencies]`:

```toml
rusttorch-data = { version = "=0.1.0", path = "crates/rusttorch-data", default-features = false }
```

Create `crates/rusttorch-data/Cargo.toml`:

```toml
[package]
name = "rusttorch-data"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
repository.workspace = true
license.workspace = true
description = "Typed datasets, samplers, batching, and loading for RustTorch"
readme = "README.md"
publish = ["crates-io"]

[features]
default = []
download-libtorch = ["rusttorch-core/download-libtorch"]
doc-only = ["rusttorch-core/doc-only"]

[dependencies]
rand.workspace = true
rand_chacha.workspace = true
rusttorch-core.workspace = true

[dev-dependencies]
tch.workspace = true

[package.metadata.docs.rs]
no-default-features = true
features = ["doc-only"]
```

Add `rusttorch-data.workspace = true` to the root dependencies. Root acquisition remains owned by `rusttorch-core`; do not add `rusttorch-data/download-libtorch` to root defaults.

- [ ] **Step 4: Move the implementation and install the facade shim**

Move the complete current `src/data.rs` implementation into `crates/rusttorch-data/src/lib.rs`. Make only these semantic-neutral changes:

```rust
//! Typed datasets, samplers, batching, and loading for RustTorch.

#![deny(missing_docs)]

use rusttorch_core::{Result, RustTorchError};
```

Change rustdoc imports from `rusttorch::data` to `rusttorch_data` and tensor/result imports to `rusttorch_core`. Replace root `src/data.rs` with:

```rust
//! Data APIs re-exported from [`rusttorch_data`].

pub use rusttorch_data::*;
```

Do not retain a second copy of `Dataset`, samplers, batching, or loader code in the facade.

- [ ] **Step 5: Prove source compatibility and direct-package behavior**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --all-targets
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch --test data
cargo test -p rusttorch --test data_libtorch
cargo bench -p rusttorch --bench data_loader --no-run
RUSTDOCFLAGS="-D warnings" cargo doc -p rusttorch-data --no-deps --no-default-features --features doc-only
```

Expected: all 23 existing data tests, the LibTorch RNG-isolation test, the direct-package test, benchmark compilation, and rustdoc pass unchanged. The crate-level `missing_docs` denial makes any undocumented public data item fail this task and every later rustdoc gate.

- [ ] **Step 6: Commit**

```sh
git add Cargo.toml Cargo.lock src/data.rs crates/rusttorch-data tests/data.rs
git commit -s -m "refactor: extract RustTorch data package"
```

### Task 3: Make CI, package policy, and release provenance workspace-aware

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `scripts/check-release.py`
- Modify: `tests/test_community_health.py`
- Modify: `tests/test_release_workflow.py`
- Modify: `Cargo.lock`

**Interfaces:**
- Produces release package order: `rusttorch-core`, `rusttorch-data`, `rusttorch-cli`, `rusttorch`.
- Produces one `.crate` archive and one SLSA subject for each package.
- Extends stable/MSRV/doc-only/package-inspection gates to the two new packages.
- Does not add automatic crates.io publication or any paid GitHub feature.

- [ ] **Step 1: Write failing release-policy tests for four packages**

Change test fixtures and expectations so the canonical manifest tuple is:

```python
PACKAGE_MANIFESTS = (
    ("rusttorch-core", Path("crates/rusttorch-core/Cargo.toml")),
    ("rusttorch-data", Path("crates/rusttorch-data/Cargo.toml")),
    ("rusttorch-cli", Path("crates/rusttorch-cli/Cargo.toml")),
    ("rusttorch", Path("Cargo.toml")),
)
```

Assert all four exact archives occur once in `subjects.txt`, one `cargo package --workspace --locked` command occurs in the build job, and the release job accepts no extra or missing archive. Add fixtures proving workspace-inherited member versions resolve to `0.1.0` and malformed, missing, or conflicting inherited versions fail closed.
Extend `tests/test_community_health.py` to assert every publishable library crate
root contains `#![deny(missing_docs)]`, including `rusttorch-core` and
`rusttorch-data`, so removing the documentation policy fails ordinary CI.

- [ ] **Step 2: Run policy tests and verify the two-package assumptions fail**

Run:

```sh
.venv/bin/python -m unittest tests.test_release_workflow tests.test_community_health -v
```

Expected: failures identify missing `rusttorch-core` and `rusttorch-data` package, documentation, MSRV, archive, or subject entries.

- [ ] **Step 3: Update the fail-closed release checker**

Set `PACKAGE_MANIFESTS` to the four-package tuple above. Derive archive names from that tuple in `write_subjects`:

```python
expected_names = [f"{name}-{version}.crate" for name, _ in PACKAGE_MANIFESTS]
```

Resolve a member `package.version = { workspace = true }` from the root
`workspace.package.version`; continue accepting the root/CLI's literal version
during this migration. Reject any other table shape. Keep the existing archive
path, symlink, type, content, checksum, and exact-directory validation
unchanged. The lockfile validator must continue requiring exactly one
matching-version entry per named package.

- [ ] **Step 4: Extend CI and release archive construction**

In stable CI, add rustdoc and archive listing for both new packages using their
`doc-only` feature. Keep individual `cargo package -p NAME --locked --list`
commands only for content inspection. Remove every individual non-`--list`
package-verification command and replace them with exactly one
`cargo package --workspace --locked`, because interdependent exact-version path
dependencies are not yet published. Add a policy test that rejects any
individual non-`--list` `cargo package -p` command in stable/release CI. Extend
the MSRV matrix to:

```yaml
package:
  - rusttorch-core
  - rusttorch-data
  - rusttorch
  - rusttorch-cli
```

Replace the current package-specific `rusttorch`/`rusttorch-cli` conditions with
two exhaustive steps so every matrix value executes a check:

```yaml
- name: Check library package on the MSRV
  if: matrix.package != 'rusttorch-cli'
  run: >-
    cargo check -p ${{ matrix.package }} --all-targets --locked
    --no-default-features --features doc-only

- name: Check CLI package on the MSRV
  if: matrix.package == 'rusttorch-cli'
  run: cargo check -p rusttorch-cli --all-targets --locked
```

This intentionally uses each library package's public `doc-only` feature, not
the dependency path `tch/doc-only`. Add policy tests that enumerate all four
matrix values and prove each selects exactly one command.

In the release build job, ask Cargo to package the interdependent workspace exactly once:

```sh
cargo package --workspace --locked
```

Cargo resolves and verifies path-plus-version dependencies together in workspace mode. Copy all four exact archives into `dist` in dependency order (`core`, `data`, `cli`, `facade`), upload all four to the immutable build artifact, validate all four after download, and include all four in the GitHub release expected-file set. Preserve pinned action SHAs, least-privilege permissions, seven-day artifact retention, and the existing SLSA reusable workflow.

`cargo package --workspace` requires Cargo 1.90 or newer. Keep packaging on the
stable release job and document that this maintainer-tool requirement is
separate from the crates' Rust 1.88 compiler MSRV; the MSRV job runs `check`
only.

- [ ] **Step 5: Run policy and package checks**

Run:

```sh
.venv/bin/python -m unittest tests.test_release_workflow tests.test_community_health -v
cargo package -p rusttorch-core --locked --list
cargo package -p rusttorch-data --locked --list
cargo package -p rusttorch-cli --locked --list
cargo package -p rusttorch --locked --list
cargo package --workspace --locked
```

Expected: policy tests pass and none of the four lists contains `.venv`, native libraries, `target`, Python bytecode, downloaded models, or datasets.

- [ ] **Step 6: Commit**

```sh
git add .github/workflows scripts/check-release.py tests/test_community_health.py tests/test_release_workflow.py Cargo.lock
git commit -s -m "ci: validate the RustTorch data workspace"
```

### Task 4: Publish package documentation and run the complete foundation gate

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/pytorch-compatibility.md`
- Modify: `docs/releasing.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `compat/pytorch_api.toml`
- Modify: `scripts/check-compatibility.py`
- Modify: `tests/test_compatibility_script.py`
- Create generated: `crates/rusttorch-core/COMPATIBILITY.md`
- Create generated: `crates/rusttorch-data/COMPATIBILITY.md`
- Modify: `crates/rusttorch-core/src/lib.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Regenerate: `docs/api-coverage.md`

**Interfaces:**
- Documents direct `rusttorch_data` and facade `rusttorch::data` imports.
- Documents that extraction changes package ownership, not the supported loader scope.
- Leaves worker/prefetch/pinning/distributed/checkpoint rows as `planned` until their executable slices land.

- [ ] **Step 1: Add compiling direct and facade examples**

Use the same three-row dataset in both package READMEs and show these imports explicitly:

```rust
use rusttorch_data::{DataLoader, Dataset, SequentialSampler};
```

```rust
use rusttorch::data::{DataLoader, Dataset, SequentialSampler};
```

State that the facade is the seamless default and the direct package is for consumers that want the data layer separately.

- [ ] **Step 2: Update architecture and release documentation**

Document the dependency direction exactly as:

```text
rusttorch -> rusttorch-data -> rusttorch-core -> tch
rusttorch -------------------------------> tch
```

Document synchronized versions and the package order `core`, `data`, `cli`, `facade`. Keep the current supported DataLoader scope unchanged and link the complete-loader plan rather than claiming workers exist.

- [ ] **Step 3: Regenerate compatibility documentation**

Add checker fixtures for workspace-inherited dependency metadata. Resolve root
`dependencies.tch = { workspace = true }` through
`workspace.dependencies.tch.version`, require the raw Cargo requirement to be
exactly `=0.26.0` (then normalize the reported compatibility version to
`0.26.0`), and
resolve root `package.version = { workspace = true }` through
`workspace.package.version`. Reject missing, conflicting, or non-string
workspace values. Update only
symbol ownership/source notes required by extraction and retain existing
statuses/evidence. Extend the deterministic renderer to write a filtered
`COMPATIBILITY.md` for each direct package, and include each file from its
crate root with `#![doc = include_str!("../COMPATIBILITY.md")]`. Then run:

```sh
.venv/bin/python -m unittest tests.test_compatibility_script -v
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

Expected: all three generated compatibility documents are current and no
planned loader row becomes supported.

- [ ] **Step 4: Run the complete local CI-equivalent gate**

Run with the repository's UV-managed `.venv` and the platform's PyTorch library path exported:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
scripts/run-python-parity.sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

Expected: every command passes and `git status --short` contains only the intended documentation changes.

- [ ] **Step 5: Commit**

```sh
git add README.md docs THIRD_PARTY_NOTICES.md compat/pytorch_api.toml scripts/check-compatibility.py tests/test_compatibility_script.py crates/rusttorch-core crates/rusttorch-data
git commit -s -m "docs: introduce the RustTorch data workspace"
```
