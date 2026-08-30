# RustTorch 0.2 Foundation Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate, verify, publish, and announce the managed-runtime, compatibility-ledger, and DataLoader foundation as RustTorch 0.2.0.

**Architecture:** The release is cut only after all three first-milestone plans pass together. GitHub receives one signed-off commit and tag; crates.io receives the bootstrap CLI before the library; docs.rs builds from the published library metadata.

**Tech Stack:** Cargo packaging/publishing, Git, GitHub CLI, crates.io, docs.rs.

**Spec:** `docs/superpowers/specs/2026-08-30-pytorch-compatibility-program.md`

## Global Constraints

- Release version is exactly `0.2.0` for both `rusttorch` and `rusttorch-cli`.
- The release claims only the scopes marked supported in `compat/pytorch_api.toml`.
- No credential, downloaded LibTorch archive, target artifact, virtual environment, or model artifact enters Git.
- The previously exposed crates.io token is never reused; Cargo must obtain authorization from a newly configured secure credential.
- GitHub repository remains public and named `RustTorch`.
- A failed verification or package dry run stops the release before any publication.

---

### Task 1: Integrate milestone versions and changelog

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/rusttorch-cli/Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `CHANGELOG.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed runtime, ledger, and DataLoader plans.
- Produces: two packages at version `0.2.0` and release notes with exact supported scope.

- [ ] **Step 1: Confirm the working tree contains only intended milestone changes**

Run:

```sh
git status --short
git diff --check
```

Inspect every path and stop if credentials, `.venv`, `target`, archives, or model files appear.

- [ ] **Step 2: Set both package versions to 0.2.0**

Change the root and CLI `package.version` fields to `0.2.0`, then regenerate
the lockfile with `cargo metadata --format-version 1`.

- [ ] **Step 3: Write exact release notes**

Add a `0.2.0 - 2026-08-30` section covering managed official LibTorch
acquisition, the three setup commands, deterministic map/stream batching,
compatibility-ledger enforcement, complete public rustdoc, and the explicitly
planned worker/domain/operator surfaces. Remove README language that still
calls the package a 0.1 MVP.

- [ ] **Step 4: Run version consistency checks**

Run:

```sh
cargo metadata --no-deps --format-version 1
python3 scripts/check-compatibility.py --check
```

Expected: both packages report 0.2.0 and the ledger matches Cargo metadata.

- [ ] **Step 5: Commit**

```sh
git add Cargo.toml crates/rusttorch-cli/Cargo.toml Cargo.lock CHANGELOG.md README.md
git commit -m "chore: prepare RustTorch 0.2.0"
```

### Task 2: Run the complete release gate

**Files:**
- No source changes expected.

**Interfaces:**
- Produces: reproducible evidence that the Git commit is releaseable.

- [ ] **Step 1: Run static and unit gates**

With `LIBTORCH_USE_PYTORCH=1` and the project PyTorch library directory on the
platform loader path, run:

```sh
python3 scripts/check-compatibility.py --check
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

- [ ] **Step 2: Run parity and backend gates**

Run:

```sh
scripts/run-python-parity.sh
scripts/check-backends.sh
```

Expected: CPU passes; available MPS/CUDA backends pass; unavailable hardware is explicitly skipped.

- [ ] **Step 3: Run benchmark and package dry runs**

Run:

```sh
cargo bench -p rusttorch --bench data_loader
cargo publish -p rusttorch-cli --dry-run
cargo publish -p rusttorch --dry-run
```

Inspect package file lists and sizes; neither package may contain native runtime archives.

- [ ] **Step 4: Confirm the tree is unchanged**

Run:

```sh
git status --short
git diff --check
```

Expected: clean working tree.

### Task 3: Push GitHub release state

**Files:**
- Git refs only.

**Interfaces:**
- Produces: public `main`, tag `v0.2.0`, and GitHub release notes.

- [ ] **Step 1: Push the verified commit**

```sh
git push origin main
```

- [ ] **Step 2: Create and push the annotated tag**

```sh
git tag -a v0.2.0 -m "RustTorch 0.2.0"
git push origin v0.2.0
```

- [ ] **Step 3: Create the public GitHub release**

```sh
gh release create v0.2.0 --repo newpoluton-alt/RustTorch --title "RustTorch 0.2.0" --generate-notes --verify-tag
```

Expected: the release points at the verified tagged commit.

### Task 4: Publish both Cargo packages

**Files:**
- External package indexes only.

**Interfaces:**
- Consumes: a new secure Cargo credential and the GitHub tag.
- Produces: crates.io packages `rusttorch-cli` 0.2.0 and `rusttorch` 0.2.0; docs.rs build for the library.

- [ ] **Step 1: Verify Cargo has a non-exposed credential without printing it**

Run:

```sh
cargo owner --list rusttorch
```

If authorization fails, stop before publishing and request that the user
configure a newly issued token through Cargo's credential mechanism. Never put
the token in a command, file diff, log, or chat response.

- [ ] **Step 2: Publish the bootstrap CLI first**

```sh
cargo publish -p rusttorch-cli
```

Expected: crates.io accepts `rusttorch-cli` 0.2.0.

- [ ] **Step 3: Publish the library**

```sh
cargo publish -p rusttorch
```

Expected: crates.io accepts `rusttorch` 0.2.0.

- [ ] **Step 4: Verify public package pages**

Verify the version through the crates.io API and confirm docs.rs has queued or
built `rusttorch` 0.2.0. Confirm GitHub, crates.io, and docs.rs all link to the
public `newpoluton-alt/RustTorch` repository.

