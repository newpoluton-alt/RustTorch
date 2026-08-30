# RustTorch 0.2 Foundation Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate, verify, publish, and announce the managed-runtime, compatibility-ledger, and DataLoader foundation as RustTorch 0.2.0.

**Architecture:** The release is cut only after the runtime, compatibility,
data, and open-source foundation plans pass together. One exact release commit
is fast-forwarded to `main`; crates.io receives the bootstrap CLI before the
library; only then does a tag trigger the SLSA-backed GitHub release. docs.rs
builds from the published library metadata.

**Tech Stack:** Cargo packaging/publishing, Git, GitHub CLI, crates.io, docs.rs.

**Spec:** `docs/superpowers/specs/2026-08-30-pytorch-compatibility-program.md`

## Global Constraints

- Release version is exactly `0.2.0` for both `rusttorch` and `rusttorch-cli`.
- The release claims only the scopes marked supported in `compat/pytorch_api.toml`.
- No credential, downloaded LibTorch archive, target artifact, virtual environment, or model artifact enters Git.
- The previously exposed crates.io token is never reused; Cargo must obtain authorization from a newly configured secure credential.
- GitHub repository remains public and named `RustTorch`.
- A failed verification or package dry run stops the release before any publication.
- Open-Source Foundation Task 4's tag-only release/provenance workflow is
  merged and structurally validated before the release begins.

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

- [ ] **Step 6: Record the immutable release candidate**

Record `RELEASE_SHA=$(git rev-parse HEAD)` without abbreviating it. Every later
gate, local-main integration, push, package publication, tag, and provenance
check must refer to this exact commit. Stop if the worktree changes.

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
RUSTDOCFLAGS="-D warnings" cargo doc -p rusttorch --no-deps \
  --no-default-features --features tch/doc-only
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
cargo package -p rusttorch-cli --locked --list
cargo package -p rusttorch --locked --list
```

Inspect package file lists, built `.crate` archives, and sizes; neither package
may contain a credential, virtual environment, native runtime, target artifact,
model artifact, or internal SDD ledger.

- [ ] **Step 4: Confirm the tree is unchanged**

Run:

```sh
git status --short
git diff --check
```

Expected: clean working tree.

### Task 3: Publish the verified packages from exact public main

**Files:**
- Git refs only.

**Interfaces:**
- Produces: public `main` at `RELEASE_SHA`, then crates.io packages
  `rusttorch-cli` 0.2.0 and `rusttorch` 0.2.0.

- [ ] **Step 1: Fast-forward local and public main to the verified SHA**

Fetch `origin/main` and verify it is an ancestor of `RELEASE_SHA`. Find the
worktree that has local `main`, require it to be clean, and fast-forward it to
the foundation branch with a reviewed `--ff-only` merge. Verify both
`refs/heads/main` and the release worktree `HEAD` equal `RELEASE_SHA`, then push
the explicit refspec `RELEASE_SHA:refs/heads/main`. Never run a bare
`git push origin main` while local `main` and the verified worktree differ.

- [ ] **Step 2: Require a newly configured Cargo credential**

Do not inspect, print, or reuse any existing credential. Ask the user to create
and configure a new least-privilege crates.io credential through Cargo's
credential mechanism, then verify authorization with a read-only owner query.
If authorization fails, stop here: public `main` may remain updated, but do not
publish either crate or create/push the release tag.

- [ ] **Step 3: Publish the bootstrap CLI first**

```sh
cargo publish -p rusttorch-cli
```

Expected: crates.io accepts `rusttorch-cli` 0.2.0. Verify the public package
version before proceeding.

- [ ] **Step 4: Publish the library**

```sh
cargo publish -p rusttorch
```

Expected: crates.io accepts `rusttorch` 0.2.0 and docs.rs queues the build.

### Task 4: Tag, attest, and verify the GitHub release

**Files:**
- External package indexes only.

**Interfaces:**
- Consumes: `RELEASE_SHA`, both public crates, and the tag-only SLSA workflow.
- Produces: tag `v0.2.0`, a GitHub release containing exact `.crate` assets,
  and SLSA provenance for those assets.

- [ ] **Step 1: Create and push the tag only after both packages are public**

```sh
git tag -a v0.2.0 "$RELEASE_SHA" -m "RustTorch 0.2.0"
git push origin v0.2.0
```

Verify the tag peels to `RELEASE_SHA`. The push triggers the OpenSSF Generic
Generator; do not call `gh release create` separately because
`upload-assets: true` creates the tag release and uploads provenance.

- [ ] **Step 2: Verify the tag workflow and release assets**

Wait for the tag workflow. Require successful build, provenance, and release
jobs. Verify the GitHub release targets `v0.2.0`/`RELEASE_SHA` and contains both
versioned `.crate` assets plus the configured `.intoto.jsonl` provenance. The
release job must have used only:

```sh
GH_TOKEN="${GITHUB_TOKEN}" gh release upload "$GITHUB_REF_NAME" dist/*.crate --clobber
```

- [ ] **Step 3: Verify provenance and crates.io byte identity**

Run `slsa-verifier` v2.7.1 over both release assets with:

```text
--source-uri github.com/newpoluton-alt/RustTorch --source-tag v0.2.0
```

SLSA directly attests the GitHub release `.crate` bytes. Download each
published crates.io archive and compare its SHA-256 with the corresponding
attested release asset before claiming that the crates.io artifact is
byte-identical to the provenance subject.

- [ ] **Step 4: Verify all public package pages**

Verify the version through the crates.io API and confirm docs.rs has queued or
built `rusttorch` 0.2.0. Confirm GitHub, crates.io, and docs.rs all link to the
public `newpoluton-alt/RustTorch` repository.
