# Task 11 independent implementation review

**Verdict: CHANGES REQUESTED**

Reviewed product range:

- base: `0014b8e4c76c97c6d2142f46de22e7fedee8d19f`
- product head: `7b18995363917effbad9da3707628b531dd81e22`
- report-only head reviewed: `e9e3ddf9f283e6565a301e46ef40541be966caa7`

The product and report commits both contain the required
`Signed-off-by: newpoluton-alt <newpoluton@gmail.com>` trailer. The worktree was
clean before this review artifact was added.

## Finding

### [P1] Reject the checkpoint envelope before plan mutation or sampler callbacks

The resume build mutates the plan with `apply_batch_options` and asks it for
`checkpoint_identity()` before it calls `validate_loader_state_envelope`
(`crates/rusttorch-data/src/loader.rs:1338-1344`). For automatic batching,
`checkpoint_identity()` invokes the downstream sampler's public
`SamplerCheckpoint::kind()` and `distributed_configuration()` methods
(`crates/rusttorch-data/src/loader.rs:307-313`). Explicit batching and no-batch
plans have the same callback-shaped identity path.

This makes a structurally invalid persisted checkpoint run user component code
before its schema, identity, generation, or derivation versions have been
rejected. It contradicts the required resume order in
`.superpowers/sdd/2026-08-31-complete-data-loader/task-11-brief.md:210-224` and
the explicit acceptance criterion that wrong schema/configuration rejects
before callbacks at `task-11-brief.md:272-273`. It also invalidates the report's
claim that an early envelope failure invokes zero component callbacks at
`.superpowers/sdd/2026-08-31-complete-data-loader/task-11-report.md:105-122`.

I confirmed the behavior with an external path-dependent adversarial program,
without changing product files. Its custom public `SamplerCheckpoint` counted
`kind()` calls through `Rc<Cell<usize>>`. After producing a valid initial
checkpoint, the probe changed only `schema_version`, reset the counter, and
attempted resume. Resume returned `Err`, but the counter was `1`; the assertion
that schema rejection precedes sampler callbacks failed with:

```text
assertion `left == right` failed: sampler kind callback ran before schema rejection
  left: 1
 right: 0
```

The focused transaction test does not catch this because its malformed-envelope
branch counts dataset/sampler validation and apply methods, but not sampler-kind
or distributed-configuration calls
(`crates/rusttorch-data/tests/checkpoint_map.rs:867-901`).

Validate all envelope fields that do not require plan identity before calling
any plan/sampler method, and avoid mutating the plan until the complete envelope
and every component have validated. The stable sampler kind/distributed
identity should be obtainable without an observable downstream callback (for
example through non-callback metadata in the checkpoint-capable type state).
Add an adversarial regression that makes identity/configuration access
observable and proves malformed schema/identity/configuration invokes it zero
times.

## Requirements audit

Apart from the finding above, inspection found the requested Task 11 structure
and semantics present:

- the trailing defaulted checkpoint type state preserves the prior public map
  builder/owner/iterator arities and is carried through type-changing setters;
- exact mode is restricted to serial, memory-disabled owned map loading with
  explicit replay-safe or transactional dataset evidence;
- the active iterator invalidates a boundary on entry to `next()` and commits it
  only after a successful visible, post-pinning result and counter advance;
- resumed owners retain the validated/restored concrete plan iterator and
  transform for the first iteration, while `set_epoch` discards pending resume
  state before starting a fresh epoch;
- every concrete sampler stores and validates its required immutable
  configuration, epoch, and occurrence cursor; weighted values use bitwise
  identity; `BatchSampler` validates nested and batch-boundary state;
- `NoBatch` checkpoints the effective converter rather than the dead outer
  collator;
- replay-safe tensor conversion fallibly deep-copies private backing and each
  fetched row, and ordinary `TensorDataset` remains storage-sharing;
- component validation precedes dataset/sampler/transform/coordinator restore,
  and successful restore retains the actual restored objects;
- pin request and effective status are checkpoint identity;
- resume-only `CheckpointBuildError<F::Error>` preserves both the
  `RustTorchError` and arbitrary transform-factory source chains without
  stringification, while default and fresh builds keep the existing RustTorch
  result;
- serde is format-neutral in production, `serde_json` is test-only for
  `rusttorch-data`, the Rust 1.88 declarations remain intact, and the package
  list contains only the intended 34 files; and
- no Task 12 positive-worker barrier, Task 13 stream resume, or Task 14
  distributed coordination implementation entered the product range.

## Verification

Fresh independent commands run for this review:

- `cargo test -p rusttorch-data --test checkpoint_map --locked`: **17 passed**;
- external malformed-schema sampler-callback probe: **failed as expected**, with
  one pre-envelope callback, proving the finding;
- `cargo check --workspace --locked`: **passed**;
- compatibility generated-document check: **passed**;
- `cargo package -p rusttorch-data --list --locked`: **34 intended files** and
  no runtime, virtual-environment, native-library, fixture, or checkpoint
  output;
- `cargo fmt --all --check`: **passed**;
- raw product-range `git diff --check`: **passed**.

The report records passing warning-denied Clippy, full workspace tests,
doctests, rustdoc, Rust 1.88 member check, and workspace archive verification.
Those long gates were not redundantly rerun after the deterministic acceptance
failure above; the focused test and full workspace compilation were rerun.

Task 11 should not be approved until the pre-envelope callback/mutation path and
its missing regression evidence are corrected.
