# Task 5 report — owned serial loader builder

## Result

Implemented the owned, re-iterable serial `DataLoader` builder while preserving
the borrowed loader constructors, iterator implementation, public facade paths,
and direct borrowed error behavior.

## RED / GREEN

- RED: after adding only `loader_builder.rs`, the focused target failed to
  compile on the intended missing `LoaderError`, `DataLoader::builder`, and
  owned loader symbols (40 compiler errors). One test-only Rust 2024 reserved
  keyword was corrected before recording that intended RED.
- GREEN: `cargo test -p rusttorch-data --test loader_builder` passes 12 tests.
- Preserved facade: `cargo test -p rusttorch --test data` passes 26 tests.
- Complete workspace: `cargo test --workspace --all-targets` passes 177 tests;
  one existing test is ignored.

## API and behavior

- Added inferable `DataLoader::builder(dataset)` through the approved hidden
  dataset marker and default generic parameters; existing explicit generic
  shapes and borrowed constructors are unchanged.
- Added `DataLoaderBuilder`, `OwnedDataLoader`, named `LoaderIter`, `AutoBatch`,
  `ExplicitBatches`, `NoBatch`, and the complete non-exhaustive `LoaderError<E>`
  variant set.
- Added fresh sampler/batch-source iteration, optional exact loader lengths,
  checked batch/drop-tail arithmetic, and `set_epoch` forwarding through every
  plan.
- Added default typed collation, explicit vector/custom collation, and default
  or custom no-batch conversion.
- Added order-independent validation for sampler/shuffle and batch-sampler
  exclusions, plus validation for batch size, drop-last/no-batch, timeout,
  prefetch, and persistence controls.
- Effective prefetch is `None` for serial loading, two batches per worker when
  positive workers omit a factor, or the explicit nonzero factor.
- Owned iteration calls `Dataset::get_batch` once, checks exact cardinality
  before collation, preserves typed pipeline errors, yields failures once, and
  terminates.
- Positive-worker configuration builds, but every attempted iteration yields
  one typed unsupported configuration error and terminates. There is no serial
  fallback and no worker-support claim.

## Evidence and documentation

- Added direct focused tests and a root-facade compiling test.
- Added a compiling public builder example and complete public rustdoc.
- Updated the canonical compatibility ledger and regenerated
  `docs/api-coverage.md`, `rusttorch-core/COMPATIBILITY.md`, and
  `rusttorch-data/COMPATIBILITY.md`.
- The compatibility scope explicitly limits support to owned serial execution;
  actual workers, background prefetch, recursive pinning, timeout enforcement,
  persistence, transforms, and checkpoints remain scheduled.

## Verification

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo test -p rusttorch-data --doc`
- `RUSTDOCFLAGS='-D warnings' cargo doc -p rusttorch-data --no-deps --no-default-features --features doc-only`
- `.venv/bin/python scripts/check-compatibility.py --check`
- `git diff --check`

All listed gates pass.

## Self-review and concerns

- Reviewed every new panic/expect site: the hidden marker is uninhabited outside
  its defining module, and `BatchSampler::new` receives a stored `NonZeroUsize`
  already validated by the builder.
- No unsafe code, detached work, unbounded queue, hidden clone, string-erased
  pipeline error, network operation, or new external dependency was added;
  `thiserror` was already a workspace dependency.
- Expected interim concern: positive-worker options are inspectable but cannot
  execute until Task 7. The typed fail-once behavior is intentional and tested.
