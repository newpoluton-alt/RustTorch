# Task 11 report — versioned component checkpoints and exact serial resume

Implementation commit: `7b18995363917effbad9da3707628b531dd81e22`

## Outcome

Task 11 adds versioned, serde-compatible, exact next-visible-batch checkpoint
and resume for owned map-style serial loaders. The implementation covers every
built-in sampler, automatic batching, compositional `BatchSampler`, no-batch
conversion, replay-safe and transactional datasets, stateless and
transactional transforms, stateful coordinators, deterministic seed metadata,
and disabled, automatic, or explicit CUDA pin identity.

Checkpoint-disabled builders and loaders keep the Task 10 bounds and behavior.
Exact mode is available only from the serial, memory-disabled type state after
the caller supplies both replay evidence and an exact dataset identity.
Positive-worker barriers, prefetched replay, and stream resume remain Tasks 12
and 13; Task 14 was not implemented.

## RED-first evidence

The complete focused checkpoint test was written before product APIs. The
initial command was:

```text
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test checkpoint_map
```

It failed at compile time on the deliberately missing `Checkpointable`,
`LoaderState`, checkpoint schema/pin types, replay/transaction wrappers,
sampler cursor state, `dataset_identity`, `resume_from`, and active
`LoaderIter::checkpoint` surface. The test also exposed the missing direct
`serde_json` test dependency. Production edits began only after that missing
API boundary was recorded.

Incremental GREEN review found and fixed two real edge cases without weakening
tests:

- calling `set_epoch` on a resumed owner could previously mix the retained old
  cursor with the newly selected epoch; checkpoint-active `set_epoch` now
  deliberately discards the pending cursor/transform and starts a fresh
  iteration, proven with a random sampler and stateful transform; and
- the first explicit stateless adapter still required the opaque inner
  transform to implement `Stateless`, which downstream `FnTransform` users
  cannot do under orphan rules. Constructing `StatelessWorker`, or selecting
  `.checkpoint_stateless()`, is now itself the documented unit-state assertion,
  and opaque closure transforms resume exactly when that assertion is true.

## Public surface

The direct `rusttorch-data` package and `rusttorch::data` facade export:

- `LOADER_STATE_SCHEMA_VERSION`, `LoaderState`, `LoaderConfiguration`,
  `CheckpointPinRequest`, `CheckpointPinStatus`, and the source-preserving
  `CheckpointBuildError<FactoryError>`;
- `Checkpointable`, `ReplaySafeDataset`, `DatasetCheckpoint`, `Stateless`,
  `TransactionalCheckpoint`, and `WorkerCheckpoint`;
- `ReplaySafeMap`, `TransactionalMap`, `ReplaySafeTensorDataset`,
  `StatelessWorker`, and `TransactionalWorker`;
- `SequentialSamplerState`, `RandomSamplerState`, `RandomReplacement`,
  `SubsetRandomSamplerState`, `WeightedRandomSamplerState`,
  `DistributedSamplerState`, and `BatchSamplerState`;
- builder transitions `dataset_identity`, `resume_from`,
  `checkpoint_stateless`, and `checkpoint_transactional`; and
- active serial `LoaderIter::checkpoint`.

A trailing defaulted checkpoint type state was appended after Task 10's memory
and pin states on map builders, owners, and iterators. Tests name the old and
current complete builder, owner, serial-iterator, and worker-iterator arities.
Several cursor/type-state plumbing types and traits must be public because they
occur in public associated or return types, but are `#[doc(hidden)]`: they are
implementation plumbing rather than a second user-facing checkpoint API.

## Exact state and boundary contract

`LoaderState` records schema, exact dataset identity, epoch, serial generation,
next visible batch, consumed logical occurrences, component states, RNG and
worker-seed derivation versions, batching, worker/prefetch/order settings,
loader seed, task rank, distributed settings, stable sampler kind, requested
pin policy, and effective pin status.

The fresh boundary and every successfully pinned consumer-visible batch are
checkpointable. Entering `next` invalidates the prior live boundary until a
successful batch commits. Error, end-of-input, and a hidden `drop_last` tail
leave `checkpoint()` unavailable through `LoaderError::Checkpoint`. Logical
cursors count occurrences, including repeated replacement samples.

Concrete sampler state stores immutable configuration plus epoch/position and
regenerates deterministic sequences rather than serializing permutations or
RNG internals. Weighted values are compared by exact IEEE-754 bits.
`BatchSampler` additionally stores its inner state, batch policy, and aligned
batch/logical cursors. Corruption tests mutate every concrete sampler's config,
epoch, and cursor, including random replacement/count/seed, subset indices,
weighted bits, and both copies of every distributed setting.

Ordinary `TensorDataset` remains storage-sharing. Its fallible
`into_replay_safe` path allocates with `f_empty_like`, copies with `f_copy_`,
maps the original backend error, owns private backing, and deep-copies each
returned row. Replay safety propagates only through immutable `Arc`, concat,
subset, and all-safe stack tuple adapters. No blanket replay/checkpoint impl was
added.

## Validation transaction and error sources

Resume performs the required all-validation-before-any-apply transaction:

1. builder, pin, envelope, identity, derivation, serial/order, batch, seed/rank,
   sampler-kind, and distributed configuration validation;
2. dataset-state validation;
3. active plan/sampler configuration and cursor validation;
4. creation of the one actual serial transform followed by transform-state
   validation;
5. validation of the effective owner collator or no-batch converter; then
6. exactly one apply each, in dataset, sampler/cursor, transform, and effective
   coordinator order.

The restored sampler iterator and transform are retained for the first
iteration; resume never validates one instance and silently uses another.
Instrumented tests prove an early envelope failure invokes zero component
validation/apply callbacks, final-component rejection invokes every validation
but zero applies, and valid resume applies dataset, sampler, transform, and
coordinator exactly once.

Configuration/component failures preserve the original `RustTorchError` in
`CheckpointBuildError::Configuration`; arbitrary transform-factory failures
preserve their original value in `CheckpointBuildError::TransformFactory`.
Both `Error::source` branches are tested. Malformed JSON remains the chosen
storage library's error and never falls back to a fresh loader.

## Supported and unsupported scope

Supported here is exact RustTorch-native resume for owned map loaders in the
`SerialExecution + MemoryDisabled` state with `workers=0`, no prefetch,
`in_order=true`, and generation zero. Pinning may be disabled, automatic, or an
available explicit CUDA device and remains part of checkpoint identity.
Stateful datasets use `TransactionalMap`; immutable replay-safe data uses
`ReplaySafeMap` or `ReplaySafeTensorDataset`. Stateful transforms and
coordinators use explicit transactional contracts; opaque transforms require
an explicit stateless assertion.

Calling `.workers(0)` intentionally selects `WorkerExecution` and has no exact
build/iteration capability yet. Positive workers, checkpoint barriers,
prefetched-replay ownership, stream checkpoints, partial in-flight work,
checkpoint file naming, atomic persistence, retention, encryption, and remote
storage remain unsupported. Pinned PyTorch 2.13 exposes
`torch.utils.data.DataLoader` but no public exact iterator-checkpoint API, so
the compatibility ledger marks this as proven RustTorch-native partial scope,
not Python API parity.

## Documentation, dependencies, and packaging

Root and package READMEs now document opt-in replay evidence, typed serde state,
boundary validity, stateful/stateless component choices, resume rejection, and
the exact worker/prefetch/stream exclusions. The canonical compatibility row
maps to the real `torch.utils.data.DataLoader` symbol; generated API coverage
and package compatibility are current.

Workspace `serde` uses derive support. Production `rusttorch-data` depends only
on format-neutral serde; `serde_json` is a workspace-inherited test dependency,
and the root's existing direct test dependency was converted to workspace
inheritance. No storage format, native artifact, compile-test framework,
network runtime, unsafe code, or paid service was added. Existing third-party
notices already list the locked serde and serde_json versions.

The `rusttorch-data` package list contains 34 intended manifest, documentation,
source, and test files, including the new checkpoint source/test. It contains
no `.venv`, `target`, Python cache/tooling, LibTorch/native library, fixture,
runtime checkpoint, or generated checkpoint output. The clean committed
workspace archive gate packaged and verified all four crates: rusttorch-core,
rusttorch-data, rusttorch, and rusttorch-cli.

## Fresh final verification

Every Rust command sourced `. scripts/dev-env.sh` and used locked dependencies.

- mandatory focused sampler/dataset/builder/transform/collate/pin/current-API
  matrix: 108 passed, including 17 checkpoint tests;
- root facade `data`: 30 passed;
- `cargo check --workspace --locked`: passed;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed;
- final `cargo test --workspace --all-targets --locked`: 308 passed, 1
  intentionally ignored Python-parity dispatch test;
- `cargo test --workspace --doc --locked`: 24 passed, including 9
  dependency-free `compile_fail` bounds examples;
- warning-denied rusttorch-data doc-only rustdoc: passed;
- Rust 1.88 rusttorch-data doc-only all-target check: passed;
- compatibility write/check, `cargo fmt --all --check`, and raw
  `git diff --check`: passed;
- rusttorch-data package-list inspection: 34 intended files, no forbidden
  artifact; and
- clean `cargo package --workspace --locked`: all four archives packaged and
  verified.

One earlier non-final workspace run hit the pre-existing timing-sensitive
`persistent_early_drop_drains_stale_results_before_both_delivery_modes_restart`
stream cancellation test. Its isolated retry passed, and the final exact
all-target workspace command passed the same test and the complete workspace.

Implementation commit `7b18995363917effbad9da3707628b531dd81e22` uses the
required subject `feat(data): resume serial loaders exactly` and contains the
`Signed-off-by: newpoluton-alt <newpoluton@gmail.com>` DCO trailer.
