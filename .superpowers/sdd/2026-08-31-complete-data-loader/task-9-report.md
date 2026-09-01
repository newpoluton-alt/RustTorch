# Task 9 report — explicitly sharded streaming workers

Implementation commit: `efac63e28346b692e05022b046da312f67ce90c2`

## Outcome

Task 9 adds a documented, typed positive-worker streaming path without
changing the ordinary zero-worker `batches` or `batches_with_collate` helpers.
The public surface is `SequenceId`, `LogicalSampleId`, `WorkerRecord<T>`,
`WorkerSourceFactory`, `StreamDataLoaderBuilder`, `StreamDataLoader`, and the
named borrowing `StreamLoaderIter`.

Every worker calls the factory once per iterator generation and owns the
returned shard iterator. There is no shared source iterator or source lock.
Persistent pools retain their threads, initializer calls, and transform state,
but recreate sources with a fresh generation cancellation/deadline context and
generation-derived worker seed.

## RED-first evidence

The first `stream_workers` test file failed to compile because the stream
factory, record, identifiers, builder, loader, and iterator did not exist. The
public skeleton was added before the engine, then the coordinator and worker
protocol were connected in compile-checked increments.

Later focused RED runs found and fixed two protocol/lifecycle defects:

- a cooperative source could publish its deadline-driven end marker before the
  coordinator timer woke, incorrectly turning a timeout into exhaustion; the
  coordinator now gives a ready record precedence and then checks an armed
  expired deadline before terminal exhaustion; and
- persistent source factories initially received the pool's generation-zero
  worker seed; source creation now derives fresh `WorkerInfo` from the active
  `WorkerRunContext` for every generation.

The final evidence also extends the standard error-source-chain test for the
new typed source stage and verifies transform-panic sequence/logical metadata.

## Protocol and boundedness

- Ordered delivery requires `Some(SequenceId)`, publishes only the next
  zero-based contiguous global ID, and bounds delayed records in a `BTreeMap`.
  Missing, first-nonzero, internal/final-gap, duplicate/past, and unsequenced
  records yield one contextual `StreamProtocol` error and then exhaustion.
- Unordered delivery accepts unsequenced records and uses the required stable
  `LogicalSampleId` for deterministic task randomness. It does not retain an
  epoch-long logical-ID set.
- Workers receive separate bounded credit lanes of exactly
  `prefetch_factor`. The globally bounded result queue has exactly
  `workers * prefetch_factor` slots. A record retains its worker credit while
  in production, the result queue, or ordered reassembly, and returns it only
  on coordinator acceptance or discard. Thus those locations share one global
  unpublished-record bound; coordinator batch assembly may additionally hold
  at most `batch_size` already-accepted records.
- Credit, result, and control channels are bounded. Checked aggregate
  validation covers every lane, channel control-block allowance, worker
  bookkeeping, result slot, credit slot, and batch allocation before
  `exact_len` or worker/user callbacks.
- Records merge before coordinator collation and batching. `drop_last` removes
  at most one final global tail, intentionally unlike PyTorch iterable
  per-process tail dropping.

## Lifecycle and failure behavior

Records, failures, and end markers carry worker and generation metadata;
records and applicable failures also carry sequence and logical identity.
Source creation/iteration, transform, transform construction, initializer, and
collation errors remain typed. Source and transform panics are contained and
reported with all metadata known at the failure point.

Normal exhaustion, source/transform/collation error, panic, protocol failure,
timeout, early iterator drop, full result queue, persistent quiescence, and
owner drop all release or discard held work and join the relevant threads.
Fatal persistent workers poison and shut down their pool. Partial thread-spawn
failure drops the partially built pool, cancels it, closes its bounded lanes,
and joins every successfully spawned handle. No worker is detached or forcibly
cancelled. As documented, drop waits for an arbitrary non-cooperative source or
native call until that call returns.

## Event-driven coverage

`crates/rusttorch-data/tests/stream_workers.rs` contains 16 tests covering:

- four disjoint modulo shards and strict delayed-low global ordering;
- every required ordered protocol error shape and exact sequence context;
- unordered, unsequenced logical-ID RNG stability across worker counts;
- known/unknown lengths, ceil/floor batching, global collation, and one tail;
- typed stage failures, source and transform panics, and one-error-then-`None`;
- cooperative timeout/drop, non-cooperative join waiting, a full global result
  queue, and exact worker teardown;
- a three-worker factor-two credit probe where the two fast shards make exactly
  four records while sequence zero is blocked, proving per-worker progress and
  the asserted unpublished maximum; and
- ordered and unordered persistent reuse, source recreation, fresh tokens and
  generation seeds, pool-state continuity, early-drop drain, and safe sequence
  zero reuse.

Builder evidence covers zero workers, zero batch/prefetch values, timeout and
capacity overflow, aggregate allocation rejection before callbacks, and the
unchanged ordinary helper signatures. A root-facade test compiles and executes
the re-exported stream contracts.

## Documentation and compatibility

The module has a compiling primary-path rustdoc example and documents ordered
and unordered identity contracts, global tail behavior, and cooperative
cancellation. Root and package READMEs describe the proven scope and limitations.
Canonical compatibility adds the partial `data.stream_workers` row with exact
direct/facade symbols and focused evidence. Generated package compatibility and
API coverage are current. No dependency was added and Tasks 10–13 remain out of
scope.

## Fresh final verification

All commands used the locked development environment. No command or test
reached the 60-second interruption threshold.

- focused `stream_workers`, `worker_lifecycle`, and `map_workers`: 53 passed;
- root façade `data`: 28 passed;
- focused final `stream_workers` plus `transform`: 24 passed;
- `cargo check --workspace --all-targets --locked`: passed;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed;
- `cargo test --workspace --all-targets --locked`: 249 passed, 1 ignored;
- warning-denied default-feature rusttorch-data doctests: 14 passed;
- rusttorch-data doc-only all-target check and warning-denied rustdoc: passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed; and
- `git diff --check`: passed.

The implementation commit is DCO-signed with the required subject
`feat(data): load explicitly sharded streams`.
