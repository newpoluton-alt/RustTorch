# Task 7 second review-fix report — conclusive build validation and exact stages

Implementation under review: `eafbefc0524abbb65fb5410fad706e68714e7068`

> Correction: the aggregate described in this report omitted crossbeam's
> per-channel heap control blocks. `task-7-fix3-report.md` supersedes that
> allocation claim and documents the final conservative aggregate.

## Result

Addressed every finding in `task-7-fix-review.md` without implementing Task 8
or changing the local zero-worker execution capability.

- `build()` now completes scalar, timeout/persistence, checked product,
  crossbeam ring, and concrete queue-allocation validation before applying
  batch options or calling sampler/batch-source `set_epoch`.
- The build and `WorkerPool` defensive path share one generic validator. It
  instantiates the actual `Completion<D, F, I>` channel slot and combines every
  task slot, completion slot, and worker/vector endpoint. As corrected in the
  final fix, the conservative aggregate also includes one control-block
  allowance per channel before checking 64 MiB. Every caller-controlled
  multiplication and addition is checked.
  WorkerPool still performs fallible concrete slot and vector reservations
  before constructing threads.
- Positive-worker timeout and persistence requests now return their typed
  unsupported errors directly from `build`, before any plan callback. They
  remain scheduled for Task 8.
- Each plan exposes defaulted panic-stage constants. Automatic and no-batch
  plans identify sampler epoch/creation/refill; explicit plans identify batch
  source epoch/creation/refill; finish panics identify either collation or
  no-batch conversion. Defaults avoid imposing a new required item on custom
  `LoaderPlan` implementations.
- Batch-source refill and no-batch converter panic regressions prove exact
  stage and batch metadata, one error followed by `None`, and joined worker
  transform teardown.

## RED evidence

The safe focused RED run never invoked an iterator for an impossible capacity:

```text
cargo test -p rusttorch-data --test map_workers
FAILED. 15 passed; 6 failed; 0 ignored
```

The six failures showed:

1. sampler/batch-source `set_epoch` ran before impossible-capacity rejection;
2. a 1 KiB inline dataset error with factor 100,000 was falsely accepted by
   the prior fixed 256-byte build estimate;
3. timeout/persistence reached `set_epoch` before later rejection;
4. explicit batch-source creation was labeled as sampler creation;
5. explicit batch-source refill was labeled as sampler refill; and
6. no-batch converter panic was labeled as collation.

No test hung or attempted the known impossible crossbeam allocation path.

## GREEN evidence

The final map-worker suite has 21 passing tests. New tests use panicking and
counting sampler/batch-source setters to prove rejected builds leave callbacks
untouched. The large-inline-error regression proves build uses the actual
generic completion representation. Explicit refill and conversion regressions
prove exact lifecycle metadata and disconnect/join behavior without sleeps.

Existing worker panic conversion, typed pipeline source preservation,
deterministic routing, ordering, bounded credits, and non-`Send`/non-`Sync`
serial coverage remain green.

## Allocation ceiling documentation

The final `prefetch_factor` rustdoc, `crates/rusttorch-data/README.md`, and
canonical compatibility entry document a conservative aggregate of concrete
task/completion slots, worker/vector bookkeeping, and per-channel control-block
allowances against a 64 MiB queue-allocation ceiling before sampler or
batch-source callbacks. Generated coverage and package compatibility
documentation were regenerated.

This ceiling covers eagerly allocated transport/coordinator bookkeeping, not
arbitrary sample payload bytes. Payload memory permits remain Task 10.

## Verification

Fresh final-state gates:

- `cargo test -p rusttorch-data --test map_workers`: 21 passed;
- `cargo test -p rusttorch-data --test loader_builder --test transform --test worker_context`: 26 passed;
- `cargo test -p rusttorch --test data_libtorch`: 1 passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed;
- `cargo clippy --workspace --all-targets -- -D warnings`: passed;
- `cargo test --workspace --all-targets`: 214 passed, 1 ignored;
- `cargo test -p rusttorch-data --doc`: 12 passed;
- warning-denied rusttorch-data rustdoc: passed; and
- `git diff --check`: passed.

## Files and self-review

Changed the loader/worker implementation, focused map-worker and builder
tests, package README, canonical/generated compatibility documentation, and
this report.

Reviewed automatic sampling, explicit batch sources, no-batch conversion,
default and custom transforms, default and custom worker initialization,
serial/worker capability selection, and argument-conflict build paths. Build's
new generic sizing requires only the existing `Dataset`, `TransformFactory`,
and `WorkerInit` semantic traits; it adds no `Send`, `Sync`, `Collate`, or
`LoaderPlan` bound. All builder overload/source configurations compile in the
workspace matrix.

No unsafe code, unbounded queue, sleep-based synchronization, detached thread,
worker collation, serial fallback, pinning behavior, or Task 8 lifecycle
implementation was added. No remaining Task 7 review blocker is known.
