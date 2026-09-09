# Task 7 review-fix report — bounded capacity and coordinator panic safety

Implementation under review: `b99b7cd4340d425651c5f220e335fd8da7701096`

## Result

Addressed all three findings from `task-7-review.md` without adding Task 8
lifecycle behavior.

- Positive-worker queue sizes now pass checked multiplication, crossbeam ring
  arithmetic, a conservative 64 MiB queue-metadata budget, concrete channel
  slot-size validation, and fallible vector reservations. The result channel
  and every per-worker task channel are created before any worker thread can
  start. Caller-controlled extreme values, including one worker with
  `prefetch_factor(usize::MAX)` and other representable but impossible sizes,
  therefore return `InvalidConfiguration` instead of reaching crossbeam's
  eager allocation.
- `LoaderError::CoordinatorPanic` carries the coordinator stage and logical
  batch when known. Positive-worker sampler epoch lookup, plan/batch-source
  creation, iterator refill, and coordinator collation/conversion are protected
  by unwind boundaries. A panic stops further submission, disconnects and joins
  the pool, is yielded once as a typed error, and is followed by `None`.
- Timeout and persistence rejection remains side-effect-free and now precedes
  even custom sampler epoch access. The zero-worker serial path retains its
  local non-`Send`/non-`Sync` API and behavior.
- `THIRD_PARTY_NOTICES.md` now records the locked inventory date as
  `2026-09-01`.

## RED evidence

The initial safe focused run stopped the extreme-capacity case at `build()`;
it never called `iter()` with an impossible channel size:

```text
cargo test -p rusttorch-data --test map_workers
FAILED. 11 passed; 4 failed; 0 ignored
```

The four failures proved that representable impossible capacities built
successfully, and that initial sampler, refill sampler, and collator panics
escaped the loader. No test hung. Two additional focused RED runs established
that sampler epoch panics escaped and timeout rejection occurred after sampler
epoch access.

All regressions use panic capture, atomics, or existing worker join behavior;
none uses sleeps or attempts the known `usize::MAX` iterator/allocation path.

## GREEN evidence

The final map-worker suite has 16 tests. New coverage proves:

- `usize::MAX` and `usize::MAX / 2` prefetch factors fail at build with a typed
  `prefetch_factor` configuration error and no factory/initializer side effect;
- sampler and explicit batch-source creation panics become one typed
  `CoordinatorPanic` before worker startup;
- sampler epoch and refill panics carry precise stage/batch metadata;
- collator panic carries batch metadata, yields once, and joins both workers;
  and
- timeout rejection precedes custom sampler epoch access.

Normal typed pipeline errors continue to preserve their concrete sources.

## Verification

Fresh final-state gates:

- `cargo test -p rusttorch-data --test map_workers`: 16 passed;
- `cargo test -p rusttorch-data --test loader_builder --test transform --test worker_context`: 26 passed;
- `cargo test -p rusttorch --test data_libtorch`: 1 passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed;
- `cargo clippy --workspace --all-targets -- -D warnings`: passed;
- `cargo test --workspace --all-targets`: 209 passed, 1 ignored;
- `cargo test -p rusttorch-data --doc`: 12 passed;
- warning-denied rusttorch-data rustdoc: passed; and
- `git diff --check`: passed.

## Files and self-review

Changed only `THIRD_PARTY_NOTICES.md`, the loader error/coordination/worker
implementation, the focused map-worker tests, and this report. Generated
compatibility artifacts remained current and unchanged.

Reviewed every positive-worker user-code call, fail-once transitions, queue
credit accounting, allocation arithmetic, pre-thread construction ordering,
and disconnect/join paths. No unbounded queue, sleep-based test, unsafe code,
serial fallback, worker-side collation, timeout/persistence implementation, or
pinning behavior was added. Cooperative cancellation and persistent lifecycle
remain Task 8; pinning remains Task 10.
