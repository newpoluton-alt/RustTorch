# Task 8 review-fix report — lifecycle race closure

Implementation under review: `bdea47297c9adb5ceb385684a5be1aefbc198ce2`

Fix: `4caf4ef`

## Result

Addressed all five findings from `task-8-review.md` without adding Task 9 or
later scope.

- Cancellation, deadline arm, and deadline disarm now acquire the same waiter
  mutex used by every condition-variable wait. The documented sole lock order
  is waiter mutex followed by deadline or signal-sender mutex, and notification
  occurs before releasing the waiter mutex. Atomic cancellation checks and the
  crossbeam disconnection wake remain available for fast and queue paths.
- Coordinator submission is nonblocking. If deterministic routing finds one
  worker lane saturated during refill, the iterator retains exactly that
  tagged task as its pending submission and returns the already-ready batch.
  No later plan item, sequence, logical occurrence, or credit advances until
  the pending task is accepted. The following blocking `next()` therefore owns
  the timeout rather than internal prefetch hiding it.
- Nonzero loader timeouts that exceed the platform monotonic `Instant` range
  reject during build validation before plan or worker callbacks. Public
  `Deadline::after` and the loader's defensive arm path fail closed as already
  expired if an addition is not representable; overflow never aliases an
  unarmed deadline.
- Persistent factory observations are sorted by `WorkerInfo::id` before seed
  assertions. Thread reuse is independently compared by logical sample across
  both epochs, so concurrent callback insertion order is irrelevant.
- `WorkerContext` now has a compiling lifecycle rustdoc example covering
  `CancellationToken`, `Deadline`, both `WaitOutcome` variants,
  `LoaderCancelled`, and `WorkerContext::check`. Compatibility evidence and
  generated coverage describe the validated timeout range and the new
  saturated-refill regression.

## RED evidence

The deterministic pre-fix evidence was:

```text
worker_context::tests::cancellation_transition_holds_the_waiter_mutex
FAILED: predicate transition did not hold the waiter mutex

worker_context::tests::deadline_transitions_hold_the_waiter_mutex
FAILED: predicate transition did not hold the waiter mutex

public_cancellation_and_deadline_waits_are_notification_driven
FAILED: left None, right Some(0ns)

overflowing_timeout_rejects_before_plan_or_worker_callbacks
FAILED: set_epoch must not run

unordered_ready_batch_does_not_block_on_saturated_lane_refill
FAILED after 2.01s: ready batch was blocked by internal refill: Timeout
```

The saturated-lane RED used exactly two workers, prefetch factor two, unordered
delivery, worker-zero sequences 0/2/4, and worker-one sequences 1/3/5. A
captured generation token was used only for bounded failing-head cleanup, so
the scoped caller and all workers joined without a correctness sleep or hang.

The reviewed persistent test also reproduced its incorrect concurrent-order
seed assertion on the first exact rerun.

## GREEN evidence

The new tests prove:

- cancellation and both dynamic-deadline transitions hold the waiter mutex at
  the predicate mutation point;
- overflowing public deadlines are observably expired, while loader overflow
  rejects with the `timeout` field before sampler, factory, initializer, or
  thread side effects;
- ready unordered batches 1, 3, and 5 return promptly while worker zero's lane
  is saturated, the next call yields `Timeout { batch: 3 }`, and the following
  call is `None` with no late publication; and
- persistent seed/thread assertions pass ten consecutive exact reruns after
  keying observations rather than relying on callback order.

## Verification

Fresh final-state gates:

- private synchronization unit tests: 2 passed;
- focused `loader_builder`, `transform`, `worker_context`, `map_workers`, and
  `worker_lifecycle`: 63 passed;
- facade regression: 1 passed;
- persistent exact regression repeated ten times: 10 passed;
- `cargo check --workspace --all-targets --locked`: passed;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed;
- `cargo test --workspace --all-targets --locked`: 232 passed, 1 ignored;
- warning-denied default-feature rusttorch-data doctests: 13 passed;
- rusttorch-data doc-only check and warning-denied rustdoc: passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed; and
- `git diff --check`: passed.

The longest gate completed in approximately 30.3 seconds. No test or command
reached the 60-second interruption threshold.

## Self-review

Re-audited every condition-variable read, wait, predicate transition, and
notification; timeout/result precedence and deadline disarm on normal result,
failure, timeout, cancellation, closure, and exhaustion; and stale,
out-of-order, and quiescence loops that must retain the same deadline.

The pending submission uses only the credit released by the visible batch,
preserves plan and deterministic routing order, cannot coexist with zero
successful outstanding work on a full lane, is accepted before source
exhaustion can advance, and is cleared on every error, timeout, iterator drop,
or owner shutdown. It never crosses a persistent generation. Existing typed
sources, Task 7 capacity/callback guarantees, zero-worker local types,
generation tagging, persistent quiescence, and non-force-cancellation behavior
remain intact.
