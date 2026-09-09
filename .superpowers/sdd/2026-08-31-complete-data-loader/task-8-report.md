# Task 8 implementation report — lifecycle-safe loader workers

Base: `2b41617ea41d25ece641817509b5642d9f890130`

Implementation: `d3587fc`

## Result

Implemented cooperative worker cancellation, a fresh timeout deadline for each
blocking `Iterator::next` call, and loader-owned persistent worker pools. The
public lifecycle surface now includes cloneable `CancellationToken`,
`Deadline`, `WaitOutcome`, `LoaderCancelled`, and `WorkerContext`; task
contexts expose the active generation controls without changing the locked
version-1 deterministic seed derivation.

`Dataset::get_batch_with_context` is source-compatible and delegates to
`get_batch` exactly once by default. Serial transform factories receive
`None`; worker factories and initializers receive a pool-lifetime
`WorkerContext`, while dataset fetches and transforms receive a fresh
generation token and dynamic deadline.

Positive-worker execution uses bounded task, control, and completion channels.
Every task, result, failure, begin message, and quiescence acknowledgement is
generation-tagged. Persistent iterators cancel and quiesce every worker before
returning the pool to the owner; stale results are drained and discarded.
Pool initialization failures, panics, partial generation startup, or channel
loss poison and synchronously shut down the pool. Iterator and owner drop join
all workers that return.

## TDD evidence

RED was captured before production changes:

```text
cargo test -p rusttorch-data --test worker_lifecycle
FAILED: unresolved CancellationToken, Deadline, WaitOutcome, WorkerContext,
get_batch_with_context, and lifecycle-aware factory/initializer contracts
```

The final event-driven suite has 13 tests and uses channels, barriers, and
condition variables rather than correctness sleeps. It directly proves:

- notification-driven cancellation, absent and expiring deadlines, and
  `check`/wait behavior;
- unchanged context-free datasets plus context-aware fetch and transform wake
  on early drop and deadline;
- one timeout with no late publication and a fresh deadline for the next
  blocking call;
- first-error sibling cancellation, typed initializer error and panic, full
  result queue and idle task wait cancellation, and exact worker exit counts;
- deliberate non-cooperative fetch making drop wait until explicit release;
- serial `None` and real worker lifecycle factory/initializer contexts;
- fresh consecutive non-persistent worker seed ranges;
- persistent reuse of threads, initial seeds, initializer/factory calls, and
  transform state across epochs with fresh generation tokens and task context;
- stale generation/sequence-zero discard after early drop; and
- both ordered and completion-order persistent lifecycle behavior.

All prior Task 7 capacity, routing, callback, panic, typed error, zero-worker,
and source-compatibility regressions remain green.

## Capacity, races, and drop review

Reviewed timeout/result boundary races and deadline disarming on every return,
stale completion, first failure, timeout, cancellation, channel closure, and
normal exhaustion path. An already-ready completion wins over timeout expiry;
the same deadline remains armed while one `next` call filters stale or
out-of-order messages.

Reviewed construction and teardown for allocation failure, partial thread
spawn, partial generation begin, pool initialization failure, worker panic,
full result sends, idle task receives, fatal exits during iterator drop,
duplicate acknowledgements, and owner drop. Fatal workers retain their control
receiver after publishing the typed error and wait on the pool shutdown token,
preventing a startup race from degrading that error to `ChannelClosed`.
Quiescence counts each worker once with fallible, capacity-accounted storage.

Rust threads are not force-cancelled. A dataset, transform, system call, or
native operation that does not observe its context can delay iterator or owner
drop; the implementation waits and joins it after it returns. No detached
thread, unbounded queue, polling cancellation loop, unsafe code, stream worker,
pinning, byte permit, or checkpoint behavior was added.

## Documentation and compatibility

Updated public rustdoc, root and package READMEs, the canonical compatibility
ledger, generated package compatibility, and API coverage. The documentation
distinguishes persistent pool-lifetime callback context from generation-level
fetch/task context and makes the non-force-cancellation limitation explicit.

## Verification

Fresh final-state gates:

- focused loader matrix: 61 passed across `loader_builder`, `transform`,
  `worker_context`, `map_workers`, and `worker_lifecycle`;
- facade regression: 1 passed;
- `cargo check --workspace --all-targets --locked`: passed;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed;
- `cargo test --workspace --all-targets --locked`: 228 passed, 1 ignored;
- default-feature rusttorch-data doctests with warnings denied: 12 passed;
- rusttorch-data doc-only check and warning-denied rustdoc: passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed; and
- `git diff --check`: passed.

For completeness, an additional unsupported attempt to execute the existing
tensor doctests under the `doc-only` feature failed at link time because that
feature does not link LibTorch symbols. The required doc-only compile and
rustdoc lanes passed; default-feature doctests passed, and unrelated tensor
doctest/link behavior was not changed.
