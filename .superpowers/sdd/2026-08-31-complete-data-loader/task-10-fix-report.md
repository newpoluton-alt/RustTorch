# Task 10 fix report — cancellation precedence and disabled storage

Implementation fix commit: `e2922fbcfc8b953e098ae6f39415345966a062b2`

Reviewed range: `59559a67fedfceceb14c3a0e4b2231eb9a390ce4..e2922fbcfc8b953e098ae6f39415345966a062b2`

## Outcome

Both Task 10 review findings are fixed at their shared roots. Generation
cancellation now closes and notifies the byte budget before it wakes lifecycle
waiters, and a cancelled budget wins over a new oversize decision. Map and
stream storage now monomorphize permit and stream-waiter payloads through the
existing trailing memory policy: byte-disabled loaders retain Task 9's
zero-sized/no-allocation representation, while byte-enabled loaders retain and
charge every permit and waiter needed for strict accounting.

No public/default API was removed or renamed. The reviewed type-state,
`MemoryFootprint`, `PinMemory`, error/source-chain, ordering, lifecycle,
pin-status, facade, compatibility, and auto-trait contracts remain intact. No
Task 11 checkpoint work was introduced.

## RED-first evidence

The deterministic cancellation regression was written before its hook existed.
The first focused unit compile failed with no method named
`cancel_with_test_hook` on `WorkerRunContext`. Its event-driven interleaving
holds the only byte permit, starts a lifecycle observer, pauses centralized
cancellation after budget notification but before token notification, releases
the held permit, and attempts both a fitting and an oversize acquisition.

The public capacity regressions were also written before the storage fix. The
first focused `memory_budget` run failed in
`disabled_capacity_keeps_task9_map_and_stream_completion_shapes`: the
byte-disabled two-worker `u8` map loader with `prefetch_factor = 200_000` was
rejected because the enlarged Task 10 completion shape required `67,210,834`
bytes, above the `67,108,864`-byte ceiling. The review's base-locked probe had
already demonstrated the corresponding disabled stream regression. No test
was weakened or removed.

## Cancellation proof

`WorkerRunContext::cancel` and the test hook share one implementation. That
implementation calls `ByteBudget::cancel` first; `ByteBudget::cancel` marks the
state cancelled while holding its standard-library mutex and then notifies all
condition-variable waiters. Only after that call returns does the helper cancel
the lifecycle token. `ByteBudget::acquire_inner` now takes the same mutex and
checks `cancelled` before checking `bytes > limit`, so cancellation is the
stable result for both fitting and newly oversize requests once cancellation
has begun.

The channel-driven regression pauses exactly between those notifications. At
the pause it drops the held permit, then proves `acquire(1)` and `acquire(2)`
both return `BudgetError::Cancelled` and that the lifecycle observer has not
yet awakened. It resumes cancellation and joins both threads. Thus neither a
post-cancel permit nor the `Oversize` value that maps to a published
`MemoryLimit` can be produced in the reviewed race, and the test terminates
without a leaked permit or waiter.

The cancellation audit found all active-generation timeout, first-error,
protocol-failure, iterator-drop, quiesce, poison, shutdown, and partial-start
paths in both map and stream code converge on this helper. Pool-lifetime
shutdown tokens do not own byte budgets; their shutdown paths first cancel any
active generation through the helper and only then close the pool token.
Existing deterministic lifecycle coverage continues to prove a timeout is
returned once, the next call is `None`, blocked permit waiters terminate, and a
persistent generation restarts without a leak. Existing exact oversize tests
continue to prove one `MemoryLimit` followed by `None`.

No unsafe cancellation, detached helper, unbounded queue, correctness sleep,
or polling loop was added.

## Exact disabled/enabled storage proof

Map completion storage is parameterized by the memory policy's associated
permit payload. `MemoryDisabled::Permit` is `()`, so the disabled worker batch,
message, result-channel slot, completed-map entry, owner pool, and iterator do
not store an `Option<BytePermit>` or any permit state. `MemoryEnabled::Permit`
is the owned `BytePermit`, so the exact enabled completion type used by channel
allocation and by the 64 MiB validation calculation includes that permit.

Stream result records, buffered records, and ordered-reassembly entries use the
same associated permit payload. Stream waiter storage is a second associated
policy payload:

- disabled: `Waiters = ()`, reported waiter bytes are exactly zero, waiter
  preflight is a no-op, construction performs no reserve or resize, and waiter
  access is absent;
- enabled: `Waiters = Vec<Option<u64>>`, validation adds exactly
  `workers * size_of::<Option<u64>>()`, preflights that allocation, and
  construction uses an exact reserve followed by one slot per worker.

Capacity validation is monomorphized over those concrete policy types, so it
charges the actual result/reassembly envelope chosen for that loader rather
than a larger unconditional shape. The base-locked public regression proves
the disabled two-worker `u8`, factor-`200_000` map and batch-size-one stream
loaders both build. The same factor with byte accounting enabled is rejected
as `InvalidConfiguration { field: "prefetch_factor", .. }`; factor `100_000`
builds for each enabled loader. Existing `current_api` assertions continue to
compile the old defaulted owner/iterator arities and verify the disabled owner
and iterator auto-trait surface.

## Preserved supported and unsupported scope

Supported scope remains strict opt-in, post-transform logical-payload
accounting for owned map batches and stream records; sequence-aware ordered
admission; exact `MemoryLimit` metadata; RAII release across success, error,
drop, timeout, cancellation, quiescence, and persistent generations; the
documented recursive footprint set; post-collation recursive pinning; CUDA-zero
automatic selection; CPU/MPS automatic no-op status; explicit available-CUDA
validation; typed `LoaderError::PinMemory`; and source-compatible disabled
builders, owners, and iterators.

Unsupported scope is unchanged: this is not a process RSS cap, allocator/node
overhead and tensor shared/view backing remain outside the logical-payload
contract, the active coordinator batch remains outside the byte budget,
automatic mode does not choose non-CUDA accelerators or a current CUDA index,
and no dynamic pinning duck typing, asynchronous-transfer benchmark, or
checkpoint/resume behavior is claimed. Tasks 11–13 remain outside this range.

The persistent stream lifecycle flake documented by the adversarial review did
not reproduce during these final gates. No attempt was made to change it
because the review established it on the accepted Task 9 base.

## Fresh GREEN verification

Every Rust command sourced `. scripts/dev-env.sh` and used locked dependencies.

- deterministic cancellation-precedence unit: 1 passed;
- focused `memory_budget`: 15 passed;
- focused memory/pin/stream/lifecycle/map/builder/transform/current-API: 112
  passed;
- root facade `data`: 29 passed;
- `cargo check --workspace --locked`: passed;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed;
- `cargo test --workspace --all-targets --locked`: 290 passed, 1 ignored;
- `cargo test --workspace --doc --locked`: 17 passed;
- rusttorch-data doc-only all-target check and warning-denied rustdoc: passed;
- compatibility write/check: current with no generated diff;
- `cargo fmt --all -- --check`: passed;
- raw `git diff --check`: passed; and
- implementation commit subject/trailer and diff check: passed.

The implementation fix commit is DCO-signed.
