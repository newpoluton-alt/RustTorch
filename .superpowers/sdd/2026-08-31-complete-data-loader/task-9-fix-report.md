# Task 9 fix report — explicitly sharded streaming workers

Implementation commit: `838549e589d738c7d1e7e5af9ecd5caa93187280`

Review addressed: `.superpowers/sdd/2026-08-31-complete-data-loader/task-9-review.md`

## Result

All five independent-review findings are fixed without changing the Task 9
public API, adding a dependency, or extending scope into Task 10.

- Ordered delivery now rejects a missing first or internal sequence as one
  contextual `StreamProtocol` once the complete validated reassembly window is
  occupied by higher IDs. A worker already producing the missing low record
  holds its credit outside that window and can still make progress.
- Every generation source is explicitly destroyed inside the worker's protected
  unwind boundary before `End` is published. A blocking destructor therefore
  delays quiescence; a destructor panic becomes the generation's single
  metadata-bearing worker panic, poisons the persistent pool, and prevents a
  subsequent source generation on that worker.
- Pre-callback capacity validation now counts both simultaneously live batch
  vectors and a conservative per-record ordered `BTreeMap` node allowance in
  addition to the bounded result ring. Reassembly cleanup drains in place and
  no longer allocates a proportional temporary vector.
- The coordinator polls the result lane before honoring expiry after every
  processed completion. Queued records, failures, and `End` therefore retain
  Task 8's ready-completion precedence.
- Iterator close and drop request cancellation before returning any buffered
  credit. Typed worker failures and malformed-record protocol failures also
  cancel before returning their held credit, so teardown cannot admit another
  `source.next()` call.

The module, package README, canonical compatibility row, and generated
compatibility pages now state the exact bounded ordered-reassembly condition:
if all `workers * prefetch_factor` credits are retained by higher IDs while the
next ID is absent, the stream is rejected; intentional disorder wider than
that window requires a larger factor or an independently progressing low-ID
shard.

## RED evidence

All regressions are event-driven; changed tests contain no correctness sleep or
polling loop.

- A one-worker factor-two `[1, 2]` first gap and `[0, 2, 3]` internal gap both
  returned `Timeout` instead of `StreamProtocol` before the full-window check.
- Persistent blocking destruction acknowledged exhaustion before the source
  destructor was released. Persistent panicking destruction likewise allowed
  terminal acknowledgement before the panic. The first draft intentionally
  continued into the broken second generation and demonstrated the resulting
  hang; it was interrupted after about 38 seconds, then rewritten with explicit
  release events so the permanent regression cannot strand the suite.
- Large inline `[u8; 1_048_576]` configurations for double-batch coexistence and
  ordered reassembly were accepted and reached `exact_len` before allocation
  accounting was extended.
- A queued high/low record fixture returned `Timeout` after buffering the high
  record even though the low record was already ready. The ready error/End lane
  test was added before its generic ready-first receive seam and initially
  failed to compile with that seam absent.
- Moving cancellation behind buffered-credit release makes the barrier-driven
  first-failure regression fail immediately with `worker entered another source
  call before protocol teardown cancelled`. The analogous iterator-drop
  mutation records one forbidden extra source call.

The first full-workspace run also exposed a related publication race: source or
transform workers could cancel before placing their typed failure in the result
lane, allowing `Cancelled` to win. Source destruction now completes first, the
definitive failure is published second, and cancellation follows; the targeted
typed-failure test then passed 50 consecutive runs.

## GREEN evidence

- Ordered first-gap and internal-gap regressions pass and yield one error then
  `None`; delayed-low progress remains covered.
- Blocking and panicking source-destructor persistence regressions pass; the
  latter reports worker 0, batch 1, no record identity, then poisons the pool so
  the next generation returns `ChannelClosed`, with one source creation total.
- Both large-inline capacity boundaries return typed configuration errors
  before exact-length, factory, initializer, collator, or worker-thread effects.
- Queued low record, queued protocol error, and actual queued panic/End receive
  cases beat an expired deadline; a genuinely non-ready cooperative source still
  returns `Timeout`.
- Barrier-driven first-failure and iterator-drop tests prove cancellation wins
  before credit release and no additional source call begins.
- The focused final data gate passed 80 tests across the library,
  `stream_workers`, `worker_lifecycle`, `map_workers`, `current_api`, and
  `loader_builder`; the root facade gate passed 28 tests.

## Final verification

All commands used the locked development environment, and no final gate reached
the 60-second interruption threshold.

- `cargo check --workspace --all-targets --locked`: passed.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- `cargo test --workspace --all-targets --locked`: 259 passed, 1 ignored across
  27 suites.
- Warning-denied default-feature rusttorch-data doctests: 14 passed.
- Rusttorch-data doc-only all-target check and warning-denied rustdoc: passed.
- Compatibility write/check: generated pages current.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.

The implementation commit is DCO-signed as
`fix(data): harden sharded stream lifecycle`.
