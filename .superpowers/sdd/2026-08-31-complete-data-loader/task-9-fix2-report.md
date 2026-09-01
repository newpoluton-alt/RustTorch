# Task 9 second fix report — concrete ordered reassembly storage

Product commit: `bcff3197f901e54333b463e9cfcb24cbee2d91c3`

Review addressed: `.superpowers/sdd/2026-08-31-complete-data-loader/task-9-fix-review.md`

## Result

The final high finding is fixed at its storage root. Ordered reassembly no
longer uses `BTreeMap` or any proxy for private standard-library node layout.
Each loader owns one bounded
`Vec<(u64, BufferedRecord<TransformOutput>)>` reserved before `exact_len` or
any worker/source callback. The allocator-returned capacity is read back and
the complete aggregate is validated again. Iterator generations borrow, clear,
and reuse that exact allocation; they never preflight one shape and later
allocate another reassembly shape.

Ordered lookup and duplicate detection scan the bounded vector, exact-sequence
removal uses `swap_remove`, and terminal/full-window diagnostics take the
minimum buffered sequence. These operations preserve deterministic sequence
delivery, duplicate/past/gap semantics, one-error-then-`None`, full-window
progress detection, cancellation-before-credit-release, and allocation-free
cleanup. Linear work is explicitly bounded by `workers * prefetch_factor` and
documented as the memory/scan tradeoff for wider intentional disorder.

## RED evidence and corrected boundary

The first exact regression used the requested ordered one-worker, factor-one,
batch-one, inline `[u8; 6 * 1024 * 1024]` configuration and expected a typed
pre-callback rejection. Against the reviewed implementation it failed at that
assertion because the flat allowance accepted the configuration, while the
real `BTreeMap` first insertion could privately reserve eleven inline values.
That is the reviewed defect: the accepted configuration could allocate a
roughly 66 MiB tree leaf after callbacks despite the 64 MiB promise.

The expectation was then corrected for the root fix. A genuine one-slot vector
does not carry the tree's private eleven-value penalty: one result slot, one
reassembly tuple, and two batch buffers are roughly four 6 MiB payload slots,
comfortably below 64 MiB. Keeping an artificial 11x charge after removing the
tree would be a false rejection tied to a private implementation that no longer
exists. The final sparse regression therefore proves that this 6 MiB one-slot
configuration is accepted by the concrete vector model, with only `exact_len`
called during build and no factory, initializer, collator, source, or thread
callback.

A second one-entry regression uses inline `[u8; 17 * 1024 * 1024]`. Its concrete
result slot plus retained reassembly tuple plus two simultaneous batch vectors
already exceed 68 MiB before fixed channel/bookkeeping allowances. It returns
`InvalidConfiguration { field: "prefetch_factor", .. }` before `exact_len`,
factory, initializer, collator, source, or persistent-worker side effects.

## Allocation proof

For ordered mode, build performs these steps before the first user callback:

1. Check the requested `workers * prefetch_factor` flat-entry aggregate so an
   obviously oversized configuration is rejected without allocating it.
2. `try_reserve_exact` that many real `(sequence, buffered-record)` entries in
   the loader-owned vector.
3. Read the actual `Vec::capacity()` returned by the allocator and re-run the
   aggregate calculation using
   `capacity * size_of::<(u64, BufferedRecord<Output>)>()`.
4. Store that same vector in the loader and lend it to every iterator
   generation after `clear`; no ordered insertion can exceed the logical credit
   window, so `push` stays within the retained allocation.

The checked simultaneous aggregate remains the concrete bounded result ring,
the actual retained ordered vector, both full batch vectors, control/credit
rings, worker bookkeeping, and channel control-block allowances. Unordered
mode retains a zero-capacity reassembly vector. Close and drop pop entries in
place only after cancellation, so cleanup adds no proportional allocation.

## Verification

All commands used the locked development environment. No command or test
reached the 60-second interruption threshold.

- Exact one-entry boundaries: 2 passed.
- Rusttorch-data library plus full stream suite: 30 passed.
- Focused library/stream/lifecycle/map/API gate: 82 passed.
- Root facade data gate: 28 passed.
- `cargo check --workspace --all-targets --locked`: passed.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- `cargo test --workspace --all-targets --locked`: 261 passed, 1 ignored across
  27 suites.
- Warning-denied rusttorch-data doctests: 14 passed.
- Rusttorch-data doc-only all-target check and warning-denied rustdoc: passed.
- Compatibility write/check: generated pages current.
- `cargo fmt --all -- --check`: passed.
- Raw `git diff --check`: passed.

The product commit is DCO-signed as
`fix(data): bound ordered stream reassembly`. No public API or dependency
changed, and no Task 10 work, unsafe code, unbounded storage, detached worker,
or correctness sleep/polling loop was introduced.
