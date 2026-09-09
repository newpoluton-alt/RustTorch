# Task 9 third fix report — preserve stream-loader `Sync`

Product commit: `83415cab80fbc547aca86cc7e3518e2179fde3fd`

Review addressed: `.superpowers/sdd/2026-08-31-complete-data-loader/task-9-fix2-review.md`

## Result

The final medium auto-trait regression is fixed without changing any public
named API or runtime protocol. `StreamDataLoader` retains the exact
capacity-revalidated flat reassembly vector behind a private
`std::sync::Mutex`. `Mutex<T>` is `Sync` when `T: Send`, so a legal final output
that is `Send` but not `Sync` no longer removes `Sync` from the public loader.

The mutex is an ownership wrapper only. `iter(&mut self)` already has exclusive
loader access and obtains the vector through `Mutex::get_mut`; there is no
runtime lock acquisition, contention, or sharing of reassembly records. The
`LockResult` is handled without `unwrap`, `expect`, or a new error/panic path:
exclusive access safely recovers the inner vector from `PoisonError` with
`into_inner`.

## RED and GREEN evidence

A public `current_api` regression defines a zero-sized `CellFactory` whose
sample and identity-transform output are `std::cell::Cell<u8>`. `Cell<u8>` is
`Send` but not `Sync`, and its source is an ordinary empty worker iterator. The
compile-time assertion is:

```rust
fn assert_sync<T: Sync>() {}
assert_sync::<StreamDataLoader<CellFactory, VecCollate>>();
```

At the reviewed head this failed with compiler error E0277: `Cell<u8> cannot be
shared between threads safely`. The diagnostic traced the extra requirement
through `WorkerRecord<Cell<u8>>`, `BufferedRecord<Cell<u8>>`, and the directly
stored reassembly `Vec` into `StreamDataLoader`.

After wrapping the retained vector, the exact compile-time assertion passes.
The generated rustdoc again contains the
`impl-Sync-for-StreamDataLoader<S,C,F,I>` auto-trait implementation. No new
`Sync` bound was added to the builder, source, transform, iterator, or collator.
Compatibility evidence now records this public compile-time contract.

## Allocation and runtime proof

The allocation fix remains unchanged:

- Build computes the requested flat-entry aggregate, reserves the real
  `Vec<ReassemblyEntry<Output>>`, reads its allocator-returned capacity, and
  validates that actual capacity in the complete aggregate before `exact_len`
  or worker/source callbacks.
- Moving the already allocated vector into `Mutex::new` does not replace,
  resize, or reallocate its heap buffer.
- Each iterator obtains that same vector through exclusive `get_mut`, clears it
  without shrinking, reads the same retained capacity for worker-pool
  validation, and lends it to the coordinator for the generation.
- The shared `ReassemblyEntry<T>` alias is used by both `size_of` capacity
  accounting and iterator storage, preventing validation/storage shape drift.
- Linear exact-sequence lookup, `swap_remove`, duplicate/past/gap checks,
  full-window progress, minimum-sequence diagnostics, cancellation-first
  draining, one-error-then-`None`, and persistent lifecycle behavior are
  unchanged.

The 6 MiB single-slot vector boundary remains accepted safely, and the 17 MiB
single-slot concrete aggregate remains rejected before every callback. The
product code performs no `lock` or `try_lock` call, returns to no tree/layout
proxy, and changes no allocation arithmetic.

## Verification

All commands used the locked development environment. No command or test
reached the 60-second interruption threshold.

- Exact `Cell<u8>` public compile-time regression: passed.
- Exact one-entry allocation boundaries: 2 passed.
- Rusttorch-data library plus full stream suite: 30 passed.
- Focused library/stream/lifecycle/map/API gate: 83 passed.
- Root facade data gate: 28 passed.
- `cargo check --workspace --all-targets --locked`: passed.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- `cargo test --workspace --all-targets --locked`: 262 passed, 1 ignored across
  27 suites.
- Warning-denied rusttorch-data doctests: 14 passed.
- Rusttorch-data doc-only all-target check and warning-denied rustdoc: passed.
- Compatibility write/check: generated pages current.
- `cargo fmt --all -- --check`: passed.
- Raw `git diff --check`: passed.

The product commit is DCO-signed as `fix(data): preserve stream loader sync`.
No dependency, unsafe block, unbounded storage, detached worker, correctness
sleep/poll loop, Task 10 work, or public behavioral expansion was introduced.
