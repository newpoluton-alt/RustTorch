# Task 10 fix re-review — cancellation precedence and disabled storage

Review base: `59559a67fedfceceb14c3a0e4b2231eb9a390ce4`

Product head: `e2922fbcfc8b953e098ae6f39415345966a062b2`

Report-only HEAD: `70f10f185733b320b88aa180cd39a64d74f6124c`

Task 9 comparison base: `e5f99b93ea4f896327b8e66204ee4e32ff78c6ec`

## Finding

### Medium — disabled stream completions still narrow Task 9's exact capacity boundary

The permit and waiter parts of the prior storage finding are fixed: disabled
stream policy uses `Permit = ()` (`crates/rusttorch-data/src/memory.rs:242-260`)
and `Waiters = ()` with zero bytes, no preflight, no construction allocation,
and no access (`crates/rusttorch-data/src/stream.rs:197-228`). The map path also
returns to its exact Task 9 boundary.

The required disabled stream boundary is nevertheless not restored. The result
envelope used by every memory policy still contains the Task 10-only
`StreamFailure::MemoryLimit`, `StreamFailure::Protocol`, and
`StreamMessage::Waiting` variants (`stream.rs:302-330`). In particular, the
`Protocol` variant owns a `String`, enlarging the concrete disabled completion
type relative to Task 9's source/transform/init/panic failures and record/end
messages (`e5f99b9:crates/rusttorch-data/src/stream.rs:194-214`). Capacity
validation multiplies that common completion shape by every outstanding slot
at `stream.rs:399-401`, so the enlargement remains externally visible even
when `prefetch_bytes` is absent.

A read-only public-API binary search used the same empty `u8`, two-worker,
batch-size-one, `VecCollate`, no-`prefetch_bytes` loader shape on both trees.
The exact largest accepted factors were:

| Loader | Task 9 base `e5f99b9` | Fix HEAD `e2922fb` |
| --- | ---: | ---: |
| disabled map | 220,717 | 220,717 |
| disabled stream | 220,717 | 209,681 |

Thus factor `215,000` still builds on Task 9 and is rejected at the fix head.
The new regression only checks factor `200,000`
(`crates/rusttorch-data/tests/memory_budget.rs:77-92`), below both thresholds,
so it proves improvement rather than the report's claimed boundary preservation
(`task-10-fix-report.md:92-100`). Parameterize the byte-only stream
failure/waiting payload as well, or otherwise restore the exact disabled Task 9
completion shape, and lock an actually discriminating base/head boundary.

## Verified fixes and regression review

- The cancellation finding is fixed at its shared root.
  `WorkerRunContext::cancel_with_hook` marks/notifies `ByteBudget` before the
  lifecycle token becomes visible (`crates/rusttorch-data/src/worker.rs:130-145`).
  `ByteBudget::acquire_inner` now holds the same mutex and checks cancellation
  before either fitting admission or the oversize result
  (`crates/rusttorch-data/src/memory.rs:179-211`). The deterministic test pauses
  exactly between those notifications, drops the held permit, proves both a
  fitting and an oversize acquire return `Cancelled`, proves the lifecycle
  observer is still asleep, resumes, and joins both threads
  (`worker.rs:1020-1054`). No post-cancel permit or `MemoryLimit` error can be
  created in the reviewed interleaving.
- Active-generation map and stream cancellation paths converge on
  `WorkerRunContext::cancel`; pool shutdown first cancels an active generation
  before closing its separate shutdown token. No direct lifecycle-first budget
  cancellation remains.
- Enabled map/stream completion and reassembly shapes contain the concrete
  `BytePermit`, and enabled stream waiter accounting uses checked
  `workers * size_of::<Option<u64>>()`, exact preflight, exact reserve, and one
  initialized entry per worker (`stream.rs:231-257,396-453`). Disabled map
  permit storage is zero-sized and its exact base capacity is restored.
- Defaulted public generic arities, disabled custom-type bounds, facade exports,
  `Cell<u8>` stream owner `Sync` coverage, and current API fixtures remain
  green. The new private policy bounds introduce no observed public or
  auto-trait regression. No dependency, unsafe code, unbounded queue, detached
  product thread, correctness sleep/polling loop, or Task 11 work was added.
- The previously documented persistent stream lifecycle flake is outside this
  fix range and was not relitigated or changed.

## Verification

- Deterministic cancellation-precedence unit: 1 passed.
- Focused memory/pin/stream/lifecycle/map/builder/transform/current-API: 112
  passed.
- Root facade `data`: 29 passed.
- Workspace all-target check: passed.
- Workspace Clippy with warnings denied: passed.
- Workspace all-target tests: 290 passed, 1 ignored.
- Workspace doctests: 17 passed.
- Warning-denied rusttorch-data rustdoc: passed.
- Compatibility generated-page check, formatting, and raw `git diff --check`:
  passed.
- The product and report commits both contain `Signed-off-by` trailers. The
  worktree was clean before this permitted review artifact was written.

## Verdict

**CHANGES REQUESTED.** Cancellation precedence and policy-specific
permit/waiter storage are correct, but the disabled stream completion remains
larger than Task 9 and still rejects a real part of the accepted base capacity
range.
