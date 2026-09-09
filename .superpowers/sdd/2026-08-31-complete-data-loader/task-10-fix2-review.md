# Task 10 second-fix re-review — exact stream storage

Review base: `629eb94a388de54be76e52cd4228cb1149dbc075`

Product head: `f920abf2047c8622a802da74ab67d98c44631977`

Report-only HEAD: `d9688eb370cccc58e94d0d9ed92ce9ab1bd5c301`

Task 9 comparison base: `e5f99b93ea4f896327b8e66204ee4e32ff78c6ec`

## Findings

No material findings.

## Review

- Disabled stream storage is now selected completely by policy. In addition to
  the already-zero-sized permit and waiter state, `MemoryDisabled` supplies
  `Infallible` for both byte-waiting and byte-failure payloads, returns no
  constructors for them, and has unreachable consumers
  (`crates/rusttorch-data/src/stream.rs:209-280`). Therefore the disabled
  concrete `StreamMessage::Waiting` and `StreamFailure::Memory` variants are
  uninhabited and add no result-slot storage; no `BytePermit`, front-waiter
  vector, `MemoryLimit`, protocol `String`, or waiting sequence can be stored or
  allocated.
- Enabled streams retain the exact runtime payloads: `BytePermit`,
  `Vec<Option<u64>>`, `u64` waiting sequence, and `StreamByteFailure` with exact
  memory-limit/protocol data (`stream.rs:198-207,282-348`). Worker paths create
  those typed messages only inside enabled footprint accounting
  (`stream.rs:1196-1353`), while coordinator conversion preserves the existing
  public `MemoryLimit` metadata and `StreamProtocol` reason/sequence
  (`stream.rs:2367-2411,2632-2634`). Existing oversize, missing/nonmonotonic ID,
  unordered metadata, delayed-low, timeout, early-drop, and persistent reuse
  tests remain green.
- `StreamCompletionFor` selects permit, waiting, and failure payloads from the
  same memory policy (`stream.rs:383-427`). The checked 64 MiB calculation and
  channel preflight both use that exact alias, reassembly uses the same
  policy-selected permit, and enabled waiter bytes/preflight are added
  separately (`stream.rs:492-550`). Validation therefore charges the concrete
  runtime shape rather than a mirror or approximation.
- An independent read-only public-builder binary search confirmed the requested
  boundaries exactly:

  | Shape | Maximum accepted | First rejected |
  | --- | ---: | ---: |
  | disabled map | 220,717 | 220,718 |
  | disabled stream | 220,717 | 220,718 |
  | byte-enabled stream | 182,331 | 182,332 |

  The checked-in regression locks the same discriminating public boundaries for
  disabled map/stream and enabled stream
  (`crates/rusttorch-data/tests/memory_budget.rs:77-159`). The disabled values
  exactly match the accepted Task 9 base.
- The prior cancellation fix remains intact: centralized cancellation marks and
  notifies the byte budget before lifecycle visibility
  (`crates/rusttorch-data/src/worker.rs:130-145`), and acquisition checks the
  mutex-protected cancelled state before fitting admission or oversize
  (`crates/rusttorch-data/src/memory.rs:179-227`). Its deterministic
  between-notifications test passes, so no post-cancel permit or byte error is
  reintroduced.
- The change is private and policy-monomorphized. Existing defaulted public
  arities, disabled trait-bound behavior, facade exports, typed errors, pinning,
  and `Cell<u8>` `Send + !Sync` output behind a `Sync` stream owner remain
  unchanged. No dependency, unsafe block, unbounded queue, detached product
  thread, correctness sleep/polling loop, or Task 11 code appears in the range.
  The previously documented baseline stream flake was outside scope and was not
  relitigated.

## Verification

- Deterministic cancellation-precedence unit: 1 passed.
- Focused memory/pin/stream/lifecycle/current-API: 68 passed.
- Root facade `data`: 29 passed.
- Workspace all-target check: passed.
- Workspace all-target Clippy with warnings denied: passed.
- Workspace all-target tests: passed (290 passed, 1 ignored).
- Workspace doctests: 17 passed.
- Warning-denied rusttorch-data rustdoc: passed.
- Compatibility generated-page check, formatting, and raw `git diff --check`:
  passed.
- Product commit `f920abf2047c8622a802da74ab67d98c44631977` and report commit
  `d9688eb370cccc58e94d0d9ed92ce9ab1bd5c301` both contain `Signed-off-by`
  trailers. The worktree was clean before this permitted review artifact.

## Verdict

**APPROVED.** The residual disabled stream capacity regression is fixed at the
concrete completion type, enabled storage remains exactly charged and
functional, and the earlier cancellation correction remains intact.
