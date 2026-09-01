# Task 10 second fix report — exact disabled stream completion storage

Implementation fix commit: `f920abf2047c8622a802da74ab67d98c44631977`

Reviewed base: `629eb94`

## Outcome

The residual Task 10 storage finding is fixed at the concrete completion type.
Byte-disabled streams no longer carry storage for Task 10-only memory-limit,
byte-protocol, or ordered byte-waiting payloads. Their public capacity boundary
now exactly matches the accepted Task 9 two-worker `u8`, batch-size-one shape:
factor `220,717` builds and factor `220,718` is rejected. The corresponding
disabled map boundary remains exactly the same.

Byte-enabled streams retain the concrete permit, waiter, memory-limit,
protocol-string, and waiting-sequence storage used by their behavior. Their
actual monomorphized boundary is separately locked at factor `182,331` accepted
and `182,332` rejected.

No Task 11 code or pre-existing stream lifecycle change appears in this range.

## RED-first evidence

The old factor-`200,000` regression was replaced before production changes by
the literal Task 9 boundary for both public loaders. On the reviewed head, the
map assertions passed but the disabled stream failed at factor `220,717`:

```text
aggregate stream queue storage requires 70640100 bytes,
above the 67108864-byte safety ceiling
```

That was the intended RED. The adversarial re-review's independent binary
search measured the old disabled stream maximum as `209,681`, versus `220,717`
on Task 9. Thus the new regression distinguishes the base from the faulty head
and would fail if the unconditional Task 10 payload shape returned.

## Root cause and exact storage proof

The first fix correctly made permits and front-waiter vectors policy-specific,
but every result still used one shared `StreamFailure` containing direct
`MemoryLimit` and `Protocol { reason: String, .. }` variants and one shared
`StreamMessage` containing `Waiting { sequence }`. Capacity validation used
`size_of` on that enlarged common completion for every policy, so disabled
streams paid for enabled-only result payloads.

The stream memory policy now supplies two additional associated payloads:

- `MemoryDisabled` uses `Infallible` for both the byte-waiting message and the
  byte-only failure payload. Those variants are uninhabited in the disabled
  concrete enums and therefore occupy no completion slot storage. Disabled
  construction methods return `None`, so no enabled-only waiter, permit,
  memory-limit, protocol string, or sequence payload can be allocated or
  stored.
- `MemoryEnabled` uses `u64` for the waiting sequence and a typed
  `StreamByteFailure` for exact `MemoryLimit` and `Protocol` data. Its existing
  `BytePermit` completion/reassembly payload and `Vec<Option<u64>>` front
  waiters remain unchanged. Typed policy conversion preserves exact metadata
  and maps back to the same public loader errors.

`StreamCompletionFor<S, F, I, M>` names all three policy-selected pieces:
permit, waiting message, and byte-only failure. The 64 MiB calculation and the
result-channel preflight both use that exact alias, as does the allocated
receiver/sender and worker/coordinator path. Validation therefore measures the
actual monomorphized slot rather than undercharging a larger runtime envelope.

The public boundary proofs are:

| Shape | Maximum accepted | First rejected |
| --- | ---: | ---: |
| Task 9 disabled map | 220,717 | 220,718 |
| fixed disabled map | 220,717 | 220,718 |
| Task 9 disabled stream | 220,717 | 220,718 |
| reviewed disabled stream before this fix | 209,681 | 209,682 |
| fixed disabled stream | 220,717 | 220,718 |
| fixed byte-enabled stream | 182,331 | 182,332 |

The enabled boundary is lower because it exactly includes the permit,
byte-failure/waiting variants, and per-worker front-waiter state required for
strict accounting. The disabled and enabled tests exercise the same public
builders and safety ceiling; they do not inspect or mirror private size
helpers.

## Preserved contracts

All reviewed enabled semantics remain intact: post-transform byte accounting,
ordered admission and monotonic shard enforcement, exact `MemoryLimit` and
protocol metadata, one error followed by `None`, bounded liveness, permit
release, and persistent generations. The earlier cancellation-precedence fix
is unchanged.

Defaulted public generic arities, disabled custom-type bounds, final-batch
pinning bounds, facade exports, pin status and validation, source chains, and
the Task 9 `Send + !Sync` output behind a `Sync` stream owner remain green.
There is no new public type, dependency, unsafe block, unbounded queue,
detached worker, correctness sleep, or polling loop.

The persistent stream lifecycle flake documented as reproducible on the Task 9
base did not appear in these gates. It was not modified or relitigated.

## Fresh GREEN verification

Every Rust command sourced `. scripts/dev-env.sh` and used locked dependencies.

- exact disabled and enabled public capacity regressions: passed;
- focused memory/pin/stream/lifecycle/current-API: 68 passed;
- root facade `data`: 29 passed;
- full locked workspace check: passed;
- workspace all-target Clippy with `-D warnings`: passed;
- workspace all-target tests: 290 passed, 1 ignored;
- workspace doctests: 17 passed;
- rusttorch-data doc-only all-target check and warning-denied rustdoc: passed;
- compatibility write/check: current with no generated diff;
- formatting check and raw `git diff --check`: passed; and
- implementation commit diff and DCO trailer: passed.

The implementation fix commit is DCO-signed.
