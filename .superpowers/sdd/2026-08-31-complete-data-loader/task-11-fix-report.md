# Task 11 fix report — reject static checkpoints before plan callbacks

Product fix commit: `999eea2fb8609776efe3b0dcebe984ce64b9b88d`

## Finding addressed

The independent Task 11 review found that resumed builds called
`apply_batch_options` and then the public custom-sampler identity callbacks
`SamplerCheckpoint::kind` and `SamplerCheckpoint::distributed_configuration`
before rejecting a malformed static checkpoint envelope. A bad schema therefore
returned an error only after running downstream code, contrary to Task 11's
validate-before-mutate transaction.

## RED-first reproduction

Before changing production code, `checkpoint_map.rs` gained a public custom
`SamplerCheckpoint` implementation whose kind lookup, distributed-configuration
lookup, epoch apply, and cursor apply are all observable through independent
counters. The test creates one valid checkpoint, corrupts one static field at a
time, constructs the complete resume builder, resets every counter immediately
before `build`, and requires both an error and four zero counts.

The initial focused command was:

```text
cargo test -p rusttorch-data --test checkpoint_map static_envelope_rejects_before_public_plan_callbacks_or_mutation --locked
```

It failed on the first schema-only case exactly at the missing guarantee:

```text
assertion `left == right` failed: schema: sampler kind callback
  left: 1
 right: 0
```

The checked-in regression covers 14 independently corrupted static fields:
schema, dataset identity, iterator generation, both derivation versions,
automatic batch size, drop-last, workers, prefetch, ordering, loader seed,
task rank, pin request, and effective pin status.

## Root fix and validation order

Resume now has two explicit envelope phases.

1. Existing builder and effective-pin validation runs first.
2. A static validator checks every value available without invoking a plan or
   sampler: schema, exact identity, generation, derivation versions,
   type-state-derived automatic/explicit/no-batch configuration, serial worker
   and prefetch settings, ordering, loader seed, task rank, pin request, and
   effective pin status.
3. Only after that phase succeeds does resume obtain the sampler kind and
   distributed identity, then validate replicas, distributed policy, and kind.
4. Dataset state is validated, followed by plan/cursor state. Automatic batch
   boundary validation receives the builder's proposed batch settings as a
   read-only input; it no longer depends on mutating the plan first.
5. The one retained transform is created and validated, then the effective
   coordinator is validated.
6. Only after every envelope and component accepts the state does resume apply
   dataset state, automatic batch options plus sampler epoch/cursor, transform
   state, and effective coordinator state.

The hidden checkpoint-plan plumbing gained defaulted, additive read-only hooks
for proposed batch settings and a defaulted non-observable batch-mode constant.
Existing plan behavior remains the default; `AutoBatch` alone overrides the
metadata and proposed-setting validation. Fresh checkpoints, disabled loaders,
explicit batches, no-batch conversion, generic factory error sources, pending
cursor epoch replacement, and Task 11 bounds/signatures retain their prior
external behavior.

## GREEN evidence and callback counts

The adversarial test is green for every corruption. For each of all 14 cases,
the observed counts immediately after the rejected build are exactly:

```text
kind callbacks:                 0
distributed-config callbacks:  0
epoch applies:                  0
cursor applies:                 0
```

The existing transaction regression remains green: final-coordinator
validation failure performs all component validations with zero applies, while
a valid resume applies dataset, sampler/cursor, transform, and coordinator once
each. The full checkpoint test now passes 18 of 18 cases.

## Final verification

Every Rust command sourced `scripts/dev-env.sh` and used locked dependencies.

- adversarial regression plus complete `checkpoint_map`: 18 passed;
- relevant `loader_builder` and `current_api`: 17 passed;
- root `data` facade: 30 passed;
- `cargo check --workspace --all-targets --locked`: passed;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed;
- `cargo test --workspace --all-targets --locked`: passed (309 passed, one
  intentionally ignored Python-parity dispatch test);
- `cargo test --workspace --doc --locked`: 24 passed, including nine expected
  compile-fail bounds examples;
- warning-denied `rusttorch-data` doc-only rustdoc: passed;
- compatibility ledger/generated-document check: passed;
- `cargo fmt --all --check` and raw `git diff --check`: passed.

No manifest, package metadata, dependency, or packaged-file selection changed,
so package-list and workspace-archive gates were deliberately not rerun under
the scoped review-fix instruction.

Commit `999eea2fb8609776efe3b0dcebe984ce64b9b88d` has subject
`fix(data): validate checkpoints before plan callbacks` and the required
`Signed-off-by: newpoluton-alt <newpoluton@gmail.com>` DCO trailer.
