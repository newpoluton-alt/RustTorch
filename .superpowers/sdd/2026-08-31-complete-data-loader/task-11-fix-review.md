# Task 11 callback-order fix review

**Verdict: CHANGES REQUESTED**

Reviewed fix range:

- review base: `93159cbc78ecd6fefc512340c03d0b32347ae9dd`
- product fix: `999eea2fb8609776efe3b0dcebe984ce64b9b88d`
- fix-report head: `12d507172b0ddc801d79776e7a50b1c14e04f71a`

The product fix and report commits both contain the required
`Signed-off-by: newpoluton-alt <newpoluton@gmail.com>` trailer. The worktree was
clean before this review artifact was added.

## Finding

### [P1] Plan-identity corruption still invokes the callbacks needed to reject it

The new static-envelope phase correctly rejects its 14 covered fields before
plan callbacks. However, resume then calls
`checkpoint_identity_with_batch_options` at
`crates/rusttorch-data/src/loader.rs:1401-1403`. `AutoBatch` implements that
method by invoking the public custom sampler's `kind()` and
`distributed_configuration()` methods at
`crates/rusttorch-data/src/loader.rs:338-348`. Only afterward does
`validate_plan_loader_state_envelope` compare replicas/distributed policy and
sampler kind at `crates/rusttorch-data/src/loader.rs:3144-3168`.

Consequently, corrupting either `LoaderConfiguration::sampler_kind` or a
distributed identity field executes downstream plan callbacks before returning
the expected error. This leaves the explicit mandatory evidence incomplete:
`.superpowers/sdd/2026-08-31-complete-data-loader/task-11-brief.md:272-273`
requires wrong kind/config/world state to reject before callbacks. No exemption
for plan-identity methods was ruled, and the original independent review
required stable sampler/distributed identity to be available without an
observable downstream callback.

I reran an external path-dependent adversarial program without changing product
files. Its public custom `SamplerCheckpoint` independently counted `kind()` and
`distributed_configuration()` calls. Starting from one valid checkpoint, it
tested two otherwise unchanged states: one with a corrupted `sampler_kind`, and
one with a corrupted `replicas` value. Both builds returned `Err`, but the
strict zero-callback assertion failed with:

```text
left: [("kind", 1, 1), ("distributed", 1, 1)]
right: [("kind", 0, 0), ("distributed", 0, 0)]
```

The checked-in test at
`crates/rusttorch-data/tests/checkpoint_map.rs:779-849` cannot expose this
residual because its 14-case table stops at static fields and does not corrupt
`sampler_kind`, `replicas`, or distributed policy.

Provide sampler kind and distributed identity as non-callback metadata (or an
equivalent closed/static mechanism) so plan-identity mismatches can be rejected
without invoking downstream methods. Extend the checked-in observable-counter
regression with sampler-kind plus distributed replicas/shuffle/seed/drop-policy
corruptions. The required observation for every rejected case is zero kind,
distributed-configuration, epoch-apply, and cursor-apply calls.

## Confirmed improvements

The scoped diff otherwise addresses the first review correctly:

- all 14 static corruptions reject with kind, distributed, epoch-apply, and
  cursor-apply counters at zero;
- builder batch size and drop-last are passed into read-only identity/cursor
  validation instead of mutating `AutoBatch` first;
- `apply_batch_options` now runs only after dataset, plan/cursor, transform, and
  effective coordinator validation have all succeeded;
- the defaulted additions to hidden checkpoint-plan plumbing preserve existing
  implementations, generic arities, and disabled/fresh loader behavior;
- explicit batching and `NoBatch` retain their prior identity and effective
  coordinator paths; and
- the generic resume-build error and both source chains are untouched.

## Verification

Fresh local verification for this scoped review:

- checked-in 14-case static-envelope test: **passed**;
- original malformed-schema external probe: **passed** after the fix;
- external sampler-kind/distributed corruption probe: **failed the required
  zero-callback assertion**, observing `(1, 1)` for both cases;
- complete `checkpoint_map`, `current_api`, and `loader_builder` suites:
  **35 passed**;
- `cargo check --workspace --all-targets --locked`: **passed**;
- `cargo fmt --all --check`: **passed**;
- raw fix-range `git diff --check`: **passed**.

Long workspace gates were not redundantly rerun; the signed fix report records
their passing results. Approval remains blocked only on plan-identity corruption
executing observable downstream callbacks before rejection.
