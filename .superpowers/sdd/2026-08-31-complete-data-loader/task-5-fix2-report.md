# Task 5 fix round 2 report — configuration bounds

## Result

Fixed the residual High finding from `task-5-fix-review.md`: every explicit
batch-sampler conflict order now reaches typed build-time validation without
requiring discarded collation or conversion capabilities.

## RED / GREEN

- Added a `CustomSample` that implements neither `DefaultCollate` nor
  `DefaultConvert`, and supports loading only through the documented
  `without_batching().convert(VecCollate)` escape hatch.
- Before production changes, the focused test failed to compile on the exact
  unintended `CustomSample: DefaultCollate` requirement.
- After the fix, the focused loader-builder suite passes 15 tests.

## Root-cause fix

- Split lightweight `LoaderPlanConfiguration` behavior from operational
  `LoaderPlan<Sample, C>` behavior.
- `DataLoaderBuilder::build` now requires only the reusable sampler or batch
  source needed to configure its index plan. It validates all argument
  conflicts and applies checked batch options without evaluating discarded
  collator/converter/error bounds.
- `OwnedDataLoader::iter`, `len`, `set_epoch`, and the other usable loader
  methods retain the full `LoaderPlan` and typed `From<Dataset::Error>` bounds.
  Valid loaders therefore remain statically typed, while a configuration that
  returns `Err` cannot execute.
- The regression audits both method orders for explicit batch sampler with
  batch size, sampler, shuffle, drop-last, and no-batching/custom conversion.

## Verification

- Focused loader builder: 15 passed.
- Root facade data suite: 26 passed.
- All rusttorch-data tests: 74 passed.
- Workspace all targets: 180 passed, 1 ignored.
- rusttorch-data doctests: 10 passed.
- Workspace Clippy and rustdoc with warnings denied: clean.
- Formatting, compatibility generation/check, and diff checks: clean.

## Concerns

No correctness concern remains from this review. The configuration-only trait
is public but doc-hidden because it appears in a public builder method bound;
it exposes no execution capability. Positive-worker execution remains the
intentional, typed Task 7 deferral.
