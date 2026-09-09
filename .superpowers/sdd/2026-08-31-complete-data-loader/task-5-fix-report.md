# Task 5 fix report — independent review

## Result

Addressed all three High findings from `task-5-review.md` in a scoped follow-up.

## RED / GREEN

- Added the three regressions before production changes.
- Focused RED reproduced the order-dependent compile failure:
  `without_batching().batch_sampler(...).build()` required `(): Collate<i64>`.
- Focused GREEN: `cargo test -p rusttorch-data --test loader_builder` passes
  14 tests, including all three new regressions.

## Fixes

1. Owned-loader build now uses the sampler module's shared fallible batch-size
   validation, including the existing allocation/capacity probe. `usize::MAX`
   returns typed `InvalidConfiguration { field: "batch_size", .. }` inside
   `catch_unwind` without panicking. `AutoBatch::iter` uses a crate-private
   validated `BatchSampler` constructor and contains no `expect`.
2. Positive-worker `OwnedDataLoader::iter` does not create a plan iterator.
   The optional plan iterator is created only for serial loading, so a
   panicking `FnSampler` factory remains untouched before the required single
   typed unsupported error is yielded.
3. `NoBatch` implements `LoaderPlan<Sample, C>` for any collator type it
   intentionally ignores. `without_batching`, no-batch sampler/shuffle, and
   custom conversion preserve `C`; both no-batching/batch-sampler method orders
   now compile and return the same typed `batch_sampler` conflict.

## Verification

- Focused loader builder: 14 passed.
- Root facade data suite: 26 passed.
- All rusttorch-data tests: 73 passed.
- Workspace all targets: 179 passed, 1 ignored.
- rusttorch-data doctests: 10 passed.
- Workspace Clippy with warnings denied: clean.
- rustdoc with warnings denied: clean.
- Formatting, compatibility generation/check, and `git diff --check`: clean.

## Self-review and concerns

- The capacity proof and `BatchSampler::new` now share exactly one validation
  function, avoiding divergent builder/sampler limits.
- Positive-worker rejection remains deliberately unsupported until Task 7,
  but now performs no sampler work or side effect before that rejection.
- No unsafe code, new dependency, string-erased error, worker execution, or
  unrelated refactor was introduced.
