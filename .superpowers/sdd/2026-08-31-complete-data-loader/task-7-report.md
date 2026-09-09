# Task 7 implementation report — bounded ordered map workers

Base: `1670be3f367035f00afadf91d4868fb73fa6e711`

## Result

Implemented positive map-worker execution with checked global credits, one
bounded `crossbeam-channel` task lane per worker, one bounded result channel,
deterministic `batch_sequence % workers` routing, generation and logical
sample identities, ordered reassembly by default, and explicit completion
order through `in_order(false)`/`ordered(false)`.

Workers share the owned map dataset through `Arc`, enter a real `WorkerInfo`
scope, run the initializer and transform factory once per worker, fetch with
`Dataset::get_batch`, validate cardinality, and transform with task-local RNG
context. Collation remains on the coordinator. The first typed stage failure
or converted worker panic is yielded once; submission stops and iterator drop
disconnects channels before joining every worker.

The additive `SerialExecution`/`WorkerExecution` capability markers keep the
default zero-worker API usable with non-`Send`/non-`Sync` datasets and
transforms while applying thread-safety bounds only after `.workers(...)`.

## TDD evidence

RED was captured before production changes:

```text
cargo test -p rusttorch-data --test map_workers
FAILED: positive_workers_fetch_batches_concurrently_on_distinct_threads
Configuration(InvalidConfiguration { field: "workers", reason: "positive-worker execution is scheduled for DataLoader Task 7" })
```

GREEN uses event synchronization only (barriers, condition variables, and
channels; no sleeps) and proves:

- simultaneous entry on two distinct worker threads;
- controlled ordered and completion-order delivery;
- deterministic modulo routing across fresh generations;
- default and explicit checked outstanding-work bounds;
- one real factory/initializer call per worker, consecutive worker seeds,
  fresh generation seeds, and correct task contexts;
- typed dataset, transform, collator, factory, initializer, cardinality, and
  dataset/transform/factory/initializer panic behavior with fail-once metadata;
- preservation of local serial datasets/transforms; and
- saturated early-drop disconnect and worker joins.

## Dependency and documentation

Added locked `crossbeam-channel 0.5.16` (`MIT OR Apache-2.0`) through workspace
inheritance. `Cargo.lock` checksum and `THIRD_PARTY_NOTICES.md` inventory are
current. Rustdoc, package/root READMEs, canonical compatibility scope, and both
generated coverage documents describe shared Rust-thread map datasets,
coordinator collation, deterministic local seeds with no Philox parity, and
the remaining scheduled capabilities.

## Verification

Fresh final gates:

- `cargo test -p rusttorch-data --test map_workers`: 11 passed;
- `cargo test -p rusttorch-data --test loader_builder --test transform --test worker_context`: 26 passed;
- `cargo test -p rusttorch --test data_libtorch`: 1 passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed;
- `cargo clippy --workspace --all-targets -- -D warnings`: passed;
- `cargo test --workspace --all-targets`: 204 passed, 1 ignored;
- `cargo test -p rusttorch-data --doc`: 12 passed;
- rusttorch-data rustdoc with warnings denied: passed; and
- `git diff --check`: passed.

## Self-review and remaining scope

Reviewed boundedness, generation filtering, typed source preservation,
fail-once transitions, disconnect order, and every positive-worker caller.
No unsafe code, detached threads, unbounded queues, worker collation, serial
fallback, stream locking, or early pinning/lifecycle implementation was added.

Per the controller rulings, nonzero timeout and persistent-worker requests
still reject before sampler/factory/initializer/thread side effects until Task
8. Pinning remains stored and visibly unapplied until Task 10. Task 7 is
item-bounded; byte permits remain Task 10. Dropping waits for a currently
non-cooperative dataset/native call to return, as documented for Task 8.
