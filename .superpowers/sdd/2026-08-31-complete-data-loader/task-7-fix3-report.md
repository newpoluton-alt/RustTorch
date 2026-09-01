# Task 7 final review-fix report — crossbeam control-block accounting

Implementation under review: `a31e64c4930348903be5e8702b14db82733ad4b4`

## Result

Closed the remaining worker-heavy allocation gap without constructing an
iterator, changing execution semantics, or broadening into Task 8.

The shared build/`WorkerPool` validator now includes a checked conservative
allowance for every heap control block allocated by locked
`crossbeam-channel 0.5.16`: one task channel per worker plus one result
channel. Build therefore rejects the demonstrated 250,000-worker/factor-one
configuration before sampler or batch-source callbacks. A four-worker,
factor-one configuration remains accepted.

## Accounting formula

For `workers = W`, `prefetch_factor = F`, and checked global credit count
`O = W * F`, the validator computes:

```text
slot_bytes = O * (
    size_of(ChannelSlot<WorkerTask>)
    + size_of(ChannelSlot<Completion<D, F, I>>)
)

worker_vector_bytes = W * (
    size_of(WorkerInfo)
    + size_of((Sender<WorkerTask>, Receiver<WorkerTask>))
    + size_of(Sender<WorkerTask>)
    + size_of(JoinHandle)
)

fixed_endpoint_bytes =
    4 * size_of(Vec header)
    + size_of(Sender<Completion<D, F, I>>)
    + size_of(Receiver<Completion<D, F, I>>)

control_block_bytes = (W + 1) * 2,048

aggregate = slot_bytes
    + worker_vector_bytes
    + fixed_endpoint_bytes
    + control_block_bytes
```

Every caller-influenced multiplication and addition is checked, and the one
aggregate must not exceed 67,108,864 bytes (64 MiB). The result and task slot
sizes still instantiate the actual generic payload/error types.

The 2 KiB control allowance is deliberately an upper bound tied to the locked
crossbeam implementation. Its `Counter<array::Channel<T>>` contains two
`CachePadded<AtomicUsize>` values; crossbeam-utils uses at most 256-byte cache
alignment across its supported target cfgs. After the 512-byte padded-atomic
total, 1,536 bytes remain for counter fields, the buffer descriptor, ring
metadata, two empty `SyncWaker`s, allocator/alignment overhead, and padding.

`WorkerPool::new` calls this same validator defensively, then retains fallible
slot/vector reservation and pre-thread channel construction.

## RED and GREEN evidence

The RED regression was build-only and finished in 0.00 seconds:

```text
channel_control_blocks_are_counted_before_many_worker_callbacks ... FAILED
assertion failed: 250,000 workers with factor 1 returned Ok
```

It used a custom sampler `set_epoch` counter. The falsely accepted build
reached the callback, while the expected contract was a typed configuration
error with zero callbacks. It never called `iter()` or constructed a channel.

GREEN finished in 0.00 seconds: 250,000 workers now return
`InvalidConfiguration` with zero callbacks, while four workers/factor one
build successfully and invoke `set_epoch` exactly once. The complete
map-worker suite has 22 passing tests.

## Documentation and claim audit

Updated `prefetch_factor` rustdoc, the package README, canonical compatibility
scope, and generated compatibility/coverage documents to call the 64 MiB
calculation a conservative aggregate/upper-bound allowance rather than an
exact measurement of private crossbeam control types. The prior fix2 report
now carries a correction and uses the same final wording.

The byte ceiling covers eager transport and coordinator bookkeeping. It does
not claim arbitrary sample payload byte permits; those remain Task 10.

## Verification

Fresh final-state gates:

- `cargo test -p rusttorch-data --test map_workers`: 22 passed;
- `cargo test -p rusttorch-data --test loader_builder --test transform --test worker_context`: 26 passed;
- `cargo test -p rusttorch --test data_libtorch`: 1 passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed;
- `cargo clippy --workspace --all-targets -- -D warnings`: passed;
- `cargo test --workspace --all-targets`: 215 passed, 1 ignored;
- `cargo test -p rusttorch-data --doc`: 12 passed;
- warning-denied rusttorch-data rustdoc: passed; and
- `git diff --check`: passed.

## Self-review and remaining scope

Reviewed every checked product/sum, the `workers + 1` channel count, exact
generic slot sizing, build-before-callback ordering, and build/WorkerPool
validator parity. The nearby accepted case guards against a blanket worker
rejection. All prior exact panic stages, typed sources, deterministic routing,
bounded credits, zero-worker local API, and disconnect/join behavior remain
unchanged.

No unsafe code, unbounded queue, sleep-based synchronization, detached thread,
worker collation, serial fallback, pinning behavior, or Task 8 lifecycle
implementation was added. No remaining Task 7 review blocker is known.
