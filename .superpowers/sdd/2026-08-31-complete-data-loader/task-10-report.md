# Task 10 report — byte-bounded prefetch and recursive pinning

Implementation commit: `d7d44b9f6bc062e9160c15b15ad33fd944de9c6a`

## Outcome

Task 10 adds opt-in, strict post-transform byte accounting and recursive
post-collation pinning to both owned map loaders and explicitly sharded stream
loaders. Default item-bounded, unpinned loaders retain their prior behavior and
do not acquire `MemoryFootprint` or `PinMemory` bounds.

The public surface adds `MemoryFootprint`, `MemoryDisabled`, `MemoryEnabled`,
`PinMemory`, `PinDisabled`, `PinEnabled`, `Auto`, `Explicit`, and
`PinMemoryStatus`. Both builder families expose order-independent
`prefetch_bytes`, `pin_memory`, and `pin_memory_for` type-state transitions;
both owners report the effective byte budget and pinning status.

## RED-first evidence

The first focused compile run failed on the deliberately written tests because
the memory and pinning traits, status/type states, recursive implementations,
builder setters, enabled build bounds, owner status methods, and worker
integration did not exist. The RED tests also established the required old
explicit generic annotations and the disabled invalid-plan/prevalidation path
before production changes.

Incremental GREEN runs then exposed and fixed:

- byte policy storage that would have forced the final transform/plan bounds
  onto the disabled map `build` path;
- ordered stream admission that needed a bounded per-shard front-waiter marker
  to distinguish a slow next ID from a globally missing ID;
- missing-sequence detection whose initial all-terminal predicate was
  vacuously true; and
- strict-lint findings in the new generic signatures and checked arithmetic.

No test was weakened. The focused final run contains 110 passing tests across
memory, pinning, stream, lifecycle, map, builder, transform, and current-API
suites.

## Public type-state ruling

The preferred direct erased-operation field was not viable for the map owner:
storing a function over `P::Batch` or the final transformed associated type in
the owner forces `P: LoaderPlan` and operational transform bounds while merely
naming/building the disabled owner. That narrows the established contract in
which intentionally discarded or conflicting plans need only
`LoaderPlanConfiguration` so validation can reject them before operational
callbacks.

The accepted resolution is trailing defaulted policy generics on builders,
owners, and iterators. `PinDisabled` supplies the no-op policy without adding a
final-batch bound; enabled iterator availability follows from the enabled
build's `Batch: PinMemory` proof. Byte footprint functions are monomorphized
only by enabled builds. Existing displayed arities remain source-compatible
because the new parameters are defaulted; `current_api` names the old full
builder, owner, serial iterator, worker iterator, stream builder, stream owner,
and stream iterator arities, and also names the new complete policy arities.
This keeps `LoaderPlan` and collation error types unchanged and maps pin errors
directly to the distinct `LoaderError::PinMemory` source chain.

## Byte-footprint and permit contract

- Tensor footprint is checked `numel * element-size`. Scalars use payload
  size; `String` and `Bytes` use owned capacity. `Vec`, `Option`, tuples of
  arity two through eight, and `BTreeMap` keys plus values recursively use
  checked sums that return `usize::MAX` on overflow.
- Map workers measure the aggregate final transformed worker batch. Stream
  workers measure each final transformed record. Raw input, coordinator active
  batch assembly, collated output, and pinned output are not measured.
- A generation-local standard-library `Mutex`/`Condvar` budget grants RAII
  permits without exceeding the exact nonzero limit. Oversize values fail
  before result publication with the configured limit and exact actual size.
- Permits remain charged through bounded result queues and ordered reassembly,
  then release before coordinator collation. Success, stale drain, first
  failure, protocol failure, timeout, cancellation, iterator drop, pool
  quiescence, and owner shutdown all drop retained permits.
- Ordered map and stream paths admit by global sequence instead of first-come
  weighted acquisition. Byte-enabled ordered stream shards must additionally
  emit strictly increasing IDs. One preallocated `Option<u64>` front slot per
  worker makes a missing global ID a typed protocol failure rather than a byte
  deadlock. Byte-disabled Task 9 ordering remains unchanged.
- Generation cancellation centrally cancels the lifecycle token and budget,
  waking condition-variable waiters. There is no unsafe code, unbounded queue,
  detached worker/waiter, sleep, or polling correctness loop.

`MemoryLimit` metadata is exact: map failures carry batch and worker only;
ordered stream failures preserve worker, sequence, and logical ID without a
guessed batch; the mandatory unordered unsequenced case carries worker and
logical ID with no batch or sequence. Every terminal failure remains visible
once and is followed by `None`.

## Pinning contract

Tensor pinning consumes the tensor and delegates to `f_pin_memory(device)`, so
LibTorch failures remain available through `RustTorchError::Backend` under
`LoaderError::PinMemory`. `Vec`, `Option`, tuples two through eight, and
`BTreeMap` values recurse by move; map keys and scalar/string/`Bytes` leaves
are preserved without cloning.

Pinning runs only after successful collation/conversion and immediately before
visibility in serial map, explicit zero-worker worker-capability map, positive
map workers, and ordered or unordered streams. Automatic mode selects CUDA
device zero only when the linked runtime exposes it; CPU-only and MPS-only
runtimes report `DisabledNoAccelerator`, perform a no-op, and retain the exact
batch type. Explicit mode accepts only an available in-range CUDA device and
rejects CPU, MPS, Vulkan/other devices, or unavailable indices before source,
transform, collator, initializer, or thread side effects.

CUDA storage and forced backend-failure assertions are conditional on CUDA
being available. This host did not claim a CUDA execution result when no CUDA
device was exposed.

## Supported and unsupported scope

Supported here is strict opt-in logical-payload accounting for prefetched map
batches and stream records, sequence-aware admission, cancellation-safe
permit lifetime, the documented built-in footprint/container set, recursive
CUDA host pinning, automatic CPU/MPS no-op status, explicit CUDA validation,
typed failures, facade exports, and default/disabled source compatibility.

This is not a whole-process RSS cap: allocator/node overhead and tensor
view/shared-storage backing are outside the footprint contract. The one active
coordinator batch remains item-bounded by `batch_size` and outside the byte
budget. This task does not claim PyTorch dynamic custom `pin_memory` duck
typing, a current-CUDA-index API, accelerator pinning beyond CUDA,
asynchronous-transfer benchmarks, or loader checkpoint/resume. Tasks 11–13
were not implemented.

## Documentation and compatibility

Root and package READMEs document the strict byte-budget boundary, CUDA-zero
automatic behavior, CPU/MPS no-op status, explicit rejection, cooperative
cancellation limitation, and checkpoint exclusion. Stream module docs state
the stronger byte-enabled monotonic shard contract and bounded front waiter.
Canonical compatibility rows for loader pinning, prefetch, workers, and stream
workers contain only the proven scope; generated package compatibility and API
coverage are current. No dependency was added.

## Fresh final verification

Every Rust command sourced `. scripts/dev-env.sh` and used locked dependencies.

- focused memory/pin/stream/lifecycle/map/builder/transform/current-API: 110
  passed;
- root facade `data`: 29 passed;
- `cargo check --workspace --all-targets --locked`: passed;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed;
- `cargo test --workspace --all-targets --locked`: 287 passed, 1 ignored;
- `cargo test --workspace --doc --locked`: 17 passed, including the two new
  dependency-free `compile_fail` bounds examples;
- rusttorch-data doc-only all-target check and warning-denied rustdoc: passed;
- compatibility write/check: current;
- `cargo fmt --all -- --check`: passed; and
- raw `git diff --check`: passed.

The implementation commit is DCO-signed with the required subject
`feat(data): bound loader memory and pin batches`.
