# Full PyTorch functionality: research and gap analysis

**Date:** 2026-09-19 (Europe/Berlin).

**Status:** Research and proposed program; no new implementation or support claim.

**Repository baseline:** RustTorch 0.4.0, `a1ea86f115e9bd24feda34d5353df4d6a074bac7`.

**Execution plan:** [Full PyTorch parity implementation plan](../plans/2026-09-19-full-pytorch-parity.md).

## Finding

RustTorch has useful, tested foundations, but full PyTorch functionality requires
a framework compatibility program. Adding the remaining layer names would leave
major gaps in tensor representation, differentiation, module behavior, runtime
controls, compilation, distributed execution and interoperability. The completed
eight workstream contracts measured earlier deliveries, not the fraction of
PyTorch reproduced.

Keep LibTorch as the numerical engine. First establish which capabilities already
exist through `tch`, which require a binding, which require Rust orchestration,
and which depend on Python. Implementing a second tensor engine or compiler is
not the default solution.

The complete goal needs two explicitly reported execution profiles:

- **Native Rust:** Rust programs build, train and deploy models without a Python
  runtime. Equivalent Rust APIs preserve the relevant observable behavior.
- **Python interoperability:** an explicit, optional runtime boundary preserves
  capabilities whose contract includes Python objects, bytecode or executable
  Python packages. This does not count as native Rust implementation.

Literal Python language identity cannot be supplied by Rust types alone. If
“exact” includes arbitrary Python tensor subclasses, import hooks, pickled
classes and CPython bytecode capture, Python must be available for those paths.
If Python is prohibited everywhere, those requirements remain unsatisfied;
renaming them `python_only` does not complete the user's full-functionality goal.
This is an architectural constraint, not a reason to silently remove features.

## 1. Freeze the target before measuring progress

The repository currently pins PyTorch **2.13.0**, commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`, with `tch` **0.26.0**. The official
latest release observed during research is **2.14.0**, released September 2,
2026, commit `2b3ec34829036a65cd9d1398ea72a0167dc37470`.
[Official release](https://github.com/pytorch/pytorch/releases/tag/v2.14.0),
[release commit](https://github.com/pytorch/pytorch/commit/2b3ec34829036a65cd9d1398ea72a0167dc37470).

Use 2.14.0 as the proposed first full-coverage target, while retaining 2.13.0 as
the measured shipping baseline until a separately tested upgrade lands. Produce
an added/removed/changed API and behavior report first. The 2.14 release changes
boundary gradients, adds control-flow/runtime capabilities and changes legacy
APIs; simply changing a version string would invalidate existing assumptions.
[Release changes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0).

The current `tch` upstream README still requires LibTorch 2.13.0. Investigate a
compatible upstream binding release or a reviewed binding patch; disabling the
native version check is not upgrade evidence.
[tch upstream](https://github.com/LaurentMazare/tch-rs).

Define the target as all public capabilities in the frozen release, including
documented unstable and deprecated features that still exist. Record their
stability separately. Keep internal schemas as implementation dependencies, and
audit any internal-named API explicitly documented for users. Private internals
are not a promise to duplicate PyTorch's source architecture. Removed historical
APIs require a separate legacy-version contract. Later upstream releases enter
a version-delta queue rather than changing an active milestone's denominator.

## 2. What the repository actually measures

These counts were computed from the committed inventory, explicit map and ledger
at the baseline above. They describe records, not percentages of functionality.

| Inventory category | Count |
|---|---:|
| Documented Python identities | 9,361 |
| Canonical ATen schemas | 3,101 |
| Public-name ATen schemas | 2,261 |
| Internal-name ATen schemas | 840 |
| Total identities | 12,462 |

| Current disposition | Ledger rows | Identities mapped to those rows |
|---|---:|---:|
| `supported` | 24 | 45 |
| `partial` | 75 | 224 |
| `planned` | 34 | 11,343 |
| `python_only` | 5 | 7 |
| `not_supported` | 3 | 843 |
| Total | 141 | 12,462 |

There are **12,179 identities mapped to broad `census.*` rows** and 283 mapped
to narrower rows. All 2,261 public-name ATen schemas still have planned census
dispositions. This does **not** mean their kernels are missing: raw `tch::Tensor`
is exposed by RustTorch and already has a large generated operator surface.
Conversely, a mapping to a supported row only establishes its written scope;
it does not prove every option or backend of the mapped identity.

Sources: [inventory](../../../compat/pytorch_inventory.json),
[mapping](../../../compat/pytorch_inventory_map.toml),
[ledger](../../../compat/pytorch_api.toml),
[collection methodology](../../api-census.md),
[core reexports](../../../crates/rusttorch-core/src/lib.rs).

The existing census is a strong starting point, but not a complete behavioral
specification. It uses CPU/Linux documentation selection, plus a recorded
runtime fingerprint. It needs a union audit of other documented platform/build
surfaces, documented constants, operator registrations outside ATen, public
runtime-only capabilities and separate ecosystem repositories. For example,
`torch.dtype` exists in the snapshot, but no identity starts with `torch.uint`
or `torch.float8`; the dtype values and their support restrictions still matter.

Add a capability matrix for dtype, device, layout, memory format, overload,
defaults, aliasing, mutation, gradients, serialization, global state and feature
interactions. Shell dtypes and backend-specific unsupported combinations must
match the target's actual contract, not an imaginary all-operations/all-devices
Cartesian product. [Upstream tensor attributes](https://docs.pytorch.org/docs/2.14/tensor_attributes.html).

### Reproduce the snapshot counts

Run with the repository's Python environment; only the standard library is used:

```sh
.venv/bin/python - <<'PY'
from collections import Counter
import json
from pathlib import Path
import tomllib

root = Path('.')
symbols = json.loads((root / 'compat/pytorch_inventory.json').read_text())['symbols']
mapping = tomllib.loads((root / 'compat/pytorch_inventory_map.toml').read_text())['mapping']
rows = tomllib.loads((root / 'compat/pytorch_api.toml').read_text())['api']
by_id = {row['id']: row for row in rows}
assert {symbol['id'] for symbol in symbols} == set(mapping)
print('identities', len(symbols))
print('kinds', dict(sorted(Counter(s['kind'] for s in symbols).items())))
print('ATen visibility', dict(sorted(Counter(
    s['visibility'] for s in symbols if s['kind'] == 'aten_schema').items())))
print('ledger statuses', dict(sorted(Counter(r['status'] for r in rows).items())))
print('mapped statuses', dict(sorted(Counter(
    by_id[mapping[s['id']]]['status'] for s in symbols).items())))
print('broad census mappings', sum(v.startswith('census.') for v in mapping.values()))
PY
```

## 3. Gap map

This is a family-level analysis. The plan requires an exact symbol/option backlog
before any family is declared complete; examples below are not exhaustive lists.

| Work package | Existing foundation | Work required for full coverage | Completion evidence |
|---|---|---|---|
| W1 Inventory and conformance | Pinned semantic census, explicit dispositions, scoped tests | Current-release delta; constants/build surfaces; exact Rust binding map; per-behavior cases and backend results | No unexplained public target identities; every claim traces to executed cases |
| W2 Native runtime and bindings | `tch` tensors, CPU/CUDA/MPS helpers | Missing dtype/device representation, generators, streams/events, allocators, dispatch, thread-local guards, build/ABI capability checks | Safe lifecycle/error tests plus real execution on each claimed runtime |
| W3 Tensor operations and layouts | Broad raw bindings; selected indexing, linalg, FFT/special and alias tests | Full overload/default/out/in-place behavior, promotion, complex, sparse compressed, nested/jagged, masked, quantized/shell types and storage semantics | Differential values, metadata, mutation, aliasing and supported gradients |
| W4 Autograd and transforms | Reverse AD; restricted functional derivatives | Native forward AD; composable `torch.func`; custom backward/JVP/vmap rules; graph/saved-tensor hooks; activation recomputation | Composition tests, grad/gradgrad checks, lifetime/reentrancy and mutation failures |
| W5 Modules and model families | Common layers, recurrent/attention/Transformer models | Complete module tree/state behavior, hooks, parameter/buffer registration, lazy/meta construction, containers, parametrizations/pruning, remaining layers/functional/init/loss options | Model/state round trips, configuration tables, train/eval and all parameter/input gradients |
| W6 Optimizers and training | Seven optimizer families, five schedules, dense scaling | Remaining algorithms, closure and advanced execution modes, full state/hooks, schedule composition, device-specific AMP, sparse/complex paths, averaging | Multi-step trajectories, skipped/overflow steps, checkpoint restart and compile/distributed interactions |
| W7 Probability | Normal/Bernoulli/Categorical | Remaining distributions, constraints, transforms, transformed/independent families, KL registry, broadcasting, reparameterized gradients | Domain/shape validation, analytic checks, gradients and statistical sampling tests |
| W8 Data and process execution | Typed threaded loaders, samplers, collation, bounded streams, exact scoped resume | Process workers, spawn semantics, per-worker iterable batching, generator/seed lifecycle, pinning, IPC/sharing, error/shutdown behavior and Python adapters | Fixed sample traces, worker-failure tests and Windows/Linux/macOS execution |
| W9 State and interoperability | SafeTensors weights, typed training checkpoint, scoped PT2/ONNX | Full state_dict semantics; storage/view preservation; native generator state; trusted Python checkpoints/package objects; DLPack/array protocols; distributed checkpoint formats | Bidirectional interchange and continued training with matching next steps |
| W10 Compiler and deployment | Explicit graph, bounded PT2/ONNX subset, real TorchScript artifacts | FX/export/capture equivalents, dynamic shapes and guards, custom ops, AOTAutograd/Inductor interoperability, AOT package loading, NativeRT, full target ONNX surface | Eager/export/compiled comparison; graph-break/guard/control-flow tests; compile/run/memory measurements |
| W11 Distributed | Synchronous CPU TCP collectives and explicit replicated/sharded training | Native backend contracts, async work, autograd hooks/buckets, DDP/FSDP2, meshes/DTensor, tensor/pipeline parallelism, elastic launch, RPC and checkpoint resharding | Multi-GPU/multi-node trajectories, mixed Rust/Python peers where claimed, faults and scaling measurements |
| W12 Framework utilities | Scoped assertions/gradcheck, host timing, local RNG, typed checkpoints | Native profiler/kernel/memory events, monitor/logging, benchmark utilities, full RNG/determinism, futures, configuration/environment controls, package/hub/TensorBoard integrations | Observable event/state tests, runtime traces and documented performance overhead |
| W13 Ecosystem and extensions | Five bounded domain pipelines | Independently versioned domain/model/weight/dataset coverage; modern quantization; custom extensions; deployment/accelerator ecosystem adapters | Per-project manifests, licenses, real workload and artifact interoperability |
| W14 Documentation and release | Examples, rustdoc, nine-crate release/provenance workflow | Consistent capability navigation and limitations; complete use cases; feature/platform packaging and continuous release drift control | Executed examples, tested packages, all published docs and support claims agree |

### Binding limits are a prerequisite, not cosmetic API gaps

At the pinned dependency, `tch::Device` represents CPU, CUDA, MPS and Vulkan;
`Kind` has 20 variants. This does not represent the whole target device/dtype
catalog, including meta/XPU/MTIA/custom-device cases and newer scalar types.
Adding a RustTorch enum alone cannot make conversions through `tch` safe.
Preserve tensor ownership and existing interoperability when extending this
boundary; test unknown native enum values rather than allowing panics or casts.

The audited dependency source is `tch` commit
`4227b89b72059b0651ff83a38637693574e0a2d6`, recorded in the published crate's
`.cargo_vcs_info.json`; the relevant files are `src/wrappers/device.rs`,
`src/wrappers/kind.rs` and `torch-sys/build.rs`. Its 2,657 generated fallible
tensor methods and corresponding panic-on-error methods are paired entry
points, not 5,314 independently verified operations.

Evaluate upstream `tch` changes first. For gaps, compare PyTorch's limited stable
C/C++ ABI with a small version-checked bridge. The stable surface includes
tensor, dispatcher and accelerator facilities, but does not cover the complete
framework; Python conversions still require `libtorch_python` and the GIL.
Its availability is a candidate, not proof that RustTorch can use every feature.
[Stable ABI documentation](https://docs.pytorch.org/docs/2.14/notes/libtorch_stable_abi.html).

### Model coverage includes the module system

RustTorch's single-tensor `Module` and `VarStore` facilities do not by themselves
reproduce arbitrary input/output trees, recursive module introspection, tied
parameters, nonpersistent buffers, hooks or state-dictionary transformations.
These contracts underpin custom models, export and distributed training. Design
their Rust ownership/callback boundary before adding dependent features.
[Module semantics](https://docs.pytorch.org/docs/2.14/notes/modules.html),
[extension mechanisms](https://docs.pytorch.org/docs/2.14/notes/extending.html).

Examples of additional optimizer families include Adafactor, SparseAdam, ASGD,
LBFGS, Muon, NAdam, RAdam and Rprop. Audit schedules and each algorithm's supported
options rather than assuming every optimizer accepts every flag. Closures,
foreach/fused execution and differentiable updates are separate contracts.
[Optimizer reference](https://docs.pytorch.org/docs/2.14/optim.html).

Functional derivatives need composition: `vmap(grad(...))`, `jacfwd`, `jacrev`,
functional module calls and mutation removal. Reverse-over-reverse JVP cannot
be counted as native dual-tensor forward AD. Distribution coverage also includes
transforms and constraints, not just a class per probability density.
[Function transforms](https://docs.pytorch.org/docs/2.14/func.html),
[distributions](https://docs.pytorch.org/docs/2.14/distributions.html).

### DataLoader equivalence needs observable worker semantics

Current workers share data through Rust threads and collate in the coordinator.
The current streaming path drops the final partial batch after merging records.
PyTorch's multi-process iterable loader drops incomplete batches per worker.
The sample set can therefore differ even with the same batch size and worker
count. Process isolation, initialization, persistence, RNG ownership, timeout
and shutdown behavior also require distinct tests.
[PyTorch data semantics](https://docs.pytorch.org/docs/2.14/data.html).

Keep the useful native threaded path. Add an explicit compatible process/batching
path and Python adapters where Python pickling is part of the contract. Rust
closures are not automatically serializable. Specify a spawn-safe executable
worker/task protocol before implementing process workers. Windows must be a
first-class test host; preserve the existing queued-record/deadline regression
and use synchronization to prove event order instead of short sleeps.

### Full compilation cannot be obtained by expanding an operator switch

Dynamo captures Python through CPython frame evaluation. Native Rust needs its
own explicit capture contract; running Python capture through an adapter is a
different profile. FX transformations, symbolic shapes, guards, functionalization
and custom-op metadata are additional work beyond graph execution.
[Compiler architecture](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html).

For compiled inference, first evaluate upstream `AOTIModelPackageLoader` through
a checked bridge. Distinguish compiled AOT packages from serialized
`ExportedProgram` archives even when both use `.pt2`. Importing a few exported
operators does not implement capture, optimization or training compilation.
[AOTInductor](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html).

TorchScript remains a legacy compatibility lane, not the basis for a new full
compiler. Complete the target's supported export/ONNX/NativeRT paths and document
the runtime dependencies of each. The target release visibly warns on legacy
TorchScript APIs. [Release deprecations](https://github.com/pytorch/pytorch/releases/tag/v2.14.0).

### Distributed training needs backend and failure contracts

The existing CPU TCP implementation proves a scoped Rust training path. It does
not provide native c10d wire compatibility, NCCL/Gloo/UCC/MPI/XCCL availability,
communication overlap, automatic reducer hooks, device meshes or elastic
recovery. Audit backends against upstream's platform/build matrix rather than
advertising each everywhere. FSDP2/DTensor and distributed checkpointing depend
on the module, autograd, placement and state contracts above.
[Distributed reference](https://docs.pytorch.org/docs/2.14/distributed.html).

### Ecosystem scope must be separately pinned

The core `torch` census does not contain every torchvision model, pretrained
weight, dataset, transform or operator, nor every audio/video pipeline. Create
independent manifests for torchvision, torchaudio, torchcodec and modern
quantization in torchao. Include compatibility adapters for historical torchtext
and TorchData functionality under explicit historical pins when required.
Tokenizers, CSV and Parquet capabilities are useful RustTorch additions; they
must not inflate the core PyTorch completion measure.

Classify ExecuTorch, XLA, TorchRec, TorchRL/TensorDict and third-party compiler
backends as separately versioned integration tracks. They stay visible in the
expanded program; completing core `torch` must not be advertised as completing
the entire evolving ecosystem. A model architecture, its trained weights and
its preprocessing recipe are three separate deliverables with provenance.
[Vision](https://github.com/pytorch/vision),
[audio](https://github.com/pytorch/audio),
[codec](https://github.com/meta-pytorch/torchcodec),
[text](https://github.com/pytorch/text),
[quantization direction](https://docs.pytorch.org/docs/2.14/quantization.html).

### Documentation has an immediate consistency gap

At the baseline, `docs/pytorch-compatibility.md` still says RustTorch exposes no
TorchScript or PT2 import/export, and only accepts SafeTensors model state.
Those broad negatives conflict with the scoped 0.4.0 deployment and training
checkpoint implementation. Reconcile that page with the ledger before extending
the roadmap. Do not replace stale negatives with equally broad positive claims.

Keep user-facing guides centered on Rust use cases, shapes, code and failure
recovery. Put upstream source references and parity evidence in compatibility
pages. Every new capability needs a runnable recipe, doctest, precise limits and
a link from both the GitHub guide and the appropriate docs.rs module.

## 4. Define “done” without manufacturing a percentage

For each frozen public capability, record:

1. Exact upstream identity, overload/options, source commit and stability.
2. Native Rust binding or explicit interoperability path, with runtime needs.
3. Applicable dtype/layout/device/build combinations and expected failures.
4. Values, metadata, aliasing/mutation, gradient and state-transition contracts.
5. Executable reference and Rust cases; test seed/input provenance and tolerances.
6. Actual result: pass, fail, untested or upstream-inapplicable, with environment.
7. User example, documentation location and remaining restrictions.

Report binding availability, behavioral conformance, hardware evidence,
interchange and performance separately. A skipped hardware case remains
untested. Passing one workload does not complete every API it touches. Internal
schemas are not added to the denominator to make progress look larger.

“Full functionality for PyTorch 2.14.0” is earned only when all in-scope public
capabilities have an implemented route and all required conformance cells pass.
If some routes need Python, say so in the claim. “Full native Rust parity” cannot
include Python-mediated completions. Neither claim follows from closing a fixed
number of milestone tickets.

Numerical parity means the same mathematical and state semantics within justified
per-operation tolerances, not universal bitwise equality across devices. Require
exact equality for indices, shapes, types, state names and other discrete
contracts. Same-build RNG stream parity is a separate test from statistically
equivalent sampling; the existing local ChaCha generator proves only its own
contract. [Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
[reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## 5. Required engineering resources and unresolved decisions

The critical expertise is native bindings/ABI safety, autograd/dispatch,
compiler systems, distributed systems, numerical testing and release engineering.
After the common inventory/runtime foundation, core training, compiler,
distributed and domain work can run in parallel with shared conformance review.
Documentation and evidence are part of each contribution, not a final cleanup.

Hardware needs include Linux and Windows CPU/CUDA, macOS arm64/MPS, AMD/ROCm,
Intel/XPU, and multi-GPU/multi-node environments for distributed claims. Meta
execution needs no accelerator but does need representable meta tensors. XLA,
MTIA and vendor backends require their own accessible build/hardware evidence;
unavailable facilities are explicit blockers to those claims. Verify the exact
upstream-supported platform combinations at the target pin.

Before committing a calendar estimate, complete W1/W2 and measure representative
operator, callback, compiled-artifact and collective deliveries. They determine
whether remaining work is a safe wrapper, a frontend feature or an upstream
dependency. The present audit does not justify “another 25%,” a completion date,
or a claimed percentage. Estimate each reviewed work package after its binding
and hardware gates, record ranges/confidence, and reserve capacity for upstream
release changes.

The decisions to resolve through those gates are concrete:

- Which missing features can land upstream in `tch`, and which need a local
  version-checked bridge using the existing or stable ABI?
- Can the optional Python boundary preserve identity/autograd where required,
  or does a given feature only support offline artifact exchange?
- What Rust module/input-tree/callback contract supports state, transforms and
  distributed training while preserving existing models?
- Which native packages expose the headers and symbols needed for c10d,
  profiler and compiled-package loading on each supported platform?
- Which hardware/integration tracks have maintainers and executable CI access?

These are required feasibility results. An unsuccessful probe leaves its feature
blocked with a reason and upstream issue; it never becomes a successful milestone.

## Global constraints for the implementation plan

- Shipping baseline: RustTorch 0.4.0, PyTorch 2.13.0, `tch` 0.26.0; proposed target: PyTorch 2.14.0 at `2b3ec34829036a65cd9d1398ea72a0167dc37470`.
- Preserve the existing Rust 1.88 floor unless a reviewed dependency change requires an explicit migration.
- Use LibTorch kernels and existing RustTorch/tch capabilities before adding dependencies or implementing replacements.
- Native Rust and Python interoperability support must be reported separately.
- No support promotion from a name, reexport, inventory mapping, skipped test or successful documentation build.
- Follow `CONTRIBUTING.md`: issue-first scope/API/dependency/unsafe review, provenance, DCO sign-off, focused pull requests and required CI.
- Never create `codex/` branches; use `b/`, `f/`, `docs/`, `test/` or `chore/` as appropriate.
- Every public change includes Rust examples, executable evidence, compatibility records and GitHub/docs.rs documentation.
- Preserve bounded input parsing, recoverable errors and existing checkpoint/data compatibility; document intentional migrations.

## Research limits

This analysis inspected the committed source, census, contribution rules,
existing evidence and current official upstream references. It did not execute
12,462 capability comparisons, regenerate the 2.14 inventory, benchmark new
backends, or validate unimplemented bridges. The plan's initial gates perform
those audits. Floating documentation pages are research context; implementation
tests must cite and run against the selected immutable source and runtime pins.
